"""BenchmarkRunner: Stage 1A lifecycle on a deterministic virtual clock (design v03 E.1).

::

    VALIDATE -> PREFLIGHT -> OPEN BUNDLE -> INPUT.open -> DUT.prepare -> PREPARE_INPUTS
    -> STRESS.prepare -> STRESS.activate -> STRESS.verify -> MEASURE -> DRAIN
    -> STRESS.deactivate -> STRESS.restore -> DUT.cleanup -> INPUT.release -> FINALIZE

On any failure the cleanup steps run in reverse start order; ``stress.restore``
runs on every path once the bundle is open.  An activate/verify failure never
falls back to nominal: MEASURE is skipped and the run fails.  The runner owns
the prepared-input store and discards it before ``DUT.cleanup``.

Input pipeline: ``plan.input_pipeline.mode`` selects either the Workload path
(``payload -> Workload.preprocess -> DUTAdapter.prepare_input``) or the DUT
adapter's override (``payload -> AdapterInputOverride.prepare_from_payload``).
The adapter must offer the override named by the plan's ``implementation_id``
(and match optional version/digest pins) and the override must implement the
WorkloadInputContract semantics, or the plan is rejected in PREFLIGHT (never a
fallback).  Override-prepared inputs are checked against the Workload
reference path (``input.equivalence``) before any stress or MEASURE step; the
run continues only when equivalence is ``passed`` (fail closed).

Stage 1R: a DUT adapter with ``completion_model=polling`` runs on the realtime
path (``runner/realtime.py``): same lifecycle, host monotonic run clock, the
process pinned to ``plan.realtime.cpu_affinity``, optional untimed
preconditioning before MEASURE, backend snapshots before MEASURE and after
DRAIN, and a v4 manifest.  A blocking adapter keeps the Stage 1A virtual path
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .. import MOCK_NOTICE, REALTIME_MOCK_NOTICE
from ..contracts.ids import canonical_json_bytes, digest_of, new_run_id, to_jsonable, utc_now_iso
from ..contracts.protocols import (
    AdapterInputOverride,
    DUTAdapter,
    HardwareStressProvider,
    InputProvider,
    Workload,
)
from ..contracts.records import (
    METRIC_CATALOG_VERSION,
    REALTIME_RECORD_SCHEMA_VERSION,
    RECORD_SCHEMA_VERSION,
    AdapterInputOverrideDescriptor,
    ComparisonKeyRecord,
    DUTCapabilities,
    DUTDescriptor,
    HostInfo,
    InputDescriptor,
    EquivalenceValidation,
    InputItem,
    InputPipelineRecord,
    RealizationEvidence,
    REALIZATION_SCHEMA_VERSION,
    RunManifest,
    StressRecord,
    SupportDecision,
    WorkloadInputContract,
    input_contract_digest,
    input_pipeline_digest,
)
from ..contracts.specs import (
    BOUNDARY_SCHEMA_VERSION,
    REALTIME_BOUNDARY_SCHEMA_VERSION,
    MeasurementBoundary,
    RealtimeMeasurementBoundary,
    WorkloadSpec,
    dut_config_identity_digest,
    workload_identity_digest,
    measurement_boundary_id,
    stress_axis_digest,
    stress_level_value,
)
from ..contracts.status import (
    BENCHMARK_LEVEL,
    NOMINAL_LEVEL,
    CompletionModel,
    Coverage,
    EquivalenceStatus,
    EvidenceValidity,
    ExecutionKind,
    ExecutionStatus,
    InputPipelineMode,
    InputPipelineOwner,
    MeasurementStatus,
    NominalCheck,
    Outcome,
    RealizedCondition,
    RunClockDomain,
    StressStep,
    VerificationStatus,
)
from ..evidence.bundle import (
    MANIFEST_FILE,
    RUN_SPEC_FILE,
    BundleError,
    create_run_dir,
    ensure_experiment_dir,
    raw_file_records,
    read_json,
    run_dirs,
    write_json_exclusive,
)
from ..evidence.keys import common_fixed_identity, comparison_keys
from ..evidence.manifest import REALTIME_VOLATILE_FIELDS, VOLATILE_FIELDS, write_manifest
from ..observation.events import REALTIME_EVENT_SCHEMA_VERSION
from ..evidence.provenance import code_identity, environment
from ..observation.reader import restore_recorded_ok
from ..observation.recorder import ObservationRecorder
from .core import (
    PreparedStore,
    RequestTable,
    count_identity_violations,
    error_info,
    prepare_inputs,
    prepare_inputs_with_override,
    validate_input_equivalence,
)
from .plan import YAML_VERSION, ResolvedPlan, validate_plan
from .schedule import Schedule, ScheduleError, compute_schedule
from .realtime import MonotonicRunClock, RealtimeMeasureLoop, pin_cpus, restore_cpus, run_precondition
from .virtual import MeasureAborted, MeasureSummary, VirtualClock, VirtualMeasureLoop

VIRTUAL_HOST = HostInfo(host_kind="virtual")
REAL_HOST = HostInfo(host_kind="real")  # Stage 1R realtime runs: the actual host
# run_spec.json v2 adds the input_pipeline section; v3 (realtime runs) adds the realtime section.
RUN_SPEC_SCHEMA_VERSION = 2
REALTIME_RUN_SPEC_SCHEMA_VERSION = 3
# realized conditions whose evidence a polling adapter provides
DEVICE_CONDITIONS = (
    RealizedCondition.INPUT_RESIDENCY, RealizedCondition.BUFFER_STATE, RealizedCondition.DATA_MOVEMENT,
    RealizedCondition.RUNTIME_QUEUE_STATE, RealizedCondition.DEVICE_FREQUENCY, RealizedCondition.THERMAL_STATE,
    RealizedCondition.COMPETING_DEVICE_USERS, RealizedCondition.MEMORY_PRESSURE_BANDWIDTH,
)


@dataclass
class Components:
    input_provider: InputProvider
    workload: Workload
    dut: DUTAdapter
    stress: HardwareStressProvider


ComponentFactory = Callable[[ResolvedPlan], Components]


class PlanRejected(Exception):
    """VALIDATE/PREFLIGHT rejection: nothing is executed and no bundle is created."""

    def __init__(self, issues: list[str]) -> None:
        super().__init__("; ".join(issues))
        self.issues = list(issues)


@dataclass(frozen=True)
class Preflight:
    components: Components
    input_descriptor: InputDescriptor
    items: tuple[InputItem, ...]
    schedule: Schedule
    dut_descriptor: DUTDescriptor
    dut_capabilities: DUTCapabilities
    stress_decision: SupportDecision
    stress_provider: str
    stress_axis_digest: str
    stress_level_value: Optional[float]
    boundary: MeasurementBoundary
    boundary_id: str
    comparison_keys: ComparisonKeyRecord
    input_contract: WorkloadInputContract
    input_pipeline: InputPipelineRecord
    input_override: Optional[AdapterInputOverride]
    override_descriptor: Optional[AdapterInputOverrideDescriptor]
    realtime: bool = False
    declaration: Optional[dict[str, Any]] = None  # polling adapter measurement declaration


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    run_dir: Path
    execution_status: ExecutionStatus
    execution_reason: Optional[str]
    measurement_status: MeasurementStatus
    measurement_reason: Optional[str]
    stress_status: dict[str, Any]
    request_counts: Optional[dict[str, int]]


@dataclass
class ExperimentResult:
    outcomes: list[RunOutcome]
    rejected: Optional[list[str]] = None


class _StepFailed(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class _RunState:
    input_open_attempted: bool = False
    input_opened: bool = False
    dut_prepare_attempted: bool = False
    store: Optional[PreparedStore] = None
    preparation: Optional[dict[str, Any]] = None  # realized PREPARE_INPUTS outcome
    stress_activate_attempted: bool = False
    verify_record: Optional[StressRecord] = None
    verify_end_ns: Optional[int] = None
    table: Optional[RequestTable] = None
    measure_started: bool = False
    measure: Optional[MeasureSummary] = None
    observe_ok: bool = True
    deactivate_ns: Optional[int] = None
    restore_ok: Optional[bool] = None
    equivalence: Optional[EquivalenceValidation] = None
    failures: list[str] = field(default_factory=list)
    # Stage 1R realtime runs
    clock_info: Optional[dict[str, Any]] = None
    previous_affinity: Optional[set[int]] = None
    realized_affinity: tuple[int, ...] = ()
    precondition: Optional[dict[str, Any]] = None
    observations: dict[str, Any] = field(default_factory=dict)
    loop: Optional[RealtimeMeasureLoop] = None


def build_boundary(dut: DUTDescriptor, workload: WorkloadSpec,
                   pipeline: InputPipelineRecord) -> MeasurementBoundary:
    """Stage 1A boundary: source read, preprocess and prepare_input happen before
    MEASURE (eager preparation) and are excluded; postprocess counts only toward
    output-ready.  The effective input pipeline (owner, semantics, fused steps
    and the full pipeline digest, i.e. the implementation identity) is part of
    the boundary identity."""
    return MeasurementBoundary(
        schema_version=BOUNDARY_SCHEMA_VERSION,
        clock_domain=RunClockDomain.VIRTUAL,
        payload_read_decode="excluded",
        preprocess="excluded",
        prepare_input_h2d="excluded",
        submit=dut.submit_semantics,
        complete=dut.complete_semantics,
        postprocess="output_ready_only",
        output_ready=workload.output_ready,
        host_step_virtual_cost_ns=0,
        latency_definitions={
            "request_completion": "t_complete - t_sched",
            "dut_service": "t_complete - t_submit_begin",
            "output_ready": "t_output_ready - t_sched",
        },
        input_pipeline_owner=pipeline.owner,
        preprocess_semantics_id=pipeline.preprocess_semantics_id,
        fused_steps=pipeline.fused_steps,
        input_pipeline_digest=input_pipeline_digest(pipeline),
    )


def build_realtime_boundary(dut: DUTDescriptor, workload: WorkloadSpec, pipeline: InputPipelineRecord,
                            declaration: dict[str, Any]) -> RealtimeMeasurementBoundary:
    """Stage 1R boundary: eager preparation stays outside MEASURE; whether the
    per-request host-to-device copy is inside the DUT-service interval, how
    device time is obtained and the instrumentation profile are declared by the
    adapter.  The effective input pipeline is part of the identity, as in v2."""
    return RealtimeMeasurementBoundary(
        schema_version=REALTIME_BOUNDARY_SCHEMA_VERSION,
        clock_domain=RunClockDomain.MONOTONIC,
        payload_read_decode="excluded",
        preprocess="excluded",
        prepare_input_h2d=declaration["prepare_input_h2d"],
        submit=dut.submit_semantics,
        complete=dut.complete_semantics,
        completion_observation="runner spin-polls PollingDUTAdapter.poll(); "
                               "t_complete = first poll that reports completion",
        postprocess="output_ready_only",
        output_ready=workload.output_ready,
        latency_definitions={
            "issue_lag": "t_issue - t_sched",
            "request_completion": "t_complete - t_sched",
            "dut_service": "t_complete - t_submit_begin",
            "output_ready": "t_output_ready - t_sched",
        },
        device_timing=declaration.get("device_timing"),
        instrumentation=declaration["instrumentation"],
        input_pipeline_owner=pipeline.owner,
        preprocess_semantics_id=pipeline.preprocess_semantics_id,
        fused_steps=pipeline.fused_steps,
        input_pipeline_digest=input_pipeline_digest(pipeline),
    )


class BenchmarkRunner:
    def __init__(
        self,
        plan: ResolvedPlan,
        factory: ComponentFactory,
        *,
        run_id_factory: Callable[[], str] = new_run_id,
        now_factory: Callable[[], str] = utc_now_iso,
    ) -> None:
        self._plan = plan
        self._factory = factory
        self._run_id_factory = run_id_factory
        self._now = now_factory
        self._restore_failed = False

    @property
    def plan(self) -> ResolvedPlan:
        return self._plan

    # ------------------------------------------------------------------ VALIDATE/PREFLIGHT
    def preflight(self) -> Preflight:
        rp = self._plan
        issues = validate_plan(rp)
        if issues:
            raise PlanRejected(issues)
        try:
            components = self._factory(rp)
        except Exception as exc:
            raise PlanRejected([f"component construction failed: {type(exc).__name__}: {exc}"]) from None
        try:
            descriptor = components.input_provider.describe()
            items = tuple(components.input_provider.items())
        except Exception as exc:
            raise PlanRejected([f"input: {exc}"]) from None
        if descriptor.input_kind is not rp.input_spec.input_kind:
            raise PlanRejected(["input: provider kind does not match the input spec"])
        try:
            schedule = compute_schedule(items, rp.plan.replay, descriptor.input_kind)
        except ScheduleError as exc:
            raise PlanRejected([f"replay: {exc}"]) from None

        try:  # a real adapter queries its device here; failing to do so rejects the plan
            dut_desc = components.dut.describe()
            caps = components.dut.capabilities()
            if not isinstance(dut_desc, DUTDescriptor) or not isinstance(caps, DUTCapabilities):
                raise TypeError("DUT adapter describe()/capabilities() returned the wrong record type")
        except Exception as exc:
            raise PlanRejected([f"dut: adapter {rp.dut.adapter!r} could not describe itself: "
                                f"{type(exc).__name__}: {exc}"]) from None
        realtime = caps.completion_model is CompletionModel.POLLING
        issues = []
        if realtime:  # Stage 1R: polling adapter on the monotonic run clock
            if rp.plan.realtime is None:
                issues.append("realtime: a polling DUT adapter requires the plan's realtime section")
            if dut_desc.implementation_kind is not rp.plan.execution_kind:
                issues.append(f"execution_kind: the plan says {rp.plan.execution_kind.value} but DUT adapter "
                              f"{rp.dut.adapter!r} is {dut_desc.implementation_kind.value}")
            if rp.stress_level != NOMINAL_LEVEL:
                issues.append("stress: realtime runs are nominal only (level 'none'); actual stress is Stage 4")
        else:
            if dut_desc.implementation_kind is not ExecutionKind.MOCK:
                issues.append("dut: Stage 1A runs mock adapters only")
            if caps.completion_model is not CompletionModel.BLOCKING:
                issues.append("dut: the Stage 1A runner supports blocking completion only")
            if not caps.virtual_timing:
                issues.append("dut: the virtual runner requires an adapter with virtual timing")
            if rp.plan.realtime is not None:
                issues.append("realtime: the realtime section requires a polling DUT adapter")
        if caps.max_inflight < rp.plan.run.max_inflight:
            issues.append("dut: capability max_inflight is below run.max_inflight")
        decision = components.stress.check(rp.stress_spec, rp.stress_level, REAL_HOST if realtime else VIRTUAL_HOST)
        if not decision.supported:
            issues.append(f"stress: {decision.reason}")
        if issues:
            raise PlanRejected(issues)
        try:
            contract = components.workload.input_contract()
            if not isinstance(contract, WorkloadInputContract):
                raise TypeError("Workload.input_contract() must return a WorkloadInputContract")
        except Exception as exc:
            raise PlanRejected([f"workload: input contract unavailable: {exc}"]) from None
        pipeline, override, override_desc = self._resolve_input_pipeline(components, contract)
        declaration = None
        if realtime:
            try:
                declaration = to_jsonable(components.dut.measurement_declaration())
                boundary = build_realtime_boundary(dut_desc, rp.workload, pipeline, declaration)
            except Exception as exc:
                raise PlanRejected([f"dut: invalid measurement declaration: {type(exc).__name__}: {exc}"]) from None
        else:
            boundary = build_boundary(dut_desc, rp.workload, pipeline)
        boundary_id = measurement_boundary_id(boundary)
        axis = stress_axis_digest(rp.stress_spec)
        level_value = stress_level_value(rp.stress_spec, rp.stress_level)
        common = common_fixed_identity(
            dut_config=rp.dut,
            dut_descriptor=dut_desc,
            dut_capabilities=caps,
            workload=rp.workload,
            measurement_boundary_id=boundary_id,
            replay=rp.plan.replay,
            run=rp.plan.run,
            software_revision=code_identity()["source_tree_sha256"],
            execution_kind=rp.plan.execution_kind,
            metric_catalog_version=METRIC_CATALOG_VERSION,
            input_pipeline=pipeline,
            output_reference=rp.output_reference_content(),
            **(self._realtime_identity(declaration) if realtime else {}),
        )
        keys = comparison_keys(
            common_parts=common,
            input_descriptor=descriptor,
            stress_axis_digest=axis,
            stress_level_table=rp.stress_spec.levels,
            stress_level=rp.stress_level,
            stress_level_value=level_value,
        )
        return Preflight(
            components=components,
            input_descriptor=descriptor,
            items=items,
            schedule=schedule,
            dut_descriptor=dut_desc,
            dut_capabilities=caps,
            stress_decision=decision,
            stress_provider=getattr(components.stress, "PROVIDER", type(components.stress).__name__),
            stress_axis_digest=axis,
            stress_level_value=level_value,
            boundary=boundary,
            boundary_id=boundary_id,
            comparison_keys=keys,
            input_contract=contract,
            input_pipeline=pipeline,
            input_override=override,
            override_descriptor=override_desc,
            realtime=realtime,
            declaration=declaration,
        )

    def _realtime_identity(self, declaration: dict[str, Any]) -> dict[str, Any]:
        cfg = self._plan.plan.realtime
        return {
            "run_clock_domain": RunClockDomain.MONOTONIC,
            "telemetry": declaration["instrumentation"],
            "host_placement": {"cpu_affinity": list(cfg.cpu_affinity)},
            "realtime_harness": {"wait_policy": cfg.wait_policy.value,
                                 "precondition_requests": cfg.precondition_requests},
        }

    def _notice(self) -> Optional[str]:
        if self._plan.plan.execution_kind is not ExecutionKind.MOCK:
            return None
        return REALTIME_MOCK_NOTICE if self._plan.plan.realtime is not None else MOCK_NOTICE

    def _resolve_input_pipeline(
        self, components: Components, contract: WorkloadInputContract
    ) -> tuple[InputPipelineRecord, Optional[AdapterInputOverride], Optional[AdapterInputOverrideDescriptor]]:
        """Resolve the explicitly selected input pipeline; never auto-select or fall back."""
        rp = self._plan
        selection = rp.plan.input_pipeline
        contract_fields = {
            "preprocess_semantics_id": contract.preprocess_semantics_id,
            "input_contract_id": contract.contract_id,
            "input_contract_version": contract.contract_version,
            "input_contract_digest": input_contract_digest(contract),
        }
        if selection.mode is InputPipelineMode.WORKLOAD:
            record = InputPipelineRecord(
                mode=InputPipelineMode.WORKLOAD,
                owner=InputPipelineOwner.WORKLOAD,
                implementation_id=rp.workload.preprocess.id,
                implementation_version=rp.workload.preprocess.version,
                implementation_digest=digest_of({"workload_implementation": rp.workload.implementation,
                                                 "preprocess": rp.workload.preprocess}),
                fused_steps=(),
                **contract_fields,
            )
            return record, None, None
        adapter = rp.dut.adapter
        try:
            override = components.dut.input_override(contract, selection.implementation_id)
        except Exception as exc:
            raise PlanRejected([f"input_pipeline: DUT adapter {adapter!r} failed to provide an input override: "
                                f"{type(exc).__name__}: {exc}"]) from None
        if override is None:
            raise PlanRejected([f"input_pipeline: adapter_override requested but DUT adapter {adapter!r} "
                                f"provides no input override {selection.implementation_id!r} "
                                "(no fallback to the workload path)"])
        try:
            descriptor = override.describe()
            if not isinstance(descriptor, AdapterInputOverrideDescriptor):
                raise TypeError("AdapterInputOverride.describe() must return an AdapterInputOverrideDescriptor")
        except Exception as exc:
            raise PlanRejected([f"input_pipeline: invalid input override descriptor: {exc}"]) from None
        issues = []
        if descriptor.implementation_id != selection.implementation_id:
            issues.append(f"input_pipeline: plan selects override {selection.implementation_id!r} but DUT "
                          f"adapter {adapter!r} provides {descriptor.implementation_id!r}")
        if (selection.implementation_version is not None
                and descriptor.implementation_version != selection.implementation_version):
            issues.append(f"input_pipeline: plan pins implementation_version {selection.implementation_version!r} "
                          f"but the override is {descriptor.implementation_version!r}")
        if (selection.implementation_digest is not None
                and descriptor.implementation_digest != selection.implementation_digest):
            issues.append("input_pipeline: plan pins implementation_digest "
                          f"{selection.implementation_digest} but the override is {descriptor.implementation_digest}")
        if contract.preprocess_semantics_id not in descriptor.preprocess_semantics_ids:
            issues.append(f"input_pipeline: override {descriptor.implementation_id!r} does not implement "
                          f"preprocess semantics {contract.preprocess_semantics_id!r}")
        if issues:
            raise PlanRejected(issues)
        record = InputPipelineRecord(
            mode=InputPipelineMode.ADAPTER_OVERRIDE,
            owner=InputPipelineOwner.DUT_ADAPTER,
            implementation_id=descriptor.implementation_id,
            implementation_version=descriptor.implementation_version,
            implementation_digest=descriptor.implementation_digest,
            fused_steps=descriptor.fused_steps,
            **contract_fields,
        )
        return record, override, descriptor

    # ------------------------------------------------------------------ records
    def experiment_record(self) -> dict[str, Any]:
        rp = self._plan
        return {
            "schema_version": 1,
            "benchmark_level": BENCHMARK_LEVEL,
            "execution_kind": rp.plan.execution_kind,
            "mock_notice": self._notice(),
            "experiment_id": rp.plan.experiment_id,
            "plan_digest": rp.plan_digest,
            "resolved_plan": rp.resolved_content(),
            "config_files": {cf.role: {"relpath": cf.relpath, "sha256": cf.sha256} for cf in rp.config_files},
            "created_at_utc": self._now(),
        }

    def run_spec(self, pf: Preflight) -> dict[str, Any]:
        rp = self._plan
        reference = dict(rp.output_reference_content())
        if rp.output_reference_entries is not None:
            reference["entries"] = list(rp.output_reference_entries)
        spec = {
            "schema_version": REALTIME_RUN_SPEC_SCHEMA_VERSION if pf.realtime else RUN_SPEC_SCHEMA_VERSION,
            "benchmark_level": BENCHMARK_LEVEL,
            "execution_kind": rp.plan.execution_kind,
            "mock_notice": self._notice(),
            "experiment_id": rp.plan.experiment_id,
            "plan_digest": rp.plan_digest,
            "resolved_plan": rp.resolved_content(),
            "config_files": {cf.role: {"relpath": cf.relpath, "sha256": cf.sha256} for cf in rp.config_files},
            "input_descriptor": pf.input_descriptor,
            "dut_descriptor": pf.dut_descriptor,
            "dut_capabilities": pf.dut_capabilities,
            "measurement_boundary": pf.boundary,
            "measurement_boundary_id": pf.boundary_id,
            "schedule": pf.schedule.describe(),
            "stress": {
                "stress_axis_digest": pf.stress_axis_digest,
                "stress_level": rp.stress_level,
                "stress_level_value": pf.stress_level_value,
                "provider": pf.stress_provider,
                "check": pf.stress_decision,
            },
            "evaluation": {"deadline": rp.plan.evaluation.deadline, "output_reference": reference},
            "input_pipeline": {
                "selection": rp.plan.input_pipeline,
                "effective": pf.input_pipeline,
                "pipeline_digest": input_pipeline_digest(pf.input_pipeline),
                "input_contract": pf.input_contract,
                "override": pf.override_descriptor,
            },
        }
        if pf.realtime:
            spec["realtime"] = {"config": rp.plan.realtime, "measurement_declaration": pf.declaration,
                                "clock_source": "CLOCK_MONOTONIC"}
        return spec

    def _codec(self) -> str:
        codec = self._plan.workload.codec
        return f"{codec.id}/{codec.version}"

    def _stress_failure_record(self, pf: Preflight, step: StressStep, exc: BaseException) -> StressRecord:
        return StressRecord(
            step=step,
            outcome=Outcome.FAILED,
            provider=pf.stress_provider,
            stress_axis_digest=pf.stress_axis_digest,
            stress_level=self._plan.stress_level,
            error=error_info(exc),
            details={"exception": True},
        )

    # ------------------------------------------------------------------ run
    def check_not_blocked(self, out_root: Path) -> None:
        """A stress restore failure blocks later runs (design v03 G.1).

        Checked in this process and on disk for every experiment under
        ``out_root``: each earlier run needs a manifest recording restore=ok or,
        if it never finalized, a successful restore record in its stress.jsonl.
        Runs written to a different output root are not visible here.
        """
        if self._restore_failed:
            raise PlanRejected(["a previous run failed stress restore; further runs are blocked"])
        root = Path(out_root)
        experiments = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        for exp_dir in experiments:
            for run_dir in run_dirs(exp_dir):
                manifest_path = run_dir / MANIFEST_FILE
                if manifest_path.exists():
                    try:
                        status = read_json(manifest_path)["stress"]["status"]
                        ok = status.get("restore") == "ok"
                    except (OSError, ValueError, KeyError, TypeError, AttributeError):
                        ok = False
                else:
                    ok = restore_recorded_ok(run_dir)
                if not ok:
                    raise PlanRejected([
                        f"earlier run {exp_dir.name}/{run_dir.name} has no successful stress restore; "
                        "further runs are blocked"
                    ])

    def run_once(self, out_root: Path, repetition: int = 0) -> RunOutcome:
        self.check_not_blocked(out_root)
        pf = self.preflight()
        try:
            exp_dir = ensure_experiment_dir(Path(out_root), self.experiment_record())
        except BundleError as exc:
            raise PlanRejected([str(exc)]) from None
        run_id = self._run_id_factory()
        run_dir = create_run_dir(exp_dir, run_id)
        write_json_exclusive(run_dir / RUN_SPEC_FILE, self.run_spec(pf))

        state = _RunState()
        if pf.realtime:
            # pin before any preparation so host buffers are allocated on the planned CPUs/node
            state.previous_affinity, state.realized_affinity = pin_cpus(self._plan.plan.realtime.cpu_affinity)
            clock: Any = MonotonicRunClock()
            state.clock_info = clock.describe()
            rec = ObservationRecorder(run_dir, RunClockDomain.MONOTONIC, REALTIME_EVENT_SCHEMA_VERSION)
        else:
            clock = VirtualClock(0)
            rec = ObservationRecorder(run_dir)
        try:
            return self._run_bundle(pf, rec, clock, state, run_id, run_dir, repetition)
        finally:
            if state.previous_affinity is not None:
                restore_cpus(state.previous_affinity)

    def _run_bundle(self, pf: Preflight, rec: ObservationRecorder, clock: Any, state: _RunState,
                    run_id: str, run_dir: Path, repetition: int) -> RunOutcome:
        try:
            rec.write_input_items(pf.items)
            rec.stress(clock.now, StressRecord(
                step=StressStep.CHECK,
                outcome=Outcome.OK,
                provider=pf.stress_provider,
                stress_axis_digest=pf.stress_axis_digest,
                stress_level=self._plan.stress_level,
                details={"supported": True, "reason": pf.stress_decision.reason,
                         "host_kind": (REAL_HOST if pf.realtime else VIRTUAL_HOST).host_kind, "preflight": True},
            ))
            try:
                self._forward(pf, rec, clock, state)
            except _StepFailed as failed:
                state.failures.append(failed.reason)
        finally:
            self._cleanup(pf, rec, clock, state)
        outcome = self._finalize(pf, rec, clock, state, run_id, run_dir, repetition)
        if state.restore_ok is False:
            self._restore_failed = True
        return outcome

    def run_experiment(self, out_root: Path) -> ExperimentResult:
        result = ExperimentResult(outcomes=[])
        for repetition in range(self._plan.plan.repetitions):
            try:
                result.outcomes.append(self.run_once(out_root, repetition))
            except PlanRejected as rejected:
                result.rejected = rejected.issues
                break
        return result

    # ------------------------------------------------------------------ forward steps
    def _step(self, rec: ObservationRecorder, clock: VirtualClock, name: str,
              call: Callable[[], Any], reason: str) -> Any:
        rec.event("phase", clock.now, phase=name, edge="begin")
        try:
            value = call()
        except Exception as exc:
            rec.event("phase", clock.now, phase=name, edge="end", outcome="failed", error=error_info(exc))
            raise _StepFailed(reason) from exc
        rec.event("phase", clock.now, phase=name, edge="end", outcome="ok")
        return value

    def _call_stress(self, pf: Preflight, step: StressStep,
                     call: Callable[[], StressRecord]) -> tuple[StressRecord, bool]:
        """Invoke a provider method. An exception, a non-StressRecord result or a
        record that cannot be written as canonical evidence becomes a failed
        record, so recording it can never interrupt the lifecycle.
        Returns (record, raised)."""
        try:
            record = call()
            if not isinstance(record, StressRecord):
                raise TypeError("stress provider must return a StressRecord")
            canonical_json_bytes(record)
            return record, False
        except Exception as exc:
            return self._stress_failure_record(pf, step, exc), True

    def _stress_step(self, pf: Preflight, rec: ObservationRecorder, clock: VirtualClock, name: str,
                     step: StressStep, call: Callable[[], StressRecord],
                     advance_ns: int = 0) -> StressRecord:
        """Run one forward stress step; the caller decides what a failure means."""
        rec.event("phase", clock.now, phase=name, edge="begin")
        record, raised = self._call_stress(pf, step, call)
        if not raised:
            clock.advance(advance_ns)  # the verify window elapses on the virtual clock
        rec.stress(clock.now, record)
        if record.outcome is Outcome.OK:
            rec.event("phase", clock.now, phase=name, edge="end", outcome="ok")
        else:
            rec.event("phase", clock.now, phase=name, edge="end", outcome="failed", error=record.error)
        return record

    def _observe(self, pf: Preflight, rec: ObservationRecorder, clock: VirtualClock, state: _RunState) -> None:
        record, _raised = self._call_stress(pf, StressStep.OBSERVE, pf.components.stress.observe)
        if record.outcome is not Outcome.OK:
            state.observe_ok = False
        rec.stress(clock.now, record)

    def _input_equivalence(self, pf: Preflight, rec: ObservationRecorder, clock: VirtualClock,
                           state: _RunState) -> None:
        """Validate override-prepared inputs against the Workload reference path.
        Fail closed: anything other than ``passed`` stops the run before stress."""
        c = pf.components
        assert state.store is not None
        rec.event("phase", clock.now, phase="input.equivalence", edge="begin")
        try:
            result, errors = validate_input_equivalence(c.input_provider, c.workload, c.dut, pf.items, state.store)
        except Exception as exc:
            state.equivalence = EquivalenceValidation(EquivalenceStatus.NOT_PERFORMED, None, 0,
                                                      reason=f"validation_error:{type(exc).__name__}")
            rec.event("phase", clock.now, phase="input.equivalence", edge="end", outcome="failed",
                      error=error_info(exc))
            raise _StepFailed("input_equivalence_not_established") from exc
        state.equivalence = result
        passed = result.status is EquivalenceStatus.PASSED
        rec.event("phase", clock.now, phase="input.equivalence", edge="end",
                  outcome="ok" if passed else "failed",
                  details={**to_jsonable(result), "errors": to_jsonable(errors)})
        if result.status is EquivalenceStatus.FAILED:
            # The override does not implement the Workload's input semantics: never measure.
            raise _StepFailed("input_equivalence_failed")
        if not passed:
            raise _StepFailed("input_equivalence_not_established")

    def _forward(self, pf: Preflight, rec: ObservationRecorder, clock: VirtualClock, state: _RunState) -> None:
        rp = self._plan
        c = pf.components

        state.input_open_attempted = True
        self._step(rec, clock, "input.open", c.input_provider.open, "input_open_failed")
        state.input_opened = True

        state.dut_prepare_attempted = True
        self._step(rec, clock, "dut.prepare", lambda: c.dut.prepare(rp.workload, rp.dut), "dut_prepare_failed")

        store = state.store = PreparedStore()
        override = pf.input_override
        rec.event("phase", clock.now, phase="prepare_inputs", edge="begin")
        try:
            if override is None:
                prepare_inputs(c.input_provider, c.workload, c.dut, pf.items, store)
            else:
                prepare_inputs_with_override(c.input_provider, override, pf.items, store)
        except Exception as exc:
            rec.event("phase", clock.now, phase="prepare_inputs", edge="end", outcome="failed",
                      error=error_info(exc))
            raise _StepFailed("prepare_inputs_failed") from exc
        failed = store.failed
        # relative to the planned items, so it stays comparable across selections
        state.preparation = {
            "failed_items": [{"item_index": i, "stage": f.stage} for i, f in sorted(failed.items())],
        }
        rec.event(
            "phase", clock.now, phase="prepare_inputs", edge="end",
            outcome="partial" if failed else "ok",
            details={
                "prepared": store.prepared_count,
                "failed_items": [
                    {"item_index": i, "stage": f.stage, "error": f.error} for i, f in sorted(failed.items())
                ],
            },
        )
        if override is not None:
            self._input_equivalence(pf, rec, clock, state)

        record = self._stress_step(pf, rec, clock, "stress.prepare", StressStep.PREPARE, c.stress.prepare)
        if record.outcome is not Outcome.OK:
            raise _StepFailed("stress_prepare_failed")
        state.stress_activate_attempted = True
        record = self._stress_step(pf, rec, clock, "stress.activate", StressStep.ACTIVATE, c.stress.activate)
        if record.outcome is not Outcome.OK:
            raise _StepFailed("stress_activate_failed")
        window = rp.stress_spec.verification.window_ns
        state.verify_record = self._stress_step(
            pf, rec, clock, "stress.verify", StressStep.VERIFY,
            lambda: c.stress.verify(window), advance_ns=window,
        )
        verify = state.verify_record
        if verify.outcome is not Outcome.OK or verify.verification is VerificationStatus.FAILED:
            # Never fall back to nominal: no MEASURE after a failed verification.
            raise _StepFailed("stress_verify_failed")
        state.verify_end_ns = clock.now

        state.table = RequestTable(len(pf.items))
        if pf.realtime:
            self._realtime_measure(pf, rec, clock, state, store)
            return
        loop = VirtualMeasureLoop(
            clock=clock,
            recorder=rec,
            adapter=c.dut,
            workload=c.workload,
            items=pf.items,
            store=store,
            schedule=pf.schedule,
            run=rp.plan.run,
            codec=self._codec(),
            table=state.table,
            on_measure_begin=lambda: self._observe(pf, rec, clock, state),
            on_drain_end=lambda: self._observe(pf, rec, clock, state),
        )
        state.measure_started = True
        try:
            state.measure = loop.run()
        except MeasureAborted as exc:
            raise _StepFailed("measure_aborted") from exc

    def _realtime_measure(self, pf: Preflight, rec: ObservationRecorder, clock: MonotonicRunClock,
                          state: _RunState, store: PreparedStore) -> None:
        rp = self._plan
        c = pf.components
        cfg = rp.plan.realtime
        assert cfg is not None and state.table is not None
        if state.realized_affinity != tuple(cfg.cpu_affinity):
            rec.event("phase", clock.now, phase="host.placement", edge="end", outcome="failed",
                      details={"requested": list(cfg.cpu_affinity), "realized": list(state.realized_affinity)})
            raise _StepFailed("cpu_affinity_not_realized")
        if cfg.precondition_requests:
            rec.event("phase", clock.now, phase="precondition", edge="begin",
                      details={"requests": cfg.precondition_requests})
            try:
                state.precondition = run_precondition(c.dut, store, pf.items, cfg.precondition_requests, clock,
                                                      rp.plan.run.request_timeout_ns)
            except Exception as exc:
                rec.event("phase", clock.now, phase="precondition", edge="end", outcome="failed",
                          error=error_info(exc))
                raise _StepFailed("precondition_failed") from exc
            rec.event("phase", clock.now, phase="precondition", edge="end", outcome="ok", details=state.precondition)

        def observe(point: str) -> None:
            try:
                snapshot = to_jsonable(c.dut.observe(point))
                outcome, error = "ok", None
            except Exception as exc:  # an unavailable snapshot leaves the evidence not observable
                snapshot, outcome, error = None, "failed", error_info(exc)
            state.observations[point] = snapshot
            rec.event("phase", clock.now, phase=f"dut.observe.{point}", edge="end", outcome=outcome,
                      **({"details": snapshot} if snapshot is not None else {"error": error}))

        def before_measure() -> None:
            observe("measure_begin")
            self._observe(pf, rec, clock, state)

        def after_drain() -> None:
            self._observe(pf, rec, clock, state)
            observe("drain_end")

        loop = state.loop = RealtimeMeasureLoop(
            clock=clock,
            recorder=rec,
            adapter=c.dut,
            workload=c.workload,
            items=pf.items,
            store=store,
            schedule=pf.schedule,
            run=rp.plan.run,
            codec=self._codec(),
            table=state.table,
            before_measure=before_measure,
            after_drain=after_drain,
        )
        state.measure_started = True
        try:
            state.measure = loop.run()
        except MeasureAborted as exc:
            raise _StepFailed("measure_aborted") from exc

    # ------------------------------------------------------------------ cleanup
    def _cleanup(self, pf: Preflight, rec: ObservationRecorder, clock: VirtualClock, state: _RunState) -> None:
        """Reverse-order cleanup. Every step runs even if recording an earlier step
        failed (the component call always precedes its record); the first
        recorder error is re-raised only after all steps were attempted."""
        c = pf.components
        recorder_errors: list[Exception] = []

        def skipped(step: str, why: str) -> Callable[[], None]:
            return lambda: rec.event("cleanup", clock.now, step=step, outcome="skipped", details={"reason": why})

        def stress_cleanup(step_name: str, step: StressStep, call: Callable[[], StressRecord],
                           reason: str) -> Callable[[], None]:
            def run() -> None:
                record, _raised = self._call_stress(pf, step, call)
                ok = record.outcome is Outcome.OK
                if step is StressStep.RESTORE:
                    state.restore_ok = ok
                if not ok:
                    state.failures.append(reason)
                rec.stress(clock.now, record)
                rec.event("cleanup", clock.now, step=step_name, outcome="ok" if ok else "failed",
                          **({} if ok else {"error": record.error}))
            return run

        def discard() -> None:
            discarded = state.store.discard()  # type: ignore[union-attr]
            rec.event("cleanup", clock.now, step="prepared_store.discard", outcome="ok",
                      details={"discarded": discarded})

        def component_cleanup(name: str, call: Callable[[], None], reason: str) -> Callable[[], None]:
            def run() -> None:
                try:
                    call()
                except Exception as exc:
                    state.failures.append(reason)
                    rec.event("cleanup", clock.now, step=name, outcome="failed", error=error_info(exc))
                else:
                    rec.event("cleanup", clock.now, step=name, outcome="ok")
            return run

        steps: list[Callable[[], None]] = []
        if state.stress_activate_attempted:
            state.deactivate_ns = clock.now
            steps.append(stress_cleanup("stress.deactivate", StressStep.DEACTIVATE,
                                        c.stress.deactivate, "stress_deactivate_failed"))
        else:
            steps.append(skipped("stress.deactivate", "not_activated"))
        # restore runs on every termination path after the bundle is open
        steps.append(stress_cleanup("stress.restore", StressStep.RESTORE, c.stress.restore,
                                    "stress_restore_failed"))
        steps.append(discard if state.store is not None else skipped("prepared_store.discard", "not_created"))
        steps.append(component_cleanup("dut.cleanup", c.dut.cleanup, "dut_cleanup_failed")
                     if state.dut_prepare_attempted else skipped("dut.cleanup", "not_started"))
        steps.append(component_cleanup("input.release", c.input_provider.release, "input_release_failed")
                     if state.input_open_attempted else skipped("input.release", "not_started"))
        for step in steps:
            try:
                step()
            except Exception as exc:  # recorder failure: keep cleaning up components
                recorder_errors.append(exc)
        if recorder_errors:
            raise recorder_errors[0]

    # ------------------------------------------------------------------ finalize
    def _measurement_status(self, state: _RunState, counts: Optional[dict[str, int]]) -> tuple[MeasurementStatus, Optional[str]]:
        if not state.measure_started:
            return MeasurementStatus.INVALID, "not_measured"
        if state.measure is None or counts is None:
            return MeasurementStatus.INVALID, "measure_aborted"
        if counts["unresolved"]:
            return MeasurementStatus.INVALID, "unresolved_requests"
        if count_identity_violations(counts):
            return MeasurementStatus.INVALID, "count_identity_violation"
        return MeasurementStatus.VALID, None

    def _stress_status(self, state: _RunState) -> dict[str, Any]:
        rp = self._plan
        nominal = rp.stress_level == NOMINAL_LEVEL
        record = state.verify_record
        if record is None:
            verification = VerificationStatus.NOT_APPLICABLE if nominal else VerificationStatus.UNVERIFIED
        elif record.outcome is not Outcome.OK:
            verification = VerificationStatus.FAILED
        else:
            verification = record.verification or (
                VerificationStatus.NOT_APPLICABLE if nominal else VerificationStatus.UNVERIFIED
            )
        measure = state.measure
        if nominal:
            coverage = Coverage.NOT_APPLICABLE
        elif measure is None or state.verify_end_ns is None or state.deactivate_ns is None:
            coverage = Coverage.NONE
        elif state.verify_end_ns <= measure.t_measure_begin and state.deactivate_ns >= measure.t_drain_end:
            coverage = Coverage.FULL if state.observe_ok else Coverage.PARTIAL
        else:
            coverage = Coverage.NONE
        return {
            "verification": verification,
            "coverage": coverage,
            "nominal_check": NominalCheck.NOT_PERFORMED,
            "restore": "ok" if state.restore_ok else "failed",
            "verify_end_t_run_ns": state.verify_end_ns,
            "deactivate_t_run_ns": state.deactivate_ns,
        }

    @staticmethod
    def _equivalence(pf: Preflight, state: _RunState) -> EquivalenceValidation:
        if pf.input_pipeline.mode is InputPipelineMode.WORKLOAD:
            return EquivalenceValidation(EquivalenceStatus.NOT_APPLICABLE, None, 0, reason="workload_reference_path")
        if state.equivalence is None:
            return EquivalenceValidation(EquivalenceStatus.NOT_PERFORMED, None, 0, reason="not_reached")
        return state.equivalence

    def _realtime_realization(self, pf: Preflight, state: _RunState) -> dict[str, Any]:
        """Realization evidence of a Stage 1R run.

        Runner-owned conditions (input payload and preparation, issue timing,
        stress) come from what the runner itself observed; device-side
        conditions come from the polling adapter's ``realization_evidence()``.
        A device condition the adapter does not cover is ``not_observable``.
        Continuous evidence (issue lag) has no approved equivalence rule, so it
        can never establish equivalence (``unapproved:`` rule -> unresolved).
        """
        rp = self._plan
        actual = rp.plan.execution_kind is ExecutionKind.ACTUAL
        evidence: dict[RealizedCondition, RealizationEvidence] = {}

        def add(condition: RealizedCondition, validity: EvidenceValidity, source: str, semantic: str,
                value: Any = None, unit: Optional[str] = None, scope: str = "run", clock: Optional[str] = None,
                equivalence: str = "exact") -> None:
            evidence[condition] = RealizationEvidence(condition, validity, source, semantic, unit, scope, clock,
                                                      value, equivalence)

        source = f"input:{pf.input_descriptor.provider}"
        if state.input_opened:
            add(RealizedCondition.INPUT_PAYLOAD, EvidenceValidity.OBSERVED, source,
                "INPUT.open re-read the source index and re-verified its SHA-256 and the realized items digest",
                value={"items_digest": pf.input_descriptor.items_digest, "item_count": len(pf.items)})
        else:
            add(RealizedCondition.INPUT_PAYLOAD, EvidenceValidity.NOT_OBSERVABLE, source,
                "the input source was not opened in this run")
        if state.preparation is None:
            add(RealizedCondition.INPUT_PREPARATION, EvidenceValidity.NOT_OBSERVABLE, "runner:prepare_inputs",
                "PREPARE_INPUTS did not complete in this run")
        else:
            add(RealizedCondition.INPUT_PREPARATION,
                EvidenceValidity.OBSERVED if actual else EvidenceValidity.VIRTUAL, "runner:prepare_inputs",
                "per-item outcome of the selected input pipeline on the DUT adapter; for adapter_override, "
                "the equivalence check against the Workload reference path",
                value={"pipeline_digest": input_pipeline_digest(pf.input_pipeline), **state.preparation,
                       "equivalence": self._equivalence(pf, state).status.value})
        lags = sorted(state.loop.issue_lags) if state.loop is not None else []
        if lags:
            n = len(lags)
            rank = lambda p: lags[min(n - 1, max(0, -(-p * n // 100) - 1))]  # noqa: E731 - nearest rank
            add(RealizedCondition.ISSUE_TIMING, EvidenceValidity.OBSERVED, "runner:monotonic_clock",
                "issue lag t_issue - t_sched of issued requests (nearest-rank quantiles)",
                value={"n": n, "p50": rank(50), "p99": rank(99), "max": lags[-1]}, unit="ns",
                clock=RunClockDomain.MONOTONIC.value, equivalence="unapproved:replay_fidelity_rule")
        else:
            add(RealizedCondition.ISSUE_TIMING, EvidenceValidity.NOT_OBSERVABLE, "runner:monotonic_clock",
                "no request was issued in this run", clock=RunClockDomain.MONOTONIC.value)
        add(RealizedCondition.STRESS_REALIZATION, EvidenceValidity.NOT_OBSERVABLE, f"stress:{pf.stress_provider}",
            "nominal: absence of interference is not checked (nominal_check=not_performed)")
        try:
            provided = tuple(pf.components.dut.realization_evidence())
            failure = None
        except Exception as exc:  # unavailable evidence stays unavailable; it is never invented
            provided, failure = (), f"{type(exc).__name__}: {exc}"
        counts: dict[RealizedCondition, int] = {}
        for record in provided:
            if isinstance(record, RealizationEvidence) and record.condition in DEVICE_CONDITIONS:
                counts[record.condition] = counts.get(record.condition, 0) + 1
                evidence[record.condition] = record
        for condition in DEVICE_CONDITIONS:
            if counts.get(condition, 0) != 1:
                reason = ("the DUT adapter provided duplicate evidence" if counts.get(condition)
                          else "the DUT adapter provided no evidence" + (f" ({failure})" if failure else ""))
                add(condition, EvidenceValidity.NOT_OBSERVABLE, f"dut:{rp.dut.adapter}", reason, scope="device")
        return {
            "schema_version": REALIZATION_SCHEMA_VERSION,
            "execution_kind": rp.plan.execution_kind,
            "evidence": [to_jsonable(evidence[c]) for c in RealizedCondition],
        }

    def _realtime_section(self, state: _RunState) -> dict[str, Any]:
        cfg = self._plan.plan.realtime
        assert cfg is not None
        return {
            "cpu_affinity_requested": list(cfg.cpu_affinity),
            "cpu_affinity_realized": list(state.realized_affinity),
            "wait_policy": cfg.wait_policy.value,
            "precondition_requests": cfg.precondition_requests,
            "precondition": state.precondition,
        }

    def _realization(self, pf: Preflight, state: _RunState, stress_status: dict[str, Any]) -> dict[str, Any]:
        """Generic realization evidence (one record per realized condition).

        Stage 1A records only what it genuinely observes (the realized input
        payload identity, re-verified by INPUT.open); everything produced by the
        mock adapter, the virtual clock or the virtual stressor is marked
        ``virtual``; device/host conditions are ``not_observable``.  Nothing here
        may be read as evidence about real NPU conditions.
        """
        if pf.realtime:
            return self._realtime_realization(pf, state)
        rp = self._plan
        evidence: dict[RealizedCondition, RealizationEvidence] = {}

        def add(condition: RealizedCondition, validity: EvidenceValidity, source: str, semantic: str,
                value: Any = None, unit: Optional[str] = None, scope: str = "run",
                clock: Optional[str] = None) -> None:
            evidence[condition] = RealizationEvidence(condition, validity, source, semantic, unit, scope,
                                                      clock, value)

        source = f"input:{pf.input_descriptor.provider}"
        if state.input_opened:
            add(RealizedCondition.INPUT_PAYLOAD, EvidenceValidity.OBSERVED, source,
                "INPUT.open re-read the source index and re-verified its SHA-256 and the realized items digest",
                value={"items_digest": pf.input_descriptor.items_digest, "item_count": len(pf.items)})
        else:
            add(RealizedCondition.INPUT_PAYLOAD, EvidenceValidity.NOT_OBSERVABLE, source,
                "the input source was not opened in this run")
        # The realized preparation outcome of the selected pipeline (either mode):
        # which items were prepared and which failed (they become NOT_ISSUED).
        if state.preparation is None:
            add(RealizedCondition.INPUT_PREPARATION, EvidenceValidity.NOT_OBSERVABLE, "runner:prepare_inputs",
                "PREPARE_INPUTS did not complete in this run")
        else:
            add(RealizedCondition.INPUT_PREPARATION, EvidenceValidity.VIRTUAL, "runner:prepare_inputs",
                "per-item outcome of the selected input pipeline on the mock adapter; for adapter_override, "
                "the equivalence check against the Workload reference path",
                value={"pipeline_digest": input_pipeline_digest(pf.input_pipeline), **state.preparation,
                       "equivalence": self._equivalence(pf, state).status.value})
        add(RealizedCondition.ISSUE_TIMING, EvidenceValidity.VIRTUAL, "runner:virtual_clock",
            "issue time equals scheduled time by construction on the virtual clock",
            value={"issue_lag_ns": 0}, unit="ns", clock=RunClockDomain.VIRTUAL.value)
        if rp.stress_level == NOMINAL_LEVEL:
            add(RealizedCondition.STRESS_REALIZATION, EvidenceValidity.NOT_OBSERVABLE, f"stress:{pf.stress_provider}",
                "nominal: absence of interference is not checked (nominal_check=not_performed)")
        else:
            add(RealizedCondition.STRESS_REALIZATION, EvidenceValidity.VIRTUAL, f"stress:{pf.stress_provider}",
                "virtual stressor self-report (no real resource was loaded)",
                value={"verification": stress_status["verification"], "coverage": stress_status["coverage"]})
        for condition in (RealizedCondition.INPUT_RESIDENCY, RealizedCondition.BUFFER_STATE,
                          RealizedCondition.DATA_MOVEMENT, RealizedCondition.RUNTIME_QUEUE_STATE,
                          RealizedCondition.DEVICE_FREQUENCY, RealizedCondition.THERMAL_STATE,
                          RealizedCondition.COMPETING_DEVICE_USERS, RealizedCondition.MEMORY_PRESSURE_BANDWIDTH):
            add(condition, EvidenceValidity.NOT_OBSERVABLE, "none",
                "no device: Stage 1A mock execution (virtual clock, FakeDUT)", scope="device")
        return {
            "schema_version": REALIZATION_SCHEMA_VERSION,
            "execution_kind": ExecutionKind.MOCK,
            "evidence": [to_jsonable(evidence[c]) for c in RealizedCondition],
        }

    def _manifest(self, pf: Preflight, run_id: str, run_dir: Path, repetition: int,
                  status: dict[str, Any], stress_status: dict[str, Any], state: _RunState) -> RunManifest:
        rp = self._plan
        desc = pf.input_descriptor
        workload = rp.workload
        return RunManifest(
            schema_version=REALTIME_RECORD_SCHEMA_VERSION if pf.realtime else RECORD_SCHEMA_VERSION,
            benchmark_level=BENCHMARK_LEVEL,
            execution_kind=rp.plan.execution_kind,
            mock_notice=self._notice(),
            run_id=run_id,
            experiment_id=rp.plan.experiment_id,
            plan_digest=rp.plan_digest,
            repetition=repetition,
            seed=rp.plan.seed,
            created_at_utc=self._now(),
            code_identity=code_identity(),
            environment=environment({"pyyaml_version": YAML_VERSION}),
            config_digests=rp.config_digests(),
            input={
                "input_kind": desc.input_kind,
                "input_id": desc.input_id,
                "version": desc.version,
                "provider": desc.provider,
                "input_family_digest": desc.input_family_digest,
                "input_selection_digest": desc.input_selection_digest,
                "items_digest": desc.items_digest,
                "item_count": desc.item_count,
                "source_index_sha256": desc.source_index_sha256,
                "annotations_digest": digest_of([i.annotation_digest for i in pf.items]),
                "source_timing_complete": desc.source_timing_complete,
            },
            workload={
                "workload_id": workload.workload_id,
                "implementation": workload.implementation,
                "identity_digest": workload_identity_digest(workload),
                "task": workload.task,
                "artifact_sha256": workload.artifact_sha256,
                "artifact_note": "fake workload: no model artifact" if workload.artifact_sha256 is None else None,
                "precision": workload.precision,
                "batch": workload.batch,
                "codec": self._codec(),
            },
            dut={
                "dut_config_id": rp.dut.dut_config_id,
                "adapter": rp.dut.adapter,
                "identity_digest": dut_config_identity_digest(rp.dut),
                "implementation_kind": pf.dut_descriptor.implementation_kind,
                "descriptor": pf.dut_descriptor,
                "capabilities": pf.dut_capabilities,
            },
            measurement_boundary={
                "measurement_boundary_id": pf.boundary_id,
                "definition": pf.boundary,
            },
            stress={
                "stress_id": rp.stress_spec.stress_id,
                "kind": rp.stress_spec.kind,
                "provider": pf.stress_provider,
                "stress_axis_digest": pf.stress_axis_digest,
                "stress_level": rp.stress_level,
                "stress_level_value": pf.stress_level_value,
                "status": stress_status,
            },
            status=status,
            output_reference=rp.output_reference_content(),
            input_pipeline={
                **to_jsonable(pf.input_pipeline),
                "pipeline_digest": input_pipeline_digest(pf.input_pipeline),
                "measurement_boundary_id": pf.boundary_id,
                "equivalence_validation": to_jsonable(self._equivalence(pf, state)),
            },
            realization=self._realization(pf, state, stress_status),
            comparison_keys=pf.comparison_keys,
            files=raw_file_records(run_dir),
            volatile_fields=REALTIME_VOLATILE_FIELDS if pf.realtime else VOLATILE_FIELDS,
            run_clock=state.clock_info if pf.realtime else None,
            realtime=self._realtime_section(state) if pf.realtime else None,
        )

    def _finalize(self, pf: Preflight, rec: ObservationRecorder, clock: VirtualClock, state: _RunState,
                  run_id: str, run_dir: Path, repetition: int) -> RunOutcome:
        counts = state.table.counts() if state.table is not None and state.measure is not None else None
        if state.failures:
            exec_status, exec_reason = ExecutionStatus.FAILED, state.failures[0]
        else:
            exec_status, exec_reason = ExecutionStatus.COMPLETED, None
        meas_status, meas_reason = self._measurement_status(state, counts)
        stress_status = self._stress_status(state)
        rec.event(
            "run_finished", clock.now,
            execution_status=exec_status,
            execution_reason=exec_reason,
            measurement_status=meas_status,
            measurement_reason=meas_reason,
            request_counts=counts,
            stress_status=stress_status,
        )
        rec.close()
        write_manifest(run_dir, self._manifest(
            pf, run_id, run_dir, repetition,
            status={
                "execution_status": exec_status,
                "execution_reason": exec_reason,
                "measurement_status": meas_status,
                "measurement_reason": meas_reason,
                "execution_kind": self._plan.plan.execution_kind,
                "request_counts": counts,
            },
            stress_status=stress_status,
            state=state,
        ))
        return RunOutcome(
            run_id=run_id,
            run_dir=run_dir,
            execution_status=exec_status,
            execution_reason=exec_reason,
            measurement_status=meas_status,
            measurement_reason=meas_reason,
            stress_status=stress_status,
            request_counts=counts,
        )
