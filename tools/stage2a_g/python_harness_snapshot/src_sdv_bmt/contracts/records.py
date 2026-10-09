"""Record contracts: typed input items, DUT/stress records, per-request views,
output references, metric results, comparison keys and the run manifest."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .ids import digest_of, is_sha256_hex
from .specs import parse_strict, string_tuple
from .status import (
    BENCHMARK_LEVEL,
    FUSABLE_INPUT_STEPS,
    ContrastType,
    ContrastValidity,
    EvidenceValidity,
    GuardStatus,
    RealizedCondition,
    UNIT_TO_NS,
    CompletionModel,
    EquivalenceStatus,
    ExecutionKind,
    InputKind,
    InputPipelineMode,
    InputPipelineOwner,
    MetricStatus,
    Outcome,
    RequestState,
    RunClockDomain,
    StressStep,
    TimeUnit,
    TimingSemantic,
    VerificationStatus,
)

# v2: manifest gained the input_pipeline section (design change "Explicit DUT
# Adapter Preprocess Override").
# v3: manifest gained the realization section and fine-grained comparison
# components (design update "Factor Isolation").  Older bundles are rejected
# with a version error.
RECORD_SCHEMA_VERSION = 3
# v4: Stage 1R realtime bundles (monotonic run clock) add the run_clock and
# realtime sections.  v3 stays the Stage 1A virtual-clock manifest; a v3 bundle
# is never reinterpreted as v4 or vice versa.
REALTIME_RECORD_SCHEMA_VERSION = 4
MANIFEST_SCHEMA_VERSIONS = (RECORD_SCHEMA_VERSION, REALTIME_RECORD_SCHEMA_VERSION)
REALIZATION_SCHEMA_VERSION = 1
CONTRAST_VALIDITY_SCHEMA_VERSION = 1

# Identity of the Stage 1A portable metric catalog (part of the comparison
# identity: "metric/evaluator definition").
METRIC_CATALOG_VERSION = "bmt-portable-v1-stage1a"

# Raw evidence files of a run bundle. Their SHA-256 digests are recorded in
# manifest.json; evaluations never modify them.
RAW_EVIDENCE_FILES = (
    "run_spec.json",
    "input_items.jsonl",
    "events.jsonl",
    "stress.jsonl",
    "outputs.bin",
)


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceTiming:
    value: int
    unit: TimeUnit
    semantic: TimingSemantic
    clock_source: str

    def __post_init__(self) -> None:
        if type(self.value) is not int:
            raise ValueError("SourceTiming.value must be an integer")

    def to_ns(self) -> int:
        return self.value * UNIT_TO_NS[self.unit]


@dataclass(frozen=True)
class ScenarioCoordinates:
    """Only values present in the source; absent coordinates stay ``None``."""

    episode_id: Optional[str] = None
    stream_id: Optional[str] = None
    sensor_id: Optional[str] = None
    frame_id: Optional[str] = None


@dataclass(frozen=True)
class InputItem:
    item_index: int
    source_item_id: str
    payload_ref: str
    payload_sha256: str
    annotation_ref: Optional[str]
    annotation_digest: Optional[str]
    source_timing: Optional[SourceTiming]
    scenario_coordinates: Optional[ScenarioCoordinates]
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.item_index < 0:
            raise ValueError("item_index must be >= 0")
        if not is_sha256_hex(self.payload_sha256):
            raise ValueError("payload_sha256 must be a SHA-256 hex digest")
        if (self.annotation_ref is None) != (self.annotation_digest is None):
            raise ValueError("annotation_ref and annotation_digest must both be set or both null")


def item_identity(item: InputItem) -> dict[str, Any]:
    """Identity of an item: index, source ID, content digests, timing, coordinates.

    Location references (payload_ref/annotation_ref) and descriptive
    source_metadata are deliberately excluded.
    """
    return {
        "item_index": item.item_index,
        "source_item_id": item.source_item_id,
        "payload_sha256": item.payload_sha256,
        "annotation_digest": item.annotation_digest,
        "source_timing": item.source_timing,
        "scenario_coordinates": item.scenario_coordinates,
    }


def item_digest(item: InputItem) -> str:
    return digest_of(item_identity(item))


def items_digest(items: Sequence[InputItem]) -> str:
    return digest_of([item_identity(item) for item in items])


@dataclass(frozen=True)
class InputDescriptor:
    input_kind: InputKind
    input_id: str
    version: str
    provider: str
    input_family_digest: str
    input_selection_digest: str
    items_digest: str
    item_count: int
    source_index_sha256: str
    source_timing_complete: bool


# --------------------------------------------------------------------------
# DUT
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DUTDescriptor:
    adapter: str
    adapter_version: str
    implementation_kind: ExecutionKind
    device: str
    completion_model: CompletionModel
    clock_domain: RunClockDomain
    submit_semantics: str
    complete_semantics: str
    # Exact device/runtime identity detected by the adapter (backend-namespaced
    # keys, e.g. product name, UUID, PCI bus, driver and compiler versions).
    device_identity: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DUTCapabilities:
    completion_model: CompletionModel
    max_inflight: int
    virtual_timing: bool
    device_timing: bool


@dataclass(frozen=True)
class PreparedInfo:
    workload_identity_digest: str
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DUTRequest:
    request_id: int
    item_indices: tuple[int, ...]
    prepared_inputs: tuple[Any, ...]


@dataclass(frozen=True)
class DUTCompletion:
    """Completion of a polling (Stage 1R) request, returned by ``poll()``.

    ``device_timing`` holds device-clock-domain durations in ns keyed by portable
    segment names (``host_to_device``, ``device_execution``, ``device_to_host``);
    ``device_timing_method`` names how they were obtained.  They are never
    subtracted from host run-clock timestamps.  ``observations`` carries
    per-request backend observations under namespaced keys.
    """

    request_id: int
    ok: bool
    raw_output: Any
    error: Optional[ErrorInfo] = None
    device_timing: Optional[dict[str, int]] = None
    device_timing_method: Optional[str] = None
    observations: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ok and self.error is not None:
            raise ValueError("ok completion must not carry an error")
        if not self.ok and self.error is None:
            raise ValueError("error completion requires error info")
        if self.device_timing is not None:
            if self.device_timing_method is None:
                raise ValueError("device_timing requires device_timing_method")
            for name, value in self.device_timing.items():
                if name not in DEVICE_TIMING_SEGMENTS:
                    raise ValueError(f"unknown device timing segment {name!r}")
                if type(value) is not int or value < 0:
                    raise ValueError(f"device timing {name} must be an integer >= 0")


DEVICE_TIMING_SEGMENTS = ("host_to_device", "device_execution", "device_to_host")


# --------------------------------------------------------------------------
# Input pipeline (explicit Workload vs DUT-adapter preprocessing path)
# --------------------------------------------------------------------------

INPUT_CONTRACT_SCHEMA_VERSION = 1


def _check_text(value: object, what: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")


@dataclass(frozen=True)
class WorkloadInputContract:
    """Immutable, versioned description of the logical task input a Workload
    requires.  It is what a DUT adapter receives instead of the Workload object.

    ``logical_input`` is a tuple of ``(key, value)`` string pairs (sorted by
    key on construction) describing format/layout/parameters of the logical
    input, e.g. ``(("layout", "NCHW"), ("dtype", "int8"))``.
    """

    schema_version: int
    contract_id: str
    contract_version: str
    task: str
    preprocess_semantics_id: str
    payload_format: str
    logical_input: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.schema_version != INPUT_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"input contract schema_version must be {INPUT_CONTRACT_SCHEMA_VERSION}")
        for name in ("contract_id", "contract_version", "task", "preprocess_semantics_id", "payload_format"):
            _check_text(getattr(self, name), f"input contract {name}")
        pairs = tuple(tuple(pair) for pair in self.logical_input)
        if not all(len(p) == 2 and all(isinstance(x, str) for x in p) for p in pairs):
            raise ValueError("logical_input must be (key, value) string pairs")
        keys = [k for k, _ in pairs]
        if len(set(keys)) != len(keys):
            raise ValueError("logical_input keys must be unique")
        object.__setattr__(self, "logical_input", tuple(sorted(pairs)))

    def parameter(self, key: str) -> str:
        return dict(self.logical_input)[key]


def input_contract_digest(contract: WorkloadInputContract) -> str:
    return digest_of(contract)


@dataclass(frozen=True)
class AdapterInputOverrideDescriptor:
    """What a DUT-adapter input override claims to implement."""

    implementation_id: str
    implementation_version: str
    implementation_digest: str
    preprocess_semantics_ids: tuple[str, ...]
    fused_steps: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("implementation_id", "implementation_version"):
            _check_text(getattr(self, name), f"override {name}")
        if not is_sha256_hex(self.implementation_digest):
            raise ValueError("override implementation_digest must be a SHA-256 hex digest")
        object.__setattr__(self, "preprocess_semantics_ids", string_tuple(
            self.preprocess_semantics_ids, "override preprocess_semantics_ids", nonempty=True))
        object.__setattr__(self, "fused_steps", string_tuple(
            self.fused_steps, "override fused_steps", FUSABLE_INPUT_STEPS, nonempty=True))


@dataclass(frozen=True)
class InputPipelineRecord:
    """The effective input pipeline of a run (part of the fixed comparison identity)."""

    mode: InputPipelineMode
    owner: InputPipelineOwner
    preprocess_semantics_id: str
    input_contract_id: str
    input_contract_version: str
    input_contract_digest: str
    implementation_id: str
    implementation_version: str
    implementation_digest: str
    fused_steps: tuple[str, ...]

    def __post_init__(self) -> None:
        expected_owner = {
            InputPipelineMode.WORKLOAD: InputPipelineOwner.WORKLOAD,
            InputPipelineMode.ADAPTER_OVERRIDE: InputPipelineOwner.DUT_ADAPTER,
        }[self.mode]
        if self.owner is not expected_owner:
            raise ValueError(f"input pipeline mode {self.mode.value} requires owner {expected_owner.value}")
        override = self.mode is InputPipelineMode.ADAPTER_OVERRIDE
        object.__setattr__(self, "fused_steps", string_tuple(
            self.fused_steps, "fused_steps", FUSABLE_INPUT_STEPS, nonempty=override))
        if not override and self.fused_steps:
            raise ValueError("workload input pipeline has no fused steps")
        for name in ("input_contract_digest", "implementation_digest"):
            if not is_sha256_hex(getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256 hex digest")


def input_pipeline_digest(record: InputPipelineRecord) -> str:
    return digest_of(record)


@dataclass(frozen=True)
class EquivalenceValidation:
    """Override vs Workload reference path, per item:

    * both prepared -> compared by the DUT adapter's canonical digest
      (``validated_items``; differing digests -> ``mismatched_item_indices``)
    * override failed but the reference succeeded -> mismatched
    * reference failed but the override succeeded -> ``reference_error_item_indices``
      (inconclusive, never blamed on the override)
    * both failed -> ``both_failed_item_indices`` (consistent; the item is NOT_ISSUED)
    """

    status: EquivalenceStatus
    method: Optional[str]
    validated_items: int
    mismatched_item_indices: tuple[int, ...] = ()
    reference_error_item_indices: tuple[int, ...] = ()
    both_failed_item_indices: tuple[int, ...] = ()
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status is EquivalenceStatus.FAILED and not self.mismatched_item_indices:
            raise ValueError("failed equivalence requires mismatched items")
        if self.status is EquivalenceStatus.PASSED and (
            self.mismatched_item_indices or self.reference_error_item_indices or self.validated_items < 1
        ):
            raise ValueError("passed equivalence requires >= 1 validated item, no mismatch and no reference error")


@dataclass(frozen=True)
class ErrorInfo:
    code: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not isinstance(self.message, str):
            raise ValueError("ErrorInfo.code and ErrorInfo.message must be strings")


@dataclass(frozen=True)
class DUTResult:
    """Result of a blocking ``run()``. ``virtual_service_ns`` is the virtual-clock
    duration the call occupies; it exists only for the Stage 1A virtual runner."""

    request_id: int
    ok: bool
    raw_output: Any
    virtual_service_ns: int
    error: Optional[ErrorInfo] = None

    def __post_init__(self) -> None:
        if type(self.virtual_service_ns) is not int or self.virtual_service_ns < 0:
            raise ValueError("virtual_service_ns must be an integer >= 0")
        if self.ok and self.error is not None:
            raise ValueError("ok result must not carry an error")
        if not self.ok and self.error is None:
            raise ValueError("error result requires error info")


# --------------------------------------------------------------------------
# Stress
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HostInfo:
    host_kind: str


@dataclass(frozen=True)
class SupportDecision:
    supported: bool
    reason: str


@dataclass(frozen=True)
class StressRecord:
    """Provider-produced lifecycle record. The runner stamps seq and run-clock time."""

    step: StressStep
    outcome: Outcome
    provider: str
    stress_axis_digest: str
    stress_level: str
    requested: Optional[dict[str, Any]] = None
    achieved: Optional[dict[str, Any]] = None
    independent_evidence: Optional[dict[str, Any]] = None
    verification: Optional[VerificationStatus] = None
    error: Optional[ErrorInfo] = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A failed verification can never be reported as a successful step.
        if self.verification is VerificationStatus.FAILED and self.outcome is not Outcome.FAILED:
            raise ValueError("verification=failed requires outcome=failed")


# --------------------------------------------------------------------------
# Observation views
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputReference:
    request_id: int
    file: str
    offset: int
    length: int
    sha256: str
    codec: str


@dataclass(frozen=True)
class LateCompletion:
    result: str  # "ok" | "error"
    t_complete: int
    t_output_ready: Optional[int]
    output_ref: Optional[OutputReference]
    error: Optional[ErrorInfo]
    error_phase: Optional[str]


@dataclass(frozen=True)
class RequestRecord:
    """Per-request admission view folded from raw events."""

    request_id: int
    item_indices: tuple[int, ...]
    phase: str
    t_sched: int
    admission: str  # "issued" | "dropped" | "not_issued"
    admission_reason: Optional[str]
    t_issue: Optional[int]
    t_submit_begin: Optional[int]
    t_submit_end: Optional[int]
    state: RequestState


@dataclass(frozen=True)
class CompletionRecord:
    """Per-request terminal view (issued requests only) folded from raw events."""

    request_id: int
    result: RequestState  # COMPLETED | ERROR | TIMED_OUT
    t_complete: Optional[int]
    t_output_ready: Optional[int]
    output_ref: Optional[OutputReference]
    error: Optional[ErrorInfo]
    error_phase: Optional[str]
    t_timed_out: Optional[int]
    late_completion: Optional[LateCompletion]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricResult:
    metric_id: str
    definition_version: int
    measurement_boundary_id: Optional[str]
    unit: str
    status: MetricStatus
    value: Optional[float]
    n: Optional[int]
    window: Optional[dict[str, int]]
    method: str
    inputs: tuple[str, ...]
    notes: tuple[str, ...]
    eval_id: str
    execution_kind: ExecutionKind
    status_reason: Optional[str] = None
    censored_fraction: Optional[float] = None

    def __post_init__(self) -> None:
        if self.status is MetricStatus.VALID:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise ValueError(f"{self.metric_id}: valid metric requires a numeric value")
        elif self.value is not None:
            # Never substitute 0 or another number for a non-valid metric.
            raise ValueError(
                f"{self.metric_id}: status {self.status.value} requires value=null"
            )
        if self.status is MetricStatus.CENSORED and self.censored_fraction is None:
            raise ValueError(f"{self.metric_id}: censored metric requires censored_fraction")


def numeric_eligible(result: MetricResult) -> bool:
    """Only ``valid`` metrics may enter numeric comparison. Censored/invalid/empty
    results are report-only (Stage 1A computes no comparison at all)."""
    return result.status is MetricStatus.VALID and result.value is not None


# --------------------------------------------------------------------------
# Comparison identity and manifest
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RealizationEvidence:
    """Generic evidence about one realized execution condition.

    ``validity`` says what kind of evidence it is; ``source``/``semantic``/
    ``unit``/``scope``/``clock`` say how to read ``value``.  Values from
    different sources or semantics are never compared with each other.
    """

    condition: RealizedCondition
    validity: EvidenceValidity
    source: str
    semantic: str
    unit: Optional[str]
    scope: str
    clock: Optional[str]
    value: Any = None
    # How two values of this evidence are compared.  Stage 1A supports only
    # "exact" (canonical JSON equality); tolerance rules are a later decision.
    equivalence: str = "exact"

    def __post_init__(self) -> None:
        for name in ("source", "semantic", "scope", "equivalence"):
            _check_text(getattr(self, name), f"realization evidence {name}")
        if self.validity is EvidenceValidity.NOT_OBSERVABLE and self.value is not None:
            raise ValueError("not_observable evidence carries no value")
        if self.validity in (EvidenceValidity.OBSERVED, EvidenceValidity.SELF_REPORTED,
                             EvidenceValidity.VIRTUAL) and self.value is None:
            raise ValueError(f"{self.validity.value} evidence requires a value")


@dataclass(frozen=True)
class DesignGuardResult:
    """Did the specification vary only the intended factor?"""

    status: GuardStatus
    allowed_to_vary: tuple[str, ...]
    observed_differences: tuple[str, ...]
    violations: tuple[str, ...]
    intended_factor_varied: bool
    notes: tuple[str, ...] = ()
    # fixed identities whose value is unknown (null) on both sides
    unknown_identities: tuple[str, ...] = ()


@dataclass(frozen=True)
class RealizationGuardResult:
    """Were the intended-fixed conditions realized equivalently (as far as observable)?"""

    status: GuardStatus
    basis: str
    evidence_available: tuple[str, ...]
    mismatches: tuple[dict[str, Any], ...]
    unresolved: tuple[dict[str, Any], ...]
    # conditions that realize the intended (varied) factor; not required to be equal
    intended_factor_conditions: tuple[str, ...] = ()


@dataclass(frozen=True)
class MeasurementGuardResult:
    """Were both observations produced with the same valid measurement definition?"""

    status: GuardStatus
    boundary_identity: dict[str, Any]
    violations: tuple[str, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContrastValidityResult:
    """Structured three-layer validity of one pairwise contrast (no arithmetic)."""

    schema_version: int
    contrast_type: ContrastType
    run_a: str
    run_b: str
    design_guard: DesignGuardResult
    realization_guard: RealizationGuardResult
    measurement_guard: MeasurementGuardResult
    overall_status: ContrastValidity
    attribution_allowed: bool

    def __post_init__(self) -> None:
        if self.attribution_allowed and self.overall_status is not ContrastValidity.VALID:
            raise ValueError("attribution is allowed only for a valid contrast")


@dataclass(frozen=True)
class ComparisonKeyRecord:
    schema_version: int
    components: dict[str, Any]
    cell_digest: str
    key_stress_contrast: str
    key_scenario_contrast: Optional[str]
    key_interaction: Optional[str]


_RUN_CLOCK_REQUIRED = ("domain", "source", "anchor_monotonic_ns", "anchor_realtime_ns")
_REALTIME_REQUIRED = ("cpu_affinity_requested", "cpu_affinity_realized", "wait_policy", "precondition_requests",
                      "precondition")

_MANIFEST_REQUIRED = {
    "input": (
        "input_kind",
        "input_id",
        "input_family_digest",
        "input_selection_digest",
        "items_digest",
    ),
    "stress": ("stress_axis_digest", "stress_level", "status"),
    "measurement_boundary": ("measurement_boundary_id", "definition"),
    "status": ("execution_status", "measurement_status", "execution_kind", "request_counts"),
    "code_identity": ("source_tree_sha256",),
    "output_reference": ("kind",),
    "realization": ("schema_version", "execution_kind", "evidence"),
    "input_pipeline": (
        "mode",
        "owner",
        "preprocess_semantics_id",
        "input_contract_digest",
        "implementation_id",
        "implementation_version",
        "implementation_digest",
        "fused_steps",
        "measurement_boundary_id",
        "pipeline_digest",
        "equivalence_validation",
    ),
}


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    benchmark_level: str
    execution_kind: ExecutionKind
    mock_notice: Optional[str]
    run_id: str
    experiment_id: str
    plan_digest: str
    repetition: int
    seed: int
    created_at_utc: str
    code_identity: dict[str, Any]
    environment: dict[str, Any]
    config_digests: dict[str, str]
    input: dict[str, Any]
    workload: dict[str, Any]
    dut: dict[str, Any]
    measurement_boundary: dict[str, Any]
    stress: dict[str, Any]
    status: dict[str, Any]
    output_reference: dict[str, Any]
    input_pipeline: dict[str, Any]
    realization: dict[str, Any]
    comparison_keys: ComparisonKeyRecord
    files: dict[str, dict[str, Any]]
    volatile_fields: tuple[str, ...]
    run_clock: Optional[dict[str, Any]] = None  # v4 only
    realtime: Optional[dict[str, Any]] = None  # v4 only

    def __post_init__(self) -> None:
        if self.schema_version not in MANIFEST_SCHEMA_VERSIONS:
            raise ValueError("unsupported manifest schema_version")
        if self.benchmark_level != BENCHMARK_LEVEL:
            raise ValueError(f"benchmark_level must be {BENCHMARK_LEVEL!r}")
        if self.execution_kind is ExecutionKind.MOCK and not self.mock_notice:
            raise ValueError("mock manifest requires mock_notice")
        if self.execution_kind is ExecutionKind.ACTUAL and self.mock_notice is not None:
            raise ValueError("an actual manifest carries no mock notice")
        if self.schema_version == RECORD_SCHEMA_VERSION:
            if self.run_clock is not None or self.realtime is not None:
                raise ValueError("a v3 (virtual-clock) manifest has no run_clock/realtime section")
            if self.execution_kind is not ExecutionKind.MOCK:
                raise ValueError("a v3 (virtual-clock) manifest is mock only")
        else:
            missing = [k for k in _RUN_CLOCK_REQUIRED if k not in (self.run_clock or {})]
            missing += [f"realtime.{k}" for k in _REALTIME_REQUIRED if k not in (self.realtime or {})]
            if missing:
                raise ValueError(f"v4 manifest missing {missing}")
            if self.run_clock["domain"] != RunClockDomain.MONOTONIC.value:
                raise ValueError("a v4 manifest uses the monotonic run clock")
        if self.status.get("execution_kind") != self.execution_kind.value:
            raise ValueError("status.execution_kind must equal execution_kind")
        if self.realization.get("execution_kind") != self.execution_kind.value:
            raise ValueError("realization.execution_kind must equal execution_kind")
        # one strictly valid record for every realized condition, observed or not
        conditions = [parse_strict(RealizationEvidence, e, "realization.evidence").condition.value
                      for e in self.realization.get("evidence", ())]
        if sorted(conditions) != sorted(c.value for c in RealizedCondition):
            raise ValueError("realization.evidence needs exactly one record per realized condition")
        for section, keys in _MANIFEST_REQUIRED.items():
            content = getattr(self, section)
            missing = [k for k in keys if k not in content]
            if missing:
                raise ValueError(f"manifest.{section} missing {missing}")
        if self.input_pipeline.get("measurement_boundary_id") != self.measurement_boundary.get(
            "measurement_boundary_id"
        ):
            raise ValueError("input_pipeline.measurement_boundary_id must equal the run's boundary id")
        missing_files = [
            name
            for name in RAW_EVIDENCE_FILES
            if not is_sha256_hex(self.files.get(name, {}).get("sha256"))
        ]
        if missing_files:
            raise ValueError(f"manifest.files missing digests for {missing_files}")
