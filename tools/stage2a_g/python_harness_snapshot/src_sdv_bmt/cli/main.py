"""SDV-BMT command line and composition root.

Commands (there is deliberately no ``analyze`` command)::

    sdv-bmt validate PLAN               VALIDATE + PREFLIGHT, writes nothing
    sdv-bmt run PLAN [--out DIR]        run(s): virtual clock (blocking mock adapters)
                                        or realtime (polling adapters), writes run bundles
    sdv-bmt evaluate RUN_DIR [--eval-config FILE]
                                        bundle-only evaluation into a new eval_id

Output is JSON on stdout and carries the plan's (or the bundle's)
``execution_kind``; mock output also carries the MOCK notice, and a banner with
the execution kind goes to stderr.  Exit codes: 0 success, 1 run(s) executed but
failed, 2 rejected/invalid input (nothing executed or evaluated).

Components are wired through explicit, static factory tables: there is no
plugin discovery and no dynamic import.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from .. import MOCK_NOTICE, REALTIME_MOCK_NOTICE, STAGE, __version__
from ..contracts.ids import to_jsonable
from ..contracts.records import REALTIME_RECORD_SCHEMA_VERSION, input_pipeline_digest
from ..contracts.specs import DatasetSpec, DrivingScenarioSpec
from ..contracts.status import NOMINAL_LEVEL, ExecutionKind, ExecutionStatus
from ..dut.fake import FakeBlockingDUT, FakePollingDUT
from ..dut.nvidia import NvidiaCudaDUT
from ..evidence.bundle import MANIFEST_FILE, read_json
from ..input.dataset_fixture import DatasetFixtureProvider
from ..input.recorded_scenario_fixture import RecordedScenarioFixtureProvider
from ..metrics.evaluator import EvaluationError, evaluate_run
from ..runner.plan import PlanError, ResolvedPlan, load_evaluation_override, load_plan
from ..runner.runner import BenchmarkRunner, Components, PlanRejected
from ..stress.fake import FakeStressProvider
from ..stress.none import NoStressProvider
from ..workload.fake import FakeIdentityWorkload
from ..workload.reference import ReferenceConvNetWorkload

WORKLOADS = {FakeIdentityWorkload.IMPLEMENTATION: FakeIdentityWorkload,
             ReferenceConvNetWorkload.IMPLEMENTATION: ReferenceConvNetWorkload}
DUT_ADAPTERS = {FakeBlockingDUT.ADAPTER: FakeBlockingDUT, FakePollingDUT.ADAPTER: FakePollingDUT,
                NvidiaCudaDUT.ADAPTER: NvidiaCudaDUT}
STRESS_KINDS = {FakeStressProvider.KIND: FakeStressProvider}


def build_components(rp: ResolvedPlan) -> Components:
    spec = rp.input_spec
    if isinstance(spec, DatasetSpec):
        provider = DatasetFixtureProvider(spec, rp.input_spec_dir)
    elif isinstance(spec, DrivingScenarioSpec):
        provider = RecordedScenarioFixtureProvider(spec, rp.input_spec_dir)
    else:  # pragma: no cover - exhaustive
        raise ValueError(f"unsupported input spec {type(spec).__name__}")
    if rp.workload.implementation not in WORKLOADS:
        raise ValueError(f"unknown workload implementation {rp.workload.implementation!r}")
    if rp.dut.adapter not in DUT_ADAPTERS:
        raise ValueError(f"unknown DUT adapter {rp.dut.adapter!r}")
    workload = WORKLOADS[rp.workload.implementation](rp.workload)
    dut = DUT_ADAPTERS[rp.dut.adapter](rp.dut)
    if rp.stress_level == NOMINAL_LEVEL:
        stress = NoStressProvider(rp.stress_spec)
    elif rp.stress_spec.kind in STRESS_KINDS:
        stress = STRESS_KINDS[rp.stress_spec.kind](rp.stress_spec, rp.stress_level)
    else:
        raise ValueError(f"unsupported stress kind {rp.stress_spec.kind!r} in Stage 1A")
    return Components(input_provider=provider, workload=workload, dut=dut, stress=stress)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

EXIT_OK = 0
EXIT_RUN_FAILED = 1
EXIT_REJECTED = 2


ACTUAL_NOTICE = ("ACTUAL: executed on a real accelerator; reference-accelerator validation, "
                 "not an NPU performance result")


def _notice(kind: Optional[str], realtime: bool = False) -> Optional[str]:
    if kind == ExecutionKind.MOCK.value:
        return REALTIME_MOCK_NOTICE if realtime else MOCK_NOTICE
    return ACTUAL_NOTICE if kind == ExecutionKind.ACTUAL.value else None


def _emit(payload: dict[str, Any], kind: Optional[str], realtime: bool = False) -> None:
    document = {"execution_kind": kind or "unknown", "notice": _notice(kind, realtime), **payload}
    print(json.dumps(to_jsonable(document), indent=2, sort_keys=True, ensure_ascii=False))


def _banner(kind: Optional[str], realtime: bool = False) -> None:
    label = (kind or "unknown").upper()
    print(f"[{label}] sdv-bmt {__version__}: {_notice(kind, realtime) or 'execution kind unknown'}", file=sys.stderr)


def _plan_kind(rp: ResolvedPlan) -> tuple[str, bool]:
    return rp.plan.execution_kind.value, rp.plan.realtime is not None


def _rejected(command: str, issues: list[str], kind: Optional[str] = None, realtime: bool = False) -> int:
    _banner(kind, realtime)
    _emit({"command": command, "status": "rejected", "issues": issues}, kind, realtime)
    return EXIT_REJECTED


def cmd_validate(plan_path: str) -> int:
    try:
        rp = load_plan(plan_path)
    except PlanError as exc:
        return _rejected("validate", exc.issues)
    kind, realtime = _plan_kind(rp)
    try:
        pf = BenchmarkRunner(rp, build_components).preflight()
    except PlanRejected as exc:
        return _rejected("validate", exc.issues, kind, realtime)
    _banner(kind, realtime)
    schedule = pf.schedule.describe()
    offsets = schedule["planned_offsets_ns"]
    _emit({
        "command": "validate",
        "status": "valid",
        "experiment_id": rp.plan.experiment_id,
        "plan_digest": rp.plan_digest,
        "config_digests": rp.config_digests(),
        "input": pf.input_descriptor,
        "schedule": {
            "mode": schedule["mode"],
            "lead_ns": schedule["lead_ns"],
            "planned_requests": len(pf.items),
            "planned_offsets_ns": offsets,
            "episode_start_item_indices": schedule["episode_start_item_indices"],
        },
        "measurement_boundary_id": pf.boundary_id,
        "input_pipeline": {
            "effective": pf.input_pipeline,
            "pipeline_digest": input_pipeline_digest(pf.input_pipeline),
            "override": pf.override_descriptor,
        },
        "stress": {
            "stress_axis_digest": pf.stress_axis_digest,
            "stress_level": rp.stress_level,
            "provider": pf.stress_provider,
            "check": pf.stress_decision,
        },
        "comparison_keys": pf.comparison_keys,
    }, kind, realtime)
    return EXIT_OK


def cmd_run(plan_path: str, out: str) -> int:
    try:
        rp = load_plan(plan_path)
    except PlanError as exc:
        return _rejected("run", exc.issues)
    kind, realtime = _plan_kind(rp)
    _banner(kind, realtime)
    result = BenchmarkRunner(rp, build_components).run_experiment(Path(out))
    runs = [
        {
            "run_id": o.run_id,
            "run_dir": str(o.run_dir),
            "execution_status": o.execution_status,
            "execution_reason": o.execution_reason,
            "measurement_status": o.measurement_status,
            "measurement_reason": o.measurement_reason,
            "request_counts": o.request_counts,
        }
        for o in result.outcomes
    ]
    _emit({
        "command": "run",
        "status": "rejected" if result.rejected and not runs else "executed",
        "experiment_id": rp.plan.experiment_id,
        "plan_digest": rp.plan_digest,
        "runs": runs,
        "rejected": result.rejected,
    }, kind, realtime)
    if not runs:
        return EXIT_REJECTED
    failed = result.rejected or any(o.execution_status is not ExecutionStatus.COMPLETED for o in result.outcomes)
    return EXIT_RUN_FAILED if failed else EXIT_OK


def cmd_evaluate(run_dir: str, eval_config: Optional[str]) -> int:
    try:
        override = load_evaluation_override(eval_config) if eval_config else None
    except PlanError as exc:
        return _rejected("evaluate", exc.issues)
    try:
        outcome = evaluate_run(run_dir, override=override)
    except EvaluationError as exc:
        return _rejected("evaluate", [str(exc)])
    kind = outcome.metrics[0].execution_kind.value if outcome.metrics else None
    realtime = read_json(Path(run_dir) / MANIFEST_FILE).get("schema_version") == REALTIME_RECORD_SCHEMA_VERSION
    _banner(kind, realtime)
    _emit({
        "command": "evaluate",
        "status": "evaluated",
        "run_dir": str(Path(run_dir)),
        "eval_id": outcome.eval_id,
        "eval_dir": str(outcome.eval_dir),
        "metrics_content_sha256": outcome.metrics_content_sha256,
        "metrics": {
            m.metric_id: {"status": m.status, "value": m.value, "unit": m.unit,
                          **({"reason": m.status_reason} if m.status_reason else {}),
                          **({"censored_fraction": m.censored_fraction} if m.censored_fraction is not None else {})}
            for m in outcome.metrics
        },
    }, kind, realtime)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdv-bmt",
        description="SDV-BMT Model-level benchmark core (Stage 1A virtual clock; Stage 1R-G realtime harness).",
    )
    parser.add_argument("--version", action="version", version=f"sdv-bmt {__version__} (Stage {STAGE})")
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate", help="validate a plan (VALIDATE + PREFLIGHT); writes nothing")
    v.add_argument("plan")
    r = sub.add_parser("run", help="execute the plan (virtual clock or realtime) and write run bundles")
    r.add_argument("plan")
    r.add_argument("--out", default="runs", help="output root directory (default: runs)")
    e = sub.add_parser("evaluate", help="evaluate a run bundle into a new evaluations/<eval_id>")
    e.add_argument("run_dir")
    e.add_argument("--eval-config", default=None, help="YAML evaluation override (deadline only)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        return cmd_validate(args.plan)
    if args.command == "run":
        return cmd_run(args.plan, args.out)
    return cmd_evaluate(args.run_dir, args.eval_config)


if __name__ == "__main__":
    sys.exit(main())
