"""Bundle-only MetricEvaluator (design v03 I.5).

* Reads only the run bundle (plus the evaluator's own source for its code
  identity).  It never calls the Runner, DUT, workload, stress provider or the
  input source.
* Verifies every raw file against manifest.json before and after evaluating
  and refuses a bundle whose raw files changed.
* Writes a new ``evaluations/<eval_id>/`` directory (never overwrites).
* Deadline is an evaluation parameter and may be overridden; the output
  reference cannot (it must be designated before the run).
* Every metric result carries the bundle's ``execution_kind`` (mock or actual);
  only mock bundles carry a MOCK notice (the one recorded in their manifest).
* Metric notes describe the bundle's run clock: virtual-clock (Stage 1A) notes
  are unchanged; realtime bundles (host monotonic clock) get notes that say
  what their timestamps contain.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ..contracts.ids import digest_of, new_eval_id, to_jsonable, utc_now_iso
from ..contracts.records import MetricResult
from ..contracts.specs import DeadlineSpec, EvaluationOverride, SpecError, parse_strict
from ..contracts.status import BENCHMARK_LEVEL, ExecutionKind, MetricStatus, ReplayMode, RequestState, RunClockDomain
from ..evidence.bundle import (
    EVAL_MANIFEST_FILE,
    EVALUATIONS_DIR,
    MANIFEST_FILE,
    METRICS_FILE,
    file_record,
    read_json,
    verify_raw_files,
    write_json_exclusive,
)
from ..evidence.provenance import code_identity
from ..observation.reader import (
    BundleReadError,
    RunBundle,
    fold_requests,
    folded_counts,
    load_run_bundle,
    measure_window,
)
from .catalog import CATALOG_VERSION, METRICS, P99_MIN_SAMPLES, PERCENTILE_METHOD, PERCENTILES
from .identity import output_identity
from .portable import INF, Computed, maximum, per_second, percentile, planned_percentile, rate

S = RequestState
EVAL_SCHEMA_VERSION = 1
EVAL_VOLATILE_FIELDS = ("eval_id", "run_id", "created_at_utc", "outputs")


class EvaluationError(RuntimeError):
    """The bundle cannot be evaluated (unreadable or raw integrity violated)."""


@dataclass(frozen=True)
class EvaluationOutcome:
    eval_id: str
    eval_dir: Path
    metrics: tuple[MetricResult, ...]
    metrics_content_sha256: str


def evaluation_parameters(bundle: RunBundle, override: Optional[EvaluationOverride]) -> tuple[dict[str, Any], str]:
    spec_eval = bundle.run_spec["evaluation"]
    if override is not None:
        deadline = to_jsonable(override.deadline)
        source = "override"
    else:
        deadline = spec_eval["deadline"]
        source = "run_spec"
        if deadline is not None:
            deadline = to_jsonable(parse_strict(DeadlineSpec, deadline, "run_spec.evaluation.deadline"))
    reference = {k: v for k, v in spec_eval["output_reference"].items() if k != "entries"}
    params = {
        "catalog_version": CATALOG_VERSION,
        "percentile_method": PERCENTILE_METHOD,
        "p99_min_samples": P99_MIN_SAMPLES,
        "planned_percentile_noncompleted_value": "+inf",
        "deadline": deadline,
        "output_reference": reference,
    }
    return params, source


def canonical_metric_content(params: dict[str, Any], metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluation result without volatile identifiers (eval_id/run_id)."""
    return {
        "parameters": params,
        "metrics": [{k: v for k, v in m.items() if k != "eval_id"} for m in metrics],
    }


def _count_identity_ok(c: dict[str, int]) -> bool:
    return (
        c["planned"] == c["not_issued"] + c["dropped"] + c["issued"]
        and c["issued"] == c["completed"] + c["error"] + c["timed_out"]
        and c["unresolved"] == 0
    )


def bundle_execution_kind(bundle: RunBundle) -> ExecutionKind:
    return ExecutionKind(bundle.manifest["execution_kind"])


# Notes that depend on the run clock.  The virtual-clock notes are those of
# Stage 1A (golden snapshots); realtime notes describe host-clock timestamps.
_CLOCK_NOTES = {
    RunClockDomain.VIRTUAL.value: {
        "dut_service": "DUT-service boundary of the virtual FakeDUT; not an NPU time",
        "output_ready": "Stage 1A virtual clock: host steps cost 0 ns, so output-ready equals "
                        "request-completion for completed requests",
        "issue_lag": "virtual clock: schedule diagnostic only, not replay fidelity",
        "throughput_window": "window = [t0 (first planned t_sched), drain end] on the virtual clock",
    },
    RunClockDomain.MONOTONIC.value: {
        "dut_service": "host monotonic clock: submit call, the DUT's work as declared by the measurement "
                       "boundary, and the completion-observation (polling) delay; device-domain segment "
                       "durations are separate device_observation evidence and are not part of this value",
        "output_ready": "host monotonic clock: includes host postprocess, encode and the output write after "
                        "t_complete",
        "issue_lag": "host monotonic clock: realized issue lag of the single-threaded issuer (replay "
                     "fidelity evidence); no approved replay-fidelity rule",
        "throughput_window": "window = [t0 (first planned t_sched), drain end] on the host monotonic run clock",
    },
}


def bundle_clock_domain(bundle: RunBundle) -> str:
    return bundle.manifest["measurement_boundary"]["definition"]["clock_domain"]


def compute_metrics(bundle: RunBundle, params: dict[str, Any], eval_id: str) -> list[MetricResult]:
    manifest = bundle.manifest
    boundary_id = manifest["measurement_boundary"]["measurement_boundary_id"]

    def result(metric_id: str, c: Computed, *, window: Optional[dict[str, Any]] = None,
               inputs: tuple[str, ...] = ("events.jsonl",), notes: tuple[str, ...] = ()) -> MetricResult:
        definition = next(m for m in METRICS if m.metric_id == metric_id)
        return MetricResult(
            metric_id=metric_id,
            definition_version=definition.definition_version,
            measurement_boundary_id=boundary_id,
            unit=definition.unit,
            status=c.status,
            value=c.value,
            n=c.n,
            window=window,
            method=definition.method,
            inputs=inputs,
            notes=notes,
            eval_id=eval_id,
            execution_kind=bundle_execution_kind(bundle),
            status_reason=c.reason,
            censored_fraction=c.censored_fraction,
        )

    def blanket(status: MetricStatus, reason: str) -> list[MetricResult]:
        return [result(m.metric_id, Computed(status, None, 0, reason)) for m in METRICS]

    clock_notes = _CLOCK_NOTES[bundle_clock_domain(bundle)]
    has_measure = any(e["kind"] == "phase" and e["phase"] == "measure" and e["edge"] == "begin"
                      for e in bundle.events)
    if not has_measure:
        return blanket(MetricStatus.NOT_COLLECTED, "measure_phase_not_executed")
    status = manifest["status"]
    if status["measurement_status"] != "valid":
        return blanket(MetricStatus.INVALID, f"measurement_invalid:{status['measurement_reason']}")
    window = measure_window(bundle.events)
    if window is None:
        return blanket(MetricStatus.INVALID, "measure_window_incomplete")
    try:
        folded = fold_requests(bundle.events)
    except (BundleReadError, SpecError):
        return blanket(MetricStatus.INVALID, "event_sequence_invalid")
    counts = folded_counts(folded)
    finished = [e for e in bundle.events if e["kind"] == "run_finished"]
    if not finished or finished[-1]["request_counts"] != counts:
        return blanket(MetricStatus.INVALID, "counts_disagree_with_run_finished")
    if not _count_identity_ok(counts):
        return blanket(MetricStatus.INVALID, "count_identity_violation")

    planned = counts["planned"]
    entries = list(folded.values())
    t_sched = [r.t_sched for r, _ in entries]
    # t0 = first planned arrival; lead time before it is not part of the window.
    t0 = t_sched[0] if t_sched else window.t_measure_begin
    win = {
        "t_measure_begin_ns": window.t_measure_begin,
        "t0_ns": t0,
        "t_drain_end_ns": window.t_drain_end,
        "drain_cap_reached": window.drain_cap_reached,
    }
    completed = [(r, c) for r, c in entries if r.state is S.COMPLETED and c is not None]
    rc = [c.t_complete - r.t_sched for r, c in completed]
    dut = [c.t_complete - r.t_submit_begin for r, c in completed]
    ready = [c.t_output_ready - r.t_sched for r, c in completed]
    planned_rc = [
        (c.t_complete - r.t_sched) if (r.state is S.COMPLETED and c is not None) else INF
        for r, c in entries
    ]
    lags = [r.t_issue - r.t_sched for r, _ in entries if r.admission == "issued"]

    out: list[MetricResult] = []
    for name in ("planned", "issued", "completed", "dropped", "timed_out", "error", "not_issued"):
        out.append(result(f"bmt.count.{name}", Computed(MetricStatus.VALID, counts[name], planned), window=win))
    out.append(result("bmt.diagnostic.late_completion.count",
                      Computed(MetricStatus.VALID, counts["late_completion"], counts["timed_out"]), window=win,
                      notes=("late completions never change the count identity",)))
    for name, source in (("completion", "completed"), ("drop", "dropped"), ("timeout", "timed_out"),
                         ("error", "error")):
        out.append(result(f"bmt.rate.{name}", rate(counts[source], planned), window=win))

    for p in PERCENTILES:
        minimum = P99_MIN_SAMPLES if p == 99 else 1
        out.append(result(f"bmt.latency.request_completion.completed.p{p}", percentile(rc, p, minimum),
                          window=win, notes=("completed-only; interpret together with bmt.rate.completion",)))
        out.append(result(f"bmt.latency.request_completion.planned.p{p}",
                          planned_percentile(planned_rc, p, P99_MIN_SAMPLES if p == 99 else 0),
                          window=win, notes=("non-completed planned requests are +inf; censored results are "
                                             "report-only and carry value=null",
                                             "p99 requires at least 100 completed samples")))
        out.append(result(f"bmt.latency.dut_service.p{p}", percentile(dut, p, minimum), window=win,
                          notes=(clock_notes["dut_service"],)))
        out.append(result(f"bmt.latency.output_ready.completed.p{p}", percentile(ready, p, minimum), window=win,
                          notes=(clock_notes["output_ready"],)))
    out.append(result("bmt.issue_lag.p99", percentile(lags, 99, P99_MIN_SAMPLES), window=win,
                      notes=(clock_notes["issue_lag"],)))
    out.append(result("bmt.issue_lag.max", maximum(lags), window=win,
                      notes=(clock_notes["issue_lag"],)))

    if planned == 0:
        throughput = Computed(MetricStatus.EMPTY, None, 0, "no_planned_requests")
    else:
        throughput = per_second(counts["completed"], window.t_drain_end - t0, planned)
    out.append(result("bmt.throughput.completed_per_s", throughput, window=win,
                      notes=("configured max_inflight=1 throughput; not a maximum-throughput claim",
                             clock_notes["throughput_window"])))

    mode = bundle.run_spec["schedule"]["mode"]
    if mode == ReplayMode.ASAP.value:
        offered = Computed(MetricStatus.NOT_APPLICABLE, None, planned, "asap_has_no_offered_schedule")
    elif planned == 0:
        offered = Computed(MetricStatus.EMPTY, None, 0, "no_planned_requests")
    elif planned < 2:
        offered = Computed(MetricStatus.INVALID, None, planned, "insufficient_samples")
    else:
        offered = per_second(planned - 1, t_sched[-1] - t_sched[0], planned)
    out.append(result("bmt.offered_rate_per_s", offered, window=win, inputs=("events.jsonl", "run_spec.json")))

    deadline = params["deadline"]
    if deadline is None:
        miss = Computed(MetricStatus.NOT_APPLICABLE, None, planned, "no_deadline_parameter")
        notes: tuple[str, ...] = ()
    else:
        period = deadline["period_ns"]
        met = sum(1 for r, c in completed if c.t_complete <= r.t_sched + period)
        miss = rate(planned - met, planned)
        notes = (f"deadline_i = t_sched_i + {period} ns (policy {deadline['policy']}); evaluation parameter",
                 "miss = planned request not COMPLETED with t_complete <= deadline_i")
    out.append(result("bmt.deadline.miss_rate", miss, window=win, notes=notes))

    identity = output_identity(bundle.run_dir, bundle.run_spec["evaluation"]["output_reference"],
                               folded, bundle.input_items)
    out.append(result("bmt.output.identity_rate", identity.computed, window=win,
                      inputs=("events.jsonl", "outputs.bin", "run_spec.json", "input_items.jsonl"),
                      notes=identity.notes))
    assert [m.metric_id for m in out] == [m.metric_id for m in METRICS]
    return out


def evaluate_run(
    run_dir: Path | str,
    *,
    override: Optional[EvaluationOverride] = None,
    eval_id_factory: Callable[[], str] = new_eval_id,
    now_factory: Callable[[], str] = utc_now_iso,
) -> EvaluationOutcome:
    run_dir = Path(run_dir)
    # Integrity first: raw files must match the digests recorded at FINALIZE.
    try:
        recorded = read_json(run_dir / MANIFEST_FILE)["files"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EvaluationError(f"unreadable manifest: {exc}") from None
    problems = verify_raw_files(run_dir, recorded)
    if problems:
        raise EvaluationError("raw integrity violated: " + "; ".join(problems))
    try:
        bundle = load_run_bundle(run_dir)
    except BundleReadError as exc:
        raise EvaluationError(f"unreadable bundle: {exc}") from None
    raw = bundle.manifest["files"]
    manifest_record = file_record(run_dir / MANIFEST_FILE)

    eval_id = eval_id_factory()
    final_dir = run_dir / EVALUATIONS_DIR / eval_id
    if final_dir.exists():
        raise FileExistsError(f"evaluation {eval_id} already exists; evaluations are never overwritten")
    try:
        params, deadline_source = evaluation_parameters(bundle, override)
        metrics = compute_metrics(bundle, params, eval_id)
    except (KeyError, TypeError, SpecError) as exc:
        raise EvaluationError(f"malformed bundle content: {type(exc).__name__}: {exc}") from None
    metric_dicts = [to_jsonable(m) for m in metrics]
    content_digest = digest_of(canonical_metric_content(params, metric_dicts))

    # Raw evidence must be unchanged by evaluation; nothing is written otherwise.
    after = verify_raw_files(run_dir, raw)
    if after or file_record(run_dir / MANIFEST_FILE) != manifest_record:
        raise EvaluationError("raw files changed during evaluation: " + "; ".join(after))

    # Write into a staging directory and rename it into place, so an interrupted
    # evaluation never leaves a partial evaluations/<eval_id>/.
    staging = run_dir / EVALUATIONS_DIR / f".staging-{eval_id}"
    staging.mkdir(parents=True, exist_ok=False)
    kind = bundle_execution_kind(bundle)
    notice = bundle.manifest.get("mock_notice") if kind is ExecutionKind.MOCK else None
    write_json_exclusive(staging / METRICS_FILE, {
        "schema_version": EVAL_SCHEMA_VERSION,
        "benchmark_level": BENCHMARK_LEVEL,
        "execution_kind": kind,
        "mock_notice": notice,
        "eval_id": eval_id,
        "run_id": bundle.manifest["run_id"],
        "measurement_boundary_id": bundle.manifest["measurement_boundary"]["measurement_boundary_id"],
        "run_status": bundle.manifest["status"],
        "parameters": params,
        "metrics": metric_dicts,
    })
    write_json_exclusive(staging / EVAL_MANIFEST_FILE, {
        "schema_version": EVAL_SCHEMA_VERSION,
        "benchmark_level": BENCHMARK_LEVEL,
        "execution_kind": kind,
        "mock_notice": notice,
        "eval_id": eval_id,
        "run_id": bundle.manifest["run_id"],
        "created_at_utc": now_factory(),
        "evaluator": {
            "module": __name__,
            "code_identity": code_identity(),
            "catalog_version": CATALOG_VERSION,
            "definition_versions": {m.metric_id: m.definition_version for m in METRICS},
        },
        "protocol": {**params, "deadline_source": deadline_source},
        "protocol_digest": digest_of(params),
        "inputs": {MANIFEST_FILE: manifest_record, **raw},
        "raw_integrity": {"verified_before_evaluation": True, "verified_after_evaluation": True},
        "outputs": {METRICS_FILE: file_record(staging / METRICS_FILE)},
        "metrics_content_sha256": content_digest,
        "volatile_fields": list(EVAL_VOLATILE_FIELDS),
    })
    os.rename(staging, final_dir)
    return EvaluationOutcome(eval_id, final_dir, tuple(metrics), content_digest)
