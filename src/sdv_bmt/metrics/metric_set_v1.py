"""SDV-BMT Model-level Metric Set v1 (approved design update of 2026-10-08): catalog and computation.

Catalog ``sdv-bmt-model-metric-set-v1``; every value is computed by the Python evaluator from the
run bundle only (the native measurement plane records facts and computes no metric).  Five result
layers are kept apart: performance, timeliness (requirement satisfaction), task quality,
diagnostics, validity/evidence.  Guard results are structured validity results, never metrics.

Normative interpretation:

* Percentile latency is a distribution-characterization metric, not a worst-case timing guarantee.
* ``max_observed`` is the largest value observed in the measured population. It is not WCET.
* A deadline is an evaluation parameter; a deadline miss metric is not FTTI compliance.
* ``bmt.closed_loop_completion_rate`` is the observed configured-N closed-loop completion rate,
  not a maximum, peak or capacity claim, and never a periodic-rate requirement result.
* ``bmt.output.identity_rate`` is a determinism/corruption diagnostic, not an accuracy metric.

Shared rules with the legacy Stage 1A catalog (unchanged): nearest-rank percentiles; a p99 needs at
least 100 samples of its population, otherwise ``invalid(insufficient_samples)``; planned
percentiles treat every non-completed planned request as +inf and are ``censored`` (report-only,
``value=null``) when the rank lands on +inf; non-valid results never carry a number.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Optional

from ..contracts.ids import digest_of, sha256_hex
from ..contracts.quality import CITYSCAPES_SEMSEG_MIOU_V1, NUSCENES_DET3D_V1
from ..contracts.records import (METRIC_CATALOG_VERSION, METRIC_CATALOG_VERSION_V1, METRIC_CATALOG_VERSION_V1_2,
                                MetricSetResult)
from ..contracts.status import MetricLayer, MetricStatus, ReplayMode, RequestState
from ..observation.reader import RunBundle, read_output_blob
from .catalog import P99_MIN_SAMPLES, PERCENTILE_METHOD, PERCENTILES
from .identity import IdentityResult, output_identity
from .portable import INF, Computed, maximum, per_second, percentile, planned_percentile, rate, ratio
from .quality_detection import DetectionQuality, detection_quality
from .quality_det3d import Det3DQuality, det3d_quality, det3d_quality_parts
from .quality_segmentation import SegmentationQuality, segmentation_quality

# Metric Set v1 (frozen) and v1.1 (v0.1.2: v1 unchanged + the two segmentation quality metrics, approved 2026-11-06)
CATALOG_VERSION = METRIC_CATALOG_VERSION  # v1.1, the default catalog of new evaluations
CATALOG_VERSION_V1 = METRIC_CATALOG_VERSION_V1
CATALOG_VERSION_V1_2 = METRIC_CATALOG_VERSION_V1_2  # v1.1 + 3D detection quality (CenterPoint-Pillars)
S = RequestState
L = MetricLayer

PERCENTILE_NOTE = "percentile latency is a distribution-characterization metric, not a worst-case timing guarantee"
MAX_NOTE = "max_observed is the largest value observed in the measured population. It is not WCET."


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    layer: MetricLayer
    unit: str
    method: str
    definition_version: int = 1


# family -> (formula, population, layer, meaning)
LATENCY_FAMILIES: dict[str, tuple[str, str, MetricLayer, str]] = {
    "request_completion": ("t_complete - t_sched", "completed_requests", L.PERFORMANCE,
                           "request-completion latency"),
    "dut_service": ("t_complete - t_submit_begin", "completed_requests", L.PERFORMANCE,
                    "DUT-service latency: the submit call, the DUT/runtime work inside the declared measurement "
                    "boundary and the completion observation; not accelerator/device execution time"),
    "output_ready": ("t_output_ready - t_sched", "completed_requests", L.PERFORMANCE,
                     "output-ready latency: request completion plus the workload output stage"),
    "issue_lag": ("t_issue - t_sched", "issued_requests", L.DIAGNOSTICS,
                  "harness/replay realization (issuer fidelity); not accelerator latency"),
    "submit": ("t_submit_end - t_submit_begin", "completed_requests", L.DIAGNOSTICS,
               "host/runtime submission (enqueue) interval inside the DUT boundary"),
    "post_submit": ("t_complete - t_submit_end", "completed_requests", L.DIAGNOSTICS,
                    "end of submission to observed completion; may include device-start waiting, device execution, "
                    "transfer completion and the completion-observation delay; not pure device execution"),
    "output_stage": ("t_output_ready - t_complete", "completed_requests", L.DIAGNOSTICS,
                     "workload output stage after DUT completion (postprocess, codec, output write); may depend on "
                     "input content"),
}
# metric family -> portable device timing segment (device_observation.timing)
DEVICE_FAMILIES: dict[str, tuple[str, str]] = {
    "h2d": ("host_to_device", "device-clock duration of the declared host_to_device segment"),
    "device_execution_bracket": ("device_execution", "device-clock bracket of the declared device_execution "
                                 "segment; may include waiting for host enqueue; not device-active (busy) time"),
    "d2h": ("device_to_host", "device-clock duration of the declared device_to_host segment"),
}
COUNT_NAMES = ("planned", "issued", "completed", "dropped", "timed_out", "error", "not_issued")
RATE_NAMES = (("completion", "completed"), ("drop", "dropped"), ("timeout", "timed_out"), ("error", "error"),
              ("not_issued", "not_issued"))
COCO_METRICS = (("ap", "AP"), ("ap50", "AP50"), ("ap75", "AP75"), ("ap_small", "AP_small"),
                ("ap_medium", "AP_medium"), ("ap_large", "AP_large"), ("ar1", "AR1"), ("ar10", "AR10"),
                ("ar100", "AR100"))
COCO_MEANINGS = {
    "ap": "COCO AP averaged over IoU 0.50:0.05:0.95, area all, maxDets 100",
    "ap50": "COCO AP at IoU 0.50, area all, maxDets 100",
    "ap75": "COCO AP at IoU 0.75, area all, maxDets 100",
    "ap_small": "COCO AP (IoU 0.50:0.95) for ground truth area < 32^2, maxDets 100",
    "ap_medium": "COCO AP (IoU 0.50:0.95) for 32^2 <= area < 96^2, maxDets 100",
    "ap_large": "COCO AP (IoU 0.50:0.95) for area >= 96^2, maxDets 100",
    "ar1": "COCO AR (IoU 0.50:0.95, area all) with at most 1 detection per image and category",
    "ar10": "COCO AR (IoU 0.50:0.95, area all) with at most 10 detections per image and category",
    "ar100": "COCO AR (IoU 0.50:0.95, area all) with at most 100 detections per image and category",
}
THRESHOLD_METRICS = (
    ("precision", "ratio", "TP / (TP + FP) at the operating point"),
    ("recall", "ratio", "TP / (TP + FN) at the operating point (FN: unmatched non-crowd ground truth)"),
    ("f1", "ratio", "2 TP / (2 TP + FP + FN) at the operating point"),
    ("tp_count", "count", "detections matched to a non-crowd ground-truth box of the same category"),
    ("fp_count", "count", "detections at the operating point matched to no ground truth (incl. duplicates)"),
    ("fn_count", "count", "non-crowd ground-truth boxes matched by no detection"),
    ("fp_per_image", "1/image", "FP / evaluated images"),
    ("fn_per_gt", "ratio", "FN / non-crowd ground-truth boxes"),
)
CONDITIONAL = "quality.detection.conditional"


def _defs() -> list[MetricSpec]:
    out: list[MetricSpec] = []
    for family, (formula, population, layer, meaning) in LATENCY_FAMILIES.items():
        for p in PERCENTILES:
            out.append(MetricSpec(f"bmt.{family}.p{p}", layer, "ns",
                                  f"{PERCENTILE_METHOD}; values {formula} of {population}; {meaning}; p99 needs "
                                  f">= {P99_MIN_SAMPLES} samples, otherwise invalid(insufficient_samples)"))
        out.append(MetricSpec(f"bmt.{family}.max_observed", layer, "ns",
                              f"largest {formula} among {population}; {meaning}; {MAX_NOTE}"))
        if family == "request_completion":
            for p in PERCENTILES:
                out.append(MetricSpec(f"bmt.request_completion.planned.p{p}", L.PERFORMANCE, "ns",
                                      f"{PERCENTILE_METHOD}; values t_complete - t_sched of all planned requests, "
                                      "non-completed = +inf; p99 needs >= 100 completed samples; rank on +inf -> "
                                      "censored (report-only, value=null)"))
    for family, (segment, meaning) in DEVICE_FAMILIES.items():
        for p in PERCENTILES:
            out.append(MetricSpec(f"bmt.{family}.p{p}", L.DIAGNOSTICS, "ns",
                                  f"{PERCENTILE_METHOD}; values device_observation.timing.{segment} of COMPLETED "
                                  f"requests; {meaning}; device clock, never combined with host timestamps"))
        out.append(MetricSpec(f"bmt.{family}.max_observed", L.DIAGNOSTICS, "ns",
                              f"largest device_observation.timing.{segment} among COMPLETED requests; {meaning}; "
                              f"{MAX_NOTE}"))
    out += [
        MetricSpec("bmt.offered_rate", L.PERFORMANCE, "1/s",
                   "(planned - 1) / ((t_sched_last - t_sched_first) / 1e9) over the planned arrival times: the rate "
                   "the benchmark offered; fixed_rate and recorded replay; asap: not_applicable"),
        MetricSpec("bmt.achieved_completion_rate", L.PERFORMANCE, "1/s",
                   "count.completed / ((t_drain_end - t0) / 1e9), window [t0 = first planned t_sched, end of DRAIN] "
                   "(the denominator of the legacy bmt.throughput.completed_per_s); open-loop replay (fixed_rate, "
                   "recorded) only, asap: not_applicable"),
        MetricSpec("bmt.closed_loop_completion_rate", L.PERFORMANCE, "1/s",
                   "count.completed / ((t_drain_end - t0) / 1e9) of asap (closed-loop) replay at the configured N "
                   "(run.max_inflight); observed configured-N closed-loop completion rate, not a maximum, peak or "
                   "capacity claim; other replay modes: not_applicable"),
    ]
    out += [MetricSpec(f"bmt.count.{name}", L.PERFORMANCE, "count",
                       "count of measure-phase planned requests by state") for name in COUNT_NAMES]
    out.append(MetricSpec("bmt.late_completion_count", L.PERFORMANCE, "count",
                          "timed-out requests whose underlying call later returned (never counted as completed)"))
    out += [MetricSpec(f"bmt.{name}_rate", L.PERFORMANCE, "ratio", f"count.{src} / count.planned")
            for name, src in RATE_NAMES]
    out.append(MetricSpec("bmt.censored_fraction", L.PERFORMANCE, "ratio",
                          "(count.planned - count.completed) / count.planned: planned requests without a completed "
                          "value (+inf in the planned percentiles)"))
    for name, field in (("completion", "t_complete"), ("output_ready", "t_output_ready")):
        miss = (f"planned request not COMPLETED with {field} <= t_sched + deadline.period_ns (dropped, timed-out, "
                "errored and not-issued requests are misses)")
        out += [
            MetricSpec(f"bmt.deadline.{name}_miss_count", L.TIMELINESS, "count",
                       f"count of misses; miss = {miss}; deadline = evaluation parameter, not FTTI"),
            MetricSpec(f"bmt.deadline.{name}_miss_rate", L.TIMELINESS, "ratio",
                       f"misses / count.planned; miss = {miss}; deadline = evaluation parameter, not FTTI"),
        ]
    out += [
        MetricSpec("bmt.required_rate", L.TIMELINESS, "1/s",
                   "externally supplied required rate (evaluation parameter required_rate_hz; an experiment/"
                   "evaluation requirement, not a hardware property); not_applicable when not supplied"),
        MetricSpec("bmt.required_rate_met", L.TIMELINESS, "boolean",
                   "fixed_rate replay with a supplied required rate: 1 iff offered_rate >= required_rate and "
                   "achieved_completion_rate >= required_rate, else 0; not_applicable when offered_rate < "
                   "required_rate (the run did not offer the load to test it), for asap/closed-loop and recorded "
                   "replay, and without a required rate; drops/timeouts/errors are judged by their own metrics"),
    ]
    out.append(MetricSpec("bmt.output.identity_rate", L.DIAGNOSTICS, "ratio",
                          "COMPLETED outputs whose sha256 equals the designated reference for the same item, / "
                          "COMPLETED outputs that have a reference entry; determinism/corruption diagnostic, not "
                          "accuracy"))
    out.append(MetricSpec("quality.detection.coverage", L.TASK_QUALITY, "ratio",
                          "quality-evaluated planned items / planned requests (items of COMPLETED requests with a "
                          "valid task output and a frozen ground truth)"))
    for name, _ in COCO_METRICS:
        out.append(MetricSpec(f"{CONDITIONAL}.{name}", L.TASK_QUALITY, "ratio",
                              f"{COCO_MEANINGS[name]}; conditional population (items of COMPLETED requests with a "
                              "valid task output); ground truth frozen in the bundle"))
    for name, unit, meaning in THRESHOLD_METRICS:
        out.append(MetricSpec(f"{CONDITIONAL}.{name}", L.TASK_QUALITY, unit,
                              f"{meaning}; conditional population; operating point (score/IoU threshold, class "
                              "matching, max detections, duplicate and crowd policy) in the evaluation protocol"))
    out += [
        MetricSpec("data.item_count", L.VALIDITY_EVIDENCE, "count", "number of resolved input items (input_items.jsonl)"),
        MetricSpec("data.annotation_coverage", L.VALIDITY_EVIDENCE, "ratio",
                   "input items with an annotation reference / input items"),
        MetricSpec("data.integrity.payload_verified_rate", L.VALIDITY_EVIDENCE, "ratio",
                   "input items whose payload identity INPUT.open re-verified (index SHA-256, items digest, "
                   "file payload SHA-256) / input items; from the observed input_payload evidence"),
        MetricSpec("data.integrity.prepared_input_verified_rate", L.VALIDITY_EVIDENCE, "ratio",
                   "planned items whose prepared input the native measurement plane staged with its SHA-256 "
                   "re-verified / planned items; a recorded runtime check (per-item prepared digests are not in the "
                   "bundle)"),
        MetricSpec("data.integrity.annotation_verified_rate", L.VALIDITY_EVIDENCE, "ratio",
                   "input items whose frozen ground truth (quality_reference.jsonl) re-verifies against the item's "
                   "annotation_digest / input items; recomputed from the bundle"),
        MetricSpec("data.traceability.complete", L.VALIDITY_EVIDENCE, "boolean",
                   "1 iff every COMPLETED request traces to exactly one input item (payload and annotation digests) "
                   "and to an output whose bytes re-verify against its output_ref (and, with a frozen ground truth, "
                   "to a verified annotation), else 0"),
    ]
    return out


METRICS: tuple[MetricSpec, ...] = tuple(_defs())  # Metric Set v1 (frozen: 91 metrics)
SEGMENTATION_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("quality.segmentation.coverage", L.TASK_QUALITY, "ratio",
               "segmentation-evaluated planned items / planned requests (items of COMPLETED requests with a valid "
               "task output and a frozen ground truth)"),
    MetricSpec("quality.segmentation.conditional.miou", L.TASK_QUALITY, "ratio",
               "Cityscapes 19-class mean IoU: per class TP / (TP + FP + FN) with counts summed over the evaluated "
               "items, ground-truth pixels 255 not counted, mean over the classes with a non-empty union; "
               "conditional population (items of COMPLETED requests with a valid task output); ground truth frozen "
               "in the bundle"),
)
METRICS_V1_1: tuple[MetricSpec, ...] = METRICS + SEGMENTATION_METRICS
_DET3D_POP = ("conditional population (items of COMPLETED requests with a valid task output); ground truth frozen in "
              "the bundle; nuScenes devkit detection_cvpr_2019 definition")
DET3D_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("quality.det3d.coverage", L.TASK_QUALITY, "ratio",
               "3D-detection-evaluated planned items / planned requests (items of COMPLETED requests with a valid task "
               "output and a frozen ground truth)"),
    MetricSpec("quality.det3d.conditional.map", L.TASK_QUALITY, "ratio",
               "nuScenes mean average precision: per class the mean over center-distance thresholds 0.5/1/2/4 m of the "
               "AP (precision above 0.1 for recall above 0.1), mean over the 10 classes; " + _DET3D_POP),
    MetricSpec("quality.det3d.conditional.nds", L.TASK_QUALITY, "ratio",
               "nuScenes detection score: (5 mAP + sum of max(0, 1 - mTP error) over the 5 true-positive errors) / 10; "
               + _DET3D_POP),
    MetricSpec("quality.det3d.conditional.mate", L.TASK_QUALITY, "m",
               "mean translation error (planar center distance) of true positives at 2 m, mean over classes; " + _DET3D_POP),
    MetricSpec("quality.det3d.conditional.mase", L.TASK_QUALITY, "ratio",
               "mean scale error (1 - aligned 3D IoU) of true positives at 2 m, mean over classes; " + _DET3D_POP),
    MetricSpec("quality.det3d.conditional.maoe", L.TASK_QUALITY, "rad",
               "mean orientation error (yaw difference; period pi for barrier) of true positives at 2 m, mean over the "
               "classes that define it; " + _DET3D_POP),
    MetricSpec("quality.det3d.conditional.mave", L.TASK_QUALITY, "m/s",
               "mean planar velocity error of true positives at 2 m, mean over the classes that define it; " + _DET3D_POP),
    MetricSpec("quality.det3d.conditional.maae", L.TASK_QUALITY, "ratio",
               "mean attribute error (1 - attribute accuracy) of true positives at 2 m, mean over the classes that "
               "define it; " + _DET3D_POP),
)
METRICS_V1_2: tuple[MetricSpec, ...] = METRICS_V1_1 + DET3D_METRICS
CATALOG_METRICS = {CATALOG_VERSION_V1: METRICS, CATALOG_VERSION: METRICS_V1_1, CATALOG_VERSION_V1_2: METRICS_V1_2}
BY_ID = {m.metric_id: m for m in METRICS_V1_2}
assert len(BY_ID) == len(METRICS_V1_2)


def _requests_population(name: str, sample: int, counts: dict[str, int]) -> dict[str, Any]:
    planned = counts["planned"]
    return {"name": name, "sample_count": sample, "planned_count": planned, "completed_count": counts["completed"],
            "coverage": ratio(sample, planned) if planned else None}


@dataclass(frozen=True)
class MeasuredPart:
    """One measured run: a single run (label None) or one part of a run series (label "part-NN")."""
    bundle: RunBundle
    folded: dict
    counts: dict[str, int]
    window: Any
    label: Optional[str] = None


def compute_metric_set_v1(bundle: RunBundle, folded: dict, counts: dict[str, int], window: Any,
                          params: dict[str, Any], eval_id: str, kind: Any, boundary_id: str,
                          clock_notes: dict[str, str], catalog: str = CATALOG_VERSION_V1) -> list[MetricSetResult]:
    """All Metric Set v1 (or v1.1) results of a bundle whose measurement is valid (run-level gates are applied by
    the evaluator before this is called).  ``catalog`` v1 computes exactly the frozen v1 results; v1.1 adds the
    segmentation quality metrics, and a segmentation evaluator makes the detection quality metrics not_applicable."""
    return compute_metric_set_parts([MeasuredPart(bundle, folded, counts, window)], params, eval_id, kind,
                                    boundary_id, clock_notes, catalog)


def _series_quality_stub(cls: Any, evaluator: Optional[str]) -> Any:
    return cls(MetricStatus.NOT_SUPPORTED, f"series_quality_not_implemented:{evaluator}", (
        "the minimal run series evaluates the quality of the nuScenes 3D detection evaluator only",))


def _reference_check(parts: list[MeasuredPart]) -> Optional[tuple[int, int]]:
    """(verified rows, input items) of the frozen ground truth over all parts; None when a part has none."""
    verified = total = 0
    for p in parts:
        rows = p.bundle.quality_reference
        if rows is None:
            return None
        by_index = {item.item_index: item for item in p.bundle.input_items}
        verified += sum(1 for row in rows if (item := by_index.get(row.item_index)) is not None
                        and row.source_item_id == item.source_item_id
                        and row.annotation_digest == item.annotation_digest
                        and digest_of(row.annotation) == row.annotation_digest)
        total += len(p.bundle.input_items)
    return verified, total


def compute_metric_set_parts(parts: list[MeasuredPart], params: dict[str, Any], eval_id: str, kind: Any,
                             boundary_id: str, clock_notes: dict[str, str],
                             catalog: str = CATALOG_VERSION_V1) -> list[MetricSetResult]:
    """Metric Set results of one run (one part) or of a run series (its parts, all measurement-valid and of one
    identity, checked by the caller).  A series is one population: counts are summed, per-request distributions
    are taken over the requests of all parts, rates divide by the sum of the parts' windows (the time between parts
    is not part of any window), and task quality is evaluated once over the items of all parts."""
    single = len(parts) == 1
    first = parts[0].bundle
    counts = parts[0].counts if single else {k: sum(p.counts[k] for p in parts) for k in parts[0].counts}
    planned = counts["planned"]
    entries = [e for p in parts for e in p.folded.values()]

    def part_t_sched(p: MeasuredPart) -> list[int]:
        return [r.t_sched for r, _ in p.folded.values()]

    def part_t0(p: MeasuredPart) -> int:
        ts = part_t_sched(p)
        return ts[0] if ts else p.window.t_measure_begin

    def part_win(p: MeasuredPart) -> dict[str, Any]:
        return {"t_measure_begin_ns": p.window.t_measure_begin, "t0_ns": part_t0(p),
                "t_drain_end_ns": p.window.t_drain_end, "drain_cap_reached": p.window.drain_cap_reached}

    window_ns = sum(p.window.t_drain_end - part_t0(p) for p in parts)
    offered_n = sum(max(len(part_t_sched(p)) - 1, 0) for p in parts)
    offered_span = sum((ts[-1] - ts[0]) if len(ts) > 1 else 0 for ts in map(part_t_sched, parts))
    if single:
        win: dict[str, Any] = part_win(parts[0])
    else:
        win = {"series_parts": [{"part": p.label, **part_win(p)} for p in parts], "window_sum_ns": window_ns}
    completed = [(r, c) for r, c in entries if r.state is S.COMPLETED and c is not None]
    issued = [r for r, _ in entries if r.admission == "issued"]
    out: list[MetricSetResult] = []

    def emit(metric_id: str, c: Computed, population: dict[str, Any], *, inputs: tuple[str, ...] = ("events.jsonl",),
             notes: tuple[str, ...] = (), window_: Optional[dict[str, Any]] = win) -> None:
        spec = BY_ID[metric_id]
        out.append(MetricSetResult(
            metric_id=metric_id, layer=spec.layer, definition_version=spec.definition_version,
            measurement_boundary_id=boundary_id, unit=spec.unit, status=c.status, value=c.value,
            population=population, window=window_, method=spec.method, inputs=inputs, notes=notes,
            eval_id=eval_id, execution_kind=kind, status_reason=c.reason, censored_fraction=c.censored_fraction))

    # ---- latency families (host run clock)
    values = {
        "request_completion": [c.t_complete - r.t_sched for r, c in completed],
        "dut_service": [c.t_complete - r.t_submit_begin for r, c in completed],
        "output_ready": [c.t_output_ready - r.t_sched for r, c in completed],
        "issue_lag": [r.t_issue - r.t_sched for r in issued],
        "submit": [r.t_submit_end - r.t_submit_begin for r, c in completed],
        "post_submit": [c.t_complete - r.t_submit_end for r, c in completed],
        "output_stage": [c.t_output_ready - c.t_complete for r, c in completed],
    }
    family_notes = {
        "request_completion": ("completed-only; interpret with bmt.completion_rate and bmt.censored_fraction",),
        "dut_service": (clock_notes["dut_service"],),
        "output_ready": (clock_notes["output_ready"],),
        "issue_lag": (clock_notes["issue_lag"],),
    }
    planned_rc = [(c.t_complete - r.t_sched) if (r.state is S.COMPLETED and c is not None) else INF
                  for r, c in entries]
    for family, (_formula, population, _layer, _meaning) in LATENCY_FAMILIES.items():
        vals = values[family]
        pop = _requests_population(population, len(vals), counts)
        notes = family_notes.get(family, ())
        for p in PERCENTILES:
            emit(f"bmt.{family}.p{p}", percentile(vals, p, P99_MIN_SAMPLES if p == 99 else 1), pop,
                 notes=notes + (PERCENTILE_NOTE,))
        emit(f"bmt.{family}.max_observed", maximum(vals), pop, notes=notes + (MAX_NOTE,))
        if family == "request_completion":
            planned_pop = _requests_population("planned_requests", planned, counts)
            for p in PERCENTILES:
                emit(f"bmt.request_completion.planned.p{p}",
                     planned_percentile(planned_rc, p, P99_MIN_SAMPLES if p == 99 else 0), planned_pop,
                     notes=("non-completed planned requests are +inf; censored results are report-only and carry "
                            "value=null", "p99 requires at least 100 completed samples"))

    # ---- device-domain segments (device_observation events of COMPLETED requests)
    device_timing = first.manifest["measurement_boundary"]["definition"].get("device_timing")
    declared = tuple((device_timing or {}).get("segments") or ())
    observations: list[dict[int, list[dict[str, Any]]]] = []
    for part in parts:
        obs_part: dict[int, list[dict[str, Any]]] = {}
        for e in part.bundle.events:
            if e["kind"] == "device_observation":
                obs_part.setdefault(e["request_id"], []).append(e)
        observations.append(obs_part)
    for family, (segment, _meaning) in DEVICE_FAMILIES.items():
        ids = [f"bmt.{family}.p{p}" for p in PERCENTILES] + [f"bmt.{family}.max_observed"]
        if segment not in declared:
            reason = "no_device_timing_in_measurement_boundary" if device_timing is None else "segment_not_declared"
            for metric_id in ids:
                emit(metric_id, Computed(MetricStatus.NOT_SUPPORTED, None, 0, reason),
                     _requests_population("completed_requests_with_device_timing", 0, counts))
            continue
        seg_values: list[int] = []
        problem = None
        for part, obs_part in zip(parts, observations):
            where = "" if part.label is None else f"{part.label} "
            for r, c in part.folded.values():
                if r.state is not S.COMPLETED or c is None:
                    continue
                obs = obs_part.get(r.request_id, [])
                timing = obs[0].get("timing") if len(obs) == 1 else None
                value = timing.get(segment) if isinstance(timing, dict) else None
                if type(value) is not int or value < 0:
                    problem = f"{where}request {r.request_id}: no single device_observation with timing.{segment}"
                    break
                seg_values.append(value)
            if problem:
                break
        method = (device_timing or {}).get("method")
        pop = _requests_population("completed_requests_with_device_timing", len(seg_values), counts)
        pop["timing_method"] = method
        pop["clock"] = (device_timing or {}).get("clock")
        notes = (f"timing method {method}; device clock: never subtracted from or combined with host run-clock "
                 "timestamps",)
        for p in PERCENTILES:
            c = (Computed(MetricStatus.INVALID, None, len(seg_values), "device_observation_missing") if problem
                 else percentile(seg_values, p, P99_MIN_SAMPLES if p == 99 else 1))
            emit(f"bmt.{family}.p{p}", c, pop, notes=notes + ((problem,) if problem else (PERCENTILE_NOTE,)))
        c = (Computed(MetricStatus.INVALID, None, len(seg_values), "device_observation_missing") if problem
             else maximum(seg_values))
        emit(f"bmt.{family}.max_observed", c, pop, notes=notes + ((problem,) if problem else (MAX_NOTE,)))

    # ---- rates
    mode = first.run_spec["schedule"]["mode"]
    configured_n = first.run_spec["resolved_plan"]["run"]["max_inflight"]
    rate_pop = {"name": "planned_requests", "sample_count": planned, "planned_count": planned,
                "completed_count": counts["completed"], "coverage": 1.0 if planned else None,
                "replay_mode": mode, "configured_n": configured_n}
    if mode == ReplayMode.ASAP.value:
        offered = Computed(MetricStatus.NOT_APPLICABLE, None, planned, "asap_has_no_offered_schedule")
    elif planned == 0:
        offered = Computed(MetricStatus.EMPTY, None, 0, "no_planned_requests")
    elif offered_n < 1:
        offered = Computed(MetricStatus.INVALID, None, planned, "insufficient_samples")
    else:
        offered = per_second(offered_n, offered_span, planned)
    emit("bmt.offered_rate", offered, rate_pop, inputs=("events.jsonl", "run_spec.json"))
    if planned == 0:
        window_rate = Computed(MetricStatus.EMPTY, None, 0, "no_planned_requests")
    else:
        window_rate = per_second(counts["completed"], window_ns, planned)
    rate_notes = (clock_notes["throughput_window"],)
    if not single:
        rate_notes += ("run series: counts over the sum of the parts' windows and offered spans; the time between "
                       "parts is not part of any window",)
    if mode == ReplayMode.ASAP.value:
        achieved = Computed(MetricStatus.NOT_APPLICABLE, None, planned, "closed_loop_replay")
        closed = window_rate
    else:
        achieved = window_rate
        closed = Computed(MetricStatus.NOT_APPLICABLE, None, planned, f"open_loop_replay:{mode}")
    emit("bmt.achieved_completion_rate", achieved, rate_pop, inputs=("events.jsonl", "run_spec.json"),
         notes=rate_notes)
    emit("bmt.closed_loop_completion_rate", closed, rate_pop, inputs=("events.jsonl", "run_spec.json"),
         notes=rate_notes + (f"configured N = {configured_n}; not a maximum, peak or capacity claim and not a "
                             "periodic-rate requirement result",))

    # ---- counts and availability
    count_pop = _requests_population("planned_requests", planned, counts)
    for name in COUNT_NAMES:
        emit(f"bmt.count.{name}", Computed(MetricStatus.VALID, counts[name], planned), count_pop)
    emit("bmt.late_completion_count", Computed(MetricStatus.VALID, counts["late_completion"], counts["timed_out"]),
         _requests_population("timed_out_requests", counts["timed_out"], counts),
         notes=("late completions never change the count identity",))
    for name, source in RATE_NAMES:
        emit(f"bmt.{name}_rate", rate(counts[source], planned), count_pop)
    emit("bmt.censored_fraction", rate(planned - counts["completed"], planned), count_pop)

    # ---- timeliness: deadline views and the required rate
    deadline = params["deadline"]
    for name, field in (("completion", "t_complete"), ("output_ready", "t_output_ready")):
        if deadline is None:
            miss_count = miss_rate = Computed(MetricStatus.NOT_APPLICABLE, None, planned, "no_deadline_parameter")
            notes: tuple[str, ...] = ()
        else:
            period = deadline["period_ns"]
            met = sum(1 for r, c in completed if getattr(c, field) <= r.t_sched + period)
            miss_count = Computed(MetricStatus.VALID, planned - met, planned) if planned else Computed(
                MetricStatus.EMPTY, None, 0, "no_planned_requests")
            miss_rate = rate(planned - met, planned)
            notes = (f"deadline_i = t_sched_i + {period} ns (policy {deadline['policy']}); evaluation parameter, "
                     "not FTTI", f"miss = planned request not COMPLETED with {field} <= deadline_i")
        emit(f"bmt.deadline.{name}_miss_count", miss_count, count_pop, notes=notes)
        emit(f"bmt.deadline.{name}_miss_rate", miss_rate, count_pop, notes=notes)
    required = params["required_rate_hz"]
    if required is None:
        req = met_c = Computed(MetricStatus.NOT_APPLICABLE, None, planned, "no_required_rate_parameter")
    else:
        req = Computed(MetricStatus.VALID, required, planned)
        met_c = _required_rate_met(mode, required, offered, achieved, planned, offered_n, offered_span,
                                   counts["completed"], window_ns)
    emit("bmt.required_rate", req, rate_pop, inputs=("evaluation parameter required_rate_hz",), window_=None)
    emit("bmt.required_rate_met", met_c, rate_pop,
         inputs=("events.jsonl", "run_spec.json", "evaluation parameter required_rate_hz"),
         notes=("requirement-satisfaction view of the offered and achieved rates; drops, timeouts and errors are "
                "judged by their own metrics",))

    # ---- diagnostics: output identity
    reference = first.run_spec["evaluation"]["output_reference"]
    if single:
        identity = output_identity(first.run_dir, reference, parts[0].folded, first.input_items)
    elif reference.get("kind") == "none":
        identity = output_identity(first.run_dir, reference, {}, ())
    else:
        identity = IdentityResult(Computed(MetricStatus.NOT_SUPPORTED, None, 0, "output_reference_in_run_series"),
                                  ("an output reference is not supported in a run series",))
    emit("bmt.output.identity_rate", identity.computed,
         _requests_population("completed_requests_with_reference_entry", identity.computed.n, counts),
         inputs=("events.jsonl", "outputs.bin", "run_spec.json", "input_items.jsonl"),
         notes=identity.notes + ("determinism/corruption diagnostic against a reference designated before the run; "
                                 "not evidence of task correctness",))

    # ---- task quality (bundle-only, conditional population)
    protocol = params["quality"]
    evaluator = None if protocol is None else protocol.get("evaluator")
    v11_plus = catalog in (CATALOG_VERSION, CATALOG_VERSION_V1_2)
    segmentation = (v11_plus and protocol is not None and evaluator == CITYSCAPES_SEMSEG_MIOU_V1)
    is_det3d = (catalog == CATALOG_VERSION_V1_2 and protocol is not None and evaluator == NUSCENES_DET3D_V1)
    if single:
        bundle, folded = first, parts[0].folded
        seg = segmentation_quality(bundle, folded, protocol) if v11_plus else None
        d3 = det3d_quality(bundle, folded, protocol) if catalog == CATALOG_VERSION_V1_2 else None
    else:
        seg = None
        if v11_plus:
            seg = (_series_quality_stub(SegmentationQuality, evaluator) if segmentation
                   else segmentation_quality(first, parts[0].folded, protocol))
        d3 = (det3d_quality_parts([(p.bundle, p.folded, p.label) for p in parts], protocol)
              if catalog == CATALOG_VERSION_V1_2 else None)
    if segmentation:
        assert seg is not None
        quality = DetectionQuality(MetricStatus.NOT_APPLICABLE, f"quality_evaluator_is:{CITYSCAPES_SEMSEG_MIOU_V1}",
                                   annotation_check=seg.annotation_check)
    elif is_det3d:
        assert d3 is not None
        quality = DetectionQuality(MetricStatus.NOT_APPLICABLE, f"quality_evaluator_is:{NUSCENES_DET3D_V1}",
                                   annotation_check=d3.annotation_check)
    elif single or evaluator is None:
        quality = detection_quality(first, parts[0].folded, protocol)
    else:
        quality = _series_quality_stub(DetectionQuality, evaluator)
    if not single:  # the frozen-ground-truth check always covers every part
        quality = dataclasses.replace(quality, annotation_check=_reference_check(parts))
    _emit_quality(emit, quality, counts)

    # ---- validity / evidence: data integrity and traceability
    _emit_data(emit, parts, counts, quality)
    if seg is not None:  # v1.1: no evaluator -> not_collected; another evaluator -> not_applicable
        _emit_segmentation(emit, seg, counts)
    if d3 is not None:  # v1.2: likewise for the 3D detection quality
        _emit_det3d(emit, d3, counts)
    assert [m.metric_id for m in out] == [m.metric_id for m in CATALOG_METRICS[catalog]]
    return out


def _emit_segmentation(emit: Any, seg: SegmentationQuality, counts: dict[str, int]) -> None:
    planned = counts["planned"]
    base = {"name": "quality_evaluated_items", "sample_count": seg.evaluated_items, "planned_count": planned,
            "completed_count": counts["completed"],
            "coverage": ratio(seg.evaluated_items, planned) if planned else None, **seg.population}
    inputs = ("events.jsonl", "outputs.bin", "input_items.jsonl", "quality_reference.jsonl", "manifest.json")
    if seg.status in (MetricStatus.VALID, MetricStatus.EMPTY):
        cov = rate(seg.evaluated_items, planned)
    else:
        cov = Computed(seg.status, None, 0, seg.reason)
    emit("quality.segmentation.coverage", cov, base, inputs=inputs, notes=seg.notes)
    if seg.status is MetricStatus.VALID:
        miou = Computed(MetricStatus.VALID, seg.miou, seg.evaluated_items)
    else:
        miou = Computed(seg.status, None, seg.evaluated_items, seg.reason)
    emit("quality.segmentation.conditional.miou", miou, base, inputs=inputs, notes=seg.notes)


def _emit_det3d(emit: Any, d3: Det3DQuality, counts: dict[str, int]) -> None:
    planned = counts["planned"]
    base = {"name": "quality_evaluated_items", "sample_count": d3.evaluated_items, "planned_count": planned,
            "completed_count": counts["completed"],
            "coverage": ratio(d3.evaluated_items, planned) if planned else None, **d3.population}
    inputs = ("events.jsonl", "outputs.bin", "input_items.jsonl", "quality_reference.jsonl", "manifest.json")
    if d3.status in (MetricStatus.VALID, MetricStatus.EMPTY):
        cov = rate(d3.evaluated_items, planned)
    else:
        cov = Computed(d3.status, None, 0, d3.reason)
    emit("quality.det3d.coverage", cov, base, inputs=inputs, notes=d3.notes)
    values = {}
    if d3.status is MetricStatus.VALID and d3.result is not None:
        r = d3.result
        values = {"map": r["mean_ap"], "nds": r["nd_score"], "mate": r["tp_errors"]["trans_err"],
                  "mase": r["tp_errors"]["scale_err"], "maoe": r["tp_errors"]["orient_err"],
                  "mave": r["tp_errors"]["vel_err"], "maae": r["tp_errors"]["attr_err"]}
    class_note = ()
    if d3.result is not None:
        class_note = ("per-class mean AP over the distance thresholds: " + ", ".join(
            f"{c} {v:.6f}" for c, v in d3.result["mean_dist_aps"].items()),)
    for key in ("map", "nds", "mate", "mase", "maoe", "mave", "maae"):
        if key in values:
            c = Computed(MetricStatus.VALID, values[key], d3.evaluated_items)
        else:
            c = Computed(d3.status, None, d3.evaluated_items, d3.reason)
        emit(f"quality.det3d.conditional.{key}", c, base, inputs=inputs, notes=d3.notes + class_note)


def _required_rate_met(mode: str, required: float, offered: Computed, achieved: Computed, planned: int,
                       offered_n: int, offered_span_ns: int, completed: int, window_ns: int) -> Computed:
    """User rule of 2026-10-08: only a fixed-rate schedule tests a periodic-rate requirement."""
    if mode != ReplayMode.FIXED_RATE.value:
        reason = "closed_loop_replay" if mode == ReplayMode.ASAP.value else f"no_periodic_offered_rate:{mode}"
        return Computed(MetricStatus.NOT_APPLICABLE, None, planned, reason)
    if offered.status is not MetricStatus.VALID or achieved.status is not MetricStatus.VALID:
        return Computed(MetricStatus.INVALID, None, planned, "offered_or_achieved_rate_not_valid")
    req = Fraction(required)
    offered_exact = Fraction(offered_n * 1_000_000_000, offered_span_ns)
    achieved_exact = Fraction(completed * 1_000_000_000, window_ns)
    if offered_exact < req:
        return Computed(MetricStatus.NOT_APPLICABLE, None, planned, "offered_rate_below_required_rate")
    return Computed(MetricStatus.VALID, 1 if achieved_exact >= req else 0, planned)


def _emit_quality(emit: Any, quality: DetectionQuality, counts: dict[str, int]) -> None:
    planned = counts["planned"]
    base = {"name": "quality_evaluated_items", "sample_count": quality.evaluated_items, "planned_count": planned,
            "completed_count": counts["completed"],
            "coverage": ratio(quality.evaluated_items, planned) if planned else None, **quality.population}
    inputs = ("events.jsonl", "outputs.bin", "input_items.jsonl", "quality_reference.jsonl", "manifest.json")
    if quality.status in (MetricStatus.VALID, MetricStatus.EMPTY):
        cov = rate(quality.evaluated_items, planned)
    else:
        cov = Computed(quality.status, None, 0, quality.reason)
    emit("quality.detection.coverage", cov, base, inputs=inputs, notes=quality.notes)
    ids = [f"{CONDITIONAL}.{n}" for n, _ in COCO_METRICS] + [f"{CONDITIONAL}.{n}" for n, _u, _m in THRESHOLD_METRICS]
    if quality.status is not MetricStatus.VALID:
        for metric_id in ids:
            emit(metric_id, Computed(quality.status, None, quality.evaluated_items, quality.reason), base,
                 inputs=inputs, notes=quality.notes)
        return
    for name, stat in COCO_METRICS:
        value = quality.stats[stat]
        c = (Computed(MetricStatus.VALID, value, quality.evaluated_items) if value > -1
             else Computed(MetricStatus.EMPTY, None, quality.evaluated_items, "no_ground_truth_in_scope"))
        emit(f"{CONDITIONAL}.{name}", c, base, inputs=inputs, notes=quality.notes)
    k = quality.counts
    tp, fp, fn, gt = k["tp"], k["fp"], k["fn"], k["gt_non_crowd"]
    images = quality.evaluated_items
    point_pop = {**base, "detections_at_operating_point": k["detections_considered"],
                 "ignored_crowd_matches": k["ignored_crowd_matches"], "gt_non_crowd": gt}

    def frac(num: int, den: int, reason: str) -> Computed:
        return Computed(MetricStatus.VALID, ratio(num, den), images) if den else Computed(
            MetricStatus.EMPTY, None, images, reason)

    values = {
        "precision": frac(tp, tp + fp, "no_detections_at_operating_point"),
        "recall": frac(tp, tp + fn, "no_ground_truth"),
        "f1": frac(2 * tp, 2 * tp + fp + fn, "no_detections_and_no_ground_truth"),
        "tp_count": Computed(MetricStatus.VALID, tp, images),
        "fp_count": Computed(MetricStatus.VALID, fp, images),
        "fn_count": Computed(MetricStatus.VALID, fn, images),
        "fp_per_image": frac(fp, images, "no_evaluated_items"),
        "fn_per_gt": frac(fn, gt, "no_ground_truth"),
    }
    for name, _unit, _meaning in THRESHOLD_METRICS:
        emit(f"{CONDITIONAL}.{name}", values[name], point_pop, inputs=inputs, notes=quality.notes)


def _emit_data(emit: Any, parts: list[MeasuredPart], counts: dict[str, int], quality: DetectionQuality) -> None:
    single = len(parts) == 1
    items = [item for p in parts for item in p.bundle.input_items]
    n = len(items)
    item_pop = {"name": "input_items", "sample_count": n, "planned_count": counts["planned"]}
    emit("data.item_count", Computed(MetricStatus.VALID, n, n), item_pop, inputs=("input_items.jsonl",), window_=None)
    emit("data.annotation_coverage", rate(sum(1 for i in items if i.annotation_ref is not None), n), item_pop,
         inputs=("input_items.jsonl",), window_=None)
    payload_ok = True
    failed = 0
    prepared_ok = True
    for p in parts:
        evidence = {e["condition"]: e for e in p.bundle.manifest["realization"]["evidence"]}
        payload = evidence.get("input_payload", {})
        payload_ok = payload_ok and (payload.get("validity") == "observed" and (payload.get("value") or {}).get(
            "items_digest") == p.bundle.manifest["input"]["items_digest"])
        preparation = evidence.get("input_preparation", {})
        native = (p.bundle.manifest.get("realtime") or {}).get("measurement_plane") == "native"
        if native and preparation.get("validity") in ("observed", "virtual") and isinstance(preparation.get("value"),
                                                                                              dict):
            failed += len(preparation["value"].get("failed_items", []))
        else:
            prepared_ok = False
    if payload_ok:
        verified = Computed(MetricStatus.VALID, 1.0, n) if n else Computed(MetricStatus.EMPTY, None, 0, "no_items")
    else:
        verified = Computed(MetricStatus.NOT_COLLECTED, None, 0, "payload_verification_not_recorded")
    emit("data.integrity.payload_verified_rate", verified, item_pop, inputs=("manifest.json",), window_=None,
         notes=("INPUT.open re-reads the digest-pinned index and every file payload and fails the run on any "
                "mismatch; recorded as observed input_payload evidence",))
    if prepared_ok:
        prepared = rate(counts["planned"] - failed, counts["planned"])
    else:
        prepared = Computed(MetricStatus.NOT_COLLECTED, None, 0, "no_prepared_input_verification_recorded")
    emit("data.integrity.prepared_input_verified_rate", prepared,
         _requests_population("planned_requests", counts["planned"], counts), inputs=("manifest.json",),
         window_=None, notes=("runtime check of the native staging (SHA-256 of every staged input re-verified); the "
                              "per-item prepared digests are not recorded in the bundle",))
    if quality.annotation_check is None:
        annotation = Computed(MetricStatus.NOT_COLLECTED, None, 0, "annotation_content_not_in_bundle")
    else:
        annotation = rate(*quality.annotation_check)
    emit("data.integrity.annotation_verified_rate", annotation, item_pop,
         inputs=("input_items.jsonl", "quality_reference.jsonl"), window_=None)
    complete = True
    problems: list[str] = []
    reference_ok = quality.annotation_check is None or quality.annotation_check[0] == quality.annotation_check[1]
    for p in parts:
        n_part = len(p.bundle.input_items)
        where = "" if single else f"{p.label} "
        for request, completion in p.folded.values():
            if request.state is not S.COMPLETED or completion is None:
                continue
            ref = completion.output_ref
            if len(request.item_indices) != 1 or not 0 <= request.item_indices[0] < n_part or ref is None:
                complete = False
                problems.append(f"{where}request {request.request_id}: no single input item or no output reference")
                continue
            blob = read_output_blob(p.bundle.run_dir, ref)
            if len(blob) != ref.length or sha256_hex(blob) != ref.sha256:
                complete = False
                problems.append(f"{where}request {request.request_id}: output bytes do not re-verify")
    if not reference_ok:
        complete = False
        problems.append("frozen ground truth does not re-verify against the items' annotation digests")
    emit("data.traceability.complete", Computed(MetricStatus.VALID, 1 if complete else 0, counts["completed"]),
         _requests_population("completed_requests", counts["completed"], counts),
         inputs=("events.jsonl", "outputs.bin", "input_items.jsonl") + (
             ("quality_reference.jsonl",) if quality.annotation_check is not None else ()),
         notes=tuple(problems[:5]))
