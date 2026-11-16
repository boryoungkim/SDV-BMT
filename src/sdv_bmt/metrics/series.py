"""Bundle-only evaluation of a run series (``sdv-bmt evaluate SERIES_DIR``; minimal run series, CenterPoint part).

Reads ``series.json`` and the part bundles it names, and nothing else.  Before any metric it refuses the series when

* the series did not complete, a part is missing, or a part's ``manifest.json`` differs from the SHA-256 recorded in
  ``series.json``; any raw file of a part fails its manifest digest (checked before and after the evaluation);
* a part's plan, with its ``input.selection`` replaced by the series' declared selection, does not reproduce the
  series plan digest (so code, model, workload, DUT configuration, input index, run settings, replay, stress and
  evaluation settings are those of the series, and only the selection differs), or the code identity, realized DUT,
  environment, input pipeline, measurement boundary or quality-reference declaration differ between parts;
* the parts' selections are not the recorded partition, overlap, leave a gap, or share a source item id.

The result is one Metric Set evaluation (``metric_set_v1.compute_metric_set_parts``): counts summed, distributions
over all requests, rates over the sum of the parts' windows, quality once over all items.  The series is
measurement-valid only when every part is; otherwise every metric takes the first part's blanket status with a
``series_part_<k>:`` reason.  Per-part realization evidence (device state, clocks) is listed next to the result, not
merged.  Written to ``SERIES_DIR/evaluations/<eval_id>/`` (never overwritten).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional

from ..contracts.ids import digest_of, new_eval_id, to_jsonable, utc_now_iso
from ..contracts.records import LEGACY_METRIC_CATALOG_VERSION, MetricSetResult, raw_evidence_files
from ..contracts.specs import EvaluationOverride
from ..contracts.status import BENCHMARK_LEVEL, ExecutionKind
from ..evidence.bundle import (EVAL_MANIFEST_FILE, EVALUATIONS_DIR, MANIFEST_FILE, METRICS_FILE, file_record,
                               read_json, verify_raw_files, write_json_exclusive)
from ..evidence.provenance import code_identity
from ..observation.reader import BundleReadError, load_run_bundle
from . import metric_set_v1
from .evaluator import (_CLOCK_NOTES, CATALOGS, DEFAULT_CATALOG, EVAL_VOLATILE_FIELDS, METRIC_SET_EVAL_SCHEMA_VERSION,
                        EvaluationError, EvaluationOutcome, bundle_clock_domain, bundle_execution_kind,
                        canonical_metric_set_content, evidence_checks, measurement_gate, metric_set_context,
                        metric_set_parameters)

SERIES_FILE = "series.json"
SERIES_KIND = "sdv-bmt-run-series/1"
# manifest sections that must be identical in every part (everything else is selection- or run-specific)
SAME_IN_EVERY_PART = ("execution_kind", "benchmark_level", "experiment_id", "code_identity", "config_digests", "dut",
                      "environment", "measurement_boundary", "workload", "seed")
SAME_INPUT_FIELDS = ("input_id", "input_kind", "version", "input_family_digest", "source_index_sha256", "provider")
SAME_REFERENCE_FIELDS = ("annotation_schema", "evaluator", "schema_version", "file")


def is_series_dir(path: Path | str) -> bool:
    return (Path(path) / SERIES_FILE).is_file()


def _check_identity(series: dict[str, Any], bundles: list[Any]) -> list[str]:
    problems: list[str] = []
    declared = series["declared_selection"]
    first = bundles[0].manifest
    for k, b in enumerate(bundles):
        resolved = b.run_spec["resolved_plan"]
        as_series = {**resolved, "input": {**resolved["input"], "selection": declared}}
        if digest_of(as_series) != series["plan_digest"]:
            problems.append(f"part {k}: plan differs from the series plan in more than input.selection")
        m = b.manifest
        for key in SAME_IN_EVERY_PART:
            if m.get(key) != first.get(key):
                problems.append(f"part {k}: manifest {key} differs from part 0")
        for key in SAME_INPUT_FIELDS:
            if m["input"].get(key) != first["input"].get(key):
                problems.append(f"part {k}: input.{key} differs from part 0")
        strip = lambda p: {x: y for x, y in (p or {}).items() if x != "equivalence_validation"}  # noqa: E731
        if strip(m.get("input_pipeline")) != strip(first.get("input_pipeline")):
            problems.append(f"part {k}: input pipeline differs from part 0")
        ref, ref0 = m.get("quality_reference") or {}, first.get("quality_reference") or {}
        if any(ref.get(x) != ref0.get(x) for x in SAME_REFERENCE_FIELDS):
            problems.append(f"part {k}: quality reference declaration differs from part 0")
    return problems


def _check_selection(series: dict[str, Any], bundles: list[Any]) -> list[str]:
    problems: list[str] = []
    recorded = series["partition"]["parts"]
    if len(recorded) != len(bundles):
        return [f"series.json lists {len(recorded)} partition parts, {len(bundles)} part runs"]
    position = series["partition"]["selection"]["start"]
    seen: set[str] = set()
    for k, (part, b) in enumerate(zip(recorded, bundles)):
        sel = b.run_spec["resolved_plan"]["input"]["selection"]
        if (sel.get("start"), sel.get("count")) != (part["start"], part["count"]):
            problems.append(f"part {k}: selection {sel} is not the recorded partition part")
        if part["start"] != position:
            problems.append(f"part {k}: starts at index row {part['start']}, expected {position} (gap or overlap)")
        position = part["start"] + part["count"]
        if len(b.input_items) != part["count"]:
            problems.append(f"part {k}: {len(b.input_items)} input items, the partition says {part['count']}")
        ids = {item.source_item_id for item in b.input_items}
        if ids & seen:
            problems.append(f"part {k}: source items shared with an earlier part")
        seen |= ids
    total = series["partition"]["selection"]["count"]
    if position - series["partition"]["selection"]["start"] != total:
        problems.append("the parts do not add up to the declared selection")
    return problems


def series_context(series: dict[str, Any], bundles: list[Any]) -> dict[str, Any]:
    contexts = [metric_set_context(b) for b in bundles]
    first = contexts[0]
    total = lambda key: sum(c[key] or 0 for c in contexts)  # noqa: E731
    return {
        **first,
        "run_count": len(bundles),
        "planned_count": total("planned_count"),
        "issued_count": total("issued_count"),
        "completed_count": total("completed_count"),
        "input_selection_digest": None,
        "items_digest": None,
        "item_count": total("item_count"),
        "quality_reference": None,
        "series": {
            "series_id": series["series_id"],
            "plan_digest": series["plan_digest"],
            "declared_selection": series["declared_selection"],
            "partition_rule": series["partition"]["rule"],
            "parts": [{"part": k, "run_id": b.manifest["run_id"], "start": p["start"], "count": p["count"],
                       "input_selection_digest": c["input_selection_digest"], "items_digest": c["items_digest"],
                       "quality_reference": c["quality_reference"]}
                      for k, (b, p, c) in enumerate(zip(bundles, series["partition"]["parts"], contexts))],
            "gaps_between_parts": series["gaps_between_parts"],
        },
    }


def evaluate_series(
    series_dir: Path | str,
    *,
    override: Optional[EvaluationOverride] = None,
    catalog: str = DEFAULT_CATALOG,
    eval_id_factory: Callable[[], str] = new_eval_id,
    now_factory: Callable[[], str] = utc_now_iso,
) -> EvaluationOutcome:
    if catalog not in CATALOGS:
        raise EvaluationError(f"unknown metric catalog {catalog!r} (known: {list(CATALOGS)})")
    if catalog == LEGACY_METRIC_CATALOG_VERSION:
        raise EvaluationError("a run series is evaluated with a Metric Set catalog, not the legacy catalog")
    series_dir = Path(series_dir)
    try:
        series = read_json(series_dir / SERIES_FILE)
        if series.get("kind") != SERIES_KIND:
            raise ValueError(f"not a {SERIES_KIND} record")
    except (OSError, ValueError, AttributeError) as exc:
        raise EvaluationError(f"unreadable {SERIES_FILE}: {exc}") from None
    series_record = file_record(series_dir / SERIES_FILE)
    if series.get("status") != "completed":
        raise EvaluationError(f"series {series.get('series_id')} did not complete (status {series.get('status')})")
    run_dirs, raw_records, bundles = [], [], []
    for part in series["parts"]:
        run_dir = series_dir / part["run_dir"]
        try:
            manifest_record = file_record(run_dir / MANIFEST_FILE)
            manifest = read_json(run_dir / MANIFEST_FILE)
            names = raw_evidence_files(manifest.get("schema_version"))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise EvaluationError(f"part {part['part']}: unreadable manifest: {exc}") from None
        if manifest_record["sha256"] != part["manifest_sha256"]:
            raise EvaluationError(f"part {part['part']}: manifest.json differs from the SHA-256 in {SERIES_FILE}")
        problems = verify_raw_files(run_dir, manifest["files"], names)
        if problems:
            raise EvaluationError(f"part {part['part']}: raw integrity violated: " + "; ".join(problems))
        try:
            bundles.append(load_run_bundle(run_dir))
        except BundleReadError as exc:
            raise EvaluationError(f"part {part['part']}: unreadable bundle: {exc}") from None
        run_dirs.append(run_dir)
        raw_records.append((manifest_record, manifest["files"], names))
    issues = _check_identity(series, bundles) + _check_selection(series, bundles)
    if issues:
        raise EvaluationError("run series refused: " + "; ".join(issues))

    eval_id = eval_id_factory()
    final_dir = series_dir / EVALUATIONS_DIR / eval_id
    if final_dir.exists():
        raise FileExistsError(f"evaluation {eval_id} already exists; evaluations are never overwritten")
    first = bundles[0]
    boundary_id = first.manifest["measurement_boundary"]["measurement_boundary_id"]
    kind = bundle_execution_kind(first)
    try:
        params, sources = metric_set_parameters(first, override, catalog)
        gates = [measurement_gate(b) for b in bundles]
        blanket = next(((k, g[0]) for k, g in enumerate(gates) if g[0] is not None), None)
        if blanket is not None:
            k, (status, reason) = blanket
            metrics: list[MetricSetResult] = [MetricSetResult(
                metric_id=m.metric_id, layer=m.layer, definition_version=m.definition_version,
                measurement_boundary_id=boundary_id, unit=m.unit, status=status, value=None,
                population={"name": "not_evaluated", "sample_count": 0}, window=None, method=m.method,
                inputs=("series.json", "events.jsonl"), notes=(), eval_id=eval_id, execution_kind=kind,
                status_reason=f"series_part_{k}:{reason}") for m in metric_set_v1.CATALOG_METRICS[catalog]]
        else:
            parts = [metric_set_v1.MeasuredPart(b, g[1], g[2], g[3], f"part-{k:02d}")
                     for k, (b, g) in enumerate(zip(bundles, gates))]
            metrics = metric_set_v1.compute_metric_set_parts(parts, params, eval_id, kind, boundary_id,
                                                             _CLOCK_NOTES[bundle_clock_domain(first)], catalog)
        context = series_context(series, bundles)
        checks = [{**c, "part": k} for k, b in enumerate(bundles) for c in evidence_checks(b)]
    except (KeyError, TypeError) as exc:
        raise EvaluationError(f"malformed bundle content: {type(exc).__name__}: {exc}") from None
    metric_dicts = [to_jsonable(m) for m in metrics]
    content_digest = digest_of(canonical_metric_set_content(params, context, metric_dicts, catalog))

    for k, (run_dir, (manifest_record, raw, names)) in enumerate(zip(run_dirs, raw_records)):
        after = verify_raw_files(run_dir, raw, names)
        if after or file_record(run_dir / MANIFEST_FILE) != manifest_record:
            raise EvaluationError(f"part {k}: raw files changed during evaluation: " + "; ".join(after))
    if file_record(series_dir / SERIES_FILE) != series_record:
        raise EvaluationError(f"{SERIES_FILE} changed during evaluation")

    staging = series_dir / EVALUATIONS_DIR / f".staging-{eval_id}"
    staging.mkdir(parents=True, exist_ok=False)
    notice = first.manifest.get("mock_notice") if kind is ExecutionKind.MOCK else None
    realization = [{"part": k, "run_id": b.manifest["run_id"], "realization": b.manifest.get("realization"),
                    "run_clock": b.manifest.get("run_clock"), "realtime": b.manifest.get("realtime")}
                   for k, b in enumerate(bundles)]
    write_json_exclusive(staging / METRICS_FILE, {
        "schema_version": METRIC_SET_EVAL_SCHEMA_VERSION,
        "benchmark_level": BENCHMARK_LEVEL,
        "execution_kind": kind,
        "mock_notice": notice,
        "eval_id": eval_id,
        "run_id": None,
        "series_id": series["series_id"],
        "catalog_version": catalog,
        "measurement_boundary_id": boundary_id,
        "run_status": [{"part": k, **b.manifest["status"]} for k, b in enumerate(bundles)],
        "parameters": params,
        "context": context,
        "metrics": metric_dicts,
        "evidence_checks": checks,
        "part_realization": realization,
        "part_realization_note": "per-part device and host evidence, listed as recorded; the parts are not claimed to "
                                 "share one condition",
    })
    quality = params["quality"]
    write_json_exclusive(staging / EVAL_MANIFEST_FILE, {
        "schema_version": METRIC_SET_EVAL_SCHEMA_VERSION,
        "benchmark_level": BENCHMARK_LEVEL,
        "execution_kind": kind,
        "mock_notice": notice,
        "eval_id": eval_id,
        "run_id": None,
        "series_id": series["series_id"],
        "created_at_utc": now_factory(),
        "evaluator": {
            "module": __name__,
            "code_identity": code_identity(),
            "catalog_version": catalog,
            "definition_versions": {m.metric_id: m.definition_version for m in metric_set_v1.CATALOG_METRICS[catalog]},
            "metric_layers": {m.metric_id: m.layer.value for m in metric_set_v1.CATALOG_METRICS[catalog]},
            "quality_evaluator": None if quality is None else {
                "evaluator": quality["evaluator"], "implemented": quality["implemented"],
                "module": quality.get("module")},
        },
        "protocol": {**params, "deadline_source": sources["deadline"]},
        "protocol_digest": digest_of(params),
        "parameter_sources": sources,
        "inputs": {SERIES_FILE: series_record, **{
            f"{part['run_dir']}/{name}": record
            for part, (manifest_record, raw, _n) in zip(series["parts"], raw_records)
            for name, record in {MANIFEST_FILE: manifest_record, **raw}.items()}},
        "raw_integrity": {"verified_before_evaluation": True, "verified_after_evaluation": True},
        "outputs": {METRICS_FILE: file_record(staging / METRICS_FILE)},
        "metrics_content_sha256": content_digest,
        "volatile_fields": list(EVAL_VOLATILE_FIELDS) + ["series_id"],
    })
    os.rename(staging, final_dir)
    return EvaluationOutcome(eval_id, final_dir, tuple(metrics), content_digest, tuple(checks))
