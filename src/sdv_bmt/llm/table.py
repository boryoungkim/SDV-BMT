"""Metric table of evaluated LLM experiments (``python -m sdv_bmt.llm table``).

Reads, for every given experiment directory, its runs and the newest evaluation of every run, and writes the
metric table of ``sdv_bmt.metrics.table`` (model rows first, then run rows).  Before anything is written it
refuses a directory that is not an LLM experiment, a run whose raw files differ from its manifest.json, a run made
from another experiment config, a run without an evaluation, an evaluation whose metrics.json differs from its
eval_manifest.json record or that was made from a different manifest.json, and MOCK and ACTUAL runs in one table.

The evaluation (``metrics.json``, catalog ``sdv-bmt-llm-v1``) is nested; ``metric_records`` spreads it into metric
records.  Values are copied; names follow ``llm.<S1..S5|mmlu>.<quantity>.<statistic>``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from ..contracts.ids import digest_of
from ..metrics.table import metric_csv_bytes, table_rows
from . import LLM_RUN_SCHEMA
from .evaluate import LlmEvaluationError, verify_bundle
from .workload import file_sha256

# statistic name in the table -> key in the evaluation's statistics block
STATISTICS = (("p50", "median_p50"), ("p90", "p90"), ("p99", "p99"), ("min", "min"), ("max_observed", "max_observed"))
SPEED_QUANTITIES = (("ttft_ns", "ns"), ("tpot_ns", "ns"), ("e2e_ns", "ns"), ("decode_tps", "1/s"))
SPEED_COUNTS = ("measured", "fixed_length_ok", "length_mismatch", "error")
MMLU_COUNTS = ("correct", "incorrect_parsable", "unparsable", "failed", "not_run")
TASKS = {"speed": "llm-speed", "mmlu": "llm-mmlu"}


class LlmReportError(ValueError):
    """The inputs cannot be put into a metric table."""


def _record(metric_id: str, layer: str, unit: str, value: Any, status: Optional[str] = None) -> dict[str, Any]:
    return {"metric_id": metric_id, "layer": layer, "unit": unit, "value": value,
            "status": status or ("valid" if value is not None else "not_available")}


def _statistics(prefix: str, unit: str, block: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for name, key in STATISTICS:
        item = block.get(key)
        if block.get("status") != "valid":
            out.append(_record(f"{prefix}.{name}", "performance", unit, None, block.get("status") or "not_available"))
        elif isinstance(item, dict):  # a percentile carries its own status and sample count
            out.append(_record(f"{prefix}.{name}", "performance", unit, item.get("value"), item.get("status")))
        else:
            out.append(_record(f"{prefix}.{name}", "performance", unit, item))
    return out


def metric_records(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Spread one LLM evaluation into metric records (metric_id, layer, unit, status, value), values unchanged."""
    out: list[dict[str, Any]] = []
    status = metrics.get("run_status") or {}
    out.append(_record("llm.run.status", "validity_evidence", "", status.get("run_status")))
    for key in ("planned", "completed", "error"):
        out.append(_record(f"llm.count.{key}", "performance", "count", (status.get("request_counts") or {}).get(key)))
    controls = metrics.get("controls") or {}
    out.append(_record("llm.control.prefix_caching", "validity_evidence", "",
                       (controls.get("prefix_caching") or {}).get("verdict")))
    out.append(_record("llm.control.one_request_at_a_time", "validity_evidence", "",
                       (controls.get("concurrency") or {}).get("status")))
    out.append(_record("llm.streaming.unit", "validity_evidence", "", (metrics.get("streaming") or {}).get("unit")))
    for workload, block in (metrics.get("speed") or {}).items():
        counts = block.get("counts") or {}
        for key in SPEED_COUNTS:
            out.append(_record(f"llm.{workload}.count.{key}", "performance", "count", counts.get(key)))
        for quantity, unit in SPEED_QUANTITIES:
            out += _statistics(f"llm.{workload}.{quantity}", unit, block.get(quantity) or {})
    quality = metrics.get("quality")
    if quality is not None:
        out.append(_record("llm.mmlu.accuracy", "task_quality", "ratio", quality.get("accuracy")))
        out.append(_record("llm.mmlu.coverage", "task_quality", "ratio", quality.get("coverage")))
        out.append(_record("llm.mmlu.count.total", "task_quality", "count", quality.get("total")))
        counts = quality.get("counts") or {}
        for key in MMLU_COUNTS:
            out.append(_record(f"llm.mmlu.count.{key}", "task_quality", "count", counts.get(key)))
        times = metrics.get("mmlu_request_times") or {}
        for quantity in ("ttft_ns", "e2e_ns"):
            out += _statistics(f"llm.mmlu.{quantity}", "ns", times.get(quantity) or {})
    return out


def _read(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LlmReportError(f"{label}: unreadable {path.name}: {exc}") from None


def selected_evaluation(run_dir: Path, run_id: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The newest evaluation of a run (latest eval_manifest created_at_utc, ties: larger eval_id), verified."""
    root = run_dir / "evaluations"
    found = []
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        if path.is_dir() and (path / "eval_manifest.json").is_file() and (path / "metrics.json").is_file():
            manifest = _read(path / "eval_manifest.json", f"{label}: evaluation {path.name}")
            found.append((str(manifest.get("created_at_utc") or ""), path.name, path, manifest))
    if not found:
        raise LlmReportError(f"{label}: no evaluation (run 'python -m sdv_bmt.llm evaluate' first)")
    _created, eval_id, path, manifest = max(found)
    metrics = _read(path / "metrics.json", f"{label}: evaluation {eval_id}")
    content = {k: v for k, v in metrics.items() if k not in ("eval_id", "run_id")}
    if manifest.get("eval_id") != eval_id or metrics.get("eval_id") != eval_id:
        raise LlmReportError(f"{label}: evaluation {eval_id}: eval_id does not match its directory")
    if metrics.get("run_id") != run_id or manifest.get("run_id") != run_id:
        raise LlmReportError(f"{label}: evaluation {eval_id} belongs to another run")
    if manifest.get("manifest_sha256") != file_sha256(run_dir / "manifest.json"):
        raise LlmReportError(f"{label}: evaluation {eval_id} was made from a different manifest.json")
    if digest_of(content) != manifest.get("metrics_content_sha256"):
        raise LlmReportError(f"{label}: evaluation {eval_id}: metrics.json differs from its eval_manifest.json record")
    return metrics, manifest


def _dataset(workload: dict[str, Any], metrics: dict[str, Any]) -> Optional[str]:
    """The input data of a run from its resolved workload: the speed workload file (synthetic token IDs) or the MMLU
    item list with the item total of the evaluation."""
    kind = workload.get("kind")
    if kind == "speed" and workload.get("file"):
        return f"{Path(workload['file']).name.split('.')[0]} (synthetic token IDs)"
    if kind == "mmlu" and workload.get("items_file"):
        name = Path(workload["items_file"]).name.split(".")[0]
        total = (metrics.get("quality") or {}).get("total")
        return f"{name} ({total:,} items)" if isinstance(total, int) else name
    return None


def experiment_rows(exp_dir: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Run rows of one experiment directory and the execution kinds of its runs."""
    record = _read(exp_dir / "experiment.json", exp_dir.name) if (exp_dir / "experiment.json").is_file() else None
    if not isinstance(record, dict) or record.get("schema") != LLM_RUN_SCHEMA:
        raise LlmReportError(f"{exp_dir.name}: not an LLM experiment directory (experiment.json, {LLM_RUN_SCHEMA})")
    experiment_id = record["experiment_id"]
    runs = sorted(p for p in exp_dir.glob("run-*") if p.is_dir())
    if not runs:
        raise LlmReportError(f"{experiment_id}: no run in the experiment directory")
    rows, kinds = [], set()
    for run_dir in runs:
        label = f"{experiment_id}/{run_dir.name}"
        try:
            manifest = verify_bundle(run_dir)
        except LlmEvaluationError as exc:
            raise LlmReportError(str(exc)) from None
        spec = _read(run_dir / "run_spec.json", label)
        if spec.get("config_digest") != record.get("config_digest"):
            raise LlmReportError(f"{label}: run made from another experiment config than experiment.json")
        metrics, _manifest = selected_evaluation(run_dir, manifest["run_id"], label)
        kinds.add(manifest.get("execution_kind"))
        resolved = spec.get("resolved") or {}
        kind = (resolved.get("workload") or {}).get("kind")
        common = {"task": TASKS.get(kind, f"llm-{kind}"), "dataset": _dataset(resolved.get("workload") or {}, metrics),
                  "model": (resolved.get("model") or {}).get("name"),
                  "device": (resolved.get("device") or {}).get("name"), "experiment_id": experiment_id,
                  "run_id": manifest["run_id"], "eval_id": metrics["eval_id"],
                  "catalog_version": metrics.get("catalog_version")}
        rows += [{**common, **rec} for rec in metric_records(metrics)]
    return rows, kinds


def build_table(dirs: Sequence[Path | str]) -> list[dict[str, Any]]:
    experiments, kinds, seen = [], set(), set()
    for d in map(Path, dirs):
        rows, run_kinds = experiment_rows(d)
        experiment_id = rows[0]["experiment_id"]
        if experiment_id in seen:
            raise LlmReportError(f"{experiment_id}: experiment given twice")
        seen.add(experiment_id)
        experiments.append(rows)
        kinds |= run_kinds
    if not experiments:
        raise LlmReportError("no experiment given")
    if len(kinds) != 1:
        raise LlmReportError("MOCK and ACTUAL runs cannot share one table")
    return table_rows(experiments)


def write_table(dirs: Sequence[Path | str], out: Path | str) -> tuple[list[dict[str, Any]], bytes]:
    """Build the table and write it to ``out`` (created exclusively)."""
    rows = build_table(dirs)
    data = metric_csv_bytes(rows)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "xb") as fh:
        fh.write(data)
    return rows, data
