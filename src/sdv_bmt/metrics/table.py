"""Metric table (``metrics.csv``) and its console print.

The table has three parts, in this order:

1. ``headline`` rows: four representative metrics of every model under a readable name (``metric_name``), with
   times shown in milliseconds.  The console print shows these rows only, as one table per model.
2. ``model`` rows: the overall result of every experiment, one row per metric.
3. ``run`` rows: one row per run and metric.

Every path that writes a metric table uses these columns, so the tables of different models read the same way.

Run rows copy the metric records of an evaluation unchanged.  Model rows are the only values formed here:

* a numeric metric: the median of the valid run values, with min, max and the run counts;
* a text or boolean metric: the run value when every run has the same valid value (``differs_across_runs`` when
  they differ).

A metric with no valid run value gets the status all its runs share (for example ``insufficient_samples``), or
``no_valid_run`` when they differ.  Headline rows repeat model rows; the only change is ns -> ms (value / 10^6).
Model and headline rows describe the runs; they are not a test.
"""

from __future__ import annotations

import csv
import io
import json
import statistics
from typing import Any, Mapping, NamedTuple, Optional, Sequence

HEADLINE_ROW, MODEL_ROW, RUN_ROW = "headline", "model", "run"


class Headline(NamedTuple):
    metric_id: str
    name: str
    unit: str  # display unit; "ms" converts a value in ns


_VISION_TIMING = (
    Headline("bmt.dut_service.p50", "DUT service time (median)", "ms"),
    Headline("bmt.output_ready.p99", "Output-ready latency (p99)", "ms"),
    Headline("bmt.deadline.output_ready_miss_rate", "Deadline miss rate (output ready)", "ratio"),
)
# task -> its representative metrics.  Vision: the frame-budget view of the approved report (typical service time,
# tail of the time to a ready output, deadline misses) plus the task quality.  LLM: per device, the three speed
# metrics of the middle workload S3 (512 input / 512 output tokens) from the speed experiment and the raw tinyMMLU
# accuracy and the gp-IRT estimate (computed by tools/llm/tinymmlu_irt.py, given to the table with --irt) from the
# tinyMMLU experiment; the MMLU entry reads v1.0 bundles.
HEADLINES: dict[str, tuple[Headline, ...]] = {
    "coco-2017-object-detection": _VISION_TIMING + (
        Headline("quality.detection.conditional.ap", "COCO box AP", "ratio"),),
    "cityscapes-semantic-segmentation-19class": _VISION_TIMING + (
        Headline("quality.segmentation.conditional.miou", "Cityscapes mIoU (19 classes)", "ratio"),),
    "nuscenes-3d-detection-10class": _VISION_TIMING + (
        Headline("quality.det3d.conditional.nds", "nuScenes detection score (NDS)", "ratio"),),
    "llm-speed": (
        Headline("llm.S3.ttft_ns.p50", "Time to first token, TTFT (S3, median)", "ms"),
        Headline("llm.S3.tpot_ns.p50", "Time per output token, TPOT (S3, median)", "ms"),
        Headline("llm.S3.e2e_ns.p50", "End-to-end latency (S3, median)", "ms"),),
    "llm-mmlu": (
        Headline("llm.mmlu.accuracy", "MMLU accuracy (1,000 items, zero-shot)", "ratio"),),
    "llm-tinymmlu": (
        Headline("llm.tinymmlu.accuracy", "tinyMMLU accuracy (100 items, lm-evaluation-harness method)", "ratio"),
        Headline("llm.tinymmlu.gpirt", "tinyMMLU gp-IRT estimate of MMLU accuracy (value the harness reports)",
                 "ratio"),),
}
# a line printed under the headline table of a task
HEADLINE_NOTES = {"llm-speed": "S3 = speed workload with 512 input and 512 output tokens"}

METRIC_CSV_COLUMNS: tuple[tuple[str, str], ...] = (
    ("row", f"'{HEADLINE_ROW}': representative metric of the model (listed first); '{MODEL_ROW}': overall result "
            f"of the experiment per metric; '{RUN_ROW}': one run"),
    ("task", "task of the workload"),
    ("dataset", "input data of the run: input set identifier and item count (LLM: workload file / MMLU item list)"),
    ("model", "model (workload) name"),
    ("device", "device name as recorded in the run"),
    ("experiment_id", "experiment identifier"),
    ("run_id", "run identifier (a run series: its series identifier); empty on headline and model rows"),
    ("eval_id", "evaluation the run row was copied from; empty on headline and model rows"),
    ("catalog_version", "metric catalog of the evaluation; metric definitions can differ between catalogs"),
    ("metric_name", "headline row: readable name of the metric; empty otherwise"),
    ("metric_id", "metric identifier (on a headline row: the metric it repeats)"),
    ("layer", "result layer (performance, timeliness, task_quality, diagnostics, validity_evidence)"),
    ("unit", "unit of value (ns, ms, count, ratio, 1/s, ...); empty for a text value"),
    ("status", "run row: status in the evaluation (use value only when valid); model / headline row: valid; "
               "without a valid run value the status all runs share, else no_valid_run; differs_across_runs for "
               "a text value that differs; not_in_evaluation for a headline metric the evaluation lacks"),
    ("value", "run row: the evaluated value (empty when null); model / headline row: median of the valid run "
              "values (text: the value all runs share); headline times in ms"),
    ("min", "model / headline row, numeric metric: smallest valid run value"),
    ("max", "model / headline row, numeric metric: largest valid run value"),
    ("n_runs", "model / headline row: runs that carry the metric"),
    ("n_valid", "model / headline row: runs whose value is valid"),
)
COLUMN_NAMES = tuple(name for name, _ in METRIC_CSV_COLUMNS)
IDENTITY_FIELDS = ("task", "dataset", "model", "device", "experiment_id", "catalog_version")


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def model_rows(run_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Model rows of one experiment, in the order its metrics first appear in ``run_rows``."""
    order: list[str] = []
    by_metric: dict[str, list[Mapping[str, Any]]] = {}
    for row in run_rows:
        if row["metric_id"] not in by_metric:
            order.append(row["metric_id"])
            by_metric[row["metric_id"]] = []
        by_metric[row["metric_id"]].append(row)
    out = []
    for metric_id in order:
        rows = by_metric[metric_id]
        valid = [r["value"] for r in rows if r.get("status") == "valid" and r.get("value") is not None]
        first = rows[0]
        entry: dict[str, Any] = {"row": MODEL_ROW, **{k: first.get(k) for k in IDENTITY_FIELDS},
                                 "run_id": None, "eval_id": None, "metric_name": None, "metric_id": metric_id,
                                 "layer": first.get("layer"), "unit": first.get("unit"), "value": None,
                                 "min": None, "max": None, "n_runs": len(rows), "n_valid": len(valid)}
        for field in ("dataset", "catalog_version"):  # runs of one plan share them; never guess when they differ
            if len({r.get(field) for r in rows}) != 1:
                entry[field] = None
        if not valid:
            statuses = {r.get("status") for r in rows}
            entry["status"] = statuses.pop() if len(statuses) == 1 and None not in statuses else "no_valid_run"
        elif all(_number(v) for v in valid):
            entry.update(status="valid", value=statistics.median(valid), min=min(valid), max=max(valid))
        elif len({json.dumps(v, sort_keys=True) for v in valid}) == 1 and len(valid) == len(rows):
            entry.update(status="valid", value=valid[0])
        else:
            entry["status"] = "differs_across_runs"
        out.append(entry)
    return out


def _to_ms(value: Any) -> Any:
    return value / 1_000_000 if _number(value) else value


def headline_rows(models: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Headline rows of one experiment from its model rows (``HEADLINES`` of its task, in that order)."""
    if not models:
        return []
    by_id = {row["metric_id"]: row for row in models}
    first = models[0]
    out = []
    for headline in HEADLINES.get(first.get("task"), ()):
        source = by_id.get(headline.metric_id)
        if source is None:
            out.append({"row": HEADLINE_ROW, **{k: first.get(k) for k in IDENTITY_FIELDS}, "run_id": None,
                        "eval_id": None, "metric_name": headline.name, "metric_id": headline.metric_id,
                        "layer": None, "unit": headline.unit, "status": "not_in_evaluation", "value": None,
                        "min": None, "max": None, "n_runs": 0, "n_valid": 0})
            continue
        row = {**source, "row": HEADLINE_ROW, "metric_name": headline.name}
        if headline.unit == "ms" and source.get("unit") == "ns":
            row.update(unit="ms", value=_to_ms(source.get("value")), min=_to_ms(source.get("min")),
                       max=_to_ms(source.get("max")))
        out.append(row)
    return out


def table_rows(experiments: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Headline rows of all experiments, then all model rows, then all run rows (each part in experiment order).
    ``experiments`` holds the run rows of each experiment; a run row carries the run fields of ``COLUMN_NAMES``."""
    models = [model_rows(rows) for rows in experiments]
    runs = [{**row, "row": RUN_ROW, "metric_name": None} for rows in experiments for row in rows]
    return [row for m in models for row in headline_rows(m)] + [row for m in models for row in m] + runs


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return str(value)


def metric_csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """The table as UTF-8 CSV (header first, ``\\n`` line ends); the same rows give the same bytes."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMN_NAMES)
    for row in rows:
        writer.writerow([_cell(row.get(name)) for name in COLUMN_NAMES])
    return buffer.getvalue().encode("utf-8")


def _show(value: Any, unit: Optional[str]) -> str:
    if not _number(value):
        return _cell(value)
    if unit == "ms":
        return f"{value:.3f}"
    if unit == "ratio":
        return f"{value:.4f}"
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def _group(row: Mapping[str, Any]) -> tuple[Any, ...]:
    # an LLM model on one device is one group (its speed and MMLU experiments); a vision experiment is one group
    task = str(row.get("task") or "")
    return (row.get("model"), row.get("device")) if task.startswith("llm") else (row.get("experiment_id"),)


_DASH = "\u2013"
_COLUMNS = ("Metric", "Value", "Unit", "Valid runs", f"Min {_DASH} Max")
_ALIGN = "<><><"  # left, right, left, right, left


def _cells(row: Mapping[str, Any]) -> tuple[str, ...]:
    unit = row.get("unit") or ""
    n_runs = int(row.get("n_runs") or 0)
    valid = f"{int(row.get('n_valid') or 0)}/{n_runs}"
    if row.get("status") != "valid":
        return str(row.get("metric_name")), str(row.get("status")), "", valid, ""
    spread = ""
    if n_runs > 1 and _number(row.get("min")):
        spread = f"{_show(row.get('min'), unit)} {_DASH} {_show(row.get('max'), unit)}"
    return str(row.get("metric_name")), _show(row.get("value"), unit), unit, valid, spread


def _block(members: Sequence[Mapping[str, Any]], widths: Sequence[int]) -> list[str]:
    first = members[0]
    experiments = list(dict.fromkeys(str(r.get("experiment_id")) for r in members))
    datasets = [d for d in dict.fromkeys(r.get("dataset") for r in members) if d]
    info = (("task", ", ".join(dict.fromkeys(str(r.get("task")) for r in members))),
            ("dataset", "; ".join(datasets) or "-"),
            ("device", str(first.get("device"))),
            ("experiment" if len(experiments) == 1 else "experiments", ", ".join(experiments)))

    def line(cells: Sequence[str]) -> str:
        return "  ".join(c.ljust(w) if a == "<" else c.rjust(w) for c, w, a in zip(cells, widths, _ALIGN)).rstrip()

    width = sum(widths) + 2 * (len(widths) - 1)
    title = str(first.get("model"))
    lines = [f"\u250c\u2500 {title} " + "\u2500" * max(3, width - len(title) - 2)]
    lines += [f"\u2502 {label:<11}  {text}" for label, text in info]
    lines += ["\u2502", "\u2502 " + line(_COLUMNS), "\u2502 " + "  ".join("\u2500" * w for w in widths)]
    lines += ["\u2502 " + line(_cells(r)) for r in members]
    notes = list(dict.fromkeys(HEADLINE_NOTES[t] for t in (r.get("task") for r in members) if t in HEADLINE_NOTES))
    if notes:
        lines += ["\u2502"] + [f"\u2502 {note}" for note in notes]
    lines.append("\u2514" + "\u2500" * (width + 1))
    return lines


def format_headlines(rows: Sequence[Mapping[str, Any]]) -> str:
    """The headline rows as one framed table per model (per device for an LLM): metric name, value, unit, valid
    runs, min - max.  The columns line up across the tables of one print."""
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("row") == HEADLINE_ROW:
            groups.setdefault(_group(row), []).append(row)
    cells = [_COLUMNS] + [_cells(r) for members in groups.values() for r in members]
    widths = [max(len(c[i]) for c in cells) for i in range(len(_COLUMNS))]
    return "\n\n".join("\n".join(_block(members, widths)) for members in groups.values())


def format_console(rows: Sequence[Mapping[str, Any]], csv_path: str) -> str:
    """What the table commands print: the headline tables, how to read them, and where the full table is."""
    return (format_headlines(rows) + "\n\n"
            f"Value = median of the valid runs; Min {_DASH} Max = range of the valid run values.\n"
            "For detailed results (every metric, every run), see the CSV file:\n"
            f"  {csv_path}\n")
