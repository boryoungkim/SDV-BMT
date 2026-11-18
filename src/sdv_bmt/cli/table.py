"""``python -m sdv_bmt.cli.table`` -- metric table of evaluated experiments and run series.

    python -m sdv_bmt.cli.table DIR [DIR ...] --out FILE.csv

DIR is an experiment directory (experiment.json and its run directories) or a run series directory
(series.json).  The table (docs/metric_table.md) gives four representative metrics of every model under readable
names first, then the overall result of every experiment per metric (median of the run values with min / max),
then one row per run and metric.  The representative metrics are also printed (stderr) as ``name - value`` lines.
Stdout carries a JSON summary.  The command measures and evaluates nothing: run rows copy the newest evaluation
of every run (or run series) unchanged.

Before anything is written it refuses a raw file that differs from its manifest.json, a run series part whose
manifest.json differs from series.json, a run or series without an evaluation, an evaluation whose metrics.json
differs from its eval_manifest.json record or that was made from a different manifest.json / series.json, runs of
one experiment made from different plans, a repeated experiment, MOCK and ACTUAL runs in one table, and an
existing output file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from ..contracts.records import raw_evidence_files
from ..evidence.bundle import (EVAL_MANIFEST_FILE, EVALUATIONS_DIR, EXPERIMENT_FILE, MANIFEST_FILE, METRICS_FILE,
                               file_record, read_json, run_dirs, verify_raw_files)
from ..metrics.series import SERIES_FILE, SERIES_KIND
from ..metrics.table import HEADLINE_ROW, MODEL_ROW, format_console, metric_csv_bytes, table_rows

EXIT_OK, EXIT_REJECTED = 0, 2
RECORD_FIELDS = ("metric_id", "layer", "unit", "status", "value")


class TableError(ValueError):
    """The inputs cannot be put into a metric table."""


def _read(path: Path, label: str) -> Any:
    try:
        return read_json(path)
    except (OSError, ValueError) as exc:
        raise TableError(f"{label}: unreadable {path.name}: {exc}") from None


def _verified_manifest(run_dir: Path, label: str) -> tuple[dict[str, Any], str]:
    if not (run_dir / MANIFEST_FILE).is_file():
        raise TableError(f"{label}: no manifest.json (unfinished run)")
    manifest = _read(run_dir / MANIFEST_FILE, label)
    if not isinstance(manifest, dict):
        raise TableError(f"{label}: manifest.json is not an object")
    problems = verify_raw_files(run_dir, manifest.get("files"), raw_evidence_files(manifest.get("schema_version")))
    if problems:
        raise TableError(f"{label}: raw files differ from manifest.json: " + "; ".join(problems))
    return manifest, file_record(run_dir / MANIFEST_FILE)["sha256"]


def _newest_evaluation(target: Path, label: str) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """(eval_id, eval_manifest, metric records) of the evaluation with the latest created_at_utc (ties: larger
    eval_id), after checking metrics.json against its eval_manifest.json record."""
    root = target / EVALUATIONS_DIR
    found = []
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        if path.is_dir() and not path.name.startswith(".") and (path / EVAL_MANIFEST_FILE).is_file() \
                and (path / METRICS_FILE).is_file():
            manifest = _read(path / EVAL_MANIFEST_FILE, f"{label}: evaluation {path.name}")
            if not isinstance(manifest, dict) or manifest.get("eval_id") != path.name:
                raise TableError(f"{label}: evaluation {path.name}: eval_manifest.json does not name this eval_id")
            found.append((str(manifest.get("created_at_utc") or ""), path.name, path, manifest))
    if not found:
        raise TableError(f"{label}: no evaluation (run 'sdv-bmt evaluate' first; this command does not evaluate)")
    _created, eval_id, path, manifest = max(found)
    if (manifest.get("outputs") or {}).get(METRICS_FILE) != file_record(path / METRICS_FILE):
        raise TableError(f"{label}: evaluation {eval_id}: metrics.json differs from its eval_manifest.json record")
    doc = _read(path / METRICS_FILE, f"{label}: evaluation {eval_id}")
    records = doc.get("metrics") if isinstance(doc, dict) else None
    if not isinstance(records, list):
        raise TableError(f"{label}: evaluation {eval_id}: metrics.json has no metric list")
    ids = [r.get("metric_id") if isinstance(r, dict) else None for r in records]
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise TableError(f"{label}: evaluation {eval_id}: missing or repeated metric_id")
    return eval_id, manifest, records


def _evaluated_input(eval_manifest: dict[str, Any], name: str) -> Optional[str]:
    record = (eval_manifest.get("inputs") or {}).get(name)
    return record.get("sha256") if isinstance(record, dict) else None


def _device_name(manifest: dict[str, Any]) -> Optional[str]:
    # the free-text device string of a real adapter may carry a hardware serial; a mock one names the fake
    dut = manifest.get("dut") or {}
    descriptor = dut.get("descriptor") or {}
    identity = descriptor.get("device_identity") or {}
    if identity.get("nvidia.product_name"):
        return identity["nvidia.product_name"]
    if manifest.get("execution_kind") == "mock":
        return descriptor.get("device")
    # another accelerator (for example an NPU): vendor and family from the device identity, named by its DUT config
    vendor, family, config = identity.get("vendor"), identity.get("accelerator_family"), dut.get("dut_config_id")
    if vendor and config:
        return " ".join(str(x) for x in (vendor, family) if x) + f" ({config})"
    return None


def _dataset(manifest: dict[str, Any], item_count: Any) -> Optional[str]:
    # the input set of the run as recorded in manifest.json: "<input_id> (<n> items)"
    input_id = (manifest.get("input") or {}).get("input_id")
    if not input_id:
        return None
    return f"{input_id} ({item_count:,} items)" if isinstance(item_count, int) else str(input_id)


def _rows(manifest: dict[str, Any], experiment_id: str, run_id: str, eval_id: str,
          eval_manifest: dict[str, Any], records: list[dict[str, Any]],
          item_count: Any = None) -> list[dict[str, Any]]:
    workload = manifest.get("workload") or {}
    if item_count is None:
        item_count = (manifest.get("input") or {}).get("item_count")
    common = {"task": workload.get("task"), "dataset": _dataset(manifest, item_count),
              "model": workload.get("workload_id"), "device": _device_name(manifest),
              "experiment_id": experiment_id, "run_id": run_id, "eval_id": eval_id,
              "catalog_version": (eval_manifest.get("evaluator") or {}).get("catalog_version")}
    return [{**common, **{k: r.get(k) for k in RECORD_FIELDS}} for r in records]


def experiment_rows(exp_dir: Path) -> tuple[list[dict[str, Any]], set[Any]]:
    record = _read(exp_dir / EXPERIMENT_FILE, exp_dir.name)
    experiment_id = record.get("experiment_id") if isinstance(record, dict) else None
    if not isinstance(experiment_id, str):
        raise TableError(f"{exp_dir.name}: experiment.json has no experiment_id")
    dirs = run_dirs(exp_dir)
    if not dirs:
        raise TableError(f"{experiment_id}: no run in the experiment directory")
    rows, kinds = [], set()
    for run_dir in dirs:
        label = f"{experiment_id}/{run_dir.name}"
        manifest, manifest_sha = _verified_manifest(run_dir, label)
        if manifest.get("plan_digest") != record.get("plan_digest"):
            raise TableError(f"{label}: plan differs from experiment.json")
        eval_id, eval_manifest, records = _newest_evaluation(run_dir, label)
        if _evaluated_input(eval_manifest, MANIFEST_FILE) != manifest_sha:
            raise TableError(f"{label}: evaluation {eval_id} was made from a different manifest.json")
        kinds.add(manifest.get("execution_kind"))
        rows += _rows(manifest, experiment_id, manifest.get("run_id"), eval_id, eval_manifest, records)
    return rows, kinds


def series_rows(series_dir: Path) -> tuple[list[dict[str, Any]], set[Any]]:
    label = series_dir.name
    series = _read(series_dir / SERIES_FILE, label)
    if not isinstance(series, dict) or series.get("kind") != SERIES_KIND:
        raise TableError(f"{label}: series.json is not a {SERIES_KIND} record")
    if series.get("status") != "completed":
        raise TableError(f"{label}: run series did not complete (status {series.get('status')})")
    eval_id, eval_manifest, records = _newest_evaluation(series_dir, label)
    if _evaluated_input(eval_manifest, SERIES_FILE) != file_record(series_dir / SERIES_FILE)["sha256"]:
        raise TableError(f"{label}: evaluation {eval_id} was made from a different series.json")
    kinds, first, items = set(), None, 0
    for part in series.get("parts") or []:
        part_label = f"{label}/part-{part['part']:02d}"
        manifest, manifest_sha = _verified_manifest(series_dir / part["run_dir"], part_label)
        if manifest_sha != part.get("manifest_sha256"):
            raise TableError(f"{part_label}: manifest.json differs from the SHA-256 in series.json")
        if _evaluated_input(eval_manifest, f"{part['run_dir']}/{MANIFEST_FILE}") != manifest_sha:
            raise TableError(f"{part_label}: evaluation {eval_id} was made from a different manifest.json")
        kinds.add(manifest.get("execution_kind"))
        first = first or manifest
        count = (manifest.get("input") or {}).get("item_count")
        items = items + count if isinstance(items, int) and isinstance(count, int) else None
    if first is None:
        raise TableError(f"{label}: run series has no part")
    # the series covers the items of all its parts
    return _rows(first, series["experiment_id"], series.get("series_id"), eval_id, eval_manifest, records,
                 items), kinds


def build_table(dirs: Sequence[Path | str]) -> list[dict[str, Any]]:
    experiments, kinds, seen = [], set(), set()
    for path in map(Path, dirs):
        if (path / SERIES_FILE).is_file():
            rows, run_kinds = series_rows(path)
        elif (path / EXPERIMENT_FILE).is_file():
            rows, run_kinds = experiment_rows(path)
        else:
            raise TableError(f"{path.name}: neither an experiment directory (experiment.json) nor a run series "
                             "directory (series.json)")
        experiment_id = rows[0]["experiment_id"] if rows else path.name
        if experiment_id in seen:
            raise TableError(f"{experiment_id}: experiment given twice")
        seen.add(experiment_id)
        experiments.append(rows)
        kinds |= run_kinds
    if not experiments:
        raise TableError("no experiment given")
    if len(kinds) != 1:
        raise TableError("MOCK and ACTUAL runs cannot share one table")
    return table_rows(experiments)


def write_table(dirs: Sequence[Path | str], out: Path | str) -> tuple[list[dict[str, Any]], bytes]:
    """Build the table and write it to ``out`` (created exclusively; nothing is written when a check fails)."""
    out = Path(out)
    if out.exists():
        raise TableError(f"{out} already exists; tables are never overwritten")
    rows = build_table(dirs)
    data = metric_csv_bytes(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "xb") as fh:
        fh.write(data)
    return rows, data


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sdv_bmt.cli.table",
                                     description="metric table of evaluated experiments / run series")
    parser.add_argument("dirs", nargs="+", help="experiment directories (experiment.json) or run series "
                                                "directories (series.json), in table order")
    parser.add_argument("--out", required=True, help="new CSV file (must not exist)")
    args = parser.parse_args(argv)
    try:
        rows, data = write_table(args.dirs, args.out)
    except (TableError, OSError) as exc:
        print(json.dumps({"command": "table", "status": "rejected", "issues": [str(exc)]}, indent=2))
        return EXIT_REJECTED
    print(format_console(rows, str(Path(args.out).resolve())), file=sys.stderr)
    print(json.dumps({"command": "table", "status": "written", "table": args.out,
                      "sha256": hashlib.sha256(data).hexdigest(), "rows": len(rows),
                      "headline_rows": sum(1 for r in rows if r["row"] == HEADLINE_ROW),
                      "model_rows": sum(1 for r in rows if r["row"] == MODEL_ROW),
                      "experiments": len({r["experiment_id"] for r in rows})}, indent=2))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
