"""tools/show_metrics.py: a read-only view of metric tables (docs/metric_table.md)."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "show_metrics.py"
COLUMNS = ["row", "task", "dataset", "model", "device", "experiment_id", "run_id", "eval_id", "catalog_version",
           "metric_name", "metric_id", "layer", "unit", "status", "value", "min", "max", "n_runs", "n_valid"]


def _table(path: Path) -> Path:
    def row(kind, exp, metric, layer, unit, status, value, *, name="", run="", lo="", hi="", n="", nv=""):
        return {"row": kind, "task": "coco-2017-object-detection", "dataset": "coco-val2017-first1000 (1,000 items)",
                "model": "m", "device": "Dev", "experiment_id": exp, "run_id": run, "eval_id": "", "catalog_version":
                "cat", "metric_name": name, "metric_id": metric, "layer": layer, "unit": unit, "status": status,
                "value": value, "min": lo, "max": hi, "n_runs": n, "n_valid": nv}
    rows = [
        row("headline", "a", "bmt.dut_service.p50", "performance", "ms", "valid", "4.5", name="DUT service time (median)",
            lo="4.4", hi="4.6", n="2", nv="2"),
        row("model", "a", "bmt.dut_service.p50", "performance", "ns", "valid", "4500000", lo="4400000", hi="4600000",
            n="2", nv="2"),
        row("model", "a", "bmt.h2d.p50", "diagnostics", "ns", "not_supported", ""),
        row("model", "b", "quality.detection.conditional.ap", "task_quality", "ratio", "valid", "0.4289194068359202",
            n="1", nv="1"),
        row("run", "a", "bmt.dut_service.p50", "performance", "ns", "valid", "4400000", run="r1"),
        row("run", "a", "bmt.dut_service.p50", "performance", "ns", "valid", "4600000", run="r2"),
        row("run", "b", "quality.detection.conditional.ap", "task_quality", "ratio", "valid", "0.4289194068359202",
            run="r3"),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _run(*args):
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)], capture_output=True, text=True, timeout=60)


def test_full_view_groups_by_experiment_and_shows_ms(tmp_path):
    out = _run(_table(tmp_path / "m.metrics.csv"))
    assert out.returncode == 0, out.stderr
    text = out.stdout
    assert "rows 7 = headline 1 + model 3 + run 3" in text
    assert "DUT service time (median)" in text and "valid runs 2/2" in text
    assert text.index("-- a") < text.index("-- b")
    assert "4.500 ms  (4.400 - 4.600)" in text                    # ns shown as ms, with the run range
    assert "not_supported" in text and "status counts: not_supported 1, valid 1" in text
    assert "0.4289194 ratio" in text and "runs 2" in text and "runs 1" in text


def test_headline_only_and_refuses_other_files(tmp_path):
    table = _table(tmp_path / "m.metrics.csv")
    out = _run("--headline", table)
    assert out.returncode == 0 and "DUT service time (median)" in out.stdout
    assert "-- a" not in out.stdout and "bmt.h2d.p50" not in out.stdout
    other = tmp_path / "other.csv"
    other.write_text("a,b\n1,2\n", encoding="utf-8")
    bad = _run(other)
    assert bad.returncode == 2 and "not a metric table" in bad.stderr
