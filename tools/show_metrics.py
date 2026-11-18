#!/usr/bin/env python3
"""Show metric tables (``<model>.metrics.csv``) in a readable form.

    tools/show_metrics.py FILE.csv [FILE.csv ...]             everything: headline rows, then the model rows of
                                                              every experiment grouped by layer, then status counts
    tools/show_metrics.py --headline FILE.csv [FILE.csv ...]  per file: row counts and the headline rows only

The tables are written by ``python -m sdv_bmt.cli.table`` and ``python -m sdv_bmt.llm table`` (docs/metric_table.md).
This tool only reads them: it computes nothing and writes nothing.  Times in nanoseconds are shown in milliseconds;
the CSV keeps the exact values.  Use a value only when its status is ``valid``.  Standard library only.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

LAYERS = ("performance", "timeliness", "task_quality", "diagnostics", "validity_evidence")
REQUIRED = ("row", "experiment_id", "metric_id", "layer", "unit", "status", "value")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: not a metric table (missing columns: {', '.join(missing)})")
        return list(reader)


def number(value: str, unit: str) -> str:
    """A value for display: ns as ms with three decimals, other numbers with up to seven significant digits."""
    if value == "":
        return ""
    try:
        x = float(value)
    except ValueError:
        return value                                   # a text value
    if unit == "ns":
        return f"{x / 1e6:.3f}"
    if x.is_integer() and unit in ("count", "boolean"):
        return str(int(x))
    return f"{x:.7g}"


def shown_unit(unit: str) -> str:
    return "ms" if unit == "ns" else unit


def counts_line(rows: list[dict[str, str]]) -> str:
    c = collections.Counter(r["row"] for r in rows)
    return f"rows {len(rows)} = headline {c['headline']} + model {c['model']} + run {c['run']}"


def headline_lines(rows: list[dict[str, str]]) -> list[str]:
    out = []
    for r in rows:
        if r["row"] != "headline":
            continue
        span = f"{r['n_valid']}/{r['n_runs']}" if r.get("n_runs") else ""
        out.append(f"  {r.get('metric_name') or r['metric_id']:<42} {number(r['value'], 'ms'):>14} "
                   f"{r['unit']:<6} valid runs {span:<6} {r['experiment_id']}")
    return out


def model_lines(rows: list[dict[str, str]], experiment: str) -> list[str]:
    mine = [r for r in rows if r["row"] == "model" and r["experiment_id"] == experiment]
    by_layer: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for r in mine:
        by_layer[r["layer"]].append(r)
    out = []
    for layer in [*LAYERS, *sorted(set(by_layer) - set(LAYERS))]:
        if not by_layer.get(layer):
            continue
        out.append(f"  [{layer}: {len(by_layer[layer])}]")
        for r in by_layer[layer]:
            value = number(r["value"], r["unit"])
            lo, hi = number(r.get("min", ""), r["unit"]), number(r.get("max", ""), r["unit"])
            spread = f"  ({lo} - {hi})" if lo and hi and lo != hi else ""
            out.append(f"    {r['metric_id']:<48} {r['status']:<20} {value:>14} {shown_unit(r['unit'])}{spread}")
    statuses = collections.Counter(r["status"] for r in mine)
    out.append("  status counts: " + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())))
    return out


def show(path: Path, headline_only: bool) -> list[str]:
    rows = read_rows(path)
    out = [f"== {path}", f"  {counts_line(rows)}"]
    out += headline_lines(rows)
    if headline_only:
        return out
    experiments = list(dict.fromkeys(r["experiment_id"] for r in rows if r["row"] == "model"))
    for experiment in experiments:
        first = next(r for r in rows if r["row"] == "model" and r["experiment_id"] == experiment)
        runs = sorted({r["run_id"] for r in rows if r["row"] == "run" and r["experiment_id"] == experiment})
        out += ["", f"-- {experiment}",
                f"  model {first.get('model', '')} | device {first.get('device', '')}",
                f"  dataset {first.get('dataset', '')} | catalog {first.get('catalog_version', '')} | runs {len(runs)}"]
        out += model_lines(rows, experiment)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="show_metrics.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--headline", action="store_true", help="row counts and headline rows only")
    parser.add_argument("files", nargs="+", type=Path, help="metric tables (<model>.metrics.csv)")
    args = parser.parse_args(argv)
    status = 0
    for i, path in enumerate(args.files):
        try:
            lines = show(path, args.headline)
        except (OSError, ValueError) as exc:
            print(f"show_metrics.py: {exc}", file=sys.stderr)
            status = 2
            continue
        if i:
            print()
        print("\n".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(main())
