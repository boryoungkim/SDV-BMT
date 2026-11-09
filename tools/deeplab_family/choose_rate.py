#!/usr/bin/env python3
"""Apply the plan v03 section 4.5 replay-rate rule to the three DeepLab golden-capture bundles (tool; stdlib only).

Rule (approved 2026-11-06): from the captures (pinned table, asap, the 500 val images) take the largest DUT-service
``t_complete - t_submit_begin`` of completed requests over the three models; choose the highest rate in
{30, 20, 15, 10, 5, 2, 1} Hz whose frame period floor(1e9 / rate) is at least 1.25 x that maximum.  The values are
read from the bundles' raw events (not from an evaluation).

    python3 tools/deeplab_family/choose_rate.py RUN_DIR_R18 RUN_DIR_R50 RUN_DIR_R101 [--out FILE.json]
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

CANDIDATES_HZ = (30, 20, 15, 10, 5, 2, 1)
MARGIN = Fraction(5, 4)


def dut_service_ns(run_dir: Path) -> list[int]:
    """t_complete - t_submit_begin of the measure-phase requests that completed."""
    measured: set[int] = set()
    submit: dict[int, int] = {}
    out = []
    for line in open(run_dir / "events.jsonl", encoding="utf-8"):
        e = json.loads(line)
        kind = e.get("kind")
        if kind == "issued" and e.get("phase") == "measure":
            measured.add(e["request_id"])
        elif kind == "submit_begin":
            submit[e["request_id"]] = e["t_run_ns"]
        elif kind == "completed" and e["request_id"] in measured:
            out.append(e["t_complete"] - submit[e["request_id"]])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    per_run = {}
    for r in a.runs:
        v = sorted(dut_service_ns(Path(r)))
        if not v:
            raise SystemExit(f"{r}: no completed request")
        per_run[r] = {"completed": len(v), "max_ns": v[-1], "p50_ns": v[(len(v) + 1) // 2 - 1]}
    worst = max(x["max_ns"] for x in per_run.values())
    need = worst * MARGIN
    chosen = next((hz for hz in CANDIDATES_HZ if math.floor(Fraction(10**9, hz)) >= need), None)
    doc = {"rule": "highest of 30/20/15/10/5/2/1 Hz with floor(1e9/rate) >= 1.25 x max DUT-service of the captures",
           "captures": per_run, "max_dut_service_ns": worst, "required_period_ns": float(need),
           "rate_hz": chosen, "period_ns": None if chosen is None else math.floor(Fraction(10**9, chosen)),
           "asap_max_above_0_5_s": worst > 500_000_000}
    text = json.dumps(doc, indent=1, sort_keys=True)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if chosen is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
