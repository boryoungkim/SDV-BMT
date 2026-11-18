#!/usr/bin/env python3
"""Apply the replay-rate rule of the approved CenterPoint plan (v01 section 6 / v02 section 6) to the mini golden
capture (tool; stdlib only).

The DeepLab rule (tools/deeplab_family/choose_rate.py): take the largest DUT-service ``t_complete - t_submit_begin``
of the completed requests of the capture (pinned table, asap) and choose the highest candidate rate whose frame
period ``floor(1e9 / rate)`` is at least 1.25 x that maximum.  The CenterPoint plan keeps the formula and caps the
candidates at the LiDAR rotation rate of nuScenes: {20, 10, 5, 2, 1} Hz (the DeepLab list also has 30 and 15 Hz).
The rate is used for the mini baseline and for the full val run series.

    python3 tools/centerpoint/choose_rate.py CAPTURE_RUN_DIR [--out FILE.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deeplab_family"))
from choose_rate import MARGIN, dut_service_ns  # noqa: E402

CANDIDATES_HZ = (20, 10, 5, 2, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out")
    a = ap.parse_args()
    v = sorted(dut_service_ns(Path(a.run)))
    if not v:
        raise SystemExit(f"{a.run}: no completed request")
    worst = v[-1]
    need = worst * MARGIN
    chosen = next((hz for hz in CANDIDATES_HZ if math.floor(Fraction(10**9, hz)) >= need), None)
    doc = {"rule": "highest of 20/10/5/2/1 Hz with floor(1e9/rate) >= 1.25 x max DUT-service of the mini capture "
                   "(DeepLab formula; candidates capped at the 20 Hz LiDAR rotation rate, CenterPoint plan v01/v02 6)",
           "capture": a.run, "completed": len(v), "max_dut_service_ns": worst, "p50_dut_service_ns": v[(len(v) + 1) // 2 - 1],
           "required_period_ns": float(need), "rate_hz": chosen,
           "period_ns": None if chosen is None else math.floor(Fraction(10**9, chosen)),
           "deeplab_candidates_would_give_hz": next((hz for hz in (30, 20, 15, 10, 5, 2, 1)
                                                     if math.floor(Fraction(10**9, hz)) >= need), None)}
    text = json.dumps(doc, indent=1, sort_keys=True)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if chosen is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
