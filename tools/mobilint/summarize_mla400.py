#!/usr/bin/env python3
"""Summarize the MLA400 evidence root: one row per run bundle (from STAGE/evaluations/<run_id>.json written by
tools/mobilint/run_series_mla400.sh) and one row per experiment over its runs.  Read-only, stdlib only.

    python3 tools/mobilint/summarize_mla400.py ROOT [OUT.json]
"""
import json
import statistics
import sys
from pathlib import Path

COLS = [("dut p50", "bmt.dut_service.p50", 1e6), ("dut p99", "bmt.dut_service.p99", 1e6),
        ("dut max", "bmt.dut_service.max_observed", 1e6),
        ("out_ready p50", "bmt.output_ready.p50", 1e6), ("out_ready p99", "bmt.output_ready.p99", 1e6),
        ("out_ready max", "bmt.output_ready.max_observed", 1e6),
        ("rate/s", "bmt.achieved_completion_rate", 1), ("drop", "bmt.drop_rate", 1),
        ("miss(out_ready)", "bmt.deadline.output_ready_miss_rate", 1), ("identity", "bmt.output.identity_rate", 1),
        ("AP", "quality.detection.conditional.AP", 1)]


def load(path):
    t = path.read_text()
    return json.loads(t[t.index("{"):])["metrics"]


def fmt(v, scale):
    if v is None:
        return "-"
    return f"{v / scale:.2f}" if scale != 1 else (f"{v:.4f}" if isinstance(v, float) else str(v))


def main():
    root = Path(sys.argv[1])
    rows = []
    for stage in ("capture", "baseline"):
        ev = root / stage / "evaluations"
        for run_dir in sorted((root / stage).glob("*/run-*")):
            f = ev / f"{run_dir.name}.json"
            if not f.exists():
                print(f"missing evaluation for {run_dir}", file=sys.stderr)
                continue
            m = load(f)
            status = {}
            for x in m.values():
                status[x["status"]] = status.get(x["status"], 0) + 1
            rows.append({"stage": stage, "experiment": run_dir.parent.name, "run_id": run_dir.name,
                         "values": {k: (m[k]["value"] if k in m else None) for _, k, _ in COLS},
                         "status": {k: (m[k]["status"] if k in m else "absent") for _, k, _ in COLS},
                         "metric_status_counts": status, "metrics": len(m)})
    hdr = ["experiment", "run_id"] + [c for c, _, _ in COLS] + ["valid/total"]
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join([r["experiment"], r["run_id"][4:19]] + [fmt(r["values"][k], s) for _, k, s in COLS]
                                + [f"{r['metric_status_counts'].get('valid', 0)}/{r['metrics']}"]) + " |")
    print()
    agg = {}
    for r in rows:
        agg.setdefault(r["experiment"], []).append(r)
    hdr = ["experiment", "runs", "dut p50 (median)", "dut p99 (max)", "dut max (max)", "out_ready p99 (max)",
           "rate/s (min)", "drop (max)", "miss (max)", "identity (min)"]
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    summary = {}
    for exp, rs in agg.items():
        v = lambda k: [r["values"][k] for r in rs if r["values"][k] is not None]
        s = {"runs": len(rs),
             "dut_p50_median_ms": statistics.median(v("bmt.dut_service.p50")) / 1e6 if v("bmt.dut_service.p50") else None,
             "dut_p99_max_ms": max(v("bmt.dut_service.p99")) / 1e6 if v("bmt.dut_service.p99") else None,
             "dut_max_ms": max(v("bmt.dut_service.max_observed")) / 1e6 if v("bmt.dut_service.max_observed") else None,
             "out_ready_p99_max_ms": max(v("bmt.output_ready.p99")) / 1e6 if v("bmt.output_ready.p99") else None,
             "rate_min": min(v("bmt.achieved_completion_rate"), default=None),
             "drop_max": max(v("bmt.drop_rate"), default=None),
             "miss_max": max(v("bmt.deadline.output_ready_miss_rate"), default=None),
             "identity_min": min(v("bmt.output.identity_rate"), default=None)}
        summary[exp] = s
        f2 = lambda x, d=2: "-" if x is None else f"{x:.{d}f}"
        print(f"| {exp} | {s['runs']} | {f2(s['dut_p50_median_ms'])} | {f2(s['dut_p99_max_ms'])} | "
              f"{f2(s['dut_max_ms'])} | {f2(s['out_ready_p99_max_ms'])} | {f2(s['rate_min'])} | "
              f"{f2(s['drop_max'], 4)} | {f2(s['miss_max'], 4)} | {f2(s['identity_min'], 4)} |")
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps({"runs": rows, "per_experiment": summary}, indent=1))


if __name__ == "__main__":
    main()
