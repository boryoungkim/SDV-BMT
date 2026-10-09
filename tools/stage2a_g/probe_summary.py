"""Stage 2a-G: tabulate C++ probe runs (rt_probe JSONL) side by side (investigation tool; stdlib only).

    python3 tools/stage2a_g/probe_summary.py OUT.json PROBE.jsonl [PROBE.jsonl ...]

Descriptive quantiles only (nearest rank); no inferential statistics.  Durations are host
CLOCK_MONOTONIC differences except the CUDA-event brackets and the in-kernel %globaltimer delta,
which are device-clock durations and are never mixed with host timestamps.
"""

from __future__ import annotations

import json
import sys


def q(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    v = sorted(values)
    n = len(v)
    pick = lambda p: v[min(n - 1, max(0, -(-p * n // 100) - 1))]  # noqa: E731
    return {"n": n, "min": v[0], "p50": pick(50), "p90": pick(90), "p99": pick(99), "max": v[-1]}


def summarize(path: str) -> dict:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    h, reqs = rows[0], rows[1:]
    issued = [r for r in reqs if r["t_submit_begin"] >= 0]
    done = [r for r in reqs if r["state"] == "completed"]
    counts = {}
    for r in reqs:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    launch_index = 3 if h["instr"] == "events" else 1
    out = {
        "label": h["label"], "backend": h["backend"], "rate_hz": h["rate_hz"], "instr": h["instr"],
        "nvtx": h["nvtx"], "precondition": h["precondition"], "counts": counts,
        "micro": h["micro"], "init_ns": h["init_ns"], "stream_idle_at_measure_begin": h["stream_idle_at_measure_begin"],
        "issue_lag_ns": q([r["t_issue"] - r["t_sched"] for r in issued]),
        "issue_to_submit_begin_ns": q([r["t_submit_begin"] - r["t_issue"] for r in issued]),
        "submit_call_ns": q([r["t_submit_end"] - r["t_submit_begin"] for r in issued]),
        "dut_service_ns": q([r["t_complete"] - r["t_submit_begin"] for r in done]),
        "request_completion_ns": q([r["t_complete"] - r["t_sched"] for r in done]),
        "polls_per_request": q([r["polls"] for r in done]),
    }
    if h["backend"] == "cuda":
        api = [r for r in issued if r.get("api_after_ns")]
        if api:
            out["launch_call_ns"] = q([r["api_after_ns"][launch_index] - r["api_after_ns"][launch_index - 1]
                                       for r in api])
        out["in_kernel_globaltimer_ns"] = q([r["gt1"] - r["gt0"] for r in done if r["gt1"] > r["gt0"]])
        out["in_kernel_sm_mhz_x1000"] = q([round((r["c1"] - r["c0"]) * 1_000_000 / (r["gt1"] - r["gt0"]))
                                          for r in done if r["gt1"] > r["gt0"]])
        if h["instr"] == "events":
            out["event_h2d_ns"] = q([r["h2d_ns"] for r in done])
            out["event_kernel_ns"] = q([r["kernel_ns"] for r in done])
            out["event_d2h_ns"] = q([r["d2h_ns"] for r in done])
    return out


def main(out_path: str, paths: list[str]) -> None:
    table = [summarize(p) for p in paths]
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=1, sort_keys=True)
    cols = ("issue_lag_ns", "submit_call_ns", "dut_service_ns", "launch_call_ns")
    for t in table:
        cells = []
        for c in cols:
            v = t.get(c, {})
            cells.append(f"{c[:-3]} p50/p99/max {v.get('p50', '-')}/{v.get('p99', '-')}/{v.get('max', '-')}")
        print(f"{t['label']:24s} {t['counts']} | " + " | ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
