#!/usr/bin/env python3
"""Cross-check a cuDNN-plan (multi-kernel) run bundle against its Nsight Systems trace (analysis plane, stdlib).

The Stage 2a-G / 1R-G method (clock mapping by the first CUDA API call of each request, two independent
request-identity mappings, containment checks) generalized to requests that enqueue many kernels:

    python3 tools/stage2b_g_real/nsys_align_plan.py RUN_DIR TRACE.extract.json.gz OUT.json

1. Clock: nsys session ns -> run clock by offset = min_i(first API call of burst i - host t_submit_begin_i),
   bursts paired through the session-start/anchor wall-clock pair; residuals and early/late drift reported.
2. Identity: (a) NVTX range ``sdv-bmt.submit`` (payload = request id) -> that thread's API calls inside it ->
   correlation ids -> memcpy/kernel activities; (b) API calls inside the host submit bracket after mapping.
   Both must give the same correlation set per request (no order-based mapping).
3. Containment: every device activity starts after its enqueueing call started; the D2H copy ends before the
   successful cuEventQuery ends; host t_complete is not before that poll's mapped end.
4. Per request, on the nsys clock: pure H2D/D2H copy times; kernel count and names; kernel busy time (union of
   kernel intervals: cuDNN may fork kernels onto its own streams) and span (H2D end -> last kernel end); gaps between consecutive
   device activities, split into gaps whose kernel's launch call ended after the previous activity ended
   (the device waited for the host to enqueue: "host_enqueue_bound") and the remaining gaps; completion
   observation delay (successful poll end - D2H end).  The CUDA-event brackets of the bundle are listed next to
   these as their own (device-clock) quantities; host and device durations are never subtracted.
"""

from __future__ import annotations

import bisect
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage1r_g"))
from bundle_facts import load_run, quantiles  # noqa: E402

NVTX_RANGE, NVTX_MARK = 59, 34


def q(values) -> dict:
    return quantiles(values, (1, 50, 90, 99)) or {"n": 0}


def main(run_dir: str, extract_path: str, out_path: str) -> None:
    run = load_run(run_dir)
    x = json.load(gzip.open(extract_path, "rt", encoding="utf-8"))
    clock = run.manifest["run_clock"]
    anchor = clock["anchor_monotonic_ns"]
    program = run.run_spec["resolved_plan"]["workload"]["options"]["device_program"]
    h2d_bytes, d2h_bytes = program["input_bytes"], program["output_bytes"]
    issued = run.issued()
    api = sorted(x["api"], key=lambda a: a["start"])
    polls_ok = [a for a in api if a["name"] == "cuEventQuery" and a["returnValue"] == 0]
    enqueue_api = [a for a in api if a["name"] != "cuEventQuery"]
    kernels = {k["correlationId"]: k for k in x["kernels"]}
    memcpy = {m["correlationId"]: m for m in x["memcpy"]}
    result: dict = {"run_dir": run_dir, "run_id": run.run_id, "experiment_id": run.experiment_id,
                    "requests_issued": len(issued),
                    "trace_counts": {"api_calls_kept": len(api), "kernels": len(kernels), "memcpy": len(memcpy),
                                     "successful_polls": len(polls_ok), "nvtx": len(x.get("nvtx", []))},
                    "poll_not_ready_summary": x.get("poll_not_ready_summary"),
                    "profiler_overhead_summary": x.get("profiler_overhead_summary")}

    # ---- 1. clock mapping
    bursts: list[list[dict]] = []
    for a in enqueue_api:
        if bursts and a["start"] - bursts[-1][-1]["end"] < 1_000_000:
            bursts[-1].append(a)
        else:
            bursts.append([a])
    coarse = (clock["anchor_realtime_ns"] - clock["anchor_monotonic_ns"]) - x["session_start"][0]["utcEpochNs"]
    begins_abs = [anchor + r.t["t_submit_begin"] for r in issued]
    raw_offsets = []
    for burst in bursts:
        guess = burst[0]["start"] - coarse
        k = bisect.bisect_left(begins_abs, guess)
        best = min((j for j in (k - 1, k) if 0 <= j < len(begins_abs)), key=lambda j: abs(begins_abs[j] - guess),
                   default=None)
        if best is None or abs(begins_abs[best] - guess) > 3_000_000:
            continue
        raw_offsets.append(burst[0]["start"] - begins_abs[best])
    offset = min(raw_offsets)
    resid = [o - offset for o in raw_offsets]
    n10 = max(1, len(resid) // 10)
    result["clock"] = {"bursts": len(bursts), "paired": len(raw_offsets), "offset_estimate_ns": offset,
                       "residual_ns": q(resid),
                       "offset_first_10pct_min_minus_global_ns": min(raw_offsets[:n10]) - offset,
                       "offset_last_10pct_min_minus_global_ns": min(raw_offsets[-n10:]) - offset}

    def to_host(t: int) -> int:
        return t - offset - anchor

    # ---- 2. identity
    nvtx = x.get("nvtx", [])
    ranges = [e for e in nvtx if e["eventType"] == NVTX_RANGE and e.get("text") == "sdv-bmt.submit" and e["int64Value"] >= 0]
    by_tid = defaultdict(list)
    for a in enqueue_api:
        by_tid[a["globalTid"]].append(a)
    nvtx_map = {}
    for e in ranges:
        nvtx_map[e["int64Value"]] = {a["correlationId"] for a in by_tid[e["globalTid"]]
                                     if e["start"] <= a["start"] and a["end"] <= e["end"]}
    begins = [r.t["t_submit_begin"] for r in issued]
    assigned: dict[int, list[dict]] = defaultdict(list)
    outside = 0
    for a in enqueue_api:
        s, e = to_host(a["start"]), to_host(a["end"])
        k = bisect.bisect_right(begins, s) - 1
        if k >= 0 and begins[k] <= s and e <= issued[k].t["t_submit_end"]:
            assigned[issued[k].request_id].append(a)
        elif s >= begins[0]:
            outside += 1
    agree = sum(1 for r in issued if nvtx_map.get(r.request_id) == {a["correlationId"] for a in assigned.get(r.request_id, [])})
    result["identity"] = {"nvtx_ranges_measured": len(ranges), "nvtx_vs_bracket_agree": agree,
                          "nvtx_vs_bracket_disagree": len(issued) - agree,
                          "enqueue_calls_outside_any_bracket_after_first_submit": outside,
                          "completion_marks": sum(1 for e in nvtx if e["eventType"] == NVTX_MARK
                                                  and e.get("text") == "sdv-bmt.complete" and e["int64Value"] >= 0)}

    # ---- 3/4. containment and per-request device timeline
    poll_starts = [p["start"] for p in polls_ok]
    rows, names = [], Counter()
    violations = Counter()
    for r in issued:
        calls = {a["correlationId"]: a for a in assigned.get(r.request_id, [])}
        acts = sorted([(kernels[c], "kernel") for c in calls if c in kernels] +
                      [(memcpy[c], "memcpy") for c in calls if c in memcpy], key=lambda t: t[0]["start"])
        h2d = next((m for m, kind in acts if kind == "memcpy" and m["bytes"] == h2d_bytes and m["copyKind"] == 1), None)
        d2h = next((m for m, kind in acts if kind == "memcpy" and m["bytes"] == d2h_bytes and m["copyKind"] == 2), None)
        ks = [k for k, kind in acts if kind == "kernel"]
        if not ks or d2h is None:
            continue
        for act, _ in acts:
            if act["start"] < calls[act["correlationId"]]["start"]:
                violations["device_activity_before_its_call"] += 1
        names.update(k["name"] for k in ks)
        busy_sum = sum(k["end"] - k["start"] for k in ks)
        busy, cur_s, cur_e = 0, None, None  # union of kernel intervals (cuDNN may run kernels on its own streams)
        for k in sorted(ks, key=lambda k: k["start"]):
            if cur_e is None or k["start"] > cur_e:
                if cur_e is not None:
                    busy += cur_e - cur_s
                cur_s, cur_e = k["start"], k["end"]
            else:
                cur_e = max(cur_e, k["end"])
        busy += cur_e - cur_s
        streams = sorted({k["streamId"] for k in ks})
        enqueue_bound = other_gap = 0
        prev_end = h2d["end"] if h2d else ks[0]["start"]
        for k in sorted(ks, key=lambda k: k["start"]):
            gap = k["start"] - prev_end
            if gap > 0:
                if calls[k["correlationId"]]["end"] > prev_end:
                    enqueue_bound += gap
                else:
                    other_gap += gap
            prev_end = max(prev_end, k["end"])
        j = bisect.bisect_left(poll_starts, d2h["end"])
        poll = polls_ok[j] if j < len(polls_ok) else None
        row = {"id": r.request_id, "kernels": len(ks), "kernel_busy_ns": busy, "kernel_duration_sum_ns": busy_sum,
               "kernel_streams": len(streams),
               "span_h2d_end_to_last_kernel_end_ns": ks[-1]["end"] - (h2d["end"] if h2d else ks[0]["start"]),
               "gap_host_enqueue_bound_ns": enqueue_bound, "gap_other_ns": other_gap,
               "nsys_h2d_ns": h2d["end"] - h2d["start"] if h2d else None, "nsys_d2h_ns": d2h["end"] - d2h["start"],
               "api_span_ns": max(a["end"] for a in calls.values()) - min(a["start"] for a in calls.values()),
               "api_calls": len(calls),
               "event_h2d_ns": r.segment("host_to_device"), "event_device_execution_ns": r.segment("device_execution"),
               "event_d2h_ns": r.segment("device_to_host"), "host_submit_ns": r.submit, "host_dut_service_ns": r.dut_service,
               "last_kernel_end_minus_last_launch_call_end_ns": ks[-1]["end"] - calls[ks[-1]["correlationId"]]["end"]}
        if poll is not None and "t_complete" in r.t:
            row["success_poll_end_minus_d2h_end_ns"] = poll["end"] - d2h["end"]
            row["host_complete_minus_mapped_poll_end_ns"] = r.t["t_complete"] - to_host(poll["end"])
            if poll["end"] < d2h["end"]:
                violations["d2h_end_after_successful_poll_end"] += 1
            if r.t["t_complete"] < to_host(poll["end"]):
                violations["host_complete_before_mapped_poll_end"] += 1
        rows.append(row)

    def col(name):
        return [row[name] for row in rows if row.get(name) is not None]

    result["containment_violations"] = dict(violations) or {"none": 0}
    result["requests_with_timeline"] = len(rows)
    result["kernels_per_request"] = q(col("kernels"))
    result["kernel_names"] = dict(names.most_common())
    result["kernel_streams_per_request"] = q(col("kernel_streams"))
    result["timeline_ns"] = {k: q(col(k)) for k in (
        "kernel_busy_ns", "kernel_duration_sum_ns", "span_h2d_end_to_last_kernel_end_ns", "gap_host_enqueue_bound_ns", "gap_other_ns",
        "nsys_h2d_ns", "nsys_d2h_ns", "api_span_ns", "api_calls", "event_h2d_ns", "event_device_execution_ns",
        "event_d2h_ns", "host_submit_ns", "host_dut_service_ns", "success_poll_end_minus_d2h_end_ns",
        "host_complete_minus_mapped_poll_end_ns", "last_kernel_end_minus_last_launch_call_end_ns")}
    result["per_request"] = rows
    Path(out_path).write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    for key in ("trace_counts", "clock", "identity", "containment_violations", "kernels_per_request"):
        print(key, json.dumps(result[key], sort_keys=True)[:800])
    for k, v in result["timeline_ns"].items():
        print(f"  {k:52} p50 {v.get('p50')} p99 {v.get('p99')} max {v.get('max')}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
