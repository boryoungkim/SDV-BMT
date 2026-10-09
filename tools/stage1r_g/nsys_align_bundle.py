"""Stage 1R-G / 2b-G: cross-check an SDV-BMT realtime run bundle against its Nsight Systems trace
(analysis plane, stdlib only).  The Stage 2a-G method (v02 report section 15.2) applied to production
bundles of the native runner instead of probe facts.

    python3 tools/stage1r_g/nsys_align_bundle.py RUN_DIR TRACE.extract.json.gz OUT.json

RUN_DIR is a bundle produced under ``nsys profile --trace=cuda,nvtx`` with a DUT config declaring
``nvtx: true, profiler: nsys``; the extract comes from ``tools/stage2a_g/nsys_extract.py``.

1. Clock mapping (no visual alignment).  nsys timestamps are ns since the session start; bundle times are
   run-clock ns (CLOCK_MONOTONIC - anchor; the anchor pair is in manifest.run_clock).  Each request's
   CUDA API calls form one burst; a coarse offset (session start UTC vs the anchor's CLOCK_REALTIME /
   CLOCK_MONOTONIC pair) pairs each burst with the nearest host t_submit_begin, then
   offset_i = nsys(first API call start) - host(t_submit_begin_i), offset = min_i offset_i.  Drift: the
   minimum offset of the first and the last 10 % of requests.  (Bundles record no CLOCK_MONOTONIC_RAW
   pair, so the RAW-slew check of Stage 2a-G is not repeated here.)
2. Identity, two independent mappings: (a) NVTX ``sdv-bmt.submit`` range payload = request id -> the API
   calls of that thread inside the range -> correlation ids -> memcpy/kernel activities; (b) host submit
   bracket [t_submit_begin, t_submit_end] after the mapping.  They must agree; an order-based mapping is
   never used.  ``sdv-bmt.drop`` / ``sdv-bmt.not_issued`` marks are compared with the DROPPED /
   NOT_ISSUED request ids of the bundle.
3. Containment (validates the mapping): mapped API calls inside their host bracket; device activities
   start after their enqueueing call started; D2H ends before the successful cuEventQuery ends; the host
   t_complete is not before the mapped end of that poll.
4. Durations: CUDA-event brackets (device clock, from device_observation events) vs nsys memcpy/kernel
   durations; in-kernel %globaltimer vs nsys kernel duration; host submit duration vs nsys API span.
"""

from __future__ import annotations

import bisect
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bundle_facts import load_run, quantiles  # noqa: E402

SUBMIT_API = {"cuEventRecord", "cuMemcpyHtoDAsync_v2", "cuLaunchKernel", "cuMemcpyDtoHAsync_v2", "cuMemAlloc_v2"}
NVTX_RANGE, NVTX_MARK = 59, 34  # NVTX_EVENTS.eventType: push/pop range, mark


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
    dropped = sorted(r.request_id for r in run.requests.values() if r.outcome == "dropped")
    not_issued = sorted(r.request_id for r in run.requests.values() if r.outcome == "not_issued")
    api = sorted(x["api"], key=lambda a: a["start"])
    sub_api = [a for a in api if a["name"] in SUBMIT_API]
    polls_ok = [a for a in api if a["name"] == "cuEventQuery" and a["returnValue"] == 0]
    kernels = {k["correlationId"]: k for k in x["kernels"]}
    memcpy = {m["correlationId"]: m for m in x["memcpy"]}
    result: dict = {
        "run_dir": run_dir, "run_id": run.run_id, "experiment_id": run.experiment_id,
        "requests": {"issued": len(issued), "dropped": len(dropped), "not_issued": len(not_issued)},
        "precondition_requests": (run.manifest.get("realtime") or {}).get("precondition_requests"),
        "trace_counts": {"submit_api": len(sub_api), "kernels": len(kernels), "memcpy": len(memcpy),
                         "successful_polls": len(polls_ok), "nvtx": len(x.get("nvtx", []))},
        "poll_not_ready_summary": x.get("poll_not_ready_summary"),
        "profiler_overhead_summary": x.get("profiler_overhead_summary"),
        "expected_bytes": {"h2d": h2d_bytes, "d2h": d2h_bytes},
    }

    # ---- 1. clock mapping
    bursts: list[list[dict]] = []
    for a in sub_api:
        if bursts and a["start"] - bursts[-1][-1]["end"] < 1_000_000:
            bursts[-1].append(a)
        else:
            bursts.append([a])
    coarse = (clock["anchor_realtime_ns"] - clock["anchor_monotonic_ns"]) - x["session_start"][0]["utcEpochNs"]
    begins_abs = [anchor + r.t["t_submit_begin"] for r in issued]
    raw_offsets, coarse_err = [], []
    for burst in bursts:
        guess = burst[0]["start"] - coarse
        k = bisect.bisect_left(begins_abs, guess)
        best = min((j for j in (k - 1, k) if 0 <= j < len(begins_abs)), key=lambda j: abs(begins_abs[j] - guess),
                   default=None)
        if best is None or abs(begins_abs[best] - guess) > 3_000_000:
            continue  # precondition bursts (before MEASURE) and unmatched bursts
        coarse_err.append(guess - begins_abs[best])
        raw_offsets.append(burst[0]["start"] - begins_abs[best])
    result["api_bursts"] = len(bursts)
    result["burst_pairing"] = {"paired": len(raw_offsets), "issued": len(issued), "coarse_offset_error_ns": q(coarse_err)}
    offset = min(raw_offsets) if raw_offsets else None
    result["clock"] = {"offset_estimate_ns": offset}
    if offset is None:
        Path(out_path).write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
        print("no burst could be paired with a host t_submit_begin")
        return
    resid = [o - offset for o in raw_offsets]
    n10 = max(1, len(resid) // 10)
    result["clock"].update({
        "anchor": "first CUDA API call of each request (nsys) vs host t_submit_begin read immediately before it",
        "residual_ns": q(resid),
        "offset_first_10pct_min_minus_global_ns": min(raw_offsets[:n10]) - offset,
        "offset_last_10pct_min_minus_global_ns": min(raw_offsets[-n10:]) - offset,
        "span_ns": begins_abs[-1] - begins_abs[0],
    })

    def to_host(t: int) -> int:  # nsys session time -> run clock
        return t - offset - anchor

    # ---- 2. identity
    identity: dict = {}
    nvtx = x.get("nvtx", [])
    ranges = [e for e in nvtx if e["eventType"] == NVTX_RANGE and e.get("text") == "sdv-bmt.submit"]
    marks = defaultdict(list)
    for e in nvtx:
        if e["eventType"] == NVTX_MARK:
            marks[e.get("text")].append(e["int64Value"])
    api_by_tid = defaultdict(list)
    for a in sub_api:
        api_by_tid[a["globalTid"]].append(a)

    def activities(corr: set[int]) -> tuple[bool, list[int]]:
        return any(c in kernels for c in corr), sorted(memcpy[c]["bytes"] for c in corr if c in memcpy)

    h2d_included = run.manifest["measurement_boundary"]["definition"]["prepare_input_h2d"] == "included"
    expected_bytes = sorted([h2d_bytes, d2h_bytes]) if h2d_included else [d2h_bytes]
    nvtx_map: dict[int, set[int]] = {}
    measured_ranges = [e for e in ranges if e["int64Value"] >= 0]  # precondition requests carry negative ids
    nvtx_complete = 0
    for e in measured_ranges:
        calls = [a for a in api_by_tid[e["globalTid"]] if e["start"] <= a["start"] and a["end"] <= e["end"]]
        corr = {a["correlationId"] for a in calls}
        nvtx_map[e["int64Value"]] = corr
        has_k, nb = activities(corr)
        nvtx_complete += has_k and nb == expected_bytes
    begins = [r.t["t_submit_begin"] for r in issued]
    assigned: dict[int, list[dict]] = defaultdict(list)
    outside_after_t0 = 0
    for a in sub_api:
        s, e = to_host(a["start"]), to_host(a["end"])
        k = bisect.bisect_right(begins, s) - 1
        if k >= 0 and issued[k].t["t_submit_begin"] <= s and e <= issued[k].t["t_submit_end"]:
            assigned[issued[k].request_id].append(a)
        elif s >= begins[0]:
            outside_after_t0 += 1
    bracket_complete = 0
    agree = disagree = 0
    disagreements = []
    for r in issued:
        corr = {a["correlationId"] for a in assigned.get(r.request_id, [])}
        has_k, nb = activities(corr)
        bracket_complete += has_k and nb == expected_bytes
        if r.request_id in nvtx_map:
            if nvtx_map[r.request_id] == corr:
                agree += 1
            else:
                disagree += 1
                disagreements.append(r.request_id)
    identity["nvtx"] = {"measured_ranges": len(measured_ranges), "precondition_ranges": len(ranges) - len(measured_ranges),
                        "complete_with_expected_activities": nvtx_complete,
                        "marks": {k: len(v) for k, v in marks.items()}}
    identity["host_bracket"] = {"requests_complete": bracket_complete, "requests_incomplete": len(issued) - bracket_complete,
                                "submit_api_calls_outside_any_bracket_after_first_submit": outside_after_t0}
    identity["nvtx_vs_bracket"] = {"agree": agree, "disagree": disagree, "disagreeing_request_ids": disagreements[:20]}
    identity["admission_marks"] = {
        "drop_mark_ids_equal_dropped_ids": sorted(i for i in marks.get("sdv-bmt.drop", []) if i >= 0) == dropped,
        "not_issued_mark_ids_equal_not_issued_ids": sorted(marks.get("sdv-bmt.not_issued", [])) == not_issued,
        "dropped": len(dropped), "not_issued": len(not_issued)}
    completes = sorted(i for i in marks.get("sdv-bmt.complete", []) if i >= 0)
    identity["completion_marks_equal_completed_or_late_ids"] = completes == sorted(
        r.request_id for r in issued if r.outcome == "completed" or r.late is not None)
    result["identity"] = identity

    # ---- 3. containment, 4. durations (host-bracket assignment)
    poll_starts = [p["start"] for p in polls_ok]
    rows = []
    for r in issued:
        calls = assigned.get(r.request_id, [])
        if not calls:
            continue
        corr = {a["correlationId"]: a for a in calls}
        h2d = next((memcpy[c] for c in corr if c in memcpy and memcpy[c]["bytes"] == h2d_bytes and memcpy[c]["copyKind"] == 1), None)
        d2h = next((memcpy[c] for c in corr if c in memcpy and memcpy[c]["bytes"] == d2h_bytes and memcpy[c]["copyKind"] == 2), None)
        ker = next((kernels[c] for c in corr if c in kernels), None)
        j = bisect.bisect_left(poll_starts, max(a["end"] for a in calls))
        poll = polls_ok[j] if j < len(polls_ok) else None
        row = {"id": r.request_id, "api_span_ns": max(a["end"] for a in calls) - min(a["start"] for a in calls),
               "host_submit_ns": r.submit}
        if ker and d2h:
            row.update({
                "nsys_kernel_ns": ker["end"] - ker["start"], "nsys_d2h_ns": d2h["end"] - d2h["start"],
                "kernel_start_after_launch_call_ns": ker["start"] - corr[ker["correlationId"]]["start"],
                "event_kernel_ns": r.segment("device_execution"), "event_d2h_ns": r.segment("device_to_host"),
                "in_kernel_globaltimer_ns": r.in_kernel_ns,
            })
            if h2d:
                row.update({"nsys_h2d_ns": h2d["end"] - h2d["start"],
                            "h2d_start_after_its_call_ns": h2d["start"] - corr[h2d["correlationId"]]["start"],
                            "event_h2d_ns": r.segment("host_to_device")})
            if poll is not None and "t_complete" in r.t:
                row.update({"success_poll_end_minus_d2h_end_ns": poll["end"] - d2h["end"],
                            "host_complete_minus_mapped_poll_end_ns": r.t["t_complete"] - to_host(poll["end"]),
                            "host_complete_minus_mapped_d2h_end_ns": r.t["t_complete"] - to_host(d2h["end"]),
                            "mapped_kernel_start_minus_host_submit_end_ns": to_host(ker["start"]) - r.t["t_submit_end"]})
        rows.append(row)

    def col(name: str) -> list:
        return [row[name] for row in rows if row.get(name) is not None]

    def diff(a: str, b: str) -> list:
        return [row[a] - row[b] for row in rows if row.get(a) is not None and row.get(b) is not None]

    result["containment"] = {
        "requests_with_kernel_and_d2h": len(col("nsys_kernel_ns")),
        "h2d_start_after_its_call_ns": q(col("h2d_start_after_its_call_ns")),
        "kernel_start_after_launch_call_ns": q(col("kernel_start_after_launch_call_ns")),
        "mapped_kernel_start_minus_host_submit_end_ns": q(col("mapped_kernel_start_minus_host_submit_end_ns")),
        "success_poll_end_minus_d2h_end_ns": q(col("success_poll_end_minus_d2h_end_ns")),
        "host_complete_minus_mapped_poll_end_ns": q(col("host_complete_minus_mapped_poll_end_ns")),
        "host_complete_minus_mapped_d2h_end_ns": q(col("host_complete_minus_mapped_d2h_end_ns")),
        "violations": {
            "device_activity_before_its_call": sum(1 for v in col("h2d_start_after_its_call_ns") if v < 0)
            + sum(1 for v in col("kernel_start_after_launch_call_ns") if v < 0),
            "d2h_end_after_successful_poll_end": sum(1 for v in col("success_poll_end_minus_d2h_end_ns") if v < 0),
            "host_complete_before_mapped_poll_end": sum(1 for v in col("host_complete_minus_mapped_poll_end_ns") if v < 0),
        },
    }
    result["durations"] = {
        "event_minus_nsys_h2d_ns": q(diff("event_h2d_ns", "nsys_h2d_ns")),
        "event_minus_nsys_kernel_ns": q(diff("event_kernel_ns", "nsys_kernel_ns")),
        "event_minus_nsys_d2h_ns": q(diff("event_d2h_ns", "nsys_d2h_ns")),
        "nsys_kernel_minus_in_kernel_globaltimer_ns": q(diff("nsys_kernel_ns", "in_kernel_globaltimer_ns")),
        "host_submit_minus_nsys_api_span_ns": q(diff("host_submit_ns", "api_span_ns")),
        "nsys_h2d_ns": q(col("nsys_h2d_ns")), "nsys_kernel_ns": q(col("nsys_kernel_ns")), "nsys_d2h_ns": q(col("nsys_d2h_ns")),
    }
    result["per_request"] = rows
    Path(out_path).write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    for key in ("requests", "trace_counts", "burst_pairing", "clock", "identity", "containment", "durations"):
        print(key, json.dumps(result[key], sort_keys=True)[:1500])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
