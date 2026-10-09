"""Stage 2a-G: align a C++ probe run (host CLOCK_MONOTONIC facts) with its Nsight Systems trace and
cross-check request identity, containment and durations (investigation tool; stdlib only).

    python3 tools/stage2a_g/nsys_align.py PROBE.jsonl TRACE.extract.json.gz OUT.json

Method (no visual alignment):
1. Request identity.  (a) NVTX: each measured request pushes an ``sdv-bmt.submit`` range whose int64
   payload is the request id; CUDA API calls of the same thread inside the range belong to it and
   their correlation ids select its memcpy/kernel activities.  (b) Host bracket: after the clock
   mapping (2), the API calls whose mapped interval lies inside the host [t_submit_begin,
   t_submit_end] bracket of a request belong to it.  (c) Order (k-th launch = k-th request) is
   reported only to show how it breaks when the profiler loses records.
2. Clock mapping.  nsys timestamps are ns relative to the session; the probe's are CLOCK_MONOTONIC.
   For every request, offset_i = (nsys time of its first CUDA API call) - (host t_submit_begin, read
   immediately before that call).  offset = min_i offset_i; the spread of offset_i - offset is the host
   dispatch delay plus profiler cost.  Drift is checked by the offset of the first and last 10 % of
   requests, and against CLOCK_MONOTONIC_RAW by the raw-monotonic slew recorded by the probe.
3. Containment (validation of the mapping): mapped API calls of request i lie inside its host submit
   bracket; the cuEventQuery call that returned success ends before the host t_complete; device
   activities start after their enqueueing call started; D2H ends before the successful poll ends.
4. Durations: CUDA-event brackets vs nsys memcpy/kernel durations; in-kernel %globaltimer delta vs
   the nsys kernel duration; host submit duration vs the nsys API span.
"""

from __future__ import annotations

import bisect
import gzip
import json
import sys
from collections import defaultdict


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    v = sorted(values)
    n = len(v)
    pick = lambda p: v[min(n - 1, max(0, -(-p * n // 100) - 1))]  # noqa: E731
    return {"n": n, "min": v[0], "p01": pick(1), "p50": pick(50), "p90": pick(90), "p99": pick(99), "max": v[-1]}


def load_probe(path: str) -> tuple[dict, list[dict]]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return rows[0], rows[1:]


def main(probe_path: str, extract_path: str, out_path: str) -> None:
    header, reqs = load_probe(probe_path)
    x = json.load(gzip.open(extract_path, "rt", encoding="utf-8"))
    anchor = header["anchor_monotonic_ns"]
    issued = [r for r in reqs if r["t_submit_begin"] >= 0]
    issued.sort(key=lambda r: r["t_submit_begin"])
    api = sorted(x["api"], key=lambda a: a["start"])
    submit_names = {"cuEventRecord", "cuMemcpyHtoDAsync_v2", "cuLaunchKernel", "cuMemcpyDtoHAsync_v2"}
    sub_api = [a for a in api if a["name"] in submit_names]
    polls_ok = [a for a in api if a["name"] == "cuEventQuery" and a["returnValue"] == 0]
    kernels = {k["correlationId"]: k for k in x["kernels"]}
    memcpy = {m["correlationId"]: m for m in x["memcpy"]}
    result: dict = {"probe": header.get("label"), "probe_header": header, "requests_issued": len(issued),
                    "requests_dropped": sum(1 for r in reqs if r["state"] == "dropped"),
                    "trace_counts": {"submit_api": len(sub_api), "kernels": len(kernels), "memcpy": len(memcpy),
                                     "successful_polls": len(polls_ok), "nvtx": len(x.get("nvtx", []))},
                    "poll_not_ready_summary": x.get("poll_not_ready_summary"),
                    "profiler_overhead_summary": x.get("profiler_overhead_summary")}

    # ---- 2. clock mapping from the first submit API call of each request
    # Calls of one request form a burst (gaps < 1 ms; requests are milliseconds apart).  A coarse offset
    # from the session start (UTC) and the probe's CLOCK_REALTIME/CLOCK_MONOTONIC pair pairs each burst
    # with the nearest host t_submit_begin, so a lost record never shifts the pairing.
    bursts: list[list[dict]] = []
    for a in sub_api:
        if bursts and a["start"] - bursts[-1][-1]["end"] < 1_000_000:
            bursts[-1].append(a)
        else:
            bursts.append([a])
    result["api_bursts"] = len(bursts)
    cb = header["clocks_begin"]
    coarse = (cb["realtime_ns"] - cb["monotonic_ns"]) - x["session_start"][0]["utcEpochNs"]
    begins_abs = [anchor + r["t_submit_begin"] for r in issued]
    raw_offsets, coarse_err = [], []
    for burst in bursts:
        mono_guess = burst[0]["start"] - coarse
        k = bisect.bisect_left(begins_abs, mono_guess)
        best = min((j for j in (k - 1, k) if 0 <= j < len(begins_abs)), key=lambda j: abs(begins_abs[j] - mono_guess),
                   default=None)
        if best is None or abs(begins_abs[best] - mono_guess) > 3_000_000:
            continue  # precondition bursts (before t0) and unmatched bursts
        coarse_err.append(mono_guess - begins_abs[best])
        raw_offsets.append(burst[0]["start"] - begins_abs[best])
    result["burst_pairing"] = {"paired": len(raw_offsets), "issued": len(issued),
                               "coarse_offset_error_ns": quantiles(coarse_err)}
    offset = min(raw_offsets) if raw_offsets else None
    result["clock"] = {"offset_estimate_ns": offset}
    if offset is not None:
        resid = [o - offset for o in raw_offsets]
        n10 = max(1, len(resid) // 10)
        result["clock"].update({
            "anchor": "first CUDA API call of each request (nsys) vs host t_submit_begin read immediately before it",
            "residual_ns": quantiles(resid),
            "offset_first_10pct_min_minus_global_ns": min(raw_offsets[:n10]) - offset,
            "offset_last_10pct_min_minus_global_ns": min(raw_offsets[-n10:]) - offset,
            "span_ns": begins_abs[-1] - begins_abs[0],
        })
        ce = header["clocks_end"]
        result["clock"]["monotonic_raw_minus_monotonic_slew_over_run_ns"] = (
            (ce["monotonic_raw_ns"] - ce["monotonic_ns"]) - (cb["monotonic_raw_ns"] - cb["monotonic_ns"]))

    def to_host(t: int) -> int:  # nsys session time -> run clock (ns since the probe anchor)
        return t - offset - anchor

    # ---- 1. identity: NVTX, host bracket, order
    identity: dict = {}
    nvtx = x.get("nvtx", [])
    ranges = [e for e in nvtx if e["eventType"] == 59 and e.get("text") == "sdv-bmt.submit"]
    marks = defaultdict(list)
    for e in nvtx:
        if e["eventType"] == 34:
            marks[e.get("text")].append(e)
    by_id = {r["id"]: r for r in reqs}
    if ranges:
        api_by_tid = defaultdict(list)
        for a in sub_api:
            api_by_tid[a["globalTid"]].append(a)
        ok = mismatched = 0
        missing_kernel = []
        nvtx_offsets = []
        for e in ranges:
            rid = e["int64Value"]
            calls = [a for a in api_by_tid[e["globalTid"]] if e["start"] <= a["start"] and a["end"] <= e["end"]]
            corr = {a["correlationId"] for a in calls}
            has_k = any(c in kernels for c in corr)
            nb = sorted(memcpy[c]["bytes"] for c in corr if c in memcpy)
            if has_k and nb == [112, 3145728]:
                ok += 1
            else:
                mismatched += 1
                missing_kernel.append({"request_id": rid, "calls": len(calls), "kernel": has_k, "memcpy_bytes": nb})
            r = by_id.get(rid)
            if r is not None and r["t_submit_begin"] >= 0:
                nvtx_offsets.append(e["start"] - (anchor + r["t_submit_begin"]))
        identity["nvtx"] = {"ranges": len(ranges), "complete_with_h2d_kernel_d2h": ok, "incomplete": mismatched,
                            "incomplete_examples": missing_kernel[:10],
                            "marks": {k: len(v) for k, v in marks.items()},
                            "nvtx_range_start_minus_host_submit_begin_ns": quantiles(nvtx_offsets)}
        if offset is not None and nvtx_offsets:
            identity["nvtx"]["nvtx_offset_minus_api_offset_ns"] = quantiles([o - offset for o in nvtx_offsets])
    if offset is not None:
        # host bracket assignment, independent of order and of NVTX
        begins = [r["t_submit_begin"] for r in issued]
        assigned = defaultdict(list)
        outside = []
        for a in sub_api:
            s, e = to_host(a["start"]), to_host(a["end"])
            k = bisect.bisect_right(begins, s) - 1
            if k >= 0 and issued[k]["t_submit_begin"] <= s and e <= issued[k]["t_submit_end"]:
                assigned[issued[k]["id"]].append(a)
            elif k >= 0 or s >= 0:
                outside.append({"name": a["name"], "host_start": s, "host_end": e})
        complete = 0
        incomplete = []
        for r in issued:
            calls = assigned.get(r["id"], [])
            corr = {a["correlationId"] for a in calls}
            nb = sorted(memcpy[c]["bytes"] for c in corr if c in memcpy)
            if any(c in kernels for c in corr) and nb == [112, 3145728]:
                complete += 1
            else:
                incomplete.append({"request_id": r["id"], "calls": len(calls), "memcpy_bytes": nb})
        identity["host_bracket"] = {"requests_complete": complete, "requests_incomplete": len(incomplete),
                                    "incomplete_examples": incomplete[:10],
                                    "api_calls_outside_any_bracket_after_t0": len([o for o in outside
                                                                                  if o["host_start"] >= 0])}
        result["identity"] = identity

        # ---- 3. containment and 4. durations, per request (host-bracket assignment)
        poll_starts = [p["start"] for p in polls_ok]
        rows = []
        for r in issued:
            calls = assigned.get(r["id"], [])
            if not calls:
                continue
            corr = {a["correlationId"]: a for a in calls}
            h2d = next((memcpy[c] for c in corr if c in memcpy and memcpy[c]["bytes"] == 3145728), None)
            d2h = next((memcpy[c] for c in corr if c in memcpy and memcpy[c]["bytes"] == 112), None)
            ker = next((kernels[c] for c in corr if c in kernels), None)
            j = bisect.bisect_left(poll_starts, max(a["end"] for a in calls))
            poll = polls_ok[j] if j < len(polls_ok) else None
            row = {"id": r["id"],
                   "api_span_ns": max(a["end"] for a in calls) - min(a["start"] for a in calls),
                   "host_submit_ns": r["t_submit_end"] - r["t_submit_begin"]}
            if h2d and ker and d2h:
                row.update({
                    "nsys_h2d_ns": h2d["end"] - h2d["start"], "nsys_kernel_ns": ker["end"] - ker["start"],
                    "nsys_d2h_ns": d2h["end"] - d2h["start"],
                    "h2d_start_after_its_call_ns": h2d["start"] - corr[h2d["correlationId"]]["start"],
                    "kernel_start_after_launch_call_ns": ker["start"] - corr[ker["correlationId"]]["start"],
                    "event_h2d_ns": r["h2d_ns"], "event_kernel_ns": r["kernel_ns"], "event_d2h_ns": r["d2h_ns"],
                    "in_kernel_globaltimer_ns": r["gt1"] - r["gt0"] if r["gt1"] > r["gt0"] else None,
                })
                if poll is not None:
                    row.update({
                        "success_poll_end_minus_d2h_end_ns": poll["end"] - d2h["end"],
                        "host_complete_minus_mapped_d2h_end_ns": r["t_complete"] - to_host(d2h["end"]),
                        "host_complete_minus_mapped_poll_end_ns": r["t_complete"] - to_host(poll["end"]),
                    })
            rows.append(row)

        def col(name: str) -> list:
            return [row[name] for row in rows if row.get(name) is not None]

        def diff(a: str, b: str) -> list:
            return [row[a] - row[b] for row in rows if row.get(a) is not None and row.get(b) is not None
                    and row[a] >= 0 and row[b] >= 0]

        result["containment"] = {
            "requests_with_all_device_activities": len(col("nsys_kernel_ns")),
            "h2d_start_after_its_call_ns": quantiles(col("h2d_start_after_its_call_ns")),
            "kernel_start_after_launch_call_ns": quantiles(col("kernel_start_after_launch_call_ns")),
            "success_poll_end_minus_d2h_end_ns": quantiles(col("success_poll_end_minus_d2h_end_ns")),
            "host_complete_minus_mapped_poll_end_ns": quantiles(col("host_complete_minus_mapped_poll_end_ns")),
            "host_complete_minus_mapped_d2h_end_ns": quantiles(col("host_complete_minus_mapped_d2h_end_ns")),
            "violations": {
                "device_activity_before_its_call": sum(1 for v in col("h2d_start_after_its_call_ns") if v < 0)
                + sum(1 for v in col("kernel_start_after_launch_call_ns") if v < 0),
                "d2h_end_after_successful_poll_end": sum(1 for v in col("success_poll_end_minus_d2h_end_ns") if v < 0),
                "host_complete_before_mapped_poll_end": sum(1 for v in col("host_complete_minus_mapped_poll_end_ns")
                                                            if v < 0),
            },
        }
        result["durations"] = {
            "event_minus_nsys_h2d_ns": quantiles(diff("event_h2d_ns", "nsys_h2d_ns")),
            "event_minus_nsys_kernel_ns": quantiles(diff("event_kernel_ns", "nsys_kernel_ns")),
            "event_minus_nsys_d2h_ns": quantiles(diff("event_d2h_ns", "nsys_d2h_ns")),
            "nsys_kernel_minus_in_kernel_globaltimer_ns": quantiles(diff("nsys_kernel_ns", "in_kernel_globaltimer_ns")),
            "host_submit_minus_nsys_api_span_ns": quantiles(diff("host_submit_ns", "api_span_ns")),
            "nsys_h2d_ns": quantiles(col("nsys_h2d_ns")), "nsys_kernel_ns": quantiles(col("nsys_kernel_ns")),
            "nsys_d2h_ns": quantiles(col("nsys_d2h_ns")),
        }
    # order-based mapping, shown only as a fragility check
    launches = [a for a in sub_api if a["name"] == "cuLaunchKernel"]
    result["order_mapping"] = {"launch_records": len(launches),
                               "expected_launches": header.get("precondition", 0) + len(issued),
                               "usable": len(launches) == header.get("precondition", 0) + len(issued)}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
    print(json.dumps({k: result[k] for k in ("probe", "requests_issued", "trace_counts", "api_bursts",
                                              "burst_pairing", "order_mapping")}, indent=1))
    print(json.dumps(result.get("clock"), indent=1))
    print(json.dumps(result.get("identity"), indent=1)[:3000])
    print(json.dumps(result.get("containment"), indent=1))
    print(json.dumps(result.get("durations"), indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
