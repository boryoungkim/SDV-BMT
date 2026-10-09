"""Stage 2a-G: extract the cross-check records from an Nsight Systems SQLite export into a compact,
gzip-compressed JSON file (investigation tool; stdlib only).

    python3 tools/stage2a_g/nsys_extract.py TRACE.sqlite OUT.json.gz

Kept in full: session start, processes, CUDA driver/runtime API calls except the spin-poll
``cuEventQuery`` calls that returned "not ready", every successful ``cuEventQuery`` (the poll that
observed completion), kernels, memcpys, CUDA event records, NVTX events, string table entries used.
The not-ready polls are only summarized (count and duration quantiles): a spin-polling harness issues
millions of them, which is what makes full traces gigabytes large.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import sys


def quantiles(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    v = sorted(values)
    n = len(v)
    pick = lambda p: v[min(n - 1, max(0, -(-p * n // 100) - 1))]  # noqa: E731 nearest rank
    return {"n": n, "min": v[0], "p50": pick(50), "p90": pick(90), "p99": pick(99), "max": v[-1]}


def rows(c: sqlite3.Connection, sql: str) -> list[dict]:
    cur = c.execute(sql)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


def main(src: str, dst: str) -> None:
    c = sqlite3.connect(src)
    tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    names = dict(c.execute("select id, value from StringIds").fetchall())
    poll_id = next((k for k, v in names.items() if v == "cuEventQuery"), None)
    out: dict = {"source": src, "tables": sorted(t for t in tables if not t.startswith("ENUM_"))}
    out["session_start"] = rows(c, "select * from TARGET_INFO_SESSION_START_TIME")
    out["processes"] = rows(c, "select * from PROCESSES") if "PROCESSES" in tables else []
    api = rows(c, "select start, end, globalTid, correlationId, nameId, returnValue from CUPTI_ACTIVITY_KIND_RUNTIME"
                  + (f" where nameId != {poll_id} or returnValue = 0" if poll_id is not None else ""))
    for a in api:
        a["name"] = names.get(a.pop("nameId"))
    out["api"] = api
    if poll_id is not None:
        durations = [e - s for s, e in c.execute(
            f"select start, end from CUPTI_ACTIVITY_KIND_RUNTIME where nameId = {poll_id} and returnValue != 0")]
        out["poll_not_ready_summary"] = {"count": len(durations), "duration_ns": quantiles(durations)}
    out["kernels"] = rows(c, "select start, end, deviceId, contextId, streamId, correlationId, globalPid "
                             "from CUPTI_ACTIVITY_KIND_KERNEL")
    out["memcpy"] = rows(c, "select start, end, deviceId, contextId, streamId, correlationId, bytes, copyKind, "
                            "srcKind, dstKind from CUPTI_ACTIVITY_KIND_MEMCPY")
    if "CUPTI_ACTIVITY_KIND_CUDA_EVENT" in tables:
        out["cuda_events"] = rows(c, "select * from CUPTI_ACTIVITY_KIND_CUDA_EVENT")
    if "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION" in tables:
        out["synchronization_count"] = c.execute("select count(*) from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION").fetchone()[0]
    if "NVTX_EVENTS" in tables:
        nvtx = rows(c, "select start, end, eventType, rangeId, text, textId, domainId, globalTid, int64Value "
                       "from NVTX_EVENTS")
        for e in nvtx:
            if e["textId"] is not None:
                e["text"] = names.get(e["textId"])
        out["nvtx"] = nvtx
    if "PROFILER_OVERHEAD" in tables:
        out["profiler_overhead_summary"] = quantiles(
            [e - s for s, e in c.execute("select start, end from PROFILER_OVERHEAD")])
    if "DIAGNOSTIC_EVENT" in tables:
        out["diagnostics"] = rows(c, "select * from DIAGNOSTIC_EVENT")
    with gzip.open(dst, "wt", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"), sort_keys=True)
    print(dst, "api", len(api), "kernels", len(out["kernels"]), "memcpy", len(out["memcpy"]),
          "nvtx", len(out.get("nvtx", [])), "polls_not_ready", out.get("poll_not_ready_summary", {}).get("count"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
