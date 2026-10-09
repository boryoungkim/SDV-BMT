#!/usr/bin/env python3
"""Extract the cross-check records of a multi-kernel (cuDNN plan) run from an Nsight Systems SQLite export
(offline, stdlib only).  As tools/stage2a_g/nsys_extract.py, plus kernel names and launch geometry, which
identify the cuDNN kernels that actually ran (the realized engine), and every CUDA API call except the
not-ready spin polls (summarized only).

    python3 tools/stage2b_g_real/nsys_extract_plan.py TRACE.sqlite OUT.json.gz
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
    api = rows(c, "select start, end, globalTid, correlationId, nameId, returnValue from CUPTI_ACTIVITY_KIND_RUNTIME"
                  + (f" where nameId != {poll_id} or returnValue = 0" if poll_id is not None else ""))
    for a in api:
        a["name"] = names.get(a.pop("nameId"))
    out["api"] = api
    if poll_id is not None:
        durations = [e - s for s, e in c.execute(
            f"select start, end from CUPTI_ACTIVITY_KIND_RUNTIME where nameId = {poll_id} and returnValue != 0")]
        out["poll_not_ready_summary"] = {"count": len(durations), "duration_ns": quantiles(durations)}
    kernels = rows(c, "select start, end, deviceId, contextId, streamId, correlationId, globalPid, shortName, "
                      "gridX, gridY, gridZ, blockX, blockY, blockZ, registersPerThread, staticSharedMemory, "
                      "dynamicSharedMemory from CUPTI_ACTIVITY_KIND_KERNEL")
    for k in kernels:
        k["name"] = names.get(k.pop("shortName"))
    out["kernels"] = kernels
    out["memcpy"] = rows(c, "select start, end, deviceId, contextId, streamId, correlationId, bytes, copyKind, "
                            "srcKind, dstKind from CUPTI_ACTIVITY_KIND_MEMCPY")
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        out["memset"] = rows(c, "select start, end, streamId, correlationId, bytes from CUPTI_ACTIVITY_KIND_MEMSET")
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
    with gzip.open(dst, "wt", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"), sort_keys=True)
    print(dst, "api", len(api), "kernels", len(kernels), "memcpy", len(out["memcpy"]),
          "nvtx", len(out.get("nvtx", [])), "polls_not_ready", out.get("poll_not_ready_summary", {}).get("count"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
