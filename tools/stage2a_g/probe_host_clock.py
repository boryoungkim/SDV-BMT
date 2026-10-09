"""Stage 2a-G probe 2: host clock and scheduling fidelity (no GPU).

Measures what bounds a realtime issuer on this host: clock resolution, sleep
overshoot, busy-wait overshoot and descheduling gaps, unpinned and pinned to the
GPU-local NUMA node.  Usage: python tools/stage2a_g/probe_host_clock.py <out.json>
"""

from __future__ import annotations

import json
import os
import sys
import time

NODE0_CPUS = set(range(0, 16)) | set(range(32, 48))  # GPU 0 NUMA affinity (nvidia-smi topo)


def quantiles(values: list[int]) -> dict:
    v = sorted(values)
    n = len(v)

    def q(p: float) -> int:  # nearest rank
        return v[min(n - 1, max(0, int(-(-p * n // 1)) - 1))]

    return {"n": n, "min": v[0], "p50": q(0.50), "p90": q(0.90), "p99": q(0.99), "max": v[-1]}


def sleep_overshoot(request_ns: int, n: int) -> dict:
    out = []
    for _ in range(n):
        t0 = time.monotonic_ns()
        time.sleep(request_ns / 1e9)
        out.append(time.monotonic_ns() - t0 - request_ns)
    return quantiles(out)


def busy_overshoot(request_ns: int, n: int) -> dict:
    out = []
    for _ in range(n):
        target = time.monotonic_ns() + request_ns
        while (now := time.monotonic_ns()) < target:
            pass
        out.append(now - target)
    return quantiles(out)


def hybrid_overshoot(request_ns: int, spin_ns: int, n: int) -> dict:
    """Sleep until spin_ns before the target, then spin."""
    out = []
    for _ in range(n):
        target = time.monotonic_ns() + request_ns
        coarse = target - spin_ns - time.monotonic_ns()
        if coarse > 0:
            time.sleep(coarse / 1e9)
        while (now := time.monotonic_ns()) < target:
            pass
        out.append(now - target)
    return quantiles(out)


def gaps(duration_s: float, threshold_ns: int = 50_000) -> dict:
    """Tight loop on the monotonic clock; a gap above threshold means the thread did not run."""
    end = time.monotonic_ns() + int(duration_s * 1e9)
    last = time.monotonic_ns()
    big: list[int] = []
    reads = 0
    while last < end:
        now = time.monotonic_ns()
        reads += 1
        if now - last > threshold_ns:
            big.append(now - last)
        last = now
    total = sum(big)
    return {"duration_s": duration_s, "reads": reads, "gaps_over_50us": len(big),
            "gap_time_fraction": round(total / (duration_s * 1e9), 6),
            "gap_quantiles_ns": quantiles(big) if big else None}


def suite(label: str) -> dict:
    return {
        "label": label,
        "affinity": sorted(os.sched_getaffinity(0)),
        "loadavg_before": open("/proc/loadavg").read().split()[:3],
        "sleep_overshoot_ns": {str(r): sleep_overshoot(r, 400) for r in (100_000, 1_000_000, 5_000_000)},
        "busy_overshoot_ns": {str(r): busy_overshoot(r, 2000) for r in (100_000, 1_000_000)},
        "hybrid_overshoot_ns": {"1000000_spin200000": hybrid_overshoot(1_000_000, 200_000, 400)},
        "descheduling_gaps": gaps(3.0),
        "loadavg_after": open("/proc/loadavg").read().split()[:3],
    }


def main(path: str) -> None:
    res = {}
    for name in ("monotonic", "perf_counter", "monotonic_raw" if hasattr(time, "CLOCK_MONOTONIC_RAW") else None):
        if name is None:
            continue
        clock_id = {"monotonic": time.CLOCK_MONOTONIC, "perf_counter": time.CLOCK_MONOTONIC,
                    "monotonic_raw": getattr(time, "CLOCK_MONOTONIC_RAW", None)}[name]
        info = time.get_clock_info("monotonic" if name != "perf_counter" else "perf_counter")
        res[name] = {"implementation": info.implementation, "resolution_s": info.resolution,
                     "clock_getres_ns": int(time.clock_getres(clock_id) * 1e9)}
    reads = []
    for _ in range(20000):
        a = time.monotonic_ns()
        b = time.monotonic_ns()
        reads.append(b - a)
    res["monotonic_ns_read_cost"] = quantiles(reads)
    original = os.sched_getaffinity(0)
    result = {"clocks": res, "runs": [suite("unpinned")]}
    os.sched_setaffinity(0, NODE0_CPUS & original)
    try:
        result["runs"].append(suite("pinned_numa_node0"))
    finally:
        os.sched_setaffinity(0, original)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    for r in result["runs"]:
        print(r["label"], "sleep1ms p50/p99/max", [r["sleep_overshoot_ns"]["1000000"][k] for k in ("p50", "p99", "max")],
              "busy1ms p99/max", [r["busy_overshoot_ns"]["1000000"][k] for k in ("p99", "max")],
              "gaps", r["descheduling_gaps"]["gaps_over_50us"], r["descheduling_gaps"]["gap_time_fraction"],
              (r["descheduling_gaps"]["gap_quantiles_ns"] or {}).get("max"))


if __name__ == "__main__":
    main(sys.argv[1])
