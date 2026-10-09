"""Stage 2a-G probe 8: NVML cumulative counters (field values) over a known transfer window.

Usage: python tools/stage2a_g/probe_nvml_fields.py <out.json>

Reads throttle-time counters (ns) and PCIe TX/RX byte counters before and after a
known H2D/D2H burst on the target GPU and compares the deltas with the bytes moved.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from ctypes import byref, c_int, c_longlong, c_uint, c_ulonglong, c_void_p

sys.path.insert(0, os.path.dirname(__file__))
import gpuprobe as g  # noqa: E402

FIELDS = {
    74: "throttle_ns_sw_power_cap",
    76: "throttle_ns_sync_boost",
    251: "throttle_ns_sw_thermal_slowdown",
    252: "throttle_ns_hw_thermal_slowdown",
    197: "pcie_tx_bytes",
    198: "pcie_rx_bytes",
    94: "pcie_replay_counter",
}


class nvmlFieldValue(ctypes.Structure):
    _fields_ = [("fieldId", c_uint), ("scopeId", c_uint), ("timestamp", c_longlong), ("latencyUsec", c_longlong),
                ("valueType", c_int), ("nvmlReturn", c_int), ("value", c_ulonglong)]


g.nvml.nvmlDeviceGetFieldValues.restype = c_int
g.nvml.nvmlDeviceGetFieldValues.argtypes = [g.nvmlDevice, c_int, ctypes.POINTER(nvmlFieldValue)]


def read(h) -> tuple[dict, int]:
    arr = (nvmlFieldValue * len(FIELDS))()
    for i, fid in enumerate(FIELDS):
        arr[i].fieldId = fid
    t0 = time.monotonic_ns()
    r = g.nvml.nvmlDeviceGetFieldValues(h, len(FIELDS), arr)
    cost = time.monotonic_ns() - t0
    g.nvml_check(r, "nvmlDeviceGetFieldValues")
    out = {}
    for i, fid in enumerate(FIELDS):
        v = arr[i]
        out[FIELDS[fid]] = {"ok": v.nvmlReturn == 0, "value": v.value if v.nvmlReturn == 0 else None,
                            "nvml_return": v.nvmlReturn, "value_type": v.valueType, "timestamp_us": v.timestamp,
                            "latency_us": v.latencyUsec}
    return out, cost


def main(path: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = g.TARGET_UUID
    g.nvml_check(g.nvmlInit(), "nvmlInit")
    h = g.nvml_handle()
    g.cu_check(g.cuInit(0), "cuInit")
    _, dev = g.find_device()
    ctx = g.CUcontext()
    g.cu_check(g.cuDevicePrimaryCtxRetain(byref(ctx), dev), "retain")
    g.cu_check(g.cuCtxSetCurrent(ctx), "current")
    stream = g.CUstream()
    g.cu_check(g.cuStreamCreate(byref(stream), g.CU_STREAM_NON_BLOCKING), "stream")
    size, reps = 16 << 20, 50
    d, pinned = g.CUdeviceptr(), c_void_p()
    g.cu_check(g.cuMemAlloc(byref(d), size), "alloc")
    g.cu_check(g.cuMemAllocHost(byref(pinned), size), "pinned")
    costs = []
    idle_a, c = read(h)
    costs.append(c)
    time.sleep(1.0)
    idle_b, c = read(h)
    costs.append(c)
    before, c = read(h)
    costs.append(c)
    for _ in range(reps):
        g.cuMemcpyHtoDAsync(d, pinned, size, stream)
    g.cu_check(g.cuStreamSynchronize(stream), "sync")
    mid, c = read(h)
    costs.append(c)
    for _ in range(reps):
        g.cuMemcpyDtoHAsync(pinned, d, size, stream)
    g.cu_check(g.cuStreamSynchronize(stream), "sync")
    after, c = read(h)
    costs.append(c)

    def delta(a, b, key):
        return (b[key]["value"] - a[key]["value"]) if a[key]["ok"] and b[key]["ok"] else None

    out = {
        "fields_available": {k: v["ok"] for k, v in before.items()},
        "query_cost_ns": costs,
        "idle_1s_delta": {k: delta(idle_a, idle_b, k) for k in before},
        "h2d_bytes_moved": size * reps,
        "h2d_window_delta": {k: delta(before, mid, k) for k in before},
        "d2h_bytes_moved": size * reps,
        "d2h_window_delta": {k: delta(mid, after, k) for k in before},
        "timestamps_us": {k: before[k]["timestamp_us"] for k in before},
        "realtime_us_now": time.time_ns() // 1000,
    }
    g.cuMemFreeHost(pinned)
    g.cuMemFree(d)
    g.cuStreamDestroy(stream)
    g.cuDevicePrimaryCtxRelease(dev)
    g.nvmlShutdown()
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("fields_available", "query_cost_ns", "idle_1s_delta", "h2d_window_delta",
                                          "d2h_window_delta")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
