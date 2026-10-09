"""Stage 2a-G probe 5: CUPTI activity tracing (no hardware counters; RmProfilingAdminOnly=1).

Usage: python tools/stage2a_g/probe_cupti.py <out.json> <realtime|monotonic>

* CUPTI activity records for H2D/D2H memcpy, kernels and driver API calls
* record durations vs CUDA-event durations of the same operations
* host-domain placement: with the ``monotonic`` timestamp callback, does each
  activity fall inside the host interval [call begin, completion observed]?
* overhead: the same operation sequence with and without activity collection
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import struct
import sys
import time
from ctypes import CFUNCTYPE, POINTER, byref, c_float, c_size_t, c_uint32, c_uint64, c_void_p

sys.path.insert(0, os.path.dirname(__file__))
import gpuprobe as g  # noqa: E402

cupti = ctypes.CDLL(f"{g.CUDA_HOME}/extras/CUPTI/lib64/libcupti.so.13")
KIND_MEMCPY, KIND_DRIVER, KIND_CONCURRENT_KERNEL = 1, 4, 10
REQUEST_CB = CFUNCTYPE(None, POINTER(POINTER(ctypes.c_uint8)), POINTER(c_size_t), POINTER(c_size_t))
COMPLETE_CB = CFUNCTYPE(None, c_void_p, c_uint32, POINTER(ctypes.c_uint8), c_size_t, c_size_t)
TIMESTAMP_CB = CFUNCTYPE(c_uint64)
now = time.monotonic_ns

cupti.cuptiGetVersion.argtypes = [POINTER(c_uint32)]
cupti.cuptiActivityEnable.argtypes = [ctypes.c_int]
cupti.cuptiActivityDisable.argtypes = [ctypes.c_int]
cupti.cuptiActivityRegisterCallbacks.argtypes = [REQUEST_CB, COMPLETE_CB]
cupti.cuptiActivityFlushAll.argtypes = [c_uint32]
cupti.cuptiActivityGetNextRecord.argtypes = [POINTER(ctypes.c_uint8), c_size_t, POINTER(c_void_p)]
cupti.cuptiGetTimestamp.argtypes = [POINTER(c_uint64)]
cupti.cuptiActivityRegisterTimestampCallback.argtypes = [TIMESTAMP_CB]
cupti.cuptiGetResultString.argtypes = [ctypes.c_int, POINTER(ctypes.c_char_p)]

KERNEL = r"""
extern "C" __global__ void fma_loop(float* out, long long iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 1.0f + i * 1e-7f;
    for (long long k = 0; k < iters; ++k) { a = fmaf(a, 0.999999f, 1e-7f); }
    out[i] = a;
}
"""

_buffers: list = []
records: list[dict] = []


def check(r: int, what: str) -> None:
    if r != 0:
        s = ctypes.c_char_p()
        cupti.cuptiGetResultString(r, byref(s))
        raise g.ProbeError(f"{what}: CUPTI {r} {s.value!r}")


@REQUEST_CB
def on_request(buf, size, max_records):
    raw = ctypes.create_string_buffer(1 << 20 + 8)
    _buffers.append(raw)
    addr = ctypes.addressof(raw)
    aligned = (addr + 7) & ~7
    buf[0] = ctypes.cast(aligned, POINTER(ctypes.c_uint8))
    size[0] = (1 << 20)
    max_records[0] = 0


@COMPLETE_CB
def on_complete(ctx, stream_id, buf, size, valid):
    rec = c_void_p()
    while cupti.cuptiActivityGetNextRecord(buf, valid, byref(rec)) == 0:
        base = rec.value
        kind = struct.unpack_from("<I", (ctypes.c_char * 4).from_address(base))[0]
        if kind == KIND_MEMCPY:
            raw = bytes((ctypes.c_char * 56).from_address(base))
            copy_kind, src_kind, dst_kind = raw[4], raw[5], raw[6]
            nbytes, start, end = struct.unpack_from("<QQQ", raw, 8)
            corr = struct.unpack_from("<I", raw, 44)[0]
            records.append({"kind": "memcpy", "copy_kind": copy_kind, "src_kind": src_kind, "dst_kind": dst_kind,
                            "bytes": nbytes, "start": start, "end": end, "corr": corr})
        elif kind == KIND_CONCURRENT_KERNEL:
            raw = bytes((ctypes.c_char * 136).from_address(base))
            start, end, completed = struct.unpack_from("<QQQ", raw, 16)
            corr = struct.unpack_from("<I", raw, 92)[0]
            queued, submitted = struct.unpack_from("<QQ", raw, 120)
            records.append({"kind": "kernel", "start": start, "end": end, "corr": corr})
        elif kind == KIND_DRIVER:
            raw = bytes((ctypes.c_char * 40).from_address(base))
            cbid = struct.unpack_from("<I", raw, 4)[0]
            start, end = struct.unpack_from("<QQ", raw, 8)
            corr = struct.unpack_from("<I", raw, 32)[0]
            records.append({"kind": "driver", "cbid": cbid, "start": start, "end": end, "corr": corr})


@TIMESTAMP_CB
def monotonic_timestamp():
    return time.monotonic_ns()


def q(values):
    v = sorted(values)
    n = len(v)
    rank = lambda p: v[min(n - 1, max(0, math.ceil(p * n) - 1))]  # noqa: E731
    return {"n": n, "min": v[0], "p50": rank(0.5), "p90": rank(0.9), "p99": rank(0.99), "max": v[-1]}


def workload(ctx_objs, n: int, collect_host: bool) -> list[dict]:
    """n iterations of H2D(1 MiB pinned) -> kernel -> D2H(1 MiB), each bracketed by events."""
    stream, fn, dptr, pinned, out = ctx_objs
    size = 1 << 20
    ev = [g.CUevent() for _ in range(4)]
    for e in ev:
        g.cu_check(g.cuEventCreate(byref(e), 0), "event")
    it = ctypes.c_longlong(50_000)
    args = g.ptr_args(g.CUdeviceptr(out.value), it)
    rows = []
    for _ in range(n):
        t_begin = now()
        g.cuEventRecord(ev[0], stream)
        g.cuMemcpyHtoDAsync(dptr, pinned, size, stream)
        g.cuEventRecord(ev[1], stream)
        g.cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, stream, args, None)
        g.cuEventRecord(ev[2], stream)
        g.cuMemcpyDtoHAsync(pinned, dptr, size, stream)
        g.cuEventRecord(ev[3], stream)
        t_submitted = now()
        g.cu_check(g.cuEventSynchronize(ev[3]), "sync")
        t_done = now()
        ms = [c_float() for _ in range(3)]
        for i in range(3):
            g.cuEventElapsedTime(byref(ms[i]), ev[i], ev[i + 1])
        rows.append({"t_begin": t_begin, "t_submitted": t_submitted, "t_done": t_done,
                     "h2d_us": ms[0].value * 1000, "kernel_us": ms[1].value * 1000, "d2h_us": ms[2].value * 1000})
    for e in ev:
        g.cuEventDestroy(e)
    return rows


def main(path: str, mode: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = g.TARGET_UUID
    out: dict = {"timestamp_mode": mode}
    v = c_uint32()
    check(cupti.cuptiGetVersion(byref(v)), "version")
    out["cupti_api_version"] = v.value
    g.cu_check(g.cuInit(0), "cuInit")
    _, dev = g.find_device()
    g.cu_check(g.cuDevicePrimaryCtxSetFlags(dev, g.CU_CTX_SCHED_BLOCKING_SYNC), "flags")
    ctx = g.CUcontext()
    g.cu_check(g.cuDevicePrimaryCtxRetain(byref(ctx), dev), "retain")
    g.cu_check(g.cuCtxSetCurrent(ctx), "current")
    cubin, _ = g.compile_source(KERNEL, "cupti_probe.cu", "sm_120")
    mod, fn, stream = g.CUmodule(), g.CUfunction(), g.CUstream()
    g.cu_check(g.cuModuleLoadData(byref(mod), ctypes.c_char_p(cubin)), "load")
    g.cu_check(g.cuModuleGetFunction(byref(fn), mod, b"fma_loop"), "fn")
    g.cu_check(g.cuStreamCreate(byref(stream), g.CU_STREAM_NON_BLOCKING), "stream")
    dptr, out_buf, pinned = g.CUdeviceptr(), g.CUdeviceptr(), c_void_p()
    g.cu_check(g.cuMemAlloc(byref(dptr), 1 << 20), "alloc")
    g.cu_check(g.cuMemAlloc(byref(out_buf), 4096), "alloc")
    g.cu_check(g.cuMemAllocHost(byref(pinned), 1 << 20), "pinned")
    objs = (stream, fn, dptr, pinned, out_buf)

    workload(objs, 20, False)  # warm the device
    baseline = workload(objs, 200, True)

    if mode == "monotonic":
        check(cupti.cuptiActivityRegisterTimestampCallback(monotonic_timestamp), "timestamp callback")
    check(cupti.cuptiActivityRegisterCallbacks(on_request, on_complete), "callbacks")
    for kind in (KIND_MEMCPY, KIND_CONCURRENT_KERNEL, KIND_DRIVER):
        check(cupti.cuptiActivityEnable(kind), f"enable {kind}")
    ts = c_uint64()
    cupti.cuptiGetTimestamp(byref(ts))
    out["cuptiGetTimestamp_minus_monotonic_ns"] = ts.value - now()
    out["cuptiGetTimestamp_minus_realtime_ns"] = ts.value - time.time_ns()
    traced = workload(objs, 200, True)
    t_flush = now()
    check(cupti.cuptiActivityFlushAll(1), "flush")
    out["flush_ns"] = now() - t_flush
    for kind in (KIND_MEMCPY, KIND_CONCURRENT_KERNEL, KIND_DRIVER):
        cupti.cuptiActivityDisable(kind)

    memcpys = [r for r in records if r["kind"] == "memcpy"]
    kernels = [r for r in records if r["kind"] == "kernel"]
    h2d = [r for r in memcpys if r["copy_kind"] == 1]
    d2h = [r for r in memcpys if r["copy_kind"] == 2]
    out["record_counts"] = {"memcpy_h2d": len(h2d), "memcpy_d2h": len(d2h), "kernel": len(kernels),
                            "driver": sum(r["kind"] == "driver" for r in records)}
    out["h2d_src_dst_kinds"] = sorted({(r["src_kind"], r["dst_kind"]) for r in h2d})
    n = min(len(traced), len(h2d), len(kernels), len(d2h))
    pairs = list(zip(traced[-n:], h2d[-n:], kernels[-n:], d2h[-n:]))
    out["cupti_minus_event_us"] = {
        "h2d": q([(c["end"] - c["start"]) / 1000 - t["h2d_us"] for t, c, _, _ in pairs]),
        "kernel": q([(k["end"] - k["start"]) / 1000 - t["kernel_us"] for t, _, k, _ in pairs]),
        "d2h": q([(d["end"] - d["start"]) / 1000 - t["d2h_us"] for t, _, _, d in pairs]),
    }
    out["cupti_durations_us"] = {
        "h2d": q([(c["end"] - c["start"]) / 1000 for _, c, _, _ in pairs]),
        "kernel": q([(k["end"] - k["start"]) / 1000 for _, _, k, _ in pairs]),
        "d2h": q([(d["end"] - d["start"]) / 1000 for _, _, _, d in pairs]),
    }
    if mode == "monotonic":
        # each device activity must lie within [host begin, host completion observed]
        inside, margins_start, margins_end = 0, [], []
        for t, c, k, d in pairs:
            ok = t["t_begin"] <= c["start"] and d["end"] <= t["t_done"]
            inside += ok
            margins_start.append(c["start"] - t["t_begin"])
            margins_end.append(t["t_done"] - d["end"])
        out["host_interval_containment"] = {
            "pairs": len(pairs), "contained": inside,
            "h2d_start_after_host_begin_ns": q(margins_start),
            "completion_observed_after_d2h_end_ns": q(margins_end),
        }
    out["overhead"] = {
        "baseline_host_total_us": q([(r["t_done"] - r["t_begin"]) / 1000 for r in baseline]),
        "traced_host_total_us": q([(r["t_done"] - r["t_begin"]) / 1000 for r in traced]),
        "baseline_submit_us": q([(r["t_submitted"] - r["t_begin"]) / 1000 for r in baseline]),
        "traced_submit_us": q([(r["t_submitted"] - r["t_begin"]) / 1000 for r in traced]),
        "baseline_kernel_event_us": q([r["kernel_us"] for r in baseline]),
        "traced_kernel_event_us": q([r["kernel_us"] for r in traced]),
        "baseline_h2d_event_us": q([r["h2d_us"] for r in baseline]),
        "traced_h2d_event_us": q([r["h2d_us"] for r in traced]),
    }
    g.cuMemFreeHost(pinned)
    g.cuMemFree(dptr)
    g.cuMemFree(out_buf)
    g.cuStreamDestroy(stream)
    g.cuModuleUnload(mod)
    g.cuDevicePrimaryCtxRelease(dev)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("cupti_api_version", "record_counts", "cupti_minus_event_us")}, indent=0))
    print("containment", out.get("host_interval_containment"))
    print("overhead total p50", out["overhead"]["baseline_host_total_us"]["p50"], "->",
          out["overhead"]["traced_host_total_us"]["p50"], "submit p50", out["overhead"]["baseline_submit_us"]["p50"],
          "->", out["overhead"]["traced_submit_us"]["p50"])
    print("ts offsets", out["cuptiGetTimestamp_minus_monotonic_ns"], out["cuptiGetTimestamp_minus_realtime_ns"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
