"""Stage 2a-G probe 6 helper: H2D(1 MiB pinned) -> kernel -> D2H iterations with CUDA events,
for an Nsight Systems cross-check (no CUPTI in-process; nsys is the only CUPTI client).

Usage: nsys profile -t cuda ... python tools/stage2a_g/nsys_workload.py <out.json> [iterations]
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from ctypes import byref, c_float, c_void_p

sys.path.insert(0, os.path.dirname(__file__))
import gpuprobe as g  # noqa: E402

KERNEL = r"""
extern "C" __global__ void fma_loop(float* out, long long iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 1.0f + i * 1e-7f;
    for (long long k = 0; k < iters; ++k) { a = fmaf(a, 0.999999f, 1e-7f); }
    out[i] = a;
}
"""


def main(path: str, n: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = g.TARGET_UUID
    g.cu_check(g.cuInit(0), "cuInit")
    _, dev = g.find_device()
    g.cu_check(g.cuDevicePrimaryCtxSetFlags(dev, g.CU_CTX_SCHED_BLOCKING_SYNC), "flags")
    ctx = g.CUcontext()
    g.cu_check(g.cuDevicePrimaryCtxRetain(byref(ctx), dev), "retain")
    g.cu_check(g.cuCtxSetCurrent(ctx), "current")
    cubin, _ = g.compile_source(KERNEL, "nsys_probe.cu", "sm_120")
    mod, fn, stream = g.CUmodule(), g.CUfunction(), g.CUstream()
    g.cu_check(g.cuModuleLoadData(byref(mod), ctypes.c_char_p(cubin)), "load")
    g.cu_check(g.cuModuleGetFunction(byref(fn), mod, b"fma_loop"), "fn")
    g.cu_check(g.cuStreamCreate(byref(stream), g.CU_STREAM_NON_BLOCKING), "stream")
    size = 1 << 20
    dptr, out, pinned = g.CUdeviceptr(), g.CUdeviceptr(), c_void_p()
    g.cu_check(g.cuMemAlloc(byref(dptr), size), "alloc")
    g.cu_check(g.cuMemAlloc(byref(out), 4096), "alloc")
    g.cu_check(g.cuMemAllocHost(byref(pinned), size), "pinned")
    ev = [g.CUevent() for _ in range(4)]
    for e in ev:
        g.cu_check(g.cuEventCreate(byref(e), 0), "event")
    args = g.ptr_args(g.CUdeviceptr(out.value), ctypes.c_longlong(50_000))
    rows = []
    for i in range(n + 20):
        t_begin = time.monotonic_ns()
        g.cuEventRecord(ev[0], stream)
        g.cuMemcpyHtoDAsync(dptr, pinned, size, stream)
        g.cuEventRecord(ev[1], stream)
        g.cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, stream, args, None)
        g.cuEventRecord(ev[2], stream)
        g.cuMemcpyDtoHAsync(pinned, dptr, size, stream)
        g.cuEventRecord(ev[3], stream)
        g.cu_check(g.cuEventSynchronize(ev[3]), "sync")
        t_done = time.monotonic_ns()
        ms = [c_float() for _ in range(3)]
        for k in range(3):
            g.cuEventElapsedTime(byref(ms[k]), ev[k], ev[k + 1])
        rows.append({"i": i, "t_begin_mono": t_begin, "t_done_mono": t_done, "h2d_us": ms[0].value * 1000,
                     "kernel_us": ms[1].value * 1000, "d2h_us": ms[2].value * 1000})
    with open(path, "w") as f:
        json.dump({"warmup": 20, "rows": rows}, f)
    g.cuMemFreeHost(pinned)
    g.cuMemFree(dptr)
    g.cuMemFree(out)
    g.cuDevicePrimaryCtxRelease(dev)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200)
