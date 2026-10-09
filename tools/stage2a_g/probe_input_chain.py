"""Stage 2a-G probe 7: input-pipeline observability on the target GPU.

Usage: python tools/stage2a_g/probe_input_chain.py <out.json>

source payload -> host preprocess -> prepared logical input -> pinned runtime buffer
-> H2D -> device-resident representation (verified by D2H readback), and a device-side
("fused") preprocessing path compared with the host reference, for an exact
(power-of-two) normalization and for a rounding normalization (x / 255).
"""

from __future__ import annotations

import array
import ctypes
import hashlib
import json
import os
import sys
import time
from ctypes import byref, c_float, c_int, c_void_p

sys.path.insert(0, os.path.dirname(__file__))
import gpuprobe as g  # noqa: E402

KERNELS = r"""
extern "C" __global__ void norm_pow2(const unsigned char* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ((float)in[i] - 128.0f) * 0.0078125f;
}
extern "C" __global__ void norm_div255(const unsigned char* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __fdividef((float)in[i], 255.0f);
}
"""

sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


def payload(n: int, seed: int) -> bytes:
    out = bytearray(n)
    x = seed
    for i in range(n):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        out[i] = (x >> 16) & 0xFF
    return bytes(out)


def main(path: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = g.TARGET_UUID
    n = 256 * 256 * 3
    src = payload(n, 7)
    t0 = time.monotonic_ns()
    pow2 = array.array("f", [(v - 128) * 0.0078125 for v in src]).tobytes()
    host_pre_ns = time.monotonic_ns() - t0
    div255 = array.array("f", [v / 255.0 for v in src]).tobytes()  # double math, rounded to float32

    g.cu_check(g.cuInit(0), "cuInit")
    _, dev = g.find_device()
    ctx = g.CUcontext()
    g.cu_check(g.cuDevicePrimaryCtxRetain(byref(ctx), dev), "retain")
    g.cu_check(g.cuCtxSetCurrent(ctx), "current")
    cubin, _ = g.compile_source(KERNELS, "chain.cu", "sm_120")
    mod = g.CUmodule()
    g.cu_check(g.cuModuleLoadData(byref(mod), ctypes.c_char_p(cubin)), "load")
    fns = {}
    for k in ("norm_pow2", "norm_div255"):
        f = g.CUfunction()
        g.cu_check(g.cuModuleGetFunction(byref(f), mod, k.encode()), k)
        fns[k] = f
    stream = g.CUstream()
    g.cu_check(g.cuStreamCreate(byref(stream), g.CU_STREAM_NON_BLOCKING), "stream")
    fbytes = n * 4
    pinned, readback = c_void_p(), c_void_p()
    g.cu_check(g.cuMemAllocHost(byref(pinned), fbytes), "pinned")
    g.cu_check(g.cuMemAllocHost(byref(readback), fbytes), "readback")
    d_f, d_u8 = g.CUdeviceptr(), g.CUdeviceptr()
    g.cu_check(g.cuMemAlloc(byref(d_f), fbytes), "alloc f")
    g.cu_check(g.cuMemAlloc(byref(d_u8), n), "alloc u8")

    # host-prepared path: logical input -> pinned buffer -> device -> readback
    ctypes.memmove(pinned, pow2, fbytes)
    staged = ctypes.string_at(pinned, fbytes)
    g.cu_check(g.cuMemcpyHtoDAsync(d_f, pinned, fbytes, stream), "h2d")
    g.cu_check(g.cuMemcpyDtoHAsync(readback, d_f, fbytes, stream), "d2h")
    g.cu_check(g.cuStreamSynchronize(stream), "sync")
    device_resident = ctypes.string_at(readback, fbytes)

    # device-side preprocessing from the raw payload
    def device_pre(kernel: str) -> bytes:
        ctypes.memmove(pinned, src, n)
        g.cu_check(g.cuMemcpyHtoDAsync(d_u8, pinned, n, stream), "h2d u8")
        args = g.ptr_args(g.CUdeviceptr(d_u8.value), g.CUdeviceptr(d_f.value), c_int(n))
        g.cu_check(g.cuLaunchKernel(fns[kernel], (n + 255) // 256, 1, 1, 256, 1, 1, 0, stream, args, None), kernel)
        g.cu_check(g.cuMemcpyDtoHAsync(readback, d_f, fbytes, stream), "d2h")
        g.cu_check(g.cuStreamSynchronize(stream), "sync")
        return ctypes.string_at(readback, fbytes)

    dev_pow2 = device_pre("norm_pow2")
    dev_div = device_pre("norm_div255")
    mismatch_div = sum(1 for a, b in zip(array.array("f", div255), array.array("f", dev_div)) if a != b)
    out = {
        "elements": n,
        "host_preprocess_python_ns": host_pre_ns,
        "digests": {
            "source_payload": sha(src),
            "prepared_logical_input_host_pow2": sha(pow2),
            "pinned_runtime_buffer": sha(staged),
            "device_resident_readback": sha(device_resident),
            "device_preprocessed_pow2": sha(dev_pow2),
            "host_reference_div255": sha(div255),
            "device_preprocessed_div255": sha(dev_div),
        },
        "pinned_equals_prepared": staged == pow2,
        "device_resident_equals_prepared": device_resident == pow2,
        "device_pow2_equals_host_pow2": dev_pow2 == pow2,
        "device_div255_equals_host_div255": dev_div == div255,
        "div255_mismatching_elements": mismatch_div,
    }
    g.cuMemFree(d_f)
    g.cuMemFree(d_u8)
    g.cuMemFreeHost(pinned)
    g.cuMemFreeHost(readback)
    g.cuStreamDestroy(stream)
    g.cuModuleUnload(mod)
    g.cuDevicePrimaryCtxRelease(dev)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "digests"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
