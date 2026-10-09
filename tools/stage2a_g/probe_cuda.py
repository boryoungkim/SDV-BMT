"""Stage 2a-G probe 3: CUDA driver execution model on the target GPU (small, short).

Usage: python tools/stage2a_g/probe_cuda.py <out.json> <spin|yield|blocking_sync>

Only the GPU whose UUID is gpuprobe.TARGET_UUID is made visible to this process.
Total device work is a few seconds of tiny kernels and copies (< 100 MiB).
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import sys
import threading
import time
from ctypes import byref, c_float, c_int, c_size_t, c_uint, c_void_p

sys.path.insert(0, os.path.dirname(__file__))
import gpuprobe as g  # noqa: E402

KERNELS = r"""
extern "C" __global__ void noop() {}

extern "C" __global__ void fma_loop(float* out, long long iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 1.0f + i * 1e-7f;
    for (long long k = 0; k < iters; ++k) { a = fmaf(a, 0.999999f, 1e-7f); }
    out[i] = a;
}

extern "C" __global__ void normalize_u8(const unsigned char* in, float* out, int n, float scale, float bias) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * scale + bias;
}
"""

SCHED = {"spin": g.CU_CTX_SCHED_SPIN, "yield": g.CU_CTX_SCHED_YIELD, "blocking_sync": g.CU_CTX_SCHED_BLOCKING_SYNC}
now = time.monotonic_ns


def q(values: list[float]) -> dict:
    """Nearest-rank quantiles."""
    v = sorted(values)
    n = len(v)

    def rank(p: float):
        return v[min(n - 1, max(0, math.ceil(p * n) - 1))]

    return {"n": n, "min": v[0], "p50": rank(0.5), "p90": rank(0.9), "p99": rank(0.99), "max": v[-1]}


class Probe:
    def __init__(self, sched: str) -> None:
        self.out: dict = {"sched_mode": sched}
        self.sched = sched

    # ------------------------------------------------------------------ helpers
    def nvml_point(self, label: str) -> None:
        h = self.nvml
        sm, mem = c_uint(), c_uint()
        g.nvmlDeviceGetClockInfo(h, g.NVML_CLOCK_SM, byref(sm))
        g.nvmlDeviceGetClockInfo(h, g.NVML_CLOCK_MEM, byref(mem))
        gen, pstate = c_uint(), c_int()
        g.nvmlDeviceGetCurrPcieLinkGeneration(h, byref(gen))
        g.nvmlDeviceGetPerformanceState(h, byref(pstate))
        t = g.nvmlTemperature_v1(version=g.nvml_struct_version(g.nvmlTemperature_v1, 1), sensorType=0)
        g.nvmlDeviceGetTemperatureV(h, byref(t))
        self.out.setdefault("nvml_points", []).append(
            {"label": label, "t_mono_ns": now(), "sm_mhz": sm.value, "mem_mhz": mem.value, "pcie_gen": gen.value,
             "pstate": pstate.value, "temp_c": t.temperature,
             "foreign_processes": [p for p in g.compute_processes(h) if p["pid"] != os.getpid()]})

    def elapsed_ms(self, a, b) -> float:
        ms = c_float()
        g.cu_check(g.cuEventElapsedTime(byref(ms), a, b), "cuEventElapsedTime")
        return ms.value

    def event(self, flags: int = g.CU_EVENT_DEFAULT):
        e = g.CUevent()
        g.cu_check(g.cuEventCreate(byref(e), flags), "cuEventCreate")
        return e

    def launch(self, fn, grid: int, block: int, args, stream) -> None:
        g.cu_check(g.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, stream, args, None), "cuLaunchKernel")

    # ------------------------------------------------------------------ phases
    def setup(self) -> None:
        g.nvml_check(g.nvmlInit(), "nvmlInit")
        self.nvml = g.nvml_handle()
        self.nvml_point("before_cuInit")
        t0 = now()
        g.cu_check(g.cuInit(0), "cuInit")
        self.out["cuInit_ns"] = now() - t0
        version = c_int()
        g.cuDriverGetVersion(byref(version))
        ordinal, dev = g.find_device()
        self.dev = dev
        name = ctypes.create_string_buffer(128)
        g.cuDeviceGetName(name, 128, dev)
        bus = ctypes.create_string_buffer(32)
        g.cuDeviceGetPCIBusId(bus, 32, dev)
        total = c_size_t()
        g.cuDeviceTotalMem(byref(total), dev)
        self.out["device"] = {"cuda_driver_version": version.value, "visible_ordinal": ordinal,
                              "name": name.value.decode(), "pci_bus_id": bus.value.decode(),
                              "total_mem_bytes": total.value, "attributes": g.device_attributes(dev)}
        g.cu_check(g.cuDevicePrimaryCtxSetFlags(dev, SCHED[self.sched]), "cuDevicePrimaryCtxSetFlags")
        ctx = g.CUcontext()
        t0 = now()
        g.cu_check(g.cuDevicePrimaryCtxRetain(byref(ctx), dev), "cuDevicePrimaryCtxRetain")
        g.cu_check(g.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        free, tot = c_size_t(), c_size_t()
        g.cu_check(g.cuMemGetInfo(byref(free), byref(tot)), "cuMemGetInfo")  # forces context creation
        self.out["context_create_ns"] = now() - t0
        flags, active = c_uint(), c_int()
        g.cuDevicePrimaryCtxGetState(dev, byref(flags), byref(active))
        self.out["context"] = {"flags": flags.value, "active": active.value, "mem_free_after": free.value,
                               "mem_total": tot.value}
        self.nvml_point("after_context")

        arch = f"sm_{self.out['device']['attributes']['compute_capability_major']}" \
               f"{self.out['device']['attributes']['compute_capability_minor']}"
        t0 = now()
        cubin, log = g.compile_source(KERNELS, "probe.cu", arch, cubin=True)
        self.out["nvrtc_cubin"] = {"arch": arch, "compile_ns": now() - t0, "bytes": len(cubin),
                                   "sha256": hashlib.sha256(cubin).hexdigest(), "log": log}
        t0 = now()
        ptx, _ = g.compile_source(KERNELS, "probe.cu", "compute_" + arch[3:], cubin=False)
        self.out["nvrtc_ptx"] = {"compile_ns": now() - t0, "bytes": len(ptx)}
        mod_ptx = g.CUmodule()
        t0 = now()
        g.cu_check(g.cuModuleLoadData(byref(mod_ptx), ctypes.c_char_p(ptx)), "cuModuleLoadData(ptx)")
        self.out["module_load_ptx_jit_ns"] = now() - t0
        g.cuModuleUnload(mod_ptx)
        self.mod = g.CUmodule()
        t0 = now()
        g.cu_check(g.cuModuleLoadData(byref(self.mod), ctypes.c_char_p(cubin)), "cuModuleLoadData(cubin)")
        self.out["module_load_cubin_ns"] = now() - t0
        self.fn = {}
        for k in ("noop", "fma_loop", "normalize_u8"):
            f = g.CUfunction()
            g.cu_check(g.cuModuleGetFunction(byref(f), self.mod, k.encode()), k)
            self.fn[k] = f
        self.stream = g.CUstream()
        g.cu_check(g.cuStreamCreate(byref(self.stream), g.CU_STREAM_NON_BLOCKING), "cuStreamCreate")

    def launches(self) -> None:
        s, e0, e1 = self.stream, self.event(), self.event()
        out = g.CUdeviceptr()
        g.cu_check(g.cuMemAlloc(byref(out), 4096), "alloc")
        noop_args = None
        # cold: first launch of each kernel after module load
        t0 = now()
        g.cu_check(g.cuEventRecord(e0, s), "rec")
        self.launch(self.fn["noop"], 1, 1, noop_args, s)
        g.cu_check(g.cuEventRecord(e1, s), "rec")
        t_submit = now() - t0
        g.cu_check(g.cuEventSynchronize(e1), "sync")
        self.out["first_noop"] = {"host_submit_ns": t_submit, "host_total_ns": now() - t0,
                                  "event_ms": self.elapsed_ms(e0, e1)}
        submit, total, dev_ms = [], [], []
        for _ in range(300):
            t0 = now()
            g.cuEventRecord(e0, s)
            self.launch(self.fn["noop"], 1, 1, noop_args, s)
            g.cuEventRecord(e1, s)
            t1 = now()
            g.cuEventSynchronize(e1)
            t2 = now()
            submit.append(t1 - t0)
            total.append(t2 - t0)
            dev_ms.append(self.elapsed_ms(e0, e1))
        self.out["noop_warm"] = {"host_submit_ns": q(submit), "host_submit_to_sync_return_ns": q(total),
                                 "event_us": q([m * 1000 for m in dev_ms]),
                                 "distinct_event_values_us": sorted({round(m * 1000, 4) for m in dev_ms})[:20]}
        self.nvml_point("after_noop_burst")
        # fma calibrations: event duration vs host completion detection (sync vs spin query)
        cal = {}
        for iters in (1_000, 10_000, 100_000, 1_000_000):
            it = ctypes.c_longlong(iters)
            args = g.ptr_args(g.CUdeviceptr(out.value), it)
            first = None
            rows = {"event_us": [], "sync_detect_minus_event_us": [], "spin_detect_minus_event_us": []}
            for rep in range(30):
                spin = rep % 2 == 1
                t0 = now()
                g.cuEventRecord(e0, s)
                self.launch(self.fn["fma_loop"], 1, 32, args, s)
                g.cuEventRecord(e1, s)
                if spin:
                    while g.cuEventQuery(e1) == g.CUDA_ERROR_NOT_READY:
                        pass
                else:
                    g.cu_check(g.cuEventSynchronize(e1), "sync")
                t_detect = now() - t0
                ev = self.elapsed_ms(e0, e1) * 1000
                if first is None:
                    first = ev
                    continue
                rows["event_us"].append(ev)
                key = "spin_detect_minus_event_us" if spin else "sync_detect_minus_event_us"
                rows[key].append(t_detect / 1000 - ev)
            cal[str(iters)] = {"first_event_us": first, **{k: q(v) for k, v in rows.items()}}
        self.out["fma_loop_calibration"] = cal
        self.nvml_point("after_fma_calibration")
        # event query cost (complete event)
        costs = []
        for _ in range(2000):
            t0 = now()
            g.cuEventQuery(e1)
            costs.append(now() - t0)
        self.out["cuEventQuery_cost_ns"] = q(costs)
        g.cuMemFree(out)

    def copies(self) -> None:
        s, e0, e1 = self.stream, self.event(), self.event()
        res = {}
        for size in (4096, 65536, 1 << 20, 16 << 20):
            dptr = g.CUdeviceptr()
            g.cu_check(g.cuMemAlloc(byref(dptr), size), "alloc")
            pageable = ctypes.create_string_buffer(size)
            pinned = c_void_p()
            g.cu_check(g.cuMemAllocHost(byref(pinned), size), "allocHost")
            row = {}
            for label, host in (("pageable", ctypes.cast(pageable, c_void_p)), ("pinned", pinned)):
                for direction in ("h2d", "d2h"):
                    call, dev_us, total = [], [], []
                    for rep in range(21):
                        t0 = now()
                        g.cuEventRecord(e0, s)
                        if direction == "h2d":
                            r = g.cuMemcpyHtoDAsync(dptr, host, size, s)
                        else:
                            r = g.cuMemcpyDtoHAsync(host, dptr, size, s)
                        g.cu_check(r, "memcpyAsync")
                        t1 = now()
                        g.cuEventRecord(e1, s)
                        g.cu_check(g.cuEventSynchronize(e1), "sync")
                        t2 = now()
                        if rep == 0:
                            continue  # first copy of each kind reported separately below
                        call.append(t1 - t0)
                        total.append(t2 - t0)
                        dev_us.append(self.elapsed_ms(e0, e1) * 1000)
                    row[f"{label}_{direction}"] = {
                        "async_call_return_ns": q(call), "host_total_ns": q(total), "event_us": q(dev_us),
                        "event_gbps_p50": round(size / (q(dev_us)["p50"] * 1e3), 3),
                    }
            res[str(size)] = row
            g.cuMemFreeHost(pinned)
            g.cuMemFree(dptr)
        self.out["copies"] = res
        self.nvml_point("after_copies")

    def overlap(self) -> None:
        """Copy on one stream while a long kernel runs on another."""
        s2 = g.CUstream()
        g.cu_check(g.cuStreamCreate(byref(s2), g.CU_STREAM_NON_BLOCKING), "stream2")
        size = 16 << 20
        dptr, out = g.CUdeviceptr(), g.CUdeviceptr()
        g.cuMemAlloc(byref(dptr), size)
        g.cuMemAlloc(byref(out), 4096)
        pinned = c_void_p()
        g.cuMemAllocHost(byref(pinned), size)
        k0, k1, c0, c1 = self.event(), self.event(), self.event(), self.event()
        it = ctypes.c_longlong(2_000_000)
        args = g.ptr_args(g.CUdeviceptr(out.value), it)
        g.cuEventRecord(k0, self.stream)
        self.launch(self.fn["fma_loop"], 1, 32, args, self.stream)
        g.cuEventRecord(k1, self.stream)
        g.cuEventRecord(c0, s2)
        g.cuMemcpyHtoDAsync(dptr, pinned, size, s2)
        g.cuEventRecord(c1, s2)
        g.cuEventSynchronize(k1)
        g.cuEventSynchronize(c1)
        self.out["overlap"] = {
            "kernel_us": self.elapsed_ms(k0, k1) * 1000, "copy_us": self.elapsed_ms(c0, c1) * 1000,
            "copy_end_minus_kernel_end_us": self.elapsed_ms(k1, c1) * 1000,
            "copy_start_minus_kernel_start_us": self.elapsed_ms(k0, c0) * 1000,
        }
        g.cuMemFreeHost(pinned)
        g.cuMemFree(dptr)
        g.cuMemFree(out)
        g.cuStreamDestroy(s2)

    def allocations(self) -> None:
        res = {}
        for size in (4096, 1 << 20, 16 << 20):
            dev_alloc, dev_free, host_alloc, pool_alloc = [], [], [], []
            for _ in range(20):
                p = g.CUdeviceptr()
                t0 = now()
                g.cu_check(g.cuMemAlloc(byref(p), size), "alloc")
                t1 = now()
                g.cuMemFree(p)
                t2 = now()
                dev_alloc.append(t1 - t0)
                dev_free.append(t2 - t1)
                h = c_void_p()
                t0 = now()
                g.cu_check(g.cuMemAllocHost(byref(h), size), "allocHost")
                host_alloc.append(now() - t0)
                g.cuMemFreeHost(h)
                t0 = now()
                g.cu_check(g.cuMemAllocAsync(byref(p), size, self.stream), "allocAsync")
                pool_alloc.append(now() - t0)
                g.cuMemFreeAsync(p, self.stream)
            g.cuStreamSynchronize(self.stream)
            res[str(size)] = {"cuMemAlloc_ns": q(dev_alloc), "cuMemFree_ns": q(dev_free),
                              "cuMemAllocHost_pinned_ns": q(host_alloc), "cuMemAllocAsync_pool_ns": q(pool_alloc)}
        self.out["allocations"] = res

    def gil(self) -> None:
        """Does a thread make progress while the main thread blocks in a driver call?"""
        out = g.CUdeviceptr()
        g.cuMemAlloc(byref(out), 4096)
        e1 = self.event()
        stop = threading.Event()
        counter = [0]

        def spin() -> None:
            while not stop.is_set():
                counter[0] += 1

        it = ctypes.c_longlong(40_000_000)
        args = g.ptr_args(g.CUdeviceptr(out.value), it)
        th = threading.Thread(target=spin)
        th.start()
        time.sleep(0.05)
        before = counter[0]
        t0 = now()
        self.launch(self.fn["fma_loop"], 1, 32, args, self.stream)
        g.cuEventRecord(e1, self.stream)
        g.cu_check(g.cuEventSynchronize(e1), "sync")
        blocked_ns = now() - t0
        during = counter[0] - before
        stop.set()
        th.join()
        # reference rate: counter increments per ns with the main thread sleeping
        stop.clear()
        counter[0] = 0
        th = threading.Thread(target=spin)
        th.start()
        t0 = now()
        time.sleep(blocked_ns / 1e9)
        ref = counter[0]
        ref_ns = now() - t0
        stop.set()
        th.join()
        self.out["gil_during_cuEventSynchronize"] = {
            "blocked_ns": blocked_ns, "thread_increments_during_block": during,
            "thread_increments_same_duration_main_sleeping": ref, "reference_ns": ref_ns,
            "progress_ratio": round(during / ref, 3) if ref else None,
        }
        g.cuMemFree(out)

    def teardown(self) -> None:
        g.cuStreamDestroy(self.stream)
        g.cuModuleUnload(self.mod)
        g.cu_check(g.cuDevicePrimaryCtxRelease(self.dev), "cuDevicePrimaryCtxRelease")
        self.nvml_point("after_release")
        g.nvmlShutdown()


def main(path: str, sched: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = g.TARGET_UUID  # read by cuInit: no other GPU is visible
    p = Probe(sched)
    p.setup()
    try:
        p.launches()
        p.copies()
        p.overlap()
        p.allocations()
        p.gil()
    finally:
        p.teardown()
    with open(path, "w") as f:
        json.dump(p.out, f, indent=2)
    o = p.out
    print(sched, "ctx_ns", o["context_create_ns"], "first_noop", o["first_noop"],
          "noop_event_us", o["noop_warm"]["event_us"], "submit_ns", o["noop_warm"]["host_submit_ns"]["p50"],
          "gil", o["gil_during_cuEventSynchronize"]["progress_ratio"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
