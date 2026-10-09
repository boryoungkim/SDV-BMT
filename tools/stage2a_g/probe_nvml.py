"""Stage 2a-G probe 4: NVML sampling semantics and in-kernel frequency (target GPU only).

Usage: python tools/stage2a_g/probe_nvml.py <out.json>

* cost of each NVML query
* clock / P-state / PCIe-link ramp under ~1.5 s of back-to-back kernels and decay after
* per-process utilization sample timestamps: which host clock domain
* in-kernel effective SM frequency (clock64 cycles / %globaltimer ns) vs NVML samples
* PCIe throughput counter and energy counter behaviour during pinned copies
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import sys
import threading
import time
from ctypes import byref, c_float, c_size_t, c_uint, c_ulonglong, c_void_p

sys.path.insert(0, os.path.dirname(__file__))
import gpuprobe as g  # noqa: E402

KERNELS = r"""
extern "C" __global__ void freq_probe(unsigned long long* out, long long iters) {
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    long long c0 = clock64();
    float a = 1.0f + threadIdx.x * 1e-7f;
    for (long long k = 0; k < iters; ++k) { a = fmaf(a, 0.999999f, 1e-7f); }
    long long c1 = clock64();
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = t0; out[1] = t1; out[2] = (unsigned long long)c0; out[3] = (unsigned long long)c1;
        out[4] = (unsigned long long)(a > 1e30f);
    }
}

extern "C" __global__ void globaltimer_steps(unsigned long long* out, int n) {
    unsigned long long prev, cur;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(prev));
    int k = 0;
    while (k < n) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(cur));
        if (cur != prev) { out[k++] = cur - prev; prev = cur; }
    }
}
"""

now = time.monotonic_ns


def q(values):
    v = sorted(values)
    n = len(v)
    rank = lambda p: v[min(n - 1, max(0, math.ceil(p * n) - 1))]  # noqa: E731
    return {"n": n, "min": v[0], "p50": rank(0.5), "p90": rank(0.9), "p99": rank(0.99), "max": v[-1]}


def main(path: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = g.TARGET_UUID
    out: dict = {}
    g.nvml_check(g.nvmlInit(), "nvmlInit")
    h = g.nvml_handle()

    # ------------------------------------------------------------------ query costs
    def cost(label, fn, n=200):
        ts = []
        for _ in range(n):
            t0 = now()
            fn()
            ts.append(now() - t0)
        out.setdefault("query_cost_ns", {})[label] = q(ts)

    u = c_uint()
    ull = c_ulonglong()
    util = g.nvmlUtilization()
    mem = g.nvmlMemory_v2(version=g.nvml_struct_version(g.nvmlMemory_v2, 2))
    temp = g.nvmlTemperature_v1(version=g.nvml_struct_version(g.nvmlTemperature_v1, 1), sensorType=0)
    cost("clock_sm", lambda: g.nvmlDeviceGetClockInfo(h, g.NVML_CLOCK_SM, byref(u)))
    cost("temperature", lambda: g.nvmlDeviceGetTemperatureV(h, byref(temp)))
    cost("power", lambda: g.nvmlDeviceGetPowerUsage(h, byref(u)))
    cost("utilization", lambda: g.nvmlDeviceGetUtilizationRates(h, byref(util)))
    cost("memory_info", lambda: g.nvmlDeviceGetMemoryInfo(h, byref(mem)))
    cost("clocks_event_reasons", lambda: g.nvmlDeviceGetCurrentClocksEventReasons(h, byref(ull)))
    cost("pcie_link_gen", lambda: g.nvmlDeviceGetCurrPcieLinkGeneration(h, byref(u)))
    cost("compute_processes", lambda: g.compute_processes(h), n=50)
    cost("process_utilization", lambda: g.process_utilization(h, 0), n=50)
    cost("pcie_throughput_tx", lambda: g.nvmlDeviceGetPcieThroughput(h, g.NVML_PCIE_UTIL_TX_BYTES, byref(u)), n=20)

    # ------------------------------------------------------------------ CUDA setup
    g.cu_check(g.cuInit(0), "cuInit")
    _, dev = g.find_device()
    g.cu_check(g.cuDevicePrimaryCtxSetFlags(dev, g.CU_CTX_SCHED_BLOCKING_SYNC), "flags")
    ctx = g.CUcontext()
    g.cu_check(g.cuDevicePrimaryCtxRetain(byref(ctx), dev), "retain")
    g.cu_check(g.cuCtxSetCurrent(ctx), "current")
    cubin, _ = g.compile_source(KERNELS, "nvml_probe.cu", "sm_120")
    mod = g.CUmodule()
    g.cu_check(g.cuModuleLoadData(byref(mod), ctypes.c_char_p(cubin)), "load")
    fn_freq, fn_steps = g.CUfunction(), g.CUfunction()
    g.cu_check(g.cuModuleGetFunction(byref(fn_freq), mod, b"freq_probe"), "freq")
    g.cu_check(g.cuModuleGetFunction(byref(fn_steps), mod, b"globaltimer_steps"), "steps")
    stream = g.CUstream()
    g.cu_check(g.cuStreamCreate(byref(stream), g.CU_STREAM_NON_BLOCKING), "stream")
    dbuf = g.CUdeviceptr()
    g.cu_check(g.cuMemAlloc(byref(dbuf), 1 << 16), "alloc")
    host = (ctypes.c_ulonglong * 4096)()

    # globaltimer update granularity
    n_steps = c_uint(256)
    args = g.ptr_args(g.CUdeviceptr(dbuf.value), ctypes.c_int(256))
    g.cu_check(g.cuLaunchKernel(fn_steps, 1, 1, 1, 1, 1, 1, 0, stream, args, None), "steps launch")
    g.cu_check(g.cuStreamSynchronize(stream), "sync")
    g.cu_check(g.cuMemcpyDtoH(host, dbuf, 256 * 8), "dtoh")
    out["globaltimer_step_ns"] = q(list(host[:256]))
    del n_steps

    # ------------------------------------------------------------------ ramp / decay timeline
    samples: list[dict] = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            sm, memc, gen = c_uint(), c_uint(), c_uint()
            pst = ctypes.c_int()
            g.nvmlDeviceGetClockInfo(h, g.NVML_CLOCK_SM, byref(sm))
            g.nvmlDeviceGetClockInfo(h, g.NVML_CLOCK_MEM, byref(memc))
            g.nvmlDeviceGetCurrPcieLinkGeneration(h, byref(gen))
            g.nvmlDeviceGetPerformanceState(h, byref(pst))
            ut = g.nvmlUtilization()
            g.nvmlDeviceGetUtilizationRates(h, byref(ut))
            samples.append({"t": now(), "sm": sm.value, "mem": memc.value, "gen": gen.value, "p": pst.value,
                            "util": ut.gpu})
            time.sleep(0.005)

    th = threading.Thread(target=sampler)
    th.start()
    time.sleep(0.3)
    t_load0 = now()
    kernel_rows = []
    it = ctypes.c_longlong(500_000)  # ~1 ms at full clock
    fargs = g.ptr_args(g.CUdeviceptr(dbuf.value), it)
    while now() - t_load0 < 1_500_000_000:
        t_sub = now()
        g.cu_check(g.cuLaunchKernel(fn_freq, 1, 1, 1, 32, 1, 1, 0, stream, fargs, None), "freq launch")
        g.cu_check(g.cuStreamSynchronize(stream), "sync")
        g.cu_check(g.cuMemcpyDtoH(host, dbuf, 5 * 8), "dtoh")
        t0, t1, c0, c1 = host[0], host[1], host[2], host[3]
        kernel_rows.append({"t_submit": t_sub, "gt_ns": t1 - t0, "cycles": c1 - c0,
                            "mhz": round((c1 - c0) / (t1 - t0) * 1000, 1) if t1 > t0 else None,
                            "globaltimer_start": t0})
    t_load1 = now()
    time.sleep(2.0)
    stop.set()
    th.join()

    def window(a, b):
        return [s for s in samples if a <= s["t"] < b]

    out["load_timeline"] = {
        "load_ns": t_load1 - t_load0,
        "kernels": len(kernel_rows),
        "in_kernel_mhz": q([r["mhz"] for r in kernel_rows if r["mhz"]]),
        "first_kernels_mhz": [r["mhz"] for r in kernel_rows[:8]],
        "nvml_sm_mhz_before": q([s["sm"] for s in window(0, t_load0)]),
        "nvml_sm_mhz_during": q([s["sm"] for s in window(t_load0, t_load1)]),
        "nvml_sm_mhz_after": q([s["sm"] for s in window(t_load1, t_load1 + 2_000_000_000)]),
        "nvml_util_during": q([s["util"] for s in window(t_load0, t_load1)]),
        "nvml_util_after_first_500ms": [s["util"] for s in window(t_load1, t_load1 + 500_000_000)][::10],
        "first_sample_at_max_sm_after_load_start_ms": next(
            ((s["t"] - t_load0) / 1e6 for s in window(t_load0, t_load1) if s["sm"] >= 2000), None),
        "pstate_during": sorted({s["p"] for s in window(t_load0, t_load1)}),
        "pcie_gen_during": sorted({s["gen"] for s in window(t_load0, t_load1)}),
        "decay_sm_mhz_ms_after_load": [((s["t"] - t_load1) // 1_000_000, s["sm"], s["p"], s["gen"])
                                       for s in window(t_load1, t_load1 + 2_000_000_000)][::20],
        "sampler_period_ns": q([b["t"] - a["t"] for a, b in zip(samples, samples[1:])]),
    }

    # per-process utilization sample timestamps vs host clocks
    status, procs = g.process_utilization(h, 0)
    rt_us, mono_us = time.time_ns() // 1000, now() // 1000
    mine = [p for p in procs if p["pid"] == os.getpid()]
    out["process_utilization"] = {
        "status": status, "samples": len(procs), "own_pid_samples": len(mine),
        "pids": sorted({p["pid"] for p in procs}),
        "latest_ts_minus_realtime_us": (max(p["timestamp_us"] for p in procs) - rt_us) if procs else None,
        "latest_ts_minus_monotonic_us": (max(p["timestamp_us"] for p in procs) - mono_us) if procs else None,
        "own_sm_util_values": sorted({p["sm_util"] for p in mine}),
    }

    # PCIe throughput counter + energy during a pinned H2D burst
    size = 16 << 20
    big = g.CUdeviceptr()
    g.cu_check(g.cuMemAlloc(byref(big), size), "alloc big")
    pinned = c_void_p()
    g.cu_check(g.cuMemAllocHost(byref(pinned), size), "pinned")
    e0, e1 = g.CUevent(), g.CUevent()
    g.cuEventCreate(byref(e0), 0)
    g.cuEventCreate(byref(e1), 0)
    pcie = []
    e_before = c_ulonglong()
    g.nvmlDeviceGetTotalEnergyConsumption(h, byref(e_before))
    g.cuEventRecord(e0, stream)
    t0 = now()
    for i in range(150):
        g.cuMemcpyHtoDAsync(big, pinned, size, stream)
        if i % 10 == 0:
            rx = c_uint()
            g.nvmlDeviceGetPcieThroughput(h, g.NVML_PCIE_UTIL_RX_BYTES, byref(rx))
            tx = c_uint()
            g.nvmlDeviceGetPcieThroughput(h, g.NVML_PCIE_UTIL_TX_BYTES, byref(tx))
            pcie.append({"t_ms": (now() - t0) / 1e6, "rx_kBps": rx.value, "tx_kBps": tx.value})
    g.cuEventRecord(e1, stream)
    g.cuEventSynchronize(e1)
    ms = c_float()
    g.cuEventElapsedTime(byref(ms), e0, e1)
    e_after = c_ulonglong()
    g.nvmlDeviceGetTotalEnergyConsumption(h, byref(e_after))
    out["pcie_counter_during_h2d"] = {
        "bytes": 150 * size, "event_ms": ms.value, "event_GBps": round(150 * size / (ms.value * 1e6), 2),
        "nvml_samples": pcie, "energy_mJ_delta": e_after.value - e_before.value,
    }
    g.cuMemFreeHost(pinned)
    g.cuMemFree(big)
    g.cuMemFree(dbuf)
    g.cuStreamDestroy(stream)
    g.cuModuleUnload(mod)
    g.cuDevicePrimaryCtxRelease(dev)
    g.nvmlShutdown()
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    lt = out["load_timeline"]
    print("globaltimer step", out["globaltimer_step_ns"], "\nin-kernel MHz", lt["in_kernel_mhz"], "first", lt["first_kernels_mhz"],
          "\nnvml during", lt["nvml_sm_mhz_during"], "ramp_ms", lt["first_sample_at_max_sm_after_load_start_ms"],
          "\npstate", lt["pstate_during"], "gen", lt["pcie_gen_during"], "\ndecay", lt["decay_sm_mhz_ms_after_load"],
          "\nproc util", out["process_utilization"], "\npcie", out["pcie_counter_during_h2d"]["event_GBps"],
          out["pcie_counter_during_h2d"]["nvml_samples"][:5])


if __name__ == "__main__":
    main(sys.argv[1])
