"""Stage 2a-G probe 1: environment inventory (read-only; NVML queries, no CUDA context).

Usage: python tools/stage2a_g/probe_inventory.py <out.json>
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import pwd
import subprocess
import sys
from ctypes import byref, c_int, c_uint, c_ulonglong

sys.path.insert(0, os.path.dirname(__file__))
import gpuprobe as g  # noqa: E402


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:  # inventory only
        return f"<unavailable: {exc}>"


def owner(pid: int) -> dict:
    try:
        uid = os.stat(f"/proc/{pid}").st_uid
        with open(f"/proc/{pid}/comm") as f:
            comm = f.read().strip()
        return {"user": pwd.getpwuid(uid).pw_name, "comm": comm, "is_self_user": uid == os.getuid()}
    except Exception as exc:
        return {"user": None, "comm": None, "error": str(exc)}


def gpu(index: int) -> dict:
    h = g.nvmlDevice()
    g.nvml_check(g.nvmlDeviceGetHandleByIndex(index, byref(h)), "handle")
    pci = g.nvmlPciInfo()
    g.nvml_check(g.nvmlDeviceGetPciInfo(h, byref(pci)), "pci")
    mem = g.nvmlMemory_v2(version=g.nvml_struct_version(g.nvmlMemory_v2, 2))
    g.nvml_check(g.nvmlDeviceGetMemoryInfo(h, byref(mem)), "memory")
    out: dict = {
        "nvml_index": index,
        "name": g.nvml_string(g.nvmlDeviceGetName, h),
        "uuid": g.nvml_string(g.nvmlDeviceGetUUID, h),
        "serial": g.nvml_string(g.nvmlDeviceGetSerial, h),
        "board_part_number": g.nvml_string(g.nvmlDeviceGetBoardPartNumber, h),
        "vbios_version": g.nvml_string(g.nvmlDeviceGetVbiosVersion, h),
        "pci_bus_id": pci.busId.decode(),
        "pci_device_id": hex(pci.pciDeviceId),
        "pci_subsystem_id": hex(pci.pciSubSystemId),
        "memory_bytes": {"total": mem.total, "reserved": mem.reserved, "used": mem.used, "free": mem.free},
    }
    for key, fn in (("persistence_mode", g.nvmlDeviceGetPersistenceMode), ("compute_mode", g.nvmlDeviceGetComputeMode),
                    ("performance_state", g.nvmlDeviceGetPerformanceState)):
        v = c_int()
        r = fn(h, byref(v))
        out[key] = v.value if r == 0 else f"error:{r}"
    for key, clock in (("sm", g.NVML_CLOCK_SM), ("graphics", g.NVML_CLOCK_GRAPHICS), ("memory", g.NVML_CLOCK_MEM)):
        cur, mx = c_uint(), c_uint()
        g.nvmlDeviceGetClockInfo(h, clock, byref(cur))
        g.nvmlDeviceGetMaxClockInfo(h, clock, byref(mx))
        out.setdefault("clocks_mhz", {})[key] = {"current": cur.value, "max": mx.value}
    t = g.nvmlTemperature_v1(version=g.nvml_struct_version(g.nvmlTemperature_v1, 1), sensorType=0)
    r = g.nvmlDeviceGetTemperatureV(h, byref(t))
    out["temperature_c"] = t.temperature if r == 0 else f"error:{r}"
    p = c_uint()
    out["power_mw"] = p.value if g.nvmlDeviceGetPowerUsage(h, byref(p)) == 0 else None
    reasons = c_ulonglong()
    if g.nvmlDeviceGetCurrentClocksEventReasons(h, byref(reasons)) == 0:
        out["clocks_event_reasons"] = g.clock_event_reasons(reasons.value)
    gen, width = c_uint(), c_uint()
    g.nvmlDeviceGetCurrPcieLinkGeneration(h, byref(gen))
    g.nvmlDeviceGetCurrPcieLinkWidth(h, byref(width))
    out["pcie_link_current"] = {"generation": gen.value, "width": width.value}
    util = g.nvmlUtilization()
    out["utilization_percent"] = ({"gpu": util.gpu, "memory": util.memory}
                                  if g.nvmlDeviceGetUtilizationRates(h, byref(util)) == 0 else None)
    out["compute_processes"] = [{**p, **owner(p["pid"])} for p in g.compute_processes(h)]
    return out


def main(path: str) -> None:
    g.nvml_check(g.nvmlInit(), "nvmlInit")
    try:
        driver = ctypes.create_string_buffer(96)
        g.nvmlSystemGetDriverVersion(driver, 96)
        nvml_version = ctypes.create_string_buffer(96)
        g.nvmlSystemGetNVMLVersion(nvml_version, 96)
        cuda_driver = c_int()
        g.nvmlSystemGetCudaDriverVersion(byref(cuda_driver))
        count = c_uint()
        g.nvmlDeviceGetCount(byref(count))
        gpus = [gpu(i) for i in range(count.value)]
    finally:
        g.nvmlShutdown()
    major, minor = c_int(), c_int()
    g.nvrtcVersion(byref(major), byref(minor))
    inventory = {
        "host": {
            "hostname": platform.node(),
            "os": run(["lsb_release", "-ds"]),
            "kernel": platform.release(),
            "cpu_model": run(["sh", "-c", "lscpu | sed -n 's/^Model name:\\s*//p'"]),
            "logical_cpus": os.cpu_count(),
            "numa": run(["sh", "-c", "lscpu | grep '^NUMA node[0-9]'"]).splitlines(),
            "clocksource": open("/sys/devices/system/clocksource/clocksource0/current_clocksource").read().strip(),
            "cpufreq_governor_cpu0": run(["cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"]),
            "loadavg": open("/proc/loadavg").read().split()[:3],
            "logged_in_users": len(run(["who"]).splitlines()),
            "memory_total_kib": int(open("/proc/meminfo").readline().split()[1]),
            "python": platform.python_version(),
            "process_cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
        "nvidia": {
            "driver_version": driver.value.decode(),
            "nvml_version": nvml_version.value.decode(),
            "cuda_driver_api_version": cuda_driver.value,
            "kernel_module": open("/proc/driver/nvidia/version").readline().strip(),
            "driver_params": run(["sh", "-c", "grep -E 'RmProfilingAdminOnly|EnableStreamMemOPs' "
                                               "/proc/driver/nvidia/params"]).splitlines(),
            "gpu_topology": run(["nvidia-smi", "topo", "-m"]).splitlines()[:4],
        },
        "toolchain": {
            "cuda_home_used": g.CUDA_HOME,
            "default_cuda_symlink": os.path.realpath("/usr/local/cuda"),
            "installed_cuda": sorted(p for p in os.listdir("/usr/local") if p.startswith("cuda-")),
            "nvcc": run([f"{g.CUDA_HOME}/bin/nvcc", "--version"]).splitlines()[-2:],
            "nvrtc_version": f"{major.value}.{minor.value}",
            "nsys": run([f"{g.CUDA_HOME}/bin/nsys", "--version"]),
            "ncu": run([f"{g.CUDA_HOME}/bin/ncu", "--version"]).splitlines()[-1:],
            "cupti_libs": sorted(os.listdir(f"{g.CUDA_HOME}/extras/CUPTI/lib64")),
        },
        "python_packages_outside_project_venv": {
            name: run([sys.executable if False else "/usr/bin/python3", "-c",
                       f"import importlib.util as u; s=u.find_spec('{name}'); print(s.origin if s else 'absent')"])
            for name in ("torch", "cupy", "cuda", "pycuda", "pynvml", "tensorrt", "onnxruntime", "numpy")
        },
        "gpus": gpus,
    }
    with open(path, "w") as f:
        json.dump(inventory, f, indent=2, sort_keys=True)
    print(json.dumps({"gpus": [(x["name"], x["uuid"], x["pci_bus_id"], len(x["compute_processes"])) for x in gpus],
                      "loadavg": inventory["host"]["loadavg"]}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
