"""Stage 2a-G investigation bindings: minimal ctypes access to the CUDA driver API,
NVRTC and NVML on the development host.

Investigation tool only. It is not part of the ``sdv_bmt`` package and adds no
Python dependency (stdlib ``ctypes`` against the installed NVIDIA libraries).
Probes select the GPU by UUID and never touch any other device.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, byref, c_char, c_char_p, c_float, c_int, c_size_t, c_uint, c_uint64, c_ulonglong, c_void_p

CUDA_HOME = os.environ.get("SDV_BMT_CUDA_HOME", "/usr/local/cuda-13.0")
LIB_CUDA = "libcuda.so.1"
LIB_NVRTC = f"{CUDA_HOME}/targets/x86_64-linux/lib/libnvrtc.so.13"
LIB_NVML = "libnvidia-ml.so.1"

# GPU used by every Stage 2a-G probe (NVML index 0, PCI 0000:3D:00.0).
TARGET_UUID = os.environ.get("SDV_BMT_GPU_UUID", "GPU-f3e03bfa-61ab-acdf-0834-450de8becefc")


class ProbeError(RuntimeError):
    pass


# --------------------------------------------------------------------------- CUDA driver

cuda = ctypes.CDLL(LIB_CUDA)
CUdevice = c_int
CUcontext = c_void_p
CUstream = c_void_p
CUevent = c_void_p
CUmodule = c_void_p
CUfunction = c_void_p
CUdeviceptr = c_uint64


class CUuuid(ctypes.Structure):
    _fields_ = [("bytes", c_char * 16)]


def _proto(lib, name, restype, *argtypes):
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = list(argtypes)
    return fn


cuGetErrorName = _proto(cuda, "cuGetErrorName", c_int, c_int, POINTER(c_char_p))
cuGetErrorString = _proto(cuda, "cuGetErrorString", c_int, c_int, POINTER(c_char_p))
cuInit = _proto(cuda, "cuInit", c_int, c_uint)
cuDriverGetVersion = _proto(cuda, "cuDriverGetVersion", c_int, POINTER(c_int))
cuDeviceGetCount = _proto(cuda, "cuDeviceGetCount", c_int, POINTER(c_int))
cuDeviceGet = _proto(cuda, "cuDeviceGet", c_int, POINTER(CUdevice), c_int)
cuDeviceGetName = _proto(cuda, "cuDeviceGetName", c_int, c_char_p, c_int, CUdevice)
cuDeviceGetUuid = _proto(cuda, "cuDeviceGetUuid_v2", c_int, POINTER(CUuuid), CUdevice)
cuDeviceGetPCIBusId = _proto(cuda, "cuDeviceGetPCIBusId", c_int, c_char_p, c_int, CUdevice)
cuDeviceGetAttribute = _proto(cuda, "cuDeviceGetAttribute", c_int, POINTER(c_int), c_int, CUdevice)
cuDeviceTotalMem = _proto(cuda, "cuDeviceTotalMem_v2", c_int, POINTER(c_size_t), CUdevice)
cuDevicePrimaryCtxRetain = _proto(cuda, "cuDevicePrimaryCtxRetain", c_int, POINTER(CUcontext), CUdevice)
cuDevicePrimaryCtxRelease = _proto(cuda, "cuDevicePrimaryCtxRelease_v2", c_int, CUdevice)
cuDevicePrimaryCtxSetFlags = _proto(cuda, "cuDevicePrimaryCtxSetFlags_v2", c_int, CUdevice, c_uint)
cuDevicePrimaryCtxGetState = _proto(cuda, "cuDevicePrimaryCtxGetState", c_int, CUdevice, POINTER(c_uint), POINTER(c_int))
cuCtxSetCurrent = _proto(cuda, "cuCtxSetCurrent", c_int, CUcontext)
cuCtxSynchronize = _proto(cuda, "cuCtxSynchronize", c_int)
cuMemGetInfo = _proto(cuda, "cuMemGetInfo_v2", c_int, POINTER(c_size_t), POINTER(c_size_t))
cuMemAlloc = _proto(cuda, "cuMemAlloc_v2", c_int, POINTER(CUdeviceptr), c_size_t)
cuMemFree = _proto(cuda, "cuMemFree_v2", c_int, CUdeviceptr)
cuMemAllocHost = _proto(cuda, "cuMemAllocHost_v2", c_int, POINTER(c_void_p), c_size_t)
cuMemFreeHost = _proto(cuda, "cuMemFreeHost", c_int, c_void_p)
cuMemAllocAsync = _proto(cuda, "cuMemAllocAsync", c_int, POINTER(CUdeviceptr), c_size_t, CUstream)
cuMemFreeAsync = _proto(cuda, "cuMemFreeAsync", c_int, CUdeviceptr, CUstream)
cuMemcpyHtoD = _proto(cuda, "cuMemcpyHtoD_v2", c_int, CUdeviceptr, c_void_p, c_size_t)
cuMemcpyDtoH = _proto(cuda, "cuMemcpyDtoH_v2", c_int, c_void_p, CUdeviceptr, c_size_t)
cuMemcpyHtoDAsync = _proto(cuda, "cuMemcpyHtoDAsync_v2", c_int, CUdeviceptr, c_void_p, c_size_t, CUstream)
cuMemcpyDtoHAsync = _proto(cuda, "cuMemcpyDtoHAsync_v2", c_int, c_void_p, CUdeviceptr, c_size_t, CUstream)
cuStreamCreate = _proto(cuda, "cuStreamCreate", c_int, POINTER(CUstream), c_uint)
cuStreamDestroy = _proto(cuda, "cuStreamDestroy_v2", c_int, CUstream)
cuStreamSynchronize = _proto(cuda, "cuStreamSynchronize", c_int, CUstream)
cuStreamQuery = _proto(cuda, "cuStreamQuery", c_int, CUstream)
cuEventCreate = _proto(cuda, "cuEventCreate", c_int, POINTER(CUevent), c_uint)
cuEventDestroy = _proto(cuda, "cuEventDestroy_v2", c_int, CUevent)
cuEventRecord = _proto(cuda, "cuEventRecord", c_int, CUevent, CUstream)
cuEventQuery = _proto(cuda, "cuEventQuery", c_int, CUevent)
cuEventSynchronize = _proto(cuda, "cuEventSynchronize", c_int, CUevent)
cuEventElapsedTime = _proto(cuda, "cuEventElapsedTime_v2", c_int, POINTER(c_float), CUevent, CUevent)
cuModuleLoadData = _proto(cuda, "cuModuleLoadData", c_int, POINTER(CUmodule), c_void_p)
cuModuleUnload = _proto(cuda, "cuModuleUnload", c_int, CUmodule)
cuModuleGetFunction = _proto(cuda, "cuModuleGetFunction", c_int, POINTER(CUfunction), CUmodule, c_char_p)
cuLaunchKernel = _proto(cuda, "cuLaunchKernel", c_int, CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
                        c_uint, CUstream, POINTER(c_void_p), POINTER(c_void_p))

CUDA_SUCCESS = 0
CUDA_ERROR_NOT_READY = 600
CU_CTX_SCHED_AUTO = 0x00
CU_CTX_SCHED_SPIN = 0x01
CU_CTX_SCHED_YIELD = 0x02
CU_CTX_SCHED_BLOCKING_SYNC = 0x04
CU_EVENT_DEFAULT = 0x0
CU_EVENT_BLOCKING_SYNC = 0x1
CU_EVENT_DISABLE_TIMING = 0x2
CU_STREAM_NON_BLOCKING = 0x1

ATTRIBUTES = {
    "compute_capability_major": 75,
    "compute_capability_minor": 76,
    "multiprocessor_count": 16,
    "clock_rate_khz": 13,
    "memory_clock_rate_khz": 36,
    "global_memory_bus_width_bits": 37,
    "l2_cache_size_bytes": 38,
    "async_engine_count": 40,
    "unified_addressing": 41,
    "concurrent_kernels": 31,
    "can_map_host_memory": 19,
    "compute_mode": 20,
    "integrated": 18,
    "kernel_exec_timeout": 17,
    "pageable_memory_access": 88,
    "host_native_atomic_supported": 86,
    "memory_pools_supported": 115,
}


def cu_check(result: int, what: str) -> None:
    if result != CUDA_SUCCESS:
        name, text = c_char_p(), c_char_p()
        cuGetErrorName(result, byref(name))
        cuGetErrorString(result, byref(text))
        raise ProbeError(f"{what}: {result} {name.value!r} {text.value!r}")


def uuid_string(raw: bytes) -> str:
    h = raw.hex()
    return f"GPU-{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def find_device(uuid: str = TARGET_UUID) -> tuple[int, int]:
    """Return (ordinal, CUdevice) of the device with this UUID; never selects by index."""
    count = c_int()
    cu_check(cuDeviceGetCount(byref(count)), "cuDeviceGetCount")
    for ordinal in range(count.value):
        dev = CUdevice()
        cu_check(cuDeviceGet(byref(dev), ordinal), "cuDeviceGet")
        u = CUuuid()
        cu_check(cuDeviceGetUuid(byref(u), dev), "cuDeviceGetUuid")
        if uuid_string(u.bytes) == uuid:
            return ordinal, dev.value
    raise ProbeError(f"no CUDA device with UUID {uuid}")


def device_attributes(dev: int) -> dict[str, int]:
    out = {}
    for name, attr in ATTRIBUTES.items():
        v = c_int()
        cu_check(cuDeviceGetAttribute(byref(v), attr, dev), f"attribute {name}")
        out[name] = v.value
    return out


def ptr_args(*values) -> ctypes.Array:
    """Kernel argument array for cuLaunchKernel from ctypes scalars."""
    holders = list(values)
    array = (c_void_p * len(holders))(*[ctypes.cast(byref(v), c_void_p) for v in holders])
    array._holders = holders  # keep the scalars alive
    return array


# --------------------------------------------------------------------------- NVRTC

nvrtc = ctypes.CDLL(LIB_NVRTC)
nvrtcProgram = c_void_p
nvrtcVersion = _proto(nvrtc, "nvrtcVersion", c_int, POINTER(c_int), POINTER(c_int))
nvrtcCreateProgram = _proto(nvrtc, "nvrtcCreateProgram", c_int, POINTER(nvrtcProgram), c_char_p, c_char_p, c_int,
                            POINTER(c_char_p), POINTER(c_char_p))
nvrtcCompileProgram = _proto(nvrtc, "nvrtcCompileProgram", c_int, nvrtcProgram, c_int, POINTER(c_char_p))
nvrtcGetCUBINSize = _proto(nvrtc, "nvrtcGetCUBINSize", c_int, nvrtcProgram, POINTER(c_size_t))
nvrtcGetCUBIN = _proto(nvrtc, "nvrtcGetCUBIN", c_int, nvrtcProgram, c_char_p)
nvrtcGetPTXSize = _proto(nvrtc, "nvrtcGetPTXSize", c_int, nvrtcProgram, POINTER(c_size_t))
nvrtcGetPTX = _proto(nvrtc, "nvrtcGetPTX", c_int, nvrtcProgram, c_char_p)
nvrtcGetProgramLogSize = _proto(nvrtc, "nvrtcGetProgramLogSize", c_int, nvrtcProgram, POINTER(c_size_t))
nvrtcGetProgramLog = _proto(nvrtc, "nvrtcGetProgramLog", c_int, nvrtcProgram, c_char_p)
nvrtcDestroyProgram = _proto(nvrtc, "nvrtcDestroyProgram", c_int, POINTER(nvrtcProgram))


def compile_source(source: str, name: str, arch: str, *, cubin: bool = True) -> tuple[bytes, str]:
    prog = nvrtcProgram()
    r = nvrtcCreateProgram(byref(prog), source.encode(), name.encode(), 0, None, None)
    if r:
        raise ProbeError(f"nvrtcCreateProgram {r}")
    try:
        opts = [f"--gpu-architecture={arch}".encode(), b"--std=c++17"]
        arr = (c_char_p * len(opts))(*opts)
        r = nvrtcCompileProgram(prog, len(opts), arr)
        size = c_size_t()
        nvrtcGetProgramLogSize(prog, byref(size))
        log = ctypes.create_string_buffer(size.value)
        nvrtcGetProgramLog(prog, log)
        if r:
            raise ProbeError(f"nvrtcCompileProgram {r}: {log.value.decode(errors='replace')}")
        if cubin:
            nvrtcGetCUBINSize(prog, byref(size))
            buf = ctypes.create_string_buffer(size.value)
            nvrtcGetCUBIN(prog, buf)
        else:
            nvrtcGetPTXSize(prog, byref(size))
            buf = ctypes.create_string_buffer(size.value)
            nvrtcGetPTX(prog, buf)
        return buf.raw, log.value.decode(errors="replace")
    finally:
        nvrtcDestroyProgram(byref(prog))


# --------------------------------------------------------------------------- NVML

nvml = ctypes.CDLL(LIB_NVML)
nvmlDevice = c_void_p


class nvmlPciInfo(ctypes.Structure):
    _fields_ = [("busIdLegacy", c_char * 16), ("domain", c_uint), ("bus", c_uint), ("device", c_uint),
                ("pciDeviceId", c_uint), ("pciSubSystemId", c_uint), ("busId", c_char * 32)]


class nvmlMemory_v2(ctypes.Structure):
    _fields_ = [("version", c_uint), ("total", c_ulonglong), ("reserved", c_ulonglong), ("free", c_ulonglong),
                ("used", c_ulonglong)]


class nvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", c_uint), ("memory", c_uint)]


class nvmlProcessInfo(ctypes.Structure):
    _fields_ = [("pid", c_uint), ("usedGpuMemory", c_ulonglong), ("gpuInstanceId", c_uint),
                ("computeInstanceId", c_uint)]


class nvmlProcessUtilizationSample(ctypes.Structure):
    _fields_ = [("pid", c_uint), ("timeStamp", c_ulonglong), ("smUtil", c_uint), ("memUtil", c_uint),
                ("encUtil", c_uint), ("decUtil", c_uint)]


class nvmlTemperature_v1(ctypes.Structure):
    _fields_ = [("version", c_uint), ("sensorType", c_int), ("temperature", c_int)]


def nvml_struct_version(struct: type, version: int) -> int:
    return ctypes.sizeof(struct) | (version << 24)


nvmlErrorString = _proto(nvml, "nvmlErrorString", c_char_p, c_int)
nvmlInit = _proto(nvml, "nvmlInit_v2", c_int)
nvmlShutdown = _proto(nvml, "nvmlShutdown", c_int)
nvmlSystemGetDriverVersion = _proto(nvml, "nvmlSystemGetDriverVersion", c_int, c_char_p, c_uint)
nvmlSystemGetNVMLVersion = _proto(nvml, "nvmlSystemGetNVMLVersion", c_int, c_char_p, c_uint)
nvmlSystemGetCudaDriverVersion = _proto(nvml, "nvmlSystemGetCudaDriverVersion_v2", c_int, POINTER(c_int))
nvmlDeviceGetCount = _proto(nvml, "nvmlDeviceGetCount_v2", c_int, POINTER(c_uint))
nvmlDeviceGetHandleByIndex = _proto(nvml, "nvmlDeviceGetHandleByIndex_v2", c_int, c_uint, POINTER(nvmlDevice))
nvmlDeviceGetHandleByUUID = _proto(nvml, "nvmlDeviceGetHandleByUUID", c_int, c_char_p, POINTER(nvmlDevice))
nvmlDeviceGetName = _proto(nvml, "nvmlDeviceGetName", c_int, nvmlDevice, c_char_p, c_uint)
nvmlDeviceGetUUID = _proto(nvml, "nvmlDeviceGetUUID", c_int, nvmlDevice, c_char_p, c_uint)
nvmlDeviceGetSerial = _proto(nvml, "nvmlDeviceGetSerial", c_int, nvmlDevice, c_char_p, c_uint)
nvmlDeviceGetBoardPartNumber = _proto(nvml, "nvmlDeviceGetBoardPartNumber", c_int, nvmlDevice, c_char_p, c_uint)
nvmlDeviceGetVbiosVersion = _proto(nvml, "nvmlDeviceGetVbiosVersion", c_int, nvmlDevice, c_char_p, c_uint)
nvmlDeviceGetPciInfo = _proto(nvml, "nvmlDeviceGetPciInfo_v3", c_int, nvmlDevice, POINTER(nvmlPciInfo))
nvmlDeviceGetMemoryInfo = _proto(nvml, "nvmlDeviceGetMemoryInfo_v2", c_int, nvmlDevice, POINTER(nvmlMemory_v2))
nvmlDeviceGetClockInfo = _proto(nvml, "nvmlDeviceGetClockInfo", c_int, nvmlDevice, c_int, POINTER(c_uint))
nvmlDeviceGetMaxClockInfo = _proto(nvml, "nvmlDeviceGetMaxClockInfo", c_int, nvmlDevice, c_int, POINTER(c_uint))
nvmlDeviceGetTemperatureV = _proto(nvml, "nvmlDeviceGetTemperatureV", c_int, nvmlDevice, POINTER(nvmlTemperature_v1))
nvmlDeviceGetPowerUsage = _proto(nvml, "nvmlDeviceGetPowerUsage", c_int, nvmlDevice, POINTER(c_uint))
nvmlDeviceGetTotalEnergyConsumption = _proto(nvml, "nvmlDeviceGetTotalEnergyConsumption", c_int, nvmlDevice,
                                             POINTER(c_ulonglong))
nvmlDeviceGetPerformanceState = _proto(nvml, "nvmlDeviceGetPerformanceState", c_int, nvmlDevice, POINTER(c_int))
nvmlDeviceGetCurrentClocksEventReasons = _proto(nvml, "nvmlDeviceGetCurrentClocksEventReasons", c_int, nvmlDevice,
                                                POINTER(c_ulonglong))
nvmlDeviceGetUtilizationRates = _proto(nvml, "nvmlDeviceGetUtilizationRates", c_int, nvmlDevice,
                                       POINTER(nvmlUtilization))
nvmlDeviceGetComputeRunningProcesses = _proto(nvml, "nvmlDeviceGetComputeRunningProcesses_v3", c_int, nvmlDevice,
                                              POINTER(c_uint), POINTER(nvmlProcessInfo))
nvmlDeviceGetProcessUtilization = _proto(nvml, "nvmlDeviceGetProcessUtilization", c_int, nvmlDevice,
                                         POINTER(nvmlProcessUtilizationSample), POINTER(c_uint), c_ulonglong)
nvmlDeviceGetCurrPcieLinkGeneration = _proto(nvml, "nvmlDeviceGetCurrPcieLinkGeneration", c_int, nvmlDevice,
                                             POINTER(c_uint))
nvmlDeviceGetCurrPcieLinkWidth = _proto(nvml, "nvmlDeviceGetCurrPcieLinkWidth", c_int, nvmlDevice, POINTER(c_uint))
nvmlDeviceGetPcieThroughput = _proto(nvml, "nvmlDeviceGetPcieThroughput", c_int, nvmlDevice, c_int, POINTER(c_uint))
nvmlDeviceGetPersistenceMode = _proto(nvml, "nvmlDeviceGetPersistenceMode", c_int, nvmlDevice, POINTER(c_int))
nvmlDeviceGetComputeMode = _proto(nvml, "nvmlDeviceGetComputeMode", c_int, nvmlDevice, POINTER(c_int))

NVML_SUCCESS = 0
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_ERROR_NOT_FOUND = 6
NVML_CLOCK_GRAPHICS, NVML_CLOCK_SM, NVML_CLOCK_MEM = 0, 1, 2
NVML_PCIE_UTIL_TX_BYTES, NVML_PCIE_UTIL_RX_BYTES = 0, 1
CLOCK_EVENT_REASONS = {
    0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
    0x10: "sync_boost", 0x20: "sw_thermal_slowdown", 0x40: "hw_thermal_slowdown", 0x80: "hw_power_brake_slowdown",
    0x100: "display_clock_setting",
}


def nvml_check(result: int, what: str) -> None:
    if result != NVML_SUCCESS:
        raise ProbeError(f"{what}: NVML {result} {nvmlErrorString(result)!r}")


def nvml_string(fn, handle, size: int = 96) -> str:
    buf = ctypes.create_string_buffer(size)
    nvml_check(fn(handle, buf, size), fn.__name__)
    return buf.value.decode()


def nvml_handle(uuid: str = TARGET_UUID) -> c_void_p:
    h = nvmlDevice()
    nvml_check(nvmlDeviceGetHandleByUUID(uuid.encode(), byref(h)), "nvmlDeviceGetHandleByUUID")
    return h


def compute_processes(handle) -> list[dict]:
    count = c_uint(0)
    r = nvmlDeviceGetComputeRunningProcesses(handle, byref(count), None)
    if r not in (NVML_SUCCESS, NVML_ERROR_INSUFFICIENT_SIZE):
        nvml_check(r, "nvmlDeviceGetComputeRunningProcesses(size)")
    if count.value == 0:
        return []
    infos = (nvmlProcessInfo * (count.value + 4))()
    count = c_uint(len(infos))
    nvml_check(nvmlDeviceGetComputeRunningProcesses(handle, byref(count), infos), "nvmlDeviceGetComputeRunningProcesses")
    return [{"pid": infos[i].pid, "used_gpu_memory_bytes": infos[i].usedGpuMemory} for i in range(count.value)]


def process_utilization(handle, since_us: int) -> tuple[str, list[dict]]:
    count = c_uint(0)
    r = nvmlDeviceGetProcessUtilization(handle, None, byref(count), c_ulonglong(since_us))
    if r == NVML_ERROR_NOT_FOUND:
        return "no_samples", []
    if r not in (NVML_SUCCESS, NVML_ERROR_INSUFFICIENT_SIZE):
        return f"error:{r}:{nvmlErrorString(r).decode()}", []
    samples = (nvmlProcessUtilizationSample * max(count.value, 1))()
    r = nvmlDeviceGetProcessUtilization(handle, samples, byref(count), c_ulonglong(since_us))
    if r == NVML_ERROR_NOT_FOUND:
        return "no_samples", []
    if r != NVML_SUCCESS:
        return f"error:{r}:{nvmlErrorString(r).decode()}", []
    return "ok", [{"pid": s.pid, "timestamp_us": s.timeStamp, "sm_util": s.smUtil, "mem_util": s.memUtil}
                  for s in samples[: count.value]]


def clock_event_reasons(mask: int) -> list[str]:
    return [name for bit, name in CLOCK_EVENT_REASONS.items() if mask & bit]
