"""ctypes bindings for the NVIDIA DUT adapter: CUDA driver API, NVRTC and NVML.

Only the calls the adapter uses are declared.  Libraries are loaded on first use
(``libs()``), so importing this module never requires an NVIDIA driver.  The
NVRTC library comes from ``$SDV_BMT_CUDA_HOME`` (default ``/usr/local/cuda-13.0``):
the investigated host's default ``/usr/local/cuda`` is CUDA 12.4, which cannot
target compute capability 12.0 (Stage 2a-G).
"""

from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, byref, c_char, c_char_p, c_float, c_int, c_longlong, c_size_t, c_uint, c_uint64
from ctypes import c_ulonglong, c_void_p
from dataclasses import dataclass
from typing import Any, Optional

DEFAULT_CUDA_HOME = "/usr/local/cuda-13.0"

CUDA_SUCCESS = 0
CUDA_ERROR_NOT_READY = 600
CU_CTX_SCHED_BLOCKING_SYNC = 0x04
CU_EVENT_DEFAULT = 0x0
CU_EVENT_DISABLE_TIMING = 0x2
CU_STREAM_NON_BLOCKING = 0x1
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_CC_MAJOR = 75
CU_DEVICE_ATTRIBUTE_CC_MINOR = 76

NVML_SUCCESS = 0
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_ERROR_NOT_FOUND = 6
NVML_CLOCK_SM, NVML_CLOCK_MEM = 1, 2
# nvmlDeviceGetFieldValues ids (nvml.h, CUDA 13.0)
NVML_FIELDS = {
    74: "throttle_ns_sw_power_cap",
    76: "throttle_ns_sync_boost",
    251: "throttle_ns_sw_thermal_slowdown",
    252: "throttle_ns_hw_thermal_slowdown",
    197: "pcie_tx_bytes",
    198: "pcie_rx_bytes",
    94: "pcie_replay_counter",
}
CLOCK_EVENT_REASONS = {
    0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
    0x10: "sync_boost", 0x20: "sw_thermal_slowdown", 0x40: "hw_thermal_slowdown", 0x80: "hw_power_brake_slowdown",
    0x100: "display_clock_setting",
}


class NvidiaError(RuntimeError):
    def __init__(self, api: str, code: int, name: str, message: str) -> None:
        super().__init__(f"{api}: {code} {name} {message}")
        self.api, self.code, self.name = api, code, name


class CUuuid(ctypes.Structure):
    _fields_ = [("bytes", c_char * 16)]


class NvmlPciInfo(ctypes.Structure):
    _fields_ = [("busIdLegacy", c_char * 16), ("domain", c_uint), ("bus", c_uint), ("device", c_uint),
                ("pciDeviceId", c_uint), ("pciSubSystemId", c_uint), ("busId", c_char * 32)]


class NvmlMemoryV2(ctypes.Structure):
    _fields_ = [("version", c_uint), ("total", c_ulonglong), ("reserved", c_ulonglong), ("free", c_ulonglong),
                ("used", c_ulonglong)]


class NvmlProcessInfo(ctypes.Structure):
    _fields_ = [("pid", c_uint), ("usedGpuMemory", c_ulonglong), ("gpuInstanceId", c_uint),
                ("computeInstanceId", c_uint)]


class NvmlProcessUtilizationSample(ctypes.Structure):
    _fields_ = [("pid", c_uint), ("timeStamp", c_ulonglong), ("smUtil", c_uint), ("memUtil", c_uint),
                ("encUtil", c_uint), ("decUtil", c_uint)]


class NvmlTemperatureV1(ctypes.Structure):
    _fields_ = [("version", c_uint), ("sensorType", c_int), ("temperature", c_int)]


class NvmlFieldValue(ctypes.Structure):
    _fields_ = [("fieldId", c_uint), ("scopeId", c_uint), ("timestamp", c_longlong), ("latencyUsec", c_longlong),
                ("valueType", c_int), ("nvmlReturn", c_int), ("value", c_ulonglong)]


def nvml_struct_version(struct: type, version: int) -> int:
    return ctypes.sizeof(struct) | (version << 24)


def _declare(lib: ctypes.CDLL, table: dict[str, tuple[Any, list[Any]]]) -> None:
    for name, (restype, argtypes) in table.items():
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes


@dataclass(frozen=True)
class Libraries:
    cuda: ctypes.CDLL
    nvrtc: ctypes.CDLL
    nvml: ctypes.CDLL
    paths: dict[str, str]


_LIBS: Optional[Libraries] = None


def libs() -> Libraries:
    """Load and declare the libraries once (raises OSError when unavailable)."""
    global _LIBS
    if _LIBS is not None:
        return _LIBS
    cuda_home = os.environ.get("SDV_BMT_CUDA_HOME", DEFAULT_CUDA_HOME)
    paths = {
        "cuda_driver": "libcuda.so.1",
        "nvrtc": f"{cuda_home}/targets/x86_64-linux/lib/libnvrtc.so.13",
        "nvml": os.environ.get("SDV_BMT_NVML_LIBRARY", "libnvidia-ml.so.1"),
    }
    cuda = ctypes.CDLL(paths["cuda_driver"])
    nvrtc = ctypes.CDLL(paths["nvrtc"])
    nvml = ctypes.CDLL(paths["nvml"])
    P = POINTER
    _declare(cuda, {
        "cuGetErrorName": (c_int, [c_int, P(c_char_p)]),
        "cuGetErrorString": (c_int, [c_int, P(c_char_p)]),
        "cuInit": (c_int, [c_uint]),
        "cuDriverGetVersion": (c_int, [P(c_int)]),
        "cuDeviceGetCount": (c_int, [P(c_int)]),
        "cuDeviceGet": (c_int, [P(c_int), c_int]),
        "cuDeviceGetName": (c_int, [c_char_p, c_int, c_int]),
        "cuDeviceGetUuid_v2": (c_int, [P(CUuuid), c_int]),
        "cuDeviceGetPCIBusId": (c_int, [c_char_p, c_int, c_int]),
        "cuDeviceGetAttribute": (c_int, [P(c_int), c_int, c_int]),
        "cuDeviceTotalMem_v2": (c_int, [P(c_size_t), c_int]),
        "cuDevicePrimaryCtxRetain": (c_int, [P(c_void_p), c_int]),
        "cuDevicePrimaryCtxRelease_v2": (c_int, [c_int]),
        "cuDevicePrimaryCtxSetFlags_v2": (c_int, [c_int, c_uint]),
        "cuCtxSetCurrent": (c_int, [c_void_p]),
        "cuMemAlloc_v2": (c_int, [P(c_uint64), c_size_t]),
        "cuMemFree_v2": (c_int, [c_uint64]),
        "cuMemAllocHost_v2": (c_int, [P(c_void_p), c_size_t]),
        "cuMemFreeHost": (c_int, [c_void_p]),
        "cuMemcpyHtoD_v2": (c_int, [c_uint64, c_void_p, c_size_t]),
        "cuMemcpyDtoH_v2": (c_int, [c_void_p, c_uint64, c_size_t]),
        "cuMemcpyHtoDAsync_v2": (c_int, [c_uint64, c_void_p, c_size_t, c_void_p]),
        "cuMemcpyDtoHAsync_v2": (c_int, [c_void_p, c_uint64, c_size_t, c_void_p]),
        "cuStreamCreate": (c_int, [P(c_void_p), c_uint]),
        "cuStreamDestroy_v2": (c_int, [c_void_p]),
        "cuStreamQuery": (c_int, [c_void_p]),
        "cuStreamSynchronize": (c_int, [c_void_p]),
        "cuEventCreate": (c_int, [P(c_void_p), c_uint]),
        "cuEventDestroy_v2": (c_int, [c_void_p]),
        "cuEventRecord": (c_int, [c_void_p, c_void_p]),
        "cuEventQuery": (c_int, [c_void_p]),
        "cuEventElapsedTime_v2": (c_int, [P(c_float), c_void_p, c_void_p]),
        "cuModuleLoadData": (c_int, [P(c_void_p), c_void_p]),
        "cuModuleUnload": (c_int, [c_void_p]),
        "cuModuleGetFunction": (c_int, [P(c_void_p), c_void_p, c_char_p]),
        "cuLaunchKernel": (c_int, [c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_void_p,
                                   P(c_void_p), P(c_void_p)]),
    })
    _declare(nvrtc, {
        "nvrtcVersion": (c_int, [P(c_int), P(c_int)]),
        "nvrtcCreateProgram": (c_int, [P(c_void_p), c_char_p, c_char_p, c_int, P(c_char_p), P(c_char_p)]),
        "nvrtcCompileProgram": (c_int, [c_void_p, c_int, P(c_char_p)]),
        "nvrtcGetCUBINSize": (c_int, [c_void_p, P(c_size_t)]),
        "nvrtcGetCUBIN": (c_int, [c_void_p, c_char_p]),
        "nvrtcGetProgramLogSize": (c_int, [c_void_p, P(c_size_t)]),
        "nvrtcGetProgramLog": (c_int, [c_void_p, c_char_p]),
        "nvrtcDestroyProgram": (c_int, [P(c_void_p)]),
    })
    _declare(nvml, {
        "nvmlErrorString": (c_char_p, [c_int]),
        "nvmlInit_v2": (c_int, []),
        "nvmlShutdown": (c_int, []),
        "nvmlSystemGetDriverVersion": (c_int, [c_char_p, c_uint]),
        "nvmlSystemGetCudaDriverVersion_v2": (c_int, [P(c_int)]),
        "nvmlDeviceGetHandleByUUID": (c_int, [c_char_p, P(c_void_p)]),
        "nvmlDeviceGetName": (c_int, [c_void_p, c_char_p, c_uint]),
        "nvmlDeviceGetVbiosVersion": (c_int, [c_void_p, c_char_p, c_uint]),
        "nvmlDeviceGetBoardPartNumber": (c_int, [c_void_p, c_char_p, c_uint]),
        "nvmlDeviceGetPciInfo_v3": (c_int, [c_void_p, P(NvmlPciInfo)]),
        "nvmlDeviceGetMemoryInfo_v2": (c_int, [c_void_p, P(NvmlMemoryV2)]),
        "nvmlDeviceGetClockInfo": (c_int, [c_void_p, c_int, P(c_uint)]),
        "nvmlDeviceGetTemperatureV": (c_int, [c_void_p, P(NvmlTemperatureV1)]),
        "nvmlDeviceGetPowerUsage": (c_int, [c_void_p, P(c_uint)]),
        "nvmlDeviceGetPerformanceState": (c_int, [c_void_p, P(c_int)]),
        "nvmlDeviceGetCurrentClocksEventReasons": (c_int, [c_void_p, P(c_ulonglong)]),
        "nvmlDeviceGetCurrPcieLinkGeneration": (c_int, [c_void_p, P(c_uint)]),
        "nvmlDeviceGetCurrPcieLinkWidth": (c_int, [c_void_p, P(c_uint)]),
        "nvmlDeviceGetComputeRunningProcesses_v3": (c_int, [c_void_p, P(c_uint), P(NvmlProcessInfo)]),
        "nvmlDeviceGetProcessUtilization": (c_int, [c_void_p, P(NvmlProcessUtilizationSample), P(c_uint),
                                                    c_ulonglong]),
        "nvmlDeviceGetFieldValues": (c_int, [c_void_p, c_int, P(NvmlFieldValue)]),
    })
    _LIBS = Libraries(cuda, nvrtc, nvml, paths)
    return _LIBS


# --------------------------------------------------------------------------- checked helpers


def cu(api: str, result: int) -> None:
    if result != CUDA_SUCCESS:
        raise cuda_error(api, result)


def cuda_error(api: str, result: int) -> NvidiaError:
    lib = libs().cuda
    name, text = c_char_p(), c_char_p()
    lib.cuGetErrorName(result, byref(name))
    lib.cuGetErrorString(result, byref(text))
    return NvidiaError(api, result, (name.value or b"?").decode(), (text.value or b"").decode())


def nv(api: str, result: int) -> None:
    if result != NVML_SUCCESS:
        raise NvidiaError(api, result, "NVML", (libs().nvml.nvmlErrorString(result) or b"").decode())


def uuid_string(raw: bytes) -> str:
    h = raw.hex()
    return f"GPU-{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def nvml_text(fn: Any, handle: c_void_p, size: int = 96) -> str:
    buf = ctypes.create_string_buffer(size)
    nv(fn.__name__, fn(handle, buf, size))
    return buf.value.decode()


def kernel_args(*values: Any) -> ctypes.Array:
    """``void**`` argument array for cuLaunchKernel; the scalars are kept alive on the array."""
    array = (c_void_p * len(values))(*[ctypes.cast(byref(v), c_void_p) for v in values])
    array._keep = values  # type: ignore[attr-defined]
    return array


def compile_cubin(source: str, name: str, arch: str) -> tuple[bytes, str, list[str]]:
    """Compile CUDA C source with NVRTC to a cubin for ``arch`` (e.g. sm_120)."""
    lib = libs().nvrtc
    prog = c_void_p()
    r = lib.nvrtcCreateProgram(byref(prog), source.encode(), name.encode(), 0, None, None)
    if r:
        raise NvidiaError("nvrtcCreateProgram", r, "NVRTC", "")
    options = [f"--gpu-architecture={arch}", "--std=c++17"]
    try:
        array = (c_char_p * len(options))(*[o.encode() for o in options])
        r = lib.nvrtcCompileProgram(prog, len(options), array)
        size = c_size_t()
        lib.nvrtcGetProgramLogSize(prog, byref(size))
        log = ctypes.create_string_buffer(max(size.value, 1))
        lib.nvrtcGetProgramLog(prog, log)
        if r:
            raise NvidiaError("nvrtcCompileProgram", r, "NVRTC", log.value.decode(errors="replace"))
        lib.nvrtcGetCUBINSize(prog, byref(size))
        cubin = ctypes.create_string_buffer(size.value)
        lib.nvrtcGetCUBIN(prog, cubin)
        return cubin.raw, log.value.decode(errors="replace"), options
    finally:
        lib.nvrtcDestroyProgram(byref(prog))


def nvrtc_version() -> str:
    major, minor = c_int(), c_int()
    libs().nvrtc.nvrtcVersion(byref(major), byref(minor))
    return f"{major.value}.{minor.value}"
