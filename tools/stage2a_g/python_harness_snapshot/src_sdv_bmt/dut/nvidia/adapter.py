"""NVIDIA reference-accelerator DUT adapter (``adapter: nvidia_cuda``), Stage 1R-G.

The GPU is the first *reference* accelerator: it validates the realtime harness,
measurement boundaries and realized-condition evidence on real hardware.  It is
not the NPU target and its results do not predict NPU behaviour.

Execution path (chosen from the Stage 2a-G findings):

* device selected by UUID (``dut.device``), process primary context with
  ``CU_CTX_SCHED_BLOCKING_SYNC``; the device program (CUDA C source carried by
  the workload spec, SHA-256 pinned by ``workload.artifact_sha256``) is compiled
  with NVRTC for the device's compute capability;
* all device buffers and the pinned output buffer are allocated in
  ``prepare`` (``buffer_policy: preallocated_persistent``); ``prepare_input``
  stages each item's logical input in its own pinned host buffer
  (``residency: host_pinned``, the per-request host-to-device copy is then
  inside the DUT-service interval) or in its own device buffer verified by
  read-back (``residency: device_resident``, copy excluded);
* ``buffer_policy: per_request_alloc`` (host_pinned only) is a declared design
  option for the buffer-lifecycle characterization: ``submit`` allocates the
  device input buffer (``cuMemAlloc``) and the completing ``poll`` frees it
  (``cuMemFree``), so allocation and release are inside the DUT-service
  interval.  It is part of the DUT config and of the submit/complete semantics
  (hence of the measurement boundary);
* ``submit`` enqueues [H2D] -> kernel -> D2H on one non-blocking stream with
  CUDA events and returns; ``poll`` is ``cuEventQuery`` on the last event.

Observation: ``instrumentation: cuda_events`` brackets each segment with CUDA
events (device-clock durations; the bracket includes event overhead, Stage
2a-G); ``none`` records only a completion event.  The reference program writes
``%globaltimer``/``clock64`` at kernel start and end, giving the effective SM
clock during each request.  NVML snapshots are taken only at the lifecycle
points the runner requests (before MEASURE, after DRAIN).

NVIDIA-specific concepts (streams, events, NVML fields) stay inside this
package; the evidence it reports uses the portable realized-condition names.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import time
from ctypes import byref, c_float, c_int, c_size_t, c_uint, c_uint64, c_ulonglong, c_void_p
from dataclasses import dataclass, field
from typing import Any, Optional

from ...contracts.protocols import LifecycleError
from ...contracts.records import (
    DUTCapabilities,
    DUTCompletion,
    DUTDescriptor,
    DUTRequest,
    ErrorInfo,
    InputItem,
    PreparedInfo,
    RealizationEvidence,
    WorkloadInputContract,
)
from ...contracts.specs import DUTConfig, WorkloadSpec, parse_strict, workload_identity_digest
from ...contracts.status import CompletionModel, EvidenceValidity, ExecutionKind, RealizedCondition, RunClockDomain
from . import bindings as b

RESIDENCIES = ("host_pinned", "device_resident")
INSTRUMENTATIONS = ("cuda_events", "none")
BUFFER_POLICIES = ("preallocated_persistent", "per_request_alloc")
TIMING_METHOD = "nvidia:cuda_event_pair:bracket"
STEADY_STATE_RULE = "unapproved:steady_state_band"  # candidate rule, not approved (Stage 2a-G section 12)


@dataclass(frozen=True)
class NvidiaOptions:
    residency: str = "host_pinned"
    instrumentation: str = "cuda_events"
    context_sched: str = "blocking_sync"
    buffer_policy: str = "preallocated_persistent"

    def __post_init__(self) -> None:
        if self.residency not in RESIDENCIES:
            raise ValueError(f"residency must be one of {RESIDENCIES}")
        if self.instrumentation not in INSTRUMENTATIONS:
            raise ValueError(f"instrumentation must be one of {INSTRUMENTATIONS}")
        if self.context_sched != "blocking_sync":
            raise ValueError("context_sched supports blocking_sync only")
        if self.buffer_policy not in BUFFER_POLICIES:
            raise ValueError(f"buffer_policy must be one of {BUFFER_POLICIES}")
        if self.buffer_policy == "per_request_alloc" and self.residency != "host_pinned":
            raise ValueError("buffer_policy per_request_alloc applies to residency host_pinned only")


@dataclass(frozen=True)
class DeviceProgram:
    """Calling convention: ``entry(input, scratch_0..scratch_n, output)``, one block."""

    language: str
    entry: str
    block_threads: int
    input_bytes: int
    scratch_bytes: tuple[int, ...]
    output_bytes: int
    task_output_bytes: int
    timing_offset: Optional[int]
    source: str

    def __post_init__(self) -> None:
        if self.language != "cuda_c":
            raise ValueError("device_program.language must be cuda_c")
        if not 1 <= self.block_threads <= 1024:
            raise ValueError("device_program.block_threads must be in 1..1024")
        if self.input_bytes <= 0 or self.output_bytes <= 0 or any(s <= 0 for s in self.scratch_bytes):
            raise ValueError("device_program buffer sizes must be > 0")
        if not 0 < self.task_output_bytes <= self.output_bytes:
            raise ValueError("device_program.task_output_bytes must be in 1..output_bytes")
        if self.timing_offset is not None and self.timing_offset + 32 > self.output_bytes:
            raise ValueError("device_program.timing_offset leaves no room for 4 u64 timing words")


@dataclass
class NvidiaPreparedInput:
    item_index: int
    nbytes: int
    sha256: str
    host_ptr: Optional[int] = None
    device_ptr: Optional[int] = None


@dataclass
class _Handle:
    request_id: int
    item_index: int
    h2d_bytes: int
    transient_input: int = 0  # per_request_alloc: device input buffer to free on completion


@dataclass
class _Accumulator:
    requests: int = 0
    errors: int = 0
    in_kernel_mhz: list[float] = field(default_factory=list)
    h2d_bytes: set[int] = field(default_factory=set)


def _quantiles(values: list[float]) -> Optional[dict[str, Any]]:
    if not values:
        return None
    v = sorted(values)
    n = len(v)
    rank = lambda p: v[min(n - 1, max(0, math.ceil(p * n) - 1))]  # noqa: E731
    return {"n": n, "min": v[0], "p50": rank(0.5), "max": v[-1]}


class NvidiaCudaDUT:
    ADAPTER = "nvidia_cuda"
    VERSION = "1"

    def __init__(self, config: DUTConfig) -> None:
        if config.adapter != self.ADAPTER:
            raise ValueError(f"DUT adapter must be {self.ADAPTER!r}")
        self._config = config
        self._options = parse_strict(NvidiaOptions, config.runtime_options, "dut.runtime_options")
        self._uuid = config.device
        self._identity: Optional[dict[str, Any]] = None
        self._state = "created"
        self._nvml_handle: Optional[c_void_p] = None
        self._cu_dev: Optional[int] = None
        self._ctx_retained = False
        self._module = c_void_p()
        self._function = c_void_p()
        self._stream = c_void_p()
        self._events: dict[str, c_void_p] = {}
        self._program: Optional[DeviceProgram] = None
        self._d_input = 0
        self._d_scratch: list[int] = []
        self._d_output = 0
        self._h_output = c_void_p()
        self._device_allocs: list[int] = []  # live allocations freed at cleanup
        self._pinned_allocs: list[int] = []
        self._device_alloc_count = 0  # allocations made so far (including transient ones)
        self._pinned_alloc_count = 0
        self._cubin_sha256: Optional[str] = None
        self._transient = 0  # per_request_alloc: device input buffer of the request in flight
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._acc = _Accumulator()
        self._pending: Optional[_Handle] = None

    # ------------------------------------------------------------------ identity
    def _init_nvml(self) -> c_void_p:
        if self._nvml_handle is None:
            L = b.libs()
            b.nv("nvmlInit_v2", L.nvml.nvmlInit_v2())
            handle = c_void_p()
            b.nv("nvmlDeviceGetHandleByUUID", L.nvml.nvmlDeviceGetHandleByUUID(self._uuid.encode(), byref(handle)))
            self._nvml_handle = handle
        return self._nvml_handle

    def _find_cuda_device(self) -> int:
        if self._cu_dev is None:
            L = b.libs()
            b.cu("cuInit", L.cuda.cuInit(0))
            count = c_int()
            b.cu("cuDeviceGetCount", L.cuda.cuDeviceGetCount(byref(count)))
            for ordinal in range(count.value):
                dev = c_int()
                b.cu("cuDeviceGet", L.cuda.cuDeviceGet(byref(dev), ordinal))
                u = b.CUuuid()
                b.cu("cuDeviceGetUuid_v2", L.cuda.cuDeviceGetUuid_v2(byref(u), dev))
                if b.uuid_string(u.bytes) == self._uuid:
                    self._cu_dev = dev.value
                    break
            else:
                raise LifecycleError(f"no CUDA device with UUID {self._uuid}")
        return self._cu_dev

    def device_identity(self) -> dict[str, Any]:
        """Exact identity detected at runtime (never assumed from the product name)."""
        if self._identity is not None:
            return self._identity
        L = b.libs()
        h = self._init_nvml()
        dev = self._find_cuda_device()
        pci = b.NvmlPciInfo()
        b.nv("nvmlDeviceGetPciInfo_v3", L.nvml.nvmlDeviceGetPciInfo_v3(h, byref(pci)))
        mem = b.NvmlMemoryV2(version=b.nvml_struct_version(b.NvmlMemoryV2, 2))
        b.nv("nvmlDeviceGetMemoryInfo_v2", L.nvml.nvmlDeviceGetMemoryInfo_v2(h, byref(mem)))
        driver = ctypes.create_string_buffer(96)
        b.nv("nvmlSystemGetDriverVersion", L.nvml.nvmlSystemGetDriverVersion(driver, 96))
        cuda_api = c_int()
        b.cu("cuDriverGetVersion", L.cuda.cuDriverGetVersion(byref(cuda_api)))
        attrs = {}
        for name, attr in (("cc_major", b.CU_DEVICE_ATTRIBUTE_CC_MAJOR), ("cc_minor", b.CU_DEVICE_ATTRIBUTE_CC_MINOR),
                           ("sm_count", b.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)):
            v = c_int()
            b.cu("cuDeviceGetAttribute", L.cuda.cuDeviceGetAttribute(byref(v), attr, dev))
            attrs[name] = v.value
        self._identity = {
            "role": "reference_accelerator_validation",
            "accelerator_family": "GPU",
            "vendor": "NVIDIA",
            "nvidia.product_name": b.nvml_text(L.nvml.nvmlDeviceGetName, h),
            "nvidia.uuid": self._uuid,
            "nvidia.pci_bus_id": pci.busId.decode(),
            "nvidia.pci_device_id": hex(pci.pciDeviceId),
            "nvidia.pci_subsystem_id": hex(pci.pciSubSystemId),
            "nvidia.vbios_version": b.nvml_text(L.nvml.nvmlDeviceGetVbiosVersion, h),
            "nvidia.board_part_number": b.nvml_text(L.nvml.nvmlDeviceGetBoardPartNumber, h),
            "nvidia.memory_total_bytes": mem.total,
            "nvidia.driver_version": driver.value.decode(),
            "nvidia.cuda_driver_api_version": cuda_api.value,
            "nvidia.compute_capability": f"{attrs['cc_major']}.{attrs['cc_minor']}",
            "nvidia.sm_count": attrs["sm_count"],
            "nvidia.nvrtc_version": b.nvrtc_version(),
            "nvidia.nvrtc_library": L.paths["nvrtc"],
        }
        return self._identity

    def describe(self) -> DUTDescriptor:
        identity = self.device_identity()
        h2d = "H2D from the item's pinned host buffer, " if self._options.residency == "host_pinned" else ""
        per_request = self._options.buffer_policy == "per_request_alloc"
        alloc = "allocates the device input buffer (cuMemAlloc), " if per_request else ""
        free = "; the completing poll then frees the device input buffer (cuMemFree)" if per_request else ""
        return DUTDescriptor(
            adapter=self.ADAPTER,
            adapter_version=self.VERSION,
            implementation_kind=ExecutionKind.ACTUAL,
            device=f"{identity['nvidia.product_name']} {self._uuid}",
            completion_model=CompletionModel.POLLING,
            clock_domain=RunClockDomain.MONOTONIC,
            submit_semantics=f"NvidiaCudaDUT.submit() {alloc}enqueues {h2d}the kernel and the D2H copy of the "
                             "output on one non-blocking CUDA stream and returns",
            complete_semantics=f"cuEventQuery on the event recorded after the D2H copy returns CUDA_SUCCESS{free}",
            device_identity=identity,
        )

    def capabilities(self) -> DUTCapabilities:
        return DUTCapabilities(completion_model=CompletionModel.POLLING, max_inflight=1,
                               virtual_timing=False, device_timing=self._options.instrumentation == "cuda_events")

    def measurement_declaration(self) -> dict[str, Any]:
        events = self._options.instrumentation == "cuda_events"
        segments = (["host_to_device"] if self._options.residency == "host_pinned" else []) + \
            ["device_execution", "device_to_host"]
        return {
            "prepare_input_h2d": "included" if self._options.residency == "host_pinned" else "excluded",
            "device_timing": {
                "method": TIMING_METHOD,
                "segments": segments,
                "clock": "device (CUDA event timer, 32 ns steps on this device)",
                "semantics": "duration between two CUDA events on the request stream; includes event overhead; "
                             "never subtracted from host run-clock timestamps",
            } if events else None,
            "instrumentation": {
                "adapter": self.ADAPTER,
                "device_timing": TIMING_METHOD if events else None,
                "in_kernel_timing": "program %globaltimer/clock64 words",
                "backend_snapshots": "nvml at measure_begin and drain_end",
                "cupti": False,
                "context_sched": self._options.context_sched,
                "residency": self._options.residency,
            },
        }

    # ------------------------------------------------------------------ preparation
    def prepare(self, workload: WorkloadSpec, config: DUTConfig) -> PreparedInfo:
        if self._state != "created":
            raise LifecycleError(f"prepare() not allowed in state {self._state!r}")
        if config != self._config:
            raise ValueError("prepare() config differs from the adapter's construction config")
        program = workload.options.get("device_program")
        if program is None:
            raise ValueError("workload.options.device_program is required by the nvidia_cuda adapter")
        self._program = parse_strict(DeviceProgram, program, "workload.options.device_program")
        source_sha = hashlib.sha256(self._program.source.encode("utf-8")).hexdigest()
        if workload.artifact_sha256 != source_sha:
            raise ValueError(f"device program SHA-256 {source_sha} does not match workload.artifact_sha256")
        L = b.libs()
        identity = self.device_identity()
        dev = self._find_cuda_device()
        r = L.cuda.cuDevicePrimaryCtxSetFlags_v2(dev, b.CU_CTX_SCHED_BLOCKING_SYNC)
        if r not in (b.CUDA_SUCCESS, 708):  # 708: primary context already active (flags kept)
            b.cu("cuDevicePrimaryCtxSetFlags_v2", r)
        ctx = c_void_p()
        b.cu("cuDevicePrimaryCtxRetain", L.cuda.cuDevicePrimaryCtxRetain(byref(ctx), dev))
        self._ctx_retained = True
        b.cu("cuCtxSetCurrent", L.cuda.cuCtxSetCurrent(ctx))
        arch = "sm_" + identity["nvidia.compute_capability"].replace(".", "")
        cubin, log, options = b.compile_cubin(self._program.source, "sdv_bmt_device_program.cu", arch)
        self._cubin_sha256 = hashlib.sha256(cubin).hexdigest()
        b.cu("cuModuleLoadData", L.cuda.cuModuleLoadData(byref(self._module), ctypes.c_char_p(cubin)))
        b.cu("cuModuleGetFunction", L.cuda.cuModuleGetFunction(byref(self._function), self._module,
                                                               self._program.entry.encode()))
        b.cu("cuStreamCreate", L.cuda.cuStreamCreate(byref(self._stream), b.CU_STREAM_NON_BLOCKING))
        names = ["start", "h2d", "kernel", "done"] if self._options.instrumentation == "cuda_events" else ["done"]
        flags = b.CU_EVENT_DEFAULT if self._options.instrumentation == "cuda_events" else b.CU_EVENT_DISABLE_TIMING
        for name in names:
            event = c_void_p()
            b.cu("cuEventCreate", L.cuda.cuEventCreate(byref(event), flags))
            self._events[name] = event
        if self._options.residency == "host_pinned" and self._options.buffer_policy == "preallocated_persistent":
            self._d_input = self._device_alloc(self._program.input_bytes)
        self._d_scratch = [self._device_alloc(n) for n in self._program.scratch_bytes]
        self._d_output = self._device_alloc(self._program.output_bytes)
        self._h_output = c_void_p(self._pinned_alloc(self._program.output_bytes))
        self._state = "prepared"
        return PreparedInfo(
            workload_identity_digest=workload_identity_digest(workload),
            notes=(f"device program {self._program.entry} compiled by NVRTC {identity['nvidia.nvrtc_version']} "
                   f"for {arch}; cubin sha256 {self._cubin_sha256}; options {' '.join(options)}",
                   f"nvrtc log: {log.strip() or '(empty)'}"),
        )

    def _device_alloc(self, nbytes: int, persistent: bool = True) -> int:
        ptr = c_uint64()
        b.cu("cuMemAlloc_v2", b.libs().cuda.cuMemAlloc_v2(byref(ptr), nbytes))
        self._device_alloc_count += 1
        if persistent:
            self._device_allocs.append(ptr.value)
        return ptr.value

    def _pinned_alloc(self, nbytes: int) -> int:
        ptr = c_void_p()
        b.cu("cuMemAllocHost_v2", b.libs().cuda.cuMemAllocHost_v2(byref(ptr), nbytes))
        self._pinned_alloc_count += 1
        self._pinned_allocs.append(ptr.value)
        return ptr.value

    def _require(self, method: str) -> DeviceProgram:
        if self._state != "prepared" or self._program is None:
            raise LifecycleError(f"{method}() requires prepare() first (state {self._state!r})")
        return self._program

    def prepare_input(self, task_input: object, item: InputItem) -> NvidiaPreparedInput:
        program = self._require("prepare_input")
        if not isinstance(task_input, (bytes, bytearray)) or len(task_input) != program.input_bytes:
            raise ValueError(f"task input must be {program.input_bytes} bytes for the device program")
        L = b.libs()
        digest = hashlib.sha256(task_input).hexdigest()
        host = self._pinned_alloc(program.input_bytes)
        ctypes.memmove(host, bytes(task_input), program.input_bytes)
        if self._options.residency == "host_pinned":
            return NvidiaPreparedInput(item.item_index, program.input_bytes, digest, host_ptr=host)
        device = self._device_alloc(program.input_bytes)
        b.cu("cuMemcpyHtoD_v2", L.cuda.cuMemcpyHtoD_v2(device, c_void_p(host), program.input_bytes))
        # verify the device-resident representation by read-back (outside MEASURE)
        ctypes.memset(host, 0, program.input_bytes)
        b.cu("cuMemcpyDtoH_v2", L.cuda.cuMemcpyDtoH_v2(c_void_p(host), device, program.input_bytes))
        readback = hashlib.sha256(ctypes.string_at(host, program.input_bytes)).hexdigest()
        if readback != digest:
            raise RuntimeError(f"device-resident input of item {item.item_index} does not read back identically")
        return NvidiaPreparedInput(item.item_index, program.input_bytes, digest, device_ptr=device)

    def input_override(self, contract: WorkloadInputContract, implementation_id: str) -> None:
        return None  # no adapter preprocessing override is implemented

    def prepared_input_digest(self, prepared_input: object) -> str:
        if not isinstance(prepared_input, NvidiaPreparedInput):
            raise TypeError("not an NvidiaCudaDUT prepared input")
        return prepared_input.sha256

    # ------------------------------------------------------------------ requests
    def submit(self, request: DUTRequest) -> _Handle:
        program = self._require("submit")
        if self._pending is not None:
            raise LifecycleError("submit() while a request is outstanding (max_inflight=1)")
        prepared = request.prepared_inputs[0]
        if not isinstance(prepared, NvidiaPreparedInput) or prepared.item_index != request.item_indices[0]:
            raise ValueError("prepared input does not correspond to the request item")
        L = b.libs()
        cuda, s, ev = L.cuda, self._stream, self._events
        timing = "start" in ev
        transient = 0
        if prepared.host_ptr is not None and self._options.buffer_policy == "per_request_alloc":
            transient = self._transient = self._device_alloc(prepared.nbytes, persistent=False)
        try:
            h2d, input_ptr = 0, prepared.device_ptr
            if timing:
                b.cu("cuEventRecord", cuda.cuEventRecord(ev["start"], s))
            if prepared.host_ptr is not None:
                input_ptr = transient or self._d_input
                b.cu("cuMemcpyHtoDAsync_v2", cuda.cuMemcpyHtoDAsync_v2(input_ptr, c_void_p(prepared.host_ptr),
                                                                       prepared.nbytes, s))
                h2d = prepared.nbytes
                if timing:
                    b.cu("cuEventRecord", cuda.cuEventRecord(ev["h2d"], s))
            args = b.kernel_args(c_uint64(input_ptr), *[c_uint64(p) for p in self._d_scratch],
                                 c_uint64(self._d_output))
            b.cu("cuLaunchKernel", cuda.cuLaunchKernel(self._function, 1, 1, 1, program.block_threads, 1, 1, 0, s,
                                                       args, None))
            if timing:
                b.cu("cuEventRecord", cuda.cuEventRecord(ev["kernel"], s))
            b.cu("cuMemcpyDtoHAsync_v2", cuda.cuMemcpyDtoHAsync_v2(self._h_output, self._d_output,
                                                                   program.output_bytes, s))
            b.cu("cuEventRecord", cuda.cuEventRecord(ev["done"], s))
        except Exception:
            if transient:  # work may already be enqueued on it: free only at cleanup, after the stream drained
                self._device_allocs.append(transient)
                self._transient = 0
            raise
        self._pending = _Handle(request.request_id, prepared.item_index, h2d, transient)
        return self._pending

    def _elapsed_ns(self, a: str, z: str) -> int:
        ms = c_float()
        b.cu("cuEventElapsedTime_v2", b.libs().cuda.cuEventElapsedTime_v2(byref(ms), self._events[a], self._events[z]))
        return int(round(ms.value * 1e6))

    def poll(self, handle: object) -> Optional[DUTCompletion]:
        program = self._require("poll")
        if not isinstance(handle, _Handle) or handle is not self._pending:
            raise ValueError("unknown or stale request handle")
        r = b.libs().cuda.cuEventQuery(self._events["done"])
        if r == b.CUDA_ERROR_NOT_READY:
            return None
        self._pending = None
        if handle.transient_input:  # per_request_alloc: release inside the measured path, after completion
            b.libs().cuda.cuMemFree_v2(handle.transient_input)
            self._transient = 0
        self._acc.requests += handle.request_id >= 0
        if r != b.CUDA_SUCCESS:
            err = b.cuda_error("cuEventQuery", r)
            self._acc.errors += handle.request_id >= 0
            return DUTCompletion(handle.request_id, False, None, ErrorInfo(err.name, str(err)))
        raw = ctypes.string_at(self._h_output, program.output_bytes)
        timing = None
        if "start" in self._events:
            after_input = "h2d" if handle.h2d_bytes else "start"
            timing = {"device_execution": self._elapsed_ns(after_input, "kernel"),
                      "device_to_host": self._elapsed_ns("kernel", "done")}
            if handle.h2d_bytes:
                timing["host_to_device"] = self._elapsed_ns("start", "h2d")
        observations: dict[str, Any] = {}
        if program.timing_offset is not None:
            t0, t1, c0, c1 = (int.from_bytes(raw[program.timing_offset + 8 * k: program.timing_offset + 8 * k + 8],
                                             "little") for k in range(4))
            if t1 > t0 and c1 >= c0:
                mhz = round((c1 - c0) / (t1 - t0) * 1000.0, 3)
                observations = {"nvidia.in_kernel_globaltimer_ns": t1 - t0, "nvidia.in_kernel_sm_cycles": c1 - c0,
                                "nvidia.in_kernel_sm_mhz": mhz}
                if handle.request_id >= 0:
                    self._acc.in_kernel_mhz.append(mhz)
        if handle.request_id >= 0:
            self._acc.h2d_bytes.add(handle.h2d_bytes)
        return DUTCompletion(handle.request_id, True, raw[: program.task_output_bytes],
                             device_timing=timing, device_timing_method=TIMING_METHOD if timing else None,
                             observations=observations)

    # ------------------------------------------------------------------ observation
    def observe(self, point: str) -> dict[str, Any]:
        L = b.libs()
        nvml, h = L.nvml, self._init_nvml()
        snap: dict[str, Any] = {"point": point, "t_monotonic_ns": time.monotonic_ns(),
                                "t_realtime_us": time.time_ns() // 1000}
        u, i, ull = c_uint(), c_int(), c_ulonglong()
        if nvml.nvmlDeviceGetPerformanceState(h, byref(i)) == b.NVML_SUCCESS:
            snap["pstate"] = i.value
        for key, clock in (("sm_clock_mhz", b.NVML_CLOCK_SM), ("mem_clock_mhz", b.NVML_CLOCK_MEM)):
            if nvml.nvmlDeviceGetClockInfo(h, clock, byref(u)) == b.NVML_SUCCESS:
                snap[key] = u.value
        t = b.NvmlTemperatureV1(version=b.nvml_struct_version(b.NvmlTemperatureV1, 1), sensorType=0)
        if nvml.nvmlDeviceGetTemperatureV(h, byref(t)) == b.NVML_SUCCESS:
            snap["temperature_c"] = t.temperature
        if nvml.nvmlDeviceGetPowerUsage(h, byref(u)) == b.NVML_SUCCESS:
            snap["power_mw"] = u.value
        if nvml.nvmlDeviceGetCurrentClocksEventReasons(h, byref(ull)) == b.NVML_SUCCESS:
            snap["clocks_event_reasons"] = [n for bit, n in b.CLOCK_EVENT_REASONS.items() if ull.value & bit]
        gen, width = c_uint(), c_uint()
        if (nvml.nvmlDeviceGetCurrPcieLinkGeneration(h, byref(gen)) == b.NVML_SUCCESS
                and nvml.nvmlDeviceGetCurrPcieLinkWidth(h, byref(width)) == b.NVML_SUCCESS):
            snap["pcie_link"] = [gen.value, width.value]
        mem = b.NvmlMemoryV2(version=b.nvml_struct_version(b.NvmlMemoryV2, 2))
        if nvml.nvmlDeviceGetMemoryInfo_v2(h, byref(mem)) == b.NVML_SUCCESS:
            snap["memory_used_bytes"] = mem.used
        snap["foreign_compute_processes"] = sorted(p["pid"] for p in self._compute_processes(h)
                                                   if p["pid"] != os.getpid())
        snap["fields"] = self._fields(h)
        if self._state == "prepared":
            snap["stream_idle"] = L.cuda.cuStreamQuery(self._stream) == b.CUDA_SUCCESS
        snap["device_allocations_total"] = self._device_alloc_count
        snap["pinned_allocations_total"] = self._pinned_alloc_count
        snap["cubin_sha256"] = self._cubin_sha256
        if point == "drain_end" and "measure_begin" in self._snapshots:
            snap["process_utilization"] = self._foreign_activity(h, self._snapshots["measure_begin"]["t_realtime_us"])
        if point == "measure_begin":
            self._acc = _Accumulator()  # requests after this point belong to the measure window
        self._snapshots[point] = snap
        return snap

    def _compute_processes(self, h: c_void_p) -> list[dict[str, int]]:
        nvml = b.libs().nvml
        count = c_uint(0)
        r = nvml.nvmlDeviceGetComputeRunningProcesses_v3(h, byref(count), None)
        if r == b.NVML_SUCCESS and count.value == 0:
            return []
        if r not in (b.NVML_SUCCESS, b.NVML_ERROR_INSUFFICIENT_SIZE):
            b.nv("nvmlDeviceGetComputeRunningProcesses_v3", r)
        infos = (b.NvmlProcessInfo * (count.value + 8))()
        count = c_uint(len(infos))
        b.nv("nvmlDeviceGetComputeRunningProcesses_v3", nvml.nvmlDeviceGetComputeRunningProcesses_v3(h, byref(count),
                                                                                                     infos))
        return [{"pid": infos[k].pid, "used_gpu_memory_bytes": infos[k].usedGpuMemory} for k in range(count.value)]

    def _fields(self, h: c_void_p) -> dict[str, Optional[int]]:
        values = (b.NvmlFieldValue * len(b.NVML_FIELDS))()
        for k, fid in enumerate(b.NVML_FIELDS):
            values[k].fieldId = fid
        if b.libs().nvml.nvmlDeviceGetFieldValues(h, len(b.NVML_FIELDS), values) != b.NVML_SUCCESS:
            return {name: None for name in b.NVML_FIELDS.values()}
        return {b.NVML_FIELDS[values[k].fieldId]: (values[k].value if values[k].nvmlReturn == b.NVML_SUCCESS else None)
                for k in range(len(b.NVML_FIELDS))}

    def _foreign_activity(self, h: c_void_p, since_us: int) -> dict[str, Any]:
        nvml = b.libs().nvml
        count = c_uint(0)
        r = nvml.nvmlDeviceGetProcessUtilization(h, None, byref(count), c_ulonglong(since_us))
        if r == b.NVML_ERROR_NOT_FOUND:
            return {"status": "no_samples", "foreign_nonzero_samples": 0, "own_samples": 0}
        if r not in (b.NVML_SUCCESS, b.NVML_ERROR_INSUFFICIENT_SIZE):
            return {"status": f"error:{r}"}
        samples = (b.NvmlProcessUtilizationSample * max(count.value, 1))()
        r = nvml.nvmlDeviceGetProcessUtilization(h, samples, byref(count), c_ulonglong(since_us))
        if r == b.NVML_ERROR_NOT_FOUND:
            return {"status": "no_samples", "foreign_nonzero_samples": 0, "own_samples": 0}
        if r != b.NVML_SUCCESS:
            return {"status": f"error:{r}"}
        rows = samples[: count.value]
        own = os.getpid()
        return {
            "status": "ok",
            "foreign_nonzero_samples": sum(1 for s in rows if s.pid != own and (s.smUtil or s.memUtil)),
            "foreign_pids": sorted({s.pid for s in rows if s.pid != own}),
            "foreign_active_pids": sorted({s.pid for s in rows if s.pid != own and (s.smUtil or s.memUtil)}),
            "own_samples": sum(1 for s in rows if s.pid == own),
        }

    def realization_evidence(self) -> tuple[RealizationEvidence, ...]:
        begin, end = self._snapshots.get("measure_begin"), self._snapshots.get("drain_end")
        src = f"dut:{self.ADAPTER}"
        out: list[RealizationEvidence] = []

        def rec(condition: RealizedCondition, validity: EvidenceValidity, source: str, semantic: str,
                value: Any = None, equivalence: str = "exact", unit: Optional[str] = None,
                clock: Optional[str] = None) -> None:
            out.append(RealizationEvidence(condition, validity, source, semantic, unit, "device", clock, value,
                                           equivalence))

        location = "host_pinned" if self._options.residency == "host_pinned" else "device"
        rec(RealizedCondition.INPUT_RESIDENCY, EvidenceValidity.SELF_REPORTED, f"{src}:buffer_management",
            "where the prepared input of each request lives and the H2D bytes per request, as executed by the "
            "adapter's own copy calls (not independently traced)",
            {"prepared_input_location": location, "h2d_bytes_per_request": sorted(self._acc.h2d_bytes)})
        if begin is None or end is None:
            for condition in (RealizedCondition.BUFFER_STATE, RealizedCondition.DATA_MOVEMENT,
                              RealizedCondition.RUNTIME_QUEUE_STATE, RealizedCondition.DEVICE_FREQUENCY,
                              RealizedCondition.THERMAL_STATE, RealizedCondition.COMPETING_DEVICE_USERS,
                              RealizedCondition.MEMORY_PRESSURE_BANDWIDTH):
                rec(condition, EvidenceValidity.NOT_OBSERVABLE, src, "backend snapshot at measure_begin/drain_end "
                    "unavailable")
            return tuple(out)
        rec(RealizedCondition.BUFFER_STATE, EvidenceValidity.SELF_REPORTED, f"{src}:buffer_management",
            "buffer policy and allocations made during the measure window, counted by the adapter",
            {"policy": self._options.buffer_policy,
             "device_allocations_during_measure": end["device_allocations_total"] - begin["device_allocations_total"],
             "pinned_allocations_during_measure": end["pinned_allocations_total"] - begin["pinned_allocations_total"]})
        fields_b, fields_e = begin.get("fields", {}), end.get("fields", {})

        def counter_change(name: str) -> Optional[int]:
            a, z = fields_b.get(name), fields_e.get(name)
            return None if a is None or z is None else z - a

        def observed(condition: RealizedCondition, source: str, semantic: str, value: dict[str, Any],
                     equivalence: str = "exact", unit: Optional[str] = None) -> None:
            # an observation that did not return a value is not evidence: never compare a missing value
            missing = sorted(k for k, v in value.items() if v is None)
            if missing:
                rec(condition, EvidenceValidity.NOT_OBSERVABLE, source,
                    f"{semantic} -- not observed in this run: {', '.join(missing)}")
            else:
                rec(condition, EvidenceValidity.OBSERVED, source, semantic, value, equivalence, unit)

        observed(RealizedCondition.DATA_MOVEMENT, "nvidia:nvml",
                 "PCIe link generation/width at measure_begin and drain_end and PCIe replay counter delta",
                 {"pcie_link_measure_begin": begin.get("pcie_link"), "pcie_link_drain_end": end.get("pcie_link"),
                  "pcie_replay_delta": counter_change("pcie_replay_counter")})
        observed(RealizedCondition.RUNTIME_QUEUE_STATE, "nvidia:cuda_driver",
                 "whether the adapter's request stream had pending work at measure_begin (cuStreamQuery); by "
                 "design the adapter uses one non-blocking stream and the runner keeps at most one request in "
                 "flight (design facts, not observations)",
                 {"request_stream_idle_at_measure_begin": begin.get("stream_idle")})
        observed(RealizedCondition.DEVICE_FREQUENCY, "nvidia:program_clock64_globaltimer+nvml",
                 "effective SM clock during each measured kernel (clock64 cycles / %globaltimer ns), P-state at the "
                 "window edges, power-cap and sync-boost throttle time during the window",
                 {"in_kernel_sm_mhz": _quantiles(self._acc.in_kernel_mhz), "pstate_measure_begin": begin.get("pstate"),
                  "pstate_drain_end": end.get("pstate"),
                  "throttle_ns_sw_power_cap": counter_change("throttle_ns_sw_power_cap"),
                  "throttle_ns_sync_boost": counter_change("throttle_ns_sync_boost")},
                 equivalence=STEADY_STATE_RULE, unit="MHz")
        observed(RealizedCondition.THERMAL_STATE, "nvidia:nvml",
                 "GPU temperature and clocks event reasons at the window edges (thermal throttle counters are not "
                 "supported by this device)",
                 {"temperature_c_measure_begin": begin.get("temperature_c"),
                  "temperature_c_drain_end": end.get("temperature_c"),
                  "clocks_event_reasons_measure_begin": begin.get("clocks_event_reasons"),
                  "clocks_event_reasons_drain_end": end.get("clocks_event_reasons")},
                 equivalence=STEADY_STATE_RULE, unit="degC")
        activity = end.get("process_utilization") or {}
        users = {"foreign_compute_processes_measure_begin": begin.get("foreign_compute_processes"),
                 "foreign_compute_processes_drain_end": end.get("foreign_compute_processes")}
        if activity.get("status") == "ok":
            # sampling coverage is part of the semantic, so a sampled and an unsampled window are never compared
            observed(RealizedCondition.COMPETING_DEVICE_USERS, "nvidia:nvml",
                     "other processes with a compute context on the device at the window edges, and the number of "
                     "non-zero NVML utilization samples of other processes during the window (NVML sampling "
                     "period; activity between samples can be missed; equality does not characterize the "
                     "intensity of a foreign load)",
                     {**users, "foreign_nonzero_utilization_samples": activity.get("foreign_nonzero_samples")})
        else:
            observed(RealizedCondition.COMPETING_DEVICE_USERS, "nvidia:nvml",
                     "other processes with a compute context on the device at the window edges only; NVML "
                     f"utilization sampling of the window: {activity.get('status', 'not taken')}", users)
        rec(RealizedCondition.MEMORY_PRESSURE_BANDWIDTH, EvidenceValidity.NOT_OBSERVABLE, "nvidia:hardware_counters",
            "device DRAM bandwidth and traffic need GPU performance counters (RmProfilingAdminOnly=1 on this host); "
            "device memory use is kept in the backend snapshots")
        return tuple(out)

    # ------------------------------------------------------------------ cleanup
    def cleanup(self) -> None:
        if self._state == "cleaned":
            return
        L = b.libs()
        busy = False
        if self._stream:
            deadline = time.monotonic_ns() + 5_000_000_000
            while L.cuda.cuStreamQuery(self._stream) == b.CUDA_ERROR_NOT_READY:
                if time.monotonic_ns() > deadline:
                    busy = True  # device work still running: never free memory it may use
                    break
        if not busy:
            for ptr in self._device_allocs + ([self._transient] if self._transient else []):
                L.cuda.cuMemFree_v2(ptr)
            for ptr in self._pinned_allocs:
                L.cuda.cuMemFreeHost(c_void_p(ptr))
            for event in self._events.values():
                L.cuda.cuEventDestroy_v2(event)
            if self._stream:
                L.cuda.cuStreamDestroy_v2(self._stream)
            if self._module:
                L.cuda.cuModuleUnload(self._module)
            if self._ctx_retained and self._cu_dev is not None:
                L.cuda.cuDevicePrimaryCtxRelease_v2(self._cu_dev)
        self._device_allocs, self._pinned_allocs, self._events = [], [], {}
        self._transient = 0
        self._stream = c_void_p()
        self._module = c_void_p()
        self._ctx_retained = False
        self._pending = None
        if self._nvml_handle is not None:
            L.nvml.nvmlShutdown()
            self._nvml_handle = None
        self._state = "cleaned"
        if busy:
            raise RuntimeError("device work still running at cleanup; buffers left allocated until process exit")
