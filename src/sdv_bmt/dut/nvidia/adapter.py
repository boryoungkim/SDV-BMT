"""NVIDIA reference-accelerator DUT adapter (``adapter: nvidia_cuda``), Stage 1R-G, native measurement plane.

The GPU is the first *reference* accelerator: it validates the realtime harness, measurement boundaries
and realized-condition evidence on real hardware.  It is not the NPU target and its results do not
predict NPU behaviour.

Execution happens in the native runner ``sdv-bmt-native-nvidia`` (``native/src/nvidia/backend.hpp``):
device selection by UUID, primary context with blocking-sync scheduling, NVRTC compilation of the device
program carried by the workload spec (SHA-256 pinned by ``workload.artifact_sha256``), persistent
buffers, per-item staging of the prepared inputs (``residency: host_pinned`` keeps them in pinned host
memory and copies them per request inside DUT-service; ``device_resident`` copies them once and verifies
the device copy by read-back), submission of [H2D] -> kernel -> D2H with CUDA events on one stream,
completion by polling, NVML snapshots before MEASURE and after DRAIN.  This Python class only declares
the measurement, builds the backend part of the run specification and interprets the recorded facts
into realization evidence.  NVIDIA concepts stay in this package and in the native NVIDIA backend.

Stage 2b-G real-model baseline: a workload may instead carry a ``cudnn_plan`` device program, a model
converted offline from its ONNX artifact into an integer-only execution plan and a float32 weight file
(both SHA-256 pinned, paths relative to the repository root).  Per request the native backend then runs
[H2D] -> frequency probe -> the plan's cuDNN convolutions/poolings and NVRTC glue kernels -> D2H on the
same stream.  The cuDNN options (math type, algorithm policy, workspace limit and, for policy ``pinned``,
the per-convolution (algorithm, math type) table) are DUT runtime options, hence part of the DUT identity;
the cuDNN library version and file digests are part of the device identity; the probe is instrumentation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Optional

from ...contracts.records import DUTCapabilities, DUTDescriptor, RealizationEvidence
from ...contracts.specs import DUTConfig, WorkloadSpec, parse_strict
from ...contracts.status import CompletionModel, EvidenceValidity, ExecutionKind, RealizedCondition, RunClockDomain
from ..native import REPO_ROOT, NativeRealtimeAdapter, describe_native, native_binary, native_identity

RESIDENCIES = ("host_pinned", "device_resident")
INSTRUMENTATIONS = ("cuda_events", "none")
BUFFER_POLICIES = ("preallocated_persistent", "per_request_alloc")
PROFILERS = ("none", "nsys")
CUDNN_MATH_TYPES = ("default", "fma", "tensor_op")
CUDNN_ALGO_POLICIES = ("pinned", "heuristic")
TIMING_METHOD = "nvidia:cuda_event_pair:bracket"
STEADY_STATE_RULE = "unapproved:steady_state_band"  # candidate rule, not approved (Stage 2a-G section 12)


@dataclass(frozen=True)
class CudnnOptions:
    """cuDNN execution of a ``cudnn_plan`` program.  ``algorithms`` lists one (algorithm, math type) pair
    per convolution in plan order (cuDNN enum values) and is required by policy ``pinned``."""

    math_type: str = "default"
    algo_policy: str = "pinned"
    workspace_limit_bytes: int = 268435456
    algorithms: Optional[tuple[tuple[int, ...], ...]] = None

    def __post_init__(self) -> None:
        if self.math_type not in CUDNN_MATH_TYPES:
            raise ValueError(f"cudnn.math_type must be one of {CUDNN_MATH_TYPES}")
        if self.algo_policy not in CUDNN_ALGO_POLICIES:
            raise ValueError(f"cudnn.algo_policy must be one of {CUDNN_ALGO_POLICIES}")
        if self.workspace_limit_bytes < 0:
            raise ValueError("cudnn.workspace_limit_bytes must be >= 0")
        if self.algo_policy == "pinned":
            if not self.algorithms or any(len(a) != 2 or min(a) < 0 for a in self.algorithms):
                raise ValueError("cudnn.algorithms must list [algorithm, math_type] pairs for policy pinned")
        elif self.algorithms is not None:
            raise ValueError("cudnn.algorithms is only allowed with policy pinned")


def cudnn_table_sha256(algorithms: tuple[tuple[int, ...], ...]) -> str:
    """SHA-256 of the compact JSON of the algorithm table, as the native runner computes it."""
    text = json.dumps([list(a) for a in algorithms], separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NvidiaOptions:
    residency: str = "host_pinned"
    instrumentation: str = "cuda_events"
    context_sched: str = "blocking_sync"
    buffer_policy: str = "preallocated_persistent"
    nvtx: bool = False
    profiler: str = "none"
    cudnn: Optional[CudnnOptions] = None
    frequency_probe_cycles: Optional[int] = None

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
        if self.profiler not in PROFILERS:
            raise ValueError(f"profiler must be one of {PROFILERS}")
        if self.frequency_probe_cycles is not None:
            if self.cudnn is None:
                raise ValueError("frequency_probe_cycles applies to cudnn_plan execution (runtime_options.cudnn)")
            if self.frequency_probe_cycles < 0:
                raise ValueError("frequency_probe_cycles must be >= 0")


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


@dataclass(frozen=True)
class CudnnPlanProgram:
    """A model converted offline into a cuDNN execution plan; files are relative to the repository root."""

    language: str
    plan: str
    plan_sha256: str
    weights: str
    weights_sha256: str
    input_bytes: int
    output_bytes: int
    task_output_bytes: int
    timing_offset: Optional[int]

    def __post_init__(self) -> None:
        if self.language != "cudnn_plan":
            raise ValueError("device_program.language must be cudnn_plan")
        for path in (self.plan, self.weights):
            if not path or path.startswith("/") or ".." in path.split("/"):
                raise ValueError("device_program files must be paths inside the repository")
        if self.input_bytes <= 0 or not 0 < self.task_output_bytes <= self.output_bytes:
            raise ValueError("device_program buffer sizes are inconsistent")
        if self.timing_offset is not None and (self.timing_offset < self.task_output_bytes
                                               or self.timing_offset + 32 > self.output_bytes):
            raise ValueError("device_program.timing_offset must follow the task output and fit 4 u64 words")


def _quantiles(values: list[float]) -> Optional[dict[str, Any]]:
    if not values:
        return None
    v = sorted(values)
    n = len(v)
    rank = lambda p: v[min(n - 1, max(0, math.ceil(p * n) - 1))]  # noqa: E731
    return {"n": n, "min": v[0], "p50": rank(0.5), "max": v[-1]}


def _file_sha256(path: Any) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class NvidiaCudaDUT(NativeRealtimeAdapter):
    ADAPTER = "nvidia_cuda"
    VERSION = "3-native-cudnn-plan"
    BINARY = "sdv-bmt-native-nvidia"

    def __init__(self, config: DUTConfig) -> None:
        super().__init__(config)
        self._options = parse_strict(NvidiaOptions, config.runtime_options, "dut.runtime_options")
        self._uuid = config.device
        self._identity: Optional[dict[str, Any]] = None

    @property
    def _probe_cycles(self) -> int:
        return self._options.frequency_probe_cycles or 0

    # ------------------------------------------------------------------ identity
    def device_identity(self) -> dict[str, Any]:
        """Exact identity detected at runtime (never assumed from the product name), plus the identity
        of the native measurement plane that executes the run."""
        if self._identity is None:
            binary = native_binary(self.BINARY)
            identity = describe_native(binary, {"uuid": self._uuid})
            identity.pop("reply", None)
            identity.pop("ok", None)
            self._identity = {**identity, **native_identity(binary)}
        return self._identity

    def describe(self) -> DUTDescriptor:
        identity = self.device_identity()
        o = self._options
        h2d = "H2D from the item's pinned host buffer, " if o.residency == "host_pinned" else ""
        per_request = o.buffer_policy == "per_request_alloc"
        alloc = "allocates the device input buffer (cuMemAlloc), " if per_request else ""
        free = "; the completing poll then frees the device input buffer (cuMemFree)" if per_request else ""
        if o.cudnn is None:
            work = "the kernel"
        else:
            probe = "a frequency probe kernel, " if self._probe_cycles else ""
            work = f"{probe}the cuDNN plan (cuDNN convolutions and poolings, NVRTC glue kernels)"
        return DUTDescriptor(
            adapter=self.ADAPTER,
            adapter_version=self.VERSION,
            implementation_kind=ExecutionKind.ACTUAL,
            device=f"{identity['nvidia.product_name']} {self._uuid}",
            completion_model=CompletionModel.POLLING,
            clock_domain=RunClockDomain.MONOTONIC,
            submit_semantics=f"native runner submit: {alloc}enqueues {h2d}{work} and the D2H copy of the output "
                             "on one non-blocking CUDA stream and returns",
            complete_semantics=f"cuEventQuery on the event recorded after the D2H copy returns CUDA_SUCCESS{free}",
            device_identity=identity,
        )

    def capabilities(self) -> DUTCapabilities:
        return DUTCapabilities(completion_model=CompletionModel.POLLING, max_inflight=1, virtual_timing=False,
                               device_timing=self._options.instrumentation == "cuda_events")

    def measurement_declaration(self) -> dict[str, Any]:
        o = self._options
        events = o.instrumentation == "cuda_events"
        segments = (["host_to_device"] if o.residency == "host_pinned" else []) + ["device_execution", "device_to_host"]
        semantics = ("duration between two CUDA events on the request stream; includes event overhead; "
                     "never subtracted from host run-clock timestamps")
        if o.cudnn is not None:
            semantics += ("; with a cuDNN plan the device_execution segment spans the frequency probe and every "
                          "cuDNN and glue kernel of the request, including any gap while the host is still "
                          "enqueuing them")
        if o.cudnn is None:
            in_kernel = "program %globaltimer/clock64 words"
        elif self._probe_cycles:
            in_kernel = (f"probe kernel: one thread, {self._probe_cycles} SM cycles of clock64 spin, first in each "
                         "request's device work; %globaltimer/clock64 words")
        else:
            in_kernel = None
        instrumentation: dict[str, Any] = {
            "adapter": self.ADAPTER,
            "measurement_plane": f"native:{self.BINARY}",
            "device_timing": TIMING_METHOD if events else None,
            "in_kernel_timing": in_kernel,
            "backend_snapshots": "nvml at measure_begin and drain_end",
            "nvtx": o.nvtx,
            "profiler": o.profiler,
            "cupti_in_process": False,
            "context_sched": o.context_sched,
            "residency": o.residency,
            "buffer_policy": o.buffer_policy,
        }
        return {
            "prepare_input_h2d": "included" if o.residency == "host_pinned" else "excluded",
            "device_timing": {
                "method": TIMING_METHOD,
                "segments": segments,
                "clock": "device (CUDA event timer, 32 ns steps on this device)",
                "semantics": semantics,
            } if events else None,
            "instrumentation": instrumentation,
        }

    # ------------------------------------------------------------------ run specification (backend part)
    def backend_spec(self, workload: WorkloadSpec) -> dict[str, Any]:
        program = workload.options.get("device_program")
        if program is None:
            raise ValueError("workload.options.device_program is required by the nvidia_cuda adapter")
        if isinstance(program, dict) and program.get("language") == "cudnn_plan":
            return self._cudnn_plan_spec(workload, parse_strict(CudnnPlanProgram, program,
                                                                "workload.options.device_program"))
        if self._options.cudnn is not None:
            raise ValueError("runtime_options.cudnn applies to cudnn_plan device programs only")
        p = parse_strict(DeviceProgram, program, "workload.options.device_program")
        if workload.artifact_sha256 != hashlib.sha256(p.source.encode("utf-8")).hexdigest():
            raise ValueError("device program SHA-256 does not match workload.artifact_sha256")
        return {
            "device_uuid": self._uuid,
            "program": {"source": p.source, "entry": p.entry, "block_threads": p.block_threads,
                        "input_bytes": p.input_bytes, "scratch_bytes": list(p.scratch_bytes),
                        "output_bytes": p.output_bytes, "task_output_bytes": p.task_output_bytes,
                        "timing_offset": p.timing_offset, "artifact_sha256": workload.artifact_sha256},
        }

    def _cudnn_plan_spec(self, workload: WorkloadSpec, p: CudnnPlanProgram) -> dict[str, Any]:
        if self._options.cudnn is None:
            raise ValueError("a cudnn_plan device program requires runtime_options.cudnn")
        if self._probe_cycles and p.timing_offset is None:
            raise ValueError("frequency_probe_cycles > 0 requires device_program.timing_offset")
        plan_path, weights_path = REPO_ROOT / p.plan, REPO_ROOT / p.weights
        if _file_sha256(plan_path) != p.plan_sha256:
            raise ValueError(f"{p.plan}: SHA-256 differs from device_program.plan_sha256")
        if _file_sha256(weights_path) != p.weights_sha256:
            raise ValueError(f"{p.weights}: SHA-256 differs from device_program.weights_sha256")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("model", {}).get("onnx_sha256") != workload.artifact_sha256:
            raise ValueError("the cuDNN plan was not converted from the workload's model artifact")
        if plan["input"]["bytes"] != p.input_bytes or plan["output"]["bytes"] != p.task_output_bytes:
            raise ValueError("device_program sizes differ from the cuDNN plan")
        return {
            "device_uuid": self._uuid,
            "program": {"kind": "cudnn_plan", "plan_path": str(plan_path), "plan_sha256": p.plan_sha256,
                        "weights_path": str(weights_path), "weights_sha256": p.weights_sha256,
                        "input_bytes": p.input_bytes, "output_bytes": p.output_bytes,
                        "task_output_bytes": p.task_output_bytes, "timing_offset": p.timing_offset,
                        "artifact_sha256": workload.artifact_sha256},
        }

    # ------------------------------------------------------------------ interpretation of native facts
    def realization_evidence(self, window: Any) -> tuple[RealizationEvidence, ...]:
        """Realized device-side conditions from the facts the native runner recorded (bundle events).

        ``window.snapshots`` maps ``measure_begin``/``drain_end`` to the NVML snapshot (or None when the
        snapshot failed); ``window.device_observations`` are the measured requests' device facts.
        """
        snaps = getattr(window, "snapshots", {}) or {}
        begin, end = snaps.get("measure_begin"), snaps.get("drain_end")
        observations = [e.get("observations") or {} for e in getattr(window, "device_observations", ())]
        src = f"dut:{self.ADAPTER}"
        out: list[RealizationEvidence] = []

        def rec(condition: RealizedCondition, validity: EvidenceValidity, source: str, semantic: str,
                value: Any = None, equivalence: str = "exact", unit: Optional[str] = None) -> None:
            out.append(RealizationEvidence(condition, validity, source, semantic, unit, "device", None, value,
                                           equivalence))

        location = "host_pinned" if self._options.residency == "host_pinned" else "device"
        rec(RealizedCondition.INPUT_RESIDENCY, EvidenceValidity.SELF_REPORTED, f"{src}:buffer_management",
            "where the prepared input of each request lives and the H2D bytes per request, as executed by the "
            "native runner's own copy calls (not independently traced)",
            {"prepared_input_location": location,
             "h2d_bytes_per_request": sorted({o.get("nvidia.h2d_bytes") for o in observations
                                              if o.get("nvidia.h2d_bytes") is not None})})
        if begin is None or end is None:
            for condition in (RealizedCondition.BUFFER_STATE, RealizedCondition.DATA_MOVEMENT,
                              RealizedCondition.RUNTIME_QUEUE_STATE, RealizedCondition.DEVICE_FREQUENCY,
                              RealizedCondition.THERMAL_STATE, RealizedCondition.COMPETING_DEVICE_USERS,
                              RealizedCondition.MEMORY_PRESSURE_BANDWIDTH):
                rec(condition, EvidenceValidity.NOT_OBSERVABLE, src,
                    "backend snapshot at measure_begin/drain_end unavailable")
            return tuple(out)
        rec(RealizedCondition.BUFFER_STATE, EvidenceValidity.SELF_REPORTED, f"{src}:buffer_management",
            "buffer policy and allocations made during the measure window, counted by the native runner",
            {"policy": self._options.buffer_policy,
             "device_allocations_during_measure": end["device_allocations_total"] - begin["device_allocations_total"],
             "pinned_allocations_during_measure": end["pinned_allocations_total"] - begin["pinned_allocations_total"]})
        fields_b, fields_e = begin.get("fields", {}), end.get("fields", {})

        def counter_change(name: str) -> Optional[int]:
            a, z = fields_b.get(name), fields_e.get(name)
            return None if a is None or z is None else z - a

        def observed(condition: RealizedCondition, source: str, semantic: str, value: dict[str, Any],
                     equivalence: str = "exact", unit: Optional[str] = None) -> None:
            # an observation that returned no value is not evidence: never compare a missing value
            missing = sorted(k for k, v in value.items() if v is None)
            if missing:
                rec(condition, EvidenceValidity.NOT_OBSERVABLE, source,
                    f"{semantic} -- not observed in this run: {', '.join(missing)}")
            else:
                rec(condition, EvidenceValidity.OBSERVED, source, semantic, value, equivalence, unit)

        mhz = [round(o["nvidia.in_kernel_sm_cycles"] / o["nvidia.in_kernel_globaltimer_ns"] * 1000.0, 3)
               for o in observations
               if o.get("nvidia.in_kernel_globaltimer_ns") and o.get("nvidia.in_kernel_sm_cycles") is not None]
        observed(RealizedCondition.DATA_MOVEMENT, "nvidia:nvml",
                 "PCIe link generation/width at measure_begin and drain_end and PCIe replay counter delta",
                 {"pcie_link_measure_begin": begin.get("pcie_link"), "pcie_link_drain_end": end.get("pcie_link"),
                  "pcie_replay_delta": counter_change("pcie_replay_counter")})
        observed(RealizedCondition.RUNTIME_QUEUE_STATE, "nvidia:cuda_driver",
                 "whether the native runner's request stream had pending work at measure_begin (cuStreamQuery); "
                 "by design one non-blocking stream is used and at most one request is in flight (design facts, "
                 "not observations)",
                 {"request_stream_idle_at_measure_begin": begin.get("stream_idle")})
        if self._options.cudnn is None:
            freq_source = "nvidia:program_clock64_globaltimer+nvml"
            freq_semantic = ("effective SM clock during each measured kernel (clock64 cycles / %globaltimer ns, "
                             "computed from the recorded words), P-state at the window edges, power-cap and "
                             "sync-boost throttle time during the window")
        else:
            freq_source = "nvidia:probe_kernel_clock64_globaltimer+nvml"
            freq_semantic = ("effective SM clock of a one-thread probe kernel run first in each measured request's "
                             f"device work ({self._probe_cycles}-cycle clock64 spin / %globaltimer ns, computed from "
                             "the recorded words), P-state at the window edges, power-cap and sync-boost throttle "
                             "time during the window")
        observed(RealizedCondition.DEVICE_FREQUENCY, freq_source, freq_semantic,
                 {"in_kernel_sm_mhz": _quantiles(mhz), "pstate_measure_begin": begin.get("pstate"),
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
