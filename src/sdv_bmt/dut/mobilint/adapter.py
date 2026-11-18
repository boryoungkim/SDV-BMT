"""Mobilint ARIES NPU DUT adapter (``adapter: mobilint_aries``), native measurement plane.

The DUT is one ARIES chip of a Mobilint MLA400 card (device node ``/dev/ariesN``), running a YOLOX model
compiled offline by qbcompiler into an ``.mxq`` file (SHA-256 pinned in the workload's device program).
Execution happens in the native runner ``sdv-bmt-native-mobilint`` (``native/src/mobilint/backend.hpp``):
qbruntime Accelerator and Model with one pinned (cluster, core) in single-core mode and the asynchronous
pipeline, per-item staging of the prepared input converted once to HWC float32 host memory, submission by
``Model::inferAsync``, completion by polling ``Future::waitFor(0)``, and in collect() ``Future::get`` plus the
YOLOX head tail (sigmoid and reassembly) that the compiled model does not contain (the compiled graph ends at
the nine head convolutions).  This Python class only declares the measurement, builds the
backend part of the run specification and interprets the recorded facts into realization evidence.  Mobilint
concepts stay in this package and in the native Mobilint backend.

Measurement boundary: DUT-service ends at the poll that observes ``waitFor(0) == true``; it contains the
runtime's own input conversion (float32 -> quantized NPU input), the host-device transfers and the output
conversion (dequantization to float32), which qbruntime performs inside the inference request, before the Future
reports completion.  ``Future::get`` (it hands over the float32 outputs) and the head tail are in the output stage;
their durations are recorded per request.
qbruntime exposes no per-inference device timing, so no device segment is reported.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ...contracts.records import DUTCapabilities, DUTDescriptor, RealizationEvidence
from ...contracts.specs import DUTConfig, WorkloadSpec, parse_strict
from ...contracts.status import CompletionModel, EvidenceValidity, ExecutionKind, RealizedCondition, RunClockDomain
from ..native import NativeRealtimeAdapter, describe_native, native_binary, native_identity

CORE_MODES = ("single",)
GRAPHS = ("head9",)
DEVICE_RE = re.compile(r"^/dev/aries(\d+)$")


@dataclass(frozen=True)
class MobilintOptions:
    """Core mode and the pinned core of the chip (single-core mode on one (cluster, core))."""

    core_mode: str = "single"
    cluster: int = 0
    core: int = 0

    def __post_init__(self) -> None:
        if self.core_mode not in CORE_MODES:
            raise ValueError(f"core_mode must be one of {CORE_MODES}")
        if self.cluster not in (0, 1) or self.core not in (0, 1, 2, 3):
            raise ValueError("cluster must be 0..1 and core 0..3")


@dataclass(frozen=True)
class MxqProgram:
    """A YOLOX model compiled by qbcompiler.  ``mxq`` is a host path (compiled models are not committed; the
    SHA-256 pins the identity); ``graph: head9`` is the model cut at the nine head convolutions, whose cut ONNX
    file is pinned by ``cut_onnx_sha256`` (tools/mobilint/cut_head.py record)."""

    language: str
    mxq: str
    mxq_sha256: str
    graph: str
    cut_onnx_sha256: str
    compile_record: str
    input_bytes: int
    task_output_bytes: int

    def __post_init__(self) -> None:
        if self.language != "mobilint_mxq":
            raise ValueError("device_program.language must be mobilint_mxq")
        if self.graph not in GRAPHS:
            raise ValueError(f"device_program.graph must be one of {GRAPHS}")
        for name in ("mxq_sha256", "cut_onnx_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"device_program.{name} must be a lowercase SHA-256")
        if self.input_bytes <= 0 or self.task_output_bytes <= 0:
            raise ValueError("device_program buffer sizes must be > 0")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _quantiles(values: list[int]) -> Optional[dict[str, Any]]:
    if not values:
        return None
    v = sorted(values)
    n = len(v)
    rank = lambda p: v[min(n - 1, max(0, math.ceil(p * n) - 1))]  # noqa: E731
    return {"n": n, "min": v[0], "p50": rank(0.5), "p99": rank(0.99), "max": v[-1]}


class MobilintAriesDUT(NativeRealtimeAdapter):
    ADAPTER = "mobilint_aries"
    VERSION = "1-native-qbruntime-async-head9"
    BINARY = "sdv-bmt-native-mobilint"

    def __init__(self, config: DUTConfig) -> None:
        super().__init__(config)
        self._options = parse_strict(MobilintOptions, config.runtime_options, "dut.runtime_options")
        m = DEVICE_RE.match(config.device)
        if not m:
            raise ValueError("dut.device must be an ARIES device node /dev/ariesN")
        self._device_number = int(m.group(1))
        self._identity: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------ identity
    def device_identity(self) -> dict[str, Any]:
        """Identity detected at runtime (runtime version and library actually loaded, device node facts), plus
        the identity of the native measurement plane that executes the run."""
        if self._identity is None:
            binary = native_binary(self.BINARY)
            identity = describe_native(binary, {"device_number": self._device_number})
            identity.pop("reply", None)
            identity.pop("ok", None)
            self._identity = {**identity, **native_identity(binary)}
        return self._identity

    def describe(self) -> DUTDescriptor:
        identity = self.device_identity()
        o = self._options
        return DUTDescriptor(
            adapter=self.ADAPTER,
            adapter_version=self.VERSION,
            implementation_kind=ExecutionKind.ACTUAL,
            device=f"Mobilint ARIES {self._config.device} single core Cluster{o.cluster}/Core{o.core} "
                   f"(qbruntime {identity.get('mobilint.qbruntime_version')})",
            completion_model=CompletionModel.POLLING,
            clock_domain=RunClockDomain.MONOTONIC,
            submit_semantics="native runner submit: Model::inferAsync on the item's staged HWC float32 tensor (the "
                             "runtime quantizes the input and queues the request) and returns",
            complete_semantics="Future::waitFor(0) returns true (the runtime has dequantized the outputs to "
                               "float32 by then); Future::get (hands over the outputs) and the YOLOX head tail "
                               "(sigmoid, reassembly to anchors x 85) follow in the output stage",
            device_identity=identity,
        )

    def capabilities(self) -> DUTCapabilities:
        return DUTCapabilities(completion_model=CompletionModel.POLLING, max_inflight=1, virtual_timing=False,
                               device_timing=False)

    def measurement_declaration(self) -> dict[str, Any]:
        o = self._options
        return {
            # the runtime converts and transfers the input inside the request: it is part of DUT-service
            "prepare_input_h2d": "included",
            "device_timing": None,
            "instrumentation": {
                "adapter": self.ADAPTER,
                "measurement_plane": f"native:{self.BINARY}",
                "device_timing": None,
                "device_timing_reason": "qbruntime exposes no per-inference device timing",
                "backend_snapshots": "request state and other openers of the device node at measure_begin and "
                                     "drain_end",
                "core_mode": o.core_mode,
                "target_core": f"Cluster{o.cluster}/Core{o.core}",
                "async_pipeline": True,
                "output_stage_observations": ["mobilint.future_get_ns", "mobilint.host_tail_ns"],
                "host_tail": "sigmoid(objectness), sigmoid(classes), reassembly to anchors x 85 in collect()",
            },
        }

    # ------------------------------------------------------------------ run specification (backend part)
    def backend_spec(self, workload: WorkloadSpec) -> dict[str, Any]:
        program = workload.options.get("device_program")
        if program is None:
            raise ValueError("workload.options.device_program is required by the mobilint_aries adapter")
        p = parse_strict(MxqProgram, program, "workload.options.device_program")
        mxq = Path(p.mxq).expanduser()
        if not mxq.is_absolute():
            raise ValueError("device_program.mxq must be an absolute (or ~/) host path")
        if _file_sha256(mxq) != p.mxq_sha256:
            raise ValueError(f"{mxq}: SHA-256 differs from device_program.mxq_sha256")
        return {
            "device_number": self._device_number,
            "program": {"mxq_path": str(mxq), "mxq_sha256": p.mxq_sha256, "graph": p.graph,
                        "cut_onnx_sha256": p.cut_onnx_sha256, "artifact_sha256": workload.artifact_sha256,
                        "input_size": workload.options["input_size"], "num_anchors": workload.options["num_anchors"],
                        "input_bytes": p.input_bytes, "task_output_bytes": p.task_output_bytes},
        }

    # ------------------------------------------------------------------ interpretation of native facts
    def realization_evidence(self, window: Any) -> tuple[RealizationEvidence, ...]:
        snaps = getattr(window, "snapshots", {}) or {}
        begin, end = snaps.get("measure_begin"), snaps.get("drain_end")
        observations = [e.get("observations") or {} for e in getattr(window, "device_observations", ())]
        src = f"dut:{self.ADAPTER}"
        out: list[RealizationEvidence] = []

        def rec(condition: RealizedCondition, validity: EvidenceValidity, source: str, semantic: str,
                value: Any = None, equivalence: str = "exact", unit: Optional[str] = None) -> None:
            out.append(RealizationEvidence(condition, validity, source, semantic, unit, "device", None, value,
                                           equivalence))

        rec(RealizedCondition.INPUT_RESIDENCY, EvidenceValidity.SELF_REPORTED, f"{src}:buffer_management",
            "prepared inputs live in host memory as HWC float32 (converted once at staging); the runtime "
            "quantizes and transfers each request's input inside the inference request (not independently traced)",
            {"prepared_input_location": "host", "layout": "HWC float32"})
        rec(RealizedCondition.BUFFER_STATE, EvidenceValidity.SELF_REPORTED, f"{src}:buffer_management",
            "staged inputs and the raw output buffer are allocated before the window; Future::get hands over result "
            "arrays that the runtime allocated and filled before completion (output stage); durations of get and of "
            "the head tail per request",
            {"future_get_ns": _quantiles([o["mobilint.future_get_ns"] for o in observations
                                          if o.get("mobilint.future_get_ns") is not None]),
             "host_tail_ns": _quantiles([o["mobilint.host_tail_ns"] for o in observations
                                         if o.get("mobilint.host_tail_ns") is not None])})
        for condition, why in ((RealizedCondition.DATA_MOVEMENT, "host-NPU transfers happen inside qbruntime; no "
                                "transfer counters are exposed"),
                               (RealizedCondition.DEVICE_FREQUENCY, "no NPU clock query is exposed by qbruntime or "
                                "the device node"),
                               (RealizedCondition.THERMAL_STATE, "no NPU temperature or throttle query is exposed"),
                               (RealizedCondition.MEMORY_PRESSURE_BANDWIDTH, "no NPU memory bandwidth counters are "
                                "exposed")):
            rec(condition, EvidenceValidity.NOT_OBSERVABLE, "mobilint:qbruntime", why)
        if begin is None or end is None:
            for condition in (RealizedCondition.RUNTIME_QUEUE_STATE, RealizedCondition.COMPETING_DEVICE_USERS):
                rec(condition, EvidenceValidity.NOT_OBSERVABLE, src,
                    "backend snapshot at measure_begin/drain_end unavailable")
            return tuple(out)
        rec(RealizedCondition.RUNTIME_QUEUE_STATE, EvidenceValidity.SELF_REPORTED, f"{src}:request_state",
            "the native runner's own record of a submitted request whose Future::waitFor(0) did not yet report "
            "completion, at the window edges; by design at most one request is in flight; qbruntime exposes no "
            "queue state of its own",
            {"request_pending_measure_begin": begin.get("request_pending"),
             "request_pending_drain_end": end.get("request_pending")})
        users = {"foreign_device_openers_measure_begin": begin.get("foreign_device_openers"),
                 "foreign_device_openers_drain_end": end.get("foreign_device_openers")}
        if any(v is None for v in users.values()):
            rec(RealizedCondition.COMPETING_DEVICE_USERS, EvidenceValidity.NOT_OBSERVABLE, "host:/proc",
                "other processes holding the device node open -- /proc was not readable")
        else:
            rec(RealizedCondition.COMPETING_DEVICE_USERS, EvidenceValidity.OBSERVED, "host:/proc",
                "other processes holding the device node open at the window edges (/proc/*/fd of processes this user "
                "may read; processes of other users are not visible; activity between the edges is not observed)",
                users)
        return tuple(out)
