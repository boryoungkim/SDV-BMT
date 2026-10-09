"""Blocking FakeDUT on a virtual clock (``adapter: fake_blocking``).

``run()`` is a blocking call in virtual time: it returns immediately in real
time with ``virtual_service_ns`` from the service model, and the virtual runner
treats the call as occupying the slot for that long.  The raw output includes
the request ID and the item digest so input/output correspondence is checkable.

Service model (``runtime_options.service_model``):

* ``kind: fixed`` with ``service_ns``
* ``kind: table`` with ``default_service_ns`` and ``entries`` keyed by
  ``item_index``; an entry may override ``service_ns``, set
  ``run_outcome: error`` (request-level DUT error after ``service_ns``) and/or
  ``prepare_input_outcome: error`` (item preparation failure).

Lifecycle: created -> prepare() -> prepare_input()/run() ... -> cleanup().
``prepare_input``/``run`` before ``prepare`` or after ``cleanup`` raise
LifecycleError.  ``cleanup`` is idempotent.

Input override (``runtime_options.input_override``, optional): when present the
adapter offers ``FakeFusedPrefixOverride``, a minimal fused path that turns the
raw payload into the prepared input in one step (``sha256(prefix || payload)``)
using only the WorkloadInputContract (semantics ``fake-prefix-preprocess/1``;
the prefix comes from the contract's ``prefix_hex``).  It never sees the
Workload object.  ``divergent: true`` is a test-only knob that hashes a wrong
prefix to exercise equivalence failure.  Absent = the adapter has no override.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

from ..contracts.ids import digest_of, sha256_hex
from ..contracts.protocols import LifecycleError
from ..contracts.records import (
    AdapterInputOverrideDescriptor,
    DUTCapabilities,
    DUTCompletion,
    DUTDescriptor,
    DUTRequest,
    DUTResult,
    ErrorInfo,
    InputItem,
    PreparedInfo,
    RealizationEvidence,
    WorkloadInputContract,
    item_digest,
)
from ..contracts.specs import DUTConfig, WorkloadSpec, parse_strict, workload_identity_digest
from ..contracts.status import CompletionModel, EvidenceValidity, ExecutionKind, RealizedCondition, RunClockDomain

_OUTCOMES = ("ok", "error")


class FakeDUTInjectedError(RuntimeError):
    """Configured (injected) FakeDUT failure."""


@dataclass(frozen=True)
class FakeServiceEntry:
    item_index: int
    service_ns: Optional[int] = None
    run_outcome: str = "ok"
    error_code: Optional[str] = None
    prepare_input_outcome: str = "ok"

    def __post_init__(self) -> None:
        if self.item_index < 0:
            raise ValueError("entry.item_index must be >= 0")
        if self.service_ns is not None and self.service_ns < 0:
            raise ValueError("entry.service_ns must be >= 0")
        if self.run_outcome not in _OUTCOMES or self.prepare_input_outcome not in _OUTCOMES:
            raise ValueError(f"outcomes must be one of {_OUTCOMES}")
        if self.error_code is not None and self.run_outcome != "error":
            raise ValueError("entry.error_code requires run_outcome=error")


@dataclass(frozen=True)
class FakeServiceModel:
    kind: str
    service_ns: Optional[int] = None
    default_service_ns: Optional[int] = None
    entries: tuple[FakeServiceEntry, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "fixed":
            if self.service_ns is None or self.service_ns < 0:
                raise ValueError("fixed service model requires service_ns >= 0")
            if self.default_service_ns is not None or self.entries:
                raise ValueError("fixed service model takes no default_service_ns/entries")
        elif self.kind == "table":
            if self.default_service_ns is None or self.default_service_ns < 0:
                raise ValueError("table service model requires default_service_ns >= 0")
            if self.service_ns is not None:
                raise ValueError("table service model uses default_service_ns, not service_ns")
            indices = [e.item_index for e in self.entries]
            if len(set(indices)) != len(indices):
                raise ValueError("table entries must have unique item_index")
        else:
            raise ValueError("service_model.kind must be 'fixed' or 'table'")

    def default_ns(self) -> int:
        value = self.service_ns if self.kind == "fixed" else self.default_service_ns
        assert value is not None
        return value


@dataclass(frozen=True)
class FakeInputOverrideOptions:
    divergent: bool = False


@dataclass(frozen=True)
class FakeDUTOptions:
    service_model: FakeServiceModel
    input_override: Optional[FakeInputOverrideOptions] = None


@dataclass(frozen=True)
class FakePreparedInput:
    """DUT-side prepared input. Owned by the Runner's prepared store."""

    item_index: int
    item_digest: str
    input_sha256: str


class FakeBlockingDUT:
    ADAPTER = "fake_blocking"
    VERSION = "1"
    RAW_OUTPUT_SCHEMA = "fake_blocking-raw-v1"

    def __init__(self, config: DUTConfig) -> None:
        if config.adapter != self.ADAPTER:
            raise ValueError(f"DUT adapter must be {self.ADAPTER!r}")
        options = parse_strict(FakeDUTOptions, config.runtime_options, "dut.runtime_options")
        self._config = config
        self._model = options.service_model
        self._entries = {e.item_index: e for e in self._model.entries}
        self._override_options = options.input_override
        self._state = "created"
        self._workload_digest: Optional[str] = None

    def describe(self) -> DUTDescriptor:
        return DUTDescriptor(
            adapter=self.ADAPTER,
            adapter_version=self.VERSION,
            implementation_kind=ExecutionKind.MOCK,
            device=self._config.device,
            completion_model=CompletionModel.BLOCKING,
            clock_domain=RunClockDomain.VIRTUAL,
            submit_semantics="virtual: FakeBlockingDUT.run() is entered (blocking call begins)",
            complete_semantics="virtual: run() returns at t_submit_begin + service-model time",
        )

    def capabilities(self) -> DUTCapabilities:
        return DUTCapabilities(
            completion_model=CompletionModel.BLOCKING,
            max_inflight=1,
            virtual_timing=True,
            device_timing=False,
        )

    def prepare(self, workload: WorkloadSpec, config: DUTConfig) -> PreparedInfo:
        if self._state != "created":
            raise LifecycleError(f"prepare() not allowed in state {self._state!r}")
        if config != self._config:
            raise ValueError("prepare() config differs from the adapter's construction config")
        self._workload_digest = workload_identity_digest(workload)
        self._state = "prepared"
        return PreparedInfo(
            workload_identity_digest=self._workload_digest,
            notes=("virtual FakeDUT: no model artifact is loaded",),
        )

    def _require_prepared(self, method: str) -> None:
        if self._state != "prepared":
            raise LifecycleError(f"{method}() requires prepare() first (state {self._state!r})")

    def _check_injected_prepare_error(self, item: InputItem) -> None:
        entry = self._entries.get(item.item_index)
        if entry is not None and entry.prepare_input_outcome == "error":
            raise FakeDUTInjectedError(f"injected prepare_input error for item {item.item_index}")

    def prepare_input(self, task_input: object, item: InputItem) -> FakePreparedInput:
        self._require_prepared("prepare_input")
        if not isinstance(task_input, bytes):
            raise TypeError("FakeBlockingDUT.prepare_input expects bytes task input")
        self._check_injected_prepare_error(item)
        return FakePreparedInput(item.item_index, item_digest(item), sha256_hex(task_input))

    def input_override(
        self, contract: WorkloadInputContract, implementation_id: str
    ) -> Optional["FakeFusedPrefixOverride"]:
        if self._override_options is None or implementation_id != FakeFusedPrefixOverride.IMPLEMENTATION_ID:
            return None
        return FakeFusedPrefixOverride(self, contract, self._override_options.divergent)

    def prepared_input_digest(self, prepared_input: object) -> str:
        if not isinstance(prepared_input, FakePreparedInput):
            raise TypeError("not a FakeBlockingDUT prepared input")
        return digest_of(prepared_input)

    def run(self, request: DUTRequest) -> DUTResult:
        self._require_prepared("run")
        if len(request.item_indices) != 1 or len(request.prepared_inputs) != 1:
            raise ValueError("FakeBlockingDUT supports batch=1 only")
        prepared = request.prepared_inputs[0]
        if not isinstance(prepared, FakePreparedInput) or prepared.item_index != request.item_indices[0]:
            raise ValueError("prepared input does not correspond to the request item")
        entry = self._entries.get(prepared.item_index)
        service_ns = self._model.default_ns()
        if entry is not None and entry.service_ns is not None:
            service_ns = entry.service_ns
        if entry is not None and entry.run_outcome == "error":
            return DUTResult(
                request_id=request.request_id,
                ok=False,
                raw_output=None,
                virtual_service_ns=service_ns,
                error=ErrorInfo(
                    entry.error_code or "fake_injected_run_error",
                    f"injected run error for item {prepared.item_index}",
                ),
            )
        raw = {
            "schema": self.RAW_OUTPUT_SCHEMA,
            "request_id": request.request_id,
            "item_index": prepared.item_index,
            "item_digest": prepared.item_digest,
            "input_sha256": prepared.input_sha256,
            "value": int(prepared.input_sha256[:8], 16),
        }
        return DUTResult(request.request_id, True, raw, service_ns)

    def cleanup(self) -> None:
        self._state = "cleaned"
        self._workload_digest = None


class FakeFusedPrefixOverride:
    """Minimal fused input path: payload -> prepared input in one step.

    Implements semantics ``fake-prefix-preprocess/1`` from the contract alone:
    the prepared input digest is ``sha256(prefix || payload)`` computed in a
    single streaming pass, without materializing the preprocessed task input.
    """

    IMPLEMENTATION_ID = "fake-fused-prefix-preprocess"
    IMPLEMENTATION_VERSION = "1"
    SEMANTICS = ("fake-prefix-preprocess/1",)
    FUSED_STEPS = ("workload.preprocess", "dut.prepare_input")

    def __init__(self, dut: FakeBlockingDUT, contract: WorkloadInputContract, divergent: bool) -> None:
        self._dut = dut
        self._contract = contract
        self._divergent = divergent

    def describe(self) -> AdapterInputOverrideDescriptor:
        definition = {
            "implementation_id": self.IMPLEMENTATION_ID,
            "implementation_version": self.IMPLEMENTATION_VERSION,
            "algorithm": "prepared.input_sha256 = sha256(contract.prefix_hex bytes || payload), single pass",
            "divergent_test_variant": self._divergent,
        }
        return AdapterInputOverrideDescriptor(
            implementation_id=self.IMPLEMENTATION_ID,
            implementation_version=self.IMPLEMENTATION_VERSION,
            implementation_digest=digest_of(definition),
            preprocess_semantics_ids=self.SEMANTICS,
            fused_steps=self.FUSED_STEPS,
        )

    def prepare_from_payload(self, payload: bytes, item: InputItem) -> FakePreparedInput:
        self._dut._require_prepared("prepare_from_payload")
        if self._contract.preprocess_semantics_id not in self.SEMANTICS:
            raise ValueError(f"unsupported preprocess semantics {self._contract.preprocess_semantics_id!r}")
        if not isinstance(payload, bytes) or sha256_hex(payload) != item.payload_sha256:
            raise ValueError(f"payload of item {item.item_index} does not match payload_sha256")
        self._dut._check_injected_prepare_error(item)
        prefix = bytes.fromhex(self._contract.parameter("prefix_hex"))
        if self._divergent:
            prefix = b"divergent:" + prefix
        fused = hashlib.sha256(prefix)
        fused.update(payload)
        return FakePreparedInput(item.item_index, item_digest(item), fused.hexdigest())



# --------------------------------------------------------------------------
# Stage 1R: mock polling adapter (``adapter: fake_polling``)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FakePollingItem:
    item_index: int
    polls: int

    def __post_init__(self) -> None:
        if self.item_index < 0 or self.polls < 1:
            raise ValueError("fake_polling item entry needs item_index >= 0 and polls >= 1")


@dataclass(frozen=True)
class FakePollingOptions:
    polls_until_complete: int = 1
    items: tuple[FakePollingItem, ...] = ()
    run_error_items: tuple[int, ...] = ()
    prepare_error_items: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.polls_until_complete < 1:
            raise ValueError("polls_until_complete must be >= 1")


@dataclass
class _FakePollingHandle:
    request: DUTRequest
    polls_left: int


DEVICE_CONDITIONS_FOR_MOCK = (
    RealizedCondition.INPUT_RESIDENCY, RealizedCondition.BUFFER_STATE, RealizedCondition.DATA_MOVEMENT,
    RealizedCondition.RUNTIME_QUEUE_STATE, RealizedCondition.DEVICE_FREQUENCY, RealizedCondition.THERMAL_STATE,
    RealizedCondition.COMPETING_DEVICE_USERS, RealizedCondition.MEMORY_PRESSURE_BANDWIDTH,
)


class FakePollingDUT:
    """MOCK polling adapter for the Stage 1R realtime harness (conformance, harness floor).

    ``submit`` returns at once.  A request is reported complete by its
    ``polls_until_complete``-th ``poll`` (``items`` overrides the count per item),
    so its "service time" is the harness's own polling cost: no device, no clock,
    no accelerator work.  ``run_error_items`` complete with a DUT error and
    ``prepare_error_items`` fail preparation.  Raw outputs are those of
    FakeBlockingDUT (same schema), so the fake workload applies unchanged.
    """

    ADAPTER = "fake_polling"
    VERSION = "1"

    def __init__(self, config: DUTConfig) -> None:
        if config.adapter != self.ADAPTER:
            raise ValueError(f"DUT adapter must be {self.ADAPTER!r}")
        self._options = parse_strict(FakePollingOptions, config.runtime_options, "dut.runtime_options")
        self._polls = {e.item_index: e.polls for e in self._options.items}
        self._config = config
        self._state = "created"
        self._observed: list[str] = []

    def describe(self) -> DUTDescriptor:
        return DUTDescriptor(
            adapter=self.ADAPTER,
            adapter_version=self.VERSION,
            implementation_kind=ExecutionKind.MOCK,
            device=self._config.device,
            completion_model=CompletionModel.POLLING,
            clock_domain=RunClockDomain.MONOTONIC,
            submit_semantics="mock: FakePollingDUT.submit() records the request and returns",
            complete_semantics="mock: poll() reports completion on the configured n-th call",
        )

    def capabilities(self) -> DUTCapabilities:
        return DUTCapabilities(completion_model=CompletionModel.POLLING, max_inflight=1,
                               virtual_timing=False, device_timing=False)

    def measurement_declaration(self) -> dict:
        return {"prepare_input_h2d": "excluded", "device_timing": None,
                "instrumentation": {"adapter": self.ADAPTER, "device_timing": None, "backend_snapshots": None}}

    def prepare(self, workload: WorkloadSpec, config: DUTConfig) -> PreparedInfo:
        if self._state != "created":
            raise LifecycleError(f"prepare() not allowed in state {self._state!r}")
        self._state = "prepared"
        return PreparedInfo(workload_identity_digest=workload_identity_digest(workload),
                            notes=("mock polling FakeDUT: no model artifact, no device",))

    def prepare_input(self, task_input: object, item: InputItem) -> FakePreparedInput:
        if self._state != "prepared":
            raise LifecycleError("prepare_input() requires prepare() first")
        if not isinstance(task_input, bytes):
            raise TypeError("FakePollingDUT.prepare_input expects bytes task input")
        if item.item_index in self._options.prepare_error_items:
            raise FakeDUTInjectedError(f"injected prepare_input error for item {item.item_index}")
        return FakePreparedInput(item.item_index, item_digest(item), sha256_hex(task_input))

    def input_override(self, contract: WorkloadInputContract, implementation_id: str) -> None:
        return None

    def prepared_input_digest(self, prepared_input: object) -> str:
        if not isinstance(prepared_input, FakePreparedInput):
            raise TypeError("not a FakePollingDUT prepared input")
        return digest_of(prepared_input)

    def submit(self, request: DUTRequest) -> _FakePollingHandle:
        if self._state != "prepared":
            raise LifecycleError("submit() requires prepare() first")
        prepared = request.prepared_inputs[0]
        if not isinstance(prepared, FakePreparedInput) or prepared.item_index != request.item_indices[0]:
            raise ValueError("prepared input does not correspond to the request item")
        return _FakePollingHandle(request, self._polls.get(prepared.item_index, self._options.polls_until_complete))

    def poll(self, handle: object) -> Optional[DUTCompletion]:
        if not isinstance(handle, _FakePollingHandle):
            raise TypeError("not a FakePollingDUT handle")
        handle.polls_left -= 1
        if handle.polls_left > 0:
            return None
        request = handle.request
        prepared = request.prepared_inputs[0]
        if prepared.item_index in self._options.run_error_items:
            return DUTCompletion(request.request_id, False, None,
                                 ErrorInfo("fake_injected_run_error", f"injected run error for item {prepared.item_index}"))
        raw = {
            "schema": FakeBlockingDUT.RAW_OUTPUT_SCHEMA,
            "request_id": request.request_id,
            "item_index": prepared.item_index,
            "item_digest": prepared.item_digest,
            "input_sha256": prepared.input_sha256,
            "value": int(prepared.input_sha256[:8], 16),
        }
        return DUTCompletion(request.request_id, True, raw)

    def observe(self, point: str) -> dict:
        self._observed.append(point)
        return {"adapter": self.ADAPTER, "point": point, "mock": True}

    def realization_evidence(self) -> tuple[RealizationEvidence, ...]:
        return tuple(
            RealizationEvidence(condition, EvidenceValidity.NOT_OBSERVABLE, f"dut:{self.ADAPTER}",
                                "no device: mock polling adapter", None, "device", None)
            for condition in DEVICE_CONDITIONS_FOR_MOCK
        )

    def cleanup(self) -> None:
        self._state = "cleaned"
