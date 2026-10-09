"""Stage 1A component protocols (design v03 D.3).

Deviation note: ``HardwareStressProvider.verify`` takes ``window_ns: int``
instead of ``window_s: float`` so that every Stage 1A time value stays an
integer on the run clock.

Stage 1A adapters are blocking (``DUTAdapter.run``).  The Stage 2a-G
investigation found the CUDA driver path to be asynchronous with completion
observable by polling, so Stage 1R adds the ``PollingDUTAdapter`` extension:
``submit`` enqueues and returns, ``poll`` reports completion without blocking.
No callback contract is defined (not needed by the investigated runtime).

Input pipeline (design change "Explicit DUT Adapter Preprocess Override"):
the Workload owns the canonical input semantics and the reference
preprocessing; a DUT adapter may offer an ``AdapterInputOverride`` that
replaces only the preprocess/prepare_input execution path.  The adapter
receives an immutable ``WorkloadInputContract``, never the Workload object.
Which path runs is selected explicitly in the plan; there is no
auto-selection and no fallback.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence

from .records import (
    AdapterInputOverrideDescriptor,
    DUTCapabilities,
    DUTCompletion,
    DUTDescriptor,
    DUTRequest,
    DUTResult,
    HostInfo,
    InputDescriptor,
    InputItem,
    PreparedInfo,
    RealizationEvidence,
    StressRecord,
    SupportDecision,
    WorkloadInputContract,
)
from .specs import DUTConfig, HardwareStressSpec, WorkloadSpec

Payload = bytes


class InputValidationError(ValueError):
    """Input source/index/selection is invalid (PREFLIGHT rejection, no bundle)."""


class LifecycleError(RuntimeError):
    """A component method was called outside its permitted lifecycle order."""


class WorkloadError(ValueError):
    """Workload preprocess/postprocess/encode failed for an item or output."""


class InputProvider(Protocol):
    """Source identity, deterministic items and raw payload access only.

    An InputProvider never receives or stores DUT objects or prepared inputs.
    """

    def describe(self) -> InputDescriptor: ...

    def items(self) -> Sequence[InputItem]: ...  # deterministic, source-derived only

    def open(self) -> None: ...  # idempotent

    def payload(self, item_index: int) -> Payload: ...

    def release(self) -> None: ...  # idempotent


class Workload(Protocol):
    def input_contract(self) -> WorkloadInputContract: ...  # immutable, versioned

    def preprocess(self, payload: Payload, item: InputItem) -> object: ...  # reference path

    def postprocess(self, raw_output: object, item: InputItem) -> object: ...

    def encode(self, task_output: object) -> bytes: ...


class DUTAdapter(Protocol):
    def describe(self) -> DUTDescriptor: ...

    def capabilities(self) -> DUTCapabilities: ...

    def prepare(self, workload: WorkloadSpec, config: DUTConfig) -> PreparedInfo: ...

    def prepare_input(self, task_input: object, item: InputItem) -> object: ...

    def run(self, request: DUTRequest) -> DUTResult: ...  # Stage 1A: blocking only

    def cleanup(self) -> None: ...  # idempotent

    def input_override(
        self, contract: WorkloadInputContract, implementation_id: str
    ) -> Optional["AdapterInputOverride"]:
        """The adapter's input override ``implementation_id`` for ``contract``, or
        None if the adapter has no such override.  Side-effect free; only called
        when the plan selects ``adapter_override``."""
        ...

    def prepared_input_digest(self, prepared_input: object) -> str:
        """Canonical SHA-256 of a prepared input produced by this adapter through
        either path.  Owned by the adapter, not by an override, so an override
        cannot define its own equivalence."""
        ...


class PollingDUTAdapter(Protocol):
    """Stage 1R DUT adapter with ``completion_model=polling``.

    Identity, preparation, input-override and cleanup methods are those of
    ``DUTAdapter`` (without ``run``).  The realtime runner keeps at most
    ``max_inflight`` handles outstanding and calls ``poll`` until a completion
    is returned.  ``observe`` and ``realization_evidence`` are called only
    outside timed windows (before MEASURE and after DRAIN).
    """

    def describe(self) -> DUTDescriptor: ...

    def capabilities(self) -> DUTCapabilities: ...

    def prepare(self, workload: WorkloadSpec, config: DUTConfig) -> PreparedInfo: ...

    def prepare_input(self, task_input: object, item: InputItem) -> object: ...

    def submit(self, request: DUTRequest) -> object:
        """Enqueue the request on the device and return an opaque handle; never
        waits for completion."""
        ...

    def poll(self, handle: object) -> Optional[DUTCompletion]:
        """The completion of ``handle`` if it has completed, else None; never blocks."""
        ...

    def observe(self, point: str) -> dict[str, Any]:
        """A JSON-serializable snapshot of backend state at a lifecycle point
        (e.g. ``measure_begin``, ``drain_end``); recorded as raw evidence."""
        ...

    def realization_evidence(self) -> tuple[RealizationEvidence, ...]:
        """Evidence for the device-side realized conditions of this run, built
        from the adapter's observations (portable condition names, backend
        sources)."""
        ...

    def measurement_declaration(self) -> dict[str, Any]:
        """What the adapter's measurement covers, as a JSON-serializable mapping:

        * ``prepare_input_h2d``: ``included`` / ``excluded`` / ``inseparable``,
          whether the per-request host-to-device transfer of the prepared
          input is inside the DUT-service interval;
        * ``device_timing``: how per-request device-domain durations are
          obtained (method, portable segment names, clock, semantics), or None;
        * ``instrumentation``: the instrumentation profile.

        All three are part of the realtime measurement boundary, and
        ``instrumentation`` is also the run's ``telemetry`` identity, so runs
        under different instrumentation never compare as the same measurement.
        """
        ...

    def cleanup(self) -> None: ...

    def input_override(
        self, contract: WorkloadInputContract, implementation_id: str
    ) -> Optional["AdapterInputOverride"]: ...

    def prepared_input_digest(self, prepared_input: object) -> str: ...


class AdapterInputOverride(Protocol):
    """Adapter-owned replacement of preprocess + prepare_input (e.g. fused)."""

    def describe(self) -> AdapterInputOverrideDescriptor: ...

    def prepare_from_payload(self, payload: Payload, item: InputItem) -> object: ...


class HardwareStressProvider(Protocol):
    def check(self, spec: HardwareStressSpec, level: str, host: HostInfo) -> SupportDecision: ...

    def prepare(self) -> StressRecord: ...

    def activate(self) -> StressRecord: ...

    def verify(self, window_ns: int) -> StressRecord: ...

    def observe(self) -> StressRecord: ...

    def deactivate(self) -> StressRecord: ...

    def restore(self) -> StressRecord: ...  # every termination path; idempotent
