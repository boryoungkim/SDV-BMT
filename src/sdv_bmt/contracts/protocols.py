"""Stage 1A component protocols (design v03 D.3).

Deviation note: ``HardwareStressProvider.verify`` takes ``window_ns: int``
instead of ``window_s: float`` so that every Stage 1A time value stays an
integer on the run clock.

Stage 1A adapters are blocking (``DUTAdapter.run``) and run on the virtual
clock.  Stage 1R realtime adapters (``NativeRealtimeDUT``) drive a native
measurement plane (approved architecture: C++ measurement plane, Python
analysis plane): the native process executes DUT.prepare, input staging and
the whole MEASURE/DRAIN window and records the raw facts; the Python side
only commands lifecycle phases and interprets the recorded facts.  There is
no per-request Python call and no callback contract.

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


class AnnotationSource(Protocol):
    """Ground-truth annotation content of resolved items (Model-level Metric Set v1).

    Used only when the workload declares a task-quality evaluator: the runner freezes the returned
    rows in the bundle (``quality_reference.jsonl``) so that task quality is evaluated from the bundle
    alone.  Every row's annotation must match the item's ``annotation_digest``; the source never sees
    workload, DUT or output objects.  Separate from ``InputProvider``, whose surface stays
    source/index/payload only."""

    def reference_rows(self, items: Sequence[InputItem]) -> Sequence[dict[str, Any]]: ...


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


class NativeRealtimeDUT(Protocol):
    """Stage 1R DUT adapter backed by a native measurement-plane process (``completion_model=polling``).

    Identity, input-override and digest methods are those of ``DUTAdapter`` (without ``run``).  Lifecycle:
    ``bind_run`` (portable run facts: clock anchor, schedule, run policy, CPU placement, bundle file paths,
    output pipeline) -> ``prepare`` -> ``prepare_input`` per item -> ``commit_inputs`` -> ``run_window``
    -> ``realization_evidence`` -> ``cleanup``.
    """

    def describe(self) -> DUTDescriptor: ...

    def capabilities(self) -> DUTCapabilities: ...

    def measurement_declaration(self) -> dict[str, Any]:
        """What the adapter's measurement covers, as a JSON-serializable mapping:

        * ``prepare_input_h2d``: ``included`` / ``excluded`` / ``inseparable``,
          whether the per-request host-to-device transfer of the prepared
          input is inside the DUT-service interval;
        * ``device_timing``: how per-request device-domain durations are
          obtained (method, portable segment names, clock, semantics), or None;
        * ``instrumentation``: the instrumentation profile (including the
          measurement plane and any declared profiler).

        All three are part of the realtime measurement boundary, and
        ``instrumentation`` is also the run's ``telemetry`` identity, so runs
        under different instrumentation never compare as the same measurement.
        """
        ...

    def bind_run(self, context: Any) -> None: ...

    def prepare(self, workload: WorkloadSpec, config: DUTConfig) -> PreparedInfo: ...

    def prepare_input(self, task_input: object, item: InputItem) -> object: ...

    def commit_inputs(self, plan: Sequence[tuple[int, Optional[str], Optional[object]]],
                      output_contexts: Optional[dict[int, Any]] = None) -> dict[int, Any]:
        """Stage every planned item into device-visible memory (outside MEASURE); returns the items whose
        staging failed, keyed by item index.  ``output_contexts`` (item index -> JSON object) are handed to
        the workload's native output pipeline when the workload declares them."""
        ...

    def run_window(self, handover: Any) -> Any:
        """Precondition, backend snapshot, MEASURE/DRAIN, backend snapshot, executed natively; the native
        process appends the recorded events and outputs to the bundle files it was handed."""
        ...

    def realization_evidence(self, window: Any) -> tuple[RealizationEvidence, ...]:
        """Evidence for the device-side realized conditions, interpreted from the recorded window facts
        (portable condition names, backend sources)."""
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
