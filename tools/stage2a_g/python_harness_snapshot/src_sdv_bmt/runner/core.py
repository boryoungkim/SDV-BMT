"""Runner core data: request state machine, count identity, prepared-input store.

Request states follow design v03 E.6 (Stage 1A subset).  Count identity::

    planned = not_issued + dropped + issued
    issued  = completed + error + timed_out

A late completion is a diagnostic on a TIMED_OUT request and never changes
the counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..contracts.ids import is_sha256_hex
from ..contracts.protocols import AdapterInputOverride, DUTAdapter, InputProvider, Workload
from ..contracts.records import EquivalenceValidation, ErrorInfo, InputItem
from ..contracts.status import EquivalenceStatus, RequestState

S = RequestState

_ALLOWED = {
    S.PLANNED: {S.ISSUED, S.DROPPED, S.NOT_ISSUED},
    S.ISSUED: {S.INFLIGHT, S.ERROR},
    S.INFLIGHT: {S.COMPLETED, S.ERROR, S.TIMED_OUT},
}

_ISSUED_OR_LATER = {S.ISSUED, S.INFLIGHT, S.COMPLETED, S.ERROR, S.TIMED_OUT}


class RunnerInvariantError(RuntimeError):
    """An internal runner invariant was violated (a bug, never a DUT outcome)."""


class RequestTable:
    def __init__(self, planned: int) -> None:
        self._states = [S.PLANNED] * planned
        self._late = [False] * planned

    @property
    def planned(self) -> int:
        return len(self._states)

    def state(self, request_id: int) -> RequestState:
        return self._states[request_id]

    def transition(self, request_id: int, new: RequestState) -> None:
        current = self._states[request_id]
        if new not in _ALLOWED.get(current, set()):
            raise RunnerInvariantError(
                f"request {request_id}: illegal transition {current.value} -> {new.value}"
            )
        self._states[request_id] = new

    def mark_late_completion(self, request_id: int) -> None:
        if self._states[request_id] is not S.TIMED_OUT or self._late[request_id]:
            raise RunnerInvariantError(f"request {request_id}: unexpected late completion")
        self._late[request_id] = True

    def counts(self) -> dict[str, int]:
        states = self._states
        return {
            "planned": len(states),
            "issued": sum(s in _ISSUED_OR_LATER for s in states),
            "completed": states.count(S.COMPLETED),
            "error": states.count(S.ERROR),
            "timed_out": states.count(S.TIMED_OUT),
            "dropped": states.count(S.DROPPED),
            "not_issued": states.count(S.NOT_ISSUED),
            "late_completion": sum(self._late),
            "unresolved": sum(s in (S.PLANNED, S.ISSUED, S.INFLIGHT) for s in states),
        }


def count_identity_violations(counts: dict[str, int]) -> list[str]:
    violations = []
    if counts["planned"] != counts["not_issued"] + counts["dropped"] + counts["issued"]:
        violations.append("planned != not_issued + dropped + issued")
    if counts["issued"] != counts["completed"] + counts["error"] + counts["timed_out"]:
        violations.append("issued != completed + error + timed_out")
    return violations


# --------------------------------------------------------------------------
# prepared inputs (Runner-owned)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparationFailure:
    stage: str  # "payload" | "preprocess" | "prepare_input"
    error: ErrorInfo


class PreparedStore:
    """Prepared inputs owned by the Runner, keyed by item_index.

    Neither the InputProvider nor the Workload holds these objects.  The store
    is discarded during cleanup, before ``DUT.cleanup``.
    """

    def __init__(self) -> None:
        self._prepared: dict[int, Any] = {}
        self._failed: dict[int, PreparationFailure] = {}
        self._discarded = False

    def put(self, item_index: int, prepared: Any) -> None:
        self._check_open()
        self._prepared[item_index] = prepared

    def fail(self, item_index: int, failure: PreparationFailure) -> None:
        self._check_open()
        self._failed[item_index] = failure

    def get(self, item_index: int) -> Any:
        self._check_open()
        return self._prepared[item_index]

    def failure(self, item_index: int) -> Optional[PreparationFailure]:
        return self._failed.get(item_index)

    @property
    def prepared_count(self) -> int:
        return len(self._prepared)

    @property
    def failed(self) -> dict[int, PreparationFailure]:
        return dict(self._failed)

    @property
    def discarded(self) -> bool:
        return self._discarded

    def discard(self) -> int:
        count = len(self._prepared)
        self._prepared.clear()
        self._discarded = True
        return count

    def _check_open(self) -> None:
        if self._discarded:
            raise RunnerInvariantError("prepared store already discarded")


def error_info(exc: BaseException) -> ErrorInfo:
    return ErrorInfo(code=type(exc).__name__, message=str(exc))


def prepare_inputs(
    provider: InputProvider,
    workload: Workload,
    adapter: DUTAdapter,
    items: Sequence[InputItem],
    store: PreparedStore,
) -> None:
    """Eager PREPARE_INPUTS (design v03 D.3/F.5). Per-item failures are kept in
    the store and later become NOT_ISSUED requests; they do not abort the run."""
    for item in items:
        stage = "payload"
        try:
            payload = provider.payload(item.item_index)
            stage = "preprocess"
            task_input = workload.preprocess(payload, item)
            stage = "prepare_input"
            prepared = adapter.prepare_input(task_input, item)
        except Exception as exc:  # recorded per item; the run continues
            store.fail(item.item_index, PreparationFailure(stage, error_info(exc)))
            continue
        store.put(item.item_index, prepared)


def prepare_inputs_with_override(
    provider: InputProvider,
    override: AdapterInputOverride,
    items: Sequence[InputItem],
    store: PreparedStore,
) -> None:
    """PREPARE_INPUTS for ``input_pipeline.mode=adapter_override``: the adapter's
    override replaces preprocess + prepare_input.  A per-item failure is
    recorded (later NOT_ISSUED); the Workload path is never used as a fallback."""
    for item in items:
        stage = "payload"
        try:
            payload = provider.payload(item.item_index)
            stage = "adapter_override"
            prepared = override.prepare_from_payload(payload, item)
        except Exception as exc:
            store.fail(item.item_index, PreparationFailure(stage, error_info(exc)))
            continue
        store.put(item.item_index, prepared)


EQUIVALENCE_METHOD = "exact_adapter_prepared_input_digest"


def _adapter_digest(adapter: DUTAdapter, prepared: Any) -> str:
    digest = adapter.prepared_input_digest(prepared)
    if not is_sha256_hex(digest):
        raise ValueError(f"DUT adapter prepared_input_digest returned {digest!r}, not a SHA-256 hex digest")
    return digest


def validate_input_equivalence(
    provider: InputProvider,
    workload: Workload,
    adapter: DUTAdapter,
    items: Sequence[InputItem],
    store: PreparedStore,
) -> tuple[EquivalenceValidation, list[dict[str, Any]]]:
    """Check override-prepared inputs against the Workload reference path
    (``Workload.preprocess -> DUTAdapter.prepare_input``) for every item, using
    the DUT adapter's canonical digest (not the override's).  Reference objects
    are discarded at once; this is validation, never a fallback.

    Returns the validation record and the per-item errors seen on the reference
    side (or while digesting), which are reported, not blamed on the override.
    """
    validated = 0
    mismatched: list[int] = []
    reference_errors: list[int] = []
    both_failed: list[int] = []
    errors: list[dict[str, Any]] = []
    reference_digests: list[str] = []
    for item in items:
        index = item.item_index
        reference_digest: Optional[str] = None
        try:
            reference = adapter.prepare_input(workload.preprocess(provider.payload(index), item), item)
            reference_digest = _adapter_digest(adapter, reference)
        except Exception as exc:
            reference_failure = error_info(exc)
        if store.failure(index) is not None:
            # The override could not prepare this item (it will be NOT_ISSUED).
            (mismatched if reference_digest is not None else both_failed).append(index)
            continue
        if reference_digest is None:
            reference_errors.append(index)
            errors.append({"item_index": index, "side": "reference", "error": reference_failure})
            continue
        try:
            override_digest = _adapter_digest(adapter, store.get(index))
        except Exception as exc:
            mismatched.append(index)
            errors.append({"item_index": index, "side": "override", "error": error_info(exc)})
            continue
        validated += 1
        reference_digests.append(reference_digest)
        if override_digest != reference_digest:
            mismatched.append(index)

    def record(status: EquivalenceStatus, reason: Optional[str]) -> EquivalenceValidation:
        return EquivalenceValidation(status, EQUIVALENCE_METHOD, validated, tuple(mismatched),
                                     tuple(reference_errors), tuple(both_failed), reason)

    if mismatched:
        return record(EquivalenceStatus.FAILED, "override_not_equivalent"), errors
    if reference_errors:
        return record(EquivalenceStatus.NOT_PERFORMED, "reference_path_errors"), errors
    if validated == 0:
        return record(EquivalenceStatus.NOT_PERFORMED, "no_comparable_items"), errors
    if validated >= 2 and len(set(reference_digests)) == 1:
        # A constant digest cannot distinguish anything; equivalence is unproven.
        return record(EquivalenceStatus.NOT_PERFORMED, "degenerate_prepared_input_digest"), errors
    return record(EquivalenceStatus.PASSED, None), errors
