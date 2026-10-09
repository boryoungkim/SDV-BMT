"""Lifecycle order enforcement shared by the Stage 1A stress providers.

Order: prepare -> activate -> verify -> observe* -> deactivate -> restore.

* ``deactivate`` is allowed once activation was attempted (ok or failed).
* ``restore`` is allowed from every state and is idempotent; after a
  successful restore every other call raises LifecycleError.
* ``check`` is side-effect free and never changes the lifecycle state.
"""

from __future__ import annotations

from typing import Any, Optional

from ..contracts.protocols import LifecycleError
from ..contracts.records import ErrorInfo, StressRecord
from ..contracts.specs import HardwareStressSpec, stress_axis_digest
from ..contracts.status import Outcome, StressStep, VerificationStatus


class StressProviderBase:
    PROVIDER = ""

    def __init__(self, spec: HardwareStressSpec, level: str) -> None:
        self._spec = spec
        self._level = level
        self._axis = stress_axis_digest(spec)
        self._state = "created"
        self._activate_attempted = False

    @property
    def lifecycle_state(self) -> str:
        return self._state

    def _record(
        self,
        step: StressStep,
        outcome: Outcome,
        *,
        requested: Optional[dict[str, Any]] = None,
        achieved: Optional[dict[str, Any]] = None,
        verification: Optional[VerificationStatus] = None,
        error: Optional[ErrorInfo] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> StressRecord:
        return StressRecord(
            step=step,
            outcome=outcome,
            provider=self.PROVIDER,
            stress_axis_digest=self._axis,
            stress_level=self._level,
            requested=requested,
            achieved=achieved,
            independent_evidence=None,
            verification=verification,
            error=error,
            details=details or {},
        )

    def _require(self, step: str, allowed: set[str]) -> None:
        if self._state not in allowed:
            raise LifecycleError(f"stress {step}() not allowed in state {self._state!r}")

    @staticmethod
    def _ok(record: StressRecord) -> bool:
        return record.outcome is Outcome.OK

    # -- hooks implemented by providers -----------------------------------
    def _do_prepare(self) -> StressRecord: raise NotImplementedError
    def _do_activate(self) -> StressRecord: raise NotImplementedError
    def _do_verify(self, window_ns: int) -> StressRecord: raise NotImplementedError
    def _do_observe(self) -> StressRecord: raise NotImplementedError
    def _do_deactivate(self) -> StressRecord: raise NotImplementedError
    def _do_restore(self) -> StressRecord: raise NotImplementedError

    # -- protocol ----------------------------------------------------------
    def prepare(self) -> StressRecord:
        self._require("prepare", {"created"})
        record = self._do_prepare()
        self._state = "prepared" if self._ok(record) else "prepare_failed"
        return record

    def activate(self) -> StressRecord:
        self._require("activate", {"prepared"})
        self._activate_attempted = True
        record = self._do_activate()
        self._state = "active" if self._ok(record) else "activate_failed"
        return record

    def verify(self, window_ns: int) -> StressRecord:
        self._require("verify", {"active"})
        if type(window_ns) is not int or window_ns < 0:
            raise ValueError("window_ns must be an integer >= 0")
        record = self._do_verify(window_ns)
        self._state = "verified" if self._ok(record) else "verify_failed"
        return record

    def observe(self) -> StressRecord:
        self._require("observe", {"verified"})
        return self._do_observe()

    def deactivate(self) -> StressRecord:
        if not self._activate_attempted or self._state in {
            "deactivated", "deactivate_failed", "restored", "restore_failed"
        }:
            raise LifecycleError(f"stress deactivate() not allowed in state {self._state!r}")
        record = self._do_deactivate()
        self._state = "deactivated" if self._ok(record) else "deactivate_failed"
        return record

    def restore(self) -> StressRecord:
        if self._state == "restored":
            return self._record(StressStep.RESTORE, Outcome.OK, details={"repeat": True})
        record = self._do_restore()
        self._state = "restored" if self._ok(record) else "restore_failed"
        return record
