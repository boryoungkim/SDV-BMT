"""NoStressProvider: nominal = a specific stress axis at ``stress_level=none``.

Records "no intentional intervention" through the same lifecycle as a real
provider.  ``verify`` only waits the axis verification window (the runner
advances the virtual clock).  Whether the device was actually idle is a
separate ``nominal_check``, which Stage 1A does not perform.
"""

from __future__ import annotations

from ..contracts.records import HostInfo, StressRecord, SupportDecision
from ..contracts.specs import HardwareStressSpec
from ..contracts.status import NOMINAL_LEVEL, Outcome, StressStep, VerificationStatus
from .lifecycle import StressProviderBase


class NoStressProvider(StressProviderBase):
    PROVIDER = "none"

    def __init__(self, spec: HardwareStressSpec, level: str = NOMINAL_LEVEL) -> None:
        if level != NOMINAL_LEVEL:
            raise ValueError(f"NoStressProvider only serves level {NOMINAL_LEVEL!r}")
        super().__init__(spec, level)

    def check(self, spec: HardwareStressSpec, level: str, host: HostInfo) -> SupportDecision:
        if level != NOMINAL_LEVEL:
            return SupportDecision(False, f"NoStressProvider serves only level {NOMINAL_LEVEL!r}")
        return SupportDecision(True, "nominal: no intentional intervention")

    def _do_prepare(self) -> StressRecord:
        return self._record(StressStep.PREPARE, Outcome.OK, details={"intervention": "none"})

    def _do_activate(self) -> StressRecord:
        return self._record(StressStep.ACTIVATE, Outcome.OK, details={"intervention": "none"})

    def _do_verify(self, window_ns: int) -> StressRecord:
        return self._record(
            StressStep.VERIFY,
            Outcome.OK,
            verification=VerificationStatus.NOT_APPLICABLE,
            details={"window_ns": window_ns, "action": "wait_only"},
        )

    def _do_observe(self) -> StressRecord:
        return self._record(StressStep.OBSERVE, Outcome.OK, details={"intervention": "none"})

    def _do_deactivate(self) -> StressRecord:
        return self._record(StressStep.DEACTIVATE, Outcome.OK, details={"intervention": "none"})

    def _do_restore(self) -> StressRecord:
        return self._record(StressStep.RESTORE, Outcome.OK, details={"intervention": "none"})
