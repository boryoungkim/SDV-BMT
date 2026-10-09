"""Virtual FakeStressProvider (``kind: fake_virtual``).

No OS process, background workload, utilization or hardware evidence exists:
every record is marked ``virtual: true`` and verification can at best be
``self_reported`` (``unverified`` when the spec demands independent evidence,
which a virtual stressor cannot supply).

``provider_options``:

* ``fail_phase``: one of prepare/activate/verify/deactivate/restore — the
  provider returns a failed record at that step (failure injection).
* ``achieved_fraction`` (default 1.0): virtual self-reported achieved level as
  a fraction of the requested level; verify fails when it is below
  ``verification.achieved_min_fraction``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Optional

from ..contracts.records import ErrorInfo, HostInfo, StressRecord, SupportDecision
from ..contracts.specs import HardwareStressSpec, parse_strict, stress_level_value
from ..contracts.status import NOMINAL_LEVEL, Outcome, StressStep, VerificationStatus
from .lifecycle import StressProviderBase

FAILABLE_PHASES = ("prepare", "activate", "verify", "deactivate", "restore")


@dataclass(frozen=True)
class FakeStressOptions:
    fail_phase: Optional[str] = None
    achieved_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.fail_phase is not None and self.fail_phase not in FAILABLE_PHASES:
            raise ValueError(f"fail_phase must be one of {FAILABLE_PHASES}")
        if self.achieved_fraction < 0:
            raise ValueError("achieved_fraction must be >= 0")


def _exact(value: float) -> Fraction:
    return Fraction(repr(float(value)))


class FakeStressProvider(StressProviderBase):
    PROVIDER = "fake_virtual"
    KIND = "fake_virtual"

    def __init__(self, spec: HardwareStressSpec, level: str) -> None:
        if spec.kind != self.KIND:
            raise ValueError(f"FakeStressProvider requires kind {self.KIND!r}")
        if level == NOMINAL_LEVEL:
            raise ValueError("level 'none' is served by NoStressProvider")
        super().__init__(spec, level)
        self._options = parse_strict(FakeStressOptions, spec.provider_options, "stress.provider_options")
        value = stress_level_value(spec, level)
        assert value is not None
        self._requested = {
            "variable": spec.control.variable,
            "unit": spec.control.unit,
            "level": level,
            "value": value,
        }

    def check(self, spec: HardwareStressSpec, level: str, host: HostInfo) -> SupportDecision:
        if spec.kind != self.KIND:
            return SupportDecision(False, f"kind {spec.kind!r} is not {self.KIND!r}")
        if level not in spec.levels:
            return SupportDecision(False, f"level {level!r} is not defined by the spec")
        if host.host_kind != "virtual":
            return SupportDecision(False, "FakeStressProvider runs on the virtual host only")
        return SupportDecision(True, "virtual fake stressor (no real resource is loaded)")

    def _failure(self, step: StressStep, **kw: Any) -> StressRecord:
        return self._record(
            step,
            Outcome.FAILED,
            error=ErrorInfo("fake_injected_failure", f"injected {step.value} failure"),
            details={"virtual": True, "injected": True},
            **kw,
        )

    def _do_prepare(self) -> StressRecord:
        if self._options.fail_phase == "prepare":
            return self._failure(StressStep.PREPARE)
        return self._record(StressStep.PREPARE, Outcome.OK, details={"virtual": True})

    def _do_activate(self) -> StressRecord:
        if self._options.fail_phase == "activate":
            return self._failure(StressStep.ACTIVATE, requested=self._requested)
        return self._record(
            StressStep.ACTIVATE, Outcome.OK, requested=self._requested, details={"virtual": True}
        )

    def _achieved(self) -> dict[str, Any]:
        value = float(_exact(self._requested["value"]) * _exact(self._options.achieved_fraction))
        return {"value": value, "source": "self_report", "virtual": True}

    def _do_verify(self, window_ns: int) -> StressRecord:
        achieved = self._achieved()
        details = {"window_ns": window_ns, "virtual": True}
        if self._options.fail_phase == "verify":
            return self._failure(
                StressStep.VERIFY,
                requested=self._requested,
                achieved=achieved,
                verification=VerificationStatus.FAILED,
            )
        minimum = _exact(self._spec.verification.achieved_min_fraction) * _exact(self._requested["value"])
        if _exact(achieved["value"]) < minimum:
            return self._record(
                StressStep.VERIFY,
                Outcome.FAILED,
                requested=self._requested,
                achieved=achieved,
                verification=VerificationStatus.FAILED,
                error=ErrorInfo("achieved_below_requested", "achieved level below the verification minimum"),
                details=details,
            )
        status = (
            VerificationStatus.UNVERIFIED
            if self._spec.verification.independent is not None
            else VerificationStatus.SELF_REPORTED
        )
        return self._record(
            StressStep.VERIFY,
            Outcome.OK,
            requested=self._requested,
            achieved=achieved,
            verification=status,
            details=details,
        )

    def _do_observe(self) -> StressRecord:
        return self._record(
            StressStep.OBSERVE, Outcome.OK, achieved=self._achieved(), details={"alive": True, "virtual": True}
        )

    def _do_deactivate(self) -> StressRecord:
        if self._options.fail_phase == "deactivate":
            return self._failure(StressStep.DEACTIVATE)
        return self._record(StressStep.DEACTIVATE, Outcome.OK, details={"virtual": True})

    def _do_restore(self) -> StressRecord:
        if self._options.fail_phase == "restore":
            return self._failure(StressStep.RESTORE)
        return self._record(StressStep.RESTORE, Outcome.OK, details={"virtual": True})
