"""Enumerations and status vocabularies used by specs, records and evidence.

All values are lowercase strings so they serialize unchanged into JSON.
Python 3.10 has no ``enum.StrEnum``; ``StrEnum`` below is the equivalent mixin.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class InputKind(StrEnum):
    DATASET = "dataset"
    DRIVING_SCENARIO = "driving_scenario"


class ReplayMode(StrEnum):
    RECORDED = "recorded"
    FIXED_RATE = "fixed_rate"
    ASAP = "asap"


class TimingSemantic(StrEnum):
    CAPTURE = "capture"
    LOGGING = "logging"
    UNKNOWN = "unknown"


class TimeUnit(StrEnum):
    NS = "ns"
    US = "us"
    MS = "ms"
    S = "s"


UNIT_TO_NS: dict[TimeUnit, int] = {
    TimeUnit.NS: 1,
    TimeUnit.US: 1_000,
    TimeUnit.MS: 1_000_000,
    TimeUnit.S: 1_000_000_000,
}


class RequestState(StrEnum):
    PLANNED = "planned"
    ISSUED = "issued"
    INFLIGHT = "inflight"
    COMPLETED = "completed"
    ERROR = "error"
    TIMED_OUT = "timed_out"
    DROPPED = "dropped"
    NOT_ISSUED = "not_issued"


TERMINAL_REQUEST_STATES = frozenset(
    {
        RequestState.COMPLETED,
        RequestState.ERROR,
        RequestState.TIMED_OUT,
        RequestState.DROPPED,
        RequestState.NOT_ISSUED,
    }
)


class ExecutionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class MeasurementStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


class MetricStatus(StrEnum):
    VALID = "valid"
    EMPTY = "empty"
    CENSORED = "censored"
    INVALID = "invalid"
    NOT_COLLECTED = "not_collected"
    NOT_SUPPORTED = "not_supported"
    NOT_APPLICABLE = "not_applicable"


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    SELF_REPORTED = "self_reported"
    UNVERIFIED = "unverified"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class Coverage(StrEnum):
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"
    NOT_APPLICABLE = "not_applicable"


class NominalCheck(StrEnum):
    PASSED = "passed"
    NOT_PERFORMED = "not_performed"
    FAILED = "failed"


class ExecutionKind(StrEnum):
    MOCK = "mock"
    ACTUAL = "actual"


# Stage 1A ExperimentPlan has no evaluation_level field; the Model-level runner
# records this constant in every manifest.
BENCHMARK_LEVEL = "model"


class RunClockDomain(StrEnum):
    VIRTUAL = "virtual"
    MONOTONIC = "monotonic"  # Stage 1R: host CLOCK_MONOTONIC run clock


class CompletionModel(StrEnum):
    BLOCKING = "blocking"
    POLLING = "polling"  # Stage 1R: non-blocking submit, completion observed by polling


class WaitPolicy(StrEnum):
    """How the realtime issuer waits for t_sched and for completion."""

    SPIN = "spin"


class InputPipelineMode(StrEnum):
    """Explicitly selected input-preparation path (never auto-selected)."""

    WORKLOAD = "workload"  # payload -> Workload.preprocess -> DUTAdapter.prepare_input
    ADAPTER_OVERRIDE = "adapter_override"  # payload -> AdapterInputOverride.prepare_from_payload


class InputPipelineOwner(StrEnum):
    WORKLOAD = "workload"
    DUT_ADAPTER = "dut_adapter"


# Steps of the Workload input path an adapter override may fuse/replace.
FUSABLE_INPUT_STEPS = ("workload.preprocess", "dut.prepare_input")


class EquivalenceStatus(StrEnum):
    """Equivalence of an adapter override with the Workload reference path."""

    NOT_APPLICABLE = "not_applicable"  # workload mode: the reference path itself is used
    NOT_PERFORMED = "not_performed"
    PASSED = "passed"
    FAILED = "failed"


class StressStep(StrEnum):
    CHECK = "check"
    PREPARE = "prepare"
    ACTIVATE = "activate"
    VERIFY = "verify"
    OBSERVE = "observe"
    DEACTIVATE = "deactivate"
    RESTORE = "restore"


class Outcome(StrEnum):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"


class OnFull(StrEnum):
    DROP_NEW = "drop_new"


class OnRequestError(StrEnum):
    CONTINUE = "continue"


class OutputCapture(StrEnum):
    TASK_OUTPUT_ALL = "task_output_all"


class DeadlinePolicy(StrEnum):
    FRAME_PERIOD = "frame_period"


class ReferenceKind(StrEnum):
    NONE = "none"
    EXTERNAL_GOLDEN = "external_golden"


class StressSchedule(StrEnum):
    WHOLE_MEASUREMENT = "whole_measurement"


class OnApplyFailure(StrEnum):
    ABORT = "abort"


# Reserved stress level name for nominal (no intentional intervention) on an axis.
NOMINAL_LEVEL = "none"


# --------------------------------------------------------------------------
# Contrast validity (factor isolation): Design / Realization / Measurement guards
# --------------------------------------------------------------------------


class ContrastType(StrEnum):
    STRESS = "stress"  # only stress_level (and its value) may vary
    SCENARIO = "scenario"  # only the scenario selection (and its realized items/timing) may vary
    INTERACTION = "interaction"  # selection and stress level may vary (one pair of a 2x2)


class GuardStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_OBSERVABLE = "not_observable"  # the evidence cannot exist for this execution (e.g. mock)
    NOT_APPLICABLE = "not_applicable"  # the guard/contrast does not apply
    UNRESOLVED = "unresolved"  # evidence is partial; neither equivalence nor difference is established


class ContrastValidity(StrEnum):
    VALID = "valid"  # every layer passed: an attributed contrast is justified
    INVALID = "invalid"  # at least one layer failed
    NOT_ESTABLISHED = "not_established"  # no failure, but not every layer passed
    NOT_APPLICABLE = "not_applicable"


class RealizedCondition(StrEnum):
    """Generic realized execution conditions (not vendor-specific)."""

    INPUT_PAYLOAD = "input_payload"
    INPUT_PREPARATION = "input_preparation"
    INPUT_RESIDENCY = "input_residency"
    BUFFER_STATE = "buffer_state"
    DATA_MOVEMENT = "data_movement"  # H2D / DMA behavior
    RUNTIME_QUEUE_STATE = "runtime_queue_state"
    DEVICE_FREQUENCY = "device_frequency"
    THERMAL_STATE = "thermal_state"  # temperature / throttling
    COMPETING_DEVICE_USERS = "competing_device_users"
    ISSUE_TIMING = "issue_timing"  # realized replay issue timing
    MEMORY_PRESSURE_BANDWIDTH = "memory_pressure_bandwidth"
    STRESS_REALIZATION = "stress_realization"


class EvidenceValidity(StrEnum):
    OBSERVED = "observed"  # measured from the real artifact or system
    SELF_REPORTED = "self_reported"  # reported by the component itself, not independently checked
    VIRTUAL = "virtual"  # produced by a mock/virtual model; says nothing about real hardware
    NOT_OBSERVABLE = "not_observable"
    NOT_APPLICABLE = "not_applicable"
