"""Definition contracts (plan/spec dataclasses) and the strict mapping loader.

``parse_strict`` builds a frozen dataclass from a parsed mapping and rejects
unknown fields, missing required fields, ``null`` where not allowed, and wrong
types (``bool`` is never accepted as a number).  YAML parsing itself lives in
``sdv_bmt.runner.plan``; this module is standard-library only.

Identity digests defined here are content-based: user labels such as
``input_id``, ``stress_id``, ``workload_id`` and ``dut_config_id`` are recorded
for traceability but do not participate in the digests.
"""

from __future__ import annotations

import dataclasses
import math
import re
import types
import typing
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from .ids import digest_of, is_sha256_hex
from .status import (
    FUSABLE_INPUT_STEPS,
    NOMINAL_LEVEL,
    DeadlinePolicy,
    ExecutionKind,
    InputKind,
    InputPipelineMode,
    InputPipelineOwner,
    OnApplyFailure,
    OnFull,
    OnRequestError,
    OutputCapture,
    ReferenceKind,
    ReplayMode,
    RunClockDomain,
    StressSchedule,
    TimeUnit,
    TimingSemantic,
    WaitPolicy,
)

SCHEMA_VERSION = 1
# MeasurementBoundary v2 adds the effective input pipeline (owner, semantics,
# fused steps, pipeline digest).
BOUNDARY_SCHEMA_VERSION = 2

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Keys with a fixed meaning in a fixture source index row. A declared timing
# source field may not reuse them.
INDEX_ROW_RESERVED_KEYS = frozenset(
    {
        "source_item_id",
        "payload_text",
        "annotation",
        "metadata",
        "episode_id",
        "stream_id",
        "sensor_id",
        "frame_id",
    }
)


class SpecError(ValueError):
    """A configuration mapping violates its strict schema."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


# --------------------------------------------------------------------------
# strict mapping -> dataclass conversion
# --------------------------------------------------------------------------


def _type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, Mapping):
        return "mapping"
    if isinstance(value, list):
        return "list"
    return type(value).__name__


def parse_strict(cls: type, data: object, path: str = "$") -> Any:
    """Build dataclass ``cls`` from ``data`` with strict field and type checks."""
    if not (dataclasses.is_dataclass(cls) and isinstance(cls, type)):
        raise TypeError(f"{cls!r} is not a dataclass type")
    if not isinstance(data, Mapping):
        raise SpecError(path, f"expected a mapping, got {_type_name(data)}")
    for key in data:
        if not isinstance(key, str):
            raise SpecError(path, f"non-string key {key!r}")
    hints = typing.get_type_hints(cls)
    fields_by_name = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = sorted(k for k in data if k not in fields_by_name)
    if unknown:
        raise SpecError(path, "unknown field(s): " + ", ".join(unknown))
    kwargs: dict[str, Any] = {}
    for name, fdef in fields_by_name.items():
        sub = f"{path}.{name}"
        if name in data:
            kwargs[name] = _convert(hints[name], data[name], sub)
        elif fdef.default is MISSING and fdef.default_factory is MISSING:
            raise SpecError(sub, "required field is missing")
    try:
        return cls(**kwargs)
    except SpecError:
        raise
    except ValueError as exc:
        raise SpecError(path, str(exc)) from None


def _is_union(origin: object) -> bool:
    return origin is Union or origin is getattr(types, "UnionType", None)


def _convert(tp: Any, value: object, path: str) -> Any:
    if tp is Any:
        return plain_json(value, path)
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if _is_union(origin):
        if value is None:
            if type(None) in args:
                return None
            raise SpecError(path, "null is not allowed")
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) != 1:
            raise TypeError(f"unsupported union annotation {tp!r}")
        return _convert(non_none[0], value, path)
    if value is None:
        raise SpecError(path, "null is not allowed")
    if origin is tuple:
        if not (len(args) == 2 and args[1] is Ellipsis):
            raise TypeError(f"unsupported tuple annotation {tp!r}")
        if not isinstance(value, list):
            raise SpecError(path, f"expected a list, got {_type_name(value)}")
        return tuple(_convert(args[0], v, f"{path}[{i}]") for i, v in enumerate(value))
    if origin is dict:
        if not isinstance(value, Mapping):
            raise SpecError(path, f"expected a mapping, got {_type_name(value)}")
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SpecError(path, f"non-string key {key!r}")
            out[key] = _convert(args[1], item, f"{path}.{key}")
        return out
    if isinstance(tp, type) and issubclass(tp, Enum):
        allowed = [m.value for m in tp]
        if not isinstance(value, str) or value not in allowed:
            raise SpecError(path, f"expected one of {allowed}, got {value!r}")
        return tp(value)
    if tp is bool:
        if type(value) is not bool:
            raise SpecError(path, f"expected a boolean, got {_type_name(value)}")
        return value
    if tp is int:
        if type(value) is not int:
            raise SpecError(path, f"expected an integer, got {_type_name(value)}")
        return value
    if tp is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise SpecError(path, f"expected a finite number, got {_type_name(value)}")
        return float(value)
    if tp is str:
        if type(value) is not str:
            raise SpecError(path, f"expected a string, got {_type_name(value)}")
        return value
    if dataclasses.is_dataclass(tp):
        return parse_strict(tp, value, path)
    raise TypeError(f"unsupported annotation {tp!r} at {path}")


def plain_json(value: object, path: str = "$") -> Any:
    """Validate and deep-copy a free-form JSON-compatible value."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise SpecError(path, "non-finite number")
        return value
    if isinstance(value, list):
        return [plain_json(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SpecError(path, f"non-string key {key!r}")
            out[key] = plain_json(item, f"{path}.{key}")
        return out
    raise SpecError(path, f"unsupported value of type {_type_name(value)}")


# --------------------------------------------------------------------------
# field validation helpers (raise ValueError; parse_strict adds the path)
# --------------------------------------------------------------------------


def _check_schema_version(value: int) -> None:
    if value != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}, got {value}")


def check_identifier(value: str, what: str) -> None:
    if not _ID_RE.match(value):
        raise ValueError(f"{what} {value!r} must match {_ID_RE.pattern}")


def _check_sha256(value: str, what: str) -> None:
    if not is_sha256_hex(value):
        raise ValueError(f"{what} must be a lowercase 64-hex SHA-256 digest")


def _check_relative_path(value: str, what: str) -> None:
    if not value or value.startswith("/") or "\\" in value:
        raise ValueError(f"{what} must be a non-empty relative POSIX path, got {value!r}")


def _check_nonempty(value: str, what: str) -> None:
    if not value.strip():
        raise ValueError(f"{what} must be a non-empty string")


def string_tuple(value: object, what: str, allowed: Optional[tuple[str, ...]] = None,
                 nonempty: bool = False) -> tuple[str, ...]:
    """Validate a sequence of unique non-empty strings and return it as a tuple.
    A bare string is rejected (it is not a one-element sequence)."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise ValueError(f"{what} must be a list/tuple of strings, got {type(value).__name__}")
    items = tuple(value)
    if not all(isinstance(v, str) and v.strip() for v in items):
        raise ValueError(f"{what} must contain non-empty strings")
    if len(set(items)) != len(items):
        raise ValueError(f"{what} must not contain duplicates")
    if nonempty and not items:
        raise ValueError(f"{what} must not be empty")
    if allowed is not None:
        unknown = [v for v in items if v not in allowed]
        if unknown:
            raise ValueError(f"{what} contains unknown entries {unknown}; allowed: {list(allowed)}")
    return items


# --------------------------------------------------------------------------
# Input specs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingSpec:
    """Declared meaning of the timestamp field of a driving-scenario source."""

    source_field: str
    unit: TimeUnit
    semantic: TimingSemantic
    clock_source: str

    def __post_init__(self) -> None:
        _check_nonempty(self.source_field, "timing.source_field")
        if self.source_field in INDEX_ROW_RESERVED_KEYS:
            raise ValueError(f"timing.source_field {self.source_field!r} is a reserved index key")
        _check_nonempty(self.clock_source, "timing.clock_source")


@dataclass(frozen=True)
class InputFamilySpec:
    """What makes inputs comparable: source, schema, timing meaning, task contract."""

    source_dataset: str
    source_version: str
    task: str
    annotation_schema: Optional[str]
    preprocessing_contract: str
    streams: Optional[tuple[str, ...]] = None
    timing: Optional[TimingSpec] = None

    def __post_init__(self) -> None:
        for name in ("source_dataset", "source_version", "task", "preprocessing_contract"):
            _check_nonempty(getattr(self, name), f"family.{name}")
        if self.streams is not None:
            if not self.streams:
                raise ValueError("family.streams must not be empty")
            if len(set(self.streams)) != len(self.streams):
                raise ValueError("family.streams must not contain duplicates")


@dataclass(frozen=True)
class SourceIndexSpec:
    index: str
    index_sha256: str

    def __post_init__(self) -> None:
        _check_relative_path(self.index, "source.index")
        _check_sha256(self.index_sha256, "source.index_sha256")


@dataclass(frozen=True)
class DatasetSelectionSpec:
    start: int = 0
    count: Optional[int] = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("selection.start must be >= 0")
        if self.count is not None and self.count < 1:
            raise ValueError("selection.count must be >= 1 when given")


@dataclass(frozen=True)
class ScenarioSelectionSpec:
    stream_id: Optional[str] = None
    episodes: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.episodes is not None:
            if not self.episodes:
                raise ValueError("selection.episodes must not be empty when given")
            if len(set(self.episodes)) != len(self.episodes):
                raise ValueError("selection.episodes must not contain duplicates")


@dataclass(frozen=True)
class DatasetSpec:
    schema_version: int
    input_kind: InputKind
    input_id: str
    version: str
    family: InputFamilySpec
    source: SourceIndexSpec
    selection: DatasetSelectionSpec = field(default_factory=DatasetSelectionSpec)

    def __post_init__(self) -> None:
        _check_schema_version(self.schema_version)
        if self.input_kind is not InputKind.DATASET:
            raise ValueError("DatasetSpec requires input_kind=dataset")
        check_identifier(self.input_id, "input_id")
        _check_nonempty(self.version, "version")
        # Dataset input has no scheduling timestamps and no stream semantics.
        if self.family.timing is not None:
            raise ValueError("dataset family must not declare timing")
        if self.family.streams is not None:
            raise ValueError("dataset family must not declare streams")


@dataclass(frozen=True)
class DrivingScenarioSpec:
    schema_version: int
    input_kind: InputKind
    input_id: str
    version: str
    family: InputFamilySpec
    source: SourceIndexSpec
    selection: ScenarioSelectionSpec = field(default_factory=ScenarioSelectionSpec)

    def __post_init__(self) -> None:
        _check_schema_version(self.schema_version)
        if self.input_kind is not InputKind.DRIVING_SCENARIO:
            raise ValueError("DrivingScenarioSpec requires input_kind=driving_scenario")
        check_identifier(self.input_id, "input_id")
        _check_nonempty(self.version, "version")
        if self.family.streams is None:
            raise ValueError("driving_scenario family must declare streams")
        if self.selection.stream_id is not None and self.selection.stream_id not in self.family.streams:
            raise ValueError(
                f"selection.stream_id {self.selection.stream_id!r} is not in family.streams"
            )


InputSpec = Union[DatasetSpec, DrivingScenarioSpec]


def parse_input_spec(data: object, path: str = "$") -> InputSpec:
    if not isinstance(data, Mapping):
        raise SpecError(path, f"expected a mapping, got {_type_name(data)}")
    kind = data.get("input_kind")
    if kind == InputKind.DATASET.value:
        return parse_strict(DatasetSpec, data, path)
    if kind == InputKind.DRIVING_SCENARIO.value:
        return parse_strict(DrivingScenarioSpec, data, path)
    raise SpecError(
        f"{path}.input_kind",
        f"expected one of {[k.value for k in InputKind]}, got {kind!r}",
    )


def input_family_digest(spec: InputSpec) -> str:
    """Comparable input family (design v03 F.6): source dataset/version, stream
    schema, timing semantics, task/annotation schema, preprocessing contract."""
    return digest_of(
        {
            "schema_version": spec.schema_version,
            "input_kind": spec.input_kind,
            "family": spec.family,
        }
    )


def input_selection_digest(spec: InputSpec, resolved_selection: Mapping[str, Any]) -> str:
    """Digest of the selection as resolved against the digest-pinned source index
    (two spellings that select the same items get the same digest)."""
    return digest_of(
        {
            "input_family_digest": input_family_digest(spec),
            "source_index_sha256": spec.source.index_sha256,
            "selection": dict(resolved_selection),
        }
    )


# --------------------------------------------------------------------------
# Replay / workload / DUT / stress / boundary / run / evaluation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayPolicy:
    mode: ReplayMode
    rate_hz: Optional[float] = None
    speed: Optional[float] = None
    lead_ns: int = 0
    episode_gap_ns: Optional[int] = None

    def __post_init__(self) -> None:
        if self.lead_ns < 0:
            raise ValueError("replay.lead_ns must be >= 0")
        if self.mode is ReplayMode.FIXED_RATE:
            if self.rate_hz is None or self.rate_hz <= 0:
                raise ValueError("fixed_rate replay requires rate_hz > 0")
            if self.speed is not None:
                raise ValueError("replay.speed applies to recorded replay only")
            if self.episode_gap_ns is not None:
                raise ValueError("replay.episode_gap_ns applies to recorded replay only")
        elif self.mode is ReplayMode.RECORDED:
            if self.rate_hz is not None:
                raise ValueError("replay.rate_hz applies to fixed_rate replay only")
            if self.speed is None:
                object.__setattr__(self, "speed", 1.0)
            elif self.speed <= 0:
                raise ValueError("replay.speed must be > 0")
            if self.episode_gap_ns is not None and self.episode_gap_ns < 0:
                raise ValueError("replay.episode_gap_ns must be >= 0")
        else:
            if self.rate_hz is not None or self.speed is not None or self.episode_gap_ns is not None:
                raise ValueError("asap replay takes no rate_hz, speed or episode_gap_ns")


@dataclass(frozen=True)
class ComponentIdentity:
    id: str
    version: str

    def __post_init__(self) -> None:
        _check_nonempty(self.id, "component id")
        _check_nonempty(self.version, "component version")


@dataclass(frozen=True)
class WorkloadSpec:
    schema_version: int
    workload_id: str
    implementation: str
    task: str
    artifact_sha256: Optional[str]
    precision: str
    batch: int
    preprocess: ComponentIdentity
    postprocess: ComponentIdentity
    codec: ComponentIdentity
    output_ready: str
    quality_evaluator: Optional[str] = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_schema_version(self.schema_version)
        check_identifier(self.workload_id, "workload_id")
        _check_nonempty(self.implementation, "implementation")
        _check_nonempty(self.output_ready, "output_ready")
        if self.batch < 1:
            raise ValueError("batch must be >= 1")
        if self.artifact_sha256 is not None:
            _check_sha256(self.artifact_sha256, "artifact_sha256")


def workload_identity_digest(spec: WorkloadSpec) -> str:
    content = dataclasses.asdict(spec)
    content.pop("workload_id")
    return digest_of(content)


@dataclass(frozen=True)
class DUTConfig:
    schema_version: int
    dut_config_id: str
    adapter: str
    device: str
    max_inflight: int
    runtime_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_schema_version(self.schema_version)
        check_identifier(self.dut_config_id, "dut_config_id")
        _check_nonempty(self.adapter, "adapter")
        _check_nonempty(self.device, "device")
        if self.max_inflight < 1:
            raise ValueError("max_inflight must be >= 1")


def dut_config_identity_digest(config: DUTConfig) -> str:
    content = dataclasses.asdict(config)
    content.pop("dut_config_id")
    return digest_of(content)


@dataclass(frozen=True)
class StressControlSpec:
    variable: str
    unit: str


@dataclass(frozen=True)
class IndependentEvidenceSpec:
    source: str
    rule: str


@dataclass(frozen=True)
class StressVerificationSpec:
    window_ns: int
    achieved_min_fraction: float
    independent: Optional[IndependentEvidenceSpec] = None

    def __post_init__(self) -> None:
        if self.window_ns < 0:
            raise ValueError("verification.window_ns must be >= 0")
        if not 0.0 < self.achieved_min_fraction <= 1.0:
            raise ValueError("verification.achieved_min_fraction must be in (0, 1]")


@dataclass(frozen=True)
class HardwareStressSpec:
    schema_version: int
    stress_id: str
    kind: str
    target_resource: str
    placement: str
    control: StressControlSpec
    background_workload: Optional[dict[str, Any]]
    levels: dict[str, float]
    verification: StressVerificationSpec
    schedule: StressSchedule
    on_apply_failure: OnApplyFailure
    provider_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_schema_version(self.schema_version)
        check_identifier(self.stress_id, "stress_id")
        _check_nonempty(self.kind, "kind")
        for name, value in self.levels.items():
            check_identifier(name, "stress level name")
            if name == NOMINAL_LEVEL:
                raise ValueError(f"level name {NOMINAL_LEVEL!r} is reserved for nominal")
            if value < 0:
                raise ValueError(f"level {name!r} value must be >= 0")


def stress_axis_digest(spec: HardwareStressSpec) -> str:
    """Digest of the stress spec without its levels (and without the label)."""
    content = dataclasses.asdict(spec)
    content.pop("stress_id")
    content.pop("levels")
    return digest_of(content)


def stress_level_value(spec: HardwareStressSpec, level: str) -> Optional[float]:
    if level == NOMINAL_LEVEL:
        return None
    if level not in spec.levels:
        raise ValueError(f"stress level {level!r} is not defined by stress spec {spec.stress_id!r}")
    return spec.levels[level]


_INCLUSION = frozenset({"included", "excluded", "inseparable"})


@dataclass(frozen=True)
class MeasurementBoundary:
    """Inclusion/exclusion of each step and the meaning of each timestamp."""

    schema_version: int
    clock_domain: RunClockDomain
    payload_read_decode: str
    preprocess: str
    prepare_input_h2d: str
    submit: str
    complete: str
    postprocess: str
    output_ready: str
    host_step_virtual_cost_ns: int
    latency_definitions: dict[str, str]
    input_pipeline_owner: InputPipelineOwner
    preprocess_semantics_id: str
    fused_steps: tuple[str, ...]
    input_pipeline_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != BOUNDARY_SCHEMA_VERSION:
            raise ValueError(f"measurement boundary schema_version must be {BOUNDARY_SCHEMA_VERSION}")
        object.__setattr__(self, "fused_steps",
                           string_tuple(self.fused_steps, "fused_steps", FUSABLE_INPUT_STEPS))
        _check_sha256(self.input_pipeline_digest, "input_pipeline_digest")
        for name in ("payload_read_decode", "preprocess", "prepare_input_h2d"):
            if getattr(self, name) not in _INCLUSION:
                raise ValueError(f"{name} must be one of {sorted(_INCLUSION)}")
        _check_nonempty(self.preprocess_semantics_id, "preprocess_semantics_id")
        if self.input_pipeline_owner is InputPipelineOwner.WORKLOAD and self.fused_steps:
            raise ValueError("a workload-owned input pipeline has no fused steps")
        if self.postprocess not in ("excluded", "output_ready_only"):
            raise ValueError("postprocess must be 'excluded' or 'output_ready_only'")
        if self.host_step_virtual_cost_ns < 0:
            raise ValueError("host_step_virtual_cost_ns must be >= 0")


REALTIME_BOUNDARY_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class RealtimeMeasurementBoundary:
    """Stage 1R boundary on the host monotonic clock (schema v3).

    A separate record so that the Stage 1A virtual boundary (v2) and its
    identity stay unchanged.  ``device_timing`` and ``instrumentation`` are
    declared by the DUT adapter: device-domain durations are never subtracted
    from host timestamps, and runs with different instrumentation never share a
    boundary identity.
    """

    schema_version: int
    clock_domain: RunClockDomain
    payload_read_decode: str
    preprocess: str
    prepare_input_h2d: str
    submit: str
    complete: str
    completion_observation: str
    postprocess: str
    output_ready: str
    latency_definitions: dict[str, str]
    device_timing: Optional[dict[str, Any]]
    instrumentation: dict[str, Any]
    input_pipeline_owner: InputPipelineOwner
    preprocess_semantics_id: str
    fused_steps: tuple[str, ...]
    input_pipeline_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != REALTIME_BOUNDARY_SCHEMA_VERSION:
            raise ValueError(f"realtime boundary schema_version must be {REALTIME_BOUNDARY_SCHEMA_VERSION}")
        if self.clock_domain is not RunClockDomain.MONOTONIC:
            raise ValueError("a realtime boundary uses the monotonic run clock")
        object.__setattr__(self, "fused_steps",
                           string_tuple(self.fused_steps, "fused_steps", FUSABLE_INPUT_STEPS))
        _check_sha256(self.input_pipeline_digest, "input_pipeline_digest")
        for name in ("payload_read_decode", "preprocess", "prepare_input_h2d"):
            if getattr(self, name) not in _INCLUSION:
                raise ValueError(f"{name} must be one of {sorted(_INCLUSION)}")
        for name in ("submit", "complete", "completion_observation", "output_ready", "preprocess_semantics_id"):
            _check_nonempty(getattr(self, name), name)
        if self.postprocess not in ("excluded", "output_ready_only"):
            raise ValueError("postprocess must be 'excluded' or 'output_ready_only'")


def measurement_boundary_id(boundary: MeasurementBoundary | RealtimeMeasurementBoundary) -> str:
    return digest_of(boundary)


@dataclass(frozen=True)
class RunConfig:
    max_inflight: int
    queue_capacity: int
    on_full: OnFull
    request_timeout_ns: int
    drain_ns: int
    output_capture: OutputCapture = OutputCapture.TASK_OUTPUT_ALL
    on_request_error: OnRequestError = OnRequestError.CONTINUE

    def __post_init__(self) -> None:
        if self.max_inflight < 1:
            raise ValueError("run.max_inflight must be >= 1")
        if self.queue_capacity < 0:
            raise ValueError("run.queue_capacity must be >= 0")
        if self.request_timeout_ns <= 0:
            raise ValueError("run.request_timeout_ns must be > 0")
        if self.drain_ns < 0:
            raise ValueError("run.drain_ns must be >= 0")


@dataclass(frozen=True)
class DeadlineSpec:
    """Evaluation parameter: deadline_i = t_sched_i + period_ns."""

    policy: DeadlinePolicy
    period_ns: int

    def __post_init__(self) -> None:
        if self.period_ns <= 0:
            raise ValueError("deadline.period_ns must be > 0")


@dataclass(frozen=True)
class OutputReferenceSpec:
    kind: ReferenceKind
    path: Optional[str] = None
    sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind is ReferenceKind.NONE:
            if self.path is not None or self.sha256 is not None:
                raise ValueError("output_reference kind=none takes no path/sha256")
        else:
            if self.path is None or self.sha256 is None:
                raise ValueError("output_reference kind=external_golden requires path and sha256")
            _check_relative_path(self.path, "output_reference.path")
            _check_sha256(self.sha256, "output_reference.sha256")


@dataclass(frozen=True)
class OutputReferenceEntry:
    """One row of an external golden output reference file."""

    item_index: int
    source_item_id: str
    output_sha256: str

    def __post_init__(self) -> None:
        if self.item_index < 0:
            raise ValueError("item_index must be >= 0")
        _check_sha256(self.output_sha256, "output_sha256")


@dataclass(frozen=True)
class EvaluationSpec:
    deadline: Optional[DeadlineSpec] = None
    output_reference: OutputReferenceSpec = field(
        default_factory=lambda: OutputReferenceSpec(kind=ReferenceKind.NONE)
    )


@dataclass(frozen=True)
class EvaluationOverride:
    """Evaluate-time parameter override. The output reference is not overridable:
    it must be designated in the plan before the run."""

    schema_version: int
    deadline: Optional[DeadlineSpec]

    def __post_init__(self) -> None:
        _check_schema_version(self.schema_version)


@dataclass(frozen=True)
class InputPipelineSelection:
    """Explicit input-pipeline choice.  ``adapter_override`` must also name the
    override implementation; there is no auto-selection and no fallback."""

    mode: InputPipelineMode = InputPipelineMode.WORKLOAD
    implementation_id: Optional[str] = None
    implementation_version: Optional[str] = None  # optional pin
    implementation_digest: Optional[str] = None  # optional pin

    def __post_init__(self) -> None:
        if self.mode is InputPipelineMode.WORKLOAD:
            if any(v is not None for v in (self.implementation_id, self.implementation_version,
                                           self.implementation_digest)):
                raise ValueError("input_pipeline.implementation_* applies to adapter_override only")
            return
        if self.implementation_id is None or not self.implementation_id.strip():
            raise ValueError("input_pipeline mode adapter_override requires implementation_id")
        if self.implementation_version is not None:
            _check_nonempty(self.implementation_version, "input_pipeline.implementation_version")
        if self.implementation_digest is not None:
            _check_sha256(self.implementation_digest, "input_pipeline.implementation_digest")


@dataclass(frozen=True)
class StressSelection:
    spec: str
    level: str

    def __post_init__(self) -> None:
        _check_relative_path(self.spec, "stress.spec")
        check_identifier(self.level, "stress.level")


@dataclass(frozen=True)
class RealtimeConfig:
    """Stage 1R realtime harness settings (plan section ``realtime``).

    ``cpu_affinity`` lists the CPUs the run process is pinned to (host
    placement).  ``precondition_requests`` untimed requests run through the
    adapter after STRESS.verify and before MEASURE; they are recorded as a
    ``precondition`` phase and are never part of the planned requests.
    """

    cpu_affinity: tuple[int, ...]
    wait_policy: WaitPolicy = WaitPolicy.SPIN
    precondition_requests: int = 0

    def __post_init__(self) -> None:
        cpus = tuple(self.cpu_affinity)
        if not cpus:
            raise ValueError("realtime.cpu_affinity must list at least one CPU")
        if any(type(c) is not int or c < 0 for c in cpus) or len(set(cpus)) != len(cpus):
            raise ValueError("realtime.cpu_affinity must be distinct non-negative integers")
        object.__setattr__(self, "cpu_affinity", tuple(sorted(cpus)))
        if self.precondition_requests < 0:
            raise ValueError("realtime.precondition_requests must be >= 0")


@dataclass(frozen=True)
class ExperimentPlan:
    """Stage 1A experiment plan. There is intentionally no evaluation_level field."""

    schema_version: int
    experiment_id: str
    execution_kind: ExecutionKind
    input: str
    workload: str
    dut: str
    stress: StressSelection
    replay: ReplayPolicy
    run: RunConfig
    evaluation: EvaluationSpec = field(default_factory=EvaluationSpec)
    input_pipeline: InputPipelineSelection = field(default_factory=InputPipelineSelection)
    seed: int = 0
    repetitions: int = 1
    realtime: Optional[RealtimeConfig] = None  # Stage 1R; absent for Stage 1A virtual runs

    def __post_init__(self) -> None:
        _check_schema_version(self.schema_version)
        check_identifier(self.experiment_id, "experiment_id")
        for name in ("input", "workload", "dut"):
            _check_relative_path(getattr(self, name), name)
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.repetitions < 1:
            raise ValueError("repetitions must be >= 1")
