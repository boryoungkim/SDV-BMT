"""Strict YAML loading, ExperimentPlan resolution and static plan validation.

This is the only module allowed to import PyYAML.  The loader rejects
duplicate keys, merge keys and non-string keys; it resolves only lowercase
``true``/``false`` as booleans (YAML 1.1 ``yes/no/on/off`` stay strings) and
never converts scalars to dates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from ..contracts.ids import digest_of, loads_strict_json, sha256_hex
from ..contracts.specs import (
    DUTConfig,
    EvaluationOverride,
    ExperimentPlan,
    HardwareStressSpec,
    InputSpec,
    OutputReferenceEntry,
    SpecError,
    WorkloadSpec,
    parse_input_spec,
    parse_strict,
)
from ..contracts.status import (
    NOMINAL_LEVEL,
    ExecutionKind,
    InputKind,
    ReferenceKind,
    ReplayMode,
)


YAML_VERSION = yaml.__version__


class PlanError(ValueError):
    """The plan could not be loaded, or it failed VALIDATE."""

    def __init__(self, issues: list[str]) -> None:
        super().__init__("; ".join(issues))
        self.issues = list(issues)


# --------------------------------------------------------------------------
# strict YAML
# --------------------------------------------------------------------------


class _StrictLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.Node, deep: bool = False) -> dict:
        if not isinstance(node, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                None, None, "expected a mapping node", node.start_mark
            )
        seen: set[str] = set()
        for key_node, _value_node in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                raise yaml.constructor.ConstructorError(
                    None, None, "YAML merge keys ('<<') are not allowed", key_node.start_mark
                )
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise yaml.constructor.ConstructorError(
                    None, None, f"mapping key {key!r} is not a string", key_node.start_mark
                )
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


_REPLACED_TAGS = (
    "tag:yaml.org,2002:bool",
    "tag:yaml.org,2002:int",
    "tag:yaml.org,2002:float",
    "tag:yaml.org,2002:timestamp",
)
_StrictLoader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag not in _REPLACED_TAGS]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_StrictLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|false)$"), list("tf")
)
# Decimal only: no YAML 1.1 octal (050), hex (0x14), binary or base-60 (1:30)
# integers; underscores only between digits.  Anything else stays a string and
# is rejected by the typed field check.
_StrictLoader.add_implicit_resolver(
    "tag:yaml.org,2002:int",
    re.compile(r"^[-+]?(?:0|[1-9](?:_?[0-9])*)$"),
    list("-+0123456789"),
)
_StrictLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"^[-+]?(?:(?:[0-9](?:_?[0-9])*)?\.[0-9](?:_?[0-9])*(?:[eE][-+]?[0-9]+)?"
        r"|[0-9](?:_?[0-9])*[eE][-+]?[0-9]+)$"
    ),
    list("-+0123456789."),
)


def load_yaml_strict(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PlanError([f"{path}: cannot read: {exc}"]) from None
    try:
        return yaml.load(text, Loader=_StrictLoader)  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as exc:
        raise PlanError([f"{path}: invalid YAML: {exc}"]) from None


def _parse_file(path: Path, parse) -> Any:
    data = load_yaml_strict(path)
    try:
        return parse(data, path.name)
    except SpecError as exc:
        raise PlanError([f"{path}: {exc}"]) from None


# --------------------------------------------------------------------------
# plan resolution
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigFile:
    role: str
    relpath: str  # as written in the plan (relative to the plan file)
    path: Path  # resolved absolute path; never written into evidence
    sha256: str


@dataclass(frozen=True)
class ResolvedPlan:
    plan: ExperimentPlan
    input_spec: InputSpec
    workload: WorkloadSpec
    dut: DUTConfig
    stress_spec: HardwareStressSpec
    output_reference_entries: Optional[tuple[OutputReferenceEntry, ...]]
    config_files: tuple[ConfigFile, ...]
    input_spec_dir: Path

    @property
    def stress_level(self) -> str:
        return self.plan.stress.level

    def output_reference_content(self) -> dict[str, Any]:
        ref = self.plan.evaluation.output_reference
        if ref.kind is ReferenceKind.NONE:
            return {"kind": ReferenceKind.NONE}
        entries = list(self.output_reference_entries or ())
        return {
            "kind": ref.kind,
            "path": ref.path,
            "sha256": ref.sha256,
            "entry_count": len(entries),
            "entries_digest": digest_of(entries),
        }

    def resolved_content(self) -> dict[str, Any]:
        """Plan with defaults and referenced specs resolved; location-independent."""
        plan = self.plan
        return {
            "schema_version": plan.schema_version,
            "experiment_id": plan.experiment_id,
            "execution_kind": plan.execution_kind,
            "seed": plan.seed,
            "repetitions": plan.repetitions,
            "input": self.input_spec,
            "workload": self.workload,
            "dut": self.dut,
            "stress": {"spec": self.stress_spec, "level": plan.stress.level},
            "replay": plan.replay,
            "run": plan.run,
            "input_pipeline": plan.input_pipeline,
            "evaluation": {
                "deadline": plan.evaluation.deadline,
                "output_reference": self.output_reference_content(),
            },
            # Stage 1R realtime plans only, so Stage 1A plan digests are unchanged
            **({"realtime": plan.realtime} if plan.realtime is not None else {}),
        }

    @property
    def plan_digest(self) -> str:
        return digest_of(self.resolved_content())

    def config_digests(self) -> dict[str, str]:
        return {cf.role: cf.sha256 for cf in self.config_files}


def _load_reference(path: Path, declared_sha256: str) -> tuple[OutputReferenceEntry, ...]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PlanError([f"{path}: cannot read output reference: {exc}"]) from None
    actual = sha256_hex(data)
    if actual != declared_sha256:
        raise PlanError(
            [f"{path}: output reference sha256 mismatch (declared {declared_sha256}, actual {actual})"]
        )
    entries: list[OutputReferenceEntry] = []
    lines = data.decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for lineno, line in enumerate(lines, start=1):
        where = f"{path.name}:{lineno}"
        try:
            row = loads_strict_json(line)
            entries.append(parse_strict(OutputReferenceEntry, row, where))
        except (ValueError, SpecError) as exc:
            raise PlanError([f"{where}: {exc}"]) from None
    indices = [e.item_index for e in entries]
    if len(set(indices)) != len(indices):
        raise PlanError([f"{path}: duplicate item_index in output reference"])
    return tuple(entries)


def load_plan(plan_path: Path | str) -> ResolvedPlan:
    plan_path = Path(plan_path).resolve()
    plan = _parse_file(plan_path, lambda d, w: parse_strict(ExperimentPlan, d, w))
    base = plan_path.parent

    def resolve(relpath: str) -> Path:
        return (base / relpath).resolve()

    input_path = resolve(plan.input)
    workload_path = resolve(plan.workload)
    dut_path = resolve(plan.dut)
    stress_path = resolve(plan.stress.spec)
    input_spec = _parse_file(input_path, parse_input_spec)
    workload = _parse_file(workload_path, lambda d, w: parse_strict(WorkloadSpec, d, w))
    dut = _parse_file(dut_path, lambda d, w: parse_strict(DUTConfig, d, w))
    stress_spec = _parse_file(stress_path, lambda d, w: parse_strict(HardwareStressSpec, d, w))

    files = [
        ("plan", plan_path.name, plan_path),
        ("input", plan.input, input_path),
        ("workload", plan.workload, workload_path),
        ("dut", plan.dut, dut_path),
        ("stress", plan.stress.spec, stress_path),
    ]
    entries: Optional[tuple[OutputReferenceEntry, ...]] = None
    ref = plan.evaluation.output_reference
    if ref.kind is ReferenceKind.EXTERNAL_GOLDEN:
        assert ref.path is not None and ref.sha256 is not None
        ref_path = resolve(ref.path)
        entries = _load_reference(ref_path, ref.sha256)
        files.append(("output_reference", ref.path, ref_path))

    config_files = tuple(
        ConfigFile(role=role, relpath=rel, path=p, sha256=sha256_hex(p.read_bytes()))
        for role, rel, p in files
    )
    return ResolvedPlan(
        plan=plan,
        input_spec=input_spec,
        workload=workload,
        dut=dut,
        stress_spec=stress_spec,
        output_reference_entries=entries,
        config_files=config_files,
        input_spec_dir=input_path.parent,
    )


def load_evaluation_override(path: Path | str) -> EvaluationOverride:
    path = Path(path).resolve()
    return _parse_file(path, lambda d, w: parse_strict(EvaluationOverride, d, w))


# --------------------------------------------------------------------------
# VALIDATE (static, side-effect free)
# --------------------------------------------------------------------------


def validate_plan(rp: ResolvedPlan) -> list[str]:
    """Static Stage 1A plan rules. Returns a list of issues (empty = valid)."""
    plan = rp.plan
    issues: list[str] = []
    if plan.execution_kind is not ExecutionKind.MOCK and plan.realtime is None:
        issues.append("execution_kind: actual execution requires the realtime section (Stage 1R); "
                      "the Stage 1A virtual runner supports mock execution only")
    if rp.input_spec.input_kind is InputKind.DATASET and plan.replay.mode is ReplayMode.RECORDED:
        issues.append("replay.mode: recorded replay is not allowed for dataset input")
    if plan.run.max_inflight != 1:
        issues.append("run.max_inflight: Stage 1A supports max_inflight=1 only")
    if plan.run.queue_capacity != 0:
        issues.append("run.queue_capacity: Stage 1A supports queue_capacity=0 only")
    if plan.run.drain_ns < plan.run.request_timeout_ns:
        issues.append(
            "run.drain_ns must be >= run.request_timeout_ns so every issued request "
            "reaches a terminal state before drain ends"
        )
    if rp.dut.max_inflight != 1:
        issues.append("dut.max_inflight: Stage 1A supports max_inflight=1 only")
    if rp.workload.batch != 1:
        issues.append("workload.batch: Stage 1A supports batch=1 only")
    if rp.workload.quality_evaluator is not None:
        issues.append("workload.quality_evaluator: task quality is out of Stage 1A scope")
    level = plan.stress.level
    if level != NOMINAL_LEVEL and level not in rp.stress_spec.levels:
        issues.append(
            f"stress.level: {level!r} is neither {NOMINAL_LEVEL!r} nor a level of "
            f"stress spec {rp.stress_spec.stress_id!r}"
        )
    return issues
