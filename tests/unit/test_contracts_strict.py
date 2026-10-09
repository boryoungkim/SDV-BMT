"""WP1 gate: strict contracts — unknown/missing fields, types, Stage 1A plan shape."""

from __future__ import annotations

from pathlib import Path

import pytest

from sdv_bmt.contracts.records import ErrorInfo, MetricResult
from sdv_bmt.contracts.specs import (
    DatasetSpec,
    DrivingScenarioSpec,
    ExperimentPlan,
    HardwareStressSpec,
    ReplayPolicy,
    RunConfig,
    SpecError,
    parse_input_spec,
    parse_strict,
)
from sdv_bmt.contracts.status import ExecutionKind, MetricStatus, ReplayMode
from sdv_bmt.runner.plan import PlanError, load_yaml_strict
from tests._helpers import dataset_spec_dict, plan_dict, scenario_spec_dict, stress_spec_dict


# ---------------------------------------------------------------- unknown fields


def test_unknown_top_level_field_rejected():
    with pytest.raises(SpecError, match=r"unknown field\(s\): surprise"):
        parse_strict(ExperimentPlan, plan_dict(surprise=1))


def test_unknown_nested_field_rejected():
    data = plan_dict()
    data["run"]["prefetch"] = 4
    with pytest.raises(SpecError, match=r"\$\.run: unknown field\(s\): prefetch"):
        parse_strict(ExperimentPlan, data)


def test_evaluation_level_field_is_not_part_of_stage1a_plan():
    # M-7: Stage 1A ExperimentPlan has no evaluation_level option.
    with pytest.raises(SpecError, match="evaluation_level"):
        parse_strict(ExperimentPlan, plan_dict(evaluation_level="model"))


def test_unknown_field_in_input_family_rejected():
    data = dataset_spec_dict()
    data["family"]["weather"] = "night"
    with pytest.raises(SpecError, match=r"family: unknown field\(s\): weather"):
        parse_input_spec(data)


# ---------------------------------------------------------------- missing fields


@pytest.mark.parametrize(
    "missing", ["schema_version", "experiment_id", "execution_kind", "input", "stress", "replay", "run"]
)
def test_missing_required_plan_field_rejected(missing):
    data = plan_dict()
    del data[missing]
    with pytest.raises(SpecError, match=rf"\$\.{missing}: required field is missing"):
        parse_strict(ExperimentPlan, data)


def test_missing_required_nested_field_rejected():
    data = plan_dict()
    del data["run"]["request_timeout_ns"]
    with pytest.raises(SpecError, match=r"run\.request_timeout_ns: required field is missing"):
        parse_strict(ExperimentPlan, data)


def test_required_nullable_field_must_be_explicit():
    data = dataset_spec_dict()
    del data["family"]["annotation_schema"]
    with pytest.raises(SpecError, match="annotation_schema: required field is missing"):
        parse_input_spec(data)


def test_defaults_are_resolved():
    plan = parse_strict(ExperimentPlan, plan_dict())
    assert plan.seed == 0 and plan.repetitions == 1
    assert plan.run.output_capture.value == "task_output_all"
    assert plan.run.on_request_error.value == "continue"
    assert plan.evaluation.deadline is None
    assert plan.evaluation.output_reference.kind.value == "none"
    assert plan.execution_kind is ExecutionKind.MOCK


# ---------------------------------------------------------------- types


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("max_inflight", True, "expected an integer, got bool"),
        ("max_inflight", 1.0, "expected an integer, got float"),
        ("request_timeout_ns", "5", "expected an integer, got str"),
        ("on_full", "drop_oldest", "expected one of"),
        ("drain_ns", None, "null is not allowed"),
    ],
)
def test_wrong_types_rejected(field, value, message):
    data = plan_dict()
    data["run"][field] = value
    with pytest.raises(SpecError, match=message):
        parse_strict(ExperimentPlan, data)


def test_float_field_accepts_int_and_normalizes():
    replay = parse_strict(ReplayPolicy, {"mode": "fixed_rate", "rate_hz": 30})
    assert replay.rate_hz == 30.0 and isinstance(replay.rate_hz, float)


def test_float_field_rejects_bool():
    with pytest.raises(SpecError, match="expected a finite number, got bool"):
        parse_strict(ReplayPolicy, {"mode": "fixed_rate", "rate_hz": True})


# ---------------------------------------------------------------- replay rules


def test_recorded_replay_resolves_default_speed():
    replay = parse_strict(ReplayPolicy, {"mode": "recorded"})
    assert replay.mode is ReplayMode.RECORDED and replay.speed == 1.0


@pytest.mark.parametrize(
    "data",
    [
        {"mode": "asap", "rate_hz": 10},
        {"mode": "fixed_rate"},
        {"mode": "fixed_rate", "rate_hz": 10, "speed": 1.0},
        {"mode": "fixed_rate", "rate_hz": 10, "episode_gap_ns": 5},
        {"mode": "recorded", "rate_hz": 10},
        {"mode": "recorded", "speed": 0},
        {"mode": "recorded", "lead_ns": -1},
    ],
)
def test_mode_specific_replay_fields(data):
    with pytest.raises(SpecError):
        parse_strict(ReplayPolicy, data)


# ---------------------------------------------------------------- input specs


def test_dataset_spec_rejects_timing_and_streams():
    data = dataset_spec_dict()
    data["family"]["timing"] = scenario_spec_dict()["family"]["timing"]
    with pytest.raises(SpecError, match="dataset family must not declare timing"):
        parse_input_spec(data)
    data = dataset_spec_dict()
    data["family"]["streams"] = ["cam_front"]
    with pytest.raises(SpecError, match="dataset family must not declare streams"):
        parse_input_spec(data)


def test_scenario_spec_requires_streams_and_known_stream_selection():
    data = scenario_spec_dict()
    data["family"]["streams"] = None
    with pytest.raises(SpecError, match="must declare streams"):
        parse_input_spec(data)
    data = scenario_spec_dict(selection={"stream_id": "lidar_top"})
    with pytest.raises(SpecError, match="not in family.streams"):
        parse_input_spec(data)


def test_input_kind_dispatch():
    assert isinstance(parse_input_spec(dataset_spec_dict()), DatasetSpec)
    assert isinstance(parse_input_spec(scenario_spec_dict()), DrivingScenarioSpec)
    with pytest.raises(SpecError, match="input_kind"):
        parse_input_spec(dataset_spec_dict(input_kind="mixed"))


def test_timing_unit_and_semantic_are_typed():
    data = scenario_spec_dict()
    data["family"]["timing"]["unit"] = "frames"
    with pytest.raises(SpecError, match="unit: expected one of"):
        parse_input_spec(data)
    data = scenario_spec_dict()
    data["family"]["timing"]["semantic"] = "exposure"
    with pytest.raises(SpecError, match="semantic: expected one of"):
        parse_input_spec(data)


def test_source_index_digest_must_be_sha256():
    data = dataset_spec_dict()
    data["source"]["index_sha256"] = "abc"
    with pytest.raises(SpecError, match="index_sha256"):
        parse_input_spec(data)


# ---------------------------------------------------------------- stress spec


def test_stress_level_none_is_reserved():
    with pytest.raises(SpecError, match="reserved for nominal"):
        parse_strict(HardwareStressSpec, stress_spec_dict(levels={"none": 0.0}))


def test_stress_schedule_and_failure_policy_restricted():
    with pytest.raises(SpecError, match="schedule"):
        parse_strict(HardwareStressSpec, stress_spec_dict(schedule="burst"))
    with pytest.raises(SpecError, match="on_apply_failure"):
        parse_strict(HardwareStressSpec, stress_spec_dict(on_apply_failure="fallback_nominal"))


def test_stress_background_workload_must_be_explicit():
    data = stress_spec_dict()
    del data["background_workload"]
    with pytest.raises(SpecError, match="background_workload: required field is missing"):
        parse_strict(HardwareStressSpec, data)


# ---------------------------------------------------------------- run config


def test_run_config_basic_ranges():
    with pytest.raises(SpecError, match="request_timeout_ns must be > 0"):
        parse_strict(
            RunConfig,
            {"max_inflight": 1, "queue_capacity": 0, "on_full": "drop_new",
             "request_timeout_ns": 0, "drain_ns": 0},
        )


# ---------------------------------------------------------------- metric result rule


def _metric(status, value, **kw):
    return MetricResult(
        metric_id="bmt.test", definition_version=1, measurement_boundary_id=None,
        unit="ns", status=status, value=value, n=0, window=None, method="m",
        inputs=(), notes=(), eval_id="e", execution_kind=ExecutionKind.MOCK, **kw,
    )


def test_non_valid_metric_requires_null_value():
    with pytest.raises(ValueError, match="requires value=null"):
        _metric(MetricStatus.CENSORED, 0.0, censored_fraction=0.5)
    with pytest.raises(ValueError, match="requires value=null"):
        _metric(MetricStatus.EMPTY, 0)
    with pytest.raises(ValueError, match="requires censored_fraction"):
        _metric(MetricStatus.CENSORED, None)
    with pytest.raises(ValueError, match="requires a numeric value"):
        _metric(MetricStatus.VALID, None)
    assert _metric(MetricStatus.CENSORED, None, censored_fraction=0.5).value is None


def test_error_info_is_plain_record():
    assert ErrorInfo("c", "m").code == "c"


# ---------------------------------------------------------------- strict YAML


def _yaml(tmp_path: Path, text: str):
    path = tmp_path / "doc.yaml"
    path.write_text(text, encoding="utf-8")
    return load_yaml_strict(path)


def test_yaml_duplicate_key_rejected(tmp_path):
    with pytest.raises(PlanError, match="duplicate key 'a'"):
        _yaml(tmp_path, "a: 1\na: 2\n")


def test_yaml_nested_duplicate_key_rejected(tmp_path):
    with pytest.raises(PlanError, match="duplicate key 'mode'"):
        _yaml(tmp_path, "replay:\n  mode: asap\n  mode: recorded\n")


def test_yaml_merge_key_rejected(tmp_path):
    with pytest.raises(PlanError, match="merge keys"):
        _yaml(tmp_path, "base: &b {a: 1}\nother:\n  <<: *b\n")


def test_yaml_non_string_key_rejected(tmp_path):
    with pytest.raises(PlanError, match="is not a string"):
        _yaml(tmp_path, "1: a\n")


def test_yaml_scalars_resolve_strictly(tmp_path):
    data = _yaml(tmp_path, "a: yes\nb: on\nc: true\nd: False\ne: 2026-10-06\nf: 33_000_000\ng: null\n")
    assert data == {"a": "yes", "b": "on", "c": True, "d": "False",
                    "e": "2026-10-06", "f": 33_000_000, "g": None}


def test_yaml_multiple_documents_rejected(tmp_path):
    with pytest.raises(PlanError, match="invalid YAML"):
        _yaml(tmp_path, "a: 1\n---\nb: 2\n")


@pytest.mark.parametrize("literal", ["050_000_000", "0x14", "1:30", "0b101", "1_", "007", ".inf", ".nan"])
def test_yaml_non_decimal_numbers_are_rejected_for_numeric_fields(tmp_path, literal):
    # YAML 1.1 octal/hex/base-60/binary forms stay strings, so typed fields reject them.
    path = tmp_path / "run.yaml"
    path.write_text(f"max_inflight: 1\nqueue_capacity: 0\non_full: drop_new\n"
                    f"request_timeout_ns: {literal}\ndrain_ns: 5\n")
    with pytest.raises(SpecError, match="request_timeout_ns: expected an integer, got str"):
        parse_strict(RunConfig, load_yaml_strict(path))


def test_yaml_decimal_numbers_resolve(tmp_path):
    data = _yaml(tmp_path, "a: 33_000_000\nb: -7\nc: 0\nd: 1.5\ne: .5\nf: 2.5e-3\ng: 1e9\nh: +3\n")
    assert data == {"a": 33_000_000, "b": -7, "c": 0, "d": 1.5, "e": 0.5, "f": 0.0025, "g": 1e9, "h": 3}
    assert isinstance(data["g"], float)


def test_error_info_fields_must_be_strings():
    with pytest.raises(ValueError, match="must be strings"):
        ErrorInfo("code", float("nan"))
