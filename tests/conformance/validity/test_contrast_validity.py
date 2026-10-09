"""Contrast validity conformance (design update "Factor Isolation, Realized-Condition
Validity"): Design / Realization / Measurement guards, cases FI-1..FI-6, and the
semantics fixed after the independent review of the update.

Valid attributed contrast = Design valid AND Realization valid AND Measurement valid.
Stage 1A runs are mock: the Realization Guard is ``not_observable`` and an
attributed contrast is never established from them.  The "actual" pairs below are
runner manifests relabelled as actual with synthetic observed evidence; Stage 1A
never produces them, they exercise the guard semantics for later stages.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest

from sdv_bmt.cli.main import build_components
from sdv_bmt.contracts.ids import to_jsonable
from sdv_bmt.contracts.records import ContrastValidityResult, RealizationEvidence, RunManifest
from sdv_bmt.contracts.specs import (
    DeadlineSpec,
    DUTConfig,
    EvaluationOverride,
    EvaluationSpec,
    ReplayPolicy,
    RunConfig,
    parse_input_spec,
    parse_strict,
)
from sdv_bmt.contracts.status import (
    ContrastType,
    ContrastValidity,
    DeadlinePolicy,
    EvidenceValidity,
    GuardStatus,
    RealizedCondition,
)
from sdv_bmt.evidence.keys import identity_components
from sdv_bmt.evidence.manifest import read_manifest
from sdv_bmt.evidence.validity import INTERACTION_NOTE, evaluate_contrast_validity
from sdv_bmt.runner.plan import load_yaml_strict
from tests._helpers import CONFIGS, MS, REPO_ROOT, evaluate, load, run_plan, variant

HARDWARE_CONDITIONS = {
    "input_residency", "buffer_state", "data_movement", "runtime_queue_state", "device_frequency",
    "thermal_state", "competing_device_users", "memory_pressure_bandwidth",
}
ALL_CONDITIONS = {c.value for c in RealizedCondition}
G1 = "g1-dataset-fixed-rate-nominal"
SCENARIO = "scenario-two-episodes-fake-stress"


@dataclasses.dataclass
class Run:
    manifest: dict
    evaluation: dict


def _manifest(rp, tmp_path, name):
    return read_manifest(run_plan(rp, tmp_path / name, run_id=f"run-{name}").run_dir)


def _run(rp, tmp_path, name, factory=None, override=None) -> Run:
    out = run_plan(rp, tmp_path / name, factory=factory, run_id=f"run-{name}")
    evaluation = evaluate(out.run_dir, eval_id=f"eval-{name}", override=override)
    return Run(read_manifest(out.run_dir), json.loads((evaluation.eval_dir / "eval_manifest.json").read_text()))


def _validity(a: Run, b: Run, contrast="stress"):
    return evaluate_contrast_validity(a.manifest, b.manifest, contrast, evaluations=(a.evaluation, b.evaluation))


def _level(rp, level):
    return variant(rp, plan={"stress": dataclasses.replace(rp.plan.stress, level=level)})


def _input(rp, spec_file):
    path = CONFIGS / "inputs" / spec_file
    return variant(rp, input_spec=parse_input_spec(load_yaml_strict(path)), input_spec_dir=path.parent)


def _dut(name):
    return parse_strict(DUTConfig, load_yaml_strict(CONFIGS / "duts" / name))


def _evidence(condition, validity, value=None, source="tool-x", semantic="s", unit=None, scope="device",
              equivalence="exact"):
    return to_jsonable(RealizationEvidence(RealizedCondition(condition), EvidenceValidity(validity), source,
                                           semantic, unit, scope, None, value, equivalence))


def _actualize(run: Run) -> Run:
    """Relabel as actual with fully observed evidence showing the planned factor realized."""
    m = run.manifest
    parts = m["comparison_keys"]["components"]["common_parts"]
    for section in (m, m["status"], m["realization"], parts):
        section["execution_kind"] = "actual"
    parts["model_artifact"] = "a" * 64  # an actual workload pins its model artifact
    ids = identity_components(m["comparison_keys"])
    planned = {
        "input_payload": {"items_digest": ids["items_digest"], "item_count": m["input"]["item_count"]},
        "stress_realization": ({"nominal_check": "passed"} if ids["stress_level"] == "none"
                               else {"verification": "verified", "coverage": "full"}),
    }
    m["realization"]["evidence"] = [_evidence(c, "observed", value=planned.get(c, {"v": 1}))
                                    for c in sorted(ALL_CONDITIONS)]
    return run


def _replace_evidence(run: Run, evidence) -> Run:
    run.manifest["realization"]["evidence"] = [
        evidence if e["condition"] == evidence["condition"] else e for e in run.manifest["realization"]["evidence"]
    ]
    return run


def _actual_pair(tmp_path, level_b="none"):
    rp = load(G1)
    return _actualize(_run(rp, tmp_path, "a")), _actualize(_run(_level(rp, level_b), tmp_path, "b"))


# ---------------------------------------------------------------- FI-1


def test_fi1_valid_stress_design_contrast(tmp_path):
    base = load(G1)
    result = _validity(_run(_level(base, "none"), tmp_path, "nominal"), _run(_level(base, "L1"), tmp_path, "l1"))
    assert result.design_guard.status is GuardStatus.PASS
    assert result.design_guard.observed_differences == ("stress_level", "stress_level_value")
    assert result.design_guard.violations == () and result.design_guard.intended_factor_varied
    assert result.measurement_guard.status is GuardStatus.PASS
    assert result.measurement_guard.boundary_identity["equal"]
    assert result.realization_guard.status is GuardStatus.NOT_OBSERVABLE
    assert result.realization_guard.basis == "mock_execution"
    assert result.realization_guard.intended_factor_conditions == ("stress_realization",)
    assert result.overall_status is ContrastValidity.NOT_ESTABLISHED and not result.attribution_allowed


def test_measurement_guard_is_unresolved_without_the_evaluations(tmp_path):
    base = load(G1)
    a, b = _manifest(_level(base, "none"), tmp_path, "a"), _manifest(_level(base, "L1"), tmp_path, "b")
    result = evaluate_contrast_validity(a, b, "stress")
    assert result.measurement_guard.status is GuardStatus.UNRESOLVED
    assert result.measurement_guard.notes[0].startswith("evaluation manifests not given")
    assert result.overall_status is ContrastValidity.NOT_ESTABLISHED


# ---------------------------------------------------------------- FI-2


def test_fi2_effective_preprocessing_pipeline_differs(tmp_path):
    fused = _dut("fake-blocking-25ms-fused.yaml")
    workload_path = _run(variant(load(G1), dut=fused), tmp_path, "workload")
    override_path = _run(load("g1-adapter-override"), tmp_path, "override")
    result = _validity(workload_path, override_path)
    assert result.design_guard.status is GuardStatus.FAIL
    assert result.design_guard.violations == ("input_pipeline", "measurement_boundary_id")
    assert result.measurement_guard.status is GuardStatus.FAIL  # the boundary carries the pipeline digest
    # the realized preparation (pipeline, equivalence) differs too, and is localized
    assert [m["condition"] for m in result.realization_guard.mismatches] == ["input_preparation"]
    assert result.overall_status is ContrastValidity.INVALID and not result.attribution_allowed


def test_fi2_semantic_equivalence_does_not_make_pipelines_interchangeable(tmp_path):
    # The override passed its equivalence check, yet the contrast is still invalid:
    # equal semantics does not make the execution path performance-identical.
    override = _manifest(load("g1-adapter-override"), tmp_path, "override")
    assert override["input_pipeline"]["equivalence_validation"]["status"] == "passed"
    workload = _manifest(variant(load(G1), dut=_dut("fake-blocking-25ms-fused.yaml")), tmp_path, "workload")
    assert not evaluate_contrast_validity(workload, override, "stress").attribution_allowed


# ---------------------------------------------------------------- FI-3


def test_fi3_measurement_boundary_differs(tmp_path):
    base = load(G1)
    other_ready = dataclasses.replace(base.workload, output_ready="task output encoded in memory only")
    a = _manifest(base, tmp_path, "a")
    b = _manifest(variant(base, workload=other_ready), tmp_path, "b")
    result = evaluate_contrast_validity(a, b, "stress")
    assert result.measurement_guard.status is GuardStatus.FAIL
    assert result.measurement_guard.violations == ("measurement_boundary_id",)
    assert not result.measurement_guard.boundary_identity["equal"]
    assert result.design_guard.status is GuardStatus.FAIL  # the boundary is also a design identity
    assert result.overall_status is ContrastValidity.INVALID


def test_fi3_isolated_boundary_difference(tmp_path):
    a = _manifest(load(G1), tmp_path, "a")
    b = copy.deepcopy(a)
    b["run_id"] = "run-b"
    b["comparison_keys"]["components"]["common_parts"]["measurement_boundary_id"] = "f" * 64
    b["measurement_boundary"]["measurement_boundary_id"] = "f" * 64
    result = evaluate_contrast_validity(a, b, "stress")
    assert result.measurement_guard.violations == ("measurement_boundary_id",)
    assert result.design_guard.violations == ("measurement_boundary_id",)


# ---------------------------------------------------------------- FI-4


def test_fi4_input_family_differs_during_scenario_contrast(tmp_path):
    base = _level(load(SCENARIO), "none")
    spec_path = CONFIGS / "inputs" / "scenario-ep-a.yaml"
    ep_a = parse_input_spec(load_yaml_strict(spec_path))
    other_family = dataclasses.replace(
        parse_input_spec(load_yaml_strict(CONFIGS / "inputs" / "scenario-ep-b.yaml")),
        family=dataclasses.replace(ep_a.family, source_version="2"),
    )
    a = _manifest(variant(base, input_spec=ep_a, input_spec_dir=spec_path.parent), tmp_path, "a")
    b = _manifest(variant(base, input_spec=other_family, input_spec_dir=spec_path.parent), tmp_path, "b")
    result = evaluate_contrast_validity(a, b, "scenario")
    assert result.design_guard.status is GuardStatus.FAIL
    assert "input_family_digest" in result.design_guard.violations
    assert "input_selection_digest" not in result.design_guard.violations  # allowed to vary
    assert result.overall_status is ContrastValidity.INVALID


def test_fi4_same_family_scenario_contrast_passes_design(tmp_path):
    base = _level(load(SCENARIO), "none")
    a = _run(_input(base, "scenario-ep-a.yaml"), tmp_path, "a")
    b = _run(_input(base, "scenario-ep-b.yaml"), tmp_path, "b")
    result = _validity(a, b, "scenario")
    assert result.design_guard.status is GuardStatus.PASS
    assert result.design_guard.observed_differences == ("input_selection_digest", "items_digest")
    assert result.realization_guard.intended_factor_conditions == ("input_payload",)
    assert result.realization_guard.mismatches == ()  # preparation outcome is comparable across episodes
    assert result.measurement_guard.status is GuardStatus.PASS
    assert result.overall_status is ContrastValidity.NOT_ESTABLISHED  # mock: realization not observable


def test_scenario_contrast_is_not_applicable_to_dataset_input(tmp_path):
    a = _manifest(load(G1), tmp_path, "a")
    result = evaluate_contrast_validity(a, a, "scenario")
    assert result.design_guard.status is GuardStatus.NOT_APPLICABLE
    assert result.overall_status is ContrastValidity.NOT_APPLICABLE


def test_not_applicable_never_hides_a_failure(tmp_path):
    dataset = _manifest(load(G1), tmp_path, "dataset")
    scenario = _manifest(_level(load(SCENARIO), "none"), tmp_path, "scenario")
    result = evaluate_contrast_validity(dataset, scenario, "scenario")
    assert result.design_guard.status is GuardStatus.FAIL  # every difference is listed
    assert {"input_kind", "input_family_digest", "replay"} <= set(result.design_guard.violations)
    assert result.overall_status is ContrastValidity.INVALID
    # a measurement failure wins over a not-applicable design
    broken = copy.deepcopy(dataset)
    broken["run_id"] = "run-broken"
    broken["status"].update(measurement_status="invalid", measurement_reason="unresolved_requests")
    result = evaluate_contrast_validity(dataset, broken, "scenario")
    assert result.design_guard.status is GuardStatus.NOT_APPLICABLE
    assert result.measurement_guard.status is GuardStatus.FAIL
    assert result.overall_status is ContrastValidity.INVALID


# ---------------------------------------------------------------- FI-5


def test_fi5_all_forbidden_differences_are_reported(tmp_path):
    base = load(G1)
    changed = variant(
        base,
        dut=_dut("fake-blocking-40ms.yaml"),
        stress_spec=dataclasses.replace(base.stress_spec, placement="other-virtual-placement"),
        plan={
            "replay": parse_strict(ReplayPolicy, {"mode": "fixed_rate", "rate_hz": 10}),
            "run": parse_strict(RunConfig, {"max_inflight": 1, "queue_capacity": 0, "on_full": "drop_new",
                                            "request_timeout_ns": 500_000_000, "drain_ns": 1_000_000_000}),
            "stress": dataclasses.replace(base.plan.stress, level="L1"),
        },
    )
    result = evaluate_contrast_validity(_manifest(base, tmp_path, "a"), _manifest(changed, tmp_path, "b"), "stress")
    assert result.design_guard.status is GuardStatus.FAIL
    assert set(result.design_guard.violations) == {"dut_config", "replay", "timeout_drain", "stress_axis_digest"}
    assert {"stress_level", "stress_level_value"} <= set(result.design_guard.observed_differences)
    assert result.overall_status is ContrastValidity.INVALID and not result.attribution_allowed


def test_redefined_stress_level_is_a_design_violation(tmp_path):
    l1 = _level(load(G1), "L1")
    redefined = variant(l1, stress_spec=dataclasses.replace(l1.stress_spec, levels={"L1": 0.3, "L2": 0.5, "L3": 1.0}))
    result = evaluate_contrast_validity(_manifest(l1, tmp_path, "a"), _manifest(redefined, tmp_path, "b"), "stress")
    # same level name, different intensity: the level table is a fixed identity
    assert result.design_guard.violations == ("stress_level_table",)


@pytest.mark.parametrize(
    "component,change",
    [
        ("dut_config", {"runtime_options": {"service_time_ns": 1}}),
        ("dut_descriptor", {"adapter_version": "2"}),
        ("dut_capabilities", {"max_inflight": 4}),
    ],
)
def test_dut_identity_differences_are_localized(tmp_path, component, change):
    a = _manifest(load(G1), tmp_path, "a")
    b = copy.deepcopy(a)
    b["run_id"] = "run-b"
    parts = b["comparison_keys"]["components"]["common_parts"]
    parts[component] = {**parts[component], **change}
    assert evaluate_contrast_validity(a, b, "stress").design_guard.violations == (component,)


def test_output_reference_is_a_measurement_identity(tmp_path):
    base = load(G1)
    no_reference = variant(_level(base, "L1"), plan={"evaluation": EvaluationSpec()}, output_reference_entries=None)
    result = _validity(_run(_level(base, "none"), tmp_path, "a"), _run(no_reference, tmp_path, "b"))
    assert result.design_guard.violations == ("output_reference",)
    assert {"output_reference", "evaluation_protocol.deadline", "evaluation_protocol.output_reference"} == set(
        result.measurement_guard.violations)
    assert result.overall_status is ContrastValidity.INVALID


def test_evaluation_protocol_and_evaluator_must_match(tmp_path):
    rp = load(G1)
    later_deadline = EvaluationOverride(1, DeadlineSpec(DeadlinePolicy.FRAME_PERIOD, 120 * MS))
    a = _run(_level(rp, "none"), tmp_path, "a")
    b = _run(_level(rp, "L1"), tmp_path, "b", override=later_deadline)
    result = _validity(a, b)
    assert result.measurement_guard.violations == ("evaluation_protocol.deadline",)
    assert result.overall_status is ContrastValidity.INVALID
    c = _run(_level(rp, "L1"), tmp_path, "c")
    assert _validity(a, c).measurement_guard.status is GuardStatus.PASS
    wrong_run = evaluate_contrast_validity(a.manifest, c.manifest, "stress", evaluations=(a.evaluation, b.evaluation))
    assert "run_b:evaluation_of_another_run" in wrong_run.measurement_guard.violations
    other_evaluator = copy.deepcopy(c.evaluation)
    other_evaluator["evaluator"]["catalog_version"] = "another-catalog"
    result = evaluate_contrast_validity(a.manifest, c.manifest, "stress", evaluations=(a.evaluation, other_evaluator))
    assert result.measurement_guard.violations == ("evaluator.catalog_version",)


# ---------------------------------------------------------------- FI-6


def test_fi6_mock_replicates_never_prove_realized_conditions(tmp_path):
    rp = load(G1)
    a, b = _run(rp, tmp_path, "a"), _run(rp, tmp_path, "b")
    result = _validity(a, b)
    assert result.design_guard.status is GuardStatus.PASS
    assert result.design_guard.notes == ("intended factor did not vary (replicate pair)",)
    assert result.design_guard.unknown_identities == ("model_artifact",)  # fake workload; tolerated only for mock
    assert result.measurement_guard.status is GuardStatus.PASS
    assert result.realization_guard.status is GuardStatus.NOT_OBSERVABLE  # never PASS for mock
    assert result.realization_guard.evidence_available == ("input_payload",)
    unresolved = {u["condition"] for u in result.realization_guard.unresolved}
    assert HARDWARE_CONDITIONS <= unresolved
    assert result.overall_status is ContrastValidity.NOT_ESTABLISHED and not result.attribution_allowed
    evidence = {e["condition"]: e for e in a.manifest["realization"]["evidence"]}
    assert {evidence[c]["validity"] for c in HARDWARE_CONDITIONS} == {"not_observable"}
    assert evidence["issue_timing"]["validity"] == "virtual" and evidence["input_payload"]["validity"] == "observed"
    assert evidence["input_preparation"]["validity"] == "virtual"


def test_fi6_limitation_is_documented():
    text = (REPO_ROOT / "docs" / "measurement_protocol.md").read_text(encoding="utf-8")
    assert "A configured condition and a realized condition are not assumed to be identical" in text
    assert "not_observable" in text


# ---------------------------------------------------------------- measurement validity of runs


def test_contrast_with_an_unmeasured_run_fails_the_measurement_guard(tmp_path):
    ok = _manifest(load(G1), tmp_path, "ok")
    failed = _manifest(load("g4-stress-verify-failure"), tmp_path, "failed")
    result = evaluate_contrast_validity(ok, failed, "stress")
    assert any(v.startswith("run_b:measurement_not_valid") for v in result.measurement_guard.violations)
    assert result.overall_status is ContrastValidity.INVALID


# ---------------------------------------------------------------- realized inputs (runner evidence)


class _FailingPreprocess:
    """Workload whose preprocess fails for some items (they become NOT_ISSUED)."""

    def __init__(self, inner, bad):
        self._inner, self._bad = inner, set(bad)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def preprocess(self, payload, item):
        if item.item_index in self._bad:
            raise ValueError("corrupt payload")
        return self._inner.preprocess(payload, item)


def test_undelivered_inputs_are_a_realized_difference(tmp_path):
    def factory(rp):
        components = build_components(rp)
        return dataclasses.replace(components, workload=_FailingPreprocess(components.workload, {2, 3, 4}))

    clean, partial = _run(load(G1), tmp_path, "clean"), _run(load(G1), tmp_path, "partial", factory=factory)
    status = partial.manifest["status"]
    assert status["measurement_status"] == "valid" and status["request_counts"]["not_issued"] == 3
    evidence = {e["condition"]: e for e in partial.manifest["realization"]["evidence"]}
    assert evidence["input_preparation"]["value"]["failed_items"] == [
        {"item_index": i, "stage": "preprocess"} for i in (2, 3, 4)
    ]
    result = _validity(clean, partial)
    assert result.realization_guard.status is GuardStatus.FAIL  # a recorded difference, even for mock
    assert [m["condition"] for m in result.realization_guard.mismatches] == ["input_preparation"]
    assert result.overall_status is ContrastValidity.INVALID


@pytest.mark.parametrize("experiment,equivalence", [(G1, "not_applicable"), ("g1-adapter-override", "passed")])
def test_preparation_outcome_is_recorded_in_both_pipeline_modes(tmp_path, experiment, equivalence):
    m = _manifest(load(experiment), tmp_path, "run")
    preparation = {e["condition"]: e for e in m["realization"]["evidence"]}["input_preparation"]
    assert preparation["validity"] == "virtual"
    assert preparation["value"] == {"pipeline_digest": m["input_pipeline"]["pipeline_digest"],
                                    "failed_items": [], "equivalence": equivalence}
    assert m["status"]["request_counts"]["planned"] == m["input"]["item_count"]


# ---------------------------------------------------------------- realization semantics (synthetic actual)


def test_realization_passes_only_when_every_condition_is_observed_equal(tmp_path):
    result = _validity(*_actual_pair(tmp_path, level_b="L1"))
    assert result.realization_guard.status is GuardStatus.PASS
    assert result.realization_guard.intended_factor_conditions == ("stress_realization",)
    assert set(result.realization_guard.evidence_available) == ALL_CONDITIONS - {"stress_realization"}
    assert result.overall_status is ContrastValidity.VALID and result.attribution_allowed


def test_valid_replicate_pair_is_not_attributable(tmp_path):
    result = _validity(*_actual_pair(tmp_path))
    assert result.overall_status is ContrastValidity.VALID
    assert not result.design_guard.intended_factor_varied and not result.attribution_allowed


def test_label_only_level_change_is_a_replicate(tmp_path):
    base = load(G1)
    same_value = variant(base, stress_spec=dataclasses.replace(base.stress_spec,
                                                                levels={"L1": 0.5, "L2": 0.5, "L3": 1.0}))
    a = _actualize(_run(_level(same_value, "L1"), tmp_path, "a"))
    b = _actualize(_run(_level(same_value, "L2"), tmp_path, "b"))
    result = _validity(a, b)
    assert result.design_guard.observed_differences == ("stress_level",)
    assert result.design_guard.status is GuardStatus.PASS and not result.design_guard.intended_factor_varied
    assert result.realization_guard.intended_factor_conditions == ()  # compared as a fixed condition
    assert result.overall_status is ContrastValidity.VALID and not result.attribution_allowed


def test_realization_mismatch_fails(tmp_path):
    a, b = _actual_pair(tmp_path)
    _replace_evidence(b, _evidence("device_frequency", "observed", value={"v": 2}))
    result = _validity(a, b)
    assert result.realization_guard.status is GuardStatus.FAIL
    assert result.realization_guard.mismatches[0]["condition"] == "device_frequency"
    assert result.overall_status is ContrastValidity.INVALID


def test_partial_evidence_is_unresolved(tmp_path):
    a, b = _actual_pair(tmp_path)
    _replace_evidence(b, _evidence("thermal_state", "not_observable"))
    result = _validity(a, b)
    assert result.realization_guard.status is GuardStatus.UNRESOLVED
    assert result.realization_guard.unresolved[0]["condition"] == "thermal_state"
    assert result.overall_status is ContrastValidity.NOT_ESTABLISHED


def test_not_applicable_evidence_never_passes(tmp_path):
    a, b = _actual_pair(tmp_path, level_b="L1")
    for run in (a, b):
        run.manifest["realization"]["evidence"] = [_evidence(c, "not_applicable") for c in sorted(ALL_CONDITIONS)]
    result = _validity(a, b)
    assert result.realization_guard.status is GuardStatus.UNRESOLVED
    assert result.realization_guard.evidence_available == ()
    assert result.overall_status is ContrastValidity.NOT_ESTABLISHED and not result.attribution_allowed


def test_observed_evidence_requires_a_value(tmp_path):
    with pytest.raises(ValueError, match="requires a value"):
        RealizationEvidence(RealizedCondition.THERMAL_STATE, EvidenceValidity.OBSERVED, "x", "y", None,
                            "device", None)
    a, b = _actual_pair(tmp_path)
    for e in b.manifest["realization"]["evidence"]:
        e["value"] = None
    with pytest.raises(ValueError, match="requires a value"):
        _validity(a, b)


def test_evidence_from_different_sources_is_not_compared(tmp_path):
    a, b = _actual_pair(tmp_path)
    _replace_evidence(b, _evidence("data_movement", "observed", value={"v": 1}, source="tool-y"))
    result = _validity(a, b)
    assert result.realization_guard.status is GuardStatus.UNRESOLVED
    assert result.realization_guard.unresolved[0]["reason"] == "evidence_semantics_differ"


@pytest.mark.parametrize("value_a,value_b", [(True, 1), (1, 1.0)])
def test_evidence_values_compare_as_canonical_json(tmp_path, value_a, value_b):
    a, b = _actual_pair(tmp_path)
    _replace_evidence(a, _evidence("thermal_state", "observed", value={"v": value_a}))
    _replace_evidence(b, _evidence("thermal_state", "observed", value={"v": value_b}))
    result = _validity(a, b)
    assert result.realization_guard.status is GuardStatus.FAIL
    assert result.realization_guard.mismatches[0]["condition"] == "thermal_state"


def test_only_the_exact_equivalence_rule_is_supported(tmp_path):
    a, b = _actual_pair(tmp_path)
    for run in (a, b):
        _replace_evidence(run, _evidence("thermal_state", "observed", value={"celsius": 61.0},
                                         equivalence="abs_tolerance:1.0"))
    result = _validity(a, b)
    assert result.realization_guard.status is GuardStatus.UNRESOLVED
    assert result.realization_guard.unresolved == ({"condition": "thermal_state", "reason": "unsupported_equivalence_rule",
                                                    "equivalence": "abs_tolerance:1.0"},)


def test_duplicate_evidence_is_rejected(tmp_path):
    a, b = _actual_pair(tmp_path)
    b.manifest["realization"]["evidence"].append(_evidence("thermal_state", "observed", value={"v": 2}))
    with pytest.raises(ValueError, match="duplicate realization evidence"):
        _validity(a, b)


def test_manifest_requires_one_evidence_record_per_condition(tmp_path):
    m = _manifest(load(G1), tmp_path, "a")
    parse_strict(RunManifest, m)  # the runner's manifest is valid
    duplicate, missing = copy.deepcopy(m), copy.deepcopy(m)
    duplicate["realization"]["evidence"].append(duplicate["realization"]["evidence"][0])
    missing["realization"]["evidence"].pop()
    for bad in (duplicate, missing):
        with pytest.raises(ValueError, match="exactly one record per realized condition"):
            parse_strict(RunManifest, bad)


@pytest.mark.parametrize(
    "stress_value,validity,status,reason",
    [
        ({"verification": "self_reported", "coverage": "full"}, "observed", GuardStatus.UNRESOLVED,
         "stress_not_verified_over_the_full_window"),
        ({"verification": "verified", "coverage": "partial"}, "observed", GuardStatus.UNRESOLVED,
         "stress_not_verified_over_the_full_window"),
        ({"verification": "verified", "coverage": "full"}, "self_reported", GuardStatus.UNRESOLVED,
         "factor_evidence_not_independently_observed"),
        (None, "not_observable", GuardStatus.UNRESOLVED, "factor_evidence_not_observable"),
        ({"verification": "failed", "coverage": "full"}, "observed", GuardStatus.FAIL, "stress_not_realized"),
        ({"verification": "verified", "coverage": "none"}, "observed", GuardStatus.FAIL, "stress_not_realized"),
    ],
)
def test_intended_factor_must_be_realized_as_planned(tmp_path, stress_value, validity, status, reason):
    a, b = _actual_pair(tmp_path, level_b="L1")
    _replace_evidence(b, _evidence("stress_realization", validity, value=stress_value))
    result = _validity(a, b)
    assert result.realization_guard.intended_factor_conditions == ("stress_realization",)
    assert result.realization_guard.status is status
    entries = result.realization_guard.mismatches + result.realization_guard.unresolved
    assert {"condition": "stress_realization", "side": "run_b", "reason": reason, "validity": validity,
            "value": stress_value} in entries
    assert not result.attribution_allowed


def test_nominal_side_must_show_no_interference(tmp_path):
    a, b = _actual_pair(tmp_path, level_b="L1")
    _replace_evidence(a, _evidence("stress_realization", "observed", value={"nominal_check": "not_performed"}))
    result = _validity(a, b)
    assert result.realization_guard.unresolved == (
        {"condition": "stress_realization", "side": "run_a", "reason": "nominal_check_not_passed",
         "validity": "observed", "value": {"nominal_check": "not_performed"}},)
    assert result.overall_status is ContrastValidity.NOT_ESTABLISHED


class _ObserveFails:
    """Stress provider whose MEASURE-time observe step fails (coverage becomes partial)."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def observe(self):
        raise RuntimeError("observer lost")


def test_partially_covered_stress_is_reported_by_the_runner(tmp_path):
    def factory(rp):
        components = build_components(rp)
        return dataclasses.replace(components, stress=_ObserveFails(components.stress))

    base = load(G1)
    nominal = _run(_level(base, "none"), tmp_path, "nominal")
    partial = _run(_level(base, "L1"), tmp_path, "partial", factory=factory)
    assert partial.manifest["stress"]["status"]["coverage"] == "partial"
    assert partial.manifest["status"]["measurement_status"] == "valid"
    result = _validity(nominal, partial)
    assert {"condition": "stress_realization", "side": "run_b", "reason": "stress_not_verified_over_the_full_window",
            "validity": "virtual", "value": {"coverage": "partial", "verification": "self_reported"}
            } in result.realization_guard.unresolved
    assert not result.attribution_allowed


@pytest.mark.parametrize(
    "where,value,basis",
    [("realization", "mock", "mock_execution"), ("status", "mock", "mock_execution"),
     ("top", "emulated", "execution_kind_not_actual")],
)
def test_mock_detection_fails_closed(tmp_path, where, value, basis):
    a, b = _actual_pair(tmp_path, level_b="L1")
    (b.manifest if where == "top" else b.manifest[where])["execution_kind"] = value
    result = _validity(a, b)
    assert result.realization_guard.status is GuardStatus.NOT_OBSERVABLE
    assert result.realization_guard.basis == basis
    assert not result.attribution_allowed


def test_unknown_identity_is_unresolved_unless_both_runs_are_mock(tmp_path):
    a, b = _actual_pair(tmp_path, level_b="L1")
    for run in (a, b):
        run.manifest["comparison_keys"]["components"]["common_parts"]["model_artifact"] = None
    result = _validity(a, b)
    assert result.design_guard.status is GuardStatus.UNRESOLVED
    assert result.design_guard.unknown_identities == ("model_artifact",)
    assert result.overall_status is ContrastValidity.NOT_ESTABLISHED and not result.attribution_allowed


def test_pairwise_interaction_is_never_attributed(tmp_path):
    base = load(SCENARIO)
    a = _actualize(_run(_level(_input(base, "scenario-ep-a.yaml"), "none"), tmp_path, "a"))
    b = _actualize(_run(_level(_input(base, "scenario-ep-b.yaml"), "L1"), tmp_path, "b"))
    result = _validity(a, b, "interaction")
    assert result.design_guard.status is GuardStatus.PASS  # both cells belong to one interaction family
    assert INTERACTION_NOTE in result.design_guard.notes
    assert result.realization_guard.status is GuardStatus.PASS
    assert result.measurement_guard.status is GuardStatus.PASS
    assert result.overall_status is ContrastValidity.NOT_ESTABLISHED and not result.attribution_allowed
    # the same diagonal pair is never a single-factor contrast either
    for contrast in ("stress", "scenario"):
        assert _validity(a, b, contrast).overall_status is ContrastValidity.INVALID


def test_same_level_with_differently_realized_stress_fails(tmp_path):
    # stress_realization is exempt only when the level value varied; here it did not
    a = _manifest(_level(load(G1), "L1"), tmp_path, "a")
    b = copy.deepcopy(a)
    b["run_id"] = "run-b"
    for e in b["realization"]["evidence"]:
        if e["condition"] == "stress_realization":
            e["value"] = {**e["value"], "coverage": "partial"}
    result = evaluate_contrast_validity(a, b, "stress")
    assert result.realization_guard.intended_factor_conditions == ()
    assert result.realization_guard.mismatches[0]["condition"] == "stress_realization"


def test_validity_result_contract():
    with pytest.raises(ValueError, match="attribution is allowed only"):
        ContrastValidityResult(1, ContrastType.STRESS, "a", "b", None, None, None,  # type: ignore[arg-type]
                               ContrastValidity.NOT_ESTABLISHED, True)
    with pytest.raises(ValueError, match="carries no value"):
        RealizationEvidence(RealizedCondition.THERMAL_STATE, EvidenceValidity.NOT_OBSERVABLE, "x", "y", None,
                            "device", None, {"t": 1})
