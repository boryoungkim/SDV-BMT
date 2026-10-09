"""S1A-8: stress-axis and input-family identities keep wrong comparisons apart.

Only the stress level (stress contrast) or the scenario selection (scenario
contrast) may vary; every other factor must be identical for keys to match.
"""

from __future__ import annotations

import dataclasses

import pytest

from sdv_bmt.cli.main import build_components
from sdv_bmt.contracts.specs import ReplayPolicy, StressControlSpec, parse_input_spec, parse_strict
from sdv_bmt.contracts.status import GuardStatus
from sdv_bmt.runner.plan import load_yaml_strict
from sdv_bmt.runner.runner import BenchmarkRunner
from tests._helpers import CONFIGS, design_check, load, run_plan, variant


def keys(rp):
    return BenchmarkRunner(rp, build_components).preflight().comparison_keys


def with_level(rp, level):
    return variant(rp, plan={"stress": dataclasses.replace(rp.plan.stress, level=level)})


def with_input(rp, spec_file):
    path = CONFIGS / "inputs" / spec_file
    return variant(rp, input_spec=parse_input_spec(load_yaml_strict(path)), input_spec_dir=path.parent)


def with_axis(rp, **changes):
    return variant(rp, stress_spec=dataclasses.replace(rp.stress_spec, **changes))


SCENARIO = "scenario-two-episodes-fake-stress"


def _consistent(result, key_a, key_b):
    # Design Guard PASS <=> equal contrast key <=> no violation
    assert (result.status is GuardStatus.PASS) == (key_a == key_b)
    assert (result.status is GuardStatus.PASS) == (result.violations == ())


# ---------------------------------------------------------------- stress contrast


def test_stress_levels_on_one_axis_share_the_stress_contrast_key():
    base = load(SCENARIO)
    k = {lvl: keys(with_level(base, lvl)) for lvl in ("none", "L1", "L2", "L3")}
    assert len({v.key_stress_contrast for v in k.values()}) == 1
    assert len({v.cell_digest for v in k.values()}) == 4
    result = design_check(k["none"], k["L3"], "stress")
    assert result.status is GuardStatus.PASS
    assert result.observed_differences == ("stress_level", "stress_level_value")
    _consistent(result, k["none"].key_stress_contrast, k["L3"].key_stress_contrast)


@pytest.mark.parametrize(
    "change",
    [
        {"placement": "other-virtual-placement"},
        {"target_resource": "virtual-other"},
        {"kind": "fake_virtual_v2"},
        {"control": StressControlSpec("duty", "ratio")},
        {"provider_options": {"achieved_fraction": 0.95}},
    ],
)
def test_different_stress_axis_never_shares_a_stress_contrast(change):
    # Nominal (level none) runs use NoStress, so any axis spec yields keys.
    base = load(SCENARIO)
    a = keys(with_level(base, "none"))
    b = keys(with_level(with_axis(base, **change), "none"))
    assert a.key_stress_contrast != b.key_stress_contrast
    result = design_check(a, b, "stress")
    assert result.status is GuardStatus.FAIL and result.violations == ("stress_axis_digest",)
    _consistent(result, a.key_stress_contrast, b.key_stress_contrast)


def test_nominal_runs_of_different_axes_are_distinct():
    base = load(SCENARIO)
    a = keys(with_level(base, "none"))
    b = keys(with_level(with_axis(base, placement="elsewhere"), "none"))
    assert a.components["stress_level"] == b.components["stress_level"] == "none"
    assert a.cell_digest != b.cell_digest and a.key_stress_contrast != b.key_stress_contrast


def test_stress_contrast_requires_same_input_selection():
    base = load(SCENARIO)
    a = keys(with_input(base, "scenario-ep-a.yaml"))
    b = keys(with_level(with_input(base, "scenario-ep-b.yaml"), "L2"))
    result = design_check(a, b, "stress")
    assert result.status is GuardStatus.FAIL
    assert set(result.violations) == {"input_selection_digest", "items_digest"}


# ---------------------------------------------------------------- scenario contrast


def test_scenario_selections_of_one_family_share_the_scenario_key():
    base = load(SCENARIO)
    a = keys(with_input(base, "scenario-ep-a.yaml"))
    b = keys(with_input(base, "scenario-ep-b.yaml"))
    assert a.components["input_family_digest"] == b.components["input_family_digest"]
    assert a.key_scenario_contrast == b.key_scenario_contrast
    assert a.key_stress_contrast != b.key_stress_contrast
    result = design_check(a, b, "scenario")
    assert result.status is GuardStatus.PASS
    assert result.observed_differences == ("input_selection_digest", "items_digest")


def test_scenario_contrast_rejects_different_family():
    base = load(SCENARIO)
    a = keys(with_input(base, "scenario-ep-a.yaml"))
    spec = parse_input_spec(load_yaml_strict(CONFIGS / "inputs" / "scenario-ep-b.yaml"))
    other_family = dataclasses.replace(spec, family=dataclasses.replace(spec.family, source_version="2"))
    b = keys(variant(base, input_spec=other_family, input_spec_dir=CONFIGS / "inputs"))
    assert a.key_scenario_contrast != b.key_scenario_contrast
    result = design_check(a, b, "scenario")
    assert result.status is GuardStatus.FAIL and "input_family_digest" in result.violations


def test_scenario_contrast_requires_same_stress_level():
    base = load(SCENARIO)
    a = keys(with_input(base, "scenario-ep-a.yaml"))
    b = keys(with_level(with_input(base, "scenario-ep-b.yaml"), "L3"))
    assert design_check(a, b, "scenario").status is GuardStatus.FAIL


def test_dataset_input_has_no_scenario_or_interaction_key():
    k = keys(load("g1-dataset-fixed-rate-nominal"))
    assert k.key_scenario_contrast is None and k.key_interaction is None
    result = design_check(k, k, "scenario")
    assert result.status is GuardStatus.NOT_APPLICABLE and "driving_scenario" in result.notes[0]


# ---------------------------------------------------------------- interaction / common identity


def test_interaction_cells_share_the_interaction_key():
    base = load(SCENARIO)
    cells = [keys(with_level(with_input(base, f), lvl))
             for f in ("scenario-ep-a.yaml", "scenario-ep-b.yaml") for lvl in ("none", "L2")]
    assert len({c.key_interaction for c in cells}) == 1
    assert len({c.cell_digest for c in cells}) == 4


def test_replay_policy_change_breaks_every_key():
    base = load(SCENARIO)
    a = keys(base)
    b = keys(variant(base, plan={"replay": parse_strict(ReplayPolicy, {"mode": "recorded", "speed": 2.0,
                                                                         "episode_gap_ns": 100_000_000})}))
    assert a.components["common_fixed_identity"] != b.components["common_fixed_identity"]
    for field in ("cell_digest", "key_stress_contrast", "key_scenario_contrast", "key_interaction"):
        assert getattr(a, field) != getattr(b, field)
    assert design_check(a, b, "stress").violations == ("replay",)


def test_labels_do_not_change_identity():
    base = load(SCENARIO)
    relabeled = variant(base,
                        stress_spec=dataclasses.replace(base.stress_spec, stress_id="renamed-axis"),
                        dut=dataclasses.replace(base.dut, dut_config_id="renamed-dut"),
                        workload=dataclasses.replace(base.workload, workload_id="renamed-workload"))
    assert keys(base) == keys(relabeled)


def test_manifest_records_the_preflight_keys(tmp_path):
    rp = load(SCENARIO)
    out = run_plan(rp, tmp_path)
    from sdv_bmt.contracts.ids import to_jsonable
    from sdv_bmt.evidence.manifest import read_manifest

    assert read_manifest(out.run_dir)["comparison_keys"] == to_jsonable(keys(rp))


def test_input_pipeline_is_part_of_the_common_fixed_identity():
    base = load(SCENARIO)
    k = keys(base)
    common = k.components["common_parts"]
    assert "input_pipeline" in common and "preprocess_semantics_id" in common
    assert common["input_pipeline"] != common["workload_identity"]
