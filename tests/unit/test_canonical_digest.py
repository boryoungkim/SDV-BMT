"""WP1 gate: canonical JSON and digest determinism; content-based identities."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from sdv_bmt.contracts.ids import (
    CanonicalJSONError,
    canonical_json_bytes,
    digest_of,
    loads_strict_json,
    pretty_json_bytes,
    sha256_hex,
)
from sdv_bmt.contracts.records import (
    InputItem,
    ScenarioCoordinates,
    SourceTiming,
    item_digest,
    items_digest,
)
from sdv_bmt.contracts.specs import (
    HardwareStressSpec,
    input_family_digest,
    input_selection_digest,
    parse_input_spec,
    parse_strict,
    stress_axis_digest,
    stress_level_value,
)
from sdv_bmt.contracts.status import TimeUnit, TimingSemantic
from tests._helpers import ZERO_SHA, dataset_spec_dict, scenario_spec_dict, stress_spec_dict

SAMPLE = {"b": [1, 2.5, None, True], "a": {"z": "한글", "y": -3}, "c": "x"}


def test_canonical_json_format():
    assert canonical_json_bytes(SAMPLE) == (
        '{"a":{"y":-3,"z":"한글"},"b":[1,2.5,null,true],"c":"x"}'.encode("utf-8")
    )


def test_digest_independent_of_key_order():
    reordered = {"c": "x", "a": {"y": -3, "z": "한글"}, "b": [1, 2.5, None, True]}
    assert digest_of(SAMPLE) == digest_of(reordered)
    assert digest_of(SAMPLE) == sha256_hex(canonical_json_bytes(SAMPLE))


def test_digest_stable_across_processes_and_hash_seeds():
    expected = digest_of(SAMPLE)
    code = (
        "import json,sys;from sdv_bmt.contracts.ids import digest_of;"
        "print(digest_of(json.loads(sys.argv[1])))"
    )
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run(
            [sys.executable, "-c", code, json.dumps(SAMPLE)],
            capture_output=True, text=True, check=True, env=env,
        ).stdout.strip()
        assert out == expected


@pytest.mark.parametrize(
    "bad", [{"x": float("nan")}, {"x": float("inf")}, {1: "a"}, {"x": {1, 2}}, {"x": b"raw"}]
)
def test_canonical_json_rejects_ambiguous_values(bad):
    with pytest.raises(CanonicalJSONError):
        canonical_json_bytes(bad)


def test_dataclass_and_enum_serialization():
    timing = SourceTiming(5, TimeUnit.MS, TimingSemantic.CAPTURE, "clk")
    assert canonical_json_bytes(timing) == (
        b'{"clock_source":"clk","semantic":"capture","unit":"ms","value":5}'
    )


def test_pretty_json_is_deterministic():
    assert pretty_json_bytes(SAMPLE) == pretty_json_bytes(dict(reversed(list(SAMPLE.items()))))
    assert pretty_json_bytes(SAMPLE).endswith(b"\n")


def test_loads_strict_json_rejects_duplicates_and_nan():
    assert loads_strict_json('{"a":1}') == {"a": 1}
    with pytest.raises(ValueError, match="duplicate JSON key"):
        loads_strict_json('{"a":1,"a":2}')
    with pytest.raises(ValueError, match="non-finite"):
        loads_strict_json('{"a":NaN}')


# ---------------------------------------------------------------- input identity


def test_input_family_digest_is_label_independent_but_content_sensitive():
    base = parse_input_spec(scenario_spec_dict())
    relabeled = parse_input_spec(scenario_spec_dict(input_id="other-name", version="7"))
    assert input_family_digest(base) == input_family_digest(relabeled)

    for mutate in (
        lambda d: d["family"].__setitem__("source_version", "2"),
        lambda d: d["family"]["timing"].__setitem__("semantic", "logging"),
        lambda d: d["family"].__setitem__("preprocessing_contract", "other"),
        lambda d: d["family"].__setitem__("streams", ["cam_front", "cam_rear"]),
        lambda d: d["family"].__setitem__("task", "other-task"),
    ):
        data = scenario_spec_dict()
        mutate(data)
        assert input_family_digest(parse_input_spec(data)) != input_family_digest(base)


def test_family_excludes_index_digest_but_selection_pins_it():
    # F.6: the family is source dataset/version + schema/timing/task/preprocess
    # semantics; the concrete index a selection was resolved against is part of
    # the selection identity.
    base = parse_input_spec(scenario_spec_dict())
    other_index = parse_input_spec(scenario_spec_dict(source={"index": "x.jsonl", "index_sha256": "1" * 64}))
    assert input_family_digest(base) == input_family_digest(other_index)
    resolved = {"stream_id": "cam_front", "episodes": ["ep-a"]}
    assert input_selection_digest(base, resolved) != input_selection_digest(other_index, resolved)


def test_selection_digest_follows_the_resolved_selection():
    spec = parse_input_spec(scenario_spec_dict())
    a = input_selection_digest(spec, {"stream_id": "cam_front", "episodes": ["ep-a"]})
    assert a == input_selection_digest(spec, {"episodes": ["ep-a"], "stream_id": "cam_front"})
    assert a != input_selection_digest(spec, {"stream_id": "cam_front", "episodes": ["ep-b"]})
    assert a != input_selection_digest(spec, {"stream_id": "cam_front", "episodes": ["ep-a", "ep-b"]})


def test_equivalent_selection_spellings_share_the_selection_digest(tmp_path):
    from tests._helpers import dataset_provider

    rows = [{"source_item_id": f"s{i}", "payload_text": f"p{i}"} for i in range(5)]
    explicit = dataset_provider(tmp_path, rows, start=0, count=5).describe()
    implicit = dataset_provider(tmp_path, rows, start=0, count=None).describe()
    partial = dataset_provider(tmp_path, rows, start=0, count=4).describe()
    assert explicit.items_digest == implicit.items_digest
    assert explicit.input_selection_digest == implicit.input_selection_digest
    assert partial.input_selection_digest != explicit.input_selection_digest


def test_dataset_and_scenario_families_never_collide():
    ds = parse_input_spec(dataset_spec_dict())
    sc = parse_input_spec(scenario_spec_dict())
    assert input_family_digest(ds) != input_family_digest(sc)


def _item(**kw) -> InputItem:
    base = dict(
        item_index=0, source_item_id="a", payload_ref="idx#a", payload_sha256=ZERO_SHA,
        annotation_ref=None, annotation_digest=None, source_timing=None,
        scenario_coordinates=None, source_metadata={},
    )
    base.update(kw)
    return InputItem(**base)


def test_item_digest_covers_identity_not_location_or_metadata():
    a = _item()
    assert item_digest(a) == item_digest(_item(payload_ref="elsewhere#a", source_metadata={"k": 1}))
    assert item_digest(a) != item_digest(_item(source_item_id="b"))
    assert item_digest(a) != item_digest(_item(payload_sha256="1" * 64))
    timed = _item(source_timing=SourceTiming(1, TimeUnit.US, TimingSemantic.CAPTURE, "c"))
    assert item_digest(timed) != item_digest(a)
    assert item_digest(timed) != item_digest(
        _item(source_timing=SourceTiming(2, TimeUnit.US, TimingSemantic.CAPTURE, "c"))
    )
    coords = _item(scenario_coordinates=ScenarioCoordinates(episode_id="e"))
    assert item_digest(coords) != item_digest(a)


def test_items_digest_is_order_sensitive_and_reproducible():
    items = [_item(item_index=0, source_item_id="a"), _item(item_index=1, source_item_id="b")]
    assert items_digest(items) == items_digest(list(items))
    swapped = [_item(item_index=0, source_item_id="b"), _item(item_index=1, source_item_id="a")]
    assert items_digest(items) != items_digest(swapped)


# ---------------------------------------------------------------- stress identity


def test_stress_axis_digest_excludes_levels_and_label_only():
    base = parse_strict(HardwareStressSpec, stress_spec_dict())
    same_axis = parse_strict(
        HardwareStressSpec, stress_spec_dict(stress_id="renamed", levels={"L1": 0.3})
    )
    assert stress_axis_digest(base) == stress_axis_digest(same_axis)
    for override in (
        {"kind": "other_kind"},
        {"target_resource": "host-cpu"},
        {"placement": "core-2-3"},
        {"control": {"variable": "duty_cycle", "unit": "ratio"}},
        {"background_workload": {"artifact": "x", "sha256": ZERO_SHA}},
        {"verification": {"window_ns": 5, "achieved_min_fraction": 0.9, "independent": None}},
        {"provider_options": {"fail_phase": "verify"}},
    ):
        other = parse_strict(HardwareStressSpec, stress_spec_dict(**override))
        assert stress_axis_digest(other) != stress_axis_digest(base), override


def test_stress_level_value_lookup():
    spec = parse_strict(HardwareStressSpec, stress_spec_dict())
    assert stress_level_value(spec, "none") is None
    assert stress_level_value(spec, "L2") == 0.5
    with pytest.raises(ValueError):
        stress_level_value(spec, "L9")
