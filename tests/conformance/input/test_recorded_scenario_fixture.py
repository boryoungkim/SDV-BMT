"""Recorded scenario fixture conformance (S1A-3): typed timing, no invented metadata."""

from __future__ import annotations

import pytest

from sdv_bmt.contracts.protocols import InputValidationError
from sdv_bmt.contracts.records import ScenarioCoordinates, SourceTiming
from sdv_bmt.contracts.specs import ReplayPolicy, parse_strict
from sdv_bmt.contracts.status import InputKind, TimeUnit, TimingSemantic
from sdv_bmt.runner.schedule import ScheduleError, compute_schedule
from tests._helpers import CONFIGS, frame_rows, load_input_provider, scenario_provider


def test_typed_source_timing_and_coordinates_from_fixture():
    items = load_input_provider(CONFIGS / "inputs" / "scenario-ep-a.yaml").items()
    assert len(items) == 6
    first = items[0]
    assert first.source_timing == SourceTiming(
        1_727_000_000_000_000, TimeUnit.US, TimingSemantic.CAPTURE, "fixture-camera-clock"
    )
    assert first.scenario_coordinates == ScenarioCoordinates("ep-a", "cam_front", "cam0", "000000")
    assert [i.source_timing.value - first.source_timing.value for i in items] == [
        0, 33_412, 66_590, 100_105, 133_301, 166_874
    ]


def test_stream_filter_excludes_other_streams():
    items = load_input_provider(CONFIGS / "inputs" / "scenario-ep-a-ep-b.yaml").items()
    assert {i.scenario_coordinates.stream_id for i in items} == {"cam_front"}
    assert [i.scenario_coordinates.episode_id for i in items] == ["ep-a"] * 6 + ["ep-b"] * 5


def test_absent_coordinates_and_timing_are_not_invented(tmp_path):
    rows = [
        {"source_item_id": "x0", "payload_text": "a", "stream_id": "cam_front", "t_capture_us": 10},
        {"source_item_id": "x1", "payload_text": "b", "stream_id": "cam_front"},
        {"source_item_id": "x2", "payload_text": "c"},
    ]
    provider = scenario_provider(tmp_path, rows, selection={"stream_id": None})
    with pytest.raises(InputValidationError, match="multiple streams"):
        provider.describe()

    provider = scenario_provider(tmp_path, rows[:2])
    items = provider.items()
    assert items[0].scenario_coordinates == ScenarioCoordinates(stream_id="cam_front")
    assert items[0].scenario_coordinates.episode_id is None
    assert items[0].scenario_coordinates.sensor_id is None
    assert items[0].source_timing.value == 10
    assert items[1].source_timing is None
    assert provider.describe().source_timing_complete is False


def test_rows_without_any_coordinates_have_null_coordinates(tmp_path):
    rows = [{"source_item_id": "x0", "payload_text": "a", "t_capture_us": 1}]
    items = scenario_provider(tmp_path, rows, selection={"stream_id": None}).items()
    assert items[0].scenario_coordinates is None


def test_family_without_timing_yields_no_timing(tmp_path):
    rows = frame_rows("e", [0, 10])
    for row in rows:
        row.pop("t_capture_us")
    items = scenario_provider(tmp_path, rows, family={"timing": None}).items()
    assert all(i.source_timing is None for i in items)


def test_timestamp_regression_fixture_rejected():
    provider = load_input_provider(CONFIGS / "inputs" / "scenario-regression.yaml")
    with pytest.raises(InputValidationError, match="regression.*never re-sorted"):
        provider.describe()


def test_duplicate_timestamp_rejected(tmp_path):
    with pytest.raises(InputValidationError, match="duplicate"):
        scenario_provider(tmp_path, frame_rows("e", [0, 33_000, 33_000])).describe()


def test_timestamps_restart_per_episode_are_allowed(tmp_path):
    rows = frame_rows("e1", [0, 10, 20]) + frame_rows("e2", [0, 10], base=500)
    assert len(scenario_provider(tmp_path, rows).items()) == 5


def test_non_integer_timestamp_rejected(tmp_path):
    rows = frame_rows("e", [0, 1])
    rows[1]["t_capture_us"] = 1.5
    with pytest.raises(InputValidationError, match="integer timestamp"):
        scenario_provider(tmp_path, rows).describe()


def test_undeclared_stream_rejected(tmp_path):
    with pytest.raises(InputValidationError, match="not declared"):
        scenario_provider(tmp_path, frame_rows("e", [0], stream="lidar")).describe()


def test_episode_selection_order_and_missing_episode(tmp_path):
    rows = frame_rows("e1", [0, 10]) + frame_rows("e2", [0, 10])
    items = scenario_provider(tmp_path, rows, selection={"stream_id": "cam_front", "episodes": ["e2", "e1"]}).items()
    assert [i.scenario_coordinates.episode_id for i in items] == ["e2", "e2", "e1", "e1"]
    with pytest.raises(InputValidationError, match="'e9' has no rows"):
        scenario_provider(tmp_path, rows, selection={"stream_id": "cam_front", "episodes": ["e9"]}).describe()


def test_non_contiguous_episodes_rejected_without_explicit_selection(tmp_path):
    rows = frame_rows("e1", [0]) + frame_rows("e2", [0]) + frame_rows("e1", [50])
    rows[2]["source_item_id"] = "e1/late"
    with pytest.raises(InputValidationError, match="not contiguous"):
        scenario_provider(tmp_path, rows).describe()


@pytest.mark.parametrize("mode,extra", [("recorded", {}), ("fixed_rate", {"rate_hz": 30}), ("asap", {})])
def test_scenario_supports_all_replay_modes(mode, extra):
    items = load_input_provider(CONFIGS / "inputs" / "scenario-ep-a.yaml").items()
    schedule = compute_schedule(items, parse_strict(ReplayPolicy, {"mode": mode, **extra}), InputKind.DRIVING_SCENARIO)
    assert schedule.mode.value == mode


def test_recorded_replay_requires_complete_timing(tmp_path):
    items = scenario_provider(tmp_path, frame_rows("e", [0, None, 20])).items()
    with pytest.raises(ScheduleError, match="requires source timing for every item"):
        compute_schedule(items, parse_strict(ReplayPolicy, {"mode": "recorded"}), InputKind.DRIVING_SCENARIO)
    fixed = compute_schedule(items, parse_strict(ReplayPolicy, {"mode": "fixed_rate", "rate_hz": 10}),
                             InputKind.DRIVING_SCENARIO)
    assert fixed.offsets_ns == (0, 100_000_000, 200_000_000)


def test_scenario_digest_reproducible_and_selection_sensitive():
    a1 = load_input_provider(CONFIGS / "inputs" / "scenario-ep-a.yaml").describe()
    a2 = load_input_provider(CONFIGS / "inputs" / "scenario-ep-a.yaml").describe()
    b = load_input_provider(CONFIGS / "inputs" / "scenario-ep-b.yaml").describe()
    assert a1 == a2
    assert a1.input_family_digest == b.input_family_digest
    assert a1.input_selection_digest != b.input_selection_digest
    assert a1.items_digest != b.items_digest
