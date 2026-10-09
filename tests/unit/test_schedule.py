"""WP2 gate: exact schedule arithmetic for fixed_rate / recorded / asap."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from sdv_bmt.contracts.records import InputItem, ScenarioCoordinates, SourceTiming
from sdv_bmt.contracts.specs import ReplayPolicy, parse_strict
from sdv_bmt.contracts.status import InputKind, ReplayMode, TimeUnit, TimingSemantic
from sdv_bmt.runner.schedule import ScheduleError, compute_schedule
from tests._helpers import CONFIGS, ZERO_SHA, load_input_provider

SCN = InputKind.DRIVING_SCENARIO


def _replay(**kw) -> ReplayPolicy:
    return parse_strict(ReplayPolicy, kw)


def _items(source_times, unit=TimeUnit.US, episodes=None, clock="clk"):
    items = []
    for i, value in enumerate(source_times):
        episode = episodes[i] if episodes else "e"
        items.append(
            InputItem(
                item_index=i, source_item_id=f"i{i}", payload_ref=f"x#{i}", payload_sha256=ZERO_SHA,
                annotation_ref=None, annotation_digest=None,
                source_timing=None if value is None else SourceTiming(value, unit, TimingSemantic.CAPTURE, clock),
                scenario_coordinates=ScenarioCoordinates(episode_id=episode),
                source_metadata={},
            )
        )
    return items


# ---------------------------------------------------------------- fixed_rate


def test_fixed_rate_30hz_floor():
    s = compute_schedule(_items([None] * 4), _replay(mode="fixed_rate", rate_hz=30), SCN)
    assert s.offsets_ns == (0, 33_333_333, 66_666_666, 100_000_000)


def test_fixed_rate_20hz_exact_multiples():
    s = compute_schedule(_items([None] * 8), _replay(mode="fixed_rate", rate_hz=20), SCN)
    assert s.offsets_ns == tuple(i * 50_000_000 for i in range(8))


def test_fixed_rate_uses_exact_decimal_rate():
    s = compute_schedule(_items([None] * 5), _replay(mode="fixed_rate", rate_hz=29.97), SCN)
    expected = tuple(math.floor(Fraction(i * 10**9) / Fraction("29.97")) for i in range(5))
    assert s.offsets_ns == expected
    assert s.offsets_ns[1] == 33_366_700  # 10**9 * 100 / 2997 = 33366700.03...


def test_fixed_rate_12_5hz():
    s = compute_schedule(_items([None] * 3), _replay(mode="fixed_rate", rate_hz=12.5), SCN)
    assert s.offsets_ns == (0, 80_000_000, 160_000_000)


# ---------------------------------------------------------------- recorded


def test_recorded_g2_source_times_map_exactly():
    items = load_input_provider(CONFIGS / "inputs" / "scenario-g2-overload.yaml").items()
    s = compute_schedule(items, _replay(mode="recorded"), SCN)
    assert s.offsets_ns == (0, 33_000_000, 66_000_000, 99_000_000)


def test_recorded_jitter_reproduced_with_unit_conversion():
    items = load_input_provider(CONFIGS / "inputs" / "scenario-ep-a.yaml").items()
    s = compute_schedule(items, _replay(mode="recorded"), SCN)
    assert s.offsets_ns == (0, 33_412_000, 66_590_000, 100_105_000, 133_301_000, 166_874_000)


@pytest.mark.parametrize(
    "unit,factor", [(TimeUnit.NS, 1), (TimeUnit.US, 1_000), (TimeUnit.MS, 1_000_000), (TimeUnit.S, 10**9)]
)
def test_recorded_units(unit, factor):
    s = compute_schedule(_items([7, 10, 15], unit=unit), _replay(mode="recorded"), SCN)
    assert s.offsets_ns == (0, 3 * factor, 8 * factor)


def test_recorded_speed_scales_with_floor():
    items = _items([0, 33_333, 66_667])  # us
    assert compute_schedule(items, _replay(mode="recorded", speed=2.0), SCN).offsets_ns == (
        0, 16_666_500, 33_333_500
    )
    assert compute_schedule(_items([0, 1, 3], unit=TimeUnit.NS), _replay(mode="recorded", speed=2.0), SCN).offsets_ns == (0, 0, 1)
    assert compute_schedule(items, _replay(mode="recorded", speed=0.5), SCN).offsets_ns == (
        0, 66_666_000, 133_334_000
    )


def test_recorded_episode_gap_and_restart():
    items = _items([100, 133, 200, 5_000, 5_040], unit=TimeUnit.MS, episodes=["a", "a", "a", "b", "b"])
    s = compute_schedule(items, _replay(mode="recorded", episode_gap_ns=250_000_000), SCN)
    assert s.offsets_ns == (0, 33_000_000, 100_000_000, 350_000_000, 390_000_000)
    assert s.episode_starts == (0, 3)


def test_recorded_episode_gap_is_not_scaled_by_speed():
    items = _items([0, 100, 0, 100], unit=TimeUnit.MS, episodes=["a", "a", "b", "b"])
    s = compute_schedule(items, _replay(mode="recorded", speed=2.0, episode_gap_ns=10_000_000), SCN)
    assert s.offsets_ns == (0, 50_000_000, 60_000_000, 110_000_000)


def test_recorded_multi_episode_requires_gap():
    items = _items([0, 10, 0, 10], episodes=["a", "a", "b", "b"])
    with pytest.raises(ScheduleError, match="episode_gap_ns"):
        compute_schedule(items, _replay(mode="recorded"), SCN)


def test_recorded_two_episode_fixture():
    items = load_input_provider(CONFIGS / "inputs" / "scenario-ep-a-ep-b.yaml").items()
    s = compute_schedule(items, _replay(mode="recorded", episode_gap_ns=100_000_000), SCN)
    assert s.offsets_ns[:6] == (0, 33_412_000, 66_590_000, 100_105_000, 133_301_000, 166_874_000)
    assert s.offsets_ns[6:] == (266_874_000, 300_074_000, 333_614_000, 366_824_000, 400_354_000)


@pytest.mark.parametrize("times", [[0, 33, 30], [0, 33, 33]])
def test_recorded_regression_or_duplicate_rejected(times):
    with pytest.raises(ScheduleError, match="regression or duplicate"):
        compute_schedule(_items(times), _replay(mode="recorded"), SCN)


def test_recorded_rejects_missing_timing_and_mixed_clocks():
    with pytest.raises(ScheduleError, match="requires source timing"):
        compute_schedule(_items([0, None]), _replay(mode="recorded"), SCN)
    items = _items([0, 10])
    items[1] = InputItem(**{**items[1].__dict__, "source_timing": SourceTiming(10, TimeUnit.US, TimingSemantic.CAPTURE, "other")})
    with pytest.raises(ScheduleError, match="single source clock"):
        compute_schedule(items, _replay(mode="recorded"), SCN)


def test_recorded_rejected_for_dataset_input():
    with pytest.raises(ScheduleError, match="not allowed for dataset"):
        compute_schedule(_items([0, 1]), _replay(mode="recorded"), InputKind.DATASET)


# ---------------------------------------------------------------- asap / general


def test_asap_has_no_precomputed_offsets_and_keeps_lead():
    s = compute_schedule(_items([None, None]), _replay(mode="asap", lead_ns=5), SCN)
    assert s.mode is ReplayMode.ASAP and s.offsets_ns is None and s.lead_ns == 5


def test_schedule_is_reproducible():
    items = load_input_provider(CONFIGS / "inputs" / "scenario-ep-a-ep-b.yaml").items()
    replay = _replay(mode="recorded", episode_gap_ns=1)
    assert compute_schedule(items, replay, SCN) == compute_schedule(items, replay, SCN)


def test_item_indices_must_be_contiguous():
    items = _items([0, 1])
    items[1] = InputItem(**{**items[1].__dict__, "item_index": 5})
    with pytest.raises(ScheduleError, match="0..n-1"):
        compute_schedule(items, _replay(mode="asap"), SCN)
