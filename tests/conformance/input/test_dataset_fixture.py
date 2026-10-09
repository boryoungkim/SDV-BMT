"""Dataset fixture conformance (S1A-2): deterministic samples, no fake timing."""

from __future__ import annotations

import pytest

from sdv_bmt.contracts.ids import digest_of
from sdv_bmt.contracts.protocols import InputValidationError
from sdv_bmt.contracts.specs import parse_strict, ReplayPolicy
from sdv_bmt.contracts.status import InputKind
from sdv_bmt.runner.schedule import ScheduleError, compute_schedule
from tests._helpers import CONFIGS, dataset_provider, load_input_provider


def _rows(n, **extra):
    return [{"source_item_id": f"s{i}", "payload_text": f"p{i}", **extra} for i in range(n)]


def test_items_carry_no_timing_or_scenario_coordinates():
    provider = load_input_provider(CONFIGS / "inputs" / "dataset-fixture-128.yaml")
    items = provider.items()
    assert len(items) == 128
    assert all(i.source_timing is None and i.scenario_coordinates is None for i in items)
    assert provider.describe().input_kind is InputKind.DATASET
    assert provider.describe().source_timing_complete is False


def test_index_order_and_selection_window():
    provider = load_input_provider(CONFIGS / "inputs" / "dataset-fixture-8.yaml")
    assert [i.source_item_id for i in provider.items()] == [f"ds-{k:04d}" for k in range(8)]
    assert provider.items()[3].payload_ref == "dataset-fixture-v1.index.jsonl#ds-0003"
    assert provider.items()[3].source_metadata == {"index_line": 4}


def test_annotation_digest_is_content_digest_or_null(tmp_path):
    rows = [
        {"source_item_id": "a", "payload_text": "x", "annotation": {"label": 3}},
        {"source_item_id": "b", "payload_text": "y"},
        {"source_item_id": "c", "payload_text": "z", "annotation": None},
    ]
    items = dataset_provider(tmp_path, rows).items()
    assert items[0].annotation_digest == digest_of({"label": 3})
    assert items[0].annotation_ref == "src.index.jsonl#a/annotation"
    assert (items[1].annotation_ref, items[1].annotation_digest) == (None, None)
    assert (items[2].annotation_ref, items[2].annotation_digest) == (None, None)


def test_same_source_and_spec_give_same_items_digest(tmp_path):
    a = dataset_provider(tmp_path, _rows(5), start=1, count=3).describe()
    b = dataset_provider(tmp_path, _rows(5), start=1, count=3).describe()
    c = dataset_provider(tmp_path, _rows(5), start=0, count=3).describe()
    assert a.items_digest == b.items_digest
    assert a.input_selection_digest == b.input_selection_digest
    assert a.items_digest != c.items_digest and a.input_selection_digest != c.input_selection_digest
    assert a.input_family_digest == c.input_family_digest


@pytest.mark.parametrize(
    "rows,selection,message",
    [
        (_rows(3), {"start": 3, "count": None}, "beyond"),
        (_rows(3), {"start": 1, "count": 3}, "requests 3 items"),
        (_rows(2) + [{"source_item_id": "s0", "payload_text": "dup"}], {"start": 0}, "duplicate source_item_id"),
        ([{"source_item_id": "s0", "payload_text": "p", "t_capture_us": 5}], {"start": 0}, "unknown field"),
        ([{"source_item_id": "s0", "payload_text": "p", "episode_id": "e"}], {"start": 0}, "unknown field"),
        ([{"payload_text": "p"}], {"start": 0}, "source_item_id"),
    ],
)
def test_invalid_sources_rejected(tmp_path, rows, selection, message):
    with pytest.raises(InputValidationError, match=message):
        dataset_provider(tmp_path, rows, **selection).describe()


def test_annotation_without_schema_rejected(tmp_path):
    from sdv_bmt.contracts.specs import parse_input_spec
    from sdv_bmt.input.dataset_fixture import DatasetFixtureProvider
    from tests._helpers import dataset_spec_dict, write_index

    name, sha = write_index(tmp_path, [{"source_item_id": "a", "payload_text": "x", "annotation": {"l": 1}}])
    data = dataset_spec_dict(source={"index": name, "index_sha256": sha}, selection={"start": 0})
    data["family"]["annotation_schema"] = None
    provider = DatasetFixtureProvider(parse_input_spec(data), tmp_path)
    with pytest.raises(InputValidationError, match="annotation_schema=null"):
        provider.describe()


def test_index_digest_mismatch_rejected(tmp_path):
    provider = dataset_provider(tmp_path, _rows(2))
    (tmp_path / "src.index.jsonl").write_text('{"source_item_id":"s0","payload_text":"tampered"}\n')
    with pytest.raises(InputValidationError, match="sha256 mismatch"):
        provider.describe()


def test_blank_line_rejected(tmp_path):
    from tests._helpers import dataset_spec_dict
    from sdv_bmt.contracts.ids import sha256_hex
    from sdv_bmt.contracts.specs import parse_input_spec
    from sdv_bmt.input.dataset_fixture import DatasetFixtureProvider

    text = '{"source_item_id":"a","payload_text":"x"}\n\n{"source_item_id":"b","payload_text":"y"}\n'
    (tmp_path / "i.jsonl").write_text(text)
    spec = parse_input_spec(dataset_spec_dict(source={"index": "i.jsonl", "index_sha256": sha256_hex(text.encode())}))
    with pytest.raises(InputValidationError, match="blank line"):
        DatasetFixtureProvider(spec, tmp_path).describe()


@pytest.mark.parametrize("mode,extra", [("asap", {}), ("fixed_rate", {"rate_hz": 20})])
def test_dataset_supports_asap_and_fixed_rate(mode, extra):
    items = load_input_provider(CONFIGS / "inputs" / "dataset-fixture-8.yaml").items()
    schedule = compute_schedule(items, parse_strict(ReplayPolicy, {"mode": mode, **extra}), InputKind.DATASET)
    assert schedule.mode.value == mode


def test_dataset_recorded_replay_rejected():
    items = load_input_provider(CONFIGS / "inputs" / "dataset-fixture-8.yaml").items()
    with pytest.raises(ScheduleError, match="not allowed for dataset"):
        compute_schedule(items, parse_strict(ReplayPolicy, {"mode": "recorded"}), InputKind.DATASET)
