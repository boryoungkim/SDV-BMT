"""Input conformance common to both fixture providers (S1A-2/S1A-3/S1A-4 inputs side)."""

from __future__ import annotations

import inspect

import pytest

from sdv_bmt.contracts.ids import sha256_hex
from sdv_bmt.contracts.protocols import InputProvider, LifecycleError
from sdv_bmt.contracts.records import items_digest
from sdv_bmt.input.dataset_fixture import DatasetFixtureProvider
from sdv_bmt.input.recorded_scenario_fixture import RecordedScenarioFixtureProvider
from tests._helpers import CONFIGS, load_input_provider

SPECS = [
    "dataset-fixture-8.yaml",
    "dataset-fixture-128.yaml",
    "scenario-ep-a.yaml",
    "scenario-ep-a-ep-b.yaml",
    "scenario-g2-overload.yaml",
]


@pytest.mark.parametrize("spec", SPECS)
def test_deterministic_order_and_digest(spec):
    a = load_input_provider(CONFIGS / "inputs" / spec)
    b = load_input_provider(CONFIGS / "inputs" / spec)
    assert a.items() == b.items()
    assert a.describe() == b.describe()
    assert a.describe().items_digest == items_digest(a.items())
    assert [i.item_index for i in a.items()] == list(range(len(a.items())))


@pytest.mark.parametrize("spec", SPECS)
def test_lifecycle_open_payload_release(spec):
    p = load_input_provider(CONFIGS / "inputs" / spec)
    with pytest.raises(LifecycleError, match="before open"):
        p.payload(0)
    p.open()
    p.open()  # idempotent
    for item in p.items():
        assert sha256_hex(p.payload(item.item_index)) == item.payload_sha256
    p.release()
    p.release()  # idempotent
    with pytest.raises(LifecycleError, match="after release"):
        p.payload(0)
    with pytest.raises(LifecycleError, match="after release"):
        p.open()


@pytest.mark.parametrize("cls", [DatasetFixtureProvider, RecordedScenarioFixtureProvider])
def test_provider_interface_has_no_dut_or_prepared_input_parameter(cls):
    # M-1: the InputProvider surface is source/index/payload only.
    for name in ("describe", "items", "open", "payload", "release"):
        params = list(inspect.signature(getattr(cls, name)).parameters)
        assert params in (["self"], ["self", "item_index"]), (name, params)
    public = {n for n in dir(cls) if not n.startswith("_")}
    assert public == {"PROVIDER", "describe", "items", "open", "payload", "release"}
    proto = {n for n in dir(InputProvider) if not n.startswith("_")}
    assert proto <= public


def test_open_detects_source_change_after_preflight(tmp_path):
    import shutil

    from sdv_bmt.contracts.protocols import InputValidationError
    from sdv_bmt.contracts.specs import parse_input_spec
    from sdv_bmt.runner.plan import load_yaml_strict

    (tmp_path / "inputs").mkdir()
    (tmp_path / "fixtures").mkdir()
    shutil.copy(CONFIGS / "inputs" / "dataset-fixture-8.yaml", tmp_path / "inputs")
    shutil.copy(CONFIGS / "fixtures" / "dataset-fixture-v1.index.jsonl", tmp_path / "fixtures")
    spec_path = tmp_path / "inputs" / "dataset-fixture-8.yaml"
    provider = DatasetFixtureProvider(parse_input_spec(load_yaml_strict(spec_path)), spec_path.parent)
    provider.describe()  # preflight
    index = tmp_path / "fixtures" / "dataset-fixture-v1.index.jsonl"
    index.write_text(index.read_text().replace("sample 0000", "sample XXXX"))
    with pytest.raises(InputValidationError, match="sha256 mismatch"):
        provider.open()
