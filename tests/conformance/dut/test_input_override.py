"""DUT conformance for the explicit adapter input override (design change:
Explicit DUT Adapter Preprocess Override)."""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from sdv_bmt.contracts.protocols import AdapterInputOverride, DUTAdapter, LifecycleError
from sdv_bmt.contracts.records import AdapterInputOverrideDescriptor, WorkloadInputContract
from sdv_bmt.contracts.specs import DUTConfig, WorkloadSpec, parse_strict
from sdv_bmt.dut.fake import FakeBlockingDUT, FakeDUTInjectedError, FakeFusedPrefixOverride, FakePreparedInput
from sdv_bmt.runner.plan import load_yaml_strict
from sdv_bmt.workload.fake import FakeIdentityWorkload
from tests._helpers import CONFIGS, load_input_provider

FUSED = FakeFusedPrefixOverride.IMPLEMENTATION_ID


def _workload() -> FakeIdentityWorkload:
    spec = parse_strict(WorkloadSpec, load_yaml_strict(CONFIGS / "workloads" / "fake-identity.yaml"))
    return FakeIdentityWorkload(spec)


def _dut(override=None, entries=()) -> FakeBlockingDUT:
    options = {"service_model": {"kind": "table", "default_service_ns": 1, "entries": list(entries)}}
    if override is not None:
        options["input_override"] = override
    config = parse_strict(DUTConfig, {"schema_version": 1, "dut_config_id": "d", "adapter": "fake_blocking",
                                      "device": "v", "max_inflight": 1, "runtime_options": options})
    return FakeBlockingDUT(config)


def _prepared(dut: FakeBlockingDUT) -> FakeBlockingDUT:
    workload_spec = parse_strict(WorkloadSpec, load_yaml_strict(CONFIGS / "workloads" / "fake-identity.yaml"))
    dut.prepare(workload_spec, dut._config)
    return dut


def _provider():
    provider = load_input_provider(CONFIGS / "inputs" / "dataset-fixture-8.yaml")
    provider.open()
    return provider


def test_adapter_without_override_config_offers_none():
    assert _dut().input_override(_workload().input_contract(), FUSED) is None


def test_adapter_only_offers_the_requested_implementation():
    dut = _dut({})
    assert dut.input_override(_workload().input_contract(), "vendor-fused-resize") is None
    assert isinstance(dut.input_override(_workload().input_contract(), FUSED), FakeFusedPrefixOverride)


def test_descriptor_declares_identity_semantics_and_fused_steps():
    desc = _dut({}).input_override(_workload().input_contract(), FUSED).describe()
    assert desc.implementation_id == FUSED and desc.implementation_version == "1"
    assert desc.preprocess_semantics_ids == ("fake-prefix-preprocess/1",)
    assert desc.fused_steps == ("workload.preprocess", "dut.prepare_input")
    divergent = _dut({"divergent": True}).input_override(_workload().input_contract(), FUSED).describe()
    assert divergent.implementation_id == desc.implementation_id
    assert divergent.implementation_digest != desc.implementation_digest


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("preprocess_semantics_ids", "fake-prefix-preprocess/10", "list/tuple of strings"),  # missing-comma slip
        ("fused_steps", "workload.preprocess", "list/tuple of strings"),
        ("fused_steps", ("workload.preprocess", "gpu.resize"), "unknown entries"),
        ("fused_steps", (), "must not be empty"),
        ("preprocess_semantics_ids", ("a", "a"), "duplicates"),
    ],
)
def test_descriptor_rejects_malformed_sequences(field, value, message):
    desc = _dut({}).input_override(_workload().input_contract(), FUSED).describe()
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(desc, **{field: value})


def test_descriptor_sequences_are_frozen_tuples():
    desc = AdapterInputOverrideDescriptor(FUSED, "1", "0" * 64, ["s/1"], ["workload.preprocess", "dut.prepare_input"])
    assert isinstance(desc.fused_steps, tuple) and isinstance(desc.preprocess_semantics_ids, tuple)


def test_override_matches_the_workload_reference_path_for_every_item():
    workload, provider = _workload(), _provider()
    dut = _prepared(_dut({}))
    override = dut.input_override(workload.input_contract(), FUSED)
    for item in provider.items():
        payload = provider.payload(item.item_index)
        fused = override.prepare_from_payload(payload, item)
        reference = dut.prepare_input(workload.preprocess(payload, item), item)
        assert fused == reference
        assert dut.prepared_input_digest(fused) == dut.prepared_input_digest(reference)


def test_divergent_override_is_not_equivalent():
    workload, provider = _workload(), _provider()
    dut = _prepared(_dut({"divergent": True}))
    override = dut.input_override(workload.input_contract(), FUSED)
    item = provider.items()[0]
    payload = provider.payload(0)
    fused = override.prepare_from_payload(payload, item)
    reference = dut.prepare_input(workload.preprocess(payload, item), item)
    assert dut.prepared_input_digest(fused) != dut.prepared_input_digest(reference)


def test_override_follows_the_adapter_lifecycle():
    workload, provider = _workload(), _provider()
    dut = _dut({})
    override = dut.input_override(workload.input_contract(), FUSED)  # side-effect free before prepare
    item = provider.items()[0]
    with pytest.raises(LifecycleError, match="prepare_from_payload"):
        override.prepare_from_payload(provider.payload(0), item)
    _prepared(dut)
    assert isinstance(override.prepare_from_payload(provider.payload(0), item), FakePreparedInput)
    dut.cleanup()
    with pytest.raises(LifecycleError):
        override.prepare_from_payload(provider.payload(0), item)


def test_override_checks_payload_integrity_and_injected_errors():
    workload, provider = _workload(), _provider()
    dut = _prepared(_dut({}, entries=[{"item_index": 2, "prepare_input_outcome": "error"}]))
    override = dut.input_override(workload.input_contract(), FUSED)
    with pytest.raises(ValueError, match="does not match payload_sha256"):
        override.prepare_from_payload(provider.payload(0) + b"x", provider.items()[0])
    with pytest.raises(FakeDUTInjectedError):
        override.prepare_from_payload(provider.payload(2), provider.items()[2])


def test_override_rejects_unsupported_semantics():
    contract = dataclasses.replace(_workload().input_contract(), preprocess_semantics_id="other-preprocess/9")
    dut = _prepared(_dut({}))
    override = dut.input_override(contract, FUSED)
    assert contract.preprocess_semantics_id not in override.describe().preprocess_semantics_ids
    provider = _provider()
    with pytest.raises(ValueError, match="unsupported preprocess semantics"):
        override.prepare_from_payload(provider.payload(0), provider.items()[0])


def test_adapter_receives_only_the_immutable_contract_never_the_workload():
    params = inspect.signature(FakeBlockingDUT.input_override).parameters
    assert list(params) == ["self", "contract", "implementation_id"]
    workload = _workload()
    override = _dut({}).input_override(workload.input_contract(), FUSED)
    reachable = list(vars(override).values())
    assert not any(isinstance(v, FakeIdentityWorkload) for v in reachable)
    assert any(isinstance(v, WorkloadInputContract) for v in reachable)
    with pytest.raises(dataclasses.FrozenInstanceError):
        workload.input_contract().preprocess_semantics_id = "changed"  # type: ignore[misc]


def test_canonical_digest_belongs_to_the_adapter_not_the_override():
    assert not hasattr(FakeFusedPrefixOverride, "prepared_input_digest")
    with pytest.raises(TypeError):
        _dut({}).prepared_input_digest(b"not a prepared input")


def test_fake_classes_implement_the_protocol_surfaces():
    assert {n for n in dir(AdapterInputOverride) if not n.startswith("_")} <= set(dir(FakeFusedPrefixOverride))
    assert {n for n in dir(DUTAdapter) if not n.startswith("_")} <= set(dir(FakeBlockingDUT))
