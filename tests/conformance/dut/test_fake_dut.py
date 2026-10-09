"""DUT conformance (S1A-4 / WP3): lifecycle order, correspondence, determinism, cleanup."""

from __future__ import annotations

import hashlib
import json

import pytest

from sdv_bmt.contracts.protocols import DUTAdapter, LifecycleError
from sdv_bmt.contracts.records import DUTRequest, item_digest
from sdv_bmt.contracts.specs import DUTConfig, SpecError, WorkloadSpec, parse_strict
from sdv_bmt.contracts.status import CompletionModel, ExecutionKind, RunClockDomain
from sdv_bmt.dut.fake import FakeBlockingDUT, FakeDUTInjectedError, FakePreparedInput
from sdv_bmt.runner.plan import load_yaml_strict
from sdv_bmt.workload.fake import PREPROCESS_PREFIX, FakeIdentityWorkload
from tests._helpers import CONFIGS, load_input_provider


def _dut_config(model: dict) -> DUTConfig:
    return parse_strict(DUTConfig, {
        "schema_version": 1, "dut_config_id": "d", "adapter": "fake_blocking",
        "device": "virtual-0", "max_inflight": 1, "runtime_options": {"service_model": model},
    })


def _workload_spec() -> WorkloadSpec:
    return parse_strict(WorkloadSpec, load_yaml_strict(CONFIGS / "workloads" / "fake-identity.yaml"))


def _items():
    provider = load_input_provider(CONFIGS / "inputs" / "dataset-fixture-8.yaml")
    provider.open()
    return provider, provider.items()


FIXED = {"kind": "fixed", "service_ns": 25_000_000}


def _prepared_dut(model=FIXED):
    config = _dut_config(model)
    dut = FakeBlockingDUT(config)
    dut.prepare(_workload_spec(), config)
    return dut


def test_prepare_input_before_prepare_is_rejected():
    _provider, items = _items()
    dut = FakeBlockingDUT(_dut_config(FIXED))
    with pytest.raises(LifecycleError, match=r"prepare_input\(\) requires prepare\(\) first"):
        dut.prepare_input(b"x", items[0])
    with pytest.raises(LifecycleError, match=r"run\(\) requires prepare\(\) first"):
        dut.run(DUTRequest(0, (0,), (FakePreparedInput(0, "d", "s"),)))


def test_prepare_only_once_and_with_matching_config():
    config = _dut_config(FIXED)
    dut = FakeBlockingDUT(config)
    with pytest.raises(ValueError, match="differs"):
        dut.prepare(_workload_spec(), _dut_config({"kind": "fixed", "service_ns": 1}))
    dut.prepare(_workload_spec(), config)
    with pytest.raises(LifecycleError):
        dut.prepare(_workload_spec(), config)


def test_cleanup_is_idempotent_and_final():
    _provider, items = _items()
    dut = _prepared_dut()
    dut.cleanup()
    dut.cleanup()
    with pytest.raises(LifecycleError):
        dut.prepare_input(b"x", items[0])
    with pytest.raises(LifecycleError):
        dut.prepare(_workload_spec(), _dut_config(FIXED))


def test_output_corresponds_to_request_and_item():
    provider, items = _items()
    workload = FakeIdentityWorkload(_workload_spec())
    dut = _prepared_dut()
    item = items[5]
    prepared = dut.prepare_input(workload.preprocess(provider.payload(5), item), item)
    result = dut.run(DUTRequest(42, (5,), (prepared,)))
    assert result.ok and result.request_id == 42 and result.virtual_service_ns == 25_000_000
    assert result.raw_output["request_id"] == 42
    assert result.raw_output["item_index"] == 5
    assert result.raw_output["item_digest"] == item_digest(item)
    with pytest.raises(ValueError, match="does not correspond"):
        dut.run(DUTRequest(1, (4,), (prepared,)))
    with pytest.raises(ValueError, match="batch=1"):
        dut.run(DUTRequest(1, (5, 6), (prepared, prepared)))


def _encoded(item_index: int) -> bytes:
    provider, items = _items()
    workload = FakeIdentityWorkload(_workload_spec())
    dut = _prepared_dut()
    item = items[item_index]
    prepared = dut.prepare_input(workload.preprocess(provider.payload(item_index), item), item)
    raw = dut.run(DUTRequest(item_index, (item_index,), (prepared,))).raw_output
    return workload.encode(workload.postprocess(raw, item))


def test_output_codec_is_deterministic_and_matches_independent_derivation():
    provider, items = _items()
    for k in range(len(items)):
        blob = _encoded(k)
        assert blob == _encoded(k)
        # Independent re-derivation from the documented fake formula.
        item = items[k]
        task_input = PREPROCESS_PREFIX + provider.payload(k)
        input_sha = hashlib.sha256(task_input).hexdigest()
        expected = {
            "schema": "fake-identity-output-v1", "task": "fake-identity", "request_id": k,
            "item_index": k, "item_digest": item_digest(item), "value": int(input_sha[:8], 16),
        }
        assert blob == json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()


def test_table_model_service_times_and_errors():
    provider, items = _items()
    workload = FakeIdentityWorkload(_workload_spec())
    dut = _prepared_dut({
        "kind": "table", "default_service_ns": 10,
        "entries": [
            {"item_index": 1, "service_ns": 99},
            {"item_index": 2, "run_outcome": "error", "error_code": "E_FAKE", "service_ns": 7},
            {"item_index": 3, "prepare_input_outcome": "error"},
        ],
    })

    def run(k):
        prepared = dut.prepare_input(workload.preprocess(provider.payload(k), items[k]), items[k])
        return dut.run(DUTRequest(k, (k,), (prepared,)))

    assert run(0).virtual_service_ns == 10
    assert run(1).virtual_service_ns == 99
    failed = run(2)
    assert not failed.ok and failed.virtual_service_ns == 7 and failed.error.code == "E_FAKE"
    with pytest.raises(FakeDUTInjectedError):
        dut.prepare_input(workload.preprocess(provider.payload(3), items[3]), items[3])


@pytest.mark.parametrize(
    "model,message",
    [
        ({"kind": "random"}, "kind"),
        ({"kind": "fixed"}, "service_ns"),
        ({"kind": "fixed", "service_ns": 1, "jitter_ns": 2}, "unknown field"),
        ({"kind": "table", "default_service_ns": 1, "entries": [{"item_index": 0}, {"item_index": 0}]}, "unique"),
        ({"kind": "table", "default_service_ns": 1, "entries": [{"item_index": 0, "run_outcome": "crash"}]}, "outcomes"),
    ],
)
def test_runtime_options_are_strict(model, message):
    with pytest.raises(SpecError, match=message):
        FakeBlockingDUT(_dut_config(model))


def test_descriptor_and_capabilities_declare_mock_blocking_virtual():
    dut = FakeBlockingDUT(_dut_config(FIXED))
    d, c = dut.describe(), dut.capabilities()
    assert d.implementation_kind is ExecutionKind.MOCK
    assert d.completion_model is CompletionModel.BLOCKING and d.clock_domain is RunClockDomain.VIRTUAL
    assert c.max_inflight == 1 and c.virtual_timing and not c.device_timing


def test_fake_dut_implements_protocol_surface():
    proto = {n for n in dir(DUTAdapter) if not n.startswith("_")}
    assert proto <= {n for n in dir(FakeBlockingDUT) if not n.startswith("_")}
    assert not any(hasattr(FakeBlockingDUT, n) for n in ("submit", "poll", "callback", "submit_async"))
