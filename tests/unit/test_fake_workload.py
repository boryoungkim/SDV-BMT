"""Fake workload: deterministic preprocess/postprocess/codec and declared identity."""

from __future__ import annotations

import pytest

from sdv_bmt.contracts.protocols import WorkloadError
from sdv_bmt.contracts.records import item_digest
from sdv_bmt.contracts.specs import WorkloadSpec, parse_strict
from sdv_bmt.runner.plan import load_yaml_strict
from sdv_bmt.workload.fake import PREPROCESS_PREFIX, FakeIdentityWorkload
from tests._helpers import CONFIGS, load_input_provider


def _spec(**overrides) -> WorkloadSpec:
    data = load_yaml_strict(CONFIGS / "workloads" / "fake-identity.yaml")
    data.update(overrides)
    return parse_strict(WorkloadSpec, data)


def _item_and_payload(k=0):
    provider = load_input_provider(CONFIGS / "inputs" / "dataset-fixture-8.yaml")
    provider.open()
    return provider.items()[k], provider.payload(k)


def test_preprocess_checks_payload_integrity():
    w = FakeIdentityWorkload(_spec())
    item, payload = _item_and_payload()
    assert w.preprocess(payload, item) == PREPROCESS_PREFIX + payload
    with pytest.raises(WorkloadError, match="does not match payload_sha256"):
        w.preprocess(payload + b"x", item)


def test_postprocess_checks_correspondence():
    w = FakeIdentityWorkload(_spec())
    item, _ = _item_and_payload(2)
    raw = {"request_id": 2, "item_index": 2, "item_digest": item_digest(item), "value": 7}
    out = w.postprocess(raw, item)
    assert out["request_id"] == 2 and out["item_digest"] == item_digest(item)
    with pytest.raises(WorkloadError, match="does not correspond"):
        w.postprocess({**raw, "item_index": 3}, item)
    with pytest.raises(WorkloadError, match="does not correspond"):
        w.postprocess({**raw, "item_digest": "0" * 64}, item)


def test_encode_is_canonical_json():
    w = FakeIdentityWorkload(_spec())
    out = {"value": 1, "schema": "fake-identity-output-v1", "task": "t", "request_id": 0,
           "item_index": 0, "item_digest": "d"}
    assert w.encode(out) == (
        b'{"item_digest":"d","item_index":0,"request_id":0,'
        b'"schema":"fake-identity-output-v1","task":"t","value":1}'
    )


def test_declared_identity_must_match_implementation():
    with pytest.raises(ValueError, match="codec"):
        FakeIdentityWorkload(_spec(codec={"id": "msgpack", "version": "1"}))
    with pytest.raises(ValueError, match="artifact"):
        FakeIdentityWorkload(_spec(artifact_sha256="0" * 64))
    with pytest.raises(ValueError, match="implementation"):
        FakeIdentityWorkload(_spec(implementation="real_model"))


def test_postprocess_error_injection():
    w = FakeIdentityWorkload(_spec(options={"postprocess_error_items": [2]}))
    item, _ = _item_and_payload(2)
    raw = {"request_id": 2, "item_index": 2, "item_digest": item_digest(item), "value": 7}
    with pytest.raises(WorkloadError, match="injected"):
        w.postprocess(raw, item)
