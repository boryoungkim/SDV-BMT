"""Dataset provider with file-backed payloads (Stage 2b-G real dataset): declared digests at PREFLIGHT,
read-and-verify at open(), provider identity, and rejection of malformed indexes."""

from __future__ import annotations

import hashlib

import pytest

from sdv_bmt.contracts.protocols import InputValidationError
from tests._helpers import dataset_provider


def _file_rows(directory, payloads):
    (directory / "payloads").mkdir()
    rows = []
    for k, data in enumerate(payloads):
        rel = f"payloads/p{k}.bin"
        (directory / rel).write_bytes(data)
        rows.append({"source_item_id": f"id-{k}", "payload_file": rel,
                     "payload_sha256": hashlib.sha256(data).hexdigest(), "annotation": {"label": k}})
    return rows


def test_file_payloads_are_declared_then_verified_at_open(tmp_path):
    payloads = [b"alpha", b"beta" * 100, b"\x00\x01\x02"]
    p = dataset_provider(tmp_path, _file_rows(tmp_path, payloads))
    desc = p.describe()
    assert desc.provider == "dataset_files" and desc.item_count == 3
    assert [i.payload_sha256 for i in p.items()] == [hashlib.sha256(d).hexdigest() for d in payloads]
    p.open()
    assert [p.payload(i) for i in range(3)] == payloads
    p.release()


def test_inline_payload_index_keeps_the_fixture_provider_name(tmp_path):
    rows = [{"source_item_id": "a", "payload_text": "x", "annotation": {"label": 0}}]
    assert dataset_provider(tmp_path, rows).describe().provider == "dataset_fixture"


def test_payload_file_that_changed_is_rejected_at_open(tmp_path):
    rows = _file_rows(tmp_path, [b"one", b"two"])
    p = dataset_provider(tmp_path, rows)
    p.describe()
    (tmp_path / "payloads/p1.bin").write_bytes(b"tampered")
    with pytest.raises(InputValidationError, match="payload_sha256"):
        p.open()


def test_mixed_payload_kinds_are_rejected(tmp_path):
    rows = _file_rows(tmp_path, [b"one"]) + [{"source_item_id": "t", "payload_text": "x"}]
    with pytest.raises(InputValidationError, match="mix"):
        dataset_provider(tmp_path, rows).describe()


@pytest.mark.parametrize("rel", ["../outside.bin", "/etc/passwd", "payloads/../../x.bin"])
def test_payload_file_must_stay_inside_the_index_directory(tmp_path, rel):
    rows = [{"source_item_id": "a", "payload_file": rel, "payload_sha256": "0" * 64}]
    with pytest.raises(InputValidationError):
        dataset_provider(tmp_path, rows).describe()


def test_unknown_or_missing_file_row_fields_are_rejected(tmp_path):
    rows = _file_rows(tmp_path, [b"one"])
    bad = dict(rows[0], extra=1)
    with pytest.raises(InputValidationError, match="unknown"):
        dataset_provider(tmp_path, [bad]).describe()
    missing = {k: v for k, v in rows[0].items() if k != "payload_sha256"}
    with pytest.raises(InputValidationError, match="payload_sha256"):
        dataset_provider(tmp_path, [missing]).describe()
