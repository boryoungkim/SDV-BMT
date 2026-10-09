"""Reference accelerator conformance workload (not a task model): preprocessing exactness and
determinism, output encoding, and the independent integer CPU reference against the golden file."""

from __future__ import annotations

import array
import base64
import json
import struct
import sys

import pytest

from sdv_bmt.contracts.ids import sha256_hex
from sdv_bmt.contracts.protocols import WorkloadError
from sdv_bmt.contracts.records import InputItem
from sdv_bmt.contracts.specs import WorkloadSpec, parse_strict
from sdv_bmt.runner.plan import load_yaml_strict
from sdv_bmt.workload.reference import INPUT_BYTES, ReferenceConvNetWorkload
from tests._helpers import CONFIGS, REPO_ROOT, ZERO_SHA, load_input_provider

sys.path.insert(0, str(REPO_ROOT / "tools" / "stage1r_g"))
from reference_cpu import reference_logits  # noqa: E402


@pytest.fixture(scope="module")
def workload():
    return ReferenceConvNetWorkload(parse_strict(WorkloadSpec, load_yaml_strict(CONFIGS / "workloads" / "reference-convnet-v1.yaml")))


@pytest.fixture(scope="module")
def provider():
    p = load_input_provider(CONFIGS / "inputs" / "reference-images-128.yaml")
    p.open()
    yield p
    p.release()


def test_artifact_digest_pins_the_device_program(workload):
    spec = parse_strict(WorkloadSpec, load_yaml_strict(CONFIGS / "workloads" / "reference-convnet-v1.yaml"))
    source = spec.options["device_program"]["source"]
    assert sha256_hex(source.encode()) == spec.artifact_sha256
    assert source == (REPO_ROOT / "tools" / "stage1r_g" / "sdv_ref_convnet_v1.cu").read_text()


def test_preprocess_is_exact_and_deterministic(workload, provider):
    item = provider.items()[5]
    payload = provider.payload(5)
    a, b = workload.preprocess(payload, item), workload.preprocess(payload, item)
    assert a == b and len(a) == INPUT_BYTES
    image = base64.b64decode(payload)
    values = array.array("f")
    values.frombytes(a)
    # CHW, nearest x4: value at (c, y, x) is (pixel(c, y//4, x//4) - 128) / 128, exactly
    for c, y, x in ((0, 0, 0), (1, 7, 300), (2, 511, 511), (0, 129, 3)):
        pixel = image[((y // 4) * 128 + (x // 4)) * 3 + c]
        assert values[(c * 512 + y) * 512 + x] == (pixel - 128) / 128
    with pytest.raises(WorkloadError):
        workload.preprocess(payload + b"x", item)  # payload digest mismatch


def test_output_encoding_and_failures(workload):
    item = InputItem(3, "x", "p", ZERO_SHA, None, None, None, None)
    out = workload.postprocess(struct.pack("<10q", *([4, 9, 9] + [0] * 7)), item)
    assert out["top1"] == 1 and out["item_index"] == 3
    assert json.loads(workload.encode(out)) == out
    with pytest.raises(WorkloadError):
        workload.postprocess(b"\x00" * 79, item)


def test_cpu_reference_reproduces_the_golden_entry_of_item_0(workload, provider):
    item = provider.items()[0]
    logits = reference_logits(base64.b64decode(provider.payload(0)))
    blob = workload.encode(workload.postprocess(struct.pack("<10q", *logits), item))
    golden = json.loads((CONFIGS / "references" / "reference-images-v1.ref-convnet-v1.golden.jsonl")
                        .read_text().splitlines()[0])
    assert golden["item_index"] == 0 and sha256_hex(blob) == golden["output_sha256"]
