"""v0.1.2 DeepLabV3+ Cityscapes workload: input contract, canonical preprocessing (float32 arithmetic), postprocess,
the labelmap-u8-rowrle/1 codec and its native twin (byte-identical), and the workload/input/DUT configurations."""

from __future__ import annotations

import hashlib
import json
import random
import struct
import subprocess

import pytest

from sdv_bmt.contracts import labelmap
from sdv_bmt.contracts.protocols import WorkloadError
from sdv_bmt.contracts.records import InputItem
from sdv_bmt.contracts.specs import WorkloadSpec, parse_strict
from sdv_bmt.runner.plan import load_yaml_strict
from sdv_bmt.workload.deeplab import MEAN, STD, DeepLabCityscapesWorkload, normalize_rgb
from tests._helpers import CONFIGS, REPO_ROOT

MODELS = ("r18", "r50", "r101")


def _spec(model="r50", **options):
    data = load_yaml_strict(CONFIGS / "workloads" / f"deeplabv3plus-{model}-cityscapes-v1.yaml")
    data["options"] = {**data["options"], **options}
    return parse_strict(WorkloadSpec, data)


def _item(i=0, payload=b""):
    return InputItem(i, f"x-{i}", "p", hashlib.sha256(payload).hexdigest(), None, None, None, None, {})


def _f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


# ---------------------------------------------------------------- contract and spec checks


def test_contract_and_declared_sizes():
    w = DeepLabCityscapesWorkload(_spec())
    c = w.input_contract()
    assert (c.contract_id, c.contract_version) == ("cityscapes-rgb-norm-chw2048x1024-f32", "1")
    assert dict(c.logical_input)["shape"] == "3x1024x2048"
    with pytest.raises(ValueError, match="multiples of 8"):
        DeepLabCityscapesWorkload(_spec(input_width=2047))
    with pytest.raises(ValueError, match="num_classes must be 19"):
        DeepLabCityscapesWorkload(_spec(num_classes=20))
    program = {**_spec().options["device_program"], "task_output_bytes": 4 * 2048 * 1024}
    with pytest.raises(ValueError, match="task_output_bytes 2097152"):
        DeepLabCityscapesWorkload(_spec(device_program=program))


@pytest.mark.parametrize("model", MODELS)
def test_configured_workloads_match_their_plans(model):
    spec = _spec(model)
    DeepLabCityscapesWorkload(spec)
    program = spec.options["device_program"]
    assert program["timing_offset"] % 8 == 0 and program["output_bytes"] == program["task_output_bytes"] + 32
    plan_path = REPO_ROOT / program["plan"]
    if not plan_path.is_file():
        pytest.skip("NOT RUN: DeepLab model assets are not present on this host")
    plan = json.loads(plan_path.read_text())
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == program["plan_sha256"]
    assert plan["model"]["onnx_sha256"] == spec.artifact_sha256 and plan["weights"]["sha256"] == program["weights_sha256"]
    assert plan["input"]["bytes"] == program["input_bytes"] and plan["output"]["bytes"] == program["task_output_bytes"]


@pytest.mark.parametrize("model", MODELS)
def test_pinned_dut_configs_declare_one_entry_per_convolution(model):
    dut = load_yaml_strict(CONFIGS / "duts" / f"nvidia-rtx-pro-6000-gpu0-cudnn-deeplabv3plus-{model}.yaml")
    cudnn = dut["runtime_options"]["cudnn"]
    assert cudnn["algo_policy"] == "pinned" and cudnn["math_type"] == "default"
    assert len(cudnn["algorithms"]) == {"r18": 37, "r50": 70, "r101": 121}[model]


# ---------------------------------------------------------------- preprocessing


def test_normalization_is_float32_arithmetic():
    rgb = bytes([0, 128, 255, 17, 200, 3]) * 2  # 4 pixels
    out = struct.unpack("<12f", normalize_rgb(rgb, 2, 2))
    for c in range(3):
        for p in range(4):
            v = rgb[3 * p + c]
            assert out[4 * c + p] == _f32(_f32(float(v) - _f32(MEAN[c])) / _f32(STD[c]))


def test_preprocess_checks_payload_digest_and_size():
    w = DeepLabCityscapesWorkload(_spec())
    good = b"P6\n2048 1024\n255\n" + bytes(2048 * 1024 * 3)
    data = w.preprocess(good, _item(0, good))
    assert len(data) == 12 * 2048 * 1024
    with pytest.raises(WorkloadError, match="payload_sha256"):
        w.preprocess(good, _item(0, b"other"))
    small = b"P6\n4 2\n255\n" + bytes(24)
    with pytest.raises(WorkloadError, match="needs 2048x1024"):
        w.preprocess(small, _item(0, small))


# ---------------------------------------------------------------- postprocess and codec


def _maps(w, h):
    rng = random.Random(20261106)
    out = [bytes(w * h), bytes([18]) * (w * h)]
    blocks = bytearray(w * h)
    for y in range(h):
        for x in range(w):
            blocks[y * w + x] = (x // 7 + 3 * (y // 5)) % 19
    out.append(bytes(blocks))
    out.append(bytes((x + y) % 2 for y in range(h) for x in range(w)))  # worst case: raw rows
    out.append(bytes(rng.randrange(19) for _ in range(w * h)))
    return out


def test_codec_round_trip_and_worst_case_bound():
    w, h = 64, 9
    for labels in _maps(w, h):
        blob = labelmap.encode(labels, w, h, 7)
        assert len(blob) <= labelmap.max_encoded_bytes(w, h)
        assert labelmap.decode(blob) == (7, w, h, labels)
    worst = labelmap.encode(_maps(w, h)[3], w, h)
    assert len(worst) == labelmap.max_encoded_bytes(w, h)


def test_codec_rejects_malformed_blobs():
    blob = labelmap.encode(bytes([1, 1, 2, 2]) * 8, 32, 1, 0)
    for bad in (blob[:-1], blob + b"\0", b"XXXXXXXX" + blob[8:], blob[:16] + b"\x07" + blob[17:]):
        with pytest.raises(labelmap.CodecError):
            labelmap.decode(bad)


def test_postprocess_refuses_values_that_are_not_classes():
    w = DeepLabCityscapesWorkload(_spec())
    raw = bytearray(2048 * 1024)
    out = w.postprocess(bytes(raw), _item(3))
    assert w.encode(out) == labelmap.encode(bytes(raw), 2048, 1024, 3)
    raw[5] = 19
    with pytest.raises(WorkloadError, match="not a class index"):
        w.postprocess(bytes(raw), _item(3))
    with pytest.raises(WorkloadError, match="2097152 bytes"):
        w.postprocess(b"\0" * 10, _item(3))


def test_native_label_map_pipeline_bytes_equal_the_python_codec(native_build, tmp_path):
    """The native twin (labelmap_output.hpp) against the Python codec on synthetic maps of the full output size,
    including the worst case, and the same failure for an out-of-range value."""
    w = DeepLabCityscapesWorkload(_spec())
    W, H = 2048, 1024
    small = _maps(64, 16)
    maps = [bytes(sum(([m[(y % 16) * 64 + (x % 64)] for x in range(W)] for y in range(H)), [])) for m in small[:2]]
    maps.append(bytes((x + y) % 2 for y in range(H) for x in range(W)))  # worst case
    rng = random.Random(7)
    maps.append(bytes(rng.randrange(19) for _ in range(W * H)))
    bad = bytearray(maps[0])
    bad[1234] = 200
    maps.append(bytes(bad))
    raw_path = tmp_path / "raw.bin"
    lines, expected = [], []
    with open(raw_path, "wb") as fh:
        for i, m in enumerate(maps):
            lines.append(json.dumps({"item_index": i, "raw_path": str(raw_path), "raw_offset": fh.tell(),
                                     "raw_bytes": len(m)}) + "\n")
            fh.write(m)
            try:
                expected.append(w.encode(w.postprocess(m, _item(i))))
            except WorkloadError:
                expected.append(None)
    proc = subprocess.run([str(native_build / "sdv-bmt-native-fake"), "encode-output",
                           json.dumps(w.native_output_pipeline())], input="".join(lines), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    native = [json.loads(line) for line in proc.stdout.splitlines()]
    assert len(native) == len(expected)
    for i, (row, exp) in enumerate(zip(native, expected)):
        if exp is None:
            assert row["phase"] == "postprocess" and "not a class index" in json.dumps(row), i
        else:
            assert bytes.fromhex(row["blob_hex"]) == exp, i
    assert len(expected[2]) == labelmap.max_encoded_bytes(W, H)


def test_native_label_map_pipeline_refuses_other_parameters(native_build):
    base = DeepLabCityscapesWorkload(_spec()).native_output_pipeline()
    for changed, message in (({"codec": "labelmap-u8-raw/1"}, "codec/postprocess differ"),
                             ({"width": 0}, "width/height"), ({"num_classes": 300}, "num_classes")):
        proc = subprocess.run([str(native_build / "sdv-bmt-native-fake"), "encode-output",
                               json.dumps({**base, **changed})], input="", capture_output=True, text=True)
        assert proc.returncode != 0 and message in proc.stdout, (changed, proc.stdout)
