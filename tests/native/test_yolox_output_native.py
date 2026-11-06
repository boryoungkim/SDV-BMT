"""Native YOLOX output pipeline (``native/src/common/yolox_output.hpp``) vs the Python reference
(``YoloxCocoWorkload.postprocess`` + ``encode``): byte-identical task outputs, and the same failure, on synthetic
and adversarial raw outputs, for the 640 (YOLOX-S/M/L/X) and 416 (YOLOX-Tiny) input sizes (Stage 2b-G real-model
baseline; v0.1.2 YOLOX family)."""

from __future__ import annotations

import hashlib
import json
import random
import struct
import subprocess

import pytest

from sdv_bmt.contracts.protocols import WorkloadError
from sdv_bmt.contracts.records import InputItem
from sdv_bmt.contracts.specs import WorkloadSpec, parse_strict
from sdv_bmt.runner.plan import load_yaml_strict
from sdv_bmt.workload.yolox import YoloxCocoWorkload
from tests._helpers import CONFIGS

SIZES = [(640, 480), (500, 375), (1280, 960), (1, 1), (7, 640), (640, 640), (333, 517)]


def _workload(name):
    return YoloxCocoWorkload(parse_strict(WorkloadSpec, load_yaml_strict(CONFIGS / "workloads" / f"{name}.yaml")))


def _vectors(anchors):
    rng = random.Random(20261008)
    out = []
    for k in range(24):
        v = [0.0] * (anchors * 85)
        n_hot = rng.choice([0, 3, 40, 400, 3000])
        for a in rng.sample(range(anchors), n_hot):
            base = a * 85
            v[base:base + 4] = [rng.uniform(-1, 2), rng.uniform(-1, 2), rng.uniform(-2, 2.5), rng.uniform(-2, 2.5)]
            v[base + 4] = rng.choice([rng.random(), rng.uniform(0.005, 0.02), 1.0])
            for c in rng.sample(range(80), rng.choice([1, 3, 80])):
                v[base + 5 + c] = rng.choice([rng.random(), 0.5, 0.25])
        if k % 4 == 1:  # clusters of near-duplicate boxes for NMS, same and different classes
            for a in range(1000, 1060):
                base = a * 85
                v[base:base + 5] = [-(a - 1000) * 1.0 + 0.3, 0.2, 1.5, 1.5, 0.9 - 0.001 * (a % 7)]
                v[base + 5 + (a % 3)] = 0.95
        if k % 4 == 2:  # exact score ties
            for a in range(200, 260):
                v[a * 85 + 4] = 0.5
                v[a * 85 + 5 + 7] = 0.5
        out.append(v)
    special = [0.0] * (anchors * 85)
    special[4], special[5] = float("nan"), 0.9                    # NaN objectness: skipped
    special[85 + 4], special[85 + 5] = 0.9, float("nan")          # NaN first class score: skipped
    special[170 + 4], special[170 + 6] = 0.9, 0.9                 # ordinary
    special[255 + 4], special[255 + 5] = 0.9, -1.0                # negative score: skipped
    out.append(special)
    overflow = [0.0] * (anchors * 85)
    overflow[2], overflow[4], overflow[5] = 500.0, 0.9, 0.9      # exp overflow -> postprocess error
    out.append(overflow)
    infobj = [0.0] * (anchors * 85)
    infobj[4], infobj[5] = float("inf"), 0.9                      # infinite score -> postprocess error
    out.append(infobj)
    out.append([0.0] * (8400 * 85 if anchors != 8400 else 3549 * 85))  # raw size of the other input size
    return out


@pytest.mark.parametrize("name, anchors", [("yolox-s-coco-v1", 8400), ("yolox-tiny-coco-v1", 3549)])
def test_native_yolox_pipeline_bytes_equal_the_python_reference(native_build, tmp_path, name, anchors):
    workload = _workload(name)
    raw_path = tmp_path / "raw.bin"
    lines, expected = [], []
    with open(raw_path, "wb") as fh:
        for i, v in enumerate(_vectors(anchors)):
            raw = struct.pack(f"<{len(v)}f", *v)
            w, h = SIZES[i % len(SIZES)]
            ctx = {"image_id": 1000 + i, "width": w, "height": h}
            lines.append(json.dumps({"item_index": i, "raw_path": str(raw_path), "raw_offset": fh.tell(),
                                     "raw_bytes": len(raw), "output_context": ctx}) + "\n")
            fh.write(raw)
            item = InputItem(i, f"x-{i}", "p", hashlib.sha256(b"").hexdigest(), None, None, None, None,
                             {"index_line": i + 1, "row_metadata": {"coco_image_id": 1000 + i, "width": w, "height": h}})
            try:
                expected.append(workload.encode(workload.postprocess(raw, item)))
            except WorkloadError:
                expected.append(None)
    pipeline = json.dumps(workload.native_output_pipeline())
    proc = subprocess.run([str(native_build / "sdv-bmt-native-fake"), "encode-output", pipeline], input="".join(lines),
                          capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    native = [json.loads(line) for line in proc.stdout.splitlines()]
    assert len(native) == len(expected)
    detections = 0
    for i, (row, exp) in enumerate(zip(native, expected)):
        if exp is None:
            assert "error" in row and row["phase"] == "postprocess", (i, row)
        else:
            assert row.get("blob", "").encode("utf-8") == exp, i
            detections += len(json.loads(exp)["detections"])
    assert detections > 500 and sum(e is None for e in expected) == 3
    assert f"must be {anchors}x85 float32" in json.dumps(native[-1])


def test_native_yolox_pipeline_refuses_other_parameters(native_build):
    base = _workload("yolox-s-coco-v1").native_output_pipeline()
    for changed, message in (({"nms_iou_threshold_ppm": 500000}, "parameters differ"),
                             ({"input_size": 600}, "input_size must be a multiple of 32"),
                             ({"input_size": 0}, "input_size must be a multiple of 32"),
                             ({"input_size": 8192}, "input_size must be a multiple of 32")):
        proc = subprocess.run([str(native_build / "sdv-bmt-native-fake"), "encode-output",
                               json.dumps({**base, **changed})], input="", capture_output=True, text=True)
        assert proc.returncode != 0 and message in proc.stdout, (changed, proc.stdout)
