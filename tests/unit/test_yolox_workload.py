"""YOLOX COCO workload (Stage 2b-G real-model baseline): canonical preprocessing, decode/NMS postprocess and
codec semantics on synthetic inputs, and consistency with the configured artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

import pytest

from sdv_bmt.contracts.protocols import WorkloadError
from sdv_bmt.contracts.records import InputItem
from sdv_bmt.contracts.specs import WorkloadSpec, parse_strict
from sdv_bmt.runner.plan import load_yaml_strict
from sdv_bmt.workload import yolox as Y
from tests._helpers import CONFIGS, REPO_ROOT

SPEC = parse_strict(WorkloadSpec, load_yaml_strict(CONFIGS / "workloads" / "yolox-s-coco-v1.yaml"))


def workload():
    return Y.YoloxCocoWorkload(SPEC)


def ppm(w, h, pixel):
    data = bytes(v for y in range(h) for x in range(w) for v in pixel(x, y))
    return f"P6\n{w} {h}\n255\n".encode() + data


def item(payload, w, h, index=0, image_id=7):
    return InputItem(index, f"img-{index}", "p", hashlib.sha256(payload).hexdigest(), None, None, None, None,
                     {"index_line": 1, "row_metadata": {"coco_image_id": image_id, "width": w, "height": h}})


def floats(data):
    return struct.unpack(f"<{len(data) // 4}f", data)


def test_spec_matches_the_model_artifacts():
    program = SPEC.options["device_program"]
    onnx = REPO_ROOT / "models/yolox-s/yolox_s.onnx"
    if not onnx.exists():
        pytest.skip("NOT RUN: model artifact not present")
    assert hashlib.sha256(onnx.read_bytes()).hexdigest() == SPEC.artifact_sha256
    for key in ("plan", "weights"):
        assert hashlib.sha256((REPO_ROOT / program[key]).read_bytes()).hexdigest() == program[f"{key}_sha256"]
    plan = json.loads((REPO_ROOT / program["plan"]).read_text())
    assert plan["model"]["onnx_sha256"] == SPEC.artifact_sha256
    assert plan["input"]["bytes"] == program["input_bytes"] == Y.INPUT_BYTES
    assert plan["output"]["bytes"] == program["task_output_bytes"] == Y.RAW_BYTES


def test_category_table_is_the_sorted_coco_category_ids():
    manifest = REPO_ROOT / "datasets/coco-val2017-first1000/dataset_manifest.json"
    if not manifest.exists():
        pytest.skip("NOT RUN: dataset not present")
    ids = [c["id"] for c in json.loads(manifest.read_text())["categories"]]
    assert list(Y.COCO_CATEGORY_IDS) == sorted(ids) and len(ids) == Y.NUM_CLASSES


def test_letterbox_without_resize_is_exact_bgr_chw_with_padding():
    w, h = 640, 3
    payload = ppm(w, h, lambda x, y: (x % 256, y, 200))
    out = floats(workload().preprocess(payload, item(payload, w, h)))
    assert len(out) == 3 * 640 * 640
    plane = 640 * 640
    b, g, r = out[:plane], out[plane:2 * plane], out[2 * plane:]
    assert b[0] == 200.0 and g[640 + 5] == 1.0 and r[2 * 640 + 300] == float(300 % 256)  # BGR order, row-major
    assert b[3 * 640] == 114.0 and r[-1] == 114.0  # padding below the image


def test_letterbox_scales_by_the_short_ratio_and_pads_right_and_bottom():
    w, h = 320, 160  # r = min(640/160, 640/320) = 2.0 -> 640 x 320
    payload = ppm(w, h, lambda x, y: (10, 20, 30))
    out = floats(workload().preprocess(payload, item(payload, w, h)))
    plane = 640 * 640
    assert out[0] == 30.0 and out[plane + 319 * 640 + 639] == 20.0  # constant image stays constant
    assert out[320 * 640] == 114.0  # first padded row


def test_bilinear_resize_uses_half_pixel_centres_and_rounds_half_up():
    # 2 -> 4 pixels: source centres map to 0.25/0.75 positions; values 0 and 100
    row = Y.resize_bilinear(bytes([0, 0, 0, 100, 100, 100]), 2, 1, 4, 1)
    assert list(row[0::3]) == [0, 25, 75, 100]
    assert list(Y.resize_bilinear(bytes([1, 1, 1, 2, 2, 2]), 2, 1, 4, 1)[0::3]) == [1, 1, 2, 2]  # 1.25 -> 1, 1.75 -> 2


def test_preprocess_rejects_inconsistent_inputs():
    w, h = 4, 2
    payload = ppm(w, h, lambda x, y: (1, 2, 3))
    with pytest.raises(WorkloadError, match="payload_sha256"):
        workload().preprocess(payload + b"x", item(payload, w, h))
    with pytest.raises(WorkloadError, match="index metadata says"):
        workload().preprocess(payload, item(payload, 5, 2))
    bad = b"P5\n4 2\n255\n" + bytes(8)
    with pytest.raises(WorkloadError, match="PPM"):
        workload().preprocess(bad, item(bad, 4, 2))


def raw_with(anchors):
    """8400x85 float32 output, all objectness 0 except the given anchors {index: (x, y, lw, lh, obj, {cls: s})}."""
    v = [0.0] * (Y.NUM_ANCHORS * Y.NUM_CHANNELS)
    for a in range(Y.NUM_ANCHORS):
        v[a * 85 + 4] = 0.0
    for a, (x, y, lw, lh, obj, cls) in anchors.items():
        base = a * 85
        v[base:base + 5] = [x, y, lw, lh, obj]
        for c, s in cls.items():
            v[base + 5 + c] = s
    return struct.pack(f"<{len(v)}f", *v)


def test_decode_scale_and_quantization_follow_the_yolox_formulas():
    w, h = 1280, 960  # r = 0.5
    it = item(b"x", w, h, image_id=139)
    # anchor 6400 = stride 16 level, grid (0, 0); anchor 6401 -> grid (1, 0)
    raw = raw_with({6401: (0.5, 0.25, math.log(2.0), 0.0, 0.5, {3: 0.8})})
    out = workload().postprocess(raw, it)
    (det,) = out["detections"]
    cx, cy, bw, bh = (0.5 + 1) * 16, (0.25 + 0) * 16, 2.0 * 16, 1.0 * 16
    x1, y1 = (cx - bw / 2) / 0.5, (cy - bh / 2) / 0.5
    assert det["bbox_mpx"] == [math.floor(x1 * 1000 + 0.5), math.floor(y1 * 1000 + 0.5),
                               math.floor(bw / 0.5 * 1000 + 0.5), math.floor(bh / 0.5 * 1000 + 0.5)]
    assert det["category_id"] == Y.COCO_CATEGORY_IDS[3] and det["score_ppm"] == 400000
    assert out["image_id"] == 139 and out["item_index"] == 0 and out["schema"] == Y.OUTPUT_SCHEMA


def test_threshold_applies_to_objectness_times_best_class():
    it = item(b"x", 640, 640)
    raw = raw_with({0: (0, 0, 0, 0, 0.5, {0: 0.0199}),   # 0.00995 < 0.01 -> dropped
                    1: (0, 0, 0, 0, 0.5, {0: 0.025}),    # 0.0125 -> kept
                    2: (0, 0, 0, 0, 0.009, {0: 1.0})})   # obj below threshold -> dropped
    assert [d["score_ppm"] for d in workload().postprocess(raw, it)["detections"]] == [12500]


def test_nms_is_class_aware_and_keeps_the_higher_score():
    it = item(b"x", 640, 640)
    same = (0.0, 0.0, 1.0, 1.0)
    raw = raw_with({10: (*same, 0.9, {5: 0.9}), 11: (0.05 - 1, 0.0, 1.0, 1.0, 0.8, {5: 0.9}),
                    12: (0.05 - 2, 0.0, 1.0, 1.0, 0.7, {6: 0.9})})  # anchors 10..12: grid x 10..12
    dets = workload().postprocess(raw, it)["detections"]
    assert [(d["category_id"], d["score_ppm"]) for d in dets] == [(Y.COCO_CATEGORY_IDS[5], 810000),
                                                                   (Y.COCO_CATEGORY_IDS[6], 630000)]


def test_at_most_one_hundred_detections_in_descending_score_order():
    it = item(b"x", 640, 640)
    anchors = {a: (0, 0, -3.0, -3.0, 0.5 + a * 1e-4, {a % 80: 0.9}) for a in range(0, 6400, 37)}
    dets = workload().postprocess(raw_with(anchors), it)["detections"]
    assert len(dets) == 100
    scores = [d["score_ppm"] for d in dets]
    assert scores == sorted(scores, reverse=True)


def test_non_finite_detection_is_a_postprocess_error():
    it = item(b"x", 640, 640)
    raw = raw_with({0: (0, 0, 1000.0, 0, 0.9, {0: 0.9})})  # exp overflow -> infinite width
    with pytest.raises(WorkloadError, match="not finite"):
        workload().postprocess(raw, it)


def test_encode_is_canonical_json():
    it = item(b"x", 640, 640)
    out = workload().postprocess(raw_with({5: (0, 0, 0, 0, 0.9, {1: 0.9})}), it)
    blob = workload().encode(out)
    assert blob.startswith(b'{"detections":[{"bbox_mpx":[') and blob.endswith(b',"schema":"sdv-yolox-coco-detections/1"}')
    assert json.loads(blob) == out


def test_native_output_pipeline_declares_its_parameters():
    p = workload().native_output_pipeline()
    assert p["kind"] == "yolox_coco_v1" and p["conf_threshold_ppm"] == 10000 and p["max_detections"] == 100
    assert p["nms_iou_threshold_ppm"] == 650000 and p["input_size"] == 640 and p["num_classes"] == 80
