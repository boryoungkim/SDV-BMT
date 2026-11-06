"""YOLOX COCO workload (Stage 2b-G real-model baseline; v0.1.2 YOLOX family): canonical preprocessing,
decode/NMS postprocess and codec semantics on synthetic inputs, the input size and anchor count declared in the
workload spec, and consistency with the configured artifacts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import struct
from pathlib import Path

import pytest

from sdv_bmt.contracts.protocols import WorkloadError
from sdv_bmt.contracts.records import InputItem, input_contract_digest
from sdv_bmt.contracts.specs import WorkloadSpec, parse_strict
from sdv_bmt.runner.plan import load_yaml_strict
from sdv_bmt.workload import yolox as Y
from tests._helpers import CONFIGS, REPO_ROOT

FAMILY = ("yolox-tiny-coco-v1", "yolox-s-coco-v1", "yolox-s-coco-v1-quality", "yolox-m-coco-v1", "yolox-l-coco-v1",
          "yolox-x-coco-v1")


def spec_of(name):
    return parse_strict(WorkloadSpec, load_yaml_strict(CONFIGS / "workloads" / f"{name}.yaml"))


SPEC = spec_of("yolox-s-coco-v1")
TINY = spec_of("yolox-tiny-coco-v1")


def workload(spec=SPEC):
    return Y.YoloxCocoWorkload(spec)


def ppm(w, h, pixel):
    data = bytes(v for y in range(h) for x in range(w) for v in pixel(x, y))
    return f"P6\n{w} {h}\n255\n".encode() + data


def item(payload, w, h, index=0, image_id=7):
    return InputItem(index, f"img-{index}", "p", hashlib.sha256(payload).hexdigest(), None, None, None, None,
                     {"index_line": 1, "row_metadata": {"coco_image_id": image_id, "width": w, "height": h}})


def floats(data):
    return struct.unpack(f"<{len(data) // 4}f", data)


@pytest.mark.parametrize("name", FAMILY)
def test_spec_matches_the_model_artifacts(name):
    spec = spec_of(name)
    program = spec.options["device_program"]
    plan_path = REPO_ROOT / program["plan"]
    onnx = plan_path.parent / (Path(program["plan"]).name.split(".")[0] + ".onnx")
    if not onnx.exists():
        pytest.skip("NOT RUN: model artifact not present")
    assert hashlib.sha256(onnx.read_bytes()).hexdigest() == spec.artifact_sha256
    for key in ("plan", "weights"):
        assert hashlib.sha256((REPO_ROOT / program[key]).read_bytes()).hexdigest() == program[f"{key}_sha256"]
    plan = json.loads(plan_path.read_text())
    assert plan["model"]["onnx_sha256"] == spec.artifact_sha256 and plan["model"]["onnx_file"] == onnx.name
    size, anchors = spec.options["input_size"], spec.options["num_anchors"]
    assert plan["input"]["shape"] == [1, 3, size, size] and plan["output"]["shape"] == [1, anchors, 85]
    assert plan["input"]["bytes"] == program["input_bytes"] and plan["output"]["bytes"] == program["task_output_bytes"]
    Y.YoloxCocoWorkload(spec)  # the declared sizes agree with the device program


def test_category_table_is_the_sorted_coco_category_ids():
    manifest = REPO_ROOT / "datasets/coco-val2017-first1000/dataset_manifest.json"
    if not manifest.exists():
        pytest.skip("NOT RUN: dataset not present")
    ids = [c["id"] for c in json.loads(manifest.read_text())["categories"]]
    assert list(Y.COCO_CATEGORY_IDS) == sorted(ids) and len(ids) == Y.NUM_CLASSES


def with_options(spec, drop=(), **changes):
    options = {k: v for k, v in {**spec.options, **changes}.items() if k not in drop}
    return dataclasses.replace(spec, options=options)


def test_input_size_and_anchor_count_are_declared_and_checked():
    assert Y.anchor_count(640) == 8400 and Y.anchor_count(416) == 3549
    for bad in ({"input_size": 600}, {"input_size": 0}, {"input_size": 8192}, {"input_size": "640"},
                {"input_size": True}, {"input_size": 640.0}):
        with pytest.raises(ValueError, match="options.input_size"):
            workload(with_options(SPEC, **bad))
    with pytest.raises(ValueError, match="options.input_size"):
        workload(with_options(SPEC, drop=("input_size",)))
    with pytest.raises(ValueError, match="num_anchors must be 8400 for input_size 640"):
        workload(with_options(SPEC, num_anchors=3549))
    with pytest.raises(ValueError, match="num_anchors must be 8400"):
        workload(with_options(SPEC, drop=("num_anchors",)))
    # a 416 declaration with YOLOX-S's 640 device program
    with pytest.raises(ValueError, match="device_program must declare input_bytes 2076672 and task_output_bytes "
                                         "1206660 for input_size 416"):
        workload(with_options(SPEC, input_size=416, num_anchors=3549))


def test_input_contract_follows_the_declared_size_and_is_unchanged_for_640():
    s, t = workload().input_contract(), workload(TINY).input_contract()
    assert s.contract_id == "yolox-letterbox-bgr-chw640-f32" and t.contract_id == "yolox-letterbox-bgr-chw416-f32"
    assert input_contract_digest(s) == "c6c4de0712b855c6bbb03c9fa61f436deba1acbd7905aef01aeb029315317942"  # v0.1.0
    assert dict(t.logical_input)["shape"] == "3x416x416"
    assert dict(t.logical_input)["letterbox"].startswith("r=min(416/h,416/w)")


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


def test_tiny_letterbox_uses_the_declared_416_canvas():
    w, h = 832, 416  # r = min(416/416, 416/832) = 0.5 -> 416 x 208
    payload = ppm(w, h, lambda x, y: (10, 20, 30))
    out = floats(workload(TINY).preprocess(payload, item(payload, w, h)))
    plane = 416 * 416
    assert len(out) == 3 * plane
    assert out[0] == 30.0 and out[plane + 207 * 416 + 415] == 20.0 and out[2 * plane + 207 * 416] == 10.0
    assert out[208 * 416] == 114.0 and out[-1] == 114.0  # first padded row, last pixel


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


def raw_with(anchors, count=8400):
    """count x 85 float32 output, all objectness 0 except the given anchors {index: (x, y, lw, lh, obj, {cls: s})}."""
    v = [0.0] * (count * Y.NUM_CHANNELS)
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


def test_tiny_decode_uses_the_416_grids_and_raw_size():
    it = item(b"x", 416, 416)  # r = 1
    first_stride32 = 52 * 52 + 26 * 26  # anchor 3380 = stride 32 level, grid (0, 0)
    raw = raw_with({first_stride32 + 13 + 2: (0.5, 0.5, 0.0, 0.0, 0.9, {7: 0.9})}, count=3549)  # grid (2, 1)
    (det,) = workload(TINY).postprocess(raw, it)["detections"]
    assert det["bbox_mpx"] == [64000, 32000, 32000, 32000]  # centre (80, 48), size 32
    assert det["category_id"] == Y.COCO_CATEGORY_IDS[7] and det["score_ppm"] == 810000
    with pytest.raises(WorkloadError, match=r"raw output must be 1206660 bytes \(3549x85 float32\)"):
        workload(TINY).postprocess(raw_with({}), it)


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
    assert p == {"kind": "yolox_coco_v1", "task": "coco-2017-object-detection",  # as recorded by v0.1.0
                 "postprocess": "yolox-decode-nms-coco/1", "codec": "canonical-json-utf8/1",
                 "implementation": "native:native/src/common/yolox_output.hpp", "conf_threshold_ppm": 10000,
                 "nms_iou_threshold_ppm": 650000, "max_detections": 100, "input_size": 640, "num_classes": 80}
    assert workload(TINY).native_output_pipeline() == {**p, "input_size": 416}
