"""CenterPoint-Pillars workload, det3d codec and the bundle-only nuScenes detection evaluator (CenterPoint part)."""

from __future__ import annotations

import dataclasses
import math
import struct

import pytest

from sdv_bmt.contracts import det3d
from sdv_bmt.contracts.ids import sha256_hex
from sdv_bmt.contracts.protocols import WorkloadError
from sdv_bmt.contracts.records import InputItem
from sdv_bmt.contracts.specs import WorkloadSpec, parse_strict
from sdv_bmt.metrics import quality_det3d as Q
from sdv_bmt.runner.plan import load_yaml_strict
from sdv_bmt.workload.centerpoint import (CLASS_NAMES, HEADER, INPUT_BYTES, MAGIC, RAW_BYTES,
                                          CenterPointNuScenesWorkload)
from tests._helpers import CONFIGS


def _spec(**changes) -> WorkloadSpec:
    spec = parse_strict(WorkloadSpec, load_yaml_strict(CONFIGS / "workloads" / "centerpoint-pillar02-nus-v1.yaml"))
    return dataclasses.replace(spec, **changes)


def _payload(pillars: int = 100, magic: bytes = MAGIC, max_pillars: int = 40000) -> bytes:
    return HEADER.pack(magic, pillars, max_pillars, 20, 5, 1234, 0, 0, 0) + bytes(INPUT_BYTES)


def _item(payload: bytes, i: int = 0) -> InputItem:
    return InputItem(item_index=i, source_item_id=f"s{i}", payload_ref=f"p{i}", payload_sha256=sha256_hex(payload),
                     annotation_ref=None, annotation_digest=None, source_timing=None, scenario_coordinates=None)


def test_workload_identity_and_contract():
    w = CenterPointNuScenesWorkload(_spec())
    c = w.input_contract()
    assert (c.contract_id, c.contract_version, c.preprocess_semantics_id) == (
        "nuscenes-pillar02-p40000", "1", "pillar-payload-body/1")
    assert w.native_output_pipeline()["task_classes"] == [1, 2, 2, 1, 2, 2]
    assert CLASS_NAMES[:3] == ("car", "truck", "construction_vehicle") and len(CLASS_NAMES) == 10
    assert Q.CLASS_NAMES == tuple(sorted(CLASS_NAMES, key=Q.CLASS_NAMES.index))  # same 10 classes
    with pytest.raises(ValueError, match="implementation"):
        CenterPointNuScenesWorkload(_spec(implementation="yolox_coco"))
    program = dict(_spec().options["device_program"], input_bytes=123)
    with pytest.raises(ValueError, match="input_bytes"):
        CenterPointNuScenesWorkload(_spec(options={"device_program": program}))


def test_preprocess_strips_the_header_and_refuses_other_payloads():
    w = CenterPointNuScenesWorkload(_spec())
    p = _payload()
    assert w.preprocess(p, _item(p)) == p[HEADER.size:] and len(p) - HEADER.size == INPUT_BYTES
    for bad, message in ((_payload(magic=b"SDVPIL02"), "pillar payload"), (_payload(max_pillars=30000), "pillar payload"),
                         (_payload(pillars=40001), "pillar payload"), (p[:-1], "bytes, expected")):
        with pytest.raises(WorkloadError, match=message):
            w.preprocess(bad, _item(bad))
    with pytest.raises(WorkloadError, match="payload_sha256"):
        w.preprocess(p, _item(_payload(pillars=7)))
    with pytest.raises(WorkloadError, match="raw output"):
        w.postprocess(b"\0" * (RAW_BYTES - 4), _item(p))


def test_det3d_codec_round_trip_and_errors():
    dets = [((1.0, -2.5, -1.0, 4.2, 1.9, 1.6, 0.3, 0.1, -0.2), 0.75, 0), ((0.0,) * 9, 0.5, 9)]
    blob = det3d.encode(42, dets)
    assert blob[:8] == b"SDVCP3D1" and len(blob) == 16 + 2 * 44
    item, back = det3d.decode(blob)
    assert item == 42 and [d[2] for d in back] == [0, 9]
    assert all(abs(a - b) < 1e-6 for (x, _, _), (y, _, _) in zip(dets, back) for a, b in zip(x, y))
    for bad in (blob[:10], b"SDVXXXXX" + blob[8:], blob + b"\0"):
        with pytest.raises(det3d.CodecError):
            det3d.decode(bad)


def _identity_pose():
    return {"rotation": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]}


def test_evaluator_perfect_predictions_give_map_and_nds_one():
    """Every class has one ground-truth box equal to the prediction: AP 1 per class, TP errors 0, NDS 1."""
    dets = [((4.0 + 2.5 * k, (-1) ** k * 3.0, -1.0, 4.0, 2.0, 1.5, 0.2 * k, 0.5 * (k % 2), 0.0), 0.9, k)
            for k in range(10)]
    pose = _identity_pose()
    pred = Q.predictions_to_global(dets, pose, pose)
    gt_boxes = [{**{k: b[k] for k in ("translation", "size", "rotation", "detection_name", "attribute_name")},
                 "velocity": list(b["velocity"]), "num_pts": 12} for b in pred]
    annotation = {"boxes": gt_boxes, "bicycle_racks": [], "lidar2ego": pose, "ego2global": pose}
    r = Q.det3d_from_items([("tok", annotation, det3d.encode(0, dets))])
    assert r["mean_ap"] == pytest.approx(1.0) and r["nd_score"] == pytest.approx(1.0, abs=1e-6)
    assert all(v == pytest.approx(1.0) for v in r["mean_dist_aps"].values())
    assert r["tp_errors"]["trans_err"] == pytest.approx(0.0, abs=1e-6)
    # no prediction at all: AP 0 for every class, NDS 0 (TP errors 1)
    r0 = Q.det3d_from_items([("tok", annotation, det3d.encode(0, []))])
    assert r0["mean_ap"] == 0.0 and r0["nd_score"] == pytest.approx(0.0)


def test_evaluator_numpy_semantics_helpers():
    assert Q.np_interp([0.5, 2.0, -1.0], [0.0, 1.0], [0.0, 10.0]) == [5.0, 10.0, 0.0]
    assert Q.linspace_101()[0] == 0.0 and Q.linspace_101()[-1] == 1.0 and len(Q.linspace_101()) == 101
    assert math.isnan(Q.np_nanmean([math.nan, math.nan]))
    assert Q.angle_diff(math.pi - 0.1, -math.pi + 0.1, 2 * math.pi) == pytest.approx(-0.2)


def test_centerpoint_configs_are_consistent():
    """The mini input, the workload and the det3d evaluator declare the same task and annotation schema."""
    from sdv_bmt.contracts.quality import NUSCENES_DET3D_V1, QUALITY_EVALUATORS
    from sdv_bmt.contracts.specs import parse_input_spec
    spec = _spec()
    inp = parse_input_spec(load_yaml_strict(CONFIGS / "inputs" / "nuscenes-mini-val4.yaml"), "nuscenes-mini-val4.yaml")
    decl = QUALITY_EVALUATORS[NUSCENES_DET3D_V1]
    assert spec.quality_evaluator == NUSCENES_DET3D_V1
    assert decl.task == spec.task == inp.family.task
    assert decl.annotation_schema == inp.family.annotation_schema
    assert inp.family.preprocessing_contract == "nuscenes-pillar02-p40000/1"
    assert struct.calcsize("<8s8I") == HEADER.size == 40
