"""v0.1.2 Metric Set v1.1: the Cityscapes mIoU evaluator (cityscapes-semseg-miou/1) and the catalog that adds it.

* exact per-class counts against a direct per-pixel count, ignore label 255, classes with an empty union
* bundle-only evaluation on a minimal synthetic bundle: frozen ground truth, output digests, item binding
* v1.1 keeps every Metric Set v1 value of a detection bundle and reports the segmentation metrics not_applicable;
  the frozen v1 catalog still reproduces an archived v0.1.2 YOLOX evaluation (content digest)
"""

from __future__ import annotations

import base64
import hashlib
import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from sdv_bmt.contracts import labelmap
from sdv_bmt.contracts.ids import digest_of
from sdv_bmt.contracts.quality import CITYSCAPES_SEMSEG_MIOU_V1
from sdv_bmt.contracts.records import OutputReference, QualityReferenceRow
from sdv_bmt.contracts.status import MetricStatus, RequestState
from sdv_bmt.metrics import metric_set_v1
from sdv_bmt.metrics.quality_segmentation import ConfusionCounts, ground_truth, segmentation_protocol, segmentation_quality
from tests._helpers import METRIC_SET_V1, METRIC_SET_V1_1, REPO_ROOT, evaluate, metric_map


def _direct(pairs):
    tp, fp, fn = [0] * 19, [0] * 19, [0] * 19
    for pred, gt in pairs:
        for p, g in zip(pred, gt):
            if g == 255:
                continue
            if p == g:
                tp[p] += 1
            else:
                fp[p] += 1
                fn[g] += 1
    ious = {c: tp[c] / (tp[c] + fp[c] + fn[c]) for c in range(19) if tp[c] + fp[c] + fn[c]}
    return sum(ious.values()) / len(ious), ious


def test_counts_equal_a_direct_per_pixel_count():
    rng = random.Random(3)
    pairs = []
    for _ in range(5):
        gt = bytes(rng.choice([255] + list(range(12))) for _ in range(4000))  # classes 12..18 absent
        pred = bytes(g if g != 255 and rng.random() < 0.7 else rng.randrange(14) for g in gt)
        pairs.append((pred, gt))
    cc = ConfusionCounts()
    for pred, gt in pairs:
        cc.add(pred, gt)
    miou, ious = cc.result()
    ref, ref_ious = _direct(pairs)
    assert miou == pytest.approx(ref, abs=1e-15)
    assert {c: v for c, v in ious.items() if v is not None} == pytest.approx(ref_ious, abs=1e-15)
    assert all(ious[c] is None for c in (14, 15, 16, 17, 18))  # never predicted nor present: left out of the mean
    assert ious[12] == 0.0 or ious[12] is not None  # predicted but absent: counted (IoU 0)


def _annotation(labels, w, h):
    return {"encoding": labelmap.CODEC_ID, "width": w, "height": h,
            "labels_b64": base64.b64encode(labelmap.encode(labels, w, h)).decode()}


def test_ground_truth_is_validated():
    w, h = 8, 2
    assert ground_truth(_annotation(bytes([255, 0] * 8), w, h))[2] == bytes([255, 0] * 8)
    for bad in ({**_annotation(bytes(16), w, h), "encoding": "x"}, {**_annotation(bytes(16), w, h), "width": 9},
                _annotation(bytes([40] * 16), w, h), {**_annotation(bytes(16), w, h), "labels_b64": "!!"}):
        with pytest.raises(ValueError):
            ground_truth(bad)
    item_blob = labelmap.encode(bytes(16), w, h, 3)  # a ground truth must not carry an item index
    with pytest.raises(ValueError):
        ground_truth({**_annotation(bytes(16), w, h), "labels_b64": base64.b64encode(item_blob).decode()})


def _bundle(tmp_path, gts, preds, w, h, *, item_of=None):
    outputs, folded, items, rows = bytearray(), {}, [], []
    for i, (gt, pred) in enumerate(zip(gts, preds)):
        ann = _annotation(gt, w, h)
        items.append(SimpleNamespace(item_index=i, source_item_id=f"img{i}", annotation_digest=digest_of(ann)))
        rows.append(QualityReferenceRow(1, i, f"img{i}", digest_of(ann), ann))
        blob = labelmap.encode(pred, w, h, i if item_of is None else item_of(i))
        ref = OutputReference(i, "outputs.bin", len(outputs), len(blob), hashlib.sha256(blob).hexdigest(),
                              labelmap.CODEC_ID)
        outputs += blob
        folded[i] = (SimpleNamespace(request_id=i, item_indices=(i,), state=RequestState.COMPLETED),
                     SimpleNamespace(output_ref=ref))
    (tmp_path / "outputs.bin").write_bytes(bytes(outputs))
    bundle = SimpleNamespace(run_dir=tmp_path, input_items=items, quality_reference=rows,
                             manifest={"quality_reference": {"evaluator": CITYSCAPES_SEMSEG_MIOU_V1,
                                                             "file": "quality_reference.jsonl"}},
                             run_spec={"resolved_plan": {"workload": {"quality_evaluator": CITYSCAPES_SEMSEG_MIOU_V1}}})
    return bundle, folded


def test_bundle_only_miou(tmp_path):
    rng = random.Random(5)
    w, h = 32, 4
    gts = [bytes(rng.choice([255, 0, 1, 2, 3]) for _ in range(w * h)) for _ in range(3)]
    preds = [bytes(g if g != 255 and rng.random() < 0.8 else rng.randrange(5) for g in gt) for gt in gts]
    bundle, folded = _bundle(tmp_path, gts, preds, w, h)
    q = segmentation_quality(bundle, folded, segmentation_protocol())
    assert q.status is MetricStatus.VALID and q.evaluated_items == 3 and q.annotation_check == (3, 3)
    assert q.miou == pytest.approx(_direct(list(zip(preds, gts)))[0], abs=1e-15)


def test_output_bound_to_another_item_is_invalid(tmp_path):
    w, h = 8, 1
    bundle, folded = _bundle(tmp_path, [bytes(8)] * 2, [bytes(8)] * 2, w, h, item_of=lambda i: 1 - i)
    q = segmentation_quality(bundle, folded, segmentation_protocol())
    assert q.status is MetricStatus.INVALID and q.reason == "task_output_invalid"


def test_detection_protocol_makes_segmentation_not_applicable(tmp_path):
    bundle, folded = _bundle(tmp_path, [bytes(8)], [bytes(8)], 8, 1)
    q = segmentation_quality(bundle, folded, {"evaluator": "coco-detection-bbox/1", "implemented": True})
    assert q.status is MetricStatus.NOT_APPLICABLE


def test_catalog_v1_1_extends_v1():
    v1 = [m.metric_id for m in metric_set_v1.CATALOG_METRICS[METRIC_SET_V1]]
    v11 = [m.metric_id for m in metric_set_v1.CATALOG_METRICS[METRIC_SET_V1_1]]
    assert v11[:len(v1)] == v1 and v11[len(v1):] == ["quality.segmentation.coverage",
                                                       "quality.segmentation.conditional.miou"]


def _yolox_bundle():
    roots = sorted((REPO_ROOT / "runs" / "v0.1.2-yolox" / "baseline").glob("yolox-s-baseline/run-*"))
    if not roots:
        pytest.skip("NOT RUN: v0.1.2 YOLOX evidence is not present on this host")
    return roots[0]


def test_v1_1_keeps_every_v1_value_of_a_yolox_bundle(tmp_path):
    run = shutil.copytree(_yolox_bundle(), tmp_path / "run", ignore=shutil.ignore_patterns("evaluations"))
    m1 = metric_map(evaluate(run, "e-v1", catalog=METRIC_SET_V1))
    m11 = metric_map(evaluate(run, "e-v11", catalog=METRIC_SET_V1_1))
    for metric_id, r in m1.items():
        assert (m11[metric_id].status, m11[metric_id].value) == (r.status, r.value), metric_id
    assert m11["quality.detection.conditional.ap"].value == 0.4289194068359202
    for metric_id in ("quality.segmentation.coverage", "quality.segmentation.conditional.miou"):
        assert m11[metric_id].status is MetricStatus.NOT_APPLICABLE


def test_v1_catalog_reproduces_the_archived_yolox_evaluation(tmp_path):
    src = _yolox_bundle()
    archived = [json.loads(p.read_text()) for p in sorted((src / "evaluations").glob("*/eval_manifest.json"))]
    archived = [e for e in archived if e["evaluator"]["catalog_version"] == METRIC_SET_V1]
    assert archived
    run = shutil.copytree(src, tmp_path / "run", ignore=shutil.ignore_patterns("evaluations"))
    outcome = evaluate(run, "e-repro", catalog=METRIC_SET_V1)
    assert outcome.metrics_content_sha256 == archived[0]["metrics_content_sha256"]
