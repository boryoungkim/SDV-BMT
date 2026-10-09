"""Canonical detection quality (Metric Set v1, ``coco-detection-bbox/1``).

* The canonical COCO module computes exactly what the cross-validated analysis-plane tool
  (``tools/stage2b_g_real/coco_eval.py``, checked against pycocotools 2.0.11) computes.
* Threshold-defined TP/FP/FN follow the declared matching rules (hand-computed cases) and agree
  with the COCO ``evaluateImg`` matching path on real data.
* Quality is conditional (completed outputs only) and always comes with its coverage.
* Regression anchor: the Stage 2b-G YOLOX-S/COCO outputs give AP 0.4289194068359202 again.
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import pytest

from sdv_bmt.contracts.ids import canonical_json_bytes, digest_of, sha256_hex
from sdv_bmt.contracts.quality import COCO_2017_CATEGORY_IDS, COCO_DETECTION_BBOX_V1, QUALITY_EVALUATORS
from sdv_bmt.contracts.records import (
    CompletionRecord,
    InputItem,
    OutputReference,
    QualityReferenceRow,
    RequestRecord,
)
from sdv_bmt.contracts.specs import DetectionOperatingPoint
from sdv_bmt.contracts.status import MetricStatus, RequestState
from sdv_bmt.metrics import quality_coco
from sdv_bmt.metrics.quality_detection import detection_quality, quality_protocol
from sdv_bmt.observation.reader import RunBundle
from tests._helpers import REPO_ROOT

TOOL = REPO_ROOT / "tools" / "stage2b_g_real" / "coco_eval.py"
EVIDENCE = REPO_ROOT / "runs" / "stage2b-g-real" / "analysis" / "quality"
DUMP = EVIDENCE / "coco_eval_inputs_run-20261007T201237Z-86f737e638ed.json"
RECORDED = EVIDENCE / "quality_br-yolox-30hz_run-20261007T201237Z-86f737e638ed.json"


def _tool():
    spec = importlib.util.spec_from_file_location("stage2b_coco_eval_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _random_case(seed: int):
    rng = random.Random(seed)
    cats = [1, 2, 3, 18, 90]
    gts, dets = {}, {}
    ann_id = 1
    for img in range(1, 31):
        objs = []
        for _ in range(rng.randint(0, 6)):
            w, h = rng.uniform(4, 200), rng.uniform(4, 200)
            objs.append({"id": ann_id, "category_id": rng.choice(cats), "bbox": [rng.uniform(0, 400), rng.uniform(0, 400), w, h],
                         "area": w * h * rng.uniform(0.5, 1.0), "iscrowd": 1 if rng.random() < 0.08 else 0})
            ann_id += 1
        gts[img] = objs
        out = []
        for o in objs:  # noisy detections of real objects, sometimes duplicated or with a wrong class
            for _ in range(rng.choice((0, 1, 1, 2))):
                x, y, w, h = o["bbox"]
                out.append({"category_id": o["category_id"] if rng.random() < 0.85 else rng.choice(cats),
                            "bbox": [x + rng.uniform(-8, 8), y + rng.uniform(-8, 8), w * rng.uniform(0.8, 1.2),
                                     h * rng.uniform(0.8, 1.2)],
                            "score": round(rng.random(), 2)})  # rounded: many score ties
        for _ in range(rng.randint(0, 4)):  # pure false positives
            out.append({"category_id": rng.choice(cats), "bbox": [rng.uniform(0, 400), rng.uniform(0, 400),
                                                                    rng.uniform(4, 80), rng.uniform(4, 80)],
                        "score": round(rng.random(), 2)})
        dets[img] = out
    return gts, dets, cats


@pytest.mark.parametrize("seed", range(6))
def test_canonical_coco_module_equals_the_cross_validated_tool(seed):
    gts, dets, cats = _random_case(seed)
    ours, tool = quality_coco.evaluate(gts, dets, cats), _tool().evaluate(gts, dets, cats)
    assert ours == tool  # every statistic bit for bit, per-category AP and counts


def test_coco_module_source_is_the_tool_algorithm():
    # The evaluation functions are a verbatim port (only the module docstring differs).
    import inspect

    tool = _tool()
    for name in ("_linspace", "box_iou", "_evaluate_img", "evaluate"):
        ours = inspect.getsource(getattr(quality_coco, name)).replace('    """The 12 standard statistics (-1 where COCO has no '
                                                                     'defined value) plus per-category AP."""\n', "")
        assert ours == inspect.getsource(getattr(tool, name)), name


# ---------------------------------------------------------------- operating point (hand-computed)


def _box(x, y, w, h):
    return [float(x), float(y), float(w), float(h)]


def test_threshold_counts_duplicates_class_mismatch_and_score_cut():
    gts = {1: [{"id": 1, "category_id": 1, "bbox": _box(0, 0, 10, 10), "area": 100.0, "iscrowd": 0}]}
    dets = {1: [{"category_id": 1, "bbox": _box(0, 0, 10, 10), "score": 0.9},   # TP
                {"category_id": 1, "bbox": _box(0, 0, 10, 10), "score": 0.8},   # duplicate on a matched box: FP
                {"category_id": 2, "bbox": _box(0, 0, 10, 10), "score": 0.7},   # wrong class: FP
                {"category_id": 1, "bbox": _box(0, 0, 10, 10), "score": 0.4}]}  # below the score threshold
    k = quality_coco.threshold_counts(gts, dets, [1, 2], 0.5, 0.5, 100)
    assert (k["tp"], k["fp"], k["fn"], k["detections_considered"], k["gt_non_crowd"]) == (1, 2, 0, 3, 1)


def test_threshold_counts_crowd_regions_and_iou_boundary():
    gts = {1: [{"id": 1, "category_id": 1, "bbox": _box(0, 0, 100, 100), "area": 10000.0, "iscrowd": 1}],
           2: [{"id": 2, "category_id": 1, "bbox": _box(0, 0, 10, 10), "area": 100.0, "iscrowd": 0}],
           3: [{"id": 3, "category_id": 1, "bbox": _box(0, 0, 10, 10), "area": 100.0, "iscrowd": 0}]}
    dets = {1: [{"category_id": 1, "bbox": _box(10, 10, 10, 10), "score": 0.9},   # inside a crowd region: ignored
                {"category_id": 1, "bbox": _box(20, 20, 10, 10), "score": 0.9}],  # crowd absorbs several: ignored
            2: [{"category_id": 1, "bbox": _box(0, 0, 10, 20), "score": 0.9}],     # IoU exactly 0.5: TP
            3: [{"category_id": 1, "bbox": _box(5, 0, 10, 10), "score": 0.9}]}     # IoU 1/3 < 0.5: FP, GT is FN
    k = quality_coco.threshold_counts(gts, dets, [1], 0.5, 0.5, 100)
    assert (k["tp"], k["fp"], k["fn"], k["ignored_crowd_matches"], k["gt_non_crowd"]) == (1, 1, 1, 2, 2)


def test_threshold_counts_respect_the_per_image_detection_limit():
    gts = {1: [{"id": i + 1, "category_id": 1, "bbox": _box(20 * i, 0, 10, 10), "area": 100.0, "iscrowd": 0}
               for i in range(3)]}
    dets = {1: [{"category_id": 1, "bbox": _box(20 * i, 0, 10, 10), "score": 0.9 - 0.1 * i} for i in range(3)]}
    assert quality_coco.threshold_counts(gts, dets, [1], 0.5, 0.5, 2)["tp"] == 2
    assert quality_coco.threshold_counts(gts, dets, [1], 0.5, 0.5, 2)["fn"] == 1


# ---------------------------------------------------------------- bundle-level (synthetic in-memory bundle)


def _annotation(image_id, boxes):
    return {"file_name": f"{image_id}.jpg", "height": 100, "image_id": image_id, "width": 100,
            "objects": [{"area": float(w * h), "bbox": [float(x), float(y), float(w), float(h)], "category_id": c,
                         "id": image_id * 100 + i, "iscrowd": 0} for i, (c, x, y, w, h) in enumerate(boxes)]}


def _output(item_index, image_id, dets):
    return {"schema": "sdv-yolox-coco-detections/1", "image_id": image_id, "item_index": item_index,
            "detections": [{"bbox_mpx": [x * 1000, y * 1000, w * 1000, h * 1000], "category_id": c,
                            "score_ppm": s} for c, x, y, w, h, s in dets]}


def synthetic_bundle(tmp_path, cases, states, outputs=None, tamper=None):
    """cases: [(image_id, gt boxes, detections)], states: per item RequestState of its request."""
    items, rows, folded = [], [], {}
    blobs = bytearray()
    for i, (image_id, boxes, dets) in enumerate(cases):
        ann = _annotation(image_id, boxes)
        items.append(InputItem(i, f"{image_id:012d}", f"index.jsonl#{image_id:012d}", "0" * 64,
                               f"index.jsonl#{image_id:012d}/annotation", digest_of(ann), None, None))
        rows.append(QualityReferenceRow(1, i, f"{image_id:012d}", digest_of(ann), (tamper or {}).get(i, ann)))
        state = states[i]
        issued = state is not RequestState.DROPPED
        req = RequestRecord(i, (i,), "measure", i * 10, "issued" if issued else "dropped", None if issued else "busy",
                            i * 10 if issued else None, i * 10 if issued else None, i * 10 + 1 if issued else None, state)
        comp = None
        if state is RequestState.COMPLETED:
            blob = (outputs or {}).get(i) or canonical_json_bytes(_output(i, image_id, dets))
            ref = OutputReference(i, "outputs.bin", len(blobs), len(blob), sha256_hex(blob), "canonical-json-utf8/1")
            blobs += blob
            comp = CompletionRecord(i, state, i * 10 + 5, i * 10 + 6, ref, None, None, None, None)
        folded[i] = (req, comp)
    (tmp_path / "outputs.bin").write_bytes(bytes(blobs))
    manifest = {"quality_reference": {"evaluator": COCO_DETECTION_BBOX_V1, "file": "quality_reference.jsonl"}}
    run_spec = {"resolved_plan": {"workload": {"quality_evaluator": COCO_DETECTION_BBOX_V1}}}
    bundle = RunBundle(tmp_path, manifest, run_spec, tuple(items), (), (), tuple(rows))
    return bundle, folded


CASES = [
    (101, [(1, 0, 0, 10, 10), (3, 50, 50, 20, 20)], [(1, 0, 0, 10, 10, 900000), (3, 50, 50, 20, 20, 800000)]),
    (102, [(1, 0, 0, 30, 30)], [(1, 0, 0, 30, 30, 700000), (18, 0, 0, 30, 30, 600000)]),
    (103, [(2, 10, 10, 40, 40)], [(2, 10, 10, 40, 40, 950000)]),
]


def test_conditional_quality_with_coverage_below_one(tmp_path):
    states = [RequestState.COMPLETED, RequestState.COMPLETED, RequestState.DROPPED]
    bundle, folded = synthetic_bundle(tmp_path, CASES, states)
    q = detection_quality(bundle, folded, quality_protocol(bundle, None))
    assert q.status is MetricStatus.VALID and q.evaluated_items == 2 and q.annotation_check == (3, 3)
    assert q.population["images"] == 2 and q.population["gt_objects"] == 3
    # item 2 (dropped) is not in the conditional population: its ground truth is not a false negative
    assert (q.counts["tp"], q.counts["fp"], q.counts["fn"]) == (3, 1, 0)
    assert "conditional: 2 of 3 planned requests" in q.notes[0]
    (tmp_path / "all").mkdir()
    all_bundle, all_folded = synthetic_bundle(tmp_path / "all", CASES, [RequestState.COMPLETED] * 3)
    full = detection_quality(all_bundle, all_folded, quality_protocol(all_bundle, None))
    assert full.evaluated_items == 3 and full.counts["tp"] == 4 and full.stats != q.stats


def test_operating_point_override_changes_only_threshold_counts(tmp_path):
    bundle, folded = synthetic_bundle(tmp_path, CASES, [RequestState.COMPLETED] * 3)
    default = detection_quality(bundle, folded, quality_protocol(bundle, None))
    strict = detection_quality(bundle, folded, quality_protocol(bundle, DetectionOperatingPoint(0.75, 0.5)))
    assert default.stats == strict.stats  # AP/AR do not depend on the operating point
    assert (default.counts["tp"], strict.counts["tp"]) == (4, 3)  # the 0.7-score box falls below 0.75
    assert quality_protocol(bundle, None)["operating_point"]["score_threshold"] == 0.5


@pytest.mark.parametrize("problem", ["schema", "image_id", "digest"])
def test_invalid_task_output_invalidates_quality(tmp_path, problem):
    bad = {"schema": canonical_json_bytes({"schema": "other/1", "image_id": 101, "item_index": 0, "detections": []}),
           "image_id": canonical_json_bytes(_output(0, 999, [])),
           "digest": None}[problem]
    bundle, folded = synthetic_bundle(tmp_path, CASES, [RequestState.COMPLETED] * 3, outputs={0: bad} if bad else None)
    if problem == "digest":
        (tmp_path / "outputs.bin").write_bytes(b"X" + (tmp_path / "outputs.bin").read_bytes()[1:])
    q = detection_quality(bundle, folded, quality_protocol(bundle, None))
    assert (q.status, q.reason) == (MetricStatus.INVALID, "task_output_invalid")


def test_frozen_reference_must_match_the_item_annotation_digests(tmp_path):
    tampered = _annotation(102, [(1, 0, 0, 31, 30)])
    bundle, folded = synthetic_bundle(tmp_path, CASES, [RequestState.COMPLETED] * 3, tamper={1: tampered})
    q = detection_quality(bundle, folded, quality_protocol(bundle, None))
    assert (q.status, q.reason, q.annotation_check) == (MetricStatus.INVALID, "quality_reference_integrity", (2, 3))


def test_no_frozen_reference_means_not_collected(tmp_path):
    bundle, folded = synthetic_bundle(tmp_path, CASES, [RequestState.COMPLETED] * 3)
    from dataclasses import replace

    plain = replace(bundle, quality_reference=None, manifest={}, run_spec={"resolved_plan": {"workload": {}}})
    q = detection_quality(plain, folded, quality_protocol(plain, None))
    assert (q.status, q.reason, q.annotation_check) == (MetricStatus.NOT_COLLECTED, "quality_evaluator_not_declared",
                                                        None)


def test_declared_category_set_is_the_coco_2017_set():
    from sdv_bmt.workload.yolox import COCO_CATEGORY_IDS

    assert COCO_2017_CATEGORY_IDS == COCO_CATEGORY_IDS and len(COCO_2017_CATEGORY_IDS) == 80
    assert QUALITY_EVALUATORS[COCO_DETECTION_BBOX_V1].task == "coco-2017-object-detection"
    manifest = REPO_ROOT / "datasets" / "coco-val2017-first1000" / "dataset_manifest.json"
    if manifest.is_file():  # local dataset asset (Stage 2b-G), categories copied from instances_val2017.json
        assert tuple(c["id"] for c in json.loads(manifest.read_text())["categories"]) == COCO_2017_CATEGORY_IDS


# ---------------------------------------------------------------- regression anchor (local Stage 2b-G evidence)


@pytest.fixture(scope="module")
def baseline_inputs():
    if not (DUMP.is_file() and RECORDED.is_file()):
        pytest.skip("Stage 2b-G evidence not present on this host")
    data = json.loads(DUMP.read_text())
    gts = {int(k): v for k, v in data["gts"].items()}
    dets = {int(k): v for k, v in data["dets"].items()}
    return gts, dets, data["categories"]


def test_regression_anchor_ap_of_the_stage2b_g_baseline(baseline_inputs):
    gts, dets, cats = baseline_inputs
    assert tuple(cats) == COCO_2017_CATEGORY_IDS
    stats = quality_coco.evaluate(gts, dets, list(COCO_2017_CATEGORY_IDS))["stats"]
    recorded = json.loads(RECORDED.read_text())["conditional"]["stats"]
    assert stats == recorded
    assert stats["AP"] == 0.4289194068359202
    assert round(stats["AP50"], 5) == 0.61348 and round(stats["AP75"], 5) == 0.46807


def test_threshold_counts_agree_with_the_coco_matching_path(baseline_inputs):
    gts, dets, cats = baseline_inputs
    for score in (0.5, 0.3):
        k = quality_coco.threshold_counts(gts, dets, cats, score, 0.5, 100)
        tp = fp = npig = 0
        for img in sorted(gts):
            kept = [d for d in dets.get(img, ()) if d["score"] >= score]
            for cat in cats:
                gt = [g for g in gts[img] if g["category_id"] == cat]
                dt = sorted(({"id": n + 1, "area": d["bbox"][2] * d["bbox"][3], **d} for n, d in enumerate(kept)
                             if d["category_id"] == cat), key=lambda d: -d["score"])
                ious = [[quality_coco.box_iou(d["bbox"], g["bbox"], bool(g["iscrowd"])) for g in gt] for d in dt] \
                    if gt and dt else []
                e = quality_coco._evaluate_img(gt, dt, ious, quality_coco.AREA_RANGES[0], 100)
                if e is None:
                    continue
                t = 0  # IoU threshold 0.50
                tp += sum(1 for m, ig in zip(e["dtm"][t], e["dt_ig"][t]) if m and not ig)
                fp += sum(1 for m, ig in zip(e["dtm"][t], e["dt_ig"][t]) if not m and not ig)
                npig += sum(1 for ig in e["gt_ig"] if ig == 0)
        assert (k["tp"], k["fp"], k["fn"], k["gt_non_crowd"]) == (tp, fp, npig - tp, npig)
