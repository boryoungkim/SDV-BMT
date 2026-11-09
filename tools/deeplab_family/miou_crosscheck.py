#!/usr/bin/env python3
"""Gate 8 (plan v03 7.3): the bundle-only mIoU evaluator vs the official Cityscapes evaluation script (tools only).

Run with a tools-only environment ``<venv-tools-seg>`` (cityscapesscripts 2.2.4).  The official
``evalPixelLevelSemanticLabeling.evaluateImgLists`` (pixel-level class IoU, ``averageScoreClasses``; instance-level
scores off) is run on label-id PNGs, and its mIoU and class IoUs are compared with ``ConfusionCounts`` of
``sdv_bmt.metrics.quality_segmentation`` (the evaluator behind ``quality.segmentation.conditional.miou``).  The
label-id <-> train-id table of the dataset builder is checked against cityscapesscripts ``labels.py`` as well.

Modes:
  --synthetic N               N random train-id ground truths (with ignored regions) and noisy predictions
  --bundle RUN_DIR --gt-zip Z the GPU label maps recorded in a capture bundle against the official gtFine label-id
                              PNGs read from the gtFine zip (the bundle's own frozen ground truth must agree with them)

    <venv-tools-seg>/bin/python tools/deeplab_family/miou_crosscheck.py --synthetic 8 --out R.json
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
from sdv_bmt.contracts import labelmap  # noqa: E402
from sdv_bmt.metrics.quality_segmentation import ConfusionCounts, ground_truth  # noqa: E402
from build_cityscapes_val import LABEL_ID_TO_TRAIN_ID, TRAIN_ID_TABLE  # noqa: E402

DIFF_MAX = 1e-9
TRAIN_TO_ID = {t: i for i, t in LABEL_ID_TO_TRAIN_ID.items()}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_eval(pairs: list[tuple[np.ndarray, np.ndarray]], tmp: Path) -> dict:
    """pairs of (prediction label ids, ground-truth label ids) uint8 HxW."""
    from cityscapesscripts.evaluation import evalPixelLevelSemanticLabeling as E  # noqa: E402
    preds, gts = [], []
    for k, (p, g) in enumerate(pairs):
        pf, gf = tmp / f"img{k:04d}_pred.png", tmp / f"img{k:04d}_gt_labelIds.png"
        Image.fromarray(p).save(pf)
        Image.fromarray(g).save(gf)
        preds.append(str(pf))
        gts.append(str(gf))
    args = E.args
    args.evalInstLevelScore = False
    args.evalPixelAccuracy = False
    args.JSONOutput = False
    args.quiet = True
    res = E.evaluateImgLists(preds, gts, args)
    return {"miou": float(res["averageScoreClasses"]), "class_iou": res["classScores"]}


def check_label_table() -> list[str]:
    from cityscapesscripts.helpers.labels import labels  # noqa: E402
    problems = []
    for lab in labels:
        if lab.id < 0:
            continue
        ours = TRAIN_ID_TABLE[lab.id]
        official = lab.trainId if (0 <= lab.trainId < 19 and not lab.ignoreInEval) else 255
        if ours != official:
            problems.append(f"label id {lab.id} ({lab.name}): ours {ours}, labels.py {official}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", type=int)
    src.add_argument("--bundle")
    ap.add_argument("--gt-zip")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import cityscapesscripts  # noqa: E402

    table_problems = check_label_table()
    ours = ConfusionCounts()
    pairs = []
    if a.synthetic:
        rng = np.random.default_rng(8)
        h, w = 256, 512
        for _ in range(a.synthetic):
            gt = rng.integers(0, 19, size=(h // 16, w // 16)).repeat(16, 0).repeat(16, 1).astype(np.uint8)
            gt[rng.random((h, w)) < 0.05] = 255
            gt[:, :rng.integers(0, 64)] = 255
            pred = gt.copy()
            flip = rng.random((h, w)) < 0.15
            pred[flip] = rng.integers(0, 19, size=int(flip.sum()))
            pred[pred == 255] = rng.integers(0, 19)
            ours.add(pred.tobytes(), gt.tobytes())
            to_id = np.vectorize(lambda t: TRAIN_TO_ID.get(int(t), 0), otypes=[np.uint8])
            pairs.append((to_id(pred), to_id(gt)))  # ignored ground truth -> id 0 (unlabeled, ignoreInEval)
        source = {"synthetic": a.synthetic, "size": [w, h], "seed": 8}
    else:
        bdir = Path(a.bundle)
        if not a.gt_zip:
            raise SystemExit("--bundle needs --gt-zip")
        gz = zipfile.ZipFile(a.gt_zip)
        refs = {json.loads(l)["item_index"]: json.loads(l) for l in open(bdir / "quality_reference.jsonl")}
        items = {json.loads(l)["item_index"]: json.loads(l) for l in open(bdir / "input_items.jsonl")}
        outputs = (bdir / "outputs.bin").read_bytes()
        mismatched_gt = 0
        request_item = {}
        for line in open(bdir / "events.jsonl", encoding="utf-8"):
            e = json.loads(line)
            if e.get("kind") == "issued" and e.get("phase") == "measure":
                request_item[e["request_id"]] = e["item_indices"][0]
            elif e.get("kind") == "completed" and e["request_id"] in request_item:
                ref = e["output_ref"]
                idx = request_item[e["request_id"]]
                _, ww, hh, pred = labelmap.decode(outputs[ref["offset"]:ref["offset"] + ref["length"]])
                _, _, frozen = ground_truth(refs[idx]["annotation"])
                member = refs[idx]["annotation"]["source"]
                gt_ids = np.array(Image.open(io.BytesIO(gz.read(member))), dtype=np.uint8)
                if gt_ids.tobytes().translate(TRAIN_ID_TABLE) != frozen:
                    mismatched_gt += 1
                ours.add(pred, frozen)
                p = np.frombuffer(pred, dtype=np.uint8).reshape(hh, ww)
                pairs.append((np.vectorize(lambda t: TRAIN_TO_ID[int(t)], otypes=[np.uint8])(p), gt_ids))
        source = {"bundle": str(bdir), "items": len(pairs), "frozen_gt_differs_from_official_png": mismatched_gt,
                  "gt_zip": Path(a.gt_zip).name}
    with tempfile.TemporaryDirectory(prefix="sdv-bmt-csEval-") as tmp:
        off = official_eval(pairs, Path(tmp))
    miou, ious = ours.result()
    names = list(off["class_iou"].keys())
    class_diff = {}
    from cityscapesscripts.helpers.labels import name2label  # noqa: E402
    for name, v in off["class_iou"].items():
        tid = name2label[name].trainId
        if 0 <= tid < 19 and not name2label[name].ignoreInEval:
            mine = ious[tid]
            class_diff[name] = None if mine is None or v != v else abs(mine - v)
    diff = abs(miou - off["miou"])
    report = {"schema": "sdv-bmt-deeplab-miou-crosscheck/1", "source": source,
              "cityscapesscripts": getattr(cityscapesscripts, "__version__", "2.2.4 (pip)"),
              "label_table_problems": table_problems, "miou_ours": miou, "miou_official": off["miou"],
              "diff": diff, "class_iou_max_diff": max(v for v in class_diff.values() if v is not None),
              "criterion": f"|diff| <= {DIFF_MAX}", "evaluated_classes": len(names),
              "tool_sha256": sha256_file(Path(__file__)),
              "passed": diff <= DIFF_MAX and not table_problems
              and source.get("frozen_gt_differs_from_official_png", 0) == 0}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("miou_ours", "miou_official", "diff", "class_iou_max_diff",
                                             "label_table_problems", "passed")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
