#!/usr/bin/env python3
"""Cross-check of the stdlib COCO evaluator (coco_eval.py) against pycocotools, Stage 2b-G (offline tool).

Run with the tools-only venv that holds pycocotools (approved 2026-10-08; not the project .venv):
    .venv-tools/bin/python tools/stage2b_g_real/coco_eval_crosscheck.py INPUTS.json OUT.json
INPUTS.json is written by ``coco_quality.py --dump-inputs`` (ground truth, detections, category ids).  Both
evaluators receive exactly the same boxes, scores, categories, image set and ground-truth order.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pycocotools  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from coco_eval import STAT_NAMES, evaluate  # noqa: E402


def main() -> int:
    inputs_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    data = json.loads(inputs_path.read_text())
    cat_ids = data["categories"]
    gts = {int(k): v for k, v in data["gts"].items()}
    dets = {int(k): v for k, v in data["dets"].items()}
    gt_json = {"images": [{"id": i} for i in sorted(gts)], "categories": [{"id": c} for c in cat_ids],
               "annotations": [dict(g, image_id=i) for i in sorted(gts) for g in gts[i]]}
    results = [{"image_id": i, "category_id": d["category_id"], "bbox": d["bbox"], "score": d["score"]}
               for i in sorted(dets) for d in dets[i]]
    with contextlib.redirect_stdout(io.StringIO()) as log:
        coco_gt = COCO()
        coco_gt.dataset = gt_json
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(results)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.imgIds = sorted(gts)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    ref = dict(zip(STAT_NAMES, (float(x) for x in ev.stats)))
    ours = evaluate(gts, dets, cat_ids)
    precision = ev.eval["precision"][:, :, :, 0, 2]
    per_cat_ref = {}
    for k, c in enumerate(sorted(cat_ids)):
        s = precision[:, :, k]
        per_cat_ref[c] = float(np.mean(s[s > -1])) if (s > -1).any() else -1.0
    stat_diff = {n: abs(ref[n] - ours["stats"][n]) for n in STAT_NAMES}
    cat_diff = max(abs(per_cat_ref[c] - ours["per_category_ap"][c]) for c in per_cat_ref)
    result = {
        "schema": "sdv-bmt-stage2b-g-real-coco-crosscheck/1",
        "inputs": str(inputs_path), "inputs_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
        "pycocotools_version": getattr(pycocotools, "__version__", "unknown"), "numpy_version": np.__version__,
        "pycocotools": ref, "stdlib": ours["stats"], "abs_diff": stat_diff, "max_abs_diff": max(stat_diff.values()),
        "per_category_max_abs_diff": cat_diff, "images": ours["images"], "detections": ours["detections"],
        "pycocotools_log": log.getvalue(),
    }
    out_path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"max_abs_diff": result["max_abs_diff"], "per_category_max_abs_diff": cat_diff,
                      "pycocotools_AP": ref["AP"], "stdlib_AP": ours["stats"]["AP"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
