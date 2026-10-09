#!/usr/bin/env python3
"""Per-request evidence chain of a Metric Set v1 (manifest v5) bundle, from the bundle only (tool).

For one request: item -> payload digest -> frozen ground truth row (re-verified against the item's
annotation_digest) -> request timestamps and the derived timing views -> device segments -> output bytes
(re-verified against output_ref) -> detections -> TP/FP/FN of that image at the evaluation's operating point.

    .venv/bin/python tools/metric_set_v1/evidence_chain_v1.py RUN_DIR REQUEST_ID OUT.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from sdv_bmt.contracts.ids import digest_of  # noqa: E402
from sdv_bmt.contracts.quality import COCO_2017_CATEGORY_IDS  # noqa: E402
from sdv_bmt.metrics import quality_coco  # noqa: E402
from sdv_bmt.observation.reader import fold_requests, load_run_bundle, read_output_blob  # noqa: E402


def main() -> int:
    run_dir, rid, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    bundle = load_run_bundle(run_dir)
    request, completion = fold_requests(bundle.events)[rid]
    item = bundle.input_items[request.item_indices[0]]
    row = next(r for r in bundle.quality_reference if r.item_index == item.item_index)
    blob = read_output_blob(run_dir, completion.output_ref)
    output = json.loads(blob)
    obs = [e for e in bundle.events if e["kind"] == "device_observation" and e["request_id"] == rid]
    em = sorted((json.loads(p.read_text()) for p in (run_dir / "evaluations").glob("*/eval_manifest.json")),
                key=lambda e: e["created_at_utc"])[-1]
    point = em["protocol"]["quality"]["operating_point"]
    ann = row.annotation
    gts = {ann["image_id"]: [{k: o[k] for k in ("id", "category_id", "bbox", "area", "iscrowd")} for o in ann["objects"]]}
    dets = {ann["image_id"]: [{"category_id": d["category_id"], "bbox": [v / 1000.0 for v in d["bbox_mpx"]],
                               "score": d["score_ppm"] / 1e6} for d in output["detections"]]}
    counts = quality_coco.threshold_counts(gts, dets, list(COCO_2017_CATEGORY_IDS), point["score_threshold"],
                                           point["iou_threshold"], point["max_detections_per_image"])
    chain = {
        "run_id": bundle.manifest["run_id"], "request_id": rid, "eval_id": em["eval_id"],
        "item": {"item_index": item.item_index, "source_item_id": item.source_item_id,
                 "payload_sha256": item.payload_sha256, "annotation_digest": item.annotation_digest},
        "frozen_ground_truth": {"image_id": ann["image_id"], "objects": len(ann["objects"]),
                                "crowd": sum(1 for o in ann["objects"] if o["iscrowd"]),
                                "digest_reverified": digest_of(row.annotation) == item.annotation_digest == row.annotation_digest},
        "timestamps_ns": {"t_sched": request.t_sched, "t_issue": request.t_issue,
                          "t_submit_begin": request.t_submit_begin, "t_submit_end": request.t_submit_end,
                          "t_complete": completion.t_complete, "t_output_ready": completion.t_output_ready},
        "timing_views_ns": {"issue_lag": request.t_issue - request.t_sched,
                            "submit": request.t_submit_end - request.t_submit_begin,
                            "post_submit": completion.t_complete - request.t_submit_end,
                            "dut_service": completion.t_complete - request.t_submit_begin,
                            "request_completion": completion.t_complete - request.t_sched,
                            "output_stage": completion.t_output_ready - completion.t_complete,
                            "output_ready": completion.t_output_ready - request.t_sched},
        "device_observation": {"timing_ns": obs[0]["timing"], "timing_method": obs[0]["timing_method"],
                               "observations": obs[0]["observations"]} if obs else None,
        "output": {"sha256": completion.output_ref.sha256, "length": completion.output_ref.length,
                   "bytes_reverified": hashlib.sha256(blob).hexdigest() == completion.output_ref.sha256,
                   "image_id": output["image_id"], "detections": len(output["detections"])},
        "operating_point": {"score_threshold": point["score_threshold"], "iou_threshold": point["iou_threshold"]},
        "image_counts_at_operating_point": counts,
    }
    out.write_text(json.dumps(chain, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"request": rid, "image_id": ann["image_id"], "counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
