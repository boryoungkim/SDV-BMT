#!/usr/bin/env python3
"""Stage 2b-G real-model baseline: the raw evidence chain of one request, from the source JPEG to the task
quality contribution (analysis plane; stdlib + the sdv_bmt package; no third-party module).

    .venv/bin/python tools/stage2b_g_real/evidence_chain.py RUN_DIR REQUEST_ID INDEX.jsonl COCO_IMAGES_DIR OUT.json

Every link is re-derived from recorded data and re-verified where the bundle holds a digest:
  source JPEG (COCO file, SHA-256 from the index metadata) -> decoded PPM payload (payload_sha256, the bundle's
  input item) -> canonical preprocessing (recomputed here; the bundle records the preprocessing identity but no
  per-item prepared-input digest) -> request events (t_sched .. t_output_ready, run clock) -> device observation
  (CUDA-event segments, device clock; H2D bytes; probe words) -> encoded task output (offset, length, SHA-256 in
  outputs.bin) -> output reference (golden) row -> detections vs the image's ground truth (IoU >= 0.5, same
  category, greedy by score, crowd objects not counted) -> the latencies the evaluator derives for it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from sdv_bmt.cli.main import WORKLOADS  # noqa: E402
from sdv_bmt.contracts.records import InputItem  # noqa: E402
from sdv_bmt.contracts.specs import WorkloadSpec, parse_strict  # noqa: E402


def iou(a, b):
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def main() -> int:
    run_dir, rid, index_path, images_dir, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
    run_spec = json.loads((run_dir / "run_spec.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    events = [json.loads(line) for line in open(run_dir / "events.jsonl", encoding="utf-8")]
    req = [e for e in events if e.get("request_id") == rid]
    issued = next(e for e in req if e["kind"] == "issued")
    item_index = issued["item_indices"][0]
    item_row = next(json.loads(line) for line in open(run_dir / "input_items.jsonl", encoding="utf-8")
                    if json.loads(line)["item_index"] == item_index)
    index_rows = {json.loads(line)["source_item_id"]: json.loads(line) for line in open(index_path, encoding="utf-8")}
    row = index_rows[item_row["source_item_id"]]
    jpeg = (images_dir / row["metadata"]["jpeg_file"]).read_bytes()
    payload = (index_path.parent / row["payload_file"]).read_bytes()
    spec = parse_strict(WorkloadSpec, run_spec["resolved_plan"]["workload"])
    workload = WORKLOADS[spec.implementation](spec)
    item = InputItem(**{k: item_row[k] for k in ("item_index", "source_item_id", "payload_ref", "payload_sha256",
                                                  "annotation_ref", "annotation_digest", "source_timing",
                                                  "scenario_coordinates", "source_metadata")})
    prepared = workload.preprocess(payload, item)
    t = {}
    for e in req:
        if e["kind"] == "issued":
            t.update(t_sched=e["t_sched"], t_issue=e["t_issue"])
        elif e["kind"] in ("submit_begin", "submit_end"):
            t[f"t_{e['kind']}"] = e["t_run_ns"]
        elif e["kind"] == "completed":
            t.update(t_complete=e["t_complete"], t_output_ready=e["t_output_ready"])
            ref = e["output_ref"]
    device = next((e for e in req if e["kind"] == "device_observation"), {})
    blob = (run_dir / "outputs.bin").read_bytes()[ref["offset"]:ref["offset"] + ref["length"]]
    output = json.loads(blob)
    golden = run_spec.get("evaluation", {}).get("output_reference") or {}
    golden_row = next((g for g in golden.get("entries", []) if g["item_index"] == item_index), None)
    gts = [o for o in row["annotation"]["objects"]]
    matched, used, tp = [], set(), 0
    for d in output["detections"]:
        box = [v / 1000.0 for v in d["bbox_mpx"]]
        best, best_g = 0.0, None
        for g_i, g in enumerate(gts):
            if g_i in used or g["iscrowd"] or g["category_id"] != d["category_id"]:
                continue
            v = iou(box, g["bbox"])
            if v > best:
                best, best_g = v, g_i
        if best >= 0.5:
            used.add(best_g)
            tp += 1
        matched.append({"category_id": d["category_id"], "score": d["score_ppm"] / 1e6, "bbox_px": box,
                        "best_iou_same_category": round(best, 4), "matched_object_id": gts[best_g]["id"] if best >= 0.5 else None})
    deadline = run_spec["resolved_plan"]["evaluation"]["deadline"]
    chain = {
        "run_id": manifest["run_id"], "experiment_id": manifest["experiment_id"], "request_id": rid,
        "1_source": {"coco_image_id": row["annotation"]["image_id"], "jpeg_file": row["metadata"]["jpeg_file"],
                     "jpeg_sha256_index": row["metadata"]["jpeg_sha256"], "jpeg_sha256_now": hashlib.sha256(jpeg).hexdigest(),
                     "decoder": row["metadata"]["decoder"], "size": [row["metadata"]["width"], row["metadata"]["height"]]},
        "2_payload": {"payload_ref": item_row["payload_ref"], "payload_sha256_bundle": item_row["payload_sha256"],
                      "payload_sha256_now": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                      "annotation_digest_bundle": item_row["annotation_digest"]},
        "3_preprocess": {"semantics": run_spec["resolved_plan"]["workload"]["preprocess"],
                         "prepared_sha256_recomputed": hashlib.sha256(prepared).hexdigest(), "bytes": len(prepared),
                         "note": "the bundle records the preprocessing identity and the native re-verification "
                                 "outcome, not a per-item prepared-input digest"},
        "4_request_events_run_clock_ns": t,
        "5_device_observation": {k: device.get(k) for k in ("timing", "timing_method", "observations")},
        "6_output": {"output_ref": ref, "sha256_now": hashlib.sha256(blob).hexdigest(), "detections": len(output["detections"]),
                     "golden_row": golden_row},
        "7_quality_contribution": {"gt_objects": len(gts), "gt_crowd": sum(g["iscrowd"] for g in gts),
                                   "detections_matched_iou50": tp, "detections": matched[:15]},
        "8_latencies_ns": {"issue_lag": t["t_issue"] - t["t_sched"], "submit": t["t_submit_end"] - t["t_submit_begin"],
                           "dut_service": t["t_complete"] - t["t_submit_begin"],
                           "request_completion": t["t_complete"] - t["t_sched"],
                           "output_ready": t["t_output_ready"] - t["t_sched"],
                           "output_step": t["t_output_ready"] - t["t_complete"],
                           "deadline_ns": deadline["period_ns"] if deadline else None,
                           "deadline_met": (t["t_complete"] <= t["t_sched"] + deadline["period_ns"]) if deadline else None},
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(chain, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: chain[k] for k in ("1_source", "2_payload", "8_latencies_ns")}, sort_keys=True)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
