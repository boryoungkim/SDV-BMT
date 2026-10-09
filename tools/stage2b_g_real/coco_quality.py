#!/usr/bin/env python3
"""Task quality (COCO bbox AP) of a YOLOX run bundle, Stage 2b-G real-model baseline (analysis plane, stdlib).

Not a canonical catalog metric: the metric catalog has no ``quality.*`` entry and the workload declares no
quality evaluator (plan validation still requires ``quality_evaluator: null``).  This tool reads only:
  * the run bundle: run_spec.json (input spec: index SHA-256), input_items.jsonl (per-item annotation digest),
    events.jsonl (completed requests and their output references), outputs.bin (encoded task outputs,
    each re-verified against its recorded SHA-256);
  * the dataset index given on the command line, accepted only if its SHA-256 equals the run's declared index
    digest, and each item's annotation only if its canonical-JSON digest equals the bundle's annotation_digest.

Quality is reported as in design v03 H.6: ``conditional`` = over completed outputs only, with ``coverage`` =
completed / planned; ``planned_missing_as_empty`` = every planned image, a missing output counted as no
detections (a detection-specific missing-output policy, reported next to it, never instead of it).

    .venv/bin/python tools/stage2b_g_real/coco_quality.py RUN_DIR INDEX.jsonl OUT.json [--dump-inputs FILE.json]
        [--outputs-jsonl FILE]

``--outputs-jsonl`` replaces the bundle's outputs by task outputs written elsewhere (one JSON object per line,
same schema; e.g. the ONNX Runtime-derived outputs of validate_numerics.py) while the items and the verified
ground truth still come from the bundle -- a task-level comparison on identical images and annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from coco_eval import evaluate  # noqa: E402
from sdv_bmt.contracts.ids import digest_of  # noqa: E402


def load_inputs(run_dir: Path, index_path: Path) -> dict:
    run_spec = json.loads((run_dir / "run_spec.json").read_text())
    declared = run_spec["resolved_plan"]["input"]["source"]["index_sha256"]
    index_bytes = index_path.read_bytes()
    if hashlib.sha256(index_bytes).hexdigest() != declared:
        raise SystemExit(f"{index_path}: SHA-256 differs from the run's declared index digest {declared}")
    by_source = {}
    for line in index_bytes.decode("utf-8").splitlines():
        row = json.loads(line)
        by_source[row["source_item_id"]] = row["annotation"]
    items = {}
    for line in open(run_dir / "input_items.jsonl", encoding="utf-8"):
        it = json.loads(line)
        ann = by_source[it["source_item_id"]]
        if digest_of(ann) != it["annotation_digest"]:
            raise SystemExit(f"item {it['item_index']}: annotation digest differs from the bundle")
        items[it["item_index"]] = ann
    outputs = (run_dir / "outputs.bin").read_bytes()
    issued_item, completed = {}, {}
    for line in open(run_dir / "events.jsonl", encoding="utf-8"):
        e = json.loads(line)
        if e.get("kind") == "issued":
            issued_item[e["request_id"]] = e["item_indices"][0]
        elif e.get("kind") == "completed":
            ref = e["output_ref"]
            blob = outputs[ref["offset"]:ref["offset"] + ref["length"]]
            if hashlib.sha256(blob).hexdigest() != ref["sha256"]:
                raise SystemExit(f"request {e['request_id']}: output bytes differ from their recorded SHA-256")
            completed[issued_item[e["request_id"]]] = json.loads(blob)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return {"items": items, "completed": completed, "manifest": manifest,
            "categories": None, "index_sha256": declared}


def gt_and_dets(items: dict, completed: dict, image_items: list[int]) -> tuple[dict, dict]:
    gts, dets = {}, {}
    for i in image_items:
        ann = items[i]
        gts[ann["image_id"]] = [{"id": o["id"], "category_id": o["category_id"], "bbox": o["bbox"], "area": o["area"],
                                 "iscrowd": o["iscrowd"]} for o in ann["objects"]]
        out = completed.get(i)
        if out is not None:
            if out["image_id"] != ann["image_id"]:
                raise SystemExit(f"item {i}: output image_id {out['image_id']} differs from the annotation")
            dets[ann["image_id"]] = [{"category_id": d["category_id"],
                                      "bbox": [v / 1000.0 for v in d["bbox_mpx"]],
                                      "score": d["score_ppm"] / 1e6} for d in out["detections"]]
    return gts, dets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("index")
    ap.add_argument("out")
    ap.add_argument("--dump-inputs")
    ap.add_argument("--outputs-jsonl")
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    data = load_inputs(run_dir, Path(a.index))
    if a.outputs_jsonl:
        external = [json.loads(line) for line in open(a.outputs_jsonl, encoding="utf-8")]
        data["completed"] = {o["item_index"]: o for o in external}
    manifest_path = Path(a.index).parent / "dataset_manifest.json"
    cat_ids = [c["id"] for c in json.loads(manifest_path.read_text())["categories"]]
    items, completed = data["items"], data["completed"]
    planned = sorted(items)
    done = sorted(completed)
    gts_c, dets_c = gt_and_dets(items, completed, done)
    conditional = evaluate(gts_c, dets_c, cat_ids)
    gts_p, dets_p = gt_and_dets(items, completed, planned)
    planned_q = evaluate(gts_p, dets_p, cat_ids)
    m = data["manifest"]
    result = {
        "schema": "sdv-bmt-stage2b-g-real-coco-quality/1",
        "status": "analysis-plane result; not a canonical catalog metric",
        "run_id": m["run_id"], "experiment_id": m["experiment_id"], "run_dir": str(run_dir),
        "outputs_source": a.outputs_jsonl or "bundle outputs.bin",
        "evaluator": {"tool": "tools/stage2b_g_real/coco_quality.py + coco_eval.py (stdlib re-implementation of "
                              "pycocotools COCOeval bbox semantics)",
                      "coco_eval_sha256": hashlib.sha256((Path(__file__).parent / "coco_eval.py").read_bytes()).hexdigest(),
                      "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "params": "iouThrs linspace(.5,.95,10), recThrs linspace(0,1,101), maxDets 1/10/100, "
                                "areas all/small/medium/large, category-aware over all 80 COCO categories"},
        "inputs": {"index_sha256": data["index_sha256"], "planned_items": len(planned), "completed_items": len(done),
                   "output_detections": sum(len(o["detections"]) for o in completed.values())},
        "coverage": len(done) / len(planned),
        "conditional": conditional,
        "planned_missing_as_empty": planned_q,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    if a.dump_inputs:
        Path(a.dump_inputs).write_text(json.dumps({"categories": cat_ids, "gts": gts_c, "dets": dets_c}))
    print(json.dumps({"run_id": m["run_id"], "coverage": result["coverage"],
                      "conditional": conditional["stats"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
