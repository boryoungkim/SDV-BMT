#!/usr/bin/env python3
"""Stage 2b-G real-model baseline: numerical and task-level correctness of the GPU execution (offline tool).

Run with the system Python (onnxruntime + numpy are approved for offline tools only; the benchmark core does
not use them).  For the first N items of a COCO input selection:

  1. the canonical Workload preprocessing (sdv_bmt, stdlib) produces the input tensors;
  2. the native NVIDIA backend (tools/stage2b_g_real/build/raw_dump, same code as the runner, same DUT runtime
     options) executes them on the GPU and dumps the raw 8400x85 outputs;
  3. ONNX Runtime (CPU execution provider) runs the downloaded ONNX model on the same tensors;
  4. raw outputs are compared per channel group (box offsets, log sizes, objectness, class scores), and both
     are post-processed by the Python twin (YoloxCocoWorkload.postprocess) and compared at detection level;
  5. optionally (--bundle RUN_DIR) the bundle's native task-output bytes for the same items are compared with
     the Python twin applied to the dumped GPU raw outputs (native pipeline == Python reference on real data,
     and run-to-run determinism of the GPU raw outputs at the level of the encoded task output).

    python3 tools/stage2b_g_real/validate_numerics.py --plan configs/experiments/X.yaml --count 100 \
        --out runs/stage2b-g-real/analysis/numerics.json [--bundle RUN_DIR] [--ort-out FILE.jsonl]

``--ort-out`` additionally writes the ONNX Runtime-derived task outputs of all N items (for task-level AP).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402  (offline tool; approved)
import onnxruntime as ort  # noqa: E402

from sdv_bmt.cli.main import build_components  # noqa: E402
from sdv_bmt.contracts.ids import canonical_json_bytes  # noqa: E402
from sdv_bmt.runner.plan import load_plan  # noqa: E402

GROUPS = {"xy": slice(0, 2), "log_wh": slice(2, 4), "obj": slice(4, 5), "cls": slice(5, 85)}


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def compare_detections(gpu: list, ref: list) -> dict:
    """Greedy one-to-one matching in reference score order: same category, IoU >= 0.95."""
    used = set()
    matched, score_diff = 0, []
    for r in ref:
        best, best_j = 0.0, None
        for j, g in enumerate(gpu):
            if j in used or g["category_id"] != r["category_id"]:
                continue
            v = iou(g["bbox_mpx"], r["bbox_mpx"])
            if v > best:
                best, best_j = v, j
        if best_j is not None and best >= 0.95:
            used.add(best_j)
            matched += 1
            score_diff.append(abs(gpu[best_j]["score_ppm"] - r["score_ppm"]) / 1e6)
    return {"gpu": len(gpu), "ref": len(ref), "matched": matched, "max_score_diff": max(score_diff, default=0.0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bundle")
    ap.add_argument("--ort-out")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    rp = load_plan(a.plan)
    comps = build_components(rp)
    workload, dut, provider = comps.workload, comps.dut, comps.input_provider
    items = list(provider.items())[: a.count]
    provider.open()
    tmp = Path(tempfile.mkdtemp(prefix="sdv-bmt-numerics-", dir="/dev/shm"))
    try:
        staging = tmp / "staging.bin"
        rows, tensors = [], []
        t0 = time.time()
        with open(staging, "wb") as fh:
            for it in items:
                data = workload.preprocess(provider.payload(it.item_index), it)
                rows.append({"item_index": it.item_index, "offset": fh.tell(), "nbytes": len(data),
                             "sha256": hashlib.sha256(data).hexdigest()})
                fh.write(data)
                tensors.append(data)
        t_pre = time.time() - t0
        spec = {"device_uuid": rp.dut.device, "backend": {"name": dut.ADAPTER, "options": dict(rp.dut.runtime_options)},
                **dut.backend_spec(rp.workload)}
        (tmp / "spec.json").write_bytes(canonical_json_bytes(spec))
        (tmp / "items.json").write_text(json.dumps(rows))
        raw_path = tmp / "raw.bin"
        proc = subprocess.run([str(REPO / "tools/stage2b_g_real/build/raw_dump"), str(tmp / "spec.json"),
                               str(staging), str(tmp / "items.json"), str(raw_path)],
                              capture_output=True, text=True)
        dump = json.loads(proc.stdout.strip().splitlines()[-1])
        if not dump.get("ok"):
            raise SystemExit(f"raw_dump failed: {dump} {proc.stderr[-2000:]}")
        gpu_raw = np.fromfile(raw_path, dtype="<f4").reshape(len(items), 8400, 85)

        so = ort.SessionOptions()
        so.intra_op_num_threads = a.threads
        so.inter_op_num_threads = 1
        model = REPO / "models/yolox-s/yolox_s.onnx"
        sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
        diffs = {g: [] for g in GROUPS}
        rel = {g: [] for g in GROUPS}
        det_cmp, native_vs_python, ort_rows = [], [], []
        bundle_blobs = {}
        if a.bundle:
            outputs = (Path(a.bundle) / "outputs.bin").read_bytes()
            for line in open(Path(a.bundle) / "events.jsonl"):
                e = json.loads(line)
                if e.get("kind") == "completed":
                    ref = e["output_ref"]
                    req_item = e["request_id"]  # request i serves item i in these plans
                    bundle_blobs[req_item] = outputs[ref["offset"]:ref["offset"] + ref["length"]]
        t0 = time.time()
        for k, it in enumerate(items):
            x = np.frombuffer(tensors[k], dtype="<f4").reshape(1, 3, 640, 640)
            ref_raw = sess.run(None, {"images": x})[0][0]
            g_raw = gpu_raw[k]
            for name, sl in GROUPS.items():
                d = np.abs(g_raw[:, sl] - ref_raw[:, sl])
                diffs[name].append(float(d.max()))
                rel[name].append(float((d / np.maximum(np.abs(ref_raw[:, sl]), 1e-3)).max()))
            g_out = workload.postprocess(g_raw.astype("<f4").tobytes(), it)
            r_out = workload.postprocess(ref_raw.astype("<f4").tobytes(), it)
            det_cmp.append({"item_index": it.item_index, **compare_detections(g_out["detections"], r_out["detections"])})
            ort_rows.append(r_out)
            if it.item_index in bundle_blobs:
                native_vs_python.append(bundle_blobs[it.item_index] == workload.encode(g_out))
        t_ort = time.time() - t0
        if a.ort_out:
            with open(a.ort_out, "w", encoding="utf-8") as fh:
                for r in ort_rows:
                    fh.write(canonical_json_bytes(r).decode("utf-8") + "\n")

        def summary(values):
            v = sorted(values)
            return {"n": len(v), "max": v[-1], "p50": v[len(v) // 2], "p99": v[min(len(v) - 1, math.ceil(0.99 * len(v)) - 1)]}

        total_ref = sum(d["ref"] for d in det_cmp)
        total_gpu = sum(d["gpu"] for d in det_cmp)
        result = {
            "schema": "sdv-bmt-stage2b-g-real-numerics/1",
            "plan": str(Path(a.plan)),
            "items": len(items),
            "onnxruntime": ort.__version__, "numpy": np.__version__, "ort_providers": sess.get_providers(),
            "ort_threads": a.threads, "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "gpu_dump": {k: dump[k] for k in ("items", "cudnn_algorithm_table_sha256")},
            "preprocess_seconds": round(t_pre, 1), "ort_seconds": round(t_ort, 1),
            "raw_abs_diff_max_per_item": {g: summary(v) for g, v in diffs.items()},
            "raw_rel_diff_max_per_item": {g: summary(v) for g, v in rel.items()},
            "detections": {"gpu_total": total_gpu, "ort_total": total_ref,
                           "matched_total": sum(d["matched"] for d in det_cmp),
                           "items_identical_count": sum(1 for d in det_cmp if d["gpu"] == d["ref"] == d["matched"]),
                           "max_matched_score_diff": max(d["max_score_diff"] for d in det_cmp)},
            "per_item": det_cmp,
            "bundle": a.bundle,
            "native_blob_equals_python_twin_on_gpu_raw": (
                {"compared": len(native_vs_python), "equal": sum(native_vs_python)} if a.bundle else None),
            "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print(json.dumps({k: result[k] for k in ("items", "raw_abs_diff_max_per_item", "detections",
                                                  "native_blob_equals_python_twin_on_gpu_raw")}, sort_keys=True))
    finally:
        provider.release()
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
