#!/usr/bin/env python3
"""L1 numeric check of a compiled YOLOX .mxq on the Mobilint ARIES NPU against ONNX Runtime FP32 (offline tool).

Modelled on tools/stage2b_g_real/validate_numerics.py (the GPU check). For the first N items of the COCO subset:

  1. the canonical Workload preprocessing (sdv_bmt YoloxCocoWorkload.preprocess) produces the CHW float32 BGR
     tensor, exactly what a run stages; the NPU gets the same values transposed to HWC (the model input layout);
  2. ONNX Runtime (CPUExecutionProvider, FP32) runs the original ONNX file (SHA-256 checked against the
     workload's artifact_sha256) on the CHW tensor -> reference anchors x 85;
  3. qbruntime runs the .mxq on one pinned NPU core (sync infer) on the HWC tensor -> NPU anchors x 85. A model
     cut at the nine head convolutions (tools/mobilint/cut_head.py) returns nine HWC tensors in its own order;
     they are matched by (grid, channels) and the tail (sigmoid, concat) is applied on the host with
     cut_head.host_tail, as the C++ backend does in collect(). A full-graph model returns anchors x 85 directly;
  4. raw outputs are compared per channel group (xy, log_wh, obj, cls); both are decoded with the same Python
     postprocess (YoloxCocoWorkload.postprocess) and compared at detection level; COCO bbox AP is computed for
     both with the repo evaluator (tools/stage2b_g_real/coco_eval.py) on the same images and annotations.

The criterion (G13, decision pending) is applied by the reader, not here: the tool only reports the numbers.

Environment: a Python with qbruntime, numpy, onnxruntime and PyYAML (sdv_bmt is imported from src/; its only
runtime dependency is PyYAML). Do not run while another process uses the device.

    python3 tools/mobilint/validate_numerics.py --workload configs/workloads/yolox-s-coco-v1.yaml \\
        --index datasets/coco-val2017-first1000/index.jsonl --onnx ~/sdv_assets/yolox_onnx/yolox_s.onnx \\
        --mxq ~/sdv_assets/compile/yolox_s.mxq --dev 2 --count 1000 \\
        --out ~/sdv_assets/l1/yolox-s/numerics.json [--tensor-cache ~/sdv_assets/tensors/640]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "stage2b_g_real"))

import numpy as np  # noqa: E402  (offline tool)
import onnxruntime as ort  # noqa: E402
import qbruntime as qb  # noqa: E402
import yaml  # noqa: E402

from coco_eval import evaluate  # noqa: E402
from cut_head import BRANCHES, host_tail  # noqa: E402  (tools/mobilint, the script's directory)
from sdv_bmt.contracts.records import InputItem  # noqa: E402
from sdv_bmt.contracts.specs import WorkloadSpec, parse_strict  # noqa: E402
from sdv_bmt.workload.yolox import YoloxCocoWorkload  # noqa: E402

SCHEMA = "sdv-bmt-mobilint-numerics/1"
GROUPS = {"xy": slice(0, 2), "log_wh": slice(2, 4), "obj": slice(4, 5), "cls": slice(5, 85)}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def compare_detections(npu: list, ref: list, min_iou: float) -> dict:
    """Greedy one-to-one matching in reference score order: same category, IoU >= min_iou."""
    used = set()
    matched, score_diff = 0, []
    for r in ref:
        best, best_j = 0.0, None
        for j, g in enumerate(npu):
            if j in used or g["category_id"] != r["category_id"]:
                continue
            v = iou(g["bbox_mpx"], r["bbox_mpx"])
            if v > best:
                best, best_j = v, j
        if best_j is not None and best >= min_iou:
            used.add(best_j)
            matched += 1
            score_diff.append(abs(npu[best_j]["score_ppm"] - r["score_ppm"]) / 1e6)
    return {"npu": len(npu), "ref": len(ref), "matched": matched, "max_score_diff": max(score_diff, default=0.0)}


def summary(values):
    v = sorted(values)
    return {"n": len(v), "max": v[-1], "p50": v[len(v) // 2], "p99": v[min(len(v) - 1, math.ceil(0.99 * len(v)) - 1)]}


def to_eval(out: dict) -> list[dict]:
    return [{"category_id": d["category_id"], "bbox": [v / 1000.0 for v in d["bbox_mpx"]],
             "score": d["score_ppm"] / 1e6} for d in out["detections"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, help="workload YAML (input size, preprocessing, artifact_sha256)")
    ap.add_argument("--index", required=True, help="index.jsonl of the COCO subset (payloads next to it)")
    ap.add_argument("--onnx", required=True, help="original ONNX file (checked against artifact_sha256)")
    ap.add_argument("--mxq", required=True, help="compiled model")
    ap.add_argument("--dev", type=int, required=True, help="ARIES device number (/dev/ariesN)")
    ap.add_argument("--cluster", type=int, default=0)
    ap.add_argument("--core", type=int, default=0)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=8, help="ONNX Runtime intra-op threads")
    ap.add_argument("--tensor-cache", help="directory for preprocessed CHW tensors (reused across runs/models)")
    ap.add_argument("--out", required=True, help="numerics.json; npu_outputs.jsonl and ort_outputs.jsonl go next to it")
    a = ap.parse_args()

    wl_path = Path(a.workload)
    spec = parse_strict(WorkloadSpec, yaml.safe_load(wl_path.read_text(encoding="utf-8")), str(wl_path))
    workload = YoloxCocoWorkload(spec)
    size, anchors = spec.options["input_size"], spec.options["num_anchors"]

    onnx_path, mxq_path = Path(a.onnx).expanduser(), Path(a.mxq).expanduser()
    onnx_sha = file_sha256(onnx_path)
    if onnx_sha != spec.artifact_sha256:
        raise SystemExit(f"{onnx_path}: SHA-256 {onnx_sha} differs from the workload's artifact_sha256")
    mxq_sha = file_sha256(mxq_path)

    index_path = Path(a.index)
    index_bytes = index_path.read_bytes()
    rows = [json.loads(line) for line in index_bytes.decode("utf-8").splitlines() if line][: a.count]
    manifest = json.loads((index_path.parent / "dataset_manifest.json").read_text(encoding="utf-8"))
    cat_ids = [c["id"] for c in manifest["categories"]]
    cache = Path(a.tensor_cache).expanduser() if a.tensor_cache else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    preprocess_id = f"{spec.preprocess.id}/{spec.preprocess.version}"

    # ONNX Runtime reference session
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # NPU: one pinned core, default (vendor) input/output conversion, sync infer
    mc = qb.ModelConfig()
    mc.set_single_core_mode([qb.CoreId(getattr(qb.Cluster, f"Cluster{a.cluster}"), getattr(qb.Core, f"Core{a.core}"))])
    model = qb.Model(str(mxq_path), mc)
    acc = qb.Accelerator(a.dev)
    model.launch(acc)
    in_shape, out_shape = model.get_model_input_shape(), model.get_model_output_shape()
    target_cores = [str(c) for c in model.get_target_cores()]
    core_mode = str(model.get_core_mode())
    if len(target_cores) != 1:
        raise SystemExit(f"expected one pinned core, got {target_cores}")
    if tuple(in_shape[0]) != (size, size, 3):
        raise SystemExit(f"model input shape {in_shape} is not ({size}, {size}, 3)")
    grids = [size // s for s in (8, 16, 32)]
    if len(out_shape) == 1:
        graph = "full"
        if int(np.prod(out_shape[0])) != anchors * 85:
            raise SystemExit(f"model output shape {out_shape} does not hold {anchors} x 85 values")
    elif len(out_shape) == 9:
        graph = "head9"
        want = sorted((g, ch) for g in grids for _, ch in BRANCHES)
        got = sorted((int(s[-3]), int(s[-1])) for s in out_shape)
        if got != want or sum(g * g for g in grids) != anchors:
            raise SystemExit(f"model outputs {out_shape} are not the nine YOLOX head tensors (HWC) {want}")
    else:
        raise SystemExit(f"model has {len(out_shape)} outputs; expected 1 (full graph) or 9 (cut at the head)")

    def npu_infer(hwc: np.ndarray) -> np.ndarray:
        outs = model.infer([hwc])
        if graph == "full":
            return np.asarray(outs[0], dtype=np.float32).reshape(anchors, 85)
        by_shape = {}
        for o in outs:
            o = np.asarray(o, dtype=np.float32)
            g, ch = o.shape[-3], o.shape[-1]
            by_shape[(g, ch)] = o.reshape(g, g, ch).transpose(2, 0, 1)   # HWC -> CHW
        return host_tail([tuple(by_shape[(g, ch)] for _, ch in BRANCHES) for g in grids])

    diffs = {g: [] for g in GROUPS}
    det_cmp, npu_rows, ort_rows = [], [], []
    gts, dets_npu, dets_ort = {}, {}, {}
    t_pre = t_ort = t_npu = 0.0
    cache_hits = 0
    out_path = Path(a.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for n, row in enumerate(rows):
            payload = (index_path.parent / row["payload_file"]).read_bytes()
            if sha256(payload) != row["payload_sha256"]:
                raise SystemExit(f"{row['payload_file']}: payload SHA-256 differs from the index")
            item = InputItem(item_index=n, source_item_id=row["source_item_id"], payload_ref=row["payload_file"],
                             payload_sha256=row["payload_sha256"], annotation_ref=None, annotation_digest=None,
                             source_timing=None, scenario_coordinates=None,
                             source_metadata={"row_metadata": row["metadata"]})
            t0 = time.perf_counter()
            chw = None
            if cache:
                cpath = cache / f"{row['source_item_id']}.{row['payload_sha256'][:16]}.npy"
                if cpath.exists():
                    chw = np.load(cpath)
                    cache_hits += 1
            if chw is None:
                chw = np.frombuffer(workload.preprocess(payload, item), dtype="<f4").reshape(3, size, size)
                if cache:
                    np.save(cpath, chw)
            hwc = np.ascontiguousarray(chw.transpose(1, 2, 0))
            t_pre += time.perf_counter() - t0

            t0 = time.perf_counter()
            ref_raw = sess.run(None, {input_name: chw[np.newaxis]})[0].reshape(anchors, 85)
            t_ort += time.perf_counter() - t0
            t0 = time.perf_counter()
            npu_raw = npu_infer(hwc)
            t_npu += time.perf_counter() - t0

            for name, sl in GROUPS.items():
                diffs[name].append(float(np.abs(npu_raw[:, sl] - ref_raw[:, sl]).max()))
            n_out = workload.postprocess(npu_raw.astype("<f4").tobytes(), item)
            r_out = workload.postprocess(ref_raw.astype("<f4").tobytes(), item)
            det_cmp.append({"item_index": n, "image_id": n_out["image_id"],
                            **compare_detections(n_out["detections"], r_out["detections"], 0.5)})
            npu_rows.append(n_out)
            ort_rows.append(r_out)
            ann = row["annotation"]
            gts[ann["image_id"]] = [{"id": o["id"], "category_id": o["category_id"], "bbox": o["bbox"],
                                     "area": o["area"], "iscrowd": o["iscrowd"]} for o in ann["objects"]]
            dets_npu[ann["image_id"]] = to_eval(n_out)
            dets_ort[ann["image_id"]] = to_eval(r_out)
            if (n + 1) % 50 == 0 or n + 1 == len(rows):
                print(f"[{n + 1}/{len(rows)}] pre {t_pre:.0f}s ort {t_ort:.0f}s npu {t_npu:.1f}s", flush=True)
    finally:
        model.dispose()

    for name, data in (("npu_outputs.jsonl", npu_rows), ("ort_outputs.jsonl", ort_rows)):
        (out_path.parent / name).write_text("".join(canonical(r) + "\n" for r in data), encoding="utf-8")
    ap_npu = evaluate(gts, dets_npu, cat_ids)["stats"]
    ap_ort = evaluate(gts, dets_ort, cat_ids)["stats"]
    total_ref = sum(d["ref"] for d in det_cmp)
    result = {
        "schema": SCHEMA,
        "workload": {"file": str(wl_path), "workload_id": spec.workload_id, "input_size": size,
                     "preprocess": preprocess_id},
        "items": len(rows),
        "index": {"file": str(index_path), "sha256": sha256(index_bytes)},
        "onnx": {"file": str(onnx_path), "sha256": onnx_sha},
        "mxq": {"file": str(mxq_path), "sha256": mxq_sha},
        "npu": {"device": f"/dev/aries{a.dev}", "core_mode": core_mode,
                "target_cores": target_cores, "input_shape": [list(s) for s in in_shape],
                "output_shape": [list(s) for s in out_shape], "call": "Model.infer([hwc float32]) (sync)",
                "graph": graph,
                "host_tail": "cut_head.host_tail (sigmoid obj/cls, concat 85, levels 8/16/32)" if graph == "head9"
                             else None,
                "cut_head_sha256": file_sha256(Path(__file__).with_name("cut_head.py")) if graph == "head9" else None,
                "qbruntime": getattr(qb, "__version__", "unknown")},
        "onnxruntime": ort.__version__, "numpy": np.__version__, "ort_threads": a.threads,
        "seconds": {"preprocess": round(t_pre, 1), "ort": round(t_ort, 1), "npu": round(t_npu, 1)},
        "tensor_cache": {"dir": str(cache) if cache else None, "hits": cache_hits},
        "raw_abs_diff_max_per_item": {g: summary(v) for g, v in diffs.items()},
        "detections": {"npu_total": sum(d["npu"] for d in det_cmp), "ort_total": total_ref,
                       "matched_total_iou50": sum(d["matched"] for d in det_cmp),
                       "matched_fraction_of_ort": (sum(d["matched"] for d in det_cmp) / total_ref) if total_ref else None,
                       "max_matched_score_diff": max(d["max_score_diff"] for d in det_cmp)},
        "coco_bbox": {"npu": ap_npu, "ort_fp32": ap_ort,
                      "ap_drop_vs_ort": ap_ort["AP"] - ap_npu["AP"],
                      "evaluator": "tools/stage2b_g_real/coco_eval.py (pycocotools COCOeval bbox semantics)",
                      "coco_eval_sha256": file_sha256(REPO / "tools/stage2b_g_real/coco_eval.py")},
        "per_item": det_cmp,
        "tool": {"path": "tools/mobilint/validate_numerics.py", "sha256": file_sha256(Path(__file__))},
    }
    out_path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(canonical({"items": len(rows), "AP_npu": ap_npu["AP"], "AP_ort": ap_ort["AP"],
                     "AP_drop": result["coco_bbox"]["ap_drop_vs_ort"],
                     "raw_abs_diff_p50": {g: result["raw_abs_diff_max_per_item"][g]["p50"] for g in GROUPS},
                     "matched_fraction_of_ort": result["detections"]["matched_fraction_of_ort"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
