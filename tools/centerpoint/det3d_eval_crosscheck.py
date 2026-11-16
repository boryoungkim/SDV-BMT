"""Cross-checks of the SDV-BMT nuScenes 3D detection evaluator (tools only; cross-check env with nuscenes-devkit).

Run with ``<venv-mmseg>`` (mmdet3d 1.4.0, nuscenes-devkit 1.2.0).  On one SDV-BMT CenterPoint input (index with the
frozen ground truth and poses) and a devkit dataroot that defines the same scenes as a custom split:

  A. evaluator vs devkit on the official boxes: the official results file (global frame) evaluated by
     ``sdv_bmt.metrics.quality_det3d.evaluate`` and by the devkit ``DetectionEval``; criterion |d mAP|, |d NDS| <= 1e-4;
  B. conversion: raw head outputs -> ``decode_heads`` -> ``predictions_to_global`` vs the official mmdet3d conversion
     (``output_to_nusc_box`` + ``lidar_nusc_box_to_global`` + the attribute rule) of the same detections;
  C. per raw-output set (e.g. GPU, ORT): our evaluator and the devkit on the converted boxes;
  D. criteria across sets: |mAP/NDS(gpu) - mAP/NDS(ort)| <= 1e-3; |mAP/NDS(gpu) - official| <= 0.005.

    <venv-mmseg>/bin/python tools/centerpoint/det3d_eval_crosscheck.py --index INDEX --devkit-root ROOT \
        --version v1.0-mini --split mini_val4 --official-results RESULTS.json --official-metrics METRICS.json \
        --raw gpu=DIR --raw ort=DIR --out OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyquaternion
import torch
from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric, lidar_nusc_box_to_global, output_to_nusc_box
from mmdet3d.structures import LiDARInstance3DBoxes
from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from sdv_bmt.metrics import quality_det3d as Q  # noqa: E402
from sdv_bmt.workload.centerpoint import decode_heads  # noqa: E402


def load_index(path: Path):
    rows = [json.loads(l) for l in open(path)]
    samples = [r["source_item_id"] for r in rows]
    gt, racks, ego, poses = {}, {}, {}, {}
    for r in rows:
        boxes, rk, l2e, e2g = Q.ground_truth(r["annotation"])
        s = r["source_item_id"]
        gt[s], racks[s], ego[s], poses[s] = boxes, rk, tuple(e2g["translation"]), (l2e, e2g)
    return samples, gt, racks, ego, poses


def devkit_eval(root: Path, version: str, split: str, results: dict) -> dict:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "results.json"
        p.write_text(json.dumps(results))
        nusc = NuScenes(version=version, dataroot=str(root), verbose=False)
        ev = DetectionEval(nusc, config_factory("detection_cvpr_2019"), str(p), eval_set=split, output_dir=d,
                           verbose=False)
        metrics, _ = ev.evaluate()
        return metrics.serialize()


def to_results(pred: dict[str, list[dict]]) -> dict:
    return {"meta": {"use_camera": False, "use_lidar": True, "use_radar": False, "use_map": False,
                     "use_external": False},
            "results": {s: [{"sample_token": s, "translation": list(b["translation"]), "size": list(b["size"]),
                             "rotation": list(b["rotation"]), "velocity": list(b["velocity"]),
                             "detection_name": b["detection_name"], "detection_score": b["detection_score"],
                             "attribute_name": b["attribute_name"]} for b in boxes] for s, boxes in pred.items()}}


def summary(m: dict) -> dict:
    return {"mAP": m["mean_ap"], "NDS": m["nd_score"], "tp_errors": m["tp_errors"], "mean_dist_aps": m["mean_dist_aps"]}


def official_conversion(dets, l2e: dict, e2g: dict) -> list:
    """mmdet3d NuScenesMetric conversion of our decoded detections (LiDAR frame, bottom centre)."""
    if not dets:
        return []
    boxes = LiDARInstance3DBoxes(torch.tensor([d[0] for d in dets], dtype=torch.float32), box_dim=9)
    det = {"bboxes_3d": boxes, "scores_3d": torch.tensor([d[1] for d in dets], dtype=torch.float32),
           "labels_3d": torch.tensor([d[2] for d in dets], dtype=torch.int64)}
    nus, _ = output_to_nusc_box(det)
    mat = lambda p: np.vstack([np.hstack([pyquaternion.Quaternion(p["rotation"]).rotation_matrix,  # noqa: E731
                                          np.array(p["translation"]).reshape(3, 1)]), [0, 0, 0, 1]])
    info = {"lidar_points": {"lidar2ego": mat(l2e).tolist()}, "ego2global": mat(e2g).tolist()}
    return lidar_nusc_box_to_global(info, nus, list(Q.CLASS_NAMES), config_factory("detection_cvpr_2019"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--devkit-root", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--official-results", required=True)
    ap.add_argument("--official-metrics", required=True)
    ap.add_argument("--raw", action="append", default=[], help="name=DIR of <token>.f32 raw head outputs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    samples, gt, racks, ego, poses = load_index(Path(a.index))
    root = Path(a.devkit_root)
    report: dict = {"samples": len(samples)}

    # A: official boxes
    off = json.loads(Path(a.official_results).read_text())["results"]
    pred_off = {s: [{"translation": tuple(b["translation"]), "size": tuple(b["size"]), "rotation": tuple(b["rotation"]),
                     "velocity": tuple(b["velocity"]), "detection_name": b["detection_name"],
                     "detection_score": b["detection_score"], "attribute_name": b["attribute_name"], "num_pts": -1}
                    for b in off[s]] for s in samples}
    ours_a = Q.evaluate(samples, gt, pred_off, racks, ego)
    dev_a = json.loads(Path(a.official_metrics).read_text())
    report["A_official_boxes"] = {"ours": summary(ours_a), "devkit": summary(dev_a),
                                  "d_mAP": abs(ours_a["mean_ap"] - dev_a["mean_ap"]),
                                  "d_NDS": abs(ours_a["nd_score"] - dev_a["nd_score"]),
                                  "max_d_class_ap": max(abs(ours_a["mean_dist_aps"][c] - dev_a["mean_dist_aps"][c])
                                                        for c in Q.CLASS_NAMES)}
    report["A_official_boxes"]["pass"] = (report["A_official_boxes"]["d_mAP"] <= 1e-4 and
                                          report["A_official_boxes"]["d_NDS"] <= 1e-4)
    print("A", report["A_official_boxes"]["d_mAP"], report["A_official_boxes"]["d_NDS"], flush=True)

    # B, C: raw outputs
    sets = {}
    for spec in a.raw:
        name, d = spec.split("=", 1)
        pred, conv_diff = {}, {"translation": 0.0, "size": 0.0, "velocity": 0.0, "yaw": 0.0, "count_mismatch": 0,
                               "attr_mismatch": 0}
        for s in samples:
            dets = decode_heads((Path(d) / f"{s}.f32").read_bytes())
            l2e, e2g = poses[s]
            mine = Q.predictions_to_global(dets, l2e, e2g)
            pred[s] = mine
            if name == "gpu":
                offc = official_conversion(dets, l2e, e2g)
                if len(offc) != len(mine):
                    conv_diff["count_mismatch"] += 1
                    continue
                for b, o in zip(mine, offc):
                    conv_diff["translation"] = max(conv_diff["translation"], float(np.abs(np.array(b["translation"]) - o.center).max()))
                    conv_diff["size"] = max(conv_diff["size"], float(np.abs(np.array(b["size"]) - o.wlh).max()))
                    conv_diff["velocity"] = max(conv_diff["velocity"], float(np.abs(np.array(b["velocity"]) - o.velocity[:2]).max()))
                    yo = Q.q_yaw(tuple(o.orientation.elements))
                    conv_diff["yaw"] = max(conv_diff["yaw"], abs(Q.angle_diff(Q.q_yaw(b["rotation"]), yo, 2 * np.pi)))
        ours = Q.evaluate(samples, gt, pred, racks, ego)
        dev = devkit_eval(root, a.version, a.split, to_results(pred))
        sets[name] = {"ours": summary(ours), "devkit": summary(dev), "d_mAP_vs_devkit": abs(ours["mean_ap"] - dev["mean_ap"]),
                      "d_NDS_vs_devkit": abs(ours["nd_score"] - dev["nd_score"]),
                      "boxes": ours["pred_boxes"], "gt_boxes": ours["gt_boxes"]}
        if name == "gpu":
            sets[name]["conversion_vs_official"] = conv_diff
        print(name, summary(ours)["mAP"], summary(ours)["NDS"], "devkit", dev["mean_ap"], dev["nd_score"], flush=True)
    report["C_sets"] = sets
    if "gpu" in sets and "ort" in sets:
        g, o = sets["gpu"]["ours"], sets["ort"]["ours"]
        report["D_gpu_vs_ort"] = {"d_mAP": abs(g["mAP"] - o["mAP"]), "d_NDS": abs(g["NDS"] - o["NDS"])}
        report["D_gpu_vs_ort"]["pass"] = report["D_gpu_vs_ort"]["d_mAP"] <= 1e-3 and report["D_gpu_vs_ort"]["d_NDS"] <= 1e-3
    if "gpu" in sets:
        g = sets["gpu"]["ours"]
        report["D_gpu_vs_official"] = {"d_mAP": abs(g["mAP"] - dev_a["mean_ap"]), "d_NDS": abs(g["NDS"] - dev_a["nd_score"])}
        report["D_gpu_vs_official"]["pass"] = (report["D_gpu_vs_official"]["d_mAP"] <= 0.005 and
                                               report["D_gpu_vs_official"]["d_NDS"] <= 0.005)
    report["evaluator_sha256"] = hashlib.sha256(Path(Q.__file__).read_bytes()).hexdigest()
    report["tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path(a.out).write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: v.get("pass") for k, v in report.items() if isinstance(v, dict) and "pass" in v}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
