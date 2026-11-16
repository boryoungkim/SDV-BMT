"""Official mmdetection3d CenterPoint-Pillars reference on a nuScenes scene subset (tools only, cross-check env).

Run with the cross-check environment ``<venv-mmseg>`` (mmdet3d 1.4.0, mmcv 2.1.0 CPU, nuscenes-devkit 1.2.0), never with
the benchmark core.  For the given scenes (in the given order; samples by timestamp within a scene):

  1. the official test pipeline of ``centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py``
     (LoadPointsFromFile + LoadPointsFromMultiSweeps with 9 sweeps) on the official v2 info files;
  2. the official model (``init_model``, CPU) with the checkpoint; forward hooks record the voxelizer output (pillars,
     point counts, coordinates) and the raw CenterHead outputs;
  3. the official result formatting (``NuScenesMetric._format_lidar_bbox``: LiDAR -> global, class-range pre-filter,
     attributes) and the official nuScenes devkit evaluation (``DetectionEval``, ``detection_cvpr_2019``) on a custom
     split made of exactly these scenes.

Per sample it writes SHA-256 digests of the merged points, the voxelizer tensors and the head outputs (all samples)
and the full tensors for the first ``--dump-first`` samples.  Nothing here is used by the benchmark core.

    <venv-mmseg>/bin/python tools/centerpoint/official_run.py --data-root NUSC_ROOT --version v1.0-mini \
        --infos-dir INFOS --scenes scene-0103,scene-0553,scene-0796,scene-0916 --split-name mini_val4 \
        --checkpoint CKPT --out OUT [--dump-first 20] [--threads 16]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import mmdet3d
import mmengine
import numpy as np
import nuscenes
import torch
from mmdet3d.apis import init_model
from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric
from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval

CONFIG = "configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def tensors(obj, prefix=""):
    """Flatten nested lists/tuples/dicts of tensors into {path: ndarray}."""
    out = {}
    if isinstance(obj, torch.Tensor):
        out[prefix or "t"] = obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(tensors(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(tensors(v, f"{prefix}[{i}]"))
    return out


def subset_infos(nusc: NuScenes, infos_dir: Path, scenes: list[str]) -> dict:
    scene_of = {s["token"]: s["name"] for s in nusc.scene}
    sample_scene = {s["token"]: scene_of[s["scene_token"]] for s in nusc.sample}
    rows, metainfo = [], None
    for split in ("train", "val"):
        if not (infos_dir / f"nuscenes_infos_{split}.pkl").exists():  # e.g. trainval with the val LiDAR files only
            continue
        d = mmengine.load(infos_dir / f"nuscenes_infos_{split}.pkl")
        metainfo = metainfo or d["metainfo"]
        rows += [r for r in d["data_list"] if sample_scene[r["token"]] in scenes]
    rows.sort(key=lambda r: (scenes.index(sample_scene[r["token"]]), r["timestamp"]))
    for i, r in enumerate(rows):
        r["sample_idx"] = i
    return {"metainfo": metainfo, "data_list": rows}, sample_scene


def devkit_root(data_root: Path, version: str, work: Path, split_name: str, scenes: list[str]) -> Path:
    """A dataroot of symlinks with an added <version>/splits.json (the source tree stays untouched)."""
    (work / version).mkdir(parents=True, exist_ok=True)
    for name in ("samples", "sweeps", "maps"):
        if not (work / name).exists():
            os.symlink(data_root / name, work / name)
    for f in (data_root / version).iterdir():
        if not (work / version / f.name).exists():
            os.symlink(f, work / version / f.name)
    (work / version / "splits.json").write_text(json.dumps({split_name: scenes}))
    return work


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--infos-dir", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--split-name", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-first", type=int, default=20)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--sweeps-test-mode", action="store_true",
                    help="set test_mode=True on LoadPointsFromMultiSweeps (the 9 nearest sweeps). The checkpoint's "
                         "config omits it, so without this flag the loader draws 9 of the 10 sweeps at random")
    ap.add_argument("--seed", type=int, default=None, help="numpy seed (only matters without --sweeps-test-mode)")
    a = ap.parse_args()
    if a.seed is not None:
        np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    register_all_modules()
    data_root, out, ckpt = Path(a.data_root).resolve(), Path(a.out).resolve(), Path(a.checkpoint).resolve()
    scenes = a.scenes.split(",")
    out.mkdir(parents=True, exist_ok=True)
    (out / "dump").mkdir(exist_ok=True)

    nusc = NuScenes(version=a.version, dataroot=str(data_root), verbose=False)
    subset, sample_scene = subset_infos(nusc, Path(a.infos_dir).resolve(), scenes)
    ann = out / "infos_subset.pkl"
    mmengine.dump(subset, ann)

    cfg_path = Path(mmdet3d.__file__).parent / ".mim" / CONFIG
    cfg = Config.fromfile(str(cfg_path))
    ds_cfg = cfg.test_dataloader.dataset
    ds_cfg.data_root = str(data_root) + "/"
    ds_cfg.ann_file = str(ann)
    ds_cfg.test_mode = True
    sweep_steps = [t for t in ds_cfg.pipeline if t["type"] == "LoadPointsFromMultiSweeps"]
    assert len(sweep_steps) == 1
    if a.sweeps_test_mode:
        sweep_steps[0]["test_mode"] = True
    dataset = DATASETS.build(ds_cfg)
    assert len(dataset) == len(subset["data_list"]), (len(dataset), len(subset["data_list"]))

    model = init_model(cfg, str(ckpt), device="cpu")
    model.eval()
    cap: dict = {}
    model.data_preprocessor.register_forward_hook(lambda m, i, o: cap.__setitem__("pre", o))
    model.pts_bbox_head.register_forward_hook(lambda m, i, o: cap.__setitem__("head", o))

    records, results = [], []
    for i in range(len(dataset)):
        item = dataset[i]
        batch = pseudo_collate([item])
        with torch.no_grad():
            res = model.test_step(batch)[0]
        points = batch["inputs"]["points"][0].numpy()
        vox = cap["pre"]["inputs"]["voxels"]
        voxels, nump, coors = (vox[k].numpy() for k in ("voxels", "num_points", "coors"))
        head = tensors(cap["head"])
        p = res.pred_instances_3d
        boxes, scores, labels = p.bboxes_3d.tensor.numpy(), p.scores_3d.numpy(), p.labels_3d.numpy()
        info = subset["data_list"][i]
        records.append({
            "index": i, "token": info["token"], "scene": sample_scene[info["token"]], "timestamp": info["timestamp"],
            "points": int(points.shape[0]), "points_sha256": sha(points.astype(np.float32)),
            "pillars": int(voxels.shape[0]), "voxels_sha256": sha(voxels), "num_points_sha256": sha(nump),
            "coors_sha256": sha(coors), "head_sha256": {k: sha(v) for k, v in head.items()},
            "detections": int(boxes.shape[0]),
        })
        if i < a.dump_first:
            np.savez_compressed(out / "dump" / f"{i:03d}.npz", points=points, voxels=voxels, num_points=nump,
                                coors=coors, boxes=boxes, scores=scores, labels=labels,
                                **{f"head:{k}": v for k, v in head.items()})
        results.append({"bboxes_3d": p.bboxes_3d, "scores_3d": p.scores_3d, "labels_3d": p.labels_3d})
        if i % 20 == 0:
            print(f"{i}/{len(dataset)} points {points.shape[0]} pillars {voxels.shape[0]} dets {boxes.shape[0]}",
                  flush=True)

    metric = NuScenesMetric(data_root=str(data_root), ann_file=str(ann))
    metric.data_infos = subset["data_list"]
    res_path = metric._format_lidar_bbox(results, list(range(len(results))), list(dataset.metainfo["classes"]),
                                         str(out / "results"))
    root = devkit_root(data_root, a.version, out / "devkit_root", a.split_name, scenes)
    nusc2 = NuScenes(version=a.version, dataroot=str(root), verbose=False)
    ev = DetectionEval(nusc2, config_factory("detection_cvpr_2019"), res_path, eval_set=a.split_name,
                       output_dir=str(out / "devkit"), verbose=False)
    metrics, _ = ev.evaluate()
    summary = metrics.serialize()
    (out / "devkit_metrics.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    (out / "samples.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    env = {"mmdet3d": mmdet3d.__version__, "torch": torch.__version__, "numpy": np.__version__,
           "nuscenes_devkit": getattr(nuscenes, "__version__", "1.2.0 (pip)"), "python": platform.python_version(),
           "config": CONFIG, "config_sha256": file_sha(cfg_path), "checkpoint": ckpt.name,
           "checkpoint_sha256": file_sha(ckpt), "scenes": scenes, "split_name": a.split_name,
           "samples": len(records), "threads": a.threads, "sweeps_test_mode": a.sweeps_test_mode, "seed": a.seed,
           "sweep_loader": {k: v for k, v in sweep_steps[0].items()}, "results_nusc_sha256": file_sha(Path(res_path))}
    (out / "environment.json").write_text(json.dumps(env, indent=1, sort_keys=True))
    print("mAP %.6f NDS %.6f samples %d" % (summary["mean_ap"], summary["nd_score"], len(records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
