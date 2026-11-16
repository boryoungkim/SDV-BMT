"""Build the CenterPoint-Pillars nuScenes input: 10-sweep point clouds, hard voxelization, frozen 3D ground truth.

Offline asset preparation (outside src/ and the measurement path).  Run with the cross-check environment
``<venv-mmseg>``: numpy 1.26.4 and pyquaternion 0.9.9, the same numerical libraries the official mmdetection3d
pipeline uses, so the payloads can be compared byte for byte with the official pipeline (``official_run.py``).
The nuScenes tables are read as JSON; the devkit is not used.

Per keyframe (scenes in the given order, samples by timestamp):

* points: the official test pipeline of the checkpoint's config.  The keyframe LIDAR_TOP points (x, y, z, intensity,
  time lag 0) followed by its previous sweeps: the converter walks ``prev`` up to 10 sample_data records and the loader
  uses the first 9 (all when fewer; when there is none, 9 copies of the keyframe without the points within 1 m in x
  and y).  Each sweep: points with |x| < 1 and |y| < 1 removed, ``p[:, :3] @ lidar2sensor[:3, :3] -
  lidar2sensor[:3, 3]`` with the official float32-rounded lidar2sensor matrix, time lag ``t_key/1e6 - t_sweep/1e6``.
* pillars: mmcv hard voxelization (voxel 0.2 x 0.2 x 8 m, range [-51.2, -51.2, -5, 51.2, 51.2, 3], at most 20 points
  per pillar, at most 40,000 pillars, pillars numbered in order of their first point, float32 coordinate arithmetic).
* payload (little-endian): 40-byte header ``SDVPIL01`` + u32 pillars, max_pillars, max_points, features, merged
  points, 3 x u32 0; then voxels float32 [40000, 20, 5] (zero padded), points per pillar int32 [40000], pillar (y, x)
  int32 [40000, 2].  16,480,040 bytes; the body after the header (16,480,000 bytes) is the device input.
* annotation ``nuscenes-det3d-gt/1``: the devkit's ground-truth boxes of the sample (translation, size, rotation,
  2D velocity by ``NuScenes.box_velocity``, detection name, attribute, lidar + radar points), the bicycle racks and the
  keyframe poses (lidar2ego, ego2global: rotation quaternion, translation) the evaluation needs to put predictions
  into the global frame and to measure ego distances.
* metadata: scene, timestamp, keyframe lidar2ego and ego2global (rotation quaternion, translation).

Outputs in --out: ``index.jsonl``, ``payloads/<sample_token>.pil``, ``dataset_manifest.json``.

    <venv-mmseg>/bin/python tools/centerpoint/build_nuscenes_input.py --nusc-root NUSC --version v1.0-mini \
        --scenes scene-0103,scene-0553,scene-0796,scene-0916 --out datasets/nuscenes-mini-val4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import struct
import sys
from pathlib import Path

import numpy as np
import pyquaternion
from pyquaternion import Quaternion

VOXEL = np.array([0.2, 0.2, 8.0], dtype=np.float32)
RANGE = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], dtype=np.float32)
MAX_POINTS, MAX_PILLARS, FEATURES = 20, 40000, 5
SWEEPS_INFO, SWEEPS_USED = 10, 9
MAGIC = b"SDVPIL01"
HEADER = struct.Struct("<8s8I")
PAYLOAD_BYTES = HEADER.size + MAX_PILLARS * MAX_POINTS * FEATURES * 4 + MAX_PILLARS * 4 + MAX_PILLARS * 2 * 4
DETECTION_NAME = {
    "movable_object.barrier": "barrier", "vehicle.bicycle": "bicycle", "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus", "vehicle.car": "car", "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle", "human.pedestrian.adult": "pedestrian", "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian", "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone", "vehicle.trailer": "trailer", "vehicle.truck": "truck"}
BIKE_RACK = "static_object.bicycle_rack"


class Tables:
    def __init__(self, root: Path, version: str) -> None:
        self.root = root
        load = lambda n: json.loads((root / version / f"{n}.json").read_text())  # noqa: E731
        by = lambda rows: {r["token"]: r for r in rows}  # noqa: E731
        self.scene = load("scene")
        self.sample = by(load("sample"))
        self.sample_data = by(load("sample_data"))
        self.calibrated_sensor = by(load("calibrated_sensor"))
        self.ego_pose = by(load("ego_pose"))
        self.ann = by(load("sample_annotation"))
        self.instance = by(load("instance"))
        self.category = by(load("category"))
        self.attribute = by(load("attribute"))
        self.anns_of = {}
        for a in self.ann.values():
            self.anns_of.setdefault(a["sample_token"], []).append(a["token"])
        self.lidar_of = {}
        for sd in self.sample_data.values():
            if sd["is_key_frame"] and sd["filename"].startswith("samples/LIDAR_TOP/"):
                self.lidar_of[sd["sample_token"]] = sd["token"]

    def category_name(self, ann: dict) -> str:
        return self.category[self.instance[ann["instance_token"]]["category_token"]]["name"]


def sensor2lidar(t: Tables, sd_token: str, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat):
    """mmdet3d nuscenes_converter.obtain_sensor2top (lidar sweeps), same operations and order."""
    sd = t.sample_data[sd_token]
    cs = t.calibrated_sensor[sd["calibrated_sensor_token"]]
    pose = t.ego_pose[sd["ego_pose_token"]]
    l2e_r_s, l2e_t_s = cs["rotation"], cs["translation"]
    e2g_r_s, e2g_t_s = pose["rotation"], pose["translation"]
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    rot, trans = R.T, T
    lidar2sensor = np.eye(4)  # update_infos_to_v2
    lidar2sensor[:3, :3] = rot.T
    lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
    return np.array(lidar2sensor.astype(np.float32).tolist()), sd["timestamp"] / 1e6


def remove_close(p: np.ndarray, radius: float = 1.0) -> np.ndarray:
    keep = np.logical_not(np.logical_and(np.abs(p[:, 0]) < radius, np.abs(p[:, 1]) < radius))
    return p[keep]


def load_bin(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype=np.float32).reshape(-1, FEATURES)


def merged_points(t: Tables, sample: dict) -> np.ndarray:
    sd_key = t.sample_data[t.lidar_of[sample["token"]]]
    cs = t.calibrated_sensor[sd_key["calibrated_sensor_token"]]
    pose = t.ego_pose[sd_key["ego_pose_token"]]
    l2e_t, e2g_t = cs["translation"], pose["translation"]
    l2e_r_mat, e2g_r_mat = Quaternion(cs["rotation"]).rotation_matrix, Quaternion(pose["rotation"]).rotation_matrix
    key = load_bin(t.root / sd_key["filename"]).copy()
    key[:, 4] = 0
    sweeps, rec = [], sd_key
    while len(sweeps) < SWEEPS_INFO and rec["prev"] != "":
        sweeps.append(rec["prev"])
        rec = t.sample_data[rec["prev"]]
    parts = [key]
    if not sweeps:
        parts += [remove_close(key) for _ in range(SWEEPS_USED)]
    else:
        ts = sample["timestamp"] / 1e6
        for sd_token in sweeps[:SWEEPS_USED]:
            lidar2sensor, sweep_ts = sensor2lidar(t, sd_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat)
            p = remove_close(np.copy(load_bin(t.root / t.sample_data[sd_token]["filename"])))
            p[:, :3] = p[:, :3] @ lidar2sensor[:3, :3]
            p[:, :3] -= lidar2sensor[:3, 3]
            p[:, 4] = ts - sweep_ts
            parts.append(p)
    return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)


def hard_voxelize(points: np.ndarray):
    """mmcv hard voxelization (CPU kernel semantics), vectorized; returns voxels, num_points, coors (z, y, x)."""
    grid = np.round((RANGE[3:] - RANGE[:3]) / VOXEL).astype(np.int64)  # [512, 512, 1] (x, y, z)
    c = np.floor((points[:, :3] - RANGE[:3]) / VOXEL).astype(np.int64)  # float32 arithmetic, as the kernel
    valid = np.all((c >= 0) & (c < grid), axis=1)
    idx = np.nonzero(valid)[0]
    cz, cy, cx = c[idx, 2], c[idx, 1], c[idx, 0]
    lin = (cz * grid[1] + cy) * grid[0] + cx
    uniq, first, inverse = np.unique(lin, return_index=True, return_inverse=True)
    rank_of_uniq = np.empty(len(uniq), dtype=np.int64)
    rank_of_uniq[np.argsort(first, kind="stable")] = np.arange(len(uniq))
    rank = rank_of_uniq[inverse]  # pillar number of each valid point, in order of first appearance
    keep = rank < MAX_PILLARS
    idx, rank = idx[keep], rank[keep]
    order = np.argsort(rank, kind="stable")  # points grouped by pillar, original order inside a pillar
    r_sorted = rank[order]
    start = np.searchsorted(r_sorted, r_sorted, side="left")
    slot = np.arange(len(r_sorted)) - start
    take = slot < MAX_POINTS
    n = min(len(uniq), MAX_PILLARS)
    voxels = np.zeros((n, MAX_POINTS, FEATURES), dtype=np.float32)
    voxels[r_sorted[take], slot[take]] = points[idx[order][take]]
    num_points = np.minimum(np.bincount(rank, minlength=n), MAX_POINTS).astype(np.int32)
    first_point = idx[order][start == np.arange(len(r_sorted))]  # first point of each pillar, by pillar number
    coors = np.stack([c[first_point, 2], c[first_point, 1], c[first_point, 0]], axis=1).astype(np.int32)
    return voxels, num_points, coors


def payload(points: np.ndarray, voxels: np.ndarray, num_points: np.ndarray, coors: np.ndarray) -> bytes:
    n = voxels.shape[0]
    v = np.zeros((MAX_PILLARS, MAX_POINTS, FEATURES), dtype="<f4")
    v[:n] = voxels
    npts = np.zeros(MAX_PILLARS, dtype="<i4")
    npts[:n] = num_points
    yx = np.zeros((MAX_PILLARS, 2), dtype="<i4")
    yx[:n] = coors[:, 1:3]
    out = HEADER.pack(MAGIC, n, MAX_PILLARS, MAX_POINTS, FEATURES, points.shape[0], 0, 0, 0) + v.tobytes() + \
        npts.tobytes() + yx.tobytes()
    assert len(out) == PAYLOAD_BYTES
    return out


def box_velocity(t: Tables, ann: dict, max_time_diff: float = 1.5):
    """nuscenes NuScenes.box_velocity, 2D part; None for NaN."""
    has_prev, has_next = ann["prev"] != "", ann["next"] != ""
    if not has_prev and not has_next:
        return None
    first = t.ann[ann["prev"]] if has_prev else ann
    last = t.ann[ann["next"]] if has_next else ann
    pos_diff = np.array(last["translation"]) - np.array(first["translation"])
    time_diff = 1e-6 * t.sample[last["sample_token"]]["timestamp"] - 1e-6 * t.sample[first["sample_token"]]["timestamp"]
    if has_next and has_prev:
        max_time_diff *= 2
    if time_diff > max_time_diff:
        return None
    v = pos_diff / time_diff
    return [float(v[0]), float(v[1])]


def annotation(t: Tables, sample_token: str) -> dict:
    boxes, racks = [], []
    for tok in t.anns_of.get(sample_token, []):
        a = t.ann[tok]
        cat = t.category_name(a)
        if cat == BIKE_RACK:
            racks.append({"translation": a["translation"], "size": a["size"], "rotation": a["rotation"]})
        name = DETECTION_NAME.get(cat)
        if name is None:
            continue
        attrs = a["attribute_tokens"]
        if len(attrs) > 1:
            raise SystemExit(f"annotation {tok}: more than one attribute")
        boxes.append({"translation": a["translation"], "size": a["size"], "rotation": a["rotation"],
                      "velocity": box_velocity(t, a), "detection_name": name,
                      "attribute_name": t.attribute[attrs[0]]["name"] if attrs else "",
                      "num_pts": a["num_lidar_pts"] + a["num_radar_pts"]})
    return {"boxes": boxes, "bicycle_racks": racks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nusc-root", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--scenes", required=True, help="comma-separated scene names, in output order")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.nusc_root).resolve(), Path(a.out)
    (out / "payloads").mkdir(parents=True, exist_ok=True)
    t = Tables(root, a.version)
    scene_by_name = {s["name"]: s for s in t.scene}
    rows, stats = [], []
    for name in a.scenes.split(","):
        sc = scene_by_name[name]
        tok = sc["first_sample_token"]
        while tok:
            sample = t.sample[tok]
            pts = merged_points(t, sample)
            voxels, num_points, coors = hard_voxelize(pts)
            body = payload(pts, voxels, num_points, coors)
            rel = f"payloads/{tok}.pil"
            (out / rel).write_bytes(body)
            sd = t.sample_data[t.lidar_of[tok]]
            cs, pose = t.calibrated_sensor[sd["calibrated_sensor_token"]], t.ego_pose[sd["ego_pose_token"]]
            rows.append({
                "source_item_id": tok, "payload_file": rel, "payload_sha256": hashlib.sha256(body).hexdigest(),
                "annotation": {**annotation(t, tok),
                               "lidar2ego": {"rotation": cs["rotation"], "translation": cs["translation"]},
                               "ego2global": {"rotation": pose["rotation"], "translation": pose["translation"]}},
                "metadata": {"scene": name, "timestamp_us": sample["timestamp"],
                             "lidar2ego": {"rotation": cs["rotation"], "translation": cs["translation"]},
                             "ego2global": {"rotation": pose["rotation"], "translation": pose["translation"]},
                             "points": int(pts.shape[0]), "pillars": int(voxels.shape[0])},
            })
            stats.append({"token": tok, "points_sha256": hashlib.sha256(pts.tobytes()).hexdigest(),
                          "voxels_sha256": hashlib.sha256(np.ascontiguousarray(voxels).tobytes()).hexdigest(),
                          "num_points_sha256": hashlib.sha256(num_points.tobytes()).hexdigest(),
                          "coors_sha256": hashlib.sha256(np.ascontiguousarray(coors).tobytes()).hexdigest(),
                          "points": int(pts.shape[0]), "pillars": int(voxels.shape[0])})
            tok = sample["next"]
    index = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)
    (out / "index.jsonl").write_text(index)
    (out / "build_stats.jsonl").write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in stats))
    manifest = {"schema": "sdv-bmt-nuscenes-pillar-input/1", "version": a.version, "scenes": a.scenes.split(","),
                "samples": len(rows), "payload_bytes": PAYLOAD_BYTES,
                "index_sha256": hashlib.sha256(index.encode()).hexdigest(),
                "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "numpy": np.__version__, "pyquaternion": pyquaternion.__version__ if hasattr(pyquaternion, "__version__")
                else "0.9.9", "python": platform.python_version()}
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps({k: manifest[k] for k in ("samples", "index_sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
