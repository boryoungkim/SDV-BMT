"""nuScenes 3D detection task quality from the run bundle (``nuscenes-det3d-cvpr2019/1``), standard library only.

Predictions: the recorded ``centerpoint-det3d-f32/1`` task outputs (LiDAR frame of the keyframe, bottom centre) are
converted as mmdetection3d 1.4.0 ``NuScenesMetric`` does (``output_to_nusc_box`` + ``lidar_nusc_box_to_global`` +
the attribute rule of ``_format_lidar_bbox``): gravity centre, size (dy, dx, dz) as (w, l, h), yaw quaternion about z,
velocity (vx, vy, 0); rotated and translated by lidar2ego; dropped when the ego-frame planar radius exceeds the class
range; rotated and translated by ego2global; attribute from the planar global speed (> 0.2 m/s).

Ground truth: frozen in the bundle by the input (``nuscenes-det3d-gt/1``): the devkit's boxes (translation, size,
rotation, 2D velocity, detection name, attribute, lidar + radar points), the bicycle racks and the keyframe poses.

Evaluation: the nuScenes devkit 1.2.0 detection evaluation with ``detection_cvpr_2019`` (``DetectionEval``:
``add_center_dist``, ``filter_eval_boxes``, ``accumulate``, ``calc_ap``, ``calc_tp``, ``DetectionMetrics``), written
again with numpy's summation order (pairwise sums) and ``np.interp`` semantics.  Cross-checked against the devkit
(``tools/centerpoint/det3d_eval_crosscheck.py``).
"""

from __future__ import annotations

import bisect
import math
from typing import Any, Iterable, Optional

CLASS_NAMES = ("car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle",
               "pedestrian", "traffic_cone")  # label order of the CenterPoint head
CLASS_RANGE = {"car": 50, "truck": 50, "bus": 50, "trailer": 50, "construction_vehicle": 50, "pedestrian": 40,
               "motorcycle": 40, "bicycle": 40, "traffic_cone": 30, "barrier": 30}
DIST_THS = (0.5, 1.0, 2.0, 4.0)
DIST_TH_TP = 2.0
MIN_RECALL = 0.1
MIN_PRECISION = 0.1
MAX_BOXES_PER_SAMPLE = 500
MEAN_AP_WEIGHT = 5
TP_METRICS = ("trans_err", "scale_err", "orient_err", "vel_err", "attr_err")
NELEM = 101
DEFAULT_ATTRIBUTE = {"car": "vehicle.parked", "pedestrian": "pedestrian.moving", "trailer": "vehicle.parked",
                     "truck": "vehicle.parked", "bus": "vehicle.moving", "motorcycle": "cycle.without_rider",
                     "construction_vehicle": "vehicle.parked", "bicycle": "cycle.without_rider", "barrier": "",
                     "traffic_cone": ""}
NAN = float("nan")


# ---------------------------------------------------------------- numpy-compatible arithmetic
def np_sum(a: list[float]) -> float:
    """numpy pairwise summation of a contiguous float64 array (np.add.reduce)."""
    n = len(a)
    if n < 8:
        res = 0.0
        for v in a:
            res += v
        return res
    if n <= 128:
        r = list(a[:8])
        m = n - n % 8
        for i in range(8, m, 8):
            for k in range(8):
                r[k] += a[i + k]
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        for i in range(m, n):
            res += a[i]
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return np_sum(a[:n2]) + np_sum(a[n2:])


def np_mean(a: list[float]) -> float:
    return np_sum(a) / len(a)


def np_nanmean(a: list[float]) -> float:
    vals = [0.0 if math.isnan(v) else v for v in a]
    count = sum(1 for v in a if not math.isnan(v))
    return np_sum(vals) / count if count else NAN


def np_interp(x: Iterable[float], xp: list[float], fp: list[float], left: Optional[float] = None,
              right: Optional[float] = None) -> list[float]:
    """numpy.interp (npy_interp) for increasing xp, including repeated xp values."""
    lo = fp[0] if left is None else left
    hi = fp[-1] if right is None else right
    out = []
    n = len(xp)
    for xv in x:
        if math.isnan(xv):
            out.append(xv)
        elif n == 1:
            out.append(lo if xv < xp[0] else hi if xv > xp[0] else fp[0])
        elif xv > xp[-1]:
            out.append(hi)
        elif xv < xp[0]:
            out.append(lo)
        elif xv == xp[-1]:
            out.append(fp[-1])
        else:
            j = bisect.bisect_right(xp, xv) - 1
            slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])  # xp[j] <= xv < xp[j + 1]: never a zero width
            v = slope * (xv - xp[j]) + fp[j]
            if math.isnan(v):
                v = slope * (xv - xp[j + 1]) + fp[j + 1]
                if math.isnan(v) and fp[j] == fp[j + 1]:
                    v = fp[j]
            out.append(v)
    return out


def linspace_101() -> list[float]:
    return [k * 0.01 for k in range(NELEM - 1)] + [1.0]


# ---------------------------------------------------------------- quaternion geometry (pyquaternion conventions)
def q_norm(q):
    """pyquaternion _normalise: unchanged when |1 - sum of squares| < 1e-14."""
    ss = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
    if abs(1.0 - ss) < 1e-14:
        return tuple(q)
    n = math.sqrt(ss)
    return tuple(v / n for v in q)


def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def q_matrix(q):
    """pyquaternion rotation_matrix: (q_matrix . q_bar_matrix^T)[1:, 1:]."""
    w, x, y, z = q_norm(q)
    qm = ((w, -x, -y, -z), (x, w, -z, y), (y, z, w, -x), (z, -y, x, w))
    qb = ((w, -x, -y, -z), (x, w, z, -y), (y, -z, w, x), (z, y, -x, w))
    return tuple(tuple(qm[i][0] * qb[j][0] + qm[i][1] * qb[j][1] + qm[i][2] * qb[j][2] + qm[i][3] * qb[j][3]
                       for j in (1, 2, 3)) for i in (1, 2, 3))


def mat_vec(m, v):
    return tuple(m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2] for i in range(3))


def q_yaw(q) -> float:
    v = mat_vec(q_matrix(q), (1.0, 0.0, 0.0))
    return math.atan2(v[1], v[0])


def angle_diff(x: float, y: float, period: float) -> float:
    diff = (x - y + period / 2) % period - period / 2  # float % takes the divisor's sign, as numpy
    if diff > math.pi:
        diff -= 2 * math.pi
    return diff


def point_in_box(center, size_wlh, rot, p) -> bool:
    """nuscenes points_in_box for one point (corner-based projections)."""
    w, l, h = size_wlh
    xs = [l / 2 * s for s in (1, 1, 1, 1, -1, -1, -1, -1)]
    ys = [w / 2 * s for s in (1, -1, -1, 1, 1, -1, -1, 1)]
    zs = [h / 2 * s for s in (1, 1, -1, -1, 1, 1, -1, -1)]
    m = q_matrix(rot)
    corners = [tuple(m[r][0] * xs[c] + m[r][1] * ys[c] + m[r][2] * zs[c] + center[r] for r in range(3)) for c in range(8)]
    p1, px, py, pz = corners[0], corners[4], corners[1], corners[3]
    i = [px[k] - p1[k] for k in range(3)]
    j = [py[k] - p1[k] for k in range(3)]
    k_ = [pz[k] - p1[k] for k in range(3)]
    v = [p[k] - p1[k] for k in range(3)]
    dot = lambda a, b: a[0] * b[0] + a[1] * b[1] + a[2] * b[2]  # noqa: E731
    return (0 <= dot(i, v) <= dot(i, i)) and (0 <= dot(j, v) <= dot(j, j)) and (0 <= dot(k_, v) <= dot(k_, k_))


# ---------------------------------------------------------------- predictions: LiDAR frame -> global (mmdet3d)
def predictions_to_global(dets: list[tuple[tuple[float, ...], float, int]], lidar2ego: dict,
                          ego2global: dict) -> list[dict[str, Any]]:
    q_le, t_le = tuple(lidar2ego["rotation"]), lidar2ego["translation"]
    q_eg, t_eg = tuple(ego2global["rotation"]), ego2global["translation"]
    m_le, m_eg = q_matrix(q_le), q_matrix(q_eg)
    out = []
    for box, score, label in dets:
        x, y, zb, dx, dy, dz, yaw, vx, vy = box
        name = CLASS_NAMES[label]
        center = (x, y, zb + dz * 0.5)
        rot = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
        vel = (vx, vy, 0.0)
        center = mat_vec(m_le, center)
        rot = q_mul(q_le, rot)
        vel = mat_vec(m_le, vel)
        center = tuple(center[k] + t_le[k] for k in range(3))
        if math.sqrt(center[0] ** 2 + center[1] ** 2) > CLASS_RANGE[name]:
            continue
        center = mat_vec(m_eg, center)
        rot = q_mul(q_eg, rot)
        vel = mat_vec(m_eg, vel)
        center = tuple(center[k] + t_eg[k] for k in range(3))
        if math.sqrt(vel[0] ** 2 + vel[1] ** 2) > 0.2:
            attr = ("vehicle.moving" if name in ("car", "construction_vehicle", "bus", "truck", "trailer")
                    else "cycle.with_rider" if name in ("bicycle", "motorcycle") else DEFAULT_ATTRIBUTE[name])
        else:
            attr = ("pedestrian.standing" if name == "pedestrian" else "vehicle.stopped" if name == "bus"
                    else DEFAULT_ATTRIBUTE[name])
        out.append({"translation": center, "size": (dy, dx, dz), "rotation": rot, "velocity": (vel[0], vel[1]),
                    "detection_name": name, "detection_score": score, "attribute_name": attr, "num_pts": -1})
    return out


# ---------------------------------------------------------------- devkit evaluation
def _filter(boxes: list[dict], ego_xyz, racks: list[dict]) -> list[dict]:
    kept = []
    for b in boxes:
        ex, ey = b["translation"][0] - ego_xyz[0], b["translation"][1] - ego_xyz[1]
        if math.sqrt(ex * ex + ey * ey) < CLASS_RANGE[b["detection_name"]] and b["num_pts"] != 0:
            kept.append(b)
    out = []
    for b in kept:
        if b["detection_name"] in ("bicycle", "motorcycle") and any(
                point_in_box(r["translation"], r["size"], tuple(r["rotation"]), b["translation"]) for r in racks):
            continue
        out.append(b)
    return out


def _cummean(x: list[float]) -> list[float]:
    if all(math.isnan(v) for v in x):
        return [1.0] * len(x)
    out, s, c = [], 0.0, 0
    for v in x:
        if not math.isnan(v):
            s += v
            c += 1
        out.append(s / c if c else 0.0)
    return out


def _no_predictions() -> dict[str, list[float]]:
    return {"recall": linspace_101(), "precision": [0.0] * NELEM, "confidence": [0.0] * NELEM,
            **{m: [1.0] * NELEM for m in TP_METRICS}}


def _accumulate(gt: dict[str, list[dict]], pred: dict[str, list[dict]], order: list[str], cls: str,
                dist_th: float) -> dict[str, list[float]]:
    npos = sum(1 for s in order for b in gt[s] if b["detection_name"] == cls)
    if npos == 0:
        return _no_predictions()
    plist = [b for s in order for b in pred[s] if b["detection_name"] == cls]
    confs = [b["detection_score"] for b in plist]
    sortind = [i for (_v, i) in sorted((v, i) for i, v in enumerate(confs))][::-1]
    tp, fp, conf = [], [], []
    md = {m: [] for m in TP_METRICS}
    md_conf = []
    taken = set()
    for ind in sortind:
        p = plist[ind]
        s = p["sample"]
        min_dist, match = math.inf, None
        for gi, g in enumerate(gt[s]):
            if g["detection_name"] == cls and (s, gi) not in taken:
                d = math.sqrt((p["translation"][0] - g["translation"][0]) ** 2 +
                              (p["translation"][1] - g["translation"][1]) ** 2)
                if d < min_dist:
                    min_dist, match = d, gi
        if min_dist < dist_th:
            taken.add((s, match))
            tp.append(1)
            fp.append(0)
            conf.append(p["detection_score"])
            g = gt[s][match]
            md["trans_err"].append(math.sqrt((p["translation"][0] - g["translation"][0]) ** 2 +
                                             (p["translation"][1] - g["translation"][1]) ** 2))
            gv = g["velocity"]
            md["vel_err"].append(NAN if gv is None else math.sqrt((p["velocity"][0] - gv[0]) ** 2 +
                                                                  (p["velocity"][1] - gv[1]) ** 2))
            sa, sr = g["size"], p["size"]
            inter = min(sa[0], sr[0]) * min(sa[1], sr[1]) * min(sa[2], sr[2])
            va, vr = sa[0] * sa[1] * sa[2], sr[0] * sr[1] * sr[2]
            md["scale_err"].append(1 - inter / (va + vr - inter))
            period = math.pi if cls == "barrier" else 2 * math.pi
            md["orient_err"].append(abs(angle_diff(q_yaw(tuple(g["rotation"])), q_yaw(tuple(p["rotation"])), period)))
            md["attr_err"].append(NAN if g["attribute_name"] == "" else 1 - float(g["attribute_name"] == p["attribute_name"]))
            md_conf.append(p["detection_score"])
        else:
            tp.append(0)
            fp.append(1)
            conf.append(p["detection_score"])
    if not md["trans_err"]:
        return _no_predictions()
    ctp, cfp, a, b = [], [], 0.0, 0.0
    for t_, f_ in zip(tp, fp):
        a += t_
        b += f_
        ctp.append(a)
        cfp.append(b)
    prec = [t_ / (f_ + t_) for t_, f_ in zip(ctp, cfp)]
    rec = [t_ / float(npos) for t_ in ctp]
    rec_i = linspace_101()
    prec_i = np_interp(rec_i, rec, prec, right=0.0)
    conf_i = np_interp(rec_i, rec, conf, right=0.0)
    out = {"recall": rec_i, "precision": prec_i, "confidence": conf_i}
    xr = md_conf[::-1]
    for m in TP_METRICS:
        tmp = _cummean(md[m])
        out[m] = np_interp(conf_i[::-1], xr, tmp[::-1])[::-1]
    return out


def _calc_ap(md: dict[str, list[float]]) -> float:
    prec = [max(0.0, v - MIN_PRECISION) for v in md["precision"][round(100 * MIN_RECALL) + 1:]]
    return float(np_mean(prec)) / (1.0 - MIN_PRECISION)


def _calc_tp(md: dict[str, list[float]], metric: str) -> float:
    first = round(100 * MIN_RECALL) + 1
    nz = [i for i, c in enumerate(md["confidence"]) if c != 0]
    last = nz[-1] if nz else 0
    if last < first:
        return 1.0
    return float(np_mean(md[metric][first:last + 1]))


def evaluate(samples: list[str], gt: dict[str, list[dict]], pred: dict[str, list[dict]],
             racks: dict[str, list[dict]], ego: dict[str, tuple[float, float, float]]) -> dict[str, Any]:
    """devkit DetectionEval.evaluate on global boxes; ``samples`` fixes the order of the predictions."""
    g, p = {}, {}
    for s in samples:
        if len(pred[s]) > MAX_BOXES_PER_SAMPLE:
            raise ValueError(f"sample {s}: more than {MAX_BOXES_PER_SAMPLE} predicted boxes")
        g[s] = _filter([dict(b, sample=s) for b in gt[s]], ego[s], racks[s])
        p[s] = _filter([dict(b, sample=s) for b in pred[s]], ego[s], racks[s])
    label_ap = {c: {} for c in CLASS_NAMES}
    label_tp = {c: {} for c in CLASS_NAMES}
    for c in CLASS_NAMES:
        mds = {d: _accumulate(g, p, samples, c, d) for d in DIST_THS}
        for d in DIST_THS:
            label_ap[c][d] = _calc_ap(mds[d])
        for m in TP_METRICS:
            if (c == "traffic_cone" and m in ("attr_err", "vel_err", "orient_err")) or \
                    (c == "barrier" and m in ("attr_err", "vel_err")):
                label_tp[c][m] = NAN
            else:
                label_tp[c][m] = _calc_tp(mds[DIST_TH_TP], m)
    mean_dist_aps = {c: float(np_mean([label_ap[c][d] for d in DIST_THS])) for c in CLASS_NAMES}
    mean_ap = float(np_mean([mean_dist_aps[c] for c in CLASS_NAMES]))
    tp_errors = {m: float(np_nanmean([label_tp[c][m] for c in CLASS_NAMES])) for m in TP_METRICS}
    tp_scores = {m: max(0.0, 1.0 - tp_errors[m]) for m in TP_METRICS}
    nd = float(MEAN_AP_WEIGHT * mean_ap + np_sum(list(tp_scores.values()))) / float(MEAN_AP_WEIGHT + len(tp_scores))
    return {"mean_ap": mean_ap, "nd_score": nd, "tp_errors": tp_errors, "tp_scores": tp_scores,
            "mean_dist_aps": mean_dist_aps, "label_aps": {c: {str(d): v for d, v in label_ap[c].items()} for c in CLASS_NAMES},
            "label_tp_errors": label_tp, "samples": len(samples),
            "gt_boxes": sum(len(g[s]) for s in samples), "pred_boxes": sum(len(p[s]) for s in samples)}


def ground_truth(annotation: dict) -> tuple[list[dict], list[dict], dict, dict]:
    boxes = [{"translation": tuple(b["translation"]), "size": tuple(b["size"]), "rotation": tuple(b["rotation"]),
              "velocity": None if b["velocity"] is None else tuple(b["velocity"]), "detection_name": b["detection_name"],
              "attribute_name": b["attribute_name"], "num_pts": b["num_pts"], "detection_score": -1.0}
             for b in annotation["boxes"]]
    return boxes, annotation["bicycle_racks"], annotation["lidar2ego"], annotation["ego2global"]


# ---------------------------------------------------------------- bundle-only evaluation (Metric Set v1.2)
from dataclasses import dataclass, field  # noqa: E402

from ..contracts import det3d as _codec  # noqa: E402
from ..contracts.ids import digest_of, sha256_hex  # noqa: E402
from ..contracts.quality import NUSCENES_DET3D_V1, QUALITY_EVALUATORS  # noqa: E402
from ..contracts.status import MetricStatus, RequestState  # noqa: E402

EVALUATOR_MODULE = __name__


@dataclass(frozen=True)
class Det3DQuality:
    status: MetricStatus
    reason: Optional[str] = None
    notes: tuple[str, ...] = ()
    evaluated_items: int = 0
    result: Optional[dict[str, Any]] = None
    population: dict[str, Any] = field(default_factory=dict)
    annotation_check: Optional[tuple[int, int]] = None


def det3d_protocol() -> dict[str, Any]:
    """Evaluation-identity parameters of ``nuscenes-det3d-cvpr2019/1``."""
    return {
        "evaluator": NUSCENES_DET3D_V1,
        "implemented": True,
        "module": EVALUATOR_MODULE,
        "population": "conditional: items of COMPLETED requests with a valid task output",
        "det3d": {
            "classes": list(CLASS_NAMES), "class_range_m": CLASS_RANGE, "dist_ths_m": list(DIST_THS),
            "dist_th_tp_m": DIST_TH_TP, "min_recall": MIN_RECALL, "min_precision": MIN_PRECISION,
            "max_boxes_per_sample": MAX_BOXES_PER_SAMPLE, "mean_ap_weight": MEAN_AP_WEIGHT,
            "output_codec": _codec.CODEC_ID,
            "conversion": "mmdetection3d 1.4.0 NuScenesMetric: gravity centre, (w, l, h) = (dy, dx, dz), yaw about z, "
                          "lidar2ego, ego-frame class-range filter, ego2global, attribute from planar speed > 0.2 m/s",
            "conformance": "nuScenes devkit 1.2.0 DetectionEval with detection_cvpr_2019; cross-checked against the "
                           "devkit (tools/centerpoint/det3d_eval_crosscheck.py)",
        },
    }


def det3d_from_items(items: list[tuple[str, dict, bytes]]) -> dict[str, Any]:
    """[(source_item_id, frozen annotation, encoded output)] in evaluation order -> devkit-equivalent metrics."""
    samples, gt, pred, racks, ego = [], {}, {}, {}, {}
    for sid, annotation, blob in items:
        boxes, rk, l2e, e2g = ground_truth(annotation)
        _item, dets = _codec.decode(blob)
        samples.append(sid)
        gt[sid], racks[sid], ego[sid] = boxes, rk, tuple(e2g["translation"])
        pred[sid] = predictions_to_global(dets, l2e, e2g)
    return evaluate(samples, gt, pred, racks, ego)


def det3d_quality(bundle: Any, folded: dict, protocol: Optional[dict[str, Any]]) -> Det3DQuality:
    return det3d_quality_parts([(bundle, folded, None)], protocol)


def det3d_quality_parts(parts: list[tuple[Any, dict, Optional[str]]],
                        protocol: Optional[dict[str, Any]]) -> Det3DQuality:
    """Quality of one run (one part, label None) or of a run series (parts in series order, labelled): the items of
    all parts are evaluated once, in part order and item order; a sample may occur in one part only."""
    from ..observation.reader import read_output_blob
    planned = sum(len(folded) for _b, folded, _l in parts)
    if any(bundle.quality_reference is None for bundle, _f, _l in parts):
        reason = "quality_evaluator_not_declared" if protocol is None else "quality_reference_not_in_bundle"
        return Det3DQuality(MetricStatus.NOT_COLLECTED, reason, (
            "task quality is computed only from a ground truth frozen in the bundle (manifest v5); this bundle has none",))
    verified = total = 0
    consistent = True
    for bundle, _folded, _label in parts:
        items, rows = bundle.input_items, bundle.quality_reference
        by_index = {item.item_index: item for item in items}
        v = sum(1 for row in rows if (item := by_index.get(row.item_index)) is not None
                and row.source_item_id == item.source_item_id and row.annotation_digest == item.annotation_digest
                and digest_of(row.annotation) == row.annotation_digest)
        verified, total = verified + v, total + len(items)
        consistent = consistent and v == len(items) and len(rows) == len(items)
    check = (verified, total)
    if not consistent:
        return Det3DQuality(MetricStatus.INVALID, "quality_reference_integrity", (
            f"{verified} of {total} input items have a frozen ground truth matching their annotation_digest",),
            annotation_check=check)
    if protocol is None or protocol.get("evaluator") != NUSCENES_DET3D_V1 or not protocol.get("implemented"):
        return Det3DQuality(MetricStatus.NOT_APPLICABLE, f"quality_evaluator_is:{None if protocol is None else protocol.get('evaluator')}",
                            annotation_check=check)
    ordered: list[tuple[str, dict, bytes]] = []
    seen: set[str] = set()
    for bundle, folded, label in parts:
        items = bundle.input_items
        truth = {row.item_index: row.annotation for row in bundle.quality_reference}
        where = "" if label is None else f"{label} "
        evaluated: dict[int, tuple[str, dict, bytes]] = {}
        for request, completion in folded.values():
            if request.state is not RequestState.COMPLETED or completion is None or completion.output_ref is None:
                continue
            if len(request.item_indices) != 1:
                return Det3DQuality(MetricStatus.INVALID, "batched_request_not_supported", annotation_check=check)
            item = items[request.item_indices[0]]
            ref = completion.output_ref
            blob = read_output_blob(bundle.run_dir, ref)
            try:
                if len(blob) != ref.length or sha256_hex(blob) != ref.sha256:
                    raise ValueError("stored output does not match its output_ref")
                out_item, _ = _codec.decode(blob)
                if out_item != item.item_index:
                    raise ValueError("task output item index differs from the request's item")
            except ValueError as exc:
                return Det3DQuality(MetricStatus.INVALID, "task_output_invalid",
                                    (f"{where}request {request.request_id}: {exc}",), annotation_check=check)
            if item.item_index in evaluated or item.source_item_id in seen:
                return Det3DQuality(MetricStatus.INVALID, "duplicate_item_output", annotation_check=check)
            evaluated[item.item_index] = (item.source_item_id, truth[item.item_index], blob)
        for k in sorted(evaluated):
            seen.add(evaluated[k][0])
            ordered.append(evaluated[k])
    notes = (f"conditional: {len(ordered)} of {planned} planned requests have an evaluated task output; interpret "
             "with quality.det3d.coverage", f"ground truth frozen in the bundle, evaluator {NUSCENES_DET3D_V1}")
    if len(parts) > 1:
        notes += (f"run series: the items of the {len(parts)} parts are evaluated once, as one population",)
    if not ordered:
        return Det3DQuality(MetricStatus.EMPTY, "no_evaluated_items", notes, 0, annotation_check=check)
    result = det3d_from_items(ordered)
    population = {"samples": result["samples"], "gt_boxes_after_filter": result["gt_boxes"],
                  "pred_boxes_after_filter": result["pred_boxes"]}
    return Det3DQuality(MetricStatus.VALID, None, notes, len(ordered), result, population, check)
