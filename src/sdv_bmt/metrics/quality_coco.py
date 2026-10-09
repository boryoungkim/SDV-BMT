"""COCO bounding-box evaluation, standard library only (canonical quality evaluator ``coco-detection-bbox/1``).

A re-implementation of pycocotools ``COCOeval`` semantics for ``iouType='bbox'`` (evaluate,
accumulate, summarize).  It is a port of ``tools/stage2b_g_real/coco_eval.py`` with the same
arithmetic in the same order; that tool was cross-checked against pycocotools 2.0.11 on the
Stage 2b-G YOLOX-S/COCO baseline (12 statistics within 7.6e-15,
``runs/stage2b-g-real/analysis/coco_eval_crosscheck.json``) and that cross-check stays the
conformance evidence of this module.  Parameters are the COCO defaults: IoU thresholds
linspace(0.5, 0.95, 10), recall thresholds linspace(0, 1, 101), maxDets (1, 10, 100), area ranges
all/small/medium/large, category-aware.

``threshold_counts`` adds the TP/FP/FN counts at one operating point (score and IoU threshold) with
the matching rules of ``evaluateImg`` at a single IoU threshold over area range ``all``.

Inputs:
  gts:  {image_id: [{"id", "category_id", "bbox": [x, y, w, h], "area", "iscrowd"}, ...]}  (every evaluated image)
  dets: {image_id: [{"category_id", "bbox": [x, y, w, h], "score"}, ...]} in output order (score ties keep it)
  cat_ids: the category ids of the evaluation (pycocotools ``getCatIds()`` of the annotation set)
"""

from __future__ import annotations

import bisect
from typing import Any

STAT_NAMES = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")
AREA_RANGES = ((0.0, 1e5 ** 2), (0.0, 32.0 ** 2), (32.0 ** 2, 96.0 ** 2), (96.0 ** 2, 1e5 ** 2))
AREA_RANGE_NAMES = ("all", "small", "medium", "large")
MAX_DETS = (1, 10, 100)
EPS = 2.220446049250313e-16  # np.spacing(1)


def _linspace(start: float, stop: float, num: int) -> list[float]:
    """numpy.linspace(start, stop, num, endpoint=True) for float64, value for value."""
    step = (stop - start) / (num - 1)
    values = [i * step + start for i in range(num)]
    values[-1] = stop
    return values


IOU_THRS = _linspace(0.5, 0.95, 10)
REC_THRS = _linspace(0.0, 1.0, 101)


def box_iou(d: list[float], g: list[float], crowd: bool) -> float:
    """maskApi.c bbIou for one detection/ground-truth pair (boxes x, y, w, h)."""
    w = min(d[2] + d[0], g[2] + g[0]) - max(d[0], g[0])
    if w <= 0:
        return 0.0
    h = min(d[3] + d[1], g[3] + g[1]) - max(d[1], g[1])
    if h <= 0:
        return 0.0
    i = w * h
    u = d[2] * d[3] if crowd else d[2] * d[3] + g[2] * g[3] - i
    return i / u


def _evaluate_img(gt: list[dict], dt: list[dict], ious: list[list[float]], a_rng: tuple[float, float],
                  max_det: int) -> dict[str, Any] | None:
    if not gt and not dt:
        return None
    ignore = [1 if (g["iscrowd"] or g["area"] < a_rng[0] or g["area"] > a_rng[1]) else 0 for g in gt]
    gtind = sorted(range(len(gt)), key=lambda k: ignore[k])  # stable: non-ignored first
    gt_s = [gt[k] for k in gtind]
    gt_ig = [ignore[k] for k in gtind]
    dt_s = dt[:max_det]  # dt is already score-sorted (stable) and truncated to maxDets[-1]
    crowd = [int(g["iscrowd"]) for g in gt_s]
    T, G, D = len(IOU_THRS), len(gt_s), len(dt_s)
    gtm = [[0] * G for _ in range(T)]
    dtm = [[0] * D for _ in range(T)]
    dt_ig = [[0] * D for _ in range(T)]
    if ious:
        for t_i, t in enumerate(IOU_THRS):
            for d_i in range(D):
                iou = min(t, 1 - 1e-10)
                m = -1
                for g_i in range(G):
                    if gtm[t_i][g_i] > 0 and not crowd[g_i]:
                        continue
                    if m > -1 and gt_ig[m] == 0 and gt_ig[g_i] == 1:
                        break
                    v = ious[d_i][gtind[g_i]]
                    if v < iou:
                        continue
                    iou = v
                    m = g_i
                if m == -1:
                    continue
                dt_ig[t_i][d_i] = gt_ig[m]
                dtm[t_i][d_i] = gt_s[m]["id"]
                gtm[t_i][m] = dt_s[d_i]["id"]
    out_of_range = [d["area"] < a_rng[0] or d["area"] > a_rng[1] for d in dt_s]
    for t_i in range(T):
        for d_i in range(D):
            if dtm[t_i][d_i] == 0 and out_of_range[d_i]:
                dt_ig[t_i][d_i] = 1
    return {"dt_scores": [d["score"] for d in dt_s], "dtm": dtm, "dt_ig": dt_ig, "gt_ig": gt_ig}


def evaluate(gts: dict[int, list[dict]], dets: dict[int, list[dict]], cat_ids: list[int]) -> dict[str, Any]:
    """The 12 standard statistics (-1 where COCO has no defined value) plus per-category AP."""
    img_ids = sorted(gts)
    cat_ids = sorted(cat_ids)
    # per (image, category) lists in input order; detections get ids 1.. in input order and their box area
    g_by: dict[tuple[int, int], list[dict]] = {}
    d_by: dict[tuple[int, int], list[dict]] = {}
    for img in img_ids:
        for g in gts[img]:
            g_by.setdefault((img, g["category_id"]), []).append(g)
    next_id = 1
    for img in img_ids:
        for d in dets.get(img, ()):
            d_by.setdefault((img, d["category_id"]), []).append(
                {"id": next_id, "bbox": d["bbox"], "score": d["score"], "area": d["bbox"][2] * d["bbox"][3]})
            next_id += 1
    eval_imgs: dict[tuple[int, int, int], dict | None] = {}
    for cat in cat_ids:
        for img in img_ids:
            gt = g_by.get((img, cat), [])
            dt = sorted(d_by.get((img, cat), []), key=lambda d: -d["score"])[:MAX_DETS[-1]]
            ious = [[box_iou(d["bbox"], g["bbox"], bool(g["iscrowd"])) for g in gt] for d in dt] if gt and dt else []
            for a_i, a_rng in enumerate(AREA_RANGES):
                eval_imgs[(cat, a_i, img)] = _evaluate_img(gt, dt, ious, a_rng, MAX_DETS[-1])
    T, R, K, A, M = len(IOU_THRS), len(REC_THRS), len(cat_ids), len(AREA_RANGES), len(MAX_DETS)
    precision = [[[[[-1.0] * M for _ in range(A)] for _ in range(K)] for _ in range(R)] for _ in range(T)]
    recall = [[[[-1.0] * M for _ in range(A)] for _ in range(K)] for _ in range(T)]
    for k, cat in enumerate(cat_ids):
        for a in range(A):
            for m, max_det in enumerate(MAX_DETS):
                E = [e for e in (eval_imgs[(cat, a, img)] for img in img_ids) if e is not None]
                if not E:
                    continue
                scores = [s for e in E for s in e["dt_scores"][:max_det]]
                order = sorted(range(len(scores)), key=lambda i: -scores[i])  # stable (mergesort)
                npig = sum(1 for e in E for g in e["gt_ig"] if g == 0)
                if npig == 0:
                    continue
                for t in range(T):
                    dtm = [x for e in E for x in e["dtm"][t][:max_det]]
                    dig = [x for e in E for x in e["dt_ig"][t][:max_det]]
                    tp_sum = fp_sum = 0.0
                    rc, pr = [], []
                    for i in order:
                        if dig[i]:
                            pass
                        elif dtm[i]:
                            tp_sum += 1.0
                        else:
                            fp_sum += 1.0
                        rc.append(tp_sum / npig)
                        pr.append(tp_sum / (fp_sum + tp_sum + EPS))
                    nd = len(rc)
                    recall[t][k][a][m] = rc[-1] if nd else 0.0
                    for i in range(nd - 1, 0, -1):
                        if pr[i] > pr[i - 1]:
                            pr[i - 1] = pr[i]
                    q = [0.0] * R
                    for r_i, thr in enumerate(REC_THRS):
                        pi = bisect.bisect_left(rc, thr)
                        if pi >= nd:
                            break
                        q[r_i] = pr[pi]
                    for r_i in range(R):
                        precision[t][r_i][k][a][m] = q[r_i]

    def mean_ap(t_sel, a_i, m_i, k_sel=None) -> float:
        vals = [precision[t][r][k][a_i][m_i] for t in t_sel for r in range(R)
                for k in (range(K) if k_sel is None else [k_sel])]
        vals = [v for v in vals if v > -1]
        return sum(vals) / len(vals) if vals else -1.0

    def mean_ar(a_i, m_i) -> float:
        vals = [recall[t][k][a_i][m_i] for t in range(T) for k in range(K)]
        vals = [v for v in vals if v > -1]
        return sum(vals) / len(vals) if vals else -1.0

    all_t = list(range(T))
    t50, t75 = IOU_THRS.index(0.5), IOU_THRS.index(0.75)
    stats = [mean_ap(all_t, 0, 2), mean_ap([t50], 0, 2), mean_ap([t75], 0, 2), mean_ap(all_t, 1, 2),
             mean_ap(all_t, 2, 2), mean_ap(all_t, 3, 2), mean_ar(0, 0), mean_ar(0, 1), mean_ar(0, 2),
             mean_ar(1, 2), mean_ar(2, 2), mean_ar(3, 2)]
    per_category = {cat: mean_ap(all_t, 0, 2, k) for k, cat in enumerate(cat_ids)}
    return {"stats": dict(zip(STAT_NAMES, stats)), "per_category_ap": per_category,
            "images": len(img_ids), "detections": next_id - 1,
            "gt_objects": sum(len(v) for v in gts.values()),
            "gt_crowd": sum(1 for v in gts.values() for g in v if g["iscrowd"])}


def threshold_counts(gts: dict[int, list[dict]], dets: dict[int, list[dict]], cat_ids: list[int],
                     score_threshold: float, iou_threshold: float, max_dets_per_image: int) -> dict[str, int]:
    """TP/FP/FN at one operating point.

    Per image: the ``max_dets_per_image`` highest-scoring detections (stable), then those with
    ``score >= score_threshold`` whose category is evaluated.  Per (image, category), detections in
    descending score order (stable) are matched greedily as in ``evaluateImg`` at the single IoU
    threshold over area range ``all``: the best IoU >= threshold among ground truth not yet matched
    (non-crowd first; a crowd region may absorb several detections).  A detection matched to a
    non-crowd box is a TP; one matched only to a crowd region is ignored (neither TP nor FP); an
    unmatched detection, including a second detection on an already matched box, is a FP.  Every
    unmatched non-crowd ground-truth box is a FN; crowd boxes never count as FN.
    """
    evaluated = set(cat_ids)
    tp = fp = fn = ignored = gt_non_crowd = considered = 0
    threshold = min(iou_threshold, 1 - 1e-10)
    for img in sorted(gts):
        dt_all = list(dets.get(img, ()))
        order = sorted(range(len(dt_all)), key=lambda i: -dt_all[i]["score"])[:max_dets_per_image]
        kept = [dt_all[i] for i in order
                if dt_all[i]["score"] >= score_threshold and dt_all[i]["category_id"] in evaluated]
        considered += len(kept)
        d_by: dict[int, list[dict]] = {}
        for d in kept:
            d_by.setdefault(d["category_id"], []).append(d)
        g_by: dict[int, list[dict]] = {}
        for g in gts[img]:
            if g["category_id"] in evaluated:
                g_by.setdefault(g["category_id"], []).append(g)
        for cat in sorted(set(d_by) | set(g_by)):
            gt = g_by.get(cat, [])
            ignore = [1 if g["iscrowd"] else 0 for g in gt]
            gtind = sorted(range(len(gt)), key=lambda k: ignore[k])  # stable: non-crowd first
            gt_s = [gt[k] for k in gtind]
            gt_ig = [ignore[k] for k in gtind]
            matched = [False] * len(gt_s)
            for d in d_by.get(cat, []):
                best = threshold
                m = -1
                for g_i, g in enumerate(gt_s):
                    if matched[g_i] and not gt_ig[g_i]:
                        continue
                    if m > -1 and gt_ig[m] == 0 and gt_ig[g_i] == 1:
                        break
                    v = box_iou(d["bbox"], g["bbox"], bool(g["iscrowd"]))
                    if v < best:
                        continue
                    best = v
                    m = g_i
                if m == -1:
                    fp += 1
                elif gt_ig[m]:
                    ignored += 1
                else:
                    tp += 1
                    matched[m] = True
            gt_non_crowd += sum(1 for x in gt_ig if x == 0)
            fn += sum(1 for g_i in range(len(gt_s)) if gt_ig[g_i] == 0 and not matched[g_i])
    return {"tp": tp, "fp": fp, "fn": fn, "ignored_crowd_matches": ignored, "gt_non_crowd": gt_non_crowd,
            "detections_considered": considered, "images": len(gts)}
