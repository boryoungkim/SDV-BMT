"""Canonical object-detection task quality from the run bundle (Metric Set v1, ``coco-detection-bbox/1``).

Bundle-only: the ground truth comes from the bundle's frozen quality reference
(``quality_reference.jsonl``, manifest v5), each row re-verified against the item's
``annotation_digest`` in ``input_items.jsonl``; detections come from the completed requests' task
outputs in ``outputs.bin``, each re-verified against its ``output_ref`` digest.  No dataset index
or annotation file outside the bundle is read.

Population (conditional): the items of COMPLETED requests with a valid task output.  The result is
reported with ``quality.detection.coverage`` = evaluated items / planned requests and is never
presented as planned-population quality; there is no missing-output-as-empty variant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..contracts.ids import digest_of, loads_strict_json, sha256_hex
from ..contracts.quality import (
    COCO_2017_CATEGORY_IDS,
    COCO_DETECTION_BBOX_V1,
    DEFAULT_DETECTION_OPERATING_POINT,
    QUALITY_EVALUATORS,
)
from ..contracts.records import CompletionRecord, InputItem, RequestRecord
from ..contracts.specs import DetectionOperatingPoint
from ..contracts.status import MetricStatus, RequestState
from ..observation.reader import RunBundle, read_output_blob
from . import quality_coco

MAX_DETECTIONS_PER_IMAGE = 100
EVALUATOR_MODULE = __name__


@dataclass(frozen=True)
class DetectionQuality:
    """Outcome of the detection quality evaluation of one bundle (consumed by metric_set_v1)."""

    status: MetricStatus  # valid, or the status every quality metric takes
    reason: Optional[str]
    notes: tuple[str, ...] = ()
    evaluated_items: int = 0
    stats: dict[str, float] = field(default_factory=dict)  # COCO statistics, -1.0 = undefined
    counts: dict[str, int] = field(default_factory=dict)  # operating-point counts
    population: dict[str, Any] = field(default_factory=dict)
    # (verified rows, input items) of the frozen reference; None when the bundle has no reference
    annotation_check: Optional[tuple[int, int]] = None


def quality_protocol(bundle: RunBundle, operating_point: Optional[DetectionOperatingPoint]) -> Optional[dict[str, Any]]:
    """Evaluation-identity parameters of the declared quality evaluator (None if none is declared)."""
    reference = bundle.manifest.get("quality_reference")
    evaluator = (reference or {}).get("evaluator") or bundle.run_spec["resolved_plan"]["workload"].get(
        "quality_evaluator")
    if evaluator is None:
        return None
    if evaluator != COCO_DETECTION_BBOX_V1:
        return {"evaluator": evaluator, "implemented": False}
    point = operating_point or DEFAULT_DETECTION_OPERATING_POINT
    return {
        "evaluator": evaluator,
        "implemented": True,
        "module": EVALUATOR_MODULE,
        "population": "conditional: items of COMPLETED requests with a valid task output",
        "coco": {
            "iou_type": "bbox",
            "iou_thresholds": list(quality_coco.IOU_THRS),
            "recall_thresholds": "101 points, numpy.linspace(0.0, 1.0, 101) float64",
            "max_detections": list(quality_coco.MAX_DETS),
            "area_ranges": {name: list(rng) for name, rng in zip(quality_coco.AREA_RANGE_NAMES,
                                                                  quality_coco.AREA_RANGES)},
            "category_ids": list(COCO_2017_CATEGORY_IDS),
            "crowd": "iscrowd ground truth is an ignore region; its IoU is intersection / detection area",
            "score_ties": "stable: detections keep the task output order",
            "conformance": "port of tools/stage2b_g_real/coco_eval.py, cross-checked against pycocotools 2.0.11",
        },
        "operating_point": {
            "score_threshold": point.score_threshold,
            "iou_threshold": point.iou_threshold,
            "class_matching": "same_category",
            "max_detections_per_image": MAX_DETECTIONS_PER_IMAGE,
            "matching": "greedy per image and category in descending score order (stable): best IoU >= "
                        "iou_threshold among ground truth not yet matched, non-crowd first (COCO evaluateImg)",
            "duplicate_policy": "a non-crowd ground-truth box matches at most one detection; a further detection "
                                "on it is a false positive",
            "crowd_policy": "a detection matched only to a crowd region is ignored (neither TP nor FP); crowd "
                            "boxes never count as false negatives",
            "true_negatives": "not defined (no finite background-negative population)",
        },
    }


def _gt_objects(annotation: Any) -> tuple[int, list[dict[str, Any]]]:
    if not isinstance(annotation, dict) or type(annotation.get("image_id")) is not int:
        raise ValueError("annotation needs an integer image_id")
    objects = annotation.get("objects")
    if not isinstance(objects, list):
        raise ValueError("annotation needs an objects list")
    out = []
    for o in objects:
        bbox = o.get("bbox") if isinstance(o, dict) else None
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(type(v) in (int, float) for v in bbox)
                and type(o.get("id")) is int and type(o.get("category_id")) is int
                and type(o.get("area")) in (int, float) and o.get("iscrowd") in (0, 1)):
            raise ValueError("annotation object needs id, category_id, bbox[4], area and iscrowd 0/1")
        out.append({"id": o["id"], "category_id": o["category_id"], "bbox": bbox, "area": o["area"],
                    "iscrowd": o["iscrowd"]})
    return annotation["image_id"], out


def _detections(output: Any, schemas: tuple[str, ...], image_id: int, item: InputItem) -> list[dict[str, Any]]:
    if not isinstance(output, dict) or output.get("schema") not in schemas:
        raise ValueError(f"task output schema is not one of {list(schemas)}")
    if output.get("image_id") != image_id or output.get("item_index") != item.item_index:
        raise ValueError("task output image_id/item_index differ from the item's ground truth")
    detections = output.get("detections")
    if not isinstance(detections, list):
        raise ValueError("task output needs a detections list")
    out = []
    for d in detections:
        box = d.get("bbox_mpx") if isinstance(d, dict) else None
        if not (isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box)
                and type(d.get("category_id")) is int and type(d.get("score_ppm")) is int):
            raise ValueError("detection needs integer bbox_mpx[4], category_id and score_ppm")
        out.append({"category_id": d["category_id"], "bbox": [v / 1000.0 for v in box], "score": d["score_ppm"] / 1e6})
    return out


def detection_quality(bundle: RunBundle, folded: dict[int, tuple[RequestRecord, Optional[CompletionRecord]]],
                      protocol: Optional[dict[str, Any]]) -> DetectionQuality:
    planned = len(folded)
    if bundle.quality_reference is None:
        reason = ("quality_evaluator_not_declared" if protocol is None
                  else "quality_reference_not_in_bundle")
        return DetectionQuality(MetricStatus.NOT_COLLECTED, reason, (
            "task quality is computed only from a ground truth frozen in the bundle (manifest v5); "
            "this bundle has none",))
    items = bundle.input_items
    rows = bundle.quality_reference
    verified = 0
    by_index = {item.item_index: item for item in items}
    for row in rows:
        item = by_index.get(row.item_index)
        if (item is not None and row.source_item_id == item.source_item_id
                and row.annotation_digest == item.annotation_digest and digest_of(row.annotation) == row.annotation_digest):
            verified += 1
    check = (verified, len(items))
    if verified != len(items) or len(rows) != len(items) or len({r.item_index for r in rows}) != len(rows):
        return DetectionQuality(MetricStatus.INVALID, "quality_reference_integrity", (
            f"{verified} of {len(items)} input items have a frozen ground truth matching their annotation_digest",),
            annotation_check=check)
    if protocol is None or not protocol.get("implemented"):
        evaluator = None if protocol is None else protocol.get("evaluator")
        return DetectionQuality(MetricStatus.NOT_SUPPORTED, f"quality_evaluator_not_implemented:{evaluator}",
                                annotation_check=check)
    declared = bundle.run_spec["resolved_plan"]["workload"].get("quality_evaluator")
    if declared != bundle.manifest["quality_reference"]["evaluator"]:
        return DetectionQuality(MetricStatus.INVALID, "quality_evaluator_mismatch", (
            f"run_spec declares {declared!r}, the frozen reference was made for "
            f"{bundle.manifest['quality_reference']['evaluator']!r}",), annotation_check=check)
    declaration = QUALITY_EVALUATORS[COCO_DETECTION_BBOX_V1]
    try:
        truth = {row.item_index: _gt_objects(row.annotation) for row in rows}
    except ValueError as exc:
        return DetectionQuality(MetricStatus.INVALID, "quality_reference_schema", (str(exc),), annotation_check=check)

    gts: dict[int, list[dict[str, Any]]] = {}
    dets: dict[int, list[dict[str, Any]]] = {}
    for request, completion in folded.values():
        if request.state is not RequestState.COMPLETED or completion is None or completion.output_ref is None:
            continue
        if len(request.item_indices) != 1:
            return DetectionQuality(MetricStatus.INVALID, "batched_request_not_supported", annotation_check=check)
        item = items[request.item_indices[0]]
        image_id, objects = truth[item.item_index]
        if image_id in gts:
            return DetectionQuality(MetricStatus.INVALID, "duplicate_item_output", (
                f"request {request.request_id}: item {item.item_index} already has an evaluated output",),
                annotation_check=check)
        ref = completion.output_ref
        blob = read_output_blob(bundle.run_dir, ref)
        try:
            if len(blob) != ref.length or sha256_hex(blob) != ref.sha256:
                raise ValueError("stored output does not match its output_ref")
            detections = _detections(loads_strict_json(blob.decode("utf-8")), declaration.output_schemas,
                                     image_id, item)
        except (ValueError, UnicodeDecodeError) as exc:
            return DetectionQuality(MetricStatus.INVALID, "task_output_invalid", (
                f"request {request.request_id}: {exc}",), annotation_check=check)
        gts[image_id] = objects
        dets[image_id] = detections
    evaluated = len(gts)
    population = {"images": evaluated, "gt_objects": sum(len(v) for v in gts.values()),
                  "gt_crowd": sum(1 for v in gts.values() for g in v if g["iscrowd"]),
                  "detections": sum(len(v) for v in dets.values())}
    notes = (f"conditional: {evaluated} of {planned} planned requests have an evaluated task output; "
             "interpret with quality.detection.coverage",
             f"ground truth frozen in the bundle ({bundle.manifest['quality_reference']['file']}), "
             f"evaluator {COCO_DETECTION_BBOX_V1}")
    if evaluated == 0:
        return DetectionQuality(MetricStatus.EMPTY, "no_evaluated_items", notes, 0, population=population,
                                annotation_check=check)
    cat_ids = list(protocol["coco"]["category_ids"])
    coco = quality_coco.evaluate(gts, dets, cat_ids)
    point = protocol["operating_point"]
    counts = quality_coco.threshold_counts(gts, dets, cat_ids, point["score_threshold"], point["iou_threshold"],
                                           point["max_detections_per_image"])
    return DetectionQuality(MetricStatus.VALID, None, notes, evaluated, dict(coco["stats"]), counts,
                            population, check)
