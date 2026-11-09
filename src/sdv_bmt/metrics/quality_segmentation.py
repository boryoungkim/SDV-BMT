"""Canonical semantic-segmentation task quality from the run bundle (Metric Set v1.1, ``cityscapes-semseg-miou/1``).

Bundle-only, like the detection evaluator: the ground truth comes from the bundle's frozen quality reference
(``quality_reference.jsonl``), each row re-verified against the item's ``annotation_digest``; the predicted label
maps come from the completed requests' task outputs in ``outputs.bin``, each re-verified against its
``output_ref`` digest.  Both are ``labelmap-u8-rowrle/1`` maps (the ground truth base64-encoded in the annotation).

Metric: Cityscapes 19-class mIoU.  Per class c over all evaluated images: TP = pixels with prediction c and ground
truth c; FP = pixels with prediction c and a ground truth in 0..18 other than c; FN = pixels with ground truth c and
a prediction other than c.  Pixels whose ground truth is 255 (void / not evaluated) are not counted.  IoU_c =
TP / (TP + FP + FN); mIoU = mean of IoU_c over the classes with a non-empty union (classes left out are named in the
notes).  Integer counts are exact; only the final ratios are floating point.  This is the definition of
cityscapesScripts ``evalPixelLevelSemanticLabeling`` (class IoU, ``averageScoreClasses``) and of mmseg
``IoUMetric`` with ignore index 255.

Population (conditional): the items of COMPLETED requests with a valid task output; reported with
``quality.segmentation.coverage``; a missing output is never counted as an empty prediction.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
from typing import Any, Optional

from ..contracts.ids import digest_of, sha256_hex
from ..contracts.quality import (
    CITYSCAPES_IGNORE_LABEL,
    CITYSCAPES_NUM_CLASSES,
    CITYSCAPES_SEMSEG_MIOU_V1,
    QUALITY_EVALUATORS,
)
from ..contracts.records import CompletionRecord, RequestRecord
from ..contracts.status import MetricStatus, RequestState
from ..observation.reader import RunBundle, read_output_blob
from ..contracts import labelmap

EVALUATOR_MODULE = __name__
_CLASS_MASK = [bytes(1 if v == c else 0 for v in range(256)) for c in range(CITYSCAPES_NUM_CLASSES)]
_VALID_MASK = bytes(0 if v == CITYSCAPES_IGNORE_LABEL else 1 for v in range(256))
_GT_VALUES = bytes(range(CITYSCAPES_NUM_CLASSES)) + bytes((CITYSCAPES_IGNORE_LABEL,))
_PRED_VALUES = bytes(range(CITYSCAPES_NUM_CLASSES))


@dataclass(frozen=True)
class SegmentationQuality:
    status: MetricStatus
    reason: Optional[str]
    notes: tuple[str, ...] = ()
    evaluated_items: int = 0
    miou: Optional[float] = None
    class_iou: dict[int, Optional[float]] = field(default_factory=dict)
    counts: dict[str, list[int]] = field(default_factory=dict)  # tp / fp / fn per class
    population: dict[str, Any] = field(default_factory=dict)
    annotation_check: Optional[tuple[int, int]] = None


def segmentation_protocol() -> dict[str, Any]:
    """Evaluation-identity parameters of ``cityscapes-semseg-miou/1``."""
    return {
        "evaluator": CITYSCAPES_SEMSEG_MIOU_V1,
        "implemented": True,
        "module": EVALUATOR_MODULE,
        "population": "conditional: items of COMPLETED requests with a valid task output",
        "segmentation": {
            "num_classes": CITYSCAPES_NUM_CLASSES,
            "class_ids": "Cityscapes trainId 0..18 (road ... bicycle)",
            "ignore_label": CITYSCAPES_IGNORE_LABEL,
            "iou": "TP / (TP + FP + FN) per class, counts summed over all evaluated images; ground-truth pixels 255 "
                   "are not counted",
            "mean": "arithmetic mean over the classes with a non-empty union",
            "label_codec": labelmap.CODEC_ID,
            "conformance": "definition of cityscapesScripts evalPixelLevelSemanticLabeling (averageScoreClasses), "
                           "cross-checked once against cityscapesScripts 2.2.4 (v0.1.2 numeric gate 8)",
        },
    }


def ground_truth(annotation: Any) -> tuple[int, int, bytes]:
    """(width, height, trainId bytes) of a ``cityscapes-gtfine-trainid-labelmap/1`` annotation."""
    if not isinstance(annotation, dict) or annotation.get("encoding") != labelmap.CODEC_ID:
        raise ValueError(f"annotation needs encoding {labelmap.CODEC_ID}")
    w, h, data = annotation.get("width"), annotation.get("height"), annotation.get("labels_b64")
    if type(w) is not int or type(h) is not int or not isinstance(data, str):
        raise ValueError("annotation needs integer width/height and labels_b64")
    try:
        blob = base64.b64decode(data.encode("ascii"), validate=True)
        item, bw, bh, labels = labelmap.decode(blob)
    except (binascii.Error, UnicodeEncodeError, labelmap.CodecError) as exc:
        raise ValueError(f"annotation label map is invalid: {exc}") from None
    if (bw, bh) != (w, h) or item != labelmap.NO_ITEM:
        raise ValueError("annotation label map header differs from the annotation")
    if labels.translate(None, _GT_VALUES):
        raise ValueError("ground truth holds values other than 0..18 and 255")
    return w, h, labels


class ConfusionCounts:
    """Exact per-class TP/FP/FN over images (bit-parallel counting on byte masks)."""

    def __init__(self) -> None:
        self.tp = [0] * CITYSCAPES_NUM_CLASSES
        self.fp = [0] * CITYSCAPES_NUM_CLASSES
        self.fn = [0] * CITYSCAPES_NUM_CLASSES
        self.pixels = 0

    def add(self, pred: bytes, gt: bytes) -> None:
        if len(pred) != len(gt):
            raise ValueError("prediction and ground truth sizes differ")
        valid = int.from_bytes(gt.translate(_VALID_MASK), "little")
        self.pixels += valid.bit_count()
        for c in range(CITYSCAPES_NUM_CLASSES):
            cb = bytes((c,))
            in_pred, in_gt = cb in pred, cb in gt
            if not in_pred and not in_gt:
                continue
            p = int.from_bytes(pred.translate(_CLASS_MASK[c]), "little") if in_pred else 0
            g = int.from_bytes(gt.translate(_CLASS_MASK[c]), "little") if in_gt else 0
            tp = (p & g).bit_count()
            self.tp[c] += tp
            self.fp[c] += (p & valid).bit_count() - tp
            self.fn[c] += g.bit_count() - tp

    def result(self) -> tuple[Optional[float], dict[int, Optional[float]]]:
        ious: dict[int, Optional[float]] = {}
        for c in range(CITYSCAPES_NUM_CLASSES):
            union = self.tp[c] + self.fp[c] + self.fn[c]
            ious[c] = self.tp[c] / union if union else None
        present = [v for v in ious.values() if v is not None]
        return (sum(present) / len(present) if present else None), ious


def segmentation_quality(bundle: RunBundle, folded: dict[int, tuple[RequestRecord, Optional[CompletionRecord]]],
                         protocol: Optional[dict[str, Any]]) -> SegmentationQuality:
    planned = len(folded)
    if bundle.quality_reference is None:
        reason = "quality_evaluator_not_declared" if protocol is None else "quality_reference_not_in_bundle"
        return SegmentationQuality(MetricStatus.NOT_COLLECTED, reason, (
            "task quality is computed only from a ground truth frozen in the bundle (manifest v5); this bundle has "
            "none",))
    items = bundle.input_items
    rows = bundle.quality_reference
    by_index = {item.item_index: item for item in items}
    verified = sum(1 for row in rows if (item := by_index.get(row.item_index)) is not None
                   and row.source_item_id == item.source_item_id and row.annotation_digest == item.annotation_digest
                   and digest_of(row.annotation) == row.annotation_digest)
    check = (verified, len(items))
    if verified != len(items) or len(rows) != len(items) or len({r.item_index for r in rows}) != len(rows):
        return SegmentationQuality(MetricStatus.INVALID, "quality_reference_integrity", (
            f"{verified} of {len(items)} input items have a frozen ground truth matching their annotation_digest",),
            annotation_check=check)
    if protocol is None or protocol.get("evaluator") != CITYSCAPES_SEMSEG_MIOU_V1 or not protocol.get("implemented"):
        evaluator = None if protocol is None else protocol.get("evaluator")
        return SegmentationQuality(MetricStatus.NOT_APPLICABLE, f"quality_evaluator_is:{evaluator}",
                                   annotation_check=check)
    declared = bundle.run_spec["resolved_plan"]["workload"].get("quality_evaluator")
    if declared != bundle.manifest["quality_reference"]["evaluator"]:
        return SegmentationQuality(MetricStatus.INVALID, "quality_evaluator_mismatch", (
            f"run_spec declares {declared!r}, the frozen reference was made for "
            f"{bundle.manifest['quality_reference']['evaluator']!r}",), annotation_check=check)
    schemas = QUALITY_EVALUATORS[CITYSCAPES_SEMSEG_MIOU_V1].output_schemas
    truth = {row.item_index: row.annotation for row in rows}
    counts = ConfusionCounts()
    seen: set[int] = set()
    for request, completion in folded.values():
        if request.state is not RequestState.COMPLETED or completion is None or completion.output_ref is None:
            continue
        if len(request.item_indices) != 1:
            return SegmentationQuality(MetricStatus.INVALID, "batched_request_not_supported", annotation_check=check)
        item = items[request.item_indices[0]]
        if item.item_index in seen:
            return SegmentationQuality(MetricStatus.INVALID, "duplicate_item_output", (
                f"request {request.request_id}: item {item.item_index} already has an evaluated output",),
                annotation_check=check)
        ref = completion.output_ref
        blob = read_output_blob(bundle.run_dir, ref)
        try:
            gw, gh, gt = ground_truth(truth[item.item_index])
        except ValueError as exc:
            return SegmentationQuality(MetricStatus.INVALID, "quality_reference_schema", (str(exc),),
                                       annotation_check=check)
        try:
            if len(blob) != ref.length or sha256_hex(blob) != ref.sha256:
                raise ValueError("stored output does not match its output_ref")
            if labelmap.CODEC_ID not in schemas:  # pragma: no cover - declaration consistency
                raise ValueError("codec is not an output schema of the evaluator")
            out_item, w, h, pred = labelmap.decode(blob)
            if out_item != item.item_index or (w, h) != (gw, gh):
                raise ValueError("task output item index or size differs from the item's ground truth")
            if pred.translate(None, _PRED_VALUES):
                raise ValueError("task output holds a value that is not a class index")
        except (ValueError, labelmap.CodecError) as exc:
            return SegmentationQuality(MetricStatus.INVALID, "task_output_invalid", (
                f"request {request.request_id}: {exc}",), annotation_check=check)
        counts.add(pred, gt)
        seen.add(item.item_index)
    evaluated = len(seen)
    miou, ious = counts.result()
    missing = [c for c, v in ious.items() if v is None]
    notes = (f"conditional: {evaluated} of {planned} planned requests have an evaluated task output; interpret "
             "with quality.segmentation.coverage",
             f"ground truth frozen in the bundle ({bundle.manifest['quality_reference']['file']}), evaluator "
             f"{CITYSCAPES_SEMSEG_MIOU_V1}") + ((f"classes with an empty union left out of the mean: {missing}",)
                                                if missing else ())
    population = {"images": evaluated, "pixels_evaluated": counts.pixels,
                  "classes_in_mean": CITYSCAPES_NUM_CLASSES - len(missing)}
    if evaluated == 0 or miou is None:
        return SegmentationQuality(MetricStatus.EMPTY, "no_evaluated_items", notes, evaluated, population=population,
                                   annotation_check=check)
    return SegmentationQuality(MetricStatus.VALID, None, notes, evaluated, miou, ious,
                               {"tp": counts.tp, "fp": counts.fp, "fn": counts.fn}, population, check)
