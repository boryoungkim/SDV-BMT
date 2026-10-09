"""Declared task-quality evaluators (approved design update "Model-level Metric Set v1").

A workload declares the task-quality evaluator of its task by ID (``workload.quality_evaluator``).
A run that declares one freezes the ground truth of its planned items in the bundle
(``quality_reference.jsonl``, manifest v5), and the MetricEvaluator computes ``quality.*`` from
the bundle only.  This table is the single declaration shared by plan validation (runner) and
evaluation (metrics); the evaluator's protocol parameters are part of the evaluation identity.
"""

from __future__ import annotations

from dataclasses import dataclass

from .specs import DetectionOperatingPoint

COCO_DETECTION_BBOX_V1 = "coco-detection-bbox/1"

# Category ids of the COCO 2017 instances annotations (pycocotools ``getCatIds()`` of
# instances_val2017.json, ascending): the category set the COCO evaluation runs over.
COCO_2017_CATEGORY_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
)

# Default operating point of the threshold-defined detection metrics (user decision 2026-10-08);
# an evaluation-time override replaces it and is recorded in the evaluation protocol.
DEFAULT_DETECTION_OPERATING_POINT = DetectionOperatingPoint(score_threshold=0.5, iou_threshold=0.5)


@dataclass(frozen=True)
class QualityEvaluatorDeclaration:
    evaluator_id: str
    task: str
    annotation_schema: str
    output_schemas: tuple[str, ...]  # task-output schemas the evaluator can read


QUALITY_EVALUATORS: dict[str, QualityEvaluatorDeclaration] = {
    COCO_DETECTION_BBOX_V1: QualityEvaluatorDeclaration(
        evaluator_id=COCO_DETECTION_BBOX_V1,
        task="coco-2017-object-detection",
        annotation_schema="coco-2017-instances-bbox/1",
        output_schemas=("sdv-yolox-coco-detections/1",),
    ),
}
