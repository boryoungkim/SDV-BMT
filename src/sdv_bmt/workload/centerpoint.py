"""CenterPoint-Pillars nuScenes 3D detection workload (``implementation: centerpoint_nuscenes``).

A real task workload: the mmdetection3d CenterPoint-Pillars nuScenes checkpoint (pillar 0.2 m, circle NMS), exported
offline to ONNX (artifact, SHA-256 pinned by ``artifact_sha256``) and converted to a cuDNN plan, on nuScenes LiDAR
keyframes.  The Workload owns the canonical input semantics, the postprocess and the codec; the DUT adapter executes the
model up to the raw head outputs.  Task quality (nuScenes mAP / NDS) is computed offline from the recorded outputs and the
ground truth frozen in the bundle; it is not part of this class.

* Payload: the pillar tensor file built offline by ``tools/centerpoint/build_nuscenes_input.py`` (10-sweep point cloud,
  hard voxelization): 40-byte header ``SDVPIL01`` (pillars, max pillars 40000, max points 20, features 5, merged
  points, 3 x 0) + voxels float32 [40000, 20, 5] + points per pillar int32 [40000] + pillar (y, x) int32 [40000, 2].
* Canonical preprocessing ``pillar-payload-body/1``: the header is checked and removed; the body (16,480,000 bytes) is
  the device input.  The point-cloud merge and the voxelization are offline (outside the measured path).
* Raw output: the CenterHead outputs float32 [70, 128, 128], task by task (car | truck, construction_vehicle |
  bus, trailer | barrier | motorcycle, bicycle | pedestrian, traffic_cone): reg 2, height 1, dim 3, rot 2, vel 2,
  heatmap logits (classes).
* Postprocess ``centerpoint-decode-circlenms/1`` (mmdetection3d CenterHead.predict_by_feat with
  CenterPointBBoxCoder, circle NMS) in IEEE double, the same order in the native twin: per task, heatmap sigmoid; per
  class the 500 best cells, then the 500 best of the task (score descending; ties: lower cell / position first);
  x = ((col + reg0) * 4) * 0.2 + (-51.2), y likewise, z = height, size = exp(dim), yaw = atan2(rot0, rot1),
  velocity = vel; kept when score > 0.1 and (x, y, z) inside [-61.2, -61.2, -10, 61.2, 61.2, 10]; circle NMS in score
  order (a candidate is removed when (dx^2 + dy^2) <= r_task, r = 4, 12, 10, 1, 0.85, 0.175), at most 83 per task;
  tasks concatenated in order; z moved to the bottom centre (z - dz / 2); label = task class offset + class.
  Errors (WorkloadError, the same in the native twin): a NaN heatmap logit of a class ("NaN heatmap logit"), checked class
  by class in task order; an output detection with a value whose float32 conversion is not finite ("non-finite
  detection value"), checked in output order.  The first error in that order is raised.
* Codec ``centerpoint-det3d-f32/1`` (``contracts/det3d.py``).  The native twin is
  ``native/src/common/centerpoint_output.hpp``.
"""

from __future__ import annotations

import array
import math
import struct
import sys
from typing import Any

from ..contracts import det3d
from ..contracts.ids import sha256_hex
from ..contracts.protocols import WorkloadError
from ..contracts.records import InputItem, WorkloadInputContract
from ..contracts.specs import ComponentIdentity, WorkloadSpec

TASKS = (("car",), ("truck", "construction_vehicle"), ("bus", "trailer"), ("barrier",), ("motorcycle", "bicycle"),
         ("pedestrian", "traffic_cone"))
CLASS_NAMES = tuple(c for t in TASKS for c in t)  # label order of the head
GRID = 128
REG_CHANNELS = 10
HEAD_CHANNELS = sum(REG_CHANNELS + len(t) for t in TASKS)  # 70
MAX_NUM = 500
SCORE_THRESHOLD = 0.1
POST_CENTER_RANGE = (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0)
MIN_RADIUS = (4.0, 12.0, 10.0, 1.0, 0.85, 0.175)
POST_MAX_SIZE = 83
OUT_SIZE_FACTOR, VOXEL_XY, PC_MIN_XY = 4.0, 0.2, -51.2
PILLARS, POINTS, FEATURES = 40000, 20, 5
HEADER = struct.Struct("<8s8I")
MAGIC = b"SDVPIL01"
INPUT_BYTES = PILLARS * POINTS * FEATURES * 4 + PILLARS * 4 + PILLARS * 8  # 16,480,000
RAW_BYTES = HEAD_CHANNELS * GRID * GRID * 4  # 4,587,520
OUTPUT_SCHEMA = det3d.CODEC_ID


def _sigmoid(v: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-v))
    except OverflowError:  # exp(-v) beyond double: the native twin gets inf and the same 0.0
        return 0.0


def _exp(v: float) -> float:
    try:
        return math.exp(v)
    except OverflowError:  # beyond double: the native twin gets inf (refused below as non-finite)
        return math.inf


def _f32_finite(v: float) -> bool:
    """True when ``v`` converts to a finite float32 (the C cast, as the codec and the native twin do)."""
    if not math.isfinite(v):
        return False
    try:
        struct.pack("<f", v)
    except OverflowError:
        return False
    return True


def decode_heads(raw: bytes) -> list[tuple[tuple[float, ...], float, int]]:
    """The reference postprocess: raw head outputs -> detections [(box9, score, label)] in double."""
    a = array.array("f")
    a.frombytes(raw)
    if sys.byteorder != "little":  # pragma: no cover - the workload refuses big-endian hosts
        a.byteswap()
    hw = GRID * GRID
    out: list[tuple[tuple[float, ...], float, int]] = []
    base, label_off = 0, 0
    for t, names in enumerate(TASKS):
        n = len(names)
        heat = base + REG_CHANNELS
        cands = []
        for k in range(n):
            ch = (heat + k) * hw
            if any(math.isnan(a[ch + i]) for i in range(hw)):
                raise WorkloadError("NaN heatmap logit")
            scored = sorted(((_sigmoid(a[ch + i]), i) for i in range(hw)), key=lambda e: (-e[0], e[1]))[:MAX_NUM]
            cands += [(s, k * MAX_NUM + r, k, i) for r, (s, i) in enumerate(scored)]
        cands.sort(key=lambda e: (-e[0], e[1]))
        kept = []
        for s, _pos, k, i in cands[:MAX_NUM]:
            row, col = divmod(i, GRID)
            x = ((col + a[base * hw + i]) * OUT_SIZE_FACTOR) * VOXEL_XY + PC_MIN_XY
            y = ((row + a[(base + 1) * hw + i]) * OUT_SIZE_FACTOR) * VOXEL_XY + PC_MIN_XY
            z = a[(base + 2) * hw + i]
            if not (s > SCORE_THRESHOLD and POST_CENTER_RANGE[0] <= x <= POST_CENTER_RANGE[3]
                    and POST_CENTER_RANGE[1] <= y <= POST_CENTER_RANGE[4]
                    and POST_CENTER_RANGE[2] <= z <= POST_CENTER_RANGE[5]):
                continue
            dims = tuple(_exp(a[(base + 3 + j) * hw + i]) for j in range(3))
            yaw = math.atan2(a[(base + 6) * hw + i], a[(base + 7) * hw + i])
            vel = (a[(base + 8) * hw + i], a[(base + 9) * hw + i])
            kept.append((x, y, z, dims, yaw, vel, s, k))
        suppressed = [False] * len(kept)
        keep = []
        for p in range(len(kept)):
            if suppressed[p]:
                continue
            keep.append(p)
            for q in range(p + 1, len(kept)):
                if not suppressed[q]:
                    dx, dy = kept[p][0] - kept[q][0], kept[p][1] - kept[q][1]
                    if dx * dx + dy * dy <= MIN_RADIUS[t]:
                        suppressed[q] = True
        for p in keep[:POST_MAX_SIZE]:
            x, y, z, dims, yaw, vel, s, k = kept[p]
            box = (x, y, z - dims[2] * 0.5, dims[0], dims[1], dims[2], yaw, vel[0], vel[1])
            if not all(_f32_finite(v) for v in box):
                raise WorkloadError("non-finite detection value")
            out.append((box, s, label_off + k))
        base += REG_CHANNELS + n
        label_off += n
    return out


class CenterPointNuScenesWorkload:
    IMPLEMENTATION = "centerpoint_nuscenes"
    PREPROCESS = ComponentIdentity("pillar-payload-body", "1")
    POSTPROCESS = ComponentIdentity("centerpoint-decode-circlenms", "1")
    CODEC = ComponentIdentity("centerpoint-det3d-f32", "1")

    def __init__(self, spec: WorkloadSpec) -> None:
        if spec.implementation != self.IMPLEMENTATION:
            raise ValueError(f"workload implementation must be {self.IMPLEMENTATION!r}")
        for declared, actual, what in ((spec.preprocess, self.PREPROCESS, "preprocess"),
                                       (spec.postprocess, self.POSTPROCESS, "postprocess"),
                                       (spec.codec, self.CODEC, "codec")):
            if declared != actual:
                raise ValueError(f"workload.{what} must be {actual} for {self.IMPLEMENTATION}")
        if spec.artifact_sha256 is None or "device_program" not in spec.options:
            raise ValueError("centerpoint_nuscenes needs options.device_program and the model artifact_sha256")
        if sys.byteorder != "little":
            raise ValueError("centerpoint_nuscenes needs a little-endian host")
        program = spec.options["device_program"]
        if not isinstance(program, dict) or (program.get("input_bytes"), program.get("task_output_bytes")) != (
                INPUT_BYTES, RAW_BYTES):
            raise ValueError(f"options.device_program must declare input_bytes {INPUT_BYTES} and "
                             f"task_output_bytes {RAW_BYTES}")
        self._task = spec.task
        self._contract = WorkloadInputContract(
            schema_version=1,
            contract_id=f"nuscenes-pillar02-p{PILLARS}",
            contract_version="1",
            task=spec.task,
            preprocess_semantics_id=f"{self.PREPROCESS.id}/{self.PREPROCESS.version}",
            payload_format="SDVPIL01 pillar tensor (offline 10-sweep merge and hard voxelization)",
            logical_input=(("voxels", f"float32 [{PILLARS}, {POINTS}, {FEATURES}] (x, y, z, intensity, time lag), "
                                      "zero padded"),
                           ("num_points", f"int32 [{PILLARS}], 0 for padding"),
                           ("coors", f"int32 [{PILLARS}, 2] = (y, x) on the 512 x 512 grid"),
                           ("voxelization", "voxel 0.2 x 0.2 x 8 m, range [-51.2, -51.2, -5, 51.2, 51.2, 3], 20 points "
                                            "per pillar, pillars in order of their first point"),
                           ("byte_order", "little")),
        )

    def input_contract(self) -> WorkloadInputContract:
        return self._contract

    def preprocess(self, payload: bytes, item: InputItem) -> bytes:
        if not isinstance(payload, bytes):
            raise WorkloadError("payload must be bytes")
        if sha256_hex(payload) != item.payload_sha256:
            raise WorkloadError(f"payload of item {item.item_index} does not match payload_sha256")
        if len(payload) != HEADER.size + INPUT_BYTES:
            raise WorkloadError(f"item {item.item_index}: payload is {len(payload)} bytes, expected "
                                f"{HEADER.size + INPUT_BYTES}")
        magic, pillars, max_pillars, max_points, features, _points, *_ = HEADER.unpack_from(payload)
        if magic != MAGIC or (max_pillars, max_points, features) != (PILLARS, POINTS, FEATURES) or pillars > PILLARS:
            raise WorkloadError(f"item {item.item_index}: not a {PILLARS}x{POINTS}x{FEATURES} pillar payload")
        return payload[HEADER.size:]

    def postprocess(self, raw_output: object, item: InputItem) -> dict[str, Any]:
        if not isinstance(raw_output, (bytes, bytearray)) or len(raw_output) != RAW_BYTES:
            raise WorkloadError(f"raw output must be {RAW_BYTES} bytes (float32 [{HEAD_CHANNELS}, {GRID}, {GRID}])")
        return {"schema": OUTPUT_SCHEMA, "item_index": item.item_index, "detections": decode_heads(bytes(raw_output))}

    def native_output_pipeline(self) -> dict[str, Any]:
        """The native measurement plane's twin of postprocess + encode (``native/src/common/centerpoint_output.hpp``)."""
        return {"kind": "centerpoint_det3d_v1", "task": self._task,
                "postprocess": f"{self.POSTPROCESS.id}/{self.POSTPROCESS.version}",
                "codec": f"{self.CODEC.id}/{self.CODEC.version}",
                "implementation": "native:native/src/common/centerpoint_output.hpp",
                "task_classes": [len(t) for t in TASKS], "grid": GRID, "max_num": MAX_NUM,
                "score_threshold_x1e6": int(SCORE_THRESHOLD * 1e6), "post_max_size": POST_MAX_SIZE}

    def encode(self, task_output: object) -> bytes:
        if not isinstance(task_output, dict) or task_output.get("schema") != OUTPUT_SCHEMA:
            raise WorkloadError("encode() expects a centerpoint_nuscenes task output")
        return det3d.encode(task_output["item_index"], task_output["detections"])
