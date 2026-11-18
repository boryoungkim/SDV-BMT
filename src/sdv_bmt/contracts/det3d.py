"""3D detection output codec ``centerpoint-det3d-f32/1`` (CenterPoint-Pillars task output).

Binary, little-endian: header ``SDVCP3D1`` (8 bytes), item index u32, detection count u32, then per detection
9 float32 box values (x, y, z of the bottom centre, dx, dy, dz, yaw, vx, vy; LiDAR frame of the keyframe, metres,
radians, metres per second), float32 score and u32 label (class index in the head's task order).  44 bytes per
detection.  The native twin is ``native/src/common/centerpoint_output.hpp``.
"""

from __future__ import annotations

import struct

CODEC_ID = "centerpoint-det3d-f32/1"
MAGIC = b"SDVCP3D1"
HEADER = struct.Struct("<8sII")
RECORD = struct.Struct("<9ffI")


class CodecError(ValueError):
    pass


def encode(item_index: int, detections: list[tuple[tuple[float, ...], float, int]]) -> bytes:
    """detections: [(box9, score, label)]; values are rounded to float32 here (round to nearest even)."""
    out = [HEADER.pack(MAGIC, item_index, len(detections))]
    for box, score, label in detections:
        out.append(RECORD.pack(*box, score, label))
    return b"".join(out)


def decode(blob: bytes) -> tuple[int, list[tuple[tuple[float, ...], float, int]]]:
    if len(blob) < HEADER.size:
        raise CodecError("blob shorter than the header")
    magic, item, n = HEADER.unpack_from(blob)
    if magic != MAGIC:
        raise CodecError("not a centerpoint-det3d-f32/1 blob")
    if len(blob) != HEADER.size + n * RECORD.size:
        raise CodecError(f"blob of {len(blob)} bytes does not hold {n} detections")
    dets = []
    for k in range(n):
        v = RECORD.unpack_from(blob, HEADER.size + k * RECORD.size)
        dets.append((tuple(v[:9]), v[9], v[10]))
    return item, dets
