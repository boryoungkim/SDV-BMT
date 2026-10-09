"""YOLOX COCO 2D object detection workload (``implementation: yolox_coco``), Stage 2b-G real-model baseline.

A real task workload: the official COCO-trained YOLOX-S model (ONNX artifact, SHA-256 pinned by
``artifact_sha256``) on COCO val2017 images.  The Workload owns the canonical input semantics, the
postprocess and the codec; the DUT adapter executes the model (NVIDIA: cuDNN plan converted offline from the
ONNX file).  Task quality (COCO bbox AP) is computed offline from the recorded task outputs and the dataset
annotations; it is not part of this class.

* Payload: binary PPM (``P6``, maxval 255) holding the RGB pixels of the decoded JPEG (decoded once, offline).
* Canonical preprocessing (``yolox-letterbox-bilinear-bgr/1``), following YOLOX ``ValTransform(legacy=False)``:
  ``r = min(640/h, 640/w)``; the image is resized to ``int(w*r) x int(h*r)`` (unchanged when that is its own
  size), placed at the top-left of a 640x640 canvas filled with 114, channels in BGR order, CHW float32 with
  the raw pixel value (no normalization) -> 4,915,200 bytes.  Resize: bilinear with half-pixel centres and
  edge clamping, float64 weights, horizontal then vertical pass, ``int(v + 0.5)`` per channel value.  This is
  OpenCV ``INTER_LINEAR``'s geometry but not its 11-bit fixed-point arithmetic (values may differ by one level).
* Raw output: 8400 x 85 float32 (the ONNX ``output`` tensor, undecoded).
* Postprocess ``yolox-decode-nms-coco/1`` and codec ``canonical-json-utf8/1``: see ``postprocess``.  The native
  twin is ``native/src/common/yolox_output.hpp``; both use IEEE double arithmetic in the same order.
"""

from __future__ import annotations

import array
import math
import struct
import sys
from typing import Any

from ..contracts.ids import canonical_json_bytes, sha256_hex
from ..contracts.protocols import WorkloadError
from ..contracts.records import InputItem, WorkloadInputContract
from ..contracts.specs import ComponentIdentity, WorkloadSpec

INPUT_SIZE = 640
PAD_VALUE = 114
NUM_ANCHORS = 8400
NUM_CHANNELS = 85
NUM_CLASSES = 80
STRIDES = (8, 16, 32)
CONF_THRESHOLD = 0.01  # YOLOX COCO evaluation default (exp.test_conf)
NMS_THRESHOLD = 0.65  # YOLOX COCO evaluation default (exp.nmsthre)
MAX_DETECTIONS = 100  # COCO maxDets
INPUT_BYTES = 3 * INPUT_SIZE * INPUT_SIZE * 4
RAW_BYTES = NUM_ANCHORS * NUM_CHANNELS * 4
OUTPUT_SCHEMA = "sdv-yolox-coco-detections/1"
# COCO category id of each contiguous class index (category ids sorted ascending, as YOLOX's COCO dataset)
COCO_CATEGORY_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
                     31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                     56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
                     85, 86, 87, 88, 89, 90)


def parse_ppm(payload: bytes) -> tuple[int, int, bytes]:
    """(width, height, RGB bytes) of a binary PPM with maxval 255 (no comments)."""
    tokens: list[bytes] = []
    pos = 0
    while len(tokens) < 4:
        while pos < len(payload) and payload[pos:pos + 1].isspace():
            pos += 1
        start = pos
        while pos < len(payload) and not payload[pos:pos + 1].isspace():
            pos += 1
        if start == pos or payload[start:start + 1] == b"#":
            raise WorkloadError("payload is not a binary PPM without comments")
        tokens.append(payload[start:pos])
    if tokens[0] != b"P6" or tokens[3] != b"255" or pos >= len(payload) or not payload[pos:pos + 1].isspace():
        raise WorkloadError("payload is not a binary PPM (P6, maxval 255)")
    try:
        width, height = int(tokens[1]), int(tokens[2])
    except ValueError:
        raise WorkloadError("PPM size is not an integer") from None
    pixels = payload[pos + 1:]
    if width <= 0 or height <= 0 or len(pixels) != width * height * 3:
        raise WorkloadError(f"PPM pixel data has {len(pixels)} bytes, expected {width}x{height}x3")
    return width, height, pixels


def resize_bilinear(rgb: bytes, w: int, h: int, nw: int, nh: int) -> bytes:
    """Bilinear resize of interleaved RGB8 (half-pixel centres, edge clamp, float64 weights)."""
    sx, sy = w / nw, h / nh
    xs = []
    for x in range(nw):
        f = (x + 0.5) * sx - 0.5
        x0 = math.floor(f)
        xs.append((min(max(x0, 0), w - 1) * 3, min(max(x0 + 1, 0), w - 1) * 3, f - x0))
    cache: dict[int, list[float]] = {}

    def hrow(y: int) -> list[float]:
        row = cache.get(y)
        if row is None:
            src = rgb[y * w * 3:(y + 1) * w * 3]
            row = []
            for i0, i1, a in xs:
                b = 1.0 - a
                row.append(src[i0] * b + src[i1] * a)
                row.append(src[i0 + 1] * b + src[i1 + 1] * a)
                row.append(src[i0 + 2] * b + src[i1 + 2] * a)
            cache[y] = row
        return row

    out = bytearray(nw * nh * 3)
    stride = nw * 3
    for y in range(nh):
        f = (y + 0.5) * sy - 0.5
        y0 = math.floor(f)
        b = f - y0
        c = 1.0 - b
        r0 = hrow(min(max(y0, 0), h - 1))
        r1 = hrow(min(max(y0 + 1, 0), h - 1))
        out[y * stride:(y + 1) * stride] = bytes(min(255, int(p * c + q * b + 0.5)) for p, q in zip(r0, r1))
    return bytes(out)


def _quantize(v: float, scale: float) -> int:
    u = v * scale + 0.5
    if not math.isfinite(u) or abs(u) > 9.0e15:
        raise WorkloadError("detection value is not finite or out of range")
    return math.floor(u)


def _exp(v: float) -> float:
    try:
        return math.exp(v)
    except OverflowError:
        return math.inf


class YoloxCocoWorkload:
    IMPLEMENTATION = "yolox_coco"
    PREPROCESS = ComponentIdentity("yolox-letterbox-bilinear-bgr", "1")
    POSTPROCESS = ComponentIdentity("yolox-decode-nms-coco", "1")
    CODEC = ComponentIdentity("canonical-json-utf8", "1")

    def __init__(self, spec: WorkloadSpec) -> None:
        if spec.implementation != self.IMPLEMENTATION:
            raise ValueError(f"workload implementation must be {self.IMPLEMENTATION!r}")
        for declared, actual, what in ((spec.preprocess, self.PREPROCESS, "preprocess"),
                                       (spec.postprocess, self.POSTPROCESS, "postprocess"),
                                       (spec.codec, self.CODEC, "codec")):
            if declared != actual:
                raise ValueError(f"workload.{what} must be {actual} for {self.IMPLEMENTATION}")
        if spec.artifact_sha256 is None or "device_program" not in spec.options:
            raise ValueError("yolox_coco needs options.device_program and the model artifact_sha256")
        if sys.byteorder != "little":
            raise ValueError("yolox_coco produces little-endian float32 input")
        self._task = spec.task
        self._contract = WorkloadInputContract(
            schema_version=1,
            contract_id="yolox-letterbox-bgr-chw640-f32",
            contract_version="1",
            task=spec.task,
            preprocess_semantics_id=f"{self.PREPROCESS.id}/{self.PREPROCESS.version}",
            payload_format="ppm-p6 rgb8 (decoded image)",
            logical_input=(("dtype", "float32"), ("layout", "chw"), ("shape", "3x640x640"),
                           ("channel_order", "bgr"), ("value", "pixel value 0..255, not normalized"),
                           ("letterbox", "r=min(640/h,640/w); int(w*r) x int(h*r) at top-left; pad 114"),
                           ("resize", "bilinear, half-pixel centres, edge clamp, float64 weights, int(v+0.5)"),
                           ("byte_order", "little")),
        )

    def input_contract(self) -> WorkloadInputContract:
        return self._contract

    @staticmethod
    def _image_meta(item: InputItem) -> dict[str, int]:
        meta = (item.source_metadata or {}).get("row_metadata") or {}
        try:
            out = {"image_id": int(meta["coco_image_id"]), "width": int(meta["width"]), "height": int(meta["height"])}
        except (KeyError, TypeError, ValueError):
            raise WorkloadError(f"item {item.item_index}: index metadata lacks coco_image_id/width/height") from None
        if out["width"] <= 0 or out["height"] <= 0:
            raise WorkloadError(f"item {item.item_index}: non-positive image size")
        return out

    def native_output_context(self, item: InputItem) -> dict[str, int]:
        """Per-item facts the native output pipeline needs (image id, original size), from the index row."""
        return self._image_meta(item)

    def preprocess(self, payload: bytes, item: InputItem) -> bytes:
        if not isinstance(payload, bytes):
            raise WorkloadError("payload must be bytes")
        if sha256_hex(payload) != item.payload_sha256:
            raise WorkloadError(f"payload of item {item.item_index} does not match payload_sha256")
        w, h, rgb = parse_ppm(payload)
        meta = self._image_meta(item)
        if (w, h) != (meta["width"], meta["height"]):
            raise WorkloadError(f"item {item.item_index}: payload is {w}x{h}, index metadata says "
                                f"{meta['width']}x{meta['height']}")
        r = min(INPUT_SIZE / h, INPUT_SIZE / w)
        nw, nh = int(w * r), int(h * r)
        if not (0 < nw <= INPUT_SIZE and 0 < nh <= INPUT_SIZE):
            raise WorkloadError(f"item {item.item_index}: letterbox size {nw}x{nh} out of range")
        pixels = rgb if (nw, nh) == (w, h) else resize_bilinear(rgb, w, h, nw, nh)
        out = array.array("f")
        side = INPUT_SIZE
        for src_channel in (2, 1, 0):  # B, G, R
            plane = bytearray(bytes([PAD_VALUE]) * (side * side))
            for y in range(nh):
                plane[y * side:y * side + nw] = pixels[y * nw * 3 + src_channel:(y + 1) * nw * 3:3]
            out.extend(memoryview(plane).tolist())
        data = out.tobytes()
        if len(data) != INPUT_BYTES:  # pragma: no cover - defensive
            raise WorkloadError("preprocessed input has the wrong size")
        return data

    def postprocess(self, raw_output: object, item: InputItem) -> dict[str, Any]:
        if not isinstance(raw_output, (bytes, bytearray)) or len(raw_output) != RAW_BYTES:
            raise WorkloadError(f"raw output must be {RAW_BYTES} bytes (8400x85 float32)")
        meta = self._image_meta(item)
        v = struct.unpack(f"<{NUM_ANCHORS * NUM_CHANNELS}f", bytes(raw_output))
        r = min(INPUT_SIZE / meta["height"], INPUT_SIZE / meta["width"])
        cand: list[tuple[float, float, float, float, float, int]] = []
        a = 0
        for stride in STRIDES:
            g = INPUT_SIZE // stride
            s = float(stride)
            for p in range(g * g):
                row = a * NUM_CHANNELS
                a += 1
                obj = v[row + 4]
                if not obj >= CONF_THRESHOLD:
                    continue
                best = v[row + 5]
                cls = 0
                for j in range(1, NUM_CLASSES):
                    x = v[row + 5 + j]
                    if x > best:
                        best = x
                        cls = j
                score = obj * best
                if not score >= CONF_THRESHOLD:
                    continue
                gx, gy = float(p % g), float(p // g)
                cx, cy = (v[row] + gx) * s, (v[row + 1] + gy) * s
                bw, bh = _exp(v[row + 2]) * s, _exp(v[row + 3]) * s
                cand.append((cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0, score, cls))
        cand.sort(key=lambda c: -c[4])  # stable: ties keep anchor order
        kept: list[tuple[float, float, float, float, float, int]] = []
        for b in cand:
            if len(kept) >= MAX_DETECTIONS:
                break
            suppressed = False
            for q in kept:
                if q[5] != b[5]:
                    continue
                iw = max(0.0, min(b[2], q[2]) - max(b[0], q[0]))
                ih = max(0.0, min(b[3], q[3]) - max(b[1], q[1]))
                inter = iw * ih
                denom = (b[2] - b[0]) * (b[3] - b[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
                if denom != 0.0 and inter / denom > NMS_THRESHOLD:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(b)
        detections = []
        for x1, y1, x2, y2, score, cls in kept:
            X1, Y1, X2, Y2 = x1 / r, y1 / r, x2 / r, y2 / r
            detections.append({"bbox_mpx": [_quantize(X1, 1000.0), _quantize(Y1, 1000.0),
                                             _quantize(X2 - X1, 1000.0), _quantize(Y2 - Y1, 1000.0)],
                               "category_id": COCO_CATEGORY_IDS[cls],
                               "score_ppm": _quantize(score, 1000000.0)})
        return {"schema": OUTPUT_SCHEMA, "image_id": meta["image_id"], "item_index": item.item_index,
                "detections": detections}

    def native_output_pipeline(self) -> dict[str, Any]:
        """The native measurement plane's twin of postprocess + encode (``native/src/common/yolox_output.hpp``).

        The parameters are repeated here so the native runner can refuse a pipeline whose constants differ
        from its own (a configured and a realized pipeline are not assumed to be identical)."""
        return {"kind": "yolox_coco_v1", "task": self._task,
                "postprocess": f"{self.POSTPROCESS.id}/{self.POSTPROCESS.version}",
                "codec": f"{self.CODEC.id}/{self.CODEC.version}",
                "implementation": "native:native/src/common/yolox_output.hpp",
                "conf_threshold_ppm": 10000, "nms_iou_threshold_ppm": 650000, "max_detections": MAX_DETECTIONS,
                "input_size": INPUT_SIZE, "num_classes": NUM_CLASSES}

    def encode(self, task_output: object) -> bytes:
        if not isinstance(task_output, dict) or task_output.get("schema") != OUTPUT_SCHEMA:
            raise WorkloadError("encode() expects a yolox_coco task output")
        return canonical_json_bytes(task_output)
