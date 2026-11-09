"""DeepLabV3+ Cityscapes semantic segmentation workload (``implementation: deeplabv3plus_cityscapes``), v0.1.2.

A real task workload: an mmsegmentation Cityscapes-trained DeepLabV3+ (ResNetV1c-D8 backbone R-18, R-50 or R-101),
exported offline to ONNX (artifact, SHA-256 pinned by ``artifact_sha256``) and converted to a cuDNN plan, on the
Cityscapes val images.  The Workload owns the canonical input semantics, the postprocess and the codec; the DUT
adapter executes the model, including the final bilinear resize to the input size and the argmax over the 19
classes, and returns the uint8 label map.  Task quality (mIoU) is computed offline from the recorded outputs and the
ground truth frozen in the bundle; it is not part of this class.

* Payload: binary PPM (``P6``, maxval 255) with the RGB pixels of the decoded PNG (decoded once, offline); it must
  be exactly ``options.input_width`` x ``options.input_height`` (2048 x 1024 for Cityscapes; whole-image inference,
  no resize, as the checkpoints' ``test_cfg.mode='whole'`` with ``Resize(scale=(2048, 1024), keep_ratio=True)``).
* Canonical preprocessing ``mmseg-cityscapes-normalize-rgb/1``: per channel c (R, G, B),
  ``float32(float32(v) - float32(MEAN[c])) / float32(STD[c])`` rounded to float32 at each step (the float32
  arithmetic of mmseg's data preprocessor), CHW, RGB order, little-endian float32 -> 12 * W * H bytes.
* Raw output: W x H uint8 class indices (the device-side argmax; ties resolved to the lowest index).
* Postprocess ``labelmap-u8-check/1``: every value must be a class index (< 19).  Codec
  ``labelmap-u8-rowrle/1`` (``labelmap.py``).  The native twin is ``native/src/common/labelmap_output.hpp``.
"""

from __future__ import annotations

import struct
import sys
from typing import Any

from ..contracts.protocols import WorkloadError
from ..contracts.records import InputItem, WorkloadInputContract
from ..contracts.specs import ComponentIdentity, WorkloadSpec
from ..contracts.ids import sha256_hex
from ..contracts import labelmap
from .yolox import parse_ppm

NUM_CLASSES = 19
MEAN = (123.675, 116.28, 103.53)  # R, G, B (checkpoint img_norm_cfg, to_rgb=True)
STD = (58.395, 57.12, 57.375)
OUTPUT_SCHEMA = "sdv-labelmap-u8/1"
# Cityscapes trainId order (cityscapesscripts labels.py, trainId 0..18)
CITYSCAPES_CLASSES = ("road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
                      "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle",
                      "bicycle")


def _f32(v: float) -> float:
    return struct.unpack("<f", struct.pack("<f", v))[0]


def normalization_tables() -> list[tuple[bytes, bytes, bytes, bytes]]:
    """Per channel, four 256-entry translation tables: byte k of the float32 normalized value of each pixel value.

    float32(a) - float32(b) of small integers and float32 constants is exact in double before the single rounding,
    and double rounding of a quotient of float32 values is correctly rounded (53 >= 2*24 + 2), so this equals the
    float32 arithmetic of the reference implementation."""
    tables = []
    for mean, std in zip(MEAN, STD):
        m, s = _f32(mean), _f32(std)
        values = [struct.pack("<f", _f32(_f32(float(v) - m) / s)) for v in range(256)]
        tables.append(tuple(bytes(b[k] for b in values) for k in range(4)))
    return tables


_TABLES = normalization_tables()


def normalize_rgb(rgb: bytes, width: int, height: int) -> bytes:
    """Interleaved RGB8 pixels -> the canonical CHW float32 tensor bytes (``mmseg-cityscapes-normalize-rgb/1``)."""
    n = width * height
    if len(rgb) != 3 * n:
        raise WorkloadError(f"RGB data must be {3 * n} bytes")
    out = bytearray(4 * 3 * n)
    for c in range(3):
        plane = rgb[c::3]
        view = memoryview(out)[4 * n * c:4 * n * (c + 1)]
        for k in range(4):
            view[k::4] = plane.translate(_TABLES[c][k])
    return bytes(out)


class DeepLabCityscapesWorkload:
    IMPLEMENTATION = "deeplabv3plus_cityscapes"
    PREPROCESS = ComponentIdentity("mmseg-cityscapes-normalize-rgb", "1")
    POSTPROCESS = ComponentIdentity("labelmap-u8-check", "1")
    CODEC = ComponentIdentity("labelmap-u8-rowrle", "1")

    def __init__(self, spec: WorkloadSpec) -> None:
        if spec.implementation != self.IMPLEMENTATION:
            raise ValueError(f"workload implementation must be {self.IMPLEMENTATION!r}")
        for declared, actual, what in ((spec.preprocess, self.PREPROCESS, "preprocess"),
                                       (spec.postprocess, self.POSTPROCESS, "postprocess"),
                                       (spec.codec, self.CODEC, "codec")):
            if declared != actual:
                raise ValueError(f"workload.{what} must be {actual} for {self.IMPLEMENTATION}")
        if spec.artifact_sha256 is None or "device_program" not in spec.options:
            raise ValueError("deeplabv3plus_cityscapes needs options.device_program and the model artifact_sha256")
        if sys.byteorder != "little":
            raise ValueError("deeplabv3plus_cityscapes produces little-endian float32 input")
        w, h, classes = (spec.options.get(k) for k in ("input_width", "input_height", "num_classes"))
        if not (type(w) is int and type(h) is int and 0 < w <= 0xFFFF and 0 < h <= 0xFFFF and w % 8 == 0
                and h % 8 == 0):
            raise ValueError("deeplabv3plus_cityscapes needs options.input_width/input_height, multiples of 8")
        if classes != NUM_CLASSES:
            raise ValueError(f"options.num_classes must be {NUM_CLASSES}")
        self._w, self._h = w, h
        self._input_bytes, self._raw_bytes = 12 * w * h, w * h
        program = spec.options["device_program"]
        if not isinstance(program, dict) or (program.get("input_bytes"), program.get("task_output_bytes")) != (
                self._input_bytes, self._raw_bytes):
            raise ValueError(f"options.device_program must declare input_bytes {self._input_bytes} and "
                             f"task_output_bytes {self._raw_bytes} for a {w}x{h} input")
        self._task = spec.task
        self._invalid = bytes(range(NUM_CLASSES))
        self._contract = WorkloadInputContract(
            schema_version=1,
            contract_id=f"cityscapes-rgb-norm-chw{w}x{h}-f32",
            contract_version="1",
            task=spec.task,
            preprocess_semantics_id=f"{self.PREPROCESS.id}/{self.PREPROCESS.version}",
            payload_format="ppm-p6 rgb8 (decoded image)",
            logical_input=(("dtype", "float32"), ("layout", "chw"), ("shape", f"3x{h}x{w}"),
                           ("channel_order", "rgb"),
                           ("value", "float32((float32(v) - float32(mean_c)) / float32(std_c)), mean "
                                     f"{list(MEAN)}, std {list(STD)}"),
                           ("resize", f"none: the payload must be {w}x{h}"), ("byte_order", "little")),
        )

    def input_contract(self) -> WorkloadInputContract:
        return self._contract

    def preprocess(self, payload: bytes, item: InputItem) -> bytes:
        if not isinstance(payload, bytes):
            raise WorkloadError("payload must be bytes")
        if sha256_hex(payload) != item.payload_sha256:
            raise WorkloadError(f"payload of item {item.item_index} does not match payload_sha256")
        w, h, rgb = parse_ppm(payload)
        if (w, h) != (self._w, self._h):
            raise WorkloadError(f"item {item.item_index}: payload is {w}x{h}, the contract needs {self._w}x{self._h}")
        return normalize_rgb(rgb, w, h)

    def postprocess(self, raw_output: object, item: InputItem) -> dict[str, Any]:
        if not isinstance(raw_output, (bytes, bytearray)) or len(raw_output) != self._raw_bytes:
            raise WorkloadError(f"raw output must be {self._raw_bytes} bytes ({self._w}x{self._h} uint8)")
        raw = bytes(raw_output)
        bad = raw.translate(None, self._invalid)
        if bad:
            raise WorkloadError(f"label value {bad[0]} is not a class index")
        return {"schema": OUTPUT_SCHEMA, "item_index": item.item_index, "width": self._w, "height": self._h,
                "labels": raw}

    def native_output_pipeline(self) -> dict[str, Any]:
        """The native measurement plane's twin of postprocess + encode (``native/src/common/labelmap_output.hpp``)."""
        return {"kind": "labelmap_rowrle_v1", "task": self._task,
                "postprocess": f"{self.POSTPROCESS.id}/{self.POSTPROCESS.version}",
                "codec": f"{self.CODEC.id}/{self.CODEC.version}",
                "implementation": "native:native/src/common/labelmap_output.hpp",
                "width": self._w, "height": self._h, "num_classes": NUM_CLASSES}

    def encode(self, task_output: object) -> bytes:
        if not isinstance(task_output, dict) or task_output.get("schema") != OUTPUT_SCHEMA:
            raise WorkloadError("encode() expects a deeplabv3plus_cityscapes task output")
        return labelmap.encode(task_output["labels"], task_output["width"], task_output["height"],
                               task_output["item_index"])
