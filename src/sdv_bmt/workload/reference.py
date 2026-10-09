"""Reference accelerator conformance workload (``implementation: reference_convnet``), Stage 1R-G.

This is NOT a task model and NOT an NPU benchmark workload.  It is the smallest
deterministic workload found sufficient to exercise the real accelerator path:
meaningful input preparation, a multi-megabyte host-to-device copy, non-trivial
device execution and an output copied back.  Its arithmetic is exactly
representable, so outputs are bit-identical across runs and equal to an
independent integer CPU implementation (``tools/stage1r_g/reference_cpu.py``).

* Payload: base64 text of a 128x128 RGB uint8 image (HWC, 49,152 bytes).
* Canonical preprocessing (semantics ``sdv-ref-nearest4-pow2norm/1``): decode,
  nearest-neighbour upsample x4 to 512x512, CHW float32 with value
  ``(x - 128) / 128`` (exact in float32) -> 3,145,728 bytes.
* Device program: the CUDA C source in ``options.device_program`` (the model
  artifact, SHA-256 pinned by ``artifact_sha256``), executed by the DUT adapter.
* Raw output: 10 little-endian int64 logits (80 bytes).
* Task output ``{"schema", "logits", "top1"}``, encoded as canonical JSON.
"""

from __future__ import annotations

import array
import base64
import binascii
import struct
import sys
from typing import Any

from ..contracts.ids import canonical_json_bytes, sha256_hex
from ..contracts.protocols import WorkloadError
from ..contracts.records import InputItem, WorkloadInputContract
from ..contracts.specs import ComponentIdentity, WorkloadSpec

SIDE = 128
UPSAMPLE = 4
CHANNELS = 3
PAYLOAD_BYTES = SIDE * SIDE * CHANNELS
INPUT_BYTES = CHANNELS * (SIDE * UPSAMPLE) ** 2 * 4
NUM_CLASSES = 10
OUTPUT_SCHEMA = "sdv-ref-convnet-output/1"


class ReferenceConvNetWorkload:
    IMPLEMENTATION = "reference_convnet"
    PREPROCESS = ComponentIdentity("sdv-ref-nearest4-pow2norm", "1")
    POSTPROCESS = ComponentIdentity("sdv-ref-logits-top1", "1")
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
            raise ValueError("reference_convnet needs options.device_program and its artifact_sha256")
        if sys.byteorder != "little":
            raise ValueError("reference_convnet produces little-endian float32 input")
        self._task = spec.task
        self._contract = WorkloadInputContract(
            schema_version=1,
            contract_id="sdv-ref-rgb8-to-chw512-f32",
            contract_version="1",
            task=spec.task,
            preprocess_semantics_id=f"{self.PREPROCESS.id}/{self.PREPROCESS.version}",
            payload_format="base64-text rgb8 hwc 128x128",
            logical_input=(("dtype", "float32"), ("layout", "chw"), ("shape", "3x512x512"),
                           ("upsample", "nearest x4"), ("value", "(x-128)/128, exact in float32"),
                           ("byte_order", "little")),
        )

    def input_contract(self) -> WorkloadInputContract:
        return self._contract

    def preprocess(self, payload: bytes, item: InputItem) -> bytes:
        if not isinstance(payload, bytes):
            raise WorkloadError("payload must be bytes")
        if sha256_hex(payload) != item.payload_sha256:
            raise WorkloadError(f"payload of item {item.item_index} does not match payload_sha256")
        try:
            image = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise WorkloadError(f"item {item.item_index}: payload is not base64: {exc}") from None
        if len(image) != PAYLOAD_BYTES:
            raise WorkloadError(f"item {item.item_index}: decoded image has {len(image)} bytes, not {PAYLOAD_BYTES}")
        out = array.array("f")
        for c in range(CHANNELS):
            for y in range(SIDE):
                base = y * SIDE * CHANNELS + c
                row = array.array("f", [(image[base + x * CHANNELS] - 128) * 0.0078125
                                        for x in range(SIDE) for _ in range(UPSAMPLE)])
                for _ in range(UPSAMPLE):
                    out.extend(row)
        data = out.tobytes()
        if len(data) != INPUT_BYTES:  # pragma: no cover - defensive
            raise WorkloadError("preprocessed input has the wrong size")
        return data

    def postprocess(self, raw_output: object, item: InputItem) -> dict[str, Any]:
        if not isinstance(raw_output, (bytes, bytearray)) or len(raw_output) != NUM_CLASSES * 8:
            raise WorkloadError(f"raw output must be {NUM_CLASSES * 8} bytes (int64 logits)")
        logits = list(struct.unpack(f"<{NUM_CLASSES}q", bytes(raw_output)))
        return {"schema": OUTPUT_SCHEMA, "task": self._task, "item_index": item.item_index,
                "logits": logits, "top1": logits.index(max(logits))}

    def native_output_pipeline(self) -> dict[str, Any]:
        """The native measurement plane's twin of postprocess + encode (Stage 1R-G).

        ``native/src/common/reference_output.hpp`` implements this workload's postprocess and codec so that
        ``t_output_ready`` keeps its meaning (postprocess + encode + record) inside the native measured
        path.  This class stays the reference implementation; byte equality is checked by tests over the
        whole fixture and, in every run, by the golden output identity."""
        return {"kind": "reference_convnet_v1", "task": self._task,
                "postprocess": f"{self.POSTPROCESS.id}/{self.POSTPROCESS.version}",
                "codec": f"{self.CODEC.id}/{self.CODEC.version}",
                "implementation": "native:native/src/common/reference_output.hpp"}

    def encode(self, task_output: object) -> bytes:
        if not isinstance(task_output, dict) or task_output.get("schema") != OUTPUT_SCHEMA:
            raise WorkloadError("encode() expects a reference_convnet task output")
        return canonical_json_bytes(task_output)
