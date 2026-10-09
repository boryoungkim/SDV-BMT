"""Deterministic fake workload (``implementation: fake_identity``).

* preprocess:  verifies the payload against ``item.payload_sha256`` and returns
  ``PREPROCESS_PREFIX + payload`` (bytes).
* postprocess: verifies the raw DUT output corresponds to the item (index and
  item digest) and returns a task-output mapping that keeps ``request_id``,
  ``item_index``, ``item_digest`` and ``value``.
* encode:      canonical JSON (UTF-8) of the task output.

The declared component identities in the WorkloadSpec must match the ones
implemented here, so a spec cannot claim a codec the code does not use.

``input_contract()`` returns the immutable WorkloadInputContract (semantics
``fake-prefix-preprocess/1``: logical input = ``prefix || payload`` bytes,
prefix given as hex).  A DUT adapter override receives only this contract.

Options: ``postprocess_error_items`` injects a postprocess failure for the
listed item indices (exercises the ERROR(phase=postprocess) path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts.ids import canonical_json_bytes, sha256_hex
from ..contracts.protocols import WorkloadError
from ..contracts.records import InputItem, WorkloadInputContract, item_digest
from ..contracts.specs import ComponentIdentity, WorkloadSpec, parse_strict

PREPROCESS_PREFIX = b"sdv-bmt fake preprocess v1\n"
OUTPUT_SCHEMA = "fake-identity-output-v1"
INPUT_CONTRACT_ID = "fake-identity-input"
INPUT_CONTRACT_VERSION = "1"


@dataclass(frozen=True)
class FakeWorkloadOptions:
    postprocess_error_items: tuple[int, ...] = ()


class FakeIdentityWorkload:
    IMPLEMENTATION = "fake_identity"
    PREPROCESS = ComponentIdentity("fake-prefix-preprocess", "1")
    POSTPROCESS = ComponentIdentity("fake-identity-postprocess", "1")
    CODEC = ComponentIdentity("canonical-json-utf8", "1")

    def __init__(self, spec: WorkloadSpec) -> None:
        if spec.implementation != self.IMPLEMENTATION:
            raise ValueError(f"workload implementation must be {self.IMPLEMENTATION!r}")
        for declared, actual, what in (
            (spec.preprocess, self.PREPROCESS, "preprocess"),
            (spec.postprocess, self.POSTPROCESS, "postprocess"),
            (spec.codec, self.CODEC, "codec"),
        ):
            if declared != actual:
                raise ValueError(f"workload.{what} must be {actual} for {self.IMPLEMENTATION}")
        if spec.artifact_sha256 is not None:
            raise ValueError("fake_identity workload has no model artifact (artifact_sha256 must be null)")
        options = parse_strict(FakeWorkloadOptions, spec.options, "workload.options")
        self._task = spec.task
        self._postprocess_error_items = frozenset(options.postprocess_error_items)
        self._contract = WorkloadInputContract(
            schema_version=1,
            contract_id=INPUT_CONTRACT_ID,
            contract_version=INPUT_CONTRACT_VERSION,
            task=spec.task,
            preprocess_semantics_id=f"{self.PREPROCESS.id}/{self.PREPROCESS.version}",
            payload_format="utf8-text-fixture",
            logical_input=(
                ("encoding", "bytes"),
                ("layout", "prefix||payload"),
                ("prefix_hex", PREPROCESS_PREFIX.hex()),
            ),
        )

    def input_contract(self) -> WorkloadInputContract:
        return self._contract

    def preprocess(self, payload: bytes, item: InputItem) -> bytes:
        if not isinstance(payload, bytes):
            raise WorkloadError("payload must be bytes")
        if sha256_hex(payload) != item.payload_sha256:
            raise WorkloadError(f"payload of item {item.item_index} does not match payload_sha256")
        return PREPROCESS_PREFIX + payload

    def postprocess(self, raw_output: object, item: InputItem) -> dict[str, Any]:
        if not isinstance(raw_output, dict):
            raise WorkloadError("raw output must be a mapping")
        if raw_output.get("item_index") != item.item_index or raw_output.get("item_digest") != item_digest(item):
            raise WorkloadError(f"raw output does not correspond to item {item.item_index}")
        if item.item_index in self._postprocess_error_items:
            raise WorkloadError(f"injected postprocess error for item {item.item_index}")
        return {
            "schema": OUTPUT_SCHEMA,
            "task": self._task,
            "request_id": raw_output["request_id"],
            "item_index": item.item_index,
            "item_digest": raw_output["item_digest"],
            "value": raw_output["value"],
        }

    def encode(self, task_output: object) -> bytes:
        if not isinstance(task_output, dict) or task_output.get("schema") != OUTPUT_SCHEMA:
            raise WorkloadError("encode() expects a fake-identity task output")
        return canonical_json_bytes(task_output)
