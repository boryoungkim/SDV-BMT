"""Ground-truth annotation source of a fixture/dataset index (Model-level Metric Set v1).

When a workload declares a task-quality evaluator, the run freezes the ground truth of its
planned items in the bundle (``quality_reference.jsonl``, manifest v5), so the canonical
quality evaluation never reads a mutable external dataset location.  This source reads the
annotation objects from the digest-pinned source index and returns one row per resolved item;
each annotation must have the canonical digest the item already carries (``annotation_digest``,
part of ``items_digest``).  It is a separate component from the InputProvider, whose public
surface stays source/index/payload only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..contracts.ids import digest_of
from ..contracts.protocols import InputValidationError
from ..contracts.records import QUALITY_REFERENCE_SCHEMA_VERSION, InputItem
from ..contracts.specs import InputSpec, SpecError, plain_json
from .fixture_index import parse_index_rows, read_index_bytes, row_str


class IndexAnnotationSource:
    def __init__(self, spec: InputSpec, spec_dir: Path) -> None:
        self._spec = spec
        self._index_path = (Path(spec_dir) / spec.source.index).resolve()
        self._index_name = Path(spec.source.index).name

    def reference_rows(self, items: Sequence[InputItem]) -> list[dict[str, Any]]:
        name = self._index_name
        rows = parse_index_rows(read_index_bytes(self._index_path, self._spec.source.index_sha256), name)
        by_source = {row_str(row, "source_item_id", name, required=True): row for row in rows}
        out: list[dict[str, Any]] = []
        for item in items:
            row = by_source.get(item.source_item_id)
            if row is None:
                raise InputValidationError(f"{name}: item {item.item_index} ({item.source_item_id}) is not in the index")
            if item.annotation_digest is None or row.data.get("annotation") is None:
                raise InputValidationError(f"{name}:{row.line}: item {item.item_index} has no annotation")
            try:
                annotation = plain_json(row.data["annotation"], f"{name}:{row.line}.annotation")
            except SpecError as exc:
                raise InputValidationError(str(exc)) from None
            if digest_of(annotation) != item.annotation_digest:
                raise InputValidationError(f"{name}:{row.line}: annotation does not match the item's annotation_digest")
            out.append({
                "schema_version": QUALITY_REFERENCE_SCHEMA_VERSION,
                "item_index": item.item_index,
                "source_item_id": item.source_item_id,
                "annotation_digest": item.annotation_digest,
                "annotation": annotation,
            })
        return out
