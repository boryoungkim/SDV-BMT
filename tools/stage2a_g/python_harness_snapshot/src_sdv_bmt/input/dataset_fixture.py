"""Dataset fixture provider: deterministic samples without scheduling timestamps.

Items follow index order restricted by ``selection.start/count``.  Dataset
items never carry SourceTiming or ScenarioCoordinates: no timestamp, episode,
sensor or frame is invented, and recorded replay is rejected by validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts.ids import sha256_hex
from ..contracts.protocols import InputValidationError
from ..contracts.records import InputItem
from ..contracts.specs import DatasetSpec
from .fixture_index import (
    ROW_BASE_KEYS,
    IndexedFixtureProvider,
    check_row_keys,
    check_unique_source_ids,
    parse_index_rows,
    row_annotation,
    row_metadata,
    row_payload,
    row_str,
)


class DatasetFixtureProvider(IndexedFixtureProvider):
    PROVIDER = "dataset_fixture"

    def __init__(self, spec: DatasetSpec, spec_dir: Path) -> None:
        if not isinstance(spec, DatasetSpec):
            raise TypeError("DatasetFixtureProvider requires a DatasetSpec")
        super().__init__(spec, spec_dir)

    def _resolve(self, data: bytes) -> tuple[list[InputItem], dict[int, bytes], dict[str, Any]]:
        name = self._index_name
        spec = self._spec
        rows = parse_index_rows(data, name)
        for row in rows:
            check_row_keys(row, ROW_BASE_KEYS, name)
        ids = [row_str(row, "source_item_id", name, required=True) for row in rows]
        check_unique_source_ids(ids, name)

        start, count = spec.selection.start, spec.selection.count
        if start >= len(rows):
            raise InputValidationError(
                f"selection.start {start} is beyond the {len(rows)} rows of {name}"
            )
        end = len(rows) if count is None else start + count
        if end > len(rows):
            raise InputValidationError(
                f"selection requests {count} items from {start}, but {name} has {len(rows)} rows"
            )

        items: list[InputItem] = []
        payloads: dict[int, bytes] = {}
        for item_index, row in enumerate(rows[start:end]):
            source_item_id = ids[start + item_index]
            payload = row_payload(row, name)
            annotation_ref, annotation_digest = row_annotation(
                row, name, source_item_id, spec.family.annotation_schema
            )
            items.append(
                InputItem(
                    item_index=item_index,
                    source_item_id=source_item_id,
                    payload_ref=f"{name}#{source_item_id}",
                    payload_sha256=sha256_hex(payload),
                    annotation_ref=annotation_ref,
                    annotation_digest=annotation_digest,
                    source_timing=None,
                    scenario_coordinates=None,
                    source_metadata=row_metadata(row, name),
                )
            )
            payloads[item_index] = payload
        return items, payloads, {"start": start, "count": end - start}
