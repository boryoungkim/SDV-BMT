"""Recorded driving-scenario fixture provider.

Preserves episode/stream/sensor/frame identity and typed source timing exactly
as present in the index.  Absent values stay ``None``; nothing is estimated.
Within an episode the source timestamps must strictly increase in index order;
a regression or duplicate is rejected and never silently re-sorted.  Only one
stream is selected (merging streams needs a common-clock basis and is not part
of Stage 1A).  The provider never computes ``t_sched``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..contracts.ids import sha256_hex
from ..contracts.protocols import InputValidationError
from ..contracts.records import InputItem, ScenarioCoordinates, SourceTiming
from ..contracts.specs import DrivingScenarioSpec
from .fixture_index import (
    ROW_BASE_KEYS,
    IndexedFixtureProvider,
    IndexRow,
    check_row_keys,
    check_unique_source_ids,
    parse_index_rows,
    row_annotation,
    row_metadata,
    row_payload,
    row_str,
)

COORDINATE_KEYS = ("episode_id", "stream_id", "sensor_id", "frame_id")


class _Row:
    __slots__ = ("row", "source_item_id", "coords", "timing")

    def __init__(self, row: IndexRow, source_item_id: str, coords: dict, timing: Optional[SourceTiming]):
        self.row = row
        self.source_item_id = source_item_id
        self.coords = coords
        self.timing = timing


class RecordedScenarioFixtureProvider(IndexedFixtureProvider):
    PROVIDER = "recorded_scenario_fixture"

    def __init__(self, spec: DrivingScenarioSpec, spec_dir: Path) -> None:
        if not isinstance(spec, DrivingScenarioSpec):
            raise TypeError("RecordedScenarioFixtureProvider requires a DrivingScenarioSpec")
        super().__init__(spec, spec_dir)

    def _parse_rows(self, data: bytes) -> list[_Row]:
        name = self._index_name
        family = self._spec.family
        timing = family.timing
        allowed = ROW_BASE_KEYS | set(COORDINATE_KEYS)
        if timing is not None:
            allowed = allowed | {timing.source_field}
        parsed: list[_Row] = []
        for row in parse_index_rows(data, name):
            check_row_keys(row, frozenset(allowed), name)
            source_item_id = row_str(row, "source_item_id", name, required=True)
            coords = {key: row_str(row, key, name) for key in COORDINATE_KEYS}
            if coords["stream_id"] is not None and coords["stream_id"] not in family.streams:
                raise InputValidationError(
                    f"{name}:{row.line}: stream_id {coords['stream_id']!r} is not declared in family.streams"
                )
            source_timing = None
            if timing is not None and row.data.get(timing.source_field) is not None:
                value = row.data[timing.source_field]
                if type(value) is not int:
                    raise InputValidationError(
                        f"{name}:{row.line}: {timing.source_field} must be an integer timestamp"
                    )
                source_timing = SourceTiming(value, timing.unit, timing.semantic, timing.clock_source)
            parsed.append(_Row(row, source_item_id, coords, source_timing))
        check_unique_source_ids([p.source_item_id for p in parsed], name)
        return parsed

    def _select(self, parsed: list[_Row]) -> list[_Row]:
        selection = self._spec.selection
        if selection.stream_id is not None:
            chosen = [p for p in parsed if p.coords["stream_id"] == selection.stream_id]
        else:
            chosen = list(parsed)
            if len({p.coords["stream_id"] for p in chosen}) > 1:
                raise InputValidationError(
                    "selection spans multiple streams; set selection.stream_id "
                    "(stream merging is not supported in Stage 1A)"
                )
        if selection.episodes is not None:
            ordered: list[_Row] = []
            for episode in selection.episodes:
                block = [p for p in chosen if p.coords["episode_id"] == episode]
                if not block:
                    raise InputValidationError(f"selected episode {episode!r} has no rows")
                ordered.extend(block)
        else:
            ordered = chosen
            seen: set = set()
            previous: object = object()
            for p in ordered:
                episode = p.coords["episode_id"]
                if episode != previous:
                    if episode in seen:
                        raise InputValidationError(
                            f"episode {episode!r} is not contiguous in the index; "
                            "select episodes explicitly"
                        )
                    seen.add(episode)
                    previous = episode
        if not ordered:
            raise InputValidationError("scenario selection is empty")
        return ordered

    def _check_timing_order(self, ordered: list[_Row]) -> None:
        last: dict[Optional[str], tuple[int, int]] = {}
        for p in ordered:
            if p.timing is None:
                continue
            episode = p.coords["episode_id"]
            if episode in last and p.timing.value <= last[episode][0]:
                prev_value, prev_line = last[episode]
                kind = "duplicate" if p.timing.value == prev_value else "regression"
                raise InputValidationError(
                    f"{self._index_name}:{p.row.line}: source timestamp {kind} in episode "
                    f"{episode!r} ({p.timing.value} after {prev_value} at line {prev_line}); "
                    "timestamps are never re-sorted"
                )
            last[episode] = (p.timing.value, p.row.line)

    def _resolve(self, data: bytes) -> tuple[list[InputItem], dict[int, bytes], dict[str, Any]]:
        name = self._index_name
        ordered = self._select(self._parse_rows(data))
        self._check_timing_order(ordered)
        items: list[InputItem] = []
        payloads: dict[int, bytes] = {}
        for item_index, p in enumerate(ordered):
            payload = row_payload(p.row, name)
            annotation_ref, annotation_digest = row_annotation(
                p.row, name, p.source_item_id, self._spec.family.annotation_schema
            )
            has_coords = any(v is not None for v in p.coords.values())
            items.append(
                InputItem(
                    item_index=item_index,
                    source_item_id=p.source_item_id,
                    payload_ref=f"{name}#{p.source_item_id}",
                    payload_sha256=sha256_hex(payload),
                    annotation_ref=annotation_ref,
                    annotation_digest=annotation_digest,
                    source_timing=p.timing,
                    scenario_coordinates=ScenarioCoordinates(**p.coords) if has_coords else None,
                    source_metadata=row_metadata(p.row, name),
                )
            )
            payloads[item_index] = payload
        episodes: list = []
        for p in ordered:
            if p.coords["episode_id"] not in episodes:
                episodes.append(p.coords["episode_id"])
        resolved = {"stream_id": ordered[0].coords["stream_id"], "episodes": episodes}
        return items, payloads, resolved
