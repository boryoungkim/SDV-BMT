"""Source index reader shared by the Stage 1A input fixtures.

A fixture index is UTF-8 JSON Lines whose SHA-256 must equal the digest the
input spec declares.  Each row is a strict JSON object (duplicate keys and
unknown keys are rejected).  Blank lines are rejected except one trailing
newline.  Nothing is sorted or filled in.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..contracts.ids import digest_of, loads_strict_json, sha256_hex
from ..contracts.protocols import InputValidationError, LifecycleError
from ..contracts.records import InputDescriptor, InputItem, items_digest
from ..contracts.specs import (
    InputSpec,
    SpecError,
    input_family_digest,
    input_selection_digest,
    plain_json,
)

ROW_BASE_KEYS = frozenset({"source_item_id", "payload_text", "annotation", "metadata"})


@dataclass(frozen=True)
class IndexRow:
    line: int
    data: dict[str, Any]


def read_index_bytes(path: Path, declared_sha256: str) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise InputValidationError(f"{path.name}: cannot read source index: {exc}") from None
    actual = sha256_hex(data)
    if actual != declared_sha256:
        raise InputValidationError(
            f"{path.name}: source index sha256 mismatch "
            f"(declared {declared_sha256}, actual {actual})"
        )
    return data


def parse_index_rows(data: bytes, name: str) -> list[IndexRow]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputValidationError(f"{name}: not UTF-8: {exc}") from None
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    rows: list[IndexRow] = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            raise InputValidationError(f"{name}:{lineno}: blank line")
        try:
            obj = loads_strict_json(line)
        except ValueError as exc:
            raise InputValidationError(f"{name}:{lineno}: invalid JSON: {exc}") from None
        if not isinstance(obj, Mapping):
            raise InputValidationError(f"{name}:{lineno}: row is not a JSON object")
        rows.append(IndexRow(lineno, dict(obj)))
    if not rows:
        raise InputValidationError(f"{name}: source index is empty")
    return rows


def check_row_keys(row: IndexRow, allowed: frozenset[str], name: str) -> None:
    unknown = sorted(set(row.data) - allowed)
    if unknown:
        raise InputValidationError(f"{name}:{row.line}: unknown field(s): {', '.join(unknown)}")
    for key in ("source_item_id", "payload_text"):
        if key not in row.data:
            raise InputValidationError(f"{name}:{row.line}: required field {key!r} is missing")


def row_str(row: IndexRow, key: str, name: str, required: bool = False) -> Optional[str]:
    if key not in row.data or row.data[key] is None:
        if required:
            raise InputValidationError(f"{name}:{row.line}: {key} is required")
        return None
    value = row.data[key]
    if not isinstance(value, str) or not value:
        raise InputValidationError(f"{name}:{row.line}: {key} must be a non-empty string")
    return value


def row_payload(row: IndexRow, name: str) -> bytes:
    value = row.data["payload_text"]
    if not isinstance(value, str):
        raise InputValidationError(f"{name}:{row.line}: payload_text must be a string")
    return value.encode("utf-8")


def row_annotation(
    row: IndexRow, name: str, source_item_id: str, annotation_schema: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    """Return (annotation_ref, annotation_digest); both null when the row has none."""
    annotation = row.data.get("annotation")
    if annotation is None:
        return None, None
    if annotation_schema is None:
        raise InputValidationError(
            f"{name}:{row.line}: row has an annotation but the family declares annotation_schema=null"
        )
    try:
        content = plain_json(annotation, f"{name}:{row.line}.annotation")
    except SpecError as exc:
        raise InputValidationError(str(exc)) from None
    return f"{name}#{source_item_id}/annotation", digest_of(content)


def row_metadata(row: IndexRow, name: str) -> dict[str, Any]:
    """Provenance (index line) plus row metadata only when the source has it."""
    metadata: dict[str, Any] = {"index_line": row.line}
    if "metadata" in row.data and row.data["metadata"] is not None:
        if not isinstance(row.data["metadata"], Mapping):
            raise InputValidationError(f"{name}:{row.line}: metadata must be an object")
        try:
            metadata["row_metadata"] = plain_json(row.data["metadata"], f"{name}:{row.line}.metadata")
        except SpecError as exc:
            raise InputValidationError(str(exc)) from None
    return metadata


def check_unique_source_ids(ids: list[str], name: str) -> None:
    seen: set[str] = set()
    for sid in ids:
        if sid in seen:
            raise InputValidationError(f"{name}: duplicate source_item_id {sid!r}")
        seen.add(sid)


class IndexedFixtureProvider:
    """InputProvider lifecycle over a digest-pinned fixture index.

    ``describe()``/``items()`` read and validate the index without side effects
    (PREFLIGHT).  ``open()`` re-reads the index, re-verifies its digest and the
    resolved items, and only then exposes payloads.  The provider never holds
    workload or DUT objects.
    """

    PROVIDER = ""

    def __init__(self, spec: InputSpec, spec_dir: Path) -> None:
        self._spec = spec
        self._index_path = (Path(spec_dir) / spec.source.index).resolve()
        self._index_name = Path(spec.source.index).name
        self._items: Optional[tuple[InputItem, ...]] = None
        self._selection: Optional[dict[str, Any]] = None
        self._payloads: Optional[dict[int, bytes]] = None
        self._released = False

    def _resolve(self, data: bytes) -> tuple[list[InputItem], dict[int, bytes], dict[str, Any]]:
        """Return (items, payloads, resolved selection)."""
        raise NotImplementedError

    def _read(self) -> tuple[list[InputItem], dict[int, bytes], dict[str, Any]]:
        data = read_index_bytes(self._index_path, self._spec.source.index_sha256)
        return self._resolve(data)

    def items(self) -> tuple[InputItem, ...]:
        if self._items is None:
            items, _payloads, selection = self._read()
            self._items = tuple(items)
            self._selection = selection
        return self._items

    def describe(self) -> InputDescriptor:
        items = self.items()
        return InputDescriptor(
            input_kind=self._spec.input_kind,
            input_id=self._spec.input_id,
            version=self._spec.version,
            provider=self.PROVIDER,
            input_family_digest=input_family_digest(self._spec),
            input_selection_digest=input_selection_digest(self._spec, self._selection or {}),
            items_digest=items_digest(items),
            item_count=len(items),
            source_index_sha256=self._spec.source.index_sha256,
            source_timing_complete=bool(items) and all(i.source_timing is not None for i in items),
        )

    def open(self) -> None:
        if self._released:
            raise LifecycleError("open() after release()")
        if self._payloads is not None:
            return
        items, payloads, _selection = self._read()
        if items_digest(items) != items_digest(self.items()):
            raise InputValidationError("source items changed between preflight and open()")
        self._payloads = payloads

    def payload(self, item_index: int) -> bytes:
        if self._released:
            raise LifecycleError("payload() after release()")
        if self._payloads is None:
            raise LifecycleError("payload() before open()")
        if item_index not in self._payloads:
            raise InputValidationError(f"unknown item_index {item_index}")
        return self._payloads[item_index]

    def release(self) -> None:
        self._payloads = None
        self._released = True
