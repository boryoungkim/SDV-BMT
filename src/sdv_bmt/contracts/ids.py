"""Canonical JSON, SHA-256 digests and run/evaluation identifiers.

Canonical JSON (``sdv-bmt-canonical-json-v1``) is ``json.dumps`` with sorted
keys, compact separators, UTF-8 output and no NaN/Infinity.  ``to_jsonable``
first converts dataclasses, str-enums and tuples to plain JSON values and
rejects anything that would serialize ambiguously (non-string mapping keys,
non-finite floats, unknown types).  Digests are lowercase SHA-256 hex.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

CANONICAL_JSON_VERSION = "sdv-bmt-canonical-json-v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CanonicalJSONError(TypeError):
    """Raised when a value cannot be represented as canonical JSON."""


def to_jsonable(value: Any, _path: str = "$") -> Any:
    """Convert ``value`` to plain JSON types, rejecting ambiguous values."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Enum):
        if not isinstance(value.value, str):
            raise CanonicalJSONError(f"{_path}: enum {value!r} is not str-valued")
        return value.value
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJSONError(f"{_path}: non-finite float {value!r}")
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: to_jsonable(getattr(value, f.name), f"{_path}.{f.name}")
            for f in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJSONError(f"{_path}: non-string key {key!r}")
            out[key] = to_jsonable(item, f"{_path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item, f"{_path}[{i}]") for i, item in enumerate(value)]
    raise CanonicalJSONError(f"{_path}: unsupported type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    """Deterministic human-readable JSON (sorted keys, 2-space indent, final newline)."""
    text = json.dumps(
        to_jsonable(value),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON number {name}")


def loads_strict_json(text: str) -> Any:
    """``json.loads`` that rejects duplicate keys and NaN/Infinity."""
    return json.loads(
        text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant
    )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_of(value: Any) -> str:
    """SHA-256 of the canonical JSON encoding of ``value``."""
    return sha256_hex(canonical_json_bytes(value))


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def is_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.match(value))


def utc_now_iso() -> str:
    """Wall-clock UTC timestamp. Volatile metadata only; never used for timing."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _new_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:12]}"


def new_run_id() -> str:
    """Globally unique, volatile run ID. Not used in deterministic comparison."""
    return _new_id("run")


def new_eval_id() -> str:
    """Globally unique, volatile evaluation ID."""
    return _new_id("eval")
