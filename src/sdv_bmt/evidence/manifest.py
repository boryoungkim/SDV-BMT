"""Run manifest (design v03 I.3): identity, status, provenance and file digests.

The manifest is written last, after the raw files are closed, and is never
modified afterwards.  ``VOLATILE_FIELDS`` lists the top-level fields excluded
from deterministic (canonical) comparison between runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts.ids import digest_of, to_jsonable
from ..contracts.records import QUALITY_RECORD_SCHEMA_VERSION, RECORD_SCHEMA_VERSION, RunManifest
from .bundle import MANIFEST_FILE, read_json, write_json_exclusive

VOLATILE_FIELDS = ("run_id", "created_at_utc", "environment")
# realtime runs also carry absolute clock anchors, which differ on every run
REALTIME_VOLATILE_FIELDS = VOLATILE_FIELDS + ("run_clock",)


# sections that exist only in realtime (v4/v5) manifests; a v3 manifest keeps its Stage 1A shape
REALTIME_ONLY_SECTIONS = ("run_clock", "realtime")
# section that exists only in v5 manifests (frozen task-quality reference); v3/v4 keep their shape
QUALITY_ONLY_SECTIONS = ("quality_reference",)


def write_manifest(run_dir: Path, manifest: RunManifest) -> dict[str, Any]:
    content = to_jsonable(manifest)
    if manifest.schema_version == RECORD_SCHEMA_VERSION:
        for section in REALTIME_ONLY_SECTIONS:
            content.pop(section)
    if manifest.schema_version != QUALITY_RECORD_SCHEMA_VERSION:
        for section in QUALITY_ONLY_SECTIONS:
            content.pop(section)
    write_json_exclusive(Path(run_dir) / MANIFEST_FILE, content)
    return content


def read_manifest(run_dir: Path) -> dict[str, Any]:
    return read_json(Path(run_dir) / MANIFEST_FILE)


def canonical_manifest_content(manifest: dict[str, Any]) -> dict[str, Any]:
    volatile = set(manifest.get("volatile_fields", VOLATILE_FIELDS))
    return {k: v for k, v in manifest.items() if k not in volatile}


def canonical_manifest_digest(manifest: dict[str, Any]) -> str:
    return digest_of(canonical_manifest_content(manifest))
