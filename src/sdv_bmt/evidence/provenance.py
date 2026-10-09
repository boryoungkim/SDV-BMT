"""Code and environment provenance.

Stage 1A records a source-tree SHA-256 instead of a git commit: the
repository has no git metadata and the core does not spawn subprocesses.

``source_tree_sha256`` = SHA-256 of the canonical JSON list
``[{"path": <posix path relative to src/>, "sha256": <file sha256>}, ...]``
over every ``sdv_bmt/**/*.py`` file, sorted by path.
"""

from __future__ import annotations

import functools
import platform
from pathlib import Path
from typing import Any

from .. import STAGE, __version__
from ..contracts.ids import digest_of, file_sha256

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

SOURCE_TREE_METHOD = (
    "sha256(canonical_json([{path, sha256}] for sdv_bmt/**/*.py sorted by path, "
    "path relative to the package parent directory))"
)


def source_tree_listing(root: Path = PACKAGE_ROOT) -> list[dict[str, str]]:
    root = Path(root)
    files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return [{"path": p.relative_to(root.parent).as_posix(), "sha256": file_sha256(p)} for p in files]


def source_tree_sha256(root: Path = PACKAGE_ROOT) -> str:
    return digest_of(source_tree_listing(root))


def _git_detected(start: Path) -> bool:
    return any((parent / ".git").exists() for parent in [start, *start.parents])


@functools.lru_cache(maxsize=1)
def code_identity() -> dict[str, Any]:
    listing = source_tree_listing()
    return {
        "package": "sdv_bmt",
        "version": __version__,
        "stage": STAGE,
        "source_tree_sha256": digest_of(listing),
        "source_tree_method": SOURCE_TREE_METHOD,
        "source_file_count": len(listing),
        "vcs": {
            "git_repository_detected": _git_detected(PACKAGE_ROOT),
            "commit_recorded": False,
            "note": "Stage 1A records the source-tree SHA-256; git commit/dirty state is not queried",
        },
    }


def environment(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Interpreter/platform description (volatile across hosts; never telemetry)."""
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    info.update(extra or {})
    return info
