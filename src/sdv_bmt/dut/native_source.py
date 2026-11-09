"""Source-tree digest of the native measurement plane and the build stamp next to each native executable.

The digest is the one recorded in every bundle as ``native.source_tree_sha256``: canonical JSON of the
sorted list of ``{path, sha256}`` of every ``.hpp``/``.cpp``/extension-less file under ``native/`` outside
``build/`` (``native/src``, ``native/tests``, ``native/Makefile``).

``make -C native`` writes ``<executable>.source-tree.sha256`` next to every executable it builds (this
module run as ``python3 -m sdv_bmt.dut.native_source NATIVE_DIR [EXTRA_FILE ...]``).  A run refuses an
executable whose stamp is missing or differs from the current source tree (v0.1.2 safeguard: a source
edit that was never rebuilt would otherwise run the old executable under the new source digest).  An
offline tool built from the native sources plus its own source files (``tools/stage2b_g_real/raw_dump.cpp``)
stamps the digest of ``{native_source_tree_sha256, extra file digests}``.

Standard library and ``sdv_bmt.contracts.ids`` only, so the Makefile can run it without the package's
dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from ..contracts.ids import digest_of, sha256_hex

STAMP_SUFFIX = ".source-tree.sha256"


def native_source_tree_sha256(native_dir: Path) -> str:
    """SHA-256 over the native sources (``native/src``, ``native/tests``, ``native/Makefile``) by relative path."""
    native_dir = Path(native_dir)
    files = sorted(p for p in native_dir.rglob("*") if p.is_file() and "build" not in p.relative_to(native_dir).parts
                   and p.suffix in (".hpp", ".cpp", ""))
    listing = [{"path": str(p.relative_to(native_dir)), "sha256": sha256_hex(p.read_bytes())} for p in files]
    return digest_of(listing)


def build_source_sha256(native_dir: Path, extra_files: Sequence[Path] = ()) -> str:
    """Digest an executable's stamp records: the native source tree alone, or with the tool's own sources."""
    tree = native_source_tree_sha256(native_dir)
    if not extra_files:
        return tree
    return digest_of({"native_source_tree_sha256": tree,
                      "extra_files": [{"path": Path(p).name, "sha256": sha256_hex(Path(p).read_bytes())}
                                      for p in extra_files]})


def stamp_path(executable: Path) -> Path:
    return Path(str(executable) + STAMP_SUFFIX)


def check_stamp(executable: Path, native_dir: Path, extra_files: Sequence[Path] = ()) -> str:
    """The current source digest if ``executable`` was built from it; otherwise raise ``ValueError``."""
    current = build_source_sha256(native_dir, extra_files)
    stamp = stamp_path(executable)
    if not stamp.is_file():
        raise ValueError(f"{executable.name} has no build stamp {stamp.name}; rebuild it (make -C native)")
    recorded = stamp.read_text(encoding="utf-8").strip()
    if recorded != current:
        raise ValueError(f"{executable.name} was built from native sources {recorded[:12]}..., the current sources "
                         f"are {current[:12]}...; rebuild it (make -C native)")
    return current


def main(argv: Sequence[str]) -> int:
    if not argv:
        sys.stderr.write("usage: python3 -m sdv_bmt.dut.native_source NATIVE_DIR [EXTRA_FILE ...]\n")
        return 2
    print(build_source_sha256(Path(argv[0]), [Path(p) for p in argv[1:]]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
