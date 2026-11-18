"""Source digest of the Eagle-N executor and its build record.

The executor ``libeagle_n_ttnn.so`` is built in the BOS container from three files of this repository
(``EXECUTOR_SOURCES``). ``tools/eagle_n/build_native.sh`` writes ``<executor>.build.json`` next to the executor: the
executor's SHA-256 and the digest of those sources.  The DUT config pins both, the qualification receipt repeats the
source digest, and the adapter refuses a run unless the pins, the build record and the sources of the current
checkout agree.  A checkout can therefore only run the executor built from its own source.

Run as ``python -m sdv_bmt.dut.eagle_n.executor_source NATIVE_DIR`` to print the digest (used by the build script).
Standard library and ``sdv_bmt.contracts.ids`` only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ...contracts.ids import digest_of, sha256_hex

# Paths relative to native/.
EXECUTOR_SOURCES = ("src/eagle_n/ttnn_executor.cpp", "src/eagle_n/executor_abi.hpp", "src/common/json.hpp")
BUILD_RECORD_SUFFIX = ".build.json"
BUILD_RECORD_SCHEMA = "eagle-n-executor-build/1"
NATIVE_DIR = Path(__file__).resolve().parents[4] / "native"


def executor_source_sha256(native_dir: Path = NATIVE_DIR) -> str:
    """SHA-256 of the canonical JSON list of ``{path, sha256}`` of the executor sources."""
    native_dir = Path(native_dir)
    listing = [{"path": p, "sha256": sha256_hex((native_dir / p).read_bytes())} for p in EXECUTOR_SOURCES]
    return digest_of(listing)


def build_record_path(executor: str | Path) -> Path:
    return Path(str(executor) + BUILD_RECORD_SUFFIX)


def read_build_record(executor: str | Path) -> dict[str, Any]:
    """The build record next to ``executor``; its ``executor_sha256`` must be the SHA-256 of that file."""
    path = build_record_path(executor)
    if not path.is_file():
        raise ValueError(
            f"executor build record {path} missing (build the executor with tools/eagle_n/build_native.sh)"
        )
    record = json.loads(path.read_text())
    if record.get("schema") != BUILD_RECORD_SCHEMA:
        raise ValueError(f"{path}: schema {record.get('schema')!r} is not {BUILD_RECORD_SCHEMA}")
    for key in ("executor_sha256", "executor_source_sha256"):
        value = record.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{path}: {key} is not a lowercase SHA-256")
    if sha256_hex(Path(executor).read_bytes()) != record["executor_sha256"]:
        raise ValueError(f"{path}: executor_sha256 is not the SHA-256 of {executor}")
    return record


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m sdv_bmt.dut.eagle_n.executor_source NATIVE_DIR", file=sys.stderr)
        return 2
    print(executor_source_sha256(Path(argv[0])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
