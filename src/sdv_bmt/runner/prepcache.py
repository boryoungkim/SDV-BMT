"""Prepared-input cache: reuse ``Workload.preprocess`` results across runs (opt-in, outside every measured interval).

Preprocessing runs before the measurement and is not timed, but for large inputs it dominates the wall time of a run
(pure-Python YOLOX letterbox on 5,000 COCO images: about 12 minutes per run).  When the environment variable
``SDV_BMT_PREPARED_INPUT_CACHE`` names a directory, the Runner keeps the bytes ``Workload.preprocess`` returned there
and reuses them in later runs.  Unset, nothing changes.

Correctness rules:

* key = SHA-256 of {method, context, item}.  The context is the whole SDV-BMT source tree digest (any code change
  starts a new cache), the workload implementation, the preprocessing identity and the input contract digest (which
  carries the input size and semantics); the item is the full ``InputItem`` record, whose ``payload_sha256`` comes
  from the digest-pinned index and is re-verified against the payload file by ``InputProvider.open()`` before any
  preparation.  So an entry is used only for the same payload, the same item metadata and the same code.
* an entry is the bytes plus a sidecar with their SHA-256; a read re-hashes the bytes and treats any mismatch,
  truncation or read error as a miss (the item is preprocessed again and the entry rewritten);
* entries are written atomically (temporary file, then rename); a failed write never fails the run;
* only ``bytes`` results are cached; other result types are always preprocessed.

The run records what the cache did in the ``prepare_inputs`` end event (``details.prepared_input_cache``: method,
context digest, hits, misses, writes, rejected entries, write errors, results that were not cacheable).  The cache
directory path is not recorded.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from ..contracts.ids import digest_of, to_jsonable

CACHE_ENV = "SDV_BMT_PREPARED_INPUT_CACHE"
KEY_METHOD = "sdv-bmt-prepared-input-cache/1"


class PreparedInputCache:
    def __init__(self, root: Path | str, context: dict[str, Any]) -> None:
        self.root = Path(root)
        self.context_digest = digest_of({"method": KEY_METHOD, "context": context})
        self.stats = {"hits": 0, "misses": 0, "written": 0, "rejected": 0, "write_errors": 0, "not_cacheable": 0}

    def key(self, item: Any) -> str:
        return digest_of({"method": KEY_METHOD, "context": self.context_digest, "item": to_jsonable(item)})

    def _paths(self, key: str) -> tuple[Path, Path]:
        d = self.root / key[:2]
        return d / f"{key}.bin", d / f"{key}.sha256"

    def get(self, item: Any) -> Optional[bytes]:
        """The cached bytes of this item, or None (miss).  Never raises: any problem is a miss."""
        try:
            data_path, sum_path = self._paths(self.key(item))
            if not data_path.is_file():
                self.stats["misses"] += 1
                return None
            expected = sum_path.read_text(encoding="ascii").strip()
            data = data_path.read_bytes()
        except Exception:  # noqa: BLE001 - an unreadable entry or an unkeyable item is a miss
            data, expected = b"", ""
        if not expected or hashlib.sha256(data).hexdigest() != expected:
            self.stats["rejected"] += 1
            self.stats["misses"] += 1
            return None
        self.stats["hits"] += 1
        return data

    def put(self, item: Any, data: Any) -> None:
        """Store this item's preprocessed bytes.  Never raises: a failed write is counted."""
        if not isinstance(data, bytes):
            self.stats["not_cacheable"] += 1
            return
        data_path, sum_path = self._paths(self.key(item))
        try:
            data_path.parent.mkdir(parents=True, exist_ok=True)
            for path, content in ((data_path, data), (sum_path, hashlib.sha256(data).hexdigest().encode("ascii"))):
                fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
                try:
                    with os.fdopen(fd, "wb") as fh:
                        fh.write(content)
                    os.replace(tmp, path)
                except BaseException:
                    Path(tmp).unlink(missing_ok=True)
                    raise
            self.stats["written"] += 1
        except Exception:  # noqa: BLE001 - the run goes on with the freshly preprocessed bytes
            self.stats["write_errors"] += 1

    def record(self) -> dict[str, Any]:
        """What the run records (no path)."""
        return {"method": KEY_METHOD, "context_digest": self.context_digest, **self.stats}


def cache_from_environment(context: dict[str, Any]) -> Optional[PreparedInputCache]:
    root = os.environ.get(CACHE_ENV)
    return PreparedInputCache(root, context) if root else None
