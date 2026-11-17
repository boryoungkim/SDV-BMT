"""Fixed-length synthetic speed workload S1-S5 and workload files.

Each speed request is BOS followed by regular (non-special) token IDs drawn with a counter-based SHA-256
generator, so the token IDs are reproducible on any Python version.  No chat template is used, so the input
length is exact.  Every request of a workload has its own sequence (3 warm-up + 5 measured per S); the same
file is used for every model, device and run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

BOS_TOKEN_ID = 128000
REGULAR_VOCAB = 128000  # Llama 3 IDs 0..127999 are regular tokens; 128000..128255 are special
SPEED_SEED = 20261117
WARMUP_PER_WORKLOAD = 3
MEASURE_PER_WORKLOAD = 5


@dataclass(frozen=True)
class SpeedWorkload:
    name: str
    input_tokens: int
    output_tokens: int


SPEED_WORKLOADS = (
    SpeedWorkload("S1", 128, 64),
    SpeedWorkload("S2", 128, 256),
    SpeedWorkload("S3", 512, 512),
    SpeedWorkload("S4", 830, 256),
    SpeedWorkload("S5", 1024, 128),
)


def synthetic_token_ids(seed: int, workload: str, index: int, length: int) -> list[int]:
    ids = [BOS_TOKEN_ID]
    for pos in range(1, length):
        h = hashlib.sha256(f"{seed}:{workload}:{index}:{pos}".encode()).digest()
        ids.append(int.from_bytes(h[:8], "big") % REGULAR_VOCAB)
    return ids


def token_ids_sha256(ids: Iterable[int]) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def speed_requests(seed: int = SPEED_SEED) -> list[dict[str, Any]]:
    """The S1-S5 request list in execution order: per workload 3 warm-up then 5 measured requests."""
    out = []
    for w in SPEED_WORKLOADS:
        for k in range(WARMUP_PER_WORKLOAD + MEASURE_PER_WORKLOAD):
            phase = "warmup" if k < WARMUP_PER_WORKLOAD else "measure"
            ids = synthetic_token_ids(seed, w.name, k, w.input_tokens)
            out.append({
                "item_id": f"{w.name}-{k:02d}",
                "workload": w.name,
                "phase": phase,
                "input_token_ids": ids,
                "input_token_ids_sha256": token_ids_sha256(ids),
                "generation": {"max_tokens": w.output_tokens, "min_tokens": w.output_tokens, "ignore_eos": True,
                               "stop_token_ids": []},
                "target_output_tokens": w.output_tokens,
            })
    return out


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "x", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":"), sort_keys=True) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
