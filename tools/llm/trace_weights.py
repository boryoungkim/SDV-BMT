"""Compare two safetensors weight sets tensor by tensor (raw bytes, SHA-256 per tensor name).

Used to trace whether a furiosa-llm artifact carries the same weights as a pinned Hugging Face revision.
Reads the safetensors headers and byte ranges directly (no framework needed).

    python3 tools/llm/trace_weights.py --a <dir with *.safetensors> --b <dir with *.safetensors> --out FILE
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import struct


def tensor_hashes(d: str) -> dict[str, dict]:
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
            base = 8 + n
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                a, b = meta["data_offsets"]
                fh.seek(base + a)
                h = hashlib.sha256()
                left = b - a
                while left:
                    chunk = fh.read(min(left, 1 << 24))
                    h.update(chunk)
                    left -= len(chunk)
                out[name] = {"dtype": meta["dtype"], "shape": meta["shape"], "sha256": h.hexdigest()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    x = ap.parse_args()
    ha, hb = tensor_hashes(x.a), tensor_hashes(x.b)
    common = sorted(set(ha) & set(hb))
    same = [k for k in common if ha[k] == hb[k]]
    res = {"a": x.a, "b": x.b, "tensors_a": len(ha), "tensors_b": len(hb), "common": len(common),
           "identical": len(same), "differing": [k for k in common if ha[k] != hb[k]][:50],
           "only_a": sorted(set(ha) - set(hb))[:50], "only_b": sorted(set(hb) - set(ha))[:50],
           "all_common_identical": len(same) == len(common)}
    with open(x.out, "x") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
