#!/usr/bin/env python3
"""Read-only fingerprint of BOS SDK dependencies; no device API calls."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sdk", type=Path, default=os.environ.get("EAGLE_N_SDK_ROOT"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.sdk is None:
        p.error("the BOS SDK root is required: set EAGLE_N_SDK_ROOT or pass --sdk")
    names = [
        "ttnn/_ttnncpp.so",
        "tt_metal/libtt_metal.so",
        "lib/libdevice.so",
        "tt_stl/libtt_stl.so",
        "lib/libtracy.so.0.10.0",
    ]
    libs = {str(a.sdk / "build_Release_distributed" / n): sha(a.sdk / "build_Release_distributed" / n) for n in names}
    d = {
        "sdk_commit": subprocess.check_output(["git", "-C", str(a.sdk), "rev-parse", "HEAD"], text=True).strip(),
        "libraries_sha256": libs,
        "binary_tree": "build_Release_distributed (same as standalone validated runs)",
        "device_queried": False,
        "source_read_only": True,
    }
    a.out.write_text(json.dumps(d, indent=2) + "\n")
    print(json.dumps(d, indent=2))


if __name__ == "__main__":
    main()
