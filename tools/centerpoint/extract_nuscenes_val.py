"""Extract the nuScenes v1.0-trainval files the CenterPoint val input needs (offline, tools only; cross-check env).

Run with ``<venv-mmseg>`` (the val scene list comes from nuscenes-devkit ``nuscenes.utils.splits.val``).  From the
metadata archive the tables are extracted in full; from the LiDAR blob archives only the LIDAR_TOP files of the val
keyframes and of the sweeps the official converter reaches from them (``prev`` up to 10 sample_data records) are
extracted.  Archives are read as streams; nothing else is written.

    <venv-mmseg>/bin/python tools/centerpoint/extract_nuscenes_val.py --downloads DIR --out ROOT [--jobs 5]
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from nuscenes.utils.splits import val as VAL_SCENES

SWEEPS_INFO = 10


def needed_files(root: Path) -> set[str]:
    t = lambda n: json.loads((root / "v1.0-trainval" / f"{n}.json").read_text())  # noqa: E731
    scenes = {s["token"]: s["name"] for s in t("scene")}
    samples = {s["token"]: s for s in t("sample")}
    sd = {r["token"]: r for r in t("sample_data")}
    need = set()
    for r in sd.values():
        if not (r["is_key_frame"] and r["filename"].startswith("samples/LIDAR_TOP/")):
            continue
        if scenes[samples[r["sample_token"]]["scene_token"]] not in VAL_SCENES:
            continue
        need.add(r["filename"])
        rec, n = r, 0
        while n < SWEEPS_INFO and rec["prev"]:
            rec = sd[rec["prev"]]
            need.add(rec["filename"])
            n += 1
    return need


def extract_part(args: tuple[str, str, list[str]]) -> tuple[str, int]:
    archive, out, names = args
    want = set(names)
    n = 0
    with tarfile.open(archive, "r|gz") as tf:
        for m in tf:
            if m.name in want:
                tf.extract(m, out, filter="data")
                n += 1
    return Path(archive).name, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=5)
    a = ap.parse_args()
    dl, out = Path(a.downloads), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "v1.0-trainval" / "sample_data.json").exists():
        with tarfile.open(dl / "v1.0-trainval_meta.tgz", "r:gz") as tf:
            tf.extractall(out, filter="data")
    need = sorted(needed_files(out))
    print(f"val LIDAR_TOP files needed: {len(need)}", flush=True)
    parts = sorted(dl.glob("v1.0-trainval*_blobs_lidar.tgz"))
    with ProcessPoolExecutor(a.jobs) as ex:
        got = list(ex.map(extract_part, [(str(p), str(out), need) for p in parts]))
    total = sum(n for _, n in got)
    missing = [f for f in need if not (out / f).exists()]
    report = {"val_scenes": len(VAL_SCENES), "needed": len(need), "extracted": total, "missing": len(missing),
              "per_archive": dict(got), "first_missing": missing[:5]}
    (out / "extract_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
