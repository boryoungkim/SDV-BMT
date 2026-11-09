#!/usr/bin/env python3
"""Build the v0.1.2 DeepLab dataset: the 500 Cityscapes val images as decoded payloads with frozen ground truth.

Offline asset preparation (outside src/ and the measurement path).  PNG decoding uses the system Pillow (offline
tools only); everything else is the standard library.  Only the val members of the two official zip files are read
(``leftImg8bit/val/<city>/<name>_leftImg8bit.png`` and ``gtFine/val/<city>/<name>_gtFine_labelIds.png``); nothing
else is extracted.

For every val image (sorted by name):
  * the PNG is decoded once to RGB and written as a binary PPM payload (``P6``), whose SHA-256 is the payload
    digest of the index row;
  * the fine ground truth (label ids) is mapped to the 19 Cityscapes train ids (every other id -> 255, ignored),
    coded with ``labelmap-u8-rowrle/1`` (no item index) and stored base64 in the row's ``annotation``
    (``cityscapes-gtfine-trainid-labelmap/1``), so the annotation digest pins it and runs freeze it in the bundle.

The dataset stays private (Cityscapes terms: no redistribution, no third-party access): write it below a
directory only the user can read.  Outputs: ``index.jsonl``, ``payloads/<name>.ppm``, ``dataset_manifest.json``
in --out, and the input configuration ``configs/inputs/cityscapes-val500.yaml`` (index SHA-256).

    python3 tools/deeplab_family/build_cityscapes_val.py --left-zip .../leftImg8bit_trainvaltest.zip \
        --gt-zip .../gtFine_trainvaltest.zip --out datasets/cityscapes-val500
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image, features

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from sdv_bmt.contracts import labelmap  # noqa: E402

SCHEMA = "sdv-bmt-cityscapes-val/1"
ANNOTATION_SCHEMA = "cityscapes-gtfine-trainid-labelmap/1"
WIDTH, HEIGHT = 2048, 1024
# cityscapesscripts helpers/labels.py: label id -> train id for the 19 evaluated classes; every other id is 255
LABEL_ID_TO_TRAIN_ID = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                        26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
TRAIN_ID_TABLE = bytes(LABEL_ID_TO_TRAIN_ID.get(v, 255) for v in range(256))
LEFT = re.compile(r"^leftImg8bit/val/([a-z]+)/(\1_\d{6}_\d{6})_leftImg8bit\.png$")
INPUT_CONFIG = REPO / "configs" / "inputs" / "cityscapes-val500.yaml"


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-zip", required=True)
    ap.add_argument("--gt-zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect", type=int, default=500)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "payloads").mkdir(parents=True, exist_ok=True)
    left, gt = zipfile.ZipFile(a.left_zip), zipfile.ZipFile(a.gt_zip)
    gt_names = set(gt.namelist())
    items = sorted((m.group(2), m.group(1), name) for name in left.namelist() if (m := LEFT.match(name)))
    if len(items) != a.expect:
        raise SystemExit(f"found {len(items)} val images, expected {a.expect}")
    rows, cities = [], {}
    payload_bytes = 0
    for stem, city, member in items:
        gt_member = f"gtFine/val/{city}/{stem}_gtFine_labelIds.png"
        if gt_member not in gt_names:
            raise SystemExit(f"{gt_member} missing from {a.gt_zip}")
        png = left.read(member)
        img = Image.open(io.BytesIO(png))
        mode = img.mode
        rgb = img.convert("RGB")
        if rgb.size != (WIDTH, HEIGHT):
            raise SystemExit(f"{member}: {rgb.size}, expected {WIDTH}x{HEIGHT}")
        payload = f"P6\n{WIDTH} {HEIGHT}\n255\n".encode() + rgb.tobytes()
        (out / "payloads" / f"{stem}.ppm").write_bytes(payload)
        payload_bytes += len(payload)
        gt_png = gt.read(gt_member)
        lab = Image.open(io.BytesIO(gt_png))
        if lab.mode not in ("L", "P") or lab.size != (WIDTH, HEIGHT):
            raise SystemExit(f"{gt_member}: mode {lab.mode} size {lab.size}")
        train = lab.tobytes().translate(TRAIN_ID_TABLE)
        coded = labelmap.encode(train, WIDTH, HEIGHT)
        annotation = {"encoding": labelmap.CODEC_ID, "width": WIDTH, "height": HEIGHT,
                      "labels_b64": base64.b64encode(coded).decode("ascii"),
                      "label_space": "Cityscapes trainId 0..18, 255 = not evaluated",
                      "source": gt_member, "source_png_sha256": sha256(gt_png)}
        rows.append({"source_item_id": stem, "payload_file": f"payloads/{stem}.ppm",
                     "payload_sha256": sha256(payload), "annotation": annotation,
                     "metadata": {"city": city, "png_file": member, "png_bytes": len(png), "png_sha256": sha256(png),
                                  "decoded_mode": mode, "width": WIDTH, "height": HEIGHT,
                                  "payload_format": "ppm-p6-rgb8", "source_format": "PNG"}})
        cities[city] = cities.get(city, 0) + 1
    index = "".join(canonical(r) + "\n" for r in rows).encode("utf-8")
    (out / "index.jsonl").write_bytes(index)
    index_sha = sha256(index)
    manifest = {
        "schema": SCHEMA, "dataset": "Cityscapes val (leftImg8bit, gtFine label ids -> train ids)",
        "annotation_schema": ANNOTATION_SCHEMA,
        "source": {"left_zip": {"file": Path(a.left_zip).name, "sha256": sha256_file(Path(a.left_zip)),
                                "bytes": Path(a.left_zip).stat().st_size},
                   "gt_zip": {"file": Path(a.gt_zip).name, "sha256": sha256_file(Path(a.gt_zip)),
                              "bytes": Path(a.gt_zip).stat().st_size}},
        "selection": {"count": len(rows), "rule": "every val image, sorted by name", "cities": cities},
        "label_id_to_train_id": {str(k): v for k, v in LABEL_ID_TO_TRAIN_ID.items()},
        "decoder": f"Pillow {Image.__version__} (zlib {features.version('zlib')})",
        "python": sys.version.split()[0],
        "tool": {"path": "tools/deeplab_family/build_cityscapes_val.py", "sha256": sha256_file(Path(__file__))},
        "index_sha256": index_sha, "totals": {"payload_bytes": payload_bytes, "index_bytes": len(index)},
        "terms": "Cityscapes license: non-commercial research use; no redistribution of the data or of derived data "
                 "from which it can be recovered; keep this directory private",
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    rel = Path("../..") / Path(a.out)
    INPUT_CONFIG.write_text(f"""# v0.1.2 DeepLab input: the 500 Cityscapes val images (fine annotations), built by
# tools/deeplab_family/build_cityscapes_val.py from the official leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip
# (downloaded by the user under the Cityscapes terms; the dataset stays private and is not redistributed).
# Payloads are the decoded RGB pixels (PPM, decoded once offline by Pillow); the fine ground truth, mapped to the 19 train
# ids (255 = not evaluated), is inline in the index (labelmap-u8-rowrle/1, base64).
schema_version: 1
input_kind: dataset
input_id: cityscapes-val500
version: "1"
family:
  source_dataset: cityscapes-val-gtfine
  source_version: "1"
  task: cityscapes-semantic-segmentation-19class
  annotation_schema: {ANNOTATION_SCHEMA}
  preprocessing_contract: cityscapes-rgb-norm-chw2048x1024-f32/1
source:
  index: {rel.as_posix()}/index.jsonl
  index_sha256: {index_sha}
selection:
  start: 0
  count: {len(rows)}
""", encoding="utf-8")
    print(canonical({"items": len(rows), "cities": cities, "index_sha256": index_sha, "payload_bytes": payload_bytes,
                     "index_bytes": len(index)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
