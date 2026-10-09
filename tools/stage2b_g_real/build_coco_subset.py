#!/usr/bin/env python3
"""Build the Stage 2b-G real-model dataset: the first N COCO val2017 images (by image id) as decoded payloads.

Offline asset preparation (outside src/ and outside the measurement path).  The JPEG decode uses the system
Pillow (approved 2026-10-08 for offline tools only); everything else is the Python standard library.

For every selected image:
  * the original JPEG is read from the COCO directory and its SHA-256 recorded;
  * Pillow decodes it once and converts it to RGB (grayscale/CMYK are converted by Pillow's ``convert``);
  * the RGB pixels are written as a binary PPM (``P6``) payload file, whose SHA-256 becomes the payload
    digest of the dataset index row;
  * the image's ground-truth objects are copied verbatim from ``instances_val2017.json`` into the row's
    ``annotation`` (so the annotation digest pins them).

Output directory layout:
  index.jsonl             one canonical-JSON row per image, ordered by image id
  payloads/<stem>.ppm     decoded RGB payloads
  dataset_manifest.json   source digests, selection rule, decoder identity, categories, totals

Usage:
  python3 tools/stage2b_g_real/build_coco_subset.py --images $COCO_ROOT/val2017 \
      --annotations $COCO_ROOT/annotations/instances_val2017.json --count 1000 \
      --out datasets/coco-val2017-first1000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, features

SCHEMA = "sdv-bmt-coco-subset/1"
ANNOTATION_SCHEMA = "coco-2017-instances-bbox/1"


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ann_path = Path(args.annotations)
    ann_bytes = ann_path.read_bytes()
    coco = json.loads(ann_bytes)
    images = sorted(coco["images"], key=lambda i: i["id"])[: args.count]
    if len(images) != args.count:
        raise SystemExit(f"only {len(images)} images available")
    by_image: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    categories = sorted(({"id": c["id"], "name": c["name"], "supercategory": c["supercategory"]}
                         for c in coco["categories"]), key=lambda c: c["id"])

    out = Path(args.out)
    payload_dir = out / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=False)
    decoder = f"Pillow {Image.__version__} (libjpeg-turbo {features.version('libjpeg_turbo')})"
    rows = []
    totals = {"jpeg_bytes": 0, "payload_bytes": 0, "objects": 0, "crowd_objects": 0, "images_without_objects": 0,
              "decoded_modes": {}}
    for img in images:
        jpeg_path = Path(args.images) / img["file_name"]
        jpeg = jpeg_path.read_bytes()
        with Image.open(jpeg_path) as im:
            im.load()
            mode = im.mode
            fmt = im.format
            rgb = im.convert("RGB")
        w, h = rgb.size
        if (w, h) != (img["width"], img["height"]):
            raise SystemExit(f"{img['file_name']}: decoded size {w}x{h} differs from the annotation file")
        payload = f"P6\n{w} {h}\n255\n".encode("ascii") + rgb.tobytes()
        stem = Path(img["file_name"]).stem
        rel = f"payloads/{stem}.ppm"
        (out / rel).write_bytes(payload)
        objects = [{"id": a["id"], "category_id": a["category_id"], "bbox": a["bbox"], "area": a["area"],
                    "iscrowd": a["iscrowd"]} for a in sorted(by_image.get(img["id"], []), key=lambda a: a["id"])]
        totals["jpeg_bytes"] += len(jpeg)
        totals["payload_bytes"] += len(payload)
        totals["objects"] += len(objects)
        totals["crowd_objects"] += sum(o["iscrowd"] for o in objects)
        totals["images_without_objects"] += not objects
        totals["decoded_modes"][mode] = totals["decoded_modes"].get(mode, 0) + 1
        rows.append({
            "source_item_id": stem,
            "payload_file": rel,
            "payload_sha256": sha256(payload),
            "annotation": {"image_id": img["id"], "file_name": img["file_name"], "width": w, "height": h,
                           "objects": objects},
            "metadata": {"coco_image_id": img["id"], "width": w, "height": h, "jpeg_file": img["file_name"],
                         "jpeg_sha256": sha256(jpeg), "jpeg_bytes": len(jpeg), "source_format": fmt,
                         "decoded_mode": mode, "decoder": decoder, "payload_format": "ppm-p6-rgb8"},
        })
    index_text = "".join(canonical(r) + "\n" for r in rows)
    (out / "index.jsonl").write_text(index_text, encoding="utf-8")
    manifest = {
        "schema": SCHEMA,
        "dataset": "COCO 2017 val (instances)",
        "source": {"images_dir": str(Path(args.images)), "annotation_file": str(ann_path),
                   "annotation_file_sha256": sha256(ann_bytes), "annotation_info": coco.get("info")},
        "selection": {"rule": "first N images by ascending image id", "count": args.count,
                      "first_image_id": images[0]["id"], "last_image_id": images[-1]["id"]},
        "annotation_schema": ANNOTATION_SCHEMA,
        "decoder": decoder,
        "python": sys.version.split()[0],
        "tool": {"path": "tools/stage2b_g_real/build_coco_subset.py", "sha256": sha256(Path(__file__).read_bytes())},
        "index_sha256": sha256(index_text.encode("utf-8")),
        "categories": categories,
        "totals": totals,
        "licenses_note": "COCO annotations CC BY 4.0; images under their Flickr licenses (see annotation file)",
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(canonical({"index_sha256": manifest["index_sha256"], "rows": len(rows), "totals": totals}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
