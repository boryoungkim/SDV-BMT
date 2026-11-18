#!/usr/bin/env python3
"""Calibration data for compiling YOLOX to the Mobilint ARIES NPU with qbcompiler (offline tool).

The calibration tensors must be exactly what an SDV-BMT run stages, so the compiler's quantization ranges are
measured on the runtime transform. For every selected COCO train2017 image:

  1. the JPEG is downloaded once (http://images.cocodataset.org/train2017/<file>) into --cache, or read from it;
  2. Pillow decodes it and converts it to RGB, written as a binary PPM payload, as
     tools/stage2b_g_real/build_coco_subset.py does for the evaluation subset;
  3. the canonical Workload preprocessing (sdv_bmt YoloxCocoWorkload.preprocess, yolox-letterbox-bilinear-bgr/1)
     turns it into the CHW float32 BGR tensor of the workload's input size;
  4. the tensor is transposed to HWC (the NPU input layout) and saved as one float32 .npy file.

No image of the evaluation subset (--exclude-index) may be selected. The selection is a seeded sample of the
annotation file's images sorted by image id, so it is reproducible from (annotation file, seed, count).

Output directory:
  npy/<stem>.npy        HWC float32 tensors (S x S x 3)
  calib_list.txt        absolute paths of the .npy files, one per line (the order of the manifest)
  calib_manifest.json   selection rule, seed, per-image JPEG / payload / tensor SHA-256, the list digest,
                        workload identity, decoder and tool digest (goes into the compile record)

Usage (repo venv with Pillow 11.1.0 and numpy):
  python3 tools/mobilint/make_calib.py --workload configs/workloads/yolox-tiny-coco-v1.yaml \\
      --annotations $COCO_ROOT/annotations/instances_train2017.json \\
      --exclude-index datasets/coco-val2017-first1000/index.jsonl \\
      --count 100 --seed 0 --cache $COCO_ROOT/train2017-calib --out calib/yolox-416
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402  (offline tool)
import yaml  # noqa: E402
from PIL import Image, features  # noqa: E402

from sdv_bmt.contracts.records import InputItem  # noqa: E402
from sdv_bmt.contracts.specs import WorkloadSpec, parse_strict  # noqa: E402
from sdv_bmt.workload.yolox import YoloxCocoWorkload  # noqa: E402

URL = "http://images.cocodataset.org/train2017/{}"
SCHEMA = "sdv-bmt-mobilint-calib/1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, help="workload YAML (gives input_size and the preprocessing)")
    ap.add_argument("--annotations", required=True, help="COCO instances_train2017.json")
    ap.add_argument("--exclude-index", required=True, help="index.jsonl of the evaluation subset")
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", required=True, help="directory holding (or receiving) the train2017 JPEGs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    wl_path = Path(a.workload)
    spec = parse_strict(WorkloadSpec, yaml.safe_load(wl_path.read_text(encoding="utf-8")), str(wl_path))
    workload = YoloxCocoWorkload(spec)
    size = spec.options["input_size"]

    ann_path = Path(a.annotations)
    ann_bytes = ann_path.read_bytes()
    images = sorted(json.loads(ann_bytes)["images"], key=lambda i: i["id"])
    excluded = {json.loads(line)["metadata"]["coco_image_id"]
                for line in Path(a.exclude_index).read_text(encoding="utf-8").splitlines() if line}
    pool = [im for im in images if im["id"] not in excluded]
    chosen = sorted(random.Random(a.seed).sample(pool, a.count), key=lambda i: i["id"])

    out, cache = Path(a.out), Path(a.cache)
    (out / "npy").mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=True)
    decoder = f"Pillow {Image.__version__} (libjpeg-turbo {features.version('libjpeg_turbo')})"
    rows, paths = [], []
    for n, img in enumerate(chosen):
        jpeg_path = cache / img["file_name"]
        if not jpeg_path.exists():
            with urllib.request.urlopen(URL.format(img["file_name"]), timeout=60) as r:
                jpeg_path.write_bytes(r.read())
        jpeg = jpeg_path.read_bytes()
        with Image.open(jpeg_path) as im:
            im.load()
            mode = im.mode
            rgb = im.convert("RGB")
        w, h = rgb.size
        if (w, h) != (img["width"], img["height"]):
            raise SystemExit(f"{img['file_name']}: decoded size {w}x{h} differs from the annotation file")
        payload = f"P6\n{w} {h}\n255\n".encode("ascii") + rgb.tobytes()
        item = InputItem(item_index=n, source_item_id=Path(img["file_name"]).stem, payload_ref=img["file_name"],
                         payload_sha256=sha256(payload), annotation_ref=None, annotation_digest=None,
                         source_timing=None, scenario_coordinates=None,
                         source_metadata={"row_metadata": {"coco_image_id": img["id"], "width": w, "height": h}})
        chw = np.frombuffer(workload.preprocess(payload, item), dtype="<f4").reshape(3, size, size)
        hwc = np.ascontiguousarray(chw.transpose(1, 2, 0))
        npy_path = (out / "npy" / f"{item.source_item_id}.npy").resolve()
        np.save(npy_path, hwc)
        paths.append(str(npy_path))
        rows.append({"coco_image_id": img["id"], "file_name": img["file_name"], "width": w, "height": h,
                     "decoded_mode": mode, "jpeg_sha256": sha256(jpeg), "payload_sha256": item.payload_sha256,
                     "tensor_hwc_f32_sha256": sha256(hwc.tobytes()), "npy": f"npy/{npy_path.name}"})
        print(f"[{n + 1}/{len(chosen)}] {img['file_name']} {w}x{h}", flush=True)

    list_text = "".join(p + "\n" for p in paths)
    (out / "calib_list.txt").write_text(list_text, encoding="utf-8")
    manifest = {
        "schema": SCHEMA,
        "purpose": "qbcompiler calibration data for YOLOX on Mobilint aries-rb (plan decision D07)",
        "workload": {"file": str(wl_path), "workload_id": spec.workload_id, "input_size": size,
                     "preprocess": f"{spec.preprocess.id}/{spec.preprocess.version}",
                     "tensor": f"HWC float32 {size}x{size}x3, BGR, 0..255, letterbox top-left, pad 114"},
        "source": {"annotation_file": str(ann_path), "annotation_file_sha256": sha256(ann_bytes),
                   "image_url": URL.format("<file_name>")},
        "selection": {"rule": "random.Random(seed).sample of images sorted by id, excluding the evaluation subset, "
                              "then sorted by id", "seed": a.seed, "count": a.count,
                      "excluded_index": str(a.exclude_index), "excluded_images": len(excluded)},
        "decoder": decoder,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "tool": {"path": "tools/mobilint/make_calib.py", "sha256": sha256(Path(__file__).read_bytes())},
        "image_list_sha256": sha256("".join(f"{r['coco_image_id']}\n" for r in rows).encode("ascii")),
        "tensors_sha256": sha256(canonical([r["tensor_hwc_f32_sha256"] for r in rows]).encode("utf-8")),
        "images": rows,
    }
    (out / "calib_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8")
    print(canonical({"count": len(rows), "image_list_sha256": manifest["image_list_sha256"],
                     "tensors_sha256": manifest["tensors_sha256"], "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
