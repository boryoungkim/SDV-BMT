#!/usr/bin/env python3
"""Official-implementation cross-check of the DeepLabV3+ re-definition (plan v03 section 7.2, option B; tools only).

Run with a dedicated environment ``<venv-mmseg>`` (torch 2.1.2 CPU, mmcv 2.1.0 official CPU wheel,
mmengine 0.10.7, mmsegmentation 1.2.2), never with the benchmark core.  For each image file:
  * official: mmsegmentation 1.2.2 builds the model from its own packaged config
    (``deeplabv3plus_<m>-d8_4xb2-80k_cityscapes-512x1024.py``) with ``init_model(config, None)``; the checkpoint's
    state_dict is loaded with ``torch.load(weights_only=True)`` and ``strict=True`` (no pickle code execution); the
    official test pipeline (LoadImageFromFile, Resize (2048, 1024) keep_ratio, PackSegInputs) and data preprocessor
    produce the input tensor, and ``model.predict`` gives the full-resolution class logits and the label map;
  * re-definition (``deeplab_model.py``) on the same input tensor: full-resolution logits before and after folding
    BatchNorm, and the label map;
  * the canonical preprocessing contract (``sdv_bmt.workload.deeplab.normalize_rgb``) on the decoded RGB pixels is
    compared byte for byte with the official input tensor.

Criteria (plan v03 7.2): per image ``max|official - re-definition| / max|official|`` <= 1e-5 (unfused) and label
agreement >= 0.9999; reported for the folded model as well.

    <venv-mmseg>/bin/python tools/deeplab_family/official_crosscheck.py --model r50 \
        --checkpoint models/deeplabv3plus-r50/<file>.pth (--synthetic N | --images DIR --count N) --out REPORT.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deeplab_model as D  # noqa: E402
import inputs as I  # noqa: E402

REL_MAX = 1e-5
AGREE_MIN = 0.9999


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(D.VARIANTS))
    ap.add_argument("--checkpoint", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", type=int)
    src.add_argument("--images")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    import mmcv  # noqa: E402
    import mmengine  # noqa: E402
    import mmseg  # noqa: E402
    from mmseg.apis import init_model  # noqa: E402
    from mmseg.apis.inference import _preprare_data  # noqa: E402
    from PIL import Image  # noqa: E402

    cfg_path = Path(mmseg.__file__).parent / ".mim" / "configs" / "deeplabv3plus" / \
        f"deeplabv3plus_{a.model}-d8_4xb2-80k_cityscapes-512x1024.py"
    official = init_model(str(cfg_path), None, device="cpu")
    official.eval()
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    official.load_state_dict(state, strict=True)
    ours = D.DeepLabV3Plus(a.model).eval()
    D.load_checkpoint(ours, a.checkpoint)
    fused = copy.deepcopy(ours)
    D.fold_batchnorm(fused)
    h, w = ours.input_hw

    tmp = Path(tempfile.mkdtemp(prefix="sdv-bmt-official-"))
    if a.synthetic:
        files = []
        for i in range(a.synthetic):
            f = tmp / f"synthetic-{i}.png"
            Image.frombytes("RGB", (w, h), I.synthetic_rgb(i, h, w)).save(f)
            files.append(f)
    else:
        files = sorted(Path(a.images).rglob("*.png"))[:a.count]
        if len(files) < a.count:
            raise SystemExit(f"{a.images}: fewer than {a.count} PNG files")
    rows = []
    t0 = time.time()
    for f in files:
        data, _ = _preprare_data(str(f), official)
        data = official.data_preprocessor(data, False)
        x = data["inputs"]
        with torch.no_grad():
            pred = official.predict(x, data["data_samples"])[0]
            off_logits = pred.seg_logits.data.unsqueeze(0).double()
            off_labels = pred.pred_sem_seg.data.squeeze().to(torch.uint8)
            mine = ours.full_logits(x).double()
            mine_f = fused.full_logits(x).double()
        rgb = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8).tobytes()
        contract = torch.from_numpy(I.tensor_from_rgb(rgb, h, w))
        scale = off_logits.abs().max()
        row = {"image": f.name,
               "input_tensor_equals_contract": bool(torch.equal(contract, x.float())),
               "input_tensor_max_abs_diff": float((contract - x.float()).abs().max()),
               "rel_unfused": float((mine - off_logits).abs().max() / scale),
               "rel_folded": float((mine_f - off_logits).abs().max() / scale),
               "label_agreement_unfused": float((mine.argmax(1).squeeze().to(torch.uint8) == off_labels).double().mean()),
               "label_agreement_folded": float((mine_f.argmax(1).squeeze().to(torch.uint8) == off_labels).double().mean())}
        rows.append(row)
        print(f"[{time.time() - t0:7.1f} s] {f.name} rel {row['rel_unfused']:.3e} / folded {row['rel_folded']:.3e}",
              flush=True)
    worst = max(r["rel_unfused"] for r in rows)
    agree = min(r["label_agreement_unfused"] for r in rows)
    report = {
        "schema": "sdv-bmt-deeplab-official-crosscheck/1", "model": a.model,
        "checkpoint": {"file": Path(a.checkpoint).name, "sha256": sha256_file(Path(a.checkpoint))},
        "official": {"mmsegmentation": mmseg.__version__, "mmcv": mmcv.__version__, "mmengine": mmengine.__version__,
                     "torch": torch.__version__, "numpy": np.__version__, "config": str(cfg_path.name),
                     "config_sha256": sha256_file(cfg_path), "strict_load": "ok (weights_only=True)"},
        "tools": {"official_crosscheck.py": sha256_file(Path(__file__)), "deeplab_model.py": sha256_file(Path(D.__file__)),
                  "inputs.py": sha256_file(Path(I.__file__))},
        "inputs": "synthetic" if a.synthetic else str(a.images), "count": len(rows),
        "criteria": {"rel_unfused_max": REL_MAX, "label_agreement_min": AGREE_MIN},
        "rel_unfused_max": worst, "rel_folded_max": max(r["rel_folded"] for r in rows),
        "label_agreement_unfused_min": agree,
        "label_agreement_folded_min": min(r["label_agreement_folded"] for r in rows),
        "input_tensor_equals_contract_all": all(r["input_tensor_equals_contract"] for r in rows),
        "passed": worst <= REL_MAX and agree >= AGREE_MIN, "rows": rows,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    print(json.dumps({k: report[k] for k in ("model", "rel_unfused_max", "rel_folded_max", "label_agreement_unfused_min",
                                             "input_tensor_equals_contract_all", "passed")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
