#!/usr/bin/env python3
"""Export an mmsegmentation DeepLabV3+ Cityscapes checkpoint to ONNX (offline, CPU, tools only), v0.1.2 DeepLab part.

Run with the system Python (torch CPU and onnx are used by this offline tool only; the benchmark core does not use
them).  Steps:
  1. load the checkpoint with ``torch.load(weights_only=True)`` into the plain-PyTorch re-definition
     (``deeplab_model.py``) with ``strict=True``; ``auxiliary_head.*`` (training only) is left out;
  2. compare the checkpoint's own training config (meta) with the re-definition's structural parameters;
  3. fold every BatchNorm into its conv (float64, stored float32) and export the complete inference function
     (normalized RGB 1x3xHxW float32 -> uint8 label map 1x1xHxW: logits, bilinear resize to HxW, argmax, cast)
     with torch.onnx.export (TorchScript exporter, opset 17, constant folding);
  4. gate 1 (plan v03 section 7.3): logits of the folded torch model vs ONNX Runtime CPU (the same ONNX with the
     logits tensor exposed in memory), ``max|a-b| / max|b|`` per input <= 1e-4, and the label maps.

    python3 tools/deeplab_family/export_onnx.py --model r50 --checkpoint models/deeplabv3plus-r50/<file>.pth \
        --out models/deeplabv3plus-r50/deeplabv3plus_r50.onnx --report runs/.../export_r50.json \
        [--inputs synthetic|PPM_DIR] [--count 8]

``--inputs synthetic`` (until the Cityscapes val images are available) uses deterministic image-like uint8 inputs
normalized with the canonical contract; a PPM directory uses its first ``--count`` files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deeplab_model as D  # noqa: E402
import inputs as I  # noqa: E402

GATE1_MAX_REL = 1e-4


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_checks(meta: dict, v: D.Variant) -> list[dict]:
    """Structural parameters of the re-definition against the checkpoint's own training config text."""
    cfg = meta.get("config", "")
    flat = re.sub(r"\s+", "", cfg)
    want = {
        "backbone.type=ResNetV1c": "type='ResNetV1c'" in flat,
        f"depth={v.depth}": f"depth={v.depth}," in flat,
        "strides=(1,2,1,1)": "strides=(1,2,1,1)" in flat,
        "dilations=(1,1,2,4)": "dilations=(1,1,2,4)" in flat,
        "contract_dilation=True": "contract_dilation=True" in flat,
        "style=pytorch": "style='pytorch'" in flat,
        # 'SepASPPHead' is the name of the same head in mmseg cee9004 (2020-06, the R-50/R-101 training code): same
        # forward; its SeparableConvModule(relu_first=False) = depthwise conv+BN+ReLU, pointwise conv+BN+ReLU
        "decode_head.type=DepthwiseSeparableASPPHead (2020 name SepASPPHead)":
            "type='DepthwiseSeparableASPPHead'" in flat or "type='SepASPPHead'" in flat,
        "aspp dilations=(1,12,24,36)": "dilations=(1,12,24,36)" in flat,
        f"in_channels={v.head_in}": f"in_channels={v.head_in}," in flat,
        f"channels={v.head_channels}": f",channels={v.head_channels}," in flat,
        f"c1_in_channels={v.c1_in}": f"c1_in_channels={v.c1_in}," in flat,
        f"c1_channels={v.c1_channels}": f"c1_channels={v.c1_channels}," in flat,
        "num_classes=19": "num_classes=19," in flat,
        "align_corners=False": "align_corners=False" in flat and "align_corners=True" not in flat,
        "norm_cfg SyncBN (eps default 1e-5)": "norm_cfg=dict(type='SyncBN',requires_grad=True)" in flat
                                               and "eps=" not in flat,
        "test_cfg mode=whole": "test_cfg=dict(mode='whole')" in flat,
        "img_norm mean/std/to_rgb": "mean=[123.675,116.28,103.53],std=[58.395,57.12,57.375],to_rgb=True" in flat,
    }
    return [{"check": k, "ok": bool(ok)} for k, ok in want.items()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(D.VARIANTS))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--inputs", default="synthetic")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    import onnx  # noqa: E402  (tools only)
    import onnxruntime as ort  # noqa: E402

    ckpt = Path(a.checkpoint)
    model = D.DeepLabV3Plus(a.model).eval()
    meta = D.load_checkpoint(model, str(ckpt))
    checks = config_checks(meta, model.variant)
    if not all(c["ok"] for c in checks):
        raise SystemExit(f"checkpoint config differs from the re-definition: {[c for c in checks if not c['ok']]}")
    folded = D.fold_batchnorm(model)
    h, w = model.input_hw
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, h, w)
    with torch.no_grad():
        torch.onnx.export(model, dummy, str(out), opset_version=17, input_names=["input"], output_names=["labels"],
                          do_constant_folding=True, dynamo=False)
    onnx_model = onnx.load(str(out))
    onnx.checker.check_model(onnx_model)
    # expose the logits tensor (input of the final Resize) for the comparison, in memory only
    final_resize = [n for n in onnx_model.graph.node if n.op_type == "Resize"][-1]
    logits_name = final_resize.input[0]
    onnx_model.graph.output.append(onnx.helper.make_tensor_value_info(logits_name, onnx.TensorProto.FLOAT,
                                                                      [1, D.NUM_CLASSES, h // 4, w // 4]))
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(onnx_model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    rows = []
    for name, x in I.iter_inputs(a.inputs, a.count, h, w):
        t = torch.from_numpy(x)
        with torch.no_grad():
            ref_logits = model.logits(t).numpy()
            ref_labels = model(t).numpy()
        labels, logits = sess.run(["labels", logits_name], {"input": x})
        rel = float(np.abs(logits - ref_logits).max() / np.abs(ref_logits).max())
        rows.append({"input": name, "logits_max_rel_diff": rel, "logits_max_abs_diff": float(np.abs(logits - ref_logits).max()),
                     "label_agreement": float((labels == ref_labels).mean()), "labels_dtype": str(labels.dtype)})
    worst = max(r["logits_max_rel_diff"] for r in rows)
    report = {
        "schema": "sdv-bmt-deeplab-export/1",
        "model": a.model,
        "checkpoint": {"file": ckpt.name, "bytes": ckpt.stat().st_size, "sha256": sha256_file(ckpt),
                       "mmseg_version": meta.get("mmseg_version"), "mmcv_version": meta.get("mmcv_version"),
                       "iter": meta.get("iter")},
        "strict_load": "ok (auxiliary_head.* left out)",
        "config_checks": checks,
        "batchnorm_folded": folded,
        "onnx": {"file": out.name, "bytes": out.stat().st_size, "sha256": sha256_file(out), "opset": 17,
                 "logits_tensor": logits_name},
        "tools": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
                  "numpy": np.__version__, "python": sys.version.split()[0],
                  "export_onnx.py": sha256_file(Path(__file__)),
                  "deeplab_model.py": sha256_file(Path(D.__file__)), "inputs.py": sha256_file(Path(I.__file__))},
        "gate1": {"inputs": a.inputs, "count": len(rows), "criterion": f"max_rel <= {GATE1_MAX_REL}",
                  "max_rel": worst, "passed": worst <= GATE1_MAX_REL, "rows": rows},
    }
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("model", "onnx")} | {"gate1_max_rel": worst,
                                                                    "gate1_passed": report["gate1"]["passed"]}))
    return 0 if report["gate1"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
