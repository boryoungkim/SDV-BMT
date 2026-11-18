#!/usr/bin/env python3
"""Cut a YOLOX ONNX model at the nine head-convolution outputs for the Mobilint ARIES NPU (offline tool).

Why: the tail of the YOLOX head (per-level Sigmoid on objectness and classes, Concat to 85 channels, Reshape,
Concat over levels, Transpose) has no weights and puts box offsets and probabilities into one tensor. The graph is
therefore cut at the head convolutions (per level: box 4, objectness logit 1, class
logits 80), the last point before outputs with different value ranges are concatenated, and the tail is applied
on the host (host_tail below; the C++ backend does the same in collect()).

The cut points are found from the graph structure, not from node names, so the same code works for Tiny/S/M/L/X:
the last Transpose <- Concat over levels <- one Reshape per level <- Concat [reg, Sigmoid(obj), Sigmoid(cls)].

Check: ONNX Runtime runs the original and the cut model on the same inputs (seeded random tensors and, with
--calib-dir, real calibration tensors); host_tail(cut outputs) must equal the original output up to float rounding.
The maximum absolute difference, both SHA-256 values and the output layout go into <out>.cut.json for the compile
record.

Environment: a Python with onnx, onnxruntime and numpy (for example the qbcompiler environment).

    python3 tools/mobilint/cut_head.py --onnx ~/sdv_assets/yolox_onnx/yolox_s.onnx \\
        --out ~/sdv_assets/compile/head9/yolox_s_head9.onnx --calib-dir ~/sdv_assets/calib/yolox-640/npy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SCHEMA = "sdv-bmt-mobilint-cut/1"
BRANCHES = (("reg", 4), ("obj", 1), ("cls", 80))   # order of the per-level Concat in the YOLOX export


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sigmoid(v: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-v))).astype(np.float32)


def host_tail(levels: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """Rebuild the YOLOX `output` (anchors x 85, float32) from the nine head outputs.

    levels: per stride in the original Concat order (8, 16, 32), the tuple (reg, obj logit, cls logits), each CHW
    with shapes (4, g, g), (1, g, g), (80, g, g). Same arithmetic as the removed ONNX tail.
    """
    per_level = [np.concatenate([reg, sigmoid(obj), sigmoid(cls)], axis=0).reshape(85, -1)
                 for reg, obj, cls in levels]
    return np.ascontiguousarray(np.concatenate(per_level, axis=1).T, dtype=np.float32)


def find_head_outputs(model) -> list[str]:
    """Names of the nine cut tensors, ordered level by level as (reg, obj logit, cls logits)."""
    prod = {o: n for n in model.graph.node for o in n.output}
    graph_out = model.graph.output
    if len(graph_out) != 1:
        raise SystemExit(f"expected one graph output, got {[o.name for o in graph_out]}")
    tr = prod[graph_out[0].name]
    if tr.op_type != "Transpose":
        raise SystemExit(f"graph output is produced by {tr.op_type}, expected Transpose")
    cat = prod[tr.input[0]]
    if cat.op_type != "Concat" or len(cat.input) != 3:
        raise SystemExit(f"expected Concat of 3 levels before the Transpose, got {cat.op_type}/{len(cat.input)}")
    names = []
    for r in cat.input:
        rs = prod[r]
        if rs.op_type not in ("Reshape", "Flatten"):
            raise SystemExit(f"level tensor {r} produced by {rs.op_type}, expected Reshape")
        lvl = prod[rs.input[0]]
        if lvl.op_type != "Concat" or len(lvl.input) != 3:
            raise SystemExit(f"expected per-level Concat [reg, sig(obj), sig(cls)], got {lvl.op_type}")
        reg, so, sc = lvl.input
        if prod[so].op_type != "Sigmoid" or prod[sc].op_type != "Sigmoid" or prod[reg].op_type == "Sigmoid":
            raise SystemExit(f"unexpected per-level inputs: {[prod[x].op_type for x in lvl.input]}")
        names += [reg, prod[so].input[0], prod[sc].input[0]]
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="original YOLOX ONNX file")
    ap.add_argument("--out", required=True, help="cut ONNX file to write (<out>.cut.json is written next to it)")
    ap.add_argument("--calib-dir", help="directory of HWC float32 .npy tensors to include in the check")
    ap.add_argument("--calib-count", type=int, default=8)
    ap.add_argument("--random", type=int, default=3, help="number of seeded random inputs in the check")
    a = ap.parse_args()

    import onnx
    import onnxruntime as ort
    from onnx import utils

    src, out = Path(a.onnx).expanduser(), Path(a.out).expanduser()
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first")
    out.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(src))
    in_name = model.graph.input[0].name
    dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    if len(dims) != 4 or dims[:2] != [1, 3] or dims[2] != dims[3]:
        raise SystemExit(f"input {in_name} has shape {dims}, expected [1, 3, S, S]")
    size = dims[2]
    names = find_head_outputs(model)
    del model
    utils.extract_model(str(src), str(out), [in_name], names)

    full = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    cut = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    if [o.name for o in cut.get_outputs()] != names:
        raise SystemExit("cut model output order differs from the requested order")

    inputs = [("random-seed%d" % s, np.random.RandomState(s).uniform(0, 255, (1, 3, size, size)).astype(np.float32))
              for s in range(a.random)]
    if a.calib_dir:
        for p in sorted(Path(a.calib_dir).expanduser().glob("*.npy"))[: a.calib_count]:
            hwc = np.load(p)
            inputs.append((p.name, np.ascontiguousarray(hwc.transpose(2, 0, 1))[np.newaxis].astype(np.float32)))

    layout, checks = None, []
    for label, x in inputs:
        ref = full.run(None, {in_name: x})[0][0]
        h = cut.run(None, {in_name: x})
        shapes = [list(t.shape) for t in h]
        levels = []
        for k in range(3):
            trio = h[3 * k: 3 * k + 3]
            for (br, ch), t in zip(BRANCHES, trio):
                if t.shape[0] != 1 or t.shape[1] != ch or t.shape[2] != t.shape[3]:
                    raise SystemExit(f"level {k} {br}: shape {t.shape}, expected (1, {ch}, g, g)")
            levels.append(tuple(t[0] for t in trio))
        rebuilt = host_tail(levels)
        if rebuilt.shape != ref.shape:
            raise SystemExit(f"rebuilt {rebuilt.shape} vs original {ref.shape}")
        checks.append({"input": label, "max_abs_diff": float(np.abs(rebuilt - ref).max())})
        if layout is None:
            layout = [{"level": k, "stride": size // shapes[3 * k][2], "grid": shapes[3 * k][2],
                       "outputs": [{"branch": br, "name": names[3 * k + j], "shape_nchw": shapes[3 * k + j]}
                                   for j, (br, _) in enumerate(BRANCHES)]} for k in range(3)]
        print(f"{label}: max|host_tail(cut) - original| = {checks[-1]['max_abs_diff']:.3e}", flush=True)

    anchors = sum(lv["grid"] ** 2 for lv in layout)
    record = {
        "schema": SCHEMA,
        "reason": "cut at the head convolutions; the weight-free head tail "
                  "(Sigmoid/Concat/Reshape/Transpose) is applied on the host",
        "source": {"file": str(src), "sha256": file_sha256(src)},
        "cut": {"file": str(out), "sha256": file_sha256(out)},
        "input": {"name": in_name, "shape_nchw": dims},
        "output_names_in_order": names,
        "layout": layout,
        "anchors": anchors,
        "host_tail": "per level concat[reg, sigmoid(obj), sigmoid(cls)] -> (85, g*g); concat levels in stride "
                     "order 8, 16, 32 -> (85, anchors); transpose -> (anchors, 85) float32",
        "check": {"inputs": len(checks), "max_abs_diff": max(c["max_abs_diff"] for c in checks), "per_input": checks},
        "versions": {"onnx": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__,
                     "python": sys.version.split()[0]},
        "tool": {"path": "tools/mobilint/cut_head.py", "sha256": file_sha256(Path(__file__))},
    }
    rec_path = out.parent / (out.stem + ".cut.json")
    rec_path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cut_sha256": record["cut"]["sha256"], "anchors": anchors,
                      "max_abs_diff": record["check"]["max_abs_diff"], "record": str(rec_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
