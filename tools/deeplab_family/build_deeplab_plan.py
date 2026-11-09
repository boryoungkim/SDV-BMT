#!/usr/bin/env python3
"""Convert an exported DeepLabV3+ ONNX model into SDV-BMT cuDNN execution plans + weight blob (offline, stdlib only).

v0.1.2 DeepLab part.  The ONNX file (``export_onnx.py``) is the model artifact (identity = its SHA-256).  This tool
derives, deterministically, what the native NVIDIA backend executes with cuDNN and its glue kernels:

  <out>.plan.json        canonical plan: normalized RGB 1x3xHxW float32 -> uint8 label map HxW (the task output)
  <out>.diag.plan.json   diagnostic plan for the numeric gate only: the same steps up to the class logits
                         (1x19xH/4xW/4 float32, written to the output buffer); the final resize and argmax are left
                         out.  Same weights, same arena layout, same convolutions; only the last convolution's
                         destination differs.
  <out>.weights.f32      convolution weights and biases (BatchNorm already folded at export), float32 LE, 256-aligned

Graph patterns recognised (anything else is rejected, never approximated):
  * Conv [-> Relu] -> step ``conv`` act none / ReLU (3); Conv -> Add(that conv, residual) -> Relu -> ``conv`` with
    ``res_before_act`` (ReLU(conv + bias + residual), the addition order of the graph); grouped (depthwise) and
    dilated convolutions keep their group / dilation; pads must be symmetric
  * MaxPool (no ceil mode, no dilation) -> ``maxpool``;  GlobalAveragePool -> ``gap``
  * Resize mode linear, coordinate transformation half_pixel / pytorch_half_pixel, output sizes from constants
    (directly, or Concat(Slice(Shape(t), 0, 2), Constant[h, w]) with t's batch and channels) -> ``resize_bilinear``;
    from a 1x1 input -> ``broadcast`` (every output equals the single input value)
  * Concat on the channel axis -> zero-copy: producers write into channel slices of the concat tensor
  * ArgMax(axis 1, keepdims, first index) -> Cast(uint8) -> graph output -> ``argmax_u8`` into the output buffer
  * Shape / Slice / Constant / Concat(axis 0) nodes are allowed only inside such a resize-size expression

Usage: python3 tools/deeplab_family/build_deeplab_plan.py models/deeplabv3plus-r50/deeplabv3plus_r50.onnx \
           models/deeplabv3plus-r50/deeplabv3plus_r50
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage2b_g_real"))
import onnx_reader as R  # noqa: E402

SCHEMA = "sdv-bmt-cudnn-plan/1"
ALIGN = 256
NUM_CLASSES = 19
INT64_MAX = (1 << 63) - 1


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Fail(Exception):
    pass


def align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def convert(onnx_path: Path) -> tuple[dict, dict, bytes]:
    onnx_bytes = onnx_path.read_bytes()
    m = R.load(str(onnx_path))
    if m.opsets.get("", 0) != 17:
        raise Fail(f"expected opset 17, got {m.opsets}")
    inits = m.initializers
    graph_inputs = [i for i in m.inputs if i.name not in inits]
    if len(graph_inputs) != 1 or len(m.outputs) != 1:
        raise Fail("expected one graph input and one output")
    gin, gout = graph_inputs[0], m.outputs[0]
    in_shape = tuple(gin.shape)
    if (len(in_shape) != 4 or not all(isinstance(d, int) and d > 0 for d in in_shape) or in_shape[:2] != (1, 3)
            or gin.elem_type != R.FLOAT or in_shape[2] % 8 or in_shape[3] % 8):
        raise Fail(f"unexpected input {gin}")
    H, W = in_shape[2], in_shape[3]
    if tuple(gout.shape) != (1, 1, H, W) or gout.elem_type != R.UINT8:
        raise Fail(f"unexpected output {gout}: expected uint8 1x1x{H}x{W}")

    producer: dict[str, int] = {}
    consumers: dict[str, list[int]] = {}
    for k, n in enumerate(m.nodes):
        for o in n.outputs:
            producer[o] = k
        for i in n.inputs:
            if i:
                consumers.setdefault(i, []).append(k)
    const: dict[str, list[int]] = {}
    for n in m.nodes:
        if n.op_type == "Constant":
            t = n.attrs.get("value")
            if not isinstance(t, R.Tensor) or t.data_type != R.INT64 or len(n.outputs) != 1:
                raise Fail(f"constant {n.outputs}: only int64 tensors are supported")
            const[n.outputs[0]] = t.values()

    def ints(name: str) -> list[int]:
        if name in const:
            return const[name]
        t = inits.get(name)
        if t is None or t.data_type != R.INT64:
            raise Fail(f"{name}: expected an int64 constant")
        return t.values()

    shape: dict[str, tuple[int, ...]] = {gin.name: in_shape}
    fused: set[int] = set()
    steps: list[dict] = []
    weights = bytearray()
    concat_nodes: list[int] = []

    def add_weight(t: R.Tensor) -> int:
        if t.data_type != R.FLOAT:
            raise Fail(f"{t.name}: weights must be float32")
        weights.extend(b"\0" * (align(len(weights)) - len(weights)))
        off = len(weights)
        weights.extend(t.raw)
        return off

    def resize_sizes(n: R.Node) -> list[int]:
        """Output sizes of a Resize: a constant, or Concat(Slice(Shape(t), [0], [2], [0]), Constant[h, w])."""
        if len(n.inputs) != 4 or n.inputs[1] or n.inputs[2] or not n.inputs[3]:
            raise Fail(f"resize {n.outputs}: only the sizes input is supported")
        s = n.inputs[3]
        if s in const or s in inits:
            return ints(s)
        cat = m.nodes[producer[s]]
        if cat.op_type != "Concat" or cat.attrs.get("axis") != 0 or len(cat.inputs) != 2:
            raise Fail(f"resize {n.outputs}: unsupported sizes expression")
        sl = m.nodes[producer[cat.inputs[0]]]
        if sl.op_type != "Slice" or len(sl.inputs) != 4 or ints(sl.inputs[1]) != [0] or ints(sl.inputs[2]) != [2] \
                or ints(sl.inputs[3]) != [0]:
            raise Fail(f"resize {n.outputs}: unsupported sizes slice")
        sh = m.nodes[producer[sl.inputs[0]]]
        if sh.op_type != "Shape" or sh.inputs[0] not in shape:
            raise Fail(f"resize {n.outputs}: sizes must come from the shape of a known tensor")
        for k in (producer[s], producer[cat.inputs[0]], producer[sl.inputs[0]]):
            if any(c not in (producer[s], producer[cat.inputs[0]]) and m.nodes[c].op_type != "Resize"
                   for o in m.nodes[k].outputs for c in consumers.get(o, [])):
                raise Fail(f"resize {n.outputs}: a sizes node has other consumers")
            fused.add(k)
        return list(shape[sh.inputs[0]][:2]) + ints(cat.inputs[1])

    # ---------------------------------------------------------------- output: ArgMax -> Cast(uint8)
    cast = m.nodes[producer[gout.name]]
    if cast.op_type != "Cast" or cast.attrs.get("to") != R.UINT8:
        raise Fail("output: expected Cast(to uint8)")
    argmax = m.nodes[producer[cast.inputs[0]]]
    if (argmax.op_type != "ArgMax" or argmax.attrs.get("axis") != 1 or argmax.attrs.get("keepdims", 1) != 1
            or argmax.attrs.get("select_last_index", 0) != 0):
        raise Fail("output: expected ArgMax(axis=1, keepdims=1, first index) before the Cast")
    fused.update({producer[gout.name], producer[cast.inputs[0]]})
    final_resize = m.nodes[producer[argmax.inputs[0]]]
    if final_resize.op_type != "Resize":
        raise Fail("output: expected the class logits resized to the input size before the ArgMax")
    logits_name = final_resize.inputs[0]

    # ---------------------------------------------------------------- main pass (graph order is topological)
    for k, n in enumerate(m.nodes):
        if k in fused or n.op_type in ("Constant", "Shape", "Slice"):
            continue
        op = n.op_type
        if op == "Conv":
            x, wname, bname = n.inputs
            Wt, Bt = inits[wname], inits[bname]
            K, Cg, Rk, Sk = Wt.dims
            group = n.attrs.get("group", 1)
            pads = n.attrs.get("pads", [0, 0, 0, 0])
            strides = n.attrs.get("strides", [1, 1])
            dil = n.attrs.get("dilations", [1, 1])
            if n.attrs.get("kernel_shape") != [Rk, Sk] or pads[0] != pads[2] or pads[1] != pads[3] \
                    or n.attrs.get("auto_pad", b"NOTSET") != b"NOTSET":
                raise Fail(f"conv {n.outputs}: unsupported geometry")
            _, C, Hx, Wx = shape[x]
            if C != Cg * group or Bt.dims != (K,):
                raise Fail(f"conv {n.outputs}: channel mismatch")
            OH = (Hx + 2 * pads[0] - dil[0] * (Rk - 1) - 1) // strides[0] + 1
            OW = (Wx + 2 * pads[1] - dil[1] * (Sk - 1) - 1) // strides[1] + 1
            o = n.outputs[0]
            act, out, residual = 0, o, None
            cons = consumers.get(o, [])
            if len(cons) == 1 and m.nodes[cons[0]].op_type == "Relu":
                fused.add(cons[0])
                act, out = 3, m.nodes[cons[0]].outputs[0]
            elif len(cons) == 1 and m.nodes[cons[0]].op_type == "Add" and m.nodes[cons[0]].inputs[0] == o:
                add = m.nodes[cons[0]]
                after = consumers.get(add.outputs[0], [])
                other = add.inputs[1]
                if (len(after) != 1 or m.nodes[after[0]].op_type != "Relu" or other not in shape
                        or shape[other] != (1, K, OH, OW)):
                    raise Fail(f"conv {o}: unrecognised residual pattern")
                fused.update({cons[0], after[0]})
                act, residual, out = 3, other, m.nodes[after[0]].outputs[0]
            shape[out] = (1, K, OH, OW)
            step = {"op": "conv", "x": x, "y": out, "residual": residual, "act": act,
                    "w_offset": add_weight(Wt), "b_offset": add_weight(Bt),
                    "c": C, "h": Hx, "w": Wx, "k": K, "r": Rk, "s": Sk, "oh": OH, "ow": OW,
                    "pad_h": pads[0], "pad_w": pads[1], "stride_h": strides[0], "stride_w": strides[1],
                    "dil_h": dil[0], "dil_w": dil[1], "group": group, "onnx_node": o}
            if residual is not None:
                step["res_before_act"] = True
            steps.append(step)
        elif op == "MaxPool":
            ksz, pads, strides = n.attrs["kernel_shape"], n.attrs["pads"], n.attrs["strides"]
            if (n.attrs.get("ceil_mode", 0) != 0 or n.attrs.get("dilations", [1, 1]) != [1, 1] or len(set(pads)) != 1
                    or ksz[0] != ksz[1] or strides[0] != strides[1]):
                raise Fail(f"maxpool {n.outputs}: unsupported")
            x = n.inputs[0]
            _, C, Hx, Wx = shape[x]
            OH = (Hx + 2 * pads[0] - ksz[0]) // strides[0] + 1
            OW = (Wx + 2 * pads[1] - ksz[1]) // strides[1] + 1
            shape[n.outputs[0]] = (1, C, OH, OW)
            steps.append({"op": "maxpool", "x": x, "y": n.outputs[0], "c": C, "h": Hx, "w": Wx, "oh": OH, "ow": OW,
                          "window": ksz[0], "pad": pads[0], "stride": strides[0]})
        elif op == "GlobalAveragePool":
            x = n.inputs[0]
            _, C, Hx, Wx = shape[x]
            shape[n.outputs[0]] = (1, C, 1, 1)
            steps.append({"op": "gap", "x": x, "y": n.outputs[0], "c": C, "h": Hx, "w": Wx})
        elif op == "Resize":
            if n.attrs.get("mode") != b"linear" or n.attrs.get("coordinate_transformation_mode") not in (
                    b"half_pixel", b"pytorch_half_pixel") or n.attrs.get("antialias", 0) != 0:
                raise Fail(f"resize {n.outputs}: only bilinear half-pixel resize is supported")
            x = n.inputs[0]
            _, C, Hx, Wx = shape[x]
            sizes = resize_sizes(n)
            if len(sizes) != 4 or sizes[:2] != [1, C] or min(sizes) <= 0:
                raise Fail(f"resize {n.outputs}: unsupported sizes {sizes}")
            OH, OW = sizes[2], sizes[3]
            shape[n.outputs[0]] = (1, C, OH, OW)
            if (Hx, Wx) == (1, 1):
                steps.append({"op": "broadcast", "x": x, "y": n.outputs[0], "c": C, "oh": OH, "ow": OW})
            else:
                steps.append({"op": "resize_bilinear", "x": x, "y": n.outputs[0], "c": C, "h": Hx, "w": Wx,
                              "oh": OH, "ow": OW})
        elif op == "Concat" and n.attrs.get("axis") == 0 and all(
                m.nodes[c].op_type == "Resize" and m.nodes[c].inputs[3:] == [n.outputs[0]]
                for c in consumers.get(n.outputs[0], [])):
            continue  # a resize-size expression, checked (and fused) by resize_sizes() at its Resize
        elif op == "Concat":
            if n.attrs.get("axis") != 1:
                raise Fail(f"concat {n.outputs}: only the channel axis is supported")
            shapes = [shape[i] for i in n.inputs]
            _, _, Hx, Wx = shapes[0]
            if any(s[0] != 1 or s[2:] != (Hx, Wx) for s in shapes):
                raise Fail(f"concat {n.outputs}: shape mismatch")
            shape[n.outputs[0]] = (1, sum(s[1] for s in shapes), Hx, Wx)
            concat_nodes.append(k)
        else:
            raise Fail(f"unsupported node {op} {n.outputs}")
    for k, n in enumerate(m.nodes):  # size-expression nodes must all belong to a recognised Resize
        if n.op_type in ("Shape", "Slice") or (n.op_type == "Concat" and n.attrs.get("axis") == 0):
            if k not in fused:
                raise Fail(f"{n.op_type} {n.outputs}: not part of a recognised resize-size expression")
        elif n.op_type == "Constant":
            if any(c not in fused for c in consumers.get(n.outputs[0], [])):
                raise Fail(f"constant {n.outputs}: used outside a recognised resize-size expression")
    if logits_name not in shape or shape[logits_name] != (1, NUM_CLASSES, H // 4, W // 4):
        raise Fail(f"logits {logits_name}: expected 1x{NUM_CLASSES}x{H // 4}x{W // 4}")
    if steps[-1]["op"] != "resize_bilinear" or steps[-1]["x"] != logits_name or (steps[-1]["oh"], steps[-1]["ow"]) != (H, W):
        raise Fail("the final step must resize the logits to the input size")
    steps.append({"op": "argmax_u8", "x": steps[-1]["y"], "y": "@output", "c": NUM_CLASSES, "h": H, "w": W})

    # ---------------------------------------------------------------- placement (zero-copy concats)
    produced = {s["y"] for s in steps}
    placement: dict[str, tuple[str, int]] = {}
    for k in reversed(concat_nodes):
        n = m.nodes[k]
        out = n.outputs[0]
        root, base = placement.get(out, (out, 0))
        _, _, Hx, Wx = shape[out]
        offset = 0
        for i in n.inputs:
            if i in placement or i not in produced:
                raise Fail(f"concat {out}: input {i} cannot be placed without a copy")
            if len(consumers.get(i, [])) != 1:
                raise Fail(f"concat {out}: input {i} has other consumers")
            placement[i] = (root, base + offset * Hx * Wx)
            offset += shape[i][1]

    def root_of(t: str) -> tuple[str, int]:
        return placement.get(t, (t, 0))

    roots: list[str] = []
    for s in steps:
        if s["y"] == "@output":
            continue
        r = root_of(s["y"])[0]
        if r not in roots:
            roots.append(r)
    for k in concat_nodes:
        r = root_of(m.nodes[k].outputs[0])[0]
        if r not in roots:
            roots.append(r)
    arena: dict[str, int] = {}
    cursor = 0
    for r in roots:
        arena[r] = cursor
        cursor = align(cursor + 4 * numel(shape[r]))

    def ref(t: str) -> dict:
        if t == gin.name:
            return {"buf": "input", "offset": 0}
        if t == "@output":
            return {"buf": "output", "offset": 0}
        r, el = root_of(t)
        return {"buf": "arena", "offset": arena[r] + 4 * el}

    plan_steps = []
    for s in steps:
        p = {k: v for k, v in s.items() if k not in ("x", "y", "residual", "onnx_node")}
        p["x"], p["y"] = ref(s["x"]), ref(s["y"])
        if s["op"] == "conv":
            p["residual"] = ref(s["residual"]) if s["residual"] else None
            p["onnx_output"] = s["onnx_node"]
        plan_steps.append(p)

    weights_bytes = bytes(weights)
    counts = {op: sum(1 for s in plan_steps if s["op"] == op) for op in sorted({s["op"] for s in plan_steps})}
    common = {
        "schema": SCHEMA,
        "model": {"onnx_file": onnx_path.name, "onnx_sha256": sha256(onnx_bytes), "onnx_bytes": len(onnx_bytes),
                  "producer": f"{m.producer_name} {m.producer_version}", "opset": m.opsets[""],
                  "ir_version": m.ir_version, "nodes": len(m.nodes), "initializers": len(inits)},
        "converter": {"path": "tools/deeplab_family/build_deeplab_plan.py", "sha256": sha256(Path(__file__).read_bytes()),
                      "onnx_reader_sha256": sha256((HERE.parent / "stage2b_g_real" / "onnx_reader.py").read_bytes())},
        "input": {"name": gin.name, "shape": list(in_shape), "dtype": "float32", "layout": "NCHW",
                  "bytes": 4 * 3 * H * W},
        "weights": {"bytes": len(weights_bytes), "sha256": sha256(weights_bytes), "dtype": "float32",
                    "byte_order": "little"},
        "arena_bytes": cursor,
    }
    canonical_plan = {**common, "variant": "canonical",
                      "output": {"name": gout.name, "shape": list(gout.shape), "dtype": "uint8", "bytes": H * W,
                                 "semantics": "argmax over the 19 class logits resized to the input size"},
                      "step_counts": counts, "steps": plan_steps}
    # diagnostic plan: stop at the logits, written by the last convolution into the output buffer
    diag_steps = [dict(s) for s in plan_steps[:-2]]
    last = diag_steps[-1]
    if last["op"] != "conv" or last.get("onnx_output") != logits_name:
        raise Fail("the logits must be produced by the last convolution")
    last["y"] = {"buf": "output", "offset": 0}
    diag_counts = {op: sum(1 for s in diag_steps if s["op"] == op) for op in sorted({s["op"] for s in diag_steps})}
    diag_plan = {**common, "variant": "diagnostic_logits",
                 "output": {"name": logits_name, "shape": [1, NUM_CLASSES, H // 4, W // 4], "dtype": "float32",
                            "bytes": 4 * NUM_CLASSES * (H // 4) * (W // 4),
                            "semantics": "class logits before the final resize and argmax (numeric gate only)"},
                 "step_counts": diag_counts, "steps": diag_steps}
    return canonical_plan, diag_plan, weights_bytes


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    onnx_path, out_prefix = Path(sys.argv[1]), sys.argv[2]
    plan, diag, weights_bytes = convert(onnx_path)
    name = Path(out_prefix).name + ".weights.f32"
    plan["weights"]["file"] = diag["weights"]["file"] = name
    Path(out_prefix + ".weights.f32").write_bytes(weights_bytes)
    texts = {".plan.json": canonical(plan) + "\n", ".diag.plan.json": canonical(diag) + "\n"}
    for suffix, text in texts.items():
        Path(out_prefix + suffix).write_text(text, encoding="utf-8")
    print(canonical({"plan_sha256": sha256(texts[".plan.json"].encode()),
                     "diag_plan_sha256": sha256(texts[".diag.plan.json"].encode()),
                     "weights_sha256": plan["weights"]["sha256"], "weights_bytes": len(weights_bytes),
                     "arena_bytes": plan["arena_bytes"], "step_counts": plan["step_counts"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Fail as exc:
        raise SystemExit(f"build_deeplab_plan: {exc}")
    except (KeyError, IndexError, ValueError, TypeError) as exc:  # a reference the recognised patterns cannot resolve
        raise SystemExit(f"build_deeplab_plan: malformed or unsupported graph ({type(exc).__name__}: {exc})")
