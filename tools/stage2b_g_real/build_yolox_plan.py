#!/usr/bin/env python3
"""Convert the YOLOX-S ONNX model into an SDV-BMT cuDNN execution plan + weight blob (offline, stdlib only).

Stage 2b-G real-model baseline.  The ONNX file is the model artifact (identity = its SHA-256).  This tool
derives, deterministically, what the native NVIDIA backend executes with cuDNN and its glue kernels:

  <out>.plan.json   integer-only canonical JSON: steps, tensor placement, convolution geometry
  <out>.weights.f32 convolution weights and biases, float32 little-endian, 256-byte aligned

Graph patterns recognised (anything else is rejected, never approximated):
  * Focus stem: 8 Slice + Concat on the graph input -> step ``space_to_depth2`` (group order recorded)
  * Conv [+ Sigmoid & Mul(x, sigmoid(x)) = SiLU [+ Add residual]] -> step ``conv`` with epilogue
    act silu (+ residual); Conv + Sigmoid -> act sigmoid; bare Conv -> act none
  * MaxPool -> step ``maxpool``;  Resize nearest/asymmetric/floor with scales [1,1,2,2] -> ``upsample_nearest2x``
  * Concat on the channel axis -> zero-copy: producers write into channel slices of the concat tensor
    (batch 1, NCHW: a channel slice is contiguous); a copy step is emitted only when that is impossible
  * Reshape [1,85,-1] x3 + Concat(axis 2) + Transpose [0,2,1] -> step ``head_transpose`` into the output

Usage: python3 tools/stage2b_g_real/build_yolox_plan.py models/yolox-s/yolox_s.onnx models/yolox-s/yolox_s
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import onnx_reader as R  # noqa: E402

SCHEMA = "sdv-bmt-cudnn-plan/1"
ALIGN = 256
INT64_MAX = (1 << 63) - 1


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Fail(Exception):
    pass


def align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    onnx_path, out_prefix = Path(sys.argv[1]), sys.argv[2]
    onnx_bytes = onnx_path.read_bytes()
    m = R.load(str(onnx_path))
    if m.opsets.get("", 0) != 11:
        raise Fail(f"expected opset 11, got {m.opsets}")
    inits = m.initializers
    graph_inputs = [i for i in m.inputs if i.name not in inits]
    if len(graph_inputs) != 1 or len(m.outputs) != 1:
        raise Fail("expected one graph input and one output")
    gin, gout = graph_inputs[0], m.outputs[0]
    if tuple(gin.shape) != (1, 3, 640, 640) or gin.elem_type != R.FLOAT:
        raise Fail(f"unexpected input {gin}")
    if tuple(gout.shape) != (1, 8400, 85) or gout.elem_type != R.FLOAT:
        raise Fail(f"unexpected output {gout}")

    producer: dict[str, int] = {}
    consumers: dict[str, list[int]] = {}
    for k, n in enumerate(m.nodes):
        for o in n.outputs:
            producer[o] = k
        for i in n.inputs:
            if i:
                consumers.setdefault(i, []).append(k)

    def ints(name: str) -> list[int]:
        t = inits.get(name)
        if t is None or t.data_type != R.INT64:
            raise Fail(f"{name}: expected an int64 initializer")
        return t.values()

    shape: dict[str, tuple[int, ...]] = {gin.name: tuple(gin.shape)}
    steps: list[dict] = []
    fused: set[int] = set()
    weights = bytearray()
    concat_of: dict[str, tuple[str, int]] = {}  # tensor -> (concat output tensor, channel offset)
    concat_nodes: list[int] = []
    produced_by_step: set[str] = set()
    head_parts: dict[str, int] = {}

    def add_weight(t: R.Tensor) -> int:
        if t.data_type != R.FLOAT:
            raise Fail(f"{t.name}: weights must be float32")
        weights.extend(b"\0" * (align(len(weights)) - len(weights)))
        off = len(weights)
        weights.extend(t.raw)
        return off

    def only_consumers(t: str) -> list[int]:
        return consumers.get(t, [])

    # ---------------------------------------------------------------- Focus (space_to_depth2)
    def slice_params(n: R.Node) -> tuple[str, int, int]:
        data, starts, ends, axes, steps_ = n.inputs
        s, e, a, st = ints(starts), ints(ends), ints(axes), ints(steps_)
        if len(s) != 1 or e != [INT64_MAX] or st != [2] or a[0] not in (2, 3) or s[0] not in (0, 1):
            raise Fail(f"unsupported Slice {n.outputs}")
        return data, a[0], s[0]

    for k, n in enumerate(m.nodes):
        if n.op_type != "Concat" or len(n.inputs) != 4:
            continue
        if not all(producer.get(i) is not None and m.nodes[producer[i]].op_type == "Slice" for i in n.inputs):
            continue
        order = []
        for i in n.inputs:
            s2 = m.nodes[producer[i]]
            data2, axis2, start2 = slice_params(s2)
            s1 = m.nodes[producer[data2]]
            if s1.op_type != "Slice":
                raise Fail("Focus: expected Slice(Slice(input))")
            data1, axis1, start1 = slice_params(s1)
            if data1 != gin.name or (axis1, axis2) != (2, 3):
                raise Fail("Focus: expected rows then columns of the graph input")
            order.append([start1, start2])  # [dy, dx]
            fused.update({producer[i], producer[data2]})
        if n.attrs.get("axis") != 1:
            raise Fail("Focus: concat axis must be 1")
        if sorted(map(tuple, order)) != [(0, 0), (0, 1), (1, 0), (1, 1)]:
            raise Fail(f"Focus: unexpected group order {order}")
        fused.add(k)
        c, h, w = 3, 640, 640
        shape[n.outputs[0]] = (1, 4 * c, h // 2, w // 2)
        steps.append({"op": "space_to_depth2", "x": gin.name, "y": n.outputs[0], "c": c, "h": h, "w": w,
                      "order": order})
        produced_by_step.add(n.outputs[0])
        break
    else:
        raise Fail("Focus stem not found")

    # ---------------------------------------------------------------- head (Reshape x3 + Concat + Transpose)
    tnode = m.nodes[producer[gout.name]]
    if tnode.op_type != "Transpose" or tnode.attrs.get("perm") != [0, 2, 1]:
        raise Fail("head: expected Transpose [0,2,1] at the output")
    cnode = m.nodes[producer[tnode.inputs[0]]]
    if cnode.op_type != "Concat" or cnode.attrs.get("axis") != 2:
        raise Fail("head: expected Concat(axis=2) before the Transpose")
    head_levels = []
    for i in cnode.inputs:
        r = m.nodes[producer[i]]
        if r.op_type != "Reshape" or ints(r.inputs[1]) != [1, 85, -1]:
            raise Fail("head: expected Reshape [1,85,-1]")
        head_levels.append(r.inputs[0])
        fused.add(producer[i])
    fused.update({producer[tnode.inputs[0]], producer[gout.name]})

    # ---------------------------------------------------------------- main pass
    for k, n in enumerate(m.nodes):
        if k in fused:
            continue
        op = n.op_type
        if op == "Conv":
            x, wname, bname = n.inputs
            W, B = inits[wname], inits[bname]
            K, Cg, Rk, Sk = W.dims
            group = n.attrs.get("group", 1)
            pads = n.attrs.get("pads", [0, 0, 0, 0])
            strides = n.attrs.get("strides", [1, 1])
            dil = n.attrs.get("dilations", [1, 1])
            if n.attrs.get("kernel_shape") != [Rk, Sk] or pads[0] != pads[2] or pads[1] != pads[3]:
                raise Fail(f"conv {n.outputs}: unsupported geometry")
            _, C, H, Wd = shape[x]
            if C != Cg * group or B.dims != (K,):
                raise Fail(f"conv {n.outputs}: channel mismatch")
            OH = (H + 2 * pads[0] - dil[0] * (Rk - 1) - 1) // strides[0] + 1
            OW = (Wd + 2 * pads[1] - dil[1] * (Sk - 1) - 1) // strides[1] + 1
            o = n.outputs[0]
            act, out, residual = "none", o, None
            cons = only_consumers(o)
            sig = [c for c in cons if m.nodes[c].op_type == "Sigmoid"]
            if len(cons) == 2 and len(sig) == 1:  # SiLU: Mul(o, Sigmoid(o))
                s = m.nodes[sig[0]]
                mul = [c for c in cons if c != sig[0]][0]
                mn = m.nodes[mul]
                if mn.op_type != "Mul" or sorted(mn.inputs) != sorted([o, s.outputs[0]]) or \
                        only_consumers(s.outputs[0]) != [mul]:
                    raise Fail(f"conv {o}: unrecognised activation pattern")
                fused.update({sig[0], mul})
                act, out = "silu", mn.outputs[0]
                after = only_consumers(out)
                if len(after) == 1 and m.nodes[after[0]].op_type == "Add":
                    add = m.nodes[after[0]]
                    other = [i for i in add.inputs if i != out]
                    if len(other) == 1 and other[0] in shape and shape[other[0]] == (1, K, OH, OW):
                        fused.add(after[0])
                        residual, out = other[0], add.outputs[0]
            elif len(cons) == 1 and m.nodes[cons[0]].op_type == "Sigmoid":
                fused.add(cons[0])
                act, out = "sigmoid", m.nodes[cons[0]].outputs[0]
            shape[out] = (1, K, OH, OW)
            steps.append({"op": "conv", "x": x, "y": out, "residual": residual, "act": act,
                          "w_offset": add_weight(W), "b_offset": add_weight(B),
                          "c": C, "h": H, "w": Wd, "k": K, "r": Rk, "s": Sk, "oh": OH, "ow": OW,
                          "pad_h": pads[0], "pad_w": pads[1], "stride_h": strides[0], "stride_w": strides[1],
                          "dil_h": dil[0], "dil_w": dil[1], "group": group, "onnx_node": o})
            produced_by_step.add(out)
        elif op == "MaxPool":
            ksz, pads, strides = n.attrs["kernel_shape"], n.attrs["pads"], n.attrs["strides"]
            if n.attrs.get("ceil_mode", 0) != 0 or len(set(pads)) != 1 or ksz[0] != ksz[1] or strides != [1, 1]:
                raise Fail(f"maxpool {n.outputs}: unsupported")
            x = n.inputs[0]
            _, C, H, Wd = shape[x]
            OH = (H + 2 * pads[0] - ksz[0]) // strides[0] + 1
            OW = (Wd + 2 * pads[1] - ksz[1]) // strides[1] + 1
            shape[n.outputs[0]] = (1, C, OH, OW)
            steps.append({"op": "maxpool", "x": x, "y": n.outputs[0], "c": C, "h": H, "w": Wd, "oh": OH, "ow": OW,
                          "window": ksz[0], "pad": pads[0], "stride": strides[0]})
            produced_by_step.add(n.outputs[0])
        elif op == "Resize":
            if (n.attrs.get("mode") != b"nearest" or n.attrs.get("coordinate_transformation_mode") != b"asymmetric"
                    or n.attrs.get("nearest_mode") != b"floor"):
                raise Fail("resize: unsupported mode")
            scales = inits[n.inputs[2]].values()
            if scales != [1.0, 1.0, 2.0, 2.0]:
                raise Fail(f"resize: unsupported scales {scales}")
            x = n.inputs[0]
            _, C, H, Wd = shape[x]
            shape[n.outputs[0]] = (1, C, 2 * H, 2 * Wd)
            steps.append({"op": "upsample_nearest2x", "x": x, "y": n.outputs[0], "c": C, "h": H, "w": Wd})
            produced_by_step.add(n.outputs[0])
        elif op == "Concat":
            if n.attrs.get("axis") != 1:
                raise Fail(f"concat {n.outputs}: only the channel axis is supported")
            shapes = [shape[i] for i in n.inputs]
            _, _, H, Wd = shapes[0]
            if any(s[0] != 1 or s[2:] != (H, Wd) for s in shapes):
                raise Fail(f"concat {n.outputs}: shape mismatch")
            shape[n.outputs[0]] = (1, sum(s[1] for s in shapes), H, Wd)
            concat_nodes.append(k)
        else:
            raise Fail(f"unsupported node {op} {n.outputs}")

    # ---------------------------------------------------------------- placement (zero-copy concats)
    placement: dict[str, tuple[str, int]] = {}  # tensor -> (root tensor, element offset)
    for k in reversed(concat_nodes):  # outer concats first
        n = m.nodes[k]
        out = n.outputs[0]
        root, base = placement.get(out, (out, 0))
        _, _, H, Wd = shape[out]
        offset = 0
        for i in n.inputs:
            C = shape[i][1]
            if i in placement or i not in produced_by_step and i not in {m.nodes[c].outputs[0] for c in concat_nodes}:
                # already placed into another concat (or not produced here): copy into the slice
                steps.append({"op": "copy", "x": i, "y_root": root, "y_elem_offset": base + offset * H * Wd,
                              "elements": C * H * Wd, "after": i})
            else:
                placement[i] = (root, base + offset * H * Wd)
            offset += C

    def root_of(t: str) -> tuple[str, int]:
        return placement.get(t, (t, 0))

    # tensors that own storage: roots of every produced tensor
    roots: list[str] = []
    for s in steps:
        for key in ("y",):
            if key in s:
                r = root_of(s[key])[0]
                if r not in roots:
                    roots.append(r)
    for k in concat_nodes:
        r = root_of(m.nodes[k].outputs[0])[0]
        if r not in roots:
            roots.append(r)
    arena: dict[str, int] = {}
    cursor = 0
    for r in roots:
        n_el = 1
        for d in shape[r]:
            n_el *= d
        arena[r] = cursor
        cursor = align(cursor + 4 * n_el)

    def ref(t: str) -> dict:
        if t == gin.name:
            return {"buf": "input", "offset": 0}
        r, el = root_of(t)
        return {"buf": "arena", "offset": arena[r] + 4 * el}

    # copy steps must run after their source is produced: place them right after the producing step
    ordered: list[dict] = []
    copies = [s for s in steps if s["op"] == "copy"]
    main_steps = [s for s in steps if s["op"] != "copy"]
    for s in main_steps:
        ordered.append(s)
        for c in copies:
            if c["after"] == s["y"]:
                ordered.append(c)
    if len(ordered) != len(steps):
        raise Fail("a copy step has no producer")

    plan_steps = []
    for s in ordered:
        if s["op"] == "copy":
            plan_steps.append({"op": "copy", "x": ref(s["x"]),
                               "y": {"buf": "arena", "offset": arena[s["y_root"]] + 4 * s["y_elem_offset"]},
                               "elements": s["elements"]})
            continue
        p = {k: v for k, v in s.items() if k not in ("x", "y", "residual", "onnx_node")}
        p["x"], p["y"] = ref(s["x"]), ref(s["y"])
        if s["op"] == "conv":
            p["act"] = {"none": 0, "silu": 1, "sigmoid": 2}[s["act"]]
            p["residual"] = ref(s["residual"]) if s["residual"] else None
            p["onnx_output"] = s["onnx_node"]
        plan_steps.append(p)

    level_shapes = [shape[t] for t in head_levels]
    anchors = 0
    head = {"op": "head_transpose", "channels": 85, "y": {"buf": "output", "offset": 0}, "levels": []}
    for t, (_, C, H, Wd) in zip(head_levels, level_shapes):
        if C != 85:
            raise Fail("head: 85 channels expected")
        head["levels"].append({"x": ref(t), "h": H, "w": Wd, "anchor_offset": anchors})
        anchors += H * Wd
    if anchors != 8400:
        raise Fail(f"head: {anchors} anchors")
    plan_steps.append(head)

    used = {s["op"] for s in plan_steps}
    counts = {op: sum(1 for s in plan_steps if s["op"] == op) for op in sorted(used)}
    weights_bytes = bytes(weights)
    plan = {
        "schema": SCHEMA,
        "model": {"onnx_file": onnx_path.name, "onnx_sha256": sha256(onnx_bytes), "onnx_bytes": len(onnx_bytes),
                  "producer": f"{m.producer_name} {m.producer_version}", "opset": m.opsets[""],
                  "ir_version": m.ir_version, "nodes": len(m.nodes), "initializers": len(inits)},
        "converter": {"path": "tools/stage2b_g_real/build_yolox_plan.py",
                      "sha256": sha256(Path(__file__).read_bytes()),
                      "onnx_reader_sha256": sha256((Path(__file__).parent / "onnx_reader.py").read_bytes())},
        "input": {"name": gin.name, "shape": list(gin.shape), "dtype": "float32", "layout": "NCHW",
                  "bytes": 4 * 3 * 640 * 640},
        "output": {"name": gout.name, "shape": list(gout.shape), "dtype": "float32", "bytes": 4 * 8400 * 85,
                   "anchor_strides": [8, 16, 32], "levels_hw": [[s[2], s[3]] for s in level_shapes]},
        "weights": {"file": Path(out_prefix).name + ".weights.f32", "bytes": len(weights_bytes),
                    "sha256": sha256(weights_bytes), "dtype": "float32", "byte_order": "little"},
        "arena_bytes": cursor,
        "step_counts": counts,
        "steps": plan_steps,
    }
    Path(out_prefix + ".weights.f32").write_bytes(weights_bytes)
    text = canonical(plan) + "\n"
    Path(out_prefix + ".plan.json").write_text(text, encoding="utf-8")
    print(canonical({"plan_sha256": sha256(text.encode()), "weights_sha256": plan["weights"]["sha256"],
                     "weights_bytes": len(weights_bytes), "arena_bytes": cursor, "step_counts": counts,
                     "fused_nodes": len(fused), "nodes": len(m.nodes)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Fail as exc:
        raise SystemExit(f"build_yolox_plan: {exc}")
