#!/usr/bin/env python3
"""Convert the exported CenterPoint-Pillars ONNX model into an SDV-BMT cuDNN execution plan + weight blob (offline, stdlib only).

The ONNX file (``export_onnx.py``) is the model artifact (identity = its SHA-256).  This tool derives, deterministically,
what the native NVIDIA backend executes with cuDNN and its glue kernels:

  <out>.plan.json       pillar input -> raw CenterHead outputs [1, 70, 128, 128] float32 (the task's raw output)
  <out>.weights.f32     weights and biases (BatchNorm folded at export), float32 little-endian, 256-byte aligned

Device input (the workload's input contract): voxels float32 [P, 20, 5] | points per pillar int32 [P] |
pillar (y, x) int32 [P, 2], P = 40,000 (the payload body without its 40-byte header).

Graph patterns recognised (anything else is rejected, never approximated):
  * the pillar front-end as exported from ``centerpoint_model`` (PillarFeatureNet legacy=False, PointPillarsScatter):
    one MatMul with an [11, 64] weight, Add(bias), Relu, ReduceMax(axis 1), Transpose, ScatterND into a zero canvas
    [64, 512*512 + 1], Slice to 512*512 columns, Reshape [1, 64, 512, 512].  Checked: the 11 decorated features in
    the order raw (5), cluster offset (3), pillar-centre offset (3); the pillar-centre constants (voxel size
    0.2 / 0.2 / 8 and offsets -51.1 / -51.1 / -1 as float32 bit patterns, coordinate columns x / y / z); the
    points-per-pillar clamp at 1; the point mask arange(20); the scatter index y * 512 + x with padded pillars sent
    to column 262144.  -> steps ``pillar_features`` and ``pillar_scatter``
  * Conv [-> Relu] -> ``conv`` act none / ReLU (3)
  * ConvTranspose (kernel = stride = 2, no padding) [-> Relu] -> a 1x1 ``conv`` with 4 x out channels (channel
    o * 4 + dy * 2 + dx holds weight [:, o, dy, dx]) followed by ``depth_to_space2`` (exact rearrangement)
  * Concat on the channel axis -> zero-copy: producers write into channel slices; the graph output is such a
    concat, so the head convolutions write directly into the output buffer

    python3 tools/centerpoint/build_centerpoint_plan.py models/centerpoint-pillar02/centerpoint_pillar02.onnx \
        models/centerpoint-pillar02/centerpoint_pillar02
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage2b_g_real"))
import onnx_reader as R  # noqa: E402

SCHEMA = "sdv-bmt-cudnn-plan/1"
ALIGN = 256
P, M, F, C, GRID = 40000, 20, 5, 64, 512
HEAD_CHANNELS = 70
FRONT_OPS = {"Constant", "Concat", "Clip", "Slice", "ReduceSum", "Cast", "Reshape", "Div", "Sub", "Gather", "Unsqueeze",
             "Mul", "Add", "Expand", "ConstantOfShape", "Equal", "Where", "ScatterND", "Shape", "Greater", "MatMul",
             "Relu", "ReduceMax", "Transpose"}


class Fail(Exception):
    pass


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def convert(onnx_path: Path) -> tuple[dict, bytes]:
    onnx_bytes = onnx_path.read_bytes()
    m = R.load(str(onnx_path))
    if m.opsets.get("", 0) != 17:
        raise Fail(f"expected opset 17, got {m.opsets}")
    inits = m.initializers
    gins = {i.name: i for i in m.inputs if i.name not in inits}
    expect_in = {"voxels": ((P, M, F), R.FLOAT), "num_points": ((P,), R.INT32), "coors": ((P, 2), R.INT32)}
    if set(gins) != set(expect_in) or any(tuple(gins[k].shape) != s or gins[k].elem_type != t
                                          for k, (s, t) in expect_in.items()):
        raise Fail(f"unexpected graph inputs {[(i.name, i.shape, i.elem_type) for i in gins.values()]}")
    if len(m.outputs) != 1 or tuple(m.outputs[0].shape) != (1, HEAD_CHANNELS, 128, 128) or m.outputs[0].elem_type != R.FLOAT:
        raise Fail(f"unexpected output {m.outputs}")
    gout = m.outputs[0].name

    producer, consumers = {}, {}
    for k, n in enumerate(m.nodes):
        for o in n.outputs:
            producer[o] = k
        for i in n.inputs:
            if i:
                consumers.setdefault(i, []).append(k)
    const = {n.outputs[0]: n.attrs["value"] for n in m.nodes if n.op_type == "Constant"}

    def node(t: str) -> R.Node:
        if t not in producer:
            raise Fail(f"{t}: no producer")
        return m.nodes[producer[t]]

    def only_consumer(t: str, op: str) -> R.Node:
        cs = consumers.get(t, [])
        if len(cs) != 1 or m.nodes[cs[0]].op_type != op:
            raise Fail(f"{t}: expected a single {op} consumer, got {[m.nodes[c].op_type for c in cs]}")
        return m.nodes[cs[0]]

    def cvals(t: str):
        v = const.get(t) or inits.get(t)
        if v is None:
            raise Fail(f"{t}: expected a constant")
        return v.values()

    # ------------------------------------------------------------------ pillar front-end
    mms = [n for n in m.nodes if n.op_type == "MatMul"]
    if len(mms) != 1:
        raise Fail("expected exactly one MatMul (the pillar linear layer)")
    mm = mms[0]
    w_t = inits.get(mm.inputs[1])
    if w_t is None or tuple(w_t.dims) != (11, C) or w_t.data_type != R.FLOAT:
        raise Fail("pillar linear weight must be a float32 [11, 64] initializer")
    add = only_consumer(mm.outputs[0], "Add")
    b_name = [i for i in add.inputs if i != mm.outputs[0]][0]
    b_t = inits.get(b_name)
    if b_t is None or tuple(b_t.dims) != (C,):
        raise Fail("pillar linear bias must be a [64] initializer")
    relu = only_consumer(add.outputs[0], "Relu")
    rmax = only_consumer(relu.outputs[0], "ReduceMax")
    if list(rmax.attrs.get("axes", [])) != [1] or rmax.attrs.get("keepdims") != 0:
        raise Fail("ReduceMax must reduce axis 1 without keepdims")
    tr = only_consumer(rmax.outputs[0], "Transpose")
    if list(tr.attrs.get("perm", [])) != [1, 0]:
        raise Fail("pillar features must be transposed to [64, P]")
    rs0 = only_consumer(tr.outputs[0], "Reshape")
    snd = only_consumer(rs0.outputs[0], "ScatterND")
    canvas0 = const.get(snd.inputs[0])
    if canvas0 is None or tuple(canvas0.dims) != (C, GRID * GRID + 1) or any(canvas0.raw):
        raise Fail("scatter canvas must be a zero [64, 262145] constant")
    sl = only_consumer(snd.outputs[0], "Slice")
    if cvals(sl.inputs[1]) != [0] or cvals(sl.inputs[2]) != [GRID * GRID] or cvals(sl.inputs[3]) != [1]:
        raise Fail("scatter canvas must be sliced to columns 0..262143")
    rs1 = only_consumer(sl.outputs[0], "Reshape")
    if cvals(rs1.inputs[1]) != [1, C, GRID, GRID]:
        raise Fail("canvas must be reshaped to [1, 64, 512, 512]")
    canvas = rs1.outputs[0]
    # scatter index: Where(Greater(num_points, 0), y * 512 + x, 262144)
    idx_node = node(node(snd.inputs[1]).inputs[1]) if node(snd.inputs[1]).op_type == "Concat" else None
    where = [n for n in m.nodes if n.op_type == "Where" and n.inputs[2] in const and
             tuple(const[n.inputs[2]].dims) == (P,)]
    if len(where) != 1 or set(const[where[0].inputs[2]].values()) != {GRID * GRID}:
        raise Fail("padded pillars must be scattered to column 262144")
    lin = node(where[0].inputs[1])
    mul512 = node(lin.inputs[0])
    if lin.op_type != "Add" or mul512.op_type != "Mul" or cvals(mul512.inputs[1]) != [GRID]:
        raise Fail("scatter index must be y * 512 + x")
    gy, gx = node(node(mul512.inputs[0]).inputs[0]), node(node(lin.inputs[1]).inputs[0])
    if gy.inputs[0] != "coors" or gx.inputs[0] != "coors" or cvals(gy.inputs[1]) != [0] or cvals(gx.inputs[1]) != [1]:
        raise Fail("scatter index must use coors[:, 0] (y) and coors[:, 1] (x)")
    del idx_node
    # decorated features: Mul(Concat(voxels, f_cluster, f_center), mask) -> MatMul
    mask_mul = node(mm.inputs[0])
    feats = node(mask_mul.inputs[0])
    if mask_mul.op_type != "Mul" or feats.op_type != "Concat" or feats.inputs[0] != "voxels" or len(feats.inputs) != 3:
        raise Fail("pillar features must be Concat(voxels, cluster offset, centre offset) * mask")
    fcl = node(feats.inputs[1])
    if fcl.op_type != "Sub" or node(fcl.inputs[0]).op_type != "Slice" or node(fcl.inputs[1]).op_type != "Div":
        raise Fail("second feature group must be voxels[..., :3] - mean")
    div = node(fcl.inputs[1])
    clip = node(node(node(div.inputs[1]).inputs[0]).inputs[0])
    if clip.op_type != "Clip" or clip.inputs[0] != "num_points" or cvals(clip.inputs[1]) != [1]:
        raise Fail("the mean must divide by max(num_points, 1)")
    gt = [n for n in m.nodes if n.op_type == "Greater" and n.inputs[1] in const and tuple(const[n.inputs[1]].dims) == (1, M)]
    if len(gt) != 1 or const[gt[0].inputs[1]].values() != list(range(M)):
        raise Fail("point mask must compare num_points with arange(20)")
    # pillar-centre offsets: Sub(Gather(voxels, axis 2, k), Add(Mul(Unsqueeze(Cast(Gather(coors_zyx, axis 1, j))), v), o))
    centre = {}
    for n in m.nodes:
        if n.op_type != "Sub" or not n.inputs[0] in producer:
            continue
        g = node(n.inputs[0])
        if g.op_type != "Gather" or g.inputs[0] != "voxels" or g.attrs.get("axis") != 2:
            continue
        a = node(n.inputs[1])
        mu = node(a.inputs[0])
        gc = node(node(node(mu.inputs[0]).inputs[0]).inputs[0])
        k, j = cvals(g.inputs[1])[0], cvals(gc.inputs[1])[0]
        centre[k] = (j, f32_bits(cvals(mu.inputs[1])[0]), f32_bits(cvals(a.inputs[1])[0]))
    want = {0: (2, f32_bits(0.2), f32_bits(-51.1)), 1: (1, f32_bits(0.2), f32_bits(-51.1)),
            2: (0, f32_bits(8.0), f32_bits(-1.0))}
    if centre != want:
        raise Fail(f"pillar-centre constants differ: {centre} != {want}")
    first_conv = next(k for k, n in enumerate(m.nodes) if n.op_type == "Conv")
    bad = sorted({n.op_type for n in m.nodes[:first_conv]} - FRONT_OPS)
    if bad or producer[canvas] >= first_conv:
        raise Fail(f"unexpected front-end operators {bad}")

    weights = bytearray()

    def add_weight(raw: bytes) -> int:
        weights.extend(b"\0" * (align(len(weights)) - len(weights)))
        off = len(weights)
        weights.extend(raw)
        return off

    w11 = w_t.values()  # [11, 64] row-major -> [64, 11]
    w_off = add_weight(struct.pack(f"<{C * 11}f", *[w11[k * C + c] for c in range(C) for k in range(11)]))
    b_off = add_weight(b_t.raw)
    steps = [{"op": "pillar_features", "x": "voxels", "num_points": "num_points", "coors": "coors", "y": "@pfn",
              "pillars": P, "points": M, "features": F, "channels": C, "w_offset": w_off, "b_offset": b_off,
              "consts_f32_bits": [want[0][1], want[1][1], want[2][1], want[0][2], want[1][2], want[2][2]]},
             {"op": "pillar_scatter", "x": "@pfn", "num_points": "num_points", "coors": "coors", "y": canvas,
              "pillars": P, "channels": C, "h": GRID, "w": GRID}]
    shape = {canvas: (1, C, GRID, GRID), "@pfn": (C, P)}

    # ------------------------------------------------------------------ CNN part
    concat_of = {}  # member tensor -> (root, element offset)
    fused = set()
    for k in range(first_conv, len(m.nodes)):
        n = m.nodes[k]
        if k in fused or n.op_type == "Constant":
            continue
        if n.op_type in ("Conv", "ConvTranspose"):
            x = n.inputs[0]
            if x not in shape:
                raise Fail(f"{n.outputs[0]}: input shape unknown")
            _, c, h, w = shape[x]
            wt, bt = inits.get(n.inputs[1]), inits.get(n.inputs[2]) if len(n.inputs) > 2 else None
            if wt is None or bt is None:
                raise Fail(f"{n.outputs[0]}: weight and bias must be initializers")
            out = n.outputs[0]
            act = 0
            cs = consumers.get(out, [])
            if len(cs) == 1 and m.nodes[cs[0]].op_type == "Relu":
                fused.add(cs[0])
                act = 3
                out = m.nodes[cs[0]].outputs[0]
            if n.op_type == "Conv":
                kk, cin_g, r, s = wt.dims
                group = n.attrs.get("group", 1)
                st = n.attrs.get("strides", [1, 1])
                pads = n.attrs.get("pads", [0, 0, 0, 0])
                dil = n.attrs.get("dilations", [1, 1])
                if pads[0] != pads[2] or pads[1] != pads[3] or cin_g * group != c:
                    raise Fail(f"{n.outputs[0]}: unsupported padding or grouping")
                oh = (h + 2 * pads[0] - dil[0] * (r - 1) - 1) // st[0] + 1
                ow = (w + 2 * pads[1] - dil[1] * (s - 1) - 1) // st[1] + 1
                steps.append({"op": "conv", "x": x, "y": out, "c": c, "h": h, "w": w, "k": kk, "r": r, "s": s,
                              "stride_h": st[0], "stride_w": st[1], "pad_h": pads[0], "pad_w": pads[1],
                              "dil_h": dil[0], "dil_w": dil[1], "group": group, "oh": oh, "ow": ow, "act": act,
                              "residual": None, "w_offset": add_weight(wt.raw), "b_offset": add_weight(bt.raw),
                              "onnx_node": n.outputs[0]})
                shape[out] = (1, kk, oh, ow)
            else:
                cin, cout, r, s = wt.dims
                if (r, s) != (2, 2) or n.attrs.get("strides") != [2, 2] or any(n.attrs.get("pads", [0, 0, 0, 0])) \
                        or n.attrs.get("group", 1) != 1 or any(n.attrs.get("output_padding", [0, 0])) or cin != c:
                    raise Fail(f"{n.outputs[0]}: only kernel = stride = 2 transposed convolutions are supported")
                wv, bv = wt.values(), bt.values()
                conv_w = [wv[((i * cout + o) * 2 + dy) * 2 + dx] for o in range(cout) for dy in range(2)
                          for dx in range(2) for i in range(cin)]
                conv_b = [bv[o] for o in range(cout) for _ in range(4)]
                tmp = n.outputs[0] + "#1x1"
                steps.append({"op": "conv", "x": x, "y": tmp, "c": c, "h": h, "w": w, "k": 4 * cout, "r": 1, "s": 1,
                              "stride_h": 1, "stride_w": 1, "pad_h": 0, "pad_w": 0, "dil_h": 1, "dil_w": 1,
                              "group": 1, "oh": h, "ow": w, "act": act, "residual": None,
                              "w_offset": add_weight(struct.pack(f"<{len(conv_w)}f", *conv_w)),
                              "b_offset": add_weight(struct.pack(f"<{len(conv_b)}f", *conv_b)),
                              "onnx_node": n.outputs[0] + " (as 1x1 conv)"})
                shape[tmp] = (1, 4 * cout, h, w)
                steps.append({"op": "depth_to_space2", "x": tmp, "y": out, "c": cout, "h": h, "w": w})
                shape[out] = (1, cout, 2 * h, 2 * w)
        elif n.op_type == "Concat":
            if n.attrs.get("axis") != 1:
                raise Fail(f"{n.outputs[0]}: only channel concat is supported")
            root = n.outputs[0]
            off, hw = 0, None
            for i in n.inputs:
                _, ci, hi, wi = shape[i]
                if hw is not None and hw != (hi, wi):
                    raise Fail(f"{root}: concat inputs differ in size")
                hw = (hi, wi)
                concat_of[i] = (root, off)
                off += ci * hi * wi
            shape[root] = (1, off // (hw[0] * hw[1]), hw[0], hw[1])
        else:
            raise Fail(f"unsupported node {n.op_type} {n.outputs}")
    if shape.get(gout) != (1, HEAD_CHANNELS, 128, 128):
        raise Fail("graph output shape differs")

    # ------------------------------------------------------------------ memory: arena + zero-copy concat
    def root_of(t: str) -> tuple[str, int]:
        el = 0
        while t in concat_of:
            t, o = concat_of[t]
            el += o
        return t, el

    arena, cursor = {}, 0
    for s in steps:
        for key in ("y",):
            r, _ = root_of(s[key])
            if r != gout and r not in arena:
                arena[r] = cursor
                cursor = align(cursor + 4 * numel(shape[r]))
    in_off = {"voxels": 0, "num_points": 4 * P * M * F, "coors": 4 * P * M * F + 4 * P}

    def ref(t: str) -> dict:
        if t in in_off:
            return {"buf": "input", "offset": in_off[t]}
        r, el = root_of(t)
        if r == gout:
            return {"buf": "output", "offset": 4 * el}
        return {"buf": "arena", "offset": arena[r] + 4 * el}

    plan_steps = []
    for s in steps:
        p = {k: (ref(v) if k in ("x", "y", "num_points", "coors") else v) for k, v in s.items() if k != "onnx_node"}
        if s["op"] == "conv":
            p["onnx_output"] = s["onnx_node"]
        plan_steps.append(p)
    weights_bytes = bytes(weights)
    counts = {op: sum(1 for s in plan_steps if s["op"] == op) for op in sorted({s["op"] for s in plan_steps})}
    plan = {
        "schema": SCHEMA, "variant": "canonical",
        "model": {"onnx_file": onnx_path.name, "onnx_sha256": sha256(onnx_bytes), "onnx_bytes": len(onnx_bytes),
                  "producer": f"{m.producer_name} {m.producer_version}", "opset": m.opsets[""],
                  "ir_version": m.ir_version, "nodes": len(m.nodes), "initializers": len(inits)},
        "converter": {"path": "tools/centerpoint/build_centerpoint_plan.py", "sha256": sha256(Path(__file__).read_bytes()),
                      "onnx_reader_sha256": sha256((HERE.parent / "stage2b_g_real" / "onnx_reader.py").read_bytes())},
        "input": {"name": "pillars", "bytes": in_off["coors"] + 8 * P, "dtype": "mixed",
                  "layout": "voxels float32 [40000,20,5] | num_points int32 [40000] | coors (y,x) int32 [40000,2]"},
        "output": {"name": gout, "shape": [1, HEAD_CHANNELS, 128, 128], "dtype": "float32",
                   "bytes": 4 * HEAD_CHANNELS * 128 * 128,
                   "semantics": "raw CenterHead outputs, task by task: reg 2, height 1, dim 3, rot 2, vel 2, heatmap logits"},
        "weights": {"bytes": len(weights_bytes), "sha256": sha256(weights_bytes), "dtype": "float32", "byte_order": "little"},
        "arena_bytes": cursor, "step_counts": counts, "steps": plan_steps,
    }
    return plan, weights_bytes


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    onnx_path, out = Path(sys.argv[1]), sys.argv[2]
    try:
        plan, weights = convert(onnx_path)
    except Fail as e:
        print(f"rejected: {e}", file=sys.stderr)
        return 1
    Path(out + ".weights.f32").write_bytes(weights)
    text = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    Path(out + ".plan.json").write_text(text)
    print(json.dumps({"plan_sha256": sha256(text.encode()), "weights_sha256": plan["weights"]["sha256"],
                      "arena_bytes": plan["arena_bytes"], "steps": plan["step_counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
