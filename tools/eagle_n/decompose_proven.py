"""Rewrite the R-50 tuned BF16 plan so that every operator after layer3 is built from blocks whose shapes ALREADY RAN on
this NPU (rule: split into shapes that already completed, prefer the largest).

Proven sources:
  * R-18 tuned graph: layer4 512 ch, ASPP branches 128 ch, Concat_1 640 ch, bottleneck 3x3 640->128, c1 12 ch,
    Concat_3 140 ch, sep dw 140/128, sep pw 140->128 / 128->128, conv_seg 128->19, Resize_1 128 ch;
  * R-50 bf16 ops #0-#100 (stem..layer3) in native_default form: 1x1 convs up to (M=32768, K=512, N=1024),
    (32768, 1024, 256), (131072, 256, 64) ...
R-50 head/layer4 channels are exactly 4x R-18, so every R-50 tensor wider than proven is kept as a LIST of channel
groups: 2048-ch tensors -> 4 groups of 512 (R-18 layer4 width); ASPP branches/bottleneck/sep -> 4 groups of 128;
c1 -> 4 x 12; Concat_1 -> per group g the 5 branch groups (640 = R-18 Concat_1); Concat_3 -> per group g
resize_g + c1_g (140).
A dense conv becomes (output block j) x (input group i) convs with weight sub-blocks (bias only on i=0) + Add partial
sums; depthwise convs run per group. The block size chosen is the LARGEST proven one. Exact same maths (only the BF16
summation order of partial sums changes). The proven-shape survey (tools/eagle_n/proven_survey.py) re-checks every
device call afterwards.
"""

from __future__ import annotations
import hashlib, struct


def _sha(b):
    return hashlib.sha256(b).hexdigest()


class Weights:
    def __init__(self, blob: bytearray):
        self.blob = blob
        self.cache = {}

    def put(self, raw: bytes, shape):
        key = _sha(raw)
        if key in self.cache:
            return dict(self.cache[key], shape=list(shape))
        self.blob.extend(b"\0" * (-len(self.blob) % 256))
        off = len(self.blob)
        self.blob.extend(raw)
        e = {"offset": off, "bytes": len(raw), "shape": list(shape), "sha256": key}
        self.cache[key] = e
        return dict(e)

    def read(self, e):
        return bytes(self.blob[e["offset"] : e["offset"] + e["bytes"]])


def _weight_block(w: bytes, shape, rows, cols):
    """Rows (output channels) x cols (input channels) of an OIHW FP32 weight; cols may be any index list."""
    n, cg, kh, kw = shape
    per = kh * kw * 4
    out = bytearray()
    runs = []  # contiguous column runs for fast copy
    for c in cols:
        if runs and runs[-1][1] == c:
            runs[-1][1] = c + 1
        else:
            runs.append([c, c + 1])
    for r in rows:
        base = r * cg * per
        for a, b in runs:
            out += w[base + a * per : base + b * per]
    return bytes(out)


def _bias_block(b: bytes, rows):
    return b"".join(b[r * 4 : (r + 1) * 4] for r in rows)


class Decomposer:
    """groups[value] = list of (tensor_name, [original channel indices])."""

    def __init__(self, ops, weights: Weights, decide, native_1x1, start_marker=("/backbone/layer4/", "/decode_head/")):
        self.src = ops
        self.w = weights
        self.decide = decide
        self.native = native_1x1
        self.start = start_marker
        self.out = []
        self.groups = {}
        self.shape = {}
        self.n = 0
        self.orig = {o["output"]: o["shape"] for o in ops}

    def g(self, name):
        if name not in self.groups:
            c = self.shape[name][1]
            self.groups[name] = [(name, list(range(c)))]
        return self.groups[name]

    def uniq(self, base):
        self.n += 1
        return f"{base}#d{self.n}"

    def emit(self, item):
        self.out.append(item)
        self.shape[item["output"]] = item["shape"]
        return item["output"]

    # ---- primitives
    def conv(self, op, xname, rows, cols, bias, launch, groups=1, suffix=""):
        w = self.w.read(op["weight"])
        b = self.w.read(op["bias"])
        shape = op["weight"]["shape"]
        if groups > 1:  # depthwise: rows == cols == channel subset
            wb = b"".join(w[c * shape[2] * shape[3] * 4 : (c + 1) * shape[2] * shape[3] * 4] for c in rows)
            wshape = [len(rows), 1, shape[2], shape[3]]
        else:
            wb = _weight_block(w, shape, rows, cols)
            wshape = [len(rows), len(cols), shape[2], shape[3]]
        bb = _bias_block(b, rows) if bias else bytes(4 * len(rows))
        item = {k: op[k] for k in ("kind", "kernel", "stride", "dilation", "padding")}
        item.update(
            groups=groups,
            inputs=[xname],
            output=self.uniq(op["output"] + suffix),
            shape=[1, len(rows), op["shape"][2], op["shape"][3]],
            weight=self.w.put(wb, wshape),
            bias=self.w.put(bb, [len(rows)]),
            launch=launch,
            decomposed_from=op["output"],
            role="conv_block",
            block_rows=list(rows),
            block_cols=list(cols),
            block_bias=bool(bias),
        )
        return self.emit(item)

    def add(self, a, b, like, c, role="psum"):
        return self.emit(
            {
                "kind": "Add",
                "inputs": [a, b],
                "output": self.uniq(like + "/" + role),
                "shape": [1, c] + self.shape[a][2:],
                "decomposed_from": like,
                "role": role,
            }
        )

    def concat(self, names, like, role="concat"):
        c = sum(self.shape[n][1] for n in names)
        return self.emit(
            {
                "kind": "Concat",
                "inputs": list(names),
                "output": self.uniq(like + "/cat"),
                "shape": [1, c] + self.shape[names[0]][2:],
                "decomposed_from": like,
                "role": role,
            }
        )

    def unary(self, op, x):
        item = {k: v for k, v in op.items() if k not in ("inputs", "output", "shape")}
        return self.emit(
            dict(
                item,
                inputs=[x],
                output=self.uniq(op["output"]),
                shape=[1, self.shape[x][1]] + (op["shape"][2:]),
                decomposed_from=op["output"],
                role="unary",
            )
        )

    # ---- block choice: largest proven output block for (M, K) ----
    def best_block(self, op, M, K, N, candidates):
        for B in candidates:
            if B > N or N % B and B != N:
                continue
            L = self.decide(op, M, K, B)
            if L:
                return B, L
        raise ValueError(f"no proven block for {op['output']} M={M} K={K} N={N}")

    def dense(self, op, out_block, group_hint=None):
        """Dense conv over the input groups; output in blocks of out_block (each block = one output group)."""
        x = self.g(op["inputs"][0])
        M = op["shape"][2] * op["shape"][3]
        N = op["shape"][1]
        outs = []
        for j0 in range(0, N, out_block):
            rows = list(range(j0, min(N, j0 + out_block)))
            # produce this output group from the largest proven sub-blocks, then concat (concat sizes ≤ proven)
            sub, L = self.best_block(
                op, M, max(len(c) for _, c in x), len(rows), [len(rows), 512, 256, 128, 64, 32, 19, 12]
            )
            parts = []
            for s0 in range(0, len(rows), sub):
                srows = rows[s0 : s0 + sub]
                acc = None
                for i, (xn, cols) in enumerate(x):
                    Li = self.decide(op, M, len(cols), len(srows))
                    if not Li:
                        raise ValueError(f"unproven block {op['output']} K={len(cols)} N={len(srows)}")
                    y = self.conv(op, xn, srows, cols, i == 0, Li, suffix=f"/n{srows[0]}k{cols[0]}")
                    acc = y if acc is None else self.add(acc, y, op["output"], len(srows))
                parts.append(acc)
            outs.append((parts[0] if len(parts) == 1 else self.concat(parts, op["output"], role="block_concat"), rows))
        self.groups[op["output"]] = outs

    def depthwise(self, op):
        outs = []
        for xn, cols in self.g(op["inputs"][0]):
            L = self.decide(op, op["shape"][2] * op["shape"][3], len(cols), len(cols), depthwise=True)
            if not L:
                raise ValueError(f"unproven depthwise group {op['output']} c={len(cols)}")
            outs.append((self.conv(op, xn, cols, cols, True, L, groups=len(cols), suffix=f"/c{cols[0]}"), cols))
        self.groups[op["output"]] = outs

    # ---- main --------------------------------------------------------------------------------------------------------
    def run(self, plan_input):
        self.shape[plan_input["name"]] = plan_input["shape"]
        for op in self.src:
            name = op["output"]
            if not any(m in name for m in self.start) or op["kind"] in ("ArgMax", "Cast") or name == "/Resize_output_0":
                ins = [self.g(i) for i in op["inputs"]]
                if any(len(gr) != 1 for gr in ins):  # final resize etc. need a single tensor
                    raise ValueError(f"{name}: grouped input reaches an undecomposed op")
                op = dict(op, inputs=[gr[0][0] for gr in ins])
                self.emit(op)
                continue
            k = op["kind"]
            out = op["output"]
            if k == "Conv":
                dw = op["groups"] > 1
                if dw:
                    self.depthwise(op)
                else:
                    n = op["shape"][1]
                    blk = (
                        512
                        if n == 2048
                        else (n if any(s in out for s in ("/layer4/", "/conv_seg/")) else (12 if n == 48 else 128))
                    )
                    self.dense(op, blk)
            elif k in ("Relu", "GlobalAveragePool"):
                self.groups[out] = [(self.unary(op, xn), cols) for xn, cols in self.g(op["inputs"][0])]
            elif k == "Resize":
                self.groups[out] = [(self.unary(op, xn), cols) for xn, cols in self.g(op["inputs"][0])]
            elif k == "Add":
                a, b = self.g(op["inputs"][0]), self.g(op["inputs"][1])
                if [c for _, c in a] != [c for _, c in b]:
                    raise ValueError(f"{out}: Add inputs grouped differently")
                self.groups[out] = [(self.add(x, y, out, len(c), role="residual"), c) for (x, c), (y, _) in zip(a, b)]
            elif k == "Concat":  # pair the i-th group of every input (R-18 concat shapes 640 / 140)
                ins = [self.g(i) for i in op["inputs"]]
                offs = []
                o = 0
                for i in op["inputs"]:
                    offs.append(o)
                    o += self.shape_of(i)
                if len({len(g) for g in ins}) != 1:
                    raise ValueError(f"{out}: concat inputs have different group counts")
                outs = []
                for gi in range(len(ins[0])):
                    names = [ins[t][gi][0] for t in range(len(ins))]
                    cols = [offs[t] + c for t in range(len(ins)) for c in ins[t][gi][1]]
                    outs.append((self.concat(names, out), cols))
                self.groups[out] = outs
            else:
                raise ValueError(f"{out}: no decomposition rule for {k}")
        return self.out

    def shape_of(self, name):
        return self.orig[name][1] if name in self.orig else self.shape[name][1]


def verify(src_ops, out_ops, blob, input_name):
    """Symbolic equivalence proof (host): per tensor channel provenance; every original conv weight element (row, col)
    used exactly once, bias exactly once per output channel, block weights byte-identical to the original sub-block,
    and every conv block reads exactly the original input channels it multiplies. Raises on any mismatch."""
    src = {o["output"]: o for o in src_ops}
    C = {input_name: 3}
    C.update({o["output"]: o["shape"][1] for o in src_ops})
    prov = {input_name: [(input_name, c) for c in range(3)]}
    part = {}  # tensor -> (orig conv, frozenset cols, bias count) while a sum is incomplete
    cover = {}  # orig conv -> set((row, col))
    biased = {}  # orig conv -> {row: count}

    def need_complete(t):
        if t in part:
            v, cols, b = part[t]
            o = src[v]
            K = o["weight"]["shape"][1] * (1 if o["groups"] == 1 else 1)
            full = set(range(o["weight"]["shape"][1])) if o["groups"] == 1 else None
            if (full is not None and cols != full) or b != 1:
                raise ValueError(f"incomplete partial sum {t} of {v}")

    for o in out_ops:
        k, ins, out = o["kind"], o["inputs"], o["output"]
        role = o.get("role")
        if role is None:  # untouched original op
            for t, u in zip(ins, src[out]["inputs"]):
                need_complete(t)
                exp = [(u, c) for c in range(C[u])]
                if prov[t] != exp:
                    raise ValueError(f"{out}: input {t} provenance differs from original {u}")
            prov[out] = [(out, c) for c in range(o["shape"][1])]
            continue
        v = o["decomposed_from"]
        so = src[v]
        if role == "conv_block":
            u = so["inputs"][0]
            rows, cols = o["block_rows"], o["block_cols"]
            need_complete(ins[0])
            if prov[ins[0]] != [(u, c) for c in cols]:
                raise ValueError(f"{out}: reads wrong input channels")
            dw = so["groups"] > 1
            w = bytes(blob[so["weight"]["offset"] : so["weight"]["offset"] + so["weight"]["bytes"]])
            n, cg, kh, kw = so["weight"]["shape"]
            per = kh * kw * 4
            ref = (
                b"".join(w[r * per : (r + 1) * per] for r in rows)
                if dw
                else b"".join(w[(r * cg + c) * per : (r * cg + c + 1) * per] for r in rows for c in cols)
            )
            got = bytes(blob[o["weight"]["offset"] : o["weight"]["offset"] + o["weight"]["bytes"]])
            if got != ref:
                raise ValueError(f"{out}: weight block differs from the original")
            bsrc = bytes(blob[so["bias"]["offset"] : so["bias"]["offset"] + so["bias"]["bytes"]])
            bgot = bytes(blob[o["bias"]["offset"] : o["bias"]["offset"] + o["bias"]["bytes"]])
            bref = b"".join(bsrc[r * 4 : (r + 1) * 4] for r in rows) if o["block_bias"] else bytes(4 * len(rows))
            if bgot != bref:
                raise ValueError(f"{out}: bias block differs")
            cv = cover.setdefault(v, set())
            bc = biased.setdefault(v, {})
            for r in rows:
                for c in cols if not dw else [r]:
                    if (r, c) in cv:
                        raise ValueError(f"{v}: weight ({r},{c}) used twice")
                    cv.add((r, c))
                if o["block_bias"]:
                    bc[r] = bc.get(r, 0) + 1
            prov[out] = [(v, r) for r in rows]
            if not dw:
                part[out] = (v, frozenset(cols), 1 if o["block_bias"] else 0)
        elif role == "psum":
            a, b = ins
            if prov[a] != prov[b] or a not in part or b not in part:
                raise ValueError(f"{out}: psum of different channels")
            va, ca, ba = part[a]
            vb, cb, bb = part[b]
            if va != vb or ca & cb:
                raise ValueError(f"{out}: overlapping partial sums")
            prov[out] = prov[a]
            part[out] = (va, ca | cb, ba + bb)
        elif role == "residual":
            for t in ins:
                need_complete(t)
            u0, u1 = so["inputs"]
            if (
                [c for _, c in prov[ins[0]]] != [c for _, c in prov[ins[1]]]
                or any(x != u0 for x, _ in prov[ins[0]])
                or any(x != u1 for x, _ in prov[ins[1]])
            ):
                raise ValueError(f"{out}: residual add mismatched")
            prov[out] = [(v, c) for _, c in prov[ins[0]]]
        elif role == "unary":
            need_complete(ins[0])
            u = so["inputs"][0]
            if any(x != u for x, _ in prov[ins[0]]):
                raise ValueError(f"{out}: unary input from another value")
            prov[out] = [(v, c) for _, c in prov[ins[0]]]
        elif role == "block_concat":
            for t in ins:
                need_complete(t)
            prov[out] = [p for t in ins for p in prov[t]]
        elif role == "concat":
            for t in ins:
                need_complete(t)
            offs = {}
            acc = 0
            for t in so["inputs"]:
                offs[t] = acc
                acc += C[t]
            res = []
            for t in ins:
                for x, c in prov[t]:
                    if x not in offs:
                        raise ValueError(f"{out}: concat piece not from an input of {v}")
                    res.append((v, offs[x] + c))
            prov[out] = res
        else:
            raise ValueError(f"{out}: unknown role {role}")
    # every decomposed original conv fully covered, bias once per row
    for v, cv in cover.items():
        o = src[v]
        n, cg = o["weight"]["shape"][:2]
        want = n * (cg if o["groups"] == 1 else 1)
        if len(cv) != want:
            raise ValueError(f"{v}: covered {len(cv)} of {want} weight elements")
        if any(biased[v].get(r, 0) != 1 for r in range(n)):
            raise ValueError(f"{v}: bias not exactly once per output channel")
    return {"convs_verified": len(cover), "ops": len(out_ops)}
