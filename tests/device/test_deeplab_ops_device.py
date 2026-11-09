"""v0.1.2 cuDNN-plan executor on the device (GPU 0): the DeepLab ops and safeguard 2.

A small hand-written plan uses every v0.1.2 op (dilated conv + ReLU, depthwise conv with the residual added before
the ReLU, global average pooling, a 1x1 conv on the pooled vector, broadcast into a concat slice, a plain conv into
the other slice, bilinear resize with align_corners=False, argmax to uint8).  It runs through the native backend
(``native/build/sdv-bmt-raw-dump``, stamp-checked) and is compared with a pure-Python float64 reference of the same
semantics (PyTorch source-index arithmetic for the resize).  Plans whose steps read or write outside a buffer, or
that leave part of the task output unwritten, must be refused at prepare.

NOT RUN (skipped) without the target GPU or the CUDA toolchain.
"""

from __future__ import annotations

import hashlib
import json
import random
import struct
import subprocess

import pytest

from sdv_bmt.contracts.ids import canonical_json_bytes
from sdv_bmt.dut import native_source
from sdv_bmt.dut.native import NATIVE_DIR, NativeRunnerError, describe_native, native_binary
from tests._helpers import REPO_ROOT

UUID = "GPU-f3e03bfa-61ab-acdf-0834-450de8becefc"
RAW_DUMP = NATIVE_DIR / "build" / "sdv-bmt-raw-dump"
OPTIONS = {"residency": "host_pinned", "instrumentation": "cuda_events", "context_sched": "blocking_sync",
           "cudnn": {"math_type": "default", "algo_policy": "heuristic", "workspace_limit_bytes": 268435456},
           "frequency_probe_cycles": 20000}
C_IN, H, W = 4, 8, 8
C, OH, OW = 6, 16, 24


@pytest.fixture(scope="module")
def raw_dump(native_build):
    try:
        describe_native(native_binary("sdv-bmt-native-nvidia"), {"uuid": UUID})
    except NativeRunnerError as exc:
        pytest.skip(f"NOT RUN: target GPU {UUID} not available to the native runner ({exc})")
    proc = subprocess.run(["make", "-s", "-C", str(NATIVE_DIR), "raw-dump"], capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"NOT RUN: raw-dump build failed: {proc.stderr[-500:]}")
    native_source.check_stamp(RAW_DUMP, NATIVE_DIR, [REPO_ROOT / "tools" / "stage2b_g_real" / "raw_dump.cpp"])
    return RAW_DUMP


def _weights(rng):
    blobs, offsets = bytearray(), {}

    def add(name, values):
        blobs.extend(b"\0" * ((-len(blobs)) % 256))
        offsets[name] = len(blobs)
        blobs.extend(struct.pack(f"<{len(values)}f", *values))

    w = {"a_w": [rng.uniform(-0.5, 0.5) for _ in range(C * C_IN * 9)], "a_b": [rng.uniform(-0.2, 0.2) for _ in range(C)],
         "b_w": [rng.uniform(-0.5, 0.5) for _ in range(C * 9)], "b_b": [rng.uniform(-0.2, 0.2) for _ in range(C)],
         "c_w": [rng.uniform(-0.5, 0.5) for _ in range(C * C)], "c_b": [rng.uniform(-0.2, 0.2) for _ in range(C)],
         "d_w": [rng.uniform(-0.5, 0.5) for _ in range(C * C)], "d_b": [rng.uniform(-0.2, 0.2) for _ in range(C)]}
    for k, v in w.items():
        add(k, v)
    return w, offsets, bytes(blobs)


def _plan(off, *, final="argmax", argmax_offset=0, argmax_h=OH, arena_shift=0):
    plane = H * W
    A, B, G, Gc = 0, 4 * C * plane, 4 * 2 * C * plane, 4 * 2 * C * plane + 256
    K = 4 * 2 * C * plane + 512
    R = K + 4 * 2 * C * plane
    arena = R + 4 * 2 * C * OH * OW

    def ref(buf, o=0):
        return {"buf": buf, "offset": o}

    def conv(x, y, c, h, w, k, r, pad, dil, group, act, wo, bo, residual=None):
        s = {"op": "conv", "x": x, "y": y, "residual": residual, "act": act, "w_offset": wo, "b_offset": bo,
             "c": c, "h": h, "w": w, "k": k, "r": r, "s": r, "oh": h, "ow": w, "pad_h": pad, "pad_w": pad,
             "stride_h": 1, "stride_w": 1, "dil_h": dil, "dil_w": dil, "group": group}
        if residual is not None:
            s["res_before_act"] = True
        return s

    steps = [
        conv(ref("input"), ref("arena", A + arena_shift), C_IN, H, W, C, 3, 2, 2, 1, 3, off["a_w"], off["a_b"]),
        conv(ref("arena", A), ref("arena", B), C, H, W, C, 3, 1, 1, C, 3, off["b_w"], off["b_b"], ref("arena", A)),
        {"op": "gap", "x": ref("arena", B), "y": ref("arena", G), "c": C, "h": H, "w": W},
        conv(ref("arena", G), ref("arena", Gc), C, 1, 1, C, 1, 0, 1, 1, 3, off["c_w"], off["c_b"]),
        {"op": "broadcast", "x": ref("arena", Gc), "y": ref("arena", K), "c": C, "oh": H, "ow": W},
        conv(ref("arena", B), ref("arena", K + 4 * C * plane), C, H, W, C, 1, 0, 1, 1, 0, off["d_w"], off["d_b"]),
    ]
    if final == "argmax":
        steps.append({"op": "resize_bilinear", "x": ref("arena", K), "y": ref("arena", R), "c": 2 * C, "h": H, "w": W,
                      "oh": OH, "ow": OW})
        steps.append({"op": "argmax_u8", "x": ref("arena", R), "y": ref("output", argmax_offset), "c": 2 * C,
                      "h": argmax_h, "w": OW})
        out_bytes = OH * OW
    else:
        steps.append({"op": "resize_bilinear", "x": ref("arena", K), "y": ref("output"), "c": 2 * C, "h": H, "w": W,
                      "oh": OH, "ow": OW})
        out_bytes = 4 * 2 * C * OH * OW
    return {"schema": "sdv-bmt-cudnn-plan/1", "model": {"onnx_sha256": "0" * 64}, "arena_bytes": arena,
            "input": {"bytes": 4 * C_IN * H * W}, "output": {"bytes": out_bytes}, "steps": steps}, out_bytes


def _run(raw_dump, tmp_path, plan, weights, out_bytes, inputs):
    (tmp_path / "w.f32").write_bytes(weights)
    plan["weights"] = {"sha256": hashlib.sha256(weights).hexdigest(), "bytes": len(weights)}
    plan_text = json.dumps(plan, sort_keys=True)
    (tmp_path / "plan.json").write_text(plan_text)
    program = {"kind": "cudnn_plan", "plan_path": str(tmp_path / "plan.json"),
               "plan_sha256": hashlib.sha256(plan_text.encode()).hexdigest(), "weights_path": str(tmp_path / "w.f32"),
               "weights_sha256": hashlib.sha256(weights).hexdigest(), "input_bytes": 4 * C_IN * H * W,
               "output_bytes": out_bytes + 32, "task_output_bytes": out_bytes, "timing_offset": out_bytes,
               "artifact_sha256": "0" * 64}
    spec = {"device_uuid": UUID, "backend": {"name": "nvidia_cuda", "options": OPTIONS}, "program": program}
    (tmp_path / "spec.json").write_bytes(canonical_json_bytes(spec))
    rows = []
    with open(tmp_path / "staging.bin", "wb") as fh:
        for i, data in enumerate(inputs):
            rows.append({"item_index": i, "offset": fh.tell(), "nbytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest()})
            fh.write(data)
    (tmp_path / "items.json").write_text(json.dumps(rows))
    proc = subprocess.run([str(raw_dump), str(tmp_path / "spec.json"), str(tmp_path / "staging.bin"),
                           str(tmp_path / "items.json"), str(tmp_path / "out.bin")], capture_output=True, text=True)
    reply = json.loads(proc.stdout.strip().splitlines()[-1])
    return reply, (tmp_path / "out.bin").read_bytes() if reply.get("ok") else b""


# ---------------------------------------------------------------- float64 reference of the same semantics


def _conv(x, cin, h, w, wt, b, cout, r, pad, dil, group):
    out = []
    cpg = cin // group
    for k in range(cout):
        g = k // (cout // group)
        plane = []
        for oy in range(h):
            for ox in range(w):
                s = 0.0
                for ci in range(cpg):
                    c = g * cpg + ci
                    for ky in range(r):
                        for kx in range(r):
                            iy, ix = oy - pad + ky * dil, ox - pad + kx * dil
                            if 0 <= iy < h and 0 <= ix < w:
                                s += wt[((k * cpg + ci) * r + ky) * r + kx] * x[c][iy * w + ix]
                plane.append(s + b[k])
        out.append(plane)
    return out


def _reference(x, w):
    relu = lambda p: [max(v, 0.0) for v in p]  # noqa: E731
    a = [relu(p) for p in _conv(x, C_IN, H, W, w["a_w"], w["a_b"], C, 3, 2, 2, 1)]
    bconv = _conv(a, C, H, W, w["b_w"], w["b_b"], C, 3, 1, 1, C)
    b = [relu([u + v for u, v in zip(p, q)]) for p, q in zip(bconv, a)]
    g = [[sum(p) / len(p)] for p in b]
    gc = [relu(p) for p in _conv(g, C, 1, 1, w["c_w"], w["c_b"], C, 1, 0, 1, 1)]
    k = [p * (H * W) for p in gc] + _conv(b, C, H, W, w["d_w"], w["d_b"], C, 1, 0, 1, 1)
    sh, sw = H / OH, W / OW
    res = []
    for p in k:
        out = []
        for oy in range(OH):
            fy = max(sh * (oy + 0.5) - 0.5, 0.0)
            y0 = int(fy)
            y1 = y0 + (1 if y0 < H - 1 else 0)
            ly1 = fy - y0
            for ox in range(OW):
                fx = max(sw * (ox + 0.5) - 0.5, 0.0)
                x0 = int(fx)
                x1 = x0 + (1 if x0 < W - 1 else 0)
                lx1 = fx - x0
                out.append((1 - ly1) * ((1 - lx1) * p[y0 * W + x0] + lx1 * p[y0 * W + x1])
                           + ly1 * ((1 - lx1) * p[y1 * W + x0] + lx1 * p[y1 * W + x1]))
        res.append(out)
    labels = bytes(max(range(2 * C), key=lambda c: (res[c][i], -c)) for i in range(OH * OW))
    return res, labels


def _inputs(rng, n):
    out = []
    for _ in range(n):
        x = [[rng.uniform(-2, 2) for _ in range(H * W)] for _ in range(C_IN)]
        x32 = [list(struct.unpack(f"<{H * W}f", struct.pack(f"<{H * W}f", *p))) for p in x]
        out.append((x32, struct.pack(f"<{C_IN * H * W}f", *[v for p in x32 for v in p])))
    return out


def test_deeplab_ops_match_the_reference(raw_dump, tmp_path):
    rng = random.Random(20261106)
    w, off, blob = _weights(rng)
    w32 = {k: list(struct.unpack(f"<{len(v)}f", struct.pack(f"<{len(v)}f", *v))) for k, v in w.items()}
    cases = _inputs(rng, 3)
    plan, out_bytes = _plan(off, final="logits")
    (tmp_path / "logits").mkdir()
    reply, out = _run(raw_dump, tmp_path / "logits", plan, blob, out_bytes, [d for _, d in cases])
    assert reply["ok"], reply
    facts = reply["cudnn_plan_facts"]
    assert "glue2_source_sha256" in facts and facts["cudnn_convolutions"] == 4
    worst = 0.0
    for n, (x, _) in enumerate(cases):
        ref, _ = _reference(x, w32)
        got = struct.unpack(f"<{2 * C * OH * OW}f", out[n * out_bytes:(n + 1) * out_bytes])
        scale = max(abs(v) for p in ref for v in p)
        worst = max(worst, max(abs(g - r) for g, r in zip(got, [v for p in ref for v in p])) / scale)
    assert worst < 1e-5, worst
    plan, out_bytes = _plan(off)
    (tmp_path / "labels").mkdir()
    reply, out = _run(raw_dump, tmp_path / "labels", plan, blob, out_bytes, [d for _, d in cases])
    assert reply["ok"], reply
    for n, (x, _) in enumerate(cases):
        _, labels = _reference(x, w32)
        got = out[n * out_bytes:(n + 1) * out_bytes]
        assert sum(a != b for a, b in zip(got, labels)) == 0, n


@pytest.mark.parametrize("change,message", [
    ({"arena_shift": 1 << 30}, "writes outside the arena buffer"),
    ({"argmax_offset": 8}, "writes outside the task output buffer"),  # would overwrite the timing words
    ({"argmax_h": OH - 1}, "do not write the whole task output"),     # last row never written
])
def test_plans_outside_their_buffers_or_partly_unwritten_are_refused(raw_dump, tmp_path, change, message):
    rng = random.Random(1)
    _, off, blob = _weights(rng)
    plan, out_bytes = _plan(off, **change)
    reply, _ = _run(raw_dump, tmp_path, plan, blob, out_bytes, [d for _, d in _inputs(rng, 1)])
    assert not reply["ok"] and message in reply["error"], reply
