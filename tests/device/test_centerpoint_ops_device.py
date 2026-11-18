"""cuDNN-plan executor on the device (GPU 0): the CenterPoint-Pillars ops (CenterPoint part).

A small hand-written plan runs pillar_features (the 11 decorated point features, the linear layer, ReLU, maximum
over the points), pillar_scatter (zero fill of the BEV canvas, then the non-empty pillars' columns) and depth_to_space2
into the task output.  It runs through the native backend (``native/build/sdv-bmt-raw-dump``, stamp-checked) and is
compared bit for bit with a pure-Python reference that rounds every IEEE operation to float32 in the kernels' order
(double rounding is exact for single +, -, *, /).  Plans outside the ops' limits must be refused at prepare.

NOT RUN (skipped) without the target GPU or the CUDA toolchain.
"""

from __future__ import annotations

import hashlib
import json
import random
import struct
import subprocess

from sdv_bmt.contracts.ids import canonical_json_bytes
from tests.device.test_deeplab_ops_device import OPTIONS, UUID, raw_dump  # noqa: F401  (fixture re-used)

P, M, C, H, W = 9, 6, 32, 4, 6
CONSTS = (0.2, 0.2, 8.0, -51.1, -51.1, -1.0)  # vx, vy, vz, x offset, y offset, z offset (voxel centre terms)


def f32(v: float) -> float:
    return struct.unpack("<f", struct.pack("<f", v))[0]


def _bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _case(rng):
    counts = [M, 3, 1, 0, M, 2, 5, 4, 1]  # pillar 3 is padding (no points); pillar 8 lies outside the canvas
    coords = [(0, 0), (1, 2), (3, 5), (2, 2), (2, 3), (0, 5), (3, 0), (1, 4), (1, W)]
    vox = []
    for p in range(P):
        for t in range(M):
            if t < counts[p]:
                y, x = coords[p]
                vox += [f32(-51.2 + 0.2 * x + rng.uniform(0, 0.2)), f32(-51.2 + 0.2 * y + rng.uniform(0, 0.2)),
                        f32(rng.uniform(-3, 2)), f32(rng.uniform(0, 255)), f32(rng.choice([0.0, 0.05, 0.4]))]
            else:
                vox += [0.0] * 5
    data = (struct.pack(f"<{P * M * 5}f", *vox) + struct.pack(f"<{P}i", *counts)
            + struct.pack(f"<{2 * P}i", *[v for yx in coords for v in yx]))
    return vox, counts, coords, data


def _reference(vox, counts, coords, w, b):
    vx, vy, vz, xo, yo, zo = (f32(c) for c in CONSTS)
    feat = [[0.0] * P for _ in range(C)]
    for p in range(P):
        n = counts[p]
        v = [vox[(p * M + t) * 5:(p * M + t + 1) * 5] for t in range(M)]
        mean = []
        for k in range(3):
            s = 0.0
            for t in range(M):
                s = f32(s + v[t][k])
            mean.append(f32(s / float(max(n, 1))))
        cy, cx = float(coords[p][0]), float(coords[p][1])
        rows = []
        for t in range(M):
            if t < n:
                f = list(v[t]) + [f32(v[t][k] - mean[k]) for k in range(3)]
                f += [f32(v[t][0] - f32(f32(cx * vx) + xo)), f32(v[t][1] - f32(f32(cy * vy) + yo)),
                      f32(v[t][2] - f32(f32(0.0 * vz) + zo))]
            else:
                f = [0.0] * 11
            rows.append(f)
        for c in range(C):
            m = 0.0
            for f in rows:
                acc = 0.0
                for k in range(11):
                    acc = f32(acc + f32(f[k] * w[c * 11 + k]))
                acc = f32(acc + b[c])
                if acc > m:
                    m = acc
            feat[c][p] = m
    canvas = [[[0.0] * W for _ in range(H)] for _ in range(C)]
    for p in range(P):
        y, x = coords[p]
        if counts[p] > 0 and 0 <= y < H and 0 <= x < W:
            for c in range(C):
                canvas[c][y][x] = feat[c][p]
    out = []
    for c in range(C // 4):
        for oy in range(2 * H):
            for ox in range(2 * W):
                out.append(canvas[c * 4 + (oy % 2) * 2 + (ox % 2)][oy // 2][ox // 2])
    return out


def _plan(w_off, b_off, *, channels=C, points=M):
    in_n, in_c = 4 * P * M * 5, 4 * P * M * 5 + 4 * P
    canvas = ((4 * C * P + 255) // 256) * 256
    arena = canvas + 4 * C * H * W
    out_bytes = 4 * C * H * W

    def ref(buf, o=0):
        return {"buf": buf, "offset": o}

    steps = [
        {"op": "pillar_features", "x": ref("input"), "num_points": ref("input", in_n), "coors": ref("input", in_c),
         "y": ref("arena"), "pillars": P, "points": points, "features": 5, "channels": channels,
         "w_offset": w_off, "b_offset": b_off, "consts_f32_bits": [_bits(c) for c in CONSTS]},
        {"op": "pillar_scatter", "x": ref("arena"), "num_points": ref("input", in_n), "coors": ref("input", in_c),
         "y": ref("arena", canvas), "pillars": P, "channels": C, "h": H, "w": W},
        {"op": "depth_to_space2", "x": ref("arena", canvas), "y": ref("output"), "c": C // 4, "h": H, "w": W},
    ]
    return {"schema": "sdv-bmt-cudnn-plan/1", "model": {"onnx_sha256": "0" * 64}, "arena_bytes": arena,
            "input": {"bytes": 4 * P * M * 5 + 12 * P}, "output": {"bytes": out_bytes}, "steps": steps}, out_bytes


def _run(raw_dump, tmp_path, plan, weights, out_bytes, inputs):  # noqa: F811
    (tmp_path / "w.f32").write_bytes(weights)
    plan["weights"] = {"sha256": hashlib.sha256(weights).hexdigest(), "bytes": len(weights)}
    plan_text = json.dumps(plan, sort_keys=True)
    (tmp_path / "plan.json").write_text(plan_text)
    program = {"kind": "cudnn_plan", "plan_path": str(tmp_path / "plan.json"),
               "plan_sha256": hashlib.sha256(plan_text.encode()).hexdigest(), "weights_path": str(tmp_path / "w.f32"),
               "weights_sha256": hashlib.sha256(weights).hexdigest(), "input_bytes": plan["input"]["bytes"],
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


def _weights(rng):
    w = [f32(rng.uniform(-0.3, 0.3)) for _ in range(C * 11)]
    b = [f32(rng.uniform(-0.5, 0.5)) for _ in range(C)]
    b_off = ((4 * len(w) + 255) // 256) * 256
    blob = struct.pack(f"<{len(w)}f", *w) + b"\0" * (b_off - 4 * len(w)) + struct.pack(f"<{C}f", *b)
    return w, b, 0, b_off, blob


def test_centerpoint_ops_match_the_float32_reference(raw_dump, tmp_path):  # noqa: F811
    rng = random.Random(20261115)
    w, b, w_off, b_off, blob = _weights(rng)
    cases = [_case(rng) for _ in range(3)]
    plan, out_bytes = _plan(w_off, b_off)
    reply, out = _run(raw_dump, tmp_path, plan, blob, out_bytes, [c[3] for c in cases])
    assert reply.get("ok"), reply
    assert reply["cudnn_plan_facts"]["cudnn_convolutions"] == 0
    for k, (vox, counts, coords, _data) in enumerate(cases):
        got = list(struct.unpack(f"<{out_bytes // 4}f", out[k * out_bytes:(k + 1) * out_bytes]))
        assert got == _reference(vox, counts, coords, w, b), k


def test_centerpoint_ops_outside_their_limits_are_refused(raw_dump, tmp_path):  # noqa: F811
    rng = random.Random(7)
    w, b, w_off, b_off, blob = _weights(rng)
    case = _case(rng)
    for kwargs, message in (({"channels": 16}, "32..1024 channels"), ({"points": 40}, "1..32 points")):
        plan, out_bytes = _plan(w_off, b_off, **kwargs)
        sub = tmp_path / str(kwargs["channels"] if "channels" in kwargs else kwargs["points"])
        sub.mkdir()
        reply, _ = _run(raw_dump, sub, plan, blob, out_bytes, [case[3]])
        assert not reply.get("ok") and message in str(reply), (kwargs, reply)
