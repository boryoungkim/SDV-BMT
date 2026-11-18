"""CenterPoint-Pillars numeric gate: GPU cuDNN plan vs ONNX Runtime CPU on the raw CenterHead outputs (offline tool).

Run with the system Python (onnxruntime and numpy are used by offline tools only; the benchmark core does not use them).
For each item of an SDV-BMT CenterPoint input index:
  * the device input is the payload body (voxels | num_points | coors, the 40-byte header removed);
  * the native NVIDIA backend (``native/build/sdv-bmt-raw-dump``, the runner's backend code, refused unless its build
    stamp matches the current native sources and raw_dump.cpp) runs the plan with the given DUT runtime options;
  * ONNX Runtime (CPU EP) runs the ONNX model artifact (SHA-256 checked against the plan) on the same tensors.
Per item and per channel group (heatmap logits / regression outputs) it reports max|gpu - ort| / max|ort|.  Criteria
(plan v02, approved 2026-11-13): heatmap <= 1e-4, regression <= 1e-3.  The raw outputs of both are kept (OUT_DIR/gpu,
OUT_DIR/ort) for the output-twin and quality checks.

    python3 tools/centerpoint/validate_numerics.py --index INDEX --onnx MODEL.onnx --plan PLAN.json --weights W.f32 \
        --dut configs/duts/nvidia-rtx-pro-6000-gpu0-cudnn-heuristic.yaml --out-dir DIR [--count N] [--chunk 20]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from sdv_bmt.contracts.ids import canonical_json_bytes  # noqa: E402
from sdv_bmt.contracts.specs import ComponentIdentity, DUTConfig, WorkloadSpec  # noqa: E402
from sdv_bmt.dut import native_source  # noqa: E402
from sdv_bmt.dut.nvidia.adapter import NvidiaCudaDUT  # noqa: E402

NATIVE_DIR = REPO / "native"
RAW_DUMP = NATIVE_DIR / "build" / "sdv-bmt-raw-dump"
RAW_DUMP_SRC = REPO / "tools" / "stage2b_g_real" / "raw_dump.cpp"
HEADER = 40  # payload header: magic 8 B + 8 x u32
P, M, F = 40000, 20, 5
TASK_CLASSES = [1, 2, 2, 1, 2, 2]
REG_PER_TASK = 10  # reg 2, height 1, dim 3, rot 2, vel 2
HEAT_MAX, REG_MAX = 1e-4, 1e-3


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def channel_groups() -> tuple[list[int], list[int]]:
    heat, reg, c = [], [], 0
    for n in TASK_CLASSES:
        reg += list(range(c, c + REG_PER_TASK))
        c += REG_PER_TASK
        heat += list(range(c, c + n))
        c += n
    return heat, reg


def quantiles(v):
    a = np.sort(np.asarray(v, dtype=np.float64))
    pick = lambda p: float(a[min(len(a) - 1, max(0, int(np.ceil(p * len(a))) - 1))])  # noqa: E731
    return {"n": int(len(a)), "min": float(a[0]), "p50": pick(0.5), "p99": pick(0.99), "max": float(a[-1])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dut", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out_dir)
    (out / "gpu").mkdir(parents=True, exist_ok=True)
    (out / "ort").mkdir(parents=True, exist_ok=True)
    stamp = native_source.check_stamp(RAW_DUMP, NATIVE_DIR, [RAW_DUMP_SRC])
    plan_path, weights_path, onnx_path = (Path(os.path.abspath(x)) for x in (a.plan, a.weights, a.onnx))  # keep symlinks
    plan = json.loads(plan_path.read_text())
    onnx_sha = sha256_file(onnx_path)
    if plan["model"]["onnx_sha256"] != onnx_sha:
        raise SystemExit("the plan was not converted from this ONNX file")
    in_bytes, out_bytes = plan["input"]["bytes"], plan["output"]["bytes"]
    program = {"language": "cudnn_plan", "plan": str(plan_path.relative_to(REPO)), "plan_sha256": sha256_file(plan_path),
               "weights": str(weights_path.relative_to(REPO)), "weights_sha256": sha256_file(weights_path),
               "input_bytes": in_bytes, "output_bytes": out_bytes + 32, "task_output_bytes": out_bytes,
               "timing_offset": out_bytes}
    ident = ComponentIdentity(id="numeric-gate", version="1")
    wl = WorkloadSpec(schema_version=1, workload_id="centerpoint-numeric-gate", implementation="none",
                      task="numeric-gate", artifact_sha256=onnx_sha, precision="fp32", batch=1, preprocess=ident,
                      postprocess=ident, codec=ident, output_ready="raw", options={"device_program": program})
    cfg = DUTConfig(**{k: v for k, v in yaml.safe_load(Path(a.dut).read_text()).items()})
    dut = NvidiaCudaDUT(cfg)
    spec = {"device_uuid": cfg.device, "backend": {"name": dut.ADAPTER, "options": dict(cfg.runtime_options)},
            **dut.backend_spec(wl)}

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    idx_path = Path(a.index)
    rows = [json.loads(l) for l in open(idx_path)]
    if a.count:
        rows = rows[: a.count]
    heat_ch, reg_ch = channel_groups()
    per, facts = [], None
    for c0 in range(0, len(rows), a.chunk):
        chunk = rows[c0: c0 + a.chunk]
        bodies = [(idx_path.parent / r["payload_file"]).read_bytes()[HEADER:] for r in chunk]
        assert all(len(b) == in_bytes for b in bodies)
        with tempfile.TemporaryDirectory(prefix="cp-rd-") as d:
            d = Path(d)
            items = []
            with open(d / "staging.bin", "wb") as fh:
                for k, b in enumerate(bodies):
                    items.append({"item_index": c0 + k, "offset": fh.tell(), "nbytes": len(b),
                                  "sha256": hashlib.sha256(b).hexdigest()})
                    fh.write(b)
            (d / "spec.json").write_bytes(canonical_json_bytes(spec))
            (d / "items.json").write_text(json.dumps(items))
            proc = subprocess.run([str(RAW_DUMP), str(d / "spec.json"), str(d / "staging.bin"), str(d / "items.json"),
                                   str(d / "out.bin")], capture_output=True, text=True)
            try:
                reply = json.loads(proc.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError):
                raise SystemExit(f"raw_dump gave no reply: {proc.stderr[-3000:]}") from None
            if not reply.get("ok"):
                raise SystemExit(f"raw_dump failed: {reply} {proc.stderr[-3000:]}")
            f = reply["cudnn_plan_facts"]
            if facts is None:
                facts = f
            elif f["cudnn_algorithm_table_sha256"] != facts["cudnn_algorithm_table_sha256"]:
                raise SystemExit("cuDNN algorithm table changed between chunks")
            raw = (d / "out.bin").read_bytes()
        for k, (r, b) in enumerate(zip(chunk, bodies)):
            g = np.frombuffer(raw[k * out_bytes:(k + 1) * out_bytes], "<f4").reshape(70, 128, 128)
            v = np.frombuffer(b, "<f4", P * M * F).reshape(P, M, F)
            n = np.frombuffer(b, "<i4", P, P * M * F * 4)
            yx = np.frombuffer(b, "<i4", P * 2, P * M * F * 4 + P * 4).reshape(P, 2)
            o = sess.run(["heads"], {"voxels": v, "num_points": n, "coors": yx})[0][0]
            tok = r["source_item_id"]
            (out / "gpu" / f"{tok}.f32").write_bytes(g.tobytes())
            (out / "ort" / f"{tok}.f32").write_bytes(o.astype("<f4").tobytes())
            rec = {"token": tok,
                   "heatmap_rel": float(np.abs(g[heat_ch] - o[heat_ch]).max() / np.abs(o[heat_ch]).max()),
                   "regression_rel": float(max(np.abs(g[c] - o[c]).max() / np.abs(o[c]).max() for c in reg_ch)),
                   "identical_values": float(np.mean(g == o))}
            per.append(rec)
        print(f"{min(c0 + a.chunk, len(rows))}/{len(rows)} worst heat {max(p['heatmap_rel'] for p in per):.3e} "
              f"reg {max(p['regression_rel'] for p in per):.3e}", flush=True)
    worst_h, worst_r = max(p["heatmap_rel"] for p in per), max(p["regression_rel"] for p in per)
    report = {"criteria": {"heatmap_rel_max": HEAT_MAX, "regression_rel_max": REG_MAX,
                           "definition": "per item and channel: max|gpu-ort| / max|ort| (regression: worst channel)"},
              "items": len(per), "worst_heatmap_rel": worst_h, "worst_regression_rel": worst_r,
              "heatmap_rel": quantiles([p["heatmap_rel"] for p in per]),
              "regression_rel": quantiles([p["regression_rel"] for p in per]),
              "pass": worst_h <= HEAT_MAX and worst_r <= REG_MAX, "per_item": per,
              "plan_sha256": program["plan_sha256"], "weights_sha256": program["weights_sha256"], "onnx_sha256": onnx_sha,
              "raw_dump": {"sha256": sha256_file(RAW_DUMP), "stamp": stamp}, "onnxruntime": ort.__version__,
              "dut_config": a.dut, "cudnn_plan_facts": facts,
              "tool_sha256": sha256_file(Path(__file__))}
    (out / "numerics_gpu_vs_ort.json").write_text(json.dumps(report, indent=1))
    print("worst heatmap", worst_h, "worst regression", worst_r, "pass", report["pass"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
