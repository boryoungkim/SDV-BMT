#!/usr/bin/env python3
"""Obtain a DeepLab model's cuDNN algorithm table with the heuristic policy and write the pinned DUT config (offline).

v0.1.2 DeepLab part, plan v03 section 8.1 step 4.  The heuristic choice depends on the convolution geometry, the
device and the cuDNN library only (not on pixel values), so it is taken from one ``raw_dump`` prepare of the
canonical plan (the runner's backend code, stamp-checked) with the heuristic DUT config.  The probe run of the
benchmark (heuristic policy, asap, the val images) must later realize the same table SHA-256; if it does not, the
numeric gates are repeated with the probe's table (plan v03 section 8.2).

    python3 tools/deeplab_family/pin_table.py --model r50 --heuristic-dut configs/duts/nvidia-rtx-pro-6000-gpu0-cudnn-heuristic.yaml \
        --out-dut configs/duts/nvidia-rtx-pro-6000-gpu0-cudnn-deeplabv3plus-r50.yaml --report REPORT.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdv_bmt.contracts.specs import DUTConfig, WorkloadSpec, parse_strict  # noqa: E402
from sdv_bmt.dut.nvidia.adapter import NvidiaCudaDUT, cudnn_table_sha256  # noqa: E402
from sdv_bmt.runner.plan import _parse_file  # noqa: E402
import inputs as I  # noqa: E402
from validate_numerics import RawDump  # noqa: E402

ALGO_NAMES = {0: "IMPLICIT_GEMM", 1: "IMPLICIT_PRECOMP_GEMM", 2: "GEMM", 3: "DIRECT", 4: "FFT", 5: "FFT_TILING",
              6: "WINOGRAD", 7: "WINOGRAD_NONFUSED"}
NAMES = {"r18": "ResNet-18", "r50": "ResNet-50", "r101": "ResNet-101"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(NAMES))
    ap.add_argument("--heuristic-dut", required=True)
    ap.add_argument("--out-dut", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    wpath = REPO / f"configs/workloads/deeplabv3plus-{a.model}-cityscapes-v1.yaml"
    w = _parse_file(wpath, lambda d, where: parse_strict(WorkloadSpec, d, where))
    cfg = _parse_file(Path(a.heuristic_dut).resolve(), lambda d, where: parse_strict(DUTConfig, d, where))
    if cfg.runtime_options["cudnn"]["algo_policy"] != "heuristic":
        raise SystemExit("--heuristic-dut must use cudnn.algo_policy heuristic")
    H, W = w.options["input_height"], w.options["input_width"]
    rd = RawDump(NvidiaCudaDUT(cfg), cfg, Path("/dev/shm"))
    rd.run("canonical", w, [(0, I.tensor_from_rgb(I.synthetic_rgb(0, H, W), H, W).tobytes())], H * W)
    facts = rd.facts["canonical"]
    table = [tuple(r) for r in facts["cudnn_algorithm_table"]]
    sha = cudnn_table_sha256(tuple(table))
    if sha != facts["cudnn_algorithm_table_sha256"]:
        raise SystemExit("table SHA-256 differs from the native runner's")
    counts: dict[str, int] = {}
    for algo, _ in table:
        counts[ALGO_NAMES.get(algo, str(algo))] = counts.get(ALGO_NAMES.get(algo, str(algo)), 0) + 1
    maths = sorted({m for _, m in table})
    rows = "\n".join(f"      - [{algo}, {math}]" for algo, math in table)
    legend = ", ".join(f"{x} = {ALGO_NAMES.get(x, '?')}" for x in sorted({t[0] for t in table}))
    text = f"""# v0.1.2 DeepLab part (ACTUAL): GPU 0 executing the DeepLabV3+ {NAMES[a.model]} cuDNN plan with a pinned algorithm table
# (cuDNN 9.1.0). The runner fails DUT.prepare if cuDNN rejects a pinned entry, and records the heuristic's own choice next
# to it. Prepared inputs in pinned host memory (per-request H2D inside DUT-service), CUDA-event segment timing, frequency
# probe (20000 SM cycles) first in each request. Same runtime options as the YOLOX pinned configurations.
# Math type 0 = CUDNN_DEFAULT_MATH (cuDNN 9.1.0 documentation: TF32 Tensor Core operation is permitted on Ampere and later;
# the table records the math type provided, not which kernels ran).
schema_version: 1
dut_config_id: nvidia-rtx-pro-6000-gpu0-cudnn-deeplabv3plus-{a.model}
adapter: nvidia_cuda
device: {cfg.device}
max_inflight: 1
runtime_options:
  residency: host_pinned
  instrumentation: cuda_events
  context_sched: blocking_sync
  cudnn:
    math_type: default
    algo_policy: pinned
    workspace_limit_bytes: {cfg.runtime_options["cudnn"]["workspace_limit_bytes"]}
    # (algorithm, math type) per convolution in plan order, cuDNN 9.1.0 enum values: {legend};
    # math type {maths}. Obtained from the heuristic policy (cudnnGetConvolutionForwardAlgorithm_v7, first supported
    # deterministic result within the workspace limit) on this device and library by tools/deeplab_family/pin_table.py
    # (raw_dump prepare of the canonical plan); table SHA-256 {sha}.
    algorithms:
{rows}
  frequency_probe_cycles: {cfg.runtime_options.get("frequency_probe_cycles", 0)}
"""
    Path(a.out_dut).write_text(text, encoding="utf-8")
    report = {"model": a.model, "table_sha256": sha, "convolutions": len(table), "algorithm_counts": counts,
              "math_types": maths, "workspace_bytes": facts["cudnn_workspace_bytes"],
              "heuristic_agreements": facts["cudnn_heuristic_agreements"], "plan_sha256": facts["plan_sha256"],
              "raw_dump_sha256": rd.sha256, "raw_dump_stamp": rd.source, "detail": facts["cudnn_algorithm_detail"],
              "glue_source_sha256": facts["glue_source_sha256"], "glue2_source_sha256": facts.get("glue2_source_sha256"),
              "launches_per_request": facts["plan_launches_per_request"], "arena_bytes": facts["arena_bytes"]}
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("model", "table_sha256", "convolutions", "algorithm_counts", "math_types",
                                             "workspace_bytes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
