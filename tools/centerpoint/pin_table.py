#!/usr/bin/env python3
"""Obtain the CenterPoint cuDNN algorithm table with the heuristic policy and write the pinned DUT config (offline).

Same procedure as the DeepLab family (tools/deeplab_family/pin_table.py): the heuristic choice depends on the
convolution geometry, the device and the cuDNN library only (not on the input values), so it is taken from one
``raw_dump`` prepare of the canonical plan (the runner's backend code, stamp-checked) with the heuristic DUT config, on
the first item of the given input index.  The probe run (heuristic policy, asap, the mini input) must later realize the
same table SHA-256.

    python3 tools/centerpoint/pin_table.py --index datasets/nuscenes-mini-val4/index.jsonl \
        --heuristic-dut configs/duts/nvidia-rtx-pro-6000-gpu0-cudnn-heuristic.yaml \
        --out-dut configs/duts/nvidia-rtx-pro-6000-gpu0-cudnn-centerpoint-pillar02.yaml --report REPORT.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "deeplab_family"))
from sdv_bmt.contracts.specs import DUTConfig, WorkloadSpec, parse_strict  # noqa: E402
from sdv_bmt.dut.nvidia.adapter import NvidiaCudaDUT, cudnn_table_sha256  # noqa: E402
from sdv_bmt.runner.plan import _parse_file  # noqa: E402
from sdv_bmt.workload.centerpoint import HEADER, INPUT_BYTES, RAW_BYTES  # noqa: E402
from validate_numerics import RawDump  # noqa: E402

ALGO_NAMES = {0: "IMPLICIT_GEMM", 1: "IMPLICIT_PRECOMP_GEMM", 2: "GEMM", 3: "DIRECT", 4: "FFT", 5: "FFT_TILING",
              6: "WINOGRAD", 7: "WINOGRAD_NONFUSED"}
WORKLOAD = REPO / "configs/workloads/centerpoint-pillar02-nus-v1.yaml"
DUT_ID = "nvidia-rtx-pro-6000-gpu0-cudnn-centerpoint-pillar02"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--heuristic-dut", required=True)
    ap.add_argument("--out-dut", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    w = _parse_file(WORKLOAD, lambda d, where: parse_strict(WorkloadSpec, d, where))
    cfg = _parse_file(Path(a.heuristic_dut).resolve(), lambda d, where: parse_strict(DUTConfig, d, where))
    if cfg.runtime_options["cudnn"]["algo_policy"] != "heuristic":
        raise SystemExit("--heuristic-dut must use cudnn.algo_policy heuristic")
    index = Path(a.index)
    row = json.loads(index.read_text(encoding="utf-8").splitlines()[0])
    body = (index.parent / row["payload_file"]).read_bytes()[HEADER.size:]
    if len(body) != INPUT_BYTES:
        raise SystemExit(f"payload body is {len(body)} bytes, expected {INPUT_BYTES}")
    rd = RawDump(NvidiaCudaDUT(cfg), cfg, Path("/dev/shm"))
    rd.run("canonical", w, [(0, body)], RAW_BYTES)
    facts = rd.facts["canonical"]
    if facts["plan_sha256"] != w.options["device_program"]["plan_sha256"]:
        raise SystemExit("realized plan SHA-256 differs from the workload's")
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
    text = f"""# CenterPoint part (ACTUAL): GPU 0 executing the CenterPoint-Pillars (pillar02) cuDNN plan with a pinned algorithm
# table (cuDNN 9.1.0). The runner fails DUT.prepare if cuDNN rejects a pinned entry, and records the heuristic's own
# choice next to it. Prepared inputs in pinned host memory (per-request H2D inside DUT-service), CUDA-event segment
# timing, frequency probe (20000 SM cycles) first in each request. Same runtime options as the YOLOX and DeepLab pinned
# configurations. Math type 0 = CUDNN_DEFAULT_MATH (cuDNN 9.1.0 documentation: TF32 Tensor Core operation is permitted on
# Ampere and later; the table records the math type provided, not which kernels ran).
schema_version: 1
dut_config_id: {DUT_ID}
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
    # deterministic result within the workspace limit) on this device and library by tools/centerpoint/pin_table.py
    # (raw_dump prepare of the canonical plan); table SHA-256 {sha}.
    algorithms:
{rows}
  frequency_probe_cycles: {cfg.runtime_options.get("frequency_probe_cycles", 0)}
"""
    Path(a.out_dut).write_text(text, encoding="utf-8")
    report = {"table_sha256": sha, "convolutions": len(table), "algorithm_counts": counts, "math_types": maths,
              "workspace_bytes": facts["cudnn_workspace_bytes"], "heuristic_agreements": facts["cudnn_heuristic_agreements"],
              "plan_sha256": facts["plan_sha256"], "raw_dump_sha256": rd.sha256, "raw_dump_stamp": rd.source,
              "detail": facts["cudnn_algorithm_detail"], "glue3_source_sha256": facts.get("glue3_source_sha256"),
              "launches_per_request": facts["plan_launches_per_request"], "arena_bytes": facts["arena_bytes"],
              "index_item": row["source_item_id"]}
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("table_sha256", "convolutions", "algorithm_counts", "math_types",
                                             "workspace_bytes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
