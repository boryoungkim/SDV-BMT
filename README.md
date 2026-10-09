# SDV-BMT — Model-level Accelerator Benchmark Core

**v0.1.0** (2026-10-09) · research benchmark framework · [Apache-2.0](LICENSE) · [changelog](CHANGELOG.md) ·
[third-party notices](THIRD_PARTY_NOTICES.md)

> **Disclaimer.** SDV-BMT v0.1 is a research benchmark framework. Its Model-level measurements do not
> establish WCET, FTTI, ISO/PAS 8800 compliance, ISO 26262 compliance, SOTIF compliance, or vehicle-level
> safety. Percentile latency is distribution characterization, not a worst-case timing guarantee.
> `max_observed` is not WCET.

## What SDV-BMT is

SDV-BMT runs one model workload on an accelerator (the device under test, DUT) under a planned request
schedule and records **evidence**: what was planned, what was actually realized, and exactly how each number
was measured. Metrics are computed afterwards from that evidence alone.

The long-term targets are automotive NPUs. Version 0.1.0 validates the measurement system on an NVIDIA GPU, the
first *reference accelerator*. GPU results are not NPU results, are not a GPU ranking, and predict nothing about
NPU timing.

## Research question

> Can a Model-level benchmark distinguish the planned configuration from the realized execution conditions,
> keep its measurement boundaries valid, and localize observed latency variation — without attributing a
> difference beyond what the evidence supports?

The guiding rules: *Control what is planned. Verify what is realized. Localize what changed. Do not attribute
beyond the evidence.* — *No attributed delta without validated factor isolation.* — *A configured condition and a
realized condition are not assumed to be identical.*

## Architecture

```text
Python control plane            plan loading, strict validation, preflight, identities
        │  resolved run specification (phase protocol, one native process per run)
        ▼
C++ native measurement plane    realtime issuer, admission (N=1, queue 0, drop_new), timeouts,
                                host-clock timestamps, DUT backend (fake or NVIDIA)  →  facts only
        │  append-only evidence bundle: events, outputs, input items, manifest, realization evidence
        ▼
Python evaluation plane         Model-level Metric Set v1, task quality, Design / Realization /
                                Measurement Guards, analysis tools
```

- **C++ records facts; Python computes meaning.** The native runner computes no metric; percentiles,
  deadlines, task quality and statistics are evaluated in Python from the bundle only.
- No Python runs in the per-request measured path; the native issuer spins on the host monotonic clock.
- Portable semantics are kept apart from vendor observation: CUDA, cuDNN, NVML and NVTX appear only in
  `native/src/nvidia/`, `src/sdv_bmt/dut/nvidia/` and namespaced evidence keys (`nvidia.*`).
- Timeouts follow the Stage 1A semantics on real time: events are decided in the order of their semantic
  timestamps (arrival `t_sched`, timeout `t_submit_begin + request_timeout_ns`, completion `t_complete`); the tie
  order completion → timeout → arrival applies only to equal timestamps; a timed-out request keeps the slot until
  its late completion.

## Model-level Metric Set v1

Catalog `sdv-bmt-model-metric-set-v1` (91 metrics), computed by `sdv-bmt evaluate` from a bundle. Every result
carries its layer and its population (planned / completed / sample count, coverage).

| layer | metrics (examples) |
|---|---|
| performance | `bmt.request_completion.*`, `bmt.dut_service.*`, `bmt.output_ready.*` (p50 / p90 / p99 / `max_observed`), `bmt.offered_rate`, `bmt.achieved_completion_rate`, `bmt.closed_loop_completion_rate`, state counts and rates |
| timeliness | `bmt.deadline.{completion,output_ready}_miss_{count,rate}`, `bmt.required_rate`, `bmt.required_rate_met` |
| task quality | `quality.detection.coverage`, `quality.detection.conditional.{ap, ap50, ap75, ap_small, ap_medium, ap_large, ar1, ar10, ar100, precision, recall, f1, tp_count, fp_count, fn_count, fp_per_image, fn_per_gt}` |
| diagnostics | `bmt.issue_lag.*`, `bmt.submit.*`, `bmt.post_submit.*`, `bmt.output_stage.*`, `bmt.h2d.*`, `bmt.device_execution_bracket.*`, `bmt.d2h.*`, `bmt.output.identity_rate` |
| validity / evidence | `data.item_count`, `data.annotation_coverage`, `data.integrity.*_verified_rate`, `data.traceability.complete` |

Rules: nearest-rank percentiles; a p99 needs at least 100 samples, otherwise `invalid(insufficient_samples)`;
planned percentiles count non-completed requests as +∞ and are `censored` (report-only) when the rank lands
there; a non-valid result never carries a number; DUT-service is not device execution time and the device
execution bracket is not device busy time; deadlines are evaluation parameters, not FTTI; detection quality is
*conditional* on completed outputs and always reported with its coverage; output identity is a determinism
diagnostic, not accuracy. Definitions: [`docs/metrics_catalog.md`](docs/metrics_catalog.md) §8 (Korean).

## Evidence and validity

- A run writes an append-only bundle: `run_spec.json`, `input_items.jsonl`, `events.jsonl`, `stress.jsonl`,
  `outputs.bin`, and — when the workload declares a task-quality evaluator — `quality_reference.jsonl` with the
  frozen ground truth; `manifest.json` (written last) records every raw file's SHA-256, the identities, the
  measurement boundary and one realization-evidence record per realized condition (`observed`,
  `self_reported`, `virtual`, `not_observable`, `not_applicable`).
- Evaluation never modifies raw files and writes a new `evaluations/<eval_id>/`.
- `evaluate_contrast_validity` checks a pair of runs with three guards: **Design** (only the intended factor
  varied), **Realization** (the fixed conditions were realized equivalently, as far as observed), and
  **Measurement** (same boundary, catalog, protocol and evaluator). Missing evidence is never equivalence; a
  numeric difference is never an attributable factor effect unless all three pass. Guard results are structured
  validity results, not metrics.

Details: [`docs/architecture.md`](docs/architecture.md), [`docs/measurement_protocol.md`](docs/measurement_protocol.md),
[`docs/evidence.md`](docs/evidence.md) (Korean).

## Reference implementation in v0.1.0

| part | status |
|---|---|
| deterministic core (virtual clock, FakeDUT) | MOCK; golden regression cases G-1..G-5 |
| realtime harness, fake native backend | MOCK; harness-floor and conformance tests |
| NVIDIA backend | ACTUAL on one reference GPU (NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition): CUDA 13.0 (headers, NVRTC, driver API, NVML), cuDNN 9.1 plan executor with a pinned algorithm table, CUDA event brackets, optional Nsight Systems cross-check |
| real-model workload | YOLOX-S (official ONNX, converted offline to a cuDNN plan) on the first 1,000 COCO val2017 images, FP32, batch 1, N = 1 |

## Quick start (no GPU needed)

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"          # runtime: PyYAML; dev: pytest
make -C native portable                    # fake native runner + C++ unit tests (g++, no CUDA)

# deterministic core (MOCK, virtual clock)
.venv/bin/sdv-bmt validate configs/experiments/g2-recorded-scenario-overload.yaml
.venv/bin/sdv-bmt run      configs/experiments/g2-recorded-scenario-overload.yaml --out runs/demo
.venv/bin/sdv-bmt evaluate runs/demo/g2-recorded-scenario-overload/<run_id>

# realtime harness with the fake native backend (MOCK, host clock)
.venv/bin/sdv-bmt run configs/experiments/r1-fake-native-harness-floor.yaml --out runs/demo
```

The realtime example pins the native runner (`realtime.cpu_affinity`) to CPUs 0–15 and 32–47 of the
reference host; on a machine without those CPUs the run stops with `cpu_affinity_not_realized`. Edit that
list to CPUs available on your machine to run it locally.

`sdv-bmt evaluate` uses Metric Set v1; `--catalog bmt-portable-v1-stage1a` reproduces evaluations of the
earlier Stage 1A catalog. `--eval-config FILE` overrides the deadline and may set `required_rate_hz` and
`quality_operating_point`. Exit codes: `0` success, `1` run executed but failed, `2` rejected.

## Testing

| scope | command | GPU tests |
|---|---|---|
| CPU / portable (also GitHub Actions) | `make -C native portable && native/build/sdv-bmt-native-tests` and `SDV_BMT_MAKE_TARGETS=portable .venv/bin/python -m pytest -q -rs` | reported NOT RUN (skipped), never passed |
| local GPU release validation | `make -C native && native/build/sdv-bmt-native-tests` and `.venv/bin/python -m pytest -q` | run on the GPU named by UUID in `configs/duts/nvidia-*.yaml` and `tests/device/` |

The NVIDIA build expects CUDA 13.0 in `/usr/local/cuda-13.0` and cuDNN 9 from the pip package
`nvidia-cudnn-cu12`; override with `make -C native CUDA_HOME=... CUDNN_ROOT=...`. Tests that need the local COCO
subset, the YOLOX files or local evidence bundles skip when those are absent.

## Reproducing the GPU reference result

Large or third-party artifacts are not in the repository (`.gitignore`: `models/`, `datasets/`, `runs/`).

```bash
# 1. model (Apache-2.0, 35,858,002 bytes)
mkdir -p models/yolox-s
curl -L -o models/yolox-s/yolox_s.onnx \
    https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
sha256sum models/yolox-s/yolox_s.onnx   # c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063

# 2. cuDNN plan + float32 weights (stdlib, deterministic; SHA-256 pinned in configs/workloads/yolox-s-coco-v1*.yaml)
python3 tools/stage2b_g_real/build_yolox_plan.py models/yolox-s/yolox_s.onnx models/yolox-s/yolox_s

# 3. COCO subset: first 1,000 val2017 images by image id, decoded once to PPM (needs Pillow)
python3 tools/stage2b_g_real/build_coco_subset.py --images $COCO_ROOT/val2017 \
    --annotations $COCO_ROOT/annotations/instances_val2017.json --count 1000 --out datasets/coco-val2017-first1000

# 4. set your GPU's UUID in configs/duts/nvidia-rtx-pro-6000-gpu0-cudnn-yolox-s.yaml (device:), then
make -C native
.venv/bin/sdv-bmt run configs/experiments/br-yolox-30hz-msv1.yaml --out runs/gpu
.venv/bin/sdv-bmt evaluate runs/gpu/br-yolox-30hz-msv1/<run_id>
```

The input specs pin the subset index by SHA-256. It was built with Pillow 11.1.0 / libjpeg-turbo 3.1.0; another
JPEG decoder can produce different pixels and therefore a different digest, which the run rejects.

## Example: canonical GPU result (v0.1.0 reference evidence)

YOLOX-S on the first 1,000 COCO val2017 images, one NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, fixed rate 30 Hz, N = 1 (run `run-20261009T044411Z-003b1415f3b3`, no foreign GPU process, shared host). Latencies in µs, nearest-rank over the 1,000 completed requests:

| metric | p50 | p90 | p99 | max_observed |
|---|---:|---:|---:|---:|
| request completion | 4,876.2 | 5,390.7 | 6,529.0 | 9,423.0 |
| DUT service | 4,874.4 | 5,355.3 | 6,397.0 | 7,634.3 |
| output ready | 5,066.4 | 5,678.3 | 6,832.9 | 10,380.3 |
| device execution bracket (device clock) | 4,490.6 | 4,834.4 | 5,500.5 | 7,008.4 |

Completed 1,000 / 1,000, no drops, timeouts or errors; completion and output-ready deadline misses 0 (deadline 33.333333 ms); achieved completion rate 30.025/s at an offered 30.000/s. COCO bbox quality (conditional, coverage 1.0): AP 0.42892, AP50 0.61348, AP75 0.46807; at score ≥ 0.5 / IoU ≥ 0.5 precision 0.83621, recall 0.50288. This is a reference-accelerator result: not an NPU result, not a ranking, not a worst-case bound. Full table, setup, earlier runs, guard results and limitations: [`docs/results/v0.1.0-gpu-reference-result.md`](docs/results/v0.1.0-gpu-reference-result.md).

## Repository layout

| path | content |
|---|---|
| `src/sdv_bmt/` | Python core: contracts, input providers, workloads, DUT adapters, runner, evidence, metrics, CLI |
| `native/` | C++ native measurement plane (`src/common`, `src/fake`, `src/nvidia`) and its unit tests |
| `configs/` | experiment plans, inputs, workloads, DUT configs, stress specs, output references, small fixtures |
| `tests/` | unit, conformance, golden, architecture, e2e, native and device tests |
| `tools/` | offline asset preparation, cross-checks and analysis (outside the core's code identity) |
| `docs/` | architecture, measurement protocol, evidence, metric catalog (Korean), `docs/results/` (English) |
| `docs/design/` | Stage 0 design basis (v03) and Stage 1A implementation authorization, public editions (Korean) |

## Roadmap

v0.1.0 covers the deterministic core, the native realtime harness, the NVIDIA GPU reference backend, the real
YOLOX-S / COCO baseline and Model-level Metric Set v1. Planned next, each behind its own review gate: NPU SDK
and observability investigation (Stage 2a-N), NPU realtime harness (1R-N) and NPU baseline (2b-N); later the
Driving Scenario stage, hardware stress, and the Scenario × Stress matrix.

Not in v0.1.0: NPU integration, driving scenarios, stress sweeps, system-level (Autoware) evaluation, WCET or
FTTI claims, functional-safety compliance claims, calibration, energy per inference, DRAM bandwidth, AI error
rates per hour or km, TTC / PET / DST, SPFM / LFM / PMHF / FIT.

## Citing and license

If you use SDV-BMT in your research, please cite it with the metadata in [`CITATION.cff`](CITATION.cff):
Boryoung Kim, *SDV-BMT: Model-level Accelerator Benchmark Core*, version 0.1.0, 2026-10-09.

Copyright 2026 Boryoung Kim. Licensed under the Apache License, Version 2.0 ([`LICENSE`](LICENSE)). Third-party
components, models and data keep their own terms: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
