# YOLOX on the Mobilint MLA400: reproduction

How to rebuild and rerun the MLA400 port of the five YOLOX models (Tiny, S, M, L, X), from the ONNX models to
the baseline runs.

**No results on this page.** The Mobilint SDK license (§8) forbids publishing performance, accuracy or benchmark
results without Mobilint's written consent, so they are only in the evidence root. For the same reason this
directory contains no Mobilint software, headers or compiled models (`.mxq`).

## What the port is

| item | value |
|---|---|
| DUT | one ARIES chip of the MLA400 (`/dev/aries2` on the reference host), single core Cluster0/Core0 (`configs/duts/mobilint-mla400-aries2-single-c0c0.yaml`) |
| Models | the ONNX files of the GPU workloads (`artifact_sha256` checked), compiled to int8 by qbcompiler with the vendor default quantization, 100 calibration images |
| Graph | cut at the nine head convolutions, the last point before outputs with different value ranges are concatenated. The weight-free tail (sigmoid on objectness and classes, concat to anchors × 85) runs on the host, in the backend's `collect()`, i.e. in the output stage, outside DUT-service |
| Contract | same preprocessing, postprocessing, codec and raw output (anchors × 85 float32) as the GPU workloads |
| Plans | per model: `capture` (asap), `capture-identity` (second capture against the NPU golden), `baseline` (30 Hz, deadline = frame period) |

## Requirements

| component | version used | note |
|---|---|---|
| MLA400 card, Aries driver, firmware | driver 1.14.0, firmware 1.2.7 | `mobilint-cli status` |
| qbruntime C++ | 1.5.0 | headers and library from the Mobilint portal (`/usr/include/qbruntime`, `/usr/lib/x86_64-linux-gnu`) |
| qbcompiler | 1.4.1 | Python environment, here `qbc14-env`; it also needs `onnx` and `onnxruntime` for `cut_head.py` |
| qbruntime Python | 1.5.3 | Python environment, here `zoo-env`, with `onnxruntime`, `numpy` and `PyYAML` (L1), plus `opencv-python` for the L0 diagnostic |
| repo `.venv` | as for the GPU reference | `sdv-bmt` itself |

The Mobilint software comes from dl.mobilint.com and needs an account; its license applies. On a host that also
has an older qbruntime C++ in `/usr/local`, `build_native.sh` still links 1.5.0 (see the script header).

Inputs are prepared as for the GPU reference (repository README):
- the five YOLOX ONNX files;
- the COCO val2017 set `datasets/coco-val2017` (all 5,000 images), the same index for all five models; Tiny's
  input config only declares the 416 contract. `datasets/coco-val2017-first1000` (its first 1000 index rows) is only
  needed to compare a new NPU golden with the 1000-image golden and to reproduce v1.0;
- COCO train2017 annotations and images, for calibration.

## Steps

`$A` = a working directory outside the repository (here `~/sdv_assets`).
`$ROOT` = a new evidence root, absolute path (here `~/sdv_evidence/mla400-yolox-single`).
Run every command from the repository root.

| # | step | environment | command | time |
|---|---|---|---|---|
| 1 | Calibration sets (416 for Tiny, 640 for the others) | repo `.venv` | `python3 tools/mobilint/make_calib.py --workload configs/workloads/yolox-tiny-coco-v1.yaml --annotations $COCO/annotations/instances_train2017.json --exclude-index datasets/coco-val2017/index.jsonl --count 100 --seed 0 --cache $COCO/train2017-calib --out $A/calib/yolox-416`, then the same with `yolox-s-coco-v1.yaml` and `--out $A/calib/yolox-640` | minutes |
| 2 | Cut the five graphs | `qbc14-env` | for each model: `python3 tools/mobilint/cut_head.py --onnx $A/yolox_onnx/yolox_<m>.onnx --out $A/compile/head9/yolox_<m>_head9.onnx --calib-dir $A/calib/yolox-<416\|640>/npy` | seconds each |
| 3 | Compile | `qbc14-env` | `tools/mobilint/compile_head9.sh $A/compile/head9 $A/calib` | tens of minutes |
| 4 | L1: NPU vs ONNX Runtime FP32, first 1000 images | `zoo-env` | for each model: `python3 tools/mobilint/validate_numerics.py --workload configs/workloads/yolox-<m>-coco-v1.yaml --index datasets/coco-val2017/index.jsonl --onnx $A/yolox_onnx/yolox_<m>.onnx --mxq $A/compile/head9/yolox_<m>_head9.mxq --dev 2 --count 1000 --out $ROOT/numerics/yolox-<m>/numerics.json` | minutes each |
| 5 | Build the native runner | repo `.venv` | `tools/mobilint/build_native.sh` | seconds |
| 6 | Record the environment | repo `.venv` | `python3 tools/stage2b_g_real/logcmd.py $ROOT --label environment -- bash tools/mobilint/capture_environment_mla400.sh $ROOT 2` | seconds |
| 7 | Validate the plans | repo `.venv` | `python3 tools/mobilint/sdv_bmt_mla400.py validate configs/experiments/<plan>.yaml` for the 15 `*-mla400-*` plans | seconds |
| 8 | Capture, golden check, identity, 25 baselines | repo `.venv` | `bash tools/mobilint/run_all_mla400.sh $ROOT` | hours |
| 9 | Summary | repo `.venv` | `python3 tools/stage2b_g_real/logcmd.py $ROOT --label summary -- python3 tools/mobilint/summarize_mla400.py $ROOT $ROOT/analysis/summary.json` | seconds |

Notes on the steps:
- **Steps 4 and 8 use the NPU.** Do not run them while another process holds the device; `run_series_mla400.sh`
  checks this and stops.
- **Step 1:** the calibration images come from train2017 only, so they can never be val2017 images; the excluded
  index does not change the selection (with the 1000-image index the same 100 images are chosen).
- **The workloads pin `/dev/aries2`.** On another host, pick the first chip of the MLA400 and its NUMA node:
  `udevadm info --query=path --name=/dev/ariesN` and `/sys/.../numa_node`. Then change the DUT config and
  `realtime.cpu_affinity` in the plans.

## Checks that must hold

- **Step 2:** each `.cut.json` records the SHA-256 of the cut ONNX. It must equal `cut_onnx_sha256` in the workload.
  The check against the full graph must report a maximum difference at float rounding level.
- **Step 3: the compiled-model SHA-256 changes on every compile.** qbcompiler 1.4.1 writes the compile time and a
  build id into the `.mxq` header. A recompile gives the same configuration and bit-identical NPU outputs (checked
  with `digest_probe` digests on fixed inputs). After a recompile, set `mxq_sha256` in the workload to the new file
  and rerun step 4.
- **Step 4:** the L1 AP of the NPU on the first 1000 images is the reference value for step 8.
- **Step 8:**
  - every capture completes all 5,000 requests;
  - the re-extracted goldens are identical to
    `configs/references/<input>.yolox-<m>-coco-v1-mla400-q8.npu-capture.golden.jsonl`; the script prints
    `identical` for each model;
  - the first 1000 rows of each golden are identical to the 1000-image golden
    (`coco-val2017-first1000*-mla400-q8.npu-capture.golden.jsonl`), whose capture AP equals the L1 NPU AP exactly
    (computed by `sdv-bmt evaluate`, or by `tools/stage2b_g_real/coco_quality.py` for S). So the L1 check holds for
    those 1000 images; the capture AP on all 5,000 images has an L1 counterpart only if step 4 runs with
    `--count 5000`;
  - every identity rate is 1.0.
- **A new input set:** on the first capture of an input without a committed NPU golden, `run_all_mla400.sh` writes
  the golden to `configs/references/` and prints the first-1000-row comparison instead of `identical`. Then
  `python3 tools/mobilint/set_mla400_plans.py --input <input>` points the identity and baseline plans at the new
  goldens (`--check` must then exit 0 with no output).

## Diagnostics (not part of the measured flow)

Two diagnostics were run outside this directory. Their sources and outputs are in the evidence root:
- **`digest_probe.cpp`:** the completion-signal and input-read checks of the measurement boundary table (C++,
  linked like the native runner).
- **`l0_one_image.py`:** L0, one image through the vendor runtime only, with the boxes drawn.

## Tools in this directory

| file | purpose |
|---|---|
| `make_calib.py` | calibration tensors (canonical preprocessing, HWC float32, manifest with SHA-256s) |
| `cut_head.py` | cut at the nine head convolutions, ONNX Runtime check, cut record; `host_tail()` |
| `compile_head9.sh` | qbcompiler invocation (options of the port) |
| `validate_numerics.py` | L1 |
| `build_native.sh` | native runner against qbruntime 1.5.0 (runs `make -C native mobilint`) |
| `sdv_bmt_mla400.py` | `sdv-bmt` from the repository sources, used by the scripts (the shared CLI registers the adapter; `sdv-bmt` works the same) |
| `run_series_mla400.sh` | runs plans with host and NPU state logged (`COMMANDS.log`) |
| `run_all_mla400.sh` | step 8 in one command |
| `set_mla400_plans.py` | the input of the 15 plans and the NPU goldens of the identity and baseline plans |
| `capture_environment_mla400.sh` | `environment/` of the evidence root |
| `summarize_mla400.py` | per-run and per-experiment table from the evaluations |
