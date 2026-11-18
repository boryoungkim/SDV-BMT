# Mobilint MLA400 port of the YOLOX workloads: implementation notes

For those working on this repository. Read this before changing anything under
`native/src/mobilint/`, `src/sdv_bmt/dut/mobilint/`, `tools/mobilint/`, `configs/**/*mla400*` or
`tests/device/test_mobilint_*`.

This page contains **no measurement results**. The Mobilint SDK license (§8) forbids publishing performance,
accuracy or benchmark results without Mobilint's written consent. Do not add results to this repository: no
latency, AP or throughput values in code, comments, configs, commit messages or docs. Results live only in the
evidence root on the reference host.

## 1. What the port is

| item | value |
|---|---|
| DUT | One ARIES chip of a Mobilint MLA400 card, device node `/dev/ariesN`; on the reference host `/dev/aries2`, the first of the card's four chips. Single-core mode on one pinned (cluster, core), Cluster0/Core0. Config: `configs/duts/mobilint-mla400-aries2-single-c0c0.yaml` |
| Adapter | `mobilint_aries`, class `MobilintAriesDUT` (`src/sdv_bmt/dut/mobilint/adapter.py`), a subclass of `NativeRealtimeAdapter` (`src/sdv_bmt/dut/native.py`) |
| Native runner | `sdv-bmt-native-mobilint` = `native/src/common/{loop,runner}.hpp` + `native/src/mobilint/backend.hpp` + `native/src/main_mobilint.cpp`, linked against qbruntime C++ 1.5.0 |
| Models | YOLOX-Tiny/S/M/L/X: the same ONNX files as the GPU workloads (`artifact_sha256` unchanged), compiled offline by qbcompiler 1.4.1 to int8 `.mxq` files for `aries-rb` |
| Workloads | `configs/workloads/yolox-{tiny,s,m,l,x}-coco-v1-mla400-q8.yaml`, plus `yolox-s-…-q8-quality.yaml` |
| Plans | Per model, in `configs/experiments/`: `…-mla400-capture.yaml` (asap, produces the golden), `…-mla400-capture-identity.yaml` (second capture against the golden), `…-mla400-baseline.yaml` (fixed rate, deadline = frame period) |
| Goldens | `configs/references/*.yolox-*-coco-v1-mla400-q8.npu-capture.golden.jsonl`, in the same format as the GPU goldens (one SHA-256 of the encoded output per item) |
| Reproduction | `tools/mobilint/README.md`, step by step from the ONNX files to the baseline runs |

The contract of the GPU workloads is kept unchanged: input `yolox-letterbox-bgr-chw{S}-f32/1`, raw output
`anchors × 85` float32 (anchor-major, `[x, y, log w, log h, obj, 80 classes]`, obj/classes after sigmoid, strides
8 → 16 → 32), the shared decode/NMS/codec (`native/src/common/yolox_output.hpp`) and COCO bbox AP. Only the
device part differs.

## 2. Data flow of one request

```
Python (adapter)                         native runner (backend.hpp)                       qbruntime / NPU
prepare: backend_spec ─────────────────► prepare: sha256(.mxq) == spec?  Accelerator(dev),
         (mxq path + sha256,                       ModelConfig single core (cluster, core),
          cut sha256, options)                     Model(.mxq), launch; map the 9 outputs by shape
stage (per item, outside timing) ──────► stage: CHW float32 → HWC float32, kept per slot
                                         ─ MEASURE ─────────────────────────────────────
                                         submit:  Model::inferAsync(hwc)   ──────────────► quantize, H2D, run, D2H,
                                         poll:    Future::waitFor(0)       ◄────────────── dequantize (inside the runtime)
                                         ── t_complete (DUT-service ends here) ──
                                         collect: Future::get() → 9 float32 tensors
                                                  host tail: sigmoid(obj, cls), reassemble anchors × 85
                                                  → shared YOLOX decode + NMS + codec
                                         ── t_output_ready (output stage ends here) ──
```

- **DUT-service** = `t_complete − t_submit_begin`. It contains the runtime's input quantization and host→device
  copy, NPU execution, the device→host copy and dequantization, and the completion poll. qbruntime does all of
  these inside the request, and they cannot be separated. That dequantization is inside DUT-service was checked on
  the device: requests with int8 outputs and with float32 outputs differ in the time to completion, not in the
  duration of `Future::get`.
- **Output stage**: `Future::get` (it only hands over the float32 outputs), the host tail and the shared
  decode/NMS/encode. The durations of `get` and of the tail are recorded per request as `mobilint.future_get_ns`
  and `mobilint.host_tail_ns`.
- **Not observable** (declared, never estimated):
  - device-side timing (the `bmt.h2d.*`, `bmt.device_execution_bracket.*` and `bmt.d2h.*` metrics are
    `not_supported`);
  - data movement, NPU frequency, thermal state and memory bandwidth during a run;
  - the split of DUT-service into its steps.

## 3. Metric support

Status of each metric of the Model-level Metric Set in the MLA400 baseline runs (fixed rate, deadline = frame
period). This section gives status only, no values. It was checked with catalog v1.1, the default (93 metrics),
and catalog v1.2 (101 metrics: v1.1 plus 8 `quality.det3d.*`).

| status | v1.1 | v1.2 | meaning |
|---|---:|---:|---|
| valid | 76 | 76 | measured and reported |
| not_supported | 12 | 12 | the device runtime gives no such observation |
| not_applicable | 5 | 13 | not meaningful for this plan or task, whatever the device (a GPU run with the same plan settings has the same set) |

As on every DUT, percentile metrics over planned requests (`bmt.request_completion.planned.*`) become `censored`
when dropped requests reach their rank, as the catalog defines.

**Valid (76)**

| group | metrics |
|---|---|
| request counts (7) | `bmt.count.{planned,issued,not_issued,completed,dropped,timed_out,error}` |
| rates (8) | `bmt.offered_rate`, `bmt.achieved_completion_rate`, `bmt.completion_rate`, `bmt.drop_rate`, `bmt.timeout_rate`, `bmt.error_rate`, `bmt.not_issued_rate`, `bmt.censored_fraction` |
| deadline (5) | `bmt.deadline.{completion,output_ready}_miss_{count,rate}`, `bmt.late_completion_count` |
| latency distributions (31) | `bmt.{dut_service,submit,post_submit,issue_lag,output_stage,output_ready,request_completion}.{p50,p90,p99,max_observed}`, `bmt.request_completion.planned.{p50,p90,p99}` |
| output (1) | `bmt.output.identity_rate` (against the NPU golden) |
| data (6) | `data.item_count`, `data.annotation_coverage`, `data.traceability.complete`, `data.integrity.{payload,prepared_input,annotation}_verified_rate` |
| 2D detection quality (18) | `quality.detection.coverage`, `quality.detection.conditional.{ap,ap50,ap75,ap_small,ap_medium,ap_large,ar1,ar10,ar100,precision,recall,f1,tp_count,fp_count,fn_count,fp_per_image,fn_per_gt}` |

**Not supported (12)**: `bmt.h2d.*`, `bmt.device_execution_bracket.*` and `bmt.d2h.*` (each `p50`, `p90`, `p99`,
`max_observed`). qbruntime 1.5.0 performs the transfers and the execution inside the inference request and exposes
no per-inference device timing. These steps are still inside DUT-service, which is measured. This is the only
difference from the GPU reference, whose NVIDIA backend reports device-side timing.

**Not applicable (5 in v1.1, 13 in v1.2)**

| metrics | why |
|---|---|
| `bmt.closed_loop_completion_rate` | fixed-rate replay, not closed loop |
| `bmt.required_rate`, `bmt.required_rate_met` | the plans declare no required rate |
| `quality.segmentation.{coverage,conditional.miou}` | detection workload |
| `quality.det3d.{coverage,conditional.map,nds,mate,mase,maoe,mave,maae}` (v1.2) | 3D detection (CenterPoint) only |

Capture runs (asap) have more `not_applicable` rate and deadline metrics, as on the GPU. In the bundles, the
backend also records per-request output-stage observations outside the catalog: `mobilint.future_get_ns` and
`mobilint.host_tail_ns`.

## 4. Problems met and how they were solved

| # | problem | resolution | where |
|---|---|---|---|
| P1 | Where to end the compiled graph. The YOLOX head tail (sigmoid on objectness and classes, concat to 85 channels, reshape, concat over strides) puts unbounded box offsets and probabilities in 0..1 into one tensor. One shared 8-bit scale for such a tensor is a known accuracy risk | The compiled graph ends at the **nine head convolutions** (per stride: box 4, objectness 1, classes 80). This is the last point before outputs with different value ranges are concatenated, and the same cut the Mobilint YOLO tutorial uses. Each output keeps its own scale. The weight-free tail runs on the host in FP32 in `collect()`, in the output stage. The cut graph plus the host tail equals the full ONNX graph in ONNX Runtime up to float rounding. All weights stay on the NPU. Do not move the tail back onto the NPU without an L1 check on the full evaluation set | `tools/mobilint/cut_head.py`, `backend.hpp` collect() |
| P2 | qbruntime returns the nine outputs in its own order and layout (HWC), not in ONNX output order | Outputs are matched by (grid size, channels) at prepare, never by position. prepare fails if a shape matches no head tensor | `backend.hpp` prepare() |
| P3 | `inferAsync` returns before the runtime reads the input buffer | Each slot's HWC buffer stays valid and unchanged until its request completes. Input conversion and transfer are therefore inside DUT-service, which is correct by definition | boundary probe; `backend.hpp` stage()/submit() |
| P4 | Is the output final when `waitFor(0)` first returns true? | Yes: checked on the device by comparing digests of `get()` at the first true poll and later, with interleaved inputs. Completion = first `waitFor(0) == true` | boundary probe |
| P5 | qbruntime prints lifecycle lines ("Model constructed/launched/disposed") to stdout, which carries the runner's JSON-lines protocol | During prepare and cleanup, stdout is redirected to stderr with `dup2`. The calls on the measured path (`inferAsync`, `waitFor`, `get`) print nothing | `backend.hpp` (stdout guard) |
| P6 | The reference host has two qbruntime C++ installs: 1.5.0 in `/usr` and 1.1.0 in `/usr/local`, which the compiler and the loader search first | `tools/mobilint/build_native.sh` builds against 1.5.0 through a private symlink directory (`-isystem`, `RUNPATH`). The shared system is not modified. `describe` reports the library actually loaded, and a device test asserts it | `build_native.sh`, `tests/device/test_mobilint_native.py` |
| P7 | Integration with the shared code | With the agreement of the common-code owner: the adapter is registered in the shared CLI (`DUT_ADAPTERS` in `src/sdv_bmt/cli/main.py`), and `native/Makefile` has an optional target `mobilint`, outside `all` and `portable`, so hosts without qbruntime still build. Adding `native/src/mobilint/` changes the native source-tree digest: rebuild the other runners (`make -C native`) on their hosts after merging | `src/sdv_bmt/cli/main.py`, `native/Makefile` |
| P8 | The NPU sits on one NUMA node of a two-socket host; the GPU plans pin the harness to the other node | The MLA400 plans pin `realtime.cpu_affinity` to the CPUs of the NPU's NUMA node, found from sysfs/`udevadm` | `configs/experiments/*mla400*` |
| P9 | The compiled `.mxq` differs on every compile (the header holds a compile time and a build id) | The workload pins the `.mxq` SHA-256, and prepare refuses any other file. A recompile gives bit-identical outputs, so the goldens stay valid, but the workload's `mxq_sha256` must be updated | workloads, `backend.hpp` prepare() |
| P10 | Replay rate of the baseline plans | The baseline plans use the fixed rate of the GPU family plans, 30 Hz, for all five models, with the deadline equal to the frame period. Dropped requests and deadline misses are part of the reported metrics. A whole-chip configuration (Global8) would be a separate DUT config | baseline plans |

## 5. Rules to keep

- **Contract:** `collect()` must deliver the GPU workloads' raw output byte for byte in layout (anchors × 85
  float32). Do not change the shared decode/NMS/codec or the input preprocessing.
- **Measured path:** no allocation, logging or Python in submit/poll. The host tail and HWC conversion stay out of
  DUT-service: HWC at staging, tail in collect.
- **Vendor code:** Mobilint headers and concepts appear only in `native/src/mobilint/` and `main_mobilint.cpp`.
  Python keeps Mobilint names in `src/sdv_bmt/dut/mobilint/`, and evidence keys are namespaced `mobilint.`. Never
  commit Mobilint software, headers, libraries or `.mxq` files.
- **Identity:** every input is pinned by SHA-256: ONNX artifact, cut ONNX, `.mxq`, dataset index and goldens. When
  one changes, update the pin and rerun capture → golden → identity.
- **Not observable means not reported:** do not derive device timing, data movement or clock values from
  DUT-service.
- **Single core only:** the adapter refuses other core modes until a Global8 DUT is added as a separate DUT
  config.
- **Shared device:** do not start a run while another process holds `/dev/ariesN` open
  (`tools/mobilint/run_series_mla400.sh` checks this).
- **No results in the repository** (see the top of this page).

## 6. Where to start

| task | start at |
|---|---|
| Rebuild the runner | `make -C native mobilint` (or `tools/mobilint/build_native.sh`), on the NPU host |
| Recompile a model | `tools/mobilint/cut_head.py` → `make_calib.py` → `compile_head9.sh` → update `mxq_sha256` in the workload |
| Check numerics after a recompile | `tools/mobilint/validate_numerics.py` (L1, NPU vs ONNX Runtime FP32) |
| Rerun everything | `tools/mobilint/run_all_mla400.sh` (capture, golden comparison, identity, baselines) |
| Device tests | `.venv/bin/python -m pytest tests/device/test_mobilint_native.py` (skipped, reported NOT RUN, without the NPU) |
| Change backend behaviour | `native/src/mobilint/backend.hpp`. Rebuild, rerun the device tests and an identity run |

## 7. Open items

- Shared-code request: add `qbruntime` to the vendor-header boundary test.
- `describe()` reads the sysfs path `/sys/class/ariesN`; the correct path is `/sys/class/aries/ariesN`. Only
  descriptive fields are affected. It is to be fixed together with the Global8 DUT.
- Global8 (whole chip, 8 cores) as a second DUT, if scheduled.
