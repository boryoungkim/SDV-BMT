# BOS Eagle-N port of the DeepLabV3+ workloads: implementation notes

For those working on this repository. Read this before changing anything under `native/src/eagle_n/`,
`src/sdv_bmt/dut/eagle_n/`, `tools/eagle_n/`, `configs/**/*eagle-n*` or `tests/*/test_eagle_n_*`.

This page contains **no measurement results**: no latency, rate, margin or accuracy values. Results live only in the
evidence root on the reference host.

## 1. What the port is

| item | value |
|---|---|
| DUT | One BOS Eagle-N board, device 0 (`/dev/bos/0`). Config: `configs/duts/eagle-n-<model>[-bf16].yaml` |
| Adapter | `bos_eagle_n`, class `EagleNDUT` (`src/sdv_bmt/dut/eagle_n/adapter.py`), a subclass of `NativeRealtimeAdapter` |
| Native runner | `sdv-bmt-native-eagle-n` = `native/src/common/{loop,runner}.hpp` + `native/src/eagle_n/backend.hpp` + `native/src/main_eagle_n.cpp`; it needs no SDK and loads the executor at run time |
| Executor | `libeagle_n_ttnn.so`, built from `native/src/eagle_n/ttnn_executor.cpp` against the BOS SDK (TT-Metal / TTNN) inside the pinned BOS container; loaded through a six-function C ABI (`executor_abi.hpp`). One executor serves the three models |
| Models | DeepLabV3+ R-18, R-50 and R-101: the same ONNX files as the GPU workloads (`artifact_sha256` unchanged) |
| Precision | BF16 activations and weights, FP32 accumulation (`bf16-conv-fp32dst`); the only precision accepted |
| Graph | A manual TTNN C++ graph built from a plan exported offline (`tools/eagle_n/export_plan.py`). R-50 and R-101 use the tuned graph with layer4 and the head split into blocks whose shapes already ran on this NPU (`decompose_proven.py`) |
| Workloads | `configs/workloads/deeplabv3plus-<model>-cityscapes-v1-eagle-n[-bf16].yaml` |
| Plans | Per model: `…-capture.yaml` (asap, produces the golden) and `…-baseline.yaml` (GPU family rate, deadline = frame period). Both are the GPU plans with this workload and DUT, the host CPU affinity and one request timeout and drain for every Eagle-N plan |
| Goldens | `configs/references/cityscapes-val500.deeplabv3plus-<model>-cityscapes-v1.eagle-n[-bf16]-capture.golden.jsonl` |
| Generator | `tools/eagle_n/configure.py` writes every Eagle-N config; do not edit them by hand |
| Reproduction | `tools/eagle_n/README.md` |

The contract of the GPU workloads is unchanged: input `cityscapes-rgb-norm-chw2048x1024-f32`, the uint8 label map
as raw output, the shared codec and the Cityscapes mIoU evaluator. Only the device part differs.

## 2. Data flow of one request

```
prepare (once)  plan + weights loaded and checked; constants converted to the plan precision on the device;
                one warm graph run; the graph is captured as a trace
stage (per item, outside timing)  canonical CHW FP32 -> NHWC BF16, kept in staged host memory
submit          H2D of the staged input, trace replay, blocking D2H of the uint32 labels   (synchronous)
poll            returns complete once submit has returned
collect         every label checked < 19, narrowed to uint8 in a preallocated buffer       (output stage)
```

- **DUT-service** = `t_complete − t_submit_begin`: H2D, trace replay (including the final resize and ArgMax on the
  device) and D2H. Submit is synchronous, so host arrivals and timeouts are serviced only after submit returns; the
  adapter declares this in `submit_semantics` and `complete_semantics`.
- **Not observable** (declared, never estimated): device-side timing, device frequency, thermal state, memory
  bandwidth, competing users and queue state. Every realization condition is reported as `not_observable`.

## 3. Qualification

A benchmark run needs a qualification receipt (`native_qualification.json`) for the plan and the executor:

1. **L0**: an L1-memory preflight under `NO_DISPATCH` (no program reaches the cores), a proven-shape survey of every
   SDK call (`proven_survey.py`), and one golden image replayed several times with identical labels.
2. **L1**: the 500 validation images on the NPU, merged (`l1_merge.py`), and the ONNX Runtime FP32 reference on the
   CPU on byte-identical inputs (`l1_ort.py`).
3. **Accuracy criterion**: a BF16 plan is accepted when its L1 NPU mIoU is at most 0.01 below the GPU FP32 mIoU of
   the same model. `qualify_native.py` enforces it, together with a sanity bound against ONNX Runtime.
4. **Receipt** (`qualify_native.py`): binds the executor SHA-256, the executor source digest, the plan, weights and
   index SHA-256 to that L1 evidence. `configure.py` then pins the receipt's SHA-256 in the workload
   (`qualification_sha256`).

**Executor tie.** `build_native.sh` writes `<executor>.build.json` with the executor SHA-256 and the digest of its
three source files (`ttnn_executor.cpp`, `executor_abi.hpp`, `common/json.hpp`; `src/sdv_bmt/dut/eagle_n/
executor_source.py`). The DUT config pins both values. The adapter refuses a run unless the executor file, the build
record, the DUT pins, the receipt and the sources of the current checkout agree. A checkout can therefore only run
the executor built from its own source, and any executor change requires L1 again.

The adapter also refuses a benchmark run when a diagnostic switch of the executor is set in the environment
(`EAGLE_N_PREFLIGHT_OUT`, `EAGLE_N_SYNC_EACH_OP`).

## 4. Problems met and how they were solved

| # | problem | resolution | where |
|---|---|---|---|
| P1 | The R-18 tuned conv settings applied to R-50 exceeded the L1 memory of a core | The exporter writes a launch decision per conv into the plan: the R-18 tuned setting only where R-18 already ran an equal or larger conv of the same kind, otherwise the default setting; the executor refuses a tuned non-R-18 plan without these decisions | `export_plan.py`, `ttnn_executor.cpp` |
| P2 | A conv shape that had never run on the board hung the device, although the L1 preflight was clean | Policy: only call shapes that already completed. `proven_survey.py` checks every SDK call of a plan against completed runs; `decompose_proven.py` rewrites layer4 and the head of R-50 and R-101 into blocks of proven shapes, with a symbolic equivalence check | `proven_survey.py`, `decompose_proven.py` |
| P3 | Submit is synchronous (H2D, replay and D2H return together) | Declared, not hidden: completion is observed by the poll after submit returns, and the semantics name the consequence for arrivals and timeouts | `adapter.py`, `backend.hpp` |
| P4 | 500 staged BF16 inputs need several GiB of host memory | Explicit host memory budgets in the DUT config; the backend and the launcher refuse a run that would exceed them | `configure.py`, `backend.hpp`, `in_container.py` |
| P5 | Killing a process that holds the NPU can leave the cores wedged until the host is rebooted | The launcher never kills a container that holds the NPU; it reports the hang and keeps its private mounts | `in_container.py` |
| P6 | A rebuilt executor is a different binary | Build record, two DUT pins and the receipt bind the executor to its source and to its L1 evidence (section 3) | `build_native.sh`, `executor_source.py` |
| P7 | The SDK runs only inside the BOS container, whose `/dev/shm` is too small for the staging file | Every device step runs through `in_container.py`, which binds a private disk directory as `/dev/shm` for `run` | `in_container.py` |

## 5. Rules to keep

- Do not add results to this repository: no times, rates, margins or accuracy values in code, comments, configs,
  commit messages or docs, and no text from which they can be inferred.
- Never kill a process or container that holds the NPU. Report a hang.
- After any change to the executor sources: rebuild the executor, rerun L1, issue a new receipt and rerun
  `configure.py`. Rebuilding the runner (`make -C native eagle-n`) needs no requalification and is required whenever
  `native/` changes.
- A new plan or plan change reaches the NPU only after the preflight and the proven-shape survey pass.
- Keep the configs generated by `configure.py`; `tests/unit/test_eagle_n_adapter.py` checks that they match.

## 6. Open items

- The executor still contains the unreachable `bfp8w` and `fp32_conv` branches; plans with those precisions are
  refused.
- The launcher uses `sudo -n` for Docker, the private hugepage mount and the NPU user check.
