# DeepLabV3+ on the Eagle-N NPU: reproduction

How to rebuild and rerun the Eagle-N port of DeepLabV3+ (R-18, R-50, R-101) through SDV-BMT, from the ONNX models to
the baseline runs. **No measurement results on this page**; they are in the evidence root.

Read the watchdog section before running any plan on the NPU.

## What the port is

| item | value |
|---|---|
| DUT | one Eagle-N A0 board, device 0 (`/dev/bos/0`), 5 × 4 Tensix grid, adapter `bos_eagle_n` (`src/sdv_bmt/dut/eagle_n/`) |
| Models | the ONNX files of the GPU DeepLabV3+ workloads (`artifact_sha256` checked, H04) |
| Precision | BF16 activations and weights, HiFi4, FP32 accumulation (`bf16-conv-fp32dst`). R-50 also has an 8-bit-weight plan (`bfp8w`), exported but **not qualified** |
| Graph | a manual TTNN C++ graph built from a plan exported offline (`tools/eagle_n/export_plan.py`). R-50 and R-101 use the tuned graph with layer4 and the head split into blocks whose shapes already ran on this NPU (`decompose_proven.py`) |
| Execution | the executor (`libeagle_n_ttnn*.so`, loaded by the native runner through a 6-function C ABI) captures a trace once at prepare and replays it per request: H2D input, trace replay, D2H labels. The final resize and argmax run on the device; the host `collect()` checks every label < 19 and casts to uint8 |
| Contract | same preprocessing, postprocessing, codec and raw output as the GPU workloads |
| Plans | per model: `capture-asap-t60` (L3, L4), `baseline` (2 Hz, guide) and `baseline-provisional` (lower rate, L5) |

| Model | DUT config | Workload | Capture plan | Baseline plan used for L5 |
|---|---|---|---|---|
| R-18 | `eagle-n-r18.yaml` | `deeplabv3plus-r18-cityscapes-v1-eagle-n.yaml` | `deeplabv3plus-r18-eagle-n-capture-asap-t60.yaml` | `deeplabv3plus-r18-eagle-n-baseline-provisional-0p8hz.yaml` |
| R-50 BF16 | `eagle-n-r50-bf16.yaml` | `deeplabv3plus-r50-cityscapes-v1-eagle-n-bf16.yaml` | `deeplabv3plus-r50-eagle-n-bf16-capture-asap-t60.yaml` | `deeplabv3plus-r50-eagle-n-bf16-baseline-provisional-0p25hz.yaml` |
| R-101 BF16 | `eagle-n-r101-bf16.yaml` | `deeplabv3plus-r101-cityscapes-v1-eagle-n-bf16.yaml` | `deeplabv3plus-r101-eagle-n-bf16-capture-asap-t60.yaml` | `deeplabv3plus-r101-eagle-n-bf16-baseline-provisional-0p2hz.yaml` |

The NPU goldens are in `configs/references/cityscapes-val500.deeplabv3plus-<model>-cityscapes-v1.eagle-n[-bf16]-capture.golden.jsonl`.

## ⚠ Read first: the `measure` watchdog of SDV-BMT

### What it does

`NativeDUT.run_window()` (`src/sdv_bmt/dut/native.py`, lines 306–311) gives the whole measure window one time limit
and **kills the native runner when no reply arrives in time**:

```
limit = span + n × request_timeout × (1 if replay is asap else 0) + drain + precondition × request_timeout + 120 s
```

`span` is the last schedule offset plus the lead time. `n` is the number of entries in `schedule.offsets_ns`.
For `replay: asap` the runner writes no offsets (`src/sdv_bmt/runner/runner.py`, line 915), so **n = 1 whatever the
number of items**.

With the GPU-copy capture plan (asap, request timeout 1 s, drain 1 s, 32 precondition requests) the limit is
**154 s for the whole window of 532 requests**, about 0.29 s per request. A GPU needs milliseconds per image; every
DeepLabV3+ variant on Eagle-N needs seconds.

### Why it matters on an NPU

When the limit passes, SDV-BMT kills the runner **while it holds the NPU**. This happened once, on R-18 on
2026-11-15: `no 'measure' reply within 154 s; native runner killed`. Killing a process while its kernels are
resident on the device can leave the cores wedged; recovery then needs a host reboot.

### What the port does about it: config only, `asap` kept

The shared code stays as it is. The Eagle-N capture plans keep `replay: asap`, as the guide asks, and raise only
`run.request_timeout_ns` and `run.drain_ns` from 1 s to **60 s** (`*-capture-asap-t60.yaml`). For `asap` each second
of request timeout adds 34 s to the limit (33 × T + drain):

```
T = 60 s:  0.002 + 60 + 60 + 32 × 60 + 120 = 2160 s  →  2160 / (32 + 500) = 4.06 s available per request
```

The maximum time per request of each variant, from its L1 evidence, must stay below this budget with a margin;
`run_eagle_n.sh` prints the budget before every run.

Checked on the NPU for R-50 BF16 on 2026-11-18 (`run_eagle_n.sh`-equivalent L3 with
`deeplabv3plus-r50-eagle-n-bf16-capture-asap-t60.yaml`): 500/500 completed in `asap`, no watchdog stop, no timeout,
mIoU equal to L1, and the extracted golden identical to the golden of the earlier `fixed_rate` captures. R-18 and
R-101 with their `asap-t60` plans are still to be run.

Cost of this choice: a request that hangs is declared timed out only after 60 s instead of 1 s. The 1 s of the GPU
copy cannot be kept on Eagle-N in any case, because one R-50 or R-101 request takes longer than 1 s. The value 60 s
is therefore a deviation from the GPU-copy plan and is reported as such.

Earlier evidence (L3–L5 of 2026-11-15 to 2026-11-18) was captured with provisional `fixed_rate` plans
(deviations D1, D3, D5, `*-capture-provisional-*.yaml`). They also avoid the watchdog without shared-code changes,
but they change the replay mode; the `asap` plans above replace them for main.

### Rules

1. Never run the GPU-copy capture plans (`*-eagle-n[-bf16]-capture.yaml`, asap with 1 s) on the NPU: the watchdog
   would kill the runner after 154 s. `run_eagle_n.sh` refuses any capture plan below 10 s request timeout.
2. For a slower model or a different item count, size T before running:
   `T ≥ ((32 + items) × max run time − 120) / 34` with drain = T, plus a margin of at least 10%.
3. Keep the request timeout above the maximum run time of one request.
4. Never stop a hung run from outside (`kill`, `docker stop`). The Eagle-N launcher never kills a run container;
   report the hang instead.
5. The launcher wall time (`WALL_SECONDS`, 6000 s by default) must exceed start + prepare + stage + measure limit +
   cleanup = 60 + 1800 + 600 + 2160 + 60 = 4680 s.

### Which numbers can be changed in config, and which need a shared-file change

> **SHARED FILE = maintainer decision.** Values marked "shared code" live in the core of SDV-BMT and apply to every
> DUT (GPU, Mobilint, Eagle-N). Do not edit them inside an Eagle-N change; ask the maintainer.

| Value | Current value | Where it is set | How to change it |
|---|---|---|---|
| `span` (schedule length) | from the replay mode, the rate and the item count | experiment plan: `replay.mode`, `replay.rate_hz`; input config: `selection.count` | **config** (a different rate or mode is a deviation from the guide) |
| `request_timeout` | 1 s in the GPU-copy plans; **60 s in the Eagle-N `asap` capture plans** | experiment plan: `run.request_timeout_ns` | **config** (the knob this port uses; deviation from the GPU-copy plan) |
| `drain` | same as the request timeout | experiment plan: `run.drain_ns` | **config** |
| precondition requests | 32 | experiment plan: `realtime.precondition_requests` | **config** |
| `n` for `asap` (always 1) | 1 | `src/sdv_bmt/runner/runner.py` line 915 (no offsets for asap) and `src/sdv_bmt/dut/native.py` line 309 | **⚠ SHARED CODE** |
| factor "asap only" on `n × request_timeout` | 1 for asap, 0 otherwise | `src/sdv_bmt/dut/native.py` line 310 | **⚠ SHARED CODE** |
| fixed margin | 120 s | `src/sdv_bmt/dut/native.py` line 311 | **⚠ SHARED CODE** |

The measure limit is computed inside `run_window()` itself. A plugin cannot change it without overriding
`run_window()`, which would copy the shared measurement code; the port does not do that.

**Possible tool fix (maintainer decision, not needed for this port):** for `asap`, use `n = number of selected
items`. The Eagle-N capture plans could then go back to a short request timeout.

### Other time limits of a run

The per-phase limits are class attributes of `NativeDUT`. A plugin may override them in its own adapter without
touching the shared file; Eagle-N does this for `prepare` only.

| Phase | Limit | Where it is set | How to change it |
|---|---|---|---|
| runner start | 60 s | `src/sdv_bmt/dut/native.py` line 183 (`START_TIMEOUT_S`) | shared default; a plugin can override the attribute |
| `describe` | 60 s | `src/sdv_bmt/dut/native.py` line 70 (`describe_native`) | **⚠ SHARED CODE** |
| `prepare` | 300 s by default, **1800 s for Eagle-N** (JIT kernel compile and trace capture take minutes) | default: `native.py` line 184; override: `src/sdv_bmt/dut/eagle_n/adapter.py` line 71 (`PREPARE_TIMEOUT_S`) | **Eagle-N plugin file** |
| `stage` | 600 s | `src/sdv_bmt/dut/native.py` line 185 (`STAGE_TIMEOUT_S`) | shared default; a plugin can override the attribute |
| `cleanup` | 60 s | `src/sdv_bmt/dut/native.py` line 186 (`CLEANUP_TIMEOUT_S`) | shared default; a plugin can override the attribute |
| launcher wall time | 1800 s by default; `run_eagle_n.sh` passes 6000 s | `tools/eagle_n/in_container.py --wall-seconds` | **Eagle-N tool option**. When it passes, the launcher only reports; it never kills the container |

## Requirements

| component | version used | note |
|---|---|---|
| Eagle-N A0 board, BOS kernel driver | device node `/dev/bos/0` | the board has its own power and takes the PCIe reference clock from the host |
| BOS container image | `sha256:bb29c5af20279d1ad95805ebfb2061399d409778b3ba54d87b1f102235d6ec8d` (pinned in `in_container.py`) | provides glibc 2.34 and the JIT toolchain; every device step, build and SDV-BMT command runs inside it |
| BOS SDK (TT-Metal / TTNN) | commit `6878560cb7dbd8cbef9be01868b7e40424a37308`, `build_Release_distributed` | mounted read-only from `--sdk` (default `/home/bmt/tt-metal`); library SHA-256s recorded in `<executor>.sdk.json` and checked by the adapter |
| 1 GiB hugepages | at least 1 free page in the existing pool | the launcher mounts a private hugetlbfs; it never changes the system reservation |
| repo `.venv` | Python 3.11 | the host `python3` may be older: pass `PYTHON=.venv/bin/python` to `make` |
| ONNX Runtime (L1 reference only) | 1.19.2, CPU | a separate read-only venv (`ORT_PY` in `run_l1_ort.sh`) |
| Host resources | 1 run stages about 14 GiB on the evidence file system; the runner and SDV-BMT use several GB of RAM | run nothing heavy on the host during a device run |

Inputs are prepared as for the GPU reference (repository README): the three DeepLabV3+ ONNX files and the
`datasets/cityscapes-val500` index.

## Assets (outside the repository)

`$A` = `EAGLE_N_ASSETS` (reference host: `/home/bmt/caontd/eagle_n_integration_assets`).

| path | content | pinned by |
|---|---|---|
| `$A/libeagle_n_ttnn.so` | executor used by R-18 | `executor_sha256` in `configs/duts/eagle-n-r18.yaml` and in the R-18 receipt |
| `$A/libeagle_n_ttnn.r50dev.so` | executor used by R-50 and R-101 (built from `native/src/eagle_n/ttnn_executor.cpp` of this branch) | `executor_sha256` in `eagle-n-r50-bf16.yaml`, `eagle-n-r101-bf16.yaml` and in their receipts |
| `$A/r18/`, `$A/r50_bf16/`, `$A/r101_bf16/` | `<model>.plan.json`, `<model>.weights.f32` (host constants, converted to the plan precision on load), `compile_record.json`, `native_qualification.json` (receipt), `npu-capture.golden.jsonl` | plan and weights SHA-256 in the workload and the receipt |
| `$A/kernel_cache_<model>` | TT-Metal JIT kernel cache, reused across runs | not pinned; an empty cache only makes prepare slower |

**Do not rebuild the executor** (`make -C native eagle-n-sdk`, `tools/eagle_n/build_native.sh`,
`in_container.py build`) unless you intend to requalify: a rebuild changes its SHA-256, the receipts become stale and
L1 must be rerun. The R-18 executor was built on 2026-11-15 from an earlier revision of this source; a rebuild for R-18
from this branch therefore also needs requalification. Rebuilding the **runner** (`make -C native eagle-n`) is always safe and is required whenever the
native source tree changes (its stamp must equal the current digest).

The DUT and workload configs hold absolute paths of the reference host. On another host, regenerate them with
`configure.py` (step 3); executor, plan and weights keep their SHA-256, so the receipts stay valid as long as the files
are the same.

## Steps

`$A` = assets root, `$E` = evidence root (`EAGLE_N_EVIDENCE`), `<m>` = `r18`, `r50` or `r101`.
Run every command from the repository root, as the user that owns the evidence root, after `sudo -v` for any step
that starts a container. **Device** = the step uses the NPU: one at a time, NPU idle, no heavy host process.

| # | step | where | command | device |
|---|---|---|---|---|
| 1 | Build the native runner | host | `make -C native eagle-n PYTHON=.venv/bin/python` | no |
| 2 | Export the plan (once per model; no device) | host, repo `.venv` | R-18: `python3 tools/eagle_n/export_plan.py --variant r18 --onnx <onnx> --out $A/r18`; R-50 / R-101: `python3 tools/eagle_n/export_plan.py --variant <m> --precision bf16 --graph tuned --envelope $A/r18/r18.plan.json --onnx <onnx> --out $A/<m>_bf16` (the tuned export runs the proven-block decomposition) | no |
| 3 | Generate the configs | host | R-18: `python3 tools/eagle_n/configure.py --variant r18`; R-50 / R-101: `python3 tools/eagle_n/configure.py --variant <m> --precision bf16 --executor $A/libeagle_n_ttnn.r50dev.so` | no |
| 4 | L0 safety gates for a new or changed plan: L1-memory preflight (no program reaches the cores) and proven-shape survey | container | `python3 tools/eagle_n/in_container.py l1 0:0 --preflight --dut <dut> --workload <workload> --kernel-cache $A/kernel_cache_<m>`, then `python3 tools/eagle_n/proven_survey.py --candidate <new preflight.json> --proven <preflight.json of plans that already ran> --out <survey.json>`: **0 unproven calls** required | no (NO_DISPATCH) |
| 5 | L0: one golden image, several identical trace replays | container | `python3 tools/eagle_n/in_container.py l1 0:0 --golden-extra 2 --dut <dut> --workload <workload> --kernel-cache $A/kernel_cache_<m>` | **yes** |
| 6 | L1: val500 on the NPU in two parts, merged | container + host | `in_container.py l1 0:250 …` and `in_container.py l1 250:250 …` (same options as step 5), then `python3 tools/eagle_n/l1_merge.py <part1>/l1 <part2>/l1 --out $E/l1_merged_<m>_bf16_<stamp>` | **yes** |
| 7 | L1: ONNX Runtime FP32 reference | host CPU | `VARIANT=<m> NPU_DIRS="<part1>/l1 <part2>/l1" NPU_MIOU=<merged NPU mIoU> tools/eagle_n/run_l1_ort.sh` | no (CPU heavy: not during a device run) |
| 8 | Qualification receipt (internal guard) | host | `python3 tools/eagle_n/qualify_native.py --dut <dut> --workload <workload> --l1-npu <l1_merged dir> --l1-ort <l1_ort dir>` | no |
| 9 | L2: describe + validate | container | `tools/eagle_n/run_eagle_n.sh <m> l2` | no |
| 10 | L3: capture (asap) + evaluate; mIoU must equal L1 | container | `tools/eagle_n/run_eagle_n.sh <m> l3` | **yes** |
| 11 | First golden (only for a new model or plan) | host | copy `<run>/capture.golden.jsonl` to `$A/<m>…/npu-capture.golden.jsonl` and to `configs/references/…eagle-n…-capture.golden.jsonl`, then rerun step 3 to produce the baseline plan with the golden | no |
| 12 | L4: second capture, identity 1.0 against the golden | container | `tools/eagle_n/run_eagle_n.sh <m> l4` | **yes** |
| 13 | L5: baselines (5 runs) | container | `tools/eagle_n/run_eagle_n.sh <m> l5` (`L5_RUNS`, `L5_START=k` to continue after a reviewed stop) | **yes** |

`tools/eagle_n/run_eagle_n.sh <m> check` runs the host checks of steps 9–13 without Docker or device: receipt,
artifacts and runner stamp consistent, capture plan asap with a watchdog-safe timeout, golden present, and the
expected mIoU taken from the receipt.

## Checks that must hold

- **Steps 4–5:** the preflight reports no op over the L1 bank size; the survey finds 0 unproven SDK calls; the golden
  replays give identical labels and agree with ONNX Runtime FP32 at ≥ 0.99 (internal floor).
- **Steps 6–8:** the merged NPU mIoU is within the proposed G15 bound of the GPU FP32 reference (drop ≤ 0.01 for BF16;
  G15 is a maintainer decision). The receipt binds executor, plan, weights and index SHA-256 to that L1 evidence.
- **Step 10:** 500/500 completed, 0 timed out, 0 error, and the evaluate mIoU equals the L1 NPU mIoU in the receipt
  exactly (`CHECK_OK`).
- **Step 12:** the re-extracted golden equals the reference golden: identity rate 1.0 (`IDENTITY_OK`).
- **Step 13:** every run: identity 1.0, mIoU equal to L1, 0 deadline misses (`CHECK_OK`).

## Deviations from the guide (decisions for the maintainer)

| ID | Model | Level | Guide | This port | Plan |
|---|---|---|---|---|---|
| watchdog | all | L3, L4 | GPU-copy capture: asap, request timeout 1 s | asap, request timeout and drain 60 s (config only; see the watchdog section) | `*-capture-asap-t60.yaml` |
| D2 | R-18 | L5 | 2 Hz baseline | 0.8 Hz, deadline 1.25 s | `deeplabv3plus-r18-eagle-n-baseline-provisional-0p8hz.yaml` |
| D4 | R-50 | L5 | 2 Hz baseline | 0.25 Hz, deadline, timeout and drain 4 s | `deeplabv3plus-r50-eagle-n-bf16-baseline-provisional-0p25hz.yaml` |
| D6 | R-101 | L5 | 2 Hz baseline | 0.2 Hz, deadline, timeout and drain 5 s | `deeplabv3plus-r101-eagle-n-bf16-baseline-provisional-0p2hz.yaml` |
| D1, D3, D5 | all | L3, L4 | asap | `fixed_rate` (evidence before 2026-11-18; replaced by `*-capture-asap-t60.yaml`) | `*-capture-provisional-*.yaml` |

D2, D4 and D6 exist because one request takes longer than the 0.5 s period of 2 Hz; the rate keeps the period at
≥ 1.25 × the maximum DUT-service, the guide's own rule. The 2 Hz plans (`*-baseline.yaml`) are kept as generated.

## Safety rules

- New or changed plans reach the NPU only after step 4 passes. A device hang in prepare was traced to one op shape
  outside the proven envelope; the survey exists to prevent that.
- Nothing is retried automatically. The first failure stops the scripts; review before rerunning.
- Never kill a process or container that holds the NPU. If the board stops answering, the reference host recovers
  only by a host reboot through the BIOS PCIe settings page; a driver warm reset did not recover it.

## Tools in this directory

| file | purpose |
|---|---|
| `run_eagle_n.sh` | L2–L5 for one model through SDV-BMT (`check`, `l2`, `l3`, `l4`, `l5`) |
| `eagle_n_checks.py` | host checks used by `run_eagle_n.sh`: pins from the receipt, watchdog budget, result and identity checks |
| `in_container.py` | launcher: every Docker step (`build`, `build-runner`, `describe`, `validate`, `run`, `evaluate`, `l1`), receipt and staging prechecks, private hugepage mount, never kills a run container |
| `export_plan.py` | ONNX (H04-checked) → device plan + constants; `--graph tuned` uses `decompose_proven.py` |
| `decompose_proven.py` | rewrites layer4 and the head of R-50/R-101 into blocks of proven shapes |
| `configure.py` | DUT, workload and experiment configs from the assets (one memory budget for all BF16 plans) |
| `l1_native.py`, `l1_merge.py` | L0/L1 on the NPU (inside the container) and the exact merge of the parts |
| `l1_ort.py`, `run_l1_ort.sh` | L1 ONNX Runtime FP32 reference on the CPU |
| `proven_survey.py` | proven-shape survey of every SDK call of a plan against plans that already ran |
| `qualify_native.py` | qualification receipt from the L1 evidence |
| `build_native.sh`, `build_runner.sh` | executor and runner builds inside the container (executor: requalification needed) |
| `run_l3_capture.sh`, `run_l3_capture_provisional.sh`, `run_l4_second_capture.sh`, `run_l5_provisional.sh`, `run_l1_two_parts.sh`, `run_tuned_trial.sh` | R-18 scripts of 2026-11-15, kept for the R-18 evidence; superseded by `run_eagle_n.sh` |
