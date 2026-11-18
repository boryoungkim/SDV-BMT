# DeepLabV3+ R-18 on Eagle-N: handover release (2026-11-16, updated 2026-11-18)

Branch `cao/eagle-n-deeplab`, based on SDV-BMT v0.1.2 (`f3c8da0`) with `main` merged in (`f11d95a`). SDV-BMT
measurement code (runner, scheduler, metrics, bundle schema, evaluator) is **unmodified**. The only edits to shared
files are the two registration lines listed in [Shared files](#5-shared-files).

Measurement results (latencies, rates) are not published here; they are in the evidence root. How to rerun every
level for R-18, R-50 and R-101 is in [`tools/eagle_n/README.md`](../../tools/eagle_n/README.md).

## 1. Status

| Level | Goal | Result | Status |
|---|---|---|---|
| L0 | One image on the NPU | Labels identical to the validated output (`621089…`) | Passed |
| L1 | Offline numerics, val500 | mIoU NPU 0.767565 / ORT FP32 0.767566 (Δ −7e-7); label agreement 99.82 % | Done, G15 pending |
| L2 | `describe` + `validate` | Passed (inside the BOS container) | Passed |
| L3 | Capture + evaluate | 500/500; mIoU equal to L1 exactly | **Provisional (D1)** |
| L4 | NPU golden + 2nd capture | Golden `af30e654…`; identity rate 1.0 | Passed with the D1 plan |
| L5 | 5 baselines | 5/5: 0 drops/timeouts/errors/deadline misses; identity 1.0 | **Provisional (D2)** |

- **D1:** the evidence capture ran at `fixed_rate` 0.8 Hz instead of `asap` because of the measure watchdog (issue 10).
  For main the capture plan is `deeplabv3plus-r18-eagle-n-capture-asap-t60.yaml`: `asap` kept, request timeout and
  drain 60 s, config only. L3/L4 of R-18 with this plan are still to be rerun.
- **D2:** the baselines ran at 0.8 Hz with a 1.25 s deadline, not 2 Hz, because one request takes longer than the
  0.5 s period of 2 Hz (above the guide's 400 ms limit).

Executor: `libeagle_n_ttnn.so` sha256 `58d9b284…`. Precision: BF16 activations/weights, conv HiFi4 + FP32 accumulate.
Graph: manual TTNN C++ graph (Plan C / D01, approved by the leader), tuned with labels kept bit-identical at every step.

## 2. What the branch contains

| Kind | Path | Purpose |
|---|---|---|
| Guide slot (new) | `src/sdv_bmt/dut/eagle_n/` | `bos_eagle_n` adapter (strict config, identity, backend spec) |
| Guide slot (new) | `native/src/eagle_n/`, `native/src/main_eagle_n.cpp` | Native backend (prepare/stage/submit/poll/collect) + TTNN executor source |
| Files added | `configs/duts/eagle-n-*.yaml`, `configs/workloads/*-eagle-n*.yaml`, `configs/experiments/*eagle-n*` | R-18, R-50 BF16, R-101 BF16 configs; capture (`asap-t60`), baseline and provisional plans |
| Files added | `configs/references/…eagle-n…-capture.golden.jsonl` | NPU goldens (R-18, R-50 BF16, R-101 BF16; 500 rows each) |
| Guide slot (new) | `tools/eagle_n/` | export, configure, build, L1, receipt, container launcher, `run_eagle_n.sh` (L2–L5), README |
| Files added | `tests/unit/test_eagle_n_*.py`, `tests/device/test_eagle_n_native.py` | Adapter / native contract / device tests (no local assets needed on CI) |
| Outside guide | `tests/eagle_n/*.cpp` | Native backend guard + mock contract tests (could move to `native/tests/`) |
| Shared, edited | `src/sdv_bmt/cli/main.py`, `native/Makefile` | Register `EagleNDUT`; optional `eagle-n` targets (not in `all`/`portable`) |
| Outside guide | `docs/eagle_n/`, `README_EAGLE_N.txt` | Notes. `README_EAGLE_N.txt` §3 describes the 12/11 state (before L1–L5) |

**Not in the repo, on this server only:**
- `/home/bmt/caontd/eagle_n_integration_assets/`: executor `.so`, plan/weights, receipts, kernel caches, lock
- `datasets/cityscapes-val500` (Cityscapes terms; build with `tools/deeplab_family/build_cityscapes_val.py`)
- the BOS SDK

## 3. Reproduce on this server

```bash
git clone -b cao/eagle-n-deeplab https://github.com/boryoungkim/SDV-BMT.git && cd SDV-BMT
/home/bmt/caontd/eagle_n_porting/plan_a/compiler/python311/python/bin/python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'
make -C native portable PYTHON="$PWD/.venv/bin/python" && native/build/sdv-bmt-native-tests
SDV_BMT_MAKE_TARGETS=portable .venv/bin/python -B -m pytest -q
make -C native eagle-n PYTHON="$PWD/.venv/bin/python"      # Eagle-N runner (never eagle-n-sdk: it rebuilds the executor)
mkdir -p datasets && ln -s /home/bmt/caontd/SDV-BMT-eagle-n/datasets/cityscapes-val500 datasets/
# NPU runs (BOS container, device must be idle); see tools/eagle_n/README.md
sudo -v
tools/eagle_n/run_eagle_n.sh r18 check      # host checks only
tools/eagle_n/run_eagle_n.sh r18 l3         # capture + evaluate (asap-t60)
tools/eagle_n/run_eagle_n.sh r18 l4         # second capture, identity 1.0
tools/eagle_n/run_eagle_n.sh r18 l5         # 5 baselines (provisional rate)
```

What to expect from a clean clone: native tests with 0 failures and pytest with 0 failed (device tests report NOT RUN
without the hardware).

Device runs are for account `bmt`. Other Linux accounts cannot open `device0.lock` (mode 600) or the dataset
(mode 700). Evidence is written to `/home/bmt/caontd/eagle_n_integration_evidence/`.

## 4. Issues hit and lessons learned

Source of each issue:
- **Guide**: Boryoung's porting guide
- **Tool**: original SDV-BMT behaviour
- **Internal**: our code or our guard
- **Env**: server or SDK

| # | Stage | Problem | Cause (source) | Fix | Lesson |
|---|---|---|---|---|---|
| 1 | Dataset | `build_cityscapes_val.py` crashed writing the manifest | Pillow 7.0 has no `PIL.features.version` (Env) | Re-ran the same tool with a shim; `index.jsonl` sha matches the guide | Check the index sha256 against the guide, not the exit code |
| 2 | L2 | `validate` / `describe` rejected on the host | Host glibc too old; `libtracy.so` not on the host (Env) | Run through `tools/eagle_n/in_container.py` | Every SDK-touching step goes through the BOS container |
| 3 | L0 trial | CB 16.9 MB > L1 1.57 MB per core; host→device `write_tensor` rejected in blocking mode | Decoder 140-channel concat, unsplit depthwise; `write_tensor` must be non-blocking (Internal/SDK) | Row-major concat, spatial split with halo, M ≤ 512; `blocking=false` + FIFO on CQ0 | Size every op against the L1 bank; check bundle `request_counts`, since exit 0 ≠ success |
| 4 | Tuning | Labels 94.93 % right after fusing the tail | Per-chunk uint32 typecast on a TILE index map is wrong after the first 32 rows (Internal/SDK) | Keep BF16 per chunk; typecast once on the full 1024×2048 map | Gate every optimization on the label hash `621089…` |
| 5 | Tuning | Conv on TILE input was faster but changed 545 pixels | Sequence-dependent wrong untilize at 20-core boundaries; not reproducible in isolation (SDK, suspected) | Rejected (`CONV_TILE_INPUT=False`) | A faster graph that changes labels is a precision change and needs approval |
| 6 | Trial | Run aborted | Disk full: VS Code cpptools cache 7.4 GB (Env) | Cleared the cache; later deleted 16 GB of R-50 `.npy` (hashes kept) | Check free disk before runs: staging needs ~14 GB |
| 7 | L3 prep | Staging budget refused 500 items | SDV-BMT stages 500 inputs; docker `/dev/shm` is 64 MB; our 4 GiB budget was too small (Tool + Internal) | RSS probe (11.6 MiB per item) → budgets 6.5 / 7 GiB for every BF16 plan; private disk dir mounted as `/dev/shm` | Measure RSS per staged item before choosing budgets |
| 8 | L3 #1 | `dut.prepare` failed: "JSON non-integer number" | Native JSON parser accepts integers only; our receipt had floats (Internal vs Tool) | Receipt writes decimals as strings; self-check | Files read by native code must follow `native/src/common/json.hpp` rules |
| 9 | L3 #2 | Precondition failed: `SlotBusy` after the 1st warm-up | The runner warms up with submit + poll and never collects; our backend required collect (Internal) | Submit refused only while a request is not final; guard test updated | The mock test had `precondition_requests=0`. Test with the real plan's warm-up count |
| 10 | L3 #3 | Native runner killed after 154 s | asap measure watchdog uses n = 1 (`native.py` `run_window`): 154 s for any item count (Tool) | Shared code **not changed**. Evidence: `fixed_rate` 0.8 Hz (D1). For main: `asap` kept, request timeout and drain 60 s → limit 2160 s (config only) | A DUT that needs about 1 s or more per request cannot use the GPU-copy asap plan; size the timeout with the README formula |
| 11 | L3 | Launcher cleanup failed; hugepage mount left | sudo timestamp (15 min) expired during the run (Env) | Script refreshes `sudo -n -v` every 60 s | Keep sudo alive in long scripts; unmount by hand if a run is interrupted |
| 12 | L5 #1 | `validate` refused all 5 runs; script re-evaluated the old bundle 5 times | Golden path must be relative (Tool); script did not stop on failure (Internal) | Golden under `configs/references/`, relative path; script stops if validate fails or no new run folder | Launchers must fail fast; never let a failed step reuse an older result |
| 13 | Margin | Request completion close to the 1 s `request_timeout` of the GPU plan | Timeout kept from the GPU plan (Guide) | Kept for the D1/D2 evidence; the `asap-t60` capture plan uses 60 s | A small slowdown would turn into timeouts at the 1 s limit |
| 14 | Handover | 2 tests fail after L3 | Tests assert "no receipt yet"; receipt now exists legitimately (Internal) | Tests check that every receipt is genuine (pins + evidence sha) and that the launcher refuses unqualified or stale graphs | Update guard tests when a gate is passed |
| 15 | Every run | `dut.prepare` takes minutes | JIT per process; the kernel cache is not reused across processes (Env) | Pinned kernel cache dir per graph | Count a few minutes of prepare per run when planning device time |
| 16 | CI | GitHub CI failed on the Eagle-N tests | Tests read the local assets, absent on CI runners (Internal) | Tests that need assets report NOT RUN; mock contract tests use synthetic plans | Run the suite in a clean clone with local paths hidden before pushing |

## 5. Shared files

Guide §5 says shared files must not be edited by us, and the exact lines should be sent instead. After the merge of
`main` (`f11d95a`) they are:
- `src/sdv_bmt/cli/main.py`: `from ..dut.eagle_n import EagleNDUT` and the `EagleNDUT` entry in `DUT_ADAPTERS`.
- `native/Makefile`: `$(wildcard src/eagle_n/*.cpp)` in `DIGEST_SRCS` and the optional `eagle-n` / `eagle-n-sdk` targets.

## 6. Open decisions

1. **Latency**: DUT-service is above the 400 ms limit for 2 Hz. Do we optimize further, or accept a lower rate (D2)?
2. **Capture timeout**: accept the 60 s request timeout of the `asap-t60` capture plans, or change the watchdog
   formula in the tool (n = selected items) so a shorter timeout can be used.
3. **D2**: accept the 0.8 Hz baselines, or rerun them later.
4. **G15**: what is the BF16 accuracy acceptance criterion? Measured Δ mIoU is −7e-7.
5. **Receipt and docs**: keep the internal qualification receipt? Where should `docs/eagle_n/` go?

## 7. R-50 and R-101

R-50 BF16 and R-101 BF16 are in the branch since 2026-11-18, with the same ladder and tools as R-18 (see
`tools/eagle_n/README.md`). The FP32-conv R-50 path and the untuned plain r50/r101 configs were removed (never
qualified). R-50 also has an 8-bit-weight plan (`bfp8w`) that is exported but not qualified.
