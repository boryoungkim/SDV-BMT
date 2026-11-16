# DeepLabV3+ R-18 on Eagle-N: handover release (2026-11-16)

Branch `cao/eagle-n-deeplab`, based on SDV-BMT v0.1.2 (`f3c8da0`). SDV-BMT measurement code (runner, scheduler,
metrics, bundle schema, evaluator) is **unmodified**. The only edits to shared files are in a separate commit for review
(see [Shared files](#shared-files-pending-boryoung)).

## 1. Status

| Level | Goal | Result | Status |
|---|---|---|---|
| L0 | One image on the NPU | Labels identical to the validated output (`621089…`) | Passed |
| L1 | Offline numerics, val500 | mIoU NPU 0.767565 / ORT FP32 0.767566 (Δ −7e-7); label agreement 99.82 % | Done, G15 pending |
| L2 | `describe` + `validate` | Passed (inside the BOS container) | Passed |
| L3 | Capture + evaluate | 500/500; mIoU 0.7675648837 = L1 exactly; DUT-service p50 985.95 ms | **Provisional (D1)** |
| L4 | NPU golden + 2nd capture | Golden `af30e654…`; identity rate 1.0 | Passed |
| L5 | 5 baselines | 5/5: 0 drops/timeouts/errors/deadline misses; identity 1.0; DUT-service p50 985.938 ms (±0.007) | **Provisional (D2)** |

- **D1:** the capture ran at `fixed_rate` 0.8 Hz instead of `asap` because of the measure watchdog; see issue 9.
- **D2:** the baselines ran at 0.8 Hz with a 1.25 s deadline, not 2 Hz, because DUT-service is 986 ms, above the
  400 ms limit for 2 Hz.

Executor: `libeagle_n_ttnn.so` sha256 `58d9b284…`. Precision: BF16 activations/weights, conv HiFi4 + FP32 accumulate.
Graph: manual TTNN C++ graph (Plan C / D01, approved by the leader), tuned 5872 → 968 ms with labels bit-identical.

## 2. What the branch contains

| Kind | Path | Purpose |
|---|---|---|
| Guide slot (new) | `src/sdv_bmt/dut/eagle_n/` | `bos_eagle_n` adapter (strict config, identity, backend spec) |
| Guide slot (new) | `native/src/eagle_n/`, `native/src/main_eagle_n.cpp` | Native backend (prepare/stage/submit/poll/collect) + TTNN executor source |
| Files added | `configs/duts/eagle-n-*.yaml`, `configs/workloads/*-eagle-n.yaml`, `configs/experiments/*eagle-n*` | R-18/50/101 configs; R-18 capture/baseline + provisional 0.8 Hz plans |
| Files added | `configs/references/…r18…eagle-n-capture.golden.jsonl` | L4 NPU golden (500 rows) |
| Guide slot (new) | `tools/eagle_n/` | export plan, configure, build, L1 (`l1_native/l1_ort/l1_merge`), receipt, container launcher, L3–L5 scripts |
| Files added | `tests/unit/test_eagle_n_*.py`, `tests/device/test_eagle_n_native.py` | Adapter / native contract / device tests |
| Outside guide | `tests/eagle_n/*.cpp` | Native backend guard + mock contract tests (could move to `native/tests/`) |
| Shared, edited | `src/sdv_bmt/cli/main.py`, `native/Makefile` | Register `EagleNDUT`; optional `eagle-n` targets (not in `all`/`portable`) |
| Outside guide | `docs/eagle_n/`, `README_EAGLE_N.txt` | Notes. `README_EAGLE_N.txt` §3 describes the 12/11 state (before L1–L5) |

**Not in the repo, on this server only:**
- `/home/bmt/caontd/eagle_n_integration_assets/`: executor `.so`, plan/weights, receipt, kernel cache, lock
- `datasets/cityscapes-val500` (Cityscapes terms; build with `tools/deeplab_family/build_cityscapes_val.py`)
- the BOS SDK

## 3. Reproduce on this server

```bash
git clone -b cao/eagle-n-deeplab https://github.com/boryoungkim/SDV-BMT.git && cd SDV-BMT
/home/bmt/caontd/eagle_n_porting/plan_a/compiler/python311/python/bin/python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'
make -C native portable PYTHON="$PWD/.venv/bin/python" && native/build/sdv-bmt-native-tests
SDV_BMT_MAKE_TARGETS=portable PYTHON="$PWD/.venv/bin/python" .venv/bin/python -B -m pytest -q
mkdir -p datasets && ln -s /home/bmt/caontd/SDV-BMT-eagle-n/datasets/cityscapes-val500 datasets/
# NPU runs (BOS container, device must be idle):
sudo -v
bash tools/eagle_n/run_l3_capture_provisional.sh    # L3 capture + evaluate
N=5 bash tools/eagle_n/run_l5_provisional.sh         # L5 baselines
```

What to expect from a clean clone, checked 2026-11-16:
- pytest: 637 passed, 52 skipped, 0 failed.
- Native tests: 0 failures.
- Runner source stamp: `66545323…`.

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
| 3 | L0 trial | CB 16.9 MB > L1 1.57 MB per core; host→device `write_tensor` rejected in blocking mode | Decoder 140-channel concat, unsplit depthwise; `write_tensor` must be non-blocking (Internal/SDK) | Row-major concat, spatial split with halo, M ≤ 512; `blocking=false` + FIFO on CQ0 | Size every op against 1.5 MB L1; check bundle `request_counts`, since exit 0 ≠ success |
| 4 | Tuning | Labels 94.93 % right after fusing the tail | Per-chunk uint32 typecast on a TILE index map is wrong after the first 32 rows (Internal/SDK) | Keep BF16 per chunk; typecast once on the full 1024×2048 map | Gate every optimization on the label hash `621089…` |
| 5 | Tuning | Conv on TILE input: 798 ms but 545 px differ | Sequence-dependent wrong untilize at 20-core boundaries; not reproducible in isolation (SDK, suspected) | Rejected (`CONV_TILE_INPUT=False`); frozen at 968 ms | A faster graph that changes labels is a precision change and needs approval |
| 6 | Trial | Run aborted | Disk full: VS Code cpptools cache 7.4 GB (Env) | Cleared the cache; later deleted 16 GB of R-50 `.npy` (hashes kept) | Check free disk before runs: staging needs ~14 GB |
| 7 | L3 prep | Staging budget refused 500 items | SDV-BMT stages 500 × 24 MiB FP32 = 11.7 GiB; docker `/dev/shm` is 64 MB; our 4 GiB budget was too small (Tool + Internal) | RSS probe (11.6 MiB per item) → R-18 budgets 6.5 / 7 GiB; private disk dir mounted as `/dev/shm` | Measure RSS per staged item before choosing budgets |
| 8 | L3 #1 | `dut.prepare` failed: "JSON non-integer number" | Native JSON parser accepts integers only; our receipt had floats (Internal vs Tool) | Receipt writes decimals as strings; self-check | Files read by native code must follow `native/src/common/json.hpp` rules |
| 9 | L3 #2 | Precondition failed: `SlotBusy` after the 1st warm-up | The runner warms up with submit + poll and never collects; our backend required collect (Internal) | Submit refused only while a request is not final; guard test updated | The mock test had `precondition_requests=0`. Test with the real plan's warm-up count |
| 10 | L3 #3 | Native runner killed after 154 s | asap measure watchdog uses n = 1 (`native.py:307-311`): 154 s for any item count; R-18 needs ~525 s (Tool) | **Not changed.** Provisional `fixed_rate` 0.8 Hz → limit 777 s (D1) | Slow DUTs (≈1 s/item) cannot use asap until the tool uses n = items |
| 11 | L3 | Launcher cleanup failed; hugepage mount left | sudo timestamp (15 min) expired during the 18 min run (Env) | Script refreshes `sudo -n -v` every 60 s | Keep sudo alive in long scripts; unmount by hand if a run is interrupted |
| 12 | L5 #1 | `validate` refused all 5 runs; script re-evaluated the old bundle 5 times | Golden path must be relative (Tool); script did not stop on failure (Internal) | Golden under `configs/references/`, relative path; script stops if validate fails or no new run folder | Launchers must fail fast; never let a failed step reuse an older result |
| 13 | Margin | request_completion max 986.4 ms vs `request_timeout` 1 s | Timeout kept from the GPU plan (Guide) | Unchanged; 13.6 ms margin reported | Any slowdown of 1.4 % turns into timeouts at the 1 s limit |
| 14 | Handover | 2 tests fail after L3 | Tests assert "no receipt yet"; receipt now exists legitimately (Internal) | Tests now check that the R-18 receipt is genuine (pins + evidence sha), R-50/R-101 have none, and the launcher refuses unqualified or stale graphs | Update guard tests when a gate is passed |
| 15 | Every run | ~165–200 s `dut.prepare` | JIT per process; the kernel cache is not reused across processes (Env) | Pinned kernel cache dir per graph | Count ~3.5 min of prepare per run when planning device time |

## 5. Shared files (pending Boryoung)

Guide §5 says shared files must not be edited by us, and the exact lines should be sent instead. The branch has them in
their own commit, so they can be dropped once the lines are added upstream.
- `src/sdv_bmt/cli/main.py`: `from ..dut.eagle_n import EagleNDUT` and the `DUT_ADAPTERS` entry. `ACTUAL_NOTICE` is
  restored to the v0.1.2 text.
- `native/Makefile`: `DIGEST_SRCS += $(wildcard src/eagle_n/*.cpp)` and the optional `eagle-n` / `eagle-n-sdk` targets.

## 6. Open decisions

1. **Latency**: 986 ms DUT-service is above the 400 ms limit for 2 Hz. Do we optimize further, or set the rate?
2. **asap watchdog**: should the tool use n = selected items? This is needed to rerun L3 as asap.
3. **D1 / D2**: accept the 0.8 Hz runs, or rerun them later.
4. **G15**: what is the BF16 accuracy acceptance criterion? Measured Δ mIoU is −7e-7.
5. **Receipt and docs**: keep the internal qualification receipt? Where should `docs/eagle_n/` go?

## 7. Notes for R-50 / R-101

- R-50 passed standalone only in **FP32**. BF16 failed the full-model gate (layer3 5.9 % > 5 %). Decide its precision
  before L0.
- `configs/duts/eagle-n-r50.yaml` still pins an old executor sha. Re-run `configure.py --variant r50` after the build.
- R-50 has 3.3× the weights of R-18 (165 MB) and a heavier graph. Before L3, re-measure:
  - staging/RSS per item (issue 7)
  - the timeout margin (issue 13)
  - the watchdog window (issue 10)
