# Changelog

All notable changes to SDV-BMT are recorded here. Versions follow semantic versioning.

## [Unreleased]

### Added

- **3D detection (LiDAR): CenterPoint-Pillars**, the mmdetection3d nuScenes checkpoint (pillar 0.2 m, circle NMS).
  New workload `centerpoint_nuscenes` (input contract `nuscenes-pillar02-p40000/1`): the 10-sweep point cloud and the
  voxelization are built offline (pillar tensor payloads), the device runs pillar features, BEV scatter, backbone,
  neck and head, and decode + circle NMS + encode (`centerpoint-det3d-f32/1`) run in the native output stage with a
  byte-identical Python twin. A NaN heatmap logit, or a detection value that does not fit float32, is an output error
  in both. Results: `docs/results/centerpoint-pillars-gpu-reference-result.md`; README and `docs/roadmap.md` describe
  CenterPoint, run series and the LLM path; `THIRD_PARTY_NOTICES.md` lists the checkpoint, nuScenes and the tools'
  packages.
- **Native cuDNN plan operations** for CenterPoint: pillar feature net, pillar scatter, zero fill and depth-to-space,
  in a third NVRTC source compiled only when a plan uses it.
- **Model-level Metric Set v1.2** (`sdv-bmt-model-metric-set-v1.2`, selectable; the default stays v1.1): v1.1 plus
  eight nuScenes detection metrics (`quality.det3d.coverage`, `quality.det3d.conditional.map`, `.nds`, `.mate`,
  `.mase`, `.maoe`, `.mave`, `.maae`), computed by the evaluator `nuscenes-det3d-cvpr2019/1` from the ground truth
  frozen in the bundle.
- **Run series (minimal).** An experiment may declare `series: {max_items_per_run, boundary}`; `sdv-bmt run` then
  executes the dataset selection as consecutive ordinary runs (parts) that never split a block of index rows with the
  same `metadata[boundary]` value, and writes `series.json`. `sdv-bmt evaluate SERIES_DIR` checks that the parts
  differ only in their selection and produces one result: counts summed, distributions over all requests, rates over
  the sum of the parts' windows, task quality evaluated once over all items. A series takes no output reference and
  one repetition. `series.json` records the process gap between part runs and, for real-time parts, each part's
  measurement window on the host monotonic clock and the gap between consecutive windows.
- **Tools** (offline): `tools/centerpoint/` (official info files and reference run, nuScenes input builder, model
  re-definition, ONNX export, plan converter, numeric gate, output-twin and evaluator cross-checks, cuDNN table pinning,
  experiment generator).
- **LLM path (Python measurement plane)**, `python -m sdv_bmt.llm`. It runs Llama 3.2 1B / 3B Instruct and
  Llama 3.1 8B Instruct on a GPU with vLLM and on a FuriosaAI RNGD card with furiosa-llm. Requests are
  pre-tokenized and sent one at a time. TTFT, TPOT and E2E are taken from host monotonic times of the runtime's
  stream. Two workloads are fixed: S1–S5 fixed-length synthetic token IDs, and MMLU with 1,000 items (zero-shot,
  generation-based). Prefix caching off and greedy decoding are verified from the values the runtime reports.
  Results go into their own bundle format (`sdv-bmt-llm-bundle/1`) and are evaluated from the bundle only with the
  new catalog `sdv-bmt-llm-v1`. The CNN Metric Sets are unchanged. Tools are in `tools/llm/` (RNGD build and smoke,
  baseline script, prefix-cache control, tokenizer and weight identity checks, GPU vs Hugging Face check,
  independent MMLU re-scorer) together with the package lists of the two measurement environments. Documentation:
  `docs/llm_benchmark.md`. Results: `docs/results/llm-llama3-gpu-rngd-result.md`. The architecture tests name the
  LLM package's exceptions per module (host clock and asyncio in `llm/runner.py`, asyncio and lazily imported
  runtimes in `llm/backends.py`); the rules for all other packages are unchanged.
- **Metric table and one-command runs.** `python -m sdv_bmt.cli.table DIR [DIR ...] --out FILE.csv` (CNN / LiDAR
  experiments and run series) and `python -m sdv_bmt.llm table DIR [DIR ...] --out FILE.csv` (LLM experiments) write
  one metric table (`docs/metric_table.md`): four representative metrics per model under readable names first, then
  the overall result per experiment and metric (median, min and max over the valid runs), then one row per run and
  metric copied unchanged from the newest evaluation. Every row names the task, dataset, model, device, experiment
  and catalog. Raw files and evaluation records are re-verified against the bundle manifests, and a table is never
  overwritten. The commands print the representative metrics as a table, followed by the path of the CSV file.
  `tools/run_model.sh <model>` runs one model end to end (validation, `REPS` timed runs, evaluation, table) with a
  progress bar, after a start header that names the models and their datasets. A Llama model runs on one device
  per name: `llama-<m>` on the GPU (vLLM), `llama-<m>-rngd` on the FuriosaAI RNGD card (furiosa-llm), and existing
  runs are refused per device. `all` runs the 12 models on the GPU, `all-mla400` the five Mobilint MLA400 YOLOX
  plans on the NPU host, and `centerpoint-pillar02-mini` runs CenterPoint on the 162 mini samples only (no val run
  series). `tools/show_metrics.py` shows metric tables in a readable form (headline rows, or every
  model row per experiment and layer with times in ms).

### Fixed

- **Measure watchdog of the native adapters with `asap` replay.** The time limit for the native runner's
  MEASURE/DRAIN reply counted an `asap` window as one request, so any `asap` window was killed after a fixed time
  whatever its item count (about 154 s with a 1 s request timeout and 32 precondition requests). It now allows one
  request timeout per planned item. Fixed-rate and recorded schedules are unchanged.
- **CLI notice of ACTUAL runs** no longer says "not an NPU performance result"; it says that the bundle's device
  identity names the DUT (a GPU run is reference-accelerator validation).

## [0.1.2] — 2026-11-10

Model extension on the GPU reference track: eight reference models instead of one (five YOLOX detectors on COCO,
three DeepLabV3+ segmenters on Cityscapes), segmentation quality in the metric catalog, and stricter checks before a
run starts. The measurement core (replay, admission, timeouts, evidence bundle, guards) is unchanged. GPU results
are reference-accelerator results, not NPU results.

### Added

- **YOLOX family.** YOLOX-Tiny (416 × 416), M, L and X next to YOLOX-S, all official COCO-trained ONNX models
  of the YOLOX 0.1.1rc0 release. The workload now declares its input size and anchor count
  (`options.input_size`, `options.num_anchors`), checked at start-up; the plan converter accepts S × S inputs.
  Workload, pinned DUT, experiment and golden-capture files for the four new models; an input configuration for
  Tiny's 416 × 416 preprocessing contract.
- **Semantic segmentation: DeepLabV3+ R-18, R-50 and R-101 (D8)**, the mmsegmentation Cityscapes checkpoints
  (512 × 1024, 80k), run on full 2048 × 1024 Cityscapes val images. New workload `deeplabv3plus_cityscapes`
  (input contract `cityscapes-rgb-norm-chw2048x1024-f32/1`); the device returns a uint8 train-id label map,
  encoded as `labelmap-u8-rowrle/1` by the native output stage and by a byte-identical Python twin.
- **Model-level Metric Set v1.1** (`sdv-bmt-model-metric-set-v1.1`): Metric Set v1 unchanged plus
  `quality.segmentation.coverage` and `quality.segmentation.conditional.miou`, computed by the evaluator
  `cityscapes-semseg-miou/1` from the ground truth frozen in the bundle. Its mIoU equals the official
  cityscapesScripts evaluation.
- **Native cuDNN plan operations** for DeepLab: ReLU after a residual add, global average pool, broadcast,
  bilinear resize and uint8 argmax, in a second NVRTC source that is compiled only when a plan uses it.
- **Tools** (offline, outside the core's code identity): `tools/yolox_family/make_experiments.py` and
  `tools/deeplab_family/` (ONNX export, plan converter, numeric gates, official-implementation and mIoU
  cross-checks, Cityscapes val builder, replay-rate rule, experiment generator). The replay rate of each family's
  baseline is set in one place (`make_experiments.py --rate-hz`).
- Public result summary `docs/results/v0.1.2-gpu-reference-result.md` and release notes `docs/release-notes/v0.1.2.md`.

### Changed

- **Default catalog.** `sdv-bmt evaluate` uses Metric Set v1.1. On detection bundles its 91 v1 metrics have the
  same values as v1 and the two segmentation metrics are `not_applicable`. The catalog identity differs, so the
  Measurement Guard does not pair a v1.1 evaluation with a v1 evaluation; `--catalog sdv-bmt-model-metric-set-v1`
  reproduces v0.1.0 evaluations byte for byte.
- **Checks before a run starts.** Plans that v0.1.0 accepted can now be rejected:
  - fixed-rate replay with a frame-period deadline: `period_ns` must equal `floor(1e9 / rate_hz)`;
  - the input's `family.preprocessing_contract` must equal the workload's input contract (the fake inputs now
    declare `fake-identity-input/1` instead of `fake-bytes-v1`);
  - the native runner must carry the build stamp of the current native sources (`make -C native` writes
    `<runner>.source-tree.sha256`); rebuild after changing anything under `native/`. The manifest records
    `native.source_tree_sha256`;
  - NVIDIA: every cuDNN plan step must read and write inside its buffers and the plan must write the whole task
    output (checked at prepare); `device_program.timing_offset` must be a multiple of 8.
- Package version 0.1.2 (`sdv-bmt --version`); there is no 0.1.1. The v0.1.2 evidence bundles were recorded
  while the package version was still 0.1.0 and keep that label; `code_identity`, not the version string,
  identifies the code that produced a bundle.

### Fixed

- Test `test_a_long_request_holds_the_slot_and_arrivals_are_dropped` (fake backend, no device) assumed that its
  deliberately long request finishes inside the floor plan's 100 ms request timeout; that took about 55 ms on the
  reference host but longer once on a shared CI runner. The test now allows 1 s. Test only.

### Not included in 0.1.2

Everything listed under 0.1.0, and: NPU results, other precisions or batch sizes, contrast designs that attribute
differences between models, and the measurement extensions planned for v0.2.

## [0.1.0] — 2026-10-09

First public release: **SDV-BMT — Model-level Accelerator Benchmark Core**. A research benchmark
framework; its Model-level measurements do not establish WCET, FTTI, ISO/PAS 8800, ISO 26262 or SOTIF
compliance, or vehicle-level safety. Licensed under the Apache License, Version 2.0.

### Included

- Stage 0 measurement design (design basis, public edition in `docs/design/`) and documentation in `docs/`.
- Stage 1A deterministic Model-level core: typed inputs, deterministic replay schedule (fixed rate, recorded,
  asap), N=1 / queue 0 / `drop_new` admission, timeouts with late completion, append-only evidence bundles,
  bundle-only evaluation, golden regression cases G-1..G-5 (MOCK execution on a virtual clock).
- Stage 1R realtime harness: C++ native measurement plane (`native/`) driven by the Python control plane
  through a phase protocol; host monotonic run clock; one native process per run; per-request facts recorded
  without Python in the measured path.
- NVIDIA GPU reference backend (reference accelerator, not the NPU target): CUDA event brackets, NVRTC kernels,
  a cuDNN plan executor with pinned algorithm tables, NVML window snapshots, optional NVTX / Nsight Systems
  cross-check configuration.
- Real-model reference workload: YOLOX-S (ONNX → cuDNN plan) on the first 1,000 COCO val2017 images, with
  the Python preprocessing reference and a native output pipeline twin.
- Model-level Metric Set v1 (`sdv-bmt-model-metric-set-v1`, 91 metrics in five layers: performance,
  timeliness, task quality, diagnostics, validity/evidence), every result with its population context; the
  Stage 1A catalog `bmt-portable-v1-stage1a` stays selectable and reproduces earlier evaluations byte for
  byte.
- Canonical COCO detection quality (`coco-detection-bbox/1`) evaluated from a ground truth frozen in the bundle
  (manifest v5, `quality_reference.jsonl`).
- Design / Realization / Measurement Guards for pairwise contrast validity (no attribution without validated
  factor isolation).
- Tests: unit, conformance, golden regression, architecture boundaries, native C++ unit tests, and NVIDIA
  device tests (skipped as NOT RUN without the GPU).

### Fixed (release hardening)

- Native realtime loop: events that are already due when the loop handles them are now decided in the order
  of their semantic timestamps (arrival `t_sched`, timeout `t_submit_begin + request_timeout_ns`, completion
  `t_complete`); the Stage 1A tie order completion → timeout → arrival applies only to equal timestamps.
  Before, a loop that woke after both a timeout and the completion were due recorded the request as COMPLETED
  instead of TIMED_OUT followed by a late completion, and a late-observed completion freed the slot for an
  arrival that was due earlier. Deterministic C++ tests reproduce delayed observation with a simulated clock.

### Added (release hardening)

- Evaluator evidence check `completed_beyond_request_timeout` in Metric Set v1 evaluations (diagnostic only;
  never changes recorded states or metric values; not part of `metrics_content_sha256`).
- `make -C native portable` (fake runner + C++ unit tests, no CUDA) and `SDV_BMT_MAKE_TARGETS` for the test
  fixture; GitHub Actions CPU / portable workflow.
- `LICENSE` (Apache-2.0), `README.md`, `THIRD_PARTY_NOTICES.md`, release notes, public result summary in
  `docs/results/`.

### Changed

- Package version 0.1.0 (the internal pre-release package version was 0.2.0); `sdv-bmt --version` prints
  `sdv-bmt 0.1.0`. Evidence bundles recorded during development keep the internal version label of their time
  (`0.1.0` in Stage 1A bundles, `0.2.0` from Stage 1R on) and are not relabeled; the source-tree SHA-256 in
  `code_identity`, not the version string, identifies the code that produced a bundle.
- Public configs and tools no longer name host-local dataset locations (`$COCO_ROOT` placeholders).

### Not included in 0.1.0

NPU integration, Driving Scenario stage, hardware stress sweep, Scenario × Stress matrix, Autoware /
system-level evaluation, WCET or FTTI claims, ISO/PAS 8800 / ISO 26262 / SOTIF compliance claims, a canonical
calibration formula, energy per inference, a DRAM-bandwidth metric, AI error rate per hour or per km, TTC / PET
/ DST, SPFM / LFM / PMHF / FIT.
