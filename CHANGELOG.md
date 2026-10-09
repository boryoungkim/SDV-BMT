# Changelog

All notable changes to SDV-BMT are recorded here. Versions follow semantic versioning.

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
- `LICENSE` (Apache-2.0), `README.md`, `THIRD_PARTY_NOTICES.md`, `CITATION.cff` (author, release date,
  license), release notes, public result summary in `docs/results/`.

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
