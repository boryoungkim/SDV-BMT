# GPU reference results — CenterPoint-Pillars on nuScenes (unreleased)

> Research benchmark results on a **reference accelerator** (an NVIDIA GPU), not NPU results and not a GPU
> ranking. Percentiles characterize the measured distribution; `max_observed` is the largest observed value in
> that population, not WCET. Deadlines are evaluation parameters, not FTTI. Nothing here establishes ISO/PAS 8800,
> ISO 26262 or SOTIF compliance or vehicle-level safety.

This page summarizes the evidence behind the CenterPoint-Pillars workload on `main` (after v0.1.2, not yet
released): 3D object detection on nuScenes LiDAR. The raw evidence bundles are not part of the repository (they
contain host-local details and nuScenes-derived data), and no nuScenes point cloud, annotation or derived payload
is redistributed. Every number below was taken from the bundles and their bundle-only evaluations, recorded on
2026-11-15.

## Setup

| item | value |
|---|---|
| model | CenterPoint-Pillars, the mmdetection3d checkpoint `centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus` (pillar 0.2 m, SECOND + SECONDFPN, CenterHead with 6 tasks, circle NMS); re-defined in plain PyTorch, exported to ONNX with BatchNorm folded and converted offline to a cuDNN plan |
| measured path | pillar input → 3D detections. The device runs the pillar feature net, the BEV scatter, backbone, neck and head; decode, circle NMS and encoding (`centerpoint-det3d-f32/1`) run in the native output stage. The 10-sweep point-cloud merge and the hard voxelization (at most 20 points per pillar, at most 40,000 pillars) are built offline and are **outside** the measured path |
| data | nuScenes v1.0-mini, the four scenes of the official val split (162 samples); nuScenes val, all 150 scenes (6,019 samples) |
| accelerator | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, one GPU (GPU 0), `execution_kind = actual`, role `reference_accelerator_validation` |
| software | NVIDIA driver 580.119.02 (CUDA driver API 13.0), NVRTC 13.0, cuDNN 9.1.0 (pip `nvidia-cudnn-cu12` 9.1.0.70); FP32, batch 1; cuDNN math type default (whether TF32 was used is not observed); pinned algorithm table (all 92 convolutions `IMPLICIT_PRECOMP_GEMM`) |
| host | the reference host of v0.1.x: a **shared multi-user host**. Logged right before the timed runs: 1-min load average 7.2–10.5 before the five mini runs and 6.6 before the val series; GPU 0 had no compute process (2 MiB used) |
| admission | N = 1, queue 0, `drop_new`, request timeout 1 s, 32 untimed precondition requests per run; deadline = frame period |
| catalog | Model-level Metric Set v1.2 (`sdv-bmt-model-metric-set-v1.2`): v1.1 plus the nuScenes 3D-detection metrics; quality evaluator `nuscenes-det3d-cvpr2019/1` on the ground truth frozen in each bundle |
| code identity | Python source tree `85e52d0b…`, native source tree `ea4e4ecb…` |

## 1. Numerical checks (all passed)

The criteria were fixed in the plan before the measurements and were not changed after seeing results.

| check | criterion | result |
|---|---|---|
| input builder vs official mmdetection3d 1.4.0 pipeline (merged points, pillars, points per pillar, pillar count, pillar coordinates) | byte-identical | all 6,019 val samples identical (the 162 mini samples too) |
| plain-PyTorch re-definition vs official model (head outputs) | ≤ 1e-4 | 0.0 |
| ONNX export: ONNX Runtime 1.19.2 vs PyTorch | max relative difference ≤ 1e-4 | 2.2e-5 |
| GPU vs ONNX Runtime: heatmap / regression outputs | ≤ 1e-4 / ≤ 1e-3 | 3.7e-5 / 9.4e-4 |
| GPU vs ONNX Runtime: mAP / NDS on the same samples | difference ≤ 1e-3 | 3.9e-7 / 3.8e-7 |
| native output stage vs Python twin | byte-identical | 324 / 324 real outputs; 162 / 162 golden-capture outputs |
| SDV-BMT evaluator vs nuscenes-devkit 1.2.0 (official boxes) | ≤ 1e-4 | 0.0 on mini and on all of val, per-class AP included |
| prediction-to-global conversion vs mmdetection3d | — | position 2.7e-6 m, size 9.4e-7 m, yaw 1.2e-7 rad; no count or attribute mismatch |
| final quality: GPU vs official (same samples) | ≤ 0.005 | mini 4.9e-7 / 8.3e-7, val 1.2e-5 / 6.2e-6 (mAP / NDS) |
| GPU kernels (pillar features, scatter, depth-to-space) device tests | — | bit-identical to the float32 reference |
| cuDNN algorithm table: heuristic probe vs pinned table | equal | equal |

The worst regression ratio (9.4e-4) is the velocity channel of the barrier class, whose values are tiny (largest
absolute value about 0.001), so the relative ratio is large; the absolute difference is of the order of 1e-6.

## 2. Replay rate

Rule (from the CenterPoint plan): the highest of 20, 10, 5, 2 and 1 Hz whose frame period `floor(1e9 / rate)` is
at least 1.25 × the largest DUT-service of the mini golden capture. The largest was 10.98 ms (required period
13.73 ms), so the runs use **20 Hz** (deadline 50 ms). The candidates stop at 20 Hz, the LiDAR rotation rate, so
that the input arrives at the rate the sensor produces it; with the DeepLab candidate list (up to 30 Hz) the same
rule would give 30 Hz.

## 3. Latency (ms) and timeliness

### 3.1 nuScenes mini, the four official-val scenes (162 samples), five runs

| item | five-run range (min–max) |
|---|---|
| completed / dropped / timed out / errors | 162 / 0 / 0 / 0 in every run |
| deadline misses (completion / output ready) | 0 / 0 |
| DUT-service p50 / p99 / max_observed | 10.42–10.45 / 10.51–10.59 / 10.54–10.64 |
| device execution bracket p50 / p99 | 9.78–9.81 / 9.84–9.88 |
| host-to-device copy p50 (16.48 MB input) | 0.53–0.55 |
| output stage (decode + circle NMS + encode) p50 / max_observed | 2.48–2.81 / 2.91–4.42 |
| output ready p99 / max_observed | 13.3–14.4 / 13.4–14.9 |
| achieved completion rate (1/s) | 20.091–20.092 |
| output identity against the golden capture | 1.0 in every run |

### 3.2 nuScenes val (6,019 samples), one run series

The val set does not fit one run's pinned input staging (16,480,040-byte payload per sample), so it runs as a
**run series**: consecutive ordinary runs (parts) of at most 1,000 samples in index order that never split a scene,
evaluated as one result. The seven parts hold 993, 989, 969, 968, 973, 966 and 161 samples; all completed and
were valid.

| item | value |
|---|---|
| completed / dropped / timed out / errors | 6,019 / 0 / 0 / 0 |
| deadline misses (completion / output ready) | 0 / 0 |
| DUT-service p50 / p99 / max_observed | 10.34 / 10.71 / 10.79 |
| device execution bracket p50 / p99 / max_observed | 9.79 / 9.86 / 9.89 |
| host-to-device copy p50 | 0.51 |
| output stage p50 / p99 / max_observed | 2.50 / 5.31 / 8.99 |
| output ready p99 / max_observed | 15.59 / 19.36 |
| offered / achieved completion rate (1/s) | 20.0 / 20.017, over the sum of the parts' measurement windows |

- Distributions are over all 6,019 requests together, not averages of per-part values.
- Between the end of one part's measurement and the start of the next, 200–240 s passed (52 s before the last,
  short part): the next part reads, SHA-256-checks and stages its inputs. The waits are in no measurement window.
  These gaps were computed afterwards, read-only, from the part bundles; the `series.json` of this series was
  written before the measurement windows were recorded there, and it was not modified.
- The output-stage tail on val (p99 5.31 ms, max 8.99 ms) is longer than on mini (max 2.91–4.42 ms). The cause
  (number of candidates per scene, host load) was not separated. The longest output-ready time, 19.36 ms, stays
  below half of the 50 ms deadline.

## 4. Task quality

| scope | SDV-BMT (GPU) | official mmdetection3d (CPU, same samples, nine nearest sweeps) | difference |
|---|---|---|---|
| mini, 162 samples | mAP 0.536228 / NDS 0.585987 | mAP 0.536228 / NDS 0.585986 | 4.9e-7 / 8.3e-7 |
| val, 6,019 samples | mAP 0.480113 / NDS 0.584944 | mAP 0.480101 / NDS 0.584938 | 1.2e-5 / 6.2e-6 |
| reference: model zoo | | mAP 48.70 / NDS 59.62 | official re-run − zoo: −0.69 / −1.13 points |

- val errors: mATE 0.326 m, mASE 0.263, mAOE 0.362 rad, mAVE 0.402 m/s, mAAE 0.199. Per-class AP: car 0.835,
  truck 0.480, construction_vehicle 0.109, bus 0.604, trailer 0.325, barrier 0.601, motorcycle 0.419,
  bicycle 0.144, pedestrian 0.756, traffic_cone 0.527.
- Coverage is 1.0 in every run. The five mini runs give identical quality (their outputs are byte-identical to the
  golden capture).
- The mini and val scores are not compared with each other: the four mini scenes have a skewed class mix, and
  mini has no construction_vehicle ground truth (that class AP is 0 by the devkit rule).
- The gap to the model zoo appears already in the official implementation's own re-run, so it is not an SDV-BMT
  implementation difference. One known cause: the official configuration draws nine of the ten sweeps at random;
  SDV-BMT and the official reference run here both use the nine nearest, for reproducibility (on mini, the random
  draw lowered mAP by 0.011). The environment and versions behind the published values were not checked.
- The 162 mini samples are also part of val. Inside the val series their outputs equal the mini golden capture,
  except for the 4-byte item position in each output header.

## 5. Notes

- Eight val samples reach the 40,000-pillar limit; the official pipeline cuts them the same way, and they are
  included in the byte-identity check of the inputs. A port to another device must keep the same rule for the
  pillars beyond the limit.
- A NaN heatmap value or a detection value that does not fit float32 is an output error, by the same rule in the
  native output stage and in the Python twin.
- The output stage returns boxes in the LiDAR frame; the conversion to the global frame and the attributes are
  done by the evaluator from the poses frozen in the bundle.

## 6. Limitations

- One reference GPU, FP32, batch 1, N = 1, one replay rate; no NPU.
- The voxelization and the multi-sweep merge are outside the measured path; the timings cover pillar input →
  detections only. Memory traffic and bandwidth are not observed.
- Mini: five runs, reported as min–max. Val: one run series. Timings are not compared across days.
- The host is shared; other users' load is logged only before runs, and host preemption is not observed.
- The cuDNN math type is the default, which permits TF32 on this GPU; the numerical checks show no visible
  effect, but TF32 use is not observed.
- Probe and capture runs ran in asap mode; only the fixed-rate runs are used for timing.
- The runs were not repeated with the code now on `main`. The native source tree is the same; the Python source
  tree changed afterwards (among others, a run series now records each part's measurement window in
  `series.json`, and the LLM path was added). `code_identity` in each bundle names the code that produced it.
