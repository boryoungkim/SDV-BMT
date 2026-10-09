# Measurement Protocol (Stage 1A virtual clock, Stage 1R-G realtime)

> **1–9절은 Stage 1A(MOCK)다.** 그 시간은 모두 deterministic virtual run clock 값이다. realtime replay fidelity, issue-lag floor, NPU 실행 시간을 의미하지 않는다.
>
> **10절은 Stage 1R-G realtime 경로**(host `CLOCK_MONOTONIC`, native measurement plane)다. GPU 실행 결과는 reference accelerator 검증이며 NPU 결과가 아니다.

## 1. Clock과 timestamp

| 기호 | 의미 (Stage 1A) |
|---|---|
| run clock | `run_clock_domain=virtual`, 정수 ns. 원점(0)은 run lifecycle 시작(bundle open) |
| `t_src` | `SourceTiming.value × unit_to_ns` (source domain, `ns/us/ms/s`, 정수만 허용) |
| `t_sched` | 예정 도착 시각 |
| `t_issue` | admission 결과가 확정된 시각. Stage 1A에서는 `t_sched`와 같다 |
| `t_submit_begin` | blocking `run()` 진입. `t_issue`와 같다(dispatch lag 0) |
| `t_submit_end` | `run()` 반환 = `t_submit_begin + virtual_service_ns` |
| `t_complete` | Runner가 completion을 관측한 시각 = DUT return 시각 |
| `t_output_ready` | postprocess+encode+기록이 끝난 시각. host 단계 virtual cost가 0이므로 `t_complete`와 같다 |

- Stress verify window(`verification.window_ns`)는 MEASURE 전에 virtual clock을 그만큼 진행시킨다. NoStress도 같은 window를 기다리므로 nominal/stress run의 timeline이 같다.
- MEASURE 시작 시각을 `t_measure_begin`, 첫 도착 기준을 `t0 = t_measure_begin + lead_ns`로 둔다.

## 2. Replay schedule (design v03 E.4)

| mode | `t_sched` | 허용 input |
|---|---|---|
| `fixed_rate` | `t0 + floor(i × 10^9 / rate_hz)` | dataset, driving_scenario |
| `recorded` | `t0 + base_e + floor((t_src,i − t_src,e0) / speed)` | driving_scenario만 |
| `asap` | slot이 비는 시각 (`t_sched = t_issue`) | dataset, driving_scenario |

- `rate_hz`, `speed`는 최단 decimal 표현의 정확한 값(`Fraction(repr(x))`)으로 계산한다.
- recorded: `e0`는 item이 속한 episode의 첫 item, `base_0 = 0`, `base_{e+1} = (episode e 마지막 offset) + episode_gap_ns`. gap은 run-clock 시간이며 speed로 나누지 않는다. 원본 timestamp는 수정하지 않는다.
- recorded는 모든 item에 SourceTiming이 있어야 하고, 단일 clock source여야 하며, episode가 둘 이상이면 `episode_gap_ns`가 필요하다.
- episode 안에서 timestamp가 역행하거나 중복되면 거절한다. 조용히 정렬하지 않는다.
- Dataset + recorded는 VALIDATE에서 거절한다.

## 3. Admission과 request state (N=1, queue 0, `drop_new`)

| 현재 | 사건 | 다음 | event |
|---|---|---|---|
| PLANNED | item 준비 실패 | NOT_ISSUED | `not_issued(reason=prepare_failed:<stage>)` |
| PLANNED | slot 사용 중 | DROPPED | `dropped(reason=no_free_slot)` |
| PLANNED | slot 비어 있음 | ISSUED | `issued(t_sched, t_issue)` |
| ISSUED | `run()` 진입 | INFLIGHT | `submit_begin` |
| INFLIGHT | DUT error 반환 / adapter 예외 | ERROR | `submit_end`, `error(error_phase=dut_run)` |
| INFLIGHT | 정상 반환, postprocess/encode 실패 | ERROR | `submit_end`, `error(error_phase=postprocess|encode)` |
| INFLIGHT | 정상 반환 | COMPLETED | `submit_end`, `completed(t_complete, t_output_ready, output_ref)` |
| INFLIGHT | timeout이 먼저 | TIMED_OUT | `timed_out(timeout_ns)` |
| TIMED_OUT | 이후 반환 | 변화 없음 | `submit_end`, `late_completion(result=ok|error)` |

- timeout timer는 `t_submit_begin + request_timeout_ns`.
- TIMED_OUT 후에도 slot은 underlying call이 반환될 때까지 점유된다. late completion의 output도 postprocess·encode·저장한 뒤 slot을 release한다.
- request error 정책은 `continue` 하나다. adapter 예외도 request-level ERROR다.
- MEASURE 도중 runner 내부 불변식 위반 같은 예상 밖 예외는 `measure_aborted`로 run을 실패시키고 cleanup을 수행한다(measurement invalid).

**동시각 규칙**

1. 같은 시각의 DUT return이 timeout보다 먼저 처리된다. timeout 시각에 정확히 반환되면 COMPLETED다.
2. 같은 시각 `t`에 slot이 비면 `t`에 도착한 request는 그 slot을 쓸 수 있다.

**계수 항등식** (Runner FINALIZE와 Evaluator가 각각 검사)

```text
planned = not_issued + dropped + issued
issued  = completed + error + timed_out
```

late completion은 계수를 바꾸지 않는다.

## 4. Drain

- 마지막 admission 시각 `t_last_sched` 이후 `t_last_sched + drain_ns`까지 virtual event를 처리한다.
- VALIDATE는 `drain_ns ≥ request_timeout_ns`를 요구한다. 따라서 issued request는 모두 drain 안에서 terminal state가 된다.
- cap까지 반환되지 않은 timed-out call은 `drain end` event의 `outstanding_request_ids`와 `cap_reached=true`로 남는다. late completion은 관측되지 않은 것으로 기록된다.
- drain end 시각 = 큐가 빈 마지막 event 시각, 또는 cap.

## 5. Measurement boundary

Runner가 adapter 선언(`submit`/`complete` 의미)과 workload 선언(`output_ready` 의미)으로 `MeasurementBoundary`를 만들고, 그 canonical JSON SHA-256을 `measurement_boundary_id`로 기록한다.

| 구간 | Stage 1A |
|---|---|
| payload read/decode | excluded (MEASURE 전 eager) |
| Workload preprocess | excluded |
| `prepare_input` / H2D | excluded |
| submit / complete | virtual FakeDUT `run()` 진입 / `t_submit_begin + service` 반환 |
| Workload postprocess | `output_ready_only` (request-completion·DUT-service에는 미포함) |
| host step virtual cost | 0 ns |
| input pipeline owner | `workload` 또는 `dut_adapter` (`plan.input_pipeline.mode`) |
| preprocess semantics | Workload contract의 `preprocess_semantics_id` |
| fused steps | `workload` mode는 없음, override는 adapter 선언(`workload.preprocess`, `dut.prepare_input` 중) |
| input pipeline digest | `InputPipelineRecord` 전체의 digest (implementation id/version/digest 포함) |

boundary schema는 v2다(v1에 pipeline 항목 추가). 두 mode 모두 Stage 1A에서는 준비가 MEASURE 전에 끝나므로 timing은 같다. 그래도 boundary identity가 pipeline owner, semantics, fused steps, pipeline digest를 포함하므로 mode나 override 구현이 다르면 `measurement_boundary_id`가 다르다.

latency 정의: request-completion `t_complete − t_sched`, DUT-service `t_complete − t_submit_begin`, output-ready `t_output_ready − t_sched`. `t_complete − t_sched`를 NPU latency라고 부르지 않는다.

## 6. Deadline (evaluation parameter)

- `deadline: {policy: frame_period, period_ns: P}` → `deadline_i = t_sched_i + P`.
- miss = planned request 중 `COMPLETED`이면서 `t_complete ≤ deadline_i`가 아닌 것. 분모는 planned.
- Deadline은 execution state를 바꾸지 않는다. `sdv-bmt evaluate --eval-config`로 다른 값을 같은 raw bundle에 적용할 수 있다.

## 7. Stress timing과 상태

- 순서: `check`(PREFLIGHT, t=0 기록) → `prepare` → `activate` → `verify(window)` → `observe`(MEASURE 시작, DRAIN 끝) → `deactivate` → `restore`.
- `verification`: NoStress `not_applicable`; FakeStress 성공 시 `self_reported`(spec이 independent evidence를 요구하면 `unverified`); 실패 시 `failed`.
- `coverage`: nominal `not_applicable`; MEASURE가 없으면 `none`; `verify_end ≤ t_measure_begin`이고 `deactivate ≥ t_drain_end`이며 observe가 모두 ok면 `full`, observe 실패가 있으면 `partial`.
- provider가 예외를 던지거나 기록할 수 없는 record를 돌려주면 그 단계는 실패 record로 남는다. observe 실패는 run을 실패시키지 않고 coverage만 `partial`로 낮춘다.
- `nominal_check`: Stage 1A는 장치 idle 확인 수단이 없으므로 항상 `not_performed`.

## 8. 결정성

- 같은 plan/seed는 byte 단위로 같은 `run_spec.json`, `input_items.jsonl`, `events.jsonl`, `stress.jsonl`, `outputs.bin`을 만든다(run ID는 이 파일들에 들어가지 않는다).
- Stage 1A 구성요소는 난수를 쓰지 않는다. `seed`는 plan identity로 기록만 한다.
- wall-clock 시각은 `created_at_utc`(volatile) 기록에만 쓴다.

## 9. Realized conditions and contrast validity

> **Control what is planned. Verify what is realized. Localize what changed. Do not attribute beyond the evidence.**
>
> **A configured condition and a realized condition are not assumed to be identical.**
>
> **Failure to establish factor isolation is not a benchmark failure; it is evidence that the requested attribution is not justified.**

A configured condition and a realized condition are not assumed to be identical. 같은 설정이어도 실제 입력 payload, input residency, buffer 할당/재사용 상태, H2D/DMA 동작, runtime queue 상태, device 주파수, 온도/throttling, 다른 장치 사용자, 실제 issue timing, memory 압력/대역폭, 실제 stress는 run마다 다를 수 있다.

**manifest `realization` section (schema v1)**: 조건마다 `RealizationEvidence` 하나를 기록한다.

| field | 의미 |
|---|---|
| `condition` | `input_payload`, `input_preparation`, `input_residency`, `buffer_state`, `data_movement`, `runtime_queue_state`, `device_frequency`, `thermal_state`, `competing_device_users`, `issue_timing`, `memory_pressure_bandwidth`, `stress_realization` |
| `validity` | `observed`(실제 대상에서 측정) / `self_reported`(구성요소의 자기 보고) / `virtual`(mock·virtual model의 산물, 실제 하드웨어에 대해 아무것도 말하지 않음) / `not_observable` / `not_applicable` |
| `source`, `semantic`, `unit`, `scope`, `clock` | 값을 읽는 방법. 이 중 하나라도 다르면 값을 비교하지 않는다(`unresolved`) |
| `value` | 조건 값(JSON). `not_observable`이면 없다. `observed`/`self_reported`/`virtual`이면 반드시 있어야 한다 |
| `equivalence` | 두 값을 비교하는 규칙. Stage 1A는 `exact`(canonical JSON 동일성. `true`, `1`, `1.0`은 서로 다르다)만 지원한다. 다른 규칙은 `unresolved`(`unsupported_equivalence_rule`)다. 연속값(온도, 달성 stress 등)의 tolerance 규칙은 이후 단계에서 결정한다 |

**Stage 1A가 기록하는 것** (모든 run은 mock):

| 조건 | validity | 내용 |
|---|---|---|
| `input_payload` | `observed` | `INPUT.open`이 source index를 다시 읽어 SHA-256과 실현된 items digest를 재검증한 결과(실제 fixture byte) |
| `input_preparation` | `virtual`(두 pipeline mode 모두) / `not_observable`(PREPARE_INPUTS 미완료) | 선택된 pipeline의 item별 결과다. `pipeline_digest`, `failed_items`(item index와 실패 단계. 실패한 item은 NOT_ISSUED가 된다), `equivalence`(workload mode는 `not_applicable`, adapter_override는 reference path 대비 결과). 계획한 item에 대한 상대 정보만 담으므로 episode가 달라도 비교할 수 있다 |
| `issue_timing` | `virtual` | virtual clock에서는 구성상 issue lag = 0 |
| `stress_realization` | `not_observable`(nominal: idle 확인 없음) / `virtual`(FakeStress 자기 보고) | |
| 나머지 8개(device/host 조건) | `not_observable` | 장치 없음 |

Realization Guard는 mock이 포함된 쌍(네 곳의 `execution_kind` 중 하나라도 `actual`이 아닌 쌍)에 대해 `not_observable`을 반환한다. deterministic FakeDUT가 같은 결과를 내더라도 `pass`라고 보고하지 않는다. 반대로 기록된 evidence에 차이가 있으면 mock이라도 `fail`이다. 예를 들어 같은 stress level인데 coverage가 다르거나, 한 run에서 일부 item의 준비가 실패했으면 `fail`이다. 차이는 귀속을 무효화하지만, 같다는 것만으로는 귀속을 정당화하지 못한다.

`pass`의 조건(이후 단계의 actual run에 적용):
- 12개 조건 모두 양쪽에서 값이 있는 `observed`이고, 같은 `source`/`semantic`/`unit`/`scope`/`clock`/`equivalence`로 같은 값이어야 한다.
- `not_applicable`은 동등성의 증거가 아니다. 장치에 해당 조건이 없음을 선언하는 방법은 아직 없으며 이후 단계의 설계 결정이다.
- 의도한 factor를 실현하는 조건은 run 간 동일성 비교에서만 빠진다. 각 run은 그 factor가 계획대로 실현되었음을 보여야 한다:
  - stress: `verification: verified`이고 `coverage: full`.
  - nominal: `nominal_check: passed`.
  - payload: 실현된 items digest가 계획과 같음.
- `verification: failed`, `coverage: none`, 계획과 다른 payload는 `fail`이다. 확인되지 않은 경우(자기 보고, 부분 coverage, 관측 불가)는 `unresolved`다.

## 10. Stage 1R-G realtime protocol (host monotonic clock, native measurement plane)

Stage 1A의 의미(계획 request, 상태 모델, request identity, N=1, queue 0, `drop_new`, timeout/drain 정의, append-only raw evidence, bundle-only 평가, 세 guard)를 그대로 쓴다. 실제 실행 때문에 바뀌는 것은 "어디서 시계를 읽는가"뿐이다.

### 10.1 Timestamp mapping

run clock = host `CLOCK_MONOTONIC` − run anchor(Python이 run 시작 때 읽고 native process에 넘김). Python lifecycle event와 native request event가 한 run clock 위에 있다.

| SDV-BMT 시각 | native capture point | CUDA 의미 (NVIDIA backend) | profiler 교차 검증 |
|---|---|---|---|
| `t_sched` | 시계를 읽지 않는다. `t0 + offset_i`(Python이 계산한 계획 값, `t0 = t_measure_begin + lead_ns`). `asap`이면 slot이 빈 뒤 읽은 시각 | 없음 | 없음 |
| `t_issue` | issuer loop가 `now ≥ t_sched`를 확인한 뒤 admission을 결정할 때 읽은 시각(DROPPED/NOT_ISSUED도 같다) | 없음(CUDA 호출 전) | drop/not-issued는 NVTX mark |
| `t_submit_begin` | admission 뒤, 그 request의 첫 CUDA 호출 직전 | `cuEventRecord(start)` 또는 `cuMemcpyHtoDAsync` 직전 | nsys 첫 API 호출 시작(정렬 anchor) |
| `t_submit_end` | 마지막 enqueue 호출(`cuEventRecord(done)`)이 반환된 뒤 | 모든 작업이 stream에 들어감. 완료를 뜻하지 않는다 | nsys API 구간 끝(포함 관계) |
| `t_complete` | `cuEventQuery(done)`이 처음 `CUDA_SUCCESS`를 반환한 직후 | D2H 뒤 event 완료 = 이 request의 stream 작업 전부 완료 | 성공 poll 끝, D2H activity 끝 |
| `t_output_ready` | 출력 사본 → native 출력 pipeline(postprocess + codec) → 출력 record append 뒤 | 없음(host 단계) | 없음 |

- latency 정의는 Stage 1A와 같다: issue lag `t_issue − t_sched`, request-completion `t_complete − t_sched`, DUT-service `t_complete − t_submit_begin`, output-ready `t_output_ready − t_sched`.
- **DUT-service는 순수 GPU 실행 시간이 아니다.** 제출 호출, 제출 뒤 장치 시작까지의 대기, 장치 작업, 완료 관측 지연을 모두 포함한다.
- issue lag는 측정값이다(Stage 1A에서는 구성상 0). issuer가 계획을 지키지 못한 경우 GPU 지연으로 표시하지 않는다.

### 10.2 Admission, timeout, drain (실제 시간)

- issuer는 단일 thread이며 MEASURE/DRAIN 안에서 잠들지 않는다(spin). process는 `realtime.cpu_affinity`에 고정되고, 실현된 affinity가 요청과 다르면 run이 실패한다(`cpu_affinity_not_realized`).
- **사건 순서(v0.1.0)**: loop가 처리할 때 이미 도래한 사건은 의미상 시각 순서로 결정한다 — 도착은 `t_sched`, timeout은 `t_submit_begin + request_timeout_ns`, 완료는 `t_complete`(완료를 보고한 첫 poll 직후 읽은 시각). 이른 시각이 먼저다. Stage 1A 동시각 규칙(완료 → timeout → 도착)은 시각이 같을 때만 쓴다. 따라서 loop가 늦게 깨어 timeout 도래 뒤에 완료를 관측하면 그 request는 TIMED_OUT이고 그 완료는 `late_completion`이다. slot을 점유한 request의 완료가 다음 도착의 `t_sched`보다 늦게 관측되면 그 도착은 slot이 점유된 상태에서 처리된다(`drop_new`이면 DROPPED). 한 시각 읽기까지 도래한 timeout과 도착은 다음 poll보다 먼저 처리한다(그 poll이 관측하는 완료는 더 늦기 때문). poll 도중에 도래한 사건도 관측된 완료보다 이르면 먼저다.
- 기록 시각: `t_issue`, `t_complete`, `t_output_ready`는 실제로 읽은 시각이다. `timed_out` 기록은 timeout 도래 시각(`t_submit_begin + timeout_ns`)을 쓰되, 그보다 늦은 기록이 이미 있으면 그 시각을 쓴다(기록 시각은 감소하지 않는다).
- 이 규칙 이전(v0.1.0 전) native loop는 한 반복 안에서 완료를 먼저 확인했다. loop가 timeout 도래 전부터 완료 뒤까지 멈추면 `t_complete − t_submit_begin > request_timeout_ns`인 COMPLETED가 기록될 수 있었다. 평가기의 진단 `completed_beyond_request_timeout`(`docs/metrics_catalog.md` 8.12절)이 그런 기존 bundle을 표시한다.
- timeout timer는 `t_submit_begin + request_timeout_ns`. TIMED_OUT 뒤에도 slot은 장치 작업이 끝날 때까지 점유된다(제출된 장치 작업은 취소할 수 없다). 그 완료는 `late_completion`이다.
- DRAIN cap = (issue loop가 끝난 시각) + `drain_ns`. cap 이후 첫 반복까지 처리한다(inclusive). 따라서 `drain_ns ≥ request_timeout_ns`이면 issued request는 모두 terminal state가 된다. cap에서 아직 장치 작업이 남은 TIMED_OUT request는 `outstanding_request_ids`에 남는다.
- precondition(`realtime.precondition_requests`): STRESS.verify 뒤, MEASURE 전에 실행하는 untimed request다. 음수 request id를 쓰고 request event를 만들지 않으며 `precondition` phase에 요약만 남긴다. 계획 request가 아니다. warm/cold 비교에서 이 값은 design element(`realtime_harness`)다.

### 10.3 Realtime measurement boundary (schema v3)

Stage 1A boundary(v2)와 identity를 바꾸지 않도록 별도 record(`RealtimeMeasurementBoundary`)다.

| 항목 | 값 |
|---|---|
| `clock_domain` | `monotonic` |
| payload read/decode, preprocess | excluded(MEASURE 전 eager) |
| `prepare_input_h2d` | adapter 선언. `host_pinned`: `included`(request마다 3 MiB H2D가 DUT-service 안). `device_resident`: `excluded` |
| `submit` / `complete` | adapter의 submit/complete 의미 |
| `completion_observation` | native runner가 backend를 spin-poll; `t_complete` = 완료를 보고한 첫 poll |
| `device_timing` | adapter 선언. NVIDIA `nvidia:cuda_event_pair:bracket`(segment `host_to_device`/`device_execution`/`device_to_host`, 장치 clock duration만), 계측 `none`이면 null |
| `instrumentation` | 계측 profile 전체(event bracket, NVTX, profiler, residency, buffer policy 등) |
| `output_pipeline` | 측정 경로 안에서 출력을 만든 구현(예: `kind: reference_convnet_v1`, `implementation: native:native/src/common/reference_output.hpp`, postprocess/codec id) |
| input pipeline 항목 | v2와 같다 |

계측, residency, buffer policy, profiler가 다르면 boundary ID와 `telemetry` component가 다르다. 그런 run은 같은 측정으로 비교되지 않는다(Measurement Guard `fail`).

### 10.4 Clock domain 규칙

| domain | 시계 | 쓰는 곳 | 다른 domain과의 관계 |
|---|---|---|---|
| run clock | host `CLOCK_MONOTONIC` − anchor | `t_sched`…`t_output_ready`, lifecycle event | Python과 native가 anchor 공유 |
| 장치 event timer | CUDA event(이 장치에서 32 ns 단위) | segment duration | duration만. host 시각과 빼지 않는다 |
| `%globaltimer`, `clock64` | 장치 ns, SM cycle | kernel 본문 구간, 실효 SM 주파수(Python이 계산) | duration과 비율만 |
| nsys session | `CLOCK_MONOTONIC` + 상수 | 교차 검증 | run마다 offset을 다시 추정하고 포함 관계로 검증 |
| nsys 장치 activity | CUPTI가 GPU 시각을 CPU 시각으로 변환 | 교차 검증 | 포함 관계로만 검증(오차 상한 미증명) |
| NVML 표본 | `CLOCK_REALTIME` µs | window 경계 snapshot, process utilization 표본 | MEASURE 경계 문맥만 |

- 장치 domain 시각을 host 시각에서 빼지 않는다. host duration과 장치 duration은 나란히 보여 줄 뿐 서로 빼서 "측정값"을 만들지 않는다.
- event bracket은 연산 자체보다 길다(event 기록과 queue 대기 포함). bracket 값을 "순수 장치 실행 시간"이라 부르지 않는다. 또 `device_resident`의 `device_execution` bracket은 start event부터(launch 지연 포함), `host_pinned`는 H2D 완료 event부터 잰다. residency가 다르면 boundary가 다르므로 서로 비교하지 않는다.

### 10.5 분리할 수 없는 경계 (benchmark run)

| 구간 | 상태 |
|---|---|
| 제출 → 장치 시작 대기(driver/queue) | CUDA event만으로 분리 불가. profiler run에서만 보인다 |
| 완료 관측 지연(장치 완료 → host 관측) | 분리 불가. profiler run에서만 보인다 |
| event bracket 자체의 overhead | bracket과 연산을 event만으로 나눌 수 없다 |
| host CPU 주파수, 실제 실행 core, 다른 host process의 선점 | 관측하지 않는다 |
| 다른 process의 GPU 작업 | nsys process-tree trace에 없다. NVML window 경계 표본으로만 관측한다 |

### 10.6 Harness floor와 계측 overhead

- harness floor는 fake backend(장치 작업 없음, 첫 poll에 완료)로 잰다. 결과는 harness 자신의 DUT-service와 issue lag다.
- 계측 profile(event bracket on/off, NVTX, profiler)은 measurement identity다. 비교는 profile별 run을 번갈아 실행해 run 간 변동과 함께 본다.
- profiler run은 별도 measurement 구성이다. profiler는 submit, DUT-service, issue lag, 완료 관측, admission 결과까지 바꿀 수 있다. profiler run의 숫자를 profiler 없는 run의 값으로 쓰지 않는다. 교차 검증은 같은 사건 사이의 상대 관계(포함, identity, duration 차이)에만 쓴다.

### 10.7 nsys 교차 검증 절차 (`tools/stage1r_g/nsys_align_bundle.py`)

1. DUT config `nvtx: true, profiler: nsys`인 plan을 `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-event-trace=false`로 실행한다(200 request 이하).
2. SQLite에서 교차 검증 record만 추출하고(`tools/stage2a_g/nsys_extract.py`; not-ready poll은 요약만), SHA-256을 기록한 뒤 SQLite를 지운다.
3. 시계: request마다 첫 CUDA API 호출(nsys) − host `t_submit_begin`의 최솟값을 offset으로 쓴다. 앞뒤 10% request의 offset으로 drift를 확인한다.
4. identity: NVTX payload mapping과 host submit bracket mapping이 서로 같아야 한다. 순서 mapping은 쓰지 않는다. drop/not-issued mark id 집합은 bundle의 DROPPED/NOT_ISSUED id 집합과 같아야 한다.
5. 포함 관계: API 호출은 host bracket 안, 장치 activity는 enqueue 호출 뒤, D2H 끝 ≤ 성공 poll 끝 ≤ host `t_complete`. 하나라도 어기면 mapping을 쓰지 않는다.

### 10.8 Actual run의 realization evidence

| 조건 | validity | 근거 |
|---|---|---|
| `input_payload` | observed, exact | INPUT.open의 index·items digest 재검증 |
| `input_preparation` | observed(actual) / virtual(mock), exact | Workload 전처리 + native staging의 item별 SHA-256 재검증 결과 |
| `issue_timing` | observed, `unapproved:replay_fidelity_rule` | issued event의 issue lag 분위수. 승인된 동등성 규칙이 없으므로 동등성을 세우지 못한다 |
| `stress_realization` | not_observable | nominal: 간섭 없음 확인(`nominal_check`)을 하지 않는다 |
| `input_residency` | self_reported, exact | native runner 자신의 copy 호출(독립 trace 아님) |
| `buffer_state` | self_reported, exact | buffer policy와 MEASURE 중 할당 수(native 계수) |
| `data_movement` | observed, exact | window 양 끝의 PCIe link gen/width, PCIe replay counter 변화(NVML) |
| `runtime_queue_state` | observed, exact | `measure_begin`에서 request stream이 비었는지(`cuStreamQuery`) |
| `device_frequency` | observed, `unapproved:steady_state_band` | kernel 안 clock64/%globaltimer로 계산한 실효 SM MHz 분위수, 경계 P-state, throttle 시간 |
| `thermal_state` | observed, `unapproved:steady_state_band` | 경계 온도와 clocks event reason(thermal throttle counter는 이 장치에서 미지원) |
| `competing_device_users` | observed, exact | 경계의 다른 compute process PID, window 동안 다른 process의 0이 아닌 NVML utilization 표본 수 |
| `memory_pressure_bandwidth` | not_observable | DRAM 대역폭·traffic은 GPU 성능 counter가 필요하고 이 host는 `RmProfilingAdminOnly=1` |

- snapshot이 실패하면 해당 장치 조건은 `not_observable`이다. 값이 없는 관측을 비교하지 않는다.
- 연속값 조건(issue timing, 주파수, 온도)에는 승인된 equivalence 규칙이 없다. `unapproved:<후보 규칙>`으로 기록하고 guard는 `unresolved`로 처리한다. ±X%, ±Y°C 같은 임의 threshold를 만들지 않는다.

## 11. Stage 2b-G real-model path (YOLOX-S, cuDNN plan)

10절의 의미를 그대로 쓴다. 아래는 실제 model 경로에서 달라지는 capture point의 내용과 추가 사항이다.

### 11.1 Request 하나의 장치 작업 순서

`[event start] → H2D(4,915,200 B, pinned → device) → [event h2d] → frequency probe(1 thread, 20,000 SM cycle) → plan step 순서대로(cuDNN convolution 83 + bias/activation glue 83, cuDNN max pooling 3, space-to-depth 1, upsample 2, head transpose 3) → [event kernel] → D2H(2,856,032 B: raw 출력 + probe word 32 B) → [event done]`. 모두 한 non-blocking stream이다.

- `t_submit_begin`은 첫 enqueue 직전, `t_submit_end`는 마지막 enqueue(`cuEventRecord(done)`)가 반환된 뒤다. 따라서 submit 구간에는 cuDNN 호출의 host 쪽 처리와 glue kernel launch 전부가 들어간다.
- `device_execution` segment(event h2d → event kernel)는 probe와 모든 kernel을 덮는다. host가 아직 enqueue하는 동안 장치가 기다린 시간도 들어간다. 이 segment는 장치 busy 시간이 아니다. busy와 idle의 분리는 profiler run(11.4)에서만 보인다.
- `t_output_ready`: raw 출력(pinned buffer의 view) → native YOLOX 출력 pipeline(decode, threshold, class-aware NMS, top-100, 원본 좌표, canonical JSON) → 출력 record append. 이 단계의 비용은 detection 수에 따라 달라진다.

### 11.2 Frequency probe

`device_frequency` evidence의 실효 SM clock은 각 request 장치 작업 맨 앞의 probe kernel이 잰다(clock64 cycle / %globaltimer ns, 한 SM). model kernel 안의 clock이 아니다. probe는 계측이므로 measurement declaration(`instrumentation.in_kernel_timing`)에 들어가고, probe 시간은 `device_execution` segment 안에 있다.

### 11.3 Output reference와 품질

- cuDNN 실행은 독립 CPU 구현과 bit 단위로 같을 수 없으므로 output identity의 reference는 같은 code·구성으로 미리 실행한 reference-capture run의 출력이다. identity는 결정성과 손상만 진단한다. 정확성은 ONNX Runtime과의 수치 비교(도구)로, task 품질은 COCO annotation 대비 AP(도구)로 따로 본다.
- task 품질(COCO bbox AP)은 analysis plane 결과다. metric catalog에 없다. bundle과, digest가 run의 선언과 같은 dataset index만 읽는다.
- 2026-10-08 Metric Set v1부터: workload가 `coco-detection-bbox/1`을 선언한 run은 ground truth를 bundle에 동결하고(`docs/evidence.md` 10절), task 품질은 canonical metric `quality.detection.*`이다(`docs/metrics_catalog.md` 8.7절). 기존 baseline bundle(v4)은 동결된 ground truth가 없어 위 도구 결과가 품질 근거로 남는다.

### 11.4 nsys 교차 검증 (`tools/stage2b_g_real/nsys_align_plan.py`)

10.7과 같은 절차에서 request당 API 호출이 많고 kernel이 많은 점만 다르다. NVTX range 안의 모든 API 호출과 host submit bracket 안의 API 호출이 같은 correlation 집합을 주어야 한다. request마다 kernel 수, kernel busy 합, H2D 끝 → 마지막 kernel 끝 span, "kernel launch 호출이 앞 activity 끝보다 늦게 끝난" gap(host enqueue를 기다린 시간), 완료 관측 지연을 nsys 시계로 계산하고, bundle의 event bracket은 자기 시계의 값으로 나란히 둔다.

## 12. Model-level Metric Set v1의 시간 view (기록되는 사실은 바뀌지 않음)

Metric Set v1은 새 timestamp를 기록하지 않는다. 기존 사실에서 계산한다(`docs/metrics_catalog.md` 8절).

| view | 식 (request마다, 정수 ns) | clock |
|---|---|---|
| request-completion | `t_complete − t_sched` | host run clock |
| DUT-service | `t_complete − t_submit_begin` = submit + post-submit | host run clock |
| submit | `t_submit_end − t_submit_begin` | host run clock |
| post-submit | `t_complete − t_submit_end` | host run clock |
| output-ready | `t_output_ready − t_sched` = request-completion + output stage | host run clock |
| output stage | `t_output_ready − t_complete` | host run clock |
| issue lag | `t_issue − t_sched` | host run clock |
| H2D / device-execution bracket / D2H | `device_observation.timing.*` | 장치 clock(측정 경계의 `device_timing`), host 시각과 섞지 않음 |

- deadline view 두 개: completion(`t_complete ≤ t_sched + period`)과 output-ready(`t_output_ready ≤ t_sched + period`). 둘 다 evaluation parameter이며 FTTI가 아니다.
- rate: offered rate는 planned 도착 시각에서, achieved / closed-loop completion rate는 [첫 planned `t_sched`, DRAIN 끝] window에서 계산한다(legacy throughput과 같은 분모).
