# Metric Catalogs — Model-level Metric Set v1 (기본) + Stage 1A legacy catalog

> **기본 catalog는 `sdv-bmt-model-metric-set-v1`(8절)이다**(2026-10-08 승인 "SDV-BMT Model-level Metric Set v1"). 1–7절은 legacy catalog `bmt-portable-v1-stage1a`의 정의이며 그대로 동결되어 있다: `sdv-bmt evaluate --catalog bmt-portable-v1-stage1a`가 이전 평가(G-1..G-5 golden snapshot, Stage 1R-G/2b-G 평가)를 byte 단위로 재현한다.

# Stage 1A Portable Metric Catalog (`bmt-portable-v1-stage1a`, legacy)

> **MOCK ONLY.** 모든 `MetricResult`에 `execution_kind=mock`이 붙는다. FakeDUT 값이며 NPU 성능이 아니다.

## 1. MetricResult

| field | 의미 |
|---|---|
| `metric_id`, `definition_version`, `unit`, `method` | catalog 정의 |
| `measurement_boundary_id` | 값이 의미를 갖는 측정 경계 |
| `status` | `valid` / `empty` / `censored` / `invalid` / `not_collected` / `not_supported` / `not_applicable` |
| `status_reason` | 비-valid 사유 (예: `insufficient_samples`, `rank_in_non_completed_region`) |
| `value` | `status=valid`일 때만 숫자. 그 외에는 항상 `null` (contract에서 강제) |
| `n` | 계산에 쓴 표본 수 |
| `censored_fraction` | planned percentile에서 미완료(+∞) 비율 |
| `window` | `t_measure_begin_ns`, `t0_ns`(첫 planned `t_sched`), `t_drain_end_ns`, `drain_cap_reached` |
| `inputs`, `notes` | 사용한 raw 파일, 해석 주석 |
| `eval_id`, `execution_kind` | 평가 ID(volatile), `mock` |

`numeric_eligible(result)`는 `status=valid`일 때만 참이다. Stage 1A에는 숫자 비교·Δ·검정 code path가 없다.

## 2. 공통 규칙

- percentile: nearest-rank, `rank = ceil(p/100 × n)`(1-based, 오름차순).
- p99는 **completed 표본 100개 이상**이어야 한다(design v03 H.3). 미만이면 `invalid(insufficient_samples)`. planned p99에도 같은 규칙을 적용하며, 이 검사는 censoring 판정보다 먼저 한다.
- completed-only percentile(p50/p90)은 표본 1개 이상. planned p50/p90은 completed 최소 수 조건이 없다(전부 미완료면 `censored`, fraction 1.0).
- 표본 0개는 `empty`.
- 비율은 정확한 유리수 `Fraction(a, b)`를 float로 변환(정확히 반올림).
- run이 MEASURE에 들어가지 못했으면 모든 metric이 `not_collected(measure_phase_not_executed)`.
- `measurement_status=invalid`이면 모든 metric이 `invalid(measurement_invalid:<reason>)`.
- raw event가 run_finished의 계수와 다르거나, event 순서가 불법이거나, 계수 항등식이 깨지면 모든 metric이 `invalid`.

## 3. Catalog

| metric | 단위 | 정의 |
|---|---|---|
| `bmt.count.{planned,issued,completed,dropped,timed_out,error,not_issued}` | count | measure phase request 상태별 수 |
| `bmt.diagnostic.late_completion.count` | count | timeout 이후 반환된 call 수. completed에 포함하지 않음 |
| `bmt.rate.{completion,drop,timeout,error}` | ratio | 해당 count ÷ planned |
| `bmt.latency.request_completion.completed.p{50,90,99}` | ns | COMPLETED의 `t_complete − t_sched` |
| `bmt.latency.request_completion.planned.p{50,90,99}` | ns | 모든 planned, 미완료 = +∞. p99는 completed ≥ 100 필요. rank가 +∞면 `censored`, `value=null` |
| `bmt.latency.dut_service.p{50,90,99}` | ns | COMPLETED의 `t_complete − t_submit_begin` (virtual FakeDUT 경계) |
| `bmt.latency.output_ready.completed.p{50,90,99}` | ns | COMPLETED의 `t_output_ready − t_sched` |
| `bmt.issue_lag.{p99,max}` | ns | issued의 `t_issue − t_sched`. virtual에서는 schedule diagnostic(항상 0) |
| `bmt.throughput.completed_per_s` | 1/s | completed ÷ ((`t_drain_end − t0`)/1e9), `t0` = 첫 planned `t_sched`(lead 시간 제외). configured N=1 처리량이며 최대 처리량이 아니다. 표본이 적으면 마지막 completion까지의 tail 때문에 offered rate보다 클 수 있다 |
| `bmt.offered_rate_per_s` | 1/s | (planned − 1) ÷ ((`t_sched_last − t_sched_first`)/1e9). `asap`은 `not_applicable` |
| `bmt.deadline.miss_rate` | ratio | `COMPLETED ∧ t_complete ≤ t_sched + period_ns`가 아닌 planned 비율. deadline 미지정 시 `not_applicable` |
| `bmt.output.identity_rate` | ratio | COMPLETED output sha256이 사전 지정 reference와 같은 비율(분모: reference entry가 있는 COMPLETED). reference 미지정 시 `not_applicable` |

## 4. Censoring

- planned percentile에서 미완료(dropped, timed_out, error, not_issued)는 +∞다.
- rank가 +∞ 영역이면 `status=censored`, `value=null`, `censored_fraction`을 기록한다. timeout·deadline 같은 임의 값을 대입하지 않는다.
- censored 결과는 report-only다. 숫자 Δ, 상대 Δ, 검정 대상이 아니다.
- stream-level 해석은 completion/drop/timeout rate, deadline miss, completed latency coverage를 함께 본다.

## 5. Output identity

- reference 우선순위 1인 external golden output만 구현했다. 사전 지정 nominal reference run은 구현하지 않았다.
- reference는 plan에 경로와 SHA-256으로 지정하고, run 시점에 파일 digest를 검증해 `run_spec.json`에 entry와 `entries_digest`를 내장한다. Evaluator는 bundle만 읽는다.
- evaluate 시점에는 reference를 바꿀 수 없다(사전 지정 원칙). 첫 nominal run을 암묵적 기준으로 쓰지 않는다.
- late-completion output은 비교에서 제외한다. identity는 결정성·손상 진단이며 accuracy가 아니다.
- G-1 golden reference(`configs/references/g1-dataset-fixture-8.fake-identity.golden.jsonl`)는 runner 없이 문서화된 fake 공식으로 도출했다(`docs/evidence.md` 6절).

## 6. 구현하지 않은 metric

`bmt.device_execution.*`, `bmt.power.*`, `bmt.energy.*`, `quality.*`, `tel.*`는 Stage 1A 범위가 아니다(catalog에 없음).

## 7. Stage 2b-G real-model baseline: task quality는 catalog 밖

YOLOX-S/COCO baseline의 task 품질(COCO bbox AP, AP50, AP75, 크기별 AP, AR)은 `tools/stage2b_g_real/coco_quality.py`가 bundle과 digest가 맞는 dataset index로 계산하는 **analysis plane 결과**다. design v03 H.6의 `quality.<task>.*`는 아직 catalog에 넣지 않았다(새 canonical metric은 프로젝트 검토 뒤). plan validation은 여전히 `quality_evaluator: null`을 요구한다. 계산은 completed 출력만(`conditional`)과 coverage를 함께 보고하고, detection 전용의 "누락 출력 = detection 없음" 값(`planned_missing_as_empty`)은 옆에 따로 둔다.

> 2026-10-08 Metric Set v1부터 detection task 품질은 canonical metric이다(8.7절). 다만 frozen ground truth가 없는 기존 bundle(manifest v3/v4)에서는 `not_collected`이고, 위 도구 결과가 그 bundle들의 품질 근거로 남는다.

## 8. Model-level Metric Set v1 (`sdv-bmt-model-metric-set-v1`)

승인 근거: 2026-10-08 프로젝트 결정 "APPROVED DESIGN UPDATE — SDV-BMT Model-level Metric Set v1"과 같은 날 후속 결정(threshold metric 기본 operating point score ≥ 0.5·IoU ≥ 0.5, 품질 이름 `quality.detection.conditional.*`, `bmt.required_rate_met` 규칙). 구현: `src/sdv_bmt/metrics/metric_set_v1.py`(catalog·계산), `quality_detection.py`(bundle 기반 detection 품질), `quality_coco.py`(COCO 평가 core), `src/sdv_bmt/contracts/quality.py`(품질 평가기 선언). 모든 값은 Python evaluator가 bundle만 읽어 계산한다. native runner는 사실만 기록한다.

### 8.1 결과 계층과 해석 규칙

| layer (`MetricSetResult.layer`) | 포함 |
|---|---|
| `performance` | request-completion, DUT-service, output-ready 분포, offered/achieved/closed-loop rate, 상태별 count, completion/drop/timeout/error/not-issued rate, censored fraction, late completion |
| `timeliness` | completion·output-ready deadline miss, 외부 required rate와 그 충족 여부 |
| `task_quality` | `quality.detection.coverage`, `quality.detection.conditional.*` |
| `diagnostics` | issue lag, submit / post-submit / output-stage, H2D / device-execution bracket / D2H, output identity |
| `validity_evidence` | `data.*`(item 수, annotation coverage, payload·prepared input·annotation 검증률, traceability) |

Design / Realization / Measurement Guard, overall contrast validity, attribution allowed는 metric이 아니라 구조화된 validity 결과다(`evidence/validity.py`). 성능 점수로 합치지 않는다.

규범 문장:
- **Percentile latency is a distribution-characterization metric, not a worst-case timing guarantee.**
- **`max_observed` is the largest value observed in the measured population. It is not WCET.** `worst_case`·`wcet` 이름의 metric은 없다.
- deadline은 evaluation parameter다. deadline miss는 FTTI 준수가 아니다. inference timing 결과는 FTTI가 아니다.
- closed-loop completion rate는 configured N의 관측값이며 maximum·peak·capacity 주장이 아니다.
- diagnostic은 품질 metric이 아니다. output identity는 결정성·손상 진단이며 정확도가 아니다.
- SDV-BMT Model-level 결과만으로 ISO/PAS 8800, ISO 26262, SOTIF 준수를 주장하지 않는다(외부 요구 → 할당된 model/runtime 요구 → SDV-BMT 측정 증거 → 충족/특성화).

### 8.2 공통 규칙

- percentile: nearest-rank `rank = ceil(p/100 × n)`. p50/p90은 표본 1개 이상, **p99는 그 population 표본 100개 이상**(아니면 `invalid(insufficient_samples)`). completion 기반 latency의 population은 COMPLETED request, issue lag는 issued request, 장치 segment는 장치 timing이 있는 COMPLETED request다.
- `max_observed`: 그 population의 최댓값. 표본 0개면 `empty`.
- planned percentile(`bmt.request_completion.planned.p*`)은 legacy 의미 그대로: 미완료 planned = +∞, p99는 completed ≥ 100, rank가 +∞면 `censored`(value null, report-only, 숫자 비교·Δ에 쓰지 않음).
- run 단위 gate(legacy와 같음): MEASURE 미실행 → 모든 metric `not_collected(measure_phase_not_executed)`, measurement invalid·window 불완전·event 순서 불법·run_finished 계수 불일치·계수 항등식 위반 → 모든 metric `invalid`.
- `valid`가 아니면 value는 항상 null이다. 0이나 다른 값으로 대신하지 않는다.

### 8.3 결과 형식과 population 문맥 (evaluation schema 2)

`metrics.json`(schema_version 2): `catalog_version`, `parameters`(evaluation protocol), `context`(이 평가의 공통 population·identity 문맥), `metrics`(MetricSetResult 목록).

| `MetricSetResult` field | 의미 |
|---|---|
| legacy field | `metric_id`, `definition_version`, `measurement_boundary_id`, `unit`, `status`, `value`, `window`, `method`, `inputs`, `notes`, `eval_id`, `execution_kind`, `status_reason`, `censored_fraction` (legacy의 `n`은 `population.sample_count`로 옮김) |
| `layer` | 8.1절의 계층 |
| `population` | 최소 `name`, `sample_count`. request population은 `planned_count`, `completed_count`, `coverage`(= sample_count / planned_count)도 가진다. rate는 `replay_mode`, `configured_n`; 장치 segment는 `timing_method`, `clock`; 품질은 `images`, `gt_objects`, `gt_crowd`, `detections`(operating point metric은 `detections_at_operating_point`, `ignored_crowd_matches`, `gt_non_crowd`) |

`context`: `run_count`(평가 하나 = 1 run), `planned_count`/`issued_count`/`completed_count`, `input_id`, `input_selection_digest`, `items_digest`, `item_count`, `measurement_boundary_id`, `clock_domain`, `device_timing_method`, `replay_mode`, `rate_hz`, `configured_n`, `queue_capacity`, `on_full`, bundle의 manifest schema version과 `metric_definition`(run이 기록한 catalog), `quality_reference`(v5만). evaluation protocol identity는 `eval_manifest.json`의 `protocol`·`protocol_digest`, catalog version은 `parameters.catalog_version`과 `evaluator.catalog_version`에 있다. completed-only 값은 항상 coverage와 함께 나온다.

### 8.4 Performance

| metric | 단위 | 정의 |
|---|---|---|
| `bmt.request_completion.{p50,p90,p99,max_observed}` | ns | COMPLETED의 `t_complete − t_sched` |
| `bmt.request_completion.planned.{p50,p90,p99}` | ns | planned 전체(미완료 = +∞). censoring 규칙 8.2 |
| `bmt.dut_service.{p50,p90,p99,max_observed}` | ns | COMPLETED의 `t_complete − t_submit_begin`. submit 호출, 선언된 경계 안의 DUT/runtime 작업, 완료 관측을 포함한다. accelerator/장치 실행 시간이 아니다 |
| `bmt.output_ready.{p50,p90,p99,max_observed}` | ns | COMPLETED의 `t_output_ready − t_sched` |
| `bmt.offered_rate` | 1/s | `(planned − 1) / ((t_sched_last − t_sched_first)/1e9)`: benchmark가 계획·제공한 rate. fixed_rate·recorded. asap은 `not_applicable` |
| `bmt.achieved_completion_rate` | 1/s | `completed / ((t_drain_end − t0)/1e9)`, window = [t0 = 첫 planned `t_sched`, DRAIN 끝]. legacy `bmt.throughput.completed_per_s`와 같은 분모(바꾸지 않음). 마지막 completion까지의 tail 때문에 offered rate보다 약간 클 수 있다. open-loop replay(fixed_rate, recorded)만, asap은 `not_applicable(closed_loop_replay)` |
| `bmt.closed_loop_completion_rate` | 1/s | 같은 식, asap(closed-loop)만. configured N(`run.max_inflight`, population의 `configured_n`)에서 관측한 값. maximum·peak·capacity 주장이 아니고 주기적 rate 요구 충족 근거로 쓰지 않는다. 다른 mode는 `not_applicable(open_loop_replay:<mode>)` |
| `bmt.count.{planned,issued,completed,dropped,timed_out,error,not_issued}` | count | measure phase request 상태별 수 |
| `bmt.late_completion_count` | count | timeout 뒤 반환된 call 수(completed에 넣지 않음) |
| `bmt.{completion,drop,timeout,error,not_issued}_rate` | ratio | 해당 count ÷ planned |
| `bmt.censored_fraction` | ratio | `(planned − completed)/planned` |

### 8.5 Timeliness / requirement satisfaction

| metric | 단위 | 정의 |
|---|---|---|
| `bmt.deadline.completion_miss_{count,rate}` | count / ratio | miss = `COMPLETED ∧ t_complete ≤ t_sched + period_ns`가 아닌 planned request(drop·timeout·error·not_issued는 miss). rate 분모 planned. legacy `bmt.deadline.miss_rate`와 같은 의미 |
| `bmt.deadline.output_ready_miss_{count,rate}` | count / ratio | 같은 규칙을 `t_output_ready`로: 쓸 수 있는 task 결과가 예산 안에 준비되었는가. completion view를 대체하지 않는다 |
| `bmt.required_rate` | 1/s | 외부에서 준 required rate(evaluation parameter `required_rate_hz`). 하드웨어 속성이 아니다. 없으면 `not_applicable(no_required_rate_parameter)` |
| `bmt.required_rate_met` | boolean (1/0) | 프로젝트 규칙(2026-10-08): fixed_rate run에 required rate가 있을 때 `offered_rate ≥ required_rate`이고 `achieved_completion_rate ≥ required_rate`이면 1, 아니면 0. `offered_rate < required_rate`이면 `not_applicable(offered_rate_below_required_rate)`(그 부하를 제공하지 않아 시험되지 않음, false가 아님). asap/closed-loop은 `not_applicable(closed_loop_replay)`, recorded replay는 `not_applicable(no_periodic_offered_rate:recorded)`. drop·timeout·error는 이 metric 안에서 요구하지 않고 각자의 metric으로 본다. 비교는 정확한 유리수로 한다 |

deadline과 required rate는 evaluation parameter다. deadline은 run_spec 값 또는 `--eval-config`의 override, required rate는 override에서만 온다. 출처는 `eval_manifest.json`의 `parameter_sources`에 남는다.

### 8.6 Diagnostics

| metric | 단위 | 정의 |
|---|---|---|
| `bmt.issue_lag.{p50,p90,p99,max_observed}` | ns | issued request의 `t_issue − t_sched`. harness/replay realization(issuer fidelity) 증거이며 accelerator latency가 아니다. report 표에서는 Performance 표에 진단 표시와 함께 둔다 |
| `bmt.submit.{p50,p90,p99,max_observed}` | ns | COMPLETED의 `t_submit_end − t_submit_begin`: DUT 경계 안의 host/runtime 제출(enqueue) 구간 |
| `bmt.post_submit.{…}` | ns | COMPLETED의 `t_complete − t_submit_end`: 장치 시작 대기, 장치 실행, 전송 완료, 완료 관측 지연을 runtime에 따라 포함할 수 있다. 순수 장치 실행이 아니다 |
| `bmt.output_stage.{…}` | ns | COMPLETED의 `t_output_ready − t_complete`: DUT 완료 뒤 workload의 postprocess·codec·출력 기록. 입력 내용에 따라 달라질 수 있다 |
| `bmt.h2d.{…}`, `bmt.device_execution_bracket.{…}`, `bmt.d2h.{…}` | ns (장치 clock) | COMPLETED request의 `device_observation.timing.{host_to_device, device_execution, device_to_host}`. 측정 경계(`device_timing`)가 그 segment를 선언한 경우만. 선언이 없으면 `not_supported`, 선언했는데 COMPLETED request에 값이 없으면 `invalid(device_observation_missing)`. host run-clock 시각과 빼거나 섞지 않는다. `device_execution_bracket`은 host enqueue 대기를 포함할 수 있으므로 장치 busy 시간이나 pure device execution이 아니다. 독립 검증된 device-active 시간이 생기면 별도 metric으로 추가한다 |
| `bmt.output.identity_rate` | ratio | legacy 의미 그대로(사전 지정 reference 대비 output digest 일치율). 결정성·손상 진단이며 정확성의 증거가 아니다 |

같은 request에서 `dut_service = submit + post_submit`, `output_ready = request_completion + output_stage`(정수 ns, host clock)가 성립한다. percentile은 각 분포의 것이라 percentile끼리 더하지 않는다.

### 8.7 Task quality — object detection (`coco-detection-bbox/1`)

- workload가 `quality_evaluator: coco-detection-bbox/1`을 선언한 realtime run은 planned item의 ground truth를 bundle에 동결한다(`quality_reference.jsonl`, manifest v5, `docs/evidence.md` 10절). evaluator는 그 행마다 canonical digest를 `input_items.jsonl`의 `annotation_digest`와 대조하고, completed 출력은 `output_ref` digest로 재검증한다. 외부 dataset 경로는 읽지 않는다.
- population은 **conditional**: valid task output이 있는 COMPLETED request의 item. 항상 coverage와 함께 보고한다. missing-output-as-empty(계획 전체 품질) 의미는 없다(별도 승인 전).
- 선언이 없는 bundle(기존 v3/v4 전부): 모든 품질 metric이 `not_collected(quality_evaluator_not_declared)`. reference가 item digest와 맞지 않으면 `invalid(quality_reference_integrity)`, 출력이 schema·image id·digest와 맞지 않으면 `invalid(task_output_invalid)`.

| metric | 단위 | 정의 |
|---|---|---|
| `quality.detection.coverage` | ratio | 품질 평가된 planned item ÷ planned request |
| `quality.detection.conditional.{ap, ap50, ap75}` | ratio | COCO AP: IoU 0.50:0.05:0.95 평균 / 0.50 / 0.75, area all, maxDets 100 |
| `quality.detection.conditional.{ap_small, ap_medium, ap_large}` | ratio | COCO AP(IoU 0.50:0.95), area < 32², 32²–96², ≥ 96² |
| `quality.detection.conditional.{ar1, ar10, ar100}` | ratio | COCO AR(IoU 0.50:0.95, area all), image·category당 detection 1 / 10 / 100개 |
| `quality.detection.conditional.precision` | ratio | operating point에서 `TP / (TP + FP)` |
| `quality.detection.conditional.recall` | ratio | `TP / (TP + FN)` |
| `quality.detection.conditional.f1` | ratio | `2TP / (2TP + FP + FN)` |
| `quality.detection.conditional.{tp,fp,fn}_count` | count | 아래 matching 규칙의 TP / FP / FN |
| `quality.detection.conditional.fp_per_image` | 1/image | FP ÷ 평가된 image 수 |
| `quality.detection.conditional.fn_per_gt` | ratio | FN ÷ non-crowd ground truth 수 |

COCO 평가(evaluation identity에 기록): pycocotools `COCOeval` bbox 의미의 stdlib 구현(`tools/stage2b_g_real/coco_eval.py`의 이식, 같은 산술·같은 순서). 그 도구는 YOLOX-S/COCO baseline에서 pycocotools 2.0.11과 12개 통계 7.6×10⁻¹⁵ 이내로 일치했다(`runs/stage2b-g-real/analysis/coco_eval_crosscheck.json`, conformance 증거). IoU thresholds linspace(0.5, 0.95, 10), recall thresholds linspace(0, 1, 101), maxDets 1/10/100, area all/small/medium/large, category = COCO 2017 80개(`COCO_2017_CATEGORY_IDS`), crowd = ignore region(IoU = 교집합 / detection 면적), score 동률은 출력 순서. COCO에 값이 없는 통계(예: 해당 크기 ground truth 없음)는 `empty(no_ground_truth_in_scope)`이며 −1을 값으로 내지 않는다. pycocotools는 runtime dependency가 아니다.

Operating point(기본 score ≥ 0.5, IoU ≥ 0.5; `quality_operating_point` override 가능, evaluation identity에 기록):
- image마다 score 상위 100개(안정 정렬) 중 score ≥ threshold이고 평가 category인 detection.
- (image, category)마다 score 내림차순(안정)으로 greedy matching: 아직 matched가 아닌 ground truth 중 IoU ≥ threshold인 최고 IoU(non-crowd 먼저, COCO `evaluateImg` 규칙).
- non-crowd box와 matched = TP. 이미 matched인 box에 대한 다음 detection(중복)과 unmatched detection = FP. crowd region에만 matched = 무시(TP도 FP도 아님). matched되지 않은 non-crowd box = FN. crowd box는 FN이 아니다.
- class matching = 같은 category. detection TN과 detection accuracy는 정의하지 않는다(유한한 background negative population이 없음).

### 8.8 Validity / evidence (`data.*`)

| metric | 단위 | 정의 |
|---|---|---|
| `data.item_count` | count | resolved input item 수(`input_items.jsonl`) |
| `data.annotation_coverage` | ratio | annotation reference가 있는 item ÷ item |
| `data.integrity.payload_verified_rate` | ratio | INPUT.open이 payload identity(index SHA-256, items digest, file payload SHA-256)를 재검증한 item ÷ item. `input_payload` evidence가 observed이고 items digest가 manifest와 같을 때만 valid(INPUT.open은 불일치 시 run을 실패시킨다) |
| `data.integrity.prepared_input_verified_rate` | ratio | native measurement plane이 SHA-256 재검증으로 staging한 planned item ÷ planned(= 1 − 준비 실패 비율). 기록된 runtime 검사 결과다. item별 prepared digest는 bundle에 없다. virtual runner bundle은 `not_collected` |
| `data.integrity.annotation_verified_rate` | ratio | frozen ground truth가 item의 `annotation_digest`로 재검증된 item ÷ item(bundle에서 다시 계산). v3/v4 bundle은 `not_collected(annotation_content_not_in_bundle)` |
| `data.traceability.complete` | boolean (1/0) | 모든 COMPLETED request가 정확히 한 input item(payload·annotation digest)과, `output_ref`로 재검증되는 출력 bytes(v5에서는 검증된 annotation까지)로 이어지면 1 |

representativeness, temporality, independence 같은 넓은 개념에는 측정 정의가 없으므로 scalar 값을 만들지 않는다.

### 8.9 예약·미구현·보류

- `quality.calibration.*`: namespace만 예약. ECE/MCE/Brier/detection calibration 공식은 고르지 않았다(별도 승인).
- `quality.numeric.*`(max/mean abs error, detection match rate, class agreement, score error): bundle 안에 사전 지정된 수치 reference가 없어 canonical metric으로 구현하지 않았다. Stage 2b-G의 offline 도구 결과(`validate_numerics.py`)가 analysis-plane 진단으로 남는다.
- `quality.detection.iou.*`(localization IoU 분포): 선택 항목, v1에서 구현하지 않음.
- Model-level canonical metric이 아님(별도 승인 전): WCET, FTTI, AI error rate per hour / per km, contributing AI error rate, TTC, PET, DST, SPFM, LFM, PMHF, FIT, neuron/sign structural coverage, input-noise/adversarial robustness, energy per inference, device DRAM bandwidth.

### 8.10 Legacy catalog와의 대응 (같은 의미)

| legacy (`bmt-portable-v1-stage1a`) | Metric Set v1 |
|---|---|
| `bmt.latency.request_completion.completed.p{50,90,99}` | `bmt.request_completion.p{50,90,99}` |
| `bmt.latency.request_completion.planned.p{50,90,99}` | `bmt.request_completion.planned.p{50,90,99}` |
| `bmt.latency.dut_service.p{50,90,99}` | `bmt.dut_service.p{50,90,99}` |
| `bmt.latency.output_ready.completed.p{50,90,99}` | `bmt.output_ready.p{50,90,99}` |
| `bmt.issue_lag.p99`, `bmt.issue_lag.max` | `bmt.issue_lag.p99`, `bmt.issue_lag.max_observed` |
| `bmt.throughput.completed_per_s` | fixed_rate·recorded: `bmt.achieved_completion_rate`, asap: `bmt.closed_loop_completion_rate` (같은 식·분모) |
| `bmt.offered_rate_per_s` | `bmt.offered_rate` |
| `bmt.rate.{completion,drop,timeout,error}` | `bmt.{completion,drop,timeout,error}_rate` |
| `bmt.diagnostic.late_completion.count` | `bmt.late_completion_count` |
| `bmt.deadline.miss_rate` | `bmt.deadline.completion_miss_rate` |
| `bmt.count.*`, `bmt.output.identity_rate` | 같은 이름 |

이름만 바뀐 metric은 같은 bundle에서 legacy 값과 같다(`tests/unit/test_metric_set_v1.py`). legacy metric을 v1 안에서 재정의하지 않았다. legacy 이름은 legacy catalog 안에만 있다.

### 8.11 Evaluation parameters (`--eval-config`, `EvaluationOverride`, schema_version 1)

| field | 의미 |
|---|---|
| `deadline` | 필수 key(null 가능). 주면 run_spec의 deadline을 대체한다(legacy와 같음) |
| `required_rate_hz` | 선택. 외부 required rate(> 0) |
| `quality_operating_point` | 선택. `{score_threshold, iou_threshold}`(각각 (0, 1]). 나머지 matching 규칙은 평가기 version에 고정 |

legacy catalog로 평가할 때는 `deadline`만 받는다(다른 field가 있으면 거절).

### 8.12 Evidence check (metric 아님, v0.1.0)

Metric Set v1 평가의 `metrics.json`에는 `evidence_checks` 목록이 있다. 기록된 request 사실이 run 자신의 구성과 맞는지 보는 진단이며, 기록된 request 상태나 metric 값을 바꾸지 않고 `metrics_content_sha256`에도 들어가지 않는다. legacy 평가 형식에는 없다(동결).

| check | status | 의미 |
|---|---|---|
| `completed_beyond_request_timeout` | `pass` / `flagged` / `not_applicable`(planned request 없음) / `not_evaluated`(event 순서 불법) | COMPLETED인데 `t_complete − t_submit_begin > request_timeout_ns`인 request의 수와 id(앞 50개). Stage 1A timeout 의미로는 TIMED_OUT + late completion이어야 한다. v0.1.0 native loop(의미상 시각 순서, `docs/measurement_protocol.md` 10.2절)는 이런 기록을 만들지 않으며, 이전 bundle에서 나타날 수 있다 |
