# Evidence Bundle (Stage 1A virtual, Stage 1R-G realtime)

> **Stage 1A bundle은 MOCK이다.** `experiment.json`, `run_spec.json`, `manifest.json`, `metrics.json`, `eval_manifest.json`에 모두 `execution_kind: "mock"`과 `mock_notice`가 들어간다.
>
> **Stage 1R-G realtime bundle**(8절)은 별도 schema version을 쓴다. fake native backend run은 `execution_kind: "mock"`(+ realtime mock notice), NVIDIA backend run은 `execution_kind: "actual"`(mock notice 없음, `device_identity.role = reference_accelerator_validation`)이다.

## 0. Schema version

| 문서 | version | 비고 |
|---|---:|---|
| `manifest.json` | 3 | v2: `input_pipeline` section. v3: `realization` section, `status.request_counts`, 세분화된 comparison component(design update "Factor Isolation"). reader는 다른 version을 `schema_version … is not supported`로 거절한다 |
| `run_spec.json` | 2 | v2에 `input_pipeline` section 추가 |
| measurement boundary | 2 | v2에 input pipeline owner/semantics/fused steps/pipeline digest 추가 |
| `events.jsonl`/`stress.jsonl` event | 1 | 변경 없음(`input.equivalence`는 기존 `phase` kind를 쓴다) |
| `experiment.json`, `metrics.json`, `eval_manifest.json` | 1 | 구조 변경 없음 |
| plan YAML | 1 | `input_pipeline`은 기본값이 있는 선택 field다. 다만 해석된 plan에 항상 포함되므로 design change 이전과 같은 YAML이라도 `plan_digest`가 바뀐다 |
| realtime `manifest.json` (Stage 1R-G) | 4 | `run_clock`, `realtime` section 추가. v3 bundle은 그대로 v3이며 v4로 재해석하지 않는다(8절) |
| realtime `run_spec.json` | 3 | `realtime` section(`config`, `measurement_declaration`, `clock_source`) 추가 |
| realtime event | 2 | `device_observation` kind 추가, `run_clock_domain: monotonic`. 한 bundle은 한 version만 쓴다 |
| realtime measurement boundary | 3 | 별도 record `RealtimeMeasurementBoundary`(`docs/measurement_protocol.md` 10.3절). Stage 1A boundary v2 identity는 바뀌지 않는다 |
| realtime `manifest.json` + frozen quality reference (Metric Set v1) | 5 | v4 + `quality_reference` section + raw file `quality_reference.jsonl`(10절). workload가 task-quality evaluator를 선언한 realtime run만 v5다. 그 밖의 run은 v4(realtime)·v3(virtual) 그대로이며, v3/v4 bundle을 v5로 재해석하지 않는다. run_spec(v3)과 event(v2)는 바뀌지 않는다 |
| `quality_reference.jsonl` 행 | 1 | `schema_version`, `item_index`, `source_item_id`, `annotation_digest`, `annotation`(index 행의 annotation object) |
| `metrics.json`, `eval_manifest.json` (Metric Set v1) | 2 | 기본 catalog `sdv-bmt-model-metric-set-v1`의 평가. metric마다 `layer`·`population`, 평가 단위 `context`·`catalog_version`. eval manifest에 `parameter_sources`, `bundle`, `evaluator.metric_layers`, `evaluator.quality_evaluator`. legacy catalog 평가는 계속 schema 1이다(`docs/metrics_catalog.md` 8절) |

## 1. Layout

```text
<out>/<experiment_id>/
├── experiment.json                 첫 run에서 1회 기록. 이후 run은 plan_digest가 같아야 함
└── <run_id>/                       run_id = run-<UTC>-<uuid12> (volatile)
    ├── run_spec.json               기본값까지 해석한 실행 spec (경로 독립)
    ├── input_items.jsonl           typed InputItem, item_index 순
    ├── events.jsonl                runner lifecycle·request event
    ├── stress.jsonl                stress provider record
    ├── outputs.bin                 canonical task output blob 연속 저장
    ├── manifest.json               FINALIZE에서 마지막에 기록
    └── evaluations/<eval_id>/      eval_id = eval-<UTC>-<uuid12> (volatile)
        ├── metrics.json
        └── eval_manifest.json
```

- 모든 파일은 exclusive create로 만들고 덮어쓰지 않는다.
- VALIDATE/PREFLIGHT에서 거절되면 어떤 디렉터리도 만들지 않는다.
- JSONL은 canonical JSON(정렬 key, compact, UTF-8, NaN 금지) 한 줄씩이다. JSON 문서는 정렬 key, 2-space indent다.

## 2. Raw event schema (version 1)

공통 field: `schema_version`, `kind`, `seq`, `t_run_ns`, `run_clock_domain`(`virtual`). `seq`는 `events.jsonl`과 `stress.jsonl`이 공유하며 합치면 정확히 `0..N-1`이다. `seq` 순서로 `t_run_ns`는 감소하지 않는다. event에는 run ID가 없다.

| kind | 필수 field (공통 제외) | 선택 field |
|---|---|---|
| `phase` | `phase`, `edge`(`begin`/`end`) | `outcome`(`ok`/`failed`/`partial`), `error`, `details` |
| `cleanup` | `step`, `outcome`(`ok`/`failed`/`skipped`) | `error`, `details` |
| `issued` | `request_id`, `item_indices`, `phase`, `t_sched`, `t_issue` | |
| `dropped`, `not_issued` | `request_id`, `item_indices`, `phase`, `t_sched`, `reason` | |
| `submit_begin`, `submit_end` | `request_id` | |
| `completed` | `request_id`, `t_complete`, `t_output_ready`, `output_ref` | |
| `error` | `request_id`, `error_phase`, `error` | `t_complete` |
| `timed_out` | `request_id`, `timeout_ns` | |
| `late_completion` | `request_id`, `result`, `t_complete` | `t_output_ready`, `output_ref`, `error_phase`, `error` |
| `run_finished` | `execution_status`, `execution_reason`, `measurement_status`, `measurement_reason`, `request_counts`, `stress_status` | |
| `stress` | `step`, `outcome`, `provider`, `stress_axis_digest`, `stress_level`, `requested`, `achieved`, `independent_evidence`, `verification`, `error`, `details` | |

`phase` 값: `input.open`, `dut.prepare`, `prepare_inputs`, `input.equivalence`(adapter_override만; end event의 `details`에 status/method/validated_items/mismatched_item_indices), `stress.prepare`, `stress.activate`, `stress.verify`, `measure`, `drain`. `cleanup.step` 값: `stress.deactivate`, `stress.restore`, `prepared_store.discard`, `dut.cleanup`, `input.release`.

`output_ref` = `{request_id, file, offset, length, sha256, codec}`.

Reader(`observation/reader.py`)는 strict JSON, kind별 field, request field 타입(`request_id`, `t_*`, `timeout_ns`는 음이 아닌 정수, `item_indices`는 비어 있지 않은 정수 list), seq 연속성, 시간 비감소, 단일 clock domain, `run_spec.json`의 evaluation/schedule section과 replay mode를 검사한다. `fold_requests`로 RequestRecord/CompletionRecord를 재구성하면서 불법 순서(예: timeout 없는 late completion)를 거절한다.

## 3. Manifest (`manifest.json`)

| section | 내용 |
|---|---|
| 상수 | `benchmark_level: "model"`, `execution_kind: "mock"`, `mock_notice` |
| identity | `run_id`(volatile), `experiment_id`, `plan_digest`, `repetition`, `seed`, `created_at_utc`(volatile) |
| `code_identity` | package/version/stage, `source_tree_sha256`, 계산 방법, 파일 수, `vcs`(git 감지 여부, commit 미기록) |
| `environment` | Python·platform·PyYAML version (volatile) |
| `config_digests` | plan/input/workload/dut/stress/output_reference 파일 SHA-256 |
| `input` | kind/id/version/provider, `input_family_digest`, `input_selection_digest`, `items_digest`, item 수, `source_index_sha256`, `annotations_digest`, timing 완전성 |
| `workload` | id, implementation, identity digest, task, artifact(null, fake), precision, batch, codec |
| `dut` | config id, adapter, identity digest, `implementation_kind: mock`, descriptor, capabilities |
| `measurement_boundary` | `measurement_boundary_id`와 정의 전체 |
| `stress` | stress id/kind/provider, `stress_axis_digest`, `stress_level`, `stress_level_value`, status(`verification`, `coverage`, `nominal_check`, `restore`, verify/deactivate 시각) |
| `status` | `execution_status`/`reason`, `measurement_status`/`reason`, `execution_kind`, `request_counts`(planned/issued/completed/error/timed_out/dropped/not_issued/late_completion/unresolved, MEASURE에 들어가지 못했으면 null) |
| `output_reference` | `kind`, path, SHA-256, entry 수, `entries_digest` (또는 `kind: none`) |
| `realization` | `schema_version`, `execution_kind`, `evidence`: 조건별 `RealizationEvidence`가 조건마다 정확히 1개씩 12개다(`docs/measurement_protocol.md` 9절). 중복·누락·불완전한 record는 `RunManifest` contract가 거절한다 |
| `input_pipeline` | `mode`, `owner`, `preprocess_semantics_id`, `input_contract_id/version/digest`, `implementation_id/version/digest`, `fused_steps`, `pipeline_digest`, `measurement_boundary_id`(run boundary와 같아야 함), `equivalence_validation`(`status`: `not_applicable`/`passed`/`failed`/`not_performed`, `method`, `validated_items`, `mismatched_item_indices`, `reference_error_item_indices`, `both_failed_item_indices`, `reason`) |
| `comparison_keys` | component digest, `cell_digest`, `key_stress_contrast`, `key_scenario_contrast`, `key_interaction` |
| `files` | raw 5개 파일의 `sha256`, `bytes` |
| `volatile_fields` | `run_id`, `created_at_utc`, `environment` |

`RunManifest` contract가 필수 section·field와 raw digest를 검사하고, reader는 manifest를 strict parse한다(unknown field 거절).

## 4. Identity digest

모든 digest는 canonical JSON의 SHA-256이며 label(`input_id`, `stress_id`, `workload_id`, `dut_config_id`)은 포함하지 않는다.

| digest | 내용 |
|---|---|
| `input_family_digest` | schema_version, input_kind, family(source dataset/version, task, annotation schema, preprocessing contract, streams, timing 의미). design v03 F.6 목록만 포함하며 index 파일 digest는 넣지 않는다 |
| `input_selection_digest` | family digest + source index SHA-256 + **해석된(resolved)** selection. dataset은 `{start, count(실제 개수)}`, scenario는 `{stream_id(실제), episodes(실제 순서)}`. 같은 item을 고르는 서로 다른 표기(`count: null`과 `count: 128`)는 같은 digest를 갖는다 |
| `items_digest` | item별 `{item_index, source_item_id, payload_sha256, annotation_digest, source_timing, scenario_coordinates}` 목록. `payload_ref`·`source_metadata` 제외 |
| `stress_axis_digest` | stress spec에서 `stress_id`와 `levels`를 뺀 전부(kind, target_resource, placement, control, background_workload, verification, schedule, on_apply_failure, provider_options) |
| `measurement_boundary_id` | MeasurementBoundary 전체 |
| `input_pipeline` digest | `InputPipelineRecord`(mode, owner, semantics, contract digest, implementation id/version/digest, fused steps) |
| `common_fixed_identity` | DUT config 내용, DUT descriptor, DUT capabilities, workload, **input pipeline**, boundary, replay, run policy, output capture, telemetry=`none`, host placement=`virtual`, software revision(source tree), execution kind, metric catalog version, output reference(path 제외) |

비교 key (design v03 J.3):

| key | 포함 | 제외 |
|---|---|---|
| `cell_digest` | 전부 | — |
| `key_stress_contrast` | common, input kind/family/selection/items, stress axis | `stress_level`, `stress_level_value` |
| `key_scenario_contrast` | common, input kind(driving_scenario), family, stress axis+level | selection, items |
| `key_interaction` | common, family, stress axis | selection, items, level |

dataset input에서는 scenario/interaction key가 `null`이다. key는 design identity를 요약할 뿐이다. 두 run을 비교해도 되는지는 `evidence.validity.evaluate_contrast_validity`(4a절, `docs/architecture.md` 9절)가 component별로 판정하며, metric 값은 비교하지 않는다.

`run_spec.json`의 `input_pipeline` section은 plan의 selection, 효과적 `InputPipelineRecord`와 digest, `WorkloadInputContract` 전체, override descriptor(또는 null)를 담는다.

### 4a. Comparison components와 contrast validity

`comparison_keys.components.common_parts`(모든 contrast에서 고정):

| component | 값 |
|---|---|
| `benchmark_level`, `execution_kind` | `model`, `mock` |
| `dut_config`, `dut_descriptor`, `dut_capabilities` | DUT config 내용(`dut_config_id` 제외), adapter 자기 기술(adapter version 등), capabilities. 따로 기록하므로 runtime option, adapter version, capability 변화를 구별할 수 있다 |
| `workload_identity`, `workload_task`, `model_artifact`, `precision_batch`, `postprocess_codec` | workload 전체 digest와 세부 값 |
| `preprocess_semantics_id`, `input_pipeline` | contract의 전처리 의미 ID, 효과적 input pipeline digest |
| `measurement_boundary_id`, `run_clock_domain` | boundary ID, `virtual` |
| `replay`, `admission_policy`, `timeout_drain`, `output_capture` | replay policy, N/queue/on_full/on_request_error, timeout/drain, output capture |
| `telemetry`, `host_placement`, `software_revision`, `metric_definition` | `none`, `virtual`, source-tree SHA-256, catalog version |
| `output_reference` | output reference의 kind, SHA-256, entry 수, `entries_digest`(path는 label이라 제외). run 전에 지정하며 평가 때 바꿀 수 없다 |

digest 하나로 된 component(`workload_identity`, `input_pipeline`, `stress_axis_digest`)가 다르면, guard는 어느 identity가 달라졌는지까지만 말한다. 그 안의 어느 field가 달라졌는지는 두 `run_spec.json`의 해석된 plan을 비교해 찾는다.

factor component:
- `input_kind`
- `input_family_digest`
- `input_selection_digest`
- `items_digest`
- `stress_axis_digest`
- `stress_level_table`(level → 값 표. 고정)
- `stress_level`
- `stress_level_value`

`ContrastValidityResult`는 file로 저장하지 않는다(Stage 1A에는 analysis 산출물이 없음). 다음 field로 이루어진다:
- `contrast_type`, `run_a`, `run_b`
- `design_guard`: `status`, `allowed_to_vary`, `observed_differences`, `violations`, `intended_factor_varied`, `notes`, `unknown_identities`
- `realization_guard`: `status`, `basis`(`evidence_mismatch`/`mock_execution`/`execution_kind_not_actual`/`partial_evidence`/`observed_evidence`), `evidence_available`, `mismatches`, `unresolved`, `intended_factor_conditions`. factor 실현 문제는 항목에 `side`, `reason`, `validity`, `value`를 함께 적는다
- `measurement_guard`: `status`, `boundary_identity`, `violations`(`evaluation_protocol.<key>`, `evaluator.<key>`, `run_x:evaluation_of_another_run` 포함), `notes`
- `overall_status`, `attribution_allowed`

## 5. 평가 산출물

- `metrics.json`: `parameters`(catalog version, percentile 방법, p99 최소 표본, deadline, output reference identity), run status, metric 목록.
- `eval_manifest.json`: evaluator module·`code_identity`·catalog version·metric별 definition version, protocol(parameters + deadline source), `protocol_digest`, 입력 파일(manifest와 raw 5개)의 SHA-256, 평가 전후 raw 무결성, `metrics.json` digest, `metrics_content_sha256`.
- `metrics_content_sha256` = `{parameters, metrics(eval_id 제외)}`의 canonical digest. 같은 bundle·evaluator·parameter라면 eval ID가 달라도 같다.
- volatile: `eval_id`, `run_id`, `created_at_utc`, `outputs`.
- 평가는 raw digest 검증 → bundle 해석 → metric 계산 → raw digest 재검증 순서로 진행하고, 재검증이 통과한 뒤에만 `evaluations/.staging-<eval_id>/`에 두 파일을 쓰고 `evaluations/<eval_id>/`로 rename한다. 중간에 실패하면 `<eval_id>` 디렉터리는 생기지 않는다(`.staging-*`만 남을 수 있으며 결과로 취급하지 않는다).

## 6. Fixture와 golden 출처

| 파일 | 생성 규칙 |
|---|---|
| `configs/fixtures/dataset-fixture-v1.index.jsonl` | `ds-0000`…`ds-0127`, `payload_text = "dataset fixture sample NNNN"`, `annotation = {"label": i % 10}`. timing·좌표 없음. G-1은 0–7, G-3은 0–2, G-5는 0–127을 선택 |
| `configs/fixtures/scenario-fixture-v1.index.jsonl` | `t_capture_us`(정수 µs). `ep-a`: cam_front offset 0/33412/66590/100105/133301/166874 µs와 cam_rear 6 frame. `ep-b`: cam_front 0/33200/66740/99950/133480 µs와 cam_rear 5 frame(+600 s base). `ep-g2`: cam_front 0/33000/66000/99000 µs. 모든 행에 episode/stream/sensor/frame ID 있음 |
| `configs/fixtures/scenario-regression-v1.index.jsonl` | `ep-r` cam_front 0/33000/**30000**/99000 µs (역행 거절 fixture) |
| `configs/references/g1-dataset-fixture-8.fake-identity.golden.jsonl` | item 0–7에 대해 runner 없이 도출: `task_input = b"sdv-bmt fake preprocess v1\n" + payload`, `value = int(sha256(task_input)[:8], 16)`, output = canonical JSON `{schema, task, request_id=item_index, item_index, item_digest, value}`, 그 SHA-256 |
| `tests/golden/expected/G1..G5/` | `python -m tests.golden.make_golden`으로 생성한 회귀 snapshot(events, stress, outputs digest, canonical metric). 기대 의미값은 `tests/golden/test_golden_cases.py`에 승인서 4절 기준으로 손으로 작성되어 있고 snapshot에서 오지 않는다. snapshot은 제출 전 내부 review 수정 후 한 번 재생성했다(G-1–G-4 raw snapshot은 byte 동일, metric snapshot은 `t0_ns`와 method 문구만 변경, G-5는 설정 재설계로 전체 변경) |

각 fixture의 SHA-256은 input spec(`source.index_sha256`)과 plan(`output_reference.sha256`)에 고정되어 있고 PREFLIGHT와 `INPUT.open`에서 재검증된다.

## 7. Provenance

- git이 없으므로 `source_tree_sha256` = `src/sdv_bmt/**/*.py`의 `[{path, sha256}]`(경로순) canonical JSON SHA-256.
- core는 subprocess를 쓰지 않으므로 git commit/dirty는 조회하지 않는다(`vcs.commit_recorded=false`).

## 8. Stage 1R-G realtime bundle

### 8.1 쓰는 주체

| 파일 | 쓰는 주체 |
|---|---|
| `experiment.json`, `run_spec.json`(v3), `input_items.jsonl` | Python control plane |
| `events.jsonl`(v2) | MEASURE 전후 lifecycle event는 Python recorder, precondition·backend snapshot·MEASURE/DRAIN event는 native runner. 한 번에 한 쪽만 쓴다. 인계 때 `next_seq`, 마지막 `t_run_ns`를 넘기고 native는 시간 역행을 거절한다 |
| `outputs.bin` | native runner(인계받은 offset 크기를 확인한 뒤 append) |
| `stress.jsonl` | Python(NoStress) |
| `manifest.json`(v4) | Python FINALIZE |

native runner는 MEASURE/DRAIN 동안 사실을 미리 할당한 memory에 두고 DRAIN 뒤 한 번에 append한다. native process가 MEASURE 중 죽으면 그 window의 event와 출력은 남지 않고 run은 실패로 기록된다(부분 evidence를 만들지 않는다).

### 8.2 Event v2 추가 사항

| kind / phase | 내용 |
|---|---|
| `device_observation` | `request_id`, `timing`(장치 clock segment duration ns: `host_to_device`/`device_execution`/`device_to_host`, 계측 `none`이면 null), `timing_method`(예: `nvidia:cuda_event_pair:bracket`), `observations`(namespaced 정수 사실: `nvidia.h2d_bytes`, `nvidia.in_kernel_globaltimer_ns`, `nvidia.in_kernel_sm_cycles`) |
| `phase: dut.prepare` end `details` | native prepare 사실(`cubin_sha256`, NVRTC option/log, 관측한 profiler injection 환경, 할당 수), native spec digest, 실현된 CPU affinity |
| `phase: precondition` | begin `details.requests`, end에 요약(`requested`, `completed`, `errors`) 또는 실패 |
| `phase: dut.observe.measure_begin`, `dut.observe.drain_end` | backend snapshot(`details`): P-state, SM/메모리 clock, 온도, 전력, clocks event reason, PCIe link, 메모리 사용량, 다른 compute process, NVML field 값(throttle ns, PCIe byte, replay counter), stream idle, 누적 할당 수, `cubin_sha256`, `drain_end`의 process utilization 표본 요약, snapshot 자신의 소요 시간. 실패하면 `outcome: failed` |
| `phase: measure` begin `details` | `planned_requests`, `replay_mode`, `t0_ns`, `wait_policy` |
| `phase: drain` begin/end `details` | `drain_ns`, `cap_t_run_ns` / `cap_reached`, `outstanding_request_ids` |
| `phase: host.placement` end(failed) | 요청과 실현된 CPU affinity가 다를 때 |

request event의 field는 v1과 같다. `dropped`/`not_issued` event의 `t_run_ns`는 admission 시각(`t_issue`)이다.

### 8.3 Manifest v4 추가 section

| section | 내용 |
|---|---|
| `run_clock` | `domain: monotonic`, `source: CLOCK_MONOTONIC`, `anchor_monotonic_ns`, `anchor_realtime_ns`(volatile) |
| `realtime` | `cpu_affinity_requested`, `cpu_affinity_realized`, `wait_policy`, `precondition_requests`, `precondition`(요약), `measurement_plane: native`, `native_counts`(native가 보고한 계수, 교차 검증용) |
| `dut.descriptor.device_identity` | 실행 시 감지한 장치 identity(NVIDIA: product name, UUID, PCI bus/device/subsystem id, VBIOS, board part number, memory, driver, CUDA driver API, compute capability, SM 수, NVRTC version/경로; `role`, `accelerator_family`, `vendor`)와 native measurement plane identity(`native.runner`, `native.runner_sha256`, `native.source_tree_sha256`) |
| `volatile_fields` | `run_id`, `created_at_utc`, `environment`, `run_clock` |

- `status.request_counts`는 Python이 bundle event를 folding해 만든다(source of truth). native 계수와 다르면 run이 실패한다.
- comparison component 추가·변경: `run_clock_domain: monotonic`, `telemetry` = adapter의 계측 profile, `host_placement = {cpu_affinity}`, `realtime_harness = {wait_policy, precondition_requests}`(virtual run은 null, nullable component). `dut_descriptor`가 device identity와 native runner digest를 포함하므로 native 실행 파일이 바뀌면 design identity도 바뀐다.

### 8.4 Profiler 산출물

profiler 교차 검증 run의 `.nsys-rep`와 추출본(`*.extract.json.gz`)은 bundle 밖, 같은 evidence root에 둔다. SQLite export는 추출 뒤 지우고, 지우기 전 SHA-256을 `artifacts_before_extraction.sha256`에 기록한다. profiler run의 bundle은 DUT config(`profiler: nsys`)가 달라 boundary와 identity가 다르다.

## 9. Stage 2b-G real-model bundle 추가 사항

| 위치 | 내용 |
|---|---|
| dataset index row | `payload_text` 대신 `payload_file`(index 디렉토리 안 상대 경로) + `payload_sha256`. annotation은 row 안에 inline(digest가 `items_digest`에 들어간다). InputDescriptor `provider: dataset_files` |
| `phase: dut.prepare` end `details.native.cudnn_plan` | plan·weight SHA-256, glue source/cubin SHA-256과 NVRTC option, cuDNN math type option, algorithm policy, workspace 상한·사용량, 실현된 algorithm table(`[[algo, math_type], ...]`)과 그 SHA-256, heuristic이 고른 값과의 일치 수, convolution별 상세, arena 크기, request당 장치 작업 수, probe cycle |
| `dut.descriptor.device_identity` | `nvidia.cudnn_version`, `nvidia.cudnn_cudart_version`, `nvidia.cudnn_library`, `nvidia.cudnn_library_files`(libcudnn*.so별 SHA-256) |
| `run_spec` / boundary `output_pipeline` | `kind: yolox_coco_v1`, postprocess `yolox-decode-nms-coco/1`, codec, 상수(`conf_threshold_ppm`, `nms_iou_threshold_ppm`, `max_detections`, `input_size`, `num_classes`) |
| `outputs.bin` | request마다 `{"detections":[{"bbox_mpx":[x,y,w,h],"category_id","score_ppm"}],"image_id","item_index","schema":"sdv-yolox-coco-detections/1"}`(정수만) |
| native `stage` 명령 | item별 `output_context`(`image_id`, `width`, `height`) |
| evidence root `runs/stage2b-g-real/` | `environment/`, `algo_probe/`, `capture/`, `baseline/`, `nsys/`, `analysis/`, `code_snapshots/stage1r-g-frozen/`(v01 증거의 Python·native source tree, digest 일치 확인), `COMMANDS.log` |

## 10. Model-level Metric Set v1 추가 사항 (2026-10-08 승인)

### 10.1 Frozen task-quality reference (manifest v5)

workload가 `quality_evaluator`(현재 `coco-detection-bbox/1`, `src/sdv_bmt/contracts/quality.py`)를 선언하면:

| 단계 | 동작 |
|---|---|
| VALIDATE | 평가기가 등록되어 있어야 하고, 그 task·annotation schema가 workload task·input family의 `annotation_schema`와 같아야 하며, realtime plan이어야 한다(virtual runner는 계속 거절) |
| PREFLIGHT | composition root가 만든 `IndexAnnotationSource`(`src/sdv_bmt/input/annotations.py`, InputProvider와 별개 component)가 digest-pinned index에서 planned item마다 annotation object를 읽는다. 모든 item에 annotation이 있어야 하고 canonical digest가 item의 `annotation_digest`와 같아야 한다. runner가 다시 대조한다. 실패하면 plan 거절(bundle 없음) |
| OPEN BUNDLE | `input_items.jsonl` 직후 `quality_reference.jsonl`을 한 번 exclusive create로 쓴다(canonical JSON 한 줄씩, item 순서) |
| FINALIZE | manifest `schema_version: 5`, `quality_reference` section(`file`, `schema_version`, `evaluator`, `annotation_schema`, `item_count`, `annotations_digest` = item annotation digest 목록의 digest, `input.annotations_digest`와 같음), `files`에 `quality_reference.jsonl`의 SHA-256·크기 |

- reader는 v5에서 이 파일을 strict parse(`QualityReferenceRow`)한다. 구조가 틀리면 bundle을 읽지 않는다.
- evaluator는 다른 raw 파일과 같이 평가 전후로 이 파일의 digest를 검증한다(두 catalog 모두). 행의 annotation digest가 item digest와 다르면(재봉인된 변조 포함) 품질 metric이 `invalid(quality_reference_integrity)`, `data.integrity.annotation_verified_rate` < 1이 된다.
- annotation 내용은 이제 bundle 안에 있다. item identity(`items_digest`)는 바뀌지 않는다(그 안의 `annotation_digest`가 원래부터 내용을 고정했다).

### 10.2 Evaluation schema 2 (Metric Set v1)

- `metrics.json`: `schema_version: 2`, `catalog_version`, `parameters`(catalog version, percentile 방법, p99 최소 표본, planned +∞ 규칙, deadline, `required_rate_hz`, output reference identity, `quality` 평가기 protocol: COCO parameter·category 80개·operating point·matching 규칙), `context`(population·identity 문맥), `metrics`(MetricSetResult).
- `eval_manifest.json`: `schema_version: 2`, `evaluator`(module, code identity, catalog version, metric별 definition version과 layer, 품질 평가기), `protocol`(= parameters + `deadline_source`), `protocol_digest`, `parameter_sources`(deadline / required rate / operating point의 출처), `bundle`(manifest schema version, run이 기록한 `metric_definition`, 일치 여부), 입력 파일 digest(v5는 `quality_reference.jsonl` 포함), 전후 무결성, `metrics_content_sha256` = `{catalog_version, parameters, context, metrics(eval_id 제외)}`의 canonical digest.
- Measurement Guard는 두 평가의 `protocol`(deadline_source 제외)과 evaluator identity를 비교한다. operating point·required rate·deadline이 다르거나 catalog가 다르면(legacy vs v1) `fail`이다.
- 새 run의 design identity `metric_definition`은 `sdv-bmt-model-metric-set-v1`이다. 이전 bundle은 기록된 `bmt-portable-v1-stage1a` 그대로이며, v1로 다시 평가하면 eval manifest의 `bundle.catalog_matches_bundle_metric_definition`이 false다.
- 기존 평가 디렉터리는 수정하지 않는다. 다시 평가하면 새 `eval_id`가 생긴다.
- v0.1.0: `metrics.json`(schema 2)에 `evidence_checks`(기록 사실의 일관성 진단, `docs/metrics_catalog.md` 8.12절)가 추가되었다. metric이 아니며 `metrics_content_sha256`에 들어가지 않는다. 이 key가 없는 이전 schema 2 평가는 진단 없이 계산된 것이다.
