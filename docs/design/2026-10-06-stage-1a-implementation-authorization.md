# SDV-BMT Stage 1A 구현 승인서

> **공개판.** 2026-10-06 Stage 1A 구현 승인 원본(SHA-256 `fa875432c4b84248f0e51754c97bad71b6de787edbd54f6485d6b1d0685d31cc`)과 내용이 같고 제목 표현만 다르다. 아래 '기준 설계 SHA-256'은 승인된 설계 원본의 값이며, 같은 디렉터리의 설계 공개판은 머리말에 적은 차이 때문에 digest가 다르다.

| 항목 | 내용 |
|---|---|
| 승인일 | 2026-10-06 |
| 승인자 결정 | Stage 1A / Stage 1R 분할안 승인, 다음 단계 진행 |
| 기준 설계 | `2026-10-06-stage-0-core-benchmark-redesign-v03.md` |
| 기준 설계 SHA-256 | `29846782acafe875ffc357b47482d57174b0c33edc77f97701f15feecb267794` |
| 승인 범위 | **Stage 1A — Deterministic Model-level Core** |
| 미승인 범위 | Stage 1R, 실제 NPU/SDK, realtime, telemetry, actual stress, analysis, System-level |
| 문서 성격 | 구현 착수 권한과 acceptance contract. 구현 완료 보고서가 아님 |

## 1. 구현 목적

Stage 1A의 목적은 framework 확장이 아니다. 다음 경로가 결정적이고 재평가 가능하다는 것을 증명한다.

```text
typed Input
→ deterministic schedule
→ deterministic FakeDUT execution
→ append-only raw observation
→ bundle-only MetricEvaluator
→ integrity-verifiable evidence
```

Stage 1A 결과는 NPU 성능 결과가 아니다. 모든 run과 metric에 `MOCK`을 명시한다.

## 2. 승인된 최소 범위

### 2.1 Contracts

- `DatasetSpec`
- `DrivingScenarioSpec`
- `SourceTiming`
- `ScenarioCoordinates`
- `InputItem`
- `ReplayPolicy`
- `WorkloadSpec`
- `DUTConfig`
- `MeasurementBoundary`
- Stage 1A `RunConfig`
- Request/Completion/Stress/Manifest/Metric records
- `input_family_digest`
- `input_selection_digest`
- `items_digest`
- `stress_axis_digest`
- `stress_level`
- comparison-key record

Stage 1A ExperimentPlan에는 `evaluation_level` field를 넣지 않는다. Manifest에는 `benchmark_level=model`을 상수로 기록한다.

### 2.2 Input fixtures

#### Dataset fixture

- source timing 없음
- deterministic source item ID와 payload
- annotation은 최소 fake reference 또는 null
- `asap`, `fixed_rate` 허용
- `recorded` validation 거절
- 같은 source/spec에서 동일 `items_digest`

#### Recorded Scenario fixture

- typed SourceTiming
- 최소 2개 episode 또는 episode-gap unit fixture
- timestamp jitter가 있는 schedule fixture
- source timing 역행 거절 fixture
- `recorded`, `fixed_rate`, `asap` schedule 계산 가능
- 원본에 없는 episode/sensor/timestamp를 생성하지 않음

### 2.3 Fake workload와 FakeDUT

- deterministic preprocess/postprocess/codec
- blocking FakeDUT 하나
- fixed service-time model
- table-driven service-time/error model
- deterministic output에 request ID와 item digest를 포함
- `prepare()` 이후에만 `prepare_input()` 허용
- InputProvider는 prepared/DUT object를 소유하지 않음
- async API는 구현하지 않음

### 2.4 Stress test doubles

- `NoStressProvider`
- virtual `FakeStressProvider`
- lifecycle order 검증
- prepare/activate/verify/deactivate/restore failure injection
- activate/verify 실패 시 measure request 0건
- 모든 종료 경로에서 restore record

실제 OS process, background workload, utilization, independent hardware evidence는 구현하지 않는다.

### 2.5 Virtual Runner

- deterministic virtual clock
- `asap`, `fixed_rate`, `recorded`
- `max_inflight=1`
- queue capacity 0
- `drop_new`
- completed/drop/error/timeout/late-completion semantics
- timeout 후 underlying request completion까지 slot 유지
- deterministic cleanup/finalize
- warmup/cooldown/preconditioning/cancellation/watchdog 없음

### 2.6 Observation/Evidence

필수 파일:

```text
experiment.json
<run_id>/
├── run_spec.json
├── input_items.jsonl
├── events.jsonl
├── stress.jsonl
├── outputs.bin
├── manifest.json
└── evaluations/<eval_id>/
    ├── metrics.json
    └── eval_manifest.json
```

필수 evidence:

- input family/selection/items digest
- stress axis/level
- measurement boundary ID
- source-tree SHA-256
- raw/output/evaluation file SHA-256
- execution/measurement/stress status
- `benchmark_level=model`
- `execution_kind=mock`
- output-identity reference identity

Recorder는 Stage 1A에서 deterministic append writer로 구현한다. writer thread/queue는 만들지 않는다.

### 2.7 Evaluator

- count/rate
- request-completion latency
- DUT-service latency
- output-ready latency
- deadline miss
- output identity diagnostic
- planned percentile censoring status/fraction
- raw bundle 재평가

제약:

- censored result는 `value=null`
- censored result에 숫자 Δ 또는 p-value를 계산하지 않음
- task accuracy/quality evaluator는 Stage 1A 범위가 아님
- DegradationAnalyzer는 만들지 않음

### 2.8 CLI

- `validate`
- virtual `run`
- `evaluate`

`analyze` command는 만들지 않는다.

### 2.9 Environment

- project-local `.venv` 생성 허용
- runtime dependency: PyYAML만
- dev dependency: pytest
- build backend: setuptools
- 새 dependency 추가는 scope-change approval 필요
- git 없이 source-tree SHA-256으로 provenance 기록 가능

## 3. 구현 순서와 work-package gate

### WP1 — Skeleton과 contracts

산출:

- project packaging
- strict YAML loader
- dataclasses/enum/status
- canonical JSON/digest utilities
- architecture import test

Gate:

- unknown field 거절
- 필수 field 누락 거절
- digest 결정성
- forbidden dependency import 없음

### WP2 — Input와 schedule

산출:

- Dataset fixture
- Recorded Scenario fixture
- Input conformance
- schedule calculator

Gate:

- Dataset + recorded 거절
- Scenario source timing 정확한 schedule
- timestamp 역행 거절
- jitter/episode gap expected schedule
- 동일 input의 digest 재현

### WP3 — Workload/DUT와 preparation ownership

산출:

- fake workload
- blocking FakeDUT
- Runner-owned prepared store

Gate:

- `DUT.prepare` 전 `prepare_input` 호출 거절
- InputProvider에 DUT/prepared object 없음
- deterministic output codec/hash
- cleanup idempotence

### WP4 — Virtual Runner와 stress lifecycle

산출:

- request state
- virtual event queue
- NoStress/FakeStress lifecycle
- cleanup/finalize

Gate:

- count identity
- drop/error/timeout/late-completion expected sequence
- verify failure 시 measure 0건
- restore on all paths

### WP5 — Bundle와 evaluator

산출:

- raw bundle writer/reader
- manifest/provenance
- evaluator와 eval manifest

Gate:

- bundle-only evaluation
- raw digest 불변
- 새 eval ID 생성
- expected metric 일치
- censoring rule 일치

### WP6 — Golden/E2E/docs/report

산출:

- golden files
- end-to-end tests
- architecture/measurement/metric/evidence docs
- Stage 1A implementation report

Gate:

- M.3의 모든 S1A 기준 통과
- 미실행/skip 항목 명시
- 금지 범위 파일·dependency 없음

## 4. 최소 golden cases

### G-1 Dataset fixed-rate nominal

- deterministic dataset items
- fixed-rate schedule
- service time이 period보다 짧음
- expected: all completed, no drop/timeout, exact request-completion/DUT-service values

### G-2 Recorded Scenario overload

- source times: 0, 33, 66, 99 ms
- N=1, queue 0, service time 40 ms
- expected:
  - f0 complete
  - f1 drop
  - f2 complete
  - f3 drop
  - completion rate 0.5
  - drop rate 0.5
  - frame-period deadline miss rate 1.0
  - completed DUT-service 40 ms

### G-3 Timeout and late completion

- request timeout 50 ms
- fake service time 100 ms
- expected:
  - request terminal state TIMED_OUT
  - late completion diagnostic
  - count identity에서 late completion은 추가 completed로 계산하지 않음
  - slot은 late completion까지 점유

### G-4 Stress verify failure

- FakeStress verify failure
- expected:
  - MEASURE event 없음
  - execution failed with stress reason
  - restore record 존재

### G-5 Censoring

- planned percentile rank가 dropped/timed-out 영역에 위치
- expected:
  - status `censored`
  - value null
  - censored fraction 기록
  - numeric comparison path 없음

### G-6 Re-evaluation

- 동일 raw bundle을 새 eval ID로 재평가
- expected:
  - canonical metric content 동일
  - raw file digest 불변
  - evaluator/protocol identity 기록

## 5. 완료 기준

Stage 1A는 다음이 모두 충족될 때만 완료다.

1. 같은 plan/seed가 동일 canonical event sequence를 만든다.
2. Dataset/Scenario input semantics가 conformance test로 분리된다.
3. InputProvider와 DUT preparation ownership이 분리된다.
4. golden state/metric이 expected 값과 일치한다.
5. NoStress/FakeStress가 모든 failure path에서 restore를 남긴다.
6. bundle-only re-evaluation이 결정적이다.
7. stress axis와 input family guard key가 잘못된 comparison을 막는다.
8. censored metric은 report-only다.
9. core dependency boundary가 지켜진다.
10. evidence manifest가 완전하다.
11. 모든 결과에 MOCK이 표시된다.
12. Stage 1R/actual/System-level scope leakage가 없다.

## 6. 명시적 금지 사항

다음은 승인 없이 구현하지 않는다.

- vendor SDK import 또는 device access
- realtime driver / `time.sleep` 기반 replay
- async submit/callback
- executor thread pool
- process supervisor, signals, watchdog
- GIL test
- telemetry sampler, `/proc` collector, power/energy
- actual stress process
- bounded prefetch/lookahead
- DegradationAnalyzer, Mann–Whitney, Holm, bootstrap
- plots/dashboard/database
- Autoware/ROS2/System-level
- plugin discovery
- git init, commit, remote push

## 7. 변경 관리

구현 중 다음 상황이 발생하면 범위를 임의로 넓히지 말고 design-change proposal을 제출한다.

- InputProvider가 DUT object를 가져야 한다고 판단되는 경우
- async abstraction이 Stage 1A에 필요하다고 판단되는 경우
- thread/process가 없으면 test를 작성할 수 없다고 판단되는 경우
- 새로운 runtime dependency가 필요한 경우
- raw event schema를 변경해야 하는 경우
- golden expected result를 변경해야 하는 경우
- Stage 1A 완료 기준을 완화하려는 경우

Proposal에는 문제, 최소 재현, 대안 2개 이상, measurement validity 영향, 추가 파일/dependency, 권장안을 포함한다.

## 8. Stage 1A 완료 보고서 형식

```text
1. 승인 범위 대비 구현 목록
2. 생성/변경 파일 목록
3. architecture와 lifecycle 요약
4. measurement boundary
5. test 명령과 pass/fail/skip 수
6. golden case별 expected vs actual
7. sample run bundle tree와 SHA-256
8. re-evaluation 일치 증거
9. dependency/import 검사
10. 알려진 limitation
11. 미구현 항목과 이유
12. Stage 1R/Stage 2a로 넘길 질문
13. source-tree SHA-256
```

## 9. 다음 gate

```text
Stage 1A implementation/report
        │
        ├─ reviewer acceptance
        │
        └─ Stage 2a SDK investigation approval
                 │
                 ▼
        Stage 1R scope decision
```

Stage 1A 완료가 Stage 1R 자동 승인을 의미하지 않는다.
