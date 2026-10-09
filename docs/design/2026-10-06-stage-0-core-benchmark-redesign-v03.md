# Stage 0 설계 보고서 — SDV-BMT 독립 NPU benchmark 재정의

> **공개판.** 2026-10-06에 승인된 Stage 0 설계 v03 원본(SHA-256 `29846782acafe875ffc357b47482d57174b0c33edc77f97701f15feecb267794`)과 설계 내용이 같다. 원본에서 바꾼 것은 다음뿐이다. (1) D.6 표에서 benchmark와 관계없는 개발 환경 파일 한 행을 뺐다. (2) 승인 기록의 표현을 프로젝트 결정 표현으로 바꿨다(머리 표의 승인 근거, 표기 범례의 `[승인]`, 승인 이력 표, git 작업 규칙 한 줄, 부록 A의 마지막 항목). 코드와 문서의 "design v03" 인용은 이 설계를 가리킨다. 같은 디렉터리의 Stage 1A 구현 승인서에 적힌 '기준 설계 SHA-256'은 원본의 값이다.

| 항목 | 내용 |
|---|---|
| 단계 | Stage 0: 조사·설계. 설계 lineage `core-benchmark-redesign`의 세 번째 보고서(v03) |
| 작성일 | 2026-10-06 |
| 상태 | **Stage 0 설계 승인. Stage 1A 구현 승인.** Stage 1R은 Stage 2a SDK 조사 후 별도 승인한다. 이 문서 작성 시점에는 구현, 설치, build, venv 생성, git init, 장비 실행, stress 실행을 하지 않았다 |
| 이전 설계 | 직전 `2026-10-06-stage-0-core-benchmark-redesign-v02.md`를 포함한 모든 이전 설계 문서는 수정하지 않고 보존한다. 제품 방향과 현재 승인 범위는 이 v03이 대체한다 |
| 승인 근거 | 2026-10-06 프로젝트 결정: Stage 1A / Stage 1R 분할안 승인 및 다음 단계 진행. v02 검토의 M-1–M-7을 Stage 1A 착수 전제조건으로 반영 |
| 이번 변경 | v02의 제품 방향을 유지하면서 input preparation ownership, metric boundary, comparison identity, censored metric 처리, output identity 기준을 수정했다. Stage 1을 deterministic core(Stage 1A)와 runtime-specific realtime harness(Stage 1R)로 분리했다 |

---

## 첫 페이지 요약

### 1. 제품 정의와 현재 core question

> **SDV-BMT는 독립 NPU/SDV benchmark suite다. 제품 구조는 (1) 단일 model/runtime/NPU를 측정하는 Model-level Evaluation과 (2) Autoware 같은 vehicle software stack에서 driving performance를 보는 System-level Evaluation의 두 계층을 갖는다. 현재 구현은 Model-level에 한정하며, Dataset baseline → Driving Scenario nominal → 검증된 Hardware Stress sweep → Scenario × Stress matrix 순서로 진행한다. System-level은 Phase 2 roadmap에 두고 현재 contract를 억지로 일반화하지 않는다.**

현재 핵심 scientific question은 다음이다.

> NPU, 모델, input/replay 설정을 고정했을 때, 실제 적용이 확인된 hardware stress 수준에 따라 Model-level 성능·출력 품질·장치 상태가 nominal 대비 어떻게 변하며, 그 변화가 driving scenario에 따라 다른가?

### 2. 평가 구조

| 축 | 값 | 의미 |
|---|---|---|
| Evaluation level | Model-level / System-level | 측정 대상의 경계. Stage 1A–6은 Model-level, System-level은 Phase 2 |
| Input kind | `dataset` / `driving_scenario` | 독립 sample 집합인지, source timing과 sequence 의미가 있는 driving input인지 |
| Hardware condition | `nominal` / `stress(axis, level)` | 의도적 stress가 없는지, 검증된 특정 자원 stress가 적용되었는지 |

사용자 관점의 평가 이름은 다음처럼 해석한다.

```text
Dataset evaluation
= Model-level + Dataset input + Nominal

Scenario evaluation
= Model-level + Driving Scenario input + Nominal

Hardware Stress evaluation
= Model-level + {Dataset or Driving Scenario} + Stress

Scenario × Stress degradation
= Model-level + Driving Scenario input + Stress
```

Dataset/Scenario는 input semantics이고 Stress는 hardware condition이다. 셋을 하나의 상호 배타적 `evaluation_mode`로 합치지 않는다.

### 3. 승인된 Model-level architecture

```text
InputProvider
├── DatasetProvider
└── DrivingScenarioProvider
          │ raw payload / source identity / optional source timing
          ▼
BenchmarkRunner ──► Workload.preprocess ──► DUTAdapter.prepare_input
       │                                           │
       │                                           ▼
       ├────────► HardwareStressProvider ───────► DUT execution
       │                                           │
       └────────► ObservationRecorder ◄────────────┘
                         │
                         ▼
                     run bundle
                         │ read-only
                         ▼
              MetricEvaluator
                         │
                         ▼
              DegradationAnalyzer
```

책임 경계는 다음처럼 고정한다.

- InputProvider는 source/index/payload만 책임지고 DUT object를 소유하지 않는다.
- `Workload.preprocess`와 `DUTAdapter.prepare_input`은 DUT 준비 이후 Runner가 조정한다.
- Stage 1A는 deterministic virtual clock과 blocking FakeDUT 하나만 구현한다.
- 실제 blocking/async completion model, thread/process 구조, issue-lag floor는 Stage 2a 조사 후 Stage 1R에서 필요한 경로만 구현한다.
- Evaluator와 Analyzer는 run bundle만 읽는다.
- core는 MLPerf, CARLA, Autoware, ROS2, vendor SDK, NumPy, Torch에 의존하지 않는다.

### 4. v02 검토에서 반영한 필수 수정

| ID | 반영 내용 |
|---|---|
| M-1 | InputProvider에서 DUT 변환 callable을 제거했다. `Input.open → DUT.prepare → PREPARE_INPUTS` 순서로 수정했다 |
| M-2 | `NPU latency`라는 과도한 이름을 제거했다. request-completion, DUT-service, output-ready 경계를 분리했다 |
| M-3 | stress/scenario factor의 axis identity와 level/selection을 분리했다. comparison guard는 level/selection만 제외한다 |
| M-4 | censored percentile은 report-only diagnostic으로 두고 숫자 Δ·검정에서는 제외한다 |
| M-5 | `SourceTiming`과 `ScenarioCoordinates`를 명시적 optional typed field로 정의했다 |
| M-6 | output identity 기준을 사전 지정된 eligible nominal reference 또는 external golden으로 바꿨다 |
| M-7 | Stage 1 ExperimentPlan에서 `evaluation_level` option을 제거했다. manifest에 `benchmark_level=model` 상수를 기록한다 |

### 5. 승인된 개발 단계

| 단계 | 범위 | 승인 상태 |
|---|---|---|
| 0 | architecture와 benchmark design | **승인 완료** |
| 1A | deterministic semantics/evidence core: input fixture 2종, blocking FakeDUT, virtual Runner, NoStress/FakeStress, bundle, offline evaluator | **구현 승인** |
| 2a | 대상 NPU SDK·도구·기존 코드 조사. 장비 실행 없이 completion model, input buffer, measurement boundary, cross-check 방법 결정 | 별도 조사 승인 필요 |
| 1R | 실제 SDK 특성에 맞는 realtime harness. blocking 또는 async 중 필요한 경로만 구현 | **Stage 2a 후 별도 승인** |
| 2b | 실제 NPU + workload + Dataset nominal baseline | Stage 1R 통과 후 |
| 3 | Driving Scenario 2–3종 nominal | Stage 2b 통과 후 |
| 4a/4b | stress 가능성 조사 → verified stress sweep | Stage 3 이후 |
| 5 | Scenario × Stress matrix | Stage 4b 이후 |
| 6 | 두 번째 workload 또는 accelerator | Stage 5 이후 선택 |
| Phase 2 | Autoware/System-level 설계와 driving impact 연결 | Stage 5 이후 별도 승인. Stage 6은 필수 선행조건이 아님 |

### 6. 로드맵

```text
Stage 0 — Design
  └─ 승인 완료
       │
       ▼
Stage 1A — Deterministic Core
  input → virtual execution → raw observation → offline evaluation → evidence
       │
       ├───────────────┐
       │               ▼
       │          Stage 2a — SDK Investigation
       │          completion model / buffers / boundary / profiler
       │               │
       └───────────────┤
                       ▼
Stage 1R — Runtime-specific Realtime Harness
  blocking executor 또는 async submit/callback 중 필요한 경로만 구현
                       │
                       ▼
Stage 2b — Real NPU + Dataset Baseline
  correctness / request-completion / DUT-service / configured-N throughput
                       │
                       ▼
Stage 3 — Driving Scenario Nominal
  replay fidelity / scenario sensitivity characterization
                       │
                       ▼
Stage 4 — Verified Hardware Stress Sweep
  requested / achieved / independent evidence / restore
                       │
                       ▼
Stage 5 — Scenario × Stress
  Degradation Map / Stress-Response Curve / interaction
              ┌────────┴────────┐
              ▼                 ▼
Stage 6 — Generalization   Phase 2 — System-level Design
2nd workload/NPU          Autoware boundary / pipeline & driving metrics
```

### 7. 승인 결정

- [x] **DEC-1 제품 정의와 범위**
  - Model-level / System-level 두 evaluation level
  - Dataset / Driving Scenario input 분리
  - Stress는 input과 직교하는 hardware-condition 축
- [x] **DEC-2 Model-level architecture**
  - InputProvider는 DUT를 모른다
  - Runner가 input preparation과 DUT lifecycle을 조정한다
  - System-level generic target abstraction은 만들지 않는다
- [x] **DEC-3 Stage 1 분할**
  - Stage 1A deterministic core
  - Stage 2a SDK 조사
  - Stage 1R runtime-specific harness
- [x] **DEC-4 Stage 1A 구현 범위**
  - virtual clock, blocking FakeDUT, N=1, queue 0, drop_new
  - raw bundle과 offline re-evaluation
- [x] **DEC-5 metric boundary와 censoring**
  - request-completion / DUT-service / output-ready 분리
  - censored metric은 report-only
- [x] **DEC-6 comparison identity**
  - stress axis vs level
  - scenario/input family vs selection
- [x] **DEC-7 환경 기본값**
  - `.venv`, PyYAML, pytest, setuptools 사용 가능
  - git은 Stage 1A blocker가 아니며 source-tree SHA-256을 기록
  - commit/remote 작업은 별도 승인
- [ ] **정보 요청 Q-1–Q-11**
  - Stage 2a 이후를 위해 필요하다. Stage 1A를 막지 않는다.

### 8. Stage 1A 착수 승인

Stage 1A의 승인 목적은 framework 확장이 아니라 다음 causal path의 유효성을 증명하는 것이다.

```text
typed input
→ deterministic schedule
→ deterministic FakeDUT execution
→ append-only raw observation
→ offline metric evaluation
→ integrity-verifiable evidence bundle
```

Stage 1A에서 realtime thread/process, async adapter, host telemetry, actual stressor, DegradationAnalyzer, 통계 검정, Autoware/ROS2 코드는 만들지 않는다.

> 이 v03은 Stage 0의 최종 승인 설계다. Stage 1A만 구현 승인되었으며, Stage 1R과 그 이후 단계는 해당 gate를 통과한 뒤 별도 승인한다.

---

## 0. 표기와 이번 조사

### 0.1 표기

| 표기 | 의미 |
|---|---|
| **[사실]** | 제공된 문서나 명령으로 직접 확인한 내용 |
| **[승인]** | 승인된 설계 결정 또는 구현 범위 |
| **[후보]** | 후속 gate에서 결정할 수 있는 안. 현재 구현 승인 아님 |
| **[가정]** | 실제 SDK/장비/데이터 확인 전의 전제 |
| **[미검증]** | 실행 또는 cross-check가 필요한 동작/값 |

이 v03의 normative architecture와 Stage 1A 범위는 승인된 설계다. 다만 동작과 성능은 아직 구현·실행으로 검증되지 않았다. Stage 1R 이후 항목은 해당 절에 명시된 후속 승인 대상이다.

- ID 규칙: DEC는 결정, Q는 정보 요청, EQ는 연구·공학 질문, S1A/S1R/S<n>은 단계 완료 기준이다.

### 0.2 이번 v03에서 확인한 사실

| 항목 | 결과 |
|---|---|
| 기준 문서 | v02 설계 문서와 v02 검토/Stage 1 승인안 |
| 보존 | v01, v02, 검토 문서를 수정하지 않았다 |
| 승인 | Stage 1A/1R 분할안이 승인되었고 다음 단계로 진행하기로 결정되었다. M-1–M-7은 승인된 Stage 1A 범위의 필수 설계 수정으로 반영했다 |
| 구현 상태 | source code, venv, build, git, SDK, device, stress 실행은 시작하지 않았다 |
| 무결성 | 입력 문서 SHA-256은 부록 B에 기록했다 |

v02에 기록된 host/device/package 사실은 이 v03에서 다시 검사하지 않았다. Stage 2a 착수 시 실제 대상 host와 SDK 상태를 재확인해야 한다.

### 0.3 하지 않은 것

- repository source 생성·수정
- dependency 설치, venv/build/test 실행
- git init/commit/remote 작업
- vendor SDK import·문서 조사·장비 실행
- 데이터/모델 탐색 또는 다운로드
- telemetry/stress 실행
- Autoware/ROS2/System-level 설계 착수

---

## A. Benchmark 목적과 연구·공학 질문

### A.1 용어

| 용어 | 정의 |
|---|---|
| Evaluation Level | 측정 대상의 경계다. `model`은 단일 model/runtime/NPU 경계, `system`은 Autoware 같은 vehicle software stack 경계다. Stage 1A–6은 Model-level만 구현한다 |
| Input | benchmark가 DUT에 제공하는 결정적 item 집합 또는 sequence다. `input_kind`는 `dataset` 또는 `driving_scenario`다 |
| Dataset Input | 독립 sample들의 결정적 집합/순서다. 원본 timestamp나 episode 의미를 요구하지 않는다. 기본 dataset quality/performance baseline에 쓴다 |
| Driving Scenario | 녹화된 실제 입력 sequence다. episode·sensor·frame 식별자와 가능한 경우 원본 timestamp를 보존한다. 시간 패턴과 sequence 의미가 benchmark 변수다 |
| Hardware Stress | DUT 실행과는 별도로 특정 hardware 자원에 의도적으로 가하는 통제된 부하다. 종류, 대상 자원, 강도(level)로 정의한다 |
| cell | (input, stress level, workload, DUT configuration)의 조합 하나다. replay policy와 run 설정은 experiment 안에서 고정한다 |
| run | cell을 한 번 실행한 것이다. 반복과 통계의 단위다 |
| M(i, s) | input i와 stress level s로 이루어진 cell에서, run마다 얻은 metric 값의 집합 |
| 열화 Δ_M(i, s) | M(i, s)의 대푯값에서 M(i, 0)의 대푯값을 뺀 값이다. s=0은 nominal이다. 상대값도 함께 보고한다 |

**평가 구조.** Dataset/Driving Scenario는 input semantics이고 Hardware Stress는 hardware condition이다. `Dataset evaluation`, `Scenario evaluation`, `Stress evaluation`을 하나의 상호 배타적 mode로 만들지 않는다. 같은 Dataset input에도 stress를 적용할 수 있고, Driving Scenario input에도 nominal/stress를 적용할 수 있다.

### A.2 최소 질문 (질문 1)

> NPU, 모델, replay 정책을 고정한다. 실제 적용이 확인된 hardware stress의 수준을 올리면, 주어진 driving scenario에 대한 NPU-level 성능, 출력 품질, 장치 상태가 nominal 대비 **얼마나, 어떤 모양으로** 변하는가? 그리고 그 변화가 **scenario에 따라 다른가?**


이 질문이 SDV-BMT의 현재 핵심 degradation question이다. 그 전에 두 개의 baseline question을 순서대로 확인한다.

- **Dataset baseline:** nominal hardware condition에서 모델/NPU의 기본 output quality, latency, throughput, telemetry를 신뢰할 수 있게 측정할 수 있는가?
- **Scenario baseline:** nominal hardware condition에서 driving scenario의 시간 패턴이나 workload characteristic이 model/NPU behavior를 실제로 바꾸는가? 바꾸지 않는다면 scenario effect 없음 자체를 결과로 기록한다.

| ID | 세부 질문 | 산출물 |
|---|---|---|
| EQ-1 | Δ가 run 간 변동보다 큰가(탐지) | Degradation Map의 판정 |
| EQ-2 | stress 수준에 따라 Δ가 어떻게 변하는가(시작 수준, 기울기, 급변점) | Stress-Response Curve |
| EQ-3 | Δ가 scenario마다 다른가(상호작용) | scenario 간 Δ 비교, 차이의 차이(DiD) |
| EQ-4 | 어디서 시간이 늘었는가(issue lag, service time, drop, 장치 상태) | metric 분해, telemetry Δ |
| EQ-5 | (요구치가 주어졌을 때만) scenario별로 요구치를 만족하는 stress 범위는 어디까지인가 | Robustness Envelope 후보 |

### A.3 요청 §12의 12개 질문에 대한 답

| # | 질문 | 답 | 상세 |
|---|---|---|---|
| 1 | 최소 scientific/engineering question | A.2. 산출물은 Δ의 크기와 모양, 그리고 scenario에 따른 차이다 | A.2 |
| 2 | 두 축을 왜 독립으로 두는가 | (a) 원인이 다르다. scenario는 입력 측(무엇을, 어떤 시간 패턴으로)이고, stress는 플랫폼 측(자원 가용량)이다. <br>(b) 각각 따로 통제하고 검증한다. scenario는 입력 digest로, stress는 적용 증거로 확인한다. <br>(c) 한 축만 바꾼 비교여야 원인을 그 축에 돌릴 수 있다. "야간+고부하"처럼 한 정의에 섞으면 열화를 어느 쪽 탓으로도 돌릴 수 없다. <br>(d) 작은 catalog 두 개를 조합해 matrix를 만들고, 다른 장비에서도 재사용할 수 있다 | F, G, J.3 |
| 3 | 결합해서 얻는 추가 정보 | **상호작용**이다. 같은 stress라도 scenario의 도착 간격(여유 시간)과 입력 내용에 따라 달라지는 작업량에 따라 영향이 비선형으로 달라진다. <br>예(E.5): frame 주기 33 ms에서 service time이 25 ms에서 40 ms로(+60%) 늘면, completed request-completion latency는 15 ms만 늘지만 drop은 50%, deadline miss는 100%가 된다. 같은 stress가 주기 100 ms인 scenario에서는 miss를 만들지 않는다. <br>단일 축 측정(asap timing 하나)으로는 이 차이가 보이지 않는다. robustness 범위도 scenario별로만 정의할 수 있다 | E.5, J.2 |
| 4 | 무엇이 "useful analysis"인가 | 다음 여섯 가지를 모두 갖춘 결과다. <br>① 반복과 불확실성을 근거로 run 간 변동 대비 판정이 있다. <br>② 한 축만 다른 비교다(guard 통과). <br>③ stress 적용이 독립 증거로 확인되었고 측정 구간 전체를 덮는다. <br>④ 3수준 이상의 용량-반응이 있다. <br>⑤ 기전 단서가 있다(issue lag, service time, drop, telemetry로 분해). <br>⑥ 원본에서 다시 계산해도 같은 값이 나온다. <br>"분해능 X 안에서 열화 없음"도 유효한 결과다 | J |
| 5 | BenchmarkRunner의 최소 책임 | plan 검증, lifecycle 조정, prepared-input ownership, schedule/admission, 요청–입력–완료–출력 대응, cleanup, bundle 마감이다. metric 계산, cell 비교, dataset/task/vendor 세부 해석은 하지 않는다 | C.2, E |
| 6 | blocking/async runtime에서 timestamp replay | Stage 1A는 virtual clock으로 schedule/admission semantics를 먼저 고정한다. Stage 2a에서 실제 completion model을 확인한 뒤 Stage 1R에서 blocking executor 또는 async submit/callback 중 필요한 경로만 구현한다. actual run에서는 issue lag로 replay fidelity를 판정한다 | C.3, E.5, E.8 |
| 7 | portable metric과 NPU 고유 telemetry | 장비 사이에 의미가 같은 것만 `bmt.*` metric(정의 version 포함)으로 둔다. vendor 값은 `tel.<source>.*`로, 원값과 metadata(단위, 출처, 주기, 측정 경계, clock)를 기록한다. <br>Δ는 같은 source끼리만 계산하고, 장비 간 비교에는 쓰지 않는다. 지원하지 않으면 `unavailable`이다 | H |
| 8 | stress intensity 정의와 적용 확인 | 세 층을 나눠 기록한다. 요청(제어 변수와 level), 달성(stressor 자기 보고), 독립 증거(stressor가 아닌 출처, 예: vendor telemetry, kernel 계정). <br>측정 전 검증 구간에서 요청 대비 달성과 독립 증거를 확인하고, 측정 구간 내내 유지되었는지(coverage)를 판정한다 | G.3, G.4 |
| 9 | nominal/stress 비교에서 고정할 것 | DUT 식별과 runtime·driver·firmware, compiled artifact·precision·batch, 측정 경계, input 목록 digest, replay·overload·timeout·warmup, telemetry·출력 기록 설정, host CPU 배치, code revision, metric 정의 version. <br>이 값들로 comparison key를 계산하고, 다르면 비교를 거절한다 | J.3 |
| 10 | benchmark 자체 overhead 확인 | Stage 1A는 deterministic semantics를 검증한다. Stage 1R에서 target-host harness floor와 recorder 비용을 측정하고, Stage 2b에서 telemetry/output capture A/B와 vendor boundary cross-check를 수행한다 | M.2, M.4–M.5 |
| 11 | 첫 결과로 가치를 입증하려면 | 검증 증거를 갖춘 3×4 Degradation Map과 scenario별 response curve, 그리고 (a) scenario 의존 열화 또는 (b) 명시된 detectable effect 범위 안에서 열화가 없다는 결론 중 하나다. 어느 쪽이든 raw bundle 재평가가 일치해야 한다 | K.4–K.5 |
| 12 | 지금 만들지 않을 것 | MLPerf 호환, 현재 단계의 System-level/closed-loop·safety 구현, plugin discovery, DB·web·DSL, 범용 thread framework, 여러 overload 정책, fault·DVFS·thermal 조작, envelope 공식화, 장비 간 종합 점수 | B.4, O.4 |

---

## B. 제품 범위와 현재 구현 범위

### B.1 제품 구조

SDV-BMT의 장기 제품 구조는 두 evaluation level이다.

| Evaluation level | 대상 | 대표 결과 | 구현 시점 |
|---|---|---|---|
| Model-level | 단일 model + runtime/driver + NPU 경계 | latency, throughput, deadline miss, output quality, power/energy(실측 시), NPU telemetry, degradation | **Stage 1A–6** |
| System-level | Autoware/vehicle software stack 경계 | pipeline timing, task/driving performance, model-level degradation의 system impact | **Phase 2** |

두 level의 metric을 하나의 종합 score로 합치지 않는다. System-level에서 성능이 악화되어도 NPU가 원인이라고 자동 귀속하지 않는다. CPU scheduling, ROS2 communication, pipeline backlog 등 다른 원인이 존재할 수 있기 때문이다.

### B.2 현재 1차 구현 범위: Model-level

| 영역 | 이번 제품이 하는 일 |
|---|---|
| 1 Input | `DatasetProvider` 또는 `DrivingScenarioProvider`가 결정적 입력을 제공한다. Dataset은 sample-oriented, Driving Scenario는 sequence/timestamp-oriented semantics를 유지한다 |
| 2 Hardware Stress | 실제 자원에 가하는 통제된 부하를 선언, 적용, 확인, 관측, 해제, 복구하고 그 증거를 남긴다 |
| 3 NPU / DUT | vendor runtime을 얇은 adapter로 연결한다. 요청을 실행하고 출력을 받으며, 선택적으로 telemetry를 제공한다 |
| 4 Model/NPU Metrics & Degradation Analysis | 원본 관측에서 portable metric, quality, telemetry 요약을 계산한다. nominal baseline, stress Δ, map, response curve를 만든다 |

Stage 2는 Dataset input으로 기본 baseline을 먼저 확립하고, Stage 3에서 Driving Scenario input을 추가한다. Stage 4의 stress는 input kind와 직교하며, Stage 5에서 Driving Scenario × Stress matrix를 핵심 결과로 만든다.

### B.3 Phase 2를 위해 지금 남기는 것

- 요청별 실제 prediction: task output 원본, 또는 참조와 sha256
- 입력 식별: `input_id` / `item_index`와 request의 대응
- run-clock 기준의 `t_complete`와 `t_output_ready`(actual run에서는 monotonic)
- DUT/runtime/host provenance와 measurement boundary

위 값은 Model-level 결과와 향후 System-level observation을 연결할 때 필요한 최소 evidence다. 그 밖의 Phase 2 기능(Autoware adapter, ROS2 listener, live input, simulator, driving evaluator)은 만들지 않는다. `system_impact/`와 `safety/` package도 지금 만들지 않는다.

### B.4 현재 구현의 Non-goals

- MLPerf와 MLPerf Automotive 호환, LoadGen·QSL·SUT 개념, 공식 제출 형식.
  - 기존 MLPerf/AI-BMT 코드는 이후 integration 안에서 모델 실행이나 evaluator 구현으로만 재사용할 수 있다.
- plugin 개발 목표, plugin discovery, entry point scan.
- **Stage 1A–6에서의** Autoware/ROS2 integration, 차량 closed-loop, System Impact, Safety KPI, 범용 acceptance criterion 엔진.
  - System-level Evaluation 자체는 제품 roadmap의 Phase 2이며 영구 non-goal이 아니다.
- Model-level과 System-level metric을 섞은 단일 종합 점수.
- 추측 기반 `BenchmarkTarget`/범용 SystemAdapter abstraction. 실제 Phase 2 integration 요구가 생긴 뒤 contract를 정한다.
- HW fault injection, DVFS, thermal 조작을 포함한 장비 설정 변경 일체.
- 별도의 power 계측 platform. 이미 있는 telemetry만 쓴다.
- microservice, database server, web dashboard, workflow DSL, 분산 실행.
- 장비 간 순위나 단일 종합 점수.
- LLM과 token 단위 metric.

## C. Architecture와 module 책임

### C.1 Stage 1A Model-level data flow

```text
VALIDATE / PREFLIGHT
├── resolved plan과 unknown field 검사
├── Input spec/index/digest와 replay compatibility 검사
├── DUT capability와 measurement boundary 검사
└── Stress.check (side-effect free)
        │
        ▼
OPEN BUNDLE
→ INPUT.open
→ DUT.prepare
→ PREPARE_INPUTS
    ├── InputProvider.payload(item)
    ├── Workload.preprocess(payload, item)
    └── DUTAdapter.prepare_input(task_input, item)
→ STRESS.prepare / activate / verify
→ MEASURE (virtual clock)
    ├── schedule / admission / request state transition
    ├── blocking FakeDUT.run
    ├── Workload.postprocess / encode
    └── append-only observation
→ virtual DRAIN
→ STRESS.deactivate / restore
→ DUT.cleanup
→ INPUT.release
→ FINALIZE
        │
        ▼
MetricEvaluator(bundle only) → MetricResult
```

Stage 1A의 `PREPARE_INPUTS` 결과는 Runner가 소유한다. InputProvider는 device-specific object를 저장하지 않는다. 작은 fixture 전체를 eager preparation하는 경로 하나만 지원한다.

Stage 1R에서는 Stage 2a 조사 결과에 따라 실제 run clock, blocking executor 또는 async submit/callback, recorder queue, process isolation을 필요한 범위에서만 추가한다. System-level data flow는 Phase 2에서 별도 정의한다.

### C.2 module 책임

| module | 책임 | 하지 않는 것 | 첫 구현 |
|---|---|---|---|
| InputProvider | input identity, 결정적 item 순서, source/index validation, raw payload 접근, release. `input_kind`와 timing capability를 선언한다 | task preprocessing, device input 생성, scheduling, stress/DUT 호출 | Stage 1A protocol |
| DatasetProvider | timestamp를 scheduling 의미로 요구하지 않는 sample 집합을 결정적으로 제공한다 | 가짜 timestamp/episode 생성, recorded replay 의미 부여 | Stage 1A fixture, Stage 2b 실제 |
| DrivingScenarioProvider | sequence/episode/stream identity와 검증 가능한 source timing을 보존한다 | `t_sched` 계산, 원본에 없는 timing 생성 | Stage 1A fixture, Stage 3 실제 |
| Workload task code | raw payload를 task input으로 전처리하고 raw DUT output을 task output으로 후처리·직렬화한다 | device/runtime 호출 | Stage 1A fake workload, Stage 2b 실제 |
| DUT Adapter | runtime/model 준비, 준비된 task input의 DUT-specific 변환, 실행, output 접근, cleanup, 실제 단계의 optional telemetry | scheduling, input selection, stress 로직, metric 계산 | Stage 1A blocking FakeDUT, Stage 1R/2b 실제 |
| HardwareStressProvider | capability check, prepare/activate/verify/observe/deactivate/restore와 evidence 기록 | DUT 호출, 결과 판정 | Stage 1A NoStress/FakeStress, Stage 4 실제 |
| BenchmarkRunner | plan validation, lifecycle, input preparation orchestration, schedule/admission, request state, cleanup, bundle finalize | metric/Δ 계산, dataset/task/vendor 세부 해석 | Stage 1A virtual Runner |
| ObservationRecorder | 사실을 append-only로 기록하고 output reference/hash를 보존한다 | metric 집계, 원인 해석 | Stage 1A deterministic writer |
| MetricEvaluator | bundle만 읽고 portable metric과 output identity diagnostic을 계산한다 | DUT 재실행, cell 비교 | Stage 1A |
| DegradationAnalyzer | comparison guard, run-level Δ, curve/map/interaction을 계산한다 | 원본 수정, 재측정 | Stage 4 최소, Stage 5 전체 |

### C.3 단계별 실행 모델

#### Stage 1A

- 단일 process, deterministic virtual clock.
- 실제 thread, process supervisor, signal handler, OS sleep을 쓰지 않는다.
- FakeDUT는 blocking `run()` 하나만 제공한다.
- `max_inflight=1`, queue 0, `drop_new` 하나만 지원한다.
- timeout과 late completion은 virtual event로 재현한다.
- Recorder는 canonical event 순서를 보존하는 단순 append writer다.
- host telemetry, GIL, realtime issue lag는 측정하지 않는다.

#### Stage 1R

Stage 2a에서 실제 SDK를 조사한 후 다음 중 필요한 것만 승인·구현한다.

- blocking SDK이면 issuer와 blocking executor 경로
- async SDK이면 짧은 submit과 callback/poll 경로
- real `CLOCK_MONOTONIC` scheduling과 issue-lag 측정
- bounded recorder writer queue
- 필요한 경우 run-per-process isolation/watchdog
- target host의 harness floor

blocking/async 양쪽을 사전에 모두 만들지 않는다.

#### Stage 4 실제 stress

실제 stressor의 OS process, process group, parent-death 처리, 잔존 process 검사는 Stage 4에서 실제 provider를 만들 때 구현한다. Stage 1A FakeStress는 lifecycle와 실패 주입만 가상으로 검증한다.

---

## D. 최소 contracts와 dependency 방향

### D.1 정의(plan)와 기록(record)

| 개념 | 종류 | 최소 내용 | 만드는 주체 |
|---|---|---|---|
| ExperimentPlan | 정의 | experiment_id, input·stress·workload·DUT 참조, ReplayPolicy, RunConfig, 반복/seed, evaluation 설정 | 사용자 YAML |
| DatasetSpec | 정의 | input_id, version, source dataset와 digest, sample 선택/순서, annotation 참조와 schema | YAML |
| DrivingScenarioSpec | 정의 | input_id, version, source dataset와 digest, scenario 선택, stream schema, source timing field·단위·의미, annotation 참조 | YAML |
| SourceTiming | 기록 | value, unit, semantic(`capture`/`logging`/`unknown`), clock/source identifier | Provider |
| ScenarioCoordinates | 기록 | episode_id, stream_id, sensor_id, frame_id. 원본에 있는 값만 | Provider |
| InputItem | 기록 | item_index, source_item_id, payload_ref+sha256, annotation_ref+digest, `source_timing?`, `scenario_coordinates?`, source_metadata | Provider |
| ReplayPolicy | 정의 | mode(`recorded`/`fixed_rate`/`asap`), speed/rate, lead, episode_gap | YAML |
| HardwareStressSpec | 정의 | stress_id, kind, target_resource, placement, control variable/unit, background workload identity, levels, verification, schedule | YAML |
| WorkloadSpec | 정의 | workload_id, task, artifact sha256, precision, batch, preprocess/postprocess/codec identity, quality evaluator | YAML |
| DUTConfig | 정의 | dut_config_id, adapter factory, device selection, max_inflight, runtime options | YAML |
| MeasurementBoundary | 정의/기록 | boundary_id, source decode/preprocess/prepare_input/H2D 포함 여부, submit/complete/output-ready 의미 | Workload + Adapter, manifest |
| RunConfig | 정의 | Stage 1A에서는 N=1, queue 0, drop_new, virtual timeout/drain, output capture | YAML |
| RequestRecord | 기록 | request_id, item_indices, phase, t_sched, t_issue, t_submit_begin/end, admission | Runner |
| CompletionRecord | 기록 | request_id, result, t_complete, t_output_ready, output_ref, error, optional device timing | Runner |
| StressRecord | 기록 | stress axis/level, phase, run-clock time, requested/achieved/independent evidence, verification, error | Provider |
| OutputReference | 기록 | request_id, file, offset, length, sha256, codec | Recorder |
| RunManifest | 기록 | `benchmark_level=model`, run/cell identity, input/stress axis identity, measurement_boundary_id, execution/status/provenance, file digests | Runner/Evidence |
| MetricResult | 파생 | metric_id, definition_version, unit, status, value, n, window, method, eval_id, boundary_id | Evaluator |
| DegradationResult | 파생 | contrast, baseline/comparison run IDs, metric, Δ, uncertainty, decision, guard result | Analyzer |

Stage 1 ExperimentPlan에는 `evaluation_level` field를 두지 않는다. Model-level Runner가 manifest에 `benchmark_level=model`을 상수로 기록한다. System-level은 실제 contract가 생길 때 새 schema version으로 추가한다.

Deadline은 실행 정책이 아니라 evaluation parameter다. 동일 raw observation에 다른 deadline을 적용할 수 있다.

### D.2 식별자와 factor identity

| 식별자 | 정의 |
|---|---|
| `experiment_id`, `plan_digest` | 기본값까지 해석한 plan의 canonical JSON sha256 |
| `input_kind`, `input_id` | `dataset` 또는 `driving_scenario`와 사용자 식별자 |
| `input_family_digest` | 비교 가능한 input family의 identity. source dataset/version, sensor/stream schema, source timing semantics, task/annotation semantics, preprocessing contract를 포함 |
| `input_selection_digest` | 특정 dataset subset 또는 scenario selection을 해석한 spec digest |
| `items_digest` | 실제 사용한 InputItem identity 목록의 digest. payload와 annotation digest, 필요한 timing/coordinates를 포함 |
| `stress_axis_digest` | stress spec에서 level만 제외한 digest. kind, target resource, placement, control variable/unit, background workload, verification rule, schedule를 포함 |
| `stress_level` | `none`, `L1`, `L2`, `L3` 등 axis 내부 수준 |
| `measurement_boundary_id` | 포함/제외 경계와 timestamp 의미를 canonical화한 digest 또는 versioned ID |
| `cell_digest` | resolved workload/DUT/input family+selection+items/stress axis+level/replay/run config/boundary의 digest |
| `run_id` | 전역 고유 실행 ID. deterministic comparison에는 사용하지 않는다 |
| `request_id` | run 내부 예정 순서 기반 정수 |
| `eval_id`, `analysis_id` | 평가/분석 실행마다 새 ID. 기존 결과를 덮어쓰지 않는다 |

Nominal은 “stress 정보 없음”이 아니라 **특정 `stress_axis_digest`에서 `stress_level=none`**으로 표현한다. 따라서 same-NPU axis의 nominal과 host-CPU axis의 nominal은 분석 계획에서 어느 axis의 기준인지 명확하다.

`cell_digest`에는 spec 이름만 넣지 않는다. resolved spec, realized `items_digest`, annotation/index digest, artifact digest, measurement boundary가 들어가야 한다.

```text
run_id
├── manifest
│   ├── benchmark_level=model
│   ├── input_family_digest / input_selection_digest / items_digest
│   ├── stress_axis_digest / stress_level
│   └── measurement_boundary_id
├── request_id ─► item_index ─► InputItem
├── request_id ─► CompletionRecord ─► OutputReference
└── StressRecord[]
```

### D.3 Stage 1A 최소 interface

```python
class InputProvider(Protocol):
    def describe(self) -> InputDescriptor: ...
    def items(self) -> Sequence[InputItem]: ...     # deterministic, source-derived only
    def open(self) -> None: ...                     # source/index access 준비, idempotent
    def payload(self, item_index: int) -> Payload: ...
    def release(self) -> None: ...                  # idempotent

class Workload(Protocol):
    def preprocess(self, payload: Payload, item: InputItem) -> object: ...
    def postprocess(self, raw_output: object, item: InputItem) -> object: ...
    def encode(self, task_output: object) -> bytes: ...

class DUTAdapter(Protocol):
    def describe(self) -> DUTDescriptor: ...
    def capabilities(self) -> DUTCapabilities: ...
    def prepare(self, workload: WorkloadSpec, config: DUTConfig) -> PreparedInfo: ...
    def prepare_input(self, task_input: object, item: InputItem) -> object: ...
    def run(self, request: DUTRequest) -> DUTResult: ...   # Stage 1A: blocking only
    def cleanup(self) -> None: ...

class HardwareStressProvider(Protocol):
    def check(self, spec: HardwareStressSpec, level: str, host: HostInfo) -> SupportDecision: ...
    def prepare(self) -> StressRecord: ...
    def activate(self) -> StressRecord: ...
    def verify(self, window_s: float) -> StressRecord: ...
    def observe(self) -> StressRecord: ...
    def deactivate(self) -> StressRecord: ...
    def restore(self) -> StressRecord: ...
```

Stage 1A Runner는 다음처럼 준비한다.

```text
for item in provider.items():
    payload = provider.payload(item.item_index)
    task_input = workload.preprocess(payload, item)
    prepared_input = adapter.prepare_input(task_input, item)
    prepared_store[item.item_index] = prepared_input
```

`prepared_store`는 Runner가 소유하고 cleanup 때 폐기한다. InputProvider는 `prepared_input`이나 DUT buffer를 알지 못한다.

Stage 1R에서 실제 SDK가 async인 것으로 확인되면 `submit/callback` 확장 계약을 별도 결정한다. Stage 1A protocol에 사용하지 않는 async method를 미리 넣지 않는다.

### D.4 measurement boundary

모든 actual run은 다음 항목을 명시한다.

| 구간 | 가능한 상태 |
|---|---|
| payload read/decode | included / excluded |
| Workload preprocess | included / excluded |
| DUTAdapter.prepare_input 및 H2D | included / excluded / inseparable |
| runtime submit/execute/complete | 의미를 adapter가 문서화 |
| Workload postprocess | included / excluded |
| output-ready | 어떤 형태의 output이 host에서 사용 가능한지 |

Stage 1A에서는 source read, preprocess, `prepare_input`을 모두 측정 밖에서 수행하고 다음 세 경계를 검증한다.

- request-completion: `t_sched → t_complete`
- DUT-service: `t_submit_begin → t_complete`
- output-ready: `t_sched → t_output_ready`

`device_execution`은 vendor/device timestamp의 의미, clock domain, sync 방법이 문서화되고 external profiler와 cross-check된 경우에만 추가한다.

### D.5 Stage 1A 디렉토리 범위

```text
SDV-BMT/
├── pyproject.toml  README.md  .gitignore
├── src/sdv_bmt/
│   ├── contracts/      ids.py specs.py records.py protocols.py status.py
│   ├── input/          dataset_fixture.py recorded_scenario_fixture.py
│   ├── workload/       fake.py
│   ├── dut/            fake.py
│   ├── stress/         none.py fake.py
│   ├── runner/         plan.py schedule.py core.py virtual.py runner.py
│   ├── observation/    events.py recorder.py reader.py
│   ├── metrics/        catalog.py evaluator.py portable.py identity.py
│   ├── evidence/       bundle.py manifest.py provenance.py keys.py
│   └── cli/            main.py
├── configs/            mock dataset/scenario/workload/DUT/stress/experiment
├── tests/
│   ├── unit/
│   ├── golden/
│   ├── conformance/{input,dut,stress}/
│   ├── architecture/
│   └── e2e/
└── docs/
    ├── architecture.md
    ├── measurement_protocol.md
    ├── metrics_catalog.md
    ├── evidence.md
    └── roadmap.md
```

Stage 1A에서 만들지 않는 디렉토리/모듈:

- `runner/realtime.py`, process supervisor, signal handling
- telemetry sampler/host `/proc` collector
- actual vendor integration
- `analysis/`
- `system/`, `autoware/`, `ros2/`
- plugin discovery

### D.6 dependency와 환경

| 항목 | 결정 |
|---|---|
| Python | `>=3.10` |
| runtime dependency | PyYAML만 |
| dev dependency | pytest |
| build backend | setuptools |
| contract types | frozen stdlib dataclasses와 strict loader |
| evidence | JSON/JSONL + append-only output blob |
| Stage 1A statistics | percentile/count/rate 계산에 필요한 stdlib만. cell 통계 검정은 없음 |
| provenance | git이 있으면 commit/dirty 상태, 없으면 source-tree SHA-256 |
| git | Stage 1A blocker가 아니다. `git init`, commit, remote 작업은 별도 승인 |
| vendor/task dependency | 향후 `integrations/` 전용 환경에 격리 |

의존 규칙:

1. `contracts`는 stdlib만 import한다.
2. `input`, `stress`, `dut`, `workload`는 서로 import하지 않고 `contracts`만 공유한다.
3. `metrics`는 Runner나 DUT를 import하지 않는다.
4. core는 vendor SDK, MLPerf, CARLA, Autoware, ROS2, NumPy, Torch, ONNX Runtime을 import하지 않는다.
5. 자동 plugin discovery를 사용하지 않는다.

---

## E. Runner lifecycle와 timing semantics

### E.1 Stage 1A lifecycle

```text
VALIDATE
  ├─ schema / unknown field
  ├─ input-kind × replay compatibility
  ├─ N=1 / queue=0 / drop_new
  └─ required digest / boundary fields
        │
        ▼
PREFLIGHT
  ├─ Input source/index/digest validation
  ├─ DUT capability check
  └─ Stress.check (side-effect free)
        │
        ▼
OPEN BUNDLE
→ INPUT.open
→ DUT.prepare
→ PREPARE_INPUTS
→ STRESS.prepare
→ STRESS.activate
→ STRESS.verify
→ MEASURE (virtual clock)
→ virtual DRAIN
→ STRESS.deactivate
→ STRESS.restore
→ DUT.cleanup
→ INPUT.release
→ FINALIZE
```

오류가 발생하면 이미 시작한 cleanup 단계를 역순으로 모두 시도하고 각 결과를 기록한다.

- Input source를 연 뒤 DUT를 준비하고, 그 다음 `PREPARE_INPUTS`를 수행한다.
- Stress 적용 전에 DUT와 prepared input이 모두 준비되어야 한다.
- activate 또는 verify가 실패하면 MEASURE에 들어가지 않는다. nominal로 대체하지 않는다.
- NoStress도 같은 lifecycle record를 남긴다.
- Stage 1A에는 warmup, real cooldown, process kill, signal cancellation이 없다.

### E.2 clock과 timestamp

| 기호 | 의미 | Stage 1A clock | 실제 단계 |
|---|---|---|---|
| `t_src` | `SourceTiming.value`를 ns로 변환한 source timestamp | source domain | 동일 |
| `t_sched` | 예정 발행 시각 | deterministic virtual run clock | `CLOCK_MONOTONIC` 기반 run clock |
| `t_issue` | admission 결과가 확정된 시각 | virtual | monotonic |
| `t_submit_begin`, `t_submit_end` | adapter 호출 진입/반환 | virtual | monotonic |
| `t_complete` | Runner가 completion을 관측한 시각 | virtual | monotonic |
| `t_output_ready` | 선언된 host output 형태가 준비된 시각 | virtual | monotonic |
| `t_dev_*` | device 내부 timing. 제공될 때만 | 없음 | adapter/device domain |
| anchor | run clock과 realtime/tool clock의 대응 | 불필요 | (`mono`, `real`) pair |

공통 event field는 `t_run_ns`와 `run_clock_domain`이다.

- Stage 1A는 `run_clock_domain=virtual`.
- Stage 1R/actual run은 `run_clock_domain=monotonic`이고 필요할 때 `t_mono_ns`를 동일 값으로 기록할 수 있다.
- duration은 동일 clock domain끼리만 계산한다.
- source timestamp와 run clock은 ReplayPolicy 식으로만 연결한다.
- device timestamp는 sync 근거 없이 host timestamp와 빼지 않는다.

### E.3 파생 시간과 이름

| metric concept | 정의 | 해석 |
|---|---|---|
| issue lag | `t_issue − t_sched` | Runner/replay 충실도. Stage 1A에서는 deterministic schedule diagnostic |
| dispatch lag | `t_submit_begin − t_issue` | admission 이후 adapter 호출까지 |
| submit call | `t_submit_end − t_submit_begin` | async에서는 submit 비용. blocking에서는 service 구간과 겹침 |
| DUT-service latency | `t_complete − t_submit_begin` | 선언된 DUT/runtime boundary 안의 처리 시간 |
| request-completion latency | `t_complete − t_sched` | 예정 도착부터 completion 관측까지. queue/issue/runtime 영향을 함께 포함할 수 있음 |
| output-ready latency | `t_output_ready − t_sched` | 예정 도착부터 선언된 task output 준비까지 |
| slot occupancy | slot 획득부터 release까지 | admission 자원 점유 |

이 문서에서는 `t_complete − t_sched`를 `NPU latency`라고 부르지 않는다. 순수 device execution time은 검증된 device timestamp가 있을 때만 별도 metric으로 추가한다.

### E.4 Replay policy

| mode | `t_sched` | 지원 input | 용도 |
|---|---|---|---|
| `recorded` | `t0 + floor((t_src,i − t_src,0) × unit_to_ns / speed)` | `driving_scenario`만 | source timing 간격 재현 |
| `fixed_rate` | `t0 + floor(i × 10^9 / rate)` | dataset, driving_scenario | 일정 offered rate |
| `asap` | slot이 비는 시각. `t_sched=t_issue` | dataset, driving_scenario | closed-loop service/throughput baseline |

- DatasetProvider에 `recorded`를 요청하면 validation에서 거절한다.
- Driving Scenario의 모든 required item에 유효한 SourceTiming이 없으면 `recorded`를 거절한다.
- 여러 episode의 `episode_gap`은 명시한다. 원본 timestamp를 수정하지 않는다.
- 여러 stream을 병합하려면 공통 clock 근거가 있어야 한다.
- 동일 input을 다른 replay policy로 실행한 결과는 같은 comparison key로 직접 비교하지 않는다.

### E.5 Stage 1A virtual admission semantics

Stage 1A는 실제 issuer/executor thread를 만들지 않고 아래 semantics를 virtual event로 구현한다.

1. `t_sched`에 slot이 비어 있으면 request를 ISSUED/INFLIGHT로 전환한다.
2. slot이 사용 중이면 `DROPPED(no_free_slot)`로 기록한다.
3. FakeDUT service model이 completion event를 예약한다.
4. virtual timeout이 먼저 발생하면 TIMED_OUT으로 전환한다. underlying fake request가 끝날 때까지 slot은 계속 점유한다.
5. 이후 completion은 late completion으로 기록하고 output-ready/late-completion 처리가 끝난 뒤 slot을 release한다.

예: frame period 33 ms, N=1.

| frame | `t_sched` | nominal service 25 ms | stressed service 40 ms |
|---|---:|---|---|
| f0 | 0 | complete 25 | complete 40 |
| f1 | 33 | complete 58 | drop |
| f2 | 66 | complete 91 | complete 106 |
| f3 | 99 | complete 124 | drop |

stressed case에서는 completed request의 DUT-service가 40 ms로 관측되고, completion coverage는 50%, drop rate는 50%, frame-period deadline miss는 100%다. completed latency 하나만으로 stream-level degradation을 해석하지 않는다.

### E.6 요청 상태

#### Stage 1A 구현 상태

| 현재 상태 | 사건 | 다음 상태 | 기록 |
|---|---|---|---|
| PLANNED | slot 없음 | DROPPED | `dropped(no_free_slot)` |
| PLANNED | slot 있음 | ISSUED | `issued` |
| ISSUED | `run()` 진입 | INFLIGHT | `submit_begin` |
| ISSUED/INFLIGHT | fake adapter/workload 오류 | ERROR | `error(phase, info)` |
| INFLIGHT | completion | COMPLETED | `completed(t_complete, t_output_ready, output_ref)` |
| INFLIGHT | timeout 먼저 발생 | TIMED_OUT | `timed_out` |
| TIMED_OUT | completion 도착 | 변화 없음 | `late_completion` |
| 준비/실행 실패로 발행되지 않음 | — | NOT_ISSUED | `not_issued(reason)` |

계수 항등식:

```text
planned = not_issued + dropped + issued
issued = completed + error + timed_out
```

late completion은 위 계수를 바꾸지 않는다.

#### Stage 1R 이후 후보

`ABANDONED`, signal cancellation, unknown/duplicate callback anomaly, real stall watchdog는 실제 completion model을 확인한 뒤 추가한다. Stage 1A 완료를 위해 미리 구현하지 않는다.

### E.7 Stage 1A 실행 정책

| 항목 | 정책 |
|---|---|
| admission | N=1, queue 0, `drop_new` |
| timeout | virtual `request_timeout_ns` |
| drain | 마지막 schedule 이후 virtual event queue를 `drain_ns`까지 처리 |
| request error | `continue` 하나 |
| cancellation | 미지원 |
| warmup/cooldown/preconditioning | 미지원 |
| output capture | canonical task output 전부 저장 |
| telemetry | 미지원 |

Deadline은 execution state를 바꾸지 않는다. Evaluator가 raw event를 읽어 계산한다.

### E.8 replay validation

Stage 1A의 목표는 **schedule 계산과 state semantics의 결정성**이다.

- 같은 plan/seed에서 canonical `t_sched`와 event sequence가 같아야 한다.
- Dataset + recorded replay는 실행 전에 거절되어야 한다.
- source timestamp jitter와 episode gap이 정확히 schedule에 반영되어야 한다.
- virtual clock이므로 Stage 1A 결과를 realtime replay fidelity로 해석하지 않는다.

Stage 1R에서 실제 host를 사용해 issue-lag floor, sleep/wakeup jitter, GIL 영향, threshold를 측정하고 `replay_fidelity` 판정 규칙을 승인받는다.

### E.9 actual 단계에서 결정할 timing policy

다음은 Stage 2a/Stage 1R 또는 Stage 2b pilot에서 결정한다.

- warmup request 수와 재사용 가능한 input 규칙
- model/runtime preconditioning
- temperature 기반 또는 고정 cooldown
- real request timeout과 stall watchdog
- run-per-process isolation
- recorder/telemetry overhead
- H2D가 포함되는 measurement boundary

---

## F. Input Providers: Dataset와 Driving Scenario

### F.1 역할 분리

```text
InputProvider
├── DatasetProvider
│   └── deterministic samples, no required scheduling timestamp
└── DrivingScenarioProvider
    └── deterministic sequence + optional typed source timing
```

| 구성 | 역할 |
|---|---|
| DatasetSpec | source dataset/version, subset 선택과 순서, annotation schema를 선언 |
| DrivingScenarioSpec | source dataset/version, sequence 선택, stream schema, source timing 의미를 선언 |
| dataset reader/indexer | 원본 dataset 지식을 integration에 격리하고 standard index 생성 |
| InputProvider | index validation, deterministic InputItem, raw payload 접근 |
| ReplayPolicy | InputItem을 schedule에 매핑. Provider는 sleep하거나 `t_sched`를 만들지 않음 |

### F.2 typed InputItem

```text
InputItem
├── item_index
├── source_item_id
├── payload_ref
├── payload_sha256
├── annotation_ref | null
├── annotation_digest | null
├── source_timing: SourceTiming | null
├── scenario_coordinates: ScenarioCoordinates | null
└── source_metadata
```

```text
SourceTiming
├── value
├── unit
├── semantic: capture | logging | unknown
└── clock_source
```

```text
ScenarioCoordinates
├── episode_id | null
├── stream_id | null
├── sensor_id | null
└── frame_id | null
```

원본에 없는 의미를 만들어 넣지 않는다. Dataset sample에는 가짜 timestamp, episode, sensor, frame을 넣지 않는다. Driving Scenario도 원본 또는 검증 가능한 index에 없는 값을 추정하지 않는다.

### F.3 DatasetProvider semantics

- 기본 dataset quality/performance baseline에 사용한다.
- 같은 spec과 source에서 같은 item 순서와 digest를 내야 한다.
- source timestamp가 존재해도 Dataset mode에서는 scheduling 의미로 사용하지 않는다.
- `asap`과 `fixed_rate`를 지원한다.
- `recorded`는 validation에서 거절한다.
- 서로 다른 sample set의 quality 차이를 scenario effect로 해석하지 않는다.

### F.4 DrivingScenarioProvider semantics

- sequence, episode, stream, source timing이 benchmark 의미를 가진다.
- `recorded` replay에는 모든 required item의 유효한 SourceTiming이 필요하다.
- 같은 stream에서 timestamp 역행 또는 허용되지 않은 중복이 있으면 거절한다. 조용히 정렬하지 않는다.
- timestamp 의미가 문서화되지 않았으면 semantic은 `unknown`이다.
- recorded replay는 source timestamp 간격에 따른 **DUT request arrival pattern**을 재현한다. 원 차량의 sensor-to-DUT 전달 지연이나 inference 시작 시각을 재현한다고 주장하지 않는다.
- timestamp가 없으면 `fixed_rate` 또는 `asap`만 사용할 수 있다.

### F.5 source 준비와 prepared input ownership

Stage 1A 준비 순서:

```text
InputProvider.open
→ DUTAdapter.prepare
→ for each InputItem:
      payload = InputProvider.payload
      task_input = Workload.preprocess
      prepared_input = DUTAdapter.prepare_input
      Runner-owned prepared_store에 저장
```

- InputProvider는 source/index/raw payload까지만 책임진다.
- Workload와 DUT-specific preparation은 Runner가 DUT 준비 이후 호출한다.
- prepared input의 lifetime과 release는 Runner/DUT cleanup 책임이다.
- Stage 1A는 작은 fixture 전체를 eager preparation한다.
- bounded lookahead, per-request H2D, device-resident cache는 구현하지 않는다.
- 실제 residency/prefetch 정책은 Stage 2a에서 dataset 크기, SDK buffer lifetime, H2D API를 확인한 뒤 결정한다.
- 각 준비 구간의 측정 포함 여부는 `measurement_boundary_id`에 기록한다.

### F.6 input identity

`input_family_digest`는 비교 가능한 input family를 정의한다.

- source dataset와 version
- sensor/stream schema
- source timing semantics
- task/annotation schema
- preprocessing contract identity

`input_selection_digest`는 특정 subset/scenario selection을 정의한다.

`items_digest`는 실제 사용한 item 목록으로 계산하며 다음을 포함한다.

- item_index와 source_item_id
- payload sha256
- annotation digest
- recorded scenario에서 schedule identity에 영향을 주는 SourceTiming/ScenarioCoordinates

같은 family에서 selection만 다른 Driving Scenario를 scenario contrast로 비교할 수 있다. family가 다르면 기본 scenario contrast를 거절한다.

### F.7 첫 Driving Scenario 선택 기준

- 고정-shape dense model에서는 input content가 DUT-service를 거의 바꾸지 않을 수 있다.
- scenario 차이는 다음 mechanism에서만 주장한다.
  - source timing/offered rate
  - point/token/object count, sparsity, dynamic shape
  - conditional execution/cascade/dynamic batching
  - preprocess/postprocess workload
  - output quality
- 첫 2–3개 scenario는 측정 가능한 workload/timing characteristic이 달라야 한다.
- 환경 label(야간/우천 등)은 source metadata 근거가 있을 때만 쓴다.
- Stage 3에서 scenario sensitivity가 관측되지 않으면 그것을 유효한 결과로 보고한다.

---

## G. Hardware Stress Provider

> **Stage 범위:** Stage 1A는 NoStress와 virtual FakeStress로 lifecycle/실패 semantics만 검증한다. 아래의 actual resource control, independent evidence, OS process cleanup은 Stage 4 provider에 적용한다.

### G.1 최소 lifecycle

| 단계 | 하는 일 | 기록 | 실패하면 |
|---|---|---|---|
| `check(spec, level, host)` | 지원 여부와 그 사유를 판단한다. 부작용이 없다 | `StressRecord(check)` | 미지원이면 plan을 거절한다(실행 0회) |
| `prepare` | stressor를 준비하고(모델 적재 등) idle 기준값을 잰다 | record와 idle 기준값 | run failed. DUT를 측정하지 않는다 |
| `activate` | 부하를 시작한다 | 시작 시각과 요청 수준 | run failed. nominal로 대체하지 않는다 |
| `verify(T_v)` | 요청 대비 달성과 독립 증거를 확인한다 | `verification` 상태 | `failed`이면 측정하지 않고 run failed |
| `observe` | 측정 중 주기적으로 기록한다(살아 있는지, 달성 부하, 오류) | 시계열 | stressor가 사라지면 coverage partial |
| `deactivate` | 부하를 멈추고 종료를 확인한다 | 중단 시각 | 기록한 뒤 restore로 넘어간다 |
| `restore` | 남은 process를 강제 종료하고 idle 복귀를 확인한다. 모든 경로에서 호출하며 멱등이다 | 잔존 0 확인, idle telemetry | 복구에 실패하면 이후 run을 preflight에서 막는다 |

### G.2 HardwareStressSpec

| field | 내용 |
|---|---|
| `stress_id`, `kind` | 예: `same_npu_background`, `host_cpu_contention` |
| `target_resource` | 실제로 부하를 준 자원의 이름. 결과에 그대로 쓴다 |
| `placement` | 부하를 둔 위치. 예: 같은 PE, 다른 PE 묶음, host core 집합 |
| `control` | 제어 변수와 단위 |
| `levels` | level 이름과 값의 대응 |
| `verification` | 검증 구간 길이, 허용오차, 독립 증거의 출처와 규칙 |
| `schedule` | 처음에는 `whole_measurement` 하나만 지원한다 |
| `on_apply_failure` | `abort` 하나만 허용한다 |

`stress_axis_digest`는 위 spec에서 level 값만 제외해 계산한다. kind/resource/placement/control/background workload/verification/schedule이 하나라도 다르면 다른 stress axis다.

### G.3 강도(intensity): 세 층으로 기록

| 층 | 내용 | 예(same-NPU background) |
|---|---|---|
| 요청 | 제어 변수와 수준 | 배경 요청률. calibration에서 잰 배경 단독 최대 처리량 C_bg의 25%, 50%, 100% |
| 달성 | stressor의 자기 보고 | 배경 완료 수/초. 검증 구간과 측정 구간에서 각각 잰다 |
| 독립 증거 | stressor가 아닌 출처 | vendor telemetry의 장치 사용률(Q-3), kernel 계정 등 |

- **response curve의 x축.** 순서형 level과, 검증 구간에서 잰 달성값(연속)을 함께 쓴다.
- **적용 판정의 위치.** 측정 구간에서 배경 달성값이 요청보다 낮아지는 것은 적용 실패가 아니라 경합의 결과일 수 있다.
  - 그래서 적용 판정은 배경만 도는 검증 구간에서 한다.
  - 측정 구간에서는 stressor가 살아서 진행 중인지와 달성값을 기록한다. 이 값은 상호 간섭의 증거다.
- **level 값의 조정.** pilot 뒤에 바꿀 수 있지만, 본 실험 전에 고정하고 그 변경을 기록한다.

### G.4 검증 상태와 coverage

| field | 값 | 규칙 |
|---|---|---|
| `verification` | `verified` / `self_reported` / `unverified` / `failed` / `not_applicable`(nominal) | `verified` = 달성 ≥ (1−tol) × 요청 AND 독립 증거가 규칙을 만족 |
| `coverage` | `full` / `partial` / `none` | `full`의 조건은 다음 세 가지다. <br>• verify 완료 시각 ≤ 측정 시작 <br>• deactivate 시각 ≥ 측정 종료 <br>• 그 사이 모든 observe 기록에서 진행이 확인됨 |
| `nominal_check` | `passed` / `not_performed` / `failed` | 측정 전에 idle 상태를 확인한다. 장치 사용률, 그리고 도구가 제공하면 장치를 쓰는 다른 process 목록을 본다 |

- 분석에 넣는 기본 조건(eligibility)
  - stress run: `verified`이고 coverage `full`
  - nominal run: `nominal_check`가 `failed`가 아님
- `self_reported`인 run은 기본으로 뺀다. 넣으려면 플래그를 명시해야 하며, 결과에 그 사실을 표시한다.

### G.5 stress 후보와 우선순위

| 순위 | kind | target_resource (이름 그대로 보고) | 제어 변수 | 자기 보고 | 독립 증거 | 확인 필요 |
|---|---|---|---|---|---|---|
| 1 | `same_npu_background` | NPU. placement에 따라 이름이 다르다: 같은 PE의 시분할, 또는 다른 PE를 쓸 때의 공유 memory·interconnect | 배경 요청률(C_bg 대비 비율) 또는 동시 stream 수 | 배경 완료율 | vendor telemetry의 사용률, 장치 process 목록 | 다중 process·context 지원, PE 지정, telemetry (Q-2, Q-3, Q-7) |
| 2 | `host_cpu_contention` | host CPU(지정한 core 집합) | core별 duty cycle(25/50/100%) × worker 수 | worker loop 수 | kernel 계정: `/proc/<pid>/stat`의 CPU 시간, `/proc/stat`의 core 사용률 | DUT 경로에서 host 작업의 비중. 작으면 효과가 작을 수 있다 |
| 3 | `host_device_transfer` | host↔device 전송 경로 | 배경 복사량(bytes/s) | 복사량 | vendor telemetry 또는 PCIe 계수기 | SDK의 buffer 복사 API |
| 4 | `host_memory_bandwidth` | host DRAM 대역폭 | 복사 process 수와 크기 | 달성 GB/s | uncore 계수기(perf 등) | 계수기 접근 권한과 도구. 없으면 독립 증거가 부족하다 |
| 제외 | HW fault, DVFS, thermal 조작 | — | — | — | — | 첫 구현에 넣지 않는다 |

- **same-NPU 배경 process의 부수 부하.** 배경 process도 host CPU와 전송 경로를 쓴다.
  - 그래서 결과를 "NPU contention"으로만 부르지 않고, 배경 process의 host CPU 사용량을 함께 기록한다.
  - 원인을 더 나누려면 대조 조건을 추가한다(선택). 배경 process의 host CPU 사용량만큼의 부하를 `host_cpu_contention`으로 따로 재현하는 조건이다.
- **PE 배치.** [사실] `/dev/rngd/`에 `npu0pe0`–`npu0pe7`과 `npu0pe0-1`, `npu0pe4-7` 같은 node가 있다.
  - [가정·미검증] PE 단위 또는 PE 묶음 단위로 배치할 수 있을지 모른다.
  - 그렇다면 "같은 PE 시분할"과 "다른 PE에서의 공유 자원 경합"은 서로 다른 stress kind로 따로 이름 붙인다.
- **host CPU contention과 runner.** runner thread는 stress 대상 core와 겹치지 않는 core에 고정한다. 그래도 stress가 runner를 늦추면 E.8에 따라 그 run을 무효로 처리한다.

### G.6 NoStressProvider (nominal)

- "의도적 개입 없음"을 명시적으로 기록한다.
- 장비가 실제로 idle이었는지는 `nominal_check`가 따로 확인한다.
- verify 단계에서는 같은 시간 동안 대기만 한다(E.1).

### G.7 정리 보장

- stressor는 process group, parent-death signal, pidfile로 관리한다. restore는 group 전체를 종료하고 남은 process가 0개인지 확인한다.
- preflight는 이전 run이 남긴 stressor를 검사한다. 근거는 pidfile과, 가능하면 장치 process 목록이다. 남아 있으면 실행하지 않는다.
- 장치 설정(주파수, 전력, 온도)은 바꾸지 않는다.

---

## H. Model/NPU metric, telemetry, 출력 품질

### H.1 계층과 status

| 계층 | 이름 공간 | 비교 조건 |
|---|---|---|
| portable | `bmt.<group>.<name>` | metric definition과 measurement boundary가 같아야 함 |
| task quality | `quality.<task>.<name>` | task, annotation, evaluator semantics가 같아야 함 |
| telemetry | `tel.<source>.<channel>` | 같은 source/channel 내부 Δ만 허용. cross-vendor normalization 안 함 |

MetricResult는 최소한 `metric_id`, `definition_version`, `measurement_boundary_id`, `unit`, `status`, `value`, `n`, `window`, `method`, `inputs`, `notes`를 가진다.

| status | 의미 | numeric cell analysis |
|---|---|---|
| `valid` | 숫자 값이 유효함 | 가능 |
| `empty` | 정상 관측이지만 표본 0개 | 불가 |
| `censored` | planned percentile 순위가 미완료 영역에 있음 | **불가. report-only** |
| `invalid(reason)` | replay/integrity/sample 규칙 위반 | 불가 |
| `not_collected` | 필요한 raw observation 없음 | 불가 |
| `not_supported` | 정의된 source가 제공하지 않음 | 불가 |
| `not_applicable` | 정의상 해당하지 않음 | 불가 |

`valid`가 아니면 `value=null`이다. 0으로 대체하지 않는다.

### H.2 portable metric catalog v1

| metric | 정의 | 해석 |
|---|---|---|
| `bmt.count.{planned,issued,completed,dropped,timed_out,error,not_issued}` | measure phase 요청 수 | 상태 항등식으로 검사 |
| `bmt.rate.{completion,drop,timeout,error}` | 해당 count ÷ planned | stream coverage와 failure |
| `bmt.latency.request_completion.completed.p{50,90,99}` | completed의 `t_complete − t_sched` | completed-only이므로 coverage와 함께 해석 |
| `bmt.latency.request_completion.planned.p{50,90,99}` | 미완료를 +∞로 둔 planned percentile | censored diagnostic 가능. 숫자 Δ/test에는 사용하지 않음 |
| `bmt.latency.dut_service.p{50,90,99}` | completed의 `t_complete − t_submit_begin` | runtime/driver/device boundary. 순수 NPU time이라고 단정하지 않음 |
| `bmt.latency.output_ready.completed.p{50,90,99}` | `t_output_ready − t_sched` | output-ready 의미가 선언된 경우 |
| `bmt.issue_lag.{p99,max}` | `t_issue − t_sched` | Stage 1R actual replay fidelity |
| `bmt.throughput.completed_per_s` | completed ÷ measurement window | **configured concurrency N의 closed-loop/replay throughput** |
| `bmt.offered_rate_per_s` | planned schedule의 offered rate | fixed/recorded replay |
| `bmt.deadline.miss_rate` | deadline까지 COMPLETED가 아닌 planned 비율 | evaluation parameter. 안전 요구사항 아님 |
| `bmt.output.identity_rate` | 지정 reference와 item별 canonical output hash 일치율 | determinism/corruption diagnostic. accuracy 아님 |
| `bmt.device_execution.*` | 검증된 device timestamp 차이 | Stage 2a/2b에서 semantics와 cross-check가 확보된 경우에만 추가 |
| `bmt.power.*`, `bmt.energy.*` | 검증된 power source가 있을 때 | 없으면 not_supported |

`asap` throughput을 자동으로 “maximum throughput”이라 부르지 않는다. 최대 처리량 주장이 필요하면 concurrency/offered-load sweep과 saturation 근거를 별도 실험으로 제시한다.

### H.3 percentile과 censoring

- nearest-rank를 사용하고 method를 결과에 기록한다.
- p99는 최소 100개 completed 표본이 있어야 한다. 부족하면 `invalid(insufficient_samples)`.
- planned percentile은 미완료를 +∞로 둔다.
- 해당 순위가 +∞이면 `status=censored`, `value=null`, `censored_fraction`을 보고한다.
- censored result에는 median Δ, Mann–Whitney p-value, relative Δ를 계산하지 않는다.
- `observed_max`는 worst-case guarantee가 아니다.

### H.4 telemetry

Stage 1A에서는 telemetry를 구현하지 않는다. Stage 2b 이후 실제 source가 있을 때 다음 원칙을 적용한다.

- channel 이름, unit, scope(chip/card/board/host/process), 의미, source/version, clock, requested/actual interval을 기록한다.
- missing/unavailable을 0으로 채우지 않는다.
- Δ는 같은 source와 channel에서만 계산한다.
- sampling overhead A/B를 수행한다.
- 장비 간 generalization 비교에는 vendor telemetry를 공통 score로 사용하지 않는다.

### H.5 power / energy

- power channel의 measurement scope가 문서화된 경우에만 계산한다.
- energy는 measurement window에서 time integration한다.
- sampling 간격이 너무 크거나 결측 구간이 크면 invalid다.
- request당 energy는 completed count로 나누며 replay idle 시간이 포함될 수 있음을 명시한다.
- 외부 계측 platform은 현재 범위에 넣지 않는다.

### H.6 output quality와 identity

#### Stage 1A

Stage 1A는 task quality를 구현하지 않는다. 다음만 검증한다.

- fake task output의 canonical encoding
- output blob/reference/hash integrity
- 사전 지정된 reference에 대한 `bmt.output.identity_rate`

Reference 우선순위:

1. external golden output
2. experiment plan에 사전 지정된 eligible nominal reference run
3. 별도 승인된 deterministic reference rule

“첫 nominal run”을 암묵적 기준으로 사용하지 않는다.

#### Stage 2 이후 task quality

- `quality.<task>.conditional`: completed output만 평가하고 coverage를 함께 보고한다.
- planned 전체 품질을 계산하려면 QualityEvaluator가 task-specific missing-output policy를 선언해야 한다.
- detection에서 빈 prediction으로 처리할 수 있어도 classification/segmentation에 동일 규칙을 강제하지 않는다.
- missing-output semantics를 정의할 수 없으면 conditional quality와 coverage만 보고한다.
- output identity는 accuracy/correctness 대체 지표가 아니다.
- 기존 MLPerf/AI-BMT evaluator는 의미와 boundary가 같을 때 integration 내부에서 재사용할 수 있다.

---

## I. Observation, evidence, 재평가

### I.1 Stage 1A raw observation

| 파일 | 내용 |
|---|---|
| `events.jsonl` | phase, issued, dropped, submit_begin/end, completed, error, timed_out, late_completion, not_issued, cleanup, run_finished |
| `stress.jsonl` | NoStress/FakeStress lifecycle와 verification record |
| `outputs.bin` 또는 `outputs/` | canonical task output blob |
| `input_items.jsonl` | resolved typed InputItem |
| `run_spec.json` | 기본값까지 해석한 실행 spec |
| `manifest.json` | identity, status, provenance, file digests |

공통 event field:

- `schema_version`
- `kind`
- `seq`
- `t_run_ns`
- `run_clock_domain`
- 관련 request/item/phase identity

Stage 1A Recorder는 deterministic event 순서를 보존하는 단순 append writer다. realtime producer queue와 writer thread는 만들지 않는다. Stage 1R에서 hot-path overhead가 필요해질 때 bounded queue 설계를 별도 승인한다.

### I.2 run bundle

```text
runs/<experiment_id>/
├── experiment.json
├── <run_id>/
│   ├── manifest.json
│   ├── run_spec.json
│   ├── input_items.jsonl
│   ├── events.jsonl
│   ├── stress.jsonl
│   ├── outputs.bin
│   └── evaluations/<eval_id>/
│       ├── metrics.json
│       └── eval_manifest.json
└── analyses/<analysis_id>/        # Stage 4 이후
```

Stage 1A에서는 `telemetry.jsonl`, logs directory, calibration directory를 필수로 만들지 않는다. 실제 source가 생길 때 schema version을 유지하며 optional evidence로 추가한다.

### I.3 manifest 최소 evidence

| 항목 | 기록 |
|---|---|
| benchmark level | 상수 `model` |
| code identity | git commit/dirty 상태 또는 source-tree SHA-256 |
| resolved config | plan/input/workload/DUT/stress/replay/run config와 각 digest |
| input | input kind/id, family/selection/items/index/annotation digest |
| workload | artifact identity, precision/batch 또는 fake workload identity |
| DUT | adapter identity, fake/actual, capability, runtime/device descriptor |
| measurement boundary | `measurement_boundary_id`와 각 구간 포함/제외 의미 |
| stress | `stress_axis_digest`, `stress_level`, lifecycle/verification status |
| status | execution, measurement, stress, execution_kind |
| files | raw/evaluation file SHA-256 |
| reference output | output identity reference 종류와 digest/run ID |

추정값을 넣지 않는다. unavailable은 reason과 함께 기록한다.

### I.4 status와 eligibility

| 축 | 값 |
|---|---|
| `execution_status` | completed / failed(reason) / cancelled / rejected |
| `measurement_status` | valid / invalid(reason) |
| `stress_status` | verification / coverage / nominal_check |
| `execution_kind` | mock / actual |

Run-level comparison eligibility:

- execution completed
- measurement valid
- actual stress 단계에서는 verified + full coverage
- comparison guard 통과

Metric-level numeric eligibility:

- `status=valid`
- 동일 metric definition/version
- 동일 measurement boundary
- 필요한 sample 수 충족

`censored`는 reportable하지만 numeric cell comparison에는 eligible하지 않다. Analyzer는 `not_evaluable_numeric(censored)`와 censored fraction을 출력한다.

### I.5 재평가

- Evaluator는 bundle만 읽는다.
- Runner, DUT, vendor SDK, input source를 다시 호출하지 않는다.
- 새 evaluation은 새 `eval_id` directory에 기록하고 raw file을 덮어쓰지 않는다.
- 동일 evaluator/version/parameters에서 volatile field를 제외한 결과가 결정적으로 같아야 한다.
- deadline 같은 evaluation parameter를 바꾸어도 DUT를 재실행하지 않는다.
- eval manifest는 input raw file digest, evaluator code identity, metric definition version, parameter를 기록한다.

### I.6 data volume

- 원본 dataset과 model artifact는 bundle에 복사하지 않고 source path/reference와 digest를 기록한다.
- Stage 1A output은 작으므로 모두 저장한다.
- 실제 raw tensor 전체 저장은 기본값이 아니며 별도 승인한다.
- 실제 dataset에서 prepared input을 bundle에 복사하지 않는다.

### I.7 mock과 actual

- FakeDUT 또는 FakeStress가 사용되면 `execution_kind=mock`.
- mock 표기는 manifest, metrics, CLI output에 전파한다.
- mock과 actual을 동일 DegradationAnalyzer 결과에 섞지 않는다.
- mock 결과를 NPU 성능 결과로 표현하지 않는다.

---

## J. Degradation Map과 분석 방법

### J.1 분석 단위

- 반복과 통계의 단위는 run이다.
- 한 run 안의 request를 독립 표본으로 취급하지 않는다.
- request-level distribution은 각 run의 metric으로 요약한 뒤 run-level로 비교한다.
- Stage 1A는 cell Δ/통계 검정을 구현하지 않고 raw metric과 comparison key만 검증한다.

### J.2 contrast 종류

| contrast | 유지하는 것 | 바꾸는 것 | 해석 |
|---|---|---|---|
| stress contrast | input selection/family, workload, DUT, replay, boundary, **stress axis** | stress level | 동일 자원 stress axis 내부 dose response |
| scenario contrast | input family, workload, DUT, stress axis+level, replay, boundary | driving scenario selection | input-side difference. compute mechanism이라고 자동 해석하지 않음 |
| input-set comparison | 명시한 공통 조건 | 서로 다른 dataset/subset | scenario effect와 구분 |
| interaction | 동일 input family와 stress axis | scenario selection × stress level | valid 2×2 이상의 difference-in-differences |
| response curve | input과 stress axis 고정 | ordered/realized stress level | level별 metric response |
| envelope 후보 | 실제 requirement source 포함 | stress level | threshold 만족 범위. 공식 score 아님 |

두 축이 동시에 다른 대각선 비교를 단일 원인 효과로 보고하지 않는다.

### J.3 comparison guard

#### 공통 고정 identity

- DUT device/adapter/runtime/driver/firmware/config
- workload artifact, precision, batch, preprocess/postprocess/codec identity
- `measurement_boundary_id`
- replay policy
- admission/timeout/drain 정책
- output capture와 telemetry 설정
- host/CPU placement와 software revision
- execution_kind
- metric/evaluator definition

#### stress contrast key

`key_stress_contrast`는 다음을 포함한다.

- 공통 고정 identity
- input family/selection/items digest
- `stress_axis_digest`

다음만 제외한다.

- `stress_level`

서로 다른 stress kind, resource, placement, control variable, background workload, verification rule은 같은 stress contrast에 들어갈 수 없다.

#### scenario contrast key

`key_scenario_contrast`는 다음을 포함한다.

- 공통 고정 identity
- `input_kind=driving_scenario`
- `input_family_digest`
- stress axis와 level

다음만 제외한다.

- `input_selection_digest`
- 그 selection에서 실현된 `items_digest`

따라서 source dataset/version, stream schema, source timing semantics, task/annotation contract가 다른 input은 같은 scenario contrast로 비교하지 않는다.

#### interaction key

네 cell 모두 동일한:

- common fixed identity
- input family
- stress axis

를 가져야 한다. 변화는 scenario selection과 stress level뿐이어야 한다.

Guard가 실패하면 비교를 계산하지 않고 차이 field와 사유를 `guards.json`에 남긴다.

### J.4 censoring과 numeric analysis

- `status=valid`인 run metric만 숫자 Δ와 검정에 사용한다.
- censored planned percentile은 curve/map에 diagnostic status와 censored fraction으로 표시한다.
- censored value에 임의의 timeout/deadline 값을 대입하지 않는다.
- survival analysis는 현재 범위에 넣지 않는다.
- stream-level degradation은 deadline miss, completion/drop/timeout rate와 completed latency coverage를 함께 사용한다.

### J.5 통계 프로토콜 결정 시점

다음은 Stage 4 pilot 전에 사전 고정한다.

- primary metric별 practical effect threshold `δ_min(metric)`
- randomized block을 반영하는 analysis method
- 반복 수와 detectable effect/power
- 다중 비교 family와 correction
- failed/invalid run의 보충 규칙
- interaction interval/decision rule

v02의 exact Mann–Whitney U + Holm + n=6은 **후보안**이었고 v03의 확정 protocol이 아니다. `n=6`은 첫 실행 계획의 provisional value일 뿐 power를 보장하지 않는다.

분석 protocol을 pilot 결과에 맞춰 변경할 수 있지만 본 실험 plan을 고정하기 전에 version과 이유를 기록해야 한다.

### J.6 run order, drift, carry-over

Stage 5 본 실험의 기본 계획:

- randomized complete block 또는 pilot에서 승인한 block-aware design
- block마다 nominal 포함
- run 시작 상태와 cooldown evidence 기록
- nominal drift와 stress 후 carry-over 확인
- pilot/calibration run은 본 분석에서 제외

구체적인 drift threshold는 metric별 `δ_min`과 함께 Stage 4 pilot 전에 고정한다.

### J.7 출력 형식

- `degradation_map.json/csv/md`
- `response_curves.json`
- `interaction.json`
- `guards.json`
- `analysis_manifest.json`
- optional figures는 Stage 5 별도 dependency 승인 후 생성

각 cell은 다음을 함께 표시한다.

- eligible run 수
- metric status
- median/range 또는 승인된 summary
- coverage/drop/timeout
- nominal 대비 Δ와 판정
- censored fraction 또는 not-evaluable 사유
- realized stress evidence summary

---

## K. 첫 완전한 Driving Scenario × Hardware Stress benchmark experiment

### K.1 전제와 선행 gate

이 절은 첫 NPU 실행이 아니라 첫 완전한 Scenario × Stress matrix다.

선행 조건:

1. Stage 2b Dataset baseline에서 model output correctness와 measurement boundary가 검증됨
2. vendor/profiler cross-check가 동일하거나 설명 가능한 boundary에서 통과함
3. Stage 3 Driving Scenario replay가 유효하고 scenario family/selection identity가 고정됨
4. Stage 4b stress axis가 독립 evidence로 verified되고 restore가 확인됨
5. Stage 4 pilot에서 analysis protocol과 반복 수가 고정됨

| 항목 | 필요한 것 |
|---|---|
| NPU | 실제 NPU 1개와 독점 또는 방해를 계측할 수 있는 실행 환경 |
| SDK | completion model, input buffer lifetime, concurrency, error semantics |
| workload | compiled artifact, preprocess/postprocess, reference output/quality |
| driving data | source timing, task annotation, selection mechanism, license |
| stress | 특정 axis의 kind/resource/placement/control/verification |
| telemetry | 가능한 utilization/temperature/frequency/power 등과 sampling semantics |
| external reference | vendor benchmark/profiler와 measurement boundary |

### K.2 provisional matrix

| 항목 | 계획 |
|---|---|
| Scenario | S1, S2, S3. 동일 `input_family_digest` 안의 서로 다른 selection |
| Stress | 동일 `stress_axis_digest`의 `none`, L1, L2, L3 |
| Cell | 12 |
| 반복 | provisional n=6. Stage 4 pilot에서 최종 결정 |
| request 수 | completed percentile sample 조건과 deadline-rate resolution을 만족하도록 사전 결정 |
| 고정 변수 | DUT/workload/artifact/precision/batch/replay/boundary/admission/output capture/telemetry/host placement/software |
| primary stream metrics | deadline miss rate, completion rate, drop rate, timeout rate |
| primary timing metrics | completed request-completion latency, DUT-service latency와 completion coverage |
| diagnostic | planned request-completion percentile의 censoring status/fraction |
| secondary | output-ready latency, quality, configured-N throughput, output identity diagnostic, telemetry Δ |

planned p99가 censored되면 숫자 Δ를 계산하지 않는다. stream rate와 completed latency/coverage를 함께 해석한다.

### K.3 절차

| 단계 | 내용 | 산출 |
|---|---|---|
| P0 DUT baseline | 동일 boundary에서 N=1 service/request-completion baseline과 vendor cross-check 재확인 | baseline bundle |
| P0b background calibration | background-only offered-load/concurrency sweep으로 `C_bg`와 안정성 범위 추정 | stress calibration bundle |
| P1 pilot | matrix 일부 또는 한 block을 실행해 noise, cooldown, stress level separation, invalid rate 확인 | pilot report |
| P2 analysis freeze | `δ_min(metric)`, block method, n, multiple comparison, replenishment rule 고정 | versioned analysis protocol |
| P3 plan freeze | axis/selection/level과 모든 fixed variable의 digest 고정 | final plan digest |
| P4 execution | 승인된 block/order로 본 run 실행 | raw run bundles |
| P5 evaluation/analysis | bundle만 읽어 metric, map, curve, interaction, guard 생성 | analysis bundle |
| P6 reproducibility | 새 process에서 evaluate/analyze 재실행 | deterministic match record |

Main DUT의 N=1 `asap` 결과를 최대 처리량이라고 부르지 않는다. background capacity `C_bg`가 필요하면 별도 saturation calibration으로 추정한다.

### K.4 산출물

- Degradation Map
- scenario별 Stress-Response Curve
- valid interaction contrast
- mechanism decomposition:
  - issue lag
  - request-completion
  - DUT-service
  - completion/drop/timeout/deadline
  - quality
  - telemetry
- verification summary:
  - cross-check
  - replay fidelity
  - stress requested/achieved/independent evidence
  - measurement overhead
  - drift/carry-over
  - raw re-evaluation consistency

### K.5 결과 타당성

결과로 인정하려면:

- fixed identity와 comparison guard가 통과해야 한다.
- metric별 eligible run 수가 analysis protocol을 만족해야 한다.
- actual stress run은 verified/full coverage여야 한다.
- replay fidelity가 승인 기준을 만족해야 한다.
- vendor boundary cross-check가 허용오차를 만족하거나 경계 차이를 설명해야 한다.
- raw re-evaluation이 일치해야 한다.
- censored/invalid run을 숫자 값으로 대체하지 않아야 한다.

열화가 없다는 결과도 유효하다. 단, realized stress evidence와 detectable-effect/분해능을 함께 제시한다.

### K.6 대안

| 상황 | 처리 |
|---|---|
| same-NPU background 불가 | host CPU 또는 host-device transfer처럼 독립 evidence가 가능한 axis로 변경. 자원 이름을 그대로 보고 |
| source timestamp 없음 | fixed-rate scenario만 수행하고 timing-scenario claim을 제한 |
| annotation 없음 | task quality는 not_supported. external output reference와 identity diagnostic만 사용 |
| telemetry 없음 | 해당 telemetry는 not_supported. same-NPU stress verification이 불충분하면 다른 stress axis 검토 |
| 실제 NPU 없음 | Stage 2b 이후 중단. FakeDUT 결과를 NPU 결과로 사용하지 않음 |

### K.7 설정 초안

Stage 1A와 Stage 2+ schema를 혼동하지 않는다. 아래는 장기 actual experiment 예시이며 `evaluation_level` field가 없다.

```yaml
schema_version: 1
experiment_id: dataset-nominal
execution_kind: actual
input: ../inputs/datasets/<dataset>.yaml
workload: ../workloads/<model>.yaml
dut: ../duts/<npu>-default.yaml
stress:
  spec: ../stress/none-for-target-axis.yaml
  level: none
replay:
  mode: asap
run:
  max_inflight: 1
  queue_capacity: 0
  on_full: drop_new
evaluation:
  metrics:
    - bmt.latency.request_completion.completed.p50
    - bmt.latency.dut_service.p50
    - bmt.throughput.completed_per_s
```

```yaml
schema_version: 1
experiment_id: first-scenario-stress-matrix
execution_kind: actual
inputs:
  - ../inputs/scenarios/S1.yaml
  - ../inputs/scenarios/S2.yaml
  - ../inputs/scenarios/S3.yaml
workload: ../workloads/<model>.yaml
dut: ../duts/<npu>-default.yaml
stress:
  spec: ../stress/npu-background.yaml
  levels: [none, L1, L2, L3]
replay:
  mode: recorded
  speed: 1.0
run:
  max_inflight: 1
  queue_capacity: 0
  on_full: drop_new
  output_capture: task_output_all
evaluation:
  deadline: {policy: frame_period}
  primary_metrics:
    - bmt.deadline.miss_rate
    - bmt.rate.completion
    - bmt.rate.drop
    - bmt.latency.request_completion.completed.p99
    - bmt.latency.dut_service.p99
  diagnostics:
    - bmt.latency.request_completion.planned.p99
analysis:
  protocol: ../analysis-protocols/<frozen-version>.yaml
  envelope: null
```

```yaml
stress_id: npu-bg
kind: same_npu_background
target_resource: <documented-resource>
placement: <same-PE-or-shared-resource>
background_workload:
  artifact: <path>
  sha256: <digest>
control:
  variable: offered_rate_fraction_of_capacity
  unit: ratio
  capacity_from: background-saturation-calibration
levels: {L1: 0.25, L2: 0.50, L3: 1.00}
verification:
  achieved_min_fraction: 0.90
  independent:
    source: <vendor-or-kernel-channel>
    rule: <pre-registered-rule>
schedule: whole_measurement
on_apply_failure: abort
```

---

## L. 개발 단계와 roadmap

### L.1 단계표

| 단계 | 범위 | 명시적 비범위 | 산출/Gate | 승인 상태 |
|---|---|---|---|---|
| Stage 0 | product/benchmark architecture와 measurement design | 구현 | v03 최종 설계 | **승인 완료** |
| Stage 1A | typed input, fixture 2종, fake workload, blocking FakeDUT, NoStress/FakeStress, virtual Runner, raw bundle, portable evaluator | realtime, async, telemetry, actual stress, analysis | deterministic golden + conformance + re-evaluation | **구현 승인** |
| Stage 2a | 대상 SDK/도구/기존 code read-only 조사 | 장비 execution, implementation | completion model, input residency, measurement boundary, profiler cross-check plan | 조사 범위 별도 승인 |
| Stage 1R | SDK 특성에 맞춘 realtime harness | 불필요한 completion path, stress matrix | actual schedule fidelity, harness floor, runtime-specific conformance | Stage 2a 후 별도 승인 |
| Stage 2b | actual adapter/workload/DatasetProvider, Dataset nominal baseline | Driving Scenario, stress | correctness, boundary cross-check, repeatability, overhead | Stage 1R gate 후 |
| Stage 3 | actual DrivingScenarioProvider 2–3종 nominal | stress | replay fidelity와 scenario sensitivity characterization | Stage 2b 후 |
| Stage 4a | stress feasibility와 independent evidence 조사/probe | full sweep | selected stress axis | 별도 승인 |
| Stage 4b | one input × nominal+3 levels sweep, minimal analyzer | full matrix | verified stress response, restore, analysis protocol freeze | Stage 4a 후 |
| Stage 5 | 3 Scenario × 4 Stress matrix와 full analyzer | System-level code | map, curve, interaction, guard/drift report | Stage 4b 후 |
| Stage 6 | second workload 또는 accelerator | 새 core 기능 | generalization evidence | 선택 |
| Phase 2 | Autoware/System-level boundary, system observation, pipeline/driving evaluator | Model/System 단일 score | 별도 design doc | Stage 5 이후 별도 승인 가능 |

### L.2 Stage 1A 구현 범위

#### Contracts

- DatasetSpec / DrivingScenarioSpec
- SourceTiming / ScenarioCoordinates / InputItem
- ReplayPolicy
- WorkloadSpec / DUTConfig / MeasurementBoundary / RunConfig
- Request/Completion/Stress/Manifest/Metric records
- input/stress factor identity와 comparison key record

#### Fixtures와 fake components

- timestamp 없는 DatasetProvider fixture
- timestamp, jitter, episode gap이 있는 RecordedScenarioProvider fixture
- fake workload preprocess/postprocess/codec
- blocking FakeDUT
  - fixed service time
  - table-driven service time
  - deterministic output
  - error/timeout/late-completion virtual injection
- NoStressProvider
- virtual FakeStress lifecycle/failure injection

#### Runner

- virtual clock
- asap/fixed-rate/recorded schedule
- N=1 / queue 0 / drop_new
- Stage 1A request state subset
- deterministic cleanup/finalize
- prepared input ownership

#### Evidence/Evaluator

- run_spec/manifest/input_items/events/stress/output
- file and source-tree digest
- counts/rates
- request-completion/DUT-service/output-ready latency
- deadline miss
- output identity diagnostic
- censored report-only
- raw bundle re-evaluation

#### CLI

- `validate`
- virtual `run`
- `evaluate`

`analyze`는 Stage 1A 범위가 아니다.

### L.3 Stage 1A 명시적 금지 범위

- actual NPU/vendor SDK
- realtime driver와 OS sleep
- async adapter, executor thread pool
- run-per-process supervisor, signal cancellation, watchdog
- GIL detection
- recorder writer thread/queue
- host/vendor telemetry와 power
- real OS stressor/process group
- bounded prefetch/lookahead
- DegradationAnalyzer와 통계 검정
- heatmap/curve figure
- Autoware/ROS2/System-level
- plugin discovery, DB, web/dashboard, workflow DSL

### L.4 Stage 1A 구현 순서

```text
WP1  Project skeleton + strict contracts/config loader
  ↓
WP2  Typed input fixtures + input conformance + replay validation
  ↓
WP3  Fake workload/DUT + prepared-input ownership
  ↓
WP4  Virtual Runner + request state + NoStress/FakeStress lifecycle
  ↓
WP5  Evidence bundle + offline evaluator
  ↓
WP6  Golden/E2E/architecture tests + docs + Stage 1A report
```

각 work package는 다음 단계가 요구하는 최소 test가 통과한 뒤 진행한다. 구현 중 새로운 framework layer가 필요하다고 판단되면 먼저 변경 이유와 최소 대안을 보고하고 승인 없이 범위를 넓히지 않는다.

### L.5 전체 roadmap gate

```text
Stage 0 — approved
   │
   ▼
Stage 1A — deterministic core
   │ gate: canonical events / typed input / bundle re-evaluation
   ├──────────────┐
   │              ▼
   │         Stage 2a — SDK investigation
   │              │
   └──────────────┤
                  ▼
Stage 1R — runtime-specific realtime harness
   │ gate: target-host replay fidelity / boundary / harness floor
   ▼
Stage 2b — Real NPU + Dataset baseline
   │ gate: correctness / profiler cross-check / repeatability
   ▼
Stage 3 — Driving Scenario nominal
   │ gate: scenario sensitivity characterized
   ▼
Stage 4 — verified stress sweep
   │ gate: requested-achieved-independent evidence / restore / protocol freeze
   ▼
Stage 5 — Scenario × Stress
   │ gate: valid map/curve/interaction and bounded interpretation
   ├─────────────────────┐
   ▼                     ▼
Stage 6 — generalization  Phase 2 — System-level design
```

Stage 6은 Phase 2의 필수 선행조건이 아니다. Stage 5의 Model-level evidence가 충분하면 두 작업을 별도 승인 아래 병행할 수 있다.

---

## M. 자동 테스트와 단계별 완료 기준

### M.1 Stage 1A test suite

| 종류 | 범위 |
|---|---|
| unit | strict config, digest, typed timing, schedule, state transition, metric calculation |
| golden | virtual Runner + FakeDUT canonical event sequence |
| input conformance | deterministic order/digest, no invented metadata, replay compatibility |
| DUT conformance | prepare-before-prepare_input, input/output correspondence, cleanup idempotence |
| stress conformance | lifecycle order, failure injection, restore on every path |
| architecture | forbidden imports와 package dependency |
| e2e | validate → virtual run → bundle → evaluate |

Stage 1A에는 `tests/realtime`, `tests/device`, telemetry overhead test를 만들지 않는다. skip을 pass로 세지 않는다.

### M.2 trust property 배치

| trust property | Stage 1A evidence | actual-stage evidence |
|---|---|---|
| Measurement Validity | metric equations, boundary record, FakeDUT golden truth | vendor/profiler cross-check, telemetry/output overhead |
| Repeatability | virtual canonical event와 re-evaluation byte/canonical equality | repeated actual runs, drift/carry-over |
| Comparability | factor identity와 guard key conformance | fixed-variable enforcement와 block design |
| Traceability | append-only raw bundle, file/source digests, provenance | artifact/runtime/driver/firmware/telemetry provenance |

Stage 1A는 actual NPU validity를 증명하지 않는다. actual claim은 Stage 2b 이후 external cross-check와 real adapter conformance가 있어야 한다.

### M.3 Stage 1A 완료 기준

| ID | 기준 | 통과 조건 |
|---|---|---|
| S1A-1 | deterministic canonical execution | 같은 plan/seed의 virtual run이 volatile run ID를 제외하고 동일 event sequence와 metric을 생성 |
| S1A-2 | Dataset semantics | asap/fixed-rate 실행 가능, recorded validation 거절, 가짜 timing 없음 |
| S1A-3 | Scenario timing semantics | typed SourceTiming에서 정확한 recorded schedule, jitter와 episode gap 재현 |
| S1A-4 | ownership boundary | InputProvider가 DUT/prepared object를 소유하지 않고 `DUT.prepare` 후 `prepare_input` 실행 |
| S1A-5 | golden state/metric | fixed/table service에서 count identity, drop, timeout, late completion, deadline, latency가 expected 값과 일치 |
| S1A-6 | stress lifecycle | NoStress/FakeStress의 모든 failure path에서 restore record가 존재하고 verify 실패 시 measure 0건 |
| S1A-7 | raw re-evaluation | bundle-only evaluate 결과가 같고 raw digest가 변하지 않으며 새 eval_id를 생성 |
| S1A-8 | comparison identity | 다른 stress axis와 다른 input family가 contrast key를 공유하지 않음. level/selection만 의도대로 제외 |
| S1A-9 | censoring rule | censored metric은 value null/report-only이며 숫자 Δ/p-value code path가 없음 |
| S1A-10 | dependency boundary | vendor SDK, NumPy, Torch, Autoware, ROS2, MLPerf import가 core에 없음 |
| S1A-11 | evidence completeness | benchmark_level=model, boundary ID, input/stress axis, source/file digest, MOCK가 manifest에 기록 |
| S1A-12 | no scope leakage | realtime/async/telemetry/analysis/System-level implementation 파일이 생성되지 않음 |

Stage 1A 보고서에는 다음을 포함한다.

- 생성/변경 파일 목록
- 실행한 command
- test pass/fail/skip 수
- golden scenario와 expected/actual 요약
- run bundle example과 digest
- 알려진 limitation
- Stage 1R로 넘긴 unresolved item
- source-tree SHA-256

### M.4 Stage 1R 승인·완료 조건

Stage 2a 보고서가 다음을 답해야 Stage 1R scope를 승인할 수 있다.

- runtime completion model: blocking/async/polling
- GIL behavior 또는 확인 방법
- input buffer lifetime와 H2D behavior
- supported concurrency/context/process model
- timeout/cancel/error semantics
- vendor profiler boundary와 API/CLI
- telemetry availability와 sampling cost
- device reset/isolation requirement

Stage 1R 완료 evidence 후보:

- target-host real clock schedule fidelity
- required completion path의 conformance
- harness floor
- recorder overhead
- process isolation 필요 여부의 근거
- same boundary profiler cross-check 준비

실제 필요한 항목만 승인한다.

### M.5 Stage 2b–6 완료 기준

#### Stage 2b — Dataset baseline

- actual adapter와 DatasetProvider conformance
- artifact/precision/batch/boundary provenance
- external golden output 또는 quality cross-check
- N=1 request-completion/DUT-service baseline
- configured concurrency throughput의 정확한 표기
- vendor/profiler boundary cross-check
- repeated-run variability
- telemetry/output recording overhead

#### Stage 3 — Driving Scenario nominal

- typed source timing과 recorded replay validity
- 2–3 scenario가 동일 family에서 다른 selection임을 evidence로 확인
- scenario별 nominal repeated runs
- workload/timing characteristic과 service/quality 차이 보고
- scenario sensitivity가 없어도 유효한 결론

#### Stage 4b — verified stress sweep

- actual stress conformance
- requested/achieved/independent evidence
- full coverage와 restore
- one input × nominal+3 levels
- Stage 4 pilot에서 analysis protocol freeze
- stressor host-resource confounder 기록

#### Stage 5 — Scenario × Stress

- planned run 전체 계수와 exclusion reason
- guard를 통과한 stress/scenario/interaction contrast만 계산
- censored diagnostic과 numeric result 분리
- map/curve/interaction의 bundle-only regeneration
- drift/carry-over와 bounded interpretation

#### Stage 6 — generalization

- second workload/NPU가 core contract를 바꾸지 않고 conformance 통과
- 변경이 필요하면 구현 전에 design change review
- cross-target 비교는 portable metric과 동일 boundary에 한정

#### Phase 2 gate

별도 design review에서 다음을 먼저 정한다.

- Autoware/System boundary
- source/clock synchronization
- NPU model request와 system observation linkage
- pipeline/driving metric
- CPU/ROS/network/planning confounder
- open-loop/closed-loop scope

Model-level contract를 소급해 generic target으로 확대하지 않는다.

### M.6 보고 규칙

- 실행하지 않은 test는 명시하고 pass로 세지 않는다.
- 실패를 통과시키기 위해 expected value나 threshold를 사후 완화하지 않는다.
- actual 허용오차, reference, threshold는 측정 전에 versioned config로 고정한다.
- 단계 외 기능이 필요해 보이면 구현 전에 scope-change proposal을 제출한다.
- raw observation을 수정하지 않고 evaluation/analysis result는 새 ID로 추가한다.

---

## N. 실제 NPU, SDK, 데이터 확인이 필요한 질문

| ID | 질문 | 막는 단계 |
|---|---|---|
| Q-1 | 첫 대상 NPU, 접속 방식, 실험 중 독점 가능 여부 | Stage 2a |
| Q-2 | SDK 이름/version/docs, Python API, blocking/async/polling, concurrency/context/process, buffer lifetime, GIL, timeout/cancel/error, device/PE 선택 | Stage 2a |
| Q-3 | utilization/memory/temperature/frequency/power/throttling/process telemetry의 source, unit, scope, clock, sampling cost | Stage 2a / 4a |
| Q-4 | cross-check에 사용할 vendor benchmark/profiler와 정확한 measurement boundary | Stage 2a / 2b |
| Q-5 | 실제 workload artifact, precision/input/batch, preprocess/postprocess, reference output/quality, 재사용 코드 위치/license | Stage 2a / 2b |
| Q-6 | Stage 2 Dataset과 Stage 3 recorded scenario의 이름/version/location/annotation/source timing/license, subset/selection 고정 방법 | Stage 2b / 3 |
| Q-7 | same-NPU background의 multi-process/context/PE placement 지원과 output correctness 영향 | Stage 4a |
| Q-8 | host shared-use, CPU affinity, device exclusivity 정책 | Stage 1R / 2b / 5 |
| Q-9 | 실제 latency/quality requirement 또는 threshold source. 없으면 Robustness Envelope 계산 안 함 | Stage 5 |
| Q-10 | NVIDIA GPU를 reference output 또는 Stage 6 target으로 사용 가능한지 | Stage 6 |
| Q-11 | vendor SDK Python/package 제약과 integration 전용 venv 사용 가능 여부 | Stage 1R / 2b |

Stage 1A는 위 질문의 답 없이 진행할 수 있다. 다만 실제 SDK를 추측한 interface나 dependency를 Stage 1A에 추가하지 않는다.

Git 관련 정책은 더 이상 Q-12 blocker가 아니다.

- git repository가 있으면 commit과 dirty 상태를 기록한다.
- 없으면 source-tree SHA-256을 기록한다.
- `git init`, commit, remote push는 별도 승인 없이 수행하지 않는다.

---

## O. 이전 설계 처리와 보류한 기능

### O.1 유지하는 원칙

- Input과 Hardware Stress는 독립이며 Runner에서만 결합한다.
- Dataset과 Driving Scenario는 다른 input semantics를 가진다.
- requested stress와 realized stress evidence를 분리한다.
- Recorder와 Evaluator를 분리하고 raw observation으로 재평가한다.
- measurement boundary와 clock domain을 명시한다.
- 측정하지 못한 값을 0으로 채우지 않는다.
- mock과 actual을 혼합하지 않는다.
- 한 factor의 effect를 주장할 때 다른 fixed variable을 유지한다.
- negative result를 유효한 benchmark 결과로 허용한다.
- System-level metric을 Model-level score와 합치지 않는다.

### O.2 v02에서 수정한 것

| v02 항목 | v03 변경 |
|---|---|
| InputProvider가 DUT 변환 callable을 받아 prepared input을 저장 | InputProvider는 raw source/payload만 제공. Runner가 DUT 준비 이후 preparation을 수행하고 prepared store를 소유 |
| `t_complete − t_sched`를 NPU latency로 명명 | request-completion latency로 변경. DUT-service와 output-ready 분리 |
| stress/input 전체를 comparison key에서 제거 | axis/family identity는 유지하고 level/selection만 제외 |
| censored planned percentile을 comparison eligible로 취급 | report-only diagnostic. numeric Δ/test에서 제외 |
| timing을 generic metadata로 둠 | SourceTiming과 ScenarioCoordinates typed field 추가 |
| 첫 nominal run을 output identity 기준으로 사용 | external golden 또는 사전 지정 eligible reference |
| ExperimentPlan의 evaluation_level | 제거. manifest에 `benchmark_level=model` 상수 |
| Stage 1에서 virtual+realtime, blocking+async를 모두 구현 | Stage 1A deterministic core와 Stage 1R runtime-specific harness로 분리 |
| exact Mann–Whitney/Holm/n=6을 사실상 고정 | Stage 4 pilot 전 metric-specific protocol로 별도 freeze |
| asap throughput을 maximum throughput으로 표현 | configured-N throughput으로 제한. maximum claim은 saturation evidence 필요 |

### O.3 폐기한 기존 방향

- MLPerf/LoadGen/QSL/SUT/official submission 구조
- Automotive plugin/upstream contribution을 현재 design constraint로 두는 것
- LoadGen scenario/metric compatibility
- generic `BenchmarkTarget`으로 Model/System을 조기 통합
- Stage 1의 async/blocking 양쪽 선제 구현
- Dataset sample에 fake driving timing을 부여하는 것
- vendor telemetry를 cross-vendor 단일 score로 정규화하는 것

### O.4 지금 만들지 않는 것

- realtime harness와 SDK-specific executor: Stage 1R
- actual NPU adapter/workload/dataset: Stage 2b
- actual telemetry/power/energy: Stage 2b 이후
- actual HardwareStressProvider와 OS stressor: Stage 4
- DegradationAnalyzer/statistical test/map/curve: Stage 4–5
- bounded prefetch, device-resident cache, per-request H2D policy: Stage 2a 이후
- multiple overload policies
- fault injection, DVFS, thermal manipulation
- Robustness Envelope 공식 score
- Autoware/ROS2/System-level implementation
- dashboard, database, plugin discovery, workflow DSL, distributed execution

---

## 부록 A. 이 v03에서 수행한 작업

- v02 설계 문서와 v02 검토/Stage 1 승인안을 전체 기준 문서로 사용했다.
- 이전 파일을 수정하지 않고 v03을 새 파일로 작성했다.
- M-1–M-7과 Stage 1A/1R 분할을 contracts, lifecycle, metrics, analysis, roadmap, completion criteria에 반영했다.
- 실제 repository source, environment, venv, git, device, SDK, stress workload는 변경하거나 실행하지 않았다.
- 별도 Stage 1A 구현 승인 문서를 작성했다.

## 부록 B. 입력 문서 무결성

| 파일 | SHA-256 |
|---|---|
| `2026-10-06-stage-0-core-benchmark-redesign-v01.md` | `acee7f4a71c9c5c7dc391e16dd8195d8c639f9bc761b30ebb72e5e3d4d230d55` |
| `2026-10-06-stage-0-core-benchmark-redesign-v02.md` | `58fdcd378d03cc54b7c82bf05e6acef83f7f3cd5052875d5ae2f79dd44ccc7ae` |
| `2026-10-06-stage-0-v02-review-and-stage1-approval.md` | `9e80e456b31dc87740d2fa6c66104c2c3fae8251d840f02b2976e76e898d0007` |

> 승인 범위는 Stage 1A까지다. Stage 1R, 장비 실행, 실제 stress 실행, commit/remote 작업은 후속 gate와 별도 승인 없이 시작하지 않는다.
