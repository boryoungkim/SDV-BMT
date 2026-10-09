# SDV-BMT — Architecture (Stage 1A core + Stage 1R-G realtime harness)

> **Stage 1A 경로는 MOCK이다.** virtual clock 위에서 blocking FakeDUT를 실행하며, 모든 manifest, metric, CLI 출력에 `execution_kind=mock`이 기록된다.
>
> **Stage 1R-G 경로**(10절)는 host monotonic clock 위의 native(C++) measurement plane이다. fake backend로는 MOCK, NVIDIA backend로는 실제 GPU(`execution_kind=actual`)에서 실행한다. GPU는 첫 *reference* accelerator이며, GPU 결과는 측정 시스템 검증용이다. NPU 성능 결과가 아니고 NPU 동작을 예측하지 않는다.

기준 설계: `docs/design/2026-10-06-stage-0-core-benchmark-redesign-v03.md`(공개판, 승인 원본 SHA-256 `29846782…267794`), 승인 범위: `docs/design/2026-10-06-stage-1a-implementation-authorization.md`(공개판).

## 1. 증명하려는 경로

```text
typed Input → deterministic schedule → deterministic FakeDUT execution
→ append-only raw observation → bundle-only MetricEvaluator → integrity-verifiable evidence
```

## 2. 패키지와 책임

| package | 모듈 | 책임 | 하지 않는 것 |
|---|---|---|---|
| `contracts` | `status` `ids` `specs` `records` `protocols` | enum/status, canonical JSON·digest·ID, strict spec dataclass와 loader, record, Protocol, 공통 예외 | stdlib 외 import |
| `input` | `fixture_index` `dataset_fixture` `recorded_scenario_fixture` | digest로 고정된 source index 검증, 결정적 `InputItem`, raw payload 제공 | preprocess, DUT 변환, `t_sched` 계산, 정렬·보간 |
| `workload` | `fake` | `fake_identity` preprocess/postprocess/codec, 불변 `WorkloadInputContract` 선언(canonical input 의미의 소유자) | device 호출 |
| `dut` | `fake` | blocking `FakeBlockingDUT`(fixed/table service model, error 주입), 선택적 fused input override(`FakeFusedPrefixOverride`) | scheduling, stress, metric, Workload object 접근 |
| `stress` | `lifecycle` `none` `fake` | lifecycle 순서 강제, `NoStressProvider`, virtual `FakeStressProvider` | 실제 OS process, 독립 증거 생성 |
| `observation` | `events` `recorder` `reader` | raw event schema, append-only writer, strict bundle reader, request folding | metric 집계 |
| `evidence` | `bundle` `manifest` `provenance` `keys` | bundle layout·file digest, manifest, source-tree SHA-256, comparison key/guard | metric 값 비교 |
| `runner` | `plan` `schedule` `core` `virtual` `runner` | strict YAML·plan 해석·VALIDATE, schedule, request state·prepared store, virtual MEASURE/DRAIN, lifecycle·cleanup·finalize | metric/Δ 계산 |
| `metrics` | `catalog` `portable` `identity` `evaluator` | bundle-only 평가, portable metric, output identity | Runner/DUT/input 재호출 |
| `cli` | `main` | composition root(정적 factory table), `validate`/`run`/`evaluate` | plugin discovery, `analyze` |

D.5 목록에 없는 내부 helper 모듈은 `input/fixture_index.py`, `stress/lifecycle.py` 두 개다(두 fixture/provider가 공유하는 코드). 기능 범위는 넓히지 않는다.

## 3. 의존 방향 (architecture test로 강제)

```text
contracts ◄── input, workload, dut, stress, observation
contracts, observation ◄── evidence
contracts, observation, evidence ◄── metrics, runner
(all) ◄── cli   (composition root)
```

- `contracts`는 stdlib만 import한다.
- `input`/`workload`/`dut`/`stress`는 서로 import하지 않고 `contracts`만 공유한다.
- `metrics`는 `runner`/`dut`/`input`/`stress`/`workload`를 import하지 않는다.
- `runner`는 구체 fake 구현을 import하지 않는다. Protocol만 사용하고, 구체 객체는 `cli/main.py`의 정적 table(`WORKLOADS`, `DUT_ADAPTERS`, `STRESS_KINDS`)이 조립한다.
- PyYAML은 `runner/plan.py`에서만 import한다.
- core 전체에서 금지: `threading`, `asyncio`, `concurrent`, `multiprocessing`, `subprocess`, `signal`, `socket`, `time`, `random`, `importlib`, `pkgutil`, `ctypes` 등과 vendor SDK, NumPy, Torch, ONNX Runtime, MLPerf, CARLA, ROS2, Autoware. `__import__`/`eval`/`exec`도 금지.
- Stage 1R-G 예외(승인 구조 "C++ Measurement Plane, Python Analysis Plane"): `time`은 `runner/realtime.py`(run clock)와 `dut/native.py`(native process watchdog)에서만, `subprocess`는 `dut/native.py`(native runner 실행)에서만 허용한다. `time.sleep`은 어디서도 쓰지 않는다. Python core에는 `ctypes`와 vendor Python binding이 없다. CUDA/NVML/NVRTC/NVTX는 `native/src/nvidia/backend.hpp`에서만 include한다(vendor header 검사 포함).

검사: `tests/architecture/test_dependency_boundary.py`(allowlist 기반, 위반 탐지 self-check 포함), `tests/architecture/test_scope.py`.

## 4. Lifecycle (design v03 E.1)

```text
VALIDATE  (plan 정적 규칙: mock only, dataset×recorded 거절, N=1, queue 0, drop_new,
           drain_ns ≥ request_timeout_ns, batch=1, quality evaluator 없음, level 유효성)
PREFLIGHT (component 생성, input index/digest/selection/timing 검증, schedule 계산,
           DUT capability(blocking·virtual_timing·max_inflight), Stress.check, boundary, comparison keys,
           이전 run의 restore 실패 여부)
   │   여기까지 실패하면 rejected: bundle을 만들지 않는다
   ▼
OPEN BUNDLE (experiment.json 1회 기록/검증, run_spec.json, input_items.jsonl, stress check 기록)
→ INPUT.open → DUT.prepare → PREPARE_INPUTS → (adapter_override일 때) INPUT.EQUIVALENCE
→ STRESS.prepare → STRESS.activate → STRESS.verify(window_ns만큼 virtual clock 진행)
→ MEASURE → DRAIN
→ STRESS.deactivate → STRESS.restore → prepared_store.discard → DUT.cleanup → INPUT.release
→ FINALIZE (run_finished → file close → manifest.json)
```

- 실패 시: 이미 시작한 단계의 cleanup을 역순으로 모두 시도한다. 시작하지 않은 단계는 `cleanup ... outcome=skipped`로 기록한다.
- cleanup 단계는 서로 독립이다. 각 단계는 component 호출을 먼저 하고 그 다음 기록한다. 기록(recorder)이 실패해도 남은 단계(restore, DUT cleanup, input release)를 모두 수행한 뒤 첫 recorder 오류를 다시 던진다.
- stress provider 호출은 `_call_stress`로 감싼다. 예외, StressRecord가 아닌 반환, canonical JSON으로 기록할 수 없는 record(NaN/inf 등)는 모두 실패 record로 바뀐다. 따라서 provider 결과가 lifecycle을 중단시키지 못한다.
- `stress.restore`는 bundle이 열린 뒤의 **모든 종료 경로**에서 호출된다(시작 전이어도 호출, provider가 no-op으로 기록).
- `stress.activate`/`verify` 실패 시 MEASURE에 들어가지 않으며 nominal로 대체하지 않는다. verify는 `outcome`이 ok가 아니거나 `verification=failed`이면 실패다. `StressRecord` contract는 `verification=failed`인데 `outcome=ok`인 record를 만들 수 없게 한다.
- `execution_status`는 첫 실패 사유(`stress_verify_failed`, `dut_prepare_failed`, `measure_aborted`, `stress_restore_failed` 등)를 갖는다.
- restore 실패는 이후 run을 PREFLIGHT에서 막는다. 같은 process 안에서, 그리고 같은 output root 아래 **모든 experiment**의 이전 run manifest(manifest가 없으면 `stress.jsonl`의 restore 기록)를 검사한다. 다른 output root는 보지 않는다(한계, Stage 4 실제 stressor의 pidfile/process 검사로 대체 예정).

## 5. Preparation ownership (M-1)

```text
for item in provider.items():
    payload       = provider.payload(item.item_index)
    task_input    = workload.preprocess(payload, item)
    prepared      = adapter.prepare_input(task_input, item)
    store.put(item.item_index, prepared)          # Runner 소유 PreparedStore
```

- `InputProvider`는 source/index/payload만 다룬다. DUT·prepared object를 받거나 저장하지 않는다(Protocol surface와 object-graph 검사로 test).
- `DUT.prepare()` 전에 `prepare_input()`을 부르면 `LifecycleError`.
- 준비는 MEASURE 전에 eager로 끝난다. 따라서 source read, preprocess, `prepare_input`은 측정 경계 밖이다.
- item별 준비 실패는 store에 기록되고, 해당 request는 MEASURE에서 `NOT_ISSUED(prepare_failed:<stage>)`가 된다(run은 계속).
- store는 cleanup에서 `DUT.cleanup` 전에 폐기된다.

## 5a. Input pipeline (design change: Explicit DUT Adapter Preprocess Override)

Workload는 canonical task input 의미, reference preprocessing, postprocessing, output encoding, task quality 의미의 소유자로 남는다. DUT Adapter는 preprocessing/input-preparation **실행 경로만** 명시적으로 대체할 수 있다.

| `plan.input_pipeline.mode` | 경로 | owner |
|---|---|---|
| `workload` (기본값) | `InputProvider.payload → Workload.preprocess → DUTAdapter.prepare_input` | `workload` |
| `adapter_override` (+ `implementation_id` 필수, `implementation_version`/`implementation_digest` 선택 pin) | `InputProvider.payload → AdapterInputOverride.prepare_from_payload` | `dut_adapter` |

- Adapter에는 mutable Workload object를 넘기지 않는다. `Workload.input_contract()`가 돌려주는 불변·versioned `WorkloadInputContract`(contract id/version, task, `preprocess_semantics_id`, payload format, logical input `(key, value)` 쌍)를 `DUTAdapter.input_override(contract, implementation_id)`에 넘긴다. adapter는 plan이 지정한 그 구현만 돌려주고, 없으면 None이다.
- 자동 선택은 없다. `workload` mode에서는 adapter의 override를 조회조차 하지 않는다.
- silent fallback도 없다. PREFLIGHT에서 다음 경우 plan을 거절한다(bundle을 만들지 않음): 지정한 override가 없음, version/digest pin 불일치, contract의 `preprocess_semantics_id`를 구현하지 않음, descriptor가 형식에 맞지 않음.
- descriptor의 `preprocess_semantics_ids`/`fused_steps`는 tuple로 고정되고 bare string은 거절된다. `fused_steps`는 `workload.preprocess`, `dut.prepare_input` 중에서만 고를 수 있다.
- override가 item 준비에 실패하면 `NOT_ISSUED(prepare_failed:adapter_override)`가 된다. Workload 경로로 대신 준비하지 않는다.
- `INPUT.EQUIVALENCE`: **모든 item**에 대해 Workload reference 경로(`Workload.preprocess → DUTAdapter.prepare_input`)를 실행하고, override 결과와 **DUT adapter의** canonical `prepared_input_digest`로 비교한다(method `exact_adapter_prepared_input_digest`). override 자신은 비교 함수를 제공하지 않는다. reference 객체는 비교 직후 버린다(fallback이 아님).
  - 양쪽 준비 성공 → digest 비교, 다르면 불일치
  - override 실패 + reference 성공 → 불일치
  - reference 실패 + override 성공 → `reference_error_item_indices`. override 탓으로 돌리지 않고 inconclusive로 남긴다
  - 양쪽 실패 → `both_failed_item_indices` (일관됨, item은 NOT_ISSUED)
  - digest가 SHA-256 형식이 아니거나, 비교한 item이 2개 이상인데 모든 reference digest가 같으면(퇴화한 digest) 동등성이 입증되지 않은 것으로 본다
- **fail closed**: `passed`일 때만 stress·MEASURE로 진행한다. `failed`이면 `input_equivalence_failed`, `not_performed`이면 `input_equivalence_not_established`로 run이 실패하고, restore를 포함한 cleanup만 수행한다.
- 효과적 input pipeline(`InputPipelineRecord`: mode, owner, semantics, contract digest, implementation id/version/digest, fused steps)의 digest는 measurement boundary(`input_pipeline_digest`)와 comparison key의 공통 고정 identity에 모두 들어간다. 따라서 override 구현이 다르면 boundary ID도 다르고(metric이 boundary ID를 가짐), stress/scenario contrast는 같은 pipeline 안에서만 성립한다.
- FakeDUT의 최소 override(`fake-fused-prefix-preprocess`)는 contract의 `prefix_hex`만으로 `sha256(prefix || payload)`를 한 번에 계산한다. `divergent: true`는 equivalence 실패 경로를 검증하기 위한 test 전용 설정이다.

## 6. Virtual execution

- 단일 process, 단일 thread, OS sleep 없음. `VirtualClock`은 정수 ns, 역행 불가.
- `EventQueue`는 `(t_ns, priority, insertion_seq)` 순서. priority: DUT return(0) < timeout(1) < 도착.
- blocking `FakeBlockingDUT.run()`은 즉시 `DUTResult(virtual_service_ns)`를 돌려주고, Runner가 `t_submit_begin + virtual_service_ns`에 return event를 예약한다.
- 세부 semantics는 `docs/measurement_protocol.md`.

## 7. 증거와 재평가

- Recorder: 파일을 exclusive create하고 canonical JSON line을 append·flush하는 단순 writer(thread/queue 없음). `seq`는 `events.jsonl`과 `stress.jsonl`이 공유하는 하나의 counter다.
- Evaluator: manifest의 raw file digest를 먼저 검증한 뒤 bundle만 읽고, 새 `evaluations/<eval_id>/`에 결과를 쓴다. 평가 전후로 raw digest가 같아야 한다.
- 세부: `docs/evidence.md`, `docs/metrics_catalog.md`.

## 8. Stage 1A에서 만들지 않은 것

(Stage 1R-G가 이 중 realtime 실행만 native measurement plane으로 추가했다. 10절.)

realtime driver(`runner/realtime.py`), async submit/callback, executor/thread pool, process supervisor·signal·watchdog, telemetry·`/proc`·power, 실제 stress process, bounded prefetch, DegradationAnalyzer·통계 검정, plot/dashboard/DB, Autoware/ROS2/System-level, plugin discovery, git 조작. `tests/architecture/test_scope.py`가 해당 경로·식별자·의존성이 없음을 검사한다.

## 9. Contrast validity (factor isolation)

> **Control what is planned. Verify what is realized. Localize what changed. Do not attribute beyond the evidence.**
>
> **A configured condition and a realized condition are not assumed to be identical.**
>
> **Failure to establish factor isolation is not a benchmark failure; it is evidence that the requested attribution is not justified.**

관측된 차이 = 의도한 factor 효과 + 실현 조건 차이 + 측정/harness 효과 + noise. SDV-BMT는 뒤의 세 항을 통제·관측·탐지·제한하려 한다. 다만 숨은 교란 요인이 없다고 주장하지 않는다. 관측할 수 없거나 분리할 수 없는 원인은 `unresolved`로 남긴다.

```text
Valid attributed contrast = Design valid AND Realization valid AND Measurement valid
```

`evidence/validity.py::evaluate_contrast_validity(manifest_a, manifest_b, contrast, *, evaluations=None)`는 두 run manifest와 (주어지면) 두 `eval_manifest.json`만 읽어 `ContrastValidityResult`를 돌려준다. metric 값은 보지 않는다(guard first, arithmetic second). Stage 1A에는 Δ·통계·인과 점수가 없다.

| guard | 질문 | Stage 1A 판정 근거 |
|---|---|---|
| Design | 명세가 의도한 factor만 바꿨는가 | `comparison_keys.components`의 모든 design component를 canonical JSON으로 비교한다. contrast별 허용 component(stress: `stress_level`·`stress_level_value`; scenario: `input_selection_digest`·`items_digest`; interaction: 둘 다) 외의 차이는 전부 `violations`로 나열한다. factor가 "달라졌다"는 판단은 label이 아니라 내용(`stress_level_value`, `items_digest`)으로 한다. 양쪽 모두 값이 null인 identity(예: `model_artifact`)는 `unknown_identities`에 적고, 두 run이 모두 mock이 아니면 `unresolved`다 |
| Realization | 고정하려던 조건이 실행에서 실제로 같게 실현되었는가 | manifest `realization.evidence`(조건별 일반 evidence record)를 비교한다. 비교 가능한 evidence 값이 다르면(canonical JSON 기준) `fail`이다. `pass`에는 모든 조건이 양쪽에서 값이 있는 `observed`이고, 지원되는 equivalence 규칙(Stage 1A: `exact`)으로 같아야 한다. `not_applicable`은 동등성의 증거가 되지 않는다. 의도한 factor를 실현하는 조건(stress 자체, scenario payload)은 그 factor의 내용이 실제로 달라졌을 때 **run 간 동일성 비교에서만** 빠진다. 각 run은 여전히 factor가 계획대로 실현되었음을 보여야 한다(stress: `verified`+`full` coverage, nominal: `nominal_check: passed`, payload: 실현된 items digest = 계획). 네 곳의 `execution_kind`(top-level, status, realization, identity) 중 하나라도 `actual`이 아니면 `not_observable`이다 |
| Measurement | 같은 유효한 측정 정의로 얻은 관측인가 | 다음이 같아야 한다: boundary ID(timestamp 의미, preprocess/H2D 포함, output-ready, input pipeline), run clock domain, output capture, telemetry 설정, metric catalog version, output reference. 두 run 모두 `completed`이면서 measurement가 `valid`여야 한다. `evaluations`가 주어지면 두 평가가 각자의 run을 평가했는지, 같은 protocol(deadline, output reference, percentile 방법 등)과 같은 evaluator(module, catalog, metric 정의 version, source tree)를 썼는지 확인한다. `evaluations`가 없으면 metric 정의를 확인할 수 없으므로 `unresolved`다 |

- **status**: guard는 `pass`/`fail`/`not_observable`/`not_applicable`/`unresolved` 중 하나다. overall은 `valid`/`invalid`/`not_established`/`not_applicable` 중 하나이며 다음 순서로 정한다. ① 어느 guard든 `fail`이면 `invalid`이다. `not_applicable`이 실패를 가리지 않는다. ② Design이 `not_applicable`이면 `not_applicable`이다. ③ interaction이면 `not_established`이다. ④ 세 guard가 모두 `pass`이면 `valid`이다. ⑤ 그 밖에는 `not_established`이다.
- **attribution**: `attribution_allowed`는 overall이 `valid`이고 의도한 factor가 실제로 달라졌을 때만 참이다. 다음 두 경우는 귀속할 수 없다.
  - replicate 쌍(label만 다른 쌍 포함): 의도한 factor가 달라지지 않았다.
  - interaction 쌍: 두 cell이 같은 interaction family에 속하는지만 확인한다. interaction에는 2×2의 네 cell이 모두 필요하다(design v03 J.2). 쌍 하나로는 성립하지 않는다.
- **malformed evidence**: 한 조건에 record가 둘 이상 있거나, `observed`/`self_reported`/`virtual`인데 값이 없는 record는 판정하지 않고 거절한다(`ValueError`).
- **Stage 1A의 한계**: Stage 1A run은 모두 mock이다. 따라서 Realization Guard는 `not_observable`(basis `mock_execution`)이고 overall은 최대 `not_established`다. deterministic FakeDUT 두 run이 같다는 사실은 실제 NPU의 실현 조건을 통제할 수 있다는 증거가 **아니다**.
- **pipeline 전환 금지**: stress/scenario contrast는 canonical(Workload) 전처리와 adapter override 전처리 사이를 조용히 오갈 수 없다. `preprocess_semantics_id`와 effective input pipeline digest가 design component이고, pipeline digest는 boundary에도 들어간다. 의미가 동등하다고 실행 경로의 성능이 같아지지는 않는다.
- **Stage 1A에서 하지 않는 것**: realization evidence의 실제 수집(residency, buffer, H2D/DMA, queue, frequency, thermal, 경쟁 사용자, 실제 issue timing, memory/bandwidth, 실제 stress)은 Stage 2a/1R/2b/4로 미룬다. vendor별 schema도 만들지 않는다.

## 10. Stage 1R-G — native measurement plane (GPU Reference Accelerator Track)

승인 근거: "SDV-BMT DEVELOPMENT AUTHORIZATION — GPU Reference Accelerator Track"(2026-10-07), "APPROVED ARCHITECTURE UPDATE — C++ Measurement Plane, Python Analysis Plane"(2026-10-08). 설계 근거: `docs/reviews/2026-10-08-stage-2a-g-sdk-observability-investigation-v02.md` 17–19절(내부 stage-gate 기록, 공개 저장소에는 포함되지 않음).

> **C++ records low-level execution facts; Python interprets those facts into benchmark metrics and research conclusions.**
>
> **Portable benchmark semantics must remain independent of the mechanism used by NVIDIA to observe them.**

### 10.1 경로 선택

| DUT completion model | 경로 | clock |
|---|---|---|
| `blocking` | Stage 1A virtual runner(변경 없음, golden snapshot byte 동일) | virtual |
| `polling` | realtime 분기: MEASURE/DRAIN을 native runner process가 실행 | host `CLOCK_MONOTONIC` |

- polling adapter는 plan의 `realtime` section(`cpu_affinity`, `wait_policy: spin`, `precondition_requests`)을 요구한다. plan의 `execution_kind`는 adapter의 `implementation_kind`와 같아야 한다.
- realtime run은 nominal(stress level `none`)만 허용한다. 실제 stress는 Stage 4다.
- workload는 native output pipeline(`native_output_pipeline()`)을 제공해야 한다. 없으면 PREFLIGHT에서 거절한다.

### 10.2 process / evidence 경계

```text
Python control plane                         C++ native runner (run 하나당 process 하나)
  VALIDATE, PREFLIGHT (describe: 장치 identity, context 생성 없음)
  OPEN BUNDLE, INPUT.open
  native 시작(resolved native run spec) ──────►  ready(실현된 CPU affinity)
  "prepare" ──────────────────────────────────►  DUT.prepare (context, NVRTC, buffer, profiler 환경 검사)
  Workload.preprocess → staging 파일
  "stage" ────────────────────────────────────►  item별 staging + SHA-256 재검증
  STRESS.prepare / activate / verify
  recorder 일시 정지(seq, 마지막 t, output offset 인계)
  "measure" ──────────────────────────────────►  precondition, observe, MEASURE, DRAIN, observe
                                                  (events.jsonl / outputs.bin에 직접 append)
  recorder 재개                     ◄──────────  응답(seq, 마지막 t, output offset, native 계수)
  STRESS.deactivate / restore
  "cleanup" ──────────────────────────────────►  DUT.cleanup, 종료
  INPUT.release, FINALIZE (event folding으로 계수, manifest v4)
```

- 명령과 응답은 stdin/stdout의 JSON 한 줄이다. lifecycle phase 경계에서만 오가고 request마다 오가지 않는다. MEASURE/DRAIN 동안 Python에는 request별 callback도 Python↔C++ 호출도 없다.
- bundle 파일에는 언제나 한 쪽만 쓴다(single writer). 인계 때 seq와 run clock(같은 `CLOCK_MONOTONIC` anchor)을 넘기므로 seq 연속성과 시간 비감소가 유지된다.
- native runner는 사실만 기록한다(timestamp, state, 장치 측 duration, kernel timing word, NVML snapshot, 출력 byte와 digest). issue lag, percentile, MHz 같은 파생 값은 Python이 bundle에서 계산한다. native가 응답에 넣는 request 계수는 교차 검증용이다. event folding 결과와 다르면 run이 실패한다(`native_counts_disagree_with_events`).
- watchdog: 응답이 phase별 상한 시간 안에 오지 않으면 Python이 native process를 종료하고 run을 실패로 기록한다. process가 끝나면 driver가 장치 자원을 회수한다.
- Stage 1A lifecycle 순서(DUT.prepare → PREPARE_INPUTS → STRESS → MEASURE/DRAIN → STRESS 해제 → DUT.cleanup)는 그대로다.

### 10.3 구성 요소

| 위치 | 내용 |
|---|---|
| `native/src/common/loop.hpp` | realtime MEASURE/DRAIN loop(portable 의미, vendor 코드 없음): admission(N=1, queue 0, `drop_new`), 상태 전이, timeout, inclusive DRAIN cap, spin 대기 |
| `native/src/common/runner.hpp`, `events.hpp`, `json.hpp`, `sha256.hpp`, `reference_output.hpp` | phase protocol, event v2 직렬화(DRAIN 뒤), JSON, SHA-256, reference workload 출력 pipeline(C++) |
| `native/src/nvidia/backend.hpp` | NVIDIA backend. CUDA driver API, NVRTC, NVML, NVTX는 이 파일에만 있다 |
| `native/src/fake/backend.hpp` | 장치 없는 MOCK backend(harness floor, conformance, fault 주입) |
| `src/sdv_bmt/dut/native.py` | native process 실행, phase protocol client, run spec, staging(backend 공통) |
| `src/sdv_bmt/dut/nvidia/adapter.py` | `NvidiaCudaDUT`(`nvidia_cuda`): 측정 선언, run spec의 backend 부분, 기록된 사실을 realization evidence로 해석 |
| `src/sdv_bmt/dut/fake.py` | `FakeNativeDUT`(`fake_native`, MOCK) |
| `src/sdv_bmt/runner/realtime.py` | `MonotonicRunClock`(anchor 공유), bundle에서 window 사실 읽기 |
| `src/sdv_bmt/workload/reference.py` | `ReferenceConvNetWorkload`(`reference_convnet`): reference accelerator conformance workload |

빌드: `make -C native`(system g++ 11.4, GNU make, CUDA 13.0 header·NVRTC, driver library). Python dependency는 늘지 않았다(PyYAML + pytest).

### 10.4 NVIDIA backend 실행 경로

UUID로 장치 선택 → primary context(`CU_CTX_SCHED_BLOCKING_SYNC`) → workload spec이 가진 device program을 SHA-256으로 확인한 뒤 NVRTC로 장치의 compute capability용 cubin 생성 → persistent device buffer와 pinned 출력 buffer → item별 staging(`host_pinned`: pinned host buffer, request마다 H2D가 DUT-service 안; `device_resident`: device buffer에 1회 복사하고 read-back으로 확인, H2D 제외) → submit: 한 non-blocking stream에 [H2D] → kernel → D2H와 CUDA event를 enqueue하고 반환 → 완료: 마지막 event에 `cuEventQuery` polling → NVML snapshot은 `measure_begin`과 `drain_end`에서만.

계측 profile(DUT config `runtime_options`, measurement identity의 일부):

| option | 값 | 의미 |
|---|---|---|
| `instrumentation` | `cuda_events` / `none` | segment event bracket(장치 clock duration, event 비용 포함) 사용 여부 |
| `nvtx` | `true` / `false` | NVTX request range와 admission/completion mark(domain `sdv-bmt`) |
| `profiler` | `none` / `nsys` | 선언한 profiler. prepare에서 실제 injection 환경(`CUDA_INJECTION64_PATH` 등)과 다르면 실패한다 |
| `residency` | `host_pinned` / `device_resident` | 준비된 입력의 위치. boundary의 `prepare_input_h2d`(included/excluded)를 정한다 |
| `buffer_policy` | `preallocated_persistent` / `per_request_alloc` | 후자는 buffer lifecycle 특성화 전용 선언 옵션이다(submit에서 할당, 완료 poll에서 해제) |

### 10.5 Reference accelerator conformance workload

`reference_convnet`(`configs/workloads/reference-convnet-v1.yaml`)는 승인된 실제 workload가 없어 만든 **reference accelerator conformance workload**다. task model도 NPU workload도 아니다. 128장의 합성 128×128 RGB 이미지(gradient/rings/stripes/noise), 정확한 전처리(nearest ×4 upsample, CHW float32 `(x−128)/128`, 3 MiB), 단일 block device program(pool4 → conv3×3 → conv3×3 → 정수 GAP → int64 FC, 약 3.0 ms). 모든 값이 정확히 표현되므로 GPU 출력은 CPU 정수 reference(`tools/stage1r_g/reference_cpu.py`)와 bit 단위로 같다. golden은 CPU 구현으로만 만든다. 출력 pipeline(`sdv-ref-logits-top1/1` + `canonical-json-utf8/1`)은 측정 경로 안에서 C++로 실행된다. Python 기준 구현과의 byte 동일성은 logit vector 407개(경계값 + random)와 task 문자열 2종으로 test에 고정되어 있고(`tests/native/test_native_unit.py`), 실제 GPU 출력은 매 run에서 CPU golden과 비교된다(output identity).

### 10.6 Portable 의미와 NVIDIA 관측의 분리

- core contract는 portable 의미만 쓴다: `host_to_device`, `device_execution`, `device_to_host`, request completion, residency 등.
- CUDA stream/event/context, NVML handle, NVTX, CUPTI 개념은 `native/src/nvidia/`, `src/sdv_bmt/dut/nvidia/`, 그리고 namespaced evidence(`nvidia:` source, `nvidia.` key)에만 있다.
- generic `Accelerator`/plugin framework는 만들지 않았다. 기존 DUT Adapter 구조를 그대로 쓴다. NPU backend는 Stage 2a-N 조사 뒤 같은 의미에 자기 SDK를 대응시킨다.

### 10.7 남은 한계

- 공유 host(다른 사용자의 부하, CPU governor `powersave`)이다. issue lag와 꼬리 값에 선점이 섞일 수 있다. host CPU frequency와 실제 실행 core는 기록하지 않는다(affinity 집합만 기록).
- DUT.cleanup은 stream이 5 s 안에 비지 않으면 자원을 해제하지 않고 process 종료에 맡긴다.
- profiler run은 별도 measurement 구성이며 짧게(≤200 request) 유지한다. spin-poll API 호출 때문에 trace가 크다.

## 11. Stage 2b-G real-model baseline path (YOLOX-S, cuDNN plan)

승인 근거: 2026-10-08 프로젝트 review 결정 "REVIEW DECISION — CONTINUE STAGE 2b-G"(실제 model + 실제 dataset baseline)와 같은 날의 후속 결정(목록 중 YOLOX-S 선택과 공식 `yolox_s.onnx` download, GPU runtime은 cuDNN/cuBLAS, offline 도구에 한해 Pillow·onnxruntime, tools 전용 venv의 pycocotools, 30 Hz와 100 Hz). GPU는 여전히 reference accelerator다. 이 경로의 결과는 NPU 결과가 아니다.

### 11.1 구성 요소

| 위치 | 내용 |
|---|---|
| `src/sdv_bmt/workload/yolox.py` | `YoloxCocoWorkload`(`yolox_coco`): PPM payload → YOLOX letterbox 전처리(BGR, CHW float32, 640×640, pad 114, bilinear) → raw 8400×85 출력의 decode + class-aware NMS + top-100 → canonical JSON. `native_output_context(item)`이 출력 pipeline에 원본 크기와 image id를 넘긴다 |
| `src/sdv_bmt/input/dataset_fixture.py`, `fixture_index.py` | index row가 `payload_file`(index 디렉토리 안의 상대 경로) + `payload_sha256`을 가질 수 있다. PREFLIGHT는 선언 digest를 쓰고 `open()`이 파일을 읽어 digest를 재검증한다. 이런 index의 provider 이름은 `dataset_files` |
| `native/src/nvidia/cudnn_plan.hpp` | `cudnn_plan` device program 실행기: plan(JSON)·weight 파일 SHA-256 확인, weight upload, activation arena, cuDNN descriptor, algorithm 선택(pinned 또는 heuristic), NVRTC glue kernel(space-to-depth, bias+activation(+residual), nearest ×2 upsample, copy, head transpose, frequency probe) |
| `native/src/nvidia/backend.hpp` | program kind `cuda_c`(기존, reference workload)와 `cudnn_plan`. submit은 [H2D] → probe → plan step → D2H를 한 stream에 넣는다. `describe`가 cuDNN version과 library file SHA-256을 device identity에 넣는다 |
| `native/src/common/yolox_output.hpp`, `output_pipeline.hpp` | YOLOX 출력 pipeline의 native twin, run spec의 `output_pipeline.kind`에 따른 선택. 출력 pipeline 상수(threshold, max detections)가 Python 선언과 다르면 native가 거절한다 |
| `native/src/common/loop.hpp`, `runner.hpp` | raw 출력은 backend buffer의 view로 넘긴다(측정 경로 안 복사 없음). 출력 buffer는 pipeline이 선언한 상한으로 MEASURE 전에 예약한다. `stage` 명령의 item이 `output_context`를 가질 수 있다 |
| `src/sdv_bmt/dut/nvidia/adapter.py` | `cudnn_plan` program(파일은 repository 기준 상대 경로, SHA-256 pin), runtime option `cudnn`(math type, algorithm policy, workspace 상한, pinned table)과 `frequency_probe_cycles` |

### 11.2 Offline 자산 준비와 검증 도구 (`tools/stage2b_g_real/`, code identity 밖)

| 도구 | 역할 | 의존성 |
|---|---|---|
| `build_coco_subset.py` | COCO val2017 앞 1,000장(image id 순)을 Pillow로 한 번 decode해 PPM payload와 index(annotation inline) 생성 | system Pillow(승인, 도구 전용) |
| `onnx_reader.py`, `build_yolox_plan.py` | ONNX protobuf를 직접 읽어 cuDNN plan과 weight 파일로 변환. 알 수 없는 graph pattern은 거절 | stdlib |
| `raw_dump.cpp`, `validate_numerics.py` | 같은 native backend로 raw 출력을 뽑아 ONNX Runtime(CPU) 출력과 비교, Python twin과 native 출력 byte 비교 | system onnxruntime·numpy(승인, 도구 전용) |
| `coco_eval.py`, `coco_quality.py` | COCOeval(bbox) 의미의 stdlib 재구현, bundle + digest가 맞는 index로 AP 계산 | stdlib |
| `coco_eval_crosscheck.py` | pycocotools와 같은 입력으로 비교 | `.venv-tools`의 pycocotools(승인) |
| `capture_golden.py` | 지정한 reference-capture run의 출력 digest를 output reference로 추출 | stdlib |
| `analyze_baseline.py`, `guard_cases.py`, `evidence_chain.py`, `nsys_extract_plan.py`, `nsys_align_plan.py` | 기술 통계, guard 결과, request 단위 증거 사슬, multi-kernel nsys 교차 검증 | stdlib |

### 11.3 Dependency

- Python core와 `.venv`는 그대로다(PyYAML + pytest). Python core는 cuDNN·ONNX·NumPy를 import하지 않는다(architecture test).
- NVIDIA native runner만 cuDNN 9.1.0(pip `nvidia-cudnn-cu12` 9.1.0.70, static cudart 12.4)을 link한다(`native/Makefile`의 `CUDNN_ROOT`). cuBLAS는 이 model에 GEMM이 없어 쓰지 않는다. vendor header는 `native/src/nvidia/`에만 있다.

## 12. Model-level Metric Set v1 (2026-10-08 승인)

승인 근거: 프로젝트 결정 "APPROVED DESIGN UPDATE — SDV-BMT Model-level Metric Set v1"(같은 날 후속 결정 포함). 원칙: **C++ records facts. Python computes meaning.** native measurement plane은 바뀌지 않았다(native source tree와 runner binary 그대로). percentile, deadline, task 품질, 통계는 native에 없다.

| 위치 | 역할 |
|---|---|
| `src/sdv_bmt/metrics/metric_set_v1.py` | 기본 catalog `sdv-bmt-model-metric-set-v1`의 정의(91개 metric, layer·단위·method)와 bundle-only 계산 |
| `src/sdv_bmt/metrics/quality_detection.py` | detection 품질: bundle의 frozen ground truth와 completed 출력만 읽고 digest를 재검증, conditional population, coverage, evaluation protocol |
| `src/sdv_bmt/metrics/quality_coco.py` | COCO bbox 평가 core(검증된 analysis-plane 도구의 이식)와 operating-point TP/FP/FN |
| `src/sdv_bmt/metrics/catalog.py` | legacy catalog `bmt-portable-v1-stage1a`, 동결. `evaluate_run(..., catalog=...)`로 선택 |
| `src/sdv_bmt/metrics/evaluator.py` | catalog 선택(기본 v1), run 단위 gate, evaluation schema 1(legacy)·2(v1) 산출물, v5 raw 파일 검증 |
| `src/sdv_bmt/contracts/quality.py` | 품질 평가기 선언 표(평가기 ID → task, annotation schema, 읽을 수 있는 task output schema), COCO 2017 category id, 기본 operating point. plan validation(runner)과 평가(metrics)가 같은 선언을 쓴다 |
| `src/sdv_bmt/input/annotations.py` | `IndexAnnotationSource`: digest-pinned index에서 item ground truth를 읽는 component. `InputProvider`의 public surface(M-1)는 그대로이고, composition root(`cli/main.py`)가 workload가 평가기를 선언했을 때만 만든다 |
| `src/sdv_bmt/runner/runner.py`, `runner/plan.py` | 선언된 평가기 검증, PREFLIGHT에서 ground truth 확보, OPEN BUNDLE에서 `quality_reference.jsonl` 기록, manifest v5 |
| `src/sdv_bmt/cli/main.py` | `sdv-bmt evaluate --catalog`(기본 v1), `--eval-config`의 `required_rate_hz`·`quality_operating_point` |

- 의존 방향은 그대로다: metrics는 runner·dut·input·workload를 import하지 않는다(evaluator는 dataset index를 열지 않는다). runner는 input을 import하지 않고 composition root가 넘긴 annotation source를 쓴다.
- 새 Python dependency 없음(pycocotools는 runtime dependency가 아니다). 새 subpackage 없음.
- 5개 결과 계층과 guard 결과는 `docs/metrics_catalog.md` 8절.
