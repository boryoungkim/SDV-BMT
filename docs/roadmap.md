# Roadmap and Stage Gates

기준: design v03 L절, Stage 1A 승인서 9절. 단계 완료는 다음 단계의 자동 승인이 아니다.

```text
Stage 0  design ─────────────────────────── 승인 완료
Stage 1A deterministic Model-level core ─── 승인 완료 (DC-1, DC-2 포함, 2026-10-07)
   │
   ▼  GPU Reference Accelerator Track (승인 2026-10-07, RTX PRO 6000 = 첫 reference accelerator, NPU 대체 아님)
Stage 2a-G SDK / observability investigation ── 완료 (보고서 v01, v02)
Stage 1R-G GPU realtime harness (C++ native measurement plane) ── 구현·증거 완료
Stage 2b-G GPU dataset baseline / measurement validation ── 측정 시스템 검증 부분 승인(2026-10-08)
   └─ 계속: 실제 model + dataset baseline(YOLOX-S, COCO val2017 1,000장, cuDNN) ── 보고 완료(2026-10-08)
   └─ Model-level Metric Set v1 구현 + canonical GPU metric report ── 승인(2026-10-08), 구현 완료(v0.1.0)
   │
   └── GPU track은 아직 닫히지 않았다. Stage 2a-N은 실제 model baseline 검토 전까지 시작하지 않는다
            ▼
Future: Stage 2a-N → Stage 1R-N → Stage 2b-N (실제 NPU)
Stage 3  Driving Scenario nominal
Stage 4a/4b stress feasibility → verified stress sweep
Stage 5  Scenario × Stress matrix
Stage 6  second workload/accelerator (선택)      Phase 2 System-level design (별도 승인)
```

GPU track 결과는 측정 시스템 검증(`reference_accelerator_validation`)이다. GPU 결과는 NPU timing, memory-path 가시성, stress 관측 가능성, 전처리 동등성, queue 의미, profiler 유효성을 입증하지 않는다. 이것들은 Stage 2a-N → 1R-N → 2b-N에서 다시 확인해야 한다.

## Stage 1A가 남긴 것 (재사용 대상)

- strict contract와 loader, canonical JSON/digest, factor identity와 comparison key
- typed InputItem / SourceTiming / ScenarioCoordinates와 replay schedule 계산
- request state machine, 계수 항등식, prepared-input ownership
- stress lifecycle 순서와 restore-on-all-paths 구조
- append-only bundle, strict reader, bundle-only evaluator, censoring 규칙

## Stage 2a — Memory / Data-Movement Investigation (필수 조사 항목)

NPU에서는 memory와 data movement가 1급 교란 요인이자 기전이다. Stage 2a SDK 조사는 다음을 명시적으로 조사한다. 관측할 수 없는 신호는 대체값을 만들지 말고 "관측 불가"로 기록한다.

| 항목 | 조사 내용 |
|---|---|
| Input residency | host-resident vs device-resident, persistent device buffer, buffer 재사용, zero-copy 지원 |
| Buffer lifecycle | 할당, 재사용/pooling, 소유권, lifetime, alignment/pinning |
| H2D / DMA | 명시적/암묵적 전송, 동기/비동기 전송, 실제 전송 byte(관측 가능 시), timing API, 전송/연산 overlap, H2D가 runtime 실행에 포함/제외/분리 불가인지 |
| Device memory | 용량/사용량, 할당 압력, memory pool 동작, retry/실패 관측 |
| Bandwidth / traffic | (지원 시) device DRAM 대역폭, read/write traffic, DMA throughput, interconnect traffic, 공유 SRAM/cache 압력 |
| Runtime warm state | first-run vs warm-run, allocator 상태, runtime/compiled-graph cache, device buffer cache/재사용 |
| Vendor preprocessing | fused resize/normalize/layout/quantization, 전처리와 H2D의 fusion, 중간 tensor 관측 가능성, 전처리 실행 위치 |
| Profiler visibility | 위 경계들을 profiler가 어디까지 보여 주는지 |

Stage 1R과 Stage 2b의 범위는 이 결과를 보고 정한다. 이 경로들은 지금 구현하지 않는다. 결과는 manifest `realization` section의 일반 evidence record(`source`/`semantic`/`unit`/`scope`/`clock`/`validity`/`equivalence`)로 담을 수 있게 되어 있다.

## Stage 1A가 하지 않은 것 (다음 gate의 입력)

| 항목 | 넘어가는 곳 |
|---|---|
| real `CLOCK_MONOTONIC` scheduling, issue-lag floor, sleep/wakeup jitter, GIL 영향 | Stage 1R (1R-G: native C++ spin issuer, GPU 경로에서 구현·특성화. NPU는 1R-N에서 다시) |
| blocking executor 또는 async submit/callback 중 SDK에 맞는 경로 | Stage 2a → 1R |
| recorder writer queue, run-per-process isolation, watchdog | Stage 1R(필요할 때만). 1R-G: native process가 run마다 따로 돌고 phase별 watchdog이 있다. writer queue는 필요 없었다(사실은 memory에 두고 DRAIN 뒤 기록) |
| actual adapter/workload/DatasetProvider, vendor profiler cross-check | Stage 2b |
| telemetry/power, output/telemetry overhead A/B | Stage 2b 이후 |
| 실제 stressor process 관리, 독립 evidence | Stage 4 |
| DegradationAnalyzer, 통계 protocol | Stage 4–5 |
| realization evidence의 연속값 equivalence 규칙(tolerance), 장치에 없는 조건의 `not_applicable` 선언 방법 | 여전히 미정. 2b-G는 issue timing, SM 주파수, 온도를 특성화만 했고 규칙은 만들지 않았다(`unapproved:` → `unresolved`). 규칙은 별도 승인이 필요하다 |
| interaction의 2×2(네 cell) validity 판정 | Stage 5 (쌍 단위 결과는 membership 확인일 뿐 귀속하지 않음) |
