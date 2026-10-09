"""Golden cases G-1..G-6 (Stage 1A authorization section 4, S1A-5/S1A-9/S1A-7).

Each case checks (a) hand-written expected semantics taken from the
authorization and (b) byte-level equality with the checked-in golden
snapshots under tests/golden/expected/ (regression guard).
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from sdv_bmt.contracts.ids import file_sha256, to_jsonable
from sdv_bmt.contracts.records import RAW_EVIDENCE_FILES, numeric_eligible
from sdv_bmt.contracts.status import MetricStatus as M
from sdv_bmt.evidence.provenance import source_tree_sha256
from sdv_bmt.metrics.catalog import CATALOG_VERSION
from tests._helpers import MS, REPO_ROOT, evaluate, metric_map, read_jsonl, request_trace
from tests.golden.make_golden import CASES, GOLDEN_DIR, SNAPSHOT_FILES, produce


@pytest.fixture(scope="module")
def produced(tmp_path_factory):
    work = tmp_path_factory.mktemp("golden")
    return {case: produce(case, work) for case in CASES}


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    from tests._helpers import load, run_plan

    work = tmp_path_factory.mktemp("golden-bundles")
    return {case: run_plan(load(name), work / case, run_id=f"run-{case}") for case, name in CASES.items()}


def _metrics(bundle_outcome, eval_id="eval-golden"):
    return metric_map(evaluate(bundle_outcome.run_dir, eval_id=eval_id))


def _value(metrics, metric_id):
    m = metrics[metric_id]
    assert m.status is M.VALID, (metric_id, m.status, m.status_reason)
    return m.value


# ---------------------------------------------------------------- snapshots


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("name", SNAPSHOT_FILES)
def test_matches_golden_snapshot(produced, case, name):
    golden = GOLDEN_DIR / case / name
    assert golden.is_file(), f"missing golden snapshot {golden}"
    assert produced[case][name] == golden.read_bytes(), f"{case}/{name} differs from golden"


# ---------------------------------------------------------------- G-1


def test_g1_dataset_fixed_rate_nominal(bundles):
    out = bundles["G1"]
    m = _metrics(out)
    assert _value(m, "bmt.count.planned") == 8 and _value(m, "bmt.count.completed") == 8
    for metric_id in ("bmt.count.dropped", "bmt.count.timed_out", "bmt.count.error", "bmt.count.not_issued"):
        assert _value(m, metric_id) == 0
    issued = [e for e in read_jsonl(out.run_dir / "events.jsonl") if e["kind"] == "issued"]
    completed = {e["request_id"]: e for e in read_jsonl(out.run_dir / "events.jsonl") if e["kind"] == "completed"}
    assert [e["t_sched"] for e in issued] == [i * 50 * MS for i in range(8)]
    for e in issued:  # exact per-request values
        c = completed[e["request_id"]]
        assert c["t_complete"] - e["t_sched"] == 25 * MS  # request-completion
        assert c["t_complete"] - e["t_issue"] == 25 * MS  # DUT-service (t_submit_begin == t_issue)
    for p in (50, 90):
        assert _value(m, f"bmt.latency.request_completion.completed.p{p}") == 25 * MS
        assert _value(m, f"bmt.latency.dut_service.p{p}") == 25 * MS
    assert _value(m, "bmt.deadline.miss_rate") == 0.0
    assert _value(m, "bmt.output.identity_rate") == 1.0
    assert _value(m, "bmt.offered_rate_per_s") == 20.0
    assert _value(m, "bmt.throughput.completed_per_s") == 8e9 / 375e6


# ---------------------------------------------------------------- G-2


def test_g2_recorded_scenario_overload(bundles):
    out = bundles["G2"]
    events = read_jsonl(out.run_dir / "events.jsonl")
    admission = {e["request_id"]: (e["kind"], e["t_sched"]) for e in events
                 if e["kind"] in ("issued", "dropped", "not_issued")}
    assert admission == {0: ("issued", 0), 1: ("dropped", 33 * MS), 2: ("issued", 66 * MS), 3: ("dropped", 99 * MS)}
    final = {e["request_id"]: e["kind"] for e in events if e["kind"] in ("completed", "dropped")}
    assert final == {0: "completed", 1: "dropped", 2: "completed", 3: "dropped"}
    m = _metrics(out)
    assert _value(m, "bmt.rate.completion") == 0.5
    assert _value(m, "bmt.rate.drop") == 0.5
    assert _value(m, "bmt.deadline.miss_rate") == 1.0
    assert _value(m, "bmt.latency.dut_service.p50") == 40 * MS
    assert _value(m, "bmt.latency.dut_service.p90") == 40 * MS
    assert m["bmt.latency.dut_service.p50"].n == 2


# ---------------------------------------------------------------- G-3


def test_g3_timeout_and_late_completion(bundles):
    out = bundles["G3"]
    trace = request_trace(out.run_dir)
    assert ("timed_out", 0, 50 * MS) in trace and ("late_completion", 0, 100 * MS) in trace
    # slot occupied until the late completion: the 80 ms arrival is dropped, not issued
    assert trace.index(("timed_out", 0, 50 * MS)) < trace.index(("dropped", 1, 80 * MS)) < trace.index(
        ("late_completion", 0, 100 * MS))
    m = _metrics(out)
    assert _value(m, "bmt.count.timed_out") == 2
    assert _value(m, "bmt.count.completed") == 0  # late completions are not completed
    assert _value(m, "bmt.diagnostic.late_completion.count") == 2
    planned, issued = _value(m, "bmt.count.planned"), _value(m, "bmt.count.issued")
    assert planned == _value(m, "bmt.count.not_issued") + _value(m, "bmt.count.dropped") + issued
    assert issued == _value(m, "bmt.count.completed") + _value(m, "bmt.count.error") + _value(m, "bmt.count.timed_out")
    terminal = {e["request_id"]: e["kind"] for e in read_jsonl(out.run_dir / "events.jsonl")
                if e["kind"] in ("timed_out", "dropped", "completed")}
    assert terminal == {0: "timed_out", 1: "dropped", 2: "timed_out"}


# ---------------------------------------------------------------- G-4


def test_g4_stress_verify_failure(bundles):
    out = bundles["G4"]
    events = read_jsonl(out.run_dir / "events.jsonl")
    assert not [e for e in events if e["kind"] == "phase" and e["phase"] in ("measure", "drain")]
    assert not [e for e in events if e["kind"] in ("issued", "dropped", "not_issued", "submit_begin")]
    manifest = json.loads((out.run_dir / "manifest.json").read_text())
    assert manifest["status"]["execution_status"] == "failed"
    assert manifest["status"]["execution_reason"] == "stress_verify_failed"
    stress = read_jsonl(out.run_dir / "stress.jsonl")
    assert [(r["step"], r["outcome"]) for r in stress][-3:] == [
        ("verify", "failed"), ("deactivate", "ok"), ("restore", "ok")
    ]
    assert manifest["stress"]["status"]["restore"] == "ok"
    assert manifest["stress"]["status"]["verification"] == "failed"


# ---------------------------------------------------------------- G-5


def test_g5_censoring(bundles):
    out = bundles["G5"]
    m = _metrics(out)
    # 128 planned at 20 Hz: 14 x 90 ms items each drop one arrival; 2 x 120 ms items time out
    # (100 ms) and hold the slot until their late completion, dropping two arrivals each.
    assert _value(m, "bmt.count.planned") == 128
    assert _value(m, "bmt.count.completed") == 108
    assert _value(m, "bmt.count.dropped") == 18
    assert _value(m, "bmt.count.timed_out") == 2
    assert _value(m, "bmt.diagnostic.late_completion.count") == 2
    for p in (90, 99):
        r = m[f"bmt.latency.request_completion.planned.p{p}"]
        assert r.status is M.CENSORED and r.value is None  # rank in the dropped/timed-out region
        assert r.censored_fraction == 20 / 128
        assert not numeric_eligible(r)  # report-only: never enters numeric comparison
    assert _value(m, "bmt.latency.request_completion.planned.p50") == 30 * MS
    assert _value(m, "bmt.latency.request_completion.completed.p99") == 90 * MS  # 108 >= 100 samples
    assert _value(m, "bmt.latency.dut_service.p90") == 90 * MS
    assert _value(m, "bmt.deadline.miss_rate") == 34 / 128
    assert _value(m, "bmt.throughput.completed_per_s") == 108e9 / 6380e6
    trace = request_trace(out.run_dir)
    assert ("timed_out", 115, 5850 * MS) in trace and ("dropped", 117, 5850 * MS) in trace
    assert ("late_completion", 115, 5870 * MS) in trace and ("issued", 118, 5900 * MS) in trace


def test_g5_no_numeric_comparison_code_path_exists():
    # Stage 1A has no analyzer: no module computes deltas or test statistics.
    src = REPO_ROOT / "src" / "sdv_bmt"
    assert not (src / "analysis").exists()
    text = "\n".join(p.read_text() for p in src.rglob("*.py"))
    for needle in ("def delta", "def compare_metric", "p_value", "mannwhitney", "bootstrap"):
        assert needle not in text.lower()


# ---------------------------------------------------------------- G-6


def test_g6_re_evaluation_is_deterministic_and_leaves_raw_untouched(bundles):
    run_dir = bundles["G2"].run_dir
    raw_before = {n: file_sha256(run_dir / n) for n in RAW_EVIDENCE_FILES + ("manifest.json",)}
    first = evaluate(run_dir, eval_id="eval-g6-first")
    code = ("import json,sys; from sdv_bmt.metrics.evaluator import evaluate_run; "
            "o = evaluate_run(sys.argv[1], catalog=sys.argv[2], eval_id_factory=lambda: 'eval-g6-second'); "
            "print(o.metrics_content_sha256)")
    second_digest = subprocess.run([sys.executable, "-c", code, str(run_dir), CATALOG_VERSION], capture_output=True,
                                   text=True, check=True).stdout.strip()
    assert first.metrics_content_sha256 == second_digest  # canonical metric content identical
    assert {n: file_sha256(run_dir / n) for n in raw_before} == raw_before  # raw digests unchanged
    assert {p.name for p in (run_dir / "evaluations").iterdir()} >= {"eval-g6-first", "eval-g6-second"}
    for eval_id in ("eval-g6-first", "eval-g6-second"):
        em = json.loads((run_dir / "evaluations" / eval_id / "eval_manifest.json").read_text())
        assert em["eval_id"] == eval_id
        assert em["evaluator"]["catalog_version"] == CATALOG_VERSION
        assert em["evaluator"]["code_identity"]["source_tree_sha256"] == source_tree_sha256()
        assert em["protocol_digest"] and em["metrics_content_sha256"] == second_digest
        assert em["execution_kind"] == "mock"
    a = json.loads((run_dir / "evaluations" / "eval-g6-first" / "metrics.json").read_text())
    b = json.loads((run_dir / "evaluations" / "eval-g6-second" / "metrics.json").read_text())
    strip = lambda doc: [{k: v for k, v in m.items() if k != "eval_id"} for m in doc["metrics"]]  # noqa: E731
    assert strip(a) == strip(b) and a["eval_id"] != b["eval_id"]
    assert to_jsonable(first.metrics[0])["eval_id"] == "eval-g6-first"
