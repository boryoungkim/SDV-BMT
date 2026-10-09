"""WP5 gate: bundle-only evaluation, raw digest invariance, new eval IDs,
expected metric values and the censoring rule."""

from __future__ import annotations

import builtins
import io
import json
import shutil

import pytest

from sdv_bmt.contracts.ids import file_sha256
from sdv_bmt.contracts.records import RAW_EVIDENCE_FILES, numeric_eligible
from sdv_bmt.contracts.specs import DeadlineSpec, EvaluationOverride
from sdv_bmt.contracts.status import DeadlinePolicy, MetricStatus
from sdv_bmt.evidence.provenance import PACKAGE_ROOT
from sdv_bmt.metrics.catalog import METRICS
from sdv_bmt.metrics.evaluator import EvaluationError
from tests._helpers import MS, evaluate, load, metric_map, reseal, run_plan, variant


@pytest.fixture
def g2(tmp_path):
    return run_plan(load("g2-recorded-scenario-overload"), tmp_path / "runs")


def _raw_digests(run_dir):
    return {name: file_sha256(run_dir / name) for name in RAW_EVIDENCE_FILES + ("manifest.json",)}


# ---------------------------------------------------------------- bundle-only


def test_evaluator_opens_only_bundle_files_and_its_own_source(g2, monkeypatch):
    opened = []
    real_open = builtins.open

    def tracking_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)
    monkeypatch.setattr(io, "open", tracking_open)
    from sdv_bmt.evidence import provenance

    provenance.code_identity.cache_clear()
    evaluate(g2.run_dir)
    run_dir, src = str(g2.run_dir.resolve()), str(PACKAGE_ROOT)
    outside = [p for p in opened if not (p.startswith(run_dir) or p.startswith(src))]
    assert outside == []
    assert any(p.endswith("events.jsonl") for p in opened)


def test_evaluation_of_relocated_bundle_is_identical(g2, tmp_path):
    first = evaluate(g2.run_dir)
    moved = tmp_path / "elsewhere" / "bundle"
    shutil.copytree(g2.run_dir, moved, ignore=shutil.ignore_patterns("evaluations"))
    second = evaluate(moved, eval_id="eval-test-moved")
    assert first.metrics_content_sha256 == second.metrics_content_sha256


def test_importing_the_evaluator_loads_no_execution_component():
    import ast
    import subprocess
    import sys

    code = ("import sys, sdv_bmt.metrics.evaluator; "
            "print(sorted(m for m in sys.modules if m.startswith('sdv_bmt.')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    modules = ast.literal_eval(out.strip())
    assert "sdv_bmt.metrics.evaluator" in modules
    assert not [m for m in modules if m.split(".")[1] in {"runner", "dut", "input", "stress", "workload", "cli"}]


# ---------------------------------------------------------------- raw digests / eval ids


def test_raw_digests_unchanged_and_new_eval_ids(g2):
    before = _raw_digests(g2.run_dir)
    a = evaluate(g2.run_dir, eval_id="eval-a")
    b = evaluate(g2.run_dir, eval_id="eval-b")
    assert _raw_digests(g2.run_dir) == before
    assert a.eval_dir != b.eval_dir and a.eval_dir.is_dir() and b.eval_dir.is_dir()
    assert a.metrics_content_sha256 == b.metrics_content_sha256
    em = json.loads((a.eval_dir / "eval_manifest.json").read_text())
    for name in RAW_EVIDENCE_FILES + ("manifest.json",):
        assert em["inputs"][name]["sha256"] == before[name]
    assert em["raw_integrity"] == {"verified_before_evaluation": True, "verified_after_evaluation": True}


def test_existing_eval_id_is_never_overwritten(g2):
    first = evaluate(g2.run_dir, eval_id="eval-same")
    content = (first.eval_dir / "metrics.json").read_bytes()
    with pytest.raises(FileExistsError):
        evaluate(g2.run_dir, eval_id="eval-same")
    assert (first.eval_dir / "metrics.json").read_bytes() == content


@pytest.mark.parametrize("victim", RAW_EVIDENCE_FILES)
def test_tampered_raw_file_is_refused(g2, victim):
    with open(g2.run_dir / victim, "ab") as fh:
        fh.write(b" ")
    with pytest.raises(EvaluationError, match="raw integrity violated"):
        evaluate(g2.run_dir)
    assert not (g2.run_dir / "evaluations").exists()


# ---------------------------------------------------------------- expected values


def test_g2_expected_metrics(g2):
    m = metric_map(evaluate(g2.run_dir))
    expected = {
        "bmt.count.planned": 4, "bmt.count.completed": 2, "bmt.count.dropped": 2,
        "bmt.rate.completion": 0.5, "bmt.rate.drop": 0.5, "bmt.deadline.miss_rate": 1.0,
        "bmt.latency.dut_service.p50": 40 * MS, "bmt.latency.dut_service.p90": 40 * MS,
        "bmt.latency.request_completion.completed.p50": 40 * MS,
        "bmt.latency.output_ready.completed.p50": 40 * MS,
    }
    for metric_id, value in expected.items():
        assert m[metric_id].status is MetricStatus.VALID and m[metric_id].value == value, metric_id


def test_deadline_override_changes_only_the_deadline_metric(g2):
    base = metric_map(evaluate(g2.run_dir, eval_id="eval-base"))
    override = EvaluationOverride(1, DeadlineSpec(DeadlinePolicy.FRAME_PERIOD, 120 * MS))
    other = evaluate(g2.run_dir, eval_id="eval-override", override=override)
    om = metric_map(other)
    assert om["bmt.deadline.miss_rate"].value == 0.5
    for metric_id in base:
        if metric_id != "bmt.deadline.miss_rate":
            assert (base[metric_id].status, base[metric_id].value) == (om[metric_id].status, om[metric_id].value)
    em = json.loads((other.eval_dir / "eval_manifest.json").read_text())
    assert em["protocol"]["deadline_source"] == "override"
    assert em["protocol"]["deadline"] == {"period_ns": 120 * MS, "policy": "frame_period"}


def test_output_ready_metric_uses_t_output_ready(g2):
    def shift(events):
        for e in events:
            if e["kind"] == "completed":
                e["t_output_ready"] = e["t_complete"] + 3 * MS
        return events

    reseal(g2.run_dir, shift)
    m = metric_map(evaluate(g2.run_dir))
    assert m["bmt.latency.output_ready.completed.p50"].value == 43 * MS
    assert m["bmt.latency.request_completion.completed.p50"].value == 40 * MS
    assert m["bmt.latency.dut_service.p50"].value == 40 * MS


def test_missing_event_is_detected_as_seq_gap(g2):
    reseal(g2.run_dir, lambda events: [e for e in events if not (e["kind"] == "dropped" and e["request_id"] == 3)])
    with pytest.raises(EvaluationError, match="unreadable bundle.*0..N-1"):
        evaluate(g2.run_dir)


def test_events_inconsistent_with_run_finished_make_metrics_invalid(g2):
    def relabel(events):
        for e in events:
            if e["kind"] == "dropped" and e["request_id"] == 3:
                e["kind"] = "not_issued"
        return events

    reseal(g2.run_dir, relabel)
    metrics = evaluate(g2.run_dir).metrics
    assert {(m.status, m.status_reason) for m in metrics} == {
        (MetricStatus.INVALID, "counts_disagree_with_run_finished")
    }
    assert all(m.value is None for m in metrics)


def test_illegal_request_sequence_makes_metrics_invalid(g2):
    def late_without_timeout(events):
        for e in events:
            if e["kind"] == "completed" and e["request_id"] == 2:
                e.update(kind="late_completion", result="ok")
        return events

    reseal(g2.run_dir, late_without_timeout)
    statuses = {(m.status, m.status_reason) for m in evaluate(g2.run_dir).metrics}
    assert statuses == {(MetricStatus.INVALID, "event_sequence_invalid")}


def test_failed_run_metrics_are_not_collected(tmp_path):
    out = run_plan(load("g4-stress-verify-failure"), tmp_path)
    metrics = evaluate(out.run_dir).metrics
    assert {m.status for m in metrics} == {MetricStatus.NOT_COLLECTED}
    assert all(m.value is None and m.status_reason == "measure_phase_not_executed" for m in metrics)


# ---------------------------------------------------------------- censoring


def test_censored_planned_percentiles_are_report_only(tmp_path):
    out = run_plan(load("g5-censoring"), tmp_path)
    m = metric_map(evaluate(out.run_dir))
    for p in (90, 99):
        r = m[f"bmt.latency.request_completion.planned.p{p}"]
        assert r.status is MetricStatus.CENSORED and r.value is None
        assert r.censored_fraction == 20 / 128 and r.n == 128
        assert not numeric_eligible(r)
    p50 = m["bmt.latency.request_completion.planned.p50"]
    assert p50.status is MetricStatus.VALID and p50.value == 30 * MS and p50.censored_fraction == 20 / 128
    completed_p99 = m["bmt.latency.request_completion.completed.p99"]
    assert (completed_p99.status, completed_p99.value, completed_p99.n) == (MetricStatus.VALID, 90 * MS, 108)


def test_planned_p99_needs_100_completed_samples(tmp_path):
    # H.3: the 100-sample p99 rule counts completed samples, also for the planned
    # p99, and is applied before censoring (G-2: 4 planned, 2 completed).
    out = run_plan(load("g2-recorded-scenario-overload"), tmp_path)
    r = metric_map(evaluate(out.run_dir))["bmt.latency.request_completion.planned.p99"]
    assert (r.status, r.status_reason, r.value, r.censored_fraction) == (
        MetricStatus.INVALID, "insufficient_samples", None, 0.5)


def test_planned_p99_with_99_completed_of_100_is_invalid_not_valid(tmp_path):
    import dataclasses

    from sdv_bmt.contracts.specs import DUTConfig, parse_strict

    base = load("g5-censoring")
    dut = parse_strict(DUTConfig, {
        "schema_version": 1, "dut_config_id": "d", "adapter": "fake_blocking", "device": "v",
        "max_inflight": 1, "runtime_options": {"service_model": {
            "kind": "table", "default_service_ns": 30 * MS,
            "entries": [{"item_index": 5, "run_outcome": "error", "service_ns": 30 * MS}]}},
    })
    spec = dataclasses.replace(base.input_spec, selection=dataclasses.replace(base.input_spec.selection, count=100))
    out = run_plan(variant(base, dut=dut, input_spec=spec), tmp_path)
    m = metric_map(evaluate(out.run_dir))
    assert m["bmt.count.completed"].value == 99
    for metric_id in ("bmt.latency.request_completion.planned.p99",
                      "bmt.latency.request_completion.completed.p99"):
        assert (m[metric_id].status, m[metric_id].status_reason) == (MetricStatus.INVALID, "insufficient_samples")


def test_metric_results_are_mock_and_cover_the_catalog(g2):
    outcome = evaluate(g2.run_dir)
    assert [m.metric_id for m in outcome.metrics] == [d.metric_id for d in METRICS]
    assert {m.execution_kind.value for m in outcome.metrics} == {"mock"}
    doc = json.loads((outcome.eval_dir / "metrics.json").read_text())
    assert doc["execution_kind"] == "mock" and "MOCK" in doc["mock_notice"]
    for metric in doc["metrics"]:
        if metric["status"] != "valid":
            assert metric["value"] is None


# ---------------------------------------------------------------- review regressions


def test_malformed_manifest_files_section_is_a_clean_error(g2):
    path = g2.run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["events.jsonl"] = "not-a-record"
    path.write_text(json.dumps(manifest))
    with pytest.raises(EvaluationError, match="events.jsonl: no digest record"):
        evaluate(g2.run_dir)
    manifest["files"] = ["broken"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(EvaluationError, match="files section is not an object"):
        evaluate(g2.run_dir)


def test_interrupted_evaluation_leaves_no_eval_directory(g2, monkeypatch):
    from sdv_bmt.metrics import evaluator

    real_write = evaluator.write_json_exclusive

    def failing_write(path, value):
        if path.name == "eval_manifest.json":
            raise OSError("disk full (injected)")
        return real_write(path, value)

    monkeypatch.setattr(evaluator, "write_json_exclusive", failing_write)
    with pytest.raises(OSError, match="disk full"):
        evaluate(g2.run_dir, eval_id="eval-interrupted")
    assert not (g2.run_dir / "evaluations" / "eval-interrupted").exists()
    leftovers = [p.name for p in (g2.run_dir / "evaluations").iterdir()]
    assert leftovers == [".staging-eval-interrupted"]  # clearly marked incomplete, never a result


def test_throughput_window_starts_at_first_scheduled_arrival(tmp_path):
    from sdv_bmt.contracts.specs import ReplayPolicy, parse_strict

    base = load("g2-recorded-scenario-overload")
    with_lead = variant(base, plan={"replay": parse_strict(ReplayPolicy, {"mode": "recorded", "lead_ns": 7 * MS})})
    m0 = metric_map(evaluate(run_plan(base, tmp_path / "a").run_dir))
    m7 = metric_map(evaluate(run_plan(with_lead, tmp_path / "b").run_dir))
    assert m0["bmt.throughput.completed_per_s"].value == m7["bmt.throughput.completed_per_s"].value == 2e9 / 106e6
    assert m7["bmt.throughput.completed_per_s"].window["t0_ns"] == 7 * MS
    assert m7["bmt.throughput.completed_per_s"].window["t_measure_begin_ns"] == 0
