"""Model-level Metric Set v1 (approved design update 2026-10-08): timing, rate, deadline and
availability metrics, result layers, population metadata, and consistency with the frozen legacy
catalog.  Bundles come from the deterministic virtual runner (Stage 1A G-cases), some with request
timestamps rewritten to known values (``reseal``)."""

from __future__ import annotations

import json

import pytest

from sdv_bmt.contracts.records import numeric_eligible
from sdv_bmt.contracts.specs import DeadlineSpec, EvaluationOverride, ReplayPolicy
from sdv_bmt.contracts.status import DeadlinePolicy, MetricLayer, MetricStatus, ReplayMode
from sdv_bmt.metrics import metric_set_v1
from sdv_bmt.metrics.evaluator import CATALOGS, DEFAULT_CATALOG, EvaluationError
from tests._helpers import LEGACY_CATALOG, METRIC_SET_V1, MS, evaluate, load, metric_map, reseal, run_plan, variant

M = MetricStatus


def v1(run_dir, eval_id="eval-v1", override=None):
    return metric_map(evaluate(run_dir, eval_id=eval_id, override=override, catalog=METRIC_SET_V1))


def legacy(run_dir, eval_id="eval-legacy"):
    return metric_map(evaluate(run_dir, eval_id=eval_id, catalog=LEGACY_CATALOG))


def value(metrics, metric_id):
    m = metrics[metric_id]
    assert m.status is M.VALID, (metric_id, m.status, m.status_reason)
    return m.value


def override(period_ns=None, required=None):
    deadline = None if period_ns is None else DeadlineSpec(DeadlinePolicy.FRAME_PERIOD, period_ns)
    return EvaluationOverride(schema_version=1, deadline=deadline, required_rate_hz=required)


@pytest.fixture
def g1(tmp_path):
    return run_plan(load("g1-dataset-fixed-rate-nominal"), tmp_path / "g1")


@pytest.fixture
def g1_timed(g1):
    """G1 with known per-request timestamps (8 requests, t_sched = 50 i ms):
    submit = (2+i) ms, t_complete = t_sched + (10+2i) ms, output stage = (i % 2) ms."""

    def mutate(events):
        for e in events:
            rid = e.get("request_id")
            if rid is None:
                continue
            base = rid * 50 * MS
            if e["kind"] == "submit_end":
                e["t_run_ns"] = base + (2 + rid) * MS
            elif e["kind"] == "completed":
                e["t_complete"] = base + (10 + 2 * rid) * MS
                e["t_output_ready"] = e["t_complete"] + (rid % 2) * MS
                e["t_run_ns"] = e["t_output_ready"]
        return events

    reseal(g1.run_dir, mutate)
    return g1


# ---------------------------------------------------------------- catalog


def test_catalog_identity_layers_and_default():
    assert DEFAULT_CATALOG == METRIC_SET_V1 and set(CATALOGS) == {METRIC_SET_V1, LEGACY_CATALOG}
    ids = [m.metric_id for m in metric_set_v1.METRICS]
    assert len(ids) == len(set(ids)) == 91
    layers = {m.metric_id: m.layer for m in metric_set_v1.METRICS}
    assert layers["bmt.dut_service.p99"] is MetricLayer.PERFORMANCE
    assert layers["bmt.submit.p50"] is MetricLayer.DIAGNOSTICS
    assert layers["bmt.issue_lag.max_observed"] is MetricLayer.DIAGNOSTICS
    assert layers["bmt.device_execution_bracket.p50"] is MetricLayer.DIAGNOSTICS
    assert layers["bmt.deadline.output_ready_miss_rate"] is MetricLayer.TIMELINESS
    assert layers["bmt.required_rate_met"] is MetricLayer.TIMELINESS
    assert layers["quality.detection.conditional.ap"] is MetricLayer.TASK_QUALITY
    assert layers["bmt.output.identity_rate"] is MetricLayer.DIAGNOSTICS
    assert layers["data.traceability.complete"] is MetricLayer.VALIDITY_EVIDENCE
    # the approved names exist exactly; no WCET/worst-case/capacity/pure-device names
    for family in ("request_completion", "dut_service", "output_ready", "issue_lag", "submit", "post_submit",
                   "output_stage", "h2d", "device_execution_bracket", "d2h"):
        assert {f"bmt.{family}.{s}" for s in ("p50", "p90", "p99", "max_observed")} <= set(ids)
    joined = " ".join(ids)
    for forbidden in ("wcet", "worst_case", "capacity", "peak", "maximum_throughput", "pure_device",
                      "device_busy", "accuracy", "tn_count", "ftti"):
        assert forbidden not in joined
    assert not [i for i in ids if i.startswith("quality.calibration.") or i.startswith("quality.numeric.")]


def test_unknown_catalog_is_refused(g1):
    from sdv_bmt.metrics.evaluator import evaluate_run

    with pytest.raises(EvaluationError):
        evaluate_run(g1.run_dir, catalog="bmt-portable-v2-unknown")


# ---------------------------------------------------------------- timing


def test_exact_submit_post_submit_output_stage_formulas(g1_timed):
    m = v1(g1_timed.run_dir)
    # nearest rank over 8 values: p50 = 4th smallest, p90 = 8th
    assert value(m, "bmt.submit.p50") == 5 * MS and value(m, "bmt.submit.p90") == 9 * MS
    assert value(m, "bmt.submit.max_observed") == 9 * MS
    assert value(m, "bmt.post_submit.p50") == 11 * MS and value(m, "bmt.post_submit.max_observed") == 15 * MS
    assert value(m, "bmt.output_stage.p50") == 0 and value(m, "bmt.output_stage.p90") == 1 * MS
    assert value(m, "bmt.output_stage.max_observed") == 1 * MS
    assert value(m, "bmt.dut_service.p50") == 16 * MS and value(m, "bmt.dut_service.max_observed") == 24 * MS
    assert value(m, "bmt.request_completion.p50") == 16 * MS
    assert value(m, "bmt.output_ready.p50") == 17 * MS and value(m, "bmt.output_ready.max_observed") == 25 * MS
    pop = m["bmt.submit.p50"].population
    assert pop == {"name": "completed_requests", "sample_count": 8, "planned_count": 8, "completed_count": 8,
                   "coverage": 1.0}


def test_p99_needs_100_samples_and_max_observed_is_not_wcet(g1):
    m = v1(g1.run_dir)
    for family in ("request_completion", "dut_service", "output_ready", "issue_lag", "submit", "post_submit",
                   "output_stage"):
        r = m[f"bmt.{family}.p99"]
        assert (r.status, r.status_reason, r.value) == (M.INVALID, "insufficient_samples", None)
        assert m[f"bmt.{family}.max_observed"].status is M.VALID
        assert "It is not WCET" in m[f"bmt.{family}.max_observed"].method
    assert "not a worst-case timing guarantee" in " ".join(m["bmt.dut_service.p50"].notes)


def test_censored_planned_population_is_report_only(tmp_path):
    m = v1(run_plan(load("g5-censoring"), tmp_path).run_dir)
    assert value(m, "bmt.censored_fraction") == 20 / 128
    for p in (90, 99):
        r = m[f"bmt.request_completion.planned.p{p}"]
        assert r.status is M.CENSORED and r.value is None and r.censored_fraction == 20 / 128
        assert not numeric_eligible(r)
    assert value(m, "bmt.request_completion.planned.p50") == 30 * MS
    completed_p99 = m["bmt.request_completion.p99"]  # 108 completed >= 100: valid, with its coverage
    assert completed_p99.value == 90 * MS and completed_p99.population["coverage"] == 108 / 128
    assert completed_p99.population["sample_count"] == 108 and completed_p99.population["planned_count"] == 128


def test_values_equal_the_frozen_legacy_catalog_where_semantics_are_shared(tmp_path):
    pairs = [(f"bmt.request_completion.p{p}", f"bmt.latency.request_completion.completed.p{p}") for p in (50, 90, 99)]
    pairs += [(f"bmt.request_completion.planned.p{p}", f"bmt.latency.request_completion.planned.p{p}")
              for p in (50, 90, 99)]
    pairs += [(f"bmt.dut_service.p{p}", f"bmt.latency.dut_service.p{p}") for p in (50, 90, 99)]
    pairs += [(f"bmt.output_ready.p{p}", f"bmt.latency.output_ready.completed.p{p}") for p in (50, 90, 99)]
    pairs += [("bmt.issue_lag.p99", "bmt.issue_lag.p99"), ("bmt.issue_lag.max_observed", "bmt.issue_lag.max"),
              ("bmt.offered_rate", "bmt.offered_rate_per_s"), ("bmt.completion_rate", "bmt.rate.completion"),
              ("bmt.drop_rate", "bmt.rate.drop"), ("bmt.timeout_rate", "bmt.rate.timeout"),
              ("bmt.error_rate", "bmt.rate.error"), ("bmt.late_completion_count", "bmt.diagnostic.late_completion.count"),
              ("bmt.deadline.completion_miss_rate", "bmt.deadline.miss_rate"),
              ("bmt.output.identity_rate", "bmt.output.identity_rate")]
    pairs += [(f"bmt.count.{n}", f"bmt.count.{n}") for n in metric_set_v1.COUNT_NAMES]
    for name in ("g1-dataset-fixed-rate-nominal", "g2-recorded-scenario-overload", "g3-timeout-late-completion",
                 "g5-censoring"):
        out = run_plan(load(name), tmp_path / name)
        new, old = v1(out.run_dir), legacy(out.run_dir)
        for a, b in pairs:
            assert (new[a].status, new[a].value, new[a].censored_fraction) == \
                   (old[b].status, old[b].value, old[b].censored_fraction), (name, a)
        assert new["bmt.achieved_completion_rate"].value == old["bmt.throughput.completed_per_s"].value


# ---------------------------------------------------------------- deadline


@pytest.mark.parametrize("period_ms, completion_misses, output_ready_misses", [
    (24, 0, 1),   # completion within the deadline, output ready too late (request 7: 24 ms vs 25 ms)
    (21, 2, 2),   # both views miss for requests 6 and 7
    (50, 0, 0),   # both meet
])
def test_completion_and_output_ready_deadline_views(g1_timed, period_ms, completion_misses, output_ready_misses):
    m = v1(g1_timed.run_dir, override=override(period_ms * MS))
    assert value(m, "bmt.deadline.completion_miss_count") == completion_misses
    assert value(m, "bmt.deadline.output_ready_miss_count") == output_ready_misses
    assert value(m, "bmt.deadline.completion_miss_rate") == completion_misses / 8
    assert value(m, "bmt.deadline.output_ready_miss_rate") == output_ready_misses / 8
    assert "not FTTI" in m["bmt.deadline.output_ready_miss_rate"].method


def test_dropped_and_timed_out_requests_count_as_misses(tmp_path):
    g2 = v1(run_plan(load("g2-recorded-scenario-overload"), tmp_path / "g2").run_dir, override=override(10_000 * MS))
    assert value(g2, "bmt.count.dropped") == 2
    assert value(g2, "bmt.deadline.completion_miss_count") == 2 == value(g2, "bmt.deadline.output_ready_miss_count")
    g3 = v1(run_plan(load("g3-timeout-late-completion"), tmp_path / "g3").run_dir, override=override(10_000 * MS))
    assert value(g3, "bmt.count.timed_out") == 2 and value(g3, "bmt.count.completed") == 0
    assert value(g3, "bmt.deadline.completion_miss_count") == 3  # 2 timed out + 1 dropped
    assert value(g3, "bmt.late_completion_count") == 2


def test_deadline_is_an_evaluation_time_parameter(g1_timed):
    run_dir = g1_timed.run_dir
    raw_before = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()}
    base, changed = v1(run_dir, "eval-a"), v1(run_dir, "eval-b", override=override(24 * MS))
    differing = {k for k in base if (base[k].status, base[k].value) != (changed[k].status, changed[k].value)}
    assert differing == {"bmt.deadline.output_ready_miss_count", "bmt.deadline.output_ready_miss_rate"}
    em = json.loads((run_dir / "evaluations" / "eval-b" / "eval_manifest.json").read_text())
    assert em["protocol"]["deadline"]["period_ns"] == 24 * MS and em["parameter_sources"]["deadline"] == "override"
    assert {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()} == raw_before
    none = v1(run_dir, "eval-c", override=override(None))
    assert none["bmt.deadline.completion_miss_rate"].status is M.NOT_APPLICABLE


# ---------------------------------------------------------------- rates


def test_fixed_rate_offered_and_achieved_completion_rate(g1):
    m = v1(g1.run_dir)
    assert value(m, "bmt.offered_rate") == 20.0  # 7 intervals over 350 ms
    assert value(m, "bmt.achieved_completion_rate") == 8e9 / 375e6  # window [t0, drain end]
    closed = m["bmt.closed_loop_completion_rate"]
    assert (closed.status, closed.status_reason) == (M.NOT_APPLICABLE, "open_loop_replay:fixed_rate")
    assert m["bmt.offered_rate"].population["replay_mode"] == "fixed_rate"
    assert m["bmt.offered_rate"].population["configured_n"] == 1


def test_achieved_completion_rate_with_drops(tmp_path):
    m = v1(run_plan(load("g5-censoring"), tmp_path).run_dir)
    assert value(m, "bmt.offered_rate") == 20.0
    assert value(m, "bmt.achieved_completion_rate") == 108e9 / 6380e6
    assert value(m, "bmt.drop_rate") == 18 / 128 and value(m, "bmt.completion_rate") == 108 / 128


def test_asap_reports_only_the_closed_loop_completion_rate(tmp_path):
    rp = load("g1-dataset-fixed-rate-nominal")
    rp = variant(rp, plan={"replay": ReplayPolicy(mode=ReplayMode.ASAP)})
    m = v1(run_plan(rp, tmp_path).run_dir)
    assert value(m, "bmt.closed_loop_completion_rate") == 8e9 / 200e6  # 8 x 25 ms back to back
    assert m["bmt.offered_rate"].status is M.NOT_APPLICABLE
    assert m["bmt.achieved_completion_rate"].status_reason == "closed_loop_replay"
    method = m["bmt.closed_loop_completion_rate"].method
    assert "not a maximum, peak or capacity claim" in method
    assert m["bmt.closed_loop_completion_rate"].population["configured_n"] == 1


def test_required_rate_rule(tmp_path, g1):
    met = v1(g1.run_dir, "eval-20", override=override(50 * MS, required=20.0))
    assert value(met, "bmt.required_rate") == 20.0 and value(met, "bmt.required_rate_met") == 1
    above = v1(g1.run_dir, "eval-25", override=override(50 * MS, required=25.0))
    r = above["bmt.required_rate_met"]  # the run never offered 25 Hz: not tested, not "false"
    assert (r.status, r.status_reason) == (M.NOT_APPLICABLE, "offered_rate_below_required_rate")
    none = v1(g1.run_dir, "eval-none")
    assert none["bmt.required_rate"].status is M.NOT_APPLICABLE
    assert none["bmt.required_rate_met"].status_reason == "no_required_rate_parameter"
    g5 = run_plan(load("g5-censoring"), tmp_path / "g5")  # offered 20 Hz, achieved 16.9/s (drops)
    assert value(v1(g5.run_dir, override=override(50 * MS, required=20.0)), "bmt.required_rate_met") == 0
    g2 = run_plan(load("g2-recorded-scenario-overload"), tmp_path / "g2")
    rec = v1(g2.run_dir, override=override(33 * MS, required=1.0))["bmt.required_rate_met"]
    assert (rec.status, rec.status_reason) == (M.NOT_APPLICABLE, "no_periodic_offered_rate:recorded")
    asap = run_plan(variant(load("g1-dataset-fixed-rate-nominal"), plan={"replay": ReplayPolicy(mode=ReplayMode.ASAP)}),
                    tmp_path / "asap")
    closed = v1(asap.run_dir, override=override(None, required=1.0))["bmt.required_rate_met"]
    assert (closed.status, closed.status_reason) == (M.NOT_APPLICABLE, "closed_loop_replay")


def test_legacy_catalog_refuses_metric_set_parameters(g1):
    with pytest.raises(EvaluationError):
        evaluate(g1.run_dir, override=override(50 * MS, required=20.0), catalog=LEGACY_CATALOG)


# ---------------------------------------------------------------- availability and population context


def test_counts_rates_and_population_context(tmp_path):
    out = run_plan(load("g3-timeout-late-completion"), tmp_path)
    outcome = evaluate(out.run_dir, catalog=METRIC_SET_V1)
    m = metric_map(outcome)
    assert value(m, "bmt.count.timed_out") == 2 and value(m, "bmt.timeout_rate") == 2 / 3
    assert value(m, "bmt.not_issued_rate") == 0.0 and value(m, "bmt.censored_fraction") == 1.0
    assert m["bmt.request_completion.p50"].status is M.EMPTY  # no completed sample: never a number
    doc = json.loads((outcome.eval_dir / "metrics.json").read_text())
    assert doc["schema_version"] == 2 and doc["catalog_version"] == METRIC_SET_V1
    ctx = doc["context"]
    assert ctx["planned_count"] == 3 and ctx["completed_count"] == 0 and ctx["run_count"] == 1
    assert ctx["input_selection_digest"] and ctx["items_digest"] and ctx["measurement_boundary_id"]
    assert ctx["bundle_metric_definition"] == METRIC_SET_V1 and ctx["configured_n"] == 1
    for metric in doc["metrics"]:
        assert metric["layer"] in {layer.value for layer in MetricLayer}
        assert "name" in metric["population"] and "sample_count" in metric["population"]
    em = json.loads((outcome.eval_dir / "eval_manifest.json").read_text())
    assert em["schema_version"] == 2 and em["evaluator"]["catalog_version"] == METRIC_SET_V1
    assert em["evaluator"]["metric_layers"]["bmt.submit.p50"] == "diagnostics"
    assert em["bundle"]["catalog_matches_bundle_metric_definition"] is True


def test_failed_run_metrics_are_not_collected(tmp_path):
    out = run_plan(load("g4-stress-verify-failure"), tmp_path)
    assert {m.status for m in evaluate(out.run_dir, catalog=METRIC_SET_V1).metrics} == {M.NOT_COLLECTED}


def test_device_segments_are_not_supported_without_device_timing(g1):
    m = v1(g1.run_dir)
    for family in ("h2d", "device_execution_bracket", "d2h"):
        r = m[f"bmt.{family}.p50"]
        assert (r.status, r.status_reason) == (M.NOT_SUPPORTED, "no_device_timing_in_measurement_boundary")


def test_v3_bundle_has_no_task_quality_and_identity_stays_diagnostic(g1):
    m = v1(g1.run_dir)
    q = m["quality.detection.conditional.ap"]
    assert (q.status, q.status_reason) == (M.NOT_COLLECTED, "quality_evaluator_not_declared")
    assert m["quality.detection.coverage"].status is M.NOT_COLLECTED
    assert m["data.integrity.annotation_verified_rate"].status_reason == "annotation_content_not_in_bundle"
    assert value(m, "data.item_count") == 8 and value(m, "data.annotation_coverage") == 1.0
    assert value(m, "data.integrity.payload_verified_rate") == 1.0 and value(m, "data.traceability.complete") == 1
    assert m["data.integrity.prepared_input_verified_rate"].status is M.NOT_COLLECTED  # virtual runner
    identity = m["bmt.output.identity_rate"]
    assert identity.value == 1.0 and identity.layer is MetricLayer.DIAGNOSTICS and "not accuracy" in identity.method


# ---------------------------------------------------------------- evidence check (v0.1.0 release hardening)


def _checks(outcome):
    return {c["check"]: c for c in outcome.evidence_checks}


def test_evidence_check_passes_on_consistent_bundles(g1, tmp_path):
    outcome = evaluate(g1.run_dir, catalog=METRIC_SET_V1)
    check = _checks(outcome)["completed_beyond_request_timeout"]
    assert (check["status"], check["count"], check["request_ids"]) == ("pass", 0, [])
    doc = json.loads((outcome.eval_dir / "metrics.json").read_text())
    assert doc["evidence_checks"] == list(outcome.evidence_checks)
    failed = run_plan(load("g4-stress-verify-failure"), tmp_path / "g4")
    assert _checks(evaluate(failed.run_dir, catalog=METRIC_SET_V1))["completed_beyond_request_timeout"]["status"] == \
        "not_applicable"


def test_evidence_check_flags_completed_beyond_timeout_without_repairing_state(g1):
    timeout = load("g1-dataset-fixed-rate-nominal").plan.run.request_timeout_ns

    def stretch(events):  # request 0 reported COMPLETED although its timer was due before t_complete
        begin = next(e["t_run_ns"] for e in events if e["kind"] == "submit_begin" and e["request_id"] == 0)
        for e in events:
            if e["kind"] == "completed" and e["request_id"] == 0:
                e["t_complete"] = begin + timeout + 1
        return events

    reseal(g1.run_dir, stretch)
    outcome = evaluate(g1.run_dir, catalog=METRIC_SET_V1)
    check = _checks(outcome)["completed_beyond_request_timeout"]
    assert (check["status"], check["count"], check["request_ids"], check["request_timeout_ns"]) == \
        ("flagged", 1, [0], timeout)
    m = metric_map(outcome)
    assert value(m, "bmt.count.completed") == 8 and value(m, "bmt.count.timed_out") == 0  # raw states unchanged
    from sdv_bmt.metrics.evaluator import canonical_metric_set_content

    assert "evidence_checks" not in canonical_metric_set_content({}, {}, [])
    legacy_doc = json.loads((evaluate(g1.run_dir, eval_id="eval-l", catalog=LEGACY_CATALOG).eval_dir
                             / "metrics.json").read_text())
    assert "evidence_checks" not in legacy_doc  # the legacy evaluation format is frozen


def test_evidence_check_flags_the_preserved_pre_fix_gpu_bundle(tmp_path):
    """Local evidence: the 1 ms-timeout device-test bundle recorded by the pre-fix native loop (request 20
    COMPLETED 3.477 ms after submit), preserved in runs/stage2b-g-msv1; skipped where absent."""
    import shutil

    from tests._helpers import REPO_ROOT

    src = (REPO_ROOT / "runs" / "stage2b-g-msv1" / "analysis" / "flaky_device_test" / "test_short_timeout_gives_timed0"
           / "rg-gpu-timeout-1ms" / "run-test-0")
    if not src.is_dir():
        pytest.skip("pre-fix evidence bundle not present on this host")
    shutil.copytree(src, tmp_path / "bundle", ignore=shutil.ignore_patterns("evaluations"))
    outcome = evaluate(tmp_path / "bundle", catalog=METRIC_SET_V1)
    check = _checks(outcome)["completed_beyond_request_timeout"]
    assert (check["status"], check["request_ids"]) == ("flagged", [20])
    m = metric_map(outcome)
    assert value(m, "bmt.count.completed") == 1 and value(m, "bmt.count.timed_out") == 31
