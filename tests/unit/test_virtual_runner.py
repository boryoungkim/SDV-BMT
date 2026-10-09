"""WP4 gate: virtual admission semantics, request-state sequences and count identity."""

from __future__ import annotations

import dataclasses

import pytest

from sdv_bmt.contracts.specs import DUTConfig, HardwareStressSpec, ReplayPolicy, RunConfig, WorkloadSpec, parse_strict
from sdv_bmt.contracts.status import ExecutionStatus, MeasurementStatus
from sdv_bmt.runner.core import count_identity_violations
from tests._helpers import MS, load, read_jsonl, request_trace, run_plan, variant


def _dut(model: dict) -> DUTConfig:
    return parse_strict(DUTConfig, {
        "schema_version": 1, "dut_config_id": "fake-test", "adapter": "fake_blocking",
        "device": "virtual-0", "max_inflight": 1, "runtime_options": {"service_model": model},
    })


def _fixed(ms: int) -> DUTConfig:
    return _dut({"kind": "fixed", "service_ns": ms * MS})


def _run_cfg(timeout_ms: int, drain_ms: int) -> RunConfig:
    return parse_strict(RunConfig, {
        "max_inflight": 1, "queue_capacity": 0, "on_full": "drop_new",
        "request_timeout_ns": timeout_ms * MS, "drain_ns": drain_ms * MS,
    })


def _replay(**kw) -> ReplayPolicy:
    return parse_strict(ReplayPolicy, kw)


def _events(outcome, kind):
    return [e for e in read_jsonl(outcome.run_dir / "events.jsonl") if e["kind"] == kind]


# ---------------------------------------------------------------- golden-shaped sequences


def test_g2_overload_sequence(tmp_path):
    out = run_plan(load("g2-recorded-scenario-overload"), tmp_path)
    assert request_trace(out.run_dir) == [
        ("issued", 0, 0), ("submit_begin", 0, 0),
        ("dropped", 1, 33 * MS),
        ("submit_end", 0, 40 * MS), ("completed", 0, 40 * MS),
        ("issued", 2, 66 * MS), ("submit_begin", 2, 66 * MS),
        ("dropped", 3, 99 * MS),
        ("submit_end", 2, 106 * MS), ("completed", 2, 106 * MS),
    ]
    assert out.request_counts["completed"] == 2 and out.request_counts["dropped"] == 2
    assert {e["reason"] for e in _events(out, "dropped")} == {"no_free_slot"}


def test_g3_timeout_holds_slot_until_late_completion(tmp_path):
    out = run_plan(load("g3-timeout-late-completion"), tmp_path)
    assert request_trace(out.run_dir) == [
        ("issued", 0, 0), ("submit_begin", 0, 0),
        ("timed_out", 0, 50 * MS),
        ("dropped", 1, 80 * MS),  # slot still occupied by the timed-out call
        ("submit_end", 0, 100 * MS), ("late_completion", 0, 100 * MS),
        ("issued", 2, 160 * MS), ("submit_begin", 2, 160 * MS),
        ("timed_out", 2, 210 * MS),
        ("submit_end", 2, 260 * MS), ("late_completion", 2, 260 * MS),
    ]
    counts = out.request_counts
    assert (counts["completed"], counts["timed_out"], counts["dropped"], counts["late_completion"]) == (0, 2, 1, 2)
    assert count_identity_violations(counts) == []
    late = _events(out, "late_completion")
    assert [e["result"] for e in late] == ["ok", "ok"]
    assert all(e["output_ref"]["length"] > 0 for e in late)


# ---------------------------------------------------------------- tie rules


def test_slot_freed_at_t_is_available_to_arrival_at_t(tmp_path):
    rp = variant(load("g3-timeout-late-completion"), dut=_fixed(40),
                 plan={"replay": _replay(mode="fixed_rate", rate_hz=25), "run": _run_cfg(1000, 1000)})
    out = run_plan(rp, tmp_path)
    assert out.request_counts["completed"] == 3 and out.request_counts["dropped"] == 0
    assert request_trace(out.run_dir)[2:5] == [
        ("submit_end", 0, 40 * MS), ("completed", 0, 40 * MS), ("issued", 1, 40 * MS)
    ]


def test_completion_exactly_at_timeout_counts_as_completed(tmp_path):
    rp = variant(load("g3-timeout-late-completion"), dut=_fixed(50),
                 plan={"replay": _replay(mode="fixed_rate", rate_hz=12.5), "run": _run_cfg(50, 100)})
    out = run_plan(rp, tmp_path)
    assert out.request_counts["completed"] == 3 and out.request_counts["timed_out"] == 0


# ---------------------------------------------------------------- asap


def test_asap_schedules_each_request_when_slot_frees(tmp_path):
    rp = variant(load("g3-timeout-late-completion"), dut=_fixed(25),
                 plan={"replay": _replay(mode="asap"), "run": _run_cfg(1000, 1000)})
    out = run_plan(rp, tmp_path)
    issued = _events(out, "issued")
    assert [(e["t_sched"], e["t_issue"]) for e in issued] == [(0, 0), (25 * MS, 25 * MS), (50 * MS, 50 * MS)]
    assert out.request_counts["dropped"] == 0


def test_asap_waits_for_late_completion_after_timeout(tmp_path):
    rp = variant(load("g3-timeout-late-completion"), plan={"replay": _replay(mode="asap")})
    out = run_plan(rp, tmp_path)
    assert [e["t_sched"] for e in _events(out, "issued")] == [0, 100 * MS, 200 * MS]
    assert out.request_counts["timed_out"] == 3 and out.request_counts["late_completion"] == 3


# ---------------------------------------------------------------- error / not_issued paths


def test_dut_error_is_request_level_and_run_continues(tmp_path):
    dut = _dut({"kind": "table", "default_service_ns": 10 * MS,
                "entries": [{"item_index": 1, "service_ns": 5 * MS, "run_outcome": "error", "error_code": "E_FAKE"}]})
    rp = variant(load("g1-dataset-fixed-rate-nominal"), dut=dut)
    out = run_plan(rp, tmp_path)
    errors = _events(out, "error")
    assert [(e["request_id"], e["error_phase"], e["error"]["code"], e["t_complete"]) for e in errors] == [
        (1, "dut_run", "E_FAKE", 55 * MS)
    ]
    counts = out.request_counts
    assert (counts["completed"], counts["error"]) == (7, 1)
    assert out.execution_status is ExecutionStatus.COMPLETED
    assert out.measurement_status is MeasurementStatus.VALID


def test_postprocess_error_becomes_error_with_phase(tmp_path):
    base = load("g1-dataset-fixed-rate-nominal")
    workload = dataclasses.replace(base.workload, options={"postprocess_error_items": [2]})
    out = run_plan(variant(base, workload=workload), tmp_path)
    errors = _events(out, "error")
    assert [(e["request_id"], e["error_phase"], e["error"]["code"]) for e in errors] == [
        (2, "postprocess", "WorkloadError")
    ]
    assert out.request_counts["completed"] == 7


def test_preparation_failure_becomes_not_issued(tmp_path):
    dut = _dut({"kind": "table", "default_service_ns": 10 * MS,
                "entries": [{"item_index": 3, "prepare_input_outcome": "error"}]})
    out = run_plan(variant(load("g1-dataset-fixed-rate-nominal"), dut=dut), tmp_path)
    not_issued = _events(out, "not_issued")
    assert [(e["request_id"], e["t_sched"], e["reason"]) for e in not_issued] == [
        (3, 150 * MS, "prepare_failed:prepare_input")
    ]
    counts = out.request_counts
    assert (counts["planned"], counts["not_issued"], counts["issued"], counts["completed"]) == (8, 1, 7, 7)
    assert count_identity_violations(counts) == []
    phase = [e for e in _events(out, "phase") if e["phase"] == "prepare_inputs" and e["edge"] == "end"][0]
    assert phase["outcome"] == "partial" and phase["details"]["failed_items"][0]["item_index"] == 3


def test_adapter_exception_is_request_level_error(tmp_path):
    from sdv_bmt.cli.main import build_components

    def factory(rp):
        c = build_components(rp)
        original = c.dut.run

        def run(request):
            if request.request_id == 4:
                raise RuntimeError("device hiccup")
            return original(request)

        c.dut.run = run
        return c

    out = run_plan(load("g1-dataset-fixed-rate-nominal"), tmp_path, factory=factory)
    errors = _events(out, "error")
    assert errors[0]["request_id"] == 4 and errors[0]["error"]["code"] == "adapter_exception"
    assert errors[0]["t_complete"] == 200 * MS  # zero virtual service time
    assert out.request_counts["completed"] == 7


def test_late_completion_with_postprocess_error(tmp_path):
    base = load("g3-timeout-late-completion")
    workload = dataclasses.replace(base.workload, options={"postprocess_error_items": [0]})
    out = run_plan(variant(base, workload=workload), tmp_path)
    late = _events(out, "late_completion")
    assert (late[0]["request_id"], late[0]["result"], late[0]["error_phase"]) == (0, "error", "postprocess")
    assert "output_ref" not in late[0]
    assert late[1]["result"] == "ok"
    assert out.request_counts["timed_out"] == 2 and out.request_counts["error"] == 0


def test_late_completion_with_dut_error(tmp_path):
    dut = _dut({"kind": "table", "default_service_ns": 100 * MS,
                "entries": [{"item_index": 0, "run_outcome": "error", "service_ns": 70 * MS}]})
    out = run_plan(variant(load("g3-timeout-late-completion"), dut=dut), tmp_path)
    late = _events(out, "late_completion")
    assert (late[0]["request_id"], late[0]["result"], late[0]["error_phase"], late[0]["t_complete"]) == (
        0, "error", "dut_run", 70 * MS)
    assert out.request_counts["error"] == 0 and out.request_counts["timed_out"] == 2


# ---------------------------------------------------------------- drain


def test_drain_cap_reports_outstanding_late_completion(tmp_path):
    rp = variant(load("g3-timeout-late-completion"), dut=_fixed(300),
                 plan={"replay": _replay(mode="fixed_rate", rate_hz=1), "run": _run_cfg(50, 100)})
    rp = variant(rp, input_spec=dataclasses.replace(rp.input_spec, selection=dataclasses.replace(
        rp.input_spec.selection, count=1)))
    out = run_plan(rp, tmp_path)
    drain_end = [e for e in _events(out, "phase") if e["phase"] == "drain" and e["edge"] == "end"][0]
    assert drain_end["t_run_ns"] == 100 * MS
    assert drain_end["details"] == {"cap_reached": True, "outstanding_request_ids": [0]}
    assert out.request_counts["timed_out"] == 1 and out.request_counts["late_completion"] == 0
    assert out.measurement_status is MeasurementStatus.VALID


def test_drain_must_cover_request_timeout(tmp_path):
    from sdv_bmt.runner.runner import PlanRejected

    rp = variant(load("g1-dataset-fixed-rate-nominal"), plan={"run": _run_cfg(100, 50)})
    with pytest.raises(PlanRejected, match="drain_ns must be >= run.request_timeout_ns"):
        run_plan(rp, tmp_path)


# ---------------------------------------------------------------- run clock offsets


def test_verify_window_and_lead_shift_the_schedule(tmp_path):
    base = load("g2-recorded-scenario-overload")
    stress = dataclasses.replace(base.stress_spec, verification=dataclasses.replace(
        base.stress_spec.verification, window_ns=5 * MS))
    rp = variant(base, stress_spec=stress,
                 plan={"replay": _replay(mode="recorded", lead_ns=2 * MS)})
    out = run_plan(rp, tmp_path)
    measure = [e for e in _events(out, "phase") if e["phase"] == "measure" and e["edge"] == "begin"][0]
    assert measure["t_run_ns"] == 5 * MS and measure["details"]["t0_ns"] == 7 * MS
    assert [e["t_sched"] for e in _events(out, "issued")] == [7 * MS, 73 * MS]


# ---------------------------------------------------------------- determinism and identity


RAW = ("run_spec.json", "input_items.jsonl", "events.jsonl", "stress.jsonl", "outputs.bin")


@pytest.mark.parametrize("name", [
    "g1-dataset-fixed-rate-nominal", "g2-recorded-scenario-overload",
    "g3-timeout-late-completion", "g5-censoring", "scenario-two-episodes-fake-stress",
])
def test_same_plan_gives_byte_identical_raw_files(tmp_path, name):
    rp = load(name)
    a = run_plan(rp, tmp_path / "a", run_id="run-a")
    b = run_plan(rp, tmp_path / "b", run_id="run-b")
    for f in RAW:
        assert (a.run_dir / f).read_bytes() == (b.run_dir / f).read_bytes(), f


@pytest.mark.parametrize("name", [
    "g1-dataset-fixed-rate-nominal", "g2-recorded-scenario-overload",
    "g3-timeout-late-completion", "g5-censoring", "scenario-two-episodes-fake-stress",
])
def test_count_identity_holds(tmp_path, name):
    out = run_plan(load(name), tmp_path)
    assert count_identity_violations(out.request_counts) == []
    assert out.request_counts["unresolved"] == 0
    finished = _events(out, "run_finished")[0]
    assert finished["request_counts"] == out.request_counts


def test_events_carry_no_run_id_and_use_virtual_clock(tmp_path):
    out = run_plan(load("g2-recorded-scenario-overload"), tmp_path, run_id="run-unique-xyz")
    text = (out.run_dir / "events.jsonl").read_text() + (out.run_dir / "stress.jsonl").read_text()
    assert "run-unique-xyz" not in text
    records = read_jsonl(out.run_dir / "events.jsonl") + read_jsonl(out.run_dir / "stress.jsonl")
    assert {r["run_clock_domain"] for r in records} == {"virtual"}
    assert sorted(r["seq"] for r in records) == list(range(len(records)))
