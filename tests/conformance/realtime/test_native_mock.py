"""Stage 1R: the Python control plane driving the native measurement plane with its MOCK backend.

Real host time is used, so these tests assert Stage 1A semantics with margins that do not depend on
host speed (a long request is made with ~10^6 polls; one poll costs well under 1 us on the native path).
"""

from __future__ import annotations

import dataclasses
import json
import os
import stat
import subprocess
import sys

import pytest

from sdv_bmt import REALTIME_MOCK_NOTICE
from sdv_bmt.contracts.specs import DUTConfig, RunConfig
from sdv_bmt.contracts.status import ExecutionStatus, MeasurementStatus
from sdv_bmt.evidence.validity import evaluate_contrast_validity
from sdv_bmt.observation.reader import fold_requests, load_run_bundle
from sdv_bmt.runner.plan import load_plan
from sdv_bmt.runner.runner import PlanRejected
from tests._helpers import (CONFIGS, evaluate, load, metric_map, plan_file, plan_path, read_jsonl, run_plan,
                            variant)

pytestmark = pytest.mark.usefixtures("native_build")
FLOOR = "r1-fake-native-harness-floor"
LONG = 3_000_000  # polls: tens of milliseconds or more on any host


def _with_dut(rp, **options):
    dut = DUTConfig(schema_version=1, dut_config_id="fake-native-test", adapter="fake_native", device="host-only",
                    max_inflight=1, runtime_options={"polls_until_complete": 1, **options})
    return variant(rp, dut=dut)


def _with_run(rp, **run):
    return variant(rp, plan={"run": dataclasses.replace(rp.plan.run, **run)})


def _states(run_dir):
    events = tuple(read_jsonl(run_dir / "events.jsonl"))
    return {rid: req.state.value for rid, (req, _) in fold_requests(events).items()}, events


@pytest.fixture(scope="module")
def floor_run(tmp_path_factory):
    return run_plan(load(FLOOR), tmp_path_factory.mktemp("floor"), run_id="run-floor-0")


def test_mock_native_run_is_a_valid_v4_bundle(floor_run):
    out = floor_run
    assert out.execution_status is ExecutionStatus.COMPLETED and out.measurement_status is MeasurementStatus.VALID
    assert out.request_counts["completed"] == 128 and out.request_counts["unresolved"] == 0
    bundle = load_run_bundle(out.run_dir)  # strict reader: one seq counter across Python and native writers
    manifest = bundle.manifest
    assert manifest["schema_version"] == 4 and manifest["execution_kind"] == "mock"
    assert manifest["mock_notice"] == REALTIME_MOCK_NOTICE
    assert manifest["realtime"]["measurement_plane"] == "native"
    assert manifest["realtime"]["native_counts"] == out.request_counts
    assert manifest["dut"]["descriptor"]["device_identity"]["native.runner"] == "sdv-bmt-native-fake"
    phases = [(e["phase"], e["edge"]) for e in bundle.events if e["kind"] == "phase"]
    order = [p for p, edge in phases if edge == "begin"]
    assert order.index("dut.prepare") < order.index("prepare_inputs") < order.index("stress.verify") < order.index("measure")
    definition = manifest["measurement_boundary"]["definition"]
    assert definition["clock_domain"] == "monotonic" and definition["output_pipeline"]["kind"] == "reference_convnet_v1"


def test_mock_native_bundle_evaluates_with_clock_aware_notes(floor_run):
    m = metric_map(evaluate(floor_run.run_dir))
    assert m["bmt.count.completed"].value == 128
    assert m["bmt.latency.dut_service.p50"].status.value == "valid"
    assert "host monotonic clock" in m["bmt.latency.dut_service.p50"].notes[0]
    assert "virtual" not in " ".join(m["bmt.issue_lag.p99"].notes)


def test_injected_faults_follow_stage_1a_states(tmp_path):
    rp = _with_dut(variant(load(FLOOR), input_spec=load("rg-gpu-timeout-1ms").input_spec),
                   prepare_error_items=[1], submit_error_items=[2], run_error_items=[3])
    rp = variant(rp, plan={"replay": dataclasses.replace(rp.plan.replay, rate_hz=200.0)})
    out = run_plan(rp, tmp_path)
    assert out.measurement_status is MeasurementStatus.VALID
    states, events = _states(out.run_dir)
    assert states[1] == "not_issued" and states[2] == "error" and states[3] == "error"
    assert all(states[i] == "completed" for i in range(4, 32))
    errors = {e["request_id"]: e["error_phase"] for e in events if e["kind"] == "error"}
    assert errors == {2: "dut_submit", 3: "dut_run"}
    reasons = [e["reason"] for e in events if e["kind"] == "not_issued"]
    assert reasons == ["prepare_failed:prepare_input"]
    c = out.request_counts
    assert c["planned"] == c["not_issued"] + c["dropped"] + c["issued"] and c["issued"] == c["completed"] + c["error"] + c["timed_out"]


def test_a_long_request_holds_the_slot_and_arrivals_are_dropped(tmp_path):
    rp = _with_dut(variant(load(FLOOR), input_spec=load("rg-gpu-timeout-1ms").input_spec),
                   items=[{"item_index": 0, "polls": LONG}])
    out = run_plan(rp, tmp_path)  # 1 kHz
    states, events = _states(out.run_dir)
    assert states[0] == "completed" and states[1] == "dropped"
    assert all(e["reason"] == "no_free_slot" for e in events if e["kind"] == "dropped")


def test_timeout_keeps_the_slot_until_late_completion(tmp_path):
    rp = _with_dut(variant(load(FLOOR), input_spec=load("rg-gpu-timeout-1ms").input_spec),
                   items=[{"item_index": 5, "polls": LONG}])
    rp = _with_run(rp, request_timeout_ns=1_000_000, drain_ns=1_000_000_000)
    out = run_plan(rp, tmp_path)
    assert out.measurement_status is MeasurementStatus.VALID
    states, events = _states(out.run_dir)
    assert states[5] == "timed_out" and states[6] == "dropped"
    late = [e for e in events if e["kind"] == "late_completion"]
    assert [e["request_id"] for e in late] == [5] and late[0]["result"] == "ok"
    timed = next(e for e in events if e["kind"] == "timed_out")
    begin = next(e for e in events if e["kind"] == "submit_begin" and e["request_id"] == 5)
    assert timed["t_run_ns"] - begin["t_run_ns"] >= 1_000_000  # timer = t_submit_begin + request_timeout_ns
    assert out.request_counts["late_completion"] == 1


def test_drain_cap_leaves_a_timed_out_request_outstanding_but_terminal(tmp_path):
    rp = _with_dut(variant(load(FLOOR), input_spec=load("rg-gpu-timeout-1ms").input_spec),
                   items=[{"item_index": 31, "polls": 30 * LONG}])
    rp = _with_run(rp, request_timeout_ns=1_000_000, drain_ns=1_000_000)
    out = run_plan(rp, tmp_path)
    assert out.measurement_status is MeasurementStatus.VALID  # every issued request is terminal
    assert out.request_counts["timed_out"] == 1 and out.request_counts["unresolved"] == 0
    drain_end = [e for e in read_jsonl(out.run_dir / "events.jsonl") if e["kind"] == "phase"
                 and e["phase"] == "drain" and e["edge"] == "end"][0]
    assert drain_end["details"] == {"cap_reached": True, "outstanding_request_ids": [31]}


def test_mock_native_pair_is_never_attributed(tmp_path):
    a = run_plan(load(FLOOR), tmp_path / "a", run_id="run-a")
    b = run_plan(load(FLOOR), tmp_path / "b", run_id="run-b")
    ea, eb = evaluate(a.run_dir, "eval-a"), evaluate(b.run_dir, "eval-b")
    ma, mb = (json.loads((x.run_dir / "manifest.json").read_text()) for x in (a, b))
    va, vb = (json.loads((e.eval_dir / "eval_manifest.json").read_text()) for e in (ea, eb))
    result = evaluate_contrast_validity(ma, mb, "stress", evaluations=(va, vb))
    assert result.design_guard.status.value == "pass" and result.measurement_guard.status.value == "pass"
    assert result.realization_guard.status.value == "not_observable"
    assert result.realization_guard.basis == "mock_execution"
    assert result.overall_status.value == "not_established" and not result.attribution_allowed


def test_missing_native_runner_rejects_the_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("SDV_BMT_NATIVE_DIR", str(tmp_path / "nowhere"))
    with pytest.raises(PlanRejected) as exc:
        run_plan(load(FLOOR), tmp_path / "out")
    assert "could not describe itself" in str(exc.value)
    assert not (tmp_path / "out").exists()


def test_unrealizable_cpu_affinity_fails_closed(tmp_path):
    """Placement contract: a run whose planned CPU affinity is not realized never measures.

    The tracked plan is loaded directly (not through the tests' portable-affinity helper) and pinned to one usable
    CPU plus a CPU id past this system's last logical CPU, which no host process can be given."""
    rp = load_plan(plan_path(FLOOR))
    usable = os.sched_getaffinity(0)
    absent = max(os.cpu_count() or 0, max(usable) + 1)
    requested = (min(usable), absent)
    rp = variant(rp, plan={"realtime": dataclasses.replace(rp.plan.realtime, cpu_affinity=requested)})
    out = run_plan(rp, tmp_path / "out")
    assert out.execution_status is ExecutionStatus.FAILED and out.execution_reason == "cpu_affinity_not_realized"
    assert out.measurement_status is MeasurementStatus.INVALID
    events = read_jsonl(out.run_dir / "events.jsonl")
    placement = [e for e in events if e["kind"] == "phase" and e["phase"] == "host.placement"]
    assert placement and placement[-1]["outcome"] == "failed"
    assert placement[-1]["details"]["requested"] == list(requested)
    assert placement[-1]["details"]["realized"] != list(requested)
    assert not [e for e in events if e["kind"] == "issued"]  # nothing was measured
    load_run_bundle(out.run_dir)  # the failed run is still a well-formed bundle


def test_an_unresponsive_native_runner_is_killed_and_the_run_fails_closed(tmp_path, monkeypatch, native_build):
    fake_dir = tmp_path / "bin"
    fake_dir.mkdir()
    script = fake_dir / "sdv-bmt-native-fake"
    script.write_text("#!/bin/sh\nexec sleep 30\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("SDV_BMT_NATIVE_DIR", str(fake_dir))
    from sdv_bmt.dut.fake import FakeNativeDUT
    monkeypatch.setattr(FakeNativeDUT, "START_TIMEOUT_S", 1.0)
    out = run_plan(load(FLOOR), tmp_path / "out")
    assert out.execution_status is ExecutionStatus.FAILED and out.execution_reason == "dut_prepare_failed"
    assert out.measurement_status is MeasurementStatus.INVALID
    events = read_jsonl(out.run_dir / "events.jsonl")
    failed = [e for e in events if e["kind"] == "phase" and e["phase"] == "dut.prepare" and e["edge"] == "end"]
    assert failed and failed[0]["outcome"] == "failed" and "killed" in failed[0]["error"]["message"]
    steps = {e["step"]: e["outcome"] for e in events if e["kind"] == "cleanup"}
    assert steps["stress.restore"] == "ok" and "dut.cleanup" in steps
    load_run_bundle(out.run_dir)  # the failed run is still a well-formed bundle


def test_workload_without_native_output_pipeline_is_rejected(tmp_path):
    rp = load(FLOOR)
    fake_workload = load("g1-dataset-fixed-rate-nominal")
    rp = variant(rp, workload=fake_workload.workload, input_spec=fake_workload.input_spec)
    with pytest.raises(PlanRejected) as exc:
        run_plan(rp, tmp_path)
    assert "no native output pipeline" in str(exc.value)


def _cli(*args):
    proc = subprocess.run([sys.executable, "-m", "sdv_bmt.cli.main", *args], capture_output=True, text=True)
    return proc.returncode, json.loads(proc.stdout) if proc.stdout.strip() else {}, proc.stderr


def test_cli_labels_mock_native_runs_and_evaluations(tmp_path):
    code, payload, stderr = _cli("run", str(plan_file(FLOOR, tmp_path)), "--out", str(tmp_path))
    assert code == 0 and payload["execution_kind"] == "mock" and payload["notice"] == REALTIME_MOCK_NOTICE
    assert "[MOCK]" in stderr
    code, payload, stderr = _cli("evaluate", payload["runs"][0]["run_dir"])
    assert code == 0 and payload["notice"] == REALTIME_MOCK_NOTICE and "[MOCK]" in stderr
