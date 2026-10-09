"""Stress conformance (S1A-6): lifecycle order, failure injection, restore on every path."""

from __future__ import annotations

import dataclasses

import pytest

from sdv_bmt.contracts.protocols import HardwareStressProvider, LifecycleError
from sdv_bmt.contracts.records import HostInfo
from sdv_bmt.contracts.specs import HardwareStressSpec, parse_strict
from sdv_bmt.contracts.status import ExecutionStatus, MeasurementStatus, Outcome, VerificationStatus
from sdv_bmt.runner.runner import BenchmarkRunner, PlanRejected
from sdv_bmt.stress.fake import FakeStressProvider
from sdv_bmt.stress.none import NoStressProvider
from tests._helpers import lifecycle_trace, load, read_jsonl, run_plan, stress_spec_dict, variant

HOST = HostInfo("virtual")


def _spec(**overrides) -> HardwareStressSpec:
    return parse_strict(HardwareStressSpec, stress_spec_dict(**overrides))


def _full_cycle(provider):
    return [provider.prepare(), provider.activate(), provider.verify(0), provider.observe(),
            provider.deactivate(), provider.restore()]


# ---------------------------------------------------------------- provider level


def test_nostress_records_full_lifecycle_as_nominal():
    p = NoStressProvider(_spec())
    records = _full_cycle(p)
    assert [r.step.value for r in records] == ["prepare", "activate", "verify", "observe", "deactivate", "restore"]
    assert all(r.outcome is Outcome.OK and r.stress_level == "none" for r in records)
    assert records[2].verification is VerificationStatus.NOT_APPLICABLE
    assert records[0].requested is None


def test_fakestress_success_is_self_reported_and_virtual():
    p = FakeStressProvider(_spec(), "L2")
    records = _full_cycle(p)
    assert all(r.outcome is Outcome.OK for r in records)
    verify = records[2]
    assert verify.verification is VerificationStatus.SELF_REPORTED
    assert verify.requested == {"variable": "virtual_intensity", "unit": "ratio", "level": "L2", "value": 0.5}
    assert verify.achieved == {"value": 0.5, "source": "self_report", "virtual": True}
    assert verify.independent_evidence is None


def test_fakestress_cannot_claim_independent_evidence():
    spec = _spec(verification={"window_ns": 0, "achieved_min_fraction": 0.9,
                               "independent": {"source": "vendor-telemetry", "rule": "util>=0.5"}})
    p = FakeStressProvider(spec, "L1")
    p.prepare(); p.activate()
    assert p.verify(0).verification is VerificationStatus.UNVERIFIED


def test_fakestress_achieved_below_minimum_fails_verify():
    p = FakeStressProvider(_spec(provider_options={"achieved_fraction": 0.5}), "L3")
    p.prepare(); p.activate()
    record = p.verify(0)
    assert record.outcome is Outcome.FAILED and record.verification is VerificationStatus.FAILED
    assert record.error.code == "achieved_below_requested"


@pytest.mark.parametrize("phase", ["prepare", "activate", "verify", "deactivate", "restore"])
def test_fakestress_failure_injection_per_phase(phase):
    p = FakeStressProvider(_spec(provider_options={"fail_phase": phase}), "L1")
    calls = [("prepare", p.prepare), ("activate", p.activate), ("verify", lambda: p.verify(0)),
             ("deactivate", p.deactivate), ("restore", p.restore)]
    for name, call in calls:
        if name in ("verify",) and phase in ("prepare", "activate"):
            continue  # verify needs a successful activate
        if name in ("activate", "deactivate") and phase == "prepare":
            continue  # nothing was activated, so there is nothing to deactivate
        record = call()
        assert record.outcome is (Outcome.FAILED if name == phase else Outcome.OK), name
        if name == phase:
            assert record.error.code == "fake_injected_failure"


@pytest.mark.parametrize(
    "steps",
    [["activate"], ["verify"], ["observe"], ["deactivate"], ["prepare", "verify"],
     ["prepare", "activate", "observe"], ["prepare", "prepare"]],
)
def test_out_of_order_calls_rejected(steps):
    p = FakeStressProvider(_spec(), "L1")
    calls = {"prepare": p.prepare, "activate": p.activate, "verify": lambda: p.verify(0),
             "observe": p.observe, "deactivate": p.deactivate}
    with pytest.raises(LifecycleError):
        for s in steps:
            calls[s]()


def test_restore_is_idempotent_and_final():
    p = NoStressProvider(_spec())
    p.prepare()
    assert p.restore().outcome is Outcome.OK
    again = p.restore()
    assert again.outcome is Outcome.OK and again.details == {"repeat": True}
    with pytest.raises(LifecycleError):
        p.activate()


def test_restore_allowed_without_any_prior_step():
    for p in (NoStressProvider(_spec()), FakeStressProvider(_spec(), "L1")):
        assert p.restore().outcome is Outcome.OK


def test_check_is_side_effect_free_and_decides_support():
    fake = FakeStressProvider(_spec(), "L1")
    assert fake.check(_spec(), "L1", HOST).supported
    assert not fake.check(_spec(), "L9", HOST).supported
    assert not fake.check(_spec(kind="host_cpu_contention"), "L1", HOST).supported
    assert not fake.check(_spec(), "L1", HostInfo("real-host")).supported
    assert fake.lifecycle_state == "created"
    nominal = NoStressProvider(_spec())
    assert nominal.check(_spec(), "none", HOST).supported
    assert not nominal.check(_spec(), "L1", HOST).supported
    assert nominal.lifecycle_state == "created"


def test_level_routing_guards():
    with pytest.raises(ValueError):
        NoStressProvider(_spec(), "L1")
    with pytest.raises(ValueError):
        FakeStressProvider(_spec(), "none")


@pytest.mark.parametrize("cls", [NoStressProvider, FakeStressProvider])
def test_providers_implement_protocol_surface(cls):
    proto = {n for n in dir(HardwareStressProvider) if not n.startswith("_")}
    assert proto <= {n for n in dir(cls) if not n.startswith("_")}


# ---------------------------------------------------------------- runner level


def _stress_variant(fail_phase=None, level="L1"):
    rp = load("g1-dataset-fixed-rate-nominal")
    options = {} if fail_phase is None else {"fail_phase": fail_phase}
    spec = dataclasses.replace(rp.stress_spec, provider_options=options)
    return variant(rp, stress_spec=spec, plan={"stress": dataclasses.replace(rp.plan.stress, level=level)})


def _kinds(out) -> list[str]:
    return [e["kind"] for e in read_jsonl(out.run_dir / "events.jsonl")]


def _has_restore(out) -> bool:
    stress = read_jsonl(out.run_dir / "stress.jsonl")
    cleanup = [e for e in read_jsonl(out.run_dir / "events.jsonl")
               if e["kind"] == "cleanup" and e["step"] == "stress.restore"]
    return any(r["step"] == "restore" for r in stress) and len(cleanup) == 1


def test_verify_failure_has_no_measure_and_full_cleanup(tmp_path):
    out = run_plan(_stress_variant("verify"), tmp_path)
    assert lifecycle_trace(out.run_dir) == [
        "stress:check:ok",
        "input.open:begin", "input.open:end:ok",
        "dut.prepare:begin", "dut.prepare:end:ok",
        "prepare_inputs:begin", "prepare_inputs:end:ok",
        "stress.prepare:begin", "stress:prepare:ok", "stress.prepare:end:ok",
        "stress.activate:begin", "stress:activate:ok", "stress.activate:end:ok",
        "stress.verify:begin", "stress:verify:failed", "stress.verify:end:failed",
        "stress:deactivate:ok", "cleanup:stress.deactivate:ok",
        "stress:restore:ok", "cleanup:stress.restore:ok",
        "cleanup:prepared_store.discard:ok", "cleanup:dut.cleanup:ok", "cleanup:input.release:ok",
    ]
    kinds = _kinds(out)
    assert not {"issued", "dropped", "not_issued", "submit_begin", "completed"} & set(kinds)
    assert "measure" not in {e.get("phase") for e in read_jsonl(out.run_dir / "events.jsonl")}
    assert (out.execution_status, out.execution_reason) == (ExecutionStatus.FAILED, "stress_verify_failed")
    assert (out.measurement_status, out.measurement_reason) == (MeasurementStatus.INVALID, "not_measured")
    assert out.request_counts is None
    assert out.stress_status["verification"] is VerificationStatus.FAILED


def test_prepare_failure_skips_activation_and_deactivation(tmp_path):
    out = run_plan(_stress_variant("prepare"), tmp_path)
    trace = lifecycle_trace(out.run_dir)
    assert trace[trace.index("stress.prepare:begin"):] == [
        "stress.prepare:begin", "stress:prepare:failed", "stress.prepare:end:failed",
        "cleanup:stress.deactivate:skipped",
        "stress:restore:ok", "cleanup:stress.restore:ok",
        "cleanup:prepared_store.discard:ok", "cleanup:dut.cleanup:ok", "cleanup:input.release:ok",
    ]
    assert out.execution_reason == "stress_prepare_failed"


@pytest.mark.parametrize(
    "fail_phase,reason,measured",
    [
        ("prepare", "stress_prepare_failed", False),
        ("activate", "stress_activate_failed", False),
        ("verify", "stress_verify_failed", False),
        ("deactivate", "stress_deactivate_failed", True),
        ("restore", "stress_restore_failed", True),
    ],
)
def test_restore_recorded_on_every_stress_failure_path(tmp_path, fail_phase, reason, measured):
    out = run_plan(_stress_variant(fail_phase), tmp_path)
    assert _has_restore(out)
    assert out.execution_status is ExecutionStatus.FAILED and out.execution_reason == reason
    assert ("issued" in _kinds(out)) is measured
    if not measured:
        assert out.measurement_reason == "not_measured"
    if fail_phase == "activate":
        assert "cleanup:stress.deactivate:ok" in lifecycle_trace(out.run_dir)
    assert out.stress_status["restore"] == ("failed" if fail_phase == "restore" else "ok")


def _wrapping_factory(target: str, method: str):
    from sdv_bmt.cli.main import build_components

    def factory(rp):
        c = build_components(rp)
        obj = getattr(c, target)

        def boom(*args, **kwargs):
            raise RuntimeError(f"injected {target}.{method} failure")

        setattr(obj, method, boom)
        return c

    return factory


@pytest.mark.parametrize(
    "target,method,reason,expected_tail",
    [
        ("input_provider", "open", "input_open_failed", [
            "input.open:begin", "input.open:end:failed",
            "cleanup:stress.deactivate:skipped", "stress:restore:ok", "cleanup:stress.restore:ok",
            "cleanup:prepared_store.discard:skipped", "cleanup:dut.cleanup:skipped",
            "cleanup:input.release:ok",
        ]),
        ("dut", "prepare", "dut_prepare_failed", [
            "dut.prepare:begin", "dut.prepare:end:failed",
            "cleanup:stress.deactivate:skipped", "stress:restore:ok", "cleanup:stress.restore:ok",
            "cleanup:prepared_store.discard:skipped", "cleanup:dut.cleanup:ok",
            "cleanup:input.release:ok",
        ]),
        ("dut", "cleanup", "dut_cleanup_failed", [
            "cleanup:dut.cleanup:failed", "cleanup:input.release:ok",
        ]),
        ("input_provider", "release", "input_release_failed", [
            "cleanup:dut.cleanup:ok", "cleanup:input.release:failed",
        ]),
    ],
)
def test_restore_recorded_on_component_failure_paths(tmp_path, target, method, reason, expected_tail):
    out = run_plan(load("g1-dataset-fixed-rate-nominal"), tmp_path, factory=_wrapping_factory(target, method))
    assert _has_restore(out)
    trace = lifecycle_trace(out.run_dir)
    assert trace[-len(expected_tail):] == expected_tail
    assert out.execution_status is ExecutionStatus.FAILED and out.execution_reason == reason


def test_restore_recorded_when_measure_aborts(tmp_path, monkeypatch):
    from sdv_bmt.runner import core

    original = core.PreparedStore.get

    def get(self, item_index):
        if item_index == 2:
            raise core.RunnerInvariantError("injected store corruption")
        return original(self, item_index)

    monkeypatch.setattr(core.PreparedStore, "get", get)
    out = run_plan(load("g1-dataset-fixed-rate-nominal"), tmp_path)
    assert _has_restore(out)
    assert out.execution_reason == "measure_aborted"
    assert (out.measurement_status, out.measurement_reason) == (MeasurementStatus.INVALID, "measure_aborted")
    trace = lifecycle_trace(out.run_dir)
    assert "measure:end:failed" in trace and "cleanup:stress.deactivate:ok" in trace


def test_provider_exception_becomes_failed_stress_record(tmp_path):
    out = run_plan(_stress_variant(), tmp_path, factory=_wrapping_factory("stress", "activate"))
    stress = read_jsonl(out.run_dir / "stress.jsonl")
    activate = [r for r in stress if r["step"] == "activate"][0]
    assert activate["outcome"] == "failed" and activate["details"] == {"exception": True}
    assert out.execution_reason == "stress_activate_failed" and _has_restore(out)


def test_nominal_and_fake_stress_record_identical_lifecycle_shape(tmp_path):
    nominal = run_plan(load("g1-dataset-fixed-rate-nominal"), tmp_path / "n")
    stressed = run_plan(_stress_variant(level="L1"), tmp_path / "s")
    assert lifecycle_trace(nominal.run_dir) == lifecycle_trace(stressed.run_dir)
    assert nominal.stress_status["coverage"].value == "not_applicable"
    assert stressed.stress_status["coverage"].value == "full"
    assert stressed.stress_status["verification"] is VerificationStatus.SELF_REPORTED


def test_restore_failure_blocks_further_runs(tmp_path):
    from sdv_bmt.cli.main import build_components

    rp = variant(_stress_variant("restore"), plan={"repetitions": 3})
    runner = BenchmarkRunner(rp, build_components, run_id_factory=iter(["run-1", "run-2", "run-3"]).__next__)
    result = runner.run_experiment(tmp_path)
    assert len(result.outcomes) == 1 and result.outcomes[0].execution_reason == "stress_restore_failed"
    assert result.rejected and "failed stress restore" in result.rejected[0]
    with pytest.raises(PlanRejected):
        runner.run_once(tmp_path)


# ---------------------------------------------------------------- review regressions


class _Spy:
    """Wrap a provider; record calls and optionally replace one step's record."""

    def __init__(self, inner, calls, replace=None):
        self._inner, self._calls, self._replace = inner, calls, replace or {}
        self.PROVIDER = inner.PROVIDER

    def check(self, *args):
        return self._inner.check(*args)

    def __getattr__(self, name):
        method = getattr(self._inner, name)

        def call(*args):
            self._calls.append(name)
            record = method(*args)
            return self._replace[name](record) if name in self._replace else record

        return call


def _spy_stress_factory(calls, replace):
    from sdv_bmt.cli.main import build_components

    def factory(rp):
        c = build_components(rp)
        c.stress = _Spy(c.stress, calls, replace)
        return c

    return factory


@pytest.mark.parametrize(
    "step,patch,reason",
    [
        ("deactivate", lambda r: dataclasses.replace(r, details={"elapsed": float("nan")}), "stress_deactivate_failed"),
        ("verify", lambda r: dataclasses.replace(r, achieved={"value": float("inf")}), "stress_verify_failed"),
        ("observe", lambda r: dataclasses.replace(r, details={"bad": {1, 2}}), None),
    ],
)
def test_unrecordable_stress_record_never_skips_restore_or_finalize(tmp_path, step, patch, reason):
    calls: list[str] = []
    out = run_plan(_stress_variant(level="L1"), tmp_path, factory=_spy_stress_factory(calls, {step: patch}))
    assert calls[-1] == "restore" and calls.count("restore") == 1
    assert (out.run_dir / "manifest.json").is_file()
    failed = [r for r in read_jsonl(out.run_dir / "stress.jsonl") if r["step"] == step and r["outcome"] == "failed"]
    assert failed and failed[0]["error"]["code"] == "CanonicalJSONError"
    if reason is None:  # observe failure degrades coverage but does not fail the run
        assert out.execution_status is ExecutionStatus.COMPLETED
        assert out.stress_status["coverage"].value == "partial"
    else:
        assert out.execution_reason == reason


def test_verify_reported_ok_but_failed_verification_never_measures(tmp_path):
    def contradictory(record):
        return dataclasses.replace(record, verification=VerificationStatus.FAILED)  # outcome stays ok

    calls: list[str] = []
    out = run_plan(_stress_variant(level="L1"), tmp_path,
                   factory=_spy_stress_factory(calls, {"verify": contradictory}))
    assert out.execution_reason == "stress_verify_failed"
    assert "issued" not in _kinds(out)
    assert out.measurement_reason == "not_measured"
    assert out.stress_status["verification"] is VerificationStatus.FAILED


def test_stress_record_contract_forbids_ok_outcome_with_failed_verification():
    from sdv_bmt.contracts.records import StressRecord
    from sdv_bmt.contracts.status import StressStep

    with pytest.raises(ValueError, match="verification=failed requires outcome=failed"):
        StressRecord(StressStep.VERIFY, Outcome.OK, "p", "a" * 64, "L1", verification=VerificationStatus.FAILED)


def test_recorder_failure_during_cleanup_still_cleans_every_component(tmp_path, monkeypatch):
    from sdv_bmt.cli.main import build_components
    from sdv_bmt.observation.recorder import ObservationRecorder, RecorderError

    calls: list[str] = []

    def factory(rp):
        c = build_components(rp)
        c.stress = _Spy(c.stress, calls)
        dut_cleanup, input_release = c.dut.cleanup, c.input_provider.release
        c.dut.cleanup = lambda: (calls.append("dut.cleanup"), dut_cleanup())[1]
        c.input_provider.release = lambda: (calls.append("input.release"), input_release())[1]
        return c

    real_event = ObservationRecorder.event

    def flaky_event(self, kind, t_run_ns, **fields):
        if kind == "cleanup" and fields.get("step") == "stress.deactivate":
            raise RecorderError("disk full (injected)")
        return real_event(self, kind, t_run_ns, **fields)

    monkeypatch.setattr(ObservationRecorder, "event", flaky_event)
    with pytest.raises(RecorderError, match="disk full"):
        run_plan(_stress_variant(level="L1"), tmp_path, factory=factory)
    assert calls[-4:] == ["deactivate", "restore", "dut.cleanup", "input.release"]
