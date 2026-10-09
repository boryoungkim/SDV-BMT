"""WP3 gate: Runner-owned prepared store; InputProvider holds no DUT/prepared object."""

from __future__ import annotations

import pytest

from sdv_bmt.contracts.records import ErrorInfo
from sdv_bmt.contracts.specs import DUTConfig, WorkloadSpec, parse_strict
from sdv_bmt.contracts.status import RequestState as S
from sdv_bmt.dut.fake import FakeBlockingDUT, FakePreparedInput
from sdv_bmt.runner.core import (
    PreparationFailure,
    PreparedStore,
    RequestTable,
    RunnerInvariantError,
    count_identity_violations,
    prepare_inputs,
)
from sdv_bmt.runner.plan import load_yaml_strict
from sdv_bmt.workload.fake import FakeIdentityWorkload
from tests._helpers import CONFIGS, load_input_provider


def _reachable(root, max_depth=8):
    seen, stack, out = set(), [(root, 0)], []
    while stack:
        obj, depth = stack.pop()
        if id(obj) in seen or isinstance(obj, type):
            continue
        seen.add(id(obj))
        out.append(obj)
        if depth >= max_depth:
            continue
        if isinstance(obj, dict):
            children = list(obj.keys()) + list(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            children = list(obj)
        elif hasattr(obj, "__dict__"):
            children = list(vars(obj).values())
        else:
            children = [getattr(obj, s) for s in getattr(obj, "__slots__", ()) if hasattr(obj, s)]
        stack.extend((c, depth + 1) for c in children)
    return out


def _components(dut_yaml="fake-blocking-25ms.yaml"):
    workload_spec = parse_strict(WorkloadSpec, load_yaml_strict(CONFIGS / "workloads" / "fake-identity.yaml"))
    dut_config = parse_strict(DUTConfig, load_yaml_strict(CONFIGS / "duts" / dut_yaml))
    provider = load_input_provider(CONFIGS / "inputs" / "dataset-fixture-8.yaml")
    return provider, FakeIdentityWorkload(workload_spec), FakeBlockingDUT(dut_config), workload_spec, dut_config


def test_runner_store_owns_prepared_inputs_and_provider_holds_none():
    provider, workload, dut, wspec, dconf = _components()
    provider.open()
    dut.prepare(wspec, dconf)
    store = PreparedStore()
    prepare_inputs(provider, workload, dut, provider.items(), store)
    assert store.prepared_count == 8 and store.failed == {}
    assert all(isinstance(store.get(i), FakePreparedInput) for i in range(8))
    forbidden = (FakePreparedInput, FakeBlockingDUT, FakeIdentityWorkload, PreparedStore)
    # Non-vacuity: the walker does find prepared inputs where they actually live.
    assert sum(isinstance(o, FakePreparedInput) for o in _reachable(store)) == 8
    for obj in _reachable(provider):
        assert not isinstance(obj, forbidden), type(obj)
    for obj in _reachable(workload):
        assert not isinstance(obj, (FakePreparedInput, FakeBlockingDUT, PreparedStore)), type(obj)


def test_preparation_before_dut_prepare_fails_every_item_at_prepare_input():
    provider, workload, dut, _wspec, _dconf = _components()
    provider.open()
    store = PreparedStore()
    prepare_inputs(provider, workload, dut, provider.items(), store)
    assert store.prepared_count == 0
    assert {f.stage for f in store.failed.values()} == {"prepare_input"}
    assert {f.error.code for f in store.failed.values()} == {"LifecycleError"}


def test_per_item_failures_are_recorded_and_others_prepared(tmp_path):
    provider, workload, _dut, wspec, _ = _components()
    config = parse_strict(DUTConfig, {
        "schema_version": 1, "dut_config_id": "d", "adapter": "fake_blocking", "device": "v",
        "max_inflight": 1, "runtime_options": {"service_model": {
            "kind": "table", "default_service_ns": 1,
            "entries": [{"item_index": 3, "prepare_input_outcome": "error"}]}},
    })
    dut = FakeBlockingDUT(config)
    dut.prepare(wspec, config)
    provider.open()
    store = PreparedStore()
    prepare_inputs(provider, workload, dut, provider.items(), store)
    assert store.prepared_count == 7
    assert store.failure(3).stage == "prepare_input"
    assert store.failure(3).error.code == "FakeDUTInjectedError"


def test_payload_failure_stage_when_provider_not_opened():
    provider, workload, dut, wspec, dconf = _components()
    dut.prepare(wspec, dconf)
    store = PreparedStore()
    prepare_inputs(provider, workload, dut, provider.items(), store)
    assert {f.stage for f in store.failed.values()} == {"payload"}


def test_store_discard_is_idempotent_and_final():
    store = PreparedStore()
    store.put(0, object())
    store.fail(1, PreparationFailure("preprocess", ErrorInfo("E", "m")))
    assert store.discard() == 1
    assert store.discard() == 0 and store.discarded
    with pytest.raises(RunnerInvariantError):
        store.get(0)


# ---------------------------------------------------------------- request table


def test_request_state_machine_and_count_identity():
    t = RequestTable(6)
    t.transition(0, S.ISSUED); t.transition(0, S.INFLIGHT); t.transition(0, S.COMPLETED)
    t.transition(1, S.DROPPED)
    t.transition(2, S.ISSUED); t.transition(2, S.INFLIGHT); t.transition(2, S.TIMED_OUT)
    t.mark_late_completion(2)
    t.transition(3, S.ISSUED); t.transition(3, S.INFLIGHT); t.transition(3, S.ERROR)
    t.transition(4, S.NOT_ISSUED)
    t.transition(5, S.ISSUED); t.transition(5, S.ERROR)
    counts = t.counts()
    assert counts == {"planned": 6, "issued": 4, "completed": 1, "error": 2, "timed_out": 1,
                      "dropped": 1, "not_issued": 1, "late_completion": 1, "unresolved": 0}
    assert count_identity_violations(counts) == []


@pytest.mark.parametrize(
    "path",
    [
        [S.COMPLETED],
        [S.ISSUED, S.COMPLETED],
        [S.ISSUED, S.INFLIGHT, S.COMPLETED, S.TIMED_OUT],
        [S.DROPPED, S.ISSUED],
        [S.ISSUED, S.INFLIGHT, S.TIMED_OUT, S.COMPLETED],
    ],
)
def test_illegal_transitions_rejected(path):
    t = RequestTable(1)
    with pytest.raises(RunnerInvariantError):
        for state in path:
            t.transition(0, state)


def test_late_completion_only_once_and_only_after_timeout():
    t = RequestTable(1)
    t.transition(0, S.ISSUED); t.transition(0, S.INFLIGHT)
    with pytest.raises(RunnerInvariantError):
        t.mark_late_completion(0)
    t.transition(0, S.TIMED_OUT)
    t.mark_late_completion(0)
    with pytest.raises(RunnerInvariantError):
        t.mark_late_completion(0)
    assert t.counts()["completed"] == 0 and t.counts()["timed_out"] == 1


def test_identity_violation_detected():
    counts = {"planned": 3, "issued": 2, "completed": 1, "error": 0, "timed_out": 0,
              "dropped": 0, "not_issued": 0}
    assert count_identity_violations(counts) == [
        "planned != not_issued + dropped + issued", "issued != completed + error + timed_out"
    ]
