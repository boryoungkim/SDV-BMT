"""Explicit input pipeline (design change: Explicit DUT Adapter Preprocess Override).

workload:          payload -> Workload.preprocess -> DUTAdapter.prepare_input
adapter_override:  payload -> AdapterInputOverride.prepare_from_payload
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from sdv_bmt.cli.main import build_components
from sdv_bmt.contracts.records import (
    InputPipelineRecord,
    WorkloadInputContract,
    input_contract_digest,
)
from sdv_bmt.contracts.specs import (
    DUTConfig,
    ExperimentPlan,
    InputPipelineSelection,
    SpecError,
    parse_strict,
)
from sdv_bmt.contracts.status import (
    ExecutionStatus,
    GuardStatus,
    InputPipelineMode,
    InputPipelineOwner,
    MeasurementStatus,
)
from sdv_bmt.evidence.manifest import read_manifest
from sdv_bmt.observation.reader import BundleReadError, load_run_bundle
from sdv_bmt.runner.plan import load_yaml_strict
from sdv_bmt.runner.runner import BenchmarkRunner, PlanRejected
from tests._helpers import (
    CONFIGS,
    design_check,
    evaluate,
    lifecycle_trace,
    load,
    metric_map,
    plan_dict,
    read_jsonl,
    request_trace,
    run_plan,
    variant,
)

FUSED = "fake-fused-prefix-preprocess"
OVERRIDE = InputPipelineSelection(InputPipelineMode.ADAPTER_OVERRIDE, FUSED)
SHA_A, SHA_B = "a" * 64, "b" * 64


def _dut(name: str) -> DUTConfig:
    return parse_strict(DUTConfig, load_yaml_strict(CONFIGS / "duts" / name))


def _with_dut_options(rp, **runtime):
    return variant(rp, dut=dataclasses.replace(rp.dut, runtime_options={**rp.dut.runtime_options, **runtime}))


def _with_selection(rp, **pins):
    return variant(rp, plan={"input_pipeline": dataclasses.replace(OVERRIDE, **pins)})


def _factory(calls=None, patch_override=None, patch_dut=None, patch_provider=None):
    """build_components plus optional spies/patches on the created components."""

    def factory(rp):
        c = build_components(rp)

        def wrap(obj, name, label):
            original = getattr(obj, name)

            def spy(*args, **kwargs):
                item = args[-1] if args else None
                calls.append(f"{label}[{item.item_index}]" if hasattr(item, "item_index") else label)
                return original(*args, **kwargs)

            setattr(obj, name, spy)

        if calls is not None:
            wrap(c.input_provider, "payload", "input.payload")
            wrap(c.workload, "preprocess", "workload.preprocess")
            wrap(c.dut, "prepare_input", "dut.prepare_input")
        original_override = c.dut.input_override

        def input_override(contract, implementation_id):
            if calls is not None:
                calls.append("dut.input_override")
            override = original_override(contract, implementation_id)
            if override is not None:
                if calls is not None:
                    wrap(override, "prepare_from_payload", "override.prepare_from_payload")
                if patch_override:
                    patch_override(override)
            return override

        c.dut.input_override = input_override
        if patch_dut:
            patch_dut(c.dut)
        if patch_provider:
            patch_provider(c.input_provider)
        return c

    return factory


def _equivalence(out) -> dict:
    return read_manifest(out.run_dir)["input_pipeline"]["equivalence_validation"]


# ---------------------------------------------------------------- contracts


def test_input_contract_is_immutable_versioned_and_canonical():
    contract = WorkloadInputContract(1, "c", "2", "task", "sem/1", "fmt", (("b", "2"), ("a", "1")))
    assert contract.logical_input == (("a", "1"), ("b", "2"))
    assert contract.parameter("b") == "2"
    with pytest.raises(dataclasses.FrozenInstanceError):
        contract.contract_version = "3"  # type: ignore[misc]
    same = WorkloadInputContract(1, "c", "2", "task", "sem/1", "fmt", (("a", "1"), ("b", "2")))
    assert input_contract_digest(contract) == input_contract_digest(same)
    assert input_contract_digest(contract) != input_contract_digest(dataclasses.replace(contract, contract_version="3"))
    with pytest.raises(ValueError, match="unique"):
        WorkloadInputContract(1, "c", "2", "t", "s", "f", (("a", "1"), ("a", "2")))
    with pytest.raises(ValueError, match="schema_version"):
        WorkloadInputContract(2, "c", "2", "t", "s", "f", ())


def test_pipeline_selection_is_explicit_and_strict():
    plan = parse_strict(ExperimentPlan, plan_dict())
    assert plan.input_pipeline.mode is InputPipelineMode.WORKLOAD  # default = canonical Workload path
    override = parse_strict(ExperimentPlan, plan_dict(input_pipeline={"mode": "adapter_override",
                                                                      "implementation_id": FUSED}))
    assert override.input_pipeline == OVERRIDE
    with pytest.raises(SpecError, match="requires implementation_id"):
        parse_strict(ExperimentPlan, plan_dict(input_pipeline={"mode": "adapter_override"}))
    with pytest.raises(SpecError, match="adapter_override only"):
        parse_strict(ExperimentPlan, plan_dict(input_pipeline={"mode": "workload", "implementation_id": "x"}))
    with pytest.raises(SpecError, match="adapter_override only"):
        parse_strict(ExperimentPlan, plan_dict(input_pipeline={"mode": "workload", "implementation_version": "1"}))
    with pytest.raises(SpecError, match="expected one of"):
        parse_strict(ExperimentPlan, plan_dict(input_pipeline={"mode": "auto"}))
    with pytest.raises(SpecError, match="unknown field"):
        parse_strict(ExperimentPlan, plan_dict(input_pipeline={"mode": "workload", "fallback": "workload"}))
    with pytest.raises(SpecError, match="implementation_digest"):
        parse_strict(ExperimentPlan, plan_dict(input_pipeline={"mode": "adapter_override", "implementation_id": FUSED,
                                                               "implementation_digest": "abc"}))


def test_pipeline_record_invariants():
    with pytest.raises(ValueError, match="requires owner"):
        InputPipelineRecord(InputPipelineMode.ADAPTER_OVERRIDE, InputPipelineOwner.WORKLOAD, "s", "c", "1",
                            SHA_A, "i", "1", SHA_B, ("workload.preprocess",))
    with pytest.raises(ValueError, match="no fused steps"):
        InputPipelineRecord(InputPipelineMode.WORKLOAD, InputPipelineOwner.WORKLOAD, "s", "c", "1",
                            SHA_A, "i", "1", SHA_B, ("workload.preprocess",))
    with pytest.raises(ValueError, match="must not be empty"):
        InputPipelineRecord(InputPipelineMode.ADAPTER_OVERRIDE, InputPipelineOwner.DUT_ADAPTER, "s", "c", "1",
                            SHA_A, "i", "1", SHA_B, ())
    with pytest.raises(ValueError, match="list/tuple"):
        InputPipelineRecord(InputPipelineMode.ADAPTER_OVERRIDE, InputPipelineOwner.DUT_ADAPTER, "s", "c", "1",
                            SHA_A, "i", "1", SHA_B, "workload.preprocess")
    record = InputPipelineRecord(InputPipelineMode.ADAPTER_OVERRIDE, InputPipelineOwner.DUT_ADAPTER, "s", "c", "1",
                                 SHA_A, "i", "1", SHA_B, ["workload.preprocess"])
    assert record.fused_steps == ("workload.preprocess",)


# ---------------------------------------------------------------- explicit selection, no fallback


def test_workload_mode_never_uses_an_available_override(tmp_path):
    calls: list[str] = []
    rp = variant(load("g1-dataset-fixed-rate-nominal"), dut=_dut("fake-blocking-25ms-fused.yaml"))
    out = run_plan(rp, tmp_path, factory=_factory(calls))
    assert "dut.input_override" not in calls  # no auto-selection: not even queried
    assert sum(c.startswith("workload.preprocess[") for c in calls) == 8
    assert sum(c.startswith("dut.prepare_input[") for c in calls) == 8
    assert not [c for c in calls if c.startswith("override.")]
    pipeline = read_manifest(out.run_dir)["input_pipeline"]
    assert (pipeline["mode"], pipeline["owner"], pipeline["fused_steps"]) == ("workload", "workload", [])
    assert _equivalence(out)["status"] == "not_applicable"


def test_override_mode_prepares_from_payload_then_validates_against_reference(tmp_path):
    calls: list[str] = []
    out = run_plan(load("g1-adapter-override"), tmp_path, factory=_factory(calls))
    first_reference = calls.index("workload.preprocess[0]")
    preparation, validation = calls[:first_reference], calls[first_reference:]
    assert sum(c.startswith("override.prepare_from_payload") for c in preparation) == 8
    assert not [c for c in preparation if c.startswith(("workload.preprocess", "dut.prepare_input"))]
    assert sum(c.startswith("workload.preprocess") for c in validation) == 8
    assert not [c for c in validation if c.startswith("override.")]
    trace = lifecycle_trace(out.run_dir)
    assert trace.index("prepare_inputs:end:ok") < trace.index("input.equivalence:begin") < trace.index(
        "input.equivalence:end:ok") < trace.index("stress.prepare:begin")
    assert out.execution_status is ExecutionStatus.COMPLETED


def test_override_requested_but_unavailable_is_rejected_in_preflight(tmp_path):
    with pytest.raises(PlanRejected, match="provides no input override 'fake-fused-prefix-preprocess'.*no fallback"):
        run_plan(load("reject-override-unavailable"), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_unknown_override_implementation_is_rejected(tmp_path):
    rp = _with_selection(load("g1-adapter-override"), implementation_id="vendor-fused-resize-normalize")
    with pytest.raises(PlanRejected, match="provides no input override 'vendor-fused-resize-normalize'"):
        run_plan(rp, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_implementation_pins_are_enforced(tmp_path):
    base = load("g1-adapter-override")
    pf = BenchmarkRunner(base, build_components).preflight()
    good = _with_selection(base, implementation_version="1",
                           implementation_digest=pf.input_pipeline.implementation_digest)
    BenchmarkRunner(good, build_components).preflight()
    with pytest.raises(PlanRejected, match="pins implementation_version '2'"):
        BenchmarkRunner(_with_selection(base, implementation_version="2"), build_components).preflight()
    with pytest.raises(PlanRejected, match="pins implementation_digest"):
        BenchmarkRunner(_with_selection(base, implementation_digest="0" * 64), build_components).preflight()


def test_override_without_the_contract_semantics_is_rejected(tmp_path):
    def factory(rp):
        c = build_components(rp)
        changed = dataclasses.replace(c.workload.input_contract(), preprocess_semantics_id="fake-prefix-preprocess/10")
        c.workload.input_contract = lambda: changed
        return c

    with pytest.raises(PlanRejected, match="does not implement preprocess semantics 'fake-prefix-preprocess/10'"):
        run_plan(load("g1-adapter-override"), tmp_path, factory=factory)


def test_item_failing_in_both_paths_is_not_issued_without_fallback(tmp_path):
    calls: list[str] = []
    rp = _with_dut_options(load("g1-adapter-override"), service_model={
        "kind": "table", "default_service_ns": 25_000_000,
        "entries": [{"item_index": 3, "prepare_input_outcome": "error"}]})
    out = run_plan(rp, tmp_path, factory=_factory(calls))
    not_issued = [e for e in read_jsonl(out.run_dir / "events.jsonl") if e["kind"] == "not_issued"]
    assert [(e["request_id"], e["reason"]) for e in not_issued] == [(3, "prepare_failed:adapter_override")]
    equivalence = _equivalence(out)
    assert (equivalence["status"], equivalence["validated_items"]) == ("passed", 7)
    assert equivalence["both_failed_item_indices"] == [3]
    assert out.execution_status is ExecutionStatus.COMPLETED


# ---------------------------------------------------------------- equivalence (fail closed)


def test_equivalence_failure_never_measures(tmp_path):
    out = run_plan(load("override-equivalence-failure"), tmp_path)
    assert (out.execution_status, out.execution_reason) == (ExecutionStatus.FAILED, "input_equivalence_failed")
    assert (out.measurement_status, out.measurement_reason) == (MeasurementStatus.INVALID, "not_measured")
    kinds = {e["kind"] for e in read_jsonl(out.run_dir / "events.jsonl")}
    assert not kinds & {"issued", "dropped", "submit_begin"}
    trace = lifecycle_trace(out.run_dir)
    assert "input.equivalence:end:failed" in trace and "stress.prepare:begin" not in trace
    assert "cleanup:stress.restore:ok" in trace
    equivalence = _equivalence(out)
    assert (equivalence["status"], equivalence["reason"]) == ("failed", "override_not_equivalent")
    assert equivalence["mismatched_item_indices"] == list(range(8))
    assert {m.status.value for m in evaluate(out.run_dir).metrics} == {"not_collected"}


def test_override_failing_where_reference_succeeds_is_a_mismatch(tmp_path):
    def flaky_override(override):
        original = override.prepare_from_payload

        def prepare(payload, item):
            if item.item_index < 7:
                raise RuntimeError("override cannot handle this item")
            return original(payload, item)

        override.prepare_from_payload = prepare

    out = run_plan(load("g1-adapter-override"), tmp_path, factory=_factory(patch_override=flaky_override))
    equivalence = _equivalence(out)
    assert (equivalence["status"], equivalence["mismatched_item_indices"], equivalence["validated_items"]) == (
        "failed", list(range(7)), 1)
    assert out.execution_reason == "input_equivalence_failed" and "issued" not in {
        e["kind"] for e in read_jsonl(out.run_dir / "events.jsonl")}


def test_no_comparable_items_does_not_establish_equivalence(tmp_path):
    entries = [{"item_index": i, "prepare_input_outcome": "error"} for i in range(8)]
    rp = _with_dut_options(load("g1-adapter-override"),
                           service_model={"kind": "table", "default_service_ns": 25_000_000, "entries": entries})
    out = run_plan(rp, tmp_path)
    equivalence = _equivalence(out)
    assert (equivalence["status"], equivalence["reason"]) == ("not_performed", "no_comparable_items")
    assert equivalence["both_failed_item_indices"] == list(range(8))
    assert out.execution_reason == "input_equivalence_not_established"
    assert "input.equivalence:end:failed" in lifecycle_trace(out.run_dir)


def test_reference_path_errors_are_inconclusive_not_blamed_on_the_override(tmp_path):
    def flaky_reread(provider):
        original, seen = provider.payload, set()

        def payload(item_index):
            if item_index == 2 and item_index in seen:  # second read (reference side) fails
                raise OSError("transient payload re-read failure")
            seen.add(item_index)
            return original(item_index)

        provider.payload = payload

    out = run_plan(load("g1-adapter-override"), tmp_path, factory=_factory(patch_provider=flaky_reread))
    equivalence = _equivalence(out)
    assert equivalence["status"] == "not_performed" and equivalence["reason"] == "reference_path_errors"
    assert equivalence["reference_error_item_indices"] == [2] and equivalence["mismatched_item_indices"] == []
    assert out.execution_reason == "input_equivalence_not_established"
    end = [e for e in read_jsonl(out.run_dir / "events.jsonl")
           if e["kind"] == "phase" and e["phase"] == "input.equivalence" and e["edge"] == "end"][0]
    assert end["details"]["errors"][0]["side"] == "reference"
    assert "transient payload re-read failure" in end["details"]["errors"][0]["error"]["message"]


def test_degenerate_adapter_digest_cannot_pass_a_divergent_override(tmp_path):
    def constant_digest(dut):
        dut.prepared_input_digest = lambda prepared: "c" * 64

    out = run_plan(load("override-equivalence-failure"), tmp_path, factory=_factory(patch_dut=constant_digest))
    equivalence = _equivalence(out)
    assert (equivalence["status"], equivalence["reason"]) == ("not_performed", "degenerate_prepared_input_digest")
    assert out.execution_reason == "input_equivalence_not_established"


def test_non_sha256_adapter_digest_is_inconclusive(tmp_path):
    def broken_digest(dut):
        dut.prepared_input_digest = lambda prepared: None

    out = run_plan(load("g1-adapter-override"), tmp_path, factory=_factory(patch_dut=broken_digest))
    equivalence = _equivalence(out)
    assert equivalence["status"] == "not_performed" and equivalence["reference_error_item_indices"] == list(range(8))
    assert out.execution_reason == "input_equivalence_not_established"


# ---------------------------------------------------------------- evidence


def test_manifest_and_run_spec_record_the_effective_pipeline(tmp_path):
    out = run_plan(load("g1-adapter-override"), tmp_path)
    m = read_manifest(out.run_dir)
    pipeline = m["input_pipeline"]
    assert pipeline["mode"] == "adapter_override" and pipeline["owner"] == "dut_adapter"
    assert pipeline["preprocess_semantics_id"] == "fake-prefix-preprocess/1"
    assert (pipeline["implementation_id"], pipeline["implementation_version"]) == (FUSED, "1")
    assert len(pipeline["implementation_digest"]) == 64 and len(pipeline["pipeline_digest"]) == 64
    assert pipeline["fused_steps"] == ["workload.preprocess", "dut.prepare_input"]
    assert pipeline["measurement_boundary_id"] == m["measurement_boundary"]["measurement_boundary_id"]
    assert pipeline["equivalence_validation"] == {
        "status": "passed", "method": "exact_adapter_prepared_input_digest", "validated_items": 8,
        "mismatched_item_indices": [], "reference_error_item_indices": [], "both_failed_item_indices": [],
        "reason": None}
    boundary = m["measurement_boundary"]["definition"]
    assert boundary["schema_version"] == 2 and boundary["input_pipeline_owner"] == "dut_adapter"
    assert boundary["fused_steps"] == ["workload.preprocess", "dut.prepare_input"]
    assert boundary["input_pipeline_digest"] == pipeline["pipeline_digest"]
    assert m["comparison_keys"]["components"]["common_parts"]["input_pipeline"] == pipeline["pipeline_digest"]
    run_spec = json.loads((out.run_dir / "run_spec.json").read_text())
    assert run_spec["schema_version"] == 2
    assert run_spec["input_pipeline"]["selection"]["mode"] == "adapter_override"
    assert run_spec["input_pipeline"]["selection"]["implementation_id"] == FUSED
    assert run_spec["input_pipeline"]["pipeline_digest"] == pipeline["pipeline_digest"]
    assert run_spec["input_pipeline"]["input_contract"]["preprocess_semantics_id"] == "fake-prefix-preprocess/1"
    assert run_spec["input_pipeline"]["override"]["implementation_digest"] == pipeline["implementation_digest"]
    assert run_spec["resolved_plan"]["input_pipeline"]["mode"] == "adapter_override"


def test_equivalent_override_produces_identical_requests_and_outputs(tmp_path):
    workload_run = run_plan(load("g1-dataset-fixed-rate-nominal"), tmp_path / "w")
    override_run = run_plan(load("g1-adapter-override"), tmp_path / "o")
    assert request_trace(workload_run.run_dir) == request_trace(override_run.run_dir)
    assert (workload_run.run_dir / "outputs.bin").read_bytes() == (override_run.run_dir / "outputs.bin").read_bytes()
    assert metric_map(evaluate(override_run.run_dir))["bmt.output.identity_rate"].value == 1.0


def test_previous_manifest_schema_is_rejected_with_a_version_error(tmp_path):
    out = run_plan(load("g1-adapter-override"), tmp_path)
    path = out.run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["schema_version"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(BundleReadError, match="schema_version 1 is not supported"):
        load_run_bundle(out.run_dir)


# ---------------------------------------------------------------- comparison identity


def _keys(rp):
    return BenchmarkRunner(rp, build_components).preflight().comparison_keys


def _boundary_id(rp):
    return BenchmarkRunner(rp, build_components).preflight().boundary_id


def test_contrasts_keep_the_effective_input_pipeline_fixed():
    workload_keys = _keys(variant(load("g1-dataset-fixed-rate-nominal"), dut=_dut("fake-blocking-25ms-fused.yaml")))
    override_keys = _keys(load("g1-adapter-override"))
    result = design_check(workload_keys, override_keys, "stress")
    assert result.status is GuardStatus.FAIL
    assert result.violations == ("input_pipeline", "measurement_boundary_id")


def test_stress_levels_under_one_override_remain_comparable():
    base = load("g1-adapter-override")
    keys = [_keys(variant(base, plan={"stress": dataclasses.replace(base.plan.stress, level=level)}))
            for level in ("none", "L1", "L3")]
    assert len({k.key_stress_contrast for k in keys}) == 1
    assert design_check(keys[0], keys[2], "stress").status is GuardStatus.PASS


def test_different_override_implementations_have_different_boundaries_and_keys():
    good, divergent = load("g1-adapter-override"), load("override-equivalence-failure")
    assert _boundary_id(good) != _boundary_id(divergent)  # metrics carry the boundary id
    result = design_check(_keys(good), _keys(divergent), "stress")
    assert result.status is GuardStatus.FAIL and "input_pipeline" in result.violations
