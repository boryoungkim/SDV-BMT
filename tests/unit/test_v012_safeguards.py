"""v0.1.2 start-up safeguards (silent-contamination review section 4, approved 2026-11-06): each wrong setup is
rejected before anything is measured.

1. replay rate vs frame-period deadline (plan validation)
3. native executable vs source tree (build stamp, run start)
4. input config preprocessing contract vs the workload's input contract (preflight)
and the timing_offset 8-byte alignment check of the NVIDIA device programs.  Safeguard 2 (plan memory bounds and
output coverage) runs in the native executor and is tested on the device (tests/device/test_deeplab_ops_device.py).
"""

from __future__ import annotations

import dataclasses
import stat

import pytest

from sdv_bmt.contracts.specs import DeadlineSpec, WorkloadSpec, parse_strict
from sdv_bmt.contracts.status import DeadlinePolicy, ReplayMode
from sdv_bmt.dut import native_source
from sdv_bmt.dut.native import NATIVE_DIR, NativeRunnerError, native_identity, native_source_tree_sha256
from sdv_bmt.dut.nvidia.adapter import CudnnPlanProgram, DeviceProgram
from sdv_bmt.runner.plan import frame_period_ns, load_yaml_strict, validate_plan
from sdv_bmt.runner.runner import PlanRejected
from tests._helpers import CONFIGS, load, run_plan, variant

# ---------------------------------------------------------------- timing_offset alignment


def _cudnn(**kw):
    base = dict(language="cudnn_plan", plan="models/x/p.json", plan_sha256="0" * 64, weights="models/x/w.f32",
                weights_sha256="0" * 64, input_bytes=12, output_bytes=1206696, task_output_bytes=1206660,
                timing_offset=1206664)
    return CudnnPlanProgram(**{**base, **kw})


def test_misaligned_timing_offset_is_rejected():
    _cudnn()  # aligned: accepted
    with pytest.raises(ValueError, match="not a multiple of 8"):
        _cudnn(timing_offset=1206660)  # the YOLOX-Tiny value that faulted on the device
    DeviceProgram("cuda_c", "k", 1, 4, (), 112, 80, 80, "src")
    with pytest.raises(ValueError, match="not a multiple of 8"):
        DeviceProgram("cuda_c", "k", 1, 4, (), 116, 80, 84, "src")


def test_every_configured_timing_offset_is_aligned():
    for path in sorted((CONFIGS / "workloads").glob("*.yaml")):
        program = load_yaml_strict(path).get("options", {}).get("device_program") or {}
        if program.get("timing_offset") is not None:
            assert program["timing_offset"] % 8 == 0, path.name


# ---------------------------------------------------------------- 1. replay rate vs deadline


def test_frame_period_follows_the_written_rate():
    assert [frame_period_ns(r) for r in (30, 100, 280, 12.5, 29.97, 2, 1)] == [
        33_333_333, 10_000_000, 3_571_428, 80_000_000, 33_366_700, 500_000_000, 1_000_000_000]


def test_rate_changed_without_its_deadline_is_rejected():
    rp = load("g1-dataset-fixed-rate-nominal")  # 20 Hz, frame_period 50 ms
    assert validate_plan(rp) == []
    wrong = variant(rp, plan={"replay": dataclasses.replace(rp.plan.replay, rate_hz=10.0)})
    issues = validate_plan(wrong)
    assert len(issues) == 1 and "requires period_ns 100000000" in issues[0] and "not 50000000" in issues[0]
    fixed = variant(wrong, plan={"evaluation": dataclasses.replace(
        rp.plan.evaluation, deadline=DeadlineSpec(DeadlinePolicy.FRAME_PERIOD, 100_000_000))})
    assert validate_plan(fixed) == []


def test_recorded_and_asap_replay_are_out_of_scope():
    rp = load("g2-recorded-scenario-overload")
    assert rp.plan.replay.mode is ReplayMode.RECORDED and validate_plan(rp) == []


def test_every_fixed_rate_experiment_file_has_the_matching_deadline():
    checked = 0
    for path in sorted((CONFIGS / "experiments").glob("*.yaml")):
        data = load_yaml_strict(path)
        replay, deadline = data.get("replay", {}), (data.get("evaluation") or {}).get("deadline")
        if replay.get("mode") == "fixed_rate" and deadline and deadline.get("policy") == "frame_period":
            assert deadline["period_ns"] == frame_period_ns(replay["rate_hz"]), path.name
            checked += 1
    assert checked >= 35


# ---------------------------------------------------------------- 3. native executable vs source


def test_build_stamp_mismatch_and_absence_are_refused(tmp_path):
    exe = tmp_path / "sdv-bmt-native-fake"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(NativeRunnerError, match="no build stamp"):
        native_identity(exe)
    native_source.stamp_path(exe).write_text("0" * 64 + "\n")
    with pytest.raises(NativeRunnerError, match="rebuild it"):
        native_identity(exe)
    native_source.stamp_path(exe).write_text(native_source_tree_sha256() + "\n")
    assert native_identity(exe)["native.source_tree_sha256"] == native_source_tree_sha256()


def test_source_edit_without_rebuild_is_refused(tmp_path):
    native = tmp_path / "native"
    (native / "src").mkdir(parents=True)
    (native / "src" / "a.hpp").write_text("int a;\n")
    (native / "Makefile").write_text("all:\n")
    exe = native / "build" / "runner"
    exe.parent.mkdir()
    exe.write_text("x")
    native_source.stamp_path(exe).write_text(native_source.build_source_sha256(native) + "\n")
    assert native_source.check_stamp(exe, native) == native_source.native_source_tree_sha256(native)
    (native / "src" / "a.hpp").write_text("int a = 1;\n")  # edited, not rebuilt
    with pytest.raises(ValueError, match="rebuild"):
        native_source.check_stamp(exe, native)


def test_tool_stamp_covers_its_own_source(tmp_path):
    extra = tmp_path / "tool.cpp"
    extra.write_text("int main() {}\n")
    with_tool = native_source.build_source_sha256(NATIVE_DIR, [extra])
    assert with_tool != native_source_tree_sha256()
    extra.write_text("int main() { return 0; }\n")
    assert native_source.build_source_sha256(NATIVE_DIR, [extra]) != with_tool


def test_makefile_stamps_match_the_current_sources(native_build):
    for exe in ("sdv-bmt-native-fake", "sdv-bmt-native-tests"):
        assert native_source.check_stamp(native_build / exe, NATIVE_DIR) == native_source_tree_sha256()


# ---------------------------------------------------------------- 4. input contract vs workload contract


def test_mismatched_preprocessing_contract_is_rejected(tmp_path):
    rp = load("g1-dataset-fixed-rate-nominal")
    family = dataclasses.replace(rp.input_spec.family, preprocessing_contract="fake-bytes-v1")
    with pytest.raises(PlanRejected, match="input.family.preprocessing_contract"):
        run_plan(variant(rp, input_spec=dataclasses.replace(rp.input_spec, family=family)), tmp_path)


def test_every_actual_experiment_pairs_its_input_with_the_workload_contract():
    from sdv_bmt.cli.main import WORKLOADS

    checked = 0
    for path in sorted((CONFIGS / "experiments").glob("*.yaml")):
        data = load_yaml_strict(path)
        if data.get("execution_kind") != "actual":
            continue
        input_path = (path.parent / data["input"]).resolve()
        if not input_path.is_file():  # e.g. the Cityscapes input before the dataset is built on this host
            continue
        spec = parse_strict(WorkloadSpec, load_yaml_strict((path.parent / data["workload"]).resolve()))
        contract = WORKLOADS[spec.implementation](spec).input_contract()
        family = load_yaml_strict(input_path)["family"]
        assert family["preprocessing_contract"] == f"{contract.contract_id}/{contract.contract_version}", path.name
        checked += 1
    assert checked >= 20


def test_yolox_416_input_paired_with_a_640_workload_would_be_rejected():
    from sdv_bmt.workload.yolox import YoloxCocoWorkload

    spec = parse_strict(WorkloadSpec, load_yaml_strict(CONFIGS / "workloads" / "yolox-s-coco-v1.yaml"))
    contract = YoloxCocoWorkload(spec).input_contract()
    tiny_input = load_yaml_strict(CONFIGS / "inputs" / "coco-val2017-first1000-416.yaml")
    assert tiny_input["family"]["preprocessing_contract"] != f"{contract.contract_id}/{contract.contract_version}"
