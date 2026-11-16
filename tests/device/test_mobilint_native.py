"""Device tests of the Mobilint MLA400 port (adapter mobilint_aries, native runner sdv-bmt-native-mobilint).

Run on the host of the MLA400 (``/dev/aries2``, the first chip of the card on the reference host) after
``tools/mobilint/build_native.sh``; the compiled models are host files named by the workloads
(``~/sdv_assets/compile/head9/*.mxq``).  Skipped -- and reported as NOT RUN -- when the runner is not built from
the current native sources, cannot describe the device, or the compiled model is missing.  Each run uses the
first 8 COCO images to stay short; the tests check behaviour, not performance, and assert no timing value.

    .venv/bin/python -m pytest tests/device/test_mobilint_native.py
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from sdv_bmt.cli import main as cli
from sdv_bmt.contracts.specs import DUTConfig, parse_input_spec
from sdv_bmt.contracts.status import EvidenceValidity, ExecutionStatus, MeasurementStatus, RealizedCondition
from sdv_bmt.dut.mobilint import MobilintAriesDUT
from sdv_bmt.dut.native import NativeRunnerError, describe_native, native_binary, native_identity
from sdv_bmt.runner.plan import load_yaml_strict
from tests._helpers import REPO_ROOT, load, read_jsonl, run_plan, variant

DEVICE = "/dev/aries2"
DEVICE_NUMBER = 2
CAPTURE = "yolox-s-mla400-capture"
FIRST8 = REPO_ROOT / "configs" / "inputs" / "coco-val2017-first8.yaml"
GOLDEN = REPO_ROOT / "configs" / "references" / \
    "coco-val2017-first1000.yolox-s-coco-v1-mla400-q8.npu-capture.golden.jsonl"


@pytest.fixture(scope="module", autouse=True)
def npu():
    binary = native_binary(MobilintAriesDUT.BINARY)
    try:
        native_identity(binary)  # built, and from the current native sources
        identity = describe_native(binary, {"device_number": DEVICE_NUMBER})
    except NativeRunnerError as exc:
        pytest.skip(f"NOT RUN: Mobilint native runner or {DEVICE} not available ({exc})")
    mxq = Path(load(CAPTURE).workload.options["device_program"]["mxq"]).expanduser()
    if not mxq.is_file():
        pytest.skip(f"NOT RUN: compiled model {mxq} not on this host")
    return identity


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    """The shared CLI does not register the adapter yet (request pending); tools/mobilint/sdv_bmt_mla400.py adds
    it at runtime, and so do the tests."""
    monkeypatch.setitem(cli.DUT_ADAPTERS, MobilintAriesDUT.ADAPTER, MobilintAriesDUT)


def _first8(name: str = CAPTURE):
    rp = load(name)
    spec = parse_input_spec(load_yaml_strict(FIRST8))
    return variant(rp, input_spec=spec, input_spec_dir=FIRST8.parent)


def _outputs_by_item(run_dir: Path) -> dict[int, str]:
    """item_index -> SHA-256 of its encoded task output (as tools/stage2b_g_real/capture_golden.py)."""
    outputs = (run_dir / "outputs.bin").read_bytes()
    item_of, out = {}, {}
    for e in read_jsonl(run_dir / "events.jsonl"):
        if e["kind"] == "issued":
            item_of[e["request_id"]] = e["item_indices"][0]
        elif e["kind"] == "completed":
            ref = e["output_ref"]
            blob = outputs[ref["offset"]:ref["offset"] + ref["length"]]
            assert hashlib.sha256(blob).hexdigest() == ref["sha256"]
            out[item_of[e["request_id"]]] = ref["sha256"]
    return out


@pytest.fixture(scope="module")
def capture(tmp_path_factory, npu):
    """One 8-image capture of YOLOX-S, shared by the tests that only read its bundle."""
    mp = pytest.MonkeyPatch()
    mp.setitem(cli.DUT_ADAPTERS, MobilintAriesDUT.ADAPTER, MobilintAriesDUT)
    try:
        out = run_plan(_first8(), tmp_path_factory.mktemp("capture"))
    finally:
        mp.undo()
    return out


def test_describe_identifies_the_device_and_the_runtime(npu):
    assert npu["role"] == "npu_target" and npu["accelerator_family"] == "NPU" and npu["vendor"] == "Mobilint"
    assert npu["mobilint.device_node"] == DEVICE and npu["mobilint.device_node_exists"] is True
    assert DEVICE_NUMBER in npu["mobilint.available_device_numbers"]
    assert npu["mobilint.qbruntime_version"] == "v1.5.0"
    # the runner links qbruntime 1.5.0 even where an older qbruntime is searched first (tools/mobilint/build_native.sh)
    assert npu["mobilint.qbruntime_library_realpath"].endswith("libqbruntime.so.1.5.0")


def test_capture_completes_every_request_with_the_committed_golden_outputs(capture):
    assert capture.execution_status is ExecutionStatus.COMPLETED
    assert capture.measurement_status is MeasurementStatus.VALID
    assert capture.request_counts["completed"] == 8
    golden = {r["item_index"]: r["output_sha256"] for r in read_jsonl(GOLDEN)}
    assert _outputs_by_item(capture.run_dir) == {i: golden[i] for i in range(8)}


def test_manifest_identity_and_measurement_declaration(capture):
    m = json.loads((capture.run_dir / "manifest.json").read_text())
    assert m["execution_kind"] == "actual" and m["mock_notice"] is None
    d = m["dut"]["descriptor"]
    assert d["adapter"] == "mobilint_aries" and d["completion_model"] == "polling"
    identity = d["device_identity"]
    assert identity["native.runner"] == "sdv-bmt-native-mobilint" and len(identity["native.runner_sha256"]) == 64
    definition = m["measurement_boundary"]["definition"]
    assert definition["prepare_input_h2d"] == "included"  # the runtime quantizes and transfers inside the request
    assert definition["device_timing"] is None


def test_requests_carry_output_stage_observations_and_no_device_timing(capture):
    obs = [e for e in read_jsonl(capture.run_dir / "events.jsonl") if e["kind"] == "device_observation"]
    assert len(obs) == 8
    for e in obs:
        assert e["timing"] is None
        assert e["observations"]["mobilint.future_get_ns"] > 0 and e["observations"]["mobilint.host_tail_ns"] > 0


def test_realization_evidence_declares_what_cannot_be_observed(capture):
    m = json.loads((capture.run_dir / "manifest.json").read_text())
    evidence = {e["condition"]: e["validity"] for e in m["realization"]["evidence"]}
    for condition in (RealizedCondition.DATA_MOVEMENT, RealizedCondition.DEVICE_FREQUENCY,
                      RealizedCondition.THERMAL_STATE, RealizedCondition.MEMORY_PRESSURE_BANDWIDTH):
        assert evidence[condition.value] == EvidenceValidity.NOT_OBSERVABLE.value
    assert evidence[RealizedCondition.RUNTIME_QUEUE_STATE.value] == EvidenceValidity.OBSERVED.value
    assert evidence[RealizedCondition.COMPETING_DEVICE_USERS.value] == EvidenceValidity.OBSERVED.value
    assert evidence[RealizedCondition.INPUT_RESIDENCY.value] == EvidenceValidity.SELF_REPORTED.value


def test_prepare_reports_one_core_and_the_nine_head_outputs(capture):
    end = [e for e in read_jsonl(capture.run_dir / "events.jsonl")
           if e["kind"] == "phase" and e["phase"] == "dut.prepare" and e["edge"] == "end"]
    facts = end[0]["details"]["native"]
    assert len(facts["target_cores"]) == 1
    assert len(facts["output_mapping"]) == 9


def test_a_second_capture_gives_identical_outputs(tmp_path, capture):
    again = run_plan(_first8(), tmp_path)
    assert again.measurement_status is MeasurementStatus.VALID
    assert _outputs_by_item(again.run_dir) == _outputs_by_item(capture.run_dir)


def test_a_compiled_model_with_another_sha256_is_refused(tmp_path):
    rp = _first8()
    options = copy.deepcopy(rp.workload.options)
    options["device_program"]["mxq_sha256"] = "0" * 64
    rp = variant(rp, workload=dataclasses.replace(rp.workload, options=options))
    try:
        out = run_plan(rp, tmp_path)
    except ValueError as exc:  # refused while the run specification is built
        assert "SHA-256" in str(exc)
        return
    assert out.execution_status is ExecutionStatus.FAILED  # refused before any request is issued
    assert not (out.request_counts or {}).get("completed")
    assert out.execution_reason == "dut_prepare_failed"
    end = [e for e in read_jsonl(out.run_dir / "events.jsonl")
           if e["kind"] == "phase" and e["phase"] == "dut.prepare" and e["edge"] == "end"]
    assert "SHA-256 differs" in end[0]["error"]["message"]


def test_cleanup_is_idempotent_without_prepare():
    dut = MobilintAriesDUT(DUTConfig(1, "x", "mobilint_aries", DEVICE, 1, {"core_mode": "single", "cluster": 0,
                                                                         "core": 0}))
    dut.cleanup()
    dut.cleanup()


def test_only_aries_device_nodes_and_single_core_are_accepted():
    with pytest.raises(ValueError):
        MobilintAriesDUT(DUTConfig(1, "x", "mobilint_aries", "/dev/nvidia0", 1, {}))
    with pytest.raises(Exception):
        MobilintAriesDUT(DUTConfig(1, "x", "mobilint_aries", DEVICE, 1, {"core_mode": "global8"}))
