"""Stage 1R-G device tests on the actual reference accelerator (GPU 0, RTX PRO 6000).

Skipped -- and reported as NOT RUN -- when the native NVIDIA runner cannot describe the target GPU.
Each test uses the 32-item reference input to stay short.  These results validate the measurement
system on a reference accelerator; they are not NPU results.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from sdv_bmt.contracts.specs import DUTConfig
from sdv_bmt.contracts.status import ExecutionStatus, MeasurementStatus
from sdv_bmt.dut.native import NativeRunnerError, describe_native, native_binary
from sdv_bmt.dut.nvidia import NvidiaCudaDUT
from sdv_bmt.observation.reader import fold_requests, load_run_bundle
from tests._helpers import evaluate, load, metric_map, read_jsonl, run_plan, variant

UUID = "GPU-f3e03bfa-61ab-acdf-0834-450de8becefc"


@pytest.fixture(scope="module", autouse=True)
def gpu(native_build):
    try:
        return describe_native(native_binary("sdv-bmt-native-nvidia"), {"uuid": UUID})
    except NativeRunnerError as exc:
        pytest.skip(f"NOT RUN: target GPU {UUID} not available to the native runner ({exc})")


def _small(name, **plan):
    rp = load(name)
    rp = variant(rp, input_spec=load("rg-gpu-timeout-1ms").input_spec)  # 32 reference images
    return variant(rp, plan=plan) if plan else rp


def _events(run_dir):
    return read_jsonl(run_dir / "events.jsonl")


def _states(run_dir):
    return {rid: req.state.value for rid, (req, _) in fold_requests(tuple(_events(run_dir))).items()}


def test_actual_run_end_to_end(tmp_path, gpu):
    out = run_plan(_small("rg-gpu-fixed-rate-nominal"), tmp_path)
    assert out.execution_status is ExecutionStatus.COMPLETED and out.measurement_status is MeasurementStatus.VALID
    assert out.request_counts["completed"] == 32
    bundle = load_run_bundle(out.run_dir)
    m = bundle.manifest
    assert m["execution_kind"] == "actual" and m["mock_notice"] is None and m["schema_version"] == 4
    identity = m["dut"]["descriptor"]["device_identity"]
    assert identity["role"] == "reference_accelerator_validation" and identity["accelerator_family"] == "GPU"
    assert identity["nvidia.uuid"] == UUID and identity["native.runner"] == "sdv-bmt-native-nvidia"
    metrics = metric_map(evaluate(out.run_dir))
    assert metrics["bmt.output.identity_rate"].value == 1.0  # golden from the independent CPU implementation
    obs = [e for e in bundle.events if e["kind"] == "device_observation"]
    assert len(obs) == 32
    assert set(obs[0]["timing"]) == {"host_to_device", "device_execution", "device_to_host"}
    assert obs[0]["observations"]["nvidia.h2d_bytes"] == 3145728
    evidence = {e["condition"]: e for e in m["realization"]["evidence"]}
    assert evidence["data_movement"]["validity"] == "observed"
    assert evidence["device_frequency"]["equivalence"].startswith("unapproved:")
    assert evidence["memory_pressure_bandwidth"]["validity"] == "not_observable"
    prepare = [e for e in bundle.events if e["kind"] == "phase" and e["phase"] == "dut.prepare" and e["edge"] == "end"]
    assert len(prepare[0]["details"]["native"]["cubin_sha256"]) == 64


def test_overload_drops_arrivals_while_the_slot_is_held(tmp_path):
    out = run_plan(_small("rg-gpu-overload-400hz"), tmp_path)
    assert out.measurement_status is MeasurementStatus.VALID
    c = out.request_counts
    assert c["dropped"] > 0 and c["completed"] > 0 and c["planned"] == c["dropped"] + c["issued"]


def test_short_timeout_gives_timed_out_and_late_completion(tmp_path):
    out = run_plan(load("rg-gpu-timeout-1ms"), tmp_path)
    assert out.measurement_status is MeasurementStatus.VALID
    c = out.request_counts
    assert c["timed_out"] == 32 and c["late_completion"] == 32 and c["completed"] == 0
    late = [e for e in _events(out.run_dir) if e["kind"] == "late_completion"]
    assert all(e["result"] == "ok" for e in late)


def test_drain_cap_with_work_still_on_the_device(tmp_path):
    out = run_plan(load("rg-gpu-timeout-cap"), tmp_path)
    assert out.execution_status is ExecutionStatus.COMPLETED  # DUT.cleanup waited for the stream
    assert out.measurement_status is MeasurementStatus.VALID and out.request_counts["unresolved"] == 0
    drain = [e for e in _events(out.run_dir) if e["kind"] == "phase" and e["phase"] == "drain" and e["edge"] == "end"]
    assert drain[0]["details"] == {"cap_reached": True, "outstanding_request_ids": [31]}


def test_device_resident_inputs_exclude_h2d_from_the_boundary(tmp_path):
    out = run_plan(_small("rg-gpu-device-resident"), tmp_path)
    assert out.measurement_status is MeasurementStatus.VALID
    m = json.loads((out.run_dir / "manifest.json").read_text())
    assert m["measurement_boundary"]["definition"]["prepare_input_h2d"] == "excluded"
    obs = [e for e in _events(out.run_dir) if e["kind"] == "device_observation"]
    assert "host_to_device" not in obs[0]["timing"] and obs[0]["observations"]["nvidia.h2d_bytes"] == 0
    assert metric_map(evaluate(out.run_dir))["bmt.output.identity_rate"].value == 1.0


def test_instrumentation_none_has_no_device_timing_and_another_identity(tmp_path):
    a = run_plan(_small("rg-gpu-no-events"), tmp_path / "a")
    b = run_plan(_small("rg-gpu-fixed-rate-nominal"), tmp_path / "b")
    obs = [e for e in _events(a.run_dir) if e["kind"] == "device_observation"]
    assert obs and all(e["timing"] is None and e["timing_method"] is None for e in obs)
    ma, mb = (json.loads((x.run_dir / "manifest.json").read_text()) for x in (a, b))
    assert ma["measurement_boundary"]["measurement_boundary_id"] != mb["measurement_boundary"]["measurement_boundary_id"]
    assert ma["comparison_keys"]["components"]["common_parts"]["telemetry"] != \
        mb["comparison_keys"]["components"]["common_parts"]["telemetry"]


def test_a_profiler_environment_that_contradicts_the_declaration_fails_prepare(tmp_path, monkeypatch):
    monkeypatch.setenv("NSYS_PROFILING_SESSION_ID", "0")  # observed injection, declared profiler: none
    out = run_plan(_small("rg-gpu-fixed-rate-nominal"), tmp_path)
    assert out.execution_status is ExecutionStatus.FAILED and out.execution_reason == "dut_prepare_failed"
    end = [e for e in _events(out.run_dir) if e["kind"] == "phase" and e["phase"] == "dut.prepare" and e["edge"] == "end"]
    assert "declared profiler 'none'" in end[0]["error"]["message"]


def test_per_request_allocation_is_counted_as_allocations_during_measure(tmp_path):
    out = run_plan(_small("bg-gpu-per-request-alloc", repetitions=1, realtime=dataclasses.replace(
        load("bg-gpu-per-request-alloc").plan.realtime, precondition_requests=2)), tmp_path)
    assert out.measurement_status is MeasurementStatus.VALID
    m = json.loads((out.run_dir / "manifest.json").read_text())
    buffer_state = {e["condition"]: e for e in m["realization"]["evidence"]}["buffer_state"]
    assert buffer_state["value"]["policy"] == "per_request_alloc"
    assert buffer_state["value"]["device_allocations_during_measure"] == out.request_counts["issued"]


def test_cleanup_is_idempotent_without_prepare():
    dut = NvidiaCudaDUT(DUTConfig(1, "x", "nvidia_cuda", UUID, 1, {}))
    dut.cleanup()
    dut.cleanup()


def test_yolox_cudnn_plan_end_to_end_with_pinned_algorithms(tmp_path, gpu):
    """Stage 2b-G real-model baseline path: YOLOX-S (cuDNN plan, pinned algorithm table) on 8 COCO images."""
    from sdv_bmt.contracts.specs import DUTConfig, parse_strict
    from sdv_bmt.dut.nvidia.adapter import cudnn_table_sha256
    from sdv_bmt.runner.plan import load_yaml_strict
    from tests._helpers import CONFIGS

    rp = load("br-yolox-algo-probe")
    dut = parse_strict(DUTConfig, load_yaml_strict(CONFIGS / "duts" / "nvidia-rtx-pro-6000-gpu0-cudnn-yolox-s.yaml"))
    rp = variant(rp, dut=dut)
    out = run_plan(rp, tmp_path)
    assert out.execution_status is ExecutionStatus.COMPLETED and out.measurement_status is MeasurementStatus.VALID
    assert out.request_counts["completed"] == 8
    bundle = load_run_bundle(out.run_dir)
    m = bundle.manifest
    identity = m["dut"]["descriptor"]["device_identity"]
    assert identity["nvidia.cudnn_version"] >= 90100 and identity["nvidia.cudnn_library_files"]
    assert m["measurement_boundary"]["definition"]["output_pipeline"]["kind"] == "yolox_coco_v1"
    prepare = next(e for e in bundle.events if e.get("phase") == "dut.prepare" and e.get("edge") == "end")
    facts = prepare["details"]["native"]["cudnn_plan"]
    assert facts["cudnn_algo_policy"] == "pinned" and facts["cudnn_convolutions"] == 83
    assert facts["cudnn_algorithm_table_sha256"] == cudnn_table_sha256(dut_options_table(dut))
    outputs = (out.run_dir / "outputs.bin").read_bytes()
    completed = [e for e in bundle.events if e["kind"] == "completed"]
    first = json.loads(outputs[completed[0]["output_ref"]["offset"]:][:completed[0]["output_ref"]["length"]])
    assert first["schema"] == "sdv-yolox-coco-detections/1" and first["image_id"] == 139 and first["detections"]
    freq = next(e for e in m["realization"]["evidence"] if e["condition"] == "device_frequency")
    assert freq["validity"] == "observed" and freq["source"].startswith("nvidia:probe_kernel")


def dut_options_table(dut):
    return tuple(tuple(r) for r in dut.runtime_options["cudnn"]["algorithms"])


def test_yolox_canonical_quality_from_a_frozen_reference(tmp_path, gpu):
    """Metric Set v1: a YOLOX run that declares coco-detection-bbox/1 freezes the ground truth of its 8 COCO
    images in a v5 bundle; the canonical evaluation reproduces a direct COCO evaluation of the same outputs
    against the dataset index, and a different operating point is a different evaluation protocol."""
    from sdv_bmt.contracts.ids import loads_strict_json
    from sdv_bmt.contracts.specs import DetectionOperatingPoint, EvaluationOverride, parse_strict
    from sdv_bmt.contracts.quality import COCO_2017_CATEGORY_IDS
    from sdv_bmt.evidence.validity import evaluate_contrast_validity
    from sdv_bmt.metrics import quality_coco
    from sdv_bmt.runner.plan import load_yaml_strict
    from tests._helpers import CONFIGS, METRIC_SET_V1

    rp = load("br-yolox-algo-probe")
    dut = parse_strict(DUTConfig, load_yaml_strict(CONFIGS / "duts" / "nvidia-rtx-pro-6000-gpu0-cudnn-yolox-s.yaml"))
    rp = variant(rp, dut=dut, workload=dataclasses.replace(rp.workload, quality_evaluator="coco-detection-bbox/1"))
    out = run_plan(rp, tmp_path)
    assert out.measurement_status is MeasurementStatus.VALID and out.request_counts["completed"] == 8
    bundle = load_run_bundle(out.run_dir)
    assert bundle.manifest["schema_version"] == 5 and len(bundle.quality_reference) == 8
    index = (rp.input_spec_dir / rp.input_spec.source.index).resolve()
    source = {r["source_item_id"]: r["annotation"] for r in map(json.loads, index.read_text().splitlines()[:8])}
    assert all(row.annotation == source[row.source_item_id] for row in bundle.quality_reference)
    outcome = evaluate(out.run_dir, eval_id="eval-default", catalog=METRIC_SET_V1)
    m = metric_map(outcome)
    assert m["quality.detection.coverage"].value == 1.0
    assert m["data.integrity.annotation_verified_rate"].value == 1.0 and m["data.traceability.complete"].value == 1
    # direct evaluation: dataset index ground truth + the bundle's encoded outputs
    gts, dets = {}, {}
    blob = (out.run_dir / "outputs.bin").read_bytes()
    for e in (e for e in bundle.events if e["kind"] == "completed"):
        ref = e["output_ref"]
        output = loads_strict_json(blob[ref["offset"]:ref["offset"] + ref["length"]].decode())
        ann = source[f"{output['image_id']:012d}"]
        gts[ann["image_id"]] = [{k: o[k] for k in ("id", "category_id", "bbox", "area", "iscrowd")} for o in ann["objects"]]
        dets[ann["image_id"]] = [{"category_id": d["category_id"], "bbox": [v / 1000.0 for v in d["bbox_mpx"]],
                                  "score": d["score_ppm"] / 1e6} for d in output["detections"]]
    direct = quality_coco.evaluate(gts, dets, list(COCO_2017_CATEGORY_IDS))["stats"]
    for name, stat in (("ap", "AP"), ("ap50", "AP50"), ("ap75", "AP75"), ("ar100", "AR100")):
        canonical = m[f"quality.detection.conditional.{name}"]
        assert canonical.value == (direct[stat] if direct[stat] > -1 else None), name
    strict = EvaluationOverride(schema_version=1, deadline=None,
                                quality_operating_point=DetectionOperatingPoint(score_threshold=0.75, iou_threshold=0.5))
    evaluate(out.run_dir, eval_id="eval-strict", catalog=METRIC_SET_V1, override=strict)
    manifest = json.loads((out.run_dir / "manifest.json").read_text())
    em = [json.loads((out.run_dir / "evaluations" / e / "eval_manifest.json").read_text())
          for e in ("eval-default", "eval-strict")]
    assert em[0]["parameter_sources"]["quality_operating_point"] == "evaluator_default"
    result = evaluate_contrast_validity(manifest, manifest, "stress", evaluations=tuple(em))
    assert "evaluation_protocol.quality" in result.measurement_guard.violations
