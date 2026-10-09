"""Stage 1R contracts: realtime plan section, realtime boundary (v3), manifest v3/v4 rules, events v2,
reader version consistency, recorder hand-over to the native writer."""

from __future__ import annotations

import dataclasses
import json

import pytest

from sdv_bmt.contracts.ids import sha256_hex
from sdv_bmt.contracts.records import RunManifest
from sdv_bmt.contracts.specs import RealtimeConfig, RealtimeMeasurementBoundary, SpecError, measurement_boundary_id, parse_strict
from sdv_bmt.contracts.status import InputPipelineOwner, RunClockDomain
from sdv_bmt.observation.events import EventSchemaError, make_event, validate_event
from sdv_bmt.observation.reader import BundleReadError, load_run_bundle
from sdv_bmt.observation.recorder import ObservationRecorder, RecorderError
from tests._helpers import ZERO_SHA, load, reseal, run_plan

pytestmark = pytest.mark.usefixtures("native_build")


def test_realtime_config_is_strict():
    cfg = parse_strict(RealtimeConfig, {"cpu_affinity": [3, 1, 2], "precondition_requests": 4})
    assert cfg.cpu_affinity == (1, 2, 3) and cfg.wait_policy.value == "spin"
    for bad in ({"cpu_affinity": []}, {"cpu_affinity": [1, 1]}, {"cpu_affinity": [-1]},
                {"cpu_affinity": [0], "precondition_requests": -1}, {"cpu_affinity": [0], "wait_policy": "sleep"}):
        with pytest.raises(SpecError):
            parse_strict(RealtimeConfig, bad)


def _boundary(**over):
    base = dict(schema_version=3, clock_domain=RunClockDomain.MONOTONIC, payload_read_decode="excluded",
                preprocess="excluded", prepare_input_h2d="included", submit="s", complete="c",
                completion_observation="o", postprocess="output_ready_only", output_ready="r",
                latency_definitions={}, device_timing=None, instrumentation={"x": 1},
                output_pipeline={"kind": "reference_convnet_v1"}, input_pipeline_owner=InputPipelineOwner.WORKLOAD,
                preprocess_semantics_id="p/1", fused_steps=(), input_pipeline_digest=ZERO_SHA)
    base.update(over)
    return RealtimeMeasurementBoundary(**base)


def test_realtime_boundary_rules_and_identity():
    b = _boundary()
    with pytest.raises(ValueError):
        _boundary(clock_domain=RunClockDomain.VIRTUAL)
    with pytest.raises(ValueError):
        _boundary(schema_version=2)
    with pytest.raises(ValueError):
        _boundary(prepare_input_h2d="sometimes")
    # instrumentation, H2D inclusion and the output pipeline are each part of the boundary identity
    ids = {measurement_boundary_id(x) for x in (b, _boundary(instrumentation={"x": 2}),
                                                  _boundary(prepare_input_h2d="excluded"),
                                                  _boundary(output_pipeline={"kind": "other"}))}
    assert len(ids) == 4


def test_device_observation_exists_only_in_event_schema_v2():
    e = make_event("device_observation", 0, 1, "monotonic", {"request_id": 0, "timing": None}, schema_version=2)
    assert validate_event(e) == []
    with pytest.raises(EventSchemaError):
        make_event("device_observation", 0, 1, "virtual", {"request_id": 0}, schema_version=1)
    assert any("not part of event schema version 1" in p for p in validate_event({**e, "schema_version": 1}))


@pytest.fixture(scope="module")
def realtime_bundle(tmp_path_factory):
    return run_plan(load("r1-fake-native-rate-10khz"), tmp_path_factory.mktemp("rt"), run_id="run-rt-0").run_dir


def _manifest(run_dir):
    return json.loads((run_dir / "manifest.json").read_text())


def test_v4_manifest_requires_its_realtime_sections(realtime_bundle):
    m = _manifest(realtime_bundle)
    parse_strict(RunManifest, m, "manifest")
    for section, key in (("run_clock", "anchor_monotonic_ns"), ("realtime", "cpu_affinity_realized")):
        broken = json.loads(json.dumps(m))
        del broken[section][key]
        with pytest.raises(SpecError):
            parse_strict(RunManifest, broken, "manifest")
    as_actual = dict(m, execution_kind="actual")  # an actual manifest never carries a mock notice
    with pytest.raises(SpecError):
        parse_strict(RunManifest, as_actual, "manifest")


def test_v3_manifest_stays_virtual_and_mock_only(realtime_bundle):
    m = _manifest(realtime_bundle)
    with pytest.raises(SpecError):
        parse_strict(RunManifest, dict(m, schema_version=3), "manifest")  # v3 has no run_clock / realtime
    v3 = {k: v for k, v in m.items() if k not in ("run_clock", "realtime")}
    v3["schema_version"] = 3
    parse_strict(RunManifest, v3, "manifest")  # shape accepted only because the run is mock
    with pytest.raises(SpecError):
        parse_strict(RunManifest, dict(v3, execution_kind="actual", mock_notice=None,
                                       status=dict(v3["status"], execution_kind="actual"),
                                       realization=dict(v3["realization"], execution_kind="actual")), "manifest")


def test_reader_rejects_mixed_event_versions(realtime_bundle, tmp_path):
    import shutil
    copy = tmp_path / "copy"
    shutil.copytree(realtime_bundle, copy)
    reseal(copy, lambda es: [dict(e, schema_version=1) if e["kind"] == "submit_begin" else e for e in es])
    with pytest.raises(BundleReadError):
        load_run_bundle(copy)


def test_recorder_hand_over_checks_continuity(tmp_path):
    rec = ObservationRecorder(tmp_path, RunClockDomain.MONOTONIC, 2)
    rec.event("phase", 10, phase="x", edge="begin")
    seq, last_t, offset = rec.suspend()
    assert (seq, last_t, offset) == (1, 10, 0)
    with pytest.raises(RecorderError):
        rec.event("phase", 11, phase="x", edge="end")  # no writes while another writer owns the files
    with pytest.raises(RecorderError):
        rec.resume(0, 10, 0)  # seq moved backwards
    with pytest.raises(RecorderError):
        rec.resume(1, 10, 5)  # outputs.bin does not have the reported size
    with open(tmp_path / "events.jsonl", "ab") as fh:  # the other writer appends one event
        fh.write(json.dumps(make_event("phase", 1, 20, "monotonic", {"phase": "y", "edge": "end"}, 2),
                            sort_keys=True, separators=(",", ":")).encode() + b"\n")
    rec.resume(2, 20, 0)
    rec.event("phase", 25, phase="z", edge="end")
    rec.close()
    seqs = [json.loads(line)["seq"] for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert seqs == [0, 1, 2]
