"""S1A-11 evidence completeness, experiment.json identity, restore blocking on disk,
and strict bundle reading."""

from __future__ import annotations

import dataclasses
import json
import shutil

import pytest

from sdv_bmt.cli.main import build_components
from sdv_bmt.contracts.ids import file_sha256, is_sha256_hex
from sdv_bmt.contracts.records import RAW_EVIDENCE_FILES
from sdv_bmt.evidence.manifest import canonical_manifest_content, read_manifest
from sdv_bmt.evidence.provenance import code_identity, source_tree_sha256
from sdv_bmt.observation.reader import BundleReadError, load_run_bundle
from sdv_bmt.runner.runner import BenchmarkRunner, PlanRejected
from tests._helpers import load, reseal, run_plan, variant


def test_manifest_evidence_is_complete(tmp_path):
    out = run_plan(load("g1-dataset-fixed-rate-nominal"), tmp_path)
    m = read_manifest(out.run_dir)
    assert m["benchmark_level"] == "model"
    assert m["execution_kind"] == "mock" and m["status"]["execution_kind"] == "mock"
    assert "MOCK" in m["mock_notice"]
    assert is_sha256_hex(m["measurement_boundary"]["measurement_boundary_id"])
    assert m["measurement_boundary"]["definition"]["clock_domain"] == "virtual"
    for key in ("input_family_digest", "input_selection_digest", "items_digest", "source_index_sha256",
                "annotations_digest"):
        assert is_sha256_hex(m["input"][key]), key
    assert is_sha256_hex(m["stress"]["stress_axis_digest"]) and m["stress"]["stress_level"] == "none"
    assert m["stress"]["status"]["restore"] == "ok"
    assert m["code_identity"]["source_tree_sha256"] == source_tree_sha256()
    assert m["code_identity"]["vcs"]["commit_recorded"] is False
    for name in RAW_EVIDENCE_FILES:
        assert m["files"][name]["sha256"] == file_sha256(out.run_dir / name)
    assert m["output_reference"]["kind"] == "external_golden"
    assert is_sha256_hex(m["output_reference"]["sha256"])
    assert m["status"]["execution_status"] == "completed" and m["status"]["measurement_status"] == "valid"
    assert set(m["volatile_fields"]) == {"run_id", "created_at_utc", "environment"}
    assert "evaluation_level" not in json.dumps(m)


def test_mock_marker_in_every_written_document(tmp_path):
    out = run_plan(load("g2-recorded-scenario-overload"), tmp_path)
    exp = json.loads((out.run_dir.parent / "experiment.json").read_text())
    spec = json.loads((out.run_dir / "run_spec.json").read_text())
    for doc in (exp, spec, read_manifest(out.run_dir)):
        assert doc["execution_kind"] == "mock" and "MOCK" in doc["mock_notice"]


def test_canonical_manifest_identical_across_runs(tmp_path):
    rp = load("g2-recorded-scenario-overload")
    a = read_manifest(run_plan(rp, tmp_path / "a", run_id="run-a").run_dir)
    b = read_manifest(run_plan(rp, tmp_path / "b", run_id="run-b").run_dir)
    assert a["run_id"] != b["run_id"]
    assert canonical_manifest_content(a) == canonical_manifest_content(b)


def test_code_identity_is_cached_but_reflects_source_tree():
    assert code_identity()["source_tree_sha256"] == source_tree_sha256()
    assert code_identity()["source_file_count"] >= 30


def test_experiment_json_written_once_and_guards_plan_identity(tmp_path):
    rp = load("g2-recorded-scenario-overload")
    first = run_plan(rp, tmp_path, run_id="run-1")
    exp_file = first.run_dir.parent / "experiment.json"
    before = exp_file.read_bytes()
    run_plan(rp, tmp_path, run_id="run-2")
    assert exp_file.read_bytes() == before
    changed = variant(rp, plan={"seed": 7})
    with pytest.raises(PlanRejected, match="already holds experiment plan"):
        run_plan(changed, tmp_path, run_id="run-3")
    assert not (first.run_dir.parent / "run-3").exists()


def _restore_fail(rp):
    spec = dataclasses.replace(rp.stress_spec, provider_options={"fail_phase": "restore"})
    return variant(rp, stress_spec=spec, plan={"stress": dataclasses.replace(rp.plan.stress, level="L1")})


def test_restore_failure_on_disk_blocks_later_invocations(tmp_path):
    rp = _restore_fail(load("g1-dataset-fixed-rate-nominal"))
    run_plan(rp, tmp_path, run_id="run-1")
    fresh_runner = BenchmarkRunner(rp, build_components, run_id_factory=lambda: "run-2")
    with pytest.raises(PlanRejected, match="no successful stress restore"):
        fresh_runner.run_once(tmp_path)


def test_incomplete_run_blocks_only_without_restore_record(tmp_path):
    rp = load("g1-dataset-fixed-rate-nominal")
    done = run_plan(rp, tmp_path, run_id="run-1")
    (done.run_dir / "manifest.json").unlink()  # simulate a run that never finalized
    run_plan(rp, tmp_path, run_id="run-2")  # restore record present -> not blocked
    stress = done.run_dir / "stress.jsonl"
    stress.write_text("".join(l + "\n" for l in stress.read_text().splitlines() if '"step":"restore"' not in l))
    with pytest.raises(PlanRejected, match="run-1"):
        run_plan(rp, tmp_path, run_id="run-3")


# ---------------------------------------------------------------- strict reader


def test_reader_accepts_a_fresh_bundle(tmp_path):
    out = run_plan(load("g3-timeout-late-completion"), tmp_path)
    bundle = load_run_bundle(out.run_dir)
    assert len(bundle.input_items) == 3
    assert [r["seq"] for r in sorted(bundle.events + bundle.stress, key=lambda r: r["seq"])] == list(
        range(len(bundle.events) + len(bundle.stress)))


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda es: [dict(e, t_run_ns=0) if e["kind"] == "run_finished" else e for e in es], "time decreases"),
        (lambda es: [dict(e, extra=1) if e["kind"] == "issued" else e for e in es], "unknown"),
        (lambda es: [dict(e, run_clock_domain="monotonic") if e["kind"] == "issued" else e for e in es],
         "mixed run-clock domains"),
        (lambda es: [dict(e, kind="stress") if e["kind"] == "issued" else e for e in es], "missing"),
    ],
)
def test_reader_rejects_corrupt_event_streams(tmp_path, mutate, message):
    out = run_plan(load("g3-timeout-late-completion"), tmp_path)
    reseal(out.run_dir, mutate)
    with pytest.raises(BundleReadError, match=message):
        load_run_bundle(out.run_dir)


def test_reader_rejects_unknown_manifest_field(tmp_path):
    out = run_plan(load("g3-timeout-late-completion"), tmp_path)
    path = out.run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["evaluation_level"] = "model"
    path.write_text(json.dumps(manifest))
    with pytest.raises(BundleReadError, match="unknown field"):
        load_run_bundle(out.run_dir)


def test_bundle_copy_reads_identically(tmp_path):
    out = run_plan(load("g2-recorded-scenario-overload"), tmp_path / "runs")
    copy = tmp_path / "copy"
    shutil.copytree(out.run_dir, copy)
    a, b = load_run_bundle(out.run_dir), load_run_bundle(copy)
    assert (a.events, a.stress, a.input_items, a.manifest) == (b.events, b.stress, b.input_items, b.manifest)


# ---------------------------------------------------------------- review regressions


@pytest.mark.parametrize(
    "field,value",
    [("t_sched", "33000000"), ("request_id", -1), ("t_issue", 1.5), ("item_indices", []), ("item_indices", ["0"])],
)
def test_reader_rejects_wrongly_typed_request_fields(tmp_path, field, value):
    out = run_plan(load("g2-recorded-scenario-overload"), tmp_path)
    reseal(out.run_dir, lambda es: [dict(e, **{field: value}) if e["kind"] == "issued" else e for e in es])
    with pytest.raises(BundleReadError, match=field):
        load_run_bundle(out.run_dir)


def test_restore_failure_blocks_other_experiments_in_the_same_output_root(tmp_path):
    run_plan(_restore_fail(load("g1-dataset-fixed-rate-nominal")), tmp_path, run_id="run-1")
    with pytest.raises(PlanRejected, match="g1-dataset-fixed-rate-nominal/run-1"):
        run_plan(load("g2-recorded-scenario-overload"), tmp_path, run_id="run-2")
    run_plan(load("g2-recorded-scenario-overload"), tmp_path / "elsewhere", run_id="run-3")  # other root: allowed
