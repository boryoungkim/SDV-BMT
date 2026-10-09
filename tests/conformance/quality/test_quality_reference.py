"""Metric Set v1: the frozen task-quality reference (manifest v5) and bundle-only quality evaluation.

The evidence path is exercised on the native measurement plane with its MOCK backend: the reference
conformance workload on an annotated copy of its fixture index, with a test-only quality evaluator
declaration (the detection evaluator itself is covered by tests/unit/test_quality_detection.py and,
on the GPU, by tests/device).  Tests marked "local evidence" re-use the Stage 2b-G GPU bundles and
dataset index of this host (copied, never modified) and are skipped where they are absent.
"""

from __future__ import annotations

import builtins
import dataclasses
import hashlib
import io
import json
import shutil

import pytest

from sdv_bmt.contracts.ids import canonical_json_bytes, digest_of, pretty_json_bytes
from sdv_bmt.contracts.quality import QUALITY_EVALUATORS, QualityEvaluatorDeclaration
from sdv_bmt.contracts.records import QUALITY_REFERENCE_FILE, raw_evidence_files
from sdv_bmt.contracts.specs import DatasetSelectionSpec, EvaluationOverride, SourceIndexSpec
from sdv_bmt.contracts.status import MetricStatus
from sdv_bmt.evidence.bundle import raw_file_records
from sdv_bmt.evidence.provenance import PACKAGE_ROOT
from sdv_bmt.evidence.validity import evaluate_contrast_validity
from sdv_bmt.metrics.evaluator import EvaluationError
from sdv_bmt.observation.reader import load_run_bundle
from sdv_bmt.runner.plan import validate_plan
from sdv_bmt.runner.runner import BenchmarkRunner, PlanRejected
from tests._helpers import (
    LEGACY_CATALOG,
    METRIC_SET_V1,
    REPO_ROOT,
    evaluate,
    load,
    metric_map,
    run_plan,
    variant,
)

pytestmark = pytest.mark.usefixtures("native_build")
FLOOR = "r1-fake-native-harness-floor"
TEST_EVALUATOR = "test-label-quality/1"
TEST_SCHEMA = "test-label/1"
ITEMS = 16  # first 16 items of the floor plan's fixture keep the native runs short


def small_floor():
    rp = load(FLOOR)
    return variant(rp, input_spec=dataclasses.replace(rp.input_spec, selection=DatasetSelectionSpec(0, ITEMS)))


def annotated_plan(tmp_path, monkeypatch, rows_filter=None):
    """The fake-native floor plan on an annotated copy of its index, declaring a test-only evaluator."""
    rp = small_floor()
    source = (rp.input_spec_dir / rp.input_spec.source.index).resolve()
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for i, row in enumerate(rows):
        row["annotation"] = {"label": i % 7, "boxes": [[i, i + 0.5, 2.25, 3.125]]}
    if rows_filter:
        rows = rows_filter(rows)
    text = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "annotated.index.jsonl").write_text(text)
    spec = dataclasses.replace(
        rp.input_spec,
        family=dataclasses.replace(rp.input_spec.family, annotation_schema=TEST_SCHEMA),
        source=SourceIndexSpec("annotated.index.jsonl", hashlib.sha256(text.encode()).hexdigest()))
    monkeypatch.setitem(QUALITY_EVALUATORS, TEST_EVALUATOR, QualityEvaluatorDeclaration(
        TEST_EVALUATOR, rp.workload.task, TEST_SCHEMA, ()))
    workload = dataclasses.replace(rp.workload, quality_evaluator=TEST_EVALUATOR)
    return variant(rp, input_spec=spec, input_spec_dir=fixtures, workload=workload), fixtures / "annotated.index.jsonl"


@pytest.fixture(scope="module")
def v5_source(tmp_path_factory):
    """One native mock run with a frozen quality reference; tests that alter a bundle get a copy."""
    patch = pytest.MonkeyPatch()
    try:
        tmp = tmp_path_factory.mktemp("v5")
        rp, index = annotated_plan(tmp, patch)
        out = run_plan(rp, tmp / "runs", run_id="run-v5-0")
    finally:
        patch.undo()
    return out, index


@pytest.fixture
def v5_run(v5_source, tmp_path):
    out, index = v5_source
    run_dir = tmp_path / "bundle"
    shutil.copytree(out.run_dir, run_dir, ignore=shutil.ignore_patterns("evaluations"))
    return dataclasses.replace(out, run_dir=run_dir), index


def _reseal_quality(run_dir, rows):
    (run_dir / QUALITY_REFERENCE_FILE).write_bytes(b"".join(canonical_json_bytes(r) + b"\n" for r in rows))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["files"] = raw_file_records(run_dir, raw_evidence_files(manifest["schema_version"]))
    (run_dir / "manifest.json").write_bytes(pretty_json_bytes(manifest))


# ---------------------------------------------------------------- v5 evidence


def test_a_declared_quality_evaluator_freezes_the_ground_truth_in_a_v5_bundle(v5_run):
    out, index = v5_run
    assert out.measurement_status.value == "valid" and out.request_counts["completed"] == ITEMS
    manifest = json.loads((out.run_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == 5
    q = manifest["quality_reference"]
    assert q["file"] == QUALITY_REFERENCE_FILE and q["evaluator"] == TEST_EVALUATOR
    assert q["annotation_schema"] == TEST_SCHEMA and q["item_count"] == ITEMS
    assert q["annotations_digest"] == manifest["input"]["annotations_digest"]
    assert QUALITY_REFERENCE_FILE in manifest["files"]
    bundle = load_run_bundle(out.run_dir)
    source = {json.loads(line)["source_item_id"]: json.loads(line)["annotation"] for line in index.read_text().splitlines()}
    for row, item in zip(bundle.quality_reference, bundle.input_items):
        assert (row.item_index, row.source_item_id, row.annotation_digest) == \
               (item.item_index, item.source_item_id, item.annotation_digest)
        assert row.annotation == source[item.source_item_id] and digest_of(row.annotation) == item.annotation_digest
    m = metric_map(evaluate(out.run_dir, catalog=METRIC_SET_V1))
    assert m["data.integrity.annotation_verified_rate"].value == 1.0
    assert m["data.traceability.complete"].value == 1
    q_ap = m["quality.detection.conditional.ap"]  # the test evaluator declares no implementation
    assert (q_ap.status, q_ap.status_reason) == (MetricStatus.NOT_SUPPORTED,
                                                 f"quality_evaluator_not_implemented:{TEST_EVALUATOR}")


def test_runs_without_a_quality_evaluator_stay_v4(tmp_path):
    out = run_plan(small_floor(), tmp_path)
    manifest = json.loads((out.run_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == 4 and "quality_reference" not in manifest
    assert not (out.run_dir / QUALITY_REFERENCE_FILE).exists()
    assert set(manifest["files"]) == set(raw_evidence_files(4))


def test_quality_evaluation_never_reads_the_external_index(v5_run, monkeypatch):
    out, index = v5_run
    first = evaluate(out.run_dir, eval_id="eval-before", catalog=METRIC_SET_V1)
    index.unlink(missing_ok=True)  # the mutable external dataset location is gone
    assert not index.exists()
    opened = []
    real_open = builtins.open

    def tracking_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)
    monkeypatch.setattr(io, "open", tracking_open)
    second = evaluate(out.run_dir, eval_id="eval-after", catalog=METRIC_SET_V1)
    monkeypatch.undo()
    assert first.metrics_content_sha256 == second.metrics_content_sha256
    outside = [p for p in opened if not (p.startswith(str(out.run_dir.resolve())) or p.startswith(str(PACKAGE_ROOT)))]
    assert outside == []
    assert any(p.endswith(QUALITY_REFERENCE_FILE) for p in opened)


def test_tampered_quality_reference_is_refused_and_raw_files_stay_unchanged(v5_run):
    out, _ = v5_run
    before = {n: (out.run_dir / n).read_bytes() for n in raw_evidence_files(5) + ("manifest.json",)}
    evaluate(out.run_dir, eval_id="eval-a", catalog=METRIC_SET_V1)
    evaluate(out.run_dir, eval_id="eval-b", catalog=LEGACY_CATALOG)
    assert {n: (out.run_dir / n).read_bytes() for n in before} == before
    with pytest.raises(FileExistsError):
        evaluate(out.run_dir, eval_id="eval-a", catalog=METRIC_SET_V1)
    with open(out.run_dir / QUALITY_REFERENCE_FILE, "ab") as fh:
        fh.write(b" ")
    for catalog in (METRIC_SET_V1, LEGACY_CATALOG):
        with pytest.raises(EvaluationError, match="raw integrity violated"):
            evaluate(out.run_dir, eval_id=f"eval-tampered-{len(catalog)}", catalog=catalog)


def test_resealed_annotation_change_is_detected_against_the_item_digests(v5_run):
    out, _ = v5_run
    rows = [json.loads(line) for line in (out.run_dir / QUALITY_REFERENCE_FILE).read_text().splitlines()]
    rows[5]["annotation"]["label"] = 99  # digests in manifest.files re-recorded: only the item digest can catch it
    _reseal_quality(out.run_dir, rows)
    m = metric_map(evaluate(out.run_dir, catalog=METRIC_SET_V1))
    assert m["data.integrity.annotation_verified_rate"].value == (ITEMS - 1) / ITEMS
    q = m["quality.detection.coverage"]
    assert (q.status, q.status_reason) == (MetricStatus.INVALID, "quality_reference_integrity")
    assert m["data.traceability.complete"].value == 0


def test_measurement_guard_refuses_different_evaluation_protocols_or_catalogs(v5_run):
    out, _ = v5_run
    manifest = json.loads((out.run_dir / "manifest.json").read_text())

    def em(eval_id):
        return json.loads((out.run_dir / "evaluations" / eval_id / "eval_manifest.json").read_text())

    evaluate(out.run_dir, eval_id="eval-plain", catalog=METRIC_SET_V1)
    evaluate(out.run_dir, eval_id="eval-rate", catalog=METRIC_SET_V1,
             override=EvaluationOverride(schema_version=1, deadline=None, required_rate_hz=30.0))
    evaluate(out.run_dir, eval_id="eval-legacy", catalog=LEGACY_CATALOG)
    same = evaluate_contrast_validity(manifest, manifest, "stress", evaluations=(em("eval-plain"), em("eval-plain")))
    assert same.measurement_guard.status.value == "pass"
    rate = evaluate_contrast_validity(manifest, manifest, "stress", evaluations=(em("eval-plain"), em("eval-rate")))
    assert "evaluation_protocol.required_rate_hz" in rate.measurement_guard.violations
    mixed = evaluate_contrast_validity(manifest, manifest, "stress", evaluations=(em("eval-plain"), em("eval-legacy")))
    assert "evaluator.catalog_version" in mixed.measurement_guard.violations
    assert not mixed.attribution_allowed


# ---------------------------------------------------------------- plan validation / PREFLIGHT


def test_plan_validation_of_declared_quality_evaluators(tmp_path, monkeypatch):
    rp, _ = annotated_plan(tmp_path, monkeypatch)
    assert validate_plan(rp) == []
    unknown = variant(rp, workload=dataclasses.replace(rp.workload, quality_evaluator="nope/1"))
    assert any("unknown quality evaluator" in i for i in validate_plan(unknown))
    other_task = variant(rp, workload=dataclasses.replace(rp.workload, task="other-task"))
    assert any("evaluates task" in i for i in validate_plan(other_task))
    other_schema = variant(rp, input_spec=dataclasses.replace(
        rp.input_spec, family=dataclasses.replace(rp.input_spec.family, annotation_schema="x/1")))
    assert any("reads annotation schema" in i for i in validate_plan(other_schema))
    virtual = variant(load("g1-dataset-fixed-rate-nominal"),
                      workload=dataclasses.replace(load("g1-dataset-fixed-rate-nominal").workload,
                                                   quality_evaluator="coco-detection-bbox/1"))
    assert any("out of Stage 1A (virtual runner) scope" in i for i in validate_plan(virtual))


def test_preflight_rejects_missing_ground_truth(tmp_path, monkeypatch):
    def drop_one(rows):
        rows[3].pop("annotation")
        return rows

    rp, _ = annotated_plan(tmp_path, monkeypatch, rows_filter=drop_one)
    with pytest.raises(PlanRejected, match="ground truth unavailable"):
        run_plan(rp, tmp_path / "runs")
    assert not (tmp_path / "runs").exists()

    from sdv_bmt.cli.main import build_components

    (tmp_path / "b").mkdir()
    rp_ok, _ = annotated_plan(tmp_path / "b", monkeypatch)

    def without_source(plan):
        return dataclasses.replace(build_components(plan), annotations=None)

    with pytest.raises(PlanRejected, match="no annotation source"):
        BenchmarkRunner(rp_ok, without_source).preflight()


# ---------------------------------------------------------------- old bundles (local evidence)


REAL = REPO_ROOT / "runs" / "stage2b-g-real"
REAL_RUNS = [
    "baseline/br-yolox-30hz/run-20261007T201237Z-86f737e638ed",
    "baseline/br-yolox-100hz/run-20261007T201549Z-6923db40b4b5",
    "baseline/br-yolox-asap/run-20261007T204215Z-423170ad503c",
]


def _copy_bundle(rel, target):
    src = REAL / rel
    if not src.is_dir():
        pytest.skip("Stage 2b-G evidence not present on this host")
    shutil.copytree(src, target, ignore=shutil.ignore_patterns("evaluations"))
    archived = sorted((src / "evaluations").iterdir())[0]
    return json.loads((archived / "eval_manifest.json").read_text())


@pytest.mark.parametrize("rel", REAL_RUNS)
def test_legacy_catalog_reproduces_archived_v4_evaluations(rel, tmp_path):
    archived = _copy_bundle(rel, tmp_path / "bundle")
    outcome = evaluate(tmp_path / "bundle", catalog=LEGACY_CATALOG)
    assert outcome.metrics_content_sha256 == archived["metrics_content_sha256"]
    m = metric_map(evaluate(tmp_path / "bundle", eval_id="eval-v1", catalog=METRIC_SET_V1))
    assert m["quality.detection.conditional.ap"].status_reason == "quality_evaluator_not_declared"
    assert m["bmt.dut_service.p50"].status is MetricStatus.VALID and m["bmt.h2d.p50"].status is MetricStatus.VALID


def test_resealed_gpu_bundle_as_v5_reproduces_the_regression_anchor(tmp_path):
    """Synthetic v5 copy of a v4 GPU bundle (ground truth from the digest-pinned dataset index) evaluated
    through the full canonical path: the analysis-plane COCO result of that run is reproduced exactly."""
    from sdv_bmt.contracts.specs import parse_input_spec
    from sdv_bmt.input.annotations import IndexAnnotationSource
    from sdv_bmt.runner.plan import load_yaml_strict

    spec_path = REPO_ROOT / "configs" / "inputs" / "coco-val2017-first1000.yaml"
    index = REPO_ROOT / "datasets" / "coco-val2017-first1000" / "index.jsonl"
    recorded = REAL / "analysis" / "quality" / "quality_br-yolox-30hz_run-20261007T201237Z-86f737e638ed.json"
    if not (index.is_file() and recorded.is_file()):
        pytest.skip("Stage 2b-G dataset/evidence not present on this host")
    run_dir = tmp_path / "bundle"
    _copy_bundle(REAL_RUNS[0], run_dir)
    bundle = load_run_bundle(run_dir)
    rows = IndexAnnotationSource(parse_input_spec(load_yaml_strict(spec_path)), spec_path.parent).reference_rows(
        bundle.input_items)
    run_spec = json.loads((run_dir / "run_spec.json").read_text())
    run_spec["resolved_plan"]["workload"]["quality_evaluator"] = "coco-detection-bbox/1"
    (run_dir / "run_spec.json").write_bytes(pretty_json_bytes(run_spec))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["schema_version"] = 5
    manifest["quality_reference"] = {"file": QUALITY_REFERENCE_FILE, "schema_version": 1,
                                     "evaluator": "coco-detection-bbox/1",
                                     "annotation_schema": "coco-2017-instances-bbox/1", "item_count": len(rows),
                                     "annotations_digest": digest_of([r["annotation_digest"] for r in rows])}
    (run_dir / "manifest.json").write_bytes(pretty_json_bytes(manifest))
    _reseal_quality(run_dir, rows)
    m = metric_map(evaluate(run_dir, catalog=METRIC_SET_V1))
    stats = json.loads(recorded.read_text())["conditional"]["stats"]
    for name, stat in (("ap", "AP"), ("ap50", "AP50"), ("ap75", "AP75"), ("ap_small", "AP_small"),
                       ("ap_medium", "AP_medium"), ("ap_large", "AP_large"), ("ar1", "AR1"), ("ar10", "AR10"),
                       ("ar100", "AR100")):
        assert m[f"quality.detection.conditional.{name}"].value == stats[stat], name
    assert m["quality.detection.conditional.ap"].value == 0.4289194068359202
    assert m["quality.detection.coverage"].value == 1.0
    assert m["data.integrity.annotation_verified_rate"].value == 1.0
    tp, fp, fn = (m[f"quality.detection.conditional.{k}_count"].value for k in ("tp", "fp", "fn"))
    assert tp + fn == m["quality.detection.conditional.tp_count"].population["gt_non_crowd"]
    assert m["quality.detection.conditional.precision"].value == tp / (tp + fp)
    assert m["quality.detection.conditional.f1"].value == 2 * tp / (2 * tp + fp + fn)
