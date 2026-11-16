"""Run series (minimal; CenterPoint part): deterministic partition, plan rules, execution of the parts, series.json,
and the single result of ``sdv-bmt evaluate SERIES_DIR`` (MOCK, virtual clock, fake components)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from sdv_bmt.cli.main import build_components, main
from sdv_bmt.metrics.evaluator import EvaluationError, evaluate_run
from sdv_bmt.metrics.series import evaluate_series
from sdv_bmt.runner.plan import PlanError, load_plan, validate_plan
from sdv_bmt.runner.series import SERIES_FILE, measure_window_monotonic, partition, run_series
from tests._helpers import CONFIGS, METRIC_SET_V1_1, write_index

SCENES = ["A"] * 3 + ["B"] * 4 + ["C"] * 2 + ["D"]


def _rows(scenes):
    return [{"annotation": {"label": i % 3}, "payload_text": f"series sample {i:04d}", "source_item_id": f"sr-{i:04d}",
             "metadata": {"scene": s}} for i, s in enumerate(scenes)]


def _plan(tmp_path: Path, scenes=SCENES, *, max_items=5, count=None, series=True, **extra) -> Path:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    name, sha = write_index(fixtures, _rows(scenes), "series.index.jsonl")
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    selection = {"start": 0, **({"count": count} if count is not None else {})}
    (inputs / "series-ds.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "input_kind": "dataset", "input_id": "series-ds", "version": "1",
        "family": {"source_dataset": "fixture-dataset", "source_version": "1", "task": "fake-identity",
                   "annotation_schema": "fake-label-v1", "preprocessing_contract": "fake-identity-input/1"},
        "source": {"index": f"../fixtures/{name}", "index_sha256": sha}, "selection": selection}), encoding="utf-8")
    exp = tmp_path / "experiments"
    exp.mkdir(exist_ok=True)
    rel = lambda p: Path(os.path.relpath(p, exp)).as_posix()  # noqa: E731
    plan = {"schema_version": 1, "experiment_id": "series-test", "execution_kind": "mock", "seed": 0,
            "repetitions": 1, "input": "../inputs/series-ds.yaml",
            "workload": rel(CONFIGS / "workloads" / "fake-identity.yaml"),
            "dut": rel(CONFIGS / "duts" / "fake-blocking-25ms.yaml"),
            "stress": {"spec": rel(CONFIGS / "stress" / "fake-virtual-axis.yaml"), "level": "none"},
            "replay": {"mode": "fixed_rate", "rate_hz": 20},
            "run": {"max_inflight": 1, "queue_capacity": 0, "on_full": "drop_new",
                    "request_timeout_ns": 1_000_000_000, "drain_ns": 1_000_000_000,
                    "output_capture": "task_output_all", "on_request_error": "continue"},
            "evaluation": {"deadline": {"policy": "frame_period", "period_ns": 50_000_000}}, **extra}
    if series:
        plan["series"] = {"max_items_per_run": max_items, "boundary": "scene"}
    path = exp / "series-test.yaml"
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return path


def _ids(n):
    it = iter(range(n))
    return lambda: f"run-test-{next(it)}"


def test_partition_is_greedy_in_index_order_and_never_splits_a_boundary_block(tmp_path):
    parts = partition(load_plan(_plan(tmp_path)))
    assert [(p.start, p.count, p.boundary_values) for p in parts] == [
        (0, 3, ("A",)), (3, 4, ("B",)), (7, 3, ("C", "D"))]
    parts = partition(load_plan(_plan(tmp_path, max_items=7)))
    assert [(p.start, p.count) for p in parts] == [(0, 7), (7, 3)]
    parts = partition(load_plan(_plan(tmp_path, max_items=100, count=9)))
    assert [(p.start, p.count) for p in parts] == [(0, 9)]


@pytest.mark.parametrize("scenes,max_items,message", [
    (["A"] * 6 + ["B"], 5, "has 6 items"),
    (["A", "B", "A"], 5, "more than one block"),
])
def test_partition_refuses_selections_the_rule_cannot_split(tmp_path, scenes, max_items, message):
    with pytest.raises(PlanError, match=message):
        partition(load_plan(_plan(tmp_path, scenes, max_items=max_items)))


def test_partition_needs_the_boundary_metadata(tmp_path):
    path = _plan(tmp_path)
    rows = _rows(SCENES)
    del rows[4]["metadata"]
    name, sha = write_index(tmp_path / "fixtures", rows, "series.index.jsonl")
    spec = yaml.safe_load((tmp_path / "inputs" / "series-ds.yaml").read_text())
    spec["source"]["index_sha256"] = sha
    (tmp_path / "inputs" / "series-ds.yaml").write_text(yaml.safe_dump(spec))
    with pytest.raises(PlanError, match="no metadata.scene"):
        partition(load_plan(path))


def test_series_plan_rules(tmp_path):
    assert validate_plan(load_plan(_plan(tmp_path))) == []
    issues = validate_plan(load_plan(_plan(tmp_path, repetitions=2)))
    assert any("repetitions must be 1" in i for i in issues)
    golden = CONFIGS / "references" / "g1-dataset-fixture-8.fake-identity.golden.jsonl"
    ref = {"kind": "external_golden", "path": Path(os.path.relpath(golden, tmp_path / "experiments")).as_posix(),
           "sha256": __import__("hashlib").sha256(golden.read_bytes()).hexdigest()}
    issues = validate_plan(load_plan(_plan(tmp_path, evaluation={
        "deadline": {"policy": "frame_period", "period_ns": 50_000_000}, "output_reference": ref})))
    assert any("output reference is not supported" in i for i in issues)
    # a plan without a series keeps its digest (the field is only part of series plans)
    single = load_plan(_plan(tmp_path, series=False))
    assert "series" not in single.resolved_content()


def _metrics(outcome):
    return {m.metric_id: m for m in outcome.metrics}


def test_series_runs_the_parts_and_evaluates_to_one_result(tmp_path):
    rp = load_plan(_plan(tmp_path))
    so = run_series(rp, build_components, tmp_path / "out", run_id_factory=_ids(10),
                    series_id_factory=lambda: "series-test-0")
    assert so.status == "completed" and len(so.outcomes) == 3
    record = json.loads((so.series_dir / SERIES_FILE).read_text())
    assert record["status"] == "completed" and record["plan_digest"] == rp.plan_digest
    assert [p["count"] for p in record["parts"]] == [3, 4, 3]
    assert [p["run_dir"] for p in record["parts"]] == [
        f"part-0{k}/series-test/run-test-{k}" for k in range(3)]
    assert len(record["gaps_between_parts"]) == 2 and all(g["gap_ns"] >= 0 for g in record["gaps_between_parts"])
    # virtual-clock parts have no host-monotonic measurement window
    assert all(p["measure_window"] is None for p in record["parts"])
    assert all(g["measure_gap_ns"] is None for g in record["gaps_between_parts"])

    out = evaluate_series(so.series_dir, catalog=METRIC_SET_V1_1, eval_id_factory=lambda: "eval-series-0")
    m = _metrics(out)
    assert m["bmt.count.planned"].value == 10 and m["bmt.count.completed"].value == 10
    assert m["data.item_count"].value == 10 and m["data.traceability.complete"].value == 1
    assert m["bmt.deadline.completion_miss_count"].value == 0

    # the same items as one ordinary run: same counts and the same per-request distributions
    single_rp = load_plan(_plan(tmp_path / "single", series=False))
    from tests._helpers import run_plan
    single = run_plan(single_rp, tmp_path / "out-single")
    s = _metrics(evaluate_run(single.run_dir, catalog=METRIC_SET_V1_1, eval_id_factory=lambda: "eval-single-0"))
    for key in ("bmt.count.planned", "bmt.count.completed", "bmt.dut_service.p50", "bmt.dut_service.p99",
                "bmt.dut_service.max_observed", "bmt.request_completion.p90", "bmt.completion_rate",
                "bmt.deadline.output_ready_miss_rate"):
        assert (m[key].status, m[key].value) == (s[key].status, s[key].value), key
    # rates: completed over the sum of the parts' windows (each part has its own lead and drain)
    assert m["bmt.achieved_completion_rate"].status.value == "valid"
    assert m["bmt.offered_rate"].value == pytest.approx(20.0)
    assert "series_parts" in m["bmt.count.planned"].window and len(m["bmt.count.planned"].window["series_parts"]) == 3
    metrics_doc = json.loads((out.eval_dir / "metrics.json").read_text())
    assert metrics_doc["series_id"] == "series-test-0" and metrics_doc["context"]["run_count"] == 3
    assert [p["part"] for p in metrics_doc["part_realization"]] == [0, 1, 2]


def _fake_realtime_part(d: Path, anchor: int, begin: int, drain_end: int, domain: str = "monotonic") -> Path:
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"run_clock": {"domain": domain, "anchor_monotonic_ns": anchor}}))
    events = [{"kind": "phase", "phase": "measure", "edge": "begin", "t_run_ns": begin},
              {"kind": "phase", "phase": "drain", "edge": "begin", "t_run_ns": drain_end - 5},
              {"kind": "phase", "phase": "drain", "edge": "end", "outcome": "ok", "t_run_ns": drain_end,
               "details": {"cap_reached": False, "outstanding_request_ids": []}}]
    (d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return d


def test_measure_window_on_the_host_monotonic_clock(tmp_path):
    a = measure_window_monotonic(_fake_realtime_part(tmp_path / "a", 1_000, 200, 900))
    b = measure_window_monotonic(_fake_realtime_part(tmp_path / "b", 5_000, 300, 700))
    assert a == {"measure_begin_monotonic_ns": 1_200, "drain_end_monotonic_ns": 1_900}
    assert b["measure_begin_monotonic_ns"] - a["drain_end_monotonic_ns"] == 3_400
    assert measure_window_monotonic(_fake_realtime_part(tmp_path / "v", 0, 1, 2, domain="virtual")) is None
    assert measure_window_monotonic(tmp_path / "missing") is None


def test_series_evaluation_refuses_changed_or_incomplete_series(tmp_path):
    rp = load_plan(_plan(tmp_path))
    so = run_series(rp, build_components, tmp_path / "out", run_id_factory=_ids(10),
                    series_id_factory=lambda: "series-test-1")
    with pytest.raises(EvaluationError, match="legacy catalog"):
        evaluate_series(so.series_dir, catalog="bmt-portable-v1-stage1a")
    record_path = so.series_dir / SERIES_FILE
    record = json.loads(record_path.read_text())
    tampered = dict(record, parts=[dict(record["parts"][0], manifest_sha256="0" * 64)] + record["parts"][1:])
    record_path.chmod(0o644)
    record_path.write_text(json.dumps(tampered))
    with pytest.raises(EvaluationError, match="manifest.json differs"):
        evaluate_series(so.series_dir)
    record_path.write_text(json.dumps(dict(record, status="incomplete")))
    with pytest.raises(EvaluationError, match="did not complete"):
        evaluate_series(so.series_dir)
    swapped = dict(record, parts=[record["parts"][1], record["parts"][0], record["parts"][2]])
    record_path.write_text(json.dumps(swapped))
    with pytest.raises(EvaluationError, match="refused"):
        evaluate_series(so.series_dir)


def test_cli_runs_and_evaluates_a_series(tmp_path, capsys):
    plan = _plan(tmp_path)
    assert main(["validate", str(plan)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert [p["count"] for p in doc["series"]["parts"]] == [3, 4, 3]
    assert main(["run", str(plan), "--out", str(tmp_path / "out")]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["series"]["status"] == "completed" and len(doc["runs"]) == 3
    assert main(["evaluate", doc["series"]["series_dir"]]) == 0
    ev = json.loads(capsys.readouterr().out)
    assert ev["series"] is True and ev["metrics"]["bmt.count.completed"]["value"] == 10
