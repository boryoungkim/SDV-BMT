"""Metric table (``python -m sdv_bmt.cli.table``, ``metrics.csv`` form): model rows first and formed only as
median / min / max of the valid run values, run rows equal to the evaluation records, byte-identical output, the
console print of the model rows, and the integrity refusals.  MOCK bundles on the virtual clock (fake components)."""

from __future__ import annotations

import csv
import json
import os
import shutil
import statistics
from pathlib import Path

import pytest
import yaml

from sdv_bmt.cli.main import build_components
from sdv_bmt.cli.table import TableError, build_table, main, write_table
from sdv_bmt.metrics.evaluator import evaluate_run
from sdv_bmt.metrics.series import evaluate_series
from sdv_bmt.metrics.table import (COLUMN_NAMES, HEADLINE_ROW, HEADLINES, MODEL_ROW, RUN_ROW, format_headlines,
                                   metric_csv_bytes, model_rows, table_rows)
from sdv_bmt.runner.plan import load_plan
from sdv_bmt.runner.series import run_series
from tests._helpers import CONFIGS, METRIC_SET_V1_1, load, run_plan, write_index

G1, G2 = "g1-dataset-fixed-rate-nominal", "g2-recorded-scenario-overload"
LATE, EARLY = "2026-02-01T00:00:00.000000Z", "2026-01-01T00:00:00.000000Z"


def _evaluate(run_dir: Path, eval_id: str, created: str = EARLY):
    return evaluate_run(run_dir, catalog=METRIC_SET_V1_1, eval_id_factory=lambda: eval_id, now_factory=lambda: created)


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    runs = tmp_path_factory.mktemp("table") / "runs"
    for name, run_id in ((G1, "run-a"), (G1, "run-b"), (G2, "run-c")):
        outcome = run_plan(load(name), runs, run_id=run_id)
        _evaluate(outcome.run_dir, f"eval-{run_id}")
    return runs


@pytest.fixture
def copy(root, tmp_path) -> Path:
    target = tmp_path / "runs"
    shutil.copytree(root, target)
    return target


def _records(run_dir: Path, eval_id: str) -> dict:
    doc = json.loads((run_dir / "evaluations" / eval_id / "metrics.json").read_text(encoding="utf-8"))
    return {m["metric_id"]: m for m in doc["metrics"]}


def _read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


# ---------------------------------------------------------------- the table form


def _run(metric_id, value, status="valid", run_id="r1", **extra):
    return {"task": "t", "model": "m", "device": "d", "experiment_id": "e", "run_id": run_id, "eval_id": f"ev-{run_id}",
            "catalog_version": "c", "metric_id": metric_id, "layer": "performance", "unit": "ns",
            "status": status, "value": value, **extra}


def test_model_rows_are_median_min_max_of_the_valid_run_values():
    runs = [_run("a", 5, run_id="r1"), _run("a", 1, run_id="r2"), _run("a", 4, run_id="r3"),
            _run("a", 100, status="invalid", run_id="r4"),
            _run("p99", None, status="insufficient_samples", run_id="r1"),
            _run("p99", None, status="insufficient_samples", run_id="r2"),
            _run("mixed", None, status="invalid", run_id="r1"), _run("mixed", None, status="censored", run_id="r2"),
            _run("verdict", "pass", run_id="r1"), _run("verdict", "pass", run_id="r2"),
            _run("unit", "token", run_id="r1"), _run("unit", "chunk", run_id="r2")]
    rows = {r["metric_id"]: r for r in model_rows(runs)}
    assert list(rows) == ["a", "p99", "mixed", "verdict", "unit"]
    a = rows["a"]
    assert (a["row"], a["status"], a["value"], a["min"], a["max"], a["n_runs"], a["n_valid"]) == \
        (MODEL_ROW, "valid", 4, 1, 5, 4, 3)
    assert a["run_id"] is None and a["eval_id"] is None and a["catalog_version"] == "c"
    assert (rows["p99"]["status"], rows["p99"]["value"]) == ("insufficient_samples", None)
    assert rows["mixed"]["status"] == "no_valid_run"
    assert (rows["verdict"]["status"], rows["verdict"]["value"], rows["verdict"]["min"]) == ("valid", "pass", None)
    assert (rows["unit"]["status"], rows["unit"]["value"]) == ("differs_across_runs", None)


def test_all_model_rows_come_before_the_run_rows():
    first = [_run("a", 1), _run("b", 2)]
    second = [{**_run("a", 3), "experiment_id": "e2"}]
    rows = table_rows([first, second])
    assert [r["row"] for r in rows] == [MODEL_ROW] * 3 + [RUN_ROW] * 3
    assert [r["experiment_id"] for r in rows[:3]] == ["e", "e", "e2"]


def test_csv_cells_and_bytes_are_deterministic():
    rows = table_rows([[_run("a", 0.1), _run("b", None, status="censored"), _run("c", True, unit="boolean")]])
    data = metric_csv_bytes(rows)
    assert data == metric_csv_bytes(rows)
    parsed = list(csv.DictReader(data.decode("utf-8").splitlines()))
    assert tuple(parsed[0]) == COLUMN_NAMES
    run_a = next(r for r in parsed if r["row"] == RUN_ROW and r["metric_id"] == "a")
    assert run_a["value"] == "0.1" and run_a["min"] == "" and run_a["n_runs"] == ""
    assert next(r for r in parsed if r["row"] == RUN_ROW and r["metric_id"] == "b")["value"] == ""
    assert next(r for r in parsed if r["row"] == RUN_ROW and r["metric_id"] == "c")["value"] == "true"


# ---------------------------------------------------------------- experiments


def test_run_rows_equal_the_evaluation_and_model_rows_summarize_them(root, tmp_path):
    rows, _data = write_table([root / G1, root / G2], tmp_path / "metrics.csv")
    parsed = _read_csv(tmp_path / "metrics.csv")
    assert len(parsed) == len(rows)
    kinds = [r["row"] for r in parsed]
    assert kinds == sorted(kinds)  # "model" rows first, then "run" rows
    for experiment, runs in ((G1, ("run-a", "run-b")), (G2, ("run-c",))):
        records = {run_id: _records(root / experiment / run_id, f"eval-{run_id}") for run_id in runs}
        for run_id, by_id in records.items():
            mine = {r["metric_id"]: r for r in rows if r["row"] == RUN_ROW and r["run_id"] == run_id}
            assert set(mine) == set(by_id)
            for metric_id, record in by_id.items():
                assert {k: mine[metric_id][k] for k in ("layer", "unit", "status", "value")} == \
                    {k: record.get(k) for k in ("layer", "unit", "status", "value")}
                assert mine[metric_id]["eval_id"] == f"eval-{run_id}"
                assert mine[metric_id]["catalog_version"] == METRIC_SET_V1_1
        model = {r["metric_id"]: r for r in rows if r["row"] == MODEL_ROW and r["experiment_id"] == experiment}
        assert set(model) == set(next(iter(records.values())))
        values = [records[r]["bmt.dut_service.p50"]["value"] for r in runs]
        entry = model["bmt.dut_service.p50"]
        assert (entry["value"], entry["min"], entry["max"], entry["n_runs"], entry["n_valid"]) == \
            (statistics.median(values), min(values), max(values), len(runs), len(runs))
        assert entry["task"] == "fake-identity" and entry["model"] and entry["device"]
        recorded = json.loads((root / experiment / runs[0] / "manifest.json").read_text())["input"]
        dataset = f"{recorded['input_id']} ({recorded['item_count']:,} items)"  # the input set of manifest.json
        assert {r["dataset"] for r in rows if r["experiment_id"] == experiment} == {dataset}


def test_same_inputs_give_the_same_bytes(root, tmp_path):
    _rows, first = write_table([root / G1], tmp_path / "a.csv")
    _rows, second = write_table([root / G1], tmp_path / "b.csv")
    assert first == second == (tmp_path / "a.csv").read_bytes()


def test_the_command_writes_the_table_and_prints_the_model_rows(root, tmp_path, capsys):
    out = tmp_path / "metrics.csv"
    assert main([str(root / G1), "--out", str(out)]) == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["status"] == "written" and summary["experiments"] == 1
    assert summary["rows"] == len(_read_csv(out)) and summary["model_rows"] < summary["rows"]
    assert f"see the CSV file:\n  {out.resolve()}\n" in captured.err  # where the full table is
    assert "run-a" not in captured.err and "bmt." not in captured.err  # names only, no run rows
    assert format_headlines(build_table([root / G1])) in captured.err
    assert main([str(root / G1), "--out", str(out)]) == 2  # never overwritten
    assert "already exists" in capsys.readouterr().out


def test_the_newest_evaluation_is_used(copy, tmp_path):
    _evaluate(copy / G1 / "run-a", "eval-run-a-newer", LATE)
    rows = build_table([copy / G1])
    assert {r["eval_id"] for r in rows if r["row"] == RUN_ROW and r["run_id"] == "run-a"} == {"eval-run-a-newer"}


# ---------------------------------------------------------------- refusals


def test_a_tampered_raw_file_is_refused_and_nothing_is_written(copy, tmp_path):
    path = copy / G1 / "run-b" / "events.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(TableError, match="raw files differ from manifest.json"):
        write_table([copy / G1], tmp_path / "metrics.csv")
    assert not (tmp_path / "metrics.csv").exists()


def test_a_tampered_evaluation_is_refused(copy):
    path = copy / G1 / "run-a" / "evaluations" / "eval-run-a" / "metrics.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"valid"', '"invalid"', 1), encoding="utf-8")
    with pytest.raises(TableError, match="metrics.json differs from its eval_manifest.json record"):
        build_table([copy / G1])


def test_a_run_without_evaluation_and_bad_inputs_are_refused(copy, root):
    with pytest.raises(TableError, match="neither an experiment directory"):
        build_table([root])
    with pytest.raises(TableError, match="experiment given twice"):
        build_table([root / G1, root / G1])
    shutil.rmtree(copy / G2 / "run-c" / "evaluations")
    with pytest.raises(TableError, match="no evaluation"):
        build_table([copy / G2])


# ---------------------------------------------------------------- run series


def test_a_run_series_is_one_run_row_set(tmp_path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    rows = [{"annotation": {"label": i % 3}, "payload_text": f"series sample {i:04d}", "source_item_id": f"sr-{i:04d}",
             "metadata": {"scene": s}} for i, s in enumerate("AAABBBBCCD")]
    name, sha = write_index(fixtures, rows, "series.index.jsonl")
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "ds.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "input_kind": "dataset", "input_id": "series-ds", "version": "1",
        "family": {"source_dataset": "fixture-dataset", "source_version": "1", "task": "fake-identity",
                   "annotation_schema": "fake-label-v1", "preprocessing_contract": "fake-identity-input/1"},
        "source": {"index": f"../fixtures/{name}", "index_sha256": sha}, "selection": {"start": 0}}), encoding="utf-8")
    (tmp_path / "experiments").mkdir()
    rel = lambda p: Path(os.path.relpath(p, tmp_path / "experiments")).as_posix()  # noqa: E731
    plan = {"schema_version": 1, "experiment_id": "series-test", "execution_kind": "mock", "seed": 0,
            "repetitions": 1, "input": "../inputs/ds.yaml", "workload": rel(CONFIGS / "workloads" / "fake-identity.yaml"),
            "dut": rel(CONFIGS / "duts" / "fake-blocking-25ms.yaml"),
            "stress": {"spec": rel(CONFIGS / "stress" / "fake-virtual-axis.yaml"), "level": "none"},
            "replay": {"mode": "fixed_rate", "rate_hz": 20},
            "run": {"max_inflight": 1, "queue_capacity": 0, "on_full": "drop_new",
                    "request_timeout_ns": 1_000_000_000, "drain_ns": 1_000_000_000,
                    "output_capture": "task_output_all", "on_request_error": "continue"},
            "evaluation": {"deadline": {"policy": "frame_period", "period_ns": 50_000_000}},
            "series": {"max_items_per_run": 5, "boundary": "scene"}}
    plan_path = tmp_path / "experiments" / "series-test.yaml"
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    ids = iter(f"run-s-{k}" for k in range(10))
    so = run_series(load_plan(plan_path), build_components, tmp_path / "out-runs", run_id_factory=lambda: next(ids),
                    series_id_factory=lambda: "series-test-0")
    evaluate_series(so.series_dir, catalog=METRIC_SET_V1_1, eval_id_factory=lambda: "eval-series-0")
    table = build_table([so.series_dir])
    runs = [r for r in table if r["row"] == RUN_ROW]
    assert {r["run_id"] for r in runs} == {"series-test-0"} and {r["eval_id"] for r in runs} == {"eval-series-0"}
    assert {r["dataset"] for r in table} == {"series-ds (10 items)"}  # the items of all parts
    by_id = _records(so.series_dir, "eval-series-0")
    assert {r["metric_id"]: r["value"] for r in runs} == {k: v.get("value") for k, v in by_id.items()}
    model = [r for r in table if r["row"] == MODEL_ROW]
    assert all(r["n_runs"] == 1 for r in model)
    series = so.series_dir / "series.json"
    series.write_text(series.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(TableError, match="different series.json"):
        build_table([so.series_dir])


# ---------------------------------------------------------------- headline rows (readable names)


def _task_runs(task, values_by_metric, units, experiment_id="e"):
    rows = []
    for run_index in range(len(next(iter(values_by_metric.values())))):
        for metric_id, values in values_by_metric.items():
            rows.append({**_run(metric_id, values[run_index], run_id=f"r{run_index}"), "task": task,
                         "experiment_id": experiment_id, "unit": units.get(metric_id, "ratio")})
    return rows


def test_headline_rows_come_first_with_names_and_times_in_ms():
    task = "coco-2017-object-detection"
    values = {"bmt.dut_service.p50": [4_000_000, 5_000_000, 4_500_000], "bmt.output_ready.p99": [6_000_000] * 3,
              "bmt.deadline.output_ready_miss_rate": [0.0, 0.01, 0.0],
              "quality.detection.conditional.ap": [0.43] * 3, "bmt.count.completed": [1000] * 3}
    rows = table_rows([_task_runs(task, values, {"bmt.dut_service.p50": "ns", "bmt.output_ready.p99": "ns"})])
    kinds = [r["row"] for r in rows]
    assert kinds == [HEADLINE_ROW] * 4 + [MODEL_ROW] * 5 + [RUN_ROW] * 15
    heads = rows[:4]
    assert [h["metric_name"] for h in heads] == [h.name for h in HEADLINES[task]]
    assert [h["metric_id"] for h in heads] == [h.metric_id for h in HEADLINES[task]]
    service = heads[0]
    assert (service["unit"], service["value"], service["min"], service["max"]) == ("ms", 4.5, 4.0, 5.0)
    model = next(r for r in rows if r["row"] == MODEL_ROW and r["metric_id"] == "bmt.dut_service.p50")
    assert (model["unit"], model["value"], model["metric_name"]) == ("ns", 4_500_000, None)
    assert heads[2]["value"] == 0.0 and heads[2]["unit"] == "ratio"
    assert all(r["metric_name"] is None for r in rows[4:])
    printed = format_headlines(rows)
    text = " ".join(printed.split())
    assert "Metric Value Unit Valid runs Min \u2013 Max" in text and "dataset -" in text  # no dataset given
    assert "DUT service time (median) 4.500 ms 3/3 4.000 \u2013 5.000" in text
    assert "COCO box AP" in text and "bmt." not in text
    lines = printed.splitlines()
    assert lines[0].startswith("\u250c\u2500 ") and lines[-1].startswith("\u2514\u2500")
    assert all(line.startswith("\u2502") for line in lines[1:-1])
    value_ends = {line.index(" ms") for line in lines if " ms " in line}
    assert len(value_ends) == 1  # values right-aligned in one column


def test_a_missing_headline_metric_is_shown_as_not_in_evaluation():
    rows = table_rows([_task_runs("cityscapes-semantic-segmentation-19class",
                                  {"bmt.dut_service.p50": [1_000_000]}, {"bmt.dut_service.p50": "ns"})])
    heads = {r["metric_id"]: r for r in rows if r["row"] == HEADLINE_ROW}
    assert len(heads) == 4 and heads["bmt.dut_service.p50"]["value"] == 1.0
    assert heads["quality.segmentation.conditional.miou"]["status"] == "not_in_evaluation"
    assert "Cityscapes mIoU (19 classes) not_in_evaluation 0/0" in " ".join(format_headlines(rows).split())


def test_llm_headlines_group_speed_and_mmlu_per_device():
    speed = _task_runs("llm-speed", {"llm.S3.ttft_ns.p50": [40_000_000, 42_000_000],
                                     "llm.S3.tpot_ns.p50": [11_000_000, 11_000_000],
                                     "llm.S3.e2e_ns.p50": [5_700_000_000, 5_800_000_000]},
                       {"llm.S3.ttft_ns.p50": "ns", "llm.S3.tpot_ns.p50": "ns", "llm.S3.e2e_ns.p50": "ns"}, "x.speed")
    mmlu = _task_runs("llm-mmlu", {"llm.mmlu.accuracy": [0.644]}, {}, "x.mmlu")
    rows = table_rows([speed, mmlu])
    heads = [r for r in rows if r["row"] == HEADLINE_ROW]
    assert [h["metric_id"] for h in heads] == ["llm.S3.ttft_ns.p50", "llm.S3.tpot_ns.p50", "llm.S3.e2e_ns.p50",
                                               "llm.mmlu.accuracy"]
    assert heads[0]["value"] == 41.0 and heads[3]["value"] == 0.644
    text = " ".join(format_headlines(rows).split())
    assert text.count("\u250c\u2500 ") == 1 and "experiments x.speed, x.mmlu" in text
    assert "MMLU accuracy (1,000 items, zero-shot) 0.6440 ratio 1/1" in text
    assert "S3 = speed workload with 512 input and 512 output tokens" in text


def test_device_names_come_from_the_recorded_identity():
    from sdv_bmt.cli.table import _device_name

    def manifest(identity, kind="actual", device="free text GPU-serial"):
        return {"execution_kind": kind, "dut": {"dut_config_id": "npu-card-chip0-single",
                                               "descriptor": {"device": device, "device_identity": identity}}}

    assert _device_name(manifest({"nvidia.product_name": "NVIDIA X"})) == "NVIDIA X"
    assert _device_name(manifest({"vendor": "Vendor", "accelerator_family": "NPU"})) == "Vendor NPU (npu-card-chip0-single)"
    assert _device_name(manifest({}, kind="mock", device="fake dut")) == "fake dut"
    assert _device_name(manifest({})) is None  # an actual run's free text is never used (it can carry a serial)
