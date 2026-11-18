"""LLM path (Python plane) with the fake backend: workload files, MMLU rules, bundle, evaluation."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from sdv_bmt.llm import mmlu
from sdv_bmt.llm.evaluate import LlmEvaluationError, evaluate, latest_metrics
from sdv_bmt.llm.runner import run_once
from sdv_bmt.llm.workload import (BOS_TOKEN_ID, REGULAR_VOCAB, SPEED_WORKLOADS, file_sha256, read_jsonl,
                                  speed_requests, token_ids_sha256)

REPO = Path(__file__).resolve().parents[2]
SPEED_FILE = REPO / "configs" / "llm" / "speed-s1-s5.tokens.jsonl"


def test_speed_requests_shape():
    reqs = speed_requests()
    assert len(reqs) == 40
    for w in SPEED_WORKLOADS:
        rs = [r for r in reqs if r["workload"] == w.name]
        assert [r["phase"] for r in rs] == ["warmup"] * 3 + ["measure"] * 5
        for r in rs:
            ids = r["input_token_ids"]
            assert len(ids) == w.input_tokens and ids[0] == BOS_TOKEN_ID
            assert all(0 <= t < REGULAR_VOCAB for t in ids[1:])
            assert r["generation"] == {"max_tokens": w.output_tokens, "min_tokens": w.output_tokens,
                                       "ignore_eos": True, "stop_token_ids": []}
        assert len({tuple(r["input_token_ids"]) for r in rs}) == 8


def test_speed_file_matches_generator_and_configs():
    rows = list(read_jsonl(SPEED_FILE))
    gen = speed_requests()
    assert rows == json.loads(json.dumps(gen))
    assert all(r["input_token_ids_sha256"] == token_ids_sha256(r["input_token_ids"]) for r in rows)
    sha = file_sha256(SPEED_FILE)
    for cfg in (REPO / "configs" / "llm" / "experiments").glob("*.speed.yaml"):
        assert f"sha256: {sha}" in cfg.read_text(), cfg.name


@pytest.mark.parametrize("text,expected", [
    ("B", "B"), (" C.", "C"), ("D) because", "D"), ("A\n", "A"), ("The answer is C.", "C"),
    ("the answer is (b)", "B"), ("Apple", None), ("", None), (None, None), ("E", None), ("I think so", None),
])
def test_extract_answer(text, expected):
    assert mmlu.extract_answer(text) == expected


def test_selection_is_proportional_and_deterministic():
    subjects = ["a"] * 500 + ["b"] * 300 + ["c"] * 199 + ["d"] * 1
    sel = mmlu.select_items(subjects * 2)  # 2,000 rows
    assert len(sel) == 1000 and len(set(sel)) == 1000
    assert sel == mmlu.select_items(subjects * 2)
    alloc = mmlu.largest_remainder({"a": 1000, "b": 600, "c": 398, "d": 2}, 1000)
    assert alloc == {"a": 500, "b": 300, "c": 199, "d": 1}


def test_score_and_compare_counts():
    items = [{"item_id": f"i{k}", "answer": k % 4} for k in range(4)]
    a = mmlu.score(items, {"i0": {"status": "completed", "text": "A"}, "i1": {"status": "completed", "text": "B"},
                           "i2": {"status": "completed", "text": "C"}, "i3": {"status": "completed", "text": "D"}})
    b = mmlu.score(items, {"i0": {"status": "completed", "text": "A"}, "i1": {"status": "completed", "text": "C"},
                           "i2": {"status": "completed", "text": "???"}, "i3": {"status": "error", "text": None}})
    assert a["accuracy"] == 1.0 and a["complete"]
    assert b["counts"] == {"correct": 1, "incorrect_parsable": 1, "unparsable": 1, "failed": 1, "not_run": 0}
    assert b["accuracy"] == 0.25 and not b["complete"] and b["coverage"] == 0.75
    c = mmlu.compare(a, b)
    assert c["verdict"] == "not_judged_incomplete" and c["accuracy_difference"] is None
    assert c["counts"]["ref_correct_other_wrong"] == 3 and c["counts"]["answer_mismatch_both_parsable"] == 1
    assert c["counts"]["other_unparsable"] == 1 and c["counts"]["other_not_executed"] == 1
    assert mmlu.compare(a, a)["verdict"] == "pass"


def _fake_cfg(tmp: Path, name: str, options: dict, workload: str) -> Path:
    p = tmp / f"{name}.yaml"
    p.write_text(f"schema: sdv-bmt-llm-run/1\nexperiment_id: {name}\nexecution_kind: mock\n"
                 f"model: {{name: fake, hf_id: none, revision: none, tokenizer_path: none}}\n"
                 f"backend: {{kind: fake, options: {json.dumps(options)}}}\n{workload}")
    return p


def _speed_workload() -> str:
    return f"workload: {{kind: speed, file: {SPEED_FILE}, sha256: {file_sha256(SPEED_FILE)}}}\n"


def test_fake_speed_bundle_evaluates(tmp_path):
    run = run_once(_fake_cfg(tmp_path, "fs", {}, _speed_workload()), tmp_path / "out")
    m = json.loads((run / "manifest.json").read_text())
    assert m["status"]["run_status"] == "completed" and m["execution_kind"] == "mock"
    evaluate(run)
    r = latest_metrics(run)
    assert r["catalog_version"] == "sdv-bmt-llm-v1"
    assert r["streaming"]["unit"] == "token"
    assert r["controls"]["prefix_caching"]["verdict"] == "confirmed_off"
    assert r["controls"]["concurrency"]["status"] == "pass"
    for w in SPEED_WORKLOADS:
        v = r["speed"][w.name]
        assert v["counts"] == {"measured": 5, "fixed_length_ok": 5, "length_mismatch": 0, "error": 0}
        assert v["ttft_ns"]["n"] == 5 and v["ttft_ns"]["p99"]["status"] == "insufficient_samples"
    # tampering with a raw file is refused
    with open(run / "requests.jsonl", "a") as fh:
        fh.write("\n")
    with pytest.raises(LlmEvaluationError):
        evaluate(run)


def test_length_mismatch_and_chunks_are_separated(tmp_path):
    run = run_once(_fake_cfg(tmp_path, "fm", {"ignore_eos_broken": True, "tokens_per_item": 2}, _speed_workload()),
                   tmp_path / "out")
    evaluate(run)
    r = latest_metrics(run)
    assert r["streaming"]["unit"] == "chunk" and not r["streaming"]["per_token_arrival_observable"]
    for w in SPEED_WORKLOADS:
        v = r["speed"][w.name]
        assert v["counts"]["length_mismatch"] == 5 and v["counts"]["fixed_length_ok"] == 0
        assert v["ttft_ns"]["n"] == 0


def test_fake_mmlu_bundle(tmp_path):
    rows = []
    for k in range(6):
        ids = [BOS_TOKEN_ID, 10 + k, 20 + k]
        rows.append({"item_id": f"test/{k}", "subject": "x", "answer": k % 4, "item_sha256": f"{k:064x}",
                     "input_token_ids": ids, "input_token_ids_sha256": token_ids_sha256(ids)})
    pf = tmp_path / "p.jsonl"
    pf.write_text("".join(json.dumps(r) + "\n" for r in rows))
    items = tmp_path / "items.json"
    items.write_text(json.dumps({"items": [{"item_id": r["item_id"], "subject": "x", "item_sha256": r["item_sha256"]}
                                           for r in rows]}))
    wl = f"workload: {{kind: mmlu, file: {pf}, sha256: {file_sha256(pf)}, items_file: {items}, warmup: 2}}\n"
    run = run_once(_fake_cfg(tmp_path, "fq", {"answer_letters": True}, wl), tmp_path / "out")
    reqs = list(read_jsonl(run / "requests.jsonl"))
    assert [r["phase"] for r in reqs] == ["warmup"] * 2 + ["measure"] * 6
    assert all(r["text"][0] in "ABCD" for r in reqs)
    evaluate(run)
    q = latest_metrics(run)["quality"]
    assert q["total"] == 6 and q["complete"] and q["coverage"] == 1.0
    assert sum(q["counts"].values()) == 6 and q["counts"]["unparsable"] == 0
    other = run_once(_fake_cfg(tmp_path, "fq", {"answer_letters": True}, wl), tmp_path / "out")
    evaluate(other)
    cmp_ = mmlu.compare(q, latest_metrics(other)["quality"])
    assert cmp_["verdict"] == "pass" and cmp_["accuracy_difference"] == 0.0


def test_request_times_use_observed_first_item():
    from sdv_bmt.llm.evaluate import request_times

    base = {"status": "completed", "t_done_ns": 1_000, "arrival_ns": [100, 400, 1_000], "arrival_tokens": [1, 1, 1]}
    t = request_times(base)
    assert t["ttft_ns"] == 100 and t["e2e_ns"] == 1_000 and t["n_out"] == 3
    assert t["tpot_ns"] == 450.0 and t["decode_tps"] == pytest.approx(2 / 900 * 1e9)
    chunked = dict(base, arrival_tokens=[2, 1, 1])  # first item carries 2 tokens: 2 tokens after t_first
    c = request_times(chunked)
    assert c["n_first"] == 2 and c["chunked"] and c["tpot_ns"] == 450.0
    assert request_times(dict(base, status="error")) is None
    single = request_times(dict(base, arrival_ns=[100], arrival_tokens=[1], t_done_ns=100))
    assert "tpot_ns" not in single and single["n_out"] == 1


# ---------------------------------------------------------------- metric table (python -m sdv_bmt.llm table)


def _fake_mmlu_cfg(tmp_path: Path) -> Path:
    rows = []
    for k in range(4):
        ids = [BOS_TOKEN_ID, 10 + k, 20 + k]
        rows.append({"item_id": f"test/{k}", "subject": "x", "answer": k % 4, "item_sha256": f"{k:064x}",
                     "input_token_ids": ids, "input_token_ids_sha256": token_ids_sha256(ids)})
    pf = tmp_path / "p.jsonl"
    pf.write_text("".join(json.dumps(r) + "\n" for r in rows))
    items = tmp_path / "items.json"
    items.write_text(json.dumps({"items": [{"item_id": r["item_id"], "subject": "x", "item_sha256": r["item_sha256"]}
                                           for r in rows]}))
    wl = f"workload: {{kind: mmlu, file: {pf}, sha256: {file_sha256(pf)}, items_file: {items}, warmup: 1}}\n"
    return _fake_cfg(tmp_path, "tq", {"answer_letters": True}, wl)


@pytest.fixture
def llm_experiments(tmp_path) -> tuple[Path, Path]:
    speed_cfg, out = _fake_cfg(tmp_path, "ts", {}, _speed_workload()), tmp_path / "out"
    for _ in range(2):
        evaluate(run_once(speed_cfg, out))
    evaluate(run_once(_fake_mmlu_cfg(tmp_path), out))
    return out / "ts", out / "tq"


def test_llm_metric_table_copies_the_evaluations(llm_experiments):
    from sdv_bmt.llm.table import build_table
    from sdv_bmt.metrics.table import HEADLINE_ROW, MODEL_ROW, RUN_ROW

    speed, quality = llm_experiments
    rows = build_table([speed, quality])
    kinds = [r["row"] for r in rows]
    assert kinds == [HEADLINE_ROW] * 4 + [MODEL_ROW] * kinds.count(MODEL_ROW) + [RUN_ROW] * kinds.count(RUN_ROW)
    heads = {r["metric_id"]: r for r in rows[:4]}
    assert list(heads) == ["llm.S3.ttft_ns.p50", "llm.S3.tpot_ns.p50", "llm.S3.e2e_ns.p50", "llm.mmlu.accuracy"]
    by_model = {(r["experiment_id"], r["metric_id"]): r for r in rows if r["row"] == MODEL_ROW}
    assert heads["llm.S3.ttft_ns.p50"]["value"] == by_model[("ts", "llm.S3.ttft_ns.p50")]["value"] / 1_000_000
    assert heads["llm.S3.ttft_ns.p50"]["unit"] == "ms" and heads["llm.S3.ttft_ns.p50"]["metric_name"]
    assert heads["llm.mmlu.accuracy"]["value"] == by_model[("tq", "llm.mmlu.accuracy")]["value"]
    runs = sorted(p for p in speed.glob("run-*"))
    values = []
    for run_dir in runs:
        m = latest_metrics(run_dir)
        mine = {r["metric_id"]: r for r in rows if r["row"] == RUN_ROW and r["run_id"] == run_dir.name}
        assert mine["llm.S1.ttft_ns.p50"]["value"] == m["speed"]["S1"]["ttft_ns"]["median_p50"]["value"]
        assert mine["llm.S3.decode_tps.min"]["value"] == m["speed"]["S3"]["decode_tps"]["min"]
        assert mine["llm.S5.count.fixed_length_ok"]["value"] == m["speed"]["S5"]["counts"]["fixed_length_ok"]
        assert (mine["llm.S2.e2e_ns.p99"]["status"], mine["llm.S2.e2e_ns.p99"]["value"]) == \
            ("insufficient_samples", None)
        assert mine["llm.control.prefix_caching"]["value"] == "confirmed_off"
        assert {r["eval_id"] for r in mine.values()} == {m["eval_id"]}
        assert {r["task"] for r in mine.values()} == {"llm-speed"}
        values.append(m["speed"]["S1"]["ttft_ns"]["median_p50"]["value"])
    model = {r["metric_id"]: r for r in rows if r["row"] == MODEL_ROW and r["experiment_id"] == "ts"}
    entry = model["llm.S1.ttft_ns.p50"]
    assert (entry["value"], entry["min"], entry["max"], entry["n_valid"]) == \
        ((values[0] + values[1]) / 2, min(values), max(values), 2)
    assert model["llm.S2.e2e_ns.p99"]["status"] == "insufficient_samples"
    q = latest_metrics(next(quality.glob("run-*")))["quality"]
    mmlu_rows = {r["metric_id"]: r for r in rows if r["row"] == RUN_ROW and r["experiment_id"] == "tq"}
    assert mmlu_rows["llm.mmlu.accuracy"]["value"] == q["accuracy"]
    assert mmlu_rows["llm.mmlu.count.unparsable"]["value"] == q["counts"]["unparsable"]
    assert {r["task"] for r in mmlu_rows.values()} == {"llm-mmlu"}


def test_llm_metric_table_command_and_refusals(llm_experiments, tmp_path, capsys):
    from sdv_bmt.llm.cli import main
    from sdv_bmt.llm.table import LlmReportError, build_table

    speed, quality = llm_experiments
    out = tmp_path / "metrics.csv"
    assert main(["table", str(speed), str(quality), "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["experiments"] == 2 and "ts, tq" in captured.err
    assert "MMLU accuracy" in captured.err and "llm.S3" not in captured.err
    assert f"see the CSV file:\n  {out.resolve()}\n" in captured.err
    assert out.read_text().startswith("row,task,dataset,model,device,experiment_id,run_id,eval_id,catalog_version,"
                                      "metric_name")
    datasets = {r["experiment_id"]: r["dataset"] for r in csv.DictReader(out.open())}
    assert datasets["ts"] == "speed-s1-s5 (synthetic token IDs)" and re.fullmatch(r"items \(\d+ items\)", datasets["tq"])
    assert "dataset      speed-s1-s5 (synthetic token IDs); items (" in captured.err
    assert main(["table", str(speed), "--out", str(out)]) == 2  # never overwritten
    with pytest.raises(LlmReportError, match="experiment given twice"):
        build_table([speed, speed])
    with pytest.raises(LlmReportError, match="not an LLM experiment"):
        build_table([tmp_path])
    run_dir = sorted(speed.glob("run-*"))[0]
    metrics = next(run_dir.glob("evaluations/*/metrics.json"))
    original = metrics.read_bytes()
    metrics.write_text(metrics.read_text().replace('"confirmed_off"', '"unverified"', 1))
    with pytest.raises(LlmReportError, match="metrics.json differs from its eval_manifest.json record"):
        build_table([speed])
    metrics.write_bytes(original)
    with open(run_dir / "requests.jsonl", "a") as fh:
        fh.write("\n")
    with pytest.raises(LlmReportError, match="changed or missing"):
        build_table([speed])
    evaluate(run_once(tmp_path / "tq.yaml", tmp_path / "out"))  # a further evaluated run is fine
    next_run = run_once(tmp_path / "tq.yaml", tmp_path / "out")  # ... a run without evaluation is not
    with pytest.raises(LlmReportError, match="no evaluation"):
        build_table([next_run.parent])
