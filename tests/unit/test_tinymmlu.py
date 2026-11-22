"""tinyMMLU workload with the fake backend: harness tokenization rule, scoring, bundle, evaluation, metric table."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sdv_bmt.llm import tinymmlu
from sdv_bmt.llm.backends import actual_token_logprobs
from sdv_bmt.llm.evaluate import evaluate, latest_metrics
from sdv_bmt.llm.runner import TASK_GENERATION, run_once
from sdv_bmt.llm.table import LlmReportError, build_table
from sdv_bmt.llm.workload import BOS_TOKEN_ID, file_sha256, read_jsonl, token_ids_sha256
from sdv_bmt.metrics.table import HEADLINE_ROW

REPO = Path(__file__).resolve().parents[2]


class _Enc:
    def __init__(self, ids):
        self.input_ids = ids


class WordTokenizer:
    """Stand-in tokenizer: BOS, then one token per space-separated piece (a leading space stays on the piece)."""

    def __call__(self, text):
        pieces, cur = [], ""
        for ch in text:
            if ch == " " and cur:
                pieces.append(cur)
                cur = ""
            cur += ch
        if cur:
            pieces.append(cur)
        return _Enc([BOS_TOKEN_ID] + [1000 + sum(map(ord, p)) % 5000 for p in pieces])


def test_encode_pair_follows_the_harness_rule():
    tok = WordTokenizer()
    ctx, cont = tinymmlu.encode_pair(tok, "Q? Answer:", " B")
    assert ctx == tok("Q? Answer:").input_ids and ctx[0] == BOS_TOKEN_ID
    assert ctx + cont == tok("Q? Answer: B").input_ids and len(cont) == 1
    # trailing whitespace of the context moves into the continuation
    ctx2, cont2 = tinymmlu.encode_pair(tok, "Q? Answer: ", "B")
    assert (ctx2, cont2) == (ctx, cont)
    with pytest.raises(ValueError):
        tinymmlu.encode_pair(tok, "", " A")
    reqs = tinymmlu.item_requests(tok, {"input_formatted": "Q? Answer:"})
    assert [r["choice"] for r in reqs] == list("ABCD")
    assert all(r["continuation_start"] == len(ctx) for r in reqs)


def test_score_takes_the_first_maximum_and_separates_missing():
    items = [{"item_id": f"i{k}", "answer": k} for k in range(4)]
    def out(lls, status="completed"):
        return {c: {"status": status, "continuation_logprobs": [v]} for c, v in zip("ABCD", lls)}
    outputs = {"i0": out([-1.0, -2.0, -3.0, -4.0]),           # A correct
               "i1": out([-1.0, -1.0, -3.0, -4.0]),           # tie A/B -> A (first), answer B -> incorrect
               "i2": {**out([-3.0, -2.0, -1.0, -4.0]), "D": {"status": "error", "continuation_logprobs": None}}}
    q = tinymmlu.score(items, outputs)
    assert q["counts"] == {"correct": 1, "incorrect": 1, "failed": 1, "not_run": 1}
    assert q["correct_vector"] == [1, 0, None, None]
    assert q["accuracy"] == 0.25 and q["coverage"] == 0.5 and not q["complete"]
    assert q["per_item"][1]["predicted"] == "A"
    full = tinymmlu.score(items[:1], {"i0": out([-1.0, -2.0, -3.0, -4.0])})
    assert full["complete"] and tinymmlu.compare(full, full)["verdict"] == "pass"


def test_actual_token_logprobs_reads_both_runtime_shapes():
    class LP:
        def __init__(self, v):
            self.logprob = v
    ids = [BOS_TOKEN_ID, 5, 6, 7]
    vllm_shape = [None, {5: LP(-0.1)}, {6: LP(-0.2), 9: LP(-0.05)}, {7: LP(-0.3)}]
    native_shape = [None, [(5, LP(-0.1))], [(9, LP(-0.05)), (6, LP(-0.2))], [(8, LP(-1.0))]]
    assert actual_token_logprobs(vllm_shape, ids, 2) == [-0.2, -0.3]
    assert actual_token_logprobs(native_shape, ids, 2) == [-0.2, None]


def _fake_tinymmlu(tmp: Path, n_items: int = 3) -> tuple[Path, list[dict]]:
    tok = WordTokenizer()
    rows = []
    for k in range(n_items):
        sha = f"{k:064x}"
        for r in tinymmlu.item_requests(tok, {"input_formatted": f"question {k} Answer:"}):
            ids = r["input_token_ids"]
            rows.append({"item_id": f"test/{k}", "request_key": r["choice"], "subject": "x",
                         "answer": "ABCD"[k % 4], "item_sha256": sha, "input_token_ids": ids,
                         "input_token_ids_sha256": token_ids_sha256(ids), "score_from": r["continuation_start"]})
    pf = tmp / "t.jsonl"
    pf.write_text("".join(json.dumps(r) + "\n" for r in rows))
    items = tmp / "items.json"
    items.write_text(json.dumps({"items": [{"item_id": f"test/{k}", "subject": "x", "item_sha256": f"{k:064x}"}
                                           for k in range(n_items)]}))
    cfg = tmp / "tm.yaml"
    cfg.write_text("schema: sdv-bmt-llm-run/1\nexperiment_id: tm\nexecution_kind: mock\n"
                   "model: {name: fake, hf_id: none, revision: none, tokenizer_path: none}\n"
                   "backend: {kind: fake, options: {}}\n"
                   f"workload: {{kind: tinymmlu, file: {pf}, sha256: {file_sha256(pf)}, items_file: {items}, "
                   "warmup: 2}\n")
    return cfg, rows


def test_fake_tinymmlu_bundle_evaluates_and_tabulates(tmp_path):
    cfg, rows = _fake_tinymmlu(tmp_path)
    run = run_once(cfg, tmp_path / "out")
    reqs = list(read_jsonl(run / "requests.jsonl"))
    assert [r["phase"] for r in reqs] == ["warmup"] * 2 + ["measure"] * 12
    assert all(r["generation"] == {**TASK_GENERATION["tinymmlu"], "score_from": rows[0]["score_from"]} for r in reqs)
    assert all(len(r["prompt_token_logprobs"]) == r["input_len"] - r["generation"]["score_from"] for r in reqs)
    assert all("text" not in r for r in reqs)
    assert [json.loads(x)["answer"] for x in (run / "quality_reference.jsonl").read_text().splitlines()] == list("ABC")
    evaluate(run)
    m = latest_metrics(run)
    q = m["quality"]
    assert q["scoring_rule"] == tinymmlu.SCORING_RULE_ID and q["total"] == 3 and q["complete"]
    assert len(q["correct_vector"]) == 3 and set(q["correct_vector"]) <= {0, 1}
    assert m["task_request_times"]["ttft_ns"]["n"] == 12
    rows_t = build_table([tmp_path / "out" / "tm"])
    head = [r for r in rows_t if r["row"] == HEADLINE_ROW]
    assert [r["metric_id"] for r in head] == ["llm.tinymmlu.accuracy", "llm.tinymmlu.gpirt"]
    assert head[0]["value"] == q["accuracy"] and head[1]["value"] is None
    assert {r["task"] for r in rows_t} == {"llm-tinymmlu"}
    assert any(r["metric_id"] == "llm.tinymmlu.count.incorrect" for r in rows_t)
    gp = [r for r in rows_t if r["row"] == "run" and r["metric_id"] == "llm.tinymmlu.gpirt"]
    assert gp[0]["status"] == "not_computed"

    # with a gp-IRT result file of tools/llm/tinymmlu_irt.py the value is copied after matching run and evaluation
    def irt_file(name, eval_id, sha=tinymmlu.IRT_PKL_SHA256):
        f = tmp_path / name
        f.write_text(json.dumps({"tool": tinymmlu.IRT_RESULT_TOOL, "irt_parameters": {"file_sha256": sha},
                                 "runs": [{"run_id": run.name, "eval_id": eval_id, "estimate": {"gpirt": 0.5}}]}))
        return f
    rows_g = build_table([tmp_path / "out" / "tm"], [irt_file("irt.json", m["eval_id"])])
    head = {r["metric_id"]: r for r in rows_g if r["row"] == HEADLINE_ROW}
    assert head["llm.tinymmlu.gpirt"]["value"] == 0.5 and head["llm.tinymmlu.gpirt"]["status"] == "valid"
    with pytest.raises(LlmReportError, match="computed from evaluation"):
        build_table([tmp_path / "out" / "tm"], [irt_file("old.json", "eval-other")])
    with pytest.raises(LlmReportError, match="IRT parameters"):
        build_table([tmp_path / "out" / "tm"], [irt_file("other.json", m["eval_id"], sha="0" * 64)])


def test_prompt_file_must_follow_the_item_list(tmp_path):
    cfg, rows = _fake_tinymmlu(tmp_path)
    items = tmp_path / "items.json"
    data = json.loads(items.read_text())
    data["items"].reverse()
    items.write_text(json.dumps(data))
    with pytest.raises(Exception, match="item list order"):
        run_once(cfg, tmp_path / "out")


def test_public_item_list_and_configs_agree():
    items = json.loads((REPO / "configs" / "llm" / "tinymmlu-100.items.json").read_text())
    assert items["dataset"] == tinymmlu.TINYMMLU_DATASET and len(items["items"]) == tinymmlu.TINYMMLU_TOTAL
    assert [it["item_id"] for it in items["items"]] == [f"test/{k}" for k in range(100)]
    text = json.dumps(items)
    assert "input_formatted\":" not in text and "question\":" not in text  # IDs and digests only
    cfgs = sorted((REPO / "configs" / "llm" / "experiments").glob("*.tinymmlu.yaml"))
    assert len(cfgs) == 6
    assert not list((REPO / "configs" / "llm" / "experiments").glob("*.mmlu.yaml"))
    for c in cfgs:
        t = c.read_text()
        assert "kind: tinymmlu" in t and "items_file: ../tinymmlu-100.items.json" in t, c.name
