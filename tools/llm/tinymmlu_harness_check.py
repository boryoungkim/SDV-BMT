#!/usr/bin/env python3
"""Cross-check of the SDV-BMT tinyMMLU path against lm-evaluation-harness itself (offline tool, GPU).

For one model, with the harness's own vLLM backend (``lm_eval.models.vllm_causallms.VLLM``) on the same GPU:

1. token IDs: for every (item, choice) the harness's ``_encode_pair(input_formatted, " " + letter)`` must give the same
   context + continuation token IDs and the same continuation start as the SDV-BMT request file;
2. harness run: ``simple_evaluate(tasks=["tinyMMLU"], log_samples=True)``; the documents it read must be the pinned
   items (per-item SHA-256 of the public item list), and its per-choice log-likelihoods, predictions and reported
   value (IRT++) are compared with the evaluated SDV-BMT run of the same model.

The harness's tinyBenchmarks aggregation runs in a temporary working directory holding only the pinned
tinyBenchmarks.pkl (see tools/llm/tinymmlu_irt.py), so it uses the same IRT parameters.  The harness loads the
dataset ``tinyBenchmarks/tinyMMLU`` through ``datasets`` (network or the Hugging Face cache); step 2 checks that
the documents equal the pinned items.  Run with the lm-eval environment and ``PYTHONPATH=src``::

    CUDA_VISIBLE_DEVICES=0 $LMEVAL_PY tools/llm/tinymmlu_harness_check.py --model-path $SDV_BMT_MODELS/<m> \
        --prompts $SDV_BMT_DATASETS/tinymmlu-100/<m>.prompts.jsonl --items configs/llm/tinymmlu-100.items.json \
        --run <evaluated SDV-BMT GPU run dir> --pkl <dir>/tinyBenchmarks.pkl --out <file.json>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sdv_bmt.llm import tinymmlu  # noqa: E402
from sdv_bmt.llm.evaluate import latest_metrics, verify_bundle  # noqa: E402
from sdv_bmt.llm.workload import file_sha256, read_jsonl  # noqa: E402
from tinymmlu_irt import IRT_PKL_SHA256, estimate, sha256_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("--model-path", "--prompts", "--items", "--run", "--pkl", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument("--sdv-bmt-conditions", action="store_true",
                    help="harness with batch size 1 and prefix caching off, as SDV-BMT runs (default: harness defaults)")
    a = ap.parse_args()
    if sha256_file(Path(a.pkl)) != IRT_PKL_SHA256:
        print("error: --pkl is not the pinned tinyBenchmarks.pkl", file=sys.stderr)
        return 2

    import importlib.metadata as md

    import lm_eval
    from lm_eval.models.vllm_causallms import VLLM

    rows = list(read_jsonl(Path(a.prompts)))
    items = json.loads(Path(a.items).read_text(encoding="utf-8"))
    item_sha = {it["item_id"]: it["item_sha256"] for it in items["items"]}
    verify_bundle(Path(a.run))
    ours = latest_metrics(Path(a.run))["quality"]
    if ours.get("scoring_rule") != tinymmlu.SCORING_RULE_ID:
        print("error: --run is not an evaluated tinyMMLU run", file=sys.stderr)
        return 2

    extra = {"batch_size": 1, "enable_prefix_caching": False} if a.sdv_bmt_conditions else {}
    lm = VLLM(pretrained=a.model_path, dtype="bfloat16", max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_memory_utilization, seed=0, **extra)

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copyfile(a.pkl, Path(tmp) / "tinyBenchmarks.pkl")
        os.chdir(tmp)
        try:
            res = lm_eval.simple_evaluate(model=lm, tasks=["tinyMMLU"], log_samples=True,
                                          batch_size=1 if a.sdv_bmt_conditions else "auto")
        finally:
            os.chdir(cwd)
    samples = sorted(res["samples"]["tinyMMLU"], key=lambda s: s["doc_id"])
    reported = res["results"]["tinyMMLU"]

    # 1) documents = pinned items, and token IDs of every request
    doc_problems, token_problems = [], []
    by_req = {(r["item_id"], r["request_key"]): r for r in rows}
    for s in samples:
        item_id = f"test/{s['doc_id']}"
        if tinymmlu.item_sha256(s["doc"]) != item_sha.get(item_id):
            doc_problems.append(item_id)
            continue
        for letter in tinymmlu.CHOICES:
            ctx, cont = lm._encode_pair(s["doc"]["input_formatted"], tinymmlu.TARGET_DELIMITER + letter)
            r = by_req[(item_id, letter)]
            if ctx + cont != r["input_token_ids"] or len(ctx) != r["score_from"]:
                token_problems.append(f"{item_id}#{letter}")

    # 2) per-choice log-likelihoods and predictions
    ours_items = {p["item_id"]: p for p in ours["per_item"]}
    max_abs, mismatch, harness_vector = 0.0, [], []
    for s in samples:
        item_id = f"test/{s['doc_id']}"
        lls = [float(x[0]) for x in s["filtered_resps"]]
        harness_vector.append(float(s["acc_norm"]))
        o = ours_items[item_id]
        for letter, v in zip(tinymmlu.CHOICES, lls):
            max_abs = max(max_abs, abs(v - o["loglikelihood"][letter]))
        h_pred = tinymmlu.CHOICES[max(range(4), key=lambda i: (lls[i], -i))]
        if h_pred != o["predicted"]:
            mismatch.append({"item_id": item_id, "harness": h_pred, "sdv_bmt": o["predicted"],
                             "harness_ll": lls, "sdv_bmt_ll": [o["loglikelihood"][c] for c in tinymmlu.CHOICES]})
    ours_est = estimate(ours["correct_vector"], Path(a.pkl))
    harness_est = estimate(harness_vector, Path(a.pkl))
    out = {
        "tool": "tools/llm/tinymmlu_harness_check.py", "tool_sha256": sha256_file(Path(__file__)),
        "packages": {d: md.version(d) for d in ("lm_eval", "vllm", "torch", "transformers", "tinyBenchmarks")},
        "harness_conditions": "batch 1, prefix caching off" if a.sdv_bmt_conditions else "harness defaults",
        "model_path": a.model_path, "prompts_sha256": file_sha256(Path(a.prompts)), "run": a.run,
        "harness_documents": len(samples), "documents_not_matching_pinned_items": doc_problems,
        "requests_with_different_token_ids": token_problems,
        "harness_reported": {k: v for k, v in reported.items() if isinstance(v, (int, float, str))},
        "harness_accuracy_100": sum(harness_vector) / len(harness_vector),
        "harness_estimate_recomputed": harness_est,
        "sdv_bmt_accuracy_100": ours["accuracy"], "sdv_bmt_estimate": ours_est,
        "max_abs_loglikelihood_difference": max_abs,
        "prediction_mismatches": mismatch,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: out[k] for k in ("harness_documents", "harness_reported", "harness_accuracy_100",
                                          "sdv_bmt_accuracy_100", "sdv_bmt_estimate",
                                          "max_abs_loglikelihood_difference")}, indent=1))
    print(json.dumps({"doc_problems": len(doc_problems), "token_problems": len(token_problems),
                      "prediction_mismatches": len(mismatch)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
