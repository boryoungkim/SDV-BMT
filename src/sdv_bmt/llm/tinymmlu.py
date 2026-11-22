"""tinyMMLU (100 items) with the evaluation of the lm-evaluation-harness task ``tinyMMLU``.

* Source: ``tinyBenchmarks/tinyMMLU`` config ``all`` split ``test`` (100 items) at a pinned dataset revision.
  The repository keeps only item IDs (``test/<row>``), subjects and per-item SHA-256 of the canonical item JSON,
  never the text.  Prompt token IDs are derived from the text, so they stay in local evidence.
* Prompt: the item's ``input_formatted`` text unchanged (the Open LLM Leaderboard MMLU prompt with its in-context
  examples already inside; ``num_fewshot: 0``).  No chat template.
* Scoring (harness ``output_type: multiple_choice``): for each choice ``" A" .. " D"`` (target delimiter ``" "``) the
  log-likelihood of the continuation given the context; the prediction is the first choice with the largest value.
  Tokenization follows the harness for a causal model (lm-evaluation-harness 0.4.13 ``_encode_pair``): trailing
  whitespace of the context moves into the continuation, ``context + continuation`` is encoded with the tokenizer's
  default special tokens (one BOS for Llama 3), and the continuation tokens are the part after the separately
  encoded context.  The harness metric is ``acc_norm`` (log-likelihood divided by the choice length in characters);
  the four choices have the same length, so it equals ``acc``.
* One runtime request per (item, choice): the context + continuation token IDs, one generated token, and the
  log-probability of every prompt token (``prompt_logprobs``).  The continuation log-likelihood is the sum over the
  continuation positions.
* Reported here: the raw accuracy over the 100 items (denominator always 100) and the per-item correctness vector in
  dataset order.  The harness's headline value, the tinyBenchmarks gp-IRT estimate of full-MMLU accuracy, is
  computed from that vector by ``tools/llm/tinymmlu_irt.py`` (numpy / scipy, outside the measurement code).
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from typing import Any, Optional, Sequence

TINYMMLU_DATASET = "tinyBenchmarks/tinyMMLU"
TINYMMLU_CONFIG = "all"
TINYMMLU_SPLIT = "test"
TINYMMLU_TOTAL = 100
CHOICES = ("A", "B", "C", "D")
TARGET_DELIMITER = " "
PROMPT_LOGPROBS = 1  # as the harness's vLLM backend requests them
EVALUATION_LABEL = "tinyMMLU 100 items (lm-evaluation-harness tinyMMLU: log-likelihood of ' A'..' D', acc_norm)"
SCORING_RULE_ID = "tinymmlu-harness-mc/1"
# tinyBenchmarks IRT parameters used for the gp-IRT estimate (tools/llm/tinymmlu_irt.py writes, the LLM table reads)
TB_COMMIT = "e9a8b1031b0340571beb6c9ca3a27891be09a8fd"
IRT_PKL_SHA256 = "c3b6e426dfe7b100fe6d0ee960398e10a8763254bcead3be80cc6bc15abca284"
IRT_RESULT_TOOL = "tools/llm/tinymmlu_irt.py"


def canonical_item(row: dict[str, Any]) -> dict[str, Any]:
    return {"subject": row["subject"], "question": row["question"], "choices": list(row["choices"]),
            "answer": int(row["answer"]), "input_formatted": row["input_formatted"]}


def item_sha256(row: dict[str, Any]) -> str:
    blob = json.dumps(canonical_item(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def encode_pair(tokenizer, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """(context token IDs, continuation token IDs) exactly as the harness builds them for a causal model."""
    if not context:
        raise ValueError("context must not be empty")
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = list(tokenizer(context + continuation).input_ids)
    ctx = list(tokenizer(context).input_ids)
    return ctx, whole[len(ctx):]


def item_requests(tokenizer, row: dict[str, Any]) -> list[dict[str, Any]]:
    """The four (choice) requests of one item: full token IDs and where the continuation starts."""
    out = []
    for letter in CHOICES:
        ctx, cont = encode_pair(tokenizer, row["input_formatted"], TARGET_DELIMITER + letter)
        if not cont:
            raise ValueError("empty continuation")
        out.append({"choice": letter, "input_token_ids": ctx + cont, "continuation_start": len(ctx)})
    return out


def score(items: Sequence[dict[str, Any]], outputs: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Score per-request outputs.

    ``items``: the item list in dataset order ({item_id, answer}); ``outputs``: item_id -> choice letter ->
    {status, continuation_logprobs}.  An item is scored only when all four choices completed with a log-probability
    for every continuation token; otherwise it is ``failed`` (some request ran) or ``not_run``.
    """
    per_item = []
    c = {"correct": 0, "incorrect": 0, "failed": 0, "not_run": 0}
    vector: list[Optional[int]] = []
    for it in items:
        o = outputs.get(it["item_id"]) or {}
        ll: dict[str, Optional[float]] = {}
        for letter in CHOICES:
            r = o.get(letter)
            lps = (r or {}).get("continuation_logprobs")
            ok = r is not None and r.get("status") == "completed" and lps and all(v is not None for v in lps)
            ll[letter] = float(sum(lps)) if ok else None
        if not o:
            outcome, pred = "not_run", None
        elif any(v is None for v in ll.values()):
            outcome, pred = "failed", None
        else:
            pred = max(CHOICES, key=lambda k: (ll[k], -CHOICES.index(k)))  # first maximum, as numpy argmax
            outcome = "correct" if CHOICES.index(pred) == int(it["answer"]) else "incorrect"
        c[outcome] += 1
        vector.append(None if pred is None else int(outcome == "correct"))
        per_item.append({"item_id": it["item_id"], "answer": CHOICES[int(it["answer"])], "predicted": pred,
                         "loglikelihood": ll, "outcome": outcome})
    total = len(items)
    scored = c["correct"] + c["incorrect"]
    return {
        "label": EVALUATION_LABEL,
        "scoring_rule": SCORING_RULE_ID,
        "total": total,
        "counts": c,
        "executed": scored,
        "coverage": float(Fraction(scored, total)) if total else None,
        "complete": total > 0 and scored == total,
        "accuracy": float(Fraction(c["correct"], total)) if total else None,
        "correct_vector": vector,
        "per_item": per_item,
    }


def compare(reference: dict[str, Any], other: dict[str, Any], threshold: float = 0.01) -> dict[str, Any]:
    """Per-item comparison of two ``score`` results on the same item list (reference = GPU)."""
    ref = {p["item_id"]: p for p in reference["per_item"]}
    oth = {p["item_id"]: p for p in other["per_item"]}
    if list(ref) != list(oth):
        raise ValueError("the two results do not cover the same item list")
    n = {"ref_correct_other_wrong": 0, "ref_wrong_other_correct": 0, "prediction_mismatch": 0,
         "ref_not_scored": 0, "other_not_scored": 0}
    max_abs = 0.0
    for k, a in ref.items():
        b = oth[k]
        if a["outcome"] == "correct" and b["outcome"] != "correct":
            n["ref_correct_other_wrong"] += 1
        if a["outcome"] != "correct" and b["outcome"] == "correct":
            n["ref_wrong_other_correct"] += 1
        if a["predicted"] is not None and b["predicted"] is not None and a["predicted"] != b["predicted"]:
            n["prediction_mismatch"] += 1
        n["ref_not_scored"] += a["predicted"] is None
        n["other_not_scored"] += b["predicted"] is None
        for letter in CHOICES:
            x, y = a["loglikelihood"].get(letter), b["loglikelihood"].get(letter)
            if x is not None and y is not None:
                max_abs = max(max_abs, abs(x - y))
    complete = reference["complete"] and other["complete"]
    diff = (other["accuracy"] - reference["accuracy"]) if complete else None
    return {
        "label": EVALUATION_LABEL,
        "criterion": f"|accuracy(other) - accuracy(reference)| <= {threshold} on this fixed item set "
                     "(total-quality difference only; not numerical equivalence or per-item identity)",
        "complete": complete,
        "reference_accuracy": reference["accuracy"],
        "other_accuracy": other["accuracy"],
        "accuracy_difference": diff,
        "verdict": ("not_judged_incomplete" if not complete else
                    "pass" if abs(Fraction(other["counts"]["correct"] - reference["counts"]["correct"],
                                           reference["total"])) <= Fraction(str(threshold)) else "fail"),
        "max_abs_loglikelihood_difference": max_abs,
        "counts": n,
    }
