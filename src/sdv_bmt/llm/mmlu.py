"""MMLU 1,000-item zero-shot generation-based evaluation: item selection, prompt, answer extraction, scoring.

* Source: ``cais/mmlu`` config ``all`` split ``test`` (14,042 items) at a pinned dataset revision.
* Selection: subject-proportional 1,000 items.  Per-subject counts by the largest-remainder method; inside a
  subject, items are ranked by SHA-256 of ``"42:<subject>:<row>"`` (seed 42) and the first n are taken.  The
  list order puts a subject-proportional 100-item subset first (the same rule with total 100), then the rest;
  both parts are in rank order.
* The public repository keeps only item IDs (``test/<row>``) and per-item SHA-256 of the canonical item JSON,
  never the question text.  Prompt token IDs are derived from the text, so they stay in local evidence.
* Prompt: one user message, the model's own chat template with the date fixed to 26 Jul 2024, generation
  prompt appended.  Greedy decoding, at most 16 new tokens, natural stop at the Llama 3 end tokens.
* Extraction (fixed, in order): (1) after stripping whitespace, the first character is A-D and the next
  character is absent or not a letter/digit; (2) a ``answer is X`` phrase; (3) otherwise unparsable.
* Scoring: denominator is always 1,000.  Unparsable outputs count as incorrect and are reported separately;
  items that did not run (error / not executed) are reported separately and make the evaluation incomplete.
"""

from __future__ import annotations

import hashlib
import json
import re
from fractions import Fraction
from typing import Any, Optional, Sequence

MMLU_DATASET = "cais/mmlu"
MMLU_CONFIG = "all"
MMLU_SPLIT = "test"
MMLU_SELECTION_SEED = 42
MMLU_TOTAL = 1000
MMLU_CHECK_SUBSET = 100
LETTERS = "ABCD"
DATE_STRING = "26 Jul 2024"
LLAMA3_STOP_TOKEN_IDS = (128001, 128008, 128009)
MAX_NEW_TOKENS = 16
INSTRUCTION = "Answer with only the letter (A, B, C, or D)."
EVALUATION_LABEL = "MMLU 1,000 items, zero-shot generation-based evaluation"
EXTRACTION_RULE_ID = "mmlu-letter-extract/1"

_ANSWER_IS = re.compile(r"answer is[\s:]*\(?([ABCD])(?![A-Za-z0-9])", re.IGNORECASE)


def canonical_item(row: dict[str, Any]) -> dict[str, Any]:
    return {"subject": row["subject"], "question": row["question"], "choices": list(row["choices"]),
            "answer": int(row["answer"])}


def item_sha256(row: dict[str, Any]) -> str:
    blob = json.dumps(canonical_item(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _rank_key(subject: str, row: int) -> str:
    return hashlib.sha256(f"{MMLU_SELECTION_SEED}:{subject}:{row}".encode()).hexdigest()


def largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    n = sum(counts.values())
    quotas = {s: Fraction(c * total, n) for s, c in counts.items()}
    alloc = {s: int(q) for s, q in quotas.items()}
    rest = total - sum(alloc.values())
    order = sorted(counts, key=lambda s: (-(quotas[s] - alloc[s]), s))
    for s in order[:rest]:
        alloc[s] += 1
    return alloc


def select_items(subjects: Sequence[str]) -> list[int]:
    """Row indices (into the pinned all/test split) of the 1,000 selected items in list order."""
    by_subject: dict[str, list[int]] = {}
    for row, s in enumerate(subjects):
        by_subject.setdefault(s, []).append(row)
    for s in by_subject:
        by_subject[s].sort(key=lambda r: _rank_key(s, r))
    counts = {s: len(v) for s, v in by_subject.items()}
    n_full = largest_remainder(counts, MMLU_TOTAL)
    n_head = largest_remainder(counts, MMLU_CHECK_SUBSET)
    head, tail = [], []
    for s in sorted(by_subject):
        k_head = min(n_head[s], n_full[s])
        head += by_subject[s][:k_head]
        tail += by_subject[s][k_head:n_full[s]]
    return head + tail


def subject_label(subject: str) -> str:
    return subject.replace("_", " ")


def user_message(row: dict[str, Any]) -> str:
    choices = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))
    return (f"The following is a multiple choice question about {subject_label(row['subject'])}.\n\n"
            f"{row['question']}\n{choices}\n\n{INSTRUCTION}")


def prompt_token_ids(tokenizer, row: dict[str, Any]) -> list[int]:
    """Token IDs of the chat-templated prompt (template text, then encoded without extra special tokens)."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": user_message(row)}], tokenize=False,
                                         add_generation_prompt=True, date_string=DATE_STRING)
    ids = tokenizer.encode(text, add_special_tokens=False)
    return list(ids)


def extract_answer(text: Optional[str]) -> Optional[str]:
    """Answer letter A-D, or None when the output is unparsable."""
    if text is None:
        return None
    s = text.strip()
    if s and s[0] in LETTERS and (len(s) == 1 or not s[1].isalnum()):
        return s[0]
    m = _ANSWER_IS.search(s)
    if m:
        return m.group(1).upper()
    return None


def score(items: Sequence[dict[str, Any]], outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Score per-item outputs.

    ``items``: the 1,000-item list ({item_id, answer}); ``outputs``: item_id -> {status, text}.
    Returns counts over the full list (denominator = len(items)) and per-item results.
    """
    per_item = []
    c = {"correct": 0, "incorrect_parsable": 0, "unparsable": 0, "failed": 0, "not_run": 0}
    for it in items:
        o = outputs.get(it["item_id"])
        if o is None:
            outcome, letter = "not_run", None
        elif o.get("status") != "completed":
            outcome, letter = "failed", None
        else:
            letter = extract_answer(o.get("text"))
            if letter is None:
                outcome = "unparsable"
            elif LETTERS.index(letter) == int(it["answer"]):
                outcome = "correct"
            else:
                outcome = "incorrect_parsable"
        c[outcome] += 1
        per_item.append({"item_id": it["item_id"], "answer": LETTERS[int(it["answer"])], "extracted": letter,
                         "outcome": outcome})
    total = len(items)
    executed = total - c["not_run"] - c["failed"]
    return {
        "label": EVALUATION_LABEL,
        "extraction_rule": EXTRACTION_RULE_ID,
        "total": total,
        "counts": c,
        "executed": executed,
        "coverage": float(Fraction(executed, total)) if total else None,
        "complete": total > 0 and executed == total,
        "accuracy": float(Fraction(c["correct"], total)) if total else None,
        "per_item": per_item,
    }


def compare(reference: dict[str, Any], other: dict[str, Any], threshold: float = 0.01) -> dict[str, Any]:
    """Per-item comparison of two ``score`` results on the same item list (reference = GPU)."""
    ref = {p["item_id"]: p for p in reference["per_item"]}
    oth = {p["item_id"]: p for p in other["per_item"]}
    if list(ref) != list(oth):
        raise ValueError("the two results do not cover the same item list")
    n = {"ref_correct_other_wrong": 0, "ref_wrong_other_correct": 0, "answer_mismatch_both_parsable": 0,
         "ref_unparsable": 0, "other_unparsable": 0, "ref_not_executed": 0, "other_not_executed": 0}
    for k, a in ref.items():
        b = oth[k]
        if a["outcome"] == "correct" and b["outcome"] != "correct":
            n["ref_correct_other_wrong"] += 1
        if a["outcome"] != "correct" and b["outcome"] == "correct":
            n["ref_wrong_other_correct"] += 1
        if a["extracted"] is not None and b["extracted"] is not None and a["extracted"] != b["extracted"]:
            n["answer_mismatch_both_parsable"] += 1
        n["ref_unparsable"] += a["outcome"] == "unparsable"
        n["other_unparsable"] += b["outcome"] == "unparsable"
        n["ref_not_executed"] += a["outcome"] in ("not_run", "failed")
        n["other_not_executed"] += b["outcome"] in ("not_run", "failed")
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
        "counts": n,
    }
