"""Independent re-scoring of an MMLU LLM bundle (cross-check of sdv_bmt.llm.mmlu / evaluate).

Written separately from the evaluator: answer extraction with two regular expressions instead of character
tests, counting straight from requests.jsonl + quality_reference.jsonl.  Prints per-outcome counts and checks
them, item by item, against the bundle's latest evaluation.

    python3 tools/llm/rescore_mmlu.py RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import glob
import json
import re
import sys

FIRST = re.compile(r"\A\s*([ABCD])(?:[^A-Za-z0-9]|\Z)")
PHRASE = re.compile(r"answer is[\s:]*\(?([ABCDabcd])(?![A-Za-z0-9])", re.IGNORECASE)


def letter(text):
    if text is None:
        return None
    m = FIRST.match(text)
    if m:
        return m.group(1)
    m = PHRASE.search(text.strip())
    return m.group(1).upper() if m else None


def rescore(run_dir: str) -> dict:
    gold = {}
    with open(f"{run_dir}/quality_reference.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            gold[r["item_id"]] = r["answer"]
    out = {}
    with open(f"{run_dir}/requests.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if r["phase"] == "measure":
                out[r["item_id"]] = r
    per = {}
    for k, g in gold.items():
        r = out.get(k)
        if r is None:
            per[k] = ("not_run", None)
        elif r["status"] != "completed":
            per[k] = ("failed", None)
        else:
            a = letter(r.get("text"))
            per[k] = ("unparsable", None) if a is None else (("correct" if a == g else "incorrect_parsable"), a)
    counts = {}
    for o, _ in per.values():
        counts[o] = counts.get(o, 0) + 1
    ev = json.load(open(sorted(glob.glob(f"{run_dir}/evaluations/*/metrics.json"))[-1]))["quality"]
    mism = [p["item_id"] for p in ev["per_item"] if (p["outcome"], p["extracted"]) != per[p["item_id"]]]
    return {"run": run_dir, "counts": counts, "accuracy": counts.get("correct", 0) / len(gold),
            "evaluator_counts": {k: v for k, v in ev["counts"].items() if v}, "items": len(gold),
            "per_item_mismatches": len(mism), "first_mismatch": mism[:3], "identical": not mism}


if __name__ == "__main__":
    res = [rescore(d) for d in sys.argv[1:]]
    print(json.dumps(res, indent=1))
    sys.exit(0 if all(r["identical"] for r in res) else 1)
