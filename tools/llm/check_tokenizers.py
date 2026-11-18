"""Check that two tokenizers give token IDs the same meaning (e.g. meta-llama vs a furiosa-ai artifact).

Compares the full id->token vocabulary, added/special tokens, the chat template text, and re-encodes every
prompt of an MMLU prompt file (decoded with tokenizer A) with tokenizer B.

    <venv>/bin/python tools/llm/check_tokenizers.py --a DIR --b DIR --prompts FILE --out FILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    x = ap.parse_args()
    from transformers import AutoTokenizer

    ta, tb = AutoTokenizer.from_pretrained(x.a), AutoTokenizer.from_pretrained(x.b)
    va, vb = ta.get_vocab(), tb.get_vocab()
    inv_a = {i: t for t, i in va.items()}
    inv_b = {i: t for t, i in vb.items()}
    diff_ids = [i for i in set(inv_a) | set(inv_b) if inv_a.get(i) != inv_b.get(i)]

    def tmpl(d):
        p = Path(d) / "chat_template.jinja"
        if p.exists():
            return p.read_text()
        return json.loads((Path(d) / "tokenizer_config.json").read_text()).get("chat_template")

    ca, cb = tmpl(x.a), tmpl(x.b)
    n, same, first_diff = 0, 0, None
    for line in open(x.prompts, encoding="utf-8"):
        r = json.loads(line)
        text = ta.decode(r["input_token_ids"])
        ib = tb.encode(text, add_special_tokens=False)
        n += 1
        if ib == r["input_token_ids"]:
            same += 1
        elif first_diff is None:
            first_diff = r["item_id"]
    res = {
        "a": x.a, "b": x.b,
        "vocab_size": [len(va), len(vb)], "vocab_id_to_token_identical": not diff_ids,
        "vocab_differing_ids": sorted(diff_ids)[:20],
        "special": {"a": {"bos": ta.bos_token, "eos": ta.eos_token}, "b": {"bos": tb.bos_token, "eos": tb.eos_token}},
        "chat_template_identical": ca == cb,
        "chat_template_sha256": [hashlib.sha256((ca or "").encode()).hexdigest(),
                                 hashlib.sha256((cb or "").encode()).hexdigest()],
        "mmlu_prompts_reencoded_identical": f"{same}/{n}", "first_differing_item": first_diff,
    }
    Path(x.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
