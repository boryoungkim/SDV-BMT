"""GPU (vLLM, BF16) vs Hugging Face transformers (FP32, CPU) small check.

    hf      --model DIR --mmlu-prompts FILE --out hf.json        (CPU; FP32; first token top-5, greedy 32
                                                                  tokens for 16 fixed prompts; greedy <=16
                                                                  tokens for the first 100 MMLU items)
    vllm    --model DIR --out vllm.json                          (GPU via the production VllmBackend; first
                                                                  token, greedy 32 tokens, prompt echo)
    compare --hf hf.json --vllm vllm.json --gpu-mmlu-run RUN_DIR --out compare.json

Criteria (approved 2026-11-17): first token 15/16 or more equal and every mismatch inside HF top-2; 32-token
generation identical for 12/16 or more (divergence position reported); MMLU path: extracted answers of the first
100 items equal for 95/100 or more between HF and the production GPU MMLU run (unparsable counts reported).
Prompt token IDs are built once with the model's tokenizer and passed to both sides; the vLLM side records the
prompt IDs the engine echoes back.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from sdv_bmt.llm import mmlu
from sdv_bmt.llm.workload import file_sha256, read_jsonl, token_ids_sha256

REPO = Path(__file__).resolve().parents[2]
PROMPTS = REPO / "configs" / "llm" / "hf-check-prompts.json"
STOP = list(mmlu.LLAMA3_STOP_TOKEN_IDS)


def chat_ids(tok, text: str) -> list[int]:
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                date_string=mmlu.DATE_STRING)
    return list(tok.encode(s, add_special_tokens=False))


def cmd_hf(model: str, mmlu_prompts: str, out: str, threads: int, n_mmlu: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model)
    t = time.time()
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
    m.eval()
    rec = {"side": "hf", "model": model, "dtype": "float32", "device": "cpu", "torch": torch.__version__,
           "threads": threads, "load_s": time.time() - t, "tokenizer_json_sha256": file_sha256(Path(model) / "tokenizer.json"),
           "prompts_file_sha256": file_sha256(PROMPTS), "fixed": [], "mmlu": []}
    cfg = json.loads(PROMPTS.read_text())

    def greedy(ids: list[int], max_new: int):
        x = torch.tensor([ids])
        with torch.no_grad():
            g = m.generate(x, max_new_tokens=max_new, do_sample=False, eos_token_id=STOP, pad_token_id=STOP[-1],
                           output_scores=True, return_dict_in_generate=True)
        new = g.sequences[0, len(ids):].tolist()
        lp = torch.log_softmax(g.scores[0][0].float(), -1)
        top = torch.topk(lp, 5)
        return new, [[int(i), float(v)] for v, i in zip(top.values, top.indices)]

    t = time.time()
    for p in cfg["prompts"]:
        ids = chat_ids(tok, p)
        new, top5 = greedy(ids, cfg["max_new_tokens"])
        rec["fixed"].append({"input_token_ids": ids, "input_sha256": token_ids_sha256(ids), "first_top5": top5,
                             "out_token_ids": new})
    rec["fixed_s"] = time.time() - t
    t = time.time()
    rows = list(read_jsonl(Path(mmlu_prompts)))[:n_mmlu]
    for r in rows:
        new, _ = greedy(r["input_token_ids"], mmlu.MAX_NEW_TOKENS)
        text = tok.decode(new, skip_special_tokens=True)
        rec["mmlu"].append({"item_id": r["item_id"], "input_sha256": r["input_token_ids_sha256"], "out_token_ids": new,
                            "text": text, "extracted": mmlu.extract_answer(text)})
    rec["mmlu_s"] = time.time() - t
    Path(out).write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("load_s", "fixed_s", "mmlu_s")}))


def cmd_vllm(model: str, out: str) -> None:
    from transformers import AutoTokenizer

    from sdv_bmt.llm.backends import VllmBackend

    tok = AutoTokenizer.from_pretrained(model)
    cfg = json.loads(PROMPTS.read_text())
    be = VllmBackend(options={"model_path": model, "gpu_memory_utilization": 0.5, "max_model_len": 4096, "seed": 0})
    applied = be.start()

    async def go():
        res = []
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from vllm.sampling_params import RequestOutputKind

        for k, p in enumerate(cfg["prompts"]):
            ids = chat_ids(tok, p)
            sp = SamplingParams(temperature=0.0, max_tokens=cfg["max_new_tokens"], stop_token_ids=STOP, logprobs=5,
                                detokenize=False, output_kind=RequestOutputKind.FINAL_ONLY)
            final = None
            async for o in be.engine.generate(TokensPrompt(prompt_token_ids=ids), sp, f"hfchk-{k}"):
                final = o
            c = final.outputs[0]
            first = c.logprobs[0] if c.logprobs else {}
            res.append({"input_token_ids": ids, "input_sha256": token_ids_sha256(ids),
                        "engine_prompt_token_ids_equal": list(final.prompt_token_ids) == ids,
                        "out_token_ids": list(c.token_ids),
                        "first_top5": sorted([[int(t), float(v.logprob)] for t, v in first.items()], key=lambda x: -x[1])})
        return res

    fixed = asyncio.run(go())
    be.stop()
    Path(out).write_text(json.dumps({"side": "vllm", "model": model, "applied": applied, "fixed": fixed}, indent=1, default=str))
    print(json.dumps({"n": len(fixed), "prompt_echo_equal": all(f["engine_prompt_token_ids_equal"] for f in fixed)}))


def cmd_compare(hf_path: str, vllm_path: str, gpu_run: str, out: str) -> None:
    hf, vl = json.loads(Path(hf_path).read_text()), json.loads(Path(vllm_path).read_text())
    res = {"criteria": {"first_token_min_equal": 15, "first_token_mismatch_in_hf_top2": True,
                        "gen32_min_identical": 12, "mmlu_min_agree_of_100": 95}}
    same_inputs = [a["input_token_ids"] == b["input_token_ids"] for a, b in zip(hf["fixed"], vl["fixed"])]
    first_eq, mism_in_top2, ident, diverge = 0, [], 0, []
    for a, b in zip(hf["fixed"], vl["fixed"]):
        ha, vb = a["out_token_ids"][:1], b["out_token_ids"][:1]
        if ha == vb:
            first_eq += 1
        else:
            top2 = [t for t, _ in a["first_top5"][:2]]
            mism_in_top2.append(bool(vb) and vb[0] in top2)
        if a["out_token_ids"] == b["out_token_ids"]:
            ident += 1
        else:
            pos = next((i for i, (x, y) in enumerate(zip(a["out_token_ids"], b["out_token_ids"])) if x != y),
                       min(len(a["out_token_ids"]), len(b["out_token_ids"])))
            diverge.append(pos)
    # MMLU path: HF first 100 vs the production GPU MMLU run records
    gpu = {r["item_id"]: r for r in read_jsonl(Path(gpu_run) / "requests.jsonl") if r["phase"] == "measure"}
    agree, hf_unp, gpu_unp, same_ids = 0, 0, 0, 0
    for h in hf["mmlu"]:
        g = gpu[h["item_id"]]
        ge = mmlu.extract_answer(g.get("text"))
        agree += (h["extracted"] == ge)
        hf_unp += h["extracted"] is None
        gpu_unp += ge is None
        same_ids += h["input_sha256"] == g["input_token_ids_sha256"]
    res.update({
        "prompt_token_ids_identical_fixed": all(same_inputs),
        "engine_prompt_echo_equal": all(f.get("engine_prompt_token_ids_equal") for f in vl["fixed"]),
        "first_token_equal": first_eq, "first_token_mismatch_in_hf_top2": mism_in_top2,
        "gen32_identical": ident, "gen32_divergence_positions": diverge,
        "mmlu_items": len(hf["mmlu"]), "mmlu_prompt_ids_identical": same_ids, "mmlu_answer_agree": agree,
        "mmlu_unparsable": {"hf": hf_unp, "gpu": gpu_unp},
        "precision": {"hf": "float32 CPU", "vllm": vl["applied"].get("dtype")},
        "tokenizer_json_sha256_hf": hf.get("tokenizer_json_sha256"),
    })
    res["pass"] = {
        "first_token": first_eq >= 15 and all(mism_in_top2),
        "gen32": ident >= 12,
        "mmlu_path": len(hf["mmlu"]) == 100 and same_ids == 100 and agree >= 95,
        "inputs_identical": all(same_inputs) and res["engine_prompt_echo_equal"],
    }
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hf"); h.add_argument("--model", required=True); h.add_argument("--mmlu-prompts", required=True)
    h.add_argument("--out", required=True); h.add_argument("--threads", type=int, default=32)
    h.add_argument("--n-mmlu", type=int, default=100)
    v = sub.add_parser("vllm"); v.add_argument("--model", required=True); v.add_argument("--out", required=True)
    c = sub.add_parser("compare"); c.add_argument("--hf", required=True); c.add_argument("--vllm", required=True)
    c.add_argument("--gpu-mmlu-run", required=True); c.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "hf":
        cmd_hf(a.model, a.mmlu_prompts, a.out, a.threads, a.n_mmlu)
    elif a.cmd == "vllm":
        cmd_vllm(a.model, a.out)
    else:
        cmd_compare(a.hf, a.vllm, a.gpu_mmlu_run, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
