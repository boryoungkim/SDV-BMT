"""Minimal furiosa-llm smoke on RNGD: runtime/driver compatibility, token-ID stream,
fixed output length, prefix-cache control and long-prompt support.

Run with the furiosa-llm environment's interpreter (the environment is not modified):

    HF_HOME=<hf cache> HF_HUB_OFFLINE=1 <venv>/bin/python tools/llm/rngd_smoke.py \
        --model furiosa-ai/Llama-3.1-8B-Instruct --revision <sha> --out <dir> [--prefix-cache on|off]

Timing values here are diagnostics only; they are not benchmark results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path


def _now() -> int:
    return time.perf_counter_ns()


def _smi(*args: str) -> str:
    try:
        return subprocess.run(["furiosa-smi", *args], capture_output=True, text=True, timeout=30).stdout
    except Exception as exc:  # noqa: BLE001 - record and continue
        return f"error: {exc!r}"


def synthetic_ids(n: int, seed: int, bos: int = 128000, vocab_regular: int = 128000) -> list[int]:
    """BOS followed by n-1 seeded regular (non-special) token IDs."""
    rng = random.Random(seed)
    return [bos] + [rng.randrange(0, vocab_regular) for _ in range(n - 1)]


async def stream_once(llm, ids: list[int], sp) -> dict:
    from transformers import BatchEncoding

    enc = BatchEncoding({"input_ids": ids, "attention_mask": [1] * len(ids)})
    items = []
    t0 = _now()
    async for out in llm.engine.stream_generate(enc, sp):
        t = _now()
        comps = []
        for c in out.outputs:
            comps.append({"token_ids": list(c.token_ids), "finish_reason": c.finish_reason})
        items.append({"t_ns": t - t0, "num_cached_tokens": getattr(out, "num_cached_tokens", None), "outputs": comps})
    t_end = _now()
    # Interpret token_ids as delta or cumulative: cumulative if each item's list extends the previous one.
    seqs = [it["outputs"][0]["token_ids"] if it["outputs"] else [] for it in items]
    cumulative = len(seqs) > 1 and all(len(b) > len(a) and b[: len(a)] == a for a, b in zip(seqs, seqs[1:]))
    if cumulative:
        tokens = seqs[-1]
        per_item = [len(seqs[0])] + [len(b) - len(a) for a, b in zip(seqs, seqs[1:])]
    else:
        tokens = [t for s in seqs for t in s]
        per_item = [len(s) for s in seqs]
    first_idx = next((i for i, k in enumerate(per_item) if k > 0), None)
    finish = next((it["outputs"][0]["finish_reason"] for it in reversed(items) if it["outputs"] and it["outputs"][0]["finish_reason"]), None)
    return {
        "input_len": len(ids),
        "n_items": len(items),
        "stream_mode": "cumulative" if cumulative else "delta",
        "tokens_per_item_hist": {str(k): per_item.count(k) for k in sorted(set(per_item))},
        "n_out": len(tokens),
        "finish_reason": finish,
        "ttft_ns": items[first_idx]["t_ns"] if first_idx is not None else None,
        "e2e_ns": items[-1]["t_ns"] if items else None,
        "loop_exit_ns": t_end - t0,
        "num_cached_tokens": [it["num_cached_tokens"] for it in items[:1]] + [items[-1]["num_cached_tokens"]] if items else [],
        "out_token_ids": tokens,
    }


def dp_metrics() -> dict:
    try:
        from furiosa.llm_native.llm import get_dp_metrics

        return {str(k): {a: getattr(v, a) for a in ("prompt_tokens", "generation_tokens", "kv_cache_usage", "prefix_cache_hit_rate", "wire_hit_rate")} for k, v in get_dp_metrics().items()}
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def prom_prefix_lines() -> list[str]:
    try:
        from furiosa.llm_native.llm import get_metrics_as_prometheus_string

        s = get_metrics_as_prometheus_string(True)
        return [ln for ln in s.splitlines() if "prefix" in ln.lower() or "cache" in ln.lower()][:60]
    except Exception as exc:  # noqa: BLE001
        return [f"error: {exc!r}"]


async def main_async(args) -> dict:
    import furiosa_llm
    from furiosa_llm import LLM, SamplingParams
    from furiosa_llm.metadata.config_types import PrefixCacheConfig, SchedulerConfig

    try:
        from furiosa_llm.sampling_params import RequestOutputKind
        delta = RequestOutputKind.DELTA
    except Exception:  # noqa: BLE001
        delta = None

    rec: dict = {
        "purpose": "RNGD minimal smoke (diagnostic, not a benchmark result)",
        "model": args.model,
        "revision": args.revision,
        "prefix_cache_requested": args.prefix_cache,
        "furiosa_llm_version": getattr(furiosa_llm, "__version__", None),
        "python": sys.version,
        "host": platform.node(),
        "env": {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_OFFLINE")},
        "smi_info_before": _smi("info"),
        "smi_status_before": _smi("status"),
    }
    sched = SchedulerConfig(prefix_cache_config=PrefixCacheConfig(enabled=(args.prefix_cache == "on")))
    rec["scheduler_config_passed"] = json.loads(LLM._serialize_obj(sched))
    t = _now()
    llm = LLM(args.model, revision=args.revision, devices="npu:0", scheduler_config=sched, cache_dir=args.cache_dir)
    rec["load_s"] = (_now() - t) / 1e9
    rec["artifact_id"] = getattr(llm, "artifact_id", None)
    rec["max_model_len"] = llm.max_model_len
    rec["smi_status_loaded"] = _smi("status")
    rec["default_sampling_params"] = repr(llm.get_default_sampling_params())

    def sp(max_tokens: int, fixed: bool) -> object:
        kw = dict(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=max_tokens)
        if fixed:
            kw.update(min_tokens=max_tokens, ignore_eos=True)
        if delta is not None:
            kw["output_kind"] = delta
        return SamplingParams(**kw)

    tests: dict = {}
    # 1) natural prompt via chat template (sanity of output meaning, EOS handling)
    tok = llm.tokenizer
    chat_ids = tok.apply_chat_template([{"role": "user", "content": "What is the capital of France? Answer in one word."}],
                                       add_generation_prompt=True, tokenize=True, date_string="26 Jul 2024")
    if isinstance(chat_ids, dict) or hasattr(chat_ids, "input_ids"):
        chat_ids = list(chat_ids["input_ids"])
    r = await stream_once(llm, list(chat_ids), sp(32, fixed=False))
    r["text"] = tok.decode(r["out_token_ids"], skip_special_tokens=False)
    tests["natural_chat"] = r
    # 2) fixed-length S1 (128/64) and S3 (512/512)
    for name, (n_in, n_out, seed) in {"S1": (128, 64, 11), "S3": (512, 512, 33)}.items():
        tests[f"fixed_{name}"] = await stream_once(llm, synthetic_ids(n_in, seed), sp(n_out, fixed=True))
    # 3) prefix cache: same S5-length prompt twice
    ids5 = synthetic_ids(1024, 55)
    m0 = dp_metrics()
    a = await stream_once(llm, ids5, sp(8, fixed=True))
    m1 = dp_metrics()
    b = await stream_once(llm, ids5, sp(8, fixed=True))
    m2 = dp_metrics()
    tests["prefix_repeat"] = {"first": a, "second": b, "dp_metrics": [m0, m1, m2], "prom_lines": prom_prefix_lines(),
                              "second_over_first_ttft": (b["ttft_ns"] / a["ttft_ns"]) if a["ttft_ns"] else None}
    # 4) long prompts beyond the largest prefill bucket (1024)
    for n_in in (1100, 1600, 2100):
        try:
            tests[f"long_{n_in}"] = await stream_once(llm, synthetic_ids(n_in, n_in), sp(8, fixed=True))
        except Exception as exc:  # noqa: BLE001
            tests[f"long_{n_in}"] = {"error": repr(exc)}
    for v in tests.values():
        for x in ([v] if "out_token_ids" in v else [v.get("first", {}), v.get("second", {})]):
            if "out_token_ids" in x and len(x["out_token_ids"]) > 40:
                x["out_token_ids_head"] = x["out_token_ids"][:40]
                x["out_token_ids_len"] = len(x["out_token_ids"])
                del x["out_token_ids"]
    rec["tests"] = tests
    rec["smi_status_after"] = _smi("status")
    try:
        llm.engine.shutdown()
    except Exception as exc:  # noqa: BLE001
        rec["shutdown_error"] = repr(exc)
    return rec


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--prefix-cache", choices=("on", "off"), default="off")
    p.add_argument("--cache-dir", default=os.path.join(os.environ.get("SDV_BMT_CACHE", "cache"), "furiosa-llm"))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        rec = asyncio.run(main_async(args))
        rc = 0
    except Exception as exc:  # noqa: BLE001
        import traceback

        rec = {"error": repr(exc), "traceback": traceback.format_exc()}
        rc = 1
    (out / f"smoke-prefix-{args.prefix_cache}.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("tests", "smi_info_before", "smi_status_before", "smi_status_loaded", "smi_status_after")}, indent=1, default=str)[:3000])
    return rc


if __name__ == "__main__":
    sys.exit(main())
