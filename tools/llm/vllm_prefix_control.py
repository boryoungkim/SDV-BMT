"""Diagnostic: show that vLLM's cache-reuse records react to prefix caching (positive control).

Loads the model twice (prefix caching off, then on) on the GPU selected by CUDA_VISIBLE_DEVICES and sends the
same 1,024-token prompt twice to each engine.  Records num_cached_tokens per request, the Prometheus
prefix-cache counters and TTFT (time ratio is a diagnostic only).  Not a benchmark result.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src <vllm venv>/bin/python tools/llm/vllm_prefix_control.py \
        --model <hf dir> --out <file.json>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from sdv_bmt.llm.workload import synthetic_token_ids


def counters():
    from prometheus_client import REGISTRY

    out = {}
    for m in REGISTRY.collect():
        if "prefix_cache" in m.name:
            for s in m.samples:
                out[s.name] = out.get(s.name, 0.0) + float(s.value)
    return out


async def two_requests(engine, ids):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    res = []
    for k in range(2):
        sp = SamplingParams(temperature=0.0, max_tokens=8, min_tokens=8, ignore_eos=True, detokenize=False,
                            output_kind=RequestOutputKind.DELTA)
        t0 = time.monotonic_ns()
        first, cached = None, None
        async for out in engine.generate(TokensPrompt(prompt_token_ids=ids), sp, f"pc-{k}-{t0}"):
            if first is None and out.outputs and out.outputs[0].token_ids:
                first = time.monotonic_ns() - t0
            cached = getattr(out, "num_cached_tokens", cached)
        res.append({"ttft_ns": first, "num_cached_tokens": cached})
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    ids = synthetic_token_ids(55, "prefix-control", 0, 1024)
    rec = {"purpose": "vLLM prefix-cache positive control (diagnostic)", "model": a.model}
    for enabled in (False, True):
        eng = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=a.model, dtype="bfloat16", seed=0, enable_prefix_caching=enabled, max_model_len=4096,
            gpu_memory_utilization=0.5, generation_config="vllm", disable_log_stats=False))
        c0 = counters()
        reqs = asyncio.run(two_requests(eng, ids))
        c1 = counters()
        rec[f"prefix_caching_{'on' if enabled else 'off'}"] = {
            "applied": eng.vllm_config.cache_config.enable_prefix_caching, "requests": reqs,
            "counter_delta": {k: c1.get(k, 0) - c0.get(k, 0) for k in c1},
            "second_over_first_ttft": reqs[1]["ttft_ns"] / reqs[0]["ttft_ns"]}
        eng.shutdown()
        del eng
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
