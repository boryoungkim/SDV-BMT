"""LLM runtime backends: fake (tests / CI), vLLM (GPU), furiosa-llm (RNGD).

A backend streams the output of one request as items ``StreamItem(new_token_ids, finish_reason,
num_cached_tokens)``.  The runner timestamps each item when it receives it.  ``start()`` loads the model
(outside measurement) and returns the settings the runtime reports it actually applied; values that cannot
be read back are recorded as ``"unverified"`` and never assumed.

Runtime packages are imported lazily inside each backend so this module imports in any environment.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

UNVERIFIED = "unverified"


@dataclass
class StreamItem:
    new_token_ids: list[int]
    finish_reason: Optional[str] = None
    num_cached_tokens: Optional[int] = None


@dataclass
class Backend:
    kind: str = "base"
    options: dict[str, Any] = field(default_factory=dict)

    def start(self) -> dict[str, Any]:
        raise NotImplementedError

    def stream(self, request_id: str, token_ids: list[int], gen: dict[str, Any]) -> AsyncIterator[StreamItem]:
        raise NotImplementedError

    def counters(self) -> dict[str, Any]:
        """Cache-reuse counters the runtime exposes (read before / after the run)."""
        return {}

    def stop(self) -> None:
        pass


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        return f"error: {exc!r}"


# --------------------------------------------------------------------------- fake


class FakeCodec:
    """Deterministic stand-in tokenizer for tests: token id -> printable character."""

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(32 + (i % 95)) for i in ids)


@dataclass
class FakeBackend(Backend):
    """Deterministic fake runtime (no device).  Options: tokens_per_item (default 1), natural_len (tokens before
    the fake end token when EOS is honoured, default 3), answer_letters (emit A-D first so the FakeCodec text is
    an MMLU-style answer)."""

    kind: str = "fake"

    def start(self) -> dict[str, Any]:
        return {"runtime": "fake", "execution_kind": "mock", "prefix_caching": False,
                "speculative_decoding": False, "sampling": "greedy", "num_cached_tokens_reported": True}

    async def stream(self, request_id: str, token_ids: list[int], gen: dict[str, Any]) -> AsyncIterator[StreamItem]:
        seed = int(hashlib.sha256(",".join(map(str, token_ids)).encode()).hexdigest()[:8], 16)
        max_tokens = int(gen["max_tokens"])
        honour_eos = not gen.get("ignore_eos", False) or bool(self.options.get("ignore_eos_broken"))
        natural = int(self.options.get("natural_len", 3))
        per_item = int(self.options.get("tokens_per_item", 1))
        out: list[int] = []
        if self.options.get("answer_letters"):
            out += [33 + seed % 4, 14]  # FakeCodec: 33 -> 'A' ... 36 -> 'D', 14 -> '.'
        while len(out) < max_tokens:
            out.append(10 + (seed + len(out)) % 50)
        finish = "length"
        min_tokens = 0 if self.options.get("ignore_eos_broken") else int(gen.get("min_tokens", 0))
        if honour_eos and natural < max_tokens and natural >= min_tokens:
            out = out[:natural]
            finish = "stop"
        await asyncio.sleep(0)
        yield StreamItem([], None, 0)  # an item without tokens, as furiosa-llm sends first
        for i in range(0, len(out), per_item):
            await asyncio.sleep(0)
            last = i + per_item >= len(out)
            yield StreamItem(out[i:i + per_item], finish if last else None, 0)

    def counters(self) -> dict[str, Any]:
        return {"prefix_cache_hits": 0, "prefix_cache_queries": 0}


# --------------------------------------------------------------------------- vLLM (GPU)


@dataclass
class VllmBackend(Backend):
    """vLLM AsyncLLM engine on one GPU.  Options: model_path, gpu_memory_utilization (0.5), max_model_len
    (4096), seed (0).  The GPU is selected by CUDA_VISIBLE_DEVICES set by the caller before start."""

    kind: str = "vllm"
    engine: Any = None

    def start(self) -> dict[str, Any]:
        from vllm import AsyncEngineArgs
        import vllm
        from vllm.v1.engine.async_llm import AsyncLLM

        o = self.options
        args = AsyncEngineArgs(
            model=o["model_path"], tokenizer=o["model_path"], dtype="bfloat16", quantization=None,
            seed=int(o.get("seed", 0)), enable_prefix_caching=False, max_model_len=int(o.get("max_model_len", 4096)),
            gpu_memory_utilization=float(o.get("gpu_memory_utilization", 0.5)), generation_config="vllm",
            disable_log_stats=False,
        )
        self.engine = AsyncLLM.from_engine_args(args)
        cfg = self.engine.vllm_config
        cache, model, sched = cfg.cache_config, cfg.model_config, cfg.scheduler_config
        applied = {
            "runtime": "vllm", "runtime_version": vllm.__version__, "execution_kind": "actual",
            "prefix_caching": bool(cache.enable_prefix_caching),
            "speculative_decoding": cfg.speculative_config is not None,
            "dtype": str(model.dtype), "quantization": model.quantization,
            "generation_config": getattr(model, "generation_config", UNVERIFIED),
            "max_model_len": model.max_model_len, "seed": model.seed,
            "chunked_prefill": getattr(sched, "enable_chunked_prefill", UNVERIFIED),
            "max_num_batched_tokens": getattr(sched, "max_num_batched_tokens", UNVERIFIED),
            "max_num_seqs": getattr(sched, "max_num_seqs", UNVERIFIED),
            "stream_interval": getattr(sched, "stream_interval", UNVERIFIED),
            "gpu_memory_utilization": cache.gpu_memory_utilization,
            "enforce_eager": getattr(model, "enforce_eager", UNVERIFIED),
            "compilation_mode": str(getattr(getattr(cfg, "compilation_config", None), "mode", UNVERIFIED)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "multiprocessing_engine_core": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "default(1)"),
            "num_cached_tokens_reported": True,
        }
        try:
            import torch

            applied["torch_version"] = torch.__version__
            applied["cuda_device_name"] = torch.cuda.get_device_name(0)
            applied["cuda_device_count"] = torch.cuda.device_count()
            applied["cuda_device_uuid"] = str(getattr(torch.cuda.get_device_properties(0), "uuid", UNVERIFIED))
        except Exception as exc:  # noqa: BLE001
            applied["torch_probe_error"] = repr(exc)
        return applied

    def _sampling(self, gen: dict[str, Any]):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        return SamplingParams(
            temperature=0.0, max_tokens=int(gen["max_tokens"]), min_tokens=int(gen.get("min_tokens", 0)),
            ignore_eos=bool(gen.get("ignore_eos", False)), stop_token_ids=list(gen.get("stop_token_ids") or []),
            logprobs=gen.get("logprobs"), detokenize=False, output_kind=RequestOutputKind.DELTA,
        )

    async def stream(self, request_id: str, token_ids: list[int], gen: dict[str, Any]) -> AsyncIterator[StreamItem]:
        from vllm.inputs import TokensPrompt

        sp = self._sampling(gen)
        async for out in self.engine.generate(TokensPrompt(prompt_token_ids=list(token_ids)), sp, request_id):
            c = out.outputs[0] if out.outputs else None
            item = StreamItem(list(c.token_ids) if c else [], c.finish_reason if c else None,
                              getattr(out, "num_cached_tokens", None))
            if gen.get("logprobs") and c is not None and c.logprobs:
                item.logprobs = [{int(k): float(v.logprob) for k, v in step.items()} for step in c.logprobs]  # type: ignore[attr-defined]
            yield item

    def counters(self) -> dict[str, Any]:
        try:
            from prometheus_client import REGISTRY

            out = {}
            for metric in REGISTRY.collect():
                if "prefix_cache" in metric.name:
                    for s in metric.samples:
                        if s.name.endswith("_total") or s.name.endswith(metric.name):
                            out[s.name] = out.get(s.name, 0.0) + float(s.value)
            return out or {"note": UNVERIFIED}
        except Exception as exc:  # noqa: BLE001
            return {"error": repr(exc)}

    def stop(self) -> None:
        if self.engine is not None:
            try:
                self.engine.shutdown()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- furiosa-llm (RNGD)


@dataclass
class FuriosaBackend(Backend):
    """furiosa-llm 2026.3 on RNGD npu:0 through the native engine's token-ID stream.  Options: model (HF id or
    artifact dir), revision, cache_dir.  Prefix caching is disabled through the scheduler configuration; no
    draft model is passed (speculative decoding is not exposed by the API)."""

    kind: str = "furiosa"
    llm: Any = None

    def start(self) -> dict[str, Any]:
        import furiosa_llm
        from furiosa_llm import LLM
        from furiosa_llm.metadata.config_types import PrefixCacheConfig, SchedulerConfig

        o = self.options
        sched = SchedulerConfig(prefix_cache_config=PrefixCacheConfig(enabled=False))
        self.llm = LLM(o["model"], revision=o.get("revision"), devices=o.get("devices", "npu:0"),
                       scheduler_config=sched, cache_dir=o.get("cache_dir"))
        applied: dict[str, Any] = {
            "runtime": "furiosa-llm", "runtime_version": str(getattr(furiosa_llm, "__version__", UNVERIFIED)),
            "execution_kind": "actual", "artifact_id": getattr(self.llm, "artifact_id", None),
            "max_model_len": self.llm.max_model_len,
            "scheduler_config_passed": LLM._serialize_obj(sched),
            "speculative_decoding": False,
            "speculative_decoding_basis": "API passes no draft model and num_speculative_tokens=None",
            "num_cached_tokens_reported": True,
        }
        prom = self._prom()
        flag = [ln for ln in prom.splitlines() if ln.startswith("furiosa_llm_cache_config_info")]
        if flag and 'enable_prefix_caching="false"' in flag[0]:
            applied["prefix_caching"] = False
        elif flag and 'enable_prefix_caching="true"' in flag[0]:
            applied["prefix_caching"] = True
        else:
            applied["prefix_caching"] = UNVERIFIED
        applied["prefix_caching_basis"] = flag[0] if flag else "cache_config_info metric not found"
        applied["smi_info"] = _run(["furiosa-smi", "info"])
        return applied

    @staticmethod
    def _prom() -> str:
        try:
            from furiosa.llm_native.llm import get_metrics_as_prometheus_string

            return get_metrics_as_prometheus_string(True)
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc!r}"

    async def stream(self, request_id: str, token_ids: list[int], gen: dict[str, Any]) -> AsyncIterator[StreamItem]:
        from furiosa_llm import SamplingParams
        from furiosa_llm.sampling_params import RequestOutputKind
        from transformers import BatchEncoding

        sp = SamplingParams(temperature=0.0, max_tokens=int(gen["max_tokens"]),
                            min_tokens=int(gen.get("min_tokens", 0)), ignore_eos=bool(gen.get("ignore_eos", False)),
                            stop_token_ids=list(gen.get("stop_token_ids") or []) or None,
                            output_kind=RequestOutputKind.DELTA)
        enc = BatchEncoding({"input_ids": list(token_ids), "attention_mask": [1] * len(token_ids)})
        async for out in self.llm.engine.stream_generate(enc, sp):
            c = out.outputs[0] if out.outputs else None
            yield StreamItem(list(c.token_ids) if c else [], c.finish_reason if c else None,
                             getattr(out, "num_cached_tokens", None))

    def counters(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for ln in self._prom().splitlines():
            if ln.startswith("furiosa_llm_prefix_cache_hits") or ln.startswith("furiosa_llm_prefix_cache_queries"):
                name = ln.split("{", 1)[0]
                out[name] = out.get(name, 0.0) + float(ln.rsplit(" ", 1)[-1])
        return out or {"note": UNVERIFIED}

    def stop(self) -> None:
        if self.llm is not None:
            try:
                self.llm.engine.shutdown()
            except Exception:  # noqa: BLE001
                pass


def make_backend(kind: str, options: dict[str, Any]) -> Backend:
    table = {"fake": FakeBackend, "vllm": VllmBackend, "furiosa": FuriosaBackend}
    if kind not in table:
        raise ValueError(f"unknown backend kind {kind!r}")
    return table[kind](options=dict(options))
