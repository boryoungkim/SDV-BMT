"""Bundle-only evaluation of LLM bundles with the catalog ``sdv-bmt-llm-v1``.

Reads only the bundle; verifies every raw file against manifest.json before and after; writes a new
``evaluations/<eval_id>/`` (metrics.json, eval_manifest.json).

Per request (completed, measured phase), with N = number of output token IDs received and n_first = number of
token IDs in the first stream item that carries tokens:
  TTFT = t_first - t_submit; E2E = t_done - t_submit;
  TPOT = (t_done - t_first) / (N - n_first) for N > n_first; decode throughput = (N - n_first) / (t_done - t_first).
With one token per stream item (n_first = 1) these are the usual (N - 1) forms.  If a stream item carries more
than one token ("chunk" delivery), only item arrival times are observable: t_first is the arrival of the first
item (the first token may have been produced earlier), and TPOT / throughput are averages over the observed
interval after that item; per-token arrival times are reported as not observable.

Speed (S1-S5): a request is a valid fixed-length measurement only if N equals the target and the finish
reason is the length limit; other completed requests are counted as length mismatches and kept out of the
fixed-length statistics.  Statistics: raw values, nearest-rank p50 (median), min, max, p90, p99 (only when
n >= 100), max_observed; every percentile carries its sample count n.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

from ..contracts.ids import digest_of, new_eval_id, pretty_json_bytes, utc_now_iso
from ..evidence.provenance import code_identity
from ..metrics.catalog import P99_MIN_SAMPLES, PERCENTILE_METHOD
from ..metrics.portable import nearest_rank_index
from . import LLM_BUNDLE_SCHEMA, LLM_CATALOG_VERSION
from . import mmlu as mmlu_mod
from . import tinymmlu as tinymmlu_mod
from .workload import file_sha256, read_jsonl


class LlmEvaluationError(RuntimeError):
    pass


def _score_tinymmlu(ref: list[dict[str, Any]], measured: list[dict[str, Any]]) -> dict[str, Any]:
    items = [{"item_id": r["item_id"], "answer": tinymmlu_mod.CHOICES.index(r["answer"])} for r in ref]
    outputs: dict[str, dict[str, dict[str, Any]]] = {}
    for r in measured:
        outputs.setdefault(r["item_id"], {})[r["request_key"]] = {
            "status": r["status"], "continuation_logprobs": r.get("prompt_token_logprobs")}
    q = tinymmlu_mod.score(items, outputs)
    q["input_token_ids_sha256"] = {f"{r['item_id']}#{r['request_key']}": r["input_token_ids_sha256"] for r in measured}
    return q


# workload kind -> scorer(quality reference rows, measured request records) -> quality block; a later task adds one
TASK_SCORERS = {"tinymmlu": _score_tinymmlu}


def verify_bundle(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    mpath = run_dir / "manifest.json"
    if not mpath.is_file():
        raise LlmEvaluationError(f"{run_dir}: no manifest.json (run not finalized)")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("schema") != LLM_BUNDLE_SCHEMA:
        raise LlmEvaluationError(f"{run_dir}: not an LLM bundle")
    for name, rec in manifest["files"].items():
        p = run_dir / name
        if not p.is_file() or file_sha256(p) != rec["sha256"] or p.stat().st_size != rec["bytes"]:
            raise LlmEvaluationError(f"{run_dir}: raw file {name} changed or missing")
    return manifest


def stats(values: list[float]) -> dict[str, Any]:
    n = len(values)
    if n == 0:
        return {"n": 0, "status": "empty", "values": []}
    s = sorted(values)

    def pct(p: int, min_n: int) -> dict[str, Any]:
        if n < min_n:
            return {"status": "insufficient_samples", "value": None, "n": n, "min_samples": min_n}
        return {"status": "valid", "value": s[nearest_rank_index(n, p)], "n": n}

    return {"n": n, "status": "valid", "values": list(values), "median_p50": pct(50, 1), "min": s[0],
            "max": s[-1], "p90": pct(90, 1), "p99": pct(99, P99_MIN_SAMPLES), "max_observed": s[-1]}


def request_times(r: dict[str, Any]) -> Optional[dict[str, Any]]:
    arr, toks = r["arrival_ns"], r["arrival_tokens"]
    if r["status"] != "completed" or not arr or r["t_done_ns"] is None:
        return None
    n = sum(toks)
    t_first, n_first = arr[0], toks[0]
    out = {"ttft_ns": t_first, "e2e_ns": r["t_done_ns"], "n_out": n, "n_first": n_first,
           "chunked": any(k > 1 for k in toks)}
    span = r["t_done_ns"] - t_first
    if n > n_first and span > 0:
        out["tpot_ns"] = float(Fraction(span, n - n_first))
        out["decode_tps"] = float(Fraction((n - n_first) * 1_000_000_000, span))
    return out


def _streaming_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    hist: dict[str, int] = {}
    for r in records:
        for k in r["arrival_tokens"]:
            hist[str(k)] = hist.get(str(k), 0) + 1
    chunked = any(int(k) > 1 for k in hist)
    first: dict[str, int] = {}
    for r in records:
        if r["arrival_tokens"]:
            k = str(r["arrival_tokens"][0])
            first[k] = first.get(k, 0) + 1
    return {"tokens_per_stream_item": hist, "tokens_in_first_item": first,
            "unit": "chunk" if chunked else "token",
            "per_token_arrival_observable": not chunked}


def _overlap_check(records: list[dict[str, Any]]) -> dict[str, Any]:
    spans = sorted((r["t_submit_ns"], r["t_submit_ns"] + r["t_exit_ns"]) for r in records)
    overlaps = sum(1 for a, b in zip(spans, spans[1:]) if b[0] < a[1])
    return {"check": "one_request_at_a_time", "status": "pass" if overlaps == 0 else "fail", "overlaps": overlaps}


def _controls(env: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    applied = env.get("applied", {})
    before, after = env.get("counters_before", {}), env.get("counters_after", {})
    hits = {k: after.get(k, 0) - before.get(k, 0) for k in after if "hit" in k and isinstance(after.get(k), (int, float))}
    cached = [r["num_cached_tokens"] for r in records if r.get("num_cached_tokens") is not None]
    pc = applied.get("prefix_caching")
    reuse_zero = (all(c == 0 for c in cached) if cached else None)
    if pc is False and reuse_zero is True and all(v == 0 for v in hits.values()):
        verdict = "confirmed_off"
    elif pc is True or reuse_zero is False or any(v > 0 for v in hits.values()):
        verdict = "reuse_observed_or_enabled"
    else:
        verdict = "unverified"
    return {
        "prefix_caching": {"verdict": verdict, "applied_setting": pc, "basis": applied.get("prefix_caching_basis"),
                           "num_cached_tokens_max": max(cached) if cached else None,
                           "num_cached_tokens_reported_requests": len(cached), "cache_hit_counter_delta": hits},
        "speculative_decoding": {"applied": applied.get("speculative_decoding", "unverified"),
                                 "basis": applied.get("speculative_decoding_basis", "runtime configuration read back")},
        "sampling": "greedy: temperature 0 passed on every request",
        "concurrency": _overlap_check(records),
        "in_request_kv_cache": "on (normal decode; not a control target)",
    }


def evaluate(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    manifest = verify_bundle(run_dir)
    spec = json.loads((run_dir / "run_spec.json").read_text(encoding="utf-8"))
    env = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    records = list(read_jsonl(run_dir / "requests.jsonl"))
    measured = [r for r in records if r["phase"] == "measure"]
    kind = spec["resolved"]["workload"]["kind"]
    result: dict[str, Any] = {
        "catalog_version": LLM_CATALOG_VERSION,
        "run_status": manifest["status"],
        "execution_kind": manifest["execution_kind"],
        "measurement_boundary": spec["measurement_boundary"],
        "definitions": {
            "ttft_ns": "t_first - t_submit", "e2e_ns": "t_done - t_submit",
            "tpot_ns": "(t_done - t_first) / (N - n_first), N > n_first",
            "decode_tps": "(N - n_first) / (t_done - t_first) [1/s]",
            "N": "number of output token IDs received",
            "n_first": "token IDs in the first stream item that carries tokens (1 with token-level delivery)",
            "percentile_method": PERCENTILE_METHOD,
            "p99_min_samples": P99_MIN_SAMPLES,
        },
        "streaming": _streaming_summary(measured),
        "controls": _controls(env, records),
        "notes": ["host-observed times of the whole system (runtime + driver + device); not device-internal time",
                  "percentiles describe the measured distribution; max_observed is not a WCET"],
    }
    if result["streaming"]["unit"] == "chunk":
        result["notes"].append("stream delivered chunks: t_first is the arrival of the first item with tokens "
                               "(the first token may have been produced earlier); TPOT and decode throughput use "
                               "N - n_first tokens over the interval after that item; per-token arrival times are "
                               "not observable")
    if kind == "speed":
        per_w: dict[str, Any] = {}
        for w in sorted({r["workload"] for r in measured}):
            rs = [r for r in measured if r["workload"] == w]
            ok, mismatch, err = [], [], []
            for r in rs:
                t = request_times(r)
                if t is None:
                    err.append(r["item_id"])
                elif t["n_out"] == r["target_output_tokens"] and r["finish_reason"] == "length":
                    ok.append(t)
                else:
                    mismatch.append({"item_id": r["item_id"], "n_out": t["n_out"], "finish_reason": r["finish_reason"]})
            per_w[w] = {
                "input_tokens": sorted({r["input_len"] for r in rs}),
                "target_output_tokens": sorted({r["target_output_tokens"] for r in rs}),
                "counts": {"measured": len(rs), "fixed_length_ok": len(ok), "length_mismatch": len(mismatch),
                           "error": len(err)},
                "length_mismatch": mismatch, "errors": err,
                "ttft_ns": stats([t["ttft_ns"] for t in ok]),
                "tpot_ns": stats([t["tpot_ns"] for t in ok if "tpot_ns" in t]),
                "e2e_ns": stats([t["e2e_ns"] for t in ok]),
                "decode_tps": stats([t["decode_tps"] for t in ok if "decode_tps" in t]),
            }
        result["speed"] = per_w
    elif kind == "mmlu":
        ref = list(read_jsonl(run_dir / "quality_reference.jsonl"))
        items = [{"item_id": r["item_id"], "answer": mmlu_mod.LETTERS.index(r["answer"])} for r in ref]
        outputs = {r["item_id"]: {"status": r["status"], "text": r.get("text")} for r in measured}
        q = mmlu_mod.score(items, outputs)
        q["input_token_ids_sha256"] = {r["item_id"]: r["input_token_ids_sha256"] for r in measured}
        q["finish_reasons"] = {}
        for r in measured:
            q["finish_reasons"][str(r["finish_reason"])] = q["finish_reasons"].get(str(r["finish_reason"]), 0) + 1
        result["quality"] = q
        times = [request_times(r) for r in measured]
        times = [t for t in times if t]
        result["mmlu_request_times"] = {"ttft_ns": stats([t["ttft_ns"] for t in times]),
                                        "e2e_ns": stats([t["e2e_ns"] for t in times])}
    elif kind in TASK_SCORERS:
        ref = list(read_jsonl(run_dir / "quality_reference.jsonl"))
        result["quality"] = TASK_SCORERS[kind](ref, measured)
        times = [request_times(r) for r in measured]
        times = [t for t in times if t]
        result["task_request_times"] = {"ttft_ns": stats([t["ttft_ns"] for t in times]),
                                        "e2e_ns": stats([t["e2e_ns"] for t in times])}
    else:
        raise LlmEvaluationError(f"unknown workload kind {kind!r}")
    eval_id = new_eval_id()
    edir = run_dir / "evaluations" / eval_id
    edir.mkdir(parents=True, exist_ok=False)
    content_sha = digest_of(result)
    with open(edir / "metrics.json", "xb") as fh:
        fh.write(pretty_json_bytes({"eval_id": eval_id, "run_id": manifest["run_id"], **result}))
    if verify_bundle(run_dir) != manifest:
        raise LlmEvaluationError("bundle changed during evaluation")
    with open(edir / "eval_manifest.json", "xb") as fh:
        fh.write(pretty_json_bytes({"eval_id": eval_id, "run_id": manifest["run_id"], "catalog_version":
                                    LLM_CATALOG_VERSION, "created_at_utc": utc_now_iso(),
                                    "metrics_content_sha256": content_sha,
                                    "manifest_sha256": file_sha256(run_dir / "manifest.json"),
                                    "code": code_identity()}))
    return edir


def latest_metrics(run_dir: Path) -> dict[str, Any]:
    evs = sorted((Path(run_dir) / "evaluations").glob("*/metrics.json"))
    if not evs:
        raise LlmEvaluationError(f"{run_dir}: not evaluated")
    return json.loads(evs[-1].read_text(encoding="utf-8"))
