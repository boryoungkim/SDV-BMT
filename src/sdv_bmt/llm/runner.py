"""LLM run: config -> one bundle (sequential requests, one at a time, host-clock token arrival times).

Bundle layout (every file created exclusively, manifest written last)::

    <out>/<experiment_id>/experiment.json          config digest (later runs must match)
    <out>/<experiment_id>/<run_id>/
        run_spec.json            resolved config, workload digests, measurement boundary
        environment.json         runtime settings read back after load, host state before/after, cache counters
        requests.jsonl           one record per request (appended as requests finish)
        quality_reference.jsonl  task runs only (MMLU, tinyMMLU): item_id -> reference fields (frozen copy)
        manifest.json            SHA-256 of the files above, run status, request counts
        evaluations/<eval_id>/   written by ``evaluate`` (never overwritten)

Measurement boundary (host ``time.monotonic_ns``): t_submit just before the runtime streaming call with
pre-tokenized input; one arrival time per stream item that carries new token IDs; t_done when the item with
the finish reason arrives.  Output text is decoded after t_done.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from ..contracts.ids import digest_of, new_run_id, pretty_json_bytes, utc_now_iso
from ..evidence.provenance import code_identity, environment
from . import LLM_BUNDLE_SCHEMA, LLM_RUN_SCHEMA
from .backends import FakeCodec, make_backend
from .mmlu import LETTERS, LLAMA3_STOP_TOKEN_IDS, MAX_NEW_TOKENS
from .tinymmlu import PROMPT_LOGPROBS
from .workload import file_sha256, read_jsonl

MEASUREMENT_BOUNDARY = {
    "id": "llm-host-stream/1",
    "plane": "python",
    "clock": "host time.monotonic_ns in the process that drives the runtime streaming API",
    "t_submit": "immediately before the streaming generation call is made with pre-tokenized input token IDs",
    "t_first": "receipt of the first stream item that carries at least one new output token ID",
    "t_done": "receipt of the stream item that carries the finish reason",
    "tokenization": "input tokenization and chat template are applied before the run (excluded); output text "
                    "is decoded after t_done (excluded); runtime-internal detokenization is disabled where the "
                    "runtime allows it (vLLM detokenize=False; furiosa-llm native token-ID stream)",
    "includes": "runtime scheduling, host<->device transfers, device execution, runtime IPC and the Python event "
                "loop delay until the item is received",
    "excludes": "device-internal timing (not observed)",
}

_ENV = re.compile(r"\$\{([A-Z0-9_]+)\}")


class LlmRunError(RuntimeError):
    pass


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def sub(m):
            if m.group(1) not in os.environ:
                raise LlmRunError(f"environment variable {m.group(1)} is not set (used in the run config)")
            return os.environ[m.group(1)]
        return _ENV.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """(config as written, config with ${VARS} expanded and relative file paths resolved)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != LLM_RUN_SCHEMA:
        raise LlmRunError(f"{path}: not an {LLM_RUN_SCHEMA} config")
    for key in ("experiment_id", "execution_kind", "model", "backend", "workload"):
        if key not in raw:
            raise LlmRunError(f"{path}: missing '{key}'")
    cfg = _expand(raw)
    base = Path(path).resolve().parent
    for key in ("file", "items_file"):
        if key in cfg["workload"]:
            p = Path(cfg["workload"][key])
            cfg["workload"][key] = str(p if p.is_absolute() else (base / p).resolve())
    return raw, cfg


# Task workloads: one prompt-file row per runtime request (``item_id``, ``request_key``, ``input_token_ids``,
# ``input_token_ids_sha256``, optional ``score_from``, plus the item fields the scorer needs), in execution order.
# The generation settings are fixed per task here; a task that scores prompt log-probabilities asks for them from
# the row's ``score_from`` on.  A later generation task (e.g. a dialogue response workload) adds an entry with its
# own generation settings and keeps the run, record and bundle structure unchanged.
TASK_GENERATION: dict[str, dict[str, Any]] = {
    "tinymmlu": {"max_tokens": 1, "min_tokens": 0, "ignore_eos": False, "stop_token_ids": [],
                 "prompt_logprobs": PROMPT_LOGPROBS},
}
TASK_REFERENCE_FIELDS: dict[str, tuple[str, ...]] = {"tinymmlu": ("subject", "answer")}
TASK_KEEP_TEXT: dict[str, bool] = {"tinymmlu": False}


def _items_file_check(w: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """The prompt file follows the public item list (order and per-item SHA-256)."""
    items = json.loads(Path(w["items_file"]).read_text(encoding="utf-8"))
    listed = [it["item_id"] for it in items["items"]]
    seen: list[str] = []
    for r in rows:
        if not seen or seen[-1] != r["item_id"]:
            seen.append(r["item_id"])
    if listed != seen:
        raise LlmRunError("prompt file does not follow the public item list order")
    shas = {it["item_id"]: it["item_sha256"] for it in items["items"]}
    bad = [r["item_id"] for r in rows if shas[r["item_id"]] != r["item_sha256"]]
    if bad:
        raise LlmRunError(f"item SHA-256 mismatch for {len(bad)} requests, first {bad[0]}")


def _task_requests(kind: str, w: dict[str, Any], rows: list[dict[str, Any]]):
    if "items_file" in w:
        _items_file_check(w, rows)
    base = TASK_GENERATION[kind]

    def req(r: dict[str, Any], phase: str) -> dict[str, Any]:
        gen = dict(base)
        if "prompt_logprobs" in gen:
            gen["score_from"] = int(r["score_from"])
        return {"item_id": r["item_id"], "request_key": r.get("request_key"), "workload": kind, "phase": phase,
                "input_token_ids": r["input_token_ids"], "input_token_ids_sha256": r["input_token_ids_sha256"],
                "generation": gen}

    warm = int(w.get("warmup", 3))
    reqs = [req(r, "warmup") for r in rows[:warm]] + [req(r, "measure") for r in rows]
    ref, seen = [], set()
    for r in rows:
        if r["item_id"] not in seen:
            seen.add(r["item_id"])
            ref.append({"item_id": r["item_id"], **{k: r[k] for k in TASK_REFERENCE_FIELDS[kind]}})
    return reqs, ref


def build_requests(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(request list in execution order, quality reference rows or [])."""
    w = cfg["workload"]
    path = Path(w["file"])
    actual = file_sha256(path)
    if actual != w["sha256"]:
        raise LlmRunError(f"workload file {path} SHA-256 {actual} != config {w['sha256']}")
    rows = list(read_jsonl(path))
    if w["kind"] == "speed":
        return rows, []
    if w["kind"] in TASK_GENERATION:
        return _task_requests(w["kind"], w, rows)
    if w["kind"] != "mmlu":
        raise LlmRunError(f"unknown workload kind {w['kind']!r}")
    if "items_file" in w:
        items = json.loads(Path(w["items_file"]).read_text(encoding="utf-8"))
        listed = [it["item_id"] for it in items["items"]]
        if listed != [r["item_id"] for r in rows]:
            raise LlmRunError("MMLU prompt file does not follow the public item list order")
        shas = {it["item_id"]: it["item_sha256"] for it in items["items"]}
        bad = [r["item_id"] for r in rows if shas[r["item_id"]] != r["item_sha256"]]
        if bad:
            raise LlmRunError(f"MMLU item SHA-256 mismatch for {len(bad)} items, first {bad[0]}")
    gen = {"max_tokens": MAX_NEW_TOKENS, "min_tokens": 0, "ignore_eos": False,
           "stop_token_ids": list(LLAMA3_STOP_TOKEN_IDS)}
    warm = int(w.get("warmup", 3))
    reqs = []
    for k, r in enumerate(rows[:warm]):
        reqs.append({"item_id": r["item_id"], "workload": "mmlu", "phase": "warmup",
                     "input_token_ids": r["input_token_ids"], "input_token_ids_sha256": r["input_token_ids_sha256"],
                     "generation": gen})
    for r in rows:
        reqs.append({"item_id": r["item_id"], "workload": "mmlu", "phase": "measure",
                     "input_token_ids": r["input_token_ids"], "input_token_ids_sha256": r["input_token_ids_sha256"],
                     "generation": gen})
    ref = [{"item_id": r["item_id"], "subject": r["subject"], "answer": LETTERS[int(r["answer"])]} for r in rows]
    return reqs, ref


def host_state() -> dict[str, Any]:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc!r}"
    st: dict[str, Any] = {"utc": utc_now_iso()}
    try:
        st["loadavg"] = Path("/proc/loadavg").read_text().strip()
    except OSError:
        st["loadavg"] = None
    st["nvidia_smi"] = run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu,clocks.sm,"
                                          "temperature.gpu,power.draw", "--format=csv,noheader"])
    st["nvidia_compute_apps"] = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                     "--format=csv,noheader"])
    st["furiosa_smi_status"] = run(["furiosa-smi", "status"])
    return st


def make_decoder(cfg: dict[str, Any]) -> tuple[Callable[[list[int]], str], dict[str, Any]]:
    if cfg["backend"]["kind"] == "fake":
        return FakeCodec().decode, {"tokenizer": "fake"}
    from transformers import AutoTokenizer

    tpath = cfg["model"]["tokenizer_path"]
    tok = AutoTokenizer.from_pretrained(tpath)
    info = {"tokenizer_path": tpath}
    tj = Path(tpath) / "tokenizer.json"
    if tj.exists():
        info["tokenizer_json_sha256"] = file_sha256(tj)
    return (lambda ids: tok.decode(ids, skip_special_tokens=True)), info


async def _run_requests(backend, requests: list[dict[str, Any]], decode, fh, keep_text: bool) -> dict[str, int]:
    counts = {"planned": len(requests), "completed": 0, "error": 0}
    for i, req in enumerate(requests):
        ids = req["input_token_ids"]
        arrivals: list[int] = []
        arrival_tokens: list[int] = []
        out_ids: list[int] = []
        finish: Optional[str] = None
        cached: Optional[int] = None
        t_done: Optional[int] = None
        logprobs = None
        prompt_lps = None
        err = None
        t_submit = time.monotonic_ns()
        try:
            async for item in backend.stream(f"req-{i}", ids, req["generation"]):
                t = time.monotonic_ns()
                if item.new_token_ids:
                    arrivals.append(t - t_submit)
                    arrival_tokens.append(len(item.new_token_ids))
                    out_ids.extend(item.new_token_ids)
                if item.num_cached_tokens is not None:
                    cached = item.num_cached_tokens
                if getattr(item, "logprobs", None) and logprobs is None:
                    logprobs = item.logprobs[0]
                if item.prompt_token_logprobs is not None and prompt_lps is None:
                    prompt_lps = item.prompt_token_logprobs
                if item.finish_reason is not None:
                    finish = item.finish_reason
                    t_done = t - t_submit
        except Exception as exc:  # noqa: BLE001 - recorded per request
            err = repr(exc)
        t_exit = time.monotonic_ns() - t_submit
        status = "completed" if err is None and finish is not None else "error"
        if status == "completed":
            counts["completed"] += 1
        else:
            counts["error"] += 1
            err = err or "stream ended without a finish reason"
        rec = {
            "request_index": i, "item_id": req["item_id"], "workload": req["workload"], "phase": req["phase"],
            "input_len": len(ids), "input_token_ids_sha256": req["input_token_ids_sha256"],
            "generation": req["generation"], "target_output_tokens": req.get("target_output_tokens"),
            "t_submit_ns": t_submit, "arrival_ns": arrivals, "arrival_tokens": arrival_tokens,
            "t_done_ns": t_done, "t_exit_ns": t_exit, "out_token_ids": out_ids, "finish_reason": finish,
            "num_cached_tokens": cached, "status": status, "error": err,
        }
        if req.get("request_key") is not None:
            rec["request_key"] = req["request_key"]
        if "prompt_logprobs" in req["generation"]:
            # log-probability of the actual prompt tokens from generation.score_from on (None: not returned)
            rec["prompt_token_logprobs"] = prompt_lps
            if status == "completed" and (prompt_lps is None or len(prompt_lps) !=
                                          len(ids) - req["generation"]["score_from"]):
                status, err = "error", "runtime returned no prompt log-probabilities for the scored positions"
                counts["completed"] -= 1
                counts["error"] += 1
                rec["status"], rec["error"] = status, err
        if logprobs is not None:
            rec["first_token_logprobs"] = {str(k): v for k, v in logprobs.items()}
        if keep_text:
            rec["text"] = decode(out_ids)  # after t_done: outside the measured interval
        fh.write(json.dumps(rec, separators=(",", ":"), sort_keys=True) + "\n")
        fh.flush()
    return counts


def _write_x(path: Path, value: Any) -> None:
    with open(path, "xb") as fh:
        fh.write(pretty_json_bytes(value))


def run_once(config_path: Path, out_root: Path) -> Path:
    raw, cfg = load_config(config_path)
    requests, reference = build_requests(cfg)
    exp_dir = Path(out_root) / cfg["experiment_id"]
    exp_record = {"schema": LLM_RUN_SCHEMA, "experiment_id": cfg["experiment_id"], "config_digest": digest_of(raw),
                  "config": raw}
    exp_file = exp_dir / "experiment.json"
    if exp_file.exists():
        if json.loads(exp_file.read_text())["config_digest"] != exp_record["config_digest"]:
            raise LlmRunError(f"{exp_dir} holds a different experiment config")
    else:
        exp_dir.mkdir(parents=True, exist_ok=True)
        _write_x(exp_file, exp_record)
    run_id = new_run_id()
    run_dir = exp_dir / run_id
    run_dir.mkdir()
    decode, tok_info = make_decoder(cfg)
    run_spec = {
        "schema": LLM_BUNDLE_SCHEMA, "run_id": run_id, "experiment_id": cfg["experiment_id"],
        "execution_kind": cfg["execution_kind"], "config": raw, "config_digest": exp_record["config_digest"],
        "resolved": {"model": cfg["model"], "backend": cfg["backend"], "device": cfg.get("device"),
                     "workload": cfg["workload"]},
        "workload_file_sha256": cfg["workload"]["sha256"], "request_count": len(requests),
        "measurement_boundary": MEASUREMENT_BOUNDARY, "decoder": tok_info,
        "controls_requested": {"sampling": "greedy (temperature 0)", "prefix_caching": False,
                               "speculative_decoding": False, "concurrency": 1, "batch": 1,
                               "in_request_kv_cache": "on (normal decode)"},
        "created_at_utc": utc_now_iso(), "code": code_identity(),
        "python_environment": environment(),
    }
    _write_x(run_dir / "run_spec.json", run_spec)
    files = ["run_spec.json", "environment.json", "requests.jsonl"]
    if reference:
        with open(run_dir / "quality_reference.jsonl", "x", encoding="utf-8") as fh:
            for r in reference:
                fh.write(json.dumps(r, separators=(",", ":"), sort_keys=True) + "\n")
        files.append("quality_reference.jsonl")
    env: dict[str, Any] = {"host_before": host_state()}
    backend = make_backend(cfg["backend"]["kind"], cfg["backend"].get("options", {}))
    status, reason, counts = "failed", None, {"planned": len(requests), "completed": 0, "error": 0}
    t_load = time.monotonic_ns()
    try:
        env["applied"] = backend.start()
        env["load_ns"] = time.monotonic_ns() - t_load
        env["counters_before"] = backend.counters()
        with open(run_dir / "requests.jsonl", "x", encoding="utf-8") as fh:
            kind = cfg["workload"]["kind"]
            keep_text = TASK_KEEP_TEXT.get(kind, bool(reference))
            counts = asyncio.run(_run_requests(backend, requests, decode, fh, keep_text=keep_text))
        env["counters_after"] = backend.counters()
        status = "completed" if counts["error"] == 0 else "completed_with_errors"
    except Exception as exc:  # noqa: BLE001
        reason = repr(exc)
        if not (run_dir / "requests.jsonl").exists():
            (run_dir / "requests.jsonl").write_text("")
    finally:
        backend.stop()
        env["host_after"] = host_state()
        _write_x(run_dir / "environment.json", env)
    manifest = {
        "schema": LLM_BUNDLE_SCHEMA, "bundle_kind": "llm", "execution_kind": cfg["execution_kind"],
        "experiment_id": cfg["experiment_id"], "run_id": run_id, "created_at_utc": utc_now_iso(),
        "files": {n: {"sha256": file_sha256(run_dir / n), "bytes": (run_dir / n).stat().st_size} for n in files},
        "status": {"run_status": status, "reason": reason, "request_counts": counts},
    }
    _write_x(run_dir / "manifest.json", manifest)
    return run_dir
