"""``python -m sdv_bmt.llm`` -- LLM benchmark commands.

    prepare-speed --out FILE                      write the S1-S5 token-ID workload file
    prepare-mmlu  --parquet P --tokenizer DIR --items FILE --prompts-out FILE
                                                  select the 1,000 items, write the public item list (IDs +
                                                  SHA-256, no text) and the local prompt token-ID file
    prepare-tinymmlu --parquet P --tokenizer DIR --items FILE --prompts-out FILE --revision REV
                                                  the 100 tinyMMLU items: public item list (IDs + SHA-256, no
                                                  text) and the local request file (4 choice requests per item)
    lengths PROMPTS [--max-model-len N]           token-length coverage of a prompt file
    run CONFIG [--out DIR]                        one run -> one bundle
    evaluate RUN_DIR                              bundle-only evaluation (catalog sdv-bmt-llm-v1)
    compare REF_RUN OTHER_RUN --out FILE          MMLU / tinyMMLU per-item comparison (reference = GPU); tinyMMLU:
                                                  near-tie rule (judged on the items both sides answer clearly)
    aggregate RUN_DIR... --out FILE               speed statistics over several evaluated runs
    table EXPERIMENT_DIR... --out FILE.csv [--irt IRT.json ...]
                                                  metric table (docs/metric_table.md): representative metrics
                                                  under readable names first, then the overall result per
                                                  metric, then one row per run and metric; the representative
                                                  metrics are also printed (stderr)

Run with the interpreter of the runtime's own environment and ``PYTHONPATH=src`` (the runtime environments
are not modified).  Output paths are created exclusively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Optional, Sequence

from ..contracts.ids import pretty_json_bytes, utc_now_iso
from ..metrics.table import HEADLINE_ROW, MODEL_ROW, format_console
from . import mmlu, tinymmlu
from .evaluate import evaluate, latest_metrics, stats, verify_bundle
from .runner import run_once
from .table import LlmReportError, write_table
from .workload import file_sha256, read_jsonl, speed_requests, token_ids_sha256, write_jsonl


def _write_x(path: Path, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as fh:
        fh.write(pretty_json_bytes(value))


def cmd_prepare_speed(out: str) -> int:
    write_jsonl(Path(out), speed_requests())
    print(json.dumps({"file": out, "sha256": file_sha256(Path(out))}))
    return 0


def cmd_prepare_mmlu(parquet: str, tokenizer: str, items_path: str, prompts_out: str, revision: str) -> int:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    table = pq.read_table(parquet).to_pylist()
    rows = [mmlu.canonical_item(r) for r in table]
    selected = mmlu.select_items([r["subject"] for r in rows])
    items = {
        "dataset": mmlu.MMLU_DATASET, "config": mmlu.MMLU_CONFIG, "split": mmlu.MMLU_SPLIT,
        "revision": revision, "source_file": "all/test-00000-of-00001.parquet",
        "source_file_sha256": file_sha256(Path(parquet)), "source_item_count": len(rows),
        "selection": "subject-proportional 1,000 (largest remainder); within subject rank by SHA-256 of "
                     "'42:<subject>:<row>'; list order = proportional 100-item subset first, then the rest",
        "item_sha256_method": "SHA-256 of canonical JSON {subject, question, choices, answer} (sorted keys, "
                              "compact separators, UTF-8)",
        "label": mmlu.EVALUATION_LABEL,
        "items": [{"item_id": f"test/{i}", "subject": rows[i]["subject"], "item_sha256": mmlu.item_sha256(rows[i])}
                  for i in selected],
    }
    ip = Path(items_path)
    if ip.exists():
        existing = json.loads(ip.read_text(encoding="utf-8"))
        if existing["items"] != items["items"] or existing["source_file_sha256"] != items["source_file_sha256"]:
            print(f"error: {ip} exists with a different selection", file=sys.stderr)
            return 2
    else:
        _write_x(ip, items)
    tok = AutoTokenizer.from_pretrained(tokenizer)
    out_rows = []
    for i in selected:
        ids = mmlu.prompt_token_ids(tok, rows[i])
        if ids.count(128000) != 1 or ids[0] != 128000:
            print(f"error: test/{i}: prompt must start with exactly one BOS", file=sys.stderr)
            return 2
        out_rows.append({"item_id": f"test/{i}", "subject": rows[i]["subject"], "answer": rows[i]["answer"],
                         "item_sha256": mmlu.item_sha256(rows[i]), "input_token_ids": ids,
                         "input_token_ids_sha256": token_ids_sha256(ids)})
    Path(prompts_out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(Path(prompts_out), out_rows)
    tj = Path(tokenizer) / "tokenizer.json"
    print(json.dumps({"items": items_path, "prompts": prompts_out, "prompts_sha256": file_sha256(Path(prompts_out)),
                      "tokenizer_json_sha256": file_sha256(tj) if tj.exists() else None, "count": len(out_rows)}))
    return 0


def cmd_prepare_tinymmlu(parquet: str, tokenizer: str, items_path: str, prompts_out: str, revision: str) -> int:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    rows = [tinymmlu.canonical_item(r) for r in pq.read_table(parquet).to_pylist()]
    if len(rows) != tinymmlu.TINYMMLU_TOTAL:
        print(f"error: {parquet} has {len(rows)} items, expected {tinymmlu.TINYMMLU_TOTAL}", file=sys.stderr)
        return 2
    items = {
        "dataset": tinymmlu.TINYMMLU_DATASET, "config": tinymmlu.TINYMMLU_CONFIG, "split": tinymmlu.TINYMMLU_SPLIT,
        "revision": revision, "source_file": "all/test-00000-of-00001.parquet",
        "source_file_sha256": file_sha256(Path(parquet)), "source_item_count": len(rows),
        "selection": "all items of the split, in dataset order (the order tinyBenchmarks expects)",
        "item_sha256_method": "SHA-256 of canonical JSON {subject, question, choices, answer, input_formatted} "
                              "(sorted keys, compact separators, UTF-8)",
        "label": tinymmlu.EVALUATION_LABEL,
        "items": [{"item_id": f"test/{i}", "subject": r["subject"], "item_sha256": tinymmlu.item_sha256(r)}
                  for i, r in enumerate(rows)],
    }
    ip = Path(items_path)
    if ip.exists():
        existing = json.loads(ip.read_text(encoding="utf-8"))
        if existing["items"] != items["items"] or existing["source_file_sha256"] != items["source_file_sha256"]:
            print(f"error: {ip} exists with different items", file=sys.stderr)
            return 2
    else:
        _write_x(ip, items)
    tok = AutoTokenizer.from_pretrained(tokenizer)
    out_rows = []
    for i, r in enumerate(rows):
        for q in tinymmlu.item_requests(tok, r):
            ids = q["input_token_ids"]
            out_rows.append({"item_id": f"test/{i}", "request_key": q["choice"], "subject": r["subject"],
                             "answer": tinymmlu.CHOICES[r["answer"]], "item_sha256": tinymmlu.item_sha256(r),
                             "input_token_ids": ids, "input_token_ids_sha256": token_ids_sha256(ids),
                             "score_from": q["continuation_start"]})
    Path(prompts_out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(Path(prompts_out), out_rows)
    tj = Path(tokenizer) / "tokenizer.json"
    print(json.dumps({"items": items_path, "prompts": prompts_out, "prompts_sha256": file_sha256(Path(prompts_out)),
                      "tokenizer_json_sha256": file_sha256(tj) if tj.exists() else None, "requests": len(out_rows)}))
    return 0


def cmd_lengths(prompts: str, max_model_len: Optional[int], max_new: int) -> int:
    lens = [len(r["input_token_ids"]) for r in read_jsonl(Path(prompts))]
    s = sorted(lens)
    rep = {"file": prompts, "sha256": file_sha256(Path(prompts)), "count": len(lens), "min": s[0], "max": s[-1],
           "median": s[len(s) // 2], "p99": s[min(len(s) - 1, -(-99 * len(s) // 100) - 1)],
           "over_1024_input": sum(1 for x in lens if x > 1024), "max_new_tokens": max_new}
    if max_model_len:
        fits = sum(1 for x in lens if x + max_new <= max_model_len)
        rep.update({"max_model_len": max_model_len, "fit": fits, "not_fit": len(lens) - fits})
    print(json.dumps(rep, indent=1))
    return 0


def cmd_compare(ref_run: str, other_run: str, out: str) -> int:
    a, b = latest_metrics(Path(ref_run)), latest_metrics(Path(other_run))
    verify_bundle(Path(ref_run)), verify_bundle(Path(other_run))
    if "quality" not in a or "quality" not in b:
        print("error: both runs must be evaluated MMLU or tinyMMLU runs", file=sys.stderr)
        return 2
    if a["quality"].get("label") != b["quality"].get("label"):
        print("error: the two runs use different quality evaluations", file=sys.stderr)
        return 2
    ha, hb = a["quality"]["input_token_ids_sha256"], b["quality"]["input_token_ids_sha256"]
    same_ids = ha == hb
    module = tinymmlu if a["quality"].get("scoring_rule") == tinymmlu.SCORING_RULE_ID else mmlu
    res = module.compare(a["quality"], b["quality"])
    res.update({"reference_run": ref_run, "other_run": other_run, "reference_eval": a["eval_id"],
                "other_eval": b["eval_id"], "prompt_token_ids_identical": same_ids,
                "prompt_token_ids_differing_items": [k for k in ha if ha.get(k) != hb.get(k)][:20],
                "created_at_utc": utc_now_iso()})
    if not same_ids:
        res["verdict"] = "not_judged_different_prompt_token_ids"
    _write_x(Path(out), res)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("prompt_token_ids_differing_items", "differing_items")}, indent=1))
    if res.get("summary"):
        print(res["summary"] if same_ids else "not judged: the two runs received different prompt token IDs",
              file=sys.stderr)
    return 0


def cmd_aggregate(run_dirs: Sequence[str], out: str) -> int:
    per_w: dict[str, dict[str, list]] = {}
    runs = []
    for d in run_dirs:
        verify_bundle(Path(d))
        m = latest_metrics(Path(d))
        runs.append({"run_dir": d, "eval_id": m["eval_id"], "run_status": m["run_status"],
                     "streaming": m["streaming"], "prefix_caching": m["controls"]["prefix_caching"]["verdict"]})
        for w, v in m.get("speed", {}).items():
            agg = per_w.setdefault(w, {"ttft_ns": [], "tpot_ns": [], "e2e_ns": [], "decode_tps": [],
                                       "counts": [], "run_medians_ttft": [], "run_medians_tpot": []})
            for k in ("ttft_ns", "tpot_ns", "e2e_ns", "decode_tps"):
                agg[k] += v[k]["values"]
            agg["counts"].append(v["counts"])
            agg["run_medians_ttft"].append(v["ttft_ns"].get("median_p50", {}).get("value"))
            agg["run_medians_tpot"].append(v["tpot_ns"].get("median_p50", {}).get("value"))
    summary = {}
    for w, agg in per_w.items():
        summary[w] = {k: stats(agg[k]) for k in ("ttft_ns", "tpot_ns", "e2e_ns", "decode_tps")}
        summary[w]["counts_per_run"] = agg["counts"]
        summary[w]["run_medians"] = {"ttft_ns": agg["run_medians_ttft"], "tpot_ns": agg["run_medians_tpot"]}
    _write_x(Path(out), {"runs": runs, "speed": summary, "created_at_utc": utc_now_iso()})
    print(json.dumps({w: {"n": s["ttft_ns"]["n"], "ttft_med_ms": (s["ttft_ns"]["median_p50"]["value"] or 0) / 1e6,
                          "tpot_med_ms": (s["tpot_ns"]["median_p50"]["value"] or 0) / 1e6}
                      for w, s in summary.items()}, indent=1))
    return 0


def cmd_table(exp_dirs: Sequence[str], out: str, irt: Sequence[str] = ()) -> int:
    try:
        rows, data = write_table(exp_dirs, out, irt)
    except (LlmReportError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_console(rows, str(Path(out).resolve())), file=sys.stderr)
    print(json.dumps({"table": out, "sha256": hashlib.sha256(data).hexdigest(), "rows": len(rows),
                      "headline_rows": sum(1 for r in rows if r["row"] == HEADLINE_ROW),
                      "model_rows": sum(1 for r in rows if r["row"] == MODEL_ROW),
                      "experiments": len({r["experiment_id"] for r in rows})}, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m sdv_bmt.llm", description="SDV-BMT LLM benchmark (Python plane)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prepare-speed"); s.add_argument("--out", required=True)
    m = sub.add_parser("prepare-mmlu")
    for a in ("--parquet", "--tokenizer", "--items", "--prompts-out", "--revision"):
        m.add_argument(a, required=True)
    tm = sub.add_parser("prepare-tinymmlu")
    for a in ("--parquet", "--tokenizer", "--items", "--prompts-out", "--revision"):
        tm.add_argument(a, required=True)
    ln = sub.add_parser("lengths"); ln.add_argument("prompts"); ln.add_argument("--max-model-len", type=int)
    ln.add_argument("--max-new", type=int, default=mmlu.MAX_NEW_TOKENS)
    r = sub.add_parser("run"); r.add_argument("config"); r.add_argument("--out", default="runs")
    e = sub.add_parser("evaluate"); e.add_argument("run_dir")
    c = sub.add_parser("compare"); c.add_argument("ref_run"); c.add_argument("other_run"); c.add_argument("--out", required=True)
    g = sub.add_parser("aggregate"); g.add_argument("run_dirs", nargs="+"); g.add_argument("--out", required=True)
    t = sub.add_parser("table"); t.add_argument("exp_dirs", nargs="+"); t.add_argument("--out", required=True)
    t.add_argument("--irt", action="append", default=[], help="result file of tools/llm/tinymmlu_irt.py (repeatable)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "prepare-speed":
        return cmd_prepare_speed(a.out)
    if a.cmd == "prepare-mmlu":
        return cmd_prepare_mmlu(a.parquet, a.tokenizer, a.items, a.prompts_out, a.revision)
    if a.cmd == "prepare-tinymmlu":
        return cmd_prepare_tinymmlu(a.parquet, a.tokenizer, a.items, a.prompts_out, a.revision)
    if a.cmd == "lengths":
        return cmd_lengths(a.prompts, a.max_model_len, a.max_new)
    if a.cmd == "run":
        d = run_once(Path(a.config), Path(a.out))
        print(json.dumps({"run_dir": str(d), **json.loads((d / "manifest.json").read_text())["status"]}))
        return 0
    if a.cmd == "evaluate":
        d = evaluate(Path(a.run_dir))
        print(json.dumps({"eval_dir": str(d)}))
        return 0
    if a.cmd == "compare":
        return cmd_compare(a.ref_run, a.other_run, a.out)
    if a.cmd == "table":
        return cmd_table(a.exp_dirs, a.out, a.irt)
    return cmd_aggregate(a.run_dirs, a.out)


if __name__ == "__main__":
    sys.exit(main())
