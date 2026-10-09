#!/usr/bin/env python3
"""Render the public result table of one run bundle from its newest Metric Set v1 evaluation (tool).

Every number comes from the bundle's ``evaluations/<eval_id>/metrics.json``; nothing is recomputed.  Latencies
are printed in microseconds (one decimal, nearest-rank values from the evaluation), rates with three decimals.

    .venv/bin/python tools/metric_set_v1/result_table.py RUN_DIR > table.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CATALOG = "sdv-bmt-model-metric-set-v1"


def newest_v1(run_dir: Path) -> tuple[dict, dict]:
    found = []
    for path in (run_dir / "evaluations").glob("*/metrics.json"):
        doc = json.loads(path.read_text())
        if doc.get("catalog_version") == CATALOG:
            em = json.loads((path.parent / "eval_manifest.json").read_text())
            found.append((em["created_at_utc"], doc, em))
    if not found:
        raise SystemExit(f"{run_dir}: no Metric Set v1 evaluation")
    _, doc, em = sorted(found, key=lambda x: x[0])[-1]
    return doc, em


def main() -> int:
    run_dir = Path(sys.argv[1])
    doc, em = newest_v1(run_dir)
    m = {x["metric_id"]: x for x in doc["metrics"]}

    def v(metric_id: str, fmt: str = "us") -> str:
        x = m[metric_id]
        if x["status"] != "valid":
            return f"{x['status']}"
        if fmt == "us":
            return f"{x['value'] / 1000:,.1f}"
        if fmt == "rate":
            return f"{x['value']:.3f}"
        if fmt == "int":
            return f"{int(x['value']):,}"
        if fmt == "q":
            return f"{x['value']:.5f}"
        return str(x["value"])

    ctx = doc["context"]
    print(f"Run `{doc['run_id']}`, evaluation `{doc['eval_id']}` (catalog `{CATALOG}`), "
          f"{ctx['planned_count']} planned / {ctx['completed_count']} completed requests, "
          f"{ctx['replay_mode']} {ctx['rate_hz']} Hz, configured N = {ctx['configured_n']}; "
          f"metrics content SHA-256 `{em['metrics_content_sha256']}`.")
    print()
    print("| latency (µs) | p50 | p90 | p99 | max_observed |")
    print("|---|---:|---:|---:|---:|")
    for family, label in (("request_completion", "request completion"), ("dut_service", "DUT service"),
                          ("output_ready", "output ready"), ("issue_lag", "issue lag (harness diagnostic)"),
                          ("submit", "submit (diagnostic)"), ("post_submit", "post-submit (diagnostic)"),
                          ("output_stage", "output stage (diagnostic)"), ("h2d", "H2D bracket (device clock)"),
                          ("device_execution_bracket", "device execution bracket (device clock)"),
                          ("d2h", "D2H bracket (device clock)")):
        print(f"| {label} | " + " | ".join(v(f"bmt.{family}.{s}") for s in ("p50", "p90", "p99", "max_observed"))
              + " |")
    print()
    print("| rate, availability, timeliness | value |")
    print("|---|---:|")
    for metric_id, label, fmt in (("bmt.offered_rate", "offered rate (1/s)", "rate"),
                                  ("bmt.achieved_completion_rate", "achieved completion rate (1/s)", "rate"),
                                  ("bmt.completion_rate", "completion rate", "rate"),
                                  ("bmt.drop_rate", "drop rate", "rate"), ("bmt.timeout_rate", "timeout rate", "rate"),
                                  ("bmt.error_rate", "error rate", "rate"),
                                  ("bmt.deadline.completion_miss_rate", "completion deadline miss rate", "rate"),
                                  ("bmt.deadline.output_ready_miss_rate", "output-ready deadline miss rate", "rate")):
        print(f"| {label} | {v(metric_id, fmt)} |")
    deadline = doc["parameters"]["deadline"]
    print()
    print(f"Deadline (evaluation parameter): {deadline['period_ns'] / 1e6:.6f} ms per request ({deadline['policy']}).")
    print()
    print("| task quality (COCO bbox, conditional on completed outputs) | value |")
    print("|---|---:|")
    for name, label, fmt in (("coverage", "coverage", "rate"), ("conditional.ap", "AP", "q"),
                             ("conditional.ap50", "AP50", "q"), ("conditional.ap75", "AP75", "q"),
                             ("conditional.ap_small", "AP small", "q"), ("conditional.ap_medium", "AP medium", "q"),
                             ("conditional.ap_large", "AP large", "q"), ("conditional.ar1", "AR1", "q"),
                             ("conditional.ar10", "AR10", "q"), ("conditional.ar100", "AR100", "q"),
                             ("conditional.precision", "precision (score ≥ 0.5, IoU ≥ 0.5)", "q"),
                             ("conditional.recall", "recall (score ≥ 0.5, IoU ≥ 0.5)", "q"),
                             ("conditional.f1", "F1 (score ≥ 0.5, IoU ≥ 0.5)", "q"),
                             ("conditional.tp_count", "TP", "int"), ("conditional.fp_count", "FP", "int"),
                             ("conditional.fn_count", "FN", "int")):
        print(f"| {label} | {v('quality.detection.' + name, fmt)} |")
    print()
    print(f"Output identity against the pre-designated reference: {v('bmt.output.identity_rate', 'rate')} "
          "(determinism / corruption diagnostic, not accuracy).")
    checks = {c["check"]: c for c in doc.get("evidence_checks", [])}
    if checks:
        c = checks["completed_beyond_request_timeout"]
        print(f"Evidence check `completed_beyond_request_timeout`: {c['status']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
