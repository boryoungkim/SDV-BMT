#!/usr/bin/env python3
"""Pooled descriptive quantiles of per-request facts over all runs of each experiment (analysis plane, stdlib).

    .venv/bin/python tools/stage2b_g_real/pooled_quantiles.py OUT.json ROOT [ROOT ...]

Descriptive only: the canonical metrics are per run (``sdv-bmt evaluate``); pooling the requests of the
repetitions of one plan shows the shape of the distribution and is not a substitute for them.  Host run-clock
and device-clock quantities are kept apart (never subtracted from each other).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage1r_g"))
from bundle_facts import load_run, quantiles, run_dirs_under  # noqa: E402


def main() -> int:
    out, roots = sys.argv[1], sys.argv[2:]
    groups: dict[str, list] = {}
    for root in roots:
        for d in run_dirs_under(root):
            r = load_run(d)
            groups.setdefault(r.experiment_id, []).append(r)
    result = {}
    for exp, runs in sorted(groups.items()):
        iss = [q for r in runs for q in r.issued()]
        comp = [q for r in runs for q in r.completed()]
        result[exp] = {
            "runs": len(runs), "requests_issued": len(iss), "requests_completed": len(comp),
            "host_ns": {"issue_lag": quantiles((q.issue_lag for q in iss), (50, 90, 99, 999)),
                        "submit": quantiles((q.submit for q in iss), (50, 90, 99, 999)),
                        "after_submit": quantiles((q.after_submit for q in iss), (50, 90, 99, 999)),
                        "dut_service": quantiles((q.dut_service for q in iss), (50, 90, 99, 999)),
                        "request_completion": quantiles((q.request_completion for q in comp), (50, 90, 99, 999)),
                        "output_ready": quantiles((q.host("t_sched", "t_output_ready") for q in comp), (50, 90, 99, 999)),
                        "output_step": quantiles((q.output_step for q in comp), (50, 90, 99, 999))},
            "device_ns": {"host_to_device": quantiles((q.segment("host_to_device") for q in iss), (50, 90, 99, 999)),
                          "device_execution": quantiles((q.segment("device_execution") for q in iss), (50, 90, 99, 999)),
                          "device_to_host": quantiles((q.segment("device_to_host") for q in iss), (50, 90, 99, 999)),
                          "probe_globaltimer": quantiles((q.in_kernel_ns for q in iss), (50, 99))},
            "probe_sm_mhz": quantiles((q.sm_mhz for q in iss), (1, 50, 99)),
        }
    Path(out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    us = lambda v: f"{v / 1000:.1f}"  # noqa: E731
    for exp, g in result.items():
        h, dv = g["host_ns"], g["device_ns"]
        print(exp, g["runs"], g["requests_issued"])
        for k, v in list(h.items()) + list(dv.items()):
            if v:
                print(f"   {k:20} p50 {us(v['p50'])} p90 {us(v.get('p90', 0))} p99 {us(v['p99'])} p99.9 {us(v.get('p999', 0))} max {us(v['max'])}")
        print("   probe MHz", g["probe_sm_mhz"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
