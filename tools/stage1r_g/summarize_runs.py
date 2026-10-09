"""Stage 1R-G / 2b-G: descriptive per-run summaries of realtime bundles (analysis plane; stdlib only).

    python3 tools/stage1r_g/summarize_runs.py OUT.json ROOT_OR_RUN_DIR [...]

For every run bundle found: identity (experiment, repetition, DUT config, boundary, instrumentation),
status and counts, the canonical metric values of its latest evaluation (``sdv-bmt evaluate``), and
descriptive quantiles of the recorded facts per clock domain (host run clock; device clock), the
backend snapshots at the window edges, the precondition summary and the realization validity per
condition.  Durations are computed only within one clock domain.  No inferential statistics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bundle_facts import Run, load_run, quantiles, run_dirs_under  # noqa: E402

METRICS = (
    "bmt.count.planned", "bmt.count.issued", "bmt.count.completed", "bmt.count.dropped", "bmt.count.timed_out",
    "bmt.count.error", "bmt.count.not_issued", "bmt.diagnostic.late_completion.count",
    "bmt.latency.dut_service.p50", "bmt.latency.dut_service.p90", "bmt.latency.dut_service.p99",
    "bmt.latency.request_completion.completed.p50", "bmt.latency.request_completion.completed.p99",
    "bmt.latency.output_ready.completed.p50", "bmt.latency.output_ready.completed.p99",
    "bmt.issue_lag.p99", "bmt.issue_lag.max", "bmt.throughput.completed_per_s", "bmt.offered_rate_per_s",
    "bmt.deadline.miss_rate", "bmt.output.identity_rate",
)
SNAPSHOT_KEYS = ("pstate", "sm_clock_mhz", "mem_clock_mhz", "temperature_c", "power_mw", "pcie_link",
                 "memory_used_bytes", "foreign_compute_processes", "clocks_event_reasons", "stream_idle",
                 "device_allocations_total", "pinned_allocations_total", "duration_ns", "process_utilization")


def phase_duration(run: Run, name: str) -> int | None:
    begin, end = run.phase_event(name, "begin"), run.phase_event(name, "end")
    return end["t_run_ns"] - begin["t_run_ns"] if begin and end else None


def summarize(run: Run) -> dict:
    m = run.manifest
    parts = m["comparison_keys"]["components"]["common_parts"]
    issued, completed = run.issued(), run.completed()
    segs = sorted({k for r in issued for k in (r.timing or {})})
    prepare = run.phase_event("dut.prepare", "end") or {}
    snap = {}
    for point in ("measure_begin", "drain_end"):
        s = run.snapshot(point)
        snap[point] = None if s is None else {k: s.get(k) for k in SNAPSHOT_KEYS if k in s}
    window = [e for e in run.phases if e.get("phase") == "measure" and e.get("edge") == "begin"]
    return {
        "run_dir": str(run.run_dir),
        "run_id": run.run_id,
        "experiment_id": run.experiment_id,
        "repetition": m["repetition"],
        "execution_kind": m["execution_kind"],
        "dut_config_id": run.run_spec["resolved_plan"]["dut"]["dut_config_id"],
        "measurement_boundary_id": m["measurement_boundary"]["measurement_boundary_id"],
        "telemetry": parts.get("telemetry"),
        "realtime_harness": parts.get("realtime_harness"),
        "replay": parts.get("replay"),
        "status": {k: m["status"][k] for k in ("execution_status", "execution_reason", "measurement_status",
                                               "measurement_reason")},
        "request_counts": m["status"]["request_counts"],
        "native_counts": (m.get("realtime") or {}).get("native_counts"),
        "precondition": (m.get("realtime") or {}).get("precondition"),
        "cpu_affinity_realized_equals_requested": (m.get("realtime") or {}).get("cpu_affinity_realized")
        == (m.get("realtime") or {}).get("cpu_affinity_requested"),
        "evaluation": None if run.evaluation is None else {
            "eval_id": run.evaluation["eval_id"], "metrics": {k: run.metric(k) for k in METRICS}},
        "lifecycle_host_ns": {"dut.prepare": phase_duration(run, "dut.prepare"),
                              "prepare_inputs": phase_duration(run, "prepare_inputs")},
        "native_prepare_facts": {k: v for k, v in (prepare.get("details", {}).get("native") or {}).items()
                                 if k in ("cubin_sha256", "nvrtc_options", "profiler_injection")},
        "window": window[0].get("details") if window else None,
        "host_ns": {
            "issue_lag_issued": quantiles(r.issue_lag for r in issued),
            "issue_lag_all_admitted": quantiles(r.issue_lag for r in run.requests.values()),
            "submit": quantiles(r.submit for r in issued),
            "dut_service": quantiles(r.dut_service for r in issued),
            "after_submit": quantiles(r.after_submit for r in issued),
            "output_step": quantiles(r.output_step for r in completed),
            "request_completion": quantiles(r.request_completion for r in completed),
        },
        "device_ns": {
            "timing_method": next((r.timing_method for r in issued if r.timing_method), None),
            **{f"segment.{s}": quantiles(r.segment(s) for r in issued) for s in segs},
            "in_kernel_globaltimer": quantiles(r.in_kernel_ns for r in issued),
        },
        "in_kernel_sm_mhz": quantiles(r.sm_mhz for r in issued),
        "snapshots": snap,
        "realization": {e["condition"]: f"{e['validity']}|{e['equivalence']}" for e in m["realization"]["evidence"]},
    }


def main(out: str, roots: list[str]) -> None:
    dirs = []
    for root in roots:
        p = Path(root)
        dirs += [p] if (p / "manifest.json").exists() else run_dirs_under(p)
    rows = [summarize(load_run(d)) for d in dirs]
    Path(out).write_text(json.dumps(rows, indent=1, sort_keys=True), encoding="utf-8")
    us = lambda q, k: "-" if not q else f"{q[k] / 1000:.1f}"  # noqa: E731
    print(f"{'experiment':34} rep st  cnt(c/d/t/e)  lag p50/p99/max us    submit p50/p99   DUTsvc p50/p99/max us     out-step p50")
    for r in rows:
        c, h = r["request_counts"] or {}, r["host_ns"]
        print(f"{r['experiment_id'][:34]:34} {r['repetition']:>3} {r['status']['measurement_status'][:2]} "
              f"{c.get('completed', '-')}/{c.get('dropped', '-')}/{c.get('timed_out', '-')}/{c.get('error', '-'):<4} "
              f"{us(h['issue_lag_issued'], 'p50')}/{us(h['issue_lag_issued'], 'p99')}/{us(h['issue_lag_issued'], 'max'):<10} "
              f"{us(h['submit'], 'p50')}/{us(h['submit'], 'p99'):<10} "
              f"{us(h['dut_service'], 'p50')}/{us(h['dut_service'], 'p99')}/{us(h['dut_service'], 'max'):<16} "
              f"{us(h['output_step'], 'p50')}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
