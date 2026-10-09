#!/usr/bin/env python3
"""Stage 2b-G real-model baseline: descriptive analysis of the YOLOX run bundles (analysis plane, stdlib only).

    .venv/bin/python tools/stage2b_g_real/analyze_baseline.py OUT.json RUN_ROOT [RUN_ROOT ...]

Reads every run bundle under the roots (via tools/stage1r_g/bundle_facts.py) and writes:
  * runs: per-run canonical metrics (latest ``sdv-bmt evaluate``), host run-clock facts, device-clock facts
    (CUDA-event brackets, frequency probe), NVML window-edge snapshots, cuDNN prepare facts, realization
    validity per condition, and the host load recorded just before the run (COMMANDS.log "state before");
  * variability: per experiment, the spread of per-run statistics across its repetitions;
  * correlation: Pearson coefficients as a description of co-variation (run level and request level); no test;
  * warmup, outliers, per_item: first requests vs the rest, the slowest requests with their decomposition by
    clock domain, and whether the same image is slow in every run.
Host and device durations are shown side by side and never subtracted from each other.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage1r_g"))
from bundle_facts import load_run, quantiles, run_dirs_under, spread  # noqa: E402

METRICS = ("bmt.count.planned", "bmt.count.completed", "bmt.count.dropped", "bmt.count.timed_out", "bmt.count.error",
           "bmt.count.not_issued", "bmt.rate.completion", "bmt.rate.drop", "bmt.rate.timeout", "bmt.rate.error",
           "bmt.latency.request_completion.completed.p50", "bmt.latency.request_completion.completed.p90",
           "bmt.latency.request_completion.completed.p99", "bmt.latency.request_completion.planned.p99",
           "bmt.latency.dut_service.p50", "bmt.latency.dut_service.p90", "bmt.latency.dut_service.p99",
           "bmt.latency.output_ready.completed.p50", "bmt.latency.output_ready.completed.p90",
           "bmt.latency.output_ready.completed.p99", "bmt.issue_lag.p99", "bmt.issue_lag.max",
           "bmt.throughput.completed_per_s", "bmt.offered_rate_per_s", "bmt.deadline.miss_rate",
           "bmt.output.identity_rate")
SNAP = ("pstate", "sm_clock_mhz", "mem_clock_mhz", "temperature_c", "power_mw", "pcie_link", "memory_used_bytes",
        "foreign_compute_processes", "clocks_event_reasons", "duration_ns")


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    syy = sum((p[1] - my) ** 2 for p in pairs)
    if sxx == 0 or syy == 0:
        return None
    return round(sum((p[0] - mx) * (p[1] - my) for p in pairs) / math.sqrt(sxx * syy), 3)


def load_states(roots):
    """'state before' entries of every COMMANDS.log above the runs: label -> (uptime line, gpu line)."""
    states = []
    seen = set()
    for root in roots:
        for log in [p / "COMMANDS.log" for p in [Path(root), *Path(root).parents]]:
            if log.exists() and log not in seen:
                seen.add(log)
                for line in open(log, encoding="utf-8"):
                    e = json.loads(line)
                    if str(e.get("label", "")).startswith(("state before", "host/gpu state before")):
                        states.append((e["utc"], e.get("stdout_tail", "")))
    return sorted(states)


def load_avg(text: str):
    m = re.search(r"load average: ([0-9.]+), ([0-9.]+), ([0-9.]+)", text)
    return [float(x) for x in m.groups()] if m else None


def output_detections(run) -> dict:
    """request id -> number of detections in its encoded task output."""
    blob = (run.run_dir / "outputs.bin").read_bytes()
    out = {}
    for e in run.events:
        if e.get("kind") == "completed":
            ref = e["output_ref"]
            out[e["request_id"]] = len(json.loads(blob[ref["offset"]:ref["offset"] + ref["length"]])["detections"])
    return out


def run_row(run, states) -> dict:
    m = run.manifest
    issued, completed = run.issued(), run.completed()
    prepare = (run.phase_event("dut.prepare", "end") or {}).get("details", {})
    cudnn = (prepare.get("native") or {}).get("cudnn_plan") or {}
    snaps = {p: (run.snapshot(p) or {}) for p in ("measure_begin", "drain_end")}
    fb, fe = snaps["measure_begin"].get("fields") or {}, snaps["drain_end"].get("fields") or {}
    counters = {}
    for k in ("pcie_rx_bytes", "pcie_tx_bytes", "pcie_replay_counter"):
        if fb.get(k) is not None and fe.get(k) is not None:
            raw = fe[k] - fb[k]
            counters[k] = {"raw_delta": raw, "delta_mod_2_32": raw % (1 << 32)}
    h2d = sum(r.observations.get("nvidia.h2d_bytes", 0) for r in issued)
    created = m.get("created_at_utc", "")
    before = [s for s in states if s[0] <= created]
    state = before[-1][1] if before else ""
    phase = lambda n: (lambda b, e: e["t_run_ns"] - b["t_run_ns"] if b and e else None)(  # noqa: E731
        run.phase_event(n, "begin"), run.phase_event(n, "end"))
    replay = m["comparison_keys"]["components"]["common_parts"]["replay"]
    return {
        "run_id": run.run_id, "experiment_id": run.experiment_id, "created_at_utc": created,
        "replay": replay, "status": {k: m["status"][k] for k in ("execution_status", "measurement_status")},
        "load_average_before": load_avg(state), "gpu_before": state.splitlines()[1] if state.count("\n") >= 1 else None,
        "metrics": {k: run.metric(k) for k in METRICS},
        "host_ns": {"issue_lag": quantiles(r.issue_lag for r in issued),
                    "submit": quantiles(r.submit for r in issued),
                    "after_submit": quantiles(r.after_submit for r in issued),
                    "dut_service": quantiles(r.dut_service for r in issued),
                    "output_step": quantiles(r.output_step for r in completed),
                    "request_completion": quantiles(r.request_completion for r in completed)},
        "device_ns": {"host_to_device": quantiles(r.segment("host_to_device") for r in issued),
                      "device_execution": quantiles(r.segment("device_execution") for r in issued),
                      "device_to_host": quantiles(r.segment("device_to_host") for r in issued),
                      "probe_globaltimer": quantiles(r.in_kernel_ns for r in issued)},
        "probe_sm_mhz": quantiles(r.sm_mhz for r in issued),
        "h2d_bytes_total": h2d, "nvml_counters": counters,
        "snapshots": {p: {k: s.get(k) for k in SNAP} for p, s in snaps.items()},
        "cudnn": {k: cudnn.get(k) for k in ("cudnn_algo_policy", "cudnn_algorithm_table_sha256",
                                            "cudnn_heuristic_agreements", "cudnn_workspace_bytes",
                                            "plan_launches_per_request", "glue_cubin_sha256", "plan_sha256",
                                            "weights_sha256", "frequency_probe_cycles")},
        "lifecycle_host_ns": {"dut.prepare": phase("dut.prepare"), "prepare_inputs": phase("prepare_inputs")},
        "precondition": (m.get("realtime") or {}).get("precondition"),
        "realization": {e["condition"]: f"{e['validity']}|{e['equivalence']}" for e in m["realization"]["evidence"]},
    }


def decompose(r) -> dict:
    return {"request_id": r.request_id, "item_index": r.item_index, "issue_lag": r.issue_lag, "submit": r.submit,
            "after_submit": r.after_submit, "dut_service": r.dut_service, "output_step": r.output_step,
            "h2d": r.segment("host_to_device"), "device_execution": r.segment("device_execution"),
            "d2h": r.segment("device_to_host"), "probe_mhz": r.sm_mhz}


def main() -> int:
    out, roots = sys.argv[1], sys.argv[2:]
    states = load_states(roots)
    runs = [load_run(d) for root in roots for d in run_dirs_under(root)]
    runs.sort(key=lambda r: r.manifest.get("created_at_utc", ""))
    rows = [run_row(r, states) for r in runs]
    by_exp: dict[str, list] = {}
    for r, row in zip(runs, rows):
        by_exp.setdefault(row["experiment_id"], []).append((r, row))
    variability, correlation, warmup, outliers, per_item = {}, {}, {}, {}, {}
    for exp, pairs in by_exp.items():
        series = {
            "dut_service_p50": [row["metrics"]["bmt.latency.dut_service.p50"] for _, row in pairs],
            "dut_service_p99": [row["metrics"]["bmt.latency.dut_service.p99"] for _, row in pairs],
            "request_completion_p50": [row["metrics"]["bmt.latency.request_completion.completed.p50"] for _, row in pairs],
            "output_ready_p50": [row["metrics"]["bmt.latency.output_ready.completed.p50"] for _, row in pairs],
            "output_ready_p99": [row["metrics"]["bmt.latency.output_ready.completed.p99"] for _, row in pairs],
            "issue_lag_max": [row["metrics"]["bmt.issue_lag.max"] for _, row in pairs],
            "submit_p50": [row["host_ns"]["submit"]["p50"] for _, row in pairs],
            "after_submit_p50": [row["host_ns"]["after_submit"]["p50"] for _, row in pairs],
            "output_step_p50": [row["host_ns"]["output_step"]["p50"] for _, row in pairs],
            "h2d_p50": [(row["device_ns"]["host_to_device"] or {}).get("p50") for _, row in pairs],
            "device_execution_p50": [(row["device_ns"]["device_execution"] or {}).get("p50") for _, row in pairs],
            "d2h_p50": [(row["device_ns"]["device_to_host"] or {}).get("p50") for _, row in pairs],
            "probe_mhz_p50": [(row["probe_sm_mhz"] or {}).get("p50") for _, row in pairs],
            "load_1min": [(row["load_average_before"] or [None])[0] for _, row in pairs],
        }
        variability[exp] = {k: {"min": min(v), "max": max(v), "range": max(v) - min(v), **(spread(v) or {})}
                            for k, v in series.items() if v and all(x is not None for x in v)}
        correlation[exp] = {"run_level_vs_dut_service_p50": {k: pearson(series["dut_service_p50"], v)
                                                             for k, v in series.items() if k != "dut_service_p50"},
                            "request_level": []}
        for r, row in pairs:
            iss = r.issued()
            dets = output_detections(r)
            ds = [x.dut_service for x in iss]
            correlation[exp]["request_level"].append({
                "run_id": r.run_id,
                "dut_service_vs": {"submit": pearson(ds, [x.submit for x in iss]),
                                   "after_submit": pearson(ds, [x.after_submit for x in iss]),
                                   "h2d": pearson(ds, [x.segment("host_to_device") for x in iss]),
                                   "device_execution": pearson(ds, [x.segment("device_execution") for x in iss]),
                                   "d2h": pearson(ds, [x.segment("device_to_host") for x in iss]),
                                   "probe_mhz": pearson(ds, [x.sm_mhz for x in iss])},
                "output_step_vs_detections": pearson([x.output_step for x in r.completed()],
                                                     [dets.get(x.request_id) for x in r.completed()]),
                "submit_vs_device_execution": pearson([x.submit for x in iss], [x.segment("device_execution") for x in iss]),
                "detections_per_output": quantiles(dets.values()),
            })
            first = [x for x in iss if x.request_id < 4]
            rest = [x for x in iss if x.request_id >= 4]
            warmup.setdefault(exp, []).append({
                "run_id": r.run_id,
                "first4": {"dut_service": quantiles(x.dut_service for x in first), "submit": quantiles(x.submit for x in first)},
                "rest": {"dut_service": quantiles(x.dut_service for x in rest), "submit": quantiles(x.submit for x in rest)},
                "first_request": decompose(iss[0]) if iss else None})
            top = sorted(iss, key=lambda x: -(x.dut_service or 0))[:3]
            lag = sorted(iss, key=lambda x: -(x.issue_lag or 0))[:2]
            median = sorted(iss, key=lambda x: x.dut_service or 0)[len(iss) // 2] if iss else None
            outliers.setdefault(exp, []).append({"run_id": r.run_id, "median_request": decompose(median) if median else None,
                                                 "slowest_dut_service": [decompose(x) for x in top],
                                                 "largest_issue_lag": [decompose(x) for x in lag]})
        # same image across runs: is per-item DUT-service / output step correlated between repetitions?
        if len(pairs) >= 2:
            per_run = [{x.item_index: x for x in r.completed()} for r, _ in pairs]
            common = sorted(set.intersection(*(set(p) for p in per_run)))
            corr_ds, corr_os = [], []
            for a in range(len(per_run)):
                for b in range(a + 1, len(per_run)):
                    corr_ds.append(pearson([per_run[a][i].dut_service for i in common],
                                           [per_run[b][i].dut_service for i in common]))
                    corr_os.append(pearson([per_run[a][i].output_step for i in common],
                                           [per_run[b][i].output_step for i in common]))
            per_item[exp] = {"items": len(common), "pairwise_pearson_dut_service": quantiles(corr_ds, (50,)),
                             "pairwise_pearson_output_step": quantiles(corr_os, (50,))}
    result = {"schema": "sdv-bmt-stage2b-g-real-analysis/1", "roots": roots, "runs": rows,
              "variability": variability, "correlation": correlation, "warmup": warmup,
              "outliers": outliers, "per_item": per_item}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    us = lambda v: "-" if v is None else f"{v / 1000:.1f}"  # noqa: E731
    print(f"{'experiment':16} {'run':38} load  DUT p50/p99      OR p50/p99      lag max   submit  aft-sub  H2D   exec    D2H  outstep  MHz")
    for row in rows:
        mt, h, d = row["metrics"], row["host_ns"], row["device_ns"]
        print(f"{row['experiment_id'][:16]:16} {row['run_id'][:38]:38} {(row['load_average_before'] or ['-'])[0]!s:>5} "
              f"{us(mt['bmt.latency.dut_service.p50'])}/{us(mt['bmt.latency.dut_service.p99'])}  "
              f"{us(mt['bmt.latency.output_ready.completed.p50'])}/{us(mt['bmt.latency.output_ready.completed.p99'])}  "
              f"{us(mt['bmt.issue_lag.max']):>8} {us(h['submit']['p50']):>7} {us(h['after_submit']['p50']):>7} "
              f"{us((d['host_to_device'] or {}).get('p50')):>5} {us((d['device_execution'] or {}).get('p50')):>7} "
              f"{us((d['device_to_host'] or {}).get('p50')):>5} {us(h['output_step']['p50']):>6} "
              f"{(row['probe_sm_mhz'] or {}).get('p50')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
