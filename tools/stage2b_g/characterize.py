"""Stage 2b-G diagnostic characterization of the actual GPU bundles (analysis plane; stdlib only).

    python3 tools/stage2b_g/characterize.py OUT.json

Descriptive only (nearest-rank quantiles, ranges, standard deviations, Pearson coefficients as a plain
description of co-variation): no hypothesis test, no inferential statistic, no threshold.  Host
run-clock durations and device-clock durations are never subtracted from each other; they are shown
side by side.

Sections: baseline variability (10 warm runs, 2 blocks), per-run co-variation of DUT-service with its
directly timed parts, warm vs cold, H2D contribution (host_pinned vs device_resident, and per-request
spread), buffer lifecycle (per_request_alloc), instrumentation (Stage 1R-G interleaved rounds and 2b-G
blocks), and G-E localization examples (outlier request vs the run's median request).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage1r_g"))
from bundle_facts import Request, Run, load_run, quantiles, run_dirs_under, spread  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
B, R1 = REPO / "runs" / "stage2b-g", REPO / "runs" / "stage1r-g"


def runs(root: Path) -> list[Run]:
    return [load_run(d) for d in run_dirs_under(root)]


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pts) < 3:
        return None
    mx, my = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    sxx, syy = sum((x - mx) ** 2 for x, _ in pts), sum((y - my) ** 2 for _, y in pts)
    return round(sxy / math.sqrt(sxx * syy), 3) if sxx and syy else None


def p(q: dict | None, k: str) -> int | float | None:
    return None if q is None else q[k]


def run_row(run: Run) -> dict:
    iss = run.issued()
    q = lambda f: quantiles(f(r) for r in iss)  # noqa: E731
    sb = run.snapshot("measure_begin") or {}
    return {
        "run_id": run.run_id, "experiment_id": run.experiment_id, "repetition": run.manifest["repetition"],
        "counts": run.manifest["status"]["request_counts"],
        "metrics": {k: run.metric(k) for k in (
            "bmt.latency.dut_service.p50", "bmt.latency.dut_service.p99", "bmt.latency.request_completion.completed.p50",
            "bmt.latency.request_completion.completed.p99", "bmt.latency.output_ready.completed.p50",
            "bmt.latency.output_ready.completed.p99", "bmt.issue_lag.p99", "bmt.issue_lag.max",
            "bmt.throughput.completed_per_s", "bmt.deadline.miss_rate", "bmt.output.identity_rate")},
        "host": {"issue_lag": q(lambda r: r.issue_lag), "submit": q(lambda r: r.submit), "dut_service": q(lambda r: r.dut_service),
                 "after_submit": q(lambda r: r.after_submit), "output_step": quantiles(r.output_step for r in run.completed())},
        "device": {s: q(lambda r, s=s: r.segment(s)) for s in ("host_to_device", "device_execution", "device_to_host")}
        | {"in_kernel": q(lambda r: r.in_kernel_ns)},
        "sm_mhz": quantiles(r.sm_mhz for r in iss),
        "measure_begin": {k: sb.get(k) for k in ("pstate", "sm_clock_mhz", "temperature_c", "pcie_link", "foreign_compute_processes")},
    }


def ranges(rows: list[dict], path: tuple[str, ...]) -> dict:
    vals = []
    for row in rows:
        v = row
        for k in path:
            v = None if v is None else v.get(k)
        vals.append(v)
    vals = [v for v in vals if v is not None]
    return {"min": min(vals), "max": max(vals), "range": max(vals) - min(vals), "n_runs": len(vals)} if vals else {}


def decomposition(run: Run) -> dict:
    """Per-request spread of DUT-service and of each directly timed part (one run), and their descriptive
    co-variation with DUT-service across the run's requests."""
    iss = run.issued()
    dut = [r.dut_service for r in iss]
    parts = {"submit(host)": [r.submit for r in iss], "after_submit(host)": [r.after_submit for r in iss],
             "issue_lag(host)": [r.issue_lag for r in iss],
             "host_to_device(device)": [r.segment("host_to_device") for r in iss],
             "device_execution(device)": [r.segment("device_execution") for r in iss],
             "device_to_host(device)": [r.segment("device_to_host") for r in iss],
             "in_kernel(device)": [r.in_kernel_ns for r in iss]}
    return {"run_id": run.run_id, "dut_service(host)": spread(dut),
            "parts": {k: {"spread": spread(v), "pearson_with_dut_service": pearson(v, dut)} for k, v in parts.items()}}


def delta(o: Request, m: Request) -> dict:
    def d(f):
        a, b = f(o), f(m)
        return None if a is None or b is None else a - b
    return {"dut_service(host)": d(lambda r: r.dut_service), "issue_lag(host)": d(lambda r: r.issue_lag),
            "submit(host)": d(lambda r: r.submit), "after_submit(host)": d(lambda r: r.after_submit),
            "output_step(host)": d(lambda r: r.output_step),
            "host_to_device(device)": d(lambda r: r.segment("host_to_device")),
            "device_execution(device)": d(lambda r: r.segment("device_execution")),
            "device_to_host(device)": d(lambda r: r.segment("device_to_host")),
            "in_kernel(device)": d(lambda r: r.in_kernel_ns), "sm_mhz": d(lambda r: r.sm_mhz)}


def facts(r: Request) -> dict:
    return {"request_id": r.request_id, "item_index": r.item_index, "issue_lag": r.issue_lag, "submit": r.submit,
            "after_submit": r.after_submit, "dut_service": r.dut_service, "output_step": r.output_step,
            "host_to_device": r.segment("host_to_device"), "device_execution": r.segment("device_execution"),
            "device_to_host": r.segment("device_to_host"), "in_kernel": r.in_kernel_ns, "sm_mhz": r.sm_mhz}


def outlier_example(run: Run, rank: int = 0) -> dict:
    iss = sorted(run.issued(), key=lambda r: r.dut_service)
    med, out = iss[len(iss) // 2], iss[-1 - rank]
    return {"run_id": run.run_id, "experiment_id": run.experiment_id, "outlier": facts(out), "median_request": facts(med),
            "delta_outlier_minus_median_ns": delta(out, med)}


def main(out: str) -> None:
    base = runs(B / "baseline")
    cold = runs(B / "cold_start")
    resident = runs(B / "device_resident")
    noev = runs(B / "no_events")
    alloc = runs(B / "per_request_alloc")
    ident = runs(B / "identical_input")
    nsys = runs(B / "nsys")
    gb = runs(B / "g-b-foreign-idle-context")
    res: dict = {}

    rows = [run_row(r) for r in base]
    res["baseline"] = {
        "runs": rows,
        "across_run_ranges_ns": {name: ranges(rows, path) for name, path in {
            "dut_service.p50": ("metrics", "bmt.latency.dut_service.p50"),
            "dut_service.p99": ("metrics", "bmt.latency.dut_service.p99"),
            "request_completion.p50": ("metrics", "bmt.latency.request_completion.completed.p50"),
            "output_ready.p50": ("metrics", "bmt.latency.output_ready.completed.p50"),
            "issue_lag.max": ("metrics", "bmt.issue_lag.max"),
            "submit.p50": ("host", "submit", "p50"),
            "after_submit.p50": ("host", "after_submit", "p50"),
            "output_step.p50": ("host", "output_step", "p50"),
            "h2d_bracket.p50": ("device", "host_to_device", "p50"),
            "exec_bracket.p50": ("device", "device_execution", "p50"),
            "d2h_bracket.p50": ("device", "device_to_host", "p50"),
            "in_kernel.p50": ("device", "in_kernel", "p50")}.items()},
        "per_run_p50_covariation_with_dut_service_p50": {
            part: pearson([r["host"]["dut_service"]["p50"] for r in rows],
                          [p(r[dom][key], "p50") for r in rows])
            for part, (dom, key) in {"submit(host)": ("host", "submit"), "after_submit(host)": ("host", "after_submit"),
                                     "host_to_device(device)": ("device", "host_to_device"),
                                     "device_execution(device)": ("device", "device_execution"),
                                     "in_kernel(device)": ("device", "in_kernel")}.items()},
        "per_request_decomposition": [decomposition(r) for r in base],
    }

    # warm vs cold: first requests of each run vs the rest
    def head_tail(rs: list[Run], k: int = 4) -> dict:
        head = [r.dut_service for run in rs for r in run.issued()[:k]]
        tail = [r.dut_service for run in rs for r in run.issued()[k:]]
        sub_h = [r.submit for run in rs for r in run.issued()[:k]]
        sub_t = [r.submit for run in rs for r in run.issued()[k:]]
        return {"first_requests_per_run": k, "dut_service_first": quantiles(head), "dut_service_rest": quantiles(tail),
                "submit_first": quantiles(sub_h), "submit_rest": quantiles(sub_t),
                "first_request_facts": [facts(run.issued()[0]) for run in rs]}
    res["warm_vs_cold"] = {
        "cold": {"runs": [run_row(r) for r in cold], **head_tail(cold)},
        "warm": head_tail(base),
        "rg_precondition_0_runs_1r_g": head_tail(runs(R1 / "instrumentation" / "rg-gpu-fixed-rate-nominal")),
        "note": "cold = first run of a new process (new context) without preconditioning after the GPU idled in P8; "
                "the measure_begin snapshot shows the device already at P1 (prepare and staging woke it)",
    }

    res["h2d_contribution"] = {
        "host_pinned_runs": [{k: r[k] for k in ("run_id", "metrics", "host", "device")} for r in rows],
        "device_resident_runs": [run_row(r) for r in resident],
        "dut_service_p50_ranges_ns": {"host_pinned": ranges(rows, ("metrics", "bmt.latency.dut_service.p50")),
                                      "device_resident": ranges([run_row(r) for r in resident], ("metrics", "bmt.latency.dut_service.p50"))},
        "submit_p50_ranges_ns": {"host_pinned": ranges(rows, ("host", "submit", "p50")),
                                 "device_resident": ranges([run_row(r) for r in resident], ("host", "submit", "p50"))},
        "device_resident_decomposition": [decomposition(r) for r in resident],
    }

    alloc_rows = [run_row(r) for r in alloc]
    res["buffer_lifecycle"] = {
        "per_request_alloc_runs": alloc_rows,
        "dropped_request_ids": {r.run_id: sorted(q.request_id for q in r.requests.values() if q.outcome == "dropped") for r in alloc},
        "slowest_requests": {r.run_id: [facts(q) for q in sorted(r.issued(), key=lambda q: -q.dut_service)[:6]] for r in alloc},
        "allocations_during_measure": {r.run_id: (r.snapshot("drain_end") or {}).get("device_allocations_total", 0)
                                       - (r.snapshot("measure_begin") or {}).get("device_allocations_total", 0) for r in alloc},
        "baseline_allocations_during_measure": sorted({(r.snapshot("drain_end") or {}).get("device_allocations_total", 0)
                                                       - (r.snapshot("measure_begin") or {}).get("device_allocations_total", 0) for r in base}),
    }

    instr_1rg = {name: [run_row(r) for r in runs(R1 / "instrumentation" / name)]
                 for name in ("rg-gpu-fixed-rate-nominal", "rg-gpu-no-events", "rg-gpu-nvtx")}
    instr_1rg["rg-gpu-nsys-crosscheck"] = [run_row(r) for r in runs(R1 / "nsys")]
    res["instrumentation"] = {
        "stage1r_g_interleaved": {name: {"dut_service_p50": ranges(rs, ("metrics", "bmt.latency.dut_service.p50")),
                                         "dut_service_p99": ranges(rs, ("metrics", "bmt.latency.dut_service.p99")),
                                         "dut_service_max": ranges(rs, ("host", "dut_service", "max")),
                                         "submit_p50": ranges(rs, ("host", "submit", "p50")),
                                         "issue_lag_max": ranges(rs, ("metrics", "bmt.issue_lag.max")),
                                         "dropped": [r["counts"]["dropped"] for r in rs]}
                                  for name, rs in instr_1rg.items()},
        "stage2b_g": {name: {"dut_service_p50": ranges(rs, ("metrics", "bmt.latency.dut_service.p50")),
                             "dut_service_p99": ranges(rs, ("metrics", "bmt.latency.dut_service.p99")),
                             "submit_p50": ranges(rs, ("host", "submit", "p50")),
                             "issue_lag_max": ranges(rs, ("metrics", "bmt.issue_lag.max")),
                             "dropped": [r["counts"]["dropped"] for r in rs]}
                      for name, rs in {"cuda_events (baseline)": rows, "none": [run_row(r) for r in noev],
                                       "cuda_events+nvtx under nsys": [run_row(r) for r in nsys],
                                       "baseline plan + foreign idle context (G-B)": [run_row(r) for r in gb]}.items()},
    }

    res["identical_input"] = {"runs": [run_row(r) for r in ident],
                              "decomposition": [decomposition(r) for r in ident]}
    nsys_1rg = runs(R1 / "nsys")[0]
    res["localization_examples"] = {
        "identical_input_outlier": outlier_example(max(ident, key=lambda r: max(q.dut_service for q in r.issued()))),
        "baseline_largest_outlier": outlier_example(max(base, key=lambda r: max(q.dut_service for q in r.issued()))),
        "per_request_alloc_stall": outlier_example(alloc[0]),
        "profiler_run_outlier_1r_g": outlier_example(nsys_1rg),
    }
    Path(out).write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")
    print("baseline ranges", json.dumps(res["baseline"]["across_run_ranges_ns"], sort_keys=True))
    print("baseline covariation", res["baseline"]["per_run_p50_covariation_with_dut_service_p50"])
    print("cold", json.dumps({k: v for k, v in res["warm_vs_cold"]["cold"].items() if k not in ("runs", "first_request_facts")}))
    print("warm", json.dumps({k: v for k, v in res["warm_vs_cold"]["warm"].items() if k != "first_request_facts"}))
    print("h2d", json.dumps(res["h2d_contribution"]["dut_service_p50_ranges_ns"]), json.dumps(res["h2d_contribution"]["submit_p50_ranges_ns"]))
    print("instr 1R-G", json.dumps(res["instrumentation"]["stage1r_g_interleaved"], sort_keys=True))
    print("instr 2b-G", json.dumps(res["instrumentation"]["stage2b_g"], sort_keys=True))
    for name, ex in res["localization_examples"].items():
        print("loc", name, ex["run_id"], json.dumps(ex["delta_outlier_minus_median_ns"]))
    d0 = res["baseline"]["per_request_decomposition"][0]
    print("decomp", d0["run_id"], json.dumps(d0["dut_service(host)"]), json.dumps({k: (v["spread"] or {}).get("sd") for k, v in d0["parts"].items()}),
          json.dumps({k: v["pearson_with_dut_service"] for k, v in d0["parts"].items()}))


if __name__ == "__main__":
    main(sys.argv[1])
