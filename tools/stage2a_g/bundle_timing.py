"""Stage 2a-G: host-timing quantiles of SDV-BMT realtime run bundles (Python harness), comparable with
probe_summary.py output (investigation tool; stdlib only).

    python3 tools/stage2a_g/bundle_timing.py OUT.json RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def q(values):
    if not values:
        return {"n": 0}
    v = sorted(values)
    n = len(v)
    pick = lambda p: v[min(n - 1, max(0, -(-p * n // 100) - 1))]  # noqa: E731
    return {"n": n, "min": v[0], "p50": pick(50), "p90": pick(90), "p99": pick(99), "max": v[-1]}


def summarize(run_dir: Path) -> dict:
    ev = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    req: dict = {}
    for e in ev:
        if "request_id" in e and e["kind"] != "device_observation":
            req.setdefault(e["request_id"], {})[e["kind"]] = e
    issued = [r for r in req.values() if "issued" in r]
    done = [r for r in req.values() if "completed" in r]
    drops = sorted(rid for rid, r in req.items() if "dropped" in r)
    lag = [r["issued"]["t_issue"] - r["issued"]["t_sched"] for r in issued]
    sub = [r["submit_end"]["t_run_ns"] - r["submit_begin"]["t_run_ns"] for r in issued]
    svc = [r["completed"]["t_complete"] - r["submit_begin"]["t_run_ns"] for r in done]
    ready = [r["completed"]["t_output_ready"] - r["completed"]["t_complete"] for r in done]
    worst = sorted(issued, key=lambda r: r["issued"]["t_issue"] - r["issued"]["t_sched"])[-3:]
    return {"run": str(run_dir), "issued": len(issued), "completed": len(done), "dropped": drops,
            "issue_lag_ns": q(lag), "submit_call_ns": q(sub), "dut_service_ns": q(svc),
            "output_ready_minus_complete_ns": q(ready),
            "worst_issue_lag": [{"request_id": r["issued"]["request_id"],
                                 "lag_ns": r["issued"]["t_issue"] - r["issued"]["t_sched"]} for r in worst]}


def main(out_path, dirs):
    table = [summarize(Path(d)) for d in dirs]
    Path(out_path).write_text(json.dumps(table, indent=1, sort_keys=True))
    for t in table:
        print(Path(t["run"]).name, "issued", t["issued"], "dropped", t["dropped"],
              "| lag p50/p99/max", t["issue_lag_ns"].get("p50"), t["issue_lag_ns"].get("p99"), t["issue_lag_ns"].get("max"),
              "| submit p50/p99/max", t["submit_call_ns"].get("p50"), t["submit_call_ns"].get("p99"), t["submit_call_ns"].get("max"),
              "| svc p50/p99/max", t["dut_service_ns"].get("p50"), t["dut_service_ns"].get("p99"), t["dut_service_ns"].get("max"),
              "| ready-complete p50", t["output_ready_minus_complete_ns"].get("p50"), "| worst lag", t["worst_issue_lag"][-1])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
