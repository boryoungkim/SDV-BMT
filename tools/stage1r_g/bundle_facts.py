"""Stage 1R-G / 2b-G analysis helpers: read the facts of an SDV-BMT realtime run bundle (analysis plane,
offline; stdlib only).  Nothing here is a benchmark metric: the canonical metrics come from
``sdv-bmt evaluate`` (MetricEvaluator).  These helpers expose the recorded per-request facts so that
diagnostic tables can be built, keeping the clock domains apart:

* host run clock (CLOCK_MONOTONIC - run anchor): t_sched, t_issue, t_submit_begin, t_submit_end,
  t_complete, t_output_ready and every duration between them;
* device clock: the CUDA-event bracket segments (``timing``) and the in-kernel %globaltimer / clock64
  words (``observations``) -- durations only, never subtracted from host timestamps.

Quantiles are nearest-rank (the same rank rule as the evaluator); they are descriptive only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

HOST_FIELDS = ("t_sched", "t_issue", "t_submit_begin", "t_submit_end", "t_complete", "t_output_ready")


def quantiles(values: Iterable[float], points: tuple[int, ...] = (50, 90, 99)) -> Optional[dict[str, Any]]:
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    n = len(v)
    out: dict[str, Any] = {"n": n, "min": v[0]}
    for p in points:
        out[f"p{p}"] = v[min(n - 1, max(0, math.ceil(p * n / 100) - 1))]
    out["max"] = v[-1]
    return out


def spread(values: Iterable[float]) -> Optional[dict[str, Any]]:
    """Descriptive spread: mean, population standard deviation, p10-p90 range (no inference)."""
    v = sorted(x for x in values if x is not None)
    if len(v) < 2:
        return None
    n = len(v)
    mean = sum(v) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / n)
    p10 = v[min(n - 1, max(0, math.ceil(0.10 * n) - 1))]
    p90 = v[min(n - 1, max(0, math.ceil(0.90 * n) - 1))]
    return {"n": n, "mean": mean, "sd": sd, "p10_p90_range": p90 - p10}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@dataclass
class Request:
    request_id: int
    item_index: Optional[int] = None
    outcome: str = "planned"           # completed / error / timed_out / dropped / not_issued / planned
    late: Optional[str] = None         # late completion result ("ok" / "error") of a timed-out request
    t: dict[str, int] = field(default_factory=dict)
    timing: Optional[dict[str, int]] = None        # device-clock segments (CUDA-event brackets)
    timing_method: Optional[str] = None
    observations: dict[str, int] = field(default_factory=dict)

    def host(self, a: str, b: str) -> Optional[int]:
        return self.t[b] - self.t[a] if a in self.t and b in self.t else None

    # host run-clock durations (one domain)
    @property
    def issue_lag(self) -> Optional[int]:
        return self.host("t_sched", "t_issue")

    @property
    def submit(self) -> Optional[int]:
        return self.host("t_submit_begin", "t_submit_end")

    @property
    def dut_service(self) -> Optional[int]:
        return self.host("t_submit_begin", "t_complete")

    @property
    def request_completion(self) -> Optional[int]:
        return self.host("t_sched", "t_complete")

    @property
    def output_step(self) -> Optional[int]:
        return self.host("t_complete", "t_output_ready")

    @property
    def after_submit(self) -> Optional[int]:
        """t_complete - t_submit_end: device work + submit->start wait + completion observation (host clock)."""
        return self.host("t_submit_end", "t_complete")

    # device-clock durations
    def segment(self, name: str) -> Optional[int]:
        return (self.timing or {}).get(name)

    @property
    def in_kernel_ns(self) -> Optional[int]:
        return self.observations.get("nvidia.in_kernel_globaltimer_ns")

    @property
    def sm_mhz(self) -> Optional[float]:
        g, c = self.observations.get("nvidia.in_kernel_globaltimer_ns"), self.observations.get("nvidia.in_kernel_sm_cycles")
        return round(c / g * 1000.0, 3) if g and c is not None else None


@dataclass
class Run:
    run_dir: Path
    manifest: dict[str, Any]
    run_spec: dict[str, Any]
    events: list[dict[str, Any]]
    requests: dict[int, Request]
    phases: list[dict[str, Any]]
    evaluation: Optional[dict[str, Any]] = None       # latest eval_manifest.json
    metrics: Optional[dict[str, Any]] = None          # latest metrics.json, by metric id

    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]

    @property
    def experiment_id(self) -> str:
        return self.manifest["experiment_id"]

    def snapshot(self, point: str) -> Optional[dict[str, Any]]:
        for e in self.phases:
            if e.get("phase") == f"dut.observe.{point}":
                return e.get("details") if e.get("outcome") == "ok" else None
        return None

    def phase_event(self, name: str, edge: str = "end") -> Optional[dict[str, Any]]:
        for e in self.phases:
            if e.get("phase") == name and e.get("edge") == edge:
                return e
        return None

    def metric(self, metric_id: str) -> Any:
        m = (self.metrics or {}).get(metric_id)
        return None if m is None else m.get("value")

    def issued(self) -> list[Request]:
        return [r for r in sorted(self.requests.values(), key=lambda r: r.request_id) if "t_submit_begin" in r.t]

    def completed(self) -> list[Request]:
        return [r for r in self.issued() if r.outcome == "completed"]


def _latest_evaluation(run_dir: Path) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    root = run_dir / "evaluations"
    if not root.is_dir():
        return None, None
    evals = sorted(p for p in root.iterdir() if p.is_dir())
    if not evals:
        return None, None
    last = evals[-1]
    manifest = json.loads((last / "eval_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((last / "metrics.json").read_text(encoding="utf-8"))
    return manifest, {m["metric_id"]: m for m in metrics["metrics"]}


def load_run(run_dir: Path | str) -> Run:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    run_spec = json.loads((run_dir / "run_spec.json").read_text(encoding="utf-8"))
    events = read_jsonl(run_dir / "events.jsonl")
    reqs: dict[int, Request] = {}
    phases = []

    def req(rid: int) -> Request:
        return reqs.setdefault(rid, Request(rid))

    for e in events:
        kind = e.get("kind")
        if kind == "phase":
            phases.append(e)
            continue
        rid = e.get("request_id")
        if rid is None:
            continue
        r = req(rid)
        if kind in ("issued", "dropped", "not_issued"):
            r.item_index = e["item_indices"][0]
            r.t["t_sched"] = e["t_sched"]
            if kind == "issued":
                r.t["t_issue"] = e["t_issue"]
            else:
                r.t["t_issue"] = e["t_run_ns"]
                r.outcome = kind
        elif kind == "submit_begin":
            r.t["t_submit_begin"] = e["t_run_ns"]
        elif kind == "submit_end":
            r.t["t_submit_end"] = e["t_run_ns"]
        elif kind == "completed":
            r.outcome = "completed"
            r.t["t_complete"] = e["t_complete"]
            r.t["t_output_ready"] = e["t_output_ready"]
        elif kind == "error":
            r.outcome = "error"
            r.t["t_complete"] = e["t_complete"]
        elif kind == "timed_out":
            r.outcome = "timed_out"
            r.t["t_timed_out"] = e["t_run_ns"]
        elif kind == "late_completion":
            r.late = e.get("result")
            r.t["t_complete"] = e["t_complete"]
            if e.get("t_output_ready") is not None:
                r.t["t_output_ready"] = e["t_output_ready"]
        elif kind == "device_observation":
            r.timing = e.get("timing")
            r.timing_method = e.get("timing_method")
            r.observations = dict(e.get("observations") or {})
    evaluation, metrics = _latest_evaluation(run_dir)
    return Run(run_dir, manifest, run_spec, events, reqs, phases, evaluation, metrics)


def run_dirs_under(root: Path | str) -> list[Path]:
    """Every run directory (one holding manifest.json) under ``root``, sorted by path."""
    return sorted(p.parent for p in Path(root).rglob("manifest.json") if (p.parent / "events.jsonl").exists())
