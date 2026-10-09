"""Stage 1A portable metric catalog (subset of design v03 H.2, version 1).

All latencies are integer nanoseconds on the run clock.  Percentiles use the
nearest-rank method: ``rank = ceil(p/100 * n)`` (1-based) over sorted values.
A p99 needs at least 100 completed samples, otherwise
``invalid(insufficient_samples)``; this also applies to the planned p99.
Planned request-completion percentiles treat every non-completed planned
request as +inf; if the rank lands on +inf the result is ``censored`` with
``value=null`` and is report-only.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts.records import METRIC_CATALOG_VERSION

CATALOG_VERSION = METRIC_CATALOG_VERSION
PERCENTILES = (50, 90, 99)
P99_MIN_SAMPLES = 100
PERCENTILE_METHOD = "nearest_rank: rank = ceil(p/100 * n), 1-based, over ascending values"


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    definition_version: int
    unit: str
    method: str


def _defs() -> list[MetricDefinition]:
    out = [
        MetricDefinition(f"bmt.count.{name}", 1, "count", "count of measure-phase planned requests by state")
        for name in ("planned", "issued", "completed", "dropped", "timed_out", "error", "not_issued")
    ]
    out.append(MetricDefinition(
        "bmt.diagnostic.late_completion.count", 1, "count",
        "timed-out requests whose underlying call later returned (not counted as completed)",
    ))
    out += [
        MetricDefinition(f"bmt.rate.{name}", 1, "ratio", f"count.{src} / count.planned")
        for name, src in (("completion", "completed"), ("drop", "dropped"),
                          ("timeout", "timed_out"), ("error", "error"))
    ]
    for p in PERCENTILES:
        out += [
            MetricDefinition(f"bmt.latency.request_completion.completed.p{p}", 1, "ns",
                             f"{PERCENTILE_METHOD}; values t_complete - t_sched of COMPLETED requests"),
            MetricDefinition(f"bmt.latency.request_completion.planned.p{p}", 1, "ns",
                             f"{PERCENTILE_METHOD}; values t_complete - t_sched, non-completed planned = +inf; "
                             "p99 needs >= 100 completed samples; rank on +inf -> censored (report-only)"),
            MetricDefinition(f"bmt.latency.dut_service.p{p}", 1, "ns",
                             f"{PERCENTILE_METHOD}; values t_complete - t_submit_begin of COMPLETED requests"),
            MetricDefinition(f"bmt.latency.output_ready.completed.p{p}", 1, "ns",
                             f"{PERCENTILE_METHOD}; values t_output_ready - t_sched of COMPLETED requests"),
        ]
    out += [
        MetricDefinition("bmt.issue_lag.p99", 1, "ns", f"{PERCENTILE_METHOD}; values t_issue - t_sched of issued requests"),
        MetricDefinition("bmt.issue_lag.max", 1, "ns", "max(t_issue - t_sched) of issued requests"),
        MetricDefinition("bmt.throughput.completed_per_s", 1, "1/s",
                         "count.completed / ((t_drain_end - t0) / 1e9), t0 = first planned t_sched; "
                         "configured-N throughput"),
        MetricDefinition("bmt.offered_rate_per_s", 1, "1/s",
                         "(planned - 1) / ((t_sched_last - t_sched_first) / 1e9); fixed_rate/recorded only"),
        MetricDefinition("bmt.deadline.miss_rate", 1, "ratio",
                         "planned requests not COMPLETED with t_complete <= t_sched + deadline.period_ns, / planned"),
        MetricDefinition("bmt.output.identity_rate", 1, "ratio",
                         "COMPLETED outputs whose sha256 equals the designated reference for the same item, "
                         "/ COMPLETED outputs that have a reference entry"),
    ]
    return out


METRICS: tuple[MetricDefinition, ...] = tuple(_defs())
BY_ID = {m.metric_id: m for m in METRICS}
