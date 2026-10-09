"""Pure computations for portable metrics (no I/O)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Sequence

from ..contracts.status import MetricStatus

INF = math.inf


@dataclass(frozen=True)
class Computed:
    status: MetricStatus
    value: Optional[float]
    n: int
    reason: Optional[str] = None
    censored_fraction: Optional[float] = None


def nearest_rank_index(n: int, p: int) -> int:
    """0-based index of the nearest-rank p-th percentile of n sorted values."""
    rank = -(-p * n // 100)  # ceil(p * n / 100) in integer arithmetic
    return max(rank, 1) - 1


def ratio(numerator: int, denominator: int) -> float:
    return float(Fraction(numerator, denominator))


def percentile(values: Sequence[int], p: int, min_samples: int) -> Computed:
    n = len(values)
    if n == 0:
        return Computed(MetricStatus.EMPTY, None, 0, "no_samples")
    if n < min_samples:
        return Computed(MetricStatus.INVALID, None, n, "insufficient_samples")
    return Computed(MetricStatus.VALID, sorted(values)[nearest_rank_index(n, p)], n)


def planned_percentile(values: Sequence[float], p: int, min_completed: int) -> Computed:
    """Percentile over all planned requests with non-completed values = +inf.

    ``min_completed`` is the minimum number of *completed* samples (design v03
    H.3: a p99 needs at least 100 completed samples); it is checked before the
    censoring rule.
    """
    n = len(values)
    if n == 0:
        return Computed(MetricStatus.EMPTY, None, 0, "no_samples")
    non_completed = sum(1 for v in values if v == INF)
    censored = ratio(non_completed, n)
    if n - non_completed < min_completed:
        return Computed(MetricStatus.INVALID, None, n, "insufficient_samples", censored)
    value = sorted(values)[nearest_rank_index(n, p)]
    if value == INF:
        return Computed(MetricStatus.CENSORED, None, n, "rank_in_non_completed_region", censored)
    return Computed(MetricStatus.VALID, value, n, None, censored)


def maximum(values: Sequence[int]) -> Computed:
    if not values:
        return Computed(MetricStatus.EMPTY, None, 0, "no_samples")
    return Computed(MetricStatus.VALID, max(values), len(values))


def rate(count: int, planned: int) -> Computed:
    if planned == 0:
        return Computed(MetricStatus.EMPTY, None, 0, "no_planned_requests")
    return Computed(MetricStatus.VALID, ratio(count, planned), planned)


def per_second(count: int, span_ns: int, n: int) -> Computed:
    if span_ns <= 0:
        return Computed(MetricStatus.INVALID, None, n, "zero_window")
    return Computed(MetricStatus.VALID, float(Fraction(count * 1_000_000_000, span_ns)), n)
