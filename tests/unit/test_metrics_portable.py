"""Portable metric arithmetic: nearest rank, sample rules, censoring, exact ratios."""

from __future__ import annotations

import math

import pytest

from sdv_bmt.contracts.status import MetricStatus
from sdv_bmt.metrics.portable import (
    INF,
    maximum,
    nearest_rank_index,
    per_second,
    percentile,
    planned_percentile,
    rate,
)


@pytest.mark.parametrize(
    "n,p,index",
    [(1, 50, 0), (1, 99, 0), (2, 50, 0), (2, 90, 1), (4, 50, 1), (4, 90, 3), (10, 50, 4),
     (10, 90, 8), (100, 99, 98), (100, 50, 49), (101, 99, 99), (200, 99, 197)],
)
def test_nearest_rank_index(n, p, index):
    assert nearest_rank_index(n, p) == index
    assert nearest_rank_index(n, p) == math.ceil(p * n / 100) - 1


def test_percentile_sample_rules():
    assert percentile([], 50, 1).status is MetricStatus.EMPTY
    small = percentile(list(range(99)), 99, 100)
    assert (small.status, small.value, small.reason) == (MetricStatus.INVALID, None, "insufficient_samples")
    ok = percentile(list(range(100, 0, -1)), 99, 100)
    assert (ok.status, ok.value, ok.n) == (MetricStatus.VALID, 99, 100)
    assert percentile([30, 10, 20], 50, 1).value == 20


def test_planned_percentile_censoring():
    values = [10, 20, INF, INF]
    p50 = planned_percentile(values, 50, 1)
    assert (p50.status, p50.value, p50.censored_fraction) == (MetricStatus.VALID, 20, 0.5)
    p90 = planned_percentile(values, 90, 1)
    assert (p90.status, p90.value, p90.censored_fraction) == (MetricStatus.CENSORED, None, 0.5)
    assert planned_percentile([], 50, 1).status is MetricStatus.EMPTY
    insufficient = planned_percentile(values, 99, 100)
    assert insufficient.status is MetricStatus.INVALID and insufficient.censored_fraction == 0.5
    none_censored = planned_percentile([5, 6], 99, 1)
    assert none_censored.status is MetricStatus.VALID and none_censored.censored_fraction == 0.0


def test_rates_are_exact_and_empty_when_nothing_planned():
    assert rate(1, 3).value == 1 / 3
    assert rate(0, 0).status is MetricStatus.EMPTY and rate(0, 0).value is None
    assert per_second(8, 375_000_000, 8).value == 8 / 0.375
    # IEEE division of two exactly representable integers is correctly rounded.
    assert per_second(3, 99_000_000, 4).value == 3e9 / 99e6
    assert per_second(1, 0, 1).status is MetricStatus.INVALID
    assert maximum([]).status is MetricStatus.EMPTY and maximum([3, 9, 1]).value == 9
