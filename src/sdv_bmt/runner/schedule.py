"""Deterministic replay schedule (design v03 E.4).

Offsets are integer nanoseconds relative to ``t0 = measure_begin + lead_ns``:

* ``fixed_rate``: ``offset_i = floor(i * 10**9 / rate_hz)``
* ``recorded``:   ``offset_i = base_e + floor((t_src_i - t_src_e0) * unit_to_ns / speed)``
  where ``e0`` is the first item of the item's episode, ``base_0 = 0`` and
  ``base_{e+1} = offset(last item of e) + episode_gap_ns`` (gap is run-clock
  time, not scaled by speed).  Source timestamps are never modified.
* ``asap``: no precomputed offsets; ``t_sched`` is the time the slot frees.

``rate_hz`` and ``speed`` are interpreted as the exact decimal value of their
shortest representation (``Fraction(repr(x))``), so the arithmetic is exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Optional, Sequence

from ..contracts.records import InputItem
from ..contracts.specs import ReplayPolicy
from ..contracts.status import InputKind, ReplayMode

NS_PER_S = 1_000_000_000


class ScheduleError(ValueError):
    """The input cannot be scheduled under the requested replay policy."""


def exact_decimal(value: float) -> Fraction:
    return Fraction(repr(float(value)))


@dataclass(frozen=True)
class Schedule:
    mode: ReplayMode
    lead_ns: int
    offsets_ns: Optional[tuple[int, ...]]
    episode_starts: tuple[int, ...]

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "lead_ns": self.lead_ns,
            "planned_offsets_ns": list(self.offsets_ns) if self.offsets_ns is not None else None,
            "episode_start_item_indices": list(self.episode_starts),
        }


def _fixed_rate_offsets(count: int, rate_hz: float) -> tuple[int, ...]:
    rate = exact_decimal(rate_hz)
    return tuple(math.floor(Fraction(i * NS_PER_S) / rate) for i in range(count))


def _episode_blocks(items: Sequence[InputItem]) -> list[list[InputItem]]:
    blocks: list[list[InputItem]] = []
    seen: set = set()
    previous: object = object()
    for item in items:
        coords = item.scenario_coordinates
        episode = coords.episode_id if coords is not None else None
        if episode != previous:
            if episode in seen:
                raise ScheduleError(f"episode {episode!r} is not contiguous in the item sequence")
            seen.add(episode)
            blocks.append([])
            previous = episode
        blocks[-1].append(item)
    return blocks


def _recorded_offsets(
    items: Sequence[InputItem], replay: ReplayPolicy
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    missing = [item.item_index for item in items if item.source_timing is None]
    if missing:
        raise ScheduleError(
            f"recorded replay requires source timing for every item; missing for items {missing[:10]}"
        )
    clocks = {item.source_timing.clock_source for item in items}  # type: ignore[union-attr]
    if len(clocks) != 1:
        raise ScheduleError(f"recorded replay requires a single source clock, got {sorted(clocks)}")
    blocks = _episode_blocks(items)
    if len(blocks) > 1 and replay.episode_gap_ns is None:
        raise ScheduleError("recorded replay of more than one episode requires replay.episode_gap_ns")
    assert replay.speed is not None
    speed = exact_decimal(replay.speed)
    offsets: list[int] = []
    starts: list[int] = []
    base = 0
    for block in blocks:
        starts.append(block[0].item_index)
        t_first = block[0].source_timing.to_ns()  # type: ignore[union-attr]
        previous_delta = -1
        for item in block:
            delta = item.source_timing.to_ns() - t_first  # type: ignore[union-attr]
            if delta <= previous_delta:
                raise ScheduleError(
                    f"source timestamp regression or duplicate at item {item.item_index}"
                )
            previous_delta = delta
            offsets.append(base + math.floor(Fraction(delta) / speed))
        base = offsets[-1] + (replay.episode_gap_ns or 0)
    return tuple(offsets), tuple(starts)


def compute_schedule(
    items: Sequence[InputItem], replay: ReplayPolicy, input_kind: InputKind
) -> Schedule:
    if not items:
        raise ScheduleError("no items to schedule")
    if [item.item_index for item in items] != list(range(len(items))):
        raise ScheduleError("item_index values must be 0..n-1 in order")
    if replay.mode is ReplayMode.ASAP:
        return Schedule(ReplayMode.ASAP, replay.lead_ns, None, ())
    if replay.mode is ReplayMode.FIXED_RATE:
        assert replay.rate_hz is not None
        return Schedule(
            ReplayMode.FIXED_RATE, replay.lead_ns, _fixed_rate_offsets(len(items), replay.rate_hz), ()
        )
    if input_kind is InputKind.DATASET:
        raise ScheduleError("recorded replay is not allowed for dataset input")
    offsets, starts = _recorded_offsets(items, replay)
    return Schedule(ReplayMode.RECORDED, replay.lead_ns, offsets, starts)
