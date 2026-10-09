"""Stage 1R realtime support in the Python control plane.

The realtime MEASURE/DRAIN loop runs in the native measurement plane (``native/``, C++; approved
architecture "C++ Measurement Plane, Python Analysis Plane").  Python keeps only:

* ``MonotonicRunClock``: the run clock shared with the native process.  Both read host
  ``CLOCK_MONOTONIC``; the anchor taken at run start is handed to the native process, so lifecycle
  events written by Python and request events written natively lie on one run clock.
* ``read_window_facts``: the facts of the native window read back from the bundle's own
  ``events.jsonl`` (never from an in-memory reply), for the Python interpretation of realization
  evidence.

The admission, timeout and drain semantics executed natively are those of Stage 1A (see
``native/src/common/loop.hpp`` and ``docs/measurement_protocol.md``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..observation.reader import read_jsonl
from ..observation.recorder import EVENTS_FILE

CLOCK_SOURCE = "CLOCK_MONOTONIC"


class MonotonicRunClock:
    """Run clock: ``CLOCK_MONOTONIC`` ns minus an anchor taken at run start.

    The anchor pair (monotonic, realtime) relates the run clock to wall time for tools that need it;
    durations are only ever computed on the run clock.
    """

    def __init__(self) -> None:
        self.anchor_monotonic_ns = time.monotonic_ns()
        self.anchor_realtime_ns = time.time_ns()

    @property
    def now(self) -> int:
        return time.monotonic_ns() - self.anchor_monotonic_ns

    def advance_to(self, t_ns: int) -> None:
        """Spin until the run clock reaches ``t_ns`` (used only for a stress verify window)."""
        while self.now < t_ns:
            pass

    def advance(self, delta_ns: int) -> None:
        self.advance_to(self.now + delta_ns)

    def describe(self) -> dict[str, Any]:
        return {"domain": "monotonic", "source": CLOCK_SOURCE,
                "anchor_monotonic_ns": self.anchor_monotonic_ns, "anchor_realtime_ns": self.anchor_realtime_ns}


@dataclass(frozen=True)
class WindowFacts:
    """Facts of one native measurement window, read from events.jsonl."""

    snapshots: dict[str, Optional[dict[str, Any]]] = field(default_factory=dict)
    device_observations: tuple[dict[str, Any], ...] = ()
    issue_lags_ns: tuple[int, ...] = ()


def read_window_facts(run_dir: Path) -> WindowFacts:
    snapshots: dict[str, Optional[dict[str, Any]]] = {}
    observations = []
    lags = []
    for e in read_jsonl(Path(run_dir) / EVENTS_FILE):
        kind = e.get("kind")
        if kind == "phase" and str(e.get("phase", "")).startswith("dut.observe."):
            point = e["phase"][len("dut.observe."):]
            snapshots[point] = e.get("details") if e.get("outcome") == "ok" else None
        elif kind == "device_observation":
            observations.append(e)
        elif kind == "issued":
            lags.append(e["t_issue"] - e["t_sched"])
    return WindowFacts(snapshots, tuple(observations), tuple(lags))
