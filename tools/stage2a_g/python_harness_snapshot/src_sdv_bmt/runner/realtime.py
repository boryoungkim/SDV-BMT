"""Stage 1R realtime execution: the host monotonic run clock and the MEASURE/DRAIN
loop for polling adapters (design v03 E.1-E.5 semantics on a real clock).

Admission semantics are those of the Stage 1A virtual loop (N=1, queue
capacity 0, ``drop_new``):

1. At ``t_sched`` a request whose item failed preparation becomes NOT_ISSUED;
   otherwise, if the slot is free it is ISSUED and submitted (INFLIGHT), else
   it is DROPPED(no_free_slot).
2. The request occupies the slot until its completion is observed.
3. If ``t_submit_begin + request_timeout_ns`` passes first, the request becomes
   TIMED_OUT, but the slot stays occupied until the device work completes:
   submitted device work cannot be cancelled (Stage 2a-G).
4. That later completion is a ``late_completion``; only then is the slot freed.

DRAIN services the slot until it is free or the run clock reaches
``cap = (end of the issue loop) + drain_ns``.  The first loop iteration at or
after the cap is still serviced, as the virtual loop processes events at
``t <= cap``; with ``drain_ns >= request_timeout_ns`` every issued request is
therefore terminal when DRAIN ends.  A TIMED_OUT request still on the device
at the cap is reported as outstanding.

What real time changes, and nothing else:

* Times are host ``CLOCK_MONOTONIC`` ns relative to the run anchor.
  ``t_issue >= t_sched``; issue lag is a measured fact, not zero by
  construction.  ``t_submit_begin`` is read after admission, just before
  ``submit`` is called.
* ``submit`` returns before completion: ``submit_end`` is the submit return
  time, and ``t_complete`` is the first poll that reports completion.  In one
  loop iteration a completion is checked before the timeout and before an
  arrival (the Stage 1A tie order: returns, then timeouts, then arrivals).  The
  timeout and arrival checks use the clock read at the start of the iteration;
  if the issuer is preempted, both observations are late by the preemption.
* Wait policy ``spin``: the issuer never sleeps inside MEASURE/DRAIN
  (Stage 2a-G: sleeping on a pinned, shared host has millisecond tails).
* Event lines for a request are written after ``submit`` returns, while the
  device works, so recorder cost stays outside the DUT-service interval for
  requests longer than those writes (see the harness floor in
  ``docs/measurement_protocol.md``); the recorder does not flush per line inside
  the window (flushed at DRAIN end).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from ..contracts.protocols import PollingDUTAdapter, Workload
from ..contracts.records import DUTCompletion, DUTRequest, ErrorInfo, InputItem, OutputReference
from ..contracts.specs import RunConfig
from ..contracts.status import ReplayMode, RequestState
from ..observation.recorder import ObservationRecorder
from .core import PreparedStore, RequestTable, RunnerInvariantError, error_info
from .schedule import Schedule
from .virtual import REQUEST_PHASE, MeasureAborted, MeasureSummary

S = RequestState
CLOCK_SOURCE = "CLOCK_MONOTONIC"


class MonotonicRunClock:
    """Run clock: ``CLOCK_MONOTONIC`` ns minus an anchor taken at run start.

    The anchor pair (monotonic, realtime) relates the run clock to wall time for
    tools that need it; durations are only ever computed on the run clock.
    """

    def __init__(self) -> None:
        self.anchor_monotonic_ns = time.monotonic_ns()
        self.anchor_realtime_ns = time.time_ns()

    @property
    def now(self) -> int:
        return time.monotonic_ns() - self.anchor_monotonic_ns

    def advance_to(self, t_ns: int) -> None:
        """Spin until the run clock reaches ``t_ns`` (never moves backwards)."""
        while self.now < t_ns:
            pass

    def advance(self, delta_ns: int) -> None:
        self.advance_to(self.now + delta_ns)

    def describe(self) -> dict[str, Any]:
        return {"domain": "monotonic", "source": CLOCK_SOURCE,
                "anchor_monotonic_ns": self.anchor_monotonic_ns, "anchor_realtime_ns": self.anchor_realtime_ns}


def pin_cpus(cpus: Sequence[int]) -> tuple[set[int], tuple[int, ...]]:
    """Pin this process to ``cpus``; returns (previous affinity, realized affinity)."""
    previous = os.sched_getaffinity(0)
    os.sched_setaffinity(0, set(cpus))
    return previous, tuple(sorted(os.sched_getaffinity(0)))


def restore_cpus(previous: set[int]) -> None:
    os.sched_setaffinity(0, previous)


@dataclass
class _Inflight:
    request_id: int
    item: InputItem
    handle: object
    t_issue: int
    deadline: int


class RealtimeMeasureLoop:
    """MEASURE/DRAIN for a ``PollingDUTAdapter`` on the monotonic run clock."""

    def __init__(
        self,
        *,
        clock: MonotonicRunClock,
        recorder: ObservationRecorder,
        adapter: PollingDUTAdapter,
        workload: Workload,
        items: Sequence[InputItem],
        store: PreparedStore,
        schedule: Schedule,
        run: RunConfig,
        codec: str,
        table: RequestTable,
        before_measure: Callable[[], None] = lambda: None,
        after_drain: Callable[[], None] = lambda: None,
    ) -> None:
        self._clock = clock
        self._rec = recorder
        self._adapter = adapter
        self._workload = workload
        self._items = list(items)
        self._store = store
        self._schedule = schedule
        self._run = run
        self._codec = codec
        self._table = table
        self._before_measure = before_measure
        self._after_drain = after_drain
        self._inflight: Optional[_Inflight] = None
        self._phase = "measure"
        self.issue_lags: list[int] = []  # t_issue - t_sched of issued requests

    # -- public ----------------------------------------------------------------
    def run(self) -> MeasureSummary:
        clock, rec = self._clock, self._rec
        # Backend snapshots are taken before the window opens: they never delay a request.
        self._before_measure()
        rec.set_flush_each(False)
        t_begin = clock.now
        t0 = t_begin + self._schedule.lead_ns
        rec.event("phase", t_begin, phase="measure", edge="begin",
                  details={"planned_requests": len(self._items), "replay_mode": self._schedule.mode,
                           "t0_ns": t0, "wait_policy": "spin"})
        try:
            if self._schedule.mode is ReplayMode.ASAP:
                t_last = self._issue_asap(t0)
            else:
                t_last = self._issue_scheduled(t0)
            t_issue_end = clock.now  # the last admission (and its submit) is complete
            rec.event("phase", t_issue_end, phase="measure", edge="end", outcome="ok")
            self._phase = "drain"
            cap = t_issue_end + self._run.drain_ns
            rec.event("phase", clock.now, phase="drain", edge="begin",
                      details={"drain_ns": self._run.drain_ns, "cap_t_run_ns": cap})
            while self._inflight is not None:
                now = clock.now
                self._service(now)
                if now >= cap:
                    break  # the iteration at the cap was serviced (inclusive cap)
            cap_reached = self._inflight is not None
            outstanding = (self._inflight.request_id,) if self._inflight is not None else ()
            t_end = clock.now
            rec.event("phase", t_end, phase="drain", edge="end", outcome="ok",
                      details={"cap_reached": cap_reached, "outstanding_request_ids": list(outstanding)})
            rec.set_flush_each(True)
            self._after_drain()
        except Exception as exc:
            rec.event("phase", clock.now, phase=self._phase, edge="end", outcome="failed", error=error_info(exc))
            rec.set_flush_each(True)
            raise MeasureAborted(self._phase) from exc
        return MeasureSummary(t_begin, t0, t_last, t_end, cap_reached, outstanding)

    @property
    def inflight_request_id(self) -> Optional[int]:
        return None if self._inflight is None else self._inflight.request_id

    # -- issue loops -------------------------------------------------------------
    def _issue_scheduled(self, t0: int) -> int:
        offsets = self._schedule.offsets_ns
        assert offsets is not None and len(offsets) == len(self._items)
        t_last = t0
        for request_id, item in enumerate(self._items):
            t_sched = t0 + offsets[request_id]
            while True:
                now = self._clock.now
                if self._inflight is not None:
                    self._service(now)
                if now >= t_sched:
                    break
            self._admit(request_id, item, t_sched)
            t_last = t_sched
        return t_last

    def _issue_asap(self, t0: int) -> int:
        self._clock.advance_to(t0)
        t_last = t0
        for request_id, item in enumerate(self._items):
            while self._inflight is not None:
                self._service(self._clock.now)
            t_sched = self._clock.now
            self._admit(request_id, item, t_sched)
            t_last = t_sched
        return t_last

    # -- admission and dispatch ----------------------------------------------------
    def _admit(self, request_id: int, item: InputItem, t_sched: int) -> None:
        now = self._clock.now
        common = {"request_id": request_id, "item_indices": [item.item_index],
                  "phase": REQUEST_PHASE, "t_sched": t_sched}
        failure = self._store.failure(item.item_index)
        if failure is not None:
            self._table.transition(request_id, S.NOT_ISSUED)
            self._rec.event("not_issued", now, reason=f"prepare_failed:{failure.stage}", **common)
            return
        if self._inflight is not None:
            self._table.transition(request_id, S.DROPPED)
            self._rec.event("dropped", now, reason="no_free_slot", **common)
            return
        self._table.transition(request_id, S.ISSUED)
        self.issue_lags.append(now - t_sched)
        self._table.transition(request_id, S.INFLIGHT)
        request = DUTRequest(request_id, (item.item_index,), (self._store.get(item.item_index),))
        t_submit_begin = self._clock.now
        try:
            handle = self._adapter.submit(request)
            failure_info = None
        except Exception as exc:  # submit failure is a request-level error (policy: continue)
            handle, failure_info = None, ErrorInfo("adapter_exception", f"{type(exc).__name__}: {exc}")
        t_submit_end = self._clock.now
        # write after submit returned: recorder cost overlaps device work
        self._rec.event("issued", now, t_issue=now, **common)
        self._rec.event("submit_begin", t_submit_begin, request_id=request_id)
        self._rec.event("submit_end", t_submit_end, request_id=request_id)
        if failure_info is not None:
            self._table.transition(request_id, S.ERROR)
            self._rec.event("error", self._clock.now, request_id=request_id, error_phase="dut_submit",
                            error=failure_info, t_complete=t_submit_end)
            return
        # Stage 1A timeout timer: t_submit_begin + request_timeout_ns
        self._inflight = _Inflight(request_id, item, handle, now, t_submit_begin + self._run.request_timeout_ns)

    # -- completion and timeout --------------------------------------------------------
    def _service(self, now: int) -> None:
        """One polling step for the request in the slot: completion first, then timeout."""
        inflight = self._inflight
        assert inflight is not None
        try:
            completion = self._adapter.poll(inflight.handle)
        except Exception as exc:
            completion = DUTCompletion(inflight.request_id, False, None,
                                       ErrorInfo("adapter_exception", f"{type(exc).__name__}: {exc}"))
        if completion is not None:
            self._on_completion(inflight, completion, self._clock.now)
        elif now >= inflight.deadline and self._table.state(inflight.request_id) is S.INFLIGHT:
            self._table.transition(inflight.request_id, S.TIMED_OUT)
            self._rec.event("timed_out", self._clock.now, request_id=inflight.request_id,
                            timeout_ns=self._run.request_timeout_ns)

    def _finish_output(self, inflight: _Inflight, completion: DUTCompletion
                       ) -> tuple[Optional[OutputReference], Optional[str], Optional[ErrorInfo]]:
        try:
            task_output = self._workload.postprocess(completion.raw_output, inflight.item)
        except Exception as exc:
            return None, "postprocess", error_info(exc)
        try:
            blob = self._workload.encode(task_output)
            if not isinstance(blob, bytes):
                raise TypeError("encode() must return bytes")
        except Exception as exc:
            return None, "encode", error_info(exc)
        return self._rec.output(inflight.request_id, blob, self._codec), None, None

    def _on_completion(self, inflight: _Inflight, completion: DUTCompletion, t_complete: int) -> None:
        rid = inflight.request_id
        if completion.request_id != rid:
            raise RunnerInvariantError(f"completion for request {completion.request_id} while {rid} is in the slot")
        state = self._table.state(rid)
        if state is S.INFLIGHT:
            if not completion.ok:
                self._table.transition(rid, S.ERROR)
                self._rec.event("error", self._clock.now, request_id=rid, error_phase="dut_run",
                                error=completion.error, t_complete=t_complete)
            else:
                ref, phase, err = self._finish_output(inflight, completion)
                t_ready = self._clock.now
                if ref is None:
                    self._table.transition(rid, S.ERROR)
                    self._rec.event("error", t_ready, request_id=rid, error_phase=phase, error=err,
                                    t_complete=t_complete)
                else:
                    self._table.transition(rid, S.COMPLETED)
                    self._rec.event("completed", t_ready, request_id=rid, t_complete=t_complete,
                                    t_output_ready=t_ready, output_ref=ref)
        elif state is S.TIMED_OUT:
            self._table.mark_late_completion(rid)
            if not completion.ok:
                self._rec.event("late_completion", self._clock.now, request_id=rid, result="error",
                                t_complete=t_complete, error_phase="dut_run", error=completion.error)
            else:
                ref, phase, err = self._finish_output(inflight, completion)
                t_ready = self._clock.now
                if ref is None:
                    self._rec.event("late_completion", t_ready, request_id=rid, result="error",
                                    t_complete=t_complete, error_phase=phase, error=err)
                else:
                    self._rec.event("late_completion", t_ready, request_id=rid, result="ok",
                                    t_complete=t_complete, t_output_ready=t_ready, output_ref=ref)
        else:
            raise RunnerInvariantError(f"completion for request {rid} in state {state.value}")
        if completion.device_timing is not None or completion.observations:
            self._rec.event("device_observation", self._clock.now, request_id=rid,
                            timing=completion.device_timing, timing_method=completion.device_timing_method,
                            observations=completion.observations)
        self._inflight = None


def run_precondition(adapter: PollingDUTAdapter, store: PreparedStore, items: Sequence[InputItem],
                     count: int, clock: MonotonicRunClock, timeout_ns: int) -> dict[str, Any]:
    """Untimed requests before MEASURE (device/runtime warm-up), cycling through the
    successfully prepared items.  Not planned requests: they produce no request
    events and no outputs; only this summary is recorded."""
    prepared = [i for i in items if store.failure(i.item_index) is None]
    completed = errors = 0
    if count and not prepared:
        return {"requested": count, "completed": 0, "errors": 0, "note": "no prepared item"}
    for k in range(count):
        item = prepared[k % len(prepared)]
        handle = adapter.submit(DUTRequest(-(k + 1), (item.item_index,), (store.get(item.item_index),)))
        deadline = clock.now + timeout_ns
        while True:
            completion = adapter.poll(handle)
            if completion is not None:
                completed += completion.ok
                errors += not completion.ok
                break
            if clock.now >= deadline:
                raise RuntimeError(f"precondition request {k} did not complete within {timeout_ns} ns")
    return {"requested": count, "completed": completed, "errors": errors}
