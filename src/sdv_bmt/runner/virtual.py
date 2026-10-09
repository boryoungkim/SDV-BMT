"""Deterministic virtual clock, event queue and the Stage 1A MEASURE/DRAIN loop.

Admission semantics (design v03 E.5), N=1, queue capacity 0, ``drop_new``:

1. At ``t_sched`` a request whose item failed preparation becomes NOT_ISSUED;
   otherwise, if the slot is free it is ISSUED and enters ``run()`` (INFLIGHT),
   else it is DROPPED(no_free_slot).
2. The blocking FakeDUT call occupies the slot for ``virtual_service_ns``.
3. If the timeout fires first the request becomes TIMED_OUT, but the slot stays
   occupied until the underlying call returns.
4. That later return is a ``late_completion`` (output still processed and
   stored); only then is the slot released.

Simultaneous events are ordered by (time, priority, insertion order) with
DUT returns before timeouts before arrivals: a call returning exactly at its
timeout instant completes, and a slot freed at ``t`` is available to an arrival
at ``t``.  Host-side steps (postprocess, encode, record) take zero virtual time,
so ``t_output_ready == t_complete`` in Stage 1A.

DRAIN processes queued events up to ``t_last_sched + drain_ns``.  Validation
requires ``drain_ns >= request_timeout_ns`` so every issued request reaches a
terminal state; late returns beyond the cap are reported as outstanding.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from ..contracts.protocols import DUTAdapter, Workload
from ..contracts.records import DUTRequest, DUTResult, ErrorInfo, InputItem, OutputReference
from ..contracts.specs import RunConfig
from ..contracts.status import ReplayMode, RequestState
from ..observation.recorder import ObservationRecorder
from .core import PreparedStore, RequestTable, RunnerInvariantError, error_info
from .schedule import Schedule

S = RequestState

PRIO_DUT_RETURN = 0
PRIO_TIMEOUT = 1

HOST_STEP_VIRTUAL_COST_NS = 0
REQUEST_PHASE = "measure"


class VirtualClock:
    def __init__(self, start_ns: int = 0) -> None:
        self._now = start_ns

    @property
    def now(self) -> int:
        return self._now

    def advance_to(self, t_ns: int) -> None:
        if t_ns < self._now:
            raise RunnerInvariantError(f"virtual clock cannot go backwards ({t_ns} < {self._now})")
        self._now = t_ns

    def advance(self, delta_ns: int) -> None:
        self.advance_to(self._now + delta_ns)


@dataclass(order=True, frozen=True)
class QueuedEvent:
    t_ns: int
    priority: int
    seq: int
    kind: str = field(compare=False)
    request_id: int = field(compare=False)


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[QueuedEvent] = []
        self._seq = 0
        self._cancelled: set[int] = set()

    def push(self, t_ns: int, priority: int, kind: str, request_id: int) -> int:
        handle = self._seq
        heapq.heappush(self._heap, QueuedEvent(t_ns, priority, handle, kind, request_id))
        self._seq += 1
        return handle

    def cancel(self, handle: int) -> None:
        self._cancelled.add(handle)

    def _prune(self) -> None:
        while self._heap and self._heap[0].seq in self._cancelled:
            self._cancelled.discard(heapq.heappop(self._heap).seq)

    def peek_time(self) -> Optional[int]:
        self._prune()
        return self._heap[0].t_ns if self._heap else None

    def pop(self) -> QueuedEvent:
        self._prune()
        return heapq.heappop(self._heap)

    def __len__(self) -> int:
        return sum(1 for e in self._heap if e.seq not in self._cancelled)


class MeasureAborted(RuntimeError):
    def __init__(self, phase: str) -> None:
        super().__init__(f"{phase} aborted")
        self.phase = phase


@dataclass(frozen=True)
class MeasureSummary:
    t_measure_begin: int
    t0: int
    t_last_sched: int
    t_drain_end: int
    drain_cap_reached: bool
    outstanding_request_ids: tuple[int, ...]


class VirtualMeasureLoop:
    def __init__(
        self,
        *,
        clock: VirtualClock,
        recorder: ObservationRecorder,
        adapter: DUTAdapter,
        workload: Workload,
        items: Sequence[InputItem],
        store: PreparedStore,
        schedule: Schedule,
        run: RunConfig,
        codec: str,
        table: RequestTable,
        on_measure_begin: Callable[[], None] = lambda: None,
        on_drain_end: Callable[[], None] = lambda: None,
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
        self._on_measure_begin = on_measure_begin
        self._on_drain_end = on_drain_end
        self._queue = EventQueue()
        self._slot: Optional[int] = None
        self._pending: dict[int, DUTResult] = {}
        self._timeout_handles: dict[int, int] = {}
        self._phase = "measure"

    # -- public --------------------------------------------------------------
    def run(self) -> MeasureSummary:
        clock, rec = self._clock, self._rec
        t_begin = clock.now
        t0 = t_begin + self._schedule.lead_ns
        rec.event(
            "phase", t_begin, phase="measure", edge="begin",
            details={
                "planned_requests": len(self._items),
                "replay_mode": self._schedule.mode,
                "t0_ns": t0,
            },
        )
        try:
            self._on_measure_begin()
            if self._schedule.mode is ReplayMode.ASAP:
                self._issue_asap(t0)
            else:
                self._issue_scheduled(t0)
            t_last = clock.now
            rec.event("phase", t_last, phase="measure", edge="end", outcome="ok")
            self._phase = "drain"
            cap = t_last + self._run.drain_ns
            rec.event("phase", t_last, phase="drain", edge="begin",
                      details={"drain_ns": self._run.drain_ns, "cap_t_run_ns": cap})
            self._process_until(cap)
            cap_reached = len(self._queue) > 0
            if cap_reached:
                clock.advance_to(cap)
            outstanding = tuple(sorted(self._pending))
            self._on_drain_end()
            rec.event(
                "phase", clock.now, phase="drain", edge="end", outcome="ok",
                details={"cap_reached": cap_reached, "outstanding_request_ids": list(outstanding)},
            )
        except Exception as exc:
            rec.event("phase", clock.now, phase=self._phase, edge="end", outcome="failed",
                      error=error_info(exc))
            raise MeasureAborted(self._phase) from exc
        return MeasureSummary(t_begin, t0, t_last, clock.now, cap_reached, outstanding)

    # -- issue loops -----------------------------------------------------------
    def _issue_scheduled(self, t0: int) -> None:
        offsets = self._schedule.offsets_ns
        assert offsets is not None and len(offsets) == len(self._items)
        for request_id, item in enumerate(self._items):
            t_sched = t0 + offsets[request_id]
            self._process_until(t_sched)
            self._clock.advance_to(t_sched)
            self._admit(request_id, item, t_sched)

    def _issue_asap(self, t0: int) -> None:
        self._clock.advance_to(t0)
        for request_id, item in enumerate(self._items):
            while self._slot is not None:
                if self._queue.peek_time() is None:
                    raise RunnerInvariantError("slot busy with no pending virtual event")
                self._process_next()
            self._process_until(self._clock.now)
            self._admit(request_id, item, self._clock.now)

    # -- admission and dispatch -------------------------------------------------
    def _admit(self, request_id: int, item: InputItem, t_sched: int) -> None:
        now = self._clock.now
        common = {"request_id": request_id, "item_indices": [item.item_index],
                  "phase": REQUEST_PHASE, "t_sched": t_sched}
        failure = self._store.failure(item.item_index)
        if failure is not None:
            self._table.transition(request_id, S.NOT_ISSUED)
            self._rec.event("not_issued", now, reason=f"prepare_failed:{failure.stage}", **common)
            return
        if self._slot is not None:
            self._table.transition(request_id, S.DROPPED)
            self._rec.event("dropped", now, reason="no_free_slot", **common)
            return
        self._slot = request_id
        self._table.transition(request_id, S.ISSUED)
        self._rec.event("issued", now, t_issue=now, **common)
        self._dispatch(request_id, item)

    def _dispatch(self, request_id: int, item: InputItem) -> None:
        now = self._clock.now
        self._table.transition(request_id, S.INFLIGHT)
        self._rec.event("submit_begin", now, request_id=request_id)
        request = DUTRequest(request_id, (item.item_index,), (self._store.get(item.item_index),))
        try:
            result = self._adapter.run(request)
            if not isinstance(result, DUTResult) or result.request_id != request_id:
                raise TypeError("adapter returned a result that does not match the request")
        except Exception as exc:  # adapter failure is a request-level error (policy: continue)
            result = DUTResult(request_id, False, None, 0,
                               ErrorInfo("adapter_exception", f"{type(exc).__name__}: {exc}"))
        self._pending[request_id] = result
        self._queue.push(now + result.virtual_service_ns, PRIO_DUT_RETURN, "dut_return", request_id)
        self._timeout_handles[request_id] = self._queue.push(
            now + self._run.request_timeout_ns, PRIO_TIMEOUT, "timeout", request_id
        )

    # -- virtual events ---------------------------------------------------------
    def _process_until(self, t_ns: int) -> None:
        while True:
            t_next = self._queue.peek_time()
            if t_next is None or t_next > t_ns:
                return
            self._process_next()

    def _process_next(self) -> None:
        event = self._queue.pop()
        self._clock.advance_to(event.t_ns)
        if event.kind == "dut_return":
            self._on_dut_return(event.request_id)
        elif event.kind == "timeout":
            self._on_timeout(event.request_id)
        else:  # pragma: no cover - defensive
            raise RunnerInvariantError(f"unknown virtual event {event.kind!r}")

    def _on_timeout(self, request_id: int) -> None:
        self._timeout_handles.pop(request_id, None)
        if self._table.state(request_id) is not S.INFLIGHT:
            raise RunnerInvariantError(f"timeout for request {request_id} that is not inflight")
        self._table.transition(request_id, S.TIMED_OUT)
        self._rec.event("timed_out", self._clock.now, request_id=request_id,
                        timeout_ns=self._run.request_timeout_ns)

    def _finish_output(
        self, request_id: int, item: InputItem, result: DUTResult
    ) -> tuple[Optional[OutputReference], Optional[str], Optional[ErrorInfo]]:
        try:
            task_output = self._workload.postprocess(result.raw_output, item)
        except Exception as exc:
            return None, "postprocess", error_info(exc)
        try:
            blob = self._workload.encode(task_output)
            if not isinstance(blob, bytes):
                raise TypeError("encode() must return bytes")
        except Exception as exc:
            return None, "encode", error_info(exc)
        return self._rec.output(request_id, blob, self._codec), None, None

    def _on_dut_return(self, request_id: int) -> None:
        result = self._pending.pop(request_id)
        now = self._clock.now
        item = self._items[request_id]
        self._rec.event("submit_end", now, request_id=request_id)
        state = self._table.state(request_id)
        if state is S.INFLIGHT:
            self._queue.cancel(self._timeout_handles.pop(request_id))
            if not result.ok:
                self._table.transition(request_id, S.ERROR)
                self._rec.event("error", now, request_id=request_id, error_phase="dut_run",
                                error=result.error, t_complete=now)
            else:
                ref, phase, err = self._finish_output(request_id, item, result)
                if ref is None:
                    self._table.transition(request_id, S.ERROR)
                    self._rec.event("error", now, request_id=request_id, error_phase=phase,
                                    error=err, t_complete=now)
                else:
                    self._table.transition(request_id, S.COMPLETED)
                    self._rec.event("completed", now, request_id=request_id, t_complete=now,
                                    t_output_ready=now + HOST_STEP_VIRTUAL_COST_NS, output_ref=ref)
        elif state is S.TIMED_OUT:
            self._table.mark_late_completion(request_id)
            if not result.ok:
                self._rec.event("late_completion", now, request_id=request_id, result="error",
                                t_complete=now, error_phase="dut_run", error=result.error)
            else:
                ref, phase, err = self._finish_output(request_id, item, result)
                if ref is None:
                    self._rec.event("late_completion", now, request_id=request_id, result="error",
                                    t_complete=now, error_phase=phase, error=err)
                else:
                    self._rec.event("late_completion", now, request_id=request_id, result="ok",
                                    t_complete=now, t_output_ready=now + HOST_STEP_VIRTUAL_COST_NS,
                                    output_ref=ref)
        else:
            raise RunnerInvariantError(f"DUT return for request {request_id} in state {state.value}")
        self._slot = None
