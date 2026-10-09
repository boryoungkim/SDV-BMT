"""Deterministic append-only ObservationRecorder.

A plain synchronous writer: no writer thread, no queue.  Files are created
exclusively (never overwritten), lines are canonical JSON, and run-clock time
may never go backwards.  Every write is flushed unless the realtime runner
switches per-write flushing off for MEASURE/DRAIN (Stage 1R hot path); lines
are still written in order and flushed at the end of that window.  The
recorder stores facts only; it computes no metric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, Sequence

from ..contracts.ids import canonical_json_bytes, sha256_hex
from ..contracts.records import InputItem, OutputReference, StressRecord
from ..contracts.status import RunClockDomain
from .events import EVENT_SCHEMA_VERSION, STRESS_KIND, make_event

EVENTS_FILE = "events.jsonl"
STRESS_FILE = "stress.jsonl"
OUTPUTS_FILE = "outputs.bin"
INPUT_ITEMS_FILE = "input_items.jsonl"


class RecorderError(RuntimeError):
    pass


class ObservationRecorder:
    def __init__(self, run_dir: Path, clock_domain: RunClockDomain = RunClockDomain.VIRTUAL,
                 schema_version: int = EVENT_SCHEMA_VERSION) -> None:
        self._dir = Path(run_dir)
        self._domain = clock_domain.value
        self._schema_version = schema_version
        self._flush_each = True
        self._events: BinaryIO = open(self._dir / EVENTS_FILE, "xb")
        self._stress: BinaryIO = open(self._dir / STRESS_FILE, "xb")
        self._outputs: BinaryIO = open(self._dir / OUTPUTS_FILE, "xb")
        self._seq = 0
        self._last_t = 0
        self._output_offset = 0
        self._closed = False

    @property
    def next_seq(self) -> int:
        return self._seq

    def write_input_items(self, items: Sequence[InputItem]) -> None:
        with open(self._dir / INPUT_ITEMS_FILE, "xb") as fh:
            for item in items:
                fh.write(canonical_json_bytes(item) + b"\n")

    def _append(self, fh: BinaryIO, kind: str, t_run_ns: int, fields: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RecorderError("recorder is closed")
        if type(t_run_ns) is not int or t_run_ns < self._last_t:
            raise RecorderError(f"run clock went backwards: {t_run_ns} < {self._last_t}")
        event = make_event(kind, self._seq, t_run_ns, self._domain, fields, self._schema_version)
        fh.write(canonical_json_bytes(event) + b"\n")
        if self._flush_each:
            fh.flush()
        self._seq += 1
        self._last_t = t_run_ns
        return event

    def event(self, kind: str, t_run_ns: int, **fields: Any) -> dict[str, Any]:
        if kind == STRESS_KIND:
            raise RecorderError("stress records go through stress()")
        return self._append(self._events, kind, t_run_ns, fields)

    def stress(self, t_run_ns: int, record: StressRecord) -> dict[str, Any]:
        fields = {
            "step": record.step,
            "outcome": record.outcome,
            "provider": record.provider,
            "stress_axis_digest": record.stress_axis_digest,
            "stress_level": record.stress_level,
            "requested": record.requested,
            "achieved": record.achieved,
            "independent_evidence": record.independent_evidence,
            "verification": record.verification,
            "error": record.error,
            "details": record.details,
        }
        return self._append(self._stress, STRESS_KIND, t_run_ns, fields)

    def set_flush_each(self, flush_each: bool) -> None:
        """Per-write flushing on (default) or off; switching it on flushes everything."""
        self._flush_each = flush_each
        if flush_each:
            self.flush()

    def flush(self) -> None:
        if not self._closed:
            for fh in (self._events, self._stress, self._outputs):
                fh.flush()

    def output(self, request_id: int, blob: bytes, codec: str) -> OutputReference:
        if self._closed:
            raise RecorderError("recorder is closed")
        ref = OutputReference(
            request_id=request_id,
            file=OUTPUTS_FILE,
            offset=self._output_offset,
            length=len(blob),
            sha256=sha256_hex(blob),
            codec=codec,
        )
        self._outputs.write(blob)
        if self._flush_each:
            self._outputs.flush()
        self._output_offset += len(blob)
        return ref

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for fh in (self._events, self._stress, self._outputs):
            fh.close()
