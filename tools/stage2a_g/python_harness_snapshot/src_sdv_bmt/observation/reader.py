"""Read-only access to a run bundle and per-request folding of raw events.

Checks applied on load: strict JSON (no duplicate keys/NaN), manifest and
input-item schemas, event schema per kind, one shared ``seq`` counter that is
exactly ``0..N-1`` across events.jsonl and stress.jsonl, run-clock time that
never decreases in ``seq`` order, and a single run-clock domain.

``fold_requests`` rebuilds RequestRecord/CompletionRecord views and rejects
illegal event sequences (e.g. a completion without submit, a late completion
without a timeout, two admissions for one request).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..contracts.ids import loads_strict_json
from ..contracts.records import (
    MANIFEST_SCHEMA_VERSIONS,
    REALTIME_RECORD_SCHEMA_VERSION,
    CompletionRecord,
    ErrorInfo,
    InputItem,
    LateCompletion,
    OutputReference,
    RequestRecord,
    RunManifest,
)
from ..contracts.specs import SpecError, parse_strict
from ..contracts.status import ReplayMode, RequestState
from .events import EVENT_SCHEMA_VERSION, REALTIME_EVENT_SCHEMA_VERSION, STRESS_KIND, validate_event
from .recorder import EVENTS_FILE, INPUT_ITEMS_FILE, OUTPUTS_FILE, STRESS_FILE

S = RequestState
MANIFEST_FILE = "manifest.json"
RUN_SPEC_FILE = "run_spec.json"
RUN_SPEC_SCHEMA_VERSION = 2
REALTIME_RUN_SPEC_SCHEMA_VERSION = 3  # Stage 1R: adds the realtime section


class BundleReadError(ValueError):
    pass


@dataclass(frozen=True)
class RunBundle:
    run_dir: Path
    manifest: dict[str, Any]
    run_spec: dict[str, Any]
    input_items: tuple[InputItem, ...]
    events: tuple[dict[str, Any], ...]
    stress: tuple[dict[str, Any], ...]


def _read_json(path: Path) -> Any:
    try:
        return loads_strict_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BundleReadError(f"{path.name}: {exc}") from None


def read_jsonl(path: Path) -> list[Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BundleReadError(f"{path.name}: {exc}") from None
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    out = []
    for lineno, line in enumerate(lines, start=1):
        try:
            out.append(loads_strict_json(line))
        except ValueError as exc:
            raise BundleReadError(f"{path.name}:{lineno}: {exc}") from None
    return out


def _check_records(path: Path, records: list[Any], stress: bool) -> None:
    for i, record in enumerate(records, start=1):
        problems = validate_event(record)
        if problems:
            raise BundleReadError(f"{path.name}:{i}: {'; '.join(problems)}")
        if (record["kind"] == STRESS_KIND) != stress:
            raise BundleReadError(f"{path.name}:{i}: kind {record['kind']!r} does not belong in this file")
    seqs = [r["seq"] for r in records]
    if seqs != sorted(seqs) or len(set(seqs)) != len(seqs):
        raise BundleReadError(f"{path.name}: seq is not strictly increasing")


def load_run_bundle(run_dir: Path | str) -> RunBundle:
    run_dir = Path(run_dir)
    manifest = _read_json(run_dir / MANIFEST_FILE)
    version = manifest.get("schema_version") if isinstance(manifest, dict) else None
    if version not in MANIFEST_SCHEMA_VERSIONS:
        raise BundleReadError(
            f"manifest.json schema_version {version!r} is not supported "
            f"(this reader reads {', '.join(map(str, MANIFEST_SCHEMA_VERSIONS))})"
        )
    realtime = version == REALTIME_RECORD_SCHEMA_VERSION
    expected_run_spec = REALTIME_RUN_SPEC_SCHEMA_VERSION if realtime else RUN_SPEC_SCHEMA_VERSION
    expected_events = REALTIME_EVENT_SCHEMA_VERSION if realtime else EVENT_SCHEMA_VERSION
    try:
        parse_strict(RunManifest, manifest, "manifest")
    except SpecError as exc:
        raise BundleReadError(f"manifest.json: {exc}") from None
    run_spec = _read_json(run_dir / RUN_SPEC_FILE)
    if not isinstance(run_spec, dict) or run_spec.get("schema_version") != expected_run_spec:
        raise BundleReadError(f"run_spec.json schema_version is not {expected_run_spec}")
    try:
        run_spec["evaluation"]["deadline"]
        run_spec["evaluation"]["output_reference"]["kind"]
        mode = run_spec["schedule"]["mode"]
    except (KeyError, TypeError):
        raise BundleReadError("run_spec.json lacks evaluation/schedule sections") from None
    if mode not in {m.value for m in ReplayMode}:
        raise BundleReadError(f"run_spec.json: unknown replay mode {mode!r}")
    items = []
    for i, row in enumerate(read_jsonl(run_dir / INPUT_ITEMS_FILE), start=1):
        try:
            items.append(parse_strict(InputItem, row, f"input_items:{i}"))
        except SpecError as exc:
            raise BundleReadError(str(exc)) from None
    events = read_jsonl(run_dir / EVENTS_FILE)
    stress = read_jsonl(run_dir / STRESS_FILE)
    _check_records(run_dir / EVENTS_FILE, events, stress=False)
    _check_records(run_dir / STRESS_FILE, stress, stress=True)
    merged = sorted(events + stress, key=lambda r: r["seq"])
    if [r["seq"] for r in merged] != list(range(len(merged))):
        raise BundleReadError("seq across events.jsonl and stress.jsonl is not exactly 0..N-1")
    times = [r["t_run_ns"] for r in merged]
    if any(b < a for a, b in zip(times, times[1:])):
        raise BundleReadError("run-clock time decreases in seq order")
    if len({r["run_clock_domain"] for r in merged}) > 1:
        raise BundleReadError("mixed run-clock domains in one run")
    if any(r["schema_version"] != expected_events for r in merged):
        raise BundleReadError(f"manifest v{version} requires event schema version {expected_events}")
    if not (run_dir / OUTPUTS_FILE).is_file():
        raise BundleReadError("outputs.bin is missing")
    return RunBundle(run_dir, manifest, run_spec, tuple(items), tuple(events), tuple(stress))


def read_output_blob(run_dir: Path, ref: OutputReference) -> bytes:
    with open(Path(run_dir) / ref.file, "rb") as fh:
        fh.seek(ref.offset)
        return fh.read(ref.length)


def restore_recorded_ok(run_dir: Path) -> bool:
    """True when stress.jsonl of a run contains a successful restore record."""
    path = Path(run_dir) / STRESS_FILE
    if not path.is_file():
        return False
    try:
        records = read_jsonl(path)
    except BundleReadError:
        return False
    return any(r.get("step") == "restore" and r.get("outcome") == "ok" for r in records)


# --------------------------------------------------------------------------
# folding
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasureWindow:
    t_measure_begin: int
    t_drain_end: int
    drain_cap_reached: bool
    outstanding_request_ids: tuple[int, ...]


def measure_window(events: tuple[dict[str, Any], ...]) -> Optional[MeasureWindow]:
    begin = [e for e in events if e["kind"] == "phase" and e["phase"] == "measure" and e["edge"] == "begin"]
    end = [e for e in events if e["kind"] == "phase" and e["phase"] == "drain" and e["edge"] == "end"
           and e.get("outcome") == "ok"]
    if not begin or not end:
        return None
    details = end[0].get("details", {})
    return MeasureWindow(
        begin[0]["t_run_ns"],
        end[0]["t_run_ns"],
        bool(details.get("cap_reached")),
        tuple(details.get("outstanding_request_ids", ())),
    )


def _output_ref(data: Optional[dict[str, Any]]) -> Optional[OutputReference]:
    return None if data is None else parse_strict(OutputReference, data, "output_ref")


def _error(data: Optional[dict[str, Any]]) -> Optional[ErrorInfo]:
    return None if data is None else parse_strict(ErrorInfo, data, "error")


@dataclass
class _Acc:
    admission: dict[str, Any]
    state: RequestState
    t_submit_begin: Optional[int] = None
    t_submit_end: Optional[int] = None
    completion: Optional[dict[str, Any]] = None
    t_timed_out: Optional[int] = None
    late: Optional[dict[str, Any]] = None


def fold_requests(events: tuple[dict[str, Any], ...]) -> dict[int, tuple[RequestRecord, Optional[CompletionRecord]]]:
    acc: dict[int, _Acc] = {}

    def need(event: dict[str, Any], *states: RequestState) -> _Acc:
        rid = event["request_id"]
        if rid not in acc or acc[rid].state not in states:
            current = acc[rid].state.value if rid in acc else "unknown"
            raise BundleReadError(f"request {rid}: {event['kind']} not allowed in state {current}")
        return acc[rid]

    for e in events:
        kind = e["kind"]
        if kind in ("issued", "dropped", "not_issued"):
            if e["request_id"] in acc:
                raise BundleReadError(f"request {e['request_id']}: second admission event")
            state = {"issued": S.ISSUED, "dropped": S.DROPPED, "not_issued": S.NOT_ISSUED}[kind]
            acc[e["request_id"]] = _Acc(admission=e, state=state)
        elif kind == "submit_begin":
            a = need(e, S.ISSUED)
            a.state, a.t_submit_begin = S.INFLIGHT, e["t_run_ns"]
        elif kind == "submit_end":
            a = need(e, S.INFLIGHT, S.TIMED_OUT)
            if a.t_submit_end is not None:
                raise BundleReadError(f"request {e['request_id']}: second submit_end")
            a.t_submit_end = e["t_run_ns"]
        elif kind == "completed":
            a = need(e, S.INFLIGHT)
            if a.t_submit_end is None:
                raise BundleReadError(f"request {e['request_id']}: completed before submit_end")
            a.state, a.completion = S.COMPLETED, e
        elif kind == "error":
            a = need(e, S.ISSUED, S.INFLIGHT)
            a.state, a.completion = S.ERROR, e
        elif kind == "timed_out":
            a = need(e, S.INFLIGHT)
            a.state, a.t_timed_out = S.TIMED_OUT, e["t_run_ns"]
        elif kind == "late_completion":
            a = need(e, S.TIMED_OUT)
            if a.late is not None or a.t_submit_end is None:
                raise BundleReadError(f"request {e['request_id']}: unexpected late_completion")
            a.late = e

    folded: dict[int, tuple[RequestRecord, Optional[CompletionRecord]]] = {}
    for rid in sorted(acc):
        a = acc[rid]
        adm = a.admission
        request = RequestRecord(
            request_id=rid,
            item_indices=tuple(adm["item_indices"]),
            phase=adm["phase"],
            t_sched=adm["t_sched"],
            admission=adm["kind"],
            admission_reason=adm.get("reason"),
            t_issue=adm.get("t_issue"),
            t_submit_begin=a.t_submit_begin,
            t_submit_end=a.t_submit_end,
            state=a.state,
        )
        completion = None
        if a.state in (S.COMPLETED, S.ERROR, S.TIMED_OUT):
            c = a.completion or {}
            late = None
            if a.late is not None:
                late = LateCompletion(
                    result=a.late["result"],
                    t_complete=a.late["t_complete"],
                    t_output_ready=a.late.get("t_output_ready"),
                    output_ref=_output_ref(a.late.get("output_ref")),
                    error=_error(a.late.get("error")),
                    error_phase=a.late.get("error_phase"),
                )
            completion = CompletionRecord(
                request_id=rid,
                result=a.state,
                t_complete=c.get("t_complete"),
                t_output_ready=c.get("t_output_ready"),
                output_ref=_output_ref(c.get("output_ref")),
                error=_error(c.get("error")),
                error_phase=c.get("error_phase"),
                t_timed_out=a.t_timed_out,
                late_completion=late,
            )
        folded[rid] = (request, completion)
    return folded


def folded_counts(folded: dict[int, tuple[RequestRecord, Optional[CompletionRecord]]]) -> dict[str, int]:
    states = [req.state for req, _ in folded.values()]
    issued = sum(s in (S.ISSUED, S.INFLIGHT, S.COMPLETED, S.ERROR, S.TIMED_OUT) for s in states)
    return {
        "planned": len(states),
        "issued": issued,
        "completed": states.count(S.COMPLETED),
        "error": states.count(S.ERROR),
        "timed_out": states.count(S.TIMED_OUT),
        "dropped": states.count(S.DROPPED),
        "not_issued": states.count(S.NOT_ISSUED),
        "late_completion": sum(1 for _, c in folded.values() if c is not None and c.late_completion is not None),
        "unresolved": sum(s in (S.ISSUED, S.INFLIGHT) for s in states),
    }
