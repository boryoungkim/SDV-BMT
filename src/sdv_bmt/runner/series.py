"""Run series: one experiment's dataset selection executed as consecutive parts (minimal; CenterPoint part).

A v0.2 measurement extension brought forward (approved 2026-11-13): a selection larger than the pinned-memory limit is
measured as several ordinary runs whose plans differ from the experiment's only by ``input.selection``, and
``sdv-bmt evaluate SERIES_DIR`` produces one result from them (``metrics/series.py``).  Only the splitting of one
experiment is in scope; runs of different conditions are never combined.

Partition (deterministic, from the digest-pinned index): the declared selection is walked in index order and cut into
parts of at most ``series.max_items_per_run`` items without splitting a block of consecutive rows that share
``metadata[series.boundary]``; a boundary value must occupy one contiguous block, and a block larger than the limit is
refused.  Each part is an ordinary run (its own precondition requests, its own bundle) under
``<out>/<experiment_id>/<series_id>/part-NN/``.  The series stops at the first part that is rejected, does not
complete or is not measurement-valid; ``series.json`` is written at the end in every case and records the partition,
each part's run id and manifest SHA-256, and the time between parts: the gap between the sdv-bmt process's part runs
and, for real-time parts, the gap between one part's measurement window (end of DRAIN) and the next part's (start of
MEASURE) on the host monotonic clock (each part's run-clock anchor plus its phase times).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ..contracts.ids import loads_strict_json, new_run_id, sha256_hex, utc_now_iso
from ..contracts.specs import DatasetSelectionSpec
from ..contracts.status import BENCHMARK_LEVEL, ExecutionStatus, MeasurementStatus
from ..evidence.bundle import BundleError, MANIFEST_FILE, ensure_experiment_dir, file_record, write_json_exclusive
from ..observation.reader import BundleReadError, measure_window, read_jsonl
from .plan import PlanError, ResolvedPlan
from .realtime import host_monotonic_ns
from .runner import BenchmarkRunner, ComponentFactory, PlanRejected, RunOutcome

SERIES_FILE = "series.json"
SERIES_SCHEMA_VERSION = 1
SERIES_KIND = "sdv-bmt-run-series/1"


@dataclass(frozen=True)
class SeriesPart:
    index: int
    start: int  # index row of the part's first item (selection.start of the part's plan)
    count: int
    boundary_values: tuple[str, ...]


def new_series_id() -> str:
    return "series-" + new_run_id()[len("run-"):]


def partition(rp: ResolvedPlan) -> list[SeriesPart]:
    """The parts of a run series plan.  Raises PlanError when the selection cannot be split by the rule."""
    series = rp.plan.series
    if series is None:
        raise PlanError(["the plan declares no run series"])
    spec = rp.input_spec
    index_path = rp.input_spec_dir / spec.source.index
    try:
        data = index_path.read_bytes()
    except OSError as exc:
        raise PlanError([f"series: cannot read {index_path.name}: {exc}"]) from None
    if sha256_hex(data) != spec.source.index_sha256:
        raise PlanError([f"series: {index_path.name} does not match source.index_sha256"])
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(data.decode("utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        try:
            row = loads_strict_json(line)
        except ValueError as exc:
            raise PlanError([f"series: {index_path.name}:{line_no}: {exc}"]) from None
        rows.append({"line": line_no, "metadata": row.get("metadata") if isinstance(row, dict) else None})
    start = spec.selection.start
    end = len(rows) if spec.selection.count is None else start + spec.selection.count
    if start >= len(rows) or end > len(rows):
        raise PlanError([f"series: selection [{start}, {end}) is outside the {len(rows)} index rows"])
    blocks: list[tuple[str, int, int]] = []  # (value, first row, rows)
    for pos in range(start, end):
        metadata = rows[pos]["metadata"]
        value = metadata.get(series.boundary) if isinstance(metadata, dict) else None
        if not isinstance(value, str) or not value:
            raise PlanError([f"series: index line {rows[pos]['line']} has no metadata.{series.boundary} string"])
        if blocks and blocks[-1][0] == value:
            blocks[-1] = (value, blocks[-1][1], blocks[-1][2] + 1)
        else:
            blocks.append((value, pos, 1))
    seen: set[str] = set()
    for value, first, n in blocks:
        if value in seen:
            raise PlanError([f"series: {series.boundary} {value!r} occupies more than one block of the selection "
                             f"(again at index row {first})"])
        seen.add(value)
        if n > series.max_items_per_run:
            raise PlanError([f"series: {series.boundary} {value!r} has {n} items, more than "
                             f"max_items_per_run {series.max_items_per_run}"])
    parts: list[SeriesPart] = []
    cur: list[tuple[str, int, int]] = []
    for block in blocks:
        if cur and sum(b[2] for b in cur) + block[2] > series.max_items_per_run:
            parts.append(SeriesPart(len(parts), cur[0][1], sum(b[2] for b in cur), tuple(b[0] for b in cur)))
            cur = []
        cur.append(block)
    if cur:
        parts.append(SeriesPart(len(parts), cur[0][1], sum(b[2] for b in cur), tuple(b[0] for b in cur)))
    return parts


def part_plan(rp: ResolvedPlan, part: SeriesPart) -> ResolvedPlan:
    """The part's plan: the experiment's plan with ``input.selection`` = the part's rows (nothing else differs)."""
    selection = DatasetSelectionSpec(start=part.start, count=part.count)
    return dataclasses.replace(rp, input_spec=dataclasses.replace(rp.input_spec, selection=selection))


def describe_partition(rp: ResolvedPlan, parts: list[SeriesPart]) -> dict[str, Any]:
    series = rp.plan.series
    assert series is not None
    return {
        "rule": {"max_items_per_run": series.max_items_per_run, "boundary": series.boundary,
                 "method": "index order; greedy fill up to max_items_per_run without splitting a boundary block"},
        "selection": {"start": parts[0].start if parts else rp.input_spec.selection.start,
                      "count": sum(p.count for p in parts)},
        "parts": [{"part": p.index, "start": p.start, "count": p.count, "boundary_values": list(p.boundary_values)}
                  for p in parts],
    }


def measure_window_monotonic(run_dir: Path) -> Optional[dict[str, int]]:
    """A real-time part's measurement window on the host monotonic clock: run-clock anchor + the MEASURE begin and DRAIN
    end phase times of its events.  None for a virtual-clock run or when the bundle does not record both."""
    try:
        manifest = loads_strict_json((run_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        clock = manifest.get("run_clock")
        if not isinstance(clock, dict) or clock.get("domain") != "monotonic":
            return None
        anchor = clock["anchor_monotonic_ns"]
        window = measure_window(tuple(read_jsonl(run_dir / "events.jsonl")))
    except (OSError, ValueError, KeyError, BundleReadError):
        return None
    if window is None or not isinstance(anchor, int):
        return None
    return {"measure_begin_monotonic_ns": anchor + window.t_measure_begin,
            "drain_end_monotonic_ns": anchor + window.t_drain_end}


@dataclass
class SeriesOutcome:
    series_id: str
    series_dir: Path
    status: str  # "completed" | "incomplete"
    outcomes: list[RunOutcome]
    rejected: Optional[list[str]] = None


def run_series(rp: ResolvedPlan, factory: ComponentFactory, out_root: Path, *,
               run_id_factory: Callable[[], str] = new_run_id, now_factory: Callable[[], str] = utc_now_iso,
               series_id_factory: Callable[[], str] = new_series_id,
               clock: Callable[[], int] = host_monotonic_ns) -> SeriesOutcome:
    parts = partition(rp)
    whole = BenchmarkRunner(rp, factory, run_id_factory=run_id_factory, now_factory=now_factory)
    try:
        exp_dir = ensure_experiment_dir(Path(out_root), whole.experiment_record())
    except BundleError as exc:
        raise PlanRejected([str(exc)]) from None
    series_id = series_id_factory()
    series_dir = exp_dir / series_id
    series_dir.mkdir(exist_ok=False)
    started = now_factory()
    records: list[dict[str, Any]] = []
    outcomes: list[RunOutcome] = []
    rejected: Optional[list[str]] = None
    for part in parts:
        part_root = series_dir / f"part-{part.index:02d}"
        runner = BenchmarkRunner(part_plan(rp, part), factory, run_id_factory=run_id_factory, now_factory=now_factory)
        begin_utc, begin_ns = now_factory(), clock()
        try:
            outcome = runner.run_once(part_root, 0)
        except PlanRejected as exc:
            rejected = [f"part {part.index}: {issue}" for issue in exc.issues]
            records.append({"part": part.index, "start": part.start, "count": part.count, "rejected": exc.issues,
                            "begin_utc": begin_utc, "begin_monotonic_ns": begin_ns})
            break
        end_ns, end_utc = clock(), now_factory()
        outcomes.append(outcome)
        manifest = outcome.run_dir / MANIFEST_FILE
        records.append({
            "part": part.index, "start": part.start, "count": part.count,
            "boundary_values": list(part.boundary_values),
            "run_id": outcome.run_id, "run_dir": outcome.run_dir.relative_to(series_dir).as_posix(),
            "manifest_sha256": file_record(manifest)["sha256"] if manifest.is_file() else None,
            "execution_status": outcome.execution_status, "measurement_status": outcome.measurement_status,
            "request_counts": outcome.request_counts,
            "begin_utc": begin_utc, "end_utc": end_utc, "begin_monotonic_ns": begin_ns, "end_monotonic_ns": end_ns,
            "measure_window": measure_window_monotonic(outcome.run_dir) if manifest.is_file() else None,
        })
        if (outcome.execution_status is not ExecutionStatus.COMPLETED
                or outcome.measurement_status is not MeasurementStatus.VALID or not manifest.is_file()):
            break
    done = [r for r in records if r.get("run_id") is not None]
    complete = (rejected is None and len(done) == len(parts)
                and all(r["execution_status"] is ExecutionStatus.COMPLETED
                        and r["measurement_status"] is MeasurementStatus.VALID for r in done))
    gaps = [{"after_part": a["part"], "before_part": b["part"],
             "gap_ns": b["begin_monotonic_ns"] - a["end_monotonic_ns"],
             "measure_gap_ns": (b["measure_window"]["measure_begin_monotonic_ns"]
                                - a["measure_window"]["drain_end_monotonic_ns"])
             if a["measure_window"] and b["measure_window"] else None}
            for a, b in zip(done, done[1:])]
    write_json_exclusive(series_dir / SERIES_FILE, {
        "schema_version": SERIES_SCHEMA_VERSION,
        "kind": SERIES_KIND,
        "benchmark_level": BENCHMARK_LEVEL,
        "series_id": series_id,
        "experiment_id": rp.plan.experiment_id,
        "plan_digest": rp.plan_digest,
        "execution_kind": rp.plan.execution_kind,
        "declared_selection": rp.input_spec.selection,
        "partition": describe_partition(rp, parts),
        "status": "completed" if complete else "incomplete",
        "parts": records,
        "gaps_between_parts": gaps,
        "gap_clock": "host monotonic clock of the sdv-bmt process (end of one part's run to the start of the next)",
        "measure_gap_clock": ("host monotonic clock, real-time parts only (null otherwise): end of one part's DRAIN to "
                              "the start of the next part's MEASURE; includes the next part's input open, DUT prepare, "
                              "input preparation and precondition requests"),
        "started_at_utc": started,
        "finished_at_utc": now_factory(),
    })
    return SeriesOutcome(series_id, series_dir, "completed" if complete else "incomplete", outcomes, rejected)
