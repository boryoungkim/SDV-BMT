"""Shared test helpers: repository paths and minimal valid spec mappings."""

from __future__ import annotations

import copy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "sdv_bmt"
CONFIGS = REPO_ROOT / "configs"

ZERO_SHA = "0" * 64


def dataset_spec_dict(**overrides) -> dict:
    data = {
        "schema_version": 1,
        "input_kind": "dataset",
        "input_id": "ds-test",
        "version": "1",
        "family": {
            "source_dataset": "fixture-dataset",
            "source_version": "1",
            "task": "fake-identity",
            "annotation_schema": "fake-label-v1",
            "preprocessing_contract": "fake-bytes-v1",
        },
        "source": {"index": "fixtures/ds.index.jsonl", "index_sha256": ZERO_SHA},
        "selection": {"start": 0, "count": 4},
    }
    data.update(copy.deepcopy(overrides))
    return data


def scenario_spec_dict(**overrides) -> dict:
    data = {
        "schema_version": 1,
        "input_kind": "driving_scenario",
        "input_id": "sc-test",
        "version": "1",
        "family": {
            "source_dataset": "fixture-drive",
            "source_version": "1",
            "task": "fake-identity",
            "annotation_schema": None,
            "preprocessing_contract": "fake-bytes-v1",
            "streams": ["cam_front"],
            "timing": {
                "source_field": "t_capture_us",
                "unit": "us",
                "semantic": "capture",
                "clock_source": "fixture-camera-clock",
            },
        },
        "source": {"index": "fixtures/sc.index.jsonl", "index_sha256": ZERO_SHA},
        "selection": {"stream_id": "cam_front"},
    }
    data.update(copy.deepcopy(overrides))
    return data


def stress_spec_dict(**overrides) -> dict:
    data = {
        "schema_version": 1,
        "stress_id": "fake-axis",
        "kind": "fake_virtual",
        "target_resource": "virtual-none",
        "placement": "virtual",
        "control": {"variable": "virtual_intensity", "unit": "ratio"},
        "background_workload": None,
        "levels": {"L1": 0.25, "L2": 0.5, "L3": 1.0},
        "verification": {"window_ns": 0, "achieved_min_fraction": 0.9, "independent": None},
        "schedule": "whole_measurement",
        "on_apply_failure": "abort",
        "provider_options": {},
    }
    data.update(copy.deepcopy(overrides))
    return data


def plan_dict(**overrides) -> dict:
    data = {
        "schema_version": 1,
        "experiment_id": "exp-test",
        "execution_kind": "mock",
        "input": "../inputs/x.yaml",
        "workload": "../workloads/w.yaml",
        "dut": "../duts/d.yaml",
        "stress": {"spec": "../stress/s.yaml", "level": "none"},
        "replay": {"mode": "asap"},
        "run": {
            "max_inflight": 1,
            "queue_capacity": 0,
            "on_full": "drop_new",
            "request_timeout_ns": 1_000_000_000,
            "drain_ns": 2_000_000_000,
        },
    }
    data.update(copy.deepcopy(overrides))
    return data


# ---------------------------------------------------------------- fixture builders


def write_index(directory: Path, rows: list[dict], name: str = "src.index.jsonl") -> tuple[str, str]:
    """Write a JSONL source index; return (relative name, sha256)."""
    import hashlib
    import json

    text = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return name, hashlib.sha256(text.encode("utf-8")).hexdigest()


def dataset_provider(directory: Path, rows: list[dict], **selection):
    from sdv_bmt.contracts.specs import parse_input_spec
    from sdv_bmt.input.dataset_fixture import DatasetFixtureProvider

    name, sha = write_index(directory, rows)
    spec = parse_input_spec(
        dataset_spec_dict(
            source={"index": name, "index_sha256": sha},
            selection=selection or {"start": 0, "count": None},
        )
    )
    return DatasetFixtureProvider(spec, directory)


def scenario_provider(directory: Path, rows: list[dict], selection=None, family=None):
    from sdv_bmt.contracts.specs import parse_input_spec
    from sdv_bmt.input.recorded_scenario_fixture import RecordedScenarioFixtureProvider

    name, sha = write_index(directory, rows)
    data = scenario_spec_dict(
        source={"index": name, "index_sha256": sha},
        selection=selection if selection is not None else {"stream_id": "cam_front"},
    )
    if family is not None:
        data["family"].update(family)
    return RecordedScenarioFixtureProvider(parse_input_spec(data), directory)


def frame_rows(episode: str, offsets_us: list, stream: str = "cam_front", base: int = 1_000_000,
               **extra) -> list[dict]:
    rows = []
    for k, off in enumerate(offsets_us):
        row = {
            "source_item_id": f"{episode}/{stream}/{k}",
            "episode_id": episode,
            "stream_id": stream,
            "sensor_id": "cam0",
            "frame_id": str(k),
            "payload_text": f"{episode} {stream} {k}",
        }
        if off is not None:
            row["t_capture_us"] = base + off
        row.update(extra)
        rows.append(row)
    return rows


def load_input_provider(spec_path: Path):
    """Build the fixture provider for a configs/inputs/*.yaml spec."""
    from sdv_bmt.contracts.specs import DatasetSpec, parse_input_spec
    from sdv_bmt.input.dataset_fixture import DatasetFixtureProvider
    from sdv_bmt.input.recorded_scenario_fixture import RecordedScenarioFixtureProvider
    from sdv_bmt.runner.plan import load_yaml_strict

    spec = parse_input_spec(load_yaml_strict(spec_path))
    cls = DatasetFixtureProvider if isinstance(spec, DatasetSpec) else RecordedScenarioFixtureProvider
    return cls(spec, spec_path.parent)


# ---------------------------------------------------------------- runner helpers

EXPERIMENTS = CONFIGS / "experiments"


def plan_path(name: str) -> Path:
    return EXPERIMENTS / f"{name}.yaml"


def load(name: str):
    from sdv_bmt.runner.plan import load_plan

    return portable_affinity(load_plan(plan_path(name)))


def _available_cpus() -> set[int]:
    import os

    return set(os.sched_getaffinity(0))


def _needs_portable_affinity(rp) -> bool:
    cfg = rp.plan.realtime
    return (cfg is not None and rp.plan.execution_kind.value == "mock"
            and not set(cfg.cpu_affinity) <= _available_cpus())


def portable_affinity(rp):
    """A MOCK realtime plan pinned to CPUs this host can realize; any other plan unchanged.

    The realtime plans pin the native runner to NUMA node 0 of the 64-CPU reference host, and the runner fails a
    run whose realized affinity differs from the plan (``cpu_affinity_not_realized``).  On hosts without those
    CPUs (hosted CI runners) a MOCK plan is pinned to the CPUs available to this process instead.  ACTUAL plans
    keep their pinning: on another host they fail rather than run with a different placement."""
    import dataclasses

    if not _needs_portable_affinity(rp):
        return rp
    cpus = tuple(sorted(_available_cpus()))
    return variant(rp, plan={"realtime": dataclasses.replace(rp.plan.realtime, cpu_affinity=cpus)})


def plan_file(name: str, tmp_path: Path) -> Path:
    """Plan file for CLI tests: the tracked plan, or -- where ``portable_affinity`` would change it -- a copy
    in ``tmp_path`` whose (relative) references point at the same specs, pinned to the CPUs available here."""
    import os

    import yaml
    from sdv_bmt.runner.plan import load_plan

    src = plan_path(name)
    if not _needs_portable_affinity(load_plan(src)):
        return src
    out = tmp_path / "plan" / src.name
    out.parent.mkdir(parents=True, exist_ok=True)

    def rel(ref: str) -> str:
        return Path(os.path.relpath((src.parent / ref).resolve(), out.parent.resolve())).as_posix()

    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    for key in ("input", "workload", "dut"):
        data[key] = rel(data[key])
    data["stress"]["spec"] = rel(data["stress"]["spec"])
    data["realtime"]["cpu_affinity"] = sorted(_available_cpus())
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def variant(rp, *, plan=None, **changes):
    """Return a ResolvedPlan with replaced plan fields (dict) and/or resolved specs."""
    import dataclasses

    if plan:
        changes["plan"] = dataclasses.replace(rp.plan, **plan)
    return dataclasses.replace(rp, **changes)


def run_plan(rp, out_root: Path, factory=None, run_id: str = "run-test-0"):
    from sdv_bmt.cli.main import build_components
    from sdv_bmt.runner.runner import BenchmarkRunner

    runner = BenchmarkRunner(rp, factory or build_components, run_id_factory=lambda: run_id)
    return runner.run_once(out_root)


def read_jsonl(path: Path) -> list[dict]:
    import json

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def merged_records(run_dir: Path) -> list[dict]:
    """events.jsonl + stress.jsonl in seq order."""
    records = read_jsonl(run_dir / "events.jsonl") + read_jsonl(run_dir / "stress.jsonl")
    return sorted(records, key=lambda r: r["seq"])


REQUEST_KINDS = ("issued", "dropped", "not_issued", "submit_begin", "submit_end", "completed",
                 "error", "timed_out", "late_completion")


def request_trace(run_dir: Path) -> list[tuple]:
    """(kind, request_id, t_run_ns) for request-level events in order."""
    return [(e["kind"], e["request_id"], e["t_run_ns"])
            for e in read_jsonl(run_dir / "events.jsonl") if e["kind"] in REQUEST_KINDS]


def lifecycle_trace(run_dir: Path) -> list[str]:
    """Compact lifecycle order: phase begin/end, cleanup steps and stress steps."""
    out = []
    for r in merged_records(run_dir):
        if r["kind"] == "phase":
            out.append(f"{r['phase']}:{r['edge']}" + (f":{r['outcome']}" if r["edge"] == "end" else ""))
        elif r["kind"] == "cleanup":
            out.append(f"cleanup:{r['step']}:{r['outcome']}")
        elif r["kind"] == "stress":
            out.append(f"stress:{r['step']}:{r['outcome']}")
    return out


MS = 1_000_000


# ---------------------------------------------------------------- evaluator helpers


LEGACY_CATALOG = "bmt-portable-v1-stage1a"
METRIC_SET_V1 = "sdv-bmt-model-metric-set-v1"


def evaluate(run_dir: Path, eval_id: str = "eval-test-0", override=None, catalog: str = LEGACY_CATALOG):
    """Evaluate a bundle.  The Stage 1A/1R tests verify the frozen legacy catalog (its metric IDs and
    golden snapshots are unchanged), so that is this helper's default; Metric Set v1 tests pass
    ``catalog=METRIC_SET_V1`` (the evaluator's own default)."""
    from sdv_bmt.metrics.evaluator import evaluate_run

    return evaluate_run(run_dir, override=override, catalog=catalog, eval_id_factory=lambda: eval_id)


def metric_map(outcome) -> dict:
    return {m.metric_id: m for m in outcome.metrics}


def reseal(run_dir: Path, mutate_events=None) -> None:
    """Test-only: rewrite events.jsonl through ``mutate_events`` and re-record the
    raw digests in manifest.json, producing a self-consistent synthetic bundle."""
    import json

    from sdv_bmt.contracts.ids import pretty_json_bytes
    from sdv_bmt.evidence.bundle import raw_file_records

    path = run_dir / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    if mutate_events is not None:
        events = mutate_events(events)
    path.write_text("".join(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n" for e in events))
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = raw_file_records(run_dir)
    manifest_path.write_bytes(pretty_json_bytes(manifest))


# ---------------------------------------------------------------- contrast validity helpers


def design_check(keys_a, keys_b, contrast: str):
    """Design Guard over two ComparisonKeyRecords (no run needed)."""
    from sdv_bmt.contracts.ids import to_jsonable
    from sdv_bmt.contracts.status import ContrastType
    from sdv_bmt.evidence.keys import identity_components
    from sdv_bmt.evidence.validity import design_guard

    return design_guard(identity_components(to_jsonable(keys_a)), identity_components(to_jsonable(keys_b)),
                        ContrastType(contrast))
