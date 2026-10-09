"""Run bundle layout (design v03 I.2) and file-level integrity helpers.

::

    <out>/<experiment_id>/
    ├── experiment.json            written once; later runs must match plan_digest
    └── <run_id>/
        ├── run_spec.json  input_items.jsonl  events.jsonl  stress.jsonl  outputs.bin
    ├── quality_reference.jsonl  manifest v5 only (frozen task-quality ground truth)
        ├── manifest.json          written last (FINALIZE)
        └── evaluations/<eval_id>/{metrics.json, eval_manifest.json}

All files are created exclusively; nothing is ever overwritten.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..contracts.ids import file_sha256, loads_strict_json, pretty_json_bytes
from ..contracts.records import RAW_EVIDENCE_FILES

EXPERIMENT_FILE = "experiment.json"
RUN_SPEC_FILE = "run_spec.json"
MANIFEST_FILE = "manifest.json"
EVALUATIONS_DIR = "evaluations"
METRICS_FILE = "metrics.json"
EVAL_MANIFEST_FILE = "eval_manifest.json"


class BundleError(RuntimeError):
    pass


def write_json_exclusive(path: Path, value: Any) -> None:
    with open(path, "xb") as fh:
        fh.write(pretty_json_bytes(value))


def read_json(path: Path) -> Any:
    return loads_strict_json(Path(path).read_text(encoding="utf-8"))


def ensure_experiment_dir(out_root: Path, record: dict[str, Any]) -> Path:
    """Create ``<out_root>/<experiment_id>/experiment.json`` once, or verify it."""
    exp_dir = Path(out_root) / record["experiment_id"]
    exp_file = exp_dir / EXPERIMENT_FILE
    if exp_file.exists():
        try:
            existing = read_json(exp_file)
            existing_digest = existing.get("plan_digest")
        except (OSError, ValueError, AttributeError) as exc:
            raise BundleError(f"{exp_file} is unreadable: {exc}") from None
        if existing_digest != record["plan_digest"]:
            raise BundleError(
                f"{exp_dir} already holds experiment plan {existing.get('plan_digest')}, "
                f"not {record['plan_digest']}"
            )
        return exp_dir
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(exp_file, record)
    return exp_dir


def create_run_dir(exp_dir: Path, run_id: str) -> Path:
    run_dir = Path(exp_dir) / run_id
    run_dir.mkdir(exist_ok=False)
    return run_dir


def file_record(path: Path) -> dict[str, Any]:
    return {"sha256": file_sha256(path), "bytes": Path(path).stat().st_size}


def raw_file_records(run_dir: Path, names: Sequence[str] = RAW_EVIDENCE_FILES) -> dict[str, dict[str, Any]]:
    return {name: file_record(Path(run_dir) / name) for name in names}


def verify_raw_files(run_dir: Path, recorded: dict[str, dict[str, Any]],
                     names: Sequence[str] = RAW_EVIDENCE_FILES) -> list[str]:
    """Raw files whose current SHA-256/size differ from the manifest record (``names``: the raw
    files of the manifest's schema version, see ``raw_evidence_files``)."""
    problems = []
    if not isinstance(recorded, dict):
        return ["manifest files section is not an object"]
    for name in names:
        path = Path(run_dir) / name
        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        expected = recorded.get(name)
        if not isinstance(expected, dict):
            problems.append(f"{name}: no digest record in manifest")
            continue
        actual = file_record(path)
        if actual != {"sha256": expected.get("sha256"), "bytes": expected.get("bytes")}:
            problems.append(f"{name}: digest/size mismatch")
    return problems


def run_dirs(exp_dir: Path) -> list[Path]:
    exp_dir = Path(exp_dir)
    if not exp_dir.is_dir():
        return []
    return sorted(p for p in exp_dir.iterdir() if p.is_dir() and (p / RUN_SPEC_FILE).exists())
