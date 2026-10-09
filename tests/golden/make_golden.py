"""Regenerate golden snapshots for G-1..G-5 (run manually, never from pytest).

    .venv/bin/python -m tests.golden.make_golden

Snapshots are regression guards for the canonical event sequence, the stress
lifecycle records, the output blob digest and the canonical metric content.
The semantic expected values are hand-written in test_golden_cases.py from the
Stage 1A authorization (section 4) and do not come from these snapshots.
Changing a golden expected result requires a design-change proposal
(authorization section 7).
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

from sdv_bmt.contracts.ids import pretty_json_bytes, to_jsonable
from sdv_bmt.metrics.evaluator import canonical_metric_content

GOLDEN_DIR = Path(__file__).resolve().parent / "expected"
CASES = {
    "G1": "g1-dataset-fixed-rate-nominal",
    "G2": "g2-recorded-scenario-overload",
    "G3": "g3-timeout-late-completion",
    "G4": "g4-stress-verify-failure",
    "G5": "g5-censoring",
}
SNAPSHOT_FILES = ("events.jsonl", "stress.jsonl", "outputs.sha256", "metrics.canonical.json")


def produce(case: str, work: Path) -> dict[str, bytes]:
    from tests._helpers import evaluate, load, run_plan

    out = run_plan(load(CASES[case]), work / case, run_id=f"run-golden-{case}")
    evaluation = evaluate(out.run_dir, eval_id=f"eval-golden-{case}")
    metrics = [to_jsonable(m) for m in evaluation.metrics]
    params = json.loads((evaluation.eval_dir / "metrics.json").read_text())["parameters"]
    outputs_sha = hashlib.sha256((out.run_dir / "outputs.bin").read_bytes()).hexdigest()
    return {
        "events.jsonl": (out.run_dir / "events.jsonl").read_bytes(),
        "stress.jsonl": (out.run_dir / "stress.jsonl").read_bytes(),
        "outputs.sha256": f"{outputs_sha}\n".encode(),
        "metrics.canonical.json": pretty_json_bytes(canonical_metric_content(params, metrics)),
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        for case in CASES:
            target = GOLDEN_DIR / case
            target.mkdir(parents=True, exist_ok=True)
            for name, data in produce(case, Path(tmp)).items():
                (target / name).write_bytes(data)
            print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
