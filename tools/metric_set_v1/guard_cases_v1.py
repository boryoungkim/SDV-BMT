#!/usr/bin/env python3
"""Design / Realization / Measurement Guard results for run pairs with Metric Set v1 evaluations (tool).

Uses the approved ``evaluate_contrast_validity`` unchanged, always with both evaluation manifests.  A pair is
"A B" (run directories); the evaluation of each side is its newest Metric Set v1 evaluation unless
``--legacy-b`` selects the newest legacy-catalog evaluation of B (mixed-catalog case).  The contrast type is
``stress`` (two nominal runs = a replicate design).  Writes one JSON document.

    .venv/bin/python tools/metric_set_v1/guard_cases_v1.py OUT.json CASES.json

CASES.json: {"case name": {"pairs": [[A, B], ...], "legacy_b": false}, ...}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from sdv_bmt.contracts.ids import to_jsonable  # noqa: E402
from sdv_bmt.evidence.validity import evaluate_contrast_validity  # noqa: E402

V1, LEGACY = "sdv-bmt-model-metric-set-v1", "bmt-portable-v1-stage1a"


def newest(run_dir: Path, catalog: str) -> dict:
    found = []
    for path in (run_dir / "evaluations").glob("*/eval_manifest.json"):
        em = json.loads(path.read_text())
        if em["evaluator"]["catalog_version"] == catalog:
            found.append((em["created_at_utc"], em))
    if not found:
        raise SystemExit(f"{run_dir}: no {catalog} evaluation")
    return sorted(found, key=lambda x: x[0])[-1][1]


def main() -> int:
    out, cases = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
    doc = {"schema": "sdv-bmt-metric-set-v1-guards/1", "contrast_type": "stress", "cases": {}}
    for name, spec in cases.items():
        rows = []
        for a, b in spec["pairs"]:
            ma = json.loads((Path(a) / "manifest.json").read_text())
            mb = json.loads((Path(b) / "manifest.json").read_text())
            ea = newest(Path(a), V1)
            eb = newest(Path(b), LEGACY if spec.get("legacy_b") else V1)
            r = to_jsonable(evaluate_contrast_validity(ma, mb, "stress", evaluations=(ea, eb)))
            rows.append({"run_a": r["run_a"], "run_b": r["run_b"], "eval_a": ea["eval_id"], "eval_b": eb["eval_id"],
                         "design": r["design_guard"]["status"], "design_violations": r["design_guard"]["violations"],
                         "realization": r["realization_guard"]["status"],
                         "realization_unresolved": sorted({u["condition"] for u in r["realization_guard"]["unresolved"]}),
                         "realization_mismatches": sorted({m["condition"] for m in r["realization_guard"]["mismatches"]}),
                         "realization_observed_equal": r["realization_guard"]["evidence_available"],
                         "measurement": r["measurement_guard"]["status"],
                         "measurement_violations": r["measurement_guard"]["violations"],
                         "overall": r["overall_status"], "attribution_allowed": r["attribution_allowed"],
                         "intended_factor_varied": r["design_guard"]["intended_factor_varied"]})
        summary = {}
        for key in ("design", "realization", "measurement", "overall", "attribution_allowed"):
            counts: dict[str, int] = {}
            for row in rows:
                counts[str(row[key])] = counts.get(str(row[key]), 0) + 1
            summary[key] = counts
        doc["cases"][name] = {"pairs": rows, "summary": summary}
        print(name, json.dumps(summary, sort_keys=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
