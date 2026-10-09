#!/usr/bin/env python3
"""Collect Metric Set v1 evaluations of run bundles into one JSON document for the canonical report (tool).

For every bundle directory given, the newest ``evaluations/<eval_id>`` whose metrics.json has catalog
``sdv-bmt-model-metric-set-v1`` is read (bundle-derived canonical values; nothing is recomputed here).
Per plan (experiment id) the tool adds a descriptive min-max range of each valid metric over the runs.
Descriptive only: no pooling, no test statistics.

    .venv/bin/python tools/metric_set_v1/metric_tables.py OUT.json RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CATALOG = "sdv-bmt-model-metric-set-v1"


def latest_v1(run_dir: Path) -> tuple[str, dict, dict]:
    found = []
    for path in sorted((run_dir / "evaluations").glob("*/metrics.json")):
        doc = json.loads(path.read_text())
        if doc.get("catalog_version") == CATALOG:
            em = json.loads((path.parent / "eval_manifest.json").read_text())
            found.append((em["created_at_utc"], path.parent.name, doc, em))
    if not found:
        raise SystemExit(f"{run_dir}: no Metric Set v1 evaluation")
    _, eval_id, doc, em = sorted(found)[-1]
    return eval_id, doc, em


def main() -> int:
    out = Path(sys.argv[1])
    runs = []
    for arg in sys.argv[2:]:
        run_dir = Path(arg)
        eval_id, doc, em = latest_v1(run_dir)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        runs.append({
            "run_dir": str(run_dir), "run_id": doc["run_id"], "experiment_id": manifest["experiment_id"],
            "eval_id": eval_id, "metrics_content_sha256": em["metrics_content_sha256"],
            "protocol_digest": em["protocol_digest"], "evaluator_source_tree": em["evaluator"]["code_identity"]["source_tree_sha256"],
            "manifest_schema_version": manifest["schema_version"], "context": doc["context"],
            "parameters": doc["parameters"],
            "metrics": {m["metric_id"]: {k: m[k] for k in ("layer", "status", "value", "unit", "status_reason",
                                                           "censored_fraction", "population")}
                        for m in doc["metrics"]},
        })
    plans: dict[str, dict] = {}
    for r in runs:
        plans.setdefault(r["experiment_id"], {"runs": [], "ranges": {}})["runs"].append(r["run_id"])
    for plan, entry in plans.items():
        members = [r for r in runs if r["experiment_id"] == plan]
        ids = members[0]["metrics"].keys()
        for metric_id in ids:
            vals = [r["metrics"][metric_id]["value"] for r in members if r["metrics"][metric_id]["status"] == "valid"]
            statuses = sorted({r["metrics"][metric_id]["status"] for r in members})
            entry["ranges"][metric_id] = {"min": min(vals) if vals else None, "max": max(vals) if vals else None,
                                          "valid_runs": len(vals), "runs": len(members), "statuses": statuses}
    doc = {"schema": "sdv-bmt-metric-set-v1-tables/1", "catalog_version": CATALOG, "runs": runs, "plans": plans,
           "note": "canonical values read from the bundles' Metric Set v1 evaluations; ranges are descriptive"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({p: len(e["runs"]) for p, e in plans.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
