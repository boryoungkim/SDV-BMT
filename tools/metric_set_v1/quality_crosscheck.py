#!/usr/bin/env python3
"""Cross-check of the canonical (bundle-only) detection quality against the Stage 2b-G analysis-plane result (tool).

Compares, for one run bundle, its newest Metric Set v1 evaluation (``quality.detection.*``, computed from the
ground truth frozen in the bundle) with the analysis-plane output of ``tools/stage2b_g_real/coco_quality.py``
(the same outputs evaluated against the digest-verified external dataset index).  Exact equality is required.

    .venv/bin/python tools/metric_set_v1/quality_crosscheck.py RUN_DIR ANALYSIS_QUALITY.json OUT.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PAIRS = (("ap", "AP"), ("ap50", "AP50"), ("ap75", "AP75"), ("ap_small", "AP_small"), ("ap_medium", "AP_medium"),
         ("ap_large", "AP_large"), ("ar1", "AR1"), ("ar10", "AR10"), ("ar100", "AR100"))


def main() -> int:
    run_dir, analysis_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    docs = []
    for path in sorted((run_dir / "evaluations").glob("*/metrics.json")):
        doc = json.loads(path.read_text())
        if doc.get("catalog_version") == "sdv-bmt-model-metric-set-v1":
            docs.append((json.loads((path.parent / "eval_manifest.json").read_text())["created_at_utc"], doc))
    canonical = {m["metric_id"]: m for m in sorted(docs, key=lambda x: x[0])[-1][1]["metrics"]}
    analysis = json.loads(analysis_path.read_text())
    rows = []
    for name, stat in PAIRS:
        a = canonical[f"quality.detection.conditional.{name}"]["value"]
        b = analysis["conditional"]["stats"][stat]
        rows.append({"metric": f"quality.detection.conditional.{name}", "canonical": a, "analysis_plane": b,
                     "equal": a == b})
    rows.append({"metric": "quality.detection.coverage", "canonical": canonical["quality.detection.coverage"]["value"],
                 "analysis_plane": analysis["coverage"],
                 "equal": canonical["quality.detection.coverage"]["value"] == analysis["coverage"]})
    result = {"schema": "sdv-bmt-metric-set-v1-quality-crosscheck/1", "run_dir": str(run_dir),
              "analysis_plane_file": str(analysis_path), "rows": rows, "all_equal": all(r["equal"] for r in rows)}
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"all_equal": result["all_equal"]}))
    return 0 if result["all_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
