#!/usr/bin/env python3
"""tinyBenchmarks estimates (IRT, p-IRT, gp-IRT) of full-MMLU accuracy from evaluated tinyMMLU runs (offline tool).

The lm-evaluation-harness task ``tinyMMLU`` reports ``tinyBenchmarks.evaluate(y, "mmlu")["mmlu"]["gpirt"]`` for the
per-item ``acc_norm`` vector ``y`` in dataset order.  This tool does the same from the bundle's evaluation
(``quality.correct_vector``).  tinyBenchmarks downloads its IRT parameter file from the repository's ``main`` branch
into the working directory when the file is absent; here the file is pinned instead: ``--pkl`` must match
``IRT_PKL_SHA256`` (tinyBenchmarks commit ``TB_COMMIT``) and the estimate runs in a temporary directory that holds
only that file, so nothing is downloaded.  The file is a pickle: load only the pinned one.

Run with an environment that has numpy, scipy and tinyBenchmarks (the lm-eval environment, docs/llm_benchmark.md)
and ``PYTHONPATH=src``::

    $LMEVAL_PY tools/llm/tinymmlu_irt.py --pkl <dir>/tinyBenchmarks.pkl --out <file.json> RUN_DIR [RUN_DIR ...]

The result file can be given to ``python -m sdv_bmt.llm table ... --irt <file.json>``, which puts the gp-IRT value
next to the raw accuracy (metric ``llm.tinymmlu.gpirt``) after matching run and evaluation IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from sdv_bmt.llm.evaluate import latest_metrics, verify_bundle  # noqa: E402
from sdv_bmt.llm.tinymmlu import (IRT_PKL_SHA256, IRT_RESULT_TOOL, SCORING_RULE_ID, TB_COMMIT,  # noqa: E402
                                   TINYMMLU_TOTAL)

IRT_PKL_URL = f"https://raw.githubusercontent.com/felipemaiapolo/tinyBenchmarks/{TB_COMMIT}/tinyBenchmarks/tinyBenchmarks.pkl"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def estimate(vector: Sequence[float], pkl: Path) -> dict[str, float]:
    """tinyBenchmarks.evaluate(vector, 'mmlu')['mmlu'] with the pinned parameter file."""
    import numpy as np
    import tinyBenchmarks as tb

    if len(vector) != TINYMMLU_TOTAL or any(v is None for v in vector):
        raise ValueError(f"need {TINYMMLU_TOTAL} scored items, got {len(vector)} with "
                         f"{sum(v is None for v in vector)} unscored")
    if sha256_file(pkl) != IRT_PKL_SHA256:
        raise ValueError(f"{pkl} is not the pinned tinyBenchmarks.pkl (SHA-256 {IRT_PKL_SHA256})")
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copyfile(pkl, Path(tmp) / "tinyBenchmarks.pkl")
        os.chdir(tmp)
        try:
            est = tb.evaluate(np.array(vector, dtype=float), "mmlu")["mmlu"]
        finally:
            os.chdir(cwd)
    return {k: float(v) for k, v in est.items()}


def package_versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for d in ("tinyBenchmarks", "numpy", "scipy"):
        try:
            out[d] = md.version(d)
        except md.PackageNotFoundError:
            out[d] = "missing"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    results = []
    for d in a.run_dirs:
        manifest = verify_bundle(Path(d))
        m = latest_metrics(Path(d))
        q = m.get("quality") or {}
        if q.get("scoring_rule") != SCORING_RULE_ID:
            print(f"error: {d} is not an evaluated tinyMMLU run", file=sys.stderr)
            return 2
        row = {"run_dir": d, "experiment_id": manifest["experiment_id"], "run_id": manifest["run_id"],
               "eval_id": m["eval_id"], "accuracy_100": q["accuracy"], "complete": q["complete"]}
        row["estimate"] = estimate(q["correct_vector"], Path(a.pkl)) if q["complete"] else None
        results.append(row)
        print(json.dumps({k: row[k] for k in ("experiment_id", "accuracy_100", "estimate")}))
    out = {"tool": IRT_RESULT_TOOL, "tool_sha256": sha256_file(Path(__file__)),
           "reported_value": "estimate.gpirt (what lm-evaluation-harness tinyMMLU returns)",
           "irt_parameters": {"file_sha256": IRT_PKL_SHA256, "source": IRT_PKL_URL}, "packages": package_versions(),
           "runs": results}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
