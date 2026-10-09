#!/usr/bin/env python3
"""Re-evaluate existing run bundles under Model-level Metric Set v1 without touching their evidence root (tool).

Each bundle found under SRC/<subroot> is copied -- manifest.json and the raw files its schema version records,
plus the experiment's experiment.json -- to DEST/<same relative path>; every copied file is checked byte for
byte (SHA-256) against the source and against the manifest record.  The copy is then evaluated twice:

* legacy catalog ``bmt-portable-v1-stage1a``: its ``metrics_content_sha256`` must equal the archived
  evaluation(s) of the source bundle with the same protocol (reproduction / compatibility check);
* Metric Set v1 (default catalog): the canonical re-evaluation.

The source evidence root is only read.  Writes DEST/... and a JSON summary.

    .venv/bin/python tools/metric_set_v1/reevaluate_bundles.py SRC_ROOT DEST_ROOT OUT.json SUBROOT [SUBROOT ...]
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from sdv_bmt.contracts.records import LEGACY_METRIC_CATALOG_VERSION, METRIC_CATALOG_VERSION, raw_evidence_files  # noqa: E402
from sdv_bmt.metrics.evaluator import evaluate_run  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    src_root, dest_root, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    subroots = sys.argv[4:]
    if dest_root.resolve() == src_root.resolve() or src_root.resolve() in dest_root.resolve().parents:
        raise SystemExit("DEST must be outside SRC")
    results = []
    for sub in subroots:
        for manifest_path in sorted((src_root / sub).rglob("manifest.json")):
            src = manifest_path.parent
            rel = src.relative_to(src_root)
            dst = dest_root / rel
            if dst.exists():
                raise SystemExit(f"{dst} exists; copies are never overwritten")
            manifest = json.loads(manifest_path.read_text())
            names = list(raw_evidence_files(manifest["schema_version"])) + ["manifest.json"]
            dst.mkdir(parents=True)
            exp_src = src.parent / "experiment.json"
            exp_dst = dst.parent / "experiment.json"
            if exp_src.is_file() and not exp_dst.exists():
                shutil.copy2(exp_src, exp_dst)
            copies = {}
            for name in names:
                shutil.copy2(src / name, dst / name)
                a, b = sha256(src / name), sha256(dst / name)
                recorded = manifest["files"].get(name, {}).get("sha256") if name != "manifest.json" else a
                if not (a == b == recorded):
                    raise SystemExit(f"{rel}/{name}: copy or manifest record differs")
                copies[name] = a
            archived = []
            for em_path in sorted((src / "evaluations").glob("*/eval_manifest.json")):
                em = json.loads(em_path.read_text())
                archived.append({"eval_id": em["eval_id"], "catalog_version": em["evaluator"]["catalog_version"],
                                 "protocol_digest": em["protocol_digest"],
                                 "metrics_content_sha256": em["metrics_content_sha256"]})
            legacy = evaluate_run(dst, catalog=LEGACY_METRIC_CATALOG_VERSION)
            legacy_em = json.loads((legacy.eval_dir / "eval_manifest.json").read_text())
            same_protocol = [a for a in archived if a["protocol_digest"] == legacy_em["protocol_digest"]]
            reproduced = bool(same_protocol) and all(a["metrics_content_sha256"] == legacy.metrics_content_sha256
                                                     for a in same_protocol)
            v1 = evaluate_run(dst, catalog=METRIC_CATALOG_VERSION)
            results.append({
                "source": str(src), "copy": str(dst), "run_id": manifest["run_id"],
                "experiment_id": manifest["experiment_id"], "manifest_schema_version": manifest["schema_version"],
                "copied_files_sha256": copies,
                "archived_evaluations": archived,
                "legacy": {"eval_id": legacy.eval_id, "protocol_digest": legacy_em["protocol_digest"],
                           "metrics_content_sha256": legacy.metrics_content_sha256,
                           "reproduces_archived": reproduced},
                "metric_set_v1": {"eval_id": v1.eval_id, "metrics_content_sha256": v1.metrics_content_sha256},
            })
            print(f"{rel}: copy verified, legacy reproduces archived = {reproduced}, v1 {v1.eval_id}")
    summary = {"schema": "sdv-bmt-metric-set-v1-reevaluation/1", "source_root": str(src_root),
               "dest_root": str(dest_root), "bundles": results,
               "all_copies_verified": True,
               "all_legacy_reproduced": all(r["legacy"]["reproduces_archived"] for r in results)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"bundles": len(results), "all_legacy_reproduced": summary["all_legacy_reproduced"]}))
    return 0 if summary["all_legacy_reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
