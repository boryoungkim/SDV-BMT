#!/usr/bin/env python3
"""Stage 2b-G real-model baseline: Design / Realization / Measurement Guard results on the YOLOX bundles
(analysis plane; the approved guard implementation ``evaluate_contrast_validity`` is used unchanged).

    .venv/bin/python tools/stage2b_g_real/guard_cases.py OUT.json ROOT [V01_BASELINE_RUN]

Every pair is evaluated with both runs' manifests and their latest eval_manifest.json, as a ``stress``
contrast between nominal runs (the only contrast nominal runs can form, i.e. a replicate design):
  * R-A  replicate pairs within one plan: all pairs of the 30 Hz runs, of the 100 Hz runs, of the asap runs;
  * R-B  same workload, different replay rate (30 Hz vs 100 Hz): the rate is not a factor of any approved
         contrast type, so the pair must not pass;
  * R-C  profiler configuration (nsys + NVTX, 200 images) vs the 30 Hz baseline;
  * R-D  reference capture (no output reference) vs a 100 Hz baseline run (same rate, output reference differs);
  * R-E  (optional) the v01 reference-conformance baseline vs a YOLOX 100 Hz baseline run (different workload).
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from sdv_bmt.contracts.ids import to_jsonable  # noqa: E402
from sdv_bmt.evidence.validity import evaluate_contrast_validity  # noqa: E402


def load(run_dir: Path) -> tuple[dict, dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    evals = sorted(p for p in (run_dir / "evaluations").iterdir() if p.is_dir())
    return manifest, json.loads((evals[-1] / "eval_manifest.json").read_text(encoding="utf-8"))


def brief(a: Path, b: Path) -> dict:
    ma, ea = load(a)
    mb, eb = load(b)
    r = to_jsonable(evaluate_contrast_validity(ma, mb, "stress", evaluations=(ea, eb)))
    rg = r["realization_guard"]
    return {
        "run_a": str(a.relative_to(REPO)), "run_b": str(b.relative_to(REPO)),
        "overall": r["overall_status"], "attribution_allowed": r["attribution_allowed"],
        "design": r["design_guard"]["status"], "design_violations": r["design_guard"]["violations"],
        "realization": rg["status"], "realization_basis": rg["basis"], "realization_available": rg["evidence_available"],
        "realization_mismatches": [{k: m.get(k) for k in ("condition", "reason")} for m in rg["mismatches"]],
        "realization_unresolved": sorted(f"{u['condition']}:{u['reason']}" for u in rg["unresolved"]),
        "measurement": r["measurement_guard"]["status"], "measurement_violations": r["measurement_guard"]["violations"],
    }


def summary(pairs: list[dict]) -> dict:
    return {
        "pairs": len(pairs),
        "overall": Counter(p["overall"] for p in pairs), "design": Counter(p["design"] for p in pairs),
        "realization": Counter(p["realization"] for p in pairs), "measurement": Counter(p["measurement"] for p in pairs),
        "attribution_allowed": Counter(p["attribution_allowed"] for p in pairs),
        "realization_mismatch_pairs": sum(1 for p in pairs if p["realization_mismatches"]),
        "unresolved_reasons": Counter(u for p in pairs for u in p["realization_unresolved"]),
        "available_in_every_pair": sorted(set.intersection(*(set(p["realization_available"]) for p in pairs))) if pairs else [],
        "representative": pairs[0] if pairs else None,
    }


def main() -> int:
    out, root = sys.argv[1], Path(sys.argv[2]).resolve()
    v01 = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else None
    runs = lambda pattern: sorted(p.parent for p in root.rglob("manifest.json") if p.parent.parent.name == pattern)  # noqa: E731
    r30, r100, asap = runs("br-yolox-30hz"), runs("br-yolox-100hz"), runs("br-yolox-asap")
    nsys, capture = runs("br-yolox-nsys"), runs("br-yolox-capture")
    result = {
        "note": "stress contrasts between nominal runs = replicate designs; guard implementation unchanged",
        "R-A": {"30hz": summary([brief(a, b) for a, b in itertools.combinations(r30, 2)]),
                "100hz": summary([brief(a, b) for a, b in itertools.combinations(r100, 2)]),
                "asap": summary([brief(a, b) for a, b in itertools.combinations(asap, 2)])},
        "R-B_rate_30_vs_100": summary([brief(a, b) for a, b in zip(r30, r100)]),
        "R-C_profiler_vs_baseline": summary([brief(r30[0], n) for n in nsys]),
        "R-D_capture_vs_100hz": summary([brief(c, r100[0]) for c in capture]),
    }
    if v01 is not None:
        result["R-E_v01_reference_vs_yolox"] = summary([brief(v01, r100[0])])
    Path(out).write_text(json.dumps(result, indent=1, sort_keys=True, default=dict) + "\n", encoding="utf-8")
    for case, value in result.items():
        if case == "note":
            continue
        groups = value.items() if case == "R-A" else [("", value)]
        for name, s in groups:
            print(case, name, json.dumps({k: s[k] for k in ("pairs", "overall", "design", "realization", "measurement",
                                                             "attribution_allowed")}, default=dict, sort_keys=True))
            if s["representative"]:
                print("    design_violations", s["representative"]["design_violations"][:12],
                      "measurement_violations", s["representative"]["measurement_violations"])
    print("unresolved (R-A 30 Hz):", json.dumps(result["R-A"]["30hz"]["unresolved_reasons"], default=dict, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
