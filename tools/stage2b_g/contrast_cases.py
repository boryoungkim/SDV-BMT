"""Stage 2b-G: factor-isolation cases G-A..G-D on actual GPU run bundles (analysis plane).

    .venv/bin/python tools/stage2b_g/contrast_cases.py OUT.json

Evaluates ``sdv_bmt.evidence.validity.evaluate_contrast_validity`` (Design, Realization and Measurement
Guards) on pairs of actual run manifests with their latest evaluation manifests.  Every pair is a
``stress`` contrast between two nominal runs, i.e. a replicate design (the intended factor does not
vary), which is the only contrast these nominal reference runs can form.  The guard code is the
approved Stage 1A/DC-2 implementation, unchanged.

* G-A  clean replicate candidates: every pair of the 10 warm baseline runs (two blocks of 5).
* G-B  known realized-condition mismatch: a baseline run vs a run of the *same plan file* executed while
       another process held an idle CUDA context on the GPU (tools/stage2b_g/idle_context.py).
* G-C  measurement-configuration mismatch: baseline (cuda_events) vs instrumentation none, and baseline
       vs the same plan with NVTX under Nsight Systems.
* G-D  missing / unavailable evidence: (1) the evidence that is genuinely unavailable on this host in
       every actual run (memory_pressure_bandwidth, not_observable: GPU counters are permission-gated);
       (2) a guard-behaviour check on an in-memory copy of a baseline manifest from which one observed
       record (competing_device_users) is deleted -- clearly a modified copy, never written as a bundle.
"""

from __future__ import annotations

import copy
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from sdv_bmt.contracts.ids import to_jsonable  # noqa: E402
from sdv_bmt.evidence.validity import evaluate_contrast_validity  # noqa: E402

ROOT = REPO / "runs" / "stage2b-g"


def runs(sub: str) -> list[Path]:
    return sorted(p.parent for p in (ROOT / sub).rglob("manifest.json"))


def load(run_dir: Path) -> tuple[dict, dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    evals = sorted(p for p in (run_dir / "evaluations").iterdir() if p.is_dir())
    evaluation = json.loads((evals[-1] / "eval_manifest.json").read_text(encoding="utf-8"))
    return manifest, evaluation


def contrast(a: Path | tuple[dict, dict], b: Path | tuple[dict, dict]) -> dict:
    ma, ea = load(a) if isinstance(a, Path) else a
    mb, eb = load(b) if isinstance(b, Path) else b
    result = to_jsonable(evaluate_contrast_validity(ma, mb, "stress", evaluations=(ea, eb)))
    result["run_a_dir"] = str(a.relative_to(REPO)) if isinstance(a, Path) else "in-memory"
    result["run_b_dir"] = str(b.relative_to(REPO)) if isinstance(b, Path) else "in-memory"
    return result


def brief(r: dict) -> dict:
    rg = r["realization_guard"]
    return {
        "run_a": r["run_a_dir"], "run_b": r["run_b_dir"], "overall": r["overall_status"],
        "attribution_allowed": r["attribution_allowed"],
        "design": r["design_guard"]["status"], "design_violations": r["design_guard"]["violations"],
        "intended_factor_varied": r["design_guard"]["intended_factor_varied"],
        "realization": rg["status"], "realization_basis": rg["basis"],
        "realization_available": rg["evidence_available"],
        "realization_mismatches": [{k: m.get(k) for k in ("condition", "reason", "run_a", "run_b")} for m in rg["mismatches"]],
        "realization_unresolved": [{k: u.get(k) for k in ("condition", "reason", "equivalence", "validity_a", "validity_b")
                                    if u.get(k) is not None} for u in rg["unresolved"]],
        "measurement": r["measurement_guard"]["status"], "measurement_violations": r["measurement_guard"]["violations"],
    }


def main(out: str) -> None:
    base = runs("baseline")
    assert len(base) == 10, base
    result: dict = {"note": "stress contrasts between nominal runs = replicate designs; guards unchanged"}

    # ---- G-A: all pairs of the 10 baseline runs
    pairs = [brief(contrast(a, b)) for a, b in itertools.combinations(base, 2)]
    unresolved_reasons = Counter((u["condition"], u["reason"]) for p in pairs for u in p["realization_unresolved"])
    result["G-A"] = {
        "pairs": len(pairs),
        "overall": Counter(p["overall"] for p in pairs),
        "design": Counter(p["design"] for p in pairs),
        "realization": Counter(p["realization"] for p in pairs),
        "measurement": Counter(p["measurement"] for p in pairs),
        "attribution_allowed": Counter(p["attribution_allowed"] for p in pairs),
        "any_realization_mismatch": sum(1 for p in pairs if p["realization_mismatches"]),
        "unresolved_condition_reasons": {f"{c}:{r}": n for (c, r), n in sorted(unresolved_reasons.items())},
        "available_conditions_in_every_pair": sorted(set.intersection(*(set(p["realization_available"]) for p in pairs))),
        "representative_pair": pairs[0],
    }

    # ---- G-B: baseline vs same plan with a foreign idle context
    gb = runs("g-b-foreign-idle-context")
    gb_pairs = [brief(contrast(a, b)) for a, b in zip(base[:5], gb)]
    result["G-B"] = {
        "pairs": len(gb_pairs),
        "same_plan_digest": all(load(a)[0]["plan_digest"] == load(b)[0]["plan_digest"] for a, b in zip(base[:5], gb)),
        "overall": Counter(p["overall"] for p in gb_pairs),
        "realization": Counter(p["realization"] for p in gb_pairs),
        "design": Counter(p["design"] for p in gb_pairs),
        "measurement": Counter(p["measurement"] for p in gb_pairs),
        "attribution_allowed": Counter(p["attribution_allowed"] for p in gb_pairs),
        "representative_pair": gb_pairs[0],
    }

    # ---- G-C: measurement configuration mismatch
    result["G-C"] = {
        "instrumentation_cuda_events_vs_none": brief(contrast(base[0], runs("no_events")[0])),
        "profiler_none_vs_nsys": brief(contrast(base[0], runs("nsys")[0])),
    }

    # ---- G-D: missing / unavailable evidence
    ma, ea = load(base[0])
    mb, eb = load(base[1])
    removed = copy.deepcopy(mb)
    removed["realization"]["evidence"] = [e for e in removed["realization"]["evidence"]
                                          if e["condition"] != "competing_device_users"]
    real = result["G-A"]["representative_pair"]
    result["G-D"] = {
        "genuinely_unavailable": {
            "memory_pressure_bandwidth": [u for u in real["realization_unresolved"] if u["condition"] == "memory_pressure_bandwidth"],
            "note": "not_observable in every actual run: DRAM bandwidth/traffic need GPU performance counters, "
                    "which RmProfilingAdminOnly=1 restricts to administrators on this host",
            "pair": {k: real[k] for k in ("run_a", "run_b", "overall", "realization")},
        },
        "removed_record_copy": {
            "description": "in-memory copy of run_b's manifest with the observed competing_device_users record "
                           "deleted (guard-behaviour check only; not a run, nothing written)",
            "intact_pair": {k: real[k] for k in ("overall", "realization", "realization_available")},
            "modified_pair": brief(contrast((ma, ea), (removed, eb))),
        },
    }
    Path(out).write_text(json.dumps(result, indent=1, sort_keys=True, default=dict), encoding="utf-8")
    for case in ("G-A", "G-B"):
        print(case, json.dumps({k: v for k, v in result[case].items() if k != "representative_pair"}, default=dict, sort_keys=True))
    for name, r in result["G-C"].items():
        print("G-C", name, json.dumps({k: r[k] for k in ("overall", "design", "design_violations", "realization",
                                                           "measurement", "measurement_violations")}, sort_keys=True))
    m = result["G-D"]["removed_record_copy"]["modified_pair"]
    print("G-D removed", json.dumps({k: m[k] for k in ("overall", "realization", "realization_available")}, sort_keys=True),
          [u for u in m["realization_unresolved"] if u["condition"] == "competing_device_users"])
    print("G-D real", json.dumps(result["G-D"]["genuinely_unavailable"]["memory_pressure_bandwidth"]))
    print("G-B mismatch", json.dumps(result["G-B"]["representative_pair"]["realization_mismatches"]))
    print("G-A unresolved", json.dumps(result["G-A"]["representative_pair"]["realization_unresolved"]))


if __name__ == "__main__":
    main(sys.argv[1])
