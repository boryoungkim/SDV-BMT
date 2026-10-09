"""Stage 1R-G / 2b-G evidence driver: run a plan with the sdv-bmt CLI into an output root, evaluate every
run bundle it created, and append the commands and outcomes to ``<root>/COMMANDS.log`` (tool; stdlib).

    .venv/bin/python tools/stage1r_g/run_and_evaluate.py ROOT PLAN.yaml [--prefix "nsys profile ..."]
                                                          [--label TEXT] [--no-evaluate]

``--prefix`` runs the CLI under a wrapper (e.g. Nsight Systems); the words are split on spaces.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CLI = str(REPO / ".venv" / "bin" / "sdv-bmt")


def run_dirs(root: Path) -> set[Path]:
    return {p.parent for p in root.rglob("manifest.json")} | {p.parent for p in root.rglob("run_spec.json")}


def log(root: Path, entry: dict) -> None:
    entry = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), **entry}
    with open(root / "COMMANDS.log", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("plan")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--no-evaluate", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    before = run_dirs(root)
    cmd = shlex.split(a.prefix) + [CLI, "run", a.plan, "--out", str(root)]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    created = sorted(run_dirs(root) - before)
    outcome = None
    try:  # the CLI's JSON document; a wrapper (nsys) may print text before and after it
        outcome, _ = json.JSONDecoder().raw_decode(proc.stdout[proc.stdout.index("{"):])
    except ValueError:
        pass
    log(root, {"label": a.label, "command": cmd, "returncode": proc.returncode,
               "runs": [str(p.relative_to(root)) for p in created],
               "result": None if outcome is None else [
                   {k: o.get(k) for k in ("run_id", "execution_status", "execution_reason", "measurement_status",
                                          "measurement_reason", "request_counts")}
                   for o in outcome.get("runs", [])] or outcome.get("rejected") or outcome.get("issues") or outcome.get("status"),
               "stderr_tail": proc.stderr[-2000:] if proc.returncode else ""})
    print(f"[{a.label or Path(a.plan).stem}] rc={proc.returncode} runs={len(created)}")
    if outcome is not None:
        for o in outcome.get("runs", []):
            print("  ", o.get("run_id"), o.get("execution_status"), o.get("execution_reason"),
                  o.get("measurement_status"), o.get("measurement_reason"),
                  json.dumps(o.get("request_counts"), sort_keys=True))
        if outcome.get("rejected") or outcome.get("issues"):
            print("   rejected:", outcome.get("rejected") or outcome.get("issues"))
    if proc.returncode and outcome is None:
        print(proc.stdout[-3000:], proc.stderr[-3000:])
    if not a.no_evaluate:
        for run_dir in created:
            if not (run_dir / "manifest.json").exists():
                continue
            ev = subprocess.run([CLI, "evaluate", str(run_dir)], cwd=REPO, capture_output=True, text=True)
            log(root, {"label": a.label, "command": [CLI, "evaluate", str(run_dir.relative_to(root))],
                       "returncode": ev.returncode, "stderr_tail": ev.stderr[-2000:] if ev.returncode else ""})
            if ev.returncode:
                print("   evaluate failed:", ev.stdout[-1500:], ev.stderr[-1500:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
