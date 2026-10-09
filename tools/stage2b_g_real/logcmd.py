#!/usr/bin/env python3
"""Run one command and append it, its return code and the tail of its output to ``ROOT/COMMANDS.log``.

    python3 tools/stage2b_g_real/logcmd.py ROOT --label TEXT -- COMMAND [ARGS...]

Used for the Stage 2b-G real-model asset preparation and analysis steps, so every evidence-producing command
is recorded in the same JSON-lines format as ``tools/stage1r_g/run_and_evaluate.py`` (stdlib only).
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: logcmd.py ROOT --label TEXT -- COMMAND [ARGS...]")
    split = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv[:split])
    cmd = argv[split + 1:]
    if not cmd:
        ap.error("no command")
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc)
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    ended = datetime.datetime.now(datetime.timezone.utc)
    entry = {"utc": started.isoformat(timespec="seconds"), "seconds": round((ended - started).total_seconds(), 1),
             "label": a.label, "command": cmd, "returncode": proc.returncode,
             "stdout_tail": proc.stdout[-3000:], "stderr_tail": proc.stderr[-3000:]}
    with open(root / "COMMANDS.log", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    sys.stdout.write(proc.stdout[-6000:])
    sys.stderr.write(proc.stderr[-6000:])
    print(f"[{a.label}] rc={proc.returncode} {entry['seconds']} s")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
