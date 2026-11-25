#!/usr/bin/env python3
"""Point the 15 MLA400 YOLOX plans at an input set and at the NPU goldens captured on it (tool; stdlib only).

    python3 tools/mobilint/set_mla400_plans.py --input coco-val2017 [--check]

``--input`` names the 640 input configuration (``configs/inputs/<id>.yaml``); YOLOX-Tiny uses ``<id>-416``.
Per model:

* ``yolox-<m>-mla400-capture.yaml``: ``input:`` is set (a capture has no output reference);
* ``yolox-<m>-mla400-capture-identity.yaml`` and ``-baseline.yaml``: ``input:`` and ``evaluation.output_reference``
  (path and SHA-256) are set together, and only when the golden of that input exists:
  ``configs/references/<input>.yolox-<m>-coco-v1-mla400-q8.npu-capture.golden.jsonl`` (written from the capture by
  ``tools/mobilint/run_all_mla400.sh ROOT capture``). Until then these plans keep their previous input and golden,
  so a plan never pairs an input with the golden of another input.

The "first 1000" / "all 5,000" image count in the plan comments follows the input. ``--check`` writes nothing and
exits 1 when a plan would change or a golden is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP, INPUTS, REFS = REPO / "configs" / "experiments", REPO / "configs" / "inputs", REPO / "configs" / "references"
MODELS = ("tiny", "s", "m", "l", "x")
COUNT_TEXT = {"coco-val2017": "all 5,000", "coco-val2017-first1000": "first 1000"}


def input_of(base: str, model: str) -> str:
    return f"{base}-416" if model == "tiny" else base


def golden_name(input_id: str, model: str) -> str:
    return f"{input_id}.yolox-{model}-coco-v1-mla400-q8.npu-capture.golden.jsonl"


def rewrite(text: str, input_id: str, count: str | None, golden: Path | None) -> str:
    text = re.sub(r"^input: \.\./inputs/[a-z0-9-]+\.yaml$", f"input: ../inputs/{input_id}.yaml", text, flags=re.M)
    if count:
        text = re.sub(r"(first 1000|all 5,000)(\s*(?:#\s*)?)COCO val2017 images", rf"{count}\2COCO val2017 images",
                      text)
    if golden is not None:
        text = re.sub(r"^(    path: \.\./references/)\S+$", rf"\g<1>{golden.name}", text, flags=re.M)
        text = re.sub(r"^(    sha256: )[0-9a-f]{64}$", rf"\g<1>{hashlib.sha256(golden.read_bytes()).hexdigest()}",
                      text, flags=re.M)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="640 input configuration id, e.g. coco-val2017")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    changed, missing = [], []
    for model in MODELS:
        input_id = input_of(a.input, model)
        if not (INPUTS / f"{input_id}.yaml").is_file():
            print(f"error: configs/inputs/{input_id}.yaml does not exist", file=sys.stderr)
            return 2
        golden = REFS / golden_name(input_id, model)
        for kind in ("capture", "capture-identity", "baseline"):
            path = EXP / f"yolox-{model}-mla400-{kind}.yaml"
            old = path.read_text(encoding="utf-8")
            if kind != "capture" and not golden.is_file():
                missing.append(golden.name)
                continue
            new = rewrite(old, input_id, COUNT_TEXT.get(a.input), None if kind == "capture" else golden)
            if new != old:
                changed.append(path.name)
                if not a.check:
                    path.write_text(new, encoding="utf-8")
    for name in changed:
        print(("would change " if a.check else "wrote ") + name)
    for name in sorted(set(missing)):
        print(f"golden missing (identity and baseline plans of this model unchanged): configs/references/{name}")
    return 1 if a.check and (changed or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
