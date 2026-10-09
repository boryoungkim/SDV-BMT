#!/usr/bin/env python3
"""Extract an output reference (golden) from a designated reference-capture run bundle (offline, stdlib only).

Stage 2b-G real-model baseline.  A cuDNN execution is not bit-reproducible by an independent CPU
implementation, so the output identity diagnostic compares every baseline run with the encoded task outputs
of one earlier GPU run of the same code and configuration (the reference capture).  Identity therefore
diagnoses run-to-run determinism and output corruption only; it is not correctness (numerical correctness is
checked against ONNX Runtime separately, task quality against the COCO annotations).

    python3 tools/stage2b_g_real/capture_golden.py RUN_DIR OUT.golden.jsonl [--count N]

Writes one row per completed item ``{"item_index", "output_sha256", "source_item_id"}`` (the format of
``configs/references/*.golden.jsonl``), items 0..N-1 only when --count is given; refuses runs that are not
completed/valid or have a missing item.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out")
    ap.add_argument("--count", type=int)
    a = ap.parse_args()
    run = Path(a.run_dir)
    manifest = json.loads((run / "manifest.json").read_text())
    status = manifest["status"]
    if status["execution_status"] != "completed" or status["measurement_status"] != "valid":
        raise SystemExit(f"{run}: not a completed/valid run")
    items = {}
    for line in open(run / "input_items.jsonl", encoding="utf-8"):
        row = json.loads(line)
        items[row["item_index"]] = row["source_item_id"]
    request_item = {}
    rows = {}
    outputs = (run / "outputs.bin").read_bytes()
    for line in open(run / "events.jsonl", encoding="utf-8"):
        e = json.loads(line)
        if e.get("kind") == "issued":
            request_item[e["request_id"]] = e["item_indices"][0]
        elif e.get("kind") == "completed":
            ref = e["output_ref"]
            blob = outputs[ref["offset"]:ref["offset"] + ref["length"]]
            if hashlib.sha256(blob).hexdigest() != ref["sha256"]:
                raise SystemExit("output bytes do not match their recorded SHA-256")
            index = request_item[e["request_id"]]
            rows[index] = {"item_index": index, "output_sha256": ref["sha256"], "source_item_id": items[index]}
    count = a.count if a.count is not None else len(items)
    missing = [i for i in range(count) if i not in rows]
    if missing:
        raise SystemExit(f"items without a completed output: {missing[:10]}")
    text = "".join(json.dumps(rows[i], sort_keys=True, separators=(",", ":")) + "\n" for i in range(count))
    Path(a.out).write_text(text, encoding="utf-8")
    print(json.dumps({"run_id": manifest["run_id"], "rows": count,
                      "sha256": hashlib.sha256(text.encode()).hexdigest()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
