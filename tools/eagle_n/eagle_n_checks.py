#!/usr/bin/env python3
"""Host-side checks for tools/eagle_n/run_eagle_n.sh. No device access, no Docker, no sudo.

Every pinned value comes from the files themselves, so this script holds no hash and no measurement:
  * the qualification receipt named by the workload (`options.device_program.qualification`, pinned by
    `qualification_sha256`) binds the executor, its source digest, the plan and weights SHA-256 and the L1 NPU mIoU
    (`gate.npu_miou`);
  * the NPU golden is the file named by the baseline plan (`evaluation.output_reference`).

Commands:
  pins     CAPTURE_PLAN BASELINE_PLAN   refuse unless receipt, artifacts, executor source and runner build agree
  result   BUNDLE LEVEL CAPTURE_PLAN    print the RESULT line of an evaluated bundle and check it (LEVEL l3, l4 or l5)
  identity GOLDEN SECOND OUT_JSON       compare two capture goldens row by row (L4 identity), write OUT_JSON
Exit status 0 = pass, 1 = refused, 3 = checks not met.
"""
from __future__ import annotations

import glob
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sdv_bmt.dut.eagle_n.executor_source import executor_source_sha256, read_build_record  # noqa: E402
from sdv_bmt.dut.native import native_source_tree_sha256  # noqa: E402
from sdv_bmt.runner.plan import load_plan  # noqa: E402

RUNNER_STAMP = ROOT / "native/build/sdv-bmt-native-eagle-n.source-tree.sha256"


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def receipt_of(rp) -> tuple[Path, dict]:
    p = Path(rp.workload.options["device_program"]["qualification"])
    if not p.is_file():
        raise SystemExit(f"REFUSED: qualification receipt {p} not found (L1 + qualify_native.py first)")
    return p, json.loads(p.read_text())


def cmd_pins(capture: str, baseline: str) -> int:
    cp, bp = load_plan(capture), load_plan(baseline)
    rpath, r = receipt_of(cp)
    o = cp.dut.runtime_options
    p = cp.workload.options["device_program"]
    bad = []
    if r.get("passed") is not True:
        bad.append("receipt not passed")
    if r.get("variant") != p["variant"]:
        bad.append(f"receipt variant {r.get('variant')} != workload {p['variant']}")
    for key, path, cfg in (
        ("executor_sha256", o["executor"], o["executor_sha256"]),
        ("plan_sha256", p["plan"], p["plan_sha256"]),
        ("weights_sha256", p["weights"], p["weights_sha256"]),
    ):
        actual = sha(path)
        if not (r.get(key) == cfg == actual):
            bad.append(f"{key}: receipt {str(r.get(key))[:12]} config {cfg[:12]} file {actual[:12]}")
    if r.get("executor_source_sha256") != o["executor_source_sha256"]:
        bad.append("receipt names a different executor source than the DUT config")
    try:
        build = read_build_record(o["executor"])
        if build["executor_source_sha256"] != o["executor_source_sha256"]:
            bad.append("executor build record names a different source than the DUT config")
    except ValueError as exc:
        bad.append(str(exc))
    if executor_source_sha256() != o["executor_source_sha256"]:
        bad.append(
            "executor source of this checkout differs from the DUT config pin: rebuild the executor and requalify"
        )
    if p.get("qualification_sha256") != sha(rpath):
        bad.append("receipt SHA-256 differs from the workload pin qualification_sha256: rerun configure.py")
    if bp.workload.options["device_program"] != p or bp.dut.runtime_options != o:
        bad.append("capture and baseline plans do not use the same DUT and workload")
    if not RUNNER_STAMP.is_file() or RUNNER_STAMP.read_text().split()[0] != native_source_tree_sha256():
        bad.append("runner build stale: run `make -C native eagle-n` (runner only; never eagle-n-sdk)")
    if cp.plan.replay.mode.value != "asap":
        bad.append("capture plan is not asap")
    ref = bp.plan.evaluation.output_reference if bp.plan.evaluation else None
    golden = (Path(baseline).parent / ref.path) if ref and ref.path else None
    if golden is None or not golden.is_file() or sha(golden) != ref.sha256:
        bad.append("baseline plan has no NPU golden in configs/references (or its SHA-256 differs)")
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 1
    print(
        f"PINS OK {p['variant']}: receipt {sha(rpath)[:8]} executor {o['executor_sha256'][:8]} "
        f"source {o['executor_source_sha256'][:8]} plan {p['plan_sha256'][:8]} weights {p['weights_sha256'][:8]} "
        f"golden {ref.sha256[:8]} runner current; capture asap; expected mIoU {r['gate']['npu_miou']}"
    )
    return 0


def cmd_result(bundle: str, level: str, capture: str) -> int:
    _, r = receipt_of(load_plan(capture))
    items = load_plan(capture).input_spec.selection.count
    f = sorted(glob.glob(bundle + "/evaluations/*/metrics.json"))[-1]
    m = {v["metric_id"]: v["value"] for v in json.load(open(f))["metrics"]}
    ids = [
        "quality.segmentation.conditional.miou",
        "bmt.count.completed",
        "bmt.count.timed_out",
        "bmt.count.dropped",
        "bmt.count.error",
    ]
    if level == "l5":
        ids += [
            "bmt.output.identity_rate",
            "bmt.deadline.completion_miss_count",
            "bmt.deadline.output_ready_miss_count",
        ]
    print(f"RESULT {level}: " + ", ".join(f'{k.split(".", 1)[1]}={m.get(k)}' for k in ids))
    if level == "l5":
        # The baseline replays the GPU family rate: drops and deadline misses are results, not failures. Every
        # completed output must equal the NPU golden.
        ok = (
            m.get("bmt.count.timed_out") == 0
            and m.get("bmt.count.error") == 0
            and m.get("bmt.output.identity_rate") == 1.0
        )
        print("expected: timed_out 0, error 0, identity 1.0 (drops and deadline misses are reported, not checked)")
    else:
        ok = (
            m.get("bmt.count.completed") == items
            and m.get("bmt.count.timed_out") == 0
            and m.get("bmt.count.error") == 0
            and m.get("quality.segmentation.conditional.miou") == float(r["gate"]["npu_miou"])
        )
        print(f"expected: completed {items}, timed_out 0, error 0, mIoU == receipt {r['gate']['npu_miou']}")
    print("CHECK_OK" if ok else "CHECK_NOT_OK")
    return 0 if ok else 3


def cmd_identity(golden: str, second: str, out: str) -> int:
    rows = lambda p: {json.loads(line)["item_index"]: json.loads(line) for line in open(p)}
    a, b = rows(golden), rows(second)
    same = sum(a[i] == b.get(i) for i in a)
    res = {
        "golden": golden,
        "second": second,
        "items": len(a),
        "identical": same,
        "identity_rate": same / len(a) if a else 0.0,
        "differing": [i for i in a if a[i] != b.get(i)][:20],
    }
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    ok = bool(a) and same == len(a) == len(b)
    print("IDENTITY_OK" if ok else "IDENTITY_NOT_OK")
    return 0 if ok else 3


if __name__ == "__main__":
    cmds = {"pins": (cmd_pins, 2), "result": (cmd_result, 3), "identity": (cmd_identity, 3)}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds or len(sys.argv) != 2 + cmds[sys.argv[1]][1]:
        raise SystemExit(__doc__)
    fn, _ = cmds[sys.argv[1]]
    sys.exit(fn(*sys.argv[2:]))
