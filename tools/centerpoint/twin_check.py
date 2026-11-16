"""Output twin check: Python decode_heads + codec vs the native C++ twin (encode-output helper), byte for byte (tools only).

Feeds raw head outputs (float32 [70,128,128] files, one per item) to both implementations and compares the encoded
task outputs.  The native helper is ``<native runner> encode-output SPEC`` (stdin: {"item_index","raw_path",
"raw_offset","raw_bytes"} lines).

    python3 tools/centerpoint/twin_check.py --raw-dir DIR [--raw-dir DIR2 ...] --out OUT.json
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from sdv_bmt.contracts import det3d  # noqa: E402
from sdv_bmt.workload.centerpoint import CenterPointNuScenesWorkload, RAW_BYTES, decode_heads  # noqa: E402
from sdv_bmt.contracts.specs import ComponentIdentity, WorkloadSpec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runner", default=str(REPO / "native" / "build" / "sdv-bmt-native-fake"))
    a = ap.parse_args()
    W = CenterPointNuScenesWorkload
    spec = WorkloadSpec(schema_version=1, workload_id="twin-check", implementation=W.IMPLEMENTATION, task="nuscenes-det3d",
                        artifact_sha256="0" * 64, precision="fp32", batch=1, preprocess=W.PREPROCESS,
                        postprocess=W.POSTPROCESS, codec=W.CODEC, output_ready="x",
                        options={"device_program": {"input_bytes": 16480000, "task_output_bytes": RAW_BYTES}})
    pipeline = W(spec).native_output_pipeline()
    files = [f for d in a.raw_dir for f in sorted(Path(d).glob("*.f32"))]
    lines = "".join(json.dumps({"item_index": k, "raw_path": str(f), "raw_offset": 0, "raw_bytes": RAW_BYTES}) + "\n"
                    for k, f in enumerate(files))
    proc = subprocess.run([a.runner, "encode-output", json.dumps(pipeline)], input=lines, capture_output=True, text=True)
    native = [json.loads(l) for l in proc.stdout.splitlines() if l.startswith("{")]
    if len(native) != len(files):
        raise SystemExit(f"native helper returned {len(native)} of {len(files)} lines: {proc.stderr[-2000:]}")
    same, dets = 0, 0
    first_diff = None
    for k, (f, n) in enumerate(zip(files, native)):
        py = det3d.encode(k, decode_heads(f.read_bytes()))
        nb = bytes.fromhex(n["blob_hex"]) if "blob_hex" in n else None
        if nb == py:
            same += 1
        elif first_diff is None:
            first_diff = {"file": str(f), "native": n.get("error"), "py_len": len(py), "native_len": None if nb is None else len(nb)}
        dets += det3d.decode(py)[1].__len__()
    rep = {"items": len(files), "byte_identical": same, "detections": dets, "first_difference": first_diff,
           "pass": same == len(files), "native_runner_sha256": hashlib.sha256(Path(a.runner).read_bytes()).hexdigest()}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("items", "byte_identical", "detections", "pass")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
