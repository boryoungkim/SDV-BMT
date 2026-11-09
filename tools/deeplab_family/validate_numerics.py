#!/usr/bin/env python3
"""v0.1.2 DeepLab numeric gates (plan v03 sections 7.3 and 8.1): GPU cuDNN plan vs ONNX Runtime CPU (offline tool).

Run with the system Python (onnxruntime, onnx and numpy are used by offline tools only).  For each input item:
  * the canonical Workload preprocessing (sdv_bmt, stdlib) produces the input tensor;
  * the native NVIDIA backend (``native/build/sdv-bmt-raw-dump``, the runner's backend code, refused unless its
    build stamp matches the current native sources and raw_dump.cpp) runs the canonical plan (uint8 label map) and
    the diagnostic plan (class logits before the final resize and argmax) with the given DUT runtime options;
  * ONNX Runtime (CPU EP) runs the workload's ONNX artifact (SHA-256 checked) with the logits tensor exposed.

Before any comparison the conv-condition check (8.1 step 5) must pass: the canonical and the diagnostic plan have
the same convolutions (every cuDNN descriptor value, weight/bias offsets, activation, residual), the same weights
and arena, differ only in the last convolution's destination and the two final steps, and their realized
algorithm tables and per-convolution heuristic choices on this device are identical.

Gates (failure = stop before the baselines; thresholds are never relaxed here):
  2  logits: r_i = max|GPU - ORT| / max|ORT| per item, max over items <= 1e-4 (user decision 2026-11-06)
  3  labels: identical pixels / all pixels >= 0.9999
  4  label values all in 0..18
  5  |mIoU(GPU) - mIoU(ORT)| <= 1e-4 (needs the ground truth: --input-config)
  6  |mIoU(ORT) - model zoo| <= 0.003 (--zoo-miou)
  7  native label-map blobs of a bundle (--bundle) == the Python codec of the GPU label maps (byte equal)

    python3 tools/deeplab_family/validate_numerics.py --workload configs/workloads/deeplabv3plus-r50-cityscapes-v1.yaml \
        --dut configs/duts/nvidia-rtx-pro-6000-gpu0-cudnn-deeplabv3plus-r50.yaml \
        (--input-config configs/inputs/cityscapes-val500.yaml | --synthetic N) --out REPORT.json \
        [--zoo-miou 0.8009] [--bundle RUN_DIR] [--skip-ort] [--chunk 25] [--threads 16]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdv_bmt.contracts import labelmap  # noqa: E402
from sdv_bmt.contracts.ids import canonical_json_bytes, digest_of  # noqa: E402
from sdv_bmt.contracts.specs import DUTConfig, WorkloadSpec, parse_input_spec, parse_strict  # noqa: E402
from sdv_bmt.dut import native_source  # noqa: E402
from sdv_bmt.dut.native import NATIVE_DIR  # noqa: E402
from sdv_bmt.dut.nvidia.adapter import NvidiaCudaDUT  # noqa: E402
from sdv_bmt.evidence.provenance import code_identity  # noqa: E402
from sdv_bmt.input.annotations import IndexAnnotationSource  # noqa: E402
from sdv_bmt.input.dataset_fixture import DatasetFixtureProvider  # noqa: E402
from sdv_bmt.metrics.quality_segmentation import ConfusionCounts, ground_truth  # noqa: E402
from sdv_bmt.runner.plan import _parse_file, load_yaml_strict  # noqa: E402
from sdv_bmt.workload.deeplab import DeepLabCityscapesWorkload  # noqa: E402
import inputs as I  # noqa: E402
from conv_conditions import conv_conditions  # noqa: E402

RAW_DUMP = NATIVE_DIR / "build" / "sdv-bmt-raw-dump"
RAW_DUMP_SRC = REPO / "tools" / "stage2b_g_real" / "raw_dump.cpp"
LOGIT_REL_MAX = 1e-4
LABEL_AGREEMENT_MIN = 0.9999
MIOU_DIFF_MAX = 1e-4
ZOO_DIFF_MAX = 0.003
NUM_CLASSES = 19


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quantiles(v: list[float]) -> dict:
    a = np.sort(np.asarray(v, dtype=np.float64))
    if not len(a):
        return {}
    pick = lambda p: float(a[min(len(a) - 1, max(0, int(np.ceil(p * len(a))) - 1))])  # noqa: E731
    return {"n": int(len(a)), "min": float(a[0]), "p50": pick(0.5), "p90": pick(0.9), "p99": pick(0.99),
            "max": float(a[-1])}


def diag_workload(w: WorkloadSpec, diag_plan: Path) -> WorkloadSpec:
    plan = json.loads(diag_plan.read_text(encoding="utf-8"))
    n = plan["output"]["bytes"]
    program = dict(w.options["device_program"])
    program.update(plan=str(diag_plan.relative_to(REPO)), plan_sha256=sha256_file(diag_plan), task_output_bytes=n,
                   output_bytes=n + 32, timing_offset=n)
    return dataclasses.replace(w, options={**w.options, "device_program": program})


class RawDump:
    def __init__(self, dut: NvidiaCudaDUT, dut_cfg: DUTConfig, scratch: Path) -> None:
        self.dut, self.cfg, self.scratch = dut, dut_cfg, scratch
        self.source = native_source.check_stamp(RAW_DUMP, NATIVE_DIR, [RAW_DUMP_SRC])  # refuses a stale build
        self.sha256 = sha256_file(RAW_DUMP)
        self.facts: dict[str, dict] = {}

    def run(self, tag: str, workload: WorkloadSpec, tensors: list[tuple[int, bytes]], out_bytes: int) -> list[bytes]:
        spec = {"device_uuid": self.cfg.device, "backend": {"name": self.dut.ADAPTER,
                                                           "options": dict(self.cfg.runtime_options)},
                **self.dut.backend_spec(workload)}
        d = Path(tempfile.mkdtemp(prefix=f"rd-{tag}-", dir=self.scratch))
        rows = []
        with open(d / "staging.bin", "wb") as fh:
            for idx, data in tensors:
                rows.append({"item_index": idx, "offset": fh.tell(), "nbytes": len(data),
                             "sha256": hashlib.sha256(data).hexdigest()})
                fh.write(data)
        (d / "spec.json").write_bytes(canonical_json_bytes(spec))
        (d / "items.json").write_text(json.dumps(rows))
        proc = subprocess.run([str(RAW_DUMP), str(d / "spec.json"), str(d / "staging.bin"), str(d / "items.json"),
                               str(d / "out.bin")], capture_output=True, text=True)
        try:
            reply = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise SystemExit(f"raw_dump ({tag}) gave no reply: {proc.stderr[-2000:]}") from None
        if not reply.get("ok"):
            raise SystemExit(f"raw_dump ({tag}) failed: {reply} {proc.stderr[-2000:]}")
        facts = reply["cudnn_plan_facts"]
        prev = self.facts.setdefault(tag, facts)
        for key in ("cudnn_algorithm_table_sha256", "plan_sha256", "weights_sha256"):
            if prev[key] != facts[key]:
                raise SystemExit(f"raw_dump ({tag}): {key} changed between chunks")
        raw = (d / "out.bin").read_bytes()
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
        if len(raw) != out_bytes * len(tensors):
            raise SystemExit(f"raw_dump ({tag}): {len(raw)} output bytes, expected {out_bytes * len(tensors)}")
        return [raw[k * out_bytes:(k + 1) * out_bytes] for k in range(len(tensors))]


def device_conv_conditions(rd: RawDump) -> list[str]:
    a, b = rd.facts.get("canonical"), rd.facts.get("diagnostic")
    problems = []
    if a is None or b is None:
        return ["realized tables of both plans not available"]
    for key in ("cudnn_algorithm_table_sha256", "cudnn_algorithm_table", "cudnn_convolutions", "cudnn_math_type_option",
                "cudnn_algo_policy", "cudnn_workspace_limit_bytes", "weights_sha256", "arena_bytes"):
        if a[key] != b[key]:
            problems.append(f"realized {key} differs")
    strip = lambda d: [{k: v for k, v in r.items()} for r in d]  # noqa: E731
    if strip(a["cudnn_algorithm_detail"]) != strip(b["cudnn_algorithm_detail"]):
        problems.append("per-convolution algorithm detail (incl. heuristic choice) differs")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--dut", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-config")
    src.add_argument("--synthetic", type=int)
    ap.add_argument("--count", type=int)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--zoo-miou", type=float)
    ap.add_argument("--bundle")
    ap.add_argument("--skip-ort", action="store_true", help="gate 7 only (with --bundle)")
    ap.add_argument("--scratch", default="/dev/shm")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    wpath, dpath = Path(a.workload).resolve(), Path(a.dut).resolve()
    w = _parse_file(wpath, lambda d, where: parse_strict(WorkloadSpec, d, where))
    dut_cfg = _parse_file(dpath, lambda d, where: parse_strict(DUTConfig, d, where))
    workload = DeepLabCityscapesWorkload(w)
    dut = NvidiaCudaDUT(dut_cfg)
    program = w.options["device_program"]
    canon_plan_path = REPO / program["plan"]
    diag_plan_path = canon_plan_path.with_name(canon_plan_path.name.replace(".plan.json", ".diag.plan.json"))
    canon = json.loads(canon_plan_path.read_text(encoding="utf-8"))
    diag = json.loads(diag_plan_path.read_text(encoding="utf-8"))
    wd = diag_workload(w, diag_plan_path)
    H, W = w.options["input_height"], w.options["input_width"]
    label_bytes, logit_bytes = H * W, diag["output"]["bytes"]
    plan_check = conv_conditions(canon, diag)
    if plan_check["plan_problems"]:
        raise SystemExit(f"conv-condition check failed (plans): {plan_check['plan_problems']}")
    onnx_path = canon_plan_path.parent / canon["model"]["onnx_file"]
    if sha256_file(onnx_path) != w.artifact_sha256:
        raise SystemExit(f"{onnx_path}: SHA-256 differs from the workload's artifact_sha256")

    # ---------------------------------------------------------------- inputs
    gt_rows: dict[int, dict] = {}
    if a.input_config:
        ipath = Path(a.input_config).resolve()
        spec = parse_input_spec(load_yaml_strict(ipath), str(ipath))
        contract = workload.input_contract()
        if spec.family.preprocessing_contract != f"{contract.contract_id}/{contract.contract_version}":
            raise SystemExit("input config preprocessing_contract differs from the workload's input contract")
        provider = DatasetFixtureProvider(spec, ipath.parent)
        items = list(provider.items())
        provider.open()
        if a.count:
            items = items[:a.count]
        for row in IndexAnnotationSource(spec, ipath.parent).reference_rows(items):
            gt_rows[row["item_index"]] = row["annotation"]
        load = lambda it: workload.preprocess(provider.payload(it.item_index), it)  # noqa: E731
        names = {it.item_index: it.source_item_id for it in items}
        indices = [it.item_index for it in items]
        by_index = {it.item_index: it for it in items}
        items_digest = digest_of([[it.item_index, it.source_item_id, it.payload_sha256] for it in items])
        input_desc = {"input_config": str(ipath.relative_to(REPO)), "items": len(items), "items_digest": items_digest}
    else:
        n = a.synthetic
        indices = list(range(n))
        names = {i: f"synthetic-{i}" for i in indices}
        load = None
        input_desc = {"synthetic": n, "generator": "tools/deeplab_family/inputs.py synthetic_rgb",
                      "inputs_py_sha256": sha256_file(Path(I.__file__))}

    def tensor(i: int) -> bytes:
        if load is not None:
            return load(by_index[i])
        return I.tensor_from_rgb(I.synthetic_rgb(i, H, W), H, W).tobytes()

    # ---------------------------------------------------------------- ONNX Runtime
    sess = logits_name = None
    if not a.skip_ort:
        import onnx  # noqa: E402
        import onnxruntime as ort  # noqa: E402
        m = onnx.load(str(onnx_path))
        logits_name = diag["output"]["name"]
        m.graph.output.append(onnx.helper.make_tensor_value_info(logits_name, onnx.TensorProto.FLOAT,
                                                                 [1, NUM_CLASSES, H // 4, W // 4]))
        so = ort.SessionOptions()
        so.intra_op_num_threads = a.threads
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
        ort_version = ort.__version__
    scratch = Path(a.scratch)
    rd = RawDump(dut, dut_cfg, scratch)

    bundle_blobs: dict[int, bytes] = {}
    if a.bundle:
        bdir = Path(a.bundle)
        outputs = (bdir / "outputs.bin").read_bytes()
        request_item = {}
        for line in open(bdir / "events.jsonl", encoding="utf-8"):
            e = json.loads(line)
            if e.get("kind") == "issued":
                request_item[e["request_id"]] = e["item_indices"][0]
            elif e.get("kind") == "completed" and e.get("output_ref"):
                ref = e["output_ref"]
                bundle_blobs[request_item[e["request_id"]]] = outputs[ref["offset"]:ref["offset"] + ref["length"]]

    rel, rel_rows, agree_rows = [], [], []
    same = total = bad_values = 0
    twin_equal = twin_compared = 0
    cm_gpu, cm_ort = ConfusionCounts(), ConfusionCounts()
    for start in range(0, len(indices), a.chunk):
        chunk = indices[start:start + a.chunk]
        tensors = [(i, tensor(i)) for i in chunk]
        labels = rd.run("canonical", w, tensors, label_bytes)
        logits = rd.run("diagnostic", wd, tensors, logit_bytes) if sess is not None else [None] * len(chunk)
        if start == 0:
            device_problems = device_conv_conditions(rd) if sess is not None else []
            if device_problems:
                raise SystemExit(f"conv-condition check failed (device): {device_problems}")
        for (i, data), lab, lg in zip(tensors, labels, logits):
            gpu_lab = np.frombuffer(lab, dtype=np.uint8)
            bad_values += int((gpu_lab >= NUM_CLASSES).sum())
            if i in bundle_blobs:
                twin_compared += 1
                twin_equal += bundle_blobs[i] == labelmap.encode(lab, W, H, i)
            if sess is None:
                continue
            x = np.frombuffer(data, dtype="<f4").reshape(1, 3, H, W)
            ort_lab, ort_lg = sess.run(["labels", logits_name], {"input": x})
            g = np.frombuffer(lg, dtype="<f4").reshape(ort_lg.shape).astype(np.float64)
            o = ort_lg.astype(np.float64)
            r = float(np.abs(g - o).max() / np.abs(o).max())
            rel.append(r)
            eq = int((gpu_lab == ort_lab.reshape(-1)).sum())
            same += eq
            total += gpu_lab.size
            rel_rows.append({"item_index": i, "name": names[i], "logits_max_rel": r,
                             "logits_max_abs": float(np.abs(g - o).max()), "ort_logits_max_abs": float(np.abs(o).max())})
            agree_rows.append({"item_index": i, "label_mismatch_pixels": int(gpu_lab.size - eq)})
            if i in gt_rows:
                _, _, gt = ground_truth(gt_rows[i])
                cm_gpu.add(lab, gt)
                cm_ort.add(ort_lab.astype(np.uint8).tobytes(), gt)
        print(f"[{time.time() - t0:7.1f} s] {start + len(chunk)}/{len(indices)} items", flush=True)

    gates: dict[str, dict] = {}
    if sess is not None:
        worst = max(rel)
        gates["2_logits"] = {"criterion": f"max_i r_i <= {LOGIT_REL_MAX}", "max": worst, "distribution": quantiles(rel),
                             "worst_items": sorted(rel_rows, key=lambda r: -r["logits_max_rel"])[:10],
                             "passed": worst <= LOGIT_REL_MAX}
        agreement = same / total
        gates["3_labels"] = {"criterion": f"agreement >= {LABEL_AGREEMENT_MIN}", "agreement": agreement,
                             "mismatch_pixels": total - same, "pixels": total,
                             "mismatch_per_item": quantiles([r["label_mismatch_pixels"] for r in agree_rows]),
                             "passed": agreement >= LABEL_AGREEMENT_MIN}
    gates["4_label_values"] = {"criterion": "every GPU label < 19", "out_of_range": bad_values, "passed": bad_values == 0}
    if gt_rows and sess is not None:
        mg, ig = cm_gpu.result()
        mo, io = cm_ort.result()
        gates["5_miou_gpu_vs_ort"] = {"criterion": f"|diff| <= {MIOU_DIFF_MAX}", "miou_gpu": mg, "miou_ort": mo,
                                      "diff": abs(mg - mo), "class_iou_gpu": ig, "class_iou_ort": io,
                                      "passed": abs(mg - mo) <= MIOU_DIFF_MAX}
        if a.zoo_miou is not None:
            gates["6_model_zoo"] = {"criterion": f"|mIoU(ORT) - zoo| <= {ZOO_DIFF_MAX}", "zoo": a.zoo_miou,
                                    "miou_ort": mo, "diff": abs(mo - a.zoo_miou),
                                    "passed": abs(mo - a.zoo_miou) <= ZOO_DIFF_MAX}
    if a.bundle:
        gates["7_twin"] = {"criterion": "native blob == Python codec of the GPU label map, every item",
                           "compared": twin_compared, "equal": twin_equal,
                           "passed": twin_compared == len(indices) and twin_equal == twin_compared}
    device = dut.device_identity()
    report = {
        "schema": "sdv-bmt-deeplab-numerics/1",
        "workload": str(wpath.relative_to(REPO)), "dut": str(dpath.relative_to(REPO)), "inputs": input_desc,
        "identity": {
            "python_code": code_identity()["source_tree_sha256"],
            "native_source_tree_sha256": native_source.native_source_tree_sha256(NATIVE_DIR),
            "raw_dump": {"sha256": rd.sha256, "stamp": rd.source},
            "onnx_sha256": w.artifact_sha256,
            "plan_sha256": program["plan_sha256"], "diag_plan_sha256": sha256_file(diag_plan_path),
            "weights_sha256": program["weights_sha256"],
            "algorithm_table_sha256": rd.facts["canonical"]["cudnn_algorithm_table_sha256"],
            "cudnn_algo_policy": rd.facts["canonical"]["cudnn_algo_policy"],
            "cudnn_math_type_option": rd.facts["canonical"]["cudnn_math_type_option"],
            "cudnn_version": device.get("nvidia.cudnn_version"), "driver": device.get("nvidia.driver_version"),
            "device_uuid": device.get("nvidia.uuid"),
            "onnxruntime": None if sess is None else ort_version, "numpy": np.__version__,
            "tool_sha256": {"validate_numerics.py": sha256_file(Path(__file__)), "inputs.py": sha256_file(Path(I.__file__))},
        },
        "conv_conditions": {**plan_check, "device_problems": [] if sess is None else device_conv_conditions(rd)},
        "realized": {k: {kk: v[kk] for kk in ("cudnn_algorithm_table_sha256", "cudnn_convolutions",
                                               "cudnn_heuristic_agreements", "cudnn_workspace_bytes", "arena_bytes",
                                               "plan_launches_per_request", "glue_source_sha256", "glue2_source_sha256")
                         if kk in v} for k, v in rd.facts.items()},
        "realized_table": rd.facts["canonical"]["cudnn_algorithm_table"],
        "gates": gates,
        "passed": all(g["passed"] for g in gates.values()),
        "seconds": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], **{k: g["passed"] for k, g in gates.items()},
                      "logits_max_rel": gates.get("2_logits", {}).get("max"),
                      "label_agreement": gates.get("3_labels", {}).get("agreement")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
