"""Build the Stage 1R-G reference conformance assets deterministically.

    .venv/bin/python tools/stage1r_g/build_reference_assets.py [--golden]
    .venv/bin/python tools/stage1r_g/build_reference_assets.py --repeat

Writes (all byte-deterministic):
* configs/fixtures/reference-images-v1.index.jsonl  128 synthetic 128x128 RGB images (base64 payloads)
* configs/inputs/reference-images-{32,128}.yaml     dataset input specs (first 32 / all 128) pinning the index
* configs/workloads/reference-convnet-v1.yaml       workload spec embedding the device program source
* with --golden: configs/references/reference-images-v1.ref-convnet-v1.golden.jsonl
  output SHA-256 per item from the independent CPU reference (no GPU involved)
* --repeat (only these files, Stage 2b-G repeated identical-input characterization):
  configs/fixtures/reference-images-v1-repeat0.index.jsonl  image 0 repeated 128 times (distinct item ids)
  configs/inputs/reference-images-repeat0-128.yaml          dataset input spec pinning that index
  configs/references/reference-images-v1-repeat0.ref-convnet-v1.golden.jsonl
  golden from the CPU reference of image 0 (no GPU involved)

The images are synthetic conformance inputs (gradients, rings, stripes, noise), not a dataset.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference_cpu import reference_logits  # noqa: E402

N_ITEMS = 128
SIDE = 128
INDEX = ROOT / "configs/fixtures/reference-images-v1.index.jsonl"
INPUT_SPECS = {32: ROOT / "configs/inputs/reference-images-32.yaml",
               128: ROOT / "configs/inputs/reference-images-128.yaml"}
WORKLOAD = ROOT / "configs/workloads/reference-convnet-v1.yaml"
GOLDEN = ROOT / "configs/references/reference-images-v1.ref-convnet-v1.golden.jsonl"
SOURCE = ROOT / "tools/stage1r_g/sdv_ref_convnet_v1.cu"
REPEAT_INDEX = ROOT / "configs/fixtures/reference-images-v1-repeat0.index.jsonl"
REPEAT_INPUT = ROOT / "configs/inputs/reference-images-repeat0-128.yaml"
REPEAT_GOLDEN = ROOT / "configs/references/reference-images-v1-repeat0.ref-convnet-v1.golden.jsonl"
PATTERNS = ("gradient", "rings", "stripes", "noise")


def image(i: int) -> bytes:
    px = bytearray(SIDE * SIDE * 3)
    kind = PATTERNS[i % 4]
    state = 1000 + i
    cx, cy = 40 + (i * 11) % 48, 40 + (i * 17) % 48
    for y in range(SIDE):
        for x in range(SIDE):
            if kind == "gradient":
                r, g, b = (x * 2 + i * 7) % 256, (y * 2 + i * 13) % 256, (x + y + i * 29) % 256
            elif kind == "rings":
                v = (math.isqrt((x - cx) ** 2 + (y - cy) ** 2) * (3 + i % 5)) % 256
                r, g, b = v, 255 - v, (v * 2) % 256
            elif kind == "stripes":
                v = 255 if ((x + (i % 3) * y) // (4 + (i % 7) * 3)) % 2 else 0
                r, g, b = v, (v + i * 31) % 256, (128 + x) % 256
            else:
                rgb = []
                for _ in range(3):
                    state = (1103515245 * state + 12345) & 0x7FFFFFFF
                    rgb.append((state >> 16) & 0xFF)
                r, g, b = rgb
            o = (y * SIDE + x) * 3
            px[o], px[o + 1], px[o + 2] = r, g, b
    return bytes(px)


def write_index() -> str:
    lines = []
    for i in range(N_ITEMS):
        row = {"source_item_id": f"ref-img-{i:03d}",
               "payload_text": base64.b64encode(image(i)).decode("ascii"),
               "metadata": {"pattern": PATTERNS[i % 4], "generator": "build_reference_assets.py v1"}}
        lines.append(json.dumps(row, sort_keys=True, separators=(",", ":")))
    INDEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return hashlib.sha256(INDEX.read_bytes()).hexdigest()


def write_input_spec(index_sha: str, count: int) -> None:
    INPUT_SPECS[count].write_text(f"""\
# Stage 1R-G reference accelerator conformance input: {count} synthetic 128x128 RGB images
# (gradients, rings, stripes, noise; base64 payloads). Not a task dataset.
schema_version: 1
input_kind: dataset
input_id: reference-images-{count}
version: "1"
family:
  source_dataset: sdv-bmt-synthetic-reference-images
  source_version: "1"
  task: reference-conformance
  annotation_schema: null
  preprocessing_contract: sdv-ref-rgb8-to-chw512-f32/1
source:
  index: ../fixtures/reference-images-v1.index.jsonl
  index_sha256: {index_sha}
selection:
  start: 0
  count: {count}
""", encoding="utf-8")


def write_workload() -> str:
    source = SOURCE.read_text(encoding="utf-8")
    sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    body = "".join(f"      {line}\n" if line else "\n" for line in source.rstrip("\n").split("\n"))
    WORKLOAD.write_text(f"""\
# Stage 1R-G reference accelerator conformance workload (NOT a task model, NOT an NPU workload).
# Exactly representable arithmetic: outputs are bit-identical across runs and equal to the integer
# CPU reference (tools/stage1r_g/reference_cpu.py). The device program below is the model artifact;
# artifact_sha256 is the SHA-256 of options.device_program.source (generated from
# tools/stage1r_g/sdv_ref_convnet_v1.cu by tools/stage1r_g/build_reference_assets.py).
schema_version: 1
workload_id: reference-convnet-v1
implementation: reference_convnet
task: reference-conformance
artifact_sha256: {sha}
precision: fp32-exact-int-accumulate
batch: 1
preprocess: {{id: sdv-ref-nearest4-pow2norm, version: "1"}}
postprocess: {{id: sdv-ref-logits-top1, version: "1"}}
codec: {{id: canonical-json-utf8, version: "1"}}
output_ready: canonical task output (logits, top1) encoded and appended to outputs.bin
quality_evaluator: null
options:
  device_program:
    language: cuda_c
    entry: sdv_ref_convnet_v1
    block_threads: 1024
    input_bytes: 3145728
    scratch_bytes: [196608, 1048576]
    output_bytes: 112
    task_output_bytes: 80
    timing_offset: 80
    source: |
{body}""", encoding="utf-8")
    return sha


def write_golden() -> None:
    from sdv_bmt.contracts.specs import parse_input_spec, parse_strict, WorkloadSpec
    from sdv_bmt.input.dataset_fixture import DatasetFixtureProvider
    from sdv_bmt.runner.plan import load_yaml_strict
    from sdv_bmt.workload.reference import ReferenceConvNetWorkload

    spec = parse_input_spec(load_yaml_strict(INPUT_SPECS[N_ITEMS]))
    provider = DatasetFixtureProvider(spec, INPUT_SPECS[N_ITEMS].parent)
    workload = ReferenceConvNetWorkload(parse_strict(WorkloadSpec, load_yaml_strict(WORKLOAD)))
    lines = []
    for item in provider.items():
        logits = reference_logits(image(item.item_index))
        raw = struct.pack("<10q", *logits)
        blob = workload.encode(workload.postprocess(raw, item))
        lines.append(json.dumps({"item_index": item.item_index, "output_sha256": hashlib.sha256(blob).hexdigest(),
                                 "source_item_id": item.source_item_id}, sort_keys=True, separators=(",", ":")))
        print(item.item_index, logits, flush=True)
    GOLDEN.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_repeat() -> None:
    """Image 0 repeated N_ITEMS times: identical logical input for every request."""
    from sdv_bmt.contracts.specs import parse_input_spec, parse_strict, WorkloadSpec
    from sdv_bmt.input.dataset_fixture import DatasetFixtureProvider
    from sdv_bmt.runner.plan import load_yaml_strict
    from sdv_bmt.workload.reference import ReferenceConvNetWorkload

    payload = base64.b64encode(image(0)).decode("ascii")
    rows = [json.dumps({"source_item_id": f"ref-img-000-rep-{k:03d}", "payload_text": payload,
                        "metadata": {"pattern": PATTERNS[0], "repeat_of": "ref-img-000",
                                     "generator": "build_reference_assets.py v1"}},
                       sort_keys=True, separators=(",", ":")) for k in range(N_ITEMS)]
    REPEAT_INDEX.write_text("\n".join(rows) + "\n", encoding="utf-8")
    index_sha = hashlib.sha256(REPEAT_INDEX.read_bytes()).hexdigest()
    REPEAT_INPUT.write_text(f"""\
# Stage 2b-G repeated identical-input characterization: reference image 0 repeated {N_ITEMS} times
# (distinct item ids, identical payload bytes). Synthetic conformance input, not a task dataset.
schema_version: 1
input_kind: dataset
input_id: reference-images-repeat0-{N_ITEMS}
version: "1"
family:
  source_dataset: sdv-bmt-synthetic-reference-images-repeated
  source_version: "1"
  task: reference-conformance
  annotation_schema: null
  preprocessing_contract: sdv-ref-rgb8-to-chw512-f32/1
source:
  index: ../fixtures/reference-images-v1-repeat0.index.jsonl
  index_sha256: {index_sha}
selection:
  start: 0
  count: {N_ITEMS}
""", encoding="utf-8")
    spec = parse_input_spec(load_yaml_strict(REPEAT_INPUT))
    provider = DatasetFixtureProvider(spec, REPEAT_INPUT.parent)
    workload = ReferenceConvNetWorkload(parse_strict(WorkloadSpec, load_yaml_strict(WORKLOAD)))
    raw = struct.pack("<10q", *reference_logits(image(0)))
    lines = []
    for item in provider.items():
        blob = workload.encode(workload.postprocess(raw, item))
        lines.append(json.dumps({"item_index": item.item_index, "output_sha256": hashlib.sha256(blob).hexdigest(),
                                 "source_item_id": item.source_item_id}, sort_keys=True, separators=(",", ":")))
    REPEAT_GOLDEN.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("repeat_index_sha256", index_sha)
    print("repeat_golden_sha256", hashlib.sha256(REPEAT_GOLDEN.read_bytes()).hexdigest())


def main() -> None:
    if "--repeat" in sys.argv:
        write_repeat()
        return
    index_sha = write_index()
    for count in INPUT_SPECS:
        write_input_spec(index_sha, count)
    artifact_sha = write_workload()
    print("index_sha256", index_sha)
    print("artifact_sha256", artifact_sha)
    if "--golden" in sys.argv:
        write_golden()
        print("golden_sha256", hashlib.sha256(GOLDEN.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
