"""Native CenterPoint output pipeline (``native/src/common/centerpoint_output.hpp``) vs the Python reference
(``CenterPointNuScenesWorkload.postprocess`` + ``encode``): byte-identical task outputs, and the same failure, on
synthetic and adversarial raw head outputs (CenterPoint part)."""

from __future__ import annotations

import array
import json
import math
import random
import subprocess

import pytest

from sdv_bmt.contracts import det3d
from sdv_bmt.contracts.protocols import WorkloadError
from sdv_bmt.contracts.records import InputItem
from sdv_bmt.contracts.specs import WorkloadSpec, parse_strict
from sdv_bmt.runner.plan import load_yaml_strict
from sdv_bmt.workload.centerpoint import GRID, HEAD_CHANNELS, RAW_BYTES, TASKS, CenterPointNuScenesWorkload
from tests._helpers import CONFIGS

HW = GRID * GRID


def _workload() -> CenterPointNuScenesWorkload:
    return CenterPointNuScenesWorkload(parse_strict(
        WorkloadSpec, load_yaml_strict(CONFIGS / "workloads" / "centerpoint-pillar02-nus-v1.yaml")))


def _item(i: int) -> InputItem:
    return InputItem(item_index=i, source_item_id=f"s{i}", payload_ref=f"p{i}", payload_sha256="0" * 64,
                     annotation_ref=None, annotation_digest=None, source_timing=None, scenario_coordinates=None)


def _channels():
    """(first regression channel, heatmap channels) per task."""
    out, base = [], 0
    for names in TASKS:
        out.append((base, list(range(base + 10, base + 10 + len(names)))))
        base += 10 + len(names)
    return out


def _blank(heat: float = -20.0) -> array.array:
    a = array.array("f", [0.0]) * (HEAD_CHANNELS * HW)
    for _reg, heats in _channels():
        for ch in heats:
            a[ch * HW:(ch + 1) * HW] = array.array("f", [heat]) * HW
    return a


def _vectors() -> list[tuple[str, array.array]]:
    rng = random.Random(20261115)
    vecs = [("empty", _blank())]
    a = _blank()
    for reg, heats in _channels():  # random scene: many candidates above the threshold, NMS active
        for ch in heats:
            for cell in rng.sample(range(HW), 900):
                a[ch * HW + cell] = rng.uniform(-4.0, 3.0)
        for j in range(10):
            for cell in range(0, HW, 7):
                a[(reg + j) * HW + cell] = rng.uniform(-1.0, 1.0)
    vecs.append(("random", a))
    a = _blank()
    for t, (reg, heats) in enumerate(_channels()):  # dense clusters for circle NMS, both classes of a task
        for k, ch in enumerate(heats):
            for row in range(40, 60):
                for col in range(30 + 20 * t, 45 + 20 * t):
                    a[ch * HW + row * GRID + col] = 2.0 - 0.01 * ((row + col + k) % 9)
    vecs.append(("clusters", a))
    a = _blank()
    for reg, heats in _channels():  # exact score ties within and across the classes of a task (> 500 cells)
        for ch in heats:
            for cell in range(1000, 1700):
                a[ch * HW + cell] = 0.0
    vecs.append(("ties", a))
    a = _blank()
    reg, heats = _channels()[0]  # post-center range edges (inclusive) and just outside
    for cell, (r0, r1, z) in {127 * GRID + 127: (13.5, 13.5, 10.0), 0: (-12.5, -12.5, -10.0),
                              5 * GRID + 5: (0.0, 0.0, 10.000001), 9 * GRID + 127: (13.6, 0.0, 0.0)}.items():
        a[heats[0] * HW + cell] = 5.0
        a[reg * HW + cell], a[(reg + 1) * HW + cell], a[(reg + 2) * HW + cell] = r0, r1, z
    vecs.append(("range_edges", a))
    a = _blank()
    reg, heats = _channels()[5]  # infinite logits are allowed (score 1 / 0); huge dims of a dropped candidate
    a[heats[0] * HW + 77] = math.inf
    a[heats[1] * HW + 78] = -math.inf
    a[heats[1] * HW + 500] = -9.0  # score < 0.1: dropped before its values matter
    a[(reg + 3) * HW + 500] = 1000.0
    a[(reg + 8) * HW + 500] = math.nan
    vecs.append(("infinite_logits", a))
    a = _blank()
    a[_channels()[2][1][1] * HW + 4321] = math.nan
    vecs.append(("nan_logit", a))
    a = _blank()
    reg, heats = _channels()[1]
    a[heats[0] * HW + 999] = 4.0
    a[(reg + 4) * HW + 999] = 100.0  # exp(100) does not fit float32
    vecs.append(("dims_overflow", a))
    a = _blank()
    reg, heats = _channels()[3]
    a[heats[0] * HW + 2000] = 4.0
    a[(reg + 9) * HW + 2000] = math.nan  # velocity of a kept detection
    vecs.append(("velocity_nan", a))
    return vecs


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    w = _workload()
    raw_path = tmp_path_factory.mktemp("cp") / "raw.bin"
    lines, expected = [], []
    with open(raw_path, "wb") as fh:
        for i, (name, a) in enumerate(_vectors()):
            raw = a.tobytes()
            assert len(raw) == RAW_BYTES
            lines.append(json.dumps({"item_index": i, "raw_path": str(raw_path), "raw_offset": fh.tell(),
                                     "raw_bytes": len(raw)}) + "\n")
            fh.write(raw)
            try:
                expected.append((name, w.encode(w.postprocess(raw, _item(i))), None))
            except WorkloadError as exc:
                expected.append((name, None, str(exc)))
    return w, lines, expected


def test_native_centerpoint_pipeline_matches_python(native_build, cases):
    w, lines, expected = cases
    proc = subprocess.run([str(native_build / "sdv-bmt-native-fake"), "encode-output",
                           json.dumps(w.native_output_pipeline())], input="".join(lines), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    native = [json.loads(line) for line in proc.stdout.splitlines()]
    assert len(native) == len(expected)
    for row, (name, blob, error) in zip(native, expected):
        if blob is None:
            assert row["phase"] == "postprocess" and error in json.dumps(row), (name, row, error)
        else:
            assert bytes.fromhex(row["blob_hex"]) == blob, name
    by_name = {name: (blob, error) for name, blob, error in expected}
    assert det3d.decode(by_name["empty"][0])[1] == []
    assert len(det3d.decode(by_name["random"][0])[1]) > 100
    assert by_name["nan_logit"][1] == "NaN heatmap logit"
    assert by_name["dims_overflow"][1] == by_name["velocity_nan"][1] == "non-finite detection value"
    edges = det3d.decode(by_name["range_edges"][0])[1]
    assert sorted(round(d[0][0], 4) for d in edges) == [-61.2, 61.2]  # inclusive edges kept, the others dropped
    infinite = det3d.decode(by_name["infinite_logits"][0])[1]
    assert [d[1] for d in infinite] == [1.0]


def test_native_centerpoint_pipeline_refuses_other_parameters(native_build):
    base = _workload().native_output_pipeline()
    for changed, message in (({"codec": "centerpoint-det3d-f64/1"}, "codec/postprocess differ"),
                             ({"max_num": 400}, "parameters differ"), ({"post_max_size": 100}, "parameters differ"),
                             ({"task_classes": [1, 2, 2, 1, 2, 1]}, "task classes differ")):
        proc = subprocess.run([str(native_build / "sdv-bmt-native-fake"), "encode-output",
                               json.dumps({**base, **changed})], input="", capture_output=True, text=True)
        assert proc.returncode != 0 and message in proc.stdout, (changed, proc.stdout)
