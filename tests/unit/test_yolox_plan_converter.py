"""Offline YOLOX ONNX -> cuDNN plan converter (``tools/stage2b_g_real/build_yolox_plan.py``, v0.1.2 YOLOX family):
the plan and weight files pinned by the workload specs are exactly what it produces from the downloaded models,
and graphs it does not know are rejected -- an unsupported op, an input that is not a multiple of 32, an output
whose anchor count does not fit the input -- without writing a plan."""

from __future__ import annotations

import hashlib
import subprocess
import sys

import pytest

from sdv_bmt.contracts.specs import WorkloadSpec, parse_strict
from sdv_bmt.runner.plan import load_yaml_strict
from tests._helpers import CONFIGS, REPO_ROOT

CONVERTER = REPO_ROOT / "tools" / "stage2b_g_real" / "build_yolox_plan.py"
TINY_ONNX = REPO_ROOT / "models" / "yolox-tiny" / "yolox_tiny.onnx"


def convert(onnx, prefix):
    return subprocess.run([sys.executable, "-I", str(CONVERTER), str(onnx), str(prefix)], capture_output=True,
                          text=True)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("model, workload", [("tiny", "yolox-tiny-coco-v1"), ("s", "yolox-s-coco-v1")])
def test_pinned_plan_and_weights_are_the_converter_output(model, workload, tmp_path):
    onnx = REPO_ROOT / "models" / f"yolox-{model}" / f"yolox_{model}.onnx"
    if not onnx.exists():
        pytest.skip("NOT RUN: model artifact not present")
    program = parse_strict(WorkloadSpec, load_yaml_strict(CONFIGS / "workloads" / f"{workload}.yaml")).options[
        "device_program"]
    proc = convert(onnx, tmp_path / f"yolox_{model}")
    assert proc.returncode == 0, proc.stderr
    assert sha256(tmp_path / f"yolox_{model}.plan.json") == program["plan_sha256"]
    assert sha256(tmp_path / f"yolox_{model}.weights.f32") == program["weights_sha256"]


def _dim(value):  # one TensorShapeProto dimension: dim { dim_value: value }, value < 2**14
    return b"\x0a\x03\x08" + bytes([(value & 0x7F) | 0x80, value >> 7])


@pytest.mark.parametrize("old, new, message", [
    (b"\x22\x07MaxPool", b"\x22\x07AvgPool", "unsupported node AvgPool"),        # op_type of a node
    (_dim(416) + _dim(416), _dim(400) + _dim(400), "unexpected input"),            # not a multiple of 32
    (_dim(416) + _dim(416), _dim(384) + _dim(384), "unexpected output"),           # 3549 anchors != 3024 of 384
])
def test_unknown_graphs_are_rejected(old, new, message, tmp_path):
    if not TINY_ONNX.exists():
        pytest.skip("NOT RUN: model artifact not present")
    data = TINY_ONNX.read_bytes()
    assert data.count(old) >= 1 and len(old) == len(new)
    mutated = tmp_path / "mutated.onnx"
    mutated.write_bytes(data.replace(old, new, 1))
    proc = convert(mutated, tmp_path / "out")
    assert proc.returncode != 0 and message in proc.stderr, proc.stderr
    assert not (tmp_path / "out.plan.json").exists() and not (tmp_path / "out.weights.f32").exists()
