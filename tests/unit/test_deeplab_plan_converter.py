"""v0.1.2 DeepLab converter (tools/deeplab_family/build_deeplab_plan.py) and experiment generator.

* the pinned canonical / diagnostic plans and the weight file are exactly what the converter produces from the
  pinned ONNX artifact (needs the local model files; NOT RUN otherwise)
* graphs with an unknown op or attribute are rejected without writing a plan
* the diagnostic plan differs from the canonical one only in the last convolution's destination and the final
  resize + argmax (the conv-condition check of the numeric gate)
* the experiment files are the generator's output; the probe and capture use asap; no baseline before the golden
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys

import pytest

from tests._helpers import CONFIGS, REPO_ROOT

TOOLS = REPO_ROOT / "tools" / "deeplab_family"
MODELS = ("r18", "r50", "r101")


def _module(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(TOOLS))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(TOOLS))
    return mod


def _onnx(model):
    path = REPO_ROOT / "models" / f"deeplabv3plus-{model}" / f"deeplabv3plus_{model}.onnx"
    if not path.is_file():
        pytest.skip("NOT RUN: DeepLab model assets are not present on this host")
    return path


@pytest.mark.parametrize("model", MODELS)
def test_pinned_plans_are_the_converter_output(model, tmp_path):
    onnx = _onnx(model)
    out = tmp_path / f"deeplabv3plus_{model}"
    proc = subprocess.run([sys.executable, str(TOOLS / "build_deeplab_plan.py"), str(onnx), str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    for suffix in (".plan.json", ".diag.plan.json", ".weights.f32"):
        pinned = onnx.parent / f"deeplabv3plus_{model}{suffix}"
        assert pinned.read_bytes() == (tmp_path / f"deeplabv3plus_{model}{suffix}").read_bytes(), suffix


# same-length renames of a node's op_type field (protobuf field 4: tag 0x22, length) keep the framing valid; each
# makes the graph one the converter does not know
@pytest.mark.parametrize("old,new", [(b'"\x06ArgMax', b'"\x06ArgMin'), (b'"\x04Relu', b'"\x04Celu'),
                                     (b'"\x11GlobalAveragePool', b'"\x11GlobalAveragePoox'),
                                     (b"half_pixel", b"half_pixxl")])
def test_unknown_graph_is_rejected(old, new, tmp_path):
    data = _onnx("r18").read_bytes()
    assert data.count(old) >= 1 and len(old) == len(new)
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(data.replace(old, new, 1))
    proc = subprocess.run([sys.executable, str(TOOLS / "build_deeplab_plan.py"), str(bad), str(tmp_path / "bad")],
                          capture_output=True, text=True)
    assert proc.returncode != 0 and "build_deeplab_plan:" in proc.stderr + proc.stdout
    assert not (tmp_path / "bad.plan.json").exists()


@pytest.mark.parametrize("model", MODELS)
def test_diagnostic_plan_has_the_canonical_convolutions(model):
    _onnx(model)
    vn = _module("conv_conditions")
    d = REPO_ROOT / "models" / f"deeplabv3plus-{model}"
    canon = json.loads((d / f"deeplabv3plus_{model}.plan.json").read_text())
    diag = json.loads((d / f"deeplabv3plus_{model}.diag.plan.json").read_text())
    check = vn.conv_conditions(canon, diag)
    assert check["plan_problems"] == [] and check["convolutions"] == {"r18": 37, "r50": 70, "r101": 121}[model]
    moved = dict(diag)
    moved["steps"] = [dict(s) for s in diag["steps"]]
    moved["steps"][0]["dil_h"] = 2
    assert any("dil_h differs" in p for p in vn.conv_conditions(canon, moved)["plan_problems"])


def test_experiment_files_are_the_generator_output():
    gen = _module("make_experiments")
    files = gen.render(None)
    assert len(files) == 6  # probe + capture per model; no baseline before a golden and a rate exist
    for path, text in files.items():
        assert path.read_text(encoding="utf-8") == text, path.name
        assert "mode: asap" in text and "deadline" not in text
    assert not list((CONFIGS / "experiments").glob("deeplabv3plus-*-baseline.yaml")) or all(
        (CONFIGS / "references" / gen.golden_name(m)).is_file() for m, _ in gen.MODELS)


def test_generated_baseline_deadline_follows_the_rate(tmp_path, monkeypatch):
    gen = _module("make_experiments")
    refs = tmp_path / "references"
    refs.mkdir()
    for m, _ in gen.MODELS:
        (refs / gen.golden_name(m)).write_text("{}\n")
    monkeypatch.setattr(gen, "CONFIGS", tmp_path)
    files = gen.render(2.0)
    baselines = {p.name: t for p, t in files.items() if p.name.endswith("-baseline.yaml")}
    assert len(baselines) == 3
    for text in baselines.values():
        assert "rate_hz: 2," in text and "period_ns: 500_000_000" in text
        assert hashlib.sha256(b"{}\n").hexdigest() in text
