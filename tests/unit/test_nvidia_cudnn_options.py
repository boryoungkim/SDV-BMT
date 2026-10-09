"""NVIDIA adapter options for cuDNN-plan execution (Stage 2b-G real-model baseline): strict validation of the
runtime options and the device program, and their place in the measurement declaration and DUT identity."""

from __future__ import annotations

import pytest

from sdv_bmt.contracts.specs import DUTConfig, SpecError, parse_strict
from sdv_bmt.dut.nvidia.adapter import (
    CudnnPlanProgram,
    NvidiaCudaDUT,
    NvidiaOptions,
    cudnn_table_sha256,
)
from sdv_bmt.runner.plan import load_yaml_strict
from tests._helpers import CONFIGS

PINNED = {"math_type": "default", "algo_policy": "pinned", "workspace_limit_bytes": 1 << 28,
          "algorithms": [[1, 0], [7, 0]]}


def options(**kw):
    return parse_strict(NvidiaOptions, kw, "opts")


def test_cudnn_options_parse_and_validate():
    o = options(cudnn=PINNED, frequency_probe_cycles=20000)
    assert o.cudnn.algorithms == ((1, 0), (7, 0))
    with pytest.raises(SpecError, match="pinned"):
        options(cudnn={**PINNED, "algorithms": None})
    with pytest.raises(SpecError, match="only allowed with policy pinned"):
        options(cudnn={**PINNED, "algo_policy": "heuristic"})
    with pytest.raises(SpecError, match="math_type"):
        options(cudnn={**PINNED, "math_type": "tf32"})
    with pytest.raises(SpecError, match="frequency_probe_cycles"):
        options(frequency_probe_cycles=100)
    with pytest.raises(SpecError, match="pairs"):
        options(cudnn={**PINNED, "algorithms": [[1, 0, 3]]})


def test_reference_options_are_unchanged():
    o = options(residency="host_pinned")
    assert o.cudnn is None and o.frequency_probe_cycles is None


def test_cudnn_plan_program_paths_and_sizes():
    base = {"language": "cudnn_plan", "plan": "models/x.plan.json", "plan_sha256": "0" * 64,
            "weights": "models/x.weights.f32", "weights_sha256": "0" * 64, "input_bytes": 16,
            "output_bytes": 64, "task_output_bytes": 32, "timing_offset": 32}
    assert parse_strict(CudnnPlanProgram, base).timing_offset == 32
    for bad in ({"plan": "/abs/plan.json"}, {"weights": "../w.f32"}, {"timing_offset": 16},
                {"timing_offset": 40}, {"task_output_bytes": 65}):
        with pytest.raises(SpecError):
            parse_strict(CudnnPlanProgram, {**base, **bad})


def test_algorithm_table_digest_is_the_compact_json_sha256():
    import hashlib
    assert cudnn_table_sha256(((1, 0), (7, 0))) == hashlib.sha256(b"[[1,0],[7,0]]").hexdigest()


def test_declared_dut_configs_parse_and_declare_the_probe():
    for name in ("nvidia-rtx-pro-6000-gpu0-cudnn-yolox-s", "nvidia-rtx-pro-6000-gpu0-cudnn-yolox-s-nsys",
                 "nvidia-rtx-pro-6000-gpu0-cudnn-heuristic"):
        cfg = parse_strict(DUTConfig, load_yaml_strict(CONFIGS / "duts" / f"{name}.yaml"))
        dut = NvidiaCudaDUT(cfg)
        decl = dut.measurement_declaration()
        assert decl["instrumentation"]["in_kernel_timing"].startswith("probe kernel: one thread, 20000 SM cycles")
        assert "cuDNN plan" in decl["device_timing"]["semantics"]
        if dut._options.cudnn.algo_policy == "pinned":
            assert len(dut._options.cudnn.algorithms) == 83
