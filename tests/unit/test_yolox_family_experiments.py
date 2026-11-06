"""v0.1.2 YOLOX family experiments: every model with a golden has a baseline, the baselines share one replay rate
with the deadline at its frame period, and the committed files are exactly the output of
``tools/yolox_family/make_experiments.py``, the one place where that rate is set."""

from __future__ import annotations

import math
import sys
from fractions import Fraction

from sdv_bmt.contracts.specs import ExperimentPlan, parse_strict
from sdv_bmt.contracts.status import ReplayMode
from sdv_bmt.runner.plan import load_yaml_strict
from tests._helpers import CONFIGS, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tools" / "yolox_family"))
import make_experiments as M  # noqa: E402


def plan(name):
    return parse_strict(ExperimentPlan, load_yaml_strict(CONFIGS / "experiments" / f"{name}.yaml"))


def test_baselines_share_one_replay_rate_and_a_frame_period_deadline():
    with_golden = [m for m, _, _, _, golden in M.MODELS if (CONFIGS / "references" / golden).is_file()]
    assert "s" in with_golden  # the v0.1.0 golden of YOLOX-S is tracked
    plans = [plan(f"yolox-{m}-baseline") for m in with_golden]
    rates = {p.replay.rate_hz for p in plans}
    assert len(rates) == 1 and all(p.replay.mode is ReplayMode.FIXED_RATE for p in plans)
    period = math.floor(Fraction(10 ** 9) / Fraction(repr(rates.pop())))
    for p in plans:
        assert p.evaluation.deadline.policy.value == "frame_period" and p.evaluation.deadline.period_ns == period
        assert p.evaluation.output_reference.kind.value == "external_golden"


def test_committed_experiment_files_are_the_generator_output():
    files = M.render(plan("yolox-s-baseline").replay.rate_hz)
    for path, text in files.items():
        assert path.is_file() and path.read_text(encoding="utf-8") == text, f"{path} differs from the tool output"
    probes = {p for p in files if p.name.endswith("-algo-probe.yaml")}
    assert len(probes) == len(M.MODELS)
    for path in probes:  # preparation runs: closed loop, heuristic algorithm policy
        p = plan(path.stem)
        assert p.replay.mode is ReplayMode.ASAP and p.dut.endswith("cudnn-heuristic.yaml")
