"""S1A-4 / E.1: exact component call order observed through spies.

INPUT.open -> DUT.prepare -> PREPARE_INPUTS (payload -> preprocess -> prepare_input
per item) -> STRESS.prepare/activate/verify -> MEASURE (run, postprocess, encode)
-> STRESS.deactivate/restore -> DUT.cleanup -> INPUT.release.
"""

from __future__ import annotations

from sdv_bmt.cli.main import build_components
from tests._helpers import load, read_jsonl, run_plan


def _spy_factory(calls: list[str]):
    def wrap(name, obj, method, show_item=False):
        original = getattr(obj, method)

        def spy(*args, **kwargs):
            label = f"{name}.{method}"
            if show_item:
                arg = args[0] if method == "payload" else args[-1]
                label += f"[{getattr(arg, 'item_index', arg)}]"
            calls.append(label)
            return original(*args, **kwargs)

        setattr(obj, method, spy)

    def factory(rp):
        c = build_components(rp)
        for m in ("open", "release"):
            wrap("input", c.input_provider, m)
        wrap("input", c.input_provider, "payload", show_item=True)
        wrap("workload", c.workload, "preprocess", show_item=True)
        wrap("workload", c.workload, "postprocess", show_item=True)
        wrap("workload", c.workload, "encode")
        for m in ("prepare", "cleanup"):
            wrap("dut", c.dut, m)
        wrap("dut", c.dut, "prepare_input", show_item=True)
        wrap("dut", c.dut, "run")
        for m in ("prepare", "activate", "verify", "observe", "deactivate", "restore"):
            wrap("stress", c.stress, m)
        return c

    return factory


def test_component_call_order(tmp_path):
    calls: list[str] = []
    rp = load("g3-timeout-late-completion")  # 3 items
    run_plan(rp, tmp_path, factory=_spy_factory(calls))
    expected_prefix = [
        "input.open",
        "dut.prepare",
        "input.payload[0]", "workload.preprocess[0]", "dut.prepare_input[0]",
        "input.payload[1]", "workload.preprocess[1]", "dut.prepare_input[1]",
        "input.payload[2]", "workload.preprocess[2]", "dut.prepare_input[2]",
        "stress.prepare", "stress.activate", "stress.verify",
        "stress.observe",
    ]
    assert calls[: len(expected_prefix)] == expected_prefix
    assert calls[-5:] == ["stress.observe", "stress.deactivate", "stress.restore", "dut.cleanup", "input.release"]
    measure = calls[len(expected_prefix):-5]
    assert measure == ["dut.run", "workload.postprocess[0]", "workload.encode",
                       "dut.run", "workload.postprocess[2]", "workload.encode"]
    # no payload access or preparation happens during MEASURE (eager, outside the boundary)
    assert not [c for c in measure if c.startswith(("input.", "dut.prepare"))]


def test_two_episode_recorded_run_issues_at_source_schedule(tmp_path):
    out = run_plan(load("scenario-two-episodes-fake-stress"), tmp_path)
    issued = [e["t_sched"] for e in read_jsonl(out.run_dir / "events.jsonl") if e["kind"] == "issued"]
    ms = 1_000_000
    assert issued == [round(v * ms) for v in (0, 33.412, 66.59, 100.105, 133.301, 166.874,
                                              266.874, 300.074, 333.614, 366.824, 400.354)]
    assert out.request_counts["completed"] == 11
