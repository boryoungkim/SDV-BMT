"""The measure watchdog of the native adapter: the time limit for the MEASURE/DRAIN reply of the native runner.

An asap schedule has no offsets.  The limit used to count it as one request, so a long asap window was killed
after a fixed time whatever its item count (about 154 s with a 1 s request timeout and 32 precondition requests).
"""

from __future__ import annotations

import pytest

from sdv_bmt.contracts.records import RealtimeRunContext, WindowHandover
from sdv_bmt.contracts.specs import DUTConfig
from sdv_bmt.dut.native import NativeRealtimeAdapter, measure_limit_s

S = 1_000_000_000
RUN = {"max_inflight": 1, "queue_capacity": 0, "on_full": "drop_new", "request_timeout_ns": 1 * S, "drain_ns": 1 * S}
ASAP = {"mode": "asap", "lead_ns": 2_000_000, "offsets_ns": None}


def test_asap_limit_grows_with_the_planned_items():
    assert measure_limit_s(ASAP, RUN, 32, 500) == pytest.approx(0.002 + 500 + 1 + 32 + 120)
    assert measure_limit_s(ASAP, RUN, 32, 1000) - measure_limit_s(ASAP, RUN, 32, 500) == pytest.approx(500.0)


def test_asap_limit_with_one_item_is_the_former_value():
    assert measure_limit_s(ASAP, RUN, 32, 1) == pytest.approx(154.002)
    assert measure_limit_s(ASAP, RUN, 32, 0) == measure_limit_s(ASAP, RUN, 32, 1)


def test_fixed_rate_limit_is_the_schedule_span_without_per_request_timeouts():
    offsets = [i * S // 2 for i in range(500)]  # 2 Hz
    schedule = {"mode": "fixed_rate", "lead_ns": 2_000_000, "offsets_ns": offsets}
    expected = (offsets[-1] + 2_000_000) / 1e9 + 1 + 32 + 120
    assert measure_limit_s(schedule, RUN, 32, 500) == pytest.approx(expected)
    assert measure_limit_s(schedule, RUN, 32, 7) == measure_limit_s(schedule, RUN, 32, 500)


class _Adapter(NativeRealtimeAdapter):
    ADAPTER = "measure_limit_test"


class _Process:
    alive = False

    def __init__(self) -> None:
        self.timeouts: dict[str, float] = {}

    def command(self, payload, timeout_s):
        self.timeouts[payload["command"]] = timeout_s
        if payload["command"] == "stage":
            return {"ok": True, "items": [{"item_index": row["item_index"], "ok": True} for row in payload["items"]]}
        return {"ok": True, "next_seq": 0, "last_t_run_ns": 0, "output_offset": 0}


def test_run_window_uses_the_number_of_staged_items(tmp_path):
    config = DUTConfig(schema_version=1, dut_config_id="t", adapter=_Adapter.ADAPTER, device="host-only",
                       max_inflight=1, runtime_options={})
    adapter = _Adapter(config)
    adapter.bind_run(RealtimeRunContext(anchor_monotonic_ns=0, cpu_affinity=(), schedule=ASAP, run=RUN,
                                        precondition_requests=32, events_path="", outputs_path="",
                                        output_pipeline={}))
    process = _Process()
    staging = tmp_path / "staging.bin"
    adapter._process, adapter._staging, adapter._staging_path = process, staging.open("wb"), str(staging)
    adapter._state = "prepared"
    adapter.commit_inputs([(i, "prepare_input", None) for i in range(500)])
    adapter.run_window(WindowHandover(next_seq=0, last_t_run_ns=0, output_offset=0))
    assert process.timeouts["measure"] == pytest.approx(measure_limit_s(ASAP, RUN, 32, 500))
    assert process.timeouts["measure"] > 600
