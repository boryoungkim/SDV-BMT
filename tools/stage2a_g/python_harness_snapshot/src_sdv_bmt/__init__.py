"""SDV-BMT: Model-level benchmark core.

Stage 1A: deterministic virtual-clock core.  Everything it executes is MOCK
(virtual clock, FakeDUT, NoStress/FakeStress).
Stage 1R-G: realtime harness on the host monotonic clock with a polling DUT
adapter; with the NVIDIA reference-accelerator adapter it executes on a real
GPU (``execution_kind=actual``).  GPU results validate the measurement system;
they are not NPU performance results.
"""

__version__ = "0.2.0"
STAGE = "1R-G"
MOCK_NOTICE = (
    "MOCK: Stage 1A virtual-clock execution with FakeDUT; "
    "not an NPU performance result"
)
REALTIME_MOCK_NOTICE = (
    "MOCK: realtime harness with a fake polling DUT on the host clock; "
    "no accelerator work, not a performance result"
)
