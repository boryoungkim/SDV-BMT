"""SDV-BMT: Model-level benchmark core.

Stage 1A: deterministic virtual-clock core.  Everything it executes is MOCK
(virtual clock, FakeDUT, NoStress/FakeStress).
Stage 1R-G: realtime harness on the host monotonic clock.  The measurement
plane is native (C++, ``native/``); Python keeps control, evidence
interpretation and evaluation.  With the NVIDIA reference-accelerator backend it
executes on a real GPU (``execution_kind=actual``).  GPU results validate the
measurement system; they are not NPU performance results.
"""

__version__ = "0.1.0"
STAGE = "1R-G"
MOCK_NOTICE = (
    "MOCK: Stage 1A virtual-clock execution with FakeDUT; "
    "not an NPU performance result"
)
REALTIME_MOCK_NOTICE = (
    "MOCK: native realtime harness with a fake backend on the host clock; "
    "no accelerator work, not a performance result"
)
