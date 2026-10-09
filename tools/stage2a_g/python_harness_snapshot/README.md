# Python realtime harness snapshot (Stage 2a-G comparison probe, not production code)

This is a verbatim copy of `src/sdv_bmt/` as it was when the Python realtime harness runs in
`runs/stage2a-g-cpp/python_harness/` were executed (2026-10-08).  Its source-tree SHA-256 equals the
`code_identity.source_tree_sha256` recorded in those bundles' manifests.

The Python realtime runner (`runner/realtime.py` `RealtimeMeasureLoop`), the `PollingDUTAdapter`
contract, `FakePollingDUT` and the ctypes NVIDIA adapter (`dut/nvidia/adapter.py`, `bindings.py`) were
retired from the production path after the approved architecture update "C++ Measurement Plane, Python
Analysis Plane" (see docs/reviews/2026-10-08-stage-2a-g-sdk-observability-investigation-v02.md, section 20; an internal
stage-gate record, not part of the public repository).
It is kept only as provenance of the comparison evidence; it is not importable as `sdv_bmt`.
