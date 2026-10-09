"""Stage 2b-G, case G-B: hold an *idle* CUDA context on one GPU from a separate process (analysis tool;
stdlib ctypes against the installed driver library; no dependency).

    /usr/bin/python3 tools/stage2b_g/idle_context.py GPU-UUID READY_FILE MAX_SECONDS

The process creates a context on the device selected by UUID, allocates nothing, launches nothing, writes
its pid to READY_FILE once the context exists, then waits (sleeping, no device work) until READY_FILE is
deleted or MAX_SECONDS pass, and destroys the context.  Its only purpose is a *realized* condition that
differs from the plan of an otherwise identical run: another process with a compute context on the
device.  It loads nothing on the GPU, so it is not a stress or a fault injection; NVML reports such a
process among the device's compute processes, which is what the competing_device_users evidence records.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time


def main(uuid: str, ready_file: str, max_seconds: float) -> int:
    cuda = ctypes.CDLL("libcuda.so.1")

    def check(result: int, api: str) -> None:
        if result != 0:
            raise RuntimeError(f"{api} failed: CUresult {result}")

    check(cuda.cuInit(0), "cuInit")
    count = ctypes.c_int()
    check(cuda.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
    device = None
    for k in range(count.value):
        d = ctypes.c_int()
        check(cuda.cuDeviceGet(ctypes.byref(d), k), "cuDeviceGet")
        raw = (ctypes.c_ubyte * 16)()
        check(cuda.cuDeviceGetUuid_v2(raw, d), "cuDeviceGetUuid_v2")
        b = bytes(raw)
        text = "GPU-" + "-".join((b[0:4].hex(), b[4:6].hex(), b[6:8].hex(), b[8:10].hex(), b[10:16].hex()))
        if text == uuid:
            device = d
    if device is None:
        raise RuntimeError(f"no CUDA device with UUID {uuid}")
    ctx = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, device), "cuCtxCreate_v2")
    with open(ready_file, "w", encoding="utf-8") as fh:
        fh.write(f"{os.getpid()}\n")
    deadline = time.monotonic() + max_seconds
    try:
        while os.path.exists(ready_file) and time.monotonic() < deadline:
            time.sleep(0.2)
    finally:
        cuda.cuCtxDestroy_v2(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], float(sys.argv[3])))
