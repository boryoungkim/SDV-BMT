#!/usr/bin/env python3
"""``sdv-bmt`` from the repository sources (compatibility entry point of the MLA400 scripts).

The shared CLI registers the ``mobilint_aries`` adapter (``src/sdv_bmt/cli/main.py``, ``DUT_ADAPTERS``), so this
is the plain CLI; the ``setdefault`` below is a no-op kept for older checkouts.  The official MLA400 runs were
made through this entry point, and ``run``/``evaluate`` behave exactly as ``sdv-bmt`` does:

    python3 tools/mobilint/sdv_bmt_mla400.py run configs/experiments/yolox-s-mla400-capture.yaml --out runs/...
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sdv_bmt.cli import main as cli  # noqa: E402
from sdv_bmt.dut.mobilint import MobilintAriesDUT  # noqa: E402

cli.DUT_ADAPTERS.setdefault(MobilintAriesDUT.ADAPTER, MobilintAriesDUT)

if __name__ == "__main__":
    raise SystemExit(cli.main())
