#!/usr/bin/env python3
"""``sdv-bmt`` with the Mobilint adapter registered (porting tool, until the shared CLI registers it).

``src/sdv_bmt/cli/main.py`` (DUT_ADAPTERS) is shared code owned by the common-code maintainer; the request to
add ``MobilintAriesDUT`` there is pending.  Until then this wrapper adds the adapter to the table at runtime and
runs the unchanged CLI, so ``run``/``evaluate`` behave exactly as ``sdv-bmt`` does:

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
