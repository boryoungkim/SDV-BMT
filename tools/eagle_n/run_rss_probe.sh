#!/usr/bin/env bash
# Measure host memory (VmRSS/VmHWM) of the native R18 executor, to size staging_budget_bytes and
# prepare_host_budget_bytes for a 500-item sdv-bmt run. Uses the L1 harness on 5 items, then stages the golden input
# into 32 extra slots and records the RSS growth per staged item. NPU device: run as bmt after `sudo -v`, NPU idle.
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python -B tools/eagle_n/in_container.py l1 0:5 \
    --kernel-cache /home/bmt/caontd/eagle_n_integration_assets/kernel_cache_r18_tuned \
    --rss-probe-slots 32 --wall-seconds 1800
OUT=$(ls -td /home/bmt/caontd/eagle_n_integration_evidence/l1_* | grep -v merged | head -1)
echo "evidence: $OUT"
python3 -c "import json;r=json.load(open('$OUT/l1/report.json'));print(json.dumps({'status':r['status'],'memory':r['memory'],'golden':[g['equals_validated'] for g in r['golden_checks']]},indent=1))"
