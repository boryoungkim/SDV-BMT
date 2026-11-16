#!/usr/bin/env bash
# PROVISIONAL L5 (user decision 2026-11-15, D2; deviation from guide Step 8): 5 baseline runs of R18 at 0.8 Hz,
# deadline 1.25 s, against the Eagle-N golden (af30e654…). R18 only (R50/R101 not ported yet, so no model rotation).
# Host state logged before each run into COMMANDS.log (common guide §10, as run_series.sh for the GPU); CPU load is
# recorded only; if another process holds /dev/bos/0 the launcher precheck refuses the run.
# Run as bmt after `sudo -v`, NPU idle. ~5 x 18 min ≈ 90 min.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -B"
EVID=/home/bmt/caontd/eagle_n_integration_evidence
PLAN=configs/experiments/deeplabv3plus-r18-eagle-n-baseline-provisional-0p8hz.yaml
N=${N:-5}
sudo -n -v
( while sleep 60; do sudo -n -v || exit; done ) & KEEP=$!
trap 'kill $KEEP 2>/dev/null' EXIT
test "$(sha256sum /home/bmt/caontd/eagle_n_integration_assets/libeagle_n_ttnn.so | cut -d' ' -f1)" = 58d9b284da6e58f7fde50f3cd71661c57f9d439de8ce3943c82ed689b74c93f5
test "$(sha256sum /home/bmt/caontd/eagle_n_integration_assets/r18/npu-capture.golden.jsonl | cut -d' ' -f1)" = af30e654681629e2c94044abc094be3059390a090131abc01486530a70012a9a
$PY tools/eagle_n/in_container.py validate "$PLAN" | tee /dev/stderr | grep -q '"status": "valid"' || { echo "validate not valid; stopping"; exit 1; }
for i in $(seq 1 "$N"); do
  {
    echo "=== L5 provisional run $i/$N  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uptime: $(uptime)"
    echo "busy_cpus(top 5): $(ps -eo pcpu,pid,user,comm --sort=-pcpu | sed -n 2,6p | tr '\n' ';')"
    echo "npu_users: $(sudo -n fuser /dev/bos/0 2>&1 || true)"
    echo "memory: $(free -m | sed -n 2p)"
  } | tee -a "$EVID/COMMANDS.log"
  BEFORE=$(ls -td "$EVID"/run_* | head -1)
  $PY tools/eagle_n/in_container.py run "$PLAN" \
      --kernel-cache /home/bmt/caontd/eagle_n_integration_assets/kernel_cache_r18_tuned --wall-seconds 5400 | tail -22 || true
  RUN=$(ls -td "$EVID"/run_* | head -1)
  if [ "$RUN" = "$BEFORE" ] || [ ! -f "$RUN/status.json" ]; then echo "run $i: launcher refused or did not finish; stopping"; exit 1; fi
  cat "$RUN/status.json"
  BUNDLE=$(ls -d "$RUN"/bundles/*/run-* | head -1)
  $PY tools/eagle_n/in_container.py evaluate "$BUNDLE" | grep -E 'miou"|identity|"status"' | head -12 || true
  echo "$i $RUN" >> "$EVID/l5_provisional_runs.txt"
done
