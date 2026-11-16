#!/usr/bin/env bash
# R18 L1 offline numeric check on Cityscapes val500 with the tuned native executor, in two parts
# (items 0-249, 250-499), then an exact merge (summed per-class counts). Run as bmt after `sudo -v`, NPU idle.
# Each part: one container with /dev/bos/0, one staged input at a time (12 MiB), golden check before/after.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -B"
EVID=/home/bmt/caontd/eagle_n_integration_evidence
CACHE=/home/bmt/caontd/eagle_n_integration_assets/kernel_cache_r18_tuned
mkdir -p "$CACHE"
PARTS=()
for R in 0:250 250:250; do
  echo "=== L1 part $R ==="; free -g | head -2; df -h / | tail -1
  $PY tools/eagle_n/in_container.py l1 "$R" --kernel-cache "$CACHE" --wall-seconds 5400 | grep -E '^(GOLDEN|ITEM|DONE)|Error|error|REJECTED' | tail -20
  D=$(ls -td "$EVID"/l1_* | head -1)
  echo "part dir: $D"; cat "$D/status.json"
  PARTS+=("$D/l1")
done
echo "=== merge ==="
$PY tools/eagle_n/l1_merge.py "${PARTS[@]}" --out "$EVID/l1_merged_$(date -u +%Y%m%dT%H%M%SZ)"
