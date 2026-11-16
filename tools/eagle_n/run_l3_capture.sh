#!/usr/bin/env bash
# L3 (guide Step 6): R18 capture plan (asap, 500 items, 32 warm-up requests) through `sdv-bmt run` on the NPU,
# then `sdv-bmt evaluate`; the mIoU must equal the L1 NPU mIoU 0.7675648837497185.
# Run as bmt after `sudo -v`, NPU idle. Needs ~14 GB free disk (SDV-BMT staging file, removed after staging).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -B"
EVID=/home/bmt/caontd/eagle_n_integration_evidence
PLAN=${PLAN:-configs/experiments/deeplabv3plus-r18-eagle-n-capture.yaml}
echo "plan: $PLAN"
# keep the sudo timestamp alive for the whole run (default 15 min < ~18 min run); the launcher needs sudo -n at the end
# to unmount its private hugepage mount. Stopped on exit.
sudo -n -v
( while sleep 60; do sudo -n -v || exit; done ) & KEEP=$!
trap 'kill $KEEP 2>/dev/null' EXIT

echo "=== 0/3 rebuild ONLY the native runner (container, no device; executor .so untouched) ==="
$PY tools/eagle_n/in_container.py build-runner | tail -4
test "$(sha256sum /home/bmt/caontd/eagle_n_integration_assets/libeagle_n_ttnn.so | cut -d' ' -f1)" = 58d9b284da6e58f7fde50f3cd71661c57f9d439de8ce3943c82ed689b74c93f5

echo "=== 1/3 validate (container, no device) ==="
$PY tools/eagle_n/in_container.py validate "$PLAN" | grep -E '"status"|issues' || true

echo "=== 2/3 sdv-bmt run, 500 items on the NPU (~20-30 min) ==="
$PY tools/eagle_n/in_container.py run "$PLAN" \
    --kernel-cache /home/bmt/caontd/eagle_n_integration_assets/kernel_cache_r18_tuned --wall-seconds 5400 | tail -15 || true
RUN=$(ls -td "$EVID"/run_* | head -1)
cat "$RUN/status.json"
BUNDLE=$(ls -d "$RUN"/bundles/*/run-* | head -1)
echo "bundle: $BUNDLE"

echo "=== 3/3 sdv-bmt evaluate (container, no device) ==="
$PY tools/eagle_n/in_container.py evaluate "$BUNDLE" | tail -60
echo "L1 NPU mIoU for comparison: 0.7675648837497185"
