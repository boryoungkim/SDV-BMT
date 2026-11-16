#!/usr/bin/env bash
# R18 tuned executor (2026-11-15): build -> re-pin executor hash -> validate -> single-image trial on the NPU ->
# compare labels with the validated output (621089...). Run as bmt after `sudo -v`, NPU idle.
# Build/validate expose no device; only the trial step attaches /dev/bos/0 (via in_container.py guards).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -B"
ASSETS=/home/bmt/caontd/eagle_n_integration_assets
EVID=/home/bmt/caontd/eagle_n_integration_evidence
VALIDATED=621089306df561378d3c9aa56901184eda76c79ac46a156e7deaf6ce07935168

echo "=== 1/5 build (container, no device) ==="
$PY tools/eagle_n/in_container.py build | tail -3
NEW=$(sha256sum "$ASSETS/libeagle_n_ttnn.so" | cut -d' ' -f1)
echo "executor sha256: $NEW"

echo "=== 2/5 re-pin executor hash in R18 configs ==="
$PY tools/eagle_n/configure.py --variant r18 >/dev/null
cp -p "$ASSETS/r18_trial/dut.yaml" "$ASSETS/r18_trial/dut.yaml.bak.$(date +%Y%m%dT%H%M%S)"
sed -i "s/^  executor_sha256: .*/  executor_sha256: $NEW/" "$ASSETS/r18_trial/dut.yaml"
grep -n executor_sha256 configs/duts/eagle-n-r18.yaml "$ASSETS/r18_trial/dut.yaml"

echo "=== 3/5 validate capture plan (container, no device) ==="
$PY tools/eagle_n/in_container.py validate configs/experiments/deeplabv3plus-r18-eagle-n-capture.yaml | grep -E '"status"|issues' || true

echo "=== 4/5 single-image trial on the NPU ==="
$PY tools/eagle_n/in_container.py run "$ASSETS/r18_trial/experiment.yaml" | tail -5 || true
RUN=$(ls -td "$EVID"/run_* | head -1)
BUNDLE=$(ls -d "$RUN"/bundles/*/run-* | head -1)
echo "bundle: $BUNDLE"

echo "=== 5/5 compare with validated labels ==="
$PY tools/eagle_n/compare_trial.py "$BUNDLE" --out "$RUN/verification" | grep -E '"output_sha256"|label_agreement|different_pixels|reference"'
GOT=$(python3 -c "import json;print(json.load(open('$RUN/verification/comparison.json'))['output_sha256'])")
if [ "$GOT" = "$VALIDATED" ]; then echo "RESULT: labels EQUAL validated output ($VALIDATED)"; else echo "RESULT: labels DIFFER ($GOT)"; fi
echo "service time (diagnostic, includes first-use JIT in trial mode):"
grep -rhoE '"bmt\.dut_service[^"]*"[^,}]*' "$BUNDLE" 2>/dev/null | head -5 || true
