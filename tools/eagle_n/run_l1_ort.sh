#!/usr/bin/env bash
# L1 reference part (CPU only, no device, no sudo): ORT FP32 vs NPU labels on val500.
# 1) [project Python 3.11] canonical input digests (sdv_bmt preprocessing);
# 2) [read-only onnxruntime 1.19.2 venv /home/bmt/phamdung/ultralytics/.venv, Python 3.8; nothing written there]
#    ORT labels only, each input checked byte-identical to step 1;
# 3) [project Python 3.11] mIoU with the unmodified sdv_bmt ConfusionCounts.
# Resumable: re-run with the same OUT to continue. Usage: run_l1_ort.sh [OUT_DIR]
set -euo pipefail
cd "$(dirname "$0")/../.."
EVID=/home/bmt/caontd/eagle_n_integration_evidence
ORT_PY=/home/bmt/phamdung/ultralytics/.venv/bin/python
OUT=${1:-$EVID/l1_ort_$(date -u +%Y%m%dT%H%M%SZ)}
NPU="$EVID/l1_20261114T181442_242579Z/l1 $EVID/l1_20261114T182426_347595Z/l1"
mkdir -p "$OUT"
echo "$0 $*  (out=$OUT, started $(date -u +%FT%TZ))" >> "$OUT/command.txt"
sha256sum tools/eagle_n/l1_ort.py "$0" > "$OUT/tool_sha256.txt"
SP=$($ORT_PY -I -c 'import onnxruntime,os;print(os.path.dirname(onnxruntime.__file__))')
sha256sum "$SP"/../onnxruntime-*.dist-info/RECORD "$SP"/../numpy-*.dist-info/RECORD >> "$OUT/tool_sha256.txt"
[ -f "$OUT/input_digests_py311.json" ] || .venv/bin/python -B tools/eagle_n/l1_ort.py --range 0:500 \
    --input-digests "$OUT/input_digests_py311.json"
nice -n 10 $ORT_PY -I -B tools/eagle_n/l1_ort.py --infer --range 0:500 --threads 4 \
    --expect-input-digests "$OUT/input_digests_py311.json" --npu $NPU --out "$OUT" 2>&1 | tee -a "$OUT/run.log"
.venv/bin/python -B tools/eagle_n/l1_ort.py --score --range 0:500 --out "$OUT" 2>&1 | tee -a "$OUT/run.log"
echo "finished $(date -u +%FT%TZ)" >> "$OUT/command.txt"
