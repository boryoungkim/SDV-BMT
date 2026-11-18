#!/usr/bin/env bash
# Eagle-N ladder L2-L5 through SDV-BMT for one model. See tools/eagle_n/README.md.
#
#   tools/eagle_n/run_eagle_n.sh MODEL LEVEL
#     MODEL  r18 | r50 | r101
#     LEVEL  check  host checks only (no Docker, no device)
#            l2     check + in-container describe and validate of the capture and baseline plans (no device)
#            l3     check + validate + capture (asap) on the NPU + evaluate; mIoU must equal the receipt (L1)
#            l4     l3 + identity of the new capture against the NPU golden in configs/references (must be 1.0)
#            l5     check + validate + L5_RUNS baselines (GPU family rate), each: no error or timeout, identity 1.0
#                   (drops and deadline misses are reported, not checked)
#
# Environment (defaults: the fixed Eagle-N root /opt/sdv_assets/eagle_n, README "Host setup"):
#   EAGLE_N_ASSETS    assets root (executor .so, <model>/ plan+weights+receipt, kernel caches)
#   EAGLE_N_EVIDENCE  evidence root (one directory per launcher call; COMMANDS.log)
#   EAGLE_N_SDK_ROOT  BOS SDK root, passed to tools/eagle_n/in_container.py (required for l2-l5)
#   L5_RUNS=5 L5_START=1   baseline runs for l5; restart after a reviewed stop with L5_START=k
#   WALL_SECONDS=6000      launcher wall time; it only reports, it never kills a container
#
# Rules kept from the port (README): the device levels need `sudo -v` first and an idle NPU; nothing is retried;
# the first failure stops the script; on a hang never kill the container or the runner: report it.
# The executor, plans and weights are never rebuilt or re-exported here.
set -uo pipefail
cd "$(dirname "$0")/../.."
MODEL=${1:-}; LEVEL=${2:-}
PY=${PY:-.venv/bin/python -B}
A=${EAGLE_N_ASSETS:-/opt/sdv_assets/eagle_n/eagle_n_integration_assets}
EVID=${EAGLE_N_EVIDENCE:-/opt/sdv_assets/eagle_n/eagle_n_integration_evidence}
WALL=${WALL_SECONDS:-6000}
E=configs/experiments
case "$MODEL" in
  r18)  ID=deeplabv3plus-r18-eagle-n;       KC=$A/kernel_cache_r18_tuned ;;
  r50)  ID=deeplabv3plus-r50-eagle-n-bf16;  KC=$A/kernel_cache_r50_bf16 ;;
  r101) ID=deeplabv3plus-r101-eagle-n-bf16; KC=$A/kernel_cache_r101_bf16 ;;
  *) sed -n '2,22p' "$0"; exit 2 ;;
esac
case "$LEVEL" in check|l2|l3|l4|l5) ;; *) sed -n '2,22p' "$0"; exit 2 ;; esac
CAPTURE=$E/$ID-capture.yaml
BASELINE=$E/$ID-baseline.yaml
CHK="$PY tools/eagle_n/eagle_n_checks.py"
GOLD=$($PY - "$BASELINE" <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, 'src')
from sdv_bmt.runner.plan import load_plan
ref = load_plan(sys.argv[1]).plan.evaluation.output_reference
print(Path(sys.argv[1]).parent / ref.path)
EOF
)
echo "=== Eagle-N $MODEL $LEVEL  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "capture plan:  $CAPTURE"; echo "baseline plan: $BASELINE"; echo "NPU golden:    $GOLD"
$CHK pins "$CAPTURE" "$BASELINE" || { echo "STOP: pins not satisfied (nothing started)"; exit 1; }
[ "$LEVEL" = check ] && exit 0

sudo -n -v || { echo "STOP: run 'sudo -v' first"; exit 1; }
( while sleep 60; do sudo -n -v || exit; done ) & KEEP=$!
trap 'kill $KEEP 2>/dev/null' EXIT
validate() {  # in-container validate, no device
  $PY tools/eagle_n/in_container.py validate "$1" --evidence "$EVID" | grep -q '"status": "valid"' \
    || { echo "STOP: $1 not valid (nothing run on the NPU)"; exit 1; }
  echo "validate (container, no device): valid  $1"
}
host_state() {
  { echo "=== Eagle-N $MODEL $1  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uptime: $(uptime)"
    echo "busy_cpus(top 5): $(ps -eo pcpu,pid,user,comm --sort=-pcpu | sed -n 2,6p | tr '\n' ';')"
    echo "npu_users: $(sudo -n fuser /dev/bos/0 2>&1 || true)"
    echo "memory: $(free -m | sed -n 2p)"
    echo "disk: $(df -h "$EVID" | tail -1)"; } | tee -a "$EVID/COMMANDS.log"
}
run_plan() {  # $1 plan, $2 label; sets RUN and BUNDLE
  host_state "$2"
  local before; before=$(ls -td "$EVID"/run_* 2>/dev/null | head -1)
  $PY tools/eagle_n/in_container.py run "$1" --evidence "$EVID" --kernel-cache "$KC" --wall-seconds "$WALL" | tail -12
  RUN=$(ls -td "$EVID"/run_* 2>/dev/null | head -1)
  { [ -n "$RUN" ] && [ "$RUN" != "$before" ] && [ -f "$RUN/status.json" ]; } \
    || { echo "STOP: launcher refused or did not finish (no new run dir)"; exit 1; }
  echo "run dir: $RUN"; cat "$RUN/status.json"
  grep -q '"exit_code": 0' "$RUN/status.json" \
    || { echo "STOP: run not clean -> no evaluate, no further run; report"; exit 1; }
  BUNDLE=$(ls -d "$RUN"/bundles/*/run-* | head -1); echo "bundle: $BUNDLE"
  $PY tools/eagle_n/in_container.py evaluate "$BUNDLE" --evidence "$EVID" > /dev/null
  local ev; ev=$(ls -td "$EVID"/evaluate_* | head -1)
  grep -q '"exit_code": 0' "$ev/status.json" || { echo "STOP: evaluate $ev failed"; exit 1; }
}

if [ "$LEVEL" = l2 ]; then
  DUT=$(dirname "$CAPTURE")/$(sed -n 's/^dut: *//p' "$CAPTURE")
  $PY tools/eagle_n/in_container.py describe "$DUT" --evidence "$EVID" | tail -5
  validate "$CAPTURE"; validate "$BASELINE"
  echo "=== L2 $MODEL done"; exit 0
fi

if [ "$LEVEL" = l3 ] || [ "$LEVEL" = l4 ]; then
  validate "$CAPTURE"
  run_plan "$CAPTURE" "$LEVEL capture"
  $CHK result "$BUNDLE" "$LEVEL" "$CAPTURE" | tee -a "$EVID/COMMANDS.log"
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "STOP: $LEVEL checks not met; report"; exit 1; }
  $PY tools/stage2b_g_real/capture_golden.py "$BUNDLE" "$RUN/capture.golden.jsonl" \
    || { echo "STOP: golden extraction failed"; exit 1; }
  if [ "$LEVEL" = l3 ]; then
    echo "golden of this capture: $RUN/capture.golden.jsonl ($(sha256sum "$RUN/capture.golden.jsonl" | cut -c1-12)…)"
    if [ -f "$GOLD" ]; then
      cmp -s "$GOLD" "$RUN/capture.golden.jsonl" || { echo "STOP: golden DIFFERS from $GOLD; report"; exit 1; }
      echo "same as $GOLD"
    fi
    echo "=== L3 $MODEL done"; exit 0
  fi
  $CHK identity "$GOLD" "$RUN/capture.golden.jsonl" "$RUN/identity.json" | tail -2
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "STOP: identity not 1.0; report"; exit 1; }
  echo "=== L4 $MODEL done"; exit 0
fi

# l5
N=${L5_RUNS:-5}; START=${L5_START:-1}
validate "$BASELINE"
LIST="$EVID/l5_${MODEL}_runs.txt"
for i in $(seq "$START" "$N"); do
  run_plan "$BASELINE" "l5 run $i/$N"
  $CHK result "$BUNDLE" l5 "$CAPTURE" | tee -a "$EVID/COMMANDS.log"
  [ "${PIPESTATUS[0]}" = 0 ] \
    || { echo "STOP run $i: checks not met; report (continue with L5_START=$((i + 1)) only after review)"; exit 1; }
  echo "$i $RUN $BUNDLE" >> "$LIST"
done
echo "=== L5 $MODEL done: runs $START..$N; list $LIST"
