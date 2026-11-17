#!/usr/bin/env bash
# Timed LLM baselines: per model 5 speed runs (S1-S5, 3 warm-up + 5 measured each) + 1 MMLU run, one fresh
# process per run, one device at a time.  Usage:
#   tools/llm/run_baselines.sh gpu|rngd [model ...]
# Host paths: SDV_BMT_MODELS, SDV_BMT_DATASETS, SDV_BMT_CACHE, VLLM_PY, FURIOSA_PY, HF_HOME (RNGD 8B artifact cache).
set -u
cd "$(dirname "$0")/../.."
DEV=$1; shift
MODELS=${*:-"llama-3.2-1b-instruct llama-3.2-3b-instruct llama-3.1-8b-instruct"}
OUT=${OUT:-runs/llm-llama}
REPS=${REPS:-5}
MARK=runs/TIMED_RUN_IN_PROGRESS
LOG=$OUT/baseline-$DEV-$(date +%Y%m%dT%H%M%S).log
export HF_HUB_OFFLINE=1 PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID
printf "start=%s\nsession=LLM\nwhat=%s baselines (%s)\n" "$(date -Iseconds)" "$DEV" "$MODELS" > "$MARK"
trap 'rm -f "$MARK"' EXIT

device_free() {
  if [ "$DEV" = gpu ]; then
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    [ "$used" -lt 1024 ]
  else
    furiosa-smi status | grep -q " 0.00/"
  fi
}

run_one() {  # config
  for i in $(seq 1 30); do device_free && break; echo "$(date -Iseconds) device busy, waiting" >> "$LOG"; sleep 20; done
  device_free || { echo "$(date -Iseconds) ABORT device busy: $1" >> "$LOG"; return 1; }
  echo "$(date -Iseconds) load=$(cut -d' ' -f1-3 /proc/loadavg) run $1" >> "$LOG"
  if [ "$DEV" = gpu ]; then
    out=$(CUDA_VISIBLE_DEVICES=0 VLLM_CACHE_ROOT=$SDV_BMT_CACHE/vllm "$VLLM_PY" -m sdv_bmt.llm run "$1" --out "$OUT" 2>>"$LOG.console" | tail -1)
  else
    out=$("$FURIOSA_PY" -m sdv_bmt.llm run "$1" --out "$OUT" 2>>"$LOG.console" | tail -1)
  fi
  echo "$out" >> "$LOG"
  rd=$(echo "$out" | python3 -c "import json,sys;print(json.loads(sys.stdin.read())['run_dir'])" 2>/dev/null) || return 1
  python3 -m sdv_bmt.llm evaluate "$rd" >> "$LOG" 2>&1
}

for m in $MODELS; do
  base=configs/llm/experiments/$m.$([ "$DEV" = gpu ] && echo gpu0-vllm || echo rngd-furiosa)
  for k in $(seq 1 "$REPS"); do run_one "$base.speed.yaml"; done
  run_one "$base.mmlu.yaml"
done
echo "$(date -Iseconds) done" >> "$LOG"
