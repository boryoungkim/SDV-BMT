#!/bin/bash
# Stage 2b-G real-model baseline: run plans in the given order into ROOT, recording host/GPU state before each run.
#   tools/stage2b_g_real/run_series.sh ROOT LABEL PLAN [PLAN ...]
set -u
ROOT=$1; LABEL=$2; shift 2
k=0
for PLAN in "$@"; do
  k=$((k + 1))
  NAME=$(basename "$PLAN" .yaml)
  python3 tools/stage2b_g_real/logcmd.py "$ROOT" --label "state before $LABEL #$k $NAME" -- bash -c \
    'uptime; nvidia-smi --query-gpu=index,pstate,clocks.sm,temperature.gpu,power.draw,memory.used --format=csv,noheader -i 0; nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader'
  .venv/bin/python tools/stage1r_g/run_and_evaluate.py "$ROOT" "$PLAN" --label "$LABEL #$k $NAME"
done
