#!/bin/bash
# MLA400 evidence driver (common guide section 10.2), the Mobilint counterpart of tools/stage2b_g_real/run_series.sh:
# run plans in the given order into ROOT/STAGE with tools/mobilint/sdv_bmt_mla400.py, evaluate each new bundle,
# and record every command plus the host and NPU state before each run in ROOT/COMMANDS.log (logcmd.py format).
#   tools/mobilint/run_series_mla400.sh ROOT STAGE LABEL PLAN [PLAN ...]
# ROOT must be absolute (logcmd.py runs commands from the repository root). MLA400_DEVICE (default /dev/aries2)
# is the NPU device file. CPU load is recorded only; if another process holds the NPU device open, the series
# stops before that run (exit 3) and the holders are in the log.
set -u
ROOT=$1; STAGE=$2; LABEL=$3; shift 3
case "$ROOT" in /*) ;; *) echo "ROOT must be an absolute path" >&2; exit 2;; esac
DEV=${MLA400_DEVICE:-/dev/aries2}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=$REPO/.venv/bin/python
LOGCMD="python3 $REPO/tools/stage2b_g_real/logcmd.py"
OUT=$ROOT/$STAGE
mkdir -p "$OUT/evaluations"

holders() {  # pids (other than ours) with the NPU device open
  local p
  for p in /proc/[0-9]*; do
    [ "${p#/proc/}" = "$$" ] && continue
    ls -l "$p/fd" 2>/dev/null | grep -q " -> $DEV\$" && echo "${p#/proc/} $(cat "$p/comm" 2>/dev/null)"
  done
}
export -f holders
export DEV

k=0
for PLAN in "$@"; do
  k=$((k + 1))
  NAME=$(basename "$PLAN" .yaml)
  $LOGCMD "$ROOT" --label "state before $LABEL #$k $NAME" -- bash -c \
    'uptime; echo "npu $DEV holders:"; holders; echo "top cpu:"; ps -eo pid,psr,pcpu,comm --sort=-pcpu | head -6'
  H=$(holders)
  if [ -n "$H" ]; then
    echo "[$LABEL #$k $NAME] $DEV is held by: $H -- not running; wait and rerun from this plan" >&2
    exit 3
  fi
  before=$(ls -d "$OUT"/*/run-*/ 2>/dev/null | sort)
  $LOGCMD "$ROOT" --label "$LABEL #$k $NAME" -- "$PY" tools/mobilint/sdv_bmt_mla400.py run "$PLAN" --out "$OUT" | tail -1
  after=$(ls -d "$OUT"/*/run-*/ 2>/dev/null | sort)
  new=$(comm -13 <(echo "$before") <(echo "$after"))
  if [ -z "$new" ]; then
    echo "[$LABEL #$k $NAME] no new run bundle (see $ROOT/COMMANDS.log)" >&2
    continue
  fi
  for R in $new; do
    RID=$(basename "$R")
    $LOGCMD "$ROOT" --label "evaluate $LABEL #$k $NAME" -- bash -c \
      "'$PY' tools/mobilint/sdv_bmt_mla400.py evaluate '$R' > '$OUT/evaluations/$RID.json' 2>&1; rc=\$?; tail -c 400 '$OUT/evaluations/$RID.json'; exit \$rc" | tail -1
    echo "[$LABEL #$k $NAME] $R"
  done
done
