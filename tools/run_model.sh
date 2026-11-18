#!/usr/bin/env bash
# Run one SDV-BMT model end to end on this host: validate, timed runs, evaluation, metric table.
#
#   tools/run_model.sh list                      the model names: 12 GPU / RNGD models, then 5 Mobilint MLA400 models
#   tools/run_model.sh <model> [--dry-run]       one model
#   tools/run_model.sh all [--dry-run]           the 12 GPU / RNGD models in order (a failed model does not stop the rest)
#   tools/run_model.sh all-mla400 [--dry-run]    the 5 Mobilint MLA400 models in order, on the NPU host
#
# --dry-run prints the commands and checks the configuration files; it uses no device and writes nothing.
#
# Environment:
#   R     results root (default runs/bench-YYYYMMDD); everything goes below it and nothing is overwritten
#   REPS  runs per model (default 5); the MMLU runs and the CenterPoint val series run once
#   Llama models also need SDV_BMT_MODELS, SDV_BMT_DATASETS, SDV_BMT_CACHE, VLLM_PY, FURIOSA_PY and HF_HOME
#   (the cache that holds the RNGD 8B artifact), see docs/llm_benchmark.md.
#   MLA400 models (yolox-*-mla400) run on the host of the Mobilint MLA400 card: they need the native runner
#   (tools/mobilint/build_native.sh), the device node of the DUT config and the compiled models (.mxq) named by the
#   workloads (tools/mobilint/README.md).  MLA400_DEVICE overrides the device file checked for other users
#   (default: the device of the DUT config), as in tools/mobilint/run_series_mla400.sh.
#
# Output:
#   $R/vision/<experiment>/run-*/ ...          CNN / LiDAR run bundles and their evaluations
#   $R/llm/<experiment>/run-*/ ...             LLM run bundles and their evaluations (tools/llm/run_baselines.sh)
#   $R/logs/<model>.log                        everything the commands printed
#   $R/tables/<model>.metrics.csv              metric table (docs/metric_table.md)
#   $R/tables/<model>.console.txt              the representative metrics as printed
#
# A model whose runs already exist under $R is refused (use a new R), and so is a start while another timed run
# holds runs/TIMED_RUN_IN_PROGRESS.  Progress: elapsed time against the typical duration on the reference host
# (runs), finished parts (CenterPoint val series) and finished runs (Llama).  Before every MLA400 run the script
# stops if another process holds the NPU device open (docs/mobilint/mla400-yolox-port.md, rule "Shared device").
set -u
cd "$(dirname "$0")/.."

R=${R:-runs/bench-$(date +%Y%m%d)}
REPS=${REPS:-5}
PY=${PY:-.venv/bin/python}            # PY, BMT, MARK and LLM_BASELINES can be replaced for tests of this script
BMT=${BMT:-.venv/bin/sdv-bmt}
MARK=${MARK:-runs/TIMED_RUN_IN_PROGRESS}
LLM_BASELINES=${LLM_BASELINES:-tools/llm/run_baselines.sh}
CAT=sdv-bmt-model-metric-set-v1.1          # default catalog (YOLOX, DeepLab)
CAT_DET3D=sdv-bmt-model-metric-set-v1.2    # with the nuScenes 3D detection metrics (CenterPoint)
MODELS=(yolox-tiny yolox-s yolox-m yolox-l yolox-x deeplabv3plus-r18 deeplabv3plus-r50 deeplabv3plus-r101
        centerpoint-pillar02 llama-3.2-1b-instruct llama-3.2-3b-instruct llama-3.1-8b-instruct)
MLA400=(yolox-tiny-mla400 yolox-s-mla400 yolox-m-mla400 yolox-l-mla400 yolox-x-mla400)   # Mobilint NPU host
MLA_RUNNER=${SDV_BMT_NATIVE_DIR:-native/build}/sdv-bmt-native-mobilint
NPU_DEV=
PRE_RUN=      # a check run before every timed run (MLA400: the NPU device is free)
DRY=0
OWN_MARK=0
TICKER=
LOG=/dev/null

say() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
mmss() { printf '%d:%02d' $(( $1 / 60 )) $(( $1 % 60 )); }
rule() { local n; for (( n = 0; n < $1; n++ )); do printf '─'; done; }  # n box-drawing dashes
section() { printf '\n── %s ' "$1"; rule $(( 74 - ${#1} )); printf '\n'; }  # header line of a model

banner() {  # SDV-BMT in 3-row block letters in a box (color on a terminal unless NO_COLOR is set)
  local on="" off="" sub="Model-level Accelerator Benchmark" line i
  local -a big=("█▀▀ █▀▄ █ █    █▀▄ █▀▄▀█ ▀█▀" \
                "▀▀█ █ █ ▀▄▀ ▀▀ █▀▄ █ ▀ █  █ " \
                "▀▀▀ ▀▀   ▀     ▀▀  ▀   ▀  ▀ ")   # 28 columns each
  local -a side=("" "$sub" "version $(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)")
  if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then on=$'\033[1;36m'; off=$'\033[0m'; fi
  line=$(rule $(( 2 + 28 + 4 + ${#sub} + 2 )))
  printf '\n  %s┌%s┐\n' "$on" "$line"
  for i in 0 1 2; do printf '  │  %s    %-*s  │\n' "${big[$i]}" ${#sub} "${side[$i]}"; done
  printf '  └%s┘%s\n\n' "$line" "$off"
}

typical_s() {  # typical duration of one unit on the reference host, seconds (yolox-*-mla400: the YOLOX value, an
  case $1 in   # estimate; the MLA400 runs were not timed on this host)
    yolox-*) echo 150 ;; deeplabv3plus-*) echo 460 ;; centerpoint-mini) echo 45 ;; centerpoint-series) echo 1700 ;;
    llama-gpu) echo 540 ;; llama-rngd) echo 660 ;; *) echo 0 ;;
  esac
}

# x CMD...: run a short command (output to the model log), or print it with --dry-run
x() { if [ $DRY = 1 ]; then echo "    + $*"; else "$@" >> "$LOG" 2>&1; fi; }

bar() {  # pct width: a progress bar, a line for the elapsed part and dots for the rest
  local n=$(( $1 * $2 / 100 )) out="" k
  for (( k = 0; k < $2; k++ )); do [ $k -lt $n ] && out+="━" || out+="·"; done
  printf '%s' "$out"
}

ticker() {  # label typical_s probe: progress bar every 3 s (a TTY) or every minute (a log), until killed
  local label=$1 typical=$2 probe=$3 start=$SECONDS last=-1 el pct line extra
  while :; do
    el=$(( SECONDS - start )); extra=""
    [ -n "$probe" ] && extra="  $($probe)"
    if [ "$typical" -gt 0 ]; then  # elapsed against the typical duration (an estimate, held at 99% until done)
      pct=$(( el * 100 / typical )); [ $pct -gt 99 ] && pct=99
      line="[$(date +%T)] $label  $(bar $pct 30) $(printf '%3d' $pct)%  $(mmss $el) / ~$(mmss "$typical")$extra"
    else
      line="[$(date +%T)] $label  $(mmss $el)$extra"
    fi
    if [ -t 1 ]; then printf '\r\033[K%s' "$line"
    elif [ $(( el / 60 )) -ne $last ]; then echo "$line"; last=$(( el / 60 )); fi
    sleep 3
  done
}

# watch LABEL TYPICAL_S PROBE CMD...: run CMD in the foreground (exactly as by hand: Ctrl-C reaches it) with a
# progress bar from a background ticker; returns the exit code of CMD
watch() {
  local label=$1 typical=$2 probe=$3 rc; shift 3
  if [ $DRY = 1 ]; then echo "    + $*"; return 0; fi
  ticker "$label" "$typical" "$probe" &
  TICKER=$!
  "$@" >> "$LOG" 2>&1; rc=$?
  kill "$TICKER" 2>/dev/null; wait "$TICKER" 2>/dev/null; TICKER=
  [ -t 1 ] && printf '\r\033[K'
  return $rc
}

take_mark() {  # what
  if [ -e "$MARK" ] && [ $OWN_MARK = 0 ]; then
    say "another timed run is in progress ($MARK):"; sed 's/^/    /' "$MARK"; return 3
  fi
  [ $DRY = 1 ] && return 0
  printf 'start=%s\nsession=tools/run_model.sh\nwhat=%s\n' "$(date -Iseconds)" "$1" > "$MARK"; OWN_MARK=1
}
drop_mark() { [ $OWN_MARK = 1 ] && rm -f "$MARK"; OWN_MARK=0; }
trap 'echo; say "interrupted"; [ -n "$TICKER" ] && kill "$TICKER" 2>/dev/null; drop_mark; exit 130' INT TERM

last_status() {  # execution / measurement status of the last run in the model log
  grep -o '"\(execution\|measurement\)_status": "[a-z_]*"' "$LOG" | tail -2 | sed 's/.*: "\(.*\)"/\1/' | paste -sd/
}

evaluate_dirs() {  # label catalog dir...: evaluate every given run (or series) that has no evaluation yet
  local label=$1 cat=$2 n=0 d; shift 2
  if [ $DRY = 1 ]; then echo "    + $BMT evaluate <each new run> --catalog $cat"; return 0; fi
  for d in "$@"; do
    [ -d "$d" ] && [ ! -d "$d/evaluations" ] || continue
    "$BMT" evaluate "$d" --catalog "$cat" >> "$LOG" 2>&1 || { say "$label: evaluation failed: $d (see $LOG)"; return 1; }
    n=$(( n + 1 ))
  done
  say "$label: evaluated $n ($cat)"
}

vision_runs() {  # label experiment typical catalog: validate, REPS runs, evaluation
  local label=$1 exp=$2 typical=$3 cat=$4 cfg=configs/experiments/$2.yaml i
  [ -f "$cfg" ] || { say "$label: missing $cfg"; return 1; }
  if [ $DRY = 0 ] && compgen -G "$R/vision/$exp/run-*" > /dev/null; then
    say "$label: $R/vision/$exp already has runs; use a new R"; return 3
  fi
  say "$label: validate $cfg"
  x "$BMT" validate "$cfg" || { say "$label: validate rejected (see $LOG)"; return 1; }
  for i in $(seq 1 "$REPS"); do
    if [ -n "$PRE_RUN" ] && [ $DRY = 0 ]; then $PRE_RUN || return 3; fi
    watch "$label run $i/$REPS" "$typical" "" "$BMT" run "$cfg" --out "$R/vision" \
      || { say "$label run $i/$REPS failed (see $LOG)"; return 1; }
    [ $DRY = 1 ] || say "$label run $i/$REPS  $(last_status)"
  done
  evaluate_dirs "$label" "$cat" "$R/vision/$exp"/run-*/
}

make_table() {  # name "module [subcommand]" dir...: metric table and the representative metrics
  local name=$1 module=$2 csv=$R/tables/$1.metrics.csv; shift 2
  if [ $DRY = 1 ]; then echo "    + $PY -m $module $* --out $csv"; return 0; fi
  # shellcheck disable=SC2086  # module may carry a subcommand
  if ! "$PY" -m $module "$@" --out "$csv" 2> "$R/tables/$name.console.txt" >> "$LOG"; then
    say "$name: table refused:"; cat "$R/tables/$name.console.txt"; tail -n 4 "$LOG"; return 1
  fi
  echo; cat "$R/tables/$name.console.txt"; echo
}

run_vision_model() {  # name
  local name=$1 exp=$1-baseline rc
  take_mark "$name" || return $?
  vision_runs "$name" "$exp" "$(typical_s "$name")" "$CAT"; rc=$?
  drop_mark
  [ $rc = 0 ] || return $rc
  make_table "$name" sdv_bmt.cli.table "$R/vision/$exp"
}

# ---------------------------------------------------------------- Mobilint MLA400 (on the NPU host)

yaml_value() {  # file key: the value of the first top-level or nested "key:" line, without a trailing comment
  sed -n "s/^ *$2: *//p" "$1" | head -1 | sed 's/ *#.*//'
}

npu_holders() {  # "pid command" of every other process that holds $NPU_DEV open
  local p
  for p in /proc/[0-9]*; do
    [ "${p#/proc/}" = "$$" ] && continue
    ls -l "$p/fd" 2>/dev/null | grep -q " -> $NPU_DEV\$" && echo "${p#/proc/} $(cat "$p/comm" 2>/dev/null)"
  done
}

npu_free() {  # before every MLA400 run: stop while another process holds the NPU device
  local held; held=$(npu_holders | paste -sd, -)
  [ -z "$held" ] && return 0
  say "$NPU_DEV is held by: $held -- not running; wait until it is free, then rerun this model with a new R"
  return 3
}

mla400_ready() {  # name: what an MLA400 run needs, checked before anything runs
  local name=$1 cfg=configs/experiments/$1-baseline.yaml dir dut wl mxq bad=0
  [ -f "$cfg" ] || { say "$name: missing $cfg"; return 1; }
  dir=$(dirname "$cfg")
  dut=$dir/$(yaml_value "$cfg" dut); wl=$dir/$(yaml_value "$cfg" workload)
  NPU_DEV=${MLA400_DEVICE:-$(yaml_value "$dut" device)}
  mxq=$(yaml_value "$wl" mxq); mxq=${mxq/#\~/$HOME}
  say "$name: NPU $NPU_DEV, compiled model $mxq"
  [ $DRY = 1 ] && return 0
  [ -x "$MLA_RUNNER" ] || { say "$name: $MLA_RUNNER missing; build it first: tools/mobilint/build_native.sh"; bad=1; }
  [ -e "$NPU_DEV" ] || { say "$name: $NPU_DEV does not exist (not the MLA400 host, or the driver is not loaded)"; bad=1; }
  [ -f "$mxq" ] || { say "$name: $mxq missing (compiled model, tools/mobilint/README.md steps 1-3)"; bad=1; }
  return $bad
}

run_mla400_model() {  # name
  local rc
  mla400_ready "$1" || return 1
  PRE_RUN=npu_free
  run_vision_model "$1"; rc=$?
  PRE_RUN=
  [ $rc = 0 ] && [ $DRY = 0 ] && echo "Mobilint SDK license: do not publish these results without Mobilint's written" \
    "consent (docs/mobilint/mla400-yolox-port.md)."
  return $rc
}

series_parts() { echo "parts done $(compgen -G "$R/vision/centerpoint-pillar02-val-series/series-*/part-*/manifest.json" | wc -l)/7"; }

run_centerpoint() {
  local name=centerpoint-pillar02 mini=centerpoint-pillar02-mini-baseline val=centerpoint-pillar02-val-series rc
  local cfg=configs/experiments/$val.yaml
  [ -f "$cfg" ] || { say "$name: missing $cfg"; return 1; }
  if [ $DRY = 0 ] && compgen -G "$R/vision/$val/series-*" > /dev/null; then
    say "$name: $R/vision/$val already has a run series; use a new R"; return 3
  fi
  take_mark "$name" || return $?
  vision_runs "$name mini" "$mini" "$(typical_s centerpoint-mini)" "$CAT_DET3D"; rc=$?
  if [ $rc = 0 ]; then
    say "$name: validate $cfg"
    if ! x "$BMT" validate "$cfg"; then say "$name: validate rejected (see $LOG)"; rc=1
    elif ! watch "$name val series" "$(typical_s centerpoint-series)" series_parts "$BMT" run "$cfg" --out "$R/vision"; then
      say "$name val series failed (see $LOG)"; rc=1
    else
      [ $DRY = 1 ] || say "$name val series  status $(grep -ho '"status": "[a-z_]*"' "$R/vision/$val"/series-*/series.json \
        | tail -1 | sed 's/.*: "\(.*\)"/\1/'), $(series_parts)"
      evaluate_dirs "$name val series" "$CAT_DET3D" "$R/vision/$val"/series-*/; rc=$?
    fi
  fi
  drop_mark
  [ $rc = 0 ] || return $rc
  make_table "$name" sdv_bmt.cli.table "$R/vision/$mini" "$R/vision/$val"/series-*/
}

llama_runs() {  # device: finished runs in the newest log of tools/llm/run_baselines.sh for this device
  local f; f=$(ls -t "$R/llm"/baseline-"$1"-*.log 2>/dev/null | head -1)
  [ -n "$f" ] && echo "runs done $(grep -c '"run_status": "completed"' "$f")/$(( REPS + 1 ))" || echo "starting"
}

run_llama() {  # model
  local m=$1 v missing="" c dev f done
  for v in SDV_BMT_MODELS SDV_BMT_DATASETS SDV_BMT_CACHE VLLM_PY FURIOSA_PY HF_HOME; do
    [ -n "${!v:-}" ] || missing="$missing $v"
  done
  if [ -n "$missing" ]; then
    [ $DRY = 1 ] && say "$m: not set:$missing (needed for the real run)" \
      || { say "$m: export$missing first (docs/llm_benchmark.md)"; return 1; }
  fi
  for c in configs/llm/experiments/"$m".{gpu0-vllm,rngd-furiosa}.{speed,mmlu}.yaml; do
    [ -f "$c" ] || { say "$m: missing $c"; return 1; }
  done
  if [ $DRY = 0 ] && compgen -G "$R/llm/$m.*/run-*" > /dev/null; then
    say "$m: $R/llm already has runs of this model; use a new R"; return 3
  fi
  if [ -e "$MARK" ] && [ $DRY = 0 ]; then
    say "another timed run is in progress ($MARK):"; sed 's/^/    /' "$MARK"; return 3
  fi
  for dev in gpu rngd; do  # run_baselines.sh holds runs/TIMED_RUN_IN_PROGRESS itself and waits for an idle device
    say "$m: $dev (speed $REPS runs + MMLU 1 run)"
    watch "$m $dev" "$(typical_s llama-$dev)" "llama_runs $dev" \
      env OUT="$R/llm" REPS="$REPS" "$LLM_BASELINES" "$dev" "$m" \
      || { say "$m $dev: run_baselines.sh failed (see $LOG)"; return 1; }
    [ $DRY = 1 ] && continue
    f=$(ls -t "$R/llm"/baseline-"$dev"-*.log | head -1)
    done=$(grep -c '"run_status": "completed"' "$f")
    say "$m $dev: $done/$(( REPS + 1 )) runs completed (log $f)"
    if [ "$done" -ne $(( REPS + 1 )) ] || grep -q ABORT "$f"; then say "$m $dev: incomplete, see $f"; return 1; fi
  done
  make_table "$m" "sdv_bmt.llm table" "$R/llm/$m".{gpu0-vllm,rngd-furiosa}.{speed,mmlu}
}

ELAPSED=0
run_one() {  # model -> 0 ok, 1 failed, 3 refused; ELAPSED = seconds taken
  local name=$1 start=$SECONDS rc
  LOG=$R/logs/$name.log
  [ $DRY = 1 ] && LOG=/dev/null
  case $name in
    yolox-*-mla400) run_mla400_model "$name" ;;
    yolox-*|deeplabv3plus-*) run_vision_model "$name" ;;
    centerpoint-pillar02) run_centerpoint ;;
    llama-*) run_llama "$name" ;;
  esac
  rc=$?
  ELAPSED=$(( SECONDS - start ))
  [ $DRY = 1 ] || say "$name: $([ $rc = 0 ] && echo done || echo "stopped (code $rc), log $LOG") in $(mmss $ELAPSED)"
  return $rc
}

known() { local m; for m in "${MODELS[@]}" "${MLA400[@]}"; do [ "$m" = "$1" ] && return 0; done; return 1; }

plans_of() {  # model: the plan files its runs use
  case $1 in
    centerpoint-pillar02) echo configs/experiments/centerpoint-pillar02-{mini-baseline,val-series}.yaml ;;
    llama-*) echo configs/llm/experiments/"$1".{gpu0-vllm,rngd-furiosa}.{speed,mmlu}.yaml ;;
    *) echo configs/experiments/"$1"-baseline.yaml ;;
  esac
}

thousands() { local n=$1 out=""; while [ ${#n} -gt 3 ]; do out=",${n: -3}$out"; n=${n:0:${#n}-3}; done; echo "$n$out"; }

dataset_of() {  # plan: its input data, written as the dataset line of the result table (read from the configs)
  local dir f n; dir=$(dirname "$1")
  [ -f "$1" ] || return 0
  case $1 in
    configs/llm/*)  # MMLU: the item list and its item count; speed: the workload file of synthetic token IDs
      f=$(yaml_value "$1" items_file | tr -d '"')
      if [ -n "$f" ]; then
        n=$(grep -o '"item_id"' "$dir/$f" 2>/dev/null | wc -l); f=$(basename "$f")
        echo "${f%%.*} ($(thousands "$n") items)"
      else
        f=$(basename "$(yaml_value "$1" file | tr -d '"')"); echo "${f%%.*} (synthetic token IDs)"
      fi ;;
    *)  # the input set of the plan: its identifier and item count
      f=$dir/$(yaml_value "$1" input); n=$(yaml_value "$f" count)
      echo "$(yaml_value "$f" input_id)${n:+ ($(thousands "$n") items)}" ;;
  esac
}

print_datasets() {  # model...: the input data of the given models, each once, under the models line
  local -a ds=(); local m p label=dataset
  mapfile -t ds < <(for m in "$@"; do for p in $(plans_of "$m"); do dataset_of "$p"; done; done | awk '!seen[$0]++')
  [ ${#ds[@]} -gt 1 ] && label=datasets
  printf '  %-16s %s\n' "$label" "${ds[0]:--}"
  for p in "${ds[@]:1}"; do printf '  %-16s %s\n' "" "$p"; done
}

usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; }

main() {
  local target=${1:-} arg
  shift || true
  for arg in "$@"; do
    case $arg in --dry-run) DRY=1 ;; *) usage; exit 2 ;; esac
  done
  case $target in
    ""|-h|--help) usage; exit 2 ;;
    list) printf '%s\n' "${MODELS[@]}" "${MLA400[@]}"; exit 0 ;;
  esac
  local -a group=()
  case $target in
    all) group=("${MODELS[@]}") ;;
    all-mla400) group=("${MLA400[@]}") ;;
    *) known "$target" || { say "unknown model: $target (tools/run_model.sh list)"; exit 2; } ;;
  esac
  [ $DRY = 1 ] || mkdir -p "$R/vision" "$R/llm" "$R/tables" "$R/logs"
  banner
  printf '  %-16s %s\n' "results" "$(realpath -m "$R")" "runs per model" "$REPS" "models" "$(case $target in
    all) echo "all ${#MODELS[@]} (GPU / RNGD)" ;; all-mla400) echo "all ${#MLA400[@]} (Mobilint MLA400)" ;;
    *) echo "$target" ;; esac)"
  if [ ${#group[@]} = 0 ]; then print_datasets "$target"; else print_datasets "${group[@]}"; fi
  [ $DRY = 1 ] && printf '  %-16s %s\n' "mode" "dry run (prints the commands; nothing is executed)"
  if [ ${#group[@]} = 0 ]; then section "$target"; run_one "$target"; exit $?; fi

  local k=0 m rc failed=0 status
  local -a results=()
  for m in "${group[@]}"; do
    k=$(( k + 1 ))
    section "[$k/${#group[@]}] $m"
    run_one "$m"; rc=$?
    case $rc in 0) status=ok ;; 3) status=refused; failed=1 ;; *) status=failed; failed=1 ;; esac
    results+=("$m $status $ELAPSED")
  done
  [ $DRY = 1 ] && exit 0
  summary "${results[@]}"
  exit $failed
}

summary() {  # "model status seconds"...: one line per model with its metric table
  local item csv rows path
  echo
  printf '┌─ Summary: %d models ' $#; rule 60; printf '\n'
  printf '│ %-24s %-8s %7s %6s  %s\n' Model Status Time Rows "Metric table (CSV)"
  printf '│ %s %s %s %s  %s\n' "$(rule 24)" "$(rule 8)" "$(rule 7)" "$(rule 6)" "$(rule 30)"
  for item in "$@"; do
    set -- $item
    csv=$R/tables/$1.metrics.csv rows=- path=-
    if [ "$2" = ok ] && [ -f "$csv" ]; then rows=$(( $(wc -l < "$csv") - 1 )); path=$(realpath "$csv"); fi
    printf '│ %-24s %-8s %7s %6s  %s\n' "$1" "$2" "$(mmss "$3")" "$rows" "$path"
  done
  printf '└'; rule 81; printf '\n'
  echo "For detailed results (every metric, every run), see the CSV file of each model above."
  echo "Logs of failed or refused models: $(realpath -m "$R/logs")/<model>.log"
}

main "$@"
