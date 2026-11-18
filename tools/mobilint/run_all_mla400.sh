#!/bin/bash
# The measured part of the MLA400 YOLOX port in one command, as run for the reference evidence:
#   1. capture (asap, all 1000 items) of the five models, then the NPU goldens re-extracted from them and compared
#      with the committed ones (configs/references/*mla400*.golden.jsonl);
#   2. capture-identity of the five models (second capture against the golden);
#   3. baselines: 5 rounds at 30 Hz, the model order rotated by one each round (Tiny->X, S->Tiny, ...).
# Every command, and the host and NPU state before each run, goes to ROOT/COMMANDS.log
# (tools/mobilint/run_series_mla400.sh, tools/stage2b_g_real/logcmd.py).
#
#   tools/mobilint/run_all_mla400.sh ROOT [STEP ...]        STEP = capture identity baseline (default: all three)
#
# Run from the repository root with the repo .venv active, after tools/mobilint/build_native.sh, on a host where
# no other process uses the NPU. ROOT must be absolute and new (one plan per experiment directory).
set -u
ROOT=${1:?usage: run_all_mla400.sh ROOT [STEP ...]}
shift
STEPS=${*:-capture identity baseline}
X=configs/experiments
M=(tiny s m l x)
SERIES=tools/mobilint/run_series_mla400.sh
LOGCMD="python3 tools/stage2b_g_real/logcmd.py"

for step in $STEPS; do
  case $step in
    capture)
      bash $SERIES "$ROOT" capture capture $X/yolox-{tiny,s,m,l,x}-mla400-capture.yaml || exit
      mkdir -p "$ROOT/capture/goldens"
      for m in "${M[@]}"; do
        R=$(ls -d "$ROOT"/capture/yolox-$m-mla400-capture/run-*/ | tail -1)
        REF=$(ls configs/references/*.yolox-$m-coco-v1-mla400-q8.npu-capture.golden.jsonl)
        G=$ROOT/capture/goldens/$(basename "$REF")
        $LOGCMD "$ROOT" --label "golden $m" -- python tools/stage2b_g_real/capture_golden.py "$R" "$G" > /dev/null
        new=$(sha256sum < "$G" | cut -d' ' -f1); ref=$(sha256sum < "$REF" | cut -d' ' -f1)
        echo "golden $m: $([ "$new" = "$ref" ] && echo identical || echo "DIFFERS ($new vs $ref)")"
      done
      # the S capture workload declares no quality evaluator (as on the GPU): its COCO AP comes from coco_quality.py
      R=$(ls -d "$ROOT"/capture/yolox-s-mla400-capture/run-*/ | tail -1)
      $LOGCMD "$ROOT" --label "coco_quality s capture" -- python tools/stage2b_g_real/coco_quality.py "$R" \
        datasets/coco-val2017-first1000/index.jsonl "$ROOT/capture/evaluations/s-capture.quality.json" | head -c 120
      echo ;;
    identity)
      bash $SERIES "$ROOT" capture identity $X/yolox-{tiny,s,m,l,x}-mla400-capture-identity.yaml || exit ;;
    baseline)
      for r in 0 1 2 3 4; do
        P=()
        for i in 0 1 2 3 4; do P+=("$X/yolox-${M[$(((i + r) % 5))]}-mla400-baseline.yaml"); done
        echo "=== round $((r + 1)): ${P[*]##*/}"
        bash $SERIES "$ROOT" baseline "round$((r + 1))" "${P[@]}" || exit
      done ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
done
