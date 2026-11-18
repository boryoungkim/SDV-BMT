#!/bin/bash
# Compile the cut YOLOX graphs (nine head convolutions, tools/mobilint/cut_head.py) for the Mobilint ARIES NPU
# with qbcompiler (the compile step of the MLA400 port; run in the Python environment that has qbcompiler 1.4.1).
#
#   tools/mobilint/compile_head9.sh WORKDIR CALIB_ROOT [MODEL ...]
#
# WORKDIR holds yolox_<m>_head9.onnx (from cut_head.py) and receives, per model, yolox_<m>_head9.mxq, the
# configuration qbcompiler used (yolox_<m>_head9.config.yaml) and the compile log (yolox_<m>_head9.compile.log).
# CALIB_ROOT holds the calibration sets of tools/mobilint/make_calib.py: CALIB_ROOT/yolox-416/npy (Tiny) and
# CALIB_ROOT/yolox-640/npy (the others). MODEL defaults to tiny s m l x.
#
# Options, the same for every model: target aries-rb, vendor default 8-bit quantization and inference scheme
# (single core), calibration from the given set (no random calibration), no CPU offload, compiled on the CPU.
# The compiled models are not committed; the workloads pin their SHA-256 (printed at the end).
set -u
WORK=${1:?usage: compile_head9.sh WORKDIR CALIB_ROOT [MODEL ...]}
CALIB=${2:?usage: compile_head9.sh WORKDIR CALIB_ROOT [MODEL ...]}
shift 2
MODELS=${*:-tiny s m l x}
export TF_CPP_MIN_LOG_LEVEL=2
cd "$WORK" || exit 2
command -v qbcompiler > /dev/null || { echo "qbcompiler not found (activate its Python environment)" >&2; exit 2; }
python -m pip show qbcompiler 2>/dev/null | grep -E "^(Name|Version):"
fail=0
for m in $MODELS; do
  if [ "$m" = tiny ]; then c=416; else c=640; fi
  [ -f "yolox_${m}_head9.onnx" ] || { echo "missing yolox_${m}_head9.onnx (run cut_head.py first)" >&2; fail=1; continue; }
  echo "=== $m start $(date +%T)"
  qbcompiler compile --model "yolox_${m}_head9.onnx" --output "yolox_${m}_head9.mxq" --target-device aries-rb \
    --device cpu --no-cpu-offload --no-use-random-calib --calib-data-path "$CALIB/yolox-$c/npy" \
    --config-save-path "yolox_${m}_head9.config.yaml" > "yolox_${m}_head9.compile.log" 2>&1
  rc=$?
  echo "=== $m exit $rc $(date +%T)"
  [ $rc -eq 0 ] || fail=1
done
sha256sum yolox_*_head9.onnx yolox_*_head9.mxq 2>/dev/null
exit $fail
