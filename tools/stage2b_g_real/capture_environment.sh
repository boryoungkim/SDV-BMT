#!/bin/bash
# Stage 2b-G real-model baseline: capture host/GPU/toolchain/code/asset identity into ROOT/environment (read-only queries).
# Optional environment: CUDA_HOME (default /usr/local/cuda-13.0), CUDNN_ROOT (default: native/Makefile's default),
# COCO_ANNOTATIONS (instances_val2017.json to hash; skipped when unset).
set -u
ROOT=${1:?usage: capture_environment.sh ROOT}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
CUDNN_ROOT=${CUDNN_ROOT:-/usr/local/lib/python3.10/dist-packages/nvidia/cudnn}
E=$ROOT/environment
mkdir -p $E
{
  echo "captured_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uptime: $(uptime)"
  echo "logged_in_sessions: $(who | wc -l)"
  echo "kernel: $(uname -r)"
  echo "governor_cpu0: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
  echo "clocksource: $(cat /sys/devices/system/clocksource/clocksource0/current_clocksource)"
  echo "disk_root: $(df -h / | tail -1)"
  echo "memory: $(free -g | sed -n 2p)"
  echo "nproc: $(nproc)"
} > $E/host.txt
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,persistence_mode,compute_mode --format=csv > $E/nvidia_smi.csv
nvidia-smi -q -i 0 > $E/nvidia_smi_q_gpu0.txt
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid --format=csv > $E/compute_apps_before.csv
sha256sum native/build/sdv-bmt-native-* > $E/native_build.sha256
.venv/bin/python -c "
import json
from sdv_bmt.evidence.provenance import code_identity
from sdv_bmt.dut.native import native_source_tree_sha256
print(json.dumps({'code_identity': code_identity(), 'native_source_tree_sha256': native_source_tree_sha256()}, indent=1, sort_keys=True))
" > $E/code_identity.json
sha256sum $CUDNN_ROOT/lib/libcudnn*.so.9 > $E/cudnn_libs.sha256
pip3 show nvidia-cudnn-cu12 2>/dev/null | head -3 > $E/cudnn_package.txt
sha256sum models/yolox-s/* datasets/coco-val2017-first1000/index.jsonl datasets/coco-val2017-first1000/dataset_manifest.json ${COCO_ANNOTATIONS:-} > $E/assets.sha256
sha256sum tools/stage2b_g_real/*.py tools/stage2b_g_real/*.cpp tools/stage2b_g_real/*.sh tools/stage1r_g/run_and_evaluate.py > $E/tools.sha256
python3 -c "import PIL, onnxruntime, numpy; print('Pillow', PIL.__version__); print('onnxruntime', onnxruntime.__version__); print('numpy', numpy.__version__)" > $E/offline_tool_packages.txt
g++ --version | head -1 > $E/toolchain.txt; $CUDA_HOME/bin/nvcc --version | tail -2 >> $E/toolchain.txt
echo done
