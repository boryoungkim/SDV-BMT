#!/bin/bash
# MLA400 counterpart of tools/stage2b_g_real/capture_environment.sh: host / NPU / toolchain / code / asset identity
# into ROOT/environment (read-only queries).  Run from the repository root with the repo .venv.
#   tools/mobilint/capture_environment_mla400.sh ROOT [DEVICE_NO]
set -u
ROOT=${1:?usage: capture_environment_mla400.sh ROOT [DEVICE_NO]}
DEVNO=${2:-2}
DEV=/dev/aries$DEVNO
E=$ROOT/environment
mkdir -p "$E"
{
  echo "captured_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname: $(hostname)"
  echo "uptime: $(uptime)"
  echo "logged_in_sessions: $(who | wc -l)"
  echo "os: $(. /etc/os-release; echo "$PRETTY_NAME")"
  echo "kernel: $(uname -r)"
  echo "governor_cpu0: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
  echo "governor_cpu16: $(cat /sys/devices/system/cpu/cpu16/cpufreq/scaling_governor 2>/dev/null)"
  echo "clocksource: $(cat /sys/devices/system/clocksource/clocksource0/current_clocksource)"
  echo "memory: $(free -g | sed -n 2p)"
  echo "nproc: $(nproc)"
} > "$E/host.txt"
lscpu > "$E/lscpu.txt"
numactl --hardware > "$E/numactl.txt" 2>&1 || true

# NPU: device node, PCI path, NUMA node, driver, other holders, vendor tool listing
{
  echo "device: $DEV"
  ls -l /dev/aries* 2>&1
  P=$(udevadm info --query=path --name=$DEV)
  D=$(readlink -f /sys$P/device)
  echo "sysfs_path: $P"
  echo "pci_device: $D"
  echo "numa_node: $(cat $D/numa_node 2>/dev/null)"
  echo "local_cpulist: $(cat $D/local_cpulist 2>/dev/null)"
  echo "pci_vendor_device: $(cat $D/vendor 2>/dev/null) $(cat $D/device 2>/dev/null)"
  echo "link: $(cat $D/current_link_speed 2>/dev/null) x$(cat $D/current_link_width 2>/dev/null) (max $(cat $D/max_link_speed 2>/dev/null) x$(cat $D/max_link_width 2>/dev/null))"
  echo "driver_module: $(cat /sys/module/aries/version 2>/dev/null)"
  modinfo aries 2>/dev/null | grep -E "^(filename|version|srcversion|vermagic):"
  echo "holders of $DEV:"
  for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q " -> $DEV\$" && echo "  ${p#/proc/} $(cat $p/comm)"; done
} > "$E/npu.txt" 2>&1
command -v mobilint-cli > /dev/null && { mobilint-cli devices; mobilint-cli status; } > "$E/mobilint_cli.txt" 2>&1

# runtime / compiler / Python packages
{
  echo "qbruntime C++ installs (the native runner links 1.5.0 through its RUNPATH; /usr/local holds 1.1.0):"
  ls -l /usr/lib/x86_64-linux-gnu/libqbruntime.so* /usr/local/lib/libqbruntime.so* 2>&1
  ls -l "$HOME/sdv_assets/qbruntime-1.5.0/lib/" 2>&1
  sha256sum /usr/lib/x86_64-linux-gnu/libqbruntime.so.1.5.0 2>&1
  echo "zoo-env:"; "$HOME/zoo-env/bin/python" -m pip list 2>/dev/null | grep -i -E "qbruntime|onnx|numpy|opencv"
  echo "qbc14-env:"; "$HOME/qbc14-env/bin/python" -m pip list 2>/dev/null | grep -i -E "qbcompiler|qubee|onnx|torch|numpy"
} > "$E/toolchain.txt" 2>&1
.venv/bin/python -m pip freeze > "$E/venv_pip_freeze.txt" 2>&1

# native runner: binary, linked library, describe
B=native/build/sdv-bmt-native-mobilint
{ ls -l native/build/*mobilint*; sha256sum native/build/*mobilint*; ldd "$B" | grep -E "qbruntime|libstdc"; } > "$E/native_build.txt" 2>&1
"$B" describe "{\"device_number\": $DEVNO}" > "$E/describe.json" 2> "$E/describe.stderr"

# code identity
.venv/bin/python -c "
import json
from sdv_bmt.evidence.provenance import code_identity
from sdv_bmt.dut.native import native_source_tree_sha256
print(json.dumps({'code_identity': code_identity(), 'native_source_tree_sha256': native_source_tree_sha256()}, indent=1, sort_keys=True))" > "$E/code_identity.json" 2>&1
git rev-parse HEAD > "$E/git_head.txt" 2>&1
git status --porcelain >> "$E/git_head.txt" 2>&1

# assets: compiled models, cut ONNX, references, plans, workloads, DUT config
{
  sha256sum "$HOME"/sdv_assets/compile/head9/*.mxq "$HOME"/sdv_assets/compile/head9/*.onnx 2>/dev/null
  sha256sum configs/references/*mla400*.golden.jsonl configs/experiments/*mla400*.yaml configs/workloads/*mla400*.yaml configs/duts/*mla400*.yaml
  sha256sum datasets/coco-val2017-first1000/index.jsonl 2>/dev/null
} > "$E/assets.sha256"
ls -l "$E"
