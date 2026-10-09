#!/bin/sh
# Build the Stage 2a-G C++ probe against the installed CUDA 13.0 toolkit headers and the driver libraries.
# Output: tools/stage2a_g/native/build/rt_probe (not a production artifact).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
CUDA_HOME=${SDV_BMT_CUDA_HOME:-/usr/local/cuda-13.0}
mkdir -p "$HERE/build"
g++ -O2 -std=c++17 -Wall -Wextra -o "$HERE/build/rt_probe" "$HERE/rt_probe.cpp" \
    -I"$CUDA_HOME/include" -L"$CUDA_HOME/lib64" -Wl,-rpath,"$CUDA_HOME/lib64" -lnvrtc -lcuda -ldl
echo "$HERE/build/rt_probe"
