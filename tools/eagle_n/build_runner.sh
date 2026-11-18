#!/usr/bin/env bash
# Rebuild ONLY the SDV-BMT native runner for Eagle-N (native/src/main_eagle_n.cpp + eagle_n/backend.hpp) and its
# source-tree stamp, inside the BOS RC6 build container (same clang-17 as build_native.sh). No device is opened.
# The executor libeagle_n_ttnn.so is NOT rebuilt, so its sha256 (and the L1-bound qualification receipt) is unchanged.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
CXX=${CXX:-$(command -v clang++-17 || command -v clang++ || command -v g++)}
PYTHON=${PYTHON:-$ROOT/.venv/bin/python}
mkdir -p "$ROOT/native/build"
"$CXX" --version | head -1
"$CXX" -std=c++17 -O2 -Wall -Wextra -Werror -o "$ROOT/native/build/sdv-bmt-native-eagle-n" \
  "$ROOT/native/src/main_eagle_n.cpp" -ldl
PYTHONPATH="$ROOT/src" "$PYTHON" -B -m sdv_bmt.dut.native_source "$ROOT/native" \
  > "$ROOT/native/build/sdv-bmt-native-eagle-n.source-tree.sha256"
sha256sum "$ROOT/native/build/sdv-bmt-native-eagle-n" "$ROOT/native/build/sdv-bmt-native-eagle-n.source-tree.sha256"
