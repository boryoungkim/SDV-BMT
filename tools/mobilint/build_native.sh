#!/bin/bash
# Build the Mobilint native runner (native/build/sdv-bmt-native-mobilint) against qbruntime 1.5.0, until the
# shared native/Makefile gets a target for it (request to the owner of the common code).
#
# The host has two qbruntime C++ installs: 1.5.0 in /usr (include, lib/x86_64-linux-gnu) and 1.1.0 in /usr/local,
# which the compiler and the loader search first.  The shared system is not modified: a private directory holds
# symlinks to the 1.5.0 header directory and library, the compiler sees it with -isystem (searched before
# /usr/local/include) and the executable finds the library through its RUNPATH.  ldd and the executable's
# describe output show which library was loaded.
#
#   tools/mobilint/build_native.sh            (from the repository root; python3 must import nothing but stdlib)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
QBRT_ROOT="${QBRT_ROOT:-$HOME/sdv_assets/qbruntime-1.5.0}"
QBRT_INC="${QBRT_INC:-/usr/include/qbruntime}"
QBRT_LIB="${QBRT_LIB:-/usr/lib/x86_64-linux-gnu/libqbruntime.so.1.5.0}"
CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--O2 -std=c++17 -Wall -Wextra -Werror}"
OUT="$REPO/native/build/sdv-bmt-native-mobilint"

mkdir -p "$QBRT_ROOT/include" "$QBRT_ROOT/lib" "$REPO/native/build"
ln -sfn "$QBRT_INC" "$QBRT_ROOT/include/qbruntime"
ln -sfn "$QBRT_LIB" "$QBRT_ROOT/lib/libqbruntime.so"
ln -sfn "$QBRT_LIB" "$QBRT_ROOT/lib/libqbruntime.so.1"

"$CXX" $CXXFLAGS -isystem "$QBRT_ROOT/include" -o "$OUT" "$REPO/native/src/main_mobilint.cpp" \
    -L"$QBRT_ROOT/lib" -Wl,-rpath,"$QBRT_ROOT/lib" -lqbruntime -ldl -lpthread
PYTHONPATH="$REPO/src" python3 -m sdv_bmt.dut.native_source "$REPO/native" > "$OUT.source-tree.sha256"

echo "built $OUT"
echo "source tree $(cat "$OUT.source-tree.sha256")"
ldd "$OUT" | grep -i qbruntime
