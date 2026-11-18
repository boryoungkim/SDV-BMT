#!/bin/bash
# Build the Mobilint native runner (native/build/sdv-bmt-native-mobilint) against qbruntime C++ 1.5.0, with the
# shared Makefile's target (make -C native mobilint) and its build stamp.
#
# The host has two qbruntime C++ installs: 1.5.0 in /usr (include, lib/x86_64-linux-gnu) and 1.1.0 in /usr/local,
# which the compiler and the loader search first.  The shared system is not modified: the Makefile target keeps a
# private directory (QBRT_ROOT) of symlinks to the 1.5.0 header directory and library, the compiler sees it with
# -isystem (searched before /usr/local/include) and the executable finds the library through its RUNPATH.  ldd and
# the executable's describe output show which library was loaded.
#
#   tools/mobilint/build_native.sh            (from the repository root; python3 must import nothing but stdlib)
#   QBRT_LIB=/other/libqbruntime.so.1.5.0 tools/mobilint/build_native.sh    (other install locations)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
make -C "$REPO/native" mobilint \
    QBRT_ROOT="${QBRT_ROOT:-$HOME/sdv_assets/qbruntime-1.5.0}" \
    QBRT_INC="${QBRT_INC:-/usr/include/qbruntime}" \
    QBRT_LIB="${QBRT_LIB:-/usr/lib/x86_64-linux-gnu/libqbruntime.so.1.5.0}"
echo "built $REPO/native/build/sdv-bmt-native-mobilint"
