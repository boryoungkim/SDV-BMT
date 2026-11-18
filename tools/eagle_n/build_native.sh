#!/usr/bin/env bash
# Build the Eagle-N executor and the native runner inside the pinned BOS container (tools/eagle_n/in_container.py
# build). No device is opened by this build.
#
# Writes, next to the executor:
#   <executor>.build.json  build record: executor SHA-256 and the digest of its source (sdv_bmt.dut.eagle_n.
#                          executor_source), compiler and flags. configure.py pins both values in the DUT configs.
#   <executor>.sdk.json    SDK commit and library SHA-256s (tools/eagle_n/sdk_identity.py), checked by the adapter.
# A new executor needs requalification: L1, qualify_native.py, configure.py (tools/eagle_n/README.md).
#
# Environment: EAGLE_N_SDK_ROOT (required), EAGLE_N_EXECUTOR_OUT (default: the executor in the assets root).
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
SDK=${EAGLE_N_SDK_ROOT:?set EAGLE_N_SDK_ROOT to the BOS SDK root}
OUT=${EAGLE_N_EXECUTOR_OUT:-/opt/sdv_assets/eagle_n/eagle_n_integration_assets/libeagle_n_ttnn.so}
CXX=${CXX:-$(command -v clang++-17 || command -v clang++ || command -v g++)}
PYTHON=${PYTHON:-python3}
mkdir -p "$ROOT/native/build" "$(dirname -- "$OUT")"
INCLUDES=()
for path in ttnn ttnn/api ttnn/core ttnn/cpp ttnn/cpp/ttnn ttnn/cpp/ttnn/deprecated tt_metal/api tt_metal \
    tt_metal/include tt_metal/hostdevcommon/api tt_metal/hw/inc tt_metal/third_party/umd \
    tt_metal/third_party/umd/device/api tt_metal/third_party/tracy/public tt_metal/third_party/taskflow tt_stl \
    tt_stl/tt_stl tt_eager build_Release_distributed/include install_distributed/include \
    install_distributed/include/metalium-thirdparty install_distributed/libexec/tt-metalium/ttnn \
    install_distributed/libexec/tt-metalium/tt_metal/hostdevcommon/api; do
  INCLUDES+=(-isystem "$SDK/$path")
done
B=$SDK/build_Release_distributed
FLAGS=(-std=c++20 -O2 -DNDEBUG -fPIC -shared -DFMT_HEADER_ONLY=1 -DSPDLOG_FMT_EXTERNAL -DSPDLOG_FWRITE_UNLOCKED
       -DTRACY_IMPORTS -DTRACY_TIMER_FALLBACK -Wno-int-to-pointer-cast)
"$CXX" "${FLAGS[@]}" "${INCLUDES[@]}" -o "$OUT" "$ROOT/native/src/eagle_n/ttnn_executor.cpp" \
  "$B/ttnn/_ttnncpp.so" "$B/tt_metal/libtt_metal.so" "$B/lib/libdevice.so" "$B/tt_stl/libtt_stl.so" \
  -Wl,-rpath,"$B/ttnn:$B/tt_metal:$B/lib:$B/tt_stl" -ldl -pthread
"$CXX" -std=c++17 -O2 -Wall -Wextra -Werror -o "$ROOT/native/build/sdv-bmt-native-eagle-n" \
  "$ROOT/native/src/main_eagle_n.cpp" -ldl
PYTHONPATH="$ROOT/src" "$PYTHON" -m sdv_bmt.dut.native_source "$ROOT/native" \
  > "$ROOT/native/build/sdv-bmt-native-eagle-n.source-tree.sha256"

PYTHONPATH="$ROOT/src" "$PYTHON" -B - "$OUT" "$ROOT/native" "$("$CXX" --version | head -1)" "${FLAGS[*]}" <<'PY'
import datetime, json, sys
from pathlib import Path
from sdv_bmt.contracts.ids import sha256_hex
from sdv_bmt.dut.eagle_n.executor_source import (BUILD_RECORD_SCHEMA, EXECUTOR_SOURCES, build_record_path,
                                                 executor_source_sha256)
out, native, compiler, flags = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
record = {
    "schema": BUILD_RECORD_SCHEMA,
    "executor": out.name,
    "executor_sha256": sha256_hex(out.read_bytes()),
    "executor_source_sha256": executor_source_sha256(native),
    "sources": {p: sha256_hex((native / p).read_bytes()) for p in EXECUTOR_SOURCES},
    "compiler": compiler,
    "flags": flags,
    "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
build_record_path(out).write_text(json.dumps(record, indent=1) + "\n")
print(json.dumps(record, indent=1))
PY
sha256sum "$OUT" "$ROOT/native/build/sdv-bmt-native-eagle-n"
"$PYTHON" -B "$ROOT/tools/eagle_n/sdk_identity.py" --sdk "$SDK" --out "$OUT.sdk.json"
