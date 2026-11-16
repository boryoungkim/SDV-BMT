#!/usr/bin/env bash
# Run inside the installed BOS RC6 build container. No device is opened by this build.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
SDK=${EAGLE_N_SDK_ROOT:-/home/bmt/tt-metal}
OUT=${EAGLE_N_EXECUTOR_OUT:-/home/bmt/caontd/eagle_n_integration_assets/libeagle_n_ttnn.so}
CXX=${CXX:-$(command -v clang++-17 || command -v clang++ || command -v g++)}
PYTHON=${PYTHON:-python3}
mkdir -p "$ROOT/native/build" "$(dirname -- "$OUT")"
INCLUDES=()
for path in ttnn ttnn/api ttnn/core ttnn/cpp ttnn/cpp/ttnn ttnn/cpp/ttnn/deprecated tt_metal/api tt_metal tt_metal/include tt_metal/hostdevcommon/api tt_metal/hw/inc tt_metal/third_party/umd tt_metal/third_party/umd/device/api tt_metal/third_party/tracy/public tt_metal/third_party/taskflow tt_stl tt_stl/tt_stl tt_eager build_Release_distributed/include install_distributed/include install_distributed/include/metalium-thirdparty install_distributed/libexec/tt-metalium/ttnn install_distributed/libexec/tt-metalium/tt_metal/hostdevcommon/api; do INCLUDES+=(-isystem "$SDK/$path"); done
"$CXX" -std=c++20 -O2 -DNDEBUG -fPIC -shared -DFMT_HEADER_ONLY=1 -DSPDLOG_FMT_EXTERNAL -DSPDLOG_FWRITE_UNLOCKED -DTRACY_IMPORTS -DTRACY_TIMER_FALLBACK -Wno-int-to-pointer-cast "${INCLUDES[@]}" -o "$OUT" "$ROOT/native/src/eagle_n/ttnn_executor.cpp" "$SDK/build_Release_distributed/ttnn/_ttnncpp.so" "$SDK/build_Release_distributed/tt_metal/libtt_metal.so" "$SDK/build_Release_distributed/lib/libdevice.so" "$SDK/build_Release_distributed/tt_stl/libtt_stl.so" -Wl,-rpath,"$SDK/build_Release_distributed/ttnn:$SDK/build_Release_distributed/tt_metal:$SDK/build_Release_distributed/lib:$SDK/build_Release_distributed/tt_stl" -ldl -pthread
"$CXX" -std=c++17 -O2 -Wall -Wextra -Werror -o "$ROOT/native/build/sdv-bmt-native-eagle-n" "$ROOT/native/src/main_eagle_n.cpp" -ldl
PYTHONPATH="$ROOT/src" "$PYTHON" -m sdv_bmt.dut.native_source "$ROOT/native" > "$ROOT/native/build/sdv-bmt-native-eagle-n.source-tree.sha256"
sha256sum "$OUT" "$ROOT/native/build/sdv-bmt-native-eagle-n"

"$PYTHON" -B "$ROOT/tools/eagle_n/sdk_identity.py" --sdk "$SDK" --out "$OUT.sdk.json"
