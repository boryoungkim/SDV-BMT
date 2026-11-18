// sdv-bmt-native-mobilint: native measurement plane with the Mobilint ARIES NPU backend.
#include "common/runner.hpp"
#include "mobilint/backend.hpp"

int main(int argc, char** argv) { return sdv::native_main<sdv::MobilintBackend>(argc, argv); }
