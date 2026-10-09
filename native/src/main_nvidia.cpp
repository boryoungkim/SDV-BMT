// sdv-bmt-native-nvidia: native measurement plane with the NVIDIA reference-accelerator backend.
#include "common/runner.hpp"
#include "nvidia/backend.hpp"

int main(int argc, char** argv) { return sdv::native_main<sdv::NvidiaBackend>(argc, argv); }
