// sdv-bmt-native-fake: native measurement plane with the MOCK backend (no device, no CUDA dependency).
#include "common/runner.hpp"
#include "fake/backend.hpp"

int main(int argc, char** argv) { return sdv::native_main<sdv::FakeBackend>(argc, argv); }
