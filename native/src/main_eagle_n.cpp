// Native measurement plane with BOS Eagle-N backend; no Python execution.
#include <cstdlib>
#include "common/runner.hpp"
#include "eagle_n/backend.hpp"
int main(int argc,char** argv){
    // stdout is the native JSON phase protocol, not a vendor log stream.
    // Configure only this process, before dlopen / any SDK initialization.
    ::setenv("TT_LOGGER_LEVEL","off",1);
    ::setenv("UMD_LOG_LEVEL","ERROR",1); // UMD errors go to stderr; INFO uses stdout.
    ::unsetenv("ENABLE_TRACY");
    ::unsetenv("TT_METAL_DEVICE_PROFILER");
    return sdv::native_main<sdv::EagleBackend>(argc,argv);
}
