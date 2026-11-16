// TEST ONLY binary; its name and descriptor always say MOCK.
#include "../../native/src/common/runner.hpp"
#include "../../native/src/eagle_n/backend.hpp"
int main(int argc,char**argv){return sdv::native_main<sdv::EagleNBackend<true>>(argc,argv);}
