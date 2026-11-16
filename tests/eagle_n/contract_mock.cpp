// TEST ONLY: no SDK/device. Refuses use by the ACTUAL backend (real_device=false).
#include "../../native/src/eagle_n/executor_abi.hpp"
#include <algorithm>
#include <cstdio>
#include <map>
struct State {std::map<int,uint32_t> values;};
static const char* describe(){return "{\"mock\":true,\"test_only\":true,\"eagle_n.runtime\":\"contract test; no device\"}";}
static void* prepare(const char*,char*,size_t){return new State;}
static bool stage(void*p,int slot,const void*,size_t n,char*e,size_t cap){if(n!=25165824){std::snprintf(e,cap,"bad size");return false;}static_cast<State*>(p)->values[slot]=uint32_t(slot==7?300:slot%19);return true;}
static bool run(void*p,int slot,uint32_t*out,size_t pixels,char*e,size_t cap){auto*s=static_cast<State*>(p);auto it=s->values.find(slot);if(it==s->values.end()){std::snprintf(e,cap,"missing slot");return false;}std::fill(out,out+pixels,it->second);return true;}
static bool destroy(void*p,char*,size_t){delete static_cast<State*>(p);return true;}
extern "C" const sdv::eagle::ExecutorAPI* sdv_eagle_executor_v1(){static const sdv::eagle::ExecutorAPI a{1,false,describe,prepare,stage,run,destroy};return &a;}
