// No NPU: exercise error handling of the real integration backend with MOCK ABI.
#include "../../native/src/eagle_n/backend.hpp"
#include <iostream>
int main(int argc,char**argv){
 try{
  if(argc!=2)return 2;auto spec=sdv::Json::parse(argv[1]);int checks=0;
  auto check=[&](bool x){if(!x)throw std::runtime_error("backend guard check failed at "+std::to_string(checks));++checks;};
  sdv::EagleNBackend<true> b(spec);b.prepare(spec);sdv::ErrorInfo e;sdv::DeviceFacts f;std::string_view raw;
  std::vector<char> data(25165824,0);sdv::Sha256 h;h.update(data.data(),data.size());auto sha=h.hex();
  check(!b.collect(f,raw,e)&&e.code=="OutputNotFinal");
  check(!b.stage(0,0,data.data(),data.size(),"bad",e)&&e.code=="StagingDigestMismatch");
  check(!b.submit(0,0,e)&&e.code=="UnstagedInput");
  check(b.stage(0,0,data.data(),data.size(),sha,e));check(b.submit(0,0,e));
  check(b.submit(1,0,e)); /* runner precondition contract: a completed request frees the slot without collect */check(b.poll(e)==1);check(b.collect(f,raw,e)&&raw.size()==2097152&&raw[0]==0);
  check(!b.collect(f,raw,e)&&e.code=="OutputNotFinal");
  check(b.stage(7,1,data.data(),data.size(),sha,e));check(b.submit(2,7,e));
  check(!b.collect(f,raw,e)&&e.code=="LabelOutOfRange"); // 300 must not be truncated to uint8.
  check(!b.stage(2,2,data.data(),data.size(),sha,e)&&e.code=="StagingBudgetExceeded");
  check(b.cleanup().at("eagle_n.device_closed").as_bool());check(b.cleanup().at("eagle_n.device_closed").as_bool());
  std::cout<<"{\"test_only\":true,\"npu_used\":false,\"checks\":"<<checks<<"}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
