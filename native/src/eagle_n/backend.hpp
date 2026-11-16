// Eagle-N native contract backend. SDK stays in a separately built C++ executor.
// No Python, model subprocess, file logging or allocation on a successful request.
#pragma once
#include <dlfcn.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <array>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>
#include "../common/json.hpp"
#include "../common/loop.hpp"
#include "../common/sha256.hpp"
#include "executor_abi.hpp"
namespace sdv {
namespace eagle {
inline std::string read(const std::string& path) {
    std::ifstream f(path,std::ios::binary);if(!f)throw JsonError("cannot read "+path);
    return std::string(std::istreambuf_iterator<char>(f),{});
}
inline std::string digest_file(const std::string& path) {
    std::ifstream f(path,std::ios::binary);if(!f)throw JsonError("cannot read "+path);
    Sha256 s;std::array<char,65536> b{};while(f){f.read(b.data(),b.size());s.update(b.data(),size_t(f.gcount()));}return s.hex();
}
class Library {
 public:
    Library(const std::string& path,const std::string& digest) {
        if(digest_file(path)!=digest)throw JsonError("executor SHA-256 mismatch");
        lib_=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);if(!lib_)throw JsonError(dlerror());
        auto get=reinterpret_cast<GetExecutor>(dlsym(lib_,"sdv_eagle_executor_v1"));
        if(!get){dlclose(lib_);lib_=nullptr;throw JsonError("executor ABI entry missing");}
        api=get();if(!api||api->abi!=kExecutorABI||!api->prepare||!api->stage||!api->run||!api->destroy||!api->describe){dlclose(lib_);lib_=nullptr;throw JsonError("executor ABI mismatch");}
    }
    ~Library(){if(lib_)dlclose(lib_);}
    Library(const Library&)=delete;Library& operator=(const Library&)=delete;
    const ExecutorAPI* api=nullptr;
 private:void* lib_=nullptr;
};
} // namespace eagle

template<bool ContractTest=false> class EagleNBackend {
 public:
    static constexpr const char* kName=ContractTest?"eagle_n_contract_mock":"bos_eagle_n";
    explicit EagleNBackend(const Json& s):spec_(s){}
    ~EagleNBackend(){release();}
    static Json describe(const Json& args) {
        eagle::Library lib(args.at("executor").as_string(),args.at("executor_sha256").as_string());
        if(lib.api->real_device==ContractTest)throw JsonError("executor real/mock identity mismatch");
        Json out=Json::parse(lib.api->describe());out["adapter"]=kName;out["mock"]=ContractTest;
        out["role"]="accelerator_under_test";out["accelerator_family"]="NPU";out["vendor"]="BOS Semiconductors";
        out["eagle_n.executor_sha256"]=args.at("executor_sha256");out["eagle_n.submit_semantics"]="synchronous trace replay + final D2H; poll returns completion afterwards";
        return out;
    }
    Json prepare(const Json&) {
        const Json& opt=spec_.at("backend").at("options");const Json& p=spec_.at("program");
        if(opt.at("device_id").as_int()!=0)throw JsonError("only explicitly qualified device 0 supported");
        if(p.at("input_bytes").as_int()!=25165824 || p.at("task_output_bytes").as_int()!=2097152)throw JsonError("DeepLab canonical byte contract mismatch");
        for(const char* k:{"plan","weights"})if(eagle::digest_file(p.at(std::string(k)+"_path").as_string())!=p.at(std::string(k)+"_sha256").as_string())throw JsonError(std::string(k)+" digest mismatch");
        auto plan=Json::parse(eagle::read(p.at("plan_path").as_string()));
        if(plan.at("onnx_sha256").as_string()!=p.at("artifact_sha256").as_string())throw JsonError("H04 plan identity mismatch");
        const std::string executor=opt.at("executor").as_string(),hash=opt.at("executor_sha256").as_string();
        const bool trial=opt.has("trial_unqualified")&&opt.at("trial_unqualified").as_bool();
        if(trial&&plan.at("variant").as_string()!="r18")throw JsonError("Unqualified trial restricted to R18");
        if(!ContractTest&&!trial){
            auto gate=Json::parse(eagle::read(p.at("qualification_path").as_string()));
            if(!gate.at("passed").as_bool()||gate.at("plan_sha256").as_string()!=p.at("plan_sha256").as_string()||gate.at("executor_sha256").as_string()!=hash||gate.at("weights_sha256").as_string()!=p.at("weights_sha256").as_string())throw JsonError("native model qualification absent/stale");
        }
        lib_=std::make_unique<eagle::Library>(executor,hash);
        if(lib_->api->real_device==ContractTest)throw JsonError("test executor forbidden in ACTUAL backend");
        lock_=::open(opt.at("lock_path").as_string().c_str(),O_CREAT|O_RDWR,0600);
        if(lock_<0||flock(lock_,LOCK_EX|LOCK_NB)!=0)throw JsonError("NPU integration device lock busy/unavailable");
        budget_=size_t(opt.at("staging_budget_bytes").as_int());item_bytes_=plan.at("precision").as_string()=="bf16"?12582912:25165824;
        out32_.resize(2097152);out8_.resize(2097152);
        handle_=lib_->api->prepare(spec_.dump().c_str(),error_.data(),error_.size());
        if(!handle_)throw JsonError(error_.data());
        Json f=Json::object();f["eagle_n.precision"]=plan.at("precision");f["eagle_n.onnx_sha256"]=plan.at("onnx_sha256");
        f["eagle_n.trial_unqualified"]=trial;f["eagle_n.graph_mode"]=trial?"direct graph; first-use JIT/allocation included":"captured trace replay";
        f["eagle_n.input_residency"]="staged_host_nhwc";f["eagle_n.output_bytes"]=2097152;f["eagle_n.real_device"]=!ContractTest;
        f["eagle_n.output_cast"]="collect checked uint32 to uint8";f["eagle_n.device_timing"]=nullptr;return f;
    }
    bool stage(int slot,int64_t,const char* data,size_t n,const std::string& sha,ErrorInfo& err) {
        Sha256 h;h.update(data,n);if(h.hex()!=sha){err={"StagingDigestMismatch","canonical input SHA-256 mismatch"};return false;}
        if(n!=25165824){err={"InputSizeMismatch","expected CHW FP32 25165824 bytes"};return false;}
        if(slot<0){err={"InvalidSlot","negative stage slot"};return false;}
        bool fresh=size_t(slot)>=staged_slots_.size()||!staged_slots_[size_t(slot)];
        if(fresh&&(item_bytes_>budget_||staged_bytes_>budget_-item_bytes_)){err={"StagingBudgetExceeded","staged host inputs exceed explicit memory budget"};return false;}
        if(!lib_->api->stage(handle_,slot,data,n,error_.data(),error_.size())){err={"EagleStageFailure",error_.data()};return false;}
        if(fresh){if(size_t(slot)>=staged_slots_.size())staged_slots_.resize(size_t(slot)+1,false);staged_slots_[size_t(slot)]=true;staged_bytes_+=item_bytes_;}
        return true;
    }
    bool submit(int64_t,int slot,ErrorInfo& err) {
        if(slot<0||size_t(slot)>=staged_slots_.size()||!staged_slots_[size_t(slot)]){err={"UnstagedInput","slot not staged"};return false;}
        // SDV-BMT runner contract (common/runner.hpp precondition): warm-up requests are submit+poll only, never
        // collected; a completed (final) request therefore frees the slot, as in the nvidia/fake backends.
        if(inflight_&&!final_){err={"SlotBusy","previous request not complete"};return false;}
        final_=false;inflight_=true;
        if(!lib_->api->run(handle_,slot,out32_.data(),out32_.size(),error_.data(),error_.size())){inflight_=false;err={"EagleExecutionFailure",error_.data()};return false;}
        final_=true;return true;
    }
    int poll(ErrorInfo&){return inflight_&&final_?1:0;}
    bool collect(DeviceFacts& f,std::string_view& raw,ErrorInfo& err) {
        if(!inflight_||!final_){err={"OutputNotFinal","completion not observed"};return false;}
        for(size_t i=0;i<out32_.size();++i){if(out32_[i]>=19){inflight_=false;err={"LabelOutOfRange","native ArgMax class outside 0..18"};return false;}out8_[i]=static_cast<char>(out32_[i]);}
        raw=std::string_view(out8_.data(),out8_.size());f.observe("eagle_n.raw_bytes",2097152);f.observe("eagle_n.synchronous",1);inflight_=false;return true;
    }
    void mark_admission(int64_t,const char*){}
    Json observe(const std::string& point){Json j=Json::object();j["point"]=point;j["eagle_n.profiler"]="off";j["eagle_n.device_time"]="not_observable";j["eagle_n.input_residency"]="staged_host_nhwc";return j;}
    Json cleanup(){release();Json j=Json::object();j["eagle_n.device_closed"]=closed_;if(!closed_)j["eagle_n.close_error"]=error_.data();return j;}
 private:
    void release(){if(handle_&&lib_){closed_=lib_->api->destroy(handle_,error_.data(),error_.size());handle_=nullptr;}lib_.reset();if(lock_>=0){flock(lock_,LOCK_UN);::close(lock_);lock_=-1;}}
    size_t budget_=0,item_bytes_=0,staged_bytes_=0;std::vector<bool> staged_slots_;
    Json spec_;std::unique_ptr<eagle::Library> lib_;void* handle_=nullptr;int lock_=-1;
    std::vector<uint32_t> out32_;std::vector<char> out8_;std::array<char,2048> error_{};
    bool inflight_=false,final_=false,closed_=false;
};
using EagleBackend=EagleNBackend<false>;
} // namespace sdv
