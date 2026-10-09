// SDV-BMT native runner: NVIDIA reference-accelerator backend (adapter nvidia_cuda, actual execution).
//
// NVIDIA-specific concepts (CUDA context/stream/events, NVRTC, NVML, NVTX) live only in this file.  The
// evidence it reports uses portable names (segments host_to_device / device_execution / device_to_host)
// or keys namespaced "nvidia." / sources "nvidia:...".
//
// Execution path (Stage 2a-G): device selected by UUID; primary context with CU_CTX_SCHED_BLOCKING_SYNC;
// the device program (CUDA C source carried by the workload spec, SHA-256 pinned) compiled by NVRTC for the
// device's compute capability; persistent device buffers and a pinned output buffer allocated in prepare;
// prepared inputs staged per item in pinned host memory (residency host_pinned, H2D per request inside
// DUT-service) or in device memory verified by read-back (device_resident, H2D excluded); submit enqueues
// [H2D] -> kernel -> D2H on one non-blocking stream with CUDA events; completion is cuEventQuery on the
// last event.  buffer_policy per_request_alloc (host_pinned only, characterization) allocates the device
// input buffer in submit and frees it in the completing poll.
//
// Device program kinds: "cuda_c" (one NVRTC kernel, one block, the reference conformance workload) and, since
// Stage 2b-G real-model baseline, "cudnn_plan" (a converted model executed with cuDNN and NVRTC glue kernels by
// cudnn_plan.hpp; the per-request sequence is [H2D] -> frequency probe -> plan steps -> D2H on the same stream).
//
// Instrumentation (part of the measurement identity, declared in the DUT config):
//   instrumentation cuda_events | none   segment event brackets (device-clock durations, include event cost)
//   nvtx true | false                    NVTX request ranges and admission/completion marks (domain sdv-bmt)
//   profiler none | nsys                 the declared profiler; prepare fails if the observed injection
//                                        environment differs (a configured and a realized measurement
//                                        configuration are not assumed to be identical)
#pragma once

#include <cuda.h>
#include <dlfcn.h>
#include <nvml.h>
#include <nvrtc.h>
#include <nvtx3/nvToolsExt.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../common/json.hpp"
#include "../common/loop.hpp"
#include "../common/sha256.hpp"
#include "cudnn_plan.hpp"

namespace sdv {

class CudaFailure : public std::runtime_error {
   public:
    CudaFailure(const std::string& api, CUresult r) : std::runtime_error(text(api, r)), code(name(r)) {}
    static std::string name(CUresult r) {
        const char* n = nullptr;
        cuGetErrorName(r, &n);
        return n ? n : "CUDA_ERROR_UNKNOWN";
    }
    static std::string text(const std::string& api, CUresult r) {
        const char* s = nullptr;
        cuGetErrorString(r, &s);
        return api + ": " + name(r) + " (" + std::to_string(int(r)) + ") " + (s ? s : "");
    }
    std::string code;
};

inline void cu(CUresult r, const char* api) {
    if (r != CUDA_SUCCESS) throw CudaFailure(api, r);
}

inline void nv(nvmlReturn_t r, const char* api) {
    if (r != NVML_SUCCESS) throw JsonError(std::string(api) + ": " + nvmlErrorString(r));
}

inline int64_t mono_abs_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

inline int64_t realtime_us() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return int64_t(ts.tv_sec) * 1000000LL + ts.tv_nsec / 1000;
}

inline std::string uuid_text(const CUuuid& u) {
    const unsigned char* b = reinterpret_cast<const unsigned char*>(u.bytes);
    char buf[64];
    std::snprintf(buf, sizeof buf, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0],
                  b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
    return buf;
}

inline CUdevice find_device(const std::string& uuid) {
    int n = 0;
    cu(cuDeviceGetCount(&n), "cuDeviceGetCount");
    for (int k = 0; k < n; ++k) {
        CUdevice d;
        cu(cuDeviceGet(&d, k), "cuDeviceGet");
        CUuuid u;
        cu(cuDeviceGetUuid_v2(&u, d), "cuDeviceGetUuid_v2");
        if (uuid_text(u) == uuid) return d;
    }
    throw JsonError("no CUDA device with UUID " + uuid);
}

class NvidiaBackend {
   public:
    static constexpr const char* kName = "nvidia_cuda";
    static constexpr const char* kTimingMethod = "nvidia:cuda_event_pair:bracket";

    explicit NvidiaBackend(const Json& spec) {
        uuid_ = spec.at("device_uuid").as_string();
        const Json& opt = spec.at("backend").at("options");
        auto str = [&](const char* key, const char* dflt) {
            return opt.has(key) ? opt.at(key).as_string() : std::string(dflt);
        };
        residency_ = str("residency", "host_pinned");
        instrumentation_ = str("instrumentation", "cuda_events");
        buffer_policy_ = str("buffer_policy", "preallocated_persistent");
        profiler_ = str("profiler", "none");
        if (str("context_sched", "blocking_sync") != "blocking_sync") throw JsonError("context_sched: blocking_sync only");
        nvtx_ = opt.has("nvtx") && opt.at("nvtx").as_bool();
        if (residency_ != "host_pinned" && residency_ != "device_resident") throw JsonError("bad residency");
        if (instrumentation_ != "cuda_events" && instrumentation_ != "none") throw JsonError("bad instrumentation");
        if (buffer_policy_ != "preallocated_persistent" && buffer_policy_ != "per_request_alloc")
            throw JsonError("bad buffer_policy");
        if (buffer_policy_ == "per_request_alloc" && residency_ != "host_pinned")
            throw JsonError("per_request_alloc applies to host_pinned only");
        if (profiler_ != "none" && profiler_ != "nsys") throw JsonError("bad profiler");
        events_on_ = instrumentation_ == "cuda_events";
        options_ = opt;
        const Json& p = spec.at("program");
        program_ = p;
        cudnn_plan_ = p.has("kind") && p.at("kind").as_string() == "cudnn_plan";
        if (cudnn_plan_) {
            if (!opt.has("cudnn")) throw JsonError("a cudnn_plan program requires the cudnn runtime options");
        } else {
            if (opt.has("cudnn") || opt.has("frequency_probe_cycles"))
                throw JsonError("cudnn / frequency_probe_cycles options apply to cudnn_plan programs only");
            source_ = p.at("source").as_string();
            entry_ = p.at("entry").as_string();
            block_threads_ = unsigned(p.at("block_threads").as_int());
            for (const Json& v : p.at("scratch_bytes").as_array()) scratch_bytes_.push_back(size_t(v.as_int()));
        }
        input_bytes_ = size_t(p.at("input_bytes").as_int());
        output_bytes_ = size_t(p.at("output_bytes").as_int());
        task_output_bytes_ = size_t(p.at("task_output_bytes").as_int());
        timing_offset_ = p.at("timing_offset").is_null() ? -1 : p.at("timing_offset").as_int();
        artifact_sha256_ = p.at("artifact_sha256").as_string();
    }

    // ---------------------------------------------------------------- identity (no context is created)
    static Json describe(const Json& args) {
        const std::string uuid = args.at("uuid").as_string();
        nv(nvmlInit_v2(), "nvmlInit_v2");
        nvmlDevice_t h;
        nv(nvmlDeviceGetHandleByUUID(uuid.c_str(), &h), "nvmlDeviceGetHandleByUUID");
        char name[96] = {}, vbios[96] = {}, part[96] = {}, driver[96] = {};
        nv(nvmlDeviceGetName(h, name, sizeof name), "nvmlDeviceGetName");
        nv(nvmlDeviceGetVbiosVersion(h, vbios, sizeof vbios), "nvmlDeviceGetVbiosVersion");
        nv(nvmlDeviceGetBoardPartNumber(h, part, sizeof part), "nvmlDeviceGetBoardPartNumber");
        nv(nvmlSystemGetDriverVersion(driver, sizeof driver), "nvmlSystemGetDriverVersion");
        nvmlPciInfo_t pci;
        nv(nvmlDeviceGetPciInfo_v3(h, &pci), "nvmlDeviceGetPciInfo_v3");
        nvmlMemory_v2_t mem{};
        mem.version = nvmlMemory_v2;
        nv(nvmlDeviceGetMemoryInfo_v2(h, &mem), "nvmlDeviceGetMemoryInfo_v2");
        nvmlShutdown();
        cu(cuInit(0), "cuInit");
        CUdevice d = find_device(uuid);
        int cc_major = 0, cc_minor = 0, sms = 0, cuda_api = 0, rt_major = 0, rt_minor = 0;
        cu(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, d), "attr");
        cu(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, d), "attr");
        cu(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, d), "attr");
        cu(cuDriverGetVersion(&cuda_api), "cuDriverGetVersion");
        nvrtcVersion(&rt_major, &rt_minor);
        Dl_info info{};
        std::string nvrtc_path = dladdr(reinterpret_cast<void*>(&nvrtcVersion), &info) && info.dli_fname
                                     ? info.dli_fname : "unknown";
        char hex[16];
        Json id = Json::object();
        id["role"] = "reference_accelerator_validation";
        id["accelerator_family"] = "GPU";
        id["vendor"] = "NVIDIA";
        id["nvidia.product_name"] = std::string(name);
        id["nvidia.uuid"] = uuid;
        id["nvidia.pci_bus_id"] = std::string(pci.busId);
        std::snprintf(hex, sizeof hex, "0x%x", pci.pciDeviceId);
        id["nvidia.pci_device_id"] = std::string(hex);
        std::snprintf(hex, sizeof hex, "0x%x", pci.pciSubSystemId);
        id["nvidia.pci_subsystem_id"] = std::string(hex);
        id["nvidia.vbios_version"] = std::string(vbios);
        id["nvidia.board_part_number"] = std::string(part);
        id["nvidia.memory_total_bytes"] = static_cast<unsigned long long>(mem.total);
        id["nvidia.driver_version"] = std::string(driver);
        id["nvidia.cuda_driver_api_version"] = cuda_api;
        id["nvidia.compute_capability"] = std::to_string(cc_major) + "." + std::to_string(cc_minor);
        id["nvidia.sm_count"] = sms;
        id["nvidia.nvrtc_version"] = std::to_string(rt_major) + "." + std::to_string(rt_minor);
        id["nvidia.nvrtc_library"] = nvrtc_path;
        const Json cudnn = CudnnPlanExecutor::library_identity();  // keep alive: the loop iterates its object
        for (const auto& kv : cudnn.as_object()) id[kv.first] = kv.second;
        return id;
    }

    // ---------------------------------------------------------------- DUT.prepare
    Json prepare(const Json&) {
        Json facts = Json::object();
        facts["profiler_injection"] = profiler_environment();
        bool injected = facts["profiler_injection"].at("injected").as_bool();
        if ((profiler_ == "nsys") != injected)
            throw JsonError("declared profiler '" + profiler_ + "' but a profiler injection environment is " +
                            (injected ? "present" : "absent"));
        if (!cudnn_plan_ && Sha256::of(source_) != artifact_sha256_)
            throw JsonError("device program SHA-256 does not match workload.artifact_sha256");
        cu(cuInit(0), "cuInit");
        dev_ = find_device(uuid_);
        CUresult r = cuDevicePrimaryCtxSetFlags(dev_, CU_CTX_SCHED_BLOCKING_SYNC);
        if (r != CUDA_SUCCESS && r != CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE) cu(r, "cuDevicePrimaryCtxSetFlags");
        cu(cuDevicePrimaryCtxRetain(&ctx_, dev_), "cuDevicePrimaryCtxRetain");
        ctx_retained_ = true;
        cu(cuCtxSetCurrent(ctx_), "cuCtxSetCurrent");
        int major = 0, minor = 0;
        cu(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev_), "attr");
        cu(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev_), "attr");
        std::string arch = "--gpu-architecture=sm_" + std::to_string(major) + std::to_string(minor);
        std::string cubin, log;
        if (!cudnn_plan_) {
            compile(arch, cubin, log);
            cubin_sha256_ = Sha256::of(cubin);
            cu(cuModuleLoadData(&module_, cubin.data()), "cuModuleLoadData");
            cu(cuModuleGetFunction(&fn_, module_, entry_.c_str()), "cuModuleGetFunction");
        }
        cu(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
        if (cudnn_plan_) {
            executor_ready_ = true;  // release() is safe on a partially prepared executor
            facts["cudnn_plan"] = executor_.prepare(stream_, arch, program_, options_, device_allocs_);
            cubin_sha256_ = facts["cudnn_plan"].at("glue_cubin_sha256").as_string();
        }
        unsigned flags = events_on_ ? CU_EVENT_DEFAULT : CU_EVENT_DISABLE_TIMING;
        if (events_on_) {
            cu(cuEventCreate(&ev_start_, flags), "cuEventCreate");
            cu(cuEventCreate(&ev_h2d_, flags), "cuEventCreate");
            cu(cuEventCreate(&ev_kernel_, flags), "cuEventCreate");
        }
        cu(cuEventCreate(&ev_done_, flags), "cuEventCreate");
        if (residency_ == "host_pinned" && buffer_policy_ == "preallocated_persistent") d_input_ = device_alloc(input_bytes_);
        for (size_t n : scratch_bytes_) d_scratch_.push_back(device_alloc(n));
        d_output_ = device_alloc(output_bytes_);
        h_output_ = pinned_alloc(output_bytes_);
        // kernel arguments (input, scratch..., output), built once; submit only sets the input pointer
        arg_ptrs_.assign(d_scratch_.size() + 2, 0);
        for (size_t k = 0; k < d_scratch_.size(); ++k) arg_ptrs_[k + 1] = d_scratch_[k];
        arg_ptrs_.back() = d_output_;
        for (CUdeviceptr& p : arg_ptrs_) args_.push_back(&p);
        nv(nvmlInit_v2(), "nvmlInit_v2");
        nvml_init_ = true;
        nv(nvmlDeviceGetHandleByUUID(uuid_.c_str(), &nvml_), "nvmlDeviceGetHandleByUUID");
        if (nvtx_) {
            domain_ = nvtxDomainCreateA("sdv-bmt");
            msg_submit_ = nvtxDomainRegisterStringA(domain_, "sdv-bmt.submit");
            msg_complete_ = nvtxDomainRegisterStringA(domain_, "sdv-bmt.complete");
            msg_drop_ = nvtxDomainRegisterStringA(domain_, "sdv-bmt.drop");
            msg_not_issued_ = nvtxDomainRegisterStringA(domain_, "sdv-bmt.not_issued");
        }
        prepared_ = true;
        facts["cubin_sha256"] = cubin_sha256_;
        if (!cudnn_plan_) {
            facts["nvrtc_options"] = arch + " --std=c++17";
            facts["nvrtc_log"] = log;
        }
        facts["device_allocations_total"] = device_allocs_;
        facts["pinned_allocations_total"] = pinned_allocs_;
        return facts;
    }

    // ---------------------------------------------------------------- PREPARE_INPUTS (staging)
    bool stage(int slot, int64_t item_index, const char* data, size_t n, const std::string& sha256, ErrorInfo& err) {
        try {
            if (n != input_bytes_) throw JsonError("prepared input size differs from the device program input");
            void* host = nullptr;
            cu(cuMemAllocHost(&host, n), "cuMemAllocHost");
            ++pinned_allocs_;
            std::memcpy(host, data, n);
            Staged s;
            s.item_index = item_index;
            s.nbytes = n;
            if (Sha256::of(host, n) != sha256) {
                cuMemFreeHost(host);
                throw JsonError("staged input does not match its SHA-256");
            }
            if (residency_ == "host_pinned") {
                s.host = host;
            } else {  // device_resident: copy once, verify the device representation by read-back
                s.device = device_alloc(n);
                cu(cuMemcpyHtoD(s.device, host, n), "cuMemcpyHtoD");
                std::memset(host, 0, n);
                cu(cuMemcpyDtoH(host, s.device, n), "cuMemcpyDtoH");
                bool same = Sha256::of(host, n) == sha256;
                cuMemFreeHost(host);
                if (!same) throw JsonError("device-resident input does not read back identically");
            }
            if (slot >= int(staged_.size())) staged_.resize(size_t(slot) + 1);
            staged_[size_t(slot)] = s;
            return true;
        } catch (const CudaFailure& e) {
            err = ErrorInfo{e.code, e.what()};
        } catch (const std::exception& e) {
            err = ErrorInfo{"StagingError", e.what()};
        }
        return false;
    }

    // ---------------------------------------------------------------- requests
    bool submit(int64_t rid, int slot, ErrorInfo& err) {
        const Staged& s = staged_[size_t(slot)];
        CUdeviceptr transient = 0;
        if (nvtx_) {
            nvtxEventAttributes_t a = attr(msg_submit_, rid);
            nvtxDomainRangePushEx(domain_, &a);
        }
        try {
            if (s.host && buffer_policy_ == "per_request_alloc") {
                transient = device_alloc(s.nbytes, false);
                transient_ = transient;
            }
            CUdeviceptr input = s.device;
            if (events_on_) cu(cuEventRecord(ev_start_, stream_), "cuEventRecord");
            h2d_bytes_ = 0;
            if (s.host) {
                input = transient ? transient : d_input_;
                cu(cuMemcpyHtoDAsync(input, s.host, s.nbytes, stream_), "cuMemcpyHtoDAsync");
                h2d_bytes_ = int64_t(s.nbytes);
                if (events_on_) cu(cuEventRecord(ev_h2d_, stream_), "cuEventRecord");
            }
            if (cudnn_plan_) {
                executor_.enqueue(input, d_output_, d_output_ + CUdeviceptr(timing_offset_));
            } else {
                arg_ptrs_[0] = input;
                cu(cuLaunchKernel(fn_, 1, 1, 1, block_threads_, 1, 1, 0, stream_, args_.data(), nullptr),
                   "cuLaunchKernel");
            }
            if (events_on_) cu(cuEventRecord(ev_kernel_, stream_), "cuEventRecord");
            cu(cuMemcpyDtoHAsync(h_output_, d_output_, output_bytes_, stream_), "cuMemcpyDtoHAsync");
            cu(cuEventRecord(ev_done_, stream_), "cuEventRecord");
        } catch (const std::exception& e) {
            if (nvtx_) nvtxDomainRangePop(domain_);
            if (transient) {  // work may be enqueued on it: free only at cleanup, after the stream drained
                leaked_.push_back(transient);
                transient_ = 0;
            }
            const CudaFailure* cf = dynamic_cast<const CudaFailure*>(&e);
            const CudnnFailure* df = dynamic_cast<const CudnnFailure*>(&e);
            err = ErrorInfo{cf ? cf->code : df ? df->code : "SubmitError", e.what()};
            return false;
        }
        if (nvtx_) nvtxDomainRangePop(domain_);
        current_rid_ = rid;
        return true;
    }

    int poll(ErrorInfo& err) {
        CUresult r = cuEventQuery(ev_done_);
        if (r == CUDA_ERROR_NOT_READY) return 0;
        if (transient_) {  // per_request_alloc: release inside the measured path, after completion
            cuMemFree(transient_);
            transient_ = 0;
        }
        if (r != CUDA_SUCCESS) {
            err = ErrorInfo{CudaFailure::name(r), CudaFailure::text("cuEventQuery", r)};
            return -1;
        }
        return 1;
    }

    bool collect(DeviceFacts& f, std::string_view& raw, ErrorInfo& err) {
        if (nvtx_) {
            nvtxEventAttributes_t a = attr(msg_complete_, current_rid_);
            nvtxDomainMarkEx(domain_, &a);
        }
        try {
            if (events_on_) {
                f.timing_method = kTimingMethod;
                if (h2d_bytes_) f.segment("host_to_device", elapsed(ev_start_, ev_h2d_));
                f.segment("device_execution", elapsed(h2d_bytes_ ? ev_h2d_ : ev_start_, ev_kernel_));
                f.segment("device_to_host", elapsed(ev_kernel_, ev_done_));
            }
        } catch (const CudaFailure& e) {
            err = ErrorInfo{e.code, e.what()};
            return false;
        }
        f.observe("nvidia.h2d_bytes", h2d_bytes_);
        const char* out = static_cast<const char*>(h_output_);
        if (timing_offset_ >= 0) {
            uint64_t w[4];
            std::memcpy(w, out + timing_offset_, sizeof w);
            if (w[1] > w[0] && w[3] >= w[2]) {
                f.observe("nvidia.in_kernel_globaltimer_ns", int64_t(w[1] - w[0]));
                f.observe("nvidia.in_kernel_sm_cycles", int64_t(w[3] - w[2]));
            }
        }
        raw = std::string_view(out, task_output_bytes_);  // valid until the next submit
        return true;
    }

    void mark_admission(int64_t rid, const char* outcome) {
        if (!nvtx_) return;
        nvtxEventAttributes_t a = attr(std::strcmp(outcome, "dropped") == 0 ? msg_drop_ : msg_not_issued_, rid);
        nvtxDomainMarkEx(domain_, &a);
    }

    // ---------------------------------------------------------------- backend snapshot (outside the window)
    Json observe(const std::string& point) {
        int64_t t_begin = mono_abs_ns();
        Json s = Json::object();
        s["point"] = point;
        s["t_monotonic_ns"] = t_begin;
        s["t_realtime_us"] = realtime_us();
        unsigned u = 0;
        unsigned long long ull = 0;
        nvmlPstates_t ps;
        if (nvmlDeviceGetPerformanceState(nvml_, &ps) == NVML_SUCCESS) s["pstate"] = int(ps);
        if (nvmlDeviceGetClockInfo(nvml_, NVML_CLOCK_SM, &u) == NVML_SUCCESS) s["sm_clock_mhz"] = u;
        if (nvmlDeviceGetClockInfo(nvml_, NVML_CLOCK_MEM, &u) == NVML_SUCCESS) s["mem_clock_mhz"] = u;
        nvmlTemperature_t t{};
        t.version = nvmlTemperature_v1;
        t.sensorType = NVML_TEMPERATURE_GPU;
        if (nvmlDeviceGetTemperatureV(nvml_, &t) == NVML_SUCCESS) s["temperature_c"] = t.temperature;
        if (nvmlDeviceGetPowerUsage(nvml_, &u) == NVML_SUCCESS) s["power_mw"] = u;
        if (nvmlDeviceGetCurrentClocksEventReasons(nvml_, &ull) == NVML_SUCCESS) s["clocks_event_reasons"] = reasons(ull);
        unsigned gen = 0, width = 0;
        if (nvmlDeviceGetCurrPcieLinkGeneration(nvml_, &gen) == NVML_SUCCESS &&
            nvmlDeviceGetCurrPcieLinkWidth(nvml_, &width) == NVML_SUCCESS) {
            Json link = Json::array();
            link.push(gen);
            link.push(width);
            s["pcie_link"] = link;
        }
        nvmlMemory_v2_t mem{};
        mem.version = nvmlMemory_v2;
        if (nvmlDeviceGetMemoryInfo_v2(nvml_, &mem) == NVML_SUCCESS)
            s["memory_used_bytes"] = static_cast<unsigned long long>(mem.used);
        s["foreign_compute_processes"] = foreign_processes();
        s["fields"] = fields();
        if (prepared_) s["stream_idle"] = cuStreamQuery(stream_) == CUDA_SUCCESS;
        s["device_allocations_total"] = device_allocs_;
        s["pinned_allocations_total"] = pinned_allocs_;
        s["cubin_sha256"] = cubin_sha256_;
        if (point == "measure_begin") since_us_ = s.at("t_realtime_us").as_int();
        if (point == "drain_end" && since_us_ > 0) s["process_utilization"] = foreign_activity(since_us_);
        s["duration_ns"] = mono_abs_ns() - t_begin;
        return s;
    }

    // ---------------------------------------------------------------- DUT.cleanup
    Json cleanup() {
        Json facts = Json::object();
        if (cleaned_) {
            facts["already_cleaned"] = true;
            return facts;
        }
        bool busy = false;
        if (prepared_) {
            int64_t deadline = mono_abs_ns() + 5000000000LL;
            while (cuStreamQuery(stream_) == CUDA_ERROR_NOT_READY) {
                if (mono_abs_ns() > deadline) {
                    busy = true;  // device work still running: never free memory it may use
                    break;
                }
            }
        }
        if (!busy && ctx_retained_) {
            for (const Staged& s : staged_) {
                if (s.host) cuMemFreeHost(s.host);
                if (s.device) cuMemFree(s.device);
            }
            if (executor_ready_) executor_.release();
            for (CUdeviceptr p : d_scratch_) cuMemFree(p);
            for (CUdeviceptr p : leaked_) cuMemFree(p);
            if (transient_) cuMemFree(transient_);
            if (d_input_) cuMemFree(d_input_);
            if (d_output_) cuMemFree(d_output_);
            if (h_output_) cuMemFreeHost(h_output_);
            for (CUevent e : {ev_start_, ev_h2d_, ev_kernel_, ev_done_})
                if (e) cuEventDestroy(e);
            if (stream_) cuStreamDestroy(stream_);
            if (module_) cuModuleUnload(module_);
            cuDevicePrimaryCtxRelease(dev_);
        }
        if (nvml_init_) nvmlShutdown();
        cleaned_ = true;
        facts["busy"] = busy;
        if (busy) facts["note"] = "device work still running after 5 s; resources are released at process exit";
        return facts;
    }

   private:
    struct Staged {
        int64_t item_index = 0;
        size_t nbytes = 0;
        void* host = nullptr;
        CUdeviceptr device = 0;
    };

    std::string uuid_, residency_, instrumentation_, buffer_policy_, profiler_;
    bool nvtx_ = false, events_on_ = true;
    bool cudnn_plan_ = false, executor_ready_ = false;
    Json options_, program_;
    CudnnPlanExecutor executor_;
    std::string source_, entry_, artifact_sha256_, cubin_sha256_;
    unsigned block_threads_ = 1;
    size_t input_bytes_ = 0, output_bytes_ = 0, task_output_bytes_ = 0;
    std::vector<size_t> scratch_bytes_;
    int64_t timing_offset_ = -1;

    CUdevice dev_ = 0;
    CUcontext ctx_ = nullptr;
    bool ctx_retained_ = false, prepared_ = false, cleaned_ = false, nvml_init_ = false;
    CUmodule module_ = nullptr;
    CUfunction fn_ = nullptr;
    CUstream stream_ = nullptr;
    CUevent ev_start_ = nullptr, ev_h2d_ = nullptr, ev_kernel_ = nullptr, ev_done_ = nullptr;
    CUdeviceptr d_input_ = 0, d_output_ = 0, transient_ = 0;
    std::vector<CUdeviceptr> d_scratch_, leaked_, arg_ptrs_;
    std::vector<void*> args_;
    void* h_output_ = nullptr;
    std::vector<Staged> staged_;
    int64_t device_allocs_ = 0, pinned_allocs_ = 0;
    int64_t h2d_bytes_ = 0, current_rid_ = 0, since_us_ = 0;
    nvmlDevice_t nvml_ = nullptr;
    nvtxDomainHandle_t domain_ = nullptr;
    nvtxStringHandle_t msg_submit_ = nullptr, msg_complete_ = nullptr, msg_drop_ = nullptr, msg_not_issued_ = nullptr;

    CUdeviceptr device_alloc(size_t n, bool persistent = true) {
        CUdeviceptr p = 0;
        cu(cuMemAlloc(&p, n), "cuMemAlloc");
        ++device_allocs_;
        (void)persistent;
        return p;
    }

    void* pinned_alloc(size_t n) {
        void* p = nullptr;
        cu(cuMemAllocHost(&p, n), "cuMemAllocHost");
        ++pinned_allocs_;
        return p;
    }

    int64_t elapsed(CUevent a, CUevent b) {
        float ms = 0;
        cu(cuEventElapsedTime(&ms, a, b), "cuEventElapsedTime");
        return std::llround(double(ms) * 1e6);
    }

    nvtxEventAttributes_t attr(nvtxStringHandle_t msg, int64_t id) const {
        nvtxEventAttributes_t a{};
        a.version = NVTX_VERSION;
        a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        a.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
        a.message.registered = msg;
        a.payloadType = NVTX_PAYLOAD_TYPE_INT64;
        a.payload.llValue = id;
        return a;
    }

    void compile(const std::string& arch, std::string& cubin, std::string& log) {
        nvrtcProgram prog;
        if (nvrtcCreateProgram(&prog, source_.c_str(), "sdv_bmt_device_program.cu", 0, nullptr, nullptr) !=
            NVRTC_SUCCESS)
            throw JsonError("nvrtcCreateProgram failed");
        const char* opts[] = {arch.c_str(), "--std=c++17"};
        nvrtcResult r = nvrtcCompileProgram(prog, 2, opts);
        size_t n = 0;
        nvrtcGetProgramLogSize(prog, &n);
        log.assign(n, '\0');
        if (n) nvrtcGetProgramLog(prog, log.data());
        while (!log.empty() && (log.back() == '\0' || log.back() == '\n')) log.pop_back();
        if (r != NVRTC_SUCCESS) {
            nvrtcDestroyProgram(&prog);
            throw JsonError("nvrtc compile failed: " + log);
        }
        nvrtcGetCUBINSize(prog, &n);
        cubin.assign(n, '\0');
        nvrtcGetCUBIN(prog, cubin.data());
        nvrtcDestroyProgram(&prog);
    }

    static Json profiler_environment() {
        Json j = Json::object();
        bool injected = false;
        for (const char* key : {"CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH", "NSYS_PROFILING_SESSION_ID"}) {
            const char* v = std::getenv(key);
            if (v) {
                injected = true;
                j[key] = std::string(v);
            }
        }
        j["injected"] = injected;
        return j;
    }

    static Json reasons(unsigned long long bits) {
        static const struct {
            unsigned long long bit;
            const char* name;
        } table[] = {{0x1, "gpu_idle"},          {0x2, "applications_clocks_setting"}, {0x4, "sw_power_cap"},
                     {0x8, "hw_slowdown"},       {0x10, "sync_boost"},                 {0x20, "sw_thermal_slowdown"},
                     {0x40, "hw_thermal_slowdown"}, {0x80, "hw_power_brake_slowdown"}, {0x100, "display_clock_setting"}};
        Json out = Json::array();
        for (const auto& e : table)
            if (bits & e.bit) out.push(e.name);
        return out;
    }

    Json foreign_processes() {
        Json pids = Json::array();
        unsigned count = 0;
        nvmlReturn_t r = nvmlDeviceGetComputeRunningProcesses_v3(nvml_, &count, nullptr);
        if (r == NVML_SUCCESS && count == 0) return pids;
        if (r != NVML_SUCCESS && r != NVML_ERROR_INSUFFICIENT_SIZE) throw JsonError("nvmlDeviceGetComputeRunningProcesses failed");
        std::vector<nvmlProcessInfo_t> infos(count + 8);
        count = unsigned(infos.size());
        nv(nvmlDeviceGetComputeRunningProcesses_v3(nvml_, &count, infos.data()), "nvmlDeviceGetComputeRunningProcesses_v3");
        std::vector<int64_t> list;
        for (unsigned k = 0; k < count; ++k)
            if (int64_t(infos[k].pid) != int64_t(getpid())) list.push_back(infos[k].pid);
        std::sort(list.begin(), list.end());
        for (int64_t p : list) pids.push(p);
        return pids;
    }

    Json fields() {
        static const struct {
            unsigned id;
            const char* name;
        } table[] = {{NVML_FI_DEV_CLOCKS_EVENT_REASON_SW_POWER_CAP, "throttle_ns_sw_power_cap"},
                     {NVML_FI_DEV_CLOCKS_EVENT_REASON_SYNC_BOOST, "throttle_ns_sync_boost"},
                     {NVML_FI_DEV_CLOCKS_EVENT_REASON_SW_THERM_SLOWDOWN, "throttle_ns_sw_thermal_slowdown"},
                     {NVML_FI_DEV_CLOCKS_EVENT_REASON_HW_THERM_SLOWDOWN, "throttle_ns_hw_thermal_slowdown"},
                     {NVML_FI_DEV_PCIE_COUNT_TX_BYTES, "pcie_tx_bytes"},
                     {NVML_FI_DEV_PCIE_COUNT_RX_BYTES, "pcie_rx_bytes"},
                     {NVML_FI_DEV_PCIE_REPLAY_COUNTER, "pcie_replay_counter"}};
        constexpr int n = int(sizeof table / sizeof table[0]);
        nvmlFieldValue_t v[n];
        std::memset(v, 0, sizeof v);
        for (int k = 0; k < n; ++k) v[k].fieldId = table[k].id;
        Json out = Json::object();
        bool ok = nvmlDeviceGetFieldValues(nvml_, n, v) == NVML_SUCCESS;
        for (int k = 0; k < n; ++k) {
            if (!ok || v[k].nvmlReturn != NVML_SUCCESS) {
                out[table[k].name] = nullptr;
                continue;
            }
            unsigned long long x = 0;
            switch (v[k].valueType) {
                case NVML_VALUE_TYPE_UNSIGNED_INT: x = v[k].value.uiVal; break;
                case NVML_VALUE_TYPE_UNSIGNED_LONG: x = v[k].value.ulVal; break;
                case NVML_VALUE_TYPE_UNSIGNED_LONG_LONG: x = v[k].value.ullVal; break;
                case NVML_VALUE_TYPE_SIGNED_LONG_LONG: x = static_cast<unsigned long long>(v[k].value.sllVal); break;
                case NVML_VALUE_TYPE_SIGNED_INT: x = static_cast<unsigned long long>(v[k].value.siVal); break;
                default: x = v[k].value.ullVal;
            }
            out[table[k].name] = x;
        }
        return out;
    }

    Json foreign_activity(int64_t since_us) {
        Json j = Json::object();
        unsigned count = 0;
        nvmlReturn_t r = nvmlDeviceGetProcessUtilization(nvml_, nullptr, &count, (unsigned long long)since_us);
        if (r == NVML_ERROR_NOT_FOUND) {
            j["status"] = "no_samples";
            j["foreign_nonzero_samples"] = 0;
            j["own_samples"] = 0;
            return j;
        }
        if (r != NVML_SUCCESS && r != NVML_ERROR_INSUFFICIENT_SIZE) {
            j["status"] = "error:" + std::to_string(int(r));
            return j;
        }
        std::vector<nvmlProcessUtilizationSample_t> rows(count ? count : 1);
        count = unsigned(rows.size());
        r = nvmlDeviceGetProcessUtilization(nvml_, rows.data(), &count, (unsigned long long)since_us);
        if (r == NVML_ERROR_NOT_FOUND) {
            j["status"] = "no_samples";
            j["foreign_nonzero_samples"] = 0;
            j["own_samples"] = 0;
            return j;
        }
        if (r != NVML_SUCCESS) {
            j["status"] = "error:" + std::to_string(int(r));
            return j;
        }
        int64_t own = getpid(), foreign_nonzero = 0, own_samples = 0;
        std::vector<int64_t> foreign, active;
        for (unsigned k = 0; k < count; ++k) {
            const auto& s = rows[k];
            if (int64_t(s.pid) == own) {
                ++own_samples;
                continue;
            }
            foreign.push_back(s.pid);
            if (s.smUtil || s.memUtil) {
                ++foreign_nonzero;
                active.push_back(s.pid);
            }
        }
        auto uniq = [](std::vector<int64_t> v) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            Json a = Json::array();
            for (int64_t x : v) a.push(x);
            return a;
        };
        j["status"] = "ok";
        j["foreign_nonzero_samples"] = foreign_nonzero;
        j["foreign_pids"] = uniq(foreign);
        j["foreign_active_pids"] = uniq(active);
        j["own_samples"] = own_samples;
        return j;
    }
};

}  // namespace sdv
