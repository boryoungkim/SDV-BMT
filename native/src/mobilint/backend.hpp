// SDV-BMT native runner: Mobilint ARIES NPU backend (adapter mobilint_aries, actual execution).
//
// Mobilint-specific concepts (qbruntime Accelerator/Model/ModelConfig/Future, ARIES clusters and cores, the
// /dev/ariesN device nodes) live only in this file.  Evidence keys are namespaced "mobilint.".
//
// Execution path: one ARIES chip selected by device number (/dev/ariesN); the compiled model (.mxq, SHA-256
// checked against the run specification) created with the declared core mode (single core on one pinned
// (cluster, core)) and the asynchronous pipeline enabled, then launched.  Prepared inputs (the workload's
// canonical CHW float32 BGR tensor) are converted to HWC once per item at staging, outside the measured path.
// submit = Model::inferAsync on the item's HWC tensor; poll = Future::waitFor(0); collect = Future::get (it
// hands over the nine outputs, already converted to float32 by the runtime) and the head tail the compiled model
// does not contain.
//
// Head tail: the compiled model ends at the nine head convolutions (per stride 8/16/32: box 4, objectness logit 1,
// class logits 80; HWC).
// collect() applies sigmoid to objectness and classes and writes the workload's raw contract (anchors x 85
// float32, anchor-major, strides in order 8, 16, 32) into a buffer allocated once in prepare.  The nine
// outputs are matched by (grid, channels), not by position: the runtime returns them in its own order.
//
// Measurement boundary (docs/measurement_protocol.md): DUT-service ends at the poll that observes
// waitFor(0) == true.  The runtime quantizes the input and dequantizes the outputs to float32 inside the request,
// before the Future reports completion, so both are inside DUT-service (checked on the device: requests with int8
// and with float32 outputs differ in the time to completion, not in the duration of Future::get).  Future::get
// and the head tail run in collect(), i.e. in the output stage, and their durations are recorded as observations.
// No per-inference device timing is exposed by qbruntime: the device segments are not observable (DeviceFacts
// carries no segment).
#pragma once

#include <dirent.h>
#include <dlfcn.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <qbruntime/qbruntime.h>

#include "../common/json.hpp"
#include "../common/loop.hpp"
#include "../common/sha256.hpp"

namespace sdv {

namespace mbl = mobilint;

inline int64_t mbl_mono_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

inline int64_t mbl_realtime_us() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return int64_t(ts.tv_sec) * 1000000LL + ts.tv_nsec / 1000;
}

inline std::string mbl_status(mbl::StatusCode sc) {
    return mbl::statusCodeToString(sc) + " (" + std::to_string(int(sc)) + ")";
}

inline void mbl_check(mbl::StatusCode sc, const char* api) {
    if (!sc) throw JsonError(std::string(api) + ": " + mbl_status(sc));
}

// small sysfs/proc readers: a missing file is "not observed" (null), never a guessed value
inline Json mbl_read_text(const std::string& path) {
    std::ifstream f(path);
    if (!f) return Json();
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return Json(s);
}

inline Json mbl_realpath(const std::string& path) {
    char buf[PATH_MAX];
    return ::realpath(path.c_str(), buf) ? Json(std::string(buf)) : Json();
}

// qbruntime 1.5.0 writes lifecycle log lines ("Model constructed/launched/disposed") to stdout, which carries the
// runner's JSON reply lines.  While a lifecycle call runs, file descriptor 1 points to stderr (the log lines are
// kept there); both stdio and iostream buffers are flushed before the descriptor is restored.  Requests
// (inferAsync, waitFor, get) write nothing (observed with tools/mobilint probe), so the measured path is untouched.
class StdoutToStderr {
   public:
    StdoutToStderr() {
        std::fflush(stdout);
        saved_ = ::dup(1);
        if (saved_ >= 0) ::dup2(2, 1);
    }
    ~StdoutToStderr() {
        std::fflush(stdout);
        std::cout.flush();
        if (saved_ >= 0) {
            ::dup2(saved_, 1);
            ::close(saved_);
        }
    }
    StdoutToStderr(const StdoutToStderr&) = delete;
    StdoutToStderr& operator=(const StdoutToStderr&) = delete;

   private:
    int saved_ = -1;
};

inline std::string mbl_file_sha256(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw JsonError("cannot read " + path);
    std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return Sha256::of(data);
}

inline std::string mbl_core_name(const mbl::CoreId& c) {
    std::string cl = c.cluster == mbl::Cluster::Cluster0 ? "Cluster0" : c.cluster == mbl::Cluster::Cluster1 ? "Cluster1" : "ClusterError";
    std::string co;
    switch (c.core) {
        case mbl::Core::Core0: co = "Core0"; break;
        case mbl::Core::Core1: co = "Core1"; break;
        case mbl::Core::Core2: co = "Core2"; break;
        case mbl::Core::Core3: co = "Core3"; break;
        case mbl::Core::GlobalCore: co = "GlobalCore"; break;
        default: co = "CoreError"; break;
    }
    return cl + "/" + co;
}

inline std::string mbl_core_mode_name(mbl::CoreMode m) {
    switch (m) {
        case mbl::CoreMode::Single: return "single";
        case mbl::CoreMode::Multi: return "multi";
        case mbl::CoreMode::Global4: return "global4";
        case mbl::CoreMode::Global8: return "global8";
        case mbl::CoreMode::Auto: return "auto";
        default: return "error";
    }
}

inline Json mbl_shape(const std::vector<int64_t>& s) {
    Json a = Json::array();
    for (int64_t v : s) a.push(v);
    return a;
}

class MobilintBackend {
   public:
    static constexpr const char* kName = "mobilint_aries";
    static constexpr int kBranchChannels[3] = {4, 1, 80};  // reg, obj logit, cls logits (per-level order)

    explicit MobilintBackend(const Json& spec) {
        dev_no_ = int(spec.at("device_number").as_int());
        const Json& opt = spec.at("backend").at("options");
        core_mode_ = opt.has("core_mode") ? opt.at("core_mode").as_string() : std::string("single");
        if (core_mode_ != "single") throw JsonError("core_mode: single only (one pinned cluster/core)");
        cluster_ = opt.has("cluster") ? int(opt.at("cluster").as_int()) : 0;
        core_ = opt.has("core") ? int(opt.at("core").as_int()) : 0;
        if (cluster_ < 0 || cluster_ > 1 || core_ < 0 || core_ > 3) throw JsonError("cluster must be 0..1, core 0..3");
        const Json& p = spec.at("program");
        mxq_path_ = p.at("mxq_path").as_string();
        mxq_sha256_ = p.at("mxq_sha256").as_string();
        size_ = int(p.at("input_size").as_int());
        anchors_ = int(p.at("num_anchors").as_int());
        input_bytes_ = size_t(p.at("input_bytes").as_int());
        task_output_bytes_ = size_t(p.at("task_output_bytes").as_int());
        if (size_ <= 0 || size_ % 32) throw JsonError("input_size must be a positive multiple of 32");
        int a = 0;
        for (int s : {8, 16, 32}) a += (size_ / s) * (size_ / s);
        if (a != anchors_) throw JsonError("num_anchors does not match input_size");
        if (input_bytes_ != size_t(3) * size_t(size_) * size_t(size_) * 4)
            throw JsonError("input_bytes must be 3 x S x S float32 (CHW)");
        if (task_output_bytes_ != size_t(anchors_) * 85 * 4) throw JsonError("task_output_bytes must be anchors x 85 float32");
    }

    // ---------------------------------------------------------------- identity (no model is created)
    static Json describe(const Json& args) {
        const int dev = int(args.at("device_number").as_int());
        const std::string node = "aries" + std::to_string(dev);
        Dl_info info{};
        std::string lib = dladdr(reinterpret_cast<void*>(&mbl::getQbRuntimeVersion), &info) && info.dli_fname
                              ? info.dli_fname : "unknown";
        Json id = Json::object();
        id["role"] = "npu_target";
        id["accelerator_family"] = "NPU";
        id["vendor"] = "Mobilint";
        id["mobilint.device_node"] = "/dev/" + node;
        id["mobilint.device_node_exists"] = ::access(("/dev/" + node).c_str(), F_OK) == 0;
        id["mobilint.device_number"] = int64_t(dev);
        Json names = Json::array();
        for (const std::string& n : mbl::getAvailableDevices()) names.push(n);
        id["mobilint.available_device_names"] = names;
        Json nums = Json::array();
        for (int n : mbl::getAvailableDeviceNumbers()) nums.push(int64_t(n));
        id["mobilint.available_device_numbers"] = nums;
        id["mobilint.qbruntime_version"] = mbl::getQbRuntimeVersion();
        id["mobilint.qbruntime_git_version"] = mbl::getQbRuntimeGitVersion();
        id["mobilint.qbruntime_library"] = lib;
        id["mobilint.qbruntime_library_realpath"] = mbl_realpath(lib);
        // sysfs facts of the device node, when the driver exposes them (null = not observed, never assumed)
        id["mobilint.sysfs_device"] = mbl_realpath("/sys/class/" + node + "/device");
        id["mobilint.pci_vendor"] = mbl_read_text("/sys/class/" + node + "/device/vendor");
        id["mobilint.pci_device"] = mbl_read_text("/sys/class/" + node + "/device/device");
        id["mobilint.driver_module_version"] = mbl_read_text("/sys/module/aries/version");
        return id;
    }

    // ---------------------------------------------------------------- DUT.prepare
    Json prepare(const Json&) {
        Json facts = Json::object();
        const std::string sha = mbl_file_sha256(mxq_path_);
        if (sha != mxq_sha256_) throw JsonError("compiled model SHA-256 " + sha + " differs from the specification");
        {
            StdoutToStderr quiet;  // qbruntime lifecycle log lines go to stderr, not into the reply channel
            mbl::StatusCode sc;
            acc_ = mbl::Accelerator::create(dev_no_, sc);
            mbl_check(sc, "Accelerator::create");
            if (!acc_) throw JsonError("Accelerator::create returned no accelerator");
            mbl::ModelConfig mc;
            mbl::CoreId cid{cluster_ == 0 ? mbl::Cluster::Cluster0 : mbl::Cluster::Cluster1,
                            static_cast<mbl::Core>(core_ + 1)};
            if (!mc.setSingleCoreMode(std::vector<mbl::CoreId>{cid}))
                throw JsonError("ModelConfig::setSingleCoreMode refused");
            mc.setAsyncPipelineEnabled(true);
            model_ = mbl::Model::create(mxq_path_, mc, sc);
            mbl_check(sc, "Model::create");
            if (!model_) throw JsonError("Model::create returned no model");
            mbl_check(model_->launch(*acc_), "Model::launch");
            launched_ = true;
        }

        // the model must be the cut YOLOX head: one HWC input S x S x 3, nine outputs (grid, channels)
        const auto& in = model_->getModelInputShape();
        if (in.size() != 1 || numel(in[0]) != int64_t(size_) * size_ * 3 || in[0].back() != 3)
            throw JsonError("model input is not one HWC S x S x 3 tensor");
        const auto& outs = model_->getModelOutputShape();
        if (outs.size() != 9) throw JsonError("model has " + std::to_string(outs.size()) + " outputs, expected the nine head tensors");
        const int grids[3] = {size_ / 8, size_ / 16, size_ / 32};
        int base = 0;
        for (int lv = 0; lv < 3; ++lv) {
            level_base_[lv] = base;
            base += grids[lv] * grids[lv];
        }
        bool seen[3][3] = {};
        out_map_.assign(outs.size(), OutMap{});
        for (size_t k = 0; k < outs.size(); ++k) {
            const auto& s = outs[k];
            if (s.size() < 3) throw JsonError("output " + std::to_string(k) + " is not HWC");
            int64_t h = s[s.size() - 3], w = s[s.size() - 2], c = s.back();
            bool found = false;
            for (int lv = 0; lv < 3 && !found; ++lv)
                for (int br = 0; br < 3 && !found; ++br)
                    if (h == grids[lv] && w == grids[lv] && c == kBranchChannels[br] && !seen[lv][br]) {
                        seen[lv][br] = found = true;
                        out_map_[k] = OutMap{lv, br, grids[lv], int(c)};
                    }
            if (!found) throw JsonError("output " + std::to_string(k) + " shape does not match a YOLOX head tensor");
        }
        raw_.assign(size_t(anchors_) * 85, 0.0f);  // the raw contract buffer, allocated once

        facts["mxq_sha256"] = sha;
        facts["core_mode"] = mbl_core_mode_name(model_->getCoreMode());
        Json cores = Json::array();
        for (const mbl::CoreId& c : model_->getTargetCores()) cores.push(mbl_core_name(c));
        facts["target_cores"] = cores;
        if (cores.as_array().size() != 1) throw JsonError("expected exactly one target core");
        facts["async_pipeline"] = true;
        facts["input_shape"] = mbl_shape(in[0]);
        Json os = Json::array();
        for (const auto& s : outs) os.push(mbl_shape(s));
        facts["output_shapes"] = os;
        Json om = Json::array();
        static const char* branch[3] = {"reg", "obj", "cls"};
        for (const OutMap& m : out_map_) {
            Json e = Json::object();
            e["level"] = int64_t(m.level);
            e["branch"] = branch[m.branch];
            om.push(e);
        }
        facts["output_mapping"] = om;
        facts["model_input_dtype"] = int64_t(model_->getModelInputDataType());
        facts["model_output_dtype"] = int64_t(model_->getModelOutputDataType());
        Json dn = Json::array();
        for (const std::string& n : model_->getDeviceNames()) dn.push(n);
        facts["model_device_names"] = dn;
        facts["accelerator_device_name"] = acc_->getDeviceName();
        facts["qbruntime_version"] = mbl::getQbRuntimeVersion();
        facts["host_tail"] = "sigmoid(obj), sigmoid(cls), concat [reg, obj, cls] per anchor, strides 8/16/32";
        return facts;
    }

    // ---------------------------------------------------------------- PREPARE_INPUTS (staging)
    bool stage(int slot, int64_t item_index, const char* data, size_t n, const std::string& sha256, ErrorInfo& err) {
        try {
            if (n != input_bytes_) throw JsonError("prepared input size differs from 3 x S x S float32");
            if (Sha256::of(data, n) != sha256) throw JsonError("staged input does not match its SHA-256");
            const size_t plane = size_t(size_) * size_t(size_);
            Staged s;
            s.item_index = item_index;
            s.hwc.resize(plane * 3);
            const float* chw = reinterpret_cast<const float*>(data);
            std::vector<float> tmp;
            if (reinterpret_cast<uintptr_t>(data) % alignof(float)) {  // staging buffer alignment is not assumed
                tmp.resize(plane * 3);
                std::memcpy(tmp.data(), data, n);
                chw = tmp.data();
            }
            for (size_t i = 0; i < plane; ++i)
                for (size_t c = 0; c < 3; ++c) s.hwc[i * 3 + c] = chw[c * plane + i];
            if (slot >= int(staged_.size())) staged_.resize(size_t(slot) + 1);
            staged_[size_t(slot)] = std::move(s);
            // the NDArray is a non-owning view of the slot's hwc vector, built after the vector reached its slot
            Staged& t = staged_[size_t(slot)];
            t.input.emplace_back(t.hwc.data(), std::vector<int64_t>{size_, size_, 3});
            return true;
        } catch (const std::exception& e) {
            err = ErrorInfo{"StagingError", e.what()};
        }
        return false;
    }

    // ---------------------------------------------------------------- requests
    bool submit(int64_t rid, int slot, ErrorInfo& err) {
        mbl::StatusCode sc;
        future_ = model_->inferAsync(staged_[size_t(slot)].input, sc);
        if (!sc) {
            err = ErrorInfo{"Model::inferAsync", mbl_status(sc)};
            return false;
        }
        pending_ = true;
        current_rid_ = rid;
        return true;
    }

    int poll(ErrorInfo&) {
        // waitFor(0) is the documented non-blocking completion check; errors surface in get() (collect)
        return future_.waitFor(0) ? 1 : 0;
    }

    bool collect(DeviceFacts& f, std::string_view& raw, ErrorInfo& err) {
        int64_t t0 = mbl_mono_ns();
        mbl::StatusCode sc;
        std::vector<mbl::NDArray<float>> outs = future_.get(sc);  // float32, dequantized before completion
        pending_ = false;
        int64_t t1 = mbl_mono_ns();
        if (!sc) {
            err = ErrorInfo{"Future::get", mbl_status(sc)};
            return false;
        }
        if (outs.size() != out_map_.size()) {
            err = ErrorInfo{"OutputMismatch", "Future::get returned " + std::to_string(outs.size()) + " outputs"};
            return false;
        }
        const float* src[3][3] = {};
        for (size_t k = 0; k < outs.size(); ++k) {
            const OutMap& m = out_map_[k];
            if (outs[k].size() != int64_t(m.grid) * m.grid * m.channels) {
                err = ErrorInfo{"OutputMismatch", "output " + std::to_string(k) + " has an unexpected size"};
                return false;
            }
            src[m.level][m.branch] = outs[k].data();
        }
        float* dst = raw_.data();
        for (int lv = 0; lv < 3; ++lv) {
            const int g = size_ >> (3 + lv);
            const float* reg = src[lv][0];
            const float* obj = src[lv][1];
            const float* cls = src[lv][2];
            float* row = dst + size_t(level_base_[lv]) * 85;
            for (int i = 0, cells = g * g; i < cells; ++i, row += 85) {
                std::memcpy(row, reg + size_t(i) * 4, 4 * sizeof(float));
                row[4] = sigmoid(obj[i]);
                const float* c = cls + size_t(i) * 80;
                for (int j = 0; j < 80; ++j) row[5 + j] = sigmoid(c[j]);
            }
        }
        int64_t t2 = mbl_mono_ns();
        f.observe("mobilint.future_get_ns", t1 - t0);
        f.observe("mobilint.host_tail_ns", t2 - t1);
        raw = std::string_view(reinterpret_cast<const char*>(raw_.data()), task_output_bytes_);  // until next collect
        return true;
    }

    void mark_admission(int64_t, const char*) {}

    // ---------------------------------------------------------------- backend snapshot (outside the window)
    Json observe(const std::string& point) {
        int64_t t_begin = mbl_mono_ns();
        Json s = Json::object();
        s["point"] = point;
        s["t_monotonic_ns"] = t_begin;
        s["t_realtime_us"] = mbl_realtime_us();
        // A request is in flight when it was submitted and its Future does not report completion yet.  pending_
        // alone is not enough: the untimed precondition requests are polled to completion but never collected, so
        // pending_ stays true after them.  waitFor(0) is the non-blocking completion check (safe to call repeatedly).
        s["request_pending"] = pending_ && !future_.waitFor(0);
        s["foreign_device_openers"] = foreign_openers();
        s["duration_ns"] = mbl_mono_ns() - t_begin;
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
        if (pending_) {  // a request still in flight: wait up to 5 s, then collect it so the runtime is idle
            busy = !future_.waitFor(5000);
            if (!busy) {
                mbl::StatusCode sc;
                (void)future_.get(sc);
                pending_ = false;
            }
        }
        if (!busy) {
            StdoutToStderr quiet;  // "Model disposed." and any destructor log go to stderr
            future_ = mbl::Future<float>();  // release the last request's future before the model and the accelerator
            if (launched_) {
                mbl::StatusCode sc = model_->dispose();
                facts["dispose"] = mbl_status(sc);
                launched_ = false;
            }
            model_.reset();
            acc_.reset();
            staged_.clear();  // only when idle: the runtime may still read the inputs of a running request
        }
        cleaned_ = true;
        facts["busy"] = busy;
        if (busy) facts["note"] = "an inference was still running after 5 s; resources are released at process exit";
        return facts;
    }

   private:
    struct Staged {
        int64_t item_index = 0;
        std::vector<float> hwc;
        std::vector<mbl::NDArray<float>> input;  // view of hwc, built once (submit allocates nothing for it)
    };
    struct OutMap {
        int level = 0, branch = 0, grid = 0, channels = 0;
    };

    static float sigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }

    static int64_t numel(const std::vector<int64_t>& s) {
        int64_t n = 1;
        for (int64_t v : s) n *= v;
        return n;
    }

    // other processes holding this device node open (scan of /proc/*/fd; outside the measured window)
    Json foreign_openers() const {
        const std::string node = "/dev/aries" + std::to_string(dev_no_);
        Json pids = Json::array();
        DIR* proc = ::opendir("/proc");
        if (!proc) return Json();
        const pid_t self = ::getpid();
        while (dirent* e = ::readdir(proc)) {
            char* end = nullptr;
            long pid = std::strtol(e->d_name, &end, 10);
            if (!end || *end || pid <= 0 || pid == self) continue;
            const std::string fd_dir = std::string("/proc/") + e->d_name + "/fd";
            DIR* fds = ::opendir(fd_dir.c_str());
            if (!fds) continue;  // not ours to read: unknown, not counted
            bool hit = false;
            while (dirent* fe = ::readdir(fds)) {
                char target[PATH_MAX];
                ssize_t n = ::readlink((fd_dir + "/" + fe->d_name).c_str(), target, sizeof target - 1);
                if (n > 0) {
                    target[n] = 0;
                    if (node == target) hit = true;
                }
            }
            ::closedir(fds);
            if (hit) pids.push(int64_t(pid));
        }
        ::closedir(proc);
        return pids;
    }

    int dev_no_ = 0;
    std::string core_mode_;
    int cluster_ = 0, core_ = 0;
    std::string mxq_path_, mxq_sha256_;
    int size_ = 0, anchors_ = 0;
    size_t input_bytes_ = 0, task_output_bytes_ = 0;

    std::unique_ptr<mbl::Accelerator> acc_;
    std::unique_ptr<mbl::Model> model_;
    bool launched_ = false, cleaned_ = false, pending_ = false;
    mbl::Future<float> future_;
    int64_t current_rid_ = 0;
    std::vector<Staged> staged_;
    std::vector<OutMap> out_map_;
    int level_base_[3] = {0, 0, 0};
    std::vector<float> raw_;
};

}  // namespace sdv
