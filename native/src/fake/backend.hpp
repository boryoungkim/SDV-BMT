// SDV-BMT native runner: MOCK backend without any device (adapter fake_native).
//
// Used for the harness floor and for conformance runs on hosts without a GPU.  submit records the request;
// the request is reported complete by its polls_until_complete-th poll (per-item override in "items"), so
// its "service time" is the harness's own polling cost.  The raw output is 10 little-endian int64 logits
// with a 1 at position (item_index % 10), consumed by the reference workload's output pipeline.  Injected
// faults: prepare_error_items fail staging, submit_error_items fail submit, run_error_items complete with a
// DUT error.  Nothing here is a statement about real hardware.
#pragma once

#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../common/json.hpp"
#include "../common/loop.hpp"
#include "../common/sha256.hpp"

namespace sdv {

class FakeBackend {
   public:
    static constexpr const char* kName = "fake_native";

    explicit FakeBackend(const Json& spec) {
        const Json& opt = spec.at("backend").at("options");
        if (opt.has("polls_until_complete")) polls_default_ = int(opt.at("polls_until_complete").as_int());
        if (polls_default_ < 1) throw JsonError("polls_until_complete must be >= 1");
        auto ints = [&](const char* key, std::set<int64_t>& out) {
            if (opt.has(key))
                for (const Json& v : opt.at(key).as_array()) out.insert(v.as_int());
        };
        ints("prepare_error_items", prepare_errors_);
        ints("submit_error_items", submit_errors_);
        ints("run_error_items", run_errors_);
        if (opt.has("items")) {
            for (const Json& e : opt.at("items").as_array()) polls_[e.at("item_index").as_int()] = int(e.at("polls").as_int());
        }
    }

    static Json describe(const Json&) {
        Json d = Json::object();
        d["adapter"] = kName;
        d["mock"] = true;
        return d;
    }

    Json prepare(const Json&) {
        Json r = Json::object();
        r["mock"] = true;
        return r;
    }

    bool stage(int slot, int64_t item_index, const char* data, size_t n, const std::string& sha256,
               ErrorInfo& err) {
        if (prepare_errors_.count(item_index)) {
            err = ErrorInfo{"FakeDUTInjectedError", "injected prepare_input error for item " + std::to_string(item_index)};
            return false;
        }
        std::string copy(data, n);
        if (Sha256::of(copy) != sha256) {
            err = ErrorInfo{"StagingDigestMismatch", "staged input does not match its SHA-256"};
            return false;
        }
        if (slot >= int(item_of_slot_.size())) item_of_slot_.resize(size_t(slot) + 1, -1);
        item_of_slot_[size_t(slot)] = item_index;
        inputs_ += 1;
        return true;
    }

    bool submit(int64_t /*rid*/, int slot, ErrorInfo& err) {
        int64_t item = item_of_slot_.at(size_t(slot));
        if (submit_errors_.count(item)) {
            err = ErrorInfo{"FakeDUTInjectedError", "injected submit error for item " + std::to_string(item)};
            return false;
        }
        auto it = polls_.find(item);
        polls_left_ = it == polls_.end() ? polls_default_ : it->second;
        current_item_ = item;
        return true;
    }

    int poll(ErrorInfo& err) {
        if (--polls_left_ > 0) return 0;
        if (run_errors_.count(current_item_)) {
            err = ErrorInfo{"fake_injected_run_error", "injected run error for item " + std::to_string(current_item_)};
            return -1;
        }
        return 1;
    }

    bool collect(DeviceFacts&, std::string_view& raw, ErrorInfo&) {
        int64_t logits[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        logits[current_item_ % 10] = 1;
        std::memcpy(raw_, logits, sizeof logits);
        raw = std::string_view(raw_, sizeof logits);
        return true;
    }

    void mark_admission(int64_t, const char*) {}

    Json observe(const std::string& point) {
        Json s = Json::object();
        s["adapter"] = kName;
        s["point"] = point;
        s["mock"] = true;
        return s;
    }

    Json cleanup() {
        Json r = Json::object();
        r["mock"] = true;
        return r;
    }

   private:
    int polls_default_ = 1;
    std::map<int64_t, int> polls_;
    std::set<int64_t> prepare_errors_, submit_errors_, run_errors_;
    std::vector<int64_t> item_of_slot_;
    int polls_left_ = 0;
    int64_t current_item_ = 0;
    char raw_[80] = {};
    int64_t inputs_ = 0;
};

}  // namespace sdv
