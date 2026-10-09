// SDV-BMT native runner: native implementation of the reference conformance workload's output pipeline,
// postprocess sdv-ref-logits-top1/1 + codec canonical-json-utf8/1.
//
// The Python ReferenceConvNetWorkload (src/sdv_bmt/workload/reference.py) is the reference implementation
// and owns the semantics.  This twin exists so that t_output_ready keeps its Stage 1A meaning
// (postprocess + encode + record) inside the native measurement plane.  Byte equality with the Python
// implementation is checked for every item by tests and, in every run, by the golden output identity.
//
// raw output: 10 little-endian int64 logits (80 bytes)
// task output: {"item_index", "logits", "schema", "task", "top1"} as canonical JSON (sorted keys, compact)
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

#include "json.hpp"
#include "loop.hpp"

namespace sdv {

class ReferenceConvNetOutput {
   public:
    static constexpr int kClasses = 10;
    static constexpr const char* kSchema = "sdv-ref-convnet-output/1";

    explicit ReferenceConvNetOutput(std::string task) : task_(std::move(task)) {
        static_assert(sizeof(int64_t) == 8, "int64");
        const uint16_t probe = 1;
        if (*reinterpret_cast<const unsigned char*>(&probe) != 1) throw JsonError("little-endian host required");
    }

    bool encode(int64_t /*request_id*/, int64_t item_index, std::string_view raw, std::string& blob,
                std::string& phase, ErrorInfo& err) const {
        if (raw.size() != size_t(kClasses) * 8) {
            phase = "postprocess";
            err = ErrorInfo{"WorkloadError", "raw output must be 80 bytes (int64 logits)"};
            return false;
        }
        int64_t logits[kClasses];
        std::memcpy(logits, raw.data(), sizeof logits);
        int top1 = 0;
        for (int k = 1; k < kClasses; ++k) {
            if (logits[k] > logits[top1]) top1 = k;  // first index of the maximum, as list.index(max(...))
        }
        Json out = Json::object();
        out["schema"] = kSchema;
        out["task"] = task_;
        out["item_index"] = item_index;
        Json values = Json::array();
        for (int64_t v : logits) values.push(Json(static_cast<long long>(v)));
        out["logits"] = values;
        out["top1"] = top1;
        out.write(blob);
        return true;
    }

   private:
    std::string task_;
};

}  // namespace sdv
