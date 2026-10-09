// SDV-BMT native runner: selection of the workload's native output pipeline (postprocess + codec) by the
// run specification's output_pipeline.kind.  Each pipeline is the native twin of a Python Workload's
// postprocess/encode; the Python implementation owns the semantics.
//
//   reference_convnet_v1   reference conformance workload (reference_output.hpp)
//   yolox_coco_v1          YOLOX COCO detection workload, Stage 2b-G real-model baseline (yolox_output.hpp);
//                          needs a per-item output context (image id and original size) from the stage command
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "json.hpp"
#include "loop.hpp"
#include "reference_output.hpp"
#include "yolox_output.hpp"

namespace sdv {

class OutputPipeline {
   public:
    explicit OutputPipeline(const Json& spec) : kind_(spec.at("kind").as_string()) {
        if (kind_ == "reference_convnet_v1") {
            reference_.emplace(spec.at("task").as_string());
        } else if (kind_ == "yolox_coco_v1") {
            yolox_.emplace(spec);
        } else {
            throw JsonError("unsupported output pipeline " + kind_);
        }
    }

    const std::string& kind() const { return kind_; }
    bool needs_context() const { return yolox_.has_value(); }

    void set_context(int64_t item_index, const Json& ctx) {
        if (!yolox_) throw JsonError("output pipeline " + kind_ + " takes no per-item context");
        yolox_->set_context(item_index, ctx);
    }

    // Upper bound of one encoded task output, used to reserve the output buffers before MEASURE.
    size_t max_blob_bytes() const { return yolox_ ? YoloxCocoOutput::kMaxBlobBytes : size_t(512); }

    bool encode(int64_t request_id, int64_t item_index, std::string_view raw, std::string& blob, std::string& phase,
                ErrorInfo& err) const {
        if (yolox_) return yolox_->encode(request_id, item_index, raw, blob, phase, err);
        return reference_->encode(request_id, item_index, raw, blob, phase, err);
    }

   private:
    std::string kind_;
    std::optional<ReferenceConvNetOutput> reference_;
    std::optional<YoloxCocoOutput> yolox_;
};

}  // namespace sdv
