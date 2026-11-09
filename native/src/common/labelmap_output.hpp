// SDV-BMT native runner: native implementation of the DeepLabV3+ Cityscapes segmentation workload's output
// pipeline, postprocess labelmap-u8-check/1 + codec labelmap-u8-rowrle/1 (v0.1.2 DeepLab part).
//
// The Python twin (src/sdv_bmt/workload/deeplab.py, src/sdv_bmt/contracts/labelmap.py) owns the semantics; this
// twin keeps t_output_ready = postprocess + encode + record inside the native measured path.  Byte equality with
// the Python implementation is checked by tests (synthetic label maps, including the worst case) and on real
// device outputs.
//
// raw output: width x height uint8 class indices (row-major), the device-side argmax of the model.
// postprocess: every value must be < num_classes, otherwise the request fails with a WorkloadError.
// codec: header "SDVLMRL1", item index u32, width u16, height u16 (little-endian), then per row either
//        0x01, runs u16, runs x (value u8, length u16) when 3 + 3*runs < 1 + width, else 0x00 and the raw row.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "json.hpp"
#include "loop.hpp"

namespace sdv {

class LabelmapRleOutput {
   public:
    static constexpr const char* kCodec = "labelmap-u8-rowrle/1";
    static constexpr const char* kPostprocess = "labelmap-u8-check/1";
    static constexpr size_t kHeaderBytes = 16;

    explicit LabelmapRleOutput(const Json& spec) {
        const uint16_t probe = 1;
        if (*reinterpret_cast<const unsigned char*>(&probe) != 1) throw JsonError("little-endian host required");
        if (spec.at("codec").as_string() != kCodec || spec.at("postprocess").as_string() != kPostprocess)
            throw JsonError("label map output pipeline codec/postprocess differ from the native implementation");
        const int64_t w = spec.at("width").as_int(), h = spec.at("height").as_int(), c = spec.at("num_classes").as_int();
        if (w <= 0 || w > 0xFFFF || h <= 0 || h > 0xFFFF || c <= 0 || c > 255)
            throw JsonError("label map output pipeline needs width/height in 1..65535 and num_classes in 1..255");
        width_ = size_t(w);
        height_ = size_t(h);
        classes_ = unsigned(c);
    }

    size_t max_blob_bytes() const { return kHeaderBytes + height_ * (1 + width_); }

    bool encode(int64_t /*request_id*/, int64_t item_index, std::string_view raw, std::string& blob,
                std::string& phase, ErrorInfo& err) const {
        phase = "postprocess";
        if (raw.size() != width_ * height_) {
            err = ErrorInfo{"WorkloadError", "raw output must be " + std::to_string(width_) + "x" +
                                                 std::to_string(height_) + " uint8 (" +
                                                 std::to_string(width_ * height_) + " bytes)"};
            return false;
        }
        for (unsigned char v : raw) {
            if (v >= classes_) {
                err = ErrorInfo{"WorkloadError", "label value " + std::to_string(unsigned(v)) + " is not a class index"};
                return false;
            }
        }
        if (item_index < 0 || item_index > int64_t(0xFFFFFFFFu)) {
            err = ErrorInfo{"WorkloadError", "item index out of range for the label map header"};
            return false;
        }
        phase = "encode";
        blob += "SDVLMRL1";
        put_u32(blob, uint32_t(item_index));
        put_u16(blob, uint16_t(width_));
        put_u16(blob, uint16_t(height_));
        for (size_t y = 0; y < height_; ++y) {
            const unsigned char* row = reinterpret_cast<const unsigned char*>(raw.data()) + y * width_;
            size_t runs = 1;
            for (size_t x = 1; x < width_; ++x) runs += row[x] != row[x - 1];
            if (3 + 3 * runs < 1 + width_) {
                blob.push_back(char(1));
                put_u16(blob, uint16_t(runs));
                size_t start = 0;
                for (size_t x = 1; x <= width_; ++x) {
                    if (x == width_ || row[x] != row[start]) {
                        blob.push_back(char(row[start]));
                        put_u16(blob, uint16_t(x - start));
                        start = x;
                    }
                }
            } else {
                blob.push_back(char(0));
                blob.append(reinterpret_cast<const char*>(row), width_);
            }
        }
        return true;
    }

   private:
    size_t width_ = 0, height_ = 0;
    unsigned classes_ = 0;

    static void put_u16(std::string& s, uint16_t v) {
        s.push_back(char(v & 0xFF));
        s.push_back(char(v >> 8));
    }
    static void put_u32(std::string& s, uint32_t v) {
        for (int k = 0; k < 4; ++k) s.push_back(char((v >> (8 * k)) & 0xFF));
    }
};

}  // namespace sdv
