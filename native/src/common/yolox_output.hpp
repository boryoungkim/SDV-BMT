// SDV-BMT native runner: native implementation of the YOLOX COCO detection workload's output pipeline,
// postprocess yolox-decode-nms-coco/1 + codec canonical-json-utf8/1 (Stage 2b-G real-model baseline).
//
// The Python YoloxCocoWorkload (src/sdv_bmt/workload/yolox.py) is the reference implementation and owns the
// semantics; this twin keeps t_output_ready = postprocess + encode + record inside the native measured path.
// Byte equality with the Python implementation is checked by tests (synthetic raw outputs) and by the Stage
// 2b-G analysis on real device outputs.
//
// raw output: 8400 anchors x 85 float32 (little-endian), anchor-major: [x, y, log w, log h, obj, 80 class
//             scores] in grid units, anchors ordered stride 8 (80x80), 16 (40x40), 32 (20x20), row-major.
// semantics:  scale r = min(640/height, 640/width) of the original image; an anchor is a candidate when
//             obj >= 0.01 and score = obj * max class score >= 0.01 (first index of the maximum); box center
//             = (raw + grid) * stride, size = exp(raw) * stride, corners = center -/+ size/2; class-aware
//             greedy NMS in descending score order (ties: lower anchor index first), suppressing IoU > 0.65
//             against a kept box of the same class; at most 100 detections are kept; boxes are divided by r
//             and written as [x, y, w, h] in milli-pixels and the score in parts per million, both rounded
//             as floor(v * scale + 0.5).  All arithmetic is IEEE double, in the same order as the Python twin.
// task output: {"detections":[{"bbox_mpx":[x,y,w,h],"category_id":c,"score_ppm":s},...],"image_id":i,
//              "item_index":k,"schema":"sdv-yolox-coco-detections/1"} as canonical JSON.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "json.hpp"
#include "loop.hpp"

namespace sdv {

class YoloxCocoOutput {
   public:
    static constexpr const char* kSchema = "sdv-yolox-coco-detections/1";
    static constexpr int kAnchors = 8400, kChannels = 85, kClasses = 80, kInputSize = 640, kMaxDetections = 100;
    static constexpr double kConfThreshold = 0.01, kNmsThreshold = 0.65;
    static constexpr int64_t kConfThresholdPpm = 10000, kNmsThresholdPpm = 650000;
    static constexpr size_t kMaxBlobBytes = 16384;

    explicit YoloxCocoOutput(const Json& spec) {
        const uint16_t probe = 1;
        if (*reinterpret_cast<const unsigned char*>(&probe) != 1) throw JsonError("little-endian host required");
        if (spec.at("conf_threshold_ppm").as_int() != kConfThresholdPpm ||
            spec.at("nms_iou_threshold_ppm").as_int() != kNmsThresholdPpm ||
            spec.at("max_detections").as_int() != kMaxDetections || spec.at("input_size").as_int() != kInputSize ||
            spec.at("num_classes").as_int() != kClasses)
            throw JsonError("yolox output pipeline parameters differ from the native implementation");
        cand_.reserve(kAnchors);
        kept_.reserve(kMaxDetections);
    }

    void set_context(int64_t item_index, const Json& ctx) {
        Ctx c;
        c.image_id = ctx.at("image_id").as_int();
        c.width = ctx.at("width").as_int();
        c.height = ctx.at("height").as_int();
        if (c.width <= 0 || c.height <= 0) throw JsonError("output context needs positive width/height");
        ctx_[item_index] = c;
    }

    bool encode(int64_t /*request_id*/, int64_t item_index, std::string_view raw, std::string& blob,
                std::string& phase, ErrorInfo& err) const {
        phase = "postprocess";
        if (raw.size() != size_t(kAnchors) * kChannels * 4) {
            err = ErrorInfo{"WorkloadError", "raw output must be 8400x85 float32 (2856000 bytes)"};
            return false;
        }
        auto it = ctx_.find(item_index);
        if (it == ctx_.end()) {
            err = ErrorInfo{"WorkloadError", "no output context for item " + std::to_string(item_index)};
            return false;
        }
        const Ctx& c = it->second;
        const char* base = raw.data();
        auto at = [base](size_t i) {
            float v;
            std::memcpy(&v, base + 4 * i, 4);
            return double(v);
        };
        const double r = std::min(double(kInputSize) / double(c.height), double(kInputSize) / double(c.width));
        cand_.clear();
        static const int strides[3] = {8, 16, 32};
        size_t a = 0;
        for (int stride : strides) {
            const int g = kInputSize / stride;
            for (int p = 0; p < g * g; ++p, ++a) {
                const size_t row = a * kChannels;
                const double obj = at(row + 4);
                if (!(obj >= kConfThreshold)) continue;
                double best = at(row + 5);
                int cls = 0;
                for (int j = 1; j < kClasses; ++j) {
                    const double v = at(row + 5 + size_t(j));
                    if (v > best) {
                        best = v;
                        cls = j;
                    }
                }
                const double score = obj * best;
                if (!(score >= kConfThreshold)) continue;
                const double gx = double(p % g), gy = double(p / g), s = double(stride);
                const double cx = (at(row + 0) + gx) * s, cy = (at(row + 1) + gy) * s;
                const double w = std::exp(at(row + 2)) * s, h = std::exp(at(row + 3)) * s;
                cand_.push_back(Cand{cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0, score, cls});
            }
        }
        std::stable_sort(cand_.begin(), cand_.end(), [](const Cand& x, const Cand& y) { return x.score > y.score; });
        kept_.clear();
        for (size_t i = 0; i < cand_.size() && kept_.size() < size_t(kMaxDetections); ++i) {
            const Cand& b = cand_[i];
            bool suppressed = false;
            for (size_t k : kept_) {
                const Cand& q = cand_[k];
                if (q.cls != b.cls) continue;
                const double iw = std::max(0.0, std::min(b.x2, q.x2) - std::max(b.x1, q.x1));
                const double ih = std::max(0.0, std::min(b.y2, q.y2) - std::max(b.y1, q.y1));
                const double inter = iw * ih;
                const double area_b = (b.x2 - b.x1) * (b.y2 - b.y1), area_q = (q.x2 - q.x1) * (q.y2 - q.y1);
                if (inter / (area_b + area_q - inter) > kNmsThreshold) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) kept_.push_back(i);
        }
        phase = "encode";
        blob += "{\"detections\":[";
        for (size_t n = 0; n < kept_.size(); ++n) {
            const Cand& b = cand_[kept_[n]];
            const double x1 = b.x1 / r, y1 = b.y1 / r, x2 = b.x2 / r, y2 = b.y2 / r;
            int64_t q[5];
            if (!quantize(x1, 1000.0, q[0]) || !quantize(y1, 1000.0, q[1]) || !quantize(x2 - x1, 1000.0, q[2]) ||
                !quantize(y2 - y1, 1000.0, q[3]) || !quantize(b.score, 1000000.0, q[4])) {
                phase = "postprocess";
                err = ErrorInfo{"WorkloadError", "detection value is not finite or out of range"};
                return false;
            }
            if (n) blob.push_back(',');
            blob += "{\"bbox_mpx\":[";
            for (int k = 0; k < 4; ++k) {
                if (k) blob.push_back(',');
                append_int(blob, q[k]);
            }
            blob += "],\"category_id\":";
            append_int(blob, kCategoryIds[b.cls]);
            blob += ",\"score_ppm\":";
            append_int(blob, q[4]);
            blob.push_back('}');
        }
        blob += "],\"image_id\":";
        append_int(blob, c.image_id);
        blob += ",\"item_index\":";
        append_int(blob, item_index);
        blob += ",\"schema\":\"";
        blob += kSchema;
        blob += "\"}";
        return true;
    }

    // COCO category id of each of the 80 contiguous class indices (ids sorted ascending, as in YOLOX COCO).
    static constexpr int kCategoryIds[kClasses] = {
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
        60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};

   private:
    struct Ctx {
        int64_t image_id = 0, width = 0, height = 0;
    };
    struct Cand {
        double x1, y1, x2, y2, score;
        int cls;
    };
    std::map<int64_t, Ctx> ctx_;
    mutable std::vector<Cand> cand_;  // preallocated scratch (no allocation in the measured path)
    mutable std::vector<size_t> kept_;

    static bool quantize(double v, double scale, int64_t& out) {
        const double u = v * scale + 0.5;
        if (!std::isfinite(u) || std::fabs(u) > 9.0e15) return false;
        out = int64_t(std::floor(u));
        return true;
    }

    static void append_int(std::string& out, int64_t v) {
        char buf[24];
        int n = 0;
        uint64_t m = v < 0 ? uint64_t(0) - uint64_t(v) : uint64_t(v);
        do {
            buf[n++] = char('0' + m % 10);
            m /= 10;
        } while (m);
        if (v < 0) out.push_back('-');
        while (n) out.push_back(buf[--n]);
    }
};

}  // namespace sdv
