// SDV-BMT native runner: native implementation of the CenterPoint-Pillars nuScenes workload's output pipeline,
// postprocess centerpoint-decode-circlenms/1 + codec centerpoint-det3d-f32/1.
//
// The Python CenterPointNuScenesWorkload (src/sdv_bmt/workload/centerpoint.py, decode_heads) is the reference
// implementation and owns the semantics; this twin keeps t_output_ready = postprocess + encode + record inside the
// native measured path.  Byte equality with the Python implementation is checked by tests (synthetic raw outputs)
// and on real device outputs.
//
// raw output: float32 [70, 128, 128] (little-endian), task by task (car | truck, construction_vehicle | bus,
//             trailer | barrier | motorcycle, bicycle | pedestrian, traffic_cone): reg 2, height 1, dim 3, rot 2,
//             vel 2, heatmap logits (classes).
// semantics:  IEEE double, the operations and their order of the Python twin: score = 1 / (1 + exp(-logit)); per
//             class the 500 best cells and then the 500 best of the task (score descending, ties: lower cell /
//             position first); x = ((col + reg0) * 4) * 0.2 + (-51.2), y likewise with row and reg1, z = height,
//             size = exp(dim), yaw = atan2(rot0, rot1); kept when score > 0.1 and (x, y, z) inside
//             [-61.2, -61.2, -10, 61.2, 61.2, 10]; circle NMS in candidate order (removed when dx*dx + dy*dy <=
//             r_task, r = 4, 12, 10, 1, 0.85, 0.175); at most 83 per task; z of the bottom centre = z - dz * 0.5.
// errors:     (postprocess) a NaN heatmap logit of a class ("NaN heatmap logit"), checked class by class in task
//             order; a written detection with a value whose float32 conversion is not finite ("non-finite detection
//             value"), checked in write order.  The first error in that order is reported, as in the Python twin.
// codec:      "SDVCP3D1", item index u32, count u32, then per detection 9 float32 box values, float32 score,
//             u32 label (44 bytes).  Work buffers are allocated at construction, never in the measured path.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "json.hpp"
#include "loop.hpp"

namespace sdv {

class CenterPointOutput {
   public:
    static constexpr const char* kCodec = "centerpoint-det3d-f32/1";
    static constexpr const char* kPostprocess = "centerpoint-decode-circlenms/1";
    static constexpr int kGrid = 128, kHW = kGrid * kGrid, kTasks = 6, kReg = 10, kMaxNum = 500, kPostMax = 83;
    static constexpr int kClasses[kTasks] = {1, 2, 2, 1, 2, 2};
    static constexpr double kMinRadius[kTasks] = {4.0, 12.0, 10.0, 1.0, 0.85, 0.175};
    static constexpr int kChannels = 70;
    static constexpr size_t kRawBytes = size_t(kChannels) * kHW * 4;
    static constexpr size_t kMaxBlobBytes = 16 + size_t(kTasks) * kPostMax * 44;

    explicit CenterPointOutput(const Json& spec) {
        const uint16_t probe = 1;
        if (*reinterpret_cast<const unsigned char*>(&probe) != 1) throw JsonError("little-endian host required");
        if (spec.at("codec").as_string() != kCodec || spec.at("postprocess").as_string() != kPostprocess)
            throw JsonError("CenterPoint output pipeline codec/postprocess differ from the native implementation");
        const Json::Array& tc = spec.at("task_classes").as_array();
        if (tc.size() != size_t(kTasks) || spec.at("grid").as_int() != kGrid || spec.at("max_num").as_int() != kMaxNum ||
            spec.at("post_max_size").as_int() != kPostMax || spec.at("score_threshold_x1e6").as_int() != 100000)
            throw JsonError("CenterPoint output pipeline parameters differ from the native implementation");
        for (int t = 0; t < kTasks; ++t)
            if (tc[size_t(t)].as_int() != kClasses[t]) throw JsonError("CenterPoint task classes differ");
        score_.assign(size_t(2) * kHW, 0.0);
        order_.assign(kHW, 0);
        cand_.reserve(size_t(2) * kMaxNum);
        kept_.reserve(kMaxNum);
        suppressed_.assign(kMaxNum, 0);
    }

    size_t max_blob_bytes() const { return kMaxBlobBytes; }

    bool encode(int64_t /*request_id*/, int64_t item_index, std::string_view raw, std::string& blob,
                std::string& phase, ErrorInfo& err) const {
        phase = "postprocess";
        if (raw.size() != kRawBytes) {
            err = ErrorInfo{"WorkloadError", "raw output must be " + std::to_string(kRawBytes) +
                                                 " bytes (float32 [70, 128, 128])"};
            return false;
        }
        if (item_index < 0 || item_index > int64_t(0xFFFFFFFFu)) {
            err = ErrorInfo{"WorkloadError", "item index out of range for the detection header"};
            return false;
        }
        const char* base_ptr = raw.data();
        auto at = [base_ptr](int channel, int cell) {
            float v;
            std::memcpy(&v, base_ptr + (size_t(channel) * kHW + size_t(cell)) * 4, 4);
            return double(v);
        };
        const size_t header_at = blob.size();
        blob += "SDVCP3D1";
        put_u32(blob, uint32_t(item_index));
        put_u32(blob, 0);  // count, patched below
        uint32_t count = 0;
        int base = 0, label_off = 0;
        for (int t = 0; t < kTasks; ++t) {
            const int n = kClasses[t], heat = base + kReg;
            cand_.clear();
            for (int k = 0; k < n; ++k) {
                double* s = score_.data() + size_t(k) * kHW;
                for (int i = 0; i < kHW; ++i) {
                    const double logit = at(heat + k, i);
                    if (std::isnan(logit)) {
                        err = ErrorInfo{"WorkloadError", "NaN heatmap logit"};
                        return false;
                    }
                    s[i] = 1.0 / (1.0 + std::exp(-logit));
                }
                for (int i = 0; i < kHW; ++i) order_[size_t(i)] = i;
                std::partial_sort(order_.begin(), order_.begin() + kMaxNum, order_.end(), [s](int a, int b) {
                    return s[a] > s[b] || (s[a] == s[b] && a < b);
                });
                for (int r = 0; r < kMaxNum; ++r)
                    cand_.push_back(Cand{s[order_[size_t(r)]], k * kMaxNum + r, k, order_[size_t(r)]});
            }
            std::sort(cand_.begin(), cand_.end(), [](const Cand& a, const Cand& b) {
                return a.s > b.s || (a.s == b.s && a.pos < b.pos);
            });
            kept_.clear();
            for (int c = 0; c < kMaxNum && c < int(cand_.size()); ++c) {
                const Cand& e = cand_[size_t(c)];
                const int row = e.cell / kGrid, col = e.cell % kGrid;
                const double x = ((col + at(base, e.cell)) * 4.0) * 0.2 + (-51.2);
                const double y = ((row + at(base + 1, e.cell)) * 4.0) * 0.2 + (-51.2);
                const double z = at(base + 2, e.cell);
                if (!(e.s > 0.1 && -61.2 <= x && x <= 61.2 && -61.2 <= y && y <= 61.2 && -10.0 <= z && z <= 10.0))
                    continue;
                Det d;
                d.v[0] = x;
                d.v[1] = y;
                d.v[2] = z;
                for (int j = 0; j < 3; ++j) d.v[3 + j] = std::exp(at(base + 3 + j, e.cell));
                d.v[6] = std::atan2(at(base + 6, e.cell), at(base + 7, e.cell));
                d.v[7] = at(base + 8, e.cell);
                d.v[8] = at(base + 9, e.cell);
                d.s = e.s;
                d.label = label_off + e.cls;
                kept_.push_back(d);
            }
            std::fill(suppressed_.begin(), suppressed_.end(), 0);
            int written = 0;
            for (size_t p = 0; p < kept_.size(); ++p) {
                if (suppressed_[p]) continue;
                for (size_t q = p + 1; q < kept_.size(); ++q) {
                    if (suppressed_[q]) continue;
                    const double dx = kept_[p].v[0] - kept_[q].v[0], dy = kept_[p].v[1] - kept_[q].v[1];
                    if (dx * dx + dy * dy <= kMinRadius[t]) suppressed_[q] = 1;
                }
                if (written < kPostMax) {
                    const Det& d = kept_[p];
                    double box[9] = {d.v[0], d.v[1], d.v[2] - d.v[5] * 0.5, d.v[3], d.v[4], d.v[5], d.v[6], d.v[7], d.v[8]};
                    for (double b : box)
                        if (!std::isfinite(float(b))) {
                            err = ErrorInfo{"WorkloadError", "non-finite detection value"};
                            return false;
                        }
                    phase = "encode";
                    for (double b : box) put_f32(blob, float(b));
                    put_f32(blob, float(d.s));
                    put_u32(blob, uint32_t(d.label));
                    phase = "postprocess";
                    ++written;
                    ++count;
                }
            }
            base += kReg + n;
            label_off += n;
        }
        phase = "encode";
        for (int k = 0; k < 4; ++k) blob[header_at + 12 + size_t(k)] = char((count >> (8 * k)) & 0xFF);
        return true;
    }

   private:
    struct Cand {
        double s;
        int pos, cls, cell;
    };
    struct Det {
        double v[9];
        double s;
        int label;
    };
    mutable std::vector<double> score_;
    mutable std::vector<int> order_;
    mutable std::vector<Cand> cand_;
    mutable std::vector<Det> kept_;
    mutable std::vector<char> suppressed_;

    static void put_u32(std::string& s, uint32_t v) {
        for (int k = 0; k < 4; ++k) s.push_back(char((v >> (8 * k)) & 0xFF));
    }
    static void put_f32(std::string& s, float f) {
        uint32_t u;
        std::memcpy(&u, &f, 4);
        put_u32(s, u);
    }
};

}  // namespace sdv
