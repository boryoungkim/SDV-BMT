// SDV-BMT native runner: cuDNN execution of a converted model plan (NVIDIA backend only), Stage 2b-G.
//
// A workload whose device program has kind "cudnn_plan" is executed by this executor instead of a single
// NVRTC kernel.  The plan (integer-only JSON produced offline from the ONNX model by
// tools/stage2b_g_real/build_yolox_plan.py) lists the steps; this file maps them onto:
//   conv                -> cudnnConvolutionForward (algorithm fixed at prepare, see below) followed by one
//                          glue kernel that adds the bias, applies the activation (none / SiLU / sigmoid)
//                          and the optional residual add
//   maxpool             -> cudnnPoolingForward (CUDNN_POOLING_MAX)
//   space_to_depth2, upsample_nearest2x, copy, head_transpose -> glue kernels compiled by NVRTC
// v0.1.2 (DeepLabV3+) adds, in a second glue source compiled only when a plan uses them:
//   conv epilogue "bias (+ residual) then ReLU" (act 3 and/or res_before_act), gap (global average pooling),
//   broadcast (Cx1x1 -> CxHxW), resize_bilinear (align_corners=False, PyTorch source-index arithmetic) and
//   argmax_u8 (class index of the maximum over channels, lowest index on ties, written as uint8).
// At prepare the executor checks every step's read/write byte ranges against its buffer (input, task output,
// arena, weights) and that the plan's writes to the output buffer cover the whole task output; the output
// buffer is not cleared between requests, so an unwritten part would silently keep the previous item's result.
// An optional frequency probe kernel (one thread spinning a fixed number of SM cycles while reading
// %globaltimer and clock64) runs first on the request stream and writes the four timing words that the
// backend reports as nvidia.in_kernel_* observations.
//
// cuDNN algorithm selection is a measurement-identity fact.  Policy "pinned" uses the per-convolution
// (algorithm, math type) list declared in the DUT config and fails if cuDNN rejects an entry; policy
// "heuristic" takes, for every convolution, the first cudnnGetConvolutionForwardAlgorithm_v7 result that is
// supported, deterministic and fits the workspace limit.  Either way the realized table, its SHA-256 and the
// heuristic's own choice are returned as prepare facts.  Nothing is benchmarked or tuned at run time.
#pragma once

#include <cuda.h>
#include <cudnn.h>
#include <dirent.h>
#include <dlfcn.h>
#include <nvrtc.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "../common/json.hpp"
#include "../common/sha256.hpp"

namespace sdv {

class CudnnFailure : public std::runtime_error {
   public:
    CudnnFailure(const std::string& api, cudnnStatus_t s)
        : std::runtime_error(api + ": " + cudnnGetErrorString(s) + " (" + std::to_string(int(s)) + ")"),
          code(cudnnGetErrorString(s)) {}
    std::string code;
};

inline void dnn(cudnnStatus_t s, const char* api) {
    if (s != CUDNN_STATUS_SUCCESS) throw CudnnFailure(api, s);
}

inline std::string read_file(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw JsonError("cannot read " + path);
    std::string out;
    char buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    std::fclose(f);
    return out;
}

inline std::string file_sha256(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw JsonError("cannot read " + path);
    Sha256 h;
    std::vector<char> buf(1 << 20);
    size_t n;
    while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0) h.update(buf.data(), n);
    std::fclose(f);
    return h.hex();
}

// Glue kernels (NVRTC).  Plain float32 arithmetic; SiLU is x * sigmoid(x) as in the ONNX graph (Mul(x, Sigmoid(x))).
inline const char* glue_kernel_source() {
    return R"SDV(
// SDV-BMT cuDNN-plan glue kernels v1
extern "C" __global__ void sdv_space_to_depth2(const float* __restrict__ x, float* __restrict__ y, int C, int H,
                                               int W, int g0y, int g0x, int g1y, int g1x, int g2y, int g2x,
                                               int g3y, int g3x) {
    const int OH = H / 2, OW = W / 2;
    const long n = 4L * C * OH * OW;
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int ox = (int)(i % OW), oy = (int)((i / OW) % OH), oc = (int)(i / ((long)OH * OW));
    const int g = oc / C, c = oc % C;
    const int dy = g == 0 ? g0y : g == 1 ? g1y : g == 2 ? g2y : g3y;
    const int dx = g == 0 ? g0x : g == 1 ? g1x : g == 2 ? g2x : g3x;
    y[i] = x[((long)c * H + 2 * oy + dy) * W + 2 * ox + dx];
}
extern "C" __global__ void sdv_bias_act(float* __restrict__ y, const float* __restrict__ b, int C, int HW, int act,
                                        const float* __restrict__ res) {
    const long n = (long)C * HW;
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = y[i] + b[i / HW];
    if (act == 1) {
        const float s = 1.0f / (1.0f + expf(-v));
        v = v * s;
    } else if (act == 2) {
        v = 1.0f / (1.0f + expf(-v));
    }
    if (res) v = v + res[i];
    y[i] = v;
}
extern "C" __global__ void sdv_upsample_nearest2x(const float* __restrict__ x, float* __restrict__ y, int C, int H,
                                                  int W) {
    const int OH = 2 * H, OW = 2 * W;
    const long n = (long)C * OH * OW;
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int ox = (int)(i % OW), oy = (int)((i / OW) % OH), c = (int)(i / ((long)OH * OW));
    y[i] = x[((long)c * H + oy / 2) * W + ox / 2];
}
extern "C" __global__ void sdv_copy(const float* __restrict__ x, float* __restrict__ y, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}
extern "C" __global__ void sdv_head_transpose(const float* __restrict__ x, float* __restrict__ y, int C, int N,
                                              int anchor_offset) {
    const long n = (long)C * N;
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int p = (int)(i / C), c = (int)(i % C);
    y[((long)anchor_offset + p) * C + c] = x[(long)c * N + p];
}
extern "C" __global__ void sdv_frequency_probe(unsigned long long* words, long long cycles) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    const long long c0 = clock64();
    long long c1 = c0;
    while (c1 - c0 < cycles) c1 = clock64();
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
    words[0] = t0;
    words[1] = t1;
    words[2] = (unsigned long long)c0;
    words[3] = (unsigned long long)c1;
}
)SDV";
}

// Second glue source (v0.1.2): compiled only for plans that use one of its kernels, so the first source and its
// cubin (and every fact derived from them) stay unchanged for plans that do not.
inline const char* glue2_kernel_source() {
    return R"SDV(
// SDV-BMT cuDNN-plan glue kernels v2 (v0.1.2)
extern "C" __global__ void sdv_bias_res_relu(float* __restrict__ y, const float* __restrict__ b, int C, int HW,
                                             int relu, const float* __restrict__ res) {
    const long n = (long)C * HW;
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = y[i] + b[i / HW];
    if (res) v = v + res[i];
    if (relu && v < 0.0f) v = 0.0f;
    y[i] = v;
}
extern "C" __global__ void sdv_gap(const float* __restrict__ x, float* __restrict__ y, int HW) {
    __shared__ float part[256];
    const int c = blockIdx.x;
    const float* p = x + (long)c * HW;
    float s = 0.0f;
    for (int i = threadIdx.x; i < HW; i += 256) s += p[i];
    part[threadIdx.x] = s;
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1) {
        if (threadIdx.x < w) part[threadIdx.x] += part[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[c] = part[0] / (float)HW;
}
extern "C" __global__ void sdv_broadcast(const float* __restrict__ x, float* __restrict__ y, int C, int HW) {
    const long n = (long)C * HW;
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = x[i / HW];
}
extern "C" __global__ void sdv_resize_bilinear(const float* __restrict__ x, float* __restrict__ y, int C, int H,
                                               int W, int OH, int OW, float sh, float sw) {
    const long n = (long)C * OH * OW;
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int ox = (int)(i % OW), oy = (int)((i / OW) % OH), c = (int)(i / ((long)OH * OW));
    float fy = sh * (oy + 0.5f) - 0.5f;
    if (fy < 0.0f) fy = 0.0f;
    float fx = sw * (ox + 0.5f) - 0.5f;
    if (fx < 0.0f) fx = 0.0f;
    const int y0 = (int)fy, x0 = (int)fx;
    const int y1 = y0 + (y0 < H - 1 ? 1 : 0), x1 = x0 + (x0 < W - 1 ? 1 : 0);
    const float ly1 = fy - y0, ly0 = 1.0f - ly1, lx1 = fx - x0, lx0 = 1.0f - lx1;
    const float* p = x + (long)c * H * W;
    y[i] = ly0 * (lx0 * p[(long)y0 * W + x0] + lx1 * p[(long)y0 * W + x1]) +
           ly1 * (lx0 * p[(long)y1 * W + x0] + lx1 * p[(long)y1 * W + x1]);
}
extern "C" __global__ void sdv_argmax_u8(const float* __restrict__ x, unsigned char* __restrict__ y, int C, int HW) {
    const long p = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= HW) return;
    float best = x[p];
    int k = 0;
    for (int c = 1; c < C; ++c) {
        const float v = x[(long)c * HW + p];
        if (v > best) {
            best = v;
            k = c;
        }
    }
    y[p] = (unsigned char)k;
}
)SDV";
}

class CudnnPlanExecutor {
   public:
    // ------------------------------------------------------------ identity (no device context needed)
    static Json library_identity() {
        Json id = Json::object();
        id["nvidia.cudnn_version"] = static_cast<long long>(cudnnGetVersion());
        id["nvidia.cudnn_cudart_version"] = static_cast<long long>(cudnnGetCudartVersion());
        Dl_info info{};
        std::string lib = dladdr(reinterpret_cast<void*>(&cudnnGetVersion), &info) && info.dli_fname
                              ? info.dli_fname : "unknown";
        id["nvidia.cudnn_library"] = lib;
        // every libcudnn* shared object in that directory (cuDNN loads its sub-libraries from there)
        std::string dir = lib.substr(0, lib.rfind('/'));
        std::vector<std::string> names;
        if (DIR* d = opendir(dir.c_str())) {
            while (dirent* e = readdir(d)) {
                std::string n = e->d_name;
                if (n.rfind("libcudnn", 0) == 0 && n.find(".so") != std::string::npos) names.push_back(n);
            }
            closedir(d);
        }
        std::sort(names.begin(), names.end());
        Json files = Json::array();
        for (const std::string& n : names) {
            Json f = Json::object();
            f["file"] = n;
            f["sha256"] = file_sha256(dir + "/" + n);
            files.push(f);
        }
        id["nvidia.cudnn_library_files"] = files;
        return id;
    }

    // ------------------------------------------------------------ DUT.prepare
    // program: {plan_path, plan_sha256, weights_path, weights_sha256, artifact_sha256, ...}
    // options: backend runtime options (cudnn {...}, frequency_probe_cycles)
    Json prepare(CUstream stream, const std::string& arch, const Json& program, const Json& options,
                 int64_t& device_allocs) {
        stream_ = stream;
        Json facts = Json::object();
        const std::string plan_text = read_file(program.at("plan_path").as_string());
        const std::string plan_sha = Sha256::of(plan_text);
        if (plan_sha != program.at("plan_sha256").as_string())
            throw JsonError("cuDNN plan SHA-256 does not match the workload declaration");
        Json plan = Json::parse(plan_text);
        if (plan.at("schema").as_string() != "sdv-bmt-cudnn-plan/1") throw JsonError("unsupported plan schema");
        if (plan.at("model").at("onnx_sha256").as_string() != program.at("artifact_sha256").as_string())
            throw JsonError("plan model SHA-256 differs from workload.artifact_sha256");
        const std::string weights_path = program.at("weights_path").as_string();
        const std::string weights = read_file(weights_path);
        const std::string weights_sha = Sha256::of(weights);
        if (weights_sha != program.at("weights_sha256").as_string() ||
            weights_sha != plan.at("weights").at("sha256").as_string() ||
            int64_t(weights.size()) != plan.at("weights").at("bytes").as_int())
            throw JsonError("weights file does not match its declared SHA-256/size");
        const Json& cudnn_opt = options.at("cudnn");
        const std::string math = cudnn_opt.at("math_type").as_string();
        math_ = math == "default" ? CUDNN_DEFAULT_MATH : math == "fma" ? CUDNN_FMA_MATH
              : math == "tensor_op" ? CUDNN_TENSOR_OP_MATH : throw JsonError("bad cudnn.math_type");
        policy_ = cudnn_opt.at("algo_policy").as_string();
        if (policy_ != "pinned" && policy_ != "heuristic") throw JsonError("bad cudnn.algo_policy");
        ws_limit_ = size_t(cudnn_opt.at("workspace_limit_bytes").as_int());
        probe_cycles_ = options.has("frequency_probe_cycles") ? options.at("frequency_probe_cycles").as_int() : 0;

        const size_t input_bytes = size_t(program.at("input_bytes").as_int());
        const size_t task_output_bytes = size_t(program.at("task_output_bytes").as_int());
        bool need_glue2 = false;  // a v0.1.2 op or epilogue anywhere in the plan
        for (const Json& s : plan.at("steps").as_array()) {
            const std::string& op = s.at("op").as_string();
            if (op == "gap" || op == "broadcast" || op == "resize_bilinear" || op == "argmax_u8") need_glue2 = true;
            if (op == "conv" && (s.at("act").as_int() == 3 || (s.has("res_before_act") && s.at("res_before_act").as_bool())))
                need_glue2 = true;
        }

        // device memory: weights (uploaded once), activation arena
        weights_ = alloc(weights.size(), device_allocs);
        cu_check(cuMemcpyHtoD(weights_, weights.data(), weights.size()), "cuMemcpyHtoD(weights)");
        arena_bytes_ = size_t(plan.at("arena_bytes").as_int());
        arena_ = alloc(arena_bytes_, device_allocs);
        cu_check(cuMemsetD8(arena_, 0, arena_bytes_), "cuMemsetD8(arena)");

        // glue kernels
        std::string cubin, log;
        compile_glue(arch, cubin, log);
        cu_check(cuModuleLoadData(&module_, cubin.data()), "cuModuleLoadData(glue)");
        for (auto& f : {std::make_pair(&fn_s2d_, "sdv_space_to_depth2"), std::make_pair(&fn_bias_act_, "sdv_bias_act"),
                        std::make_pair(&fn_upsample_, "sdv_upsample_nearest2x"), std::make_pair(&fn_copy_, "sdv_copy"),
                        std::make_pair(&fn_head_, "sdv_head_transpose"), std::make_pair(&fn_probe_, "sdv_frequency_probe")})
            cu_check(cuModuleGetFunction(f.first, module_, f.second), "cuModuleGetFunction(glue)");
        std::string cubin2, log2;
        if (need_glue2) {
            compile_source(glue2_kernel_source(), "sdv_bmt_cudnn_plan_glue2.cu", arch, cubin2, log2);
            cu_check(cuModuleLoadData(&module2_, cubin2.data()), "cuModuleLoadData(glue2)");
            for (auto& f : {std::make_pair(&fn_bias_res_relu_, "sdv_bias_res_relu"), std::make_pair(&fn_gap_, "sdv_gap"),
                            std::make_pair(&fn_broadcast_, "sdv_broadcast"),
                            std::make_pair(&fn_resize_, "sdv_resize_bilinear"), std::make_pair(&fn_argmax_, "sdv_argmax_u8")})
                cu_check(cuModuleGetFunction(f.first, module2_, f.second), "cuModuleGetFunction(glue2)");
        }
        // byte ranges each step reads/writes, checked against the buffers after parsing (v0.1.2 safeguard)
        std::vector<Extent> extents;
        auto extent = [&extents](const Ref& r, unsigned long long bytes, bool write) {
            extents.push_back(Extent{r.buf, r.offset, bytes, write});
        };
        const unsigned long long weight_bytes = weights.size();
        auto weight_range = [weight_bytes](unsigned long long off, unsigned long long bytes) {
            if (off > weight_bytes || bytes > weight_bytes - off)
                throw JsonError("plan step reads outside the weights file");
        };

        dnn(cudnnCreate(&handle_), "cudnnCreate");
        dnn(cudnnSetStream(handle_, reinterpret_cast<cudaStream_t>(stream_)), "cudnnSetStream");

        const Json* pinned = nullptr;
        if (policy_ == "pinned") pinned = &cudnn_opt.at("algorithms");
        size_t conv_index = 0, workspace = 0;
        Json table = Json::array(), detail = Json::array();
        int heuristic_agreements = 0;
        for (const Json& s : plan.at("steps").as_array()) {
            Step st;
            const std::string& op = s.at("op").as_string();
            if (op == "conv") {
                st.op = Op::Conv;
                st.x = ref(s.at("x"));
                st.y = ref(s.at("y"));
                st.has_res = !s.at("residual").is_null();
                if (st.has_res) st.res = ref(s.at("residual"));
                st.w_off = size_t(s.at("w_offset").as_int());
                st.b_off = size_t(s.at("b_offset").as_int());
                st.act = int(s.at("act").as_int());
                st.res_before_act = s.has("res_before_act") && s.at("res_before_act").as_bool();
                if (st.act < 0 || st.act > 3) throw JsonError("conv act must be 0 (none), 1 (SiLU), 2 (sigmoid) or 3 (ReLU)");
                if ((st.act == 1 || st.act == 2) && st.res_before_act)
                    throw JsonError("res_before_act applies to act none or ReLU only");
                st.k = int(s.at("k").as_int());
                st.hw = int(s.at("oh").as_int() * s.at("ow").as_int());
                int c = int(s.at("c").as_int()), h = int(s.at("h").as_int()), w = int(s.at("w").as_int());
                int group = int(s.at("group").as_int());
                if (group <= 0 || c % group != 0) throw JsonError("conv group must divide the input channels");
                {
                    const unsigned long long out = 4ULL * st.k * st.hw;
                    extent(st.x, 4ULL * c * h * w, false);
                    extent(st.y, out, true);
                    if (st.has_res) extent(st.res, out, false);
                    weight_range(st.w_off, 4ULL * st.k * (c / group) * s.at("r").as_int() * s.at("s").as_int());
                    weight_range(st.b_off, 4ULL * st.k);
                }
                dnn(cudnnCreateTensorDescriptor(&st.xd), "cudnnCreateTensorDescriptor");
                dnn(cudnnCreateTensorDescriptor(&st.yd), "cudnnCreateTensorDescriptor");
                dnn(cudnnCreateFilterDescriptor(&st.wd), "cudnnCreateFilterDescriptor");
                dnn(cudnnCreateConvolutionDescriptor(&st.cd), "cudnnCreateConvolutionDescriptor");
                dnn(cudnnSetTensor4dDescriptor(st.xd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c, h, w), "xDesc");
                dnn(cudnnSetFilter4dDescriptor(st.wd, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, st.k, c / group,
                                               int(s.at("r").as_int()), int(s.at("s").as_int())), "wDesc");
                dnn(cudnnSetConvolution2dDescriptor(st.cd, int(s.at("pad_h").as_int()), int(s.at("pad_w").as_int()),
                                                    int(s.at("stride_h").as_int()), int(s.at("stride_w").as_int()),
                                                    int(s.at("dil_h").as_int()), int(s.at("dil_w").as_int()),
                                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT), "convDesc");
                if (group != 1) dnn(cudnnSetConvolutionGroupCount(st.cd, group), "cudnnSetConvolutionGroupCount");
                dnn(cudnnSetConvolutionMathType(st.cd, math_), "cudnnSetConvolutionMathType");
                int n = 0, oc = 0, oh = 0, ow = 0;
                dnn(cudnnGetConvolution2dForwardOutputDim(st.cd, st.xd, st.wd, &n, &oc, &oh, &ow), "outputDim");
                if (n != 1 || oc != st.k || oh != s.at("oh").as_int() || ow != s.at("ow").as_int())
                    throw JsonError("cuDNN output geometry differs from the plan");
                dnn(cudnnSetTensor4dDescriptor(st.yd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, oc, oh, ow), "yDesc");
                // heuristic choice (always computed: it is the reference for the pinned table)
                int heur_algo = -1, heur_math = -1, heur_rank = -1;
                cudnnConvolutionFwdAlgoPerf_t perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
                int returned = 0;
                dnn(cudnnGetConvolutionForwardAlgorithm_v7(handle_, st.xd, st.wd, st.cd, st.yd,
                                                           CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returned, perf),
                    "cudnnGetConvolutionForwardAlgorithm_v7");
                for (int r = 0; r < returned; ++r) {
                    if (perf[r].status == CUDNN_STATUS_SUCCESS && perf[r].determinism == CUDNN_DETERMINISTIC &&
                        perf[r].memory <= ws_limit_) {
                        heur_algo = int(perf[r].algo);
                        heur_math = int(perf[r].mathType);
                        heur_rank = r;
                        break;
                    }
                }
                if (policy_ == "heuristic") {
                    if (heur_algo < 0) throw JsonError("no deterministic cuDNN algorithm fits the workspace limit");
                    st.algo = cudnnConvolutionFwdAlgo_t(heur_algo);
                    st.math = cudnnMathType_t(heur_math);
                } else {
                    if (conv_index >= pinned->as_array().size()) throw JsonError("cudnn.algorithms is too short");
                    const Json& e = pinned->as_array()[conv_index];
                    st.algo = cudnnConvolutionFwdAlgo_t(e.as_array().at(0).as_int());
                    st.math = cudnnMathType_t(e.as_array().at(1).as_int());
                }
                heuristic_agreements += int(st.algo) == heur_algo && int(st.math) == heur_math;
                dnn(cudnnSetConvolutionMathType(st.cd, st.math), "cudnnSetConvolutionMathType");
                dnn(cudnnGetConvolutionForwardWorkspaceSize(handle_, st.xd, st.wd, st.cd, st.yd, st.algo, &st.ws),
                    "cudnnGetConvolutionForwardWorkspaceSize");
                if (st.ws > ws_limit_) throw JsonError("pinned cuDNN algorithm exceeds the workspace limit");
                workspace = std::max(workspace, st.ws);
                Json row = Json::array();
                row.push(static_cast<long long>(st.algo));
                row.push(static_cast<long long>(st.math));
                table.push(row);
                Json d = Json::object();
                d["conv"] = static_cast<long long>(conv_index);
                d["algo"] = static_cast<long long>(st.algo);
                d["math_type"] = static_cast<long long>(st.math);
                d["workspace_bytes"] = static_cast<unsigned long long>(st.ws);
                d["heuristic_algo"] = heur_algo;
                d["heuristic_math_type"] = heur_math;
                d["heuristic_rank"] = heur_rank;
                d["heuristic_candidates"] = returned;
                detail.push(d);
                ++conv_index;
            } else if (op == "maxpool") {
                st.op = Op::MaxPool;
                st.x = ref(s.at("x"));
                st.y = ref(s.at("y"));
                int c = int(s.at("c").as_int()), win = int(s.at("window").as_int()), pad = int(s.at("pad").as_int()),
                    stride = int(s.at("stride").as_int());
                dnn(cudnnCreateTensorDescriptor(&st.xd), "cudnnCreateTensorDescriptor");
                dnn(cudnnCreateTensorDescriptor(&st.yd), "cudnnCreateTensorDescriptor");
                dnn(cudnnCreatePoolingDescriptor(&st.pd), "cudnnCreatePoolingDescriptor");
                dnn(cudnnSetTensor4dDescriptor(st.xd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c,
                                               int(s.at("h").as_int()), int(s.at("w").as_int())), "xDesc");
                dnn(cudnnSetTensor4dDescriptor(st.yd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c,
                                               int(s.at("oh").as_int()), int(s.at("ow").as_int())), "yDesc");
                dnn(cudnnSetPooling2dDescriptor(st.pd, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, win, win, pad, pad,
                                                stride, stride), "cudnnSetPooling2dDescriptor");
                extent(st.x, 4ULL * c * s.at("h").as_int() * s.at("w").as_int(), false);
                extent(st.y, 4ULL * c * s.at("oh").as_int() * s.at("ow").as_int(), true);
            } else if (op == "space_to_depth2") {
                st.op = Op::SpaceToDepth;
                st.x = ref(s.at("x"));
                st.y = ref(s.at("y"));
                st.i[0] = int(s.at("c").as_int());
                st.i[1] = int(s.at("h").as_int());
                st.i[2] = int(s.at("w").as_int());
                const Json::Array& order = s.at("order").as_array();
                if (order.size() != 4) throw JsonError("space_to_depth2 needs 4 groups");
                for (int g = 0; g < 4; ++g) {
                    st.i[3 + 2 * g] = int(order[size_t(g)].as_array().at(0).as_int());
                    st.i[4 + 2 * g] = int(order[size_t(g)].as_array().at(1).as_int());
                }
                st.n = 4L * st.i[0] * (st.i[1] / 2) * (st.i[2] / 2);
                extent(st.x, 4ULL * st.i[0] * st.i[1] * st.i[2], false);
                extent(st.y, 4ULL * st.n, true);
            } else if (op == "upsample_nearest2x") {
                st.op = Op::Upsample;
                st.x = ref(s.at("x"));
                st.y = ref(s.at("y"));
                st.i[0] = int(s.at("c").as_int());
                st.i[1] = int(s.at("h").as_int());
                st.i[2] = int(s.at("w").as_int());
                st.n = 4L * st.i[0] * st.i[1] * st.i[2];
                extent(st.x, 4ULL * st.i[0] * st.i[1] * st.i[2], false);
                extent(st.y, 4ULL * st.n, true);
            } else if (op == "copy") {
                st.op = Op::Copy;
                st.x = ref(s.at("x"));
                st.y = ref(s.at("y"));
                st.n = s.at("elements").as_int();
                extent(st.x, 4ULL * st.n, false);
                extent(st.y, 4ULL * st.n, true);
            } else if (op == "head_transpose") {
                const int channels = int(s.at("channels").as_int());
                const Ref y = ref(s.at("y"));
                for (const Json& lv : s.at("levels").as_array()) {  // one launch per level
                    Step t;
                    t.op = Op::HeadTranspose;
                    t.x = ref(lv.at("x"));
                    t.y = y;
                    t.i[0] = channels;
                    t.i[1] = int(lv.at("h").as_int() * lv.at("w").as_int());
                    t.i[2] = int(lv.at("anchor_offset").as_int());
                    t.n = long(t.i[0]) * t.i[1];
                    extent(t.x, 4ULL * t.n, false);
                    extent(Ref{y.buf, y.offset + 4ULL * t.i[2] * t.i[0]}, 4ULL * t.n, true);
                    steps_.push_back(t);
                }
                continue;
            } else if (op == "gap" || op == "broadcast" || op == "resize_bilinear" || op == "argmax_u8") {
                st.x = ref(s.at("x"));
                st.y = ref(s.at("y"));
                const long long c = s.at("c").as_int();
                if (c <= 0) throw JsonError(op + " needs c > 0");
                if (op == "gap") {
                    st.op = Op::Gap;
                    const long long h = s.at("h").as_int(), w = s.at("w").as_int();
                    st.i[0] = int(h * w);
                    st.n = long(c);
                    extent(st.x, 4ULL * c * h * w, false);
                    extent(st.y, 4ULL * c, true);
                } else if (op == "broadcast") {
                    st.op = Op::Broadcast;
                    const long long oh = s.at("oh").as_int(), ow = s.at("ow").as_int();
                    st.i[0] = int(c);
                    st.i[1] = int(oh * ow);
                    st.n = long(c * oh * ow);
                    extent(st.x, 4ULL * c, false);
                    extent(st.y, 4ULL * st.n, true);
                } else if (op == "resize_bilinear") {
                    st.op = Op::Resize;
                    const long long h = s.at("h").as_int(), w = s.at("w").as_int(), oh = s.at("oh").as_int(),
                                    ow = s.at("ow").as_int();
                    if (h <= 0 || w <= 0 || oh <= 0 || ow <= 0) throw JsonError("resize_bilinear needs positive sizes");
                    st.i[0] = int(c);
                    st.i[1] = int(h);
                    st.i[2] = int(w);
                    st.i[3] = int(oh);
                    st.i[4] = int(ow);
                    st.f[0] = float(h) / float(oh);  // PyTorch: (float)input_size / output_size
                    st.f[1] = float(w) / float(ow);
                    st.n = long(c * oh * ow);
                    extent(st.x, 4ULL * c * h * w, false);
                    extent(st.y, 4ULL * st.n, true);
                } else {
                    st.op = Op::Argmax;
                    const long long h = s.at("h").as_int(), w = s.at("w").as_int();
                    if (c > 256) throw JsonError("argmax_u8 supports at most 256 classes");
                    st.i[0] = int(c);
                    st.i[1] = int(h * w);
                    st.n = long(h * w);
                    extent(st.x, 4ULL * c * h * w, false);
                    extent(st.y, 1ULL * st.n, true);
                }
            } else {
                throw JsonError("unsupported plan step " + op);
            }
            steps_.push_back(st);
        }
        check_extents(extents, input_bytes, task_output_bytes, size_t(plan.at("arena_bytes").as_int()));
        if (policy_ == "pinned" && conv_index != pinned->as_array().size())
            throw JsonError("cudnn.algorithms has more entries than the plan has convolutions");
        workspace_bytes_ = workspace;
        if (workspace_bytes_) workspace_ = alloc(workspace_bytes_, device_allocs);

        std::string table_text;
        table.write(table_text);
        facts["plan_sha256"] = plan_sha;
        facts["weights_sha256"] = weights_sha;
        facts["glue_source_sha256"] = Sha256::of(std::string(glue_kernel_source()));
        facts["glue_cubin_sha256"] = Sha256::of(cubin);
        facts["glue_nvrtc_options"] = arch + " --std=c++17";
        facts["glue_nvrtc_log"] = log;
        if (need_glue2) {  // absent for plans without v0.1.2 ops, so their facts are unchanged
            facts["glue2_source_sha256"] = Sha256::of(std::string(glue2_kernel_source()));
            facts["glue2_cubin_sha256"] = Sha256::of(cubin2);
            facts["glue2_nvrtc_log"] = log2;
        }
        facts["cudnn_math_type_option"] = math;
        facts["cudnn_algo_policy"] = policy_;
        facts["cudnn_workspace_limit_bytes"] = static_cast<unsigned long long>(ws_limit_);
        facts["cudnn_workspace_bytes"] = static_cast<unsigned long long>(workspace_bytes_);
        facts["cudnn_algorithm_table"] = table;
        facts["cudnn_algorithm_table_sha256"] = Sha256::of(table_text);
        facts["cudnn_heuristic_agreements"] = heuristic_agreements;
        facts["cudnn_convolutions"] = static_cast<long long>(conv_index);
        facts["cudnn_algorithm_detail"] = detail;
        facts["arena_bytes"] = static_cast<unsigned long long>(arena_bytes_);
        facts["plan_launches_per_request"] = static_cast<long long>(launches_per_request());
        facts["frequency_probe_cycles"] = probe_cycles_;
        return facts;
    }

    // Number of device operations enqueued per request by this executor (cuDNN calls + glue launches).
    long launches_per_request() const {
        long n = probe_cycles_ > 0 ? 1 : 0;
        for (const Step& s : steps_) n += s.op == Op::Conv ? 2 : 1;
        return n;
    }

    // ------------------------------------------------------------ per request (inside submit)
    void enqueue(CUdeviceptr input, CUdeviceptr output, CUdeviceptr timing_words) {
        base_[0] = input;
        base_[1] = output;
        base_[2] = arena_;
        if (probe_cycles_ > 0) {
            CUdeviceptr words = timing_words;
            long long cycles = probe_cycles_;
            void* args[] = {&words, &cycles};
            cu_check(cuLaunchKernel(fn_probe_, 1, 1, 1, 1, 1, 1, 0, stream_, args, nullptr), "cuLaunchKernel(probe)");
        }
        const float one = 1.0f, zero = 0.0f;
        for (Step& s : steps_) {
            CUdeviceptr x = ptr(s.x), y = ptr(s.y);
            switch (s.op) {
                case Op::Conv: {
                    dnn(cudnnConvolutionForward(handle_, &one, s.xd, reinterpret_cast<void*>(x), s.wd,
                                                reinterpret_cast<void*>(weights_ + s.w_off), s.cd, s.algo,
                                                reinterpret_cast<void*>(workspace_), workspace_bytes_, &zero, s.yd,
                                                reinterpret_cast<void*>(y)),
                        "cudnnConvolutionForward");
                    CUdeviceptr b = weights_ + s.b_off;
                    CUdeviceptr r = s.has_res ? ptr(s.res) : 0;
                    int c = s.k, hw = s.hw, act = s.act;
                    if (act == 3 || s.res_before_act) {
                        int relu = act == 3 ? 1 : 0;
                        void* args[] = {&y, &b, &c, &hw, &relu, &r};
                        launch(fn_bias_res_relu_, long(c) * hw, args, "cuLaunchKernel(bias_res_relu)");
                    } else {
                        void* args[] = {&y, &b, &c, &hw, &act, &r};
                        launch(fn_bias_act_, long(c) * hw, args, "cuLaunchKernel(bias_act)");
                    }
                    break;
                }
                case Op::Gap: {
                    int hw = s.i[0];
                    void* args[] = {&x, &y, &hw};
                    cu_check(cuLaunchKernel(fn_gap_, unsigned(s.n), 1, 1, 256, 1, 1, 0, stream_, args, nullptr),
                             "cuLaunchKernel(gap)");
                    break;
                }
                case Op::Broadcast: {
                    void* args[] = {&x, &y, &s.i[0], &s.i[1]};
                    launch(fn_broadcast_, s.n, args, "cuLaunchKernel(broadcast)");
                    break;
                }
                case Op::Resize: {
                    void* args[] = {&x, &y, &s.i[0], &s.i[1], &s.i[2], &s.i[3], &s.i[4], &s.f[0], &s.f[1]};
                    launch(fn_resize_, s.n, args, "cuLaunchKernel(resize_bilinear)");
                    break;
                }
                case Op::Argmax: {
                    void* args[] = {&x, &y, &s.i[0], &s.i[1]};
                    launch(fn_argmax_, s.n, args, "cuLaunchKernel(argmax_u8)");
                    break;
                }
                case Op::MaxPool:
                    dnn(cudnnPoolingForward(handle_, s.pd, &one, s.xd, reinterpret_cast<void*>(x), &zero, s.yd,
                                            reinterpret_cast<void*>(y)),
                        "cudnnPoolingForward");
                    break;
                case Op::SpaceToDepth: {
                    void* args[] = {&x, &y, &s.i[0], &s.i[1], &s.i[2], &s.i[3], &s.i[4], &s.i[5],
                                    &s.i[6], &s.i[7], &s.i[8], &s.i[9], &s.i[10]};
                    launch(fn_s2d_, s.n, args, "cuLaunchKernel(space_to_depth2)");
                    break;
                }
                case Op::Upsample: {
                    void* args[] = {&x, &y, &s.i[0], &s.i[1], &s.i[2]};
                    launch(fn_upsample_, s.n, args, "cuLaunchKernel(upsample)");
                    break;
                }
                case Op::Copy: {
                    long n = s.n;
                    void* args[] = {&x, &y, &n};
                    launch(fn_copy_, s.n, args, "cuLaunchKernel(copy)");
                    break;
                }
                case Op::HeadTranspose: {
                    void* args[] = {&x, &y, &s.i[0], &s.i[1], &s.i[2]};
                    launch(fn_head_, s.n, args, "cuLaunchKernel(head_transpose)");
                    break;
                }
            }
        }
    }

    void release() {
        for (Step& s : steps_) {
            if (s.xd) cudnnDestroyTensorDescriptor(s.xd);
            if (s.yd) cudnnDestroyTensorDescriptor(s.yd);
            if (s.wd) cudnnDestroyFilterDescriptor(s.wd);
            if (s.cd) cudnnDestroyConvolutionDescriptor(s.cd);
            if (s.pd) cudnnDestroyPoolingDescriptor(s.pd);
        }
        steps_.clear();
        if (handle_) cudnnDestroy(handle_);
        handle_ = nullptr;
        for (CUdeviceptr* p : {&weights_, &arena_, &workspace_})
            if (*p) {
                cuMemFree(*p);
                *p = 0;
            }
        if (module_) cuModuleUnload(module_);
        module_ = nullptr;
        if (module2_) cuModuleUnload(module2_);
        module2_ = nullptr;
    }

   private:
    enum class Op { Conv, MaxPool, SpaceToDepth, Upsample, Copy, HeadTranspose, Gap, Broadcast, Resize, Argmax };
    struct Ref {
        int buf = 2;  // 0 input, 1 output, 2 arena
        size_t offset = 0;
    };
    struct Extent {
        int buf;
        unsigned long long offset, bytes;
        bool write;
    };
    struct Step {
        Op op = Op::Copy;
        Ref x, y, res;
        bool has_res = false, res_before_act = false;
        size_t w_off = 0, b_off = 0, ws = 0;
        int act = 0, k = 0, hw = 0;
        int i[11] = {};
        float f[2] = {};
        long n = 0;
        cudnnTensorDescriptor_t xd = nullptr, yd = nullptr;
        cudnnFilterDescriptor_t wd = nullptr;
        cudnnConvolutionDescriptor_t cd = nullptr;
        cudnnPoolingDescriptor_t pd = nullptr;
        cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        cudnnMathType_t math = CUDNN_DEFAULT_MATH;
    };

    CUstream stream_ = nullptr;
    cudnnHandle_t handle_ = nullptr;
    cudnnMathType_t math_ = CUDNN_DEFAULT_MATH;
    std::string policy_;
    size_t ws_limit_ = 0, workspace_bytes_ = 0, arena_bytes_ = 0;
    long long probe_cycles_ = 0;
    CUdeviceptr weights_ = 0, arena_ = 0, workspace_ = 0;
    CUdeviceptr base_[3] = {0, 0, 0};
    CUmodule module_ = nullptr, module2_ = nullptr;
    CUfunction fn_s2d_ = nullptr, fn_bias_act_ = nullptr, fn_upsample_ = nullptr, fn_copy_ = nullptr,
               fn_head_ = nullptr, fn_probe_ = nullptr;
    CUfunction fn_bias_res_relu_ = nullptr, fn_gap_ = nullptr, fn_broadcast_ = nullptr, fn_resize_ = nullptr,
               fn_argmax_ = nullptr;
    std::vector<Step> steps_;

    static void cu_check(CUresult r, const char* api) {
        if (r != CUDA_SUCCESS) {
            const char* n = nullptr;
            cuGetErrorName(r, &n);
            throw JsonError(std::string(api) + ": " + (n ? n : "CUDA error"));
        }
    }

    static Ref ref(const Json& j) {
        Ref r;
        const std::string& b = j.at("buf").as_string();
        r.buf = b == "input" ? 0 : b == "output" ? 1 : b == "arena" ? 2 : throw JsonError("bad buffer " + b);
        r.offset = size_t(j.at("offset").as_int());
        return r;
    }

    CUdeviceptr ptr(const Ref& r) const { return base_[r.buf] + r.offset; }

    CUdeviceptr alloc(size_t n, int64_t& count) {
        CUdeviceptr p = 0;
        cu_check(cuMemAlloc(&p, n), "cuMemAlloc");
        ++count;
        return p;
    }

    void launch(CUfunction f, long n, void** args, const char* api) {
        const unsigned threads = 256;
        const unsigned blocks = unsigned((n + threads - 1) / threads);
        cu_check(cuLaunchKernel(f, blocks, 1, 1, threads, 1, 1, 0, stream_, args, nullptr), api);
    }

    // Every read/write range inside its buffer, and the writes to the output buffer covering the whole task
    // output (v0.1.2 safeguard; checked once at prepare, nothing is measured here).
    static void check_extents(const std::vector<Extent>& extents, size_t input_bytes, size_t task_output_bytes,
                              size_t arena_bytes) {
        const unsigned long long size[3] = {input_bytes, task_output_bytes, arena_bytes};
        static const char* names[3] = {"input", "task output", "arena"};
        std::vector<std::pair<unsigned long long, unsigned long long>> out;
        for (const Extent& e : extents) {
            const unsigned long long cap = size[e.buf];
            if (e.bytes == 0 || e.offset > cap || e.bytes > cap - e.offset)
                throw JsonError(std::string("plan step ") + (e.write ? "writes" : "reads") + " outside the " +
                                names[e.buf] + " buffer (offset " + std::to_string(e.offset) + ", " +
                                std::to_string(e.bytes) + " bytes, buffer " + std::to_string(cap) + " bytes)");
            if (e.write && e.buf == 1) out.emplace_back(e.offset, e.offset + e.bytes);
        }
        std::sort(out.begin(), out.end());
        unsigned long long covered = 0;
        for (const auto& r : out) {
            if (r.first > covered) break;
            covered = std::max(covered, r.second);
        }
        if (covered < task_output_bytes)
            throw JsonError("plan steps do not write the whole task output (bytes " + std::to_string(covered) +
                            ".. are never written, task output " + std::to_string(task_output_bytes) + " bytes)");
    }

    static void compile_glue(const std::string& arch, std::string& cubin, std::string& log) {
        compile_source(glue_kernel_source(), "sdv_bmt_cudnn_plan_glue.cu", arch, cubin, log);
    }

    static void compile_source(const char* source, const char* name, const std::string& arch, std::string& cubin,
                               std::string& log) {
        nvrtcProgram prog;
        if (nvrtcCreateProgram(&prog, source, name, 0, nullptr, nullptr) != NVRTC_SUCCESS)
            throw JsonError(std::string("nvrtcCreateProgram failed: ") + name);
        const char* opts[] = {arch.c_str(), "--std=c++17"};
        nvrtcResult r = nvrtcCompileProgram(prog, 2, opts);
        size_t n = 0;
        nvrtcGetProgramLogSize(prog, &n);
        log.assign(n, '\0');
        if (n) nvrtcGetProgramLog(prog, log.data());
        while (!log.empty() && (log.back() == '\0' || log.back() == '\n')) log.pop_back();
        if (r != NVRTC_SUCCESS) {
            nvrtcDestroyProgram(&prog);
            throw JsonError(std::string("nvrtc compile failed (") + name + "): " + log);
        }
        nvrtcGetCUBINSize(prog, &n);
        cubin.assign(n, '\0');
        nvrtcGetCUBIN(prog, cubin.data());
        nvrtcDestroyProgram(&prog);
    }
};

}  // namespace sdv
