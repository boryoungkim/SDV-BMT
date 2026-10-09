// SDV-BMT Stage 2a-G probe: host cost of each CUDA driver API call in the submission sequence, under
// controlled variations (investigation tool).  For each case, `reps` sequences are issued one at a time
// (each waits for completion before the next), and the host CLOCK_MONOTONIC duration of every call is
// recorded.  Cases vary: the size of the preceding pinned H2D copy, the kernel (reference conformance
// program vs an empty kernel), and whether the copy is on the same stream as the kernel.
//
// Build: g++ -O2 -std=c++17 submit_probe.cpp -I$CUDA/include -L$CUDA/lib64 -lnvrtc -lcuda
#include <cuda.h>
#include <nvrtc.h>
#include <sched.h>
#include <time.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static void ck(CUresult r, const char* w) {
    if (r != CUDA_SUCCESS) {
        const char* n = nullptr;
        cuGetErrorName(r, &n);
        std::fprintf(stderr, "%s: %s\n", w, n ? n : "?");
        std::exit(2);
    }
}
static inline long long now() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static const char* kEmpty = "extern \"C\" __global__ void empty_kernel(const float*, float*, float*, unsigned long long*) {}";

static CUfunction compile(const std::string& src, const char* entry, CUmodule* mod) {
    nvrtcProgram p;
    nvrtcCreateProgram(&p, src.c_str(), "k.cu", 0, nullptr, nullptr);
    const char* opts[] = {"--gpu-architecture=sm_120", "--std=c++17"};
    if (nvrtcCompileProgram(p, 2, opts) != NVRTC_SUCCESS) {
        std::fprintf(stderr, "compile failed\n");
        std::exit(2);
    }
    size_t n;
    nvrtcGetCUBINSize(p, &n);
    std::string cubin(n, '\0');
    nvrtcGetCUBIN(p, cubin.data());
    nvrtcDestroyProgram(&p);
    ck(cuModuleLoadData(mod, cubin.data()), "load");
    CUfunction f;
    ck(cuModuleGetFunction(&f, *mod, entry), "fn");
    return f;
}

int main(int argc, char** argv) {
    const char* out_path = argc > 1 ? argv[1] : "submit_probe.json";
    int reps = argc > 2 ? std::atoi(argv[2]) : 200;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c = 0; c < 16; ++c) CPU_SET(c, &set);
    for (int c = 32; c < 48; ++c) CPU_SET(c, &set);
    sched_setaffinity(0, sizeof set, &set);
    ck(cuInit(0), "init");
    CUdevice dev = -1;
    int n = 0;
    cuDeviceGetCount(&n);
    for (int k = 0; k < n; ++k) {
        CUdevice d;
        cuDeviceGet(&d, k);
        char bus[32];
        cuDeviceGetPCIBusId(bus, sizeof bus, d);
        if (std::string(bus).find("3D:00.0") != std::string::npos || std::string(bus).find("3d:00.0") != std::string::npos)
            dev = d;
    }
    if (dev < 0) {
        std::fprintf(stderr, "GPU 0 (3D:00.0) not found\n");
        return 2;
    }
    CUresult r = cuDevicePrimaryCtxSetFlags(dev, CU_CTX_SCHED_BLOCKING_SYNC);
    if (r != CUDA_SUCCESS && r != CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE) ck(r, "flags");
    CUcontext ctx;
    ck(cuDevicePrimaryCtxRetain(&ctx, dev), "retain");
    ck(cuCtxSetCurrent(ctx), "current");
    std::ifstream f("tools/stage1r_g/sdv_ref_convnet_v1.cu");
    std::stringstream ss;
    ss << f.rdbuf();
    CUmodule m1, m2;
    CUfunction conv = compile(ss.str(), "sdv_ref_convnet_v1", &m1);
    CUfunction empty = compile(kEmpty, "empty_kernel", &m2);
    CUstream s1, s2;
    ck(cuStreamCreate(&s1, CU_STREAM_NON_BLOCKING), "stream");
    ck(cuStreamCreate(&s2, CU_STREAM_NON_BLOCKING), "stream");
    CUevent done;
    ck(cuEventCreate(&done, CU_EVENT_DISABLE_TIMING), "event");
    const size_t big = 3145728;
    CUdeviceptr d_in, d_s0, d_s1, d_out;
    ck(cuMemAlloc(&d_in, big), "a");
    ck(cuMemAlloc(&d_s0, 196608), "a");
    ck(cuMemAlloc(&d_s1, 1048576), "a");
    ck(cuMemAlloc(&d_out, 112), "a");
    void* h_in;
    void* h_out;
    ck(cuMemAllocHost(&h_in, big), "h");
    ck(cuMemAllocHost(&h_out, 112), "h");
    std::fill(static_cast<float*>(h_in), static_cast<float*>(h_in) + big / 4, 0.5f);
    void* args[] = {&d_in, &d_s0, &d_s1, &d_out};

    struct Case {
        const char* name;
        size_t h2d;        // bytes copied before the launch (0 = no copy)
        bool same_stream;  // copy on the kernel's stream
        CUfunction fn;
        bool d2h;
    };
    std::vector<Case> cases = {
        {"conv_no_h2d", 0, true, conv, true},
        {"conv_h2d_4KiB", 4096, true, conv, true},
        {"conv_h2d_256KiB", 262144, true, conv, true},
        {"conv_h2d_3MiB", big, true, conv, true},
        {"conv_h2d_3MiB_other_stream", big, false, conv, true},
        {"empty_no_h2d", 0, true, empty, true},
        {"empty_h2d_3MiB", big, true, empty, true},
        {"conv_h2d_3MiB_no_d2h", big, true, conv, false},
    };
    std::ofstream out(out_path);
    out << "{\"reps\":" << reps << ",\"cases\":{";
    for (size_t c = 0; c < cases.size(); ++c) {
        const Case& k = cases[c];
        std::vector<long long> t_copy, t_launch, t_d2h, t_rec;
        for (int i = 0; i < reps + 10; ++i) {
            long long a = now();
            if (k.h2d) ck(cuMemcpyHtoDAsync(d_in, h_in, k.h2d, k.same_stream ? s1 : s2), "h2d");
            long long b = now();
            ck(cuLaunchKernel(k.fn, 1, 1, 1, 1024, 1, 1, 0, s1, args, nullptr), "launch");
            long long c2 = now();
            if (k.d2h) ck(cuMemcpyDtoHAsync(h_out, d_out, 112, s1), "d2h");
            long long d = now();
            ck(cuEventRecord(done, s1), "rec");
            long long e = now();
            ck(cuCtxSynchronize(), "sync");
            if (i >= 10) {
                t_copy.push_back(b - a);
                t_launch.push_back(c2 - b);
                t_d2h.push_back(d - c2);
                t_rec.push_back(e - d);
            }
        }
        auto q = [](std::vector<long long> v) {
            std::sort(v.begin(), v.end());
            std::ostringstream os;
            os << "{\"p50\":" << v[v.size() / 2] << ",\"p90\":" << v[v.size() * 9 / 10] << ",\"max\":" << v.back()
               << ",\"min\":" << v.front() << "}";
            return os.str();
        };
        out << (c ? "," : "") << "\"" << k.name << "\":{\"h2d_call_ns\":" << q(t_copy)
            << ",\"launch_call_ns\":" << q(t_launch) << ",\"d2h_call_ns\":" << q(t_d2h)
            << ",\"event_record_call_ns\":" << q(t_rec) << "}";
        std::printf("%-28s copy p50 %7lld  launch p50 %7lld p90 %7lld  d2h p50 %6lld  rec p50 %5lld ns\n", k.name,
                    [&] { auto v = t_copy; std::sort(v.begin(), v.end()); return v[v.size() / 2]; }(),
                    [&] { auto v = t_launch; std::sort(v.begin(), v.end()); return v[v.size() / 2]; }(),
                    [&] { auto v = t_launch; std::sort(v.begin(), v.end()); return v[v.size() * 9 / 10]; }(),
                    [&] { auto v = t_d2h; std::sort(v.begin(), v.end()); return v[v.size() / 2]; }(),
                    [&] { auto v = t_rec; std::sort(v.begin(), v.end()); return v[v.size() / 2]; }());
    }
    out << "}}\n";
    cuDevicePrimaryCtxRelease(dev);
    return 0;
}
