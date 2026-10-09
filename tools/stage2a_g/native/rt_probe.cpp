// SDV-BMT Stage 2a-G probe: candidate C++ measurement plane (investigation tool, not production code).
//
// A fixed-rate (or ASAP) N=1 / queue 0 / drop_new realtime loop on CLOCK_MONOTONIC with a spin wait,
// CUDA driver API submission of the reference conformance program (H2D of the prepared input from pinned
// memory -> kernel -> D2H of the output) on one non-blocking stream, completion observed by polling
// cuEventQuery.  Options: CUDA-event segment brackets (--instr events|none), NVTX request ranges and
// marks (--nvtx 1), and a fake backend without any GPU work (--backend fake) for the harness floor.
// Per-request facts are kept in preallocated memory and written as JSON lines after the loop.
//
// Build: tools/stage2a_g/native/build.sh        Usage: rt_probe --help
#include <cuda.h>
#include <nvrtc.h>
#include <nvtx3/nvToolsExt.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

[[noreturn]] void die(const std::string& what) {
    std::fprintf(stderr, "rt_probe: %s\n", what.c_str());
    std::exit(2);
}

void cu_check(CUresult r, const char* api) {
    if (r != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(r, &name);
        die(std::string(api) + " failed: " + (name ? name : "?") + " (" + std::to_string(int(r)) + ")");
    }
}

inline int64_t clock_ns(clockid_t id) {
    timespec ts;
    clock_gettime(id, &ts);
    return int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}
inline int64_t mono_ns() { return clock_ns(CLOCK_MONOTONIC); }

struct Options {
    std::string backend = "cuda";
    std::string uuid = "GPU-f3e03bfa-61ab-acdf-0834-450de8becefc";
    std::string program = "tools/stage1r_g/sdv_ref_convnet_v1.cu";
    double rate_hz = 100.0;  // 0 = ASAP
    int count = 1000;
    int precondition = 0;
    std::string instr = "events";
    int nvtx = 0;
    std::string cpus = "0-15,32-47";
    int64_t timeout_ns = 1000000000LL;
    int64_t drain_ns = 1000000000LL;
    int64_t lead_ns = 2000000LL;
    int fake_polls = 1;
    int inputs = 4;
    std::string out = "rt_probe.jsonl";
    std::string label;
};

std::vector<int> parse_cpus(const std::string& spec) {
    std::vector<int> out;
    std::stringstream ss(spec);
    std::string part;
    while (std::getline(ss, part, ',')) {
        auto dash = part.find('-');
        if (dash == std::string::npos) {
            out.push_back(std::stoi(part));
        } else {
            for (int c = std::stoi(part.substr(0, dash)); c <= std::stoi(part.substr(dash + 1)); ++c) out.push_back(c);
        }
    }
    return out;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value for " + a);
            return argv[++i];
        };
        if (a == "--help") {
            std::puts("rt_probe [--backend cuda|fake] [--uuid U] [--program F.cu] [--rate HZ (0=asap)] [--count N]\n"
                      "         [--precondition K] [--instr events|none] [--nvtx 0|1] [--cpus 0-15,32-47]\n"
                      "         [--timeout-ns T] [--drain-ns D] [--lead-ns L] [--fake-polls P] [--inputs M]\n"
                      "         [--label S] --out FILE.jsonl");
            std::exit(0);
        } else if (a == "--backend") o.backend = next();
        else if (a == "--uuid") o.uuid = next();
        else if (a == "--program") o.program = next();
        else if (a == "--rate") o.rate_hz = std::stod(next());
        else if (a == "--count") o.count = std::stoi(next());
        else if (a == "--precondition") o.precondition = std::stoi(next());
        else if (a == "--instr") o.instr = next();
        else if (a == "--nvtx") o.nvtx = std::stoi(next());
        else if (a == "--cpus") o.cpus = next();
        else if (a == "--timeout-ns") o.timeout_ns = std::stoll(next());
        else if (a == "--drain-ns") o.drain_ns = std::stoll(next());
        else if (a == "--lead-ns") o.lead_ns = std::stoll(next());
        else if (a == "--fake-polls") o.fake_polls = std::stoi(next());
        else if (a == "--inputs") o.inputs = std::stoi(next());
        else if (a == "--out") o.out = next();
        else if (a == "--label") o.label = next();
        else die("unknown option " + a);
    }
    if (o.backend != "cuda" && o.backend != "fake") die("--backend must be cuda or fake");
    if (o.instr != "events" && o.instr != "none") die("--instr must be events or none");
    if (o.count < 1 || o.inputs < 1 || o.fake_polls < 1) die("--count, --inputs, --fake-polls must be >= 1");
    return o;
}

// ------------------------------------------------------------------ NVTX (sdv-bmt domain, payload = request id)
struct Nvtx {
    bool on = false;
    nvtxDomainHandle_t domain = nullptr;
    nvtxStringHandle_t submit = nullptr, complete = nullptr, drop = nullptr;
    void init(bool enable) {
        on = enable;
        if (!on) return;
        domain = nvtxDomainCreateA("sdv-bmt");
        submit = nvtxDomainRegisterStringA(domain, "sdv-bmt.submit");
        complete = nvtxDomainRegisterStringA(domain, "sdv-bmt.complete");
        drop = nvtxDomainRegisterStringA(domain, "sdv-bmt.drop");
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
    void push_submit(int64_t id) const {
        if (!on) return;
        auto a = attr(submit, id);
        nvtxDomainRangePushEx(domain, &a);
    }
    void pop() const {
        if (on) nvtxDomainRangePop(domain);
    }
    void mark(nvtxStringHandle_t msg, int64_t id) const {
        if (!on) return;
        auto a = attr(msg, id);
        nvtxDomainMarkEx(domain, &a);
    }
};

// ------------------------------------------------------------------ backends
struct Facts {
    int64_t polls = 0;
    int64_t h2d_ns = -1, kernel_ns = -1, d2h_ns = -1;  // CUDA-event brackets (device clock), -1 = not taken
    uint64_t gt0 = 0, gt1 = 0, c0 = 0, c1 = 0;         // in-kernel %globaltimer / clock64 words
    int error = 0;
    int64_t api[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // host CLOCK_MONOTONIC after each submit API call (absolute)
    int n_api = 0;
};

constexpr size_t kInputBytes = 3145728, kOutputBytes = 112, kTimingOffset = 80;
constexpr size_t kScratch0 = 196608, kScratch1 = 1048576;

struct CudaBackend {
    CUdevice dev{};
    CUcontext ctx{};
    CUmodule module{};
    CUfunction fn{};
    CUstream stream{};
    CUevent ev_start{}, ev_h2d{}, ev_kernel{}, ev_done{};
    CUdeviceptr d_in = 0, d_s0 = 0, d_s1 = 0, d_out = 0;
    void* h_out = nullptr;
    std::vector<void*> h_inputs;
    bool events = true;
    std::string cubin_sha;  // not computed in the probe; size recorded instead
    size_t cubin_size = 0;
    std::string name;

    static std::string read_file(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) die("cannot read " + path);
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    void init(const Options& o) {
        events = o.instr == "events";
        cu_check(cuInit(0), "cuInit");
        int n = 0;
        cu_check(cuDeviceGetCount(&n), "cuDeviceGetCount");
        bool found = false;
        for (int k = 0; k < n && !found; ++k) {
            CUdevice d;
            cu_check(cuDeviceGet(&d, k), "cuDeviceGet");
            CUuuid u;
            cu_check(cuDeviceGetUuid_v2(&u, d), "cuDeviceGetUuid_v2");
            char buf[64];
            const unsigned char* b = reinterpret_cast<const unsigned char*>(u.bytes);
            std::snprintf(buf, sizeof buf,
                          "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0], b[1], b[2],
                          b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
            if (o.uuid == buf) {
                dev = d;
                found = true;
            }
        }
        if (!found) die("no CUDA device with UUID " + o.uuid);
        char nm[128];
        cu_check(cuDeviceGetName(nm, sizeof nm, dev), "cuDeviceGetName");
        name = nm;
        CUresult r = cuDevicePrimaryCtxSetFlags(dev, CU_CTX_SCHED_BLOCKING_SYNC);
        if (r != CUDA_SUCCESS && r != CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE) cu_check(r, "cuDevicePrimaryCtxSetFlags");
        cu_check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
        cu_check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
        int major = 0, minor = 0;
        cu_check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "cc major");
        cu_check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "cc minor");
        std::string source = read_file(o.program);
        nvrtcProgram prog;
        if (nvrtcCreateProgram(&prog, source.c_str(), "sdv_bmt_device_program.cu", 0, nullptr, nullptr) !=
            NVRTC_SUCCESS)
            die("nvrtcCreateProgram failed");
        std::string arch = "--gpu-architecture=sm_" + std::to_string(major) + std::to_string(minor);
        const char* opts[] = {arch.c_str(), "--std=c++17"};
        if (nvrtcCompileProgram(prog, 2, opts) != NVRTC_SUCCESS) {
            size_t n_log = 0;
            nvrtcGetProgramLogSize(prog, &n_log);
            std::string log(n_log, '\0');
            nvrtcGetProgramLog(prog, log.data());
            die("nvrtc compile failed: " + log);
        }
        nvrtcGetCUBINSize(prog, &cubin_size);
        std::string cubin(cubin_size, '\0');
        nvrtcGetCUBIN(prog, cubin.data());
        nvrtcDestroyProgram(&prog);
        cu_check(cuModuleLoadData(&module, cubin.data()), "cuModuleLoadData");
        cu_check(cuModuleGetFunction(&fn, module, "sdv_ref_convnet_v1"), "cuModuleGetFunction");
        cu_check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
        unsigned flags = events ? CU_EVENT_DEFAULT : CU_EVENT_DISABLE_TIMING;
        if (events) {
            cu_check(cuEventCreate(&ev_start, flags), "cuEventCreate");
            cu_check(cuEventCreate(&ev_h2d, flags), "cuEventCreate");
            cu_check(cuEventCreate(&ev_kernel, flags), "cuEventCreate");
        }
        cu_check(cuEventCreate(&ev_done, flags), "cuEventCreate");
        cu_check(cuMemAlloc(&d_in, kInputBytes), "cuMemAlloc");
        cu_check(cuMemAlloc(&d_s0, kScratch0), "cuMemAlloc");
        cu_check(cuMemAlloc(&d_s1, kScratch1), "cuMemAlloc");
        cu_check(cuMemAlloc(&d_out, kOutputBytes), "cuMemAlloc");
        cu_check(cuMemAllocHost(&h_out, kOutputBytes), "cuMemAllocHost");
        // synthetic prepared inputs: values k/128, constant on 4x4 blocks (the program's input contract)
        for (int m = 0; m < o.inputs; ++m) {
            void* h = nullptr;
            cu_check(cuMemAllocHost(&h, kInputBytes), "cuMemAllocHost");
            float* f = static_cast<float*>(h);
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < 512; ++y)
                    for (int x = 0; x < 512; ++x) {
                        int k = ((c * 7 + (y / 4) * 3 + (x / 4) * 5 + m * 11) % 256) - 128;
                        f[(size_t(c) * 512 + y) * 512 + x] = float(k) * 0.0078125f;
                    }
            h_inputs.push_back(h);
        }
    }

    void submit(int64_t request, Facts& f) {
        void* h = h_inputs[size_t(request >= 0 ? request : -request) % h_inputs.size()];
        f.n_api = 0;
        auto mark = [&]() { f.api[f.n_api++] = mono_ns(); };
        if (events) {
            cu_check(cuEventRecord(ev_start, stream), "cuEventRecord");
            mark();
        }
        cu_check(cuMemcpyHtoDAsync(d_in, h, kInputBytes, stream), "cuMemcpyHtoDAsync");
        mark();
        if (events) {
            cu_check(cuEventRecord(ev_h2d, stream), "cuEventRecord");
            mark();
        }
        void* args[] = {&d_in, &d_s0, &d_s1, &d_out};
        cu_check(cuLaunchKernel(fn, 1, 1, 1, 1024, 1, 1, 0, stream, args, nullptr), "cuLaunchKernel");
        mark();
        if (events) {
            cu_check(cuEventRecord(ev_kernel, stream), "cuEventRecord");
            mark();
        }
        cu_check(cuMemcpyDtoHAsync(h_out, d_out, kOutputBytes, stream), "cuMemcpyDtoHAsync");
        mark();
        cu_check(cuEventRecord(ev_done, stream), "cuEventRecord");
        mark();
    }

    // 1 = complete, 0 = not yet, -1 = error
    int poll(Facts& f) {
        ++f.polls;
        CUresult r = cuEventQuery(ev_done);
        if (r == CUDA_ERROR_NOT_READY) return 0;
        if (r != CUDA_SUCCESS) {
            f.error = int(r);
            return -1;
        }
        return 1;
    }

    void collect(Facts& f) {
        if (events) {
            float ms = 0;
            cu_check(cuEventElapsedTime(&ms, ev_start, ev_h2d), "elapsed");
            f.h2d_ns = llround(double(ms) * 1e6);
            cu_check(cuEventElapsedTime(&ms, ev_h2d, ev_kernel), "elapsed");
            f.kernel_ns = llround(double(ms) * 1e6);
            cu_check(cuEventElapsedTime(&ms, ev_kernel, ev_done), "elapsed");
            f.d2h_ns = llround(double(ms) * 1e6);
        }
        const uint64_t* w = reinterpret_cast<const uint64_t*>(static_cast<const char*>(h_out) + kTimingOffset);
        f.gt0 = w[0];
        f.gt1 = w[1];
        f.c0 = w[2];
        f.c1 = w[3];
    }

    bool stream_idle() { return cuStreamQuery(stream) == CUDA_SUCCESS; }

    void cleanup() {
        cuStreamSynchronize(stream);
        for (void* h : h_inputs) cuMemFreeHost(h);
        cuMemFreeHost(h_out);
        cuMemFree(d_in);
        cuMemFree(d_s0);
        cuMemFree(d_s1);
        cuMemFree(d_out);
        cuStreamDestroy(stream);
        cuModuleUnload(module);
        cuDevicePrimaryCtxRelease(dev);
    }
};

struct FakeBackend {  // no device: completion is reported by the n-th poll (harness floor)
    int polls_needed = 1;
    int left = 0;
    void init(const Options& o) { polls_needed = o.fake_polls; }
    void submit(int64_t, Facts&) { left = polls_needed; }
    int poll(Facts& f) {
        ++f.polls;
        return --left <= 0 ? 1 : 0;
    }
    void collect(Facts&) {}
    bool stream_idle() { return true; }
    void cleanup() {}
};

// ------------------------------------------------------------------ realtime loop
enum State { PLANNED = 0, INFLIGHT, COMPLETED, ERROR, TIMED_OUT, DROPPED };
const char* state_name(int s) {
    static const char* n[] = {"planned", "inflight", "completed", "error", "timed_out", "dropped"};
    return n[s];
}

struct Rec {
    int64_t id = 0;
    int state = PLANNED;
    int late = 0;
    int64_t t_sched = 0, t_issue = -1, t_submit_begin = -1, t_submit_end = -1, t_complete = -1, t_timed_out = -1;
    Facts f;
};

template <class Backend>
struct Loop {
    Backend& be;
    const Options& o;
    Nvtx& nvtx;
    int64_t anchor;
    std::vector<Rec> recs;
    int64_t inflight = -1;
    int64_t deadline = 0;
    int64_t measure_begin = 0, measure_end = 0, drain_end = 0, cap = 0;
    bool cap_reached = false;

    Loop(Backend& b, const Options& opt, Nvtx& n, int64_t a) : be(b), o(opt), nvtx(n), anchor(a) {
        recs.resize(size_t(o.count));
    }
    int64_t now() const { return mono_ns() - anchor; }

    void service(int64_t t) {
        Rec& r = recs[size_t(inflight)];
        int s = be.poll(r.f);
        if (s != 0) {
            int64_t t_complete = now();
            if (s > 0) be.collect(r.f);
            nvtx.mark(nvtx.complete, r.id);
            if (r.state == INFLIGHT) {
                r.state = s > 0 ? COMPLETED : ERROR;
            } else {
                r.late = 1;  // TIMED_OUT stays TIMED_OUT
            }
            r.t_complete = t_complete;
            inflight = -1;
        } else if (t >= deadline && r.state == INFLIGHT) {
            r.state = TIMED_OUT;
            r.t_timed_out = now();
        }
    }

    void admit(int64_t i, int64_t t_sched) {
        Rec& r = recs[size_t(i)];
        r.id = i;
        r.t_sched = t_sched;
        int64_t t = now();
        r.t_issue = t;
        if (inflight >= 0) {
            r.state = DROPPED;
            nvtx.mark(nvtx.drop, i);
            return;
        }
        r.t_submit_begin = now();
        nvtx.push_submit(i);
        be.submit(i, r.f);
        nvtx.pop();
        r.t_submit_end = now();
        r.state = INFLIGHT;
        inflight = i;
        deadline = r.t_submit_begin + o.timeout_ns;
    }

    void run() {
        measure_begin = now();
        int64_t t0 = measure_begin + o.lead_ns;
        if (o.rate_hz > 0) {
            for (int64_t i = 0; i < o.count; ++i) {
                int64_t t_sched = t0 + int64_t(std::floor(double(i) * 1e9 / o.rate_hz));
                while (true) {
                    int64_t t = now();
                    if (inflight >= 0) service(t);
                    if (t >= t_sched) break;
                }
                admit(i, t_sched);
            }
        } else {
            while (now() < t0) {
            }
            for (int64_t i = 0; i < o.count; ++i) {
                while (inflight >= 0) service(now());
                admit(i, now());
            }
        }
        measure_end = now();
        cap = measure_end + o.drain_ns;
        while (inflight >= 0) {
            int64_t t = now();
            service(t);
            if (t >= cap) break;
        }
        cap_reached = inflight >= 0;
        drain_end = now();
    }
};

// ------------------------------------------------------------------ micro costs (before the loop)
template <class Backend>
void micro(Backend&, std::ostream& os, const Nvtx& nvtx) {
    const int n = 1000000;
    int64_t a = mono_ns(), sink = 0;
    for (int k = 0; k < n; ++k) sink += mono_ns();
    int64_t b = mono_ns();
    char line[512];
    int64_t c = mono_ns();
    for (int k = 0; k < 100000; ++k) {
        sink += std::snprintf(line, sizeof line,
                              "{\"kind\":\"completed\",\"request_id\":%d,\"run_clock_domain\":\"monotonic\","
                              "\"schema_version\":2,\"seq\":%d,\"t_complete\":%lld,\"t_output_ready\":%lld,"
                              "\"t_run_ns\":%lld}",
                              k, k + 40, (long long)(a + k), (long long)(a + k + 9), (long long)(a + k + 9));
    }
    int64_t d = mono_ns();
    double nvtx_ns = -1;
    if (nvtx.on) {  // one submit range (push + pop) and one mark, as issued per request
        int64_t e = mono_ns();
        for (int k = 0; k < 10000; ++k) {
            nvtx.push_submit(-1000000 - k);
            nvtx.pop();
            nvtx.mark(nvtx.complete, -1000000 - k);
        }
        nvtx_ns = double(mono_ns() - e) / 10000;
    }
    os << "\"micro\":{\"clock_gettime_monotonic_ns\":" << double(b - a) / n
       << ",\"snprintf_event_line_ns\":" << double(d - c) / 100000 << ",\"nvtx_range_and_mark_ns\":" << nvtx_ns
       << ",\"sink\":" << (sink & 1) << "}";
}

std::string clocks_json() {
    std::ostringstream os;
    int64_t m0 = clock_ns(CLOCK_MONOTONIC), raw = clock_ns(CLOCK_MONOTONIC_RAW), rt = clock_ns(CLOCK_REALTIME),
            m1 = clock_ns(CLOCK_MONOTONIC);
    os << "{\"monotonic_ns\":" << m0 << ",\"monotonic_raw_ns\":" << raw << ",\"realtime_ns\":" << rt
       << ",\"monotonic_after_ns\":" << m1 << "}";
    return os.str();
}

template <class Backend>
int run_with(Backend& be, const Options& o) {
    Nvtx nvtx;
    nvtx.init(o.nvtx != 0);
    std::vector<int> cpus = parse_cpus(o.cpus);
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus) CPU_SET(c, &set);
    if (sched_setaffinity(0, sizeof set, &set) != 0) die("sched_setaffinity failed");
    int64_t t_init0 = mono_ns();
    be.init(o);
    int64_t t_init1 = mono_ns();
    std::ostringstream header;
    header << "{\"kind\":\"probe_header\",\"label\":\"" << o.label << "\",\"backend\":\"" << o.backend
           << "\",\"rate_hz\":" << o.rate_hz << ",\"count\":" << o.count << ",\"precondition\":" << o.precondition
           << ",\"instr\":\"" << o.instr << "\",\"nvtx\":" << o.nvtx << ",\"cpus\":\"" << o.cpus
           << "\",\"timeout_ns\":" << o.timeout_ns << ",\"drain_ns\":" << o.drain_ns << ",\"lead_ns\":" << o.lead_ns
           << ",\"pid\":" << getpid() << ",\"init_ns\":" << (t_init1 - t_init0) << ",";
    micro(be, header, nvtx);
    // untimed preconditioning (negative ids)
    for (int k = 0; k < o.precondition; ++k) {
        Facts f;
        be.submit(-(k + 1), f);
        while (be.poll(f) == 0) {
        }
    }
    bool idle = be.stream_idle();
    std::string clocks_begin = clocks_json();
    int64_t anchor = mono_ns();
    Loop<Backend> loop(be, o, nvtx, anchor);
    loop.run();
    std::string clocks_end = clocks_json();
    be.cleanup();
    std::ofstream out(o.out);
    if (!out) die("cannot write " + o.out);
    header << ",\"anchor_monotonic_ns\":" << anchor << ",\"clocks_begin\":" << clocks_begin
           << ",\"clocks_end\":" << clocks_end << ",\"stream_idle_at_measure_begin\":" << (idle ? "true" : "false")
           << ",\"measure_begin\":" << loop.measure_begin << ",\"measure_end\":" << loop.measure_end
           << ",\"cap\":" << loop.cap << ",\"drain_end\":" << loop.drain_end
           << ",\"cap_reached\":" << (loop.cap_reached ? "true" : "false") << "}";
    out << header.str() << "\n";
    for (const Rec& r : loop.recs) {
        out << "{\"kind\":\"request\",\"id\":" << r.id << ",\"state\":\"" << state_name(r.state)
            << "\",\"late\":" << r.late << ",\"t_sched\":" << r.t_sched << ",\"t_issue\":" << r.t_issue
            << ",\"t_submit_begin\":" << r.t_submit_begin << ",\"t_submit_end\":" << r.t_submit_end
            << ",\"t_complete\":" << r.t_complete << ",\"t_timed_out\":" << r.t_timed_out << ",\"polls\":" << r.f.polls
            << ",\"h2d_ns\":" << r.f.h2d_ns << ",\"kernel_ns\":" << r.f.kernel_ns << ",\"d2h_ns\":" << r.f.d2h_ns
            << ",\"gt0\":" << r.f.gt0 << ",\"gt1\":" << r.f.gt1 << ",\"c0\":" << r.f.c0 << ",\"c1\":" << r.f.c1
            << ",\"error\":" << r.f.error << ",\"api_after_ns\":[";
        for (int k = 0; k < r.f.n_api; ++k) out << (k ? "," : "") << (r.f.api[k] - anchor);
        out << "]}\n";
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Options o = parse(argc, argv);
    if (o.backend == "cuda") {
        CudaBackend be;
        return run_with(be, o);
    }
    FakeBackend be;
    return run_with(be, o);
}
