// Stage 2b-G real-model baseline, offline validation tool (not part of the measurement plane).
//
// Drives the native NVIDIA backend (native/src/nvidia/backend.hpp, the same code the runner uses) outside the
// benchmark: prepare -> stage the given prepared inputs -> run each item once, sequentially -> write the raw
// task output bytes of every item to OUT.  Used to compare the GPU's raw model outputs with an independent
// framework (ONNX Runtime CPU) and to check the native output pipeline against the Python twin on real
// device outputs.  No timing is taken from this tool.
//
//   raw_dump SPEC.json STAGING.bin ITEMS.json OUT.bin
//     SPEC.json   {"device_uuid", "backend": {"name", "options"}, "program": {...}} (as in the native run spec)
//     ITEMS.json  [{"item_index", "offset", "nbytes", "sha256"}, ...]
//
// Build: g++ -O2 -std=c++17 -I/usr/local/cuda-13.0/include -I<cudnn>/include -Inative/src
//        tools/stage2b_g_real/raw_dump.cpp -o <out> -L/usr/local/cuda-13.0/lib64 -lnvrtc -lcuda -lnvidia-ml -ldl
//        -L<cudnn>/lib -l:libcudnn.so.9 (+ rpaths)
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <string>

#include "nvidia/backend.hpp"

using namespace sdv;

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: raw_dump SPEC.json STAGING.bin ITEMS.json OUT.bin\n");
        return 2;
    }
    try {
        Json spec = Json::parse(read_file(argv[1]));
        Json items = Json::parse(read_file(argv[3]));
        NvidiaBackend be(spec);
        Json facts = be.prepare(spec);
        int fd = ::open(argv[2], O_RDONLY);
        if (fd < 0) throw JsonError("cannot open staging file");
        std::string buf;
        int slot = 0;
        for (const Json& it : items.as_array()) {
            size_t n = size_t(it.at("nbytes").as_int());
            buf.resize(n);
            if (::pread(fd, buf.data(), n, off_t(it.at("offset").as_int())) != ssize_t(n)) throw JsonError("short read");
            ErrorInfo err;
            if (!be.stage(slot++, it.at("item_index").as_int(), buf.data(), n, it.at("sha256").as_string(), err))
                throw JsonError("stage failed: " + err.message);
        }
        ::close(fd);
        std::FILE* out = std::fopen(argv[4], "wb");
        if (!out) throw JsonError("cannot write output");
        for (int s = 0; s < slot; ++s) {
            ErrorInfo err;
            if (!be.submit(s, s, err)) throw JsonError("submit failed: " + err.message);
            int st = 0;
            while ((st = be.poll(err)) == 0) {
            }
            if (st < 0) throw JsonError("device error: " + err.message);
            DeviceFacts f;
            std::string_view raw;
            if (!be.collect(f, raw, err)) throw JsonError("collect failed: " + err.message);
            std::fwrite(raw.data(), 1, raw.size(), out);
        }
        std::fclose(out);
        Json r = Json::object();
        r["ok"] = true;
        r["items"] = slot;
        r["cudnn_algorithm_table_sha256"] =
            facts.has("cudnn_plan") ? facts.at("cudnn_plan").at("cudnn_algorithm_table_sha256") : Json(nullptr);
        r["cleanup"] = be.cleanup();
        std::printf("%s\n", r.dump().c_str());
        return 0;
    } catch (const std::exception& e) {
        std::printf("{\"ok\":false,\"error\":\"%s\"}\n", e.what());
        return 1;
    }
}
