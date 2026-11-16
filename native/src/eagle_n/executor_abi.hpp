// Eagle-N-only ABI: measured callbacks use caller-owned storage; no Python bridge.
#pragma once
#include <cstddef>
#include <cstdint>
namespace sdv::eagle {
constexpr uint32_t kExecutorABI = 1;
struct ExecutorAPI {
    uint32_t abi;
    bool real_device;
    const char* (*describe)();                       // JSON, outside MEASURE; static lifetime
    void* (*prepare)(const char* config, char* error, size_t cap);
    bool (*stage)(void*, int slot, const void*, size_t, char*, size_t);
    bool (*run)(void*, int slot, uint32_t* output, size_t pixels, char*, size_t); // synchronous, includes D2H
    bool (*destroy)(void*, char*, size_t);
};
using GetExecutor = const ExecutorAPI* (*)();
} // namespace sdv::eagle
