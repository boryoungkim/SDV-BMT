// SDV-BMT reference accelerator conformance program v1 (not a task model).
// in     : float32 [3][512][512] CHW; values k/128, k integer in [-128, 127], constant on 4x4 blocks
// pooled : float32 scratch [3][128][128];  f1 : float32 scratch [16][128][128]
// out    : int64 logits[10] (bytes 0..79), then uint64 timing[4]:
//          %globaltimer at start/end and clock64 at start/end, read by thread 0
// One block of 1024 threads.  Every intermediate value is exactly representable,
// so the result is bit-identical on every run and equal to the integer CPU reference.
#define IH 512
#define IW 512
#define PH 128
#define PW 128
#define C1 16
#define C2 16
#define NCLS 10

__device__ __forceinline__ int w1(int o, int c, int ky, int kx) { return ((o * 131 + c * 31 + ky * 7 + kx * 3) % 7) - 3; }
__device__ __forceinline__ int b1(int o) { return (o % 5) - 2; }
__device__ __forceinline__ int w2(int o, int c, int ky, int kx) { return ((o * 17 + c * 13 + ky * 5 + kx * 11) % 7) - 3; }
__device__ __forceinline__ int b2(int o) { return (o % 3) - 1; }
__device__ __forceinline__ int wf(int j, int o) { return ((j * 29 + o * 23) % 9) - 4; }
__device__ __forceinline__ int bf(int j) { return j - 5; }

extern "C" __global__ void __launch_bounds__(1024)
sdv_ref_convnet_v1(const float* __restrict__ in, float* __restrict__ pooled, float* __restrict__ f1,
                   unsigned long long* __restrict__ out)
{
    __shared__ unsigned long long gap[C2];
    const int tid = threadIdx.x, nt = blockDim.x;
    unsigned long long t0 = 0ull, t1 = 0ull;
    long long c0 = 0, c1 = 0;
    if (tid == 0) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        c0 = clock64();
    }
    if (tid < C2) gap[tid] = 0ull;
    // 1. 4x4 average pool [3][512][512] -> [3][128][128]
    for (int i = tid; i < 3 * PH * PW; i += nt) {
        const int c = i / (PH * PW), y = (i / PW) % PH, x = i % PW;
        const float* p = in + ((size_t)c * IH + 4 * y) * IW + 4 * x;
        float s = 0.0f;
        for (int dy = 0; dy < 4; ++dy)
            for (int dx = 0; dx < 4; ++dx) s += p[dy * IW + dx];
        pooled[i] = s * 0.0625f;
    }
    __syncthreads();
    // 2. conv 3->16, 3x3, zero padding, bias, ReLU
    for (int i = tid; i < C1 * PH * PW; i += nt) {
        const int o = i / (PH * PW), y = (i / PW) % PH, x = i % PW;
        float s = (float)b1(o) * 0.0078125f;
        for (int c = 0; c < 3; ++c)
            for (int ky = 0; ky < 3; ++ky)
                for (int kx = 0; kx < 3; ++kx) {
                    const int yy = y + ky - 1, xx = x + kx - 1;
                    if (yy >= 0 && yy < PH && xx >= 0 && xx < PW)
                        s += pooled[(c * PH + yy) * PW + xx] * (float)w1(o, c, ky, kx);
                }
        f1[i] = s > 0.0f ? s : 0.0f;
    }
    __syncthreads();
    // 3. conv 16->16, 3x3, zero padding, bias, ReLU, reduced to per-channel integer sums
    for (int o = 0; o < C2; ++o) {
        unsigned long long acc = 0ull;
        for (int p = tid; p < PH * PW; p += nt) {
            const int y = p / PW, x = p % PW;
            float s = (float)b2(o) * 0.0078125f;
            for (int c = 0; c < C1; ++c)
                for (int ky = 0; ky < 3; ++ky)
                    for (int kx = 0; kx < 3; ++kx) {
                        const int yy = y + ky - 1, xx = x + kx - 1;
                        if (yy >= 0 && yy < PH && xx >= 0 && xx < PW)
                            s += f1[(c * PH + yy) * PW + xx] * (float)w2(o, c, ky, kx);
                    }
            if (s > 0.0f) acc += (unsigned long long)(long long)(s * 128.0f);
        }
        atomicAdd(&gap[o], acc);  // integer addition: order-free and exact
    }
    __syncthreads();
    // 4. fully connected 16 -> 10 (int64)
    if (tid < NCLS) {
        long long s = (long long)bf(tid);
        for (int o = 0; o < C2; ++o) s += (long long)gap[o] * (long long)wf(tid, o);
        out[tid] = (unsigned long long)s;
    }
    __syncthreads();
    if (tid == 0) {
        c1 = clock64();
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        out[NCLS + 0] = t0;
        out[NCLS + 1] = t1;
        out[NCLS + 2] = (unsigned long long)c0;
        out[NCLS + 3] = (unsigned long long)c1;
    }
}
