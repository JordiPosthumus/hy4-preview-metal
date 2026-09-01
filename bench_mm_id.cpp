// Low-memory correctness and throughput harness for STQ1_0 mul_mm_id.
// Uses one synthetic expert, so it exercises the routed-expert Metal kernel
// without allocating Hy4's full 256-expert tensor or loading the model.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-metal.h"
#include "ggml-quants.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static const uint8_t stq_codebook[32] = {
    0xA9,0x89,0x29,0x09,0xA6,0x86,0x26,0x06,
    0x9A,0x92,0x1A,0x12,0x6A,0x62,0x4A,0x42,
    0x01,0x21,0x81,0xA1,0x04,0x24,0x84,0xA4,
    0x10,0x18,0x90,0x98,0x40,0x48,0x60,0x68,
};

int main(int argc, char ** argv) {
    const int k      = argc > 1 ? atoi(argv[1]) : 6144;
    const int m      = argc > 2 ? atoi(argv[2]) : 2048;
    const int tokens = argc > 3 ? atoi(argv[3]) : 64;
    const int nops   = argc > 4 ? atoi(argv[4]) : 32;
    const int reps   = argc > 5 ? atoi(argv[5]) : 10;

    if (k <= 0 || k % 256 != 0 || m <= 0 || tokens <= 0 ||
        nops <= 0 || nops > 256 || reps <= 0) {
        fprintf(stderr, "usage: %s [k-multiple-of-256] [m] [tokens] [ops-1..256] [repetitions]\n", argv[0]);
        return 2;
    }

    constexpr int n_mats = 1;
    constexpr int n_used = 1;
    std::mt19937 rng(43);
    std::normal_distribution<float> fdist(0.f, 1.f);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(0.25f, 0.75f);

    const size_t row_blocks = ggml_row_size(GGML_TYPE_STQ1_0, k)/sizeof(block_stq1_0);
    std::vector<block_stq1_0> wq((size_t)n_mats*m*row_blocks);
    for (auto & block : wq) {
        for (auto & q : block.qs) q = (uint8_t)byte_dist(rng);
        for (auto & s : block.sign) s = (uint8_t)byte_dist(rng);
        block.d = ggml_fp32_to_fp16(scale_dist(rng));
    }

    std::vector<float> x((size_t)k*n_used*tokens);
    for (auto & v : x) v = fdist(rng);
    std::vector<int32_t> ids((size_t)n_used*tokens, 0);

    ggml_backend_t metal = ggml_backend_metal_init();
    if (!metal) { fprintf(stderr, "Metal init failed\n"); return 1; }

    struct ggml_init_params ip = { 16ull*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * A = ggml_new_tensor_3d(ctx, GGML_TYPE_STQ1_0, k, m, n_mats);
    ggml_tensor * I = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, tokens);
    ggml_tensor * X = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, tokens);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_tensor * outs[256];
    for (int i = 0; i < nops; ++i) {
        outs[i] = ggml_mul_mat_id(ctx, A, X, I);
        ggml_build_forward_expand(graph, outs[i]);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, metal);
    ggml_backend_tensor_set(A, wq.data(), 0, ggml_nbytes(A));
    ggml_backend_tensor_set(I, ids.data(), 0, ggml_nbytes(I));
    ggml_backend_tensor_set(X, x.data(), 0, ggml_nbytes(X));

    ggml_backend_graph_compute(metal, graph);
    ggml_backend_synchronize(metal);

    std::vector<float> out(ggml_nelements(outs[nops - 1]));
    ggml_backend_tensor_get(outs[nops - 1], out.data(), 0, ggml_nbytes(outs[nops - 1]));

    double max_abs = 0.0;
    const int check_rows = std::min(m, 4);
    const int check_tokens = std::min(tokens, 4);
    for (int t = 0; t < check_tokens; ++t) {
        for (int row = 0; row < check_rows; ++row) {
            double ref = 0.0;
            const block_stq1_0 * blocks = wq.data() + (size_t)row*row_blocks;
            const float * xv = x.data() + (size_t)t*k;
            for (size_t ib = 0; ib < row_blocks; ++ib) {
                const block_stq1_0 & block = blocks[ib];
                const float d = ggml_fp16_to_fp32(block.d);
                for (int g = 0; g < 64; ++g) {
                    const uint8_t q = block.qs[g >> 1];
                    const uint8_t code = (g & 1) ? (q >> 4) : (q & 0x0f);
                    const uint8_t sign = (block.sign[g >> 3] >> (g & 7)) & 1;
                    const uint8_t packed = stq_codebook[(sign << 4) | code];
                    const int base = (int)ib*256 + (g/16)*64 + (g%16);
                    for (int lane = 0; lane < 4; ++lane) {
                        const int level = ((packed >> (2*lane)) & 3) - 1;
                        ref += (double)(d*level)*(double)xv[base + lane*16];
                    }
                }
            }
            const double err = std::fabs((double)out[(size_t)t*m + row] - ref);
            max_abs = std::max(max_abs, err);
        }
    }

    for (int i = 0; i < 6; ++i) ggml_backend_graph_compute(metal, graph);
    ggml_backend_synchronize(metal);

    double best = 1e18;
    double total = 0.0;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(metal, graph);
        ggml_backend_synchronize(metal);
        const auto t1 = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double>(t1 - t0).count();
        best = std::min(best, dt);
        total += dt;
    }

    const double bytes = (double)nops*ggml_nbytes(A);
    printf("STQ1_0 mul_mat_id %dx%d tokens=%d x %d ops: max_abs %.4g; best %.4f ms -> %6.1f GB/s (mean %.4f ms -> %6.1f GB/s)\n",
           m, k, tokens, nops, max_abs, best*1e3, bytes/best/1e9,
           total/reps*1e3, bytes/(total/reps)/1e9);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(metal);
    return 0;
}
