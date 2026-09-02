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
    const int n_mats = argc > 6 ? atoi(argv[6]) : 1;
    const int n_used = argc > 7 ? atoi(argv[7]) : 1;
    const int route_pattern = argc > 8 ? atoi(argv[8]) : 0; // 0 uniform, 1 clustered, 2 spread
    const int route_seed = argc > 9 ? atoi(argv[9]) : 44;

    if (k <= 0 || k % 256 != 0 || m <= 0 || tokens <= 0 ||
        nops <= 0 || nops > 256 || reps <= 0 || n_mats <= 0 ||
        n_used <= 0 || n_used > n_mats || route_pattern < 0 || route_pattern > 2) {
        fprintf(stderr, "usage: %s [k-multiple-of-256] [m] [tokens] [ops-1..256] [repetitions] [logical-experts] [used-experts] [route-pattern: 0=uniform,1=clustered,2=spread] [route-seed]\n", argv[0]);
        return 2;
    }

    std::mt19937 rng(43);
    std::normal_distribution<float> fdist(0.f, 1.f);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(0.25f, 0.75f);

    const size_t row_blocks = ggml_row_size(GGML_TYPE_STQ1_0, k)/sizeof(block_stq1_0);
    // Allocate one expert and alias it across the logical expert dimension.
    // This exercises map/dispatch/empty-expert behavior without allocating the
    // full expert stack. Every logical expert intentionally has identical data.
    std::vector<block_stq1_0> wq((size_t)m*row_blocks);
    for (auto & block : wq) {
        for (auto & q : block.qs) q = (uint8_t)byte_dist(rng);
        for (auto & s : block.sign) s = (uint8_t)byte_dist(rng);
        block.d = ggml_fp32_to_fp16(scale_dist(rng));
    }

    std::vector<float> x((size_t)k*n_used*tokens);
    for (auto & v : x) v = fdist(rng);
    std::vector<int32_t> ids((size_t)n_used*tokens);
    std::mt19937 route_rng(route_seed);
    std::uniform_int_distribution<int> expert_dist(0, n_mats - 1);
    for (int t = 0; t < tokens; ++t) {
        for (int u = 0; u < n_used; ++u) {
            if (route_pattern == 1) {
                ids[(size_t)t*n_used + u] = u % n_mats;
                continue;
            }
            if (route_pattern == 2) {
                ids[(size_t)t*n_used + u] = (t*n_used + u) % n_mats;
                continue;
            }
            int expert;
            bool duplicate;
            do {
                expert = expert_dist(route_rng);
                duplicate = false;
                for (int prior = 0; prior < u; ++prior) {
                    duplicate |= ids[(size_t)t*n_used + prior] == expert;
                }
            } while (duplicate);
            ids[(size_t)t*n_used + u] = expert;
        }
    }

    ggml_backend_t metal = ggml_backend_metal_init();
    if (!metal) { fprintf(stderr, "Metal init failed\n"); return 1; }

    struct ggml_init_params ip = { 16ull*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * A_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_STQ1_0, k, m);
    ggml_tensor * A = ggml_view_3d(ctx, A_storage, k, m, 1, A_storage->nb[1], 0, 0);
    // ggml_view_3d validates a dense span even when nb[2] is zero. Create the
    // valid one-expert alias first, then expose the logical expert count; the
    // backing span remains one expert because every z slice has zero stride.
    A->ne[2] = n_mats;
    ggml_tensor * I = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, tokens);
    ggml_tensor * X = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, tokens);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_tensor * outs[256];
    for (int i = 0; i < nops; ++i) {
        outs[i] = ggml_mul_mat_id(ctx, A, X, I);
        ggml_build_forward_expand(graph, outs[i]);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, metal);
    ggml_backend_tensor_set(A_storage, wq.data(), 0, ggml_nbytes(A_storage));
    ggml_backend_tensor_set(I, ids.data(), 0, ggml_nbytes(I));
    ggml_backend_tensor_set(X, x.data(), 0, ggml_nbytes(X));

    ggml_backend_graph_compute(metal, graph);
    ggml_backend_synchronize(metal);

    std::vector<float> out(ggml_nelements(outs[nops - 1]));
    ggml_backend_tensor_get(outs[nops - 1], out.data(), 0, ggml_nbytes(outs[nops - 1]));

    double max_abs = 0.0;
    const int check_rows = std::min(m, 4);
    const int check_tokens = std::min(tokens, 4);
    const int check_used = std::min(n_used, 4);
    for (int t = 0; t < check_tokens; ++t) {
        for (int u = 0; u < check_used; ++u) {
            for (int row = 0; row < check_rows; ++row) {
                double ref = 0.0;
                const block_stq1_0 * blocks = wq.data() + (size_t)row*row_blocks;
                const float * xv = x.data() + ((size_t)t*n_used + u)*k;
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
                const size_t oi = (size_t)row + (size_t)u*m + (size_t)t*n_used*m;
                const double err = std::fabs((double)out[oi] - ref);
                max_abs = std::max(max_abs, err);
            }
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

    const double bytes = (double)nops*ggml_nbytes(A_storage);
    const char * route_name = route_pattern == 0 ? "uniform" : route_pattern == 1 ? "clustered" : "spread";
    printf("STQ1_0 mul_mat_id %dx%d tokens=%d experts=%d used=%d routes=%s seed=%d x %d ops: max_abs %.4g; best %.4f ms -> %6.1f GB/s (one-expert-equivalent; mean %.4f ms -> %6.1f GB/s)\n",
           m, k, tokens, n_mats, n_used, route_name, route_seed, nops, max_abs, best*1e3, bytes/best/1e9,
           total/reps*1e3, bytes/(total/reps)/1e9);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(metal);
    return 0;
}
