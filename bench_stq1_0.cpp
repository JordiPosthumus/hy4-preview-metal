// Bandwidth benchmark: STQ1_0 matvec (decode-style, BS=1) on Metal.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml-metal.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cstdlib>
#include <chrono>

int main(int argc, char ** argv) {
    const int n_per_row = argc > 1 ? atoi(argv[1]) : 6144;
    const int n_rows    = argc > 2 ? atoi(argv[2]) : 2048;
    const int iters     = argc > 3 ? atoi(argv[3]) : 200;

    if (n_per_row <= 0 || n_per_row % 256 != 0 || n_rows <= 0 || iters <= 0) {
        fprintf(stderr, "usage: %s [columns-multiple-of-256] [rows] [iterations]\n", argv[0]);
        return 2;
    }

    std::mt19937 rng(7);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(0.25f, 0.75f);

    const size_t wq_bytes = ggml_row_size(GGML_TYPE_STQ1_0, n_per_row) * (size_t)n_rows;
    std::vector<block_stq1_0> wq(wq_bytes / sizeof(block_stq1_0));
    for (auto & block : wq) {
        for (auto & q : block.qs) q = (uint8_t) byte_dist(rng);
        for (auto & s : block.sign) s = (uint8_t) byte_dist(rng);
        block.d = ggml_fp32_to_fp16(scale_dist(rng));
    }
    printf("quantized size: %.1f MB (%.3f bytes/weight)\n", wq_bytes / 1e6, (double)wq_bytes / ((size_t)n_rows * n_per_row));

    ggml_backend_t be = ggml_backend_metal_init();
    if (!be) { printf("no metal\n"); return 1; }

    struct ggml_init_params ip = { 16 * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_STQ1_0, n_per_row, n_rows);
    const int BS = getenv("BS") ? atoi(getenv("BS")) : 1;
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_per_row, BS);
    ggml_tensor * C = ggml_mul_mat(ctx, A, X);
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, C);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    ggml_backend_tensor_set(A, wq.data(), 0, ggml_nbytes(A));
    std::vector<float> x((size_t)n_per_row*BS, 0.01f);
    ggml_backend_tensor_set(X, x.data(), 0, ggml_nbytes(X));

    // warmup
    for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(be, g);
    ggml_backend_synchronize(be);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(be, g);
    ggml_backend_synchronize(be);
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    double bytes = (double)ggml_nbytes(A) * iters;
    printf("BS=%d matvec %dx%d STQ1_0: %.2f ms/iter, effective BW = %.0f GB/s\n",
           BS, n_rows, n_per_row, sec / iters * 1e3, bytes / sec / 1e9);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    return 0;
}
