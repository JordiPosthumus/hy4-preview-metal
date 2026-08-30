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
#include <chrono>

int main() {
    const int n_per_row = 16384;
    const int n_rows    = 16384;
    const int iters     = 30;

    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.f, 1.f);

    std::vector<float> w((size_t)n_rows * n_per_row);
    for (auto & v : w) v = dist(rng);
    const size_t wq_bytes = ggml_row_size(GGML_TYPE_STQ1_0, n_per_row) * (size_t)n_rows;
    std::vector<block_stq1_0> wq(wq_bytes / sizeof(block_stq1_0));
    printf("quantizing %d x %d ...\n", n_rows, n_per_row);
    quantize_stq1_0(w.data(), wq.data(), n_rows, n_per_row, nullptr);
    printf("quantized size: %.1f MB (%.3f bytes/weight)\n", wq_bytes / 1e6, (double)wq_bytes / ((size_t)n_rows * n_per_row));

    ggml_backend_t be = ggml_backend_metal_init();
    if (!be) { printf("no metal\n"); return 1; }

    struct ggml_init_params ip = { 512 * 1024 * 1024, NULL, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_STQ1_0, n_per_row, n_rows);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_per_row, 1);
    ggml_tensor * C = ggml_mul_mat(ctx, A, X);
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, C);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    ggml_backend_tensor_set(A, wq.data(), 0, ggml_nbytes(A));
    std::vector<float> x(n_per_row, 0.01f);
    ggml_backend_tensor_set(X, x.data(), 0, ggml_nbytes(X));

    // warmup
    for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(be, g);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(be, g);
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    double bytes = (double)ggml_nbytes(A) * iters;
    printf("matvec %dx%d STQ1_0: %.2f ms/iter, effective BW = %.0f GB/s\n",
           n_rows, n_per_row, sec / iters * 1e3, bytes / sec / 1e9);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    return 0;
}
