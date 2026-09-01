// Timing harness for the STQ1_0 mul_mm (prefill / batched-decode) path on Metal.
// Builds N identical mul_mat ops sharing one STQ1_0 weight matrix + one F32 input,
// and times the whole graph. As N grows the per-op host overhead amortizes and the
// wall time converges on aggregate GPU throughput: GB/s = N * weight_bytes / wall.
// Model weights are NOT loaded.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-metal.h"
#include "ggml-quants.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

int main(int argc, char ** argv) {
    const int ne00  = argc > 1 ? atoi(argv[1]) : 6144;   // columns
    const int ne01  = argc > 2 ? atoi(argv[2]) : 2048;   // rows
    const int batch = argc > 3 ? atoi(argv[3]) : 64;     // ne11 (batch)
    const int nops  = argc > 4 ? atoi(argv[4]) : 32;     // identical ops in graph
    const int reps  = argc > 5 ? atoi(argv[5]) : 10;

    if (ne00 <= 0 || ne00 % 256 != 0 || ne01 <= 0 || batch <= 0 ||
        nops <= 0 || nops > 256 || reps <= 0) {
        fprintf(stderr, "usage: %s [columns-multiple-of-256] [rows] [batch] [ops-1..256] [repetitions]\n", argv[0]);
        return 2;
    }

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(0.25f, 0.75f);

    std::vector<block_stq1_0> wq((size_t)ne01 * ggml_row_size(GGML_TYPE_STQ1_0, ne00) / sizeof(block_stq1_0));
    for (auto & block : wq) {
        for (auto & q : block.qs) q = (uint8_t) byte_dist(rng);
        for (auto & s : block.sign) s = (uint8_t) byte_dist(rng);
        block.d = ggml_fp32_to_fp16(scale_dist(rng));
    }
    std::vector<float> x((size_t)batch * ne00);
    for (auto & v : x) v = dist(rng);

    ggml_backend_t mb = ggml_backend_metal_init();
    if (!mb) { printf("Metal init failed\n"); return 1; }

    struct ggml_init_params ip = { 16ull*1024*1024, NULL, true };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_STQ1_0, ne00, ne01);
    struct ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne00, batch);
    ggml_cgraph * g = ggml_new_graph(ctx);
    struct ggml_tensor * outs[256];
    for (int i = 0; i < nops && i < 256; ++i) {
        outs[i] = ggml_mul_mat(ctx, A, X);
        ggml_build_forward_expand(g, outs[i]);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, mb);
    ggml_backend_tensor_set(A, wq.data(), 0, ggml_nbytes(A));
    ggml_backend_tensor_set(X, x.data(), 0, ggml_nbytes(X));

    const double wbytes = (double)ggml_nbytes(A);

    for (int i = 0; i < 10; ++i) ggml_backend_graph_compute(mb, g);

    double best = 1e18, tot = 0;
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(mb, g);
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        if (dt < best) best = dt;
        tot += dt;
    }
    double best_bw = nops * wbytes / best / 1e9;
    printf("STQ1_0 mul_mat %dx%d batch=%d x %d ops: best %.4f ms -> %6.1f GB/s (agg weight bytes; mean %.4f ms -> %6.1f GB/s)\n",
           ne01, ne00, batch, nops, best*1e3, best_bw, tot/reps*1e3, nops*wbytes/(tot/reps)/1e9);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(mb);
    return 0;
}
