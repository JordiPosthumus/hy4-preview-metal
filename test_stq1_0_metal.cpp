// Correctness test: STQ1_0 matmul on Metal vs CPU vs float reference.
// Uses synthetic tensors quantized with the patched ggml encoder. No model weights needed.
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

int main(int argc, char ** argv) {
    const int n_per_row = 256 * 4;   // multiple of QK_K=256
    const int n_rows    = 64;
    const int n_cols    = 32;        // >8 -> exercises mul_mm path; also test mul_mv with 1

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.f, 1.f);

    // random weights, quantize to STQ1_0
    std::vector<float> w((size_t)n_rows * n_per_row);
    for (auto & v : w) v = dist(rng);
    std::vector<block_stq1_0> wq((size_t)n_rows * ggml_row_size(GGML_TYPE_STQ1_0, n_per_row) / sizeof(block_stq1_0));
    quantize_stq1_0(w.data(), wq.data(), n_rows, n_per_row, nullptr);

    // dequantized reference of W
    std::vector<float> wq_f((size_t)n_rows * n_per_row);
    for (int r = 0; r < n_rows; ++r) {
        dequantize_row_stq1_0(wq.data() + (size_t)r * (n_per_row / 256), wq_f.data() + (size_t)r * n_per_row, n_per_row);
    }

    for (int n_inp : {1, 8, 32}) {
        std::vector<float> x((size_t)n_inp * n_per_row);
        for (auto & v : x) v = dist(rng);

        // float reference: C[r][c] = dot(wq_f[r], x[c])
        std::vector<float> ref((size_t)n_rows * n_inp, 0.f);
        for (int c = 0; c < n_inp; ++c) {
            for (int r = 0; r < n_rows; ++r) {
                float s = 0.f;
                for (int k = 0; k < n_per_row; ++k) {
                    s += wq_f[(size_t)r * n_per_row + k] * x[(size_t)c * n_per_row + k];
                }
                ref[(size_t)c * n_rows + r] = s;
            }
        }

        struct backend_t { ggml_backend_t b; const char * name; };
        backend_t backends[2] = {
            { ggml_backend_cpu_init(), "CPU" },
            { ggml_backend_metal_init(), "Metal" },
        };
        if (!backends[1].b) { printf("Metal backend init failed\n"); return 1; }

        for (auto & be : backends) {
            struct ggml_init_params ip = { /*mem_size*/ 64 * 1024 * 1024, /*mem_buffer*/ NULL, /*no_alloc*/ true };
            struct ggml_context * ctx = ggml_init(ip);

            struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_STQ1_0, n_per_row, n_rows);
            struct ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_per_row, n_inp);
            struct ggml_tensor * C = ggml_mul_mat(ctx, A, X);
            ggml_cgraph * g = ggml_new_graph(ctx);
            ggml_build_forward_expand(g, C);

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be.b);
            ggml_backend_tensor_set(A, wq.data(), 0, ggml_nbytes(A));
            ggml_backend_tensor_set(X, x.data(), 0, ggml_nbytes(X));

            ggml_backend_graph_compute(be.b, g);

            std::vector<float> out(ggml_nbytes(C) / sizeof(float));
            ggml_backend_tensor_get(C, out.data(), 0, ggml_nbytes(C));

            double max_err = 0.f;
            for (size_t i = 0; i < out.size(); ++i) {
                max_err = fmax(max_err, fabs(out[i] - ref[i]));
            }
            printf("n_inp=%2d  %-5s  max_abs_err vs float-ref: %.4f\n", n_inp, be.name, max_err);

            ggml_free(ctx);
            ggml_backend_buffer_free(buf);
        }
        for (auto & be : backends) ggml_backend_free(be.b);
    }
    return 0;
}
