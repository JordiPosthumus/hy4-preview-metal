// Standalone STQ1_0 matvec kernel variants for Apple Metal.
// v0: element-offset mapping, 16 table gathers / 16 weights / row  (current ggml kernel)
// v2: group-aligned mapping, 4 table gathers / 16 weights / row, float4 y loads
// v3: group-aligned mapping, arithmetic codebook decode (zero gathers), float4 y loads
//
// Hang-proof: command buffers are polled with a hard timeout, never blocking waits.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

static const unsigned char stq_cb[32] = {
    0xA9,0x89,0x29,0x09,0xA6,0x86,0x26,0x06,
    0x9A,0x92,0x1A,0x12,0x6A,0x62,0x4A,0x42,
    0x01,0x21,0x81,0xA1,0x04,0x24,0x84,0xA4,
    0x10,0x18,0x90,0x98,0x40,0x48,0x60,0x68,
};

struct mvargs {
    int      ne00;
    int      ne01;
    uint64_t nb01;
    uint64_t nb11;
    int      ne0;
} __attribute__((aligned(16)));

#define CHECKPOINT(msg) do { printf("[ckpt] %s\n", msg); fflush(stdout); } while(0)

// returns false on timeout
static bool run_and_wait(id<MTLCommandQueue> q, id<MTLComputePipelineState> ps,
                         id<MTLBuffer> abuf, id<MTLBuffer> wbuf, id<MTLBuffer> ybuf, id<MTLBuffer> dbuf,
                         int grid_x, int tg_threads, double timeout_s, double * out_sec) {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:abuf offset:0 atIndex:0];
    [enc setBuffer:wbuf offset:0 atIndex:1];
    [enc setBuffer:ybuf offset:0 atIndex:2];
    [enc setBuffer:dbuf offset:0 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(grid_x, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_threads, 1, 1)];
    [enc endEncoding];
    [cb commit];

    auto t0 = std::chrono::steady_clock::now();
    while (cb.status < MTLCommandBufferStatusCompleted) {
        if (cb.status == MTLCommandBufferStatusError) return false;
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (dt > timeout_s) { printf("[TIMEOUT after %.1fs]\n", dt); fflush(stdout); return false; }
        usleep(200);
    }
    if (out_sec) {
        const double gpu_sec = cb.GPUEndTime - cb.GPUStartTime;
        const double wall_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        *out_sec = gpu_sec > 0.0 ? gpu_sec : wall_sec;
    }
    return cb.status == MTLCommandBufferStatusCompleted && !cb.error;
}

static bool thrash_and_wait(id<MTLCommandQueue> q, id<MTLComputePipelineState> ps,
                            id<MTLBuffer> buf, double timeout_s) {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(2048, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];

    auto t0 = std::chrono::steady_clock::now();
    while (cb.status < MTLCommandBufferStatusCompleted) {
        if (cb.status == MTLCommandBufferStatusError) return false;
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout_s) return false;
        usleep(200);
    }
    return cb.status == MTLCommandBufferStatusCompleted && !cb.error;
}

int main(int argc, char ** argv) {
    @autoreleasepool {
        const int ne00 = argc > 1 ? atoi(argv[1]) : 16384;
        const int ne01 = argc > 2 ? atoi(argv[2]) : 16384;
        const int blocks_per_row = ne00 / 256;
        const size_t row_bytes = 42 * blocks_per_row;
        const int iters = argc > 3 ? atoi(argv[3]) : 20;
        const char * filter = argc > 4 ? argv[4] : nullptr;
        const bool cold_cache = getenv("COLD_CACHE") != nullptr;

        CHECKPOINT("device");
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { printf("no device\n"); return 1; }

        CHECKPOINT("read shader");
        NSError * err = nil;
        NSString * src = [NSString stringWithContentsOfFile:@"kernels.metal" encoding:NSUTF8StringEncoding error:&err];
        if (!src) { printf("no shader: %s\n", err.localizedDescription.UTF8String); return 1; }

        CHECKPOINT("compile shader (runtime)");
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { printf("compile error: %s\n", err.localizedDescription.UTF8String); return 1; }
        CHECKPOINT("compiled");

        std::mt19937 rng(123);
        std::uniform_int_distribution<int> byte(0,255);
        std::normal_distribution<float> fdist(0.f, 1.f);
        std::vector<unsigned char> w((size_t)ne01 * row_bytes);
        for (auto & v : w) v = (unsigned char)byte(rng);
        for (int r = 0; r < ne01; ++r) {
            for (int b = 0; b < blocks_per_row; ++b) {
                _Float16 * d = (_Float16 *)(w.data() + (size_t)r*row_bytes + (size_t)b*42 + 40);
                *d = (_Float16)(0.25f + 0.5f*fdist(rng));   // finite, non-zero scales everywhere
            }
        }
        std::vector<float> y(ne00);
        for (auto & v : y) v = fdist(rng);

        CHECKPOINT("cpu reference");
        std::vector<double> ref(ne01, 0.0);
        for (int r = 0; r < ne01; ++r) {
            const unsigned char * bx = w.data() + (size_t)r*row_bytes;
            double s = 0.0;
            for (int b = 0; b < blocks_per_row; ++b) {
                const unsigned char * blk = bx + (size_t)b*42;
                float d = *(const _Float16 *)(blk + 40);
                for (int g = 0; g < 64; ++g) {
                    unsigned code = (g & 1) ? (blk[g >> 1] >> 4) : (blk[g >> 1] & 0xF);
                    unsigned sb   = (blk[32 + (g >> 3)] >> (g & 7)) & 1;
                    unsigned qpack = stq_cb[(sb << 4) | code];
                    int base = b*256 + (g/16)*64 + (g%16);
                    for (int m = 0; m < 4; ++m) {
                        int l = (int)((qpack >> (2*m)) & 3) - 1;
                        s += (double)l * (double)d * (double)y[base + m*16];
                    }
                }
            }
            ref[r] = s;
        }
        CHECKPOINT("cpu reference done");
        id<MTLBuffer> wbuf = [dev newBufferWithBytes:w.data() length:w.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> ybuf = [dev newBufferWithBytes:y.data() length:y.size()*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> dbuf = [dev newBufferWithLength:(size_t)ne01*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> abuf = [dev newBufferWithLength:sizeof(mvargs) options:MTLResourceStorageModeShared];

        mvargs args;
        args.ne00 = ne00; args.ne01 = ne01; args.nb01 = row_bytes; args.nb11 = (uint64_t)ne00*4; args.ne0 = ne01;
        memcpy(abuf.contents, &args, sizeof(args));

        const char * variants[] = { "kernel_mv_v0_8x2", "kernel_mv_v2_4x4", "kernel_mv_v3_4x4",
                                    "kernel_mv_v4_4x2", "kernel_mv_v4_8x2", "kernel_mv_v4_4x4", "kernel_mv_v4_8x4",
                                    "kernel_mv_v4_2x8", "kernel_mv_v4_4x8",
                                    "kernel_mv_v2_4x8", "kernel_mv_v2_8x4", "kernel_mv_v2_2x8", "kernel_mv_v4b_2x8",
                                    "kernel_mv_v2_2x16", "kernel_mv_v7_charfull_2x16",
                                    "kernel_mv_v8_fourblock_charfull_2x16" };
        const int nsgs[] = { 2, 4, 4, 2, 2, 4, 4, 8, 8, 8, 4, 8, 8, 16, 16, 16 };
        const int nr0s[] = { 8, 4, 4, 4, 8, 4, 8, 2, 4, 4, 8, 2, 2, 2, 2, 2 };

        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLComputePipelineState> thrash_ps = nil;
        id<MTLBuffer> thrash_buf = nil;
        if (cold_cache) {
            id<MTLFunction> thrash_fn = [lib newFunctionWithName:@"kernel_cache_thrash"];
            thrash_ps = [dev newComputePipelineStateWithFunction:thrash_fn error:&err];
            thrash_buf = [dev newBufferWithLength:128ull*1024*1024 options:MTLResourceStorageModeShared];
            printf("[mode] cold-cache: 128 MB cache-thrash buffer\n");
        }

        // Stabilize GPU clocks/residency before collecting per-variant timestamps.
        // Without this, the first one or two small-matrix variants can include a
        // 0.2-0.4 ms ramp-up penalty even after their local three-dispatch warmup.
        if (ne01 % 16 == 0) {
            id<MTLFunction> warm_fn = [lib newFunctionWithName:@"kernel_mv_v2_2x8"];
            id<MTLComputePipelineState> warm_ps = [dev newComputePipelineStateWithFunction:warm_fn error:&err];
            double warm_sec = 0;
            for (int i = 0; i < 64; ++i) {
                if (!run_and_wait(q, warm_ps, abuf, wbuf, ybuf, dbuf, ne01/16, 32*8, 15.0, &warm_sec)) {
                    printf("hardware warmup failed\n");
                    return 1;
                }
            }
        }

        const bool reverse = getenv("BENCH_REVERSE") != nullptr;
        for (int step = 0; step < 16; ++step) {
            const int v = reverse ? 15 - step : step;
            if (filter && !strstr(filter, variants[v])) continue;
            printf("[variant] %s\n", variants[v]); fflush(stdout);
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:variants[v]]];
            if (!fn) { printf("  MISSING\n"); continue; }
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!ps) { printf("  pipeline error: %s\n", err.localizedDescription.UTF8String); continue; }

            const int NSG = nsgs[v], NR0 = nr0s[v];
            const int tg_rows = NR0 * NSG;
            if (ne01 % tg_rows) { printf("  rows not divisible\n"); continue; }
            const int grid_x = ne01 / tg_rows;

            double sec = 0;
            if (!run_and_wait(q, ps, abuf, wbuf, ybuf, dbuf, grid_x, 32*NSG, 15.0, &sec)) {
                printf("  FAILED/TIMEOUT\n"); continue;
            }

            // correctness
            float * out = (float *)dbuf.contents;
            if (getenv("DBG") && v >= 3) {
                printf("  dbg out[0..3]: %.4f %.4f %.4f %.4f\n", out[0], out[1], out[2], out[3]);
                printf("  dbg ref[0..3]: %.4f %.4f %.4f %.4f\n", ref[0], ref[1], ref[2], ref[3]);
            }
            double max_abs = 0;
            double max_rel = 0;
            for (int r = 0; r < ne01; ++r) {
                const double ae = fabs(out[r] - ref[r]);
                const double re = ae / (fabs(ref[r]) + 1e-9);
                if (ae > max_abs) max_abs = ae;
                if (re > max_rel) max_rel = re;
            }

            // timing
            double tot = 0;
            bool ok = true;
            for (int i = 0; i < 3; ++i) ok &= run_and_wait(q, ps, abuf, wbuf, ybuf, dbuf, grid_x, 32*NSG, 15.0, &sec);
            for (int i = 0; i < iters && ok; ++i) {
                if (cold_cache) ok = thrash_and_wait(q, thrash_ps, thrash_buf, 15.0);
                if (!ok) break;
                ok = run_and_wait(q, ps, abuf, wbuf, ybuf, dbuf, grid_x, 32*NSG, 15.0, &sec);
                tot += sec;
            }
            if (!ok) { printf("  FAILED/TIMEOUT in timing\n"); continue; }

            double bytes = (double)w.size() * iters;
            printf("  max_abs_err=%.2e  max_rel_err=%.2e   %6.3f ms/iter   BW=%4.0f GB/s\n",
                   max_abs, max_rel, tot/iters*1e3, bytes/tot/1e9);
            fflush(stdout);
        }
        CHECKPOINT("done");
    }
    return 0;
}
