// Standalone STQ1_0 matvec kernel variants for Apple Metal.
// v0: element-offset mapping, 16 table gathers / 16 weights / row (first working kernel)
// v2: group-aligned mapping, 4 table gathers / 16 weights / row, float4 y loads
// v3: group-aligned mapping, arithmetic codebook decode (zero gathers), float4 y loads
// v7: group-aligned mapping, vector-valued char4 codebook + dot product (shipped decode)
#include <metal_stdlib>
using namespace metal;
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

constant uchar stq_cb[32] = {
    0xA9, 0x89, 0x29, 0x09, 0xA6, 0x86, 0x26, 0x06,
    0x9A, 0x92, 0x1A, 0x12, 0x6A, 0x62, 0x4A, 0x42,
    0x01, 0x21, 0x81, 0xA1, 0x04, 0x24, 0x84, 0xA4,
    0x10, 0x18, 0x90, 0x98, 0x40, 0x48, 0x60, 0x68,
};

constant char4 stq_cb4_full[32] = {
    char4( 0,  1,  1,  1), char4( 0,  1, -1,  1), char4( 0,  1,  1, -1), char4( 0,  1, -1, -1),
    char4( 1,  0,  1,  1), char4( 1,  0, -1,  1), char4( 1,  0,  1, -1), char4( 1,  0, -1, -1),
    char4( 1,  1,  0,  1), char4( 1, -1,  0,  1), char4( 1,  1,  0, -1), char4( 1, -1,  0, -1),
    char4( 1,  1,  1,  0), char4( 1, -1,  1,  0), char4( 1,  1, -1,  0), char4( 1, -1, -1,  0),
    char4( 0, -1, -1, -1), char4( 0, -1,  1, -1), char4( 0, -1, -1,  1), char4( 0, -1,  1,  1),
    char4(-1,  0, -1, -1), char4(-1,  0,  1, -1), char4(-1,  0, -1,  1), char4(-1,  0,  1,  1),
    char4(-1, -1,  0, -1), char4(-1,  1,  0, -1), char4(-1, -1,  0,  1), char4(-1,  1,  0,  1),
    char4(-1, -1, -1,  0), char4(-1,  1, -1,  0), char4(-1, -1,  1,  0), char4(-1,  1,  1,  0),
};

struct block_stq {
    uchar qs[32];    // 4-bit code per group of 4
    uchar sign[8];   // 1-bit table select per group
    half  d;         // scale
};

struct mvargs {
    int      ne00;   // elements per row (multiple of 256)
    int      ne01;   // number of rows
    uint64_t nb01;   // row stride, bytes
    uint64_t nb11;   // y row stride, bytes
    int      ne0;    // dst row stride (elements)
};

kernel void kernel_cache_thrash(device uint4 * buf [[buffer(0)]],
                                uint gid [[thread_position_in_grid]]) {
    const uint base = gid * 16;
    for (uint i = 0; i < 16; ++i) {
        buf[base + i] ^= uint4(gid + i, gid + i + 1, gid + i + 2, gid + i + 3);
    }
}

// decode one group's qpack byte arithmetically from (code, signbit)
inline uchar stq_decode_arith(uint code, uint b) {
    uint z = code >> 2;          // zero lane
    uint t = code & 3;           // sign pattern of the 3 non-zero lanes
    uint q = 0xAA;               // all lanes +1 (field 2)
    uint r1lane = (z < 2) ? 2u : 1u;
    uint r2lane = (z == 3) ? 2u : 3u;
    q &= ~(3u << (2*r1lane)); q |= ((t & 1) ? 0u : 2u) << (2*r1lane);
    q &= ~(3u << (2*r2lane)); q |= ((t & 2) ? 0u : 2u) << (2*r2lane);
    q &= ~(3u << (2*z));      q |= (1u << (2*z));
    return b ? (uchar)(0xAA - q) : (uchar)q;
}

// ---------------------------------------------------------------- v0: first working ggml kernel
template<int NR0, int NSG>
void mv_impl_v0(constant mvargs & args,
                device const char * src0,
                device const float * y,
                device float * dst,
                uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int nb = args.ne00 / 256;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    device const block_stq * ax[NR0];
    for (int row = 0; row < NR0; ++row) {
        ax[row] = (device const block_stq *)(src0 + (first_row + row) * args.nb01);
    }

    float yl[16];
    float sumf[NR0] = {0.f};

    const short ix = (tiisg/16);
    const short il = (tiisg%16)*16;
    const short p     = (il & 63) >> 4;
    const short chunk = il >> 6;

    device const float * yb = y + r1*(args.ne00) + ix*256 + il;

    for (int ib = ix; ib < nb; ib += 2) {
        for (short k = 0; k < 16; k++) yl[k] = yb[k];

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const block_stq * bx = ax[row] + ib;
            const float d = bx->d;
            float sum = 0.f;
            for (short k = 0; k < 16; k++) {
                const int g = chunk*16 + k;
                const uchar qs   = bx->qs[g >> 1];
                const uchar code = (g & 1) ? (qs >> 4) : (qs & 0x0F);
                const uchar sb   = (bx->sign[g >> 3] >> (g & 7)) & 0x01;
                const uchar qpack = stq_cb[((uint) sb << 4) | code];
                sum += (float) (((qpack >> (2*p)) & 3) - 1) * yl[k];
            }
            sumf[row] += d * sum;
        }
        yb += 256 * 2;
    }
    for (int row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.ne01) {
            dst[(uint64_t)r1*args.ne0 + first_row + row] = tot;
        }
    }
}

// ---------------------------------------------------- v2/v3: group-aligned thread mapping
// thread t16 of a 16-thread block-slice owns groups 4*t16 .. 4*t16+3 (16 weights):
//   chunk = t16>>2, gloc = (t16&3)*4, lane-m weight at chunk*64 + gloc + m*16
//   qs bytes 2*t16, 2*t16+1 ; sign byte t16>>1, bits (t16&1)*4 + j
template<int NR0, int NSG, int DECODE>
void mv_impl_ga(constant mvargs & args,
                device const char * src0,
                device const float * y,
                device float * dst,
                uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int nb = args.ne00 / 256;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    device const block_stq * ax[NR0];
    for (int row = 0; row < NR0; ++row) {
        ax[row] = (device const block_stq *)(src0 + (first_row + row) * args.nb01);
    }

    float sumf[NR0] = {0.f};

    const short ix  = tiisg / 16;        // block index within simdgroup (2 blocks)
    const short t16 = tiisg % 16;        // thread's group quad within a block
    const int chunk = t16 >> 2;
    const int gloc  = (t16 & 3) << 2;
    const int sbsh  = (t16 & 1) << 2;

    for (int ib = ix; ib < nb; ib += 2) {
        // y float4s for this block iteration, shared across all NR0 rows
        device const float * yb = y + r1*args.ne00 + ib*256 + chunk*64 + gloc;
        const float4 yv0 = *(device const float4 *)(yb);
        const float4 yv1 = *(device const float4 *)(yb + 16);
        const float4 yv2 = *(device const float4 *)(yb + 32);
        const float4 yv3 = *(device const float4 *)(yb + 48);

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const block_stq * bx = ax[row] + ib;
            const ushort qp = *(device const ushort *)(bx->qs + 2*t16);
            const uchar  sb = bx->sign[t16 >> 1] >> sbsh;
            const float d = bx->d;
            float sum = 0.f;
            FOR_UNROLL (short j = 0; j < 4; j++) {
                const uint code = (qp >> (8*(j >> 1) + 4*(j & 1))) & 0xF;
                const uint b    = (sb >> j) & 1;
                const float4 yy = float4(yv0[j], yv1[j], yv2[j], yv3[j]);
                if (DECODE == 2) {
                    sum += dot(float4(stq_cb4_full[(b << 4) | code]), yy);
                } else {
                    const uint qpack = DECODE == 1 ? (uint)stq_decode_arith(code, b)
                                                   : (uint)stq_cb[(b << 4) | code];
                    const float l0 = (float)((int)(qpack        & 3) - 1);
                    const float l1 = (float)((int)((qpack >> 2) & 3) - 1);
                    const float l2 = (float)((int)((qpack >> 4) & 3) - 1);
                    const float l3 = (float)((int)((qpack >> 6) & 3) - 1);
                    sum += l0*yv0[j] + l1*yv1[j] + l2*yv2[j] + l3*yv3[j];
                }
            }
            sumf[row] += d * sum;
        }
    }

    for (int row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.ne01) {
            dst[(uint64_t)r1*args.ne0 + first_row + row] = tot;
        }
    }
}

// ---------------------------------------------------- v4: q4_K-style cooperative mapping
// 8 threads per block, 4 blocks in flight per simdgroup:
//   thread t8 owns groups 8*t8 .. 8*t8+7 (32 weights)
//   qs: one uint32 at qs + 4*t8 ; sign: one byte at sign + t8
//   y: 8 float4 loads per block per thread, shared across NR0 rows
template<int NR0, int NSG, int DECODE = 0>
void mv_impl_v4(constant mvargs & args,
                device const char * src0,
                device const float * y,
                device float * dst,
                uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int nb = args.ne00 / 256;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    device const block_stq * ax[NR0];
    for (int row = 0; row < NR0; ++row) {
        ax[row] = (device const block_stq *)(src0 + (first_row + row) * args.nb01);
    }

    float sumf[NR0] = {0.f};

    const short ix = tiisg / 8;   // block index, 4 blocks in flight
    const short t8 = tiisg % 8;   // thread's group octet within a block
    const int chunk = t8 >> 1;
    const int gbase = (t8 & 1) << 3;

    for (int ib = ix; ib < nb; ib += 4) {
        // y for this block iteration: 32 weights per thread = 8 float4
        device const float * yb = y + r1*args.ne00 + ib*256 + chunk*64 + gbase;
        const float4 yv0a = *(device const float4 *)(yb);
        const float4 yv0b = *(device const float4 *)(yb + 4);
        const float4 yv1a = *(device const float4 *)(yb + 16);
        const float4 yv1b = *(device const float4 *)(yb + 20);
        const float4 yv2a = *(device const float4 *)(yb + 32);
        const float4 yv2b = *(device const float4 *)(yb + 36);
        const float4 yv3a = *(device const float4 *)(yb + 48);
        const float4 yv3b = *(device const float4 *)(yb + 52);

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const block_stq * bx = ax[row] + ib;
            // 42-byte blocks: uint loads here would be misaligned for odd ib —
            // read two 2-byte-aligned ushorts and combine instead
            const ushort qlo = *(device const ushort *)(bx->qs + 4*t8);
            const ushort qhi = *(device const ushort *)(bx->qs + 4*t8 + 2);
            const uint  q32 = (uint)qlo | ((uint)qhi << 16);
            const uchar s8  = bx->sign[t8];
            const float d = bx->d;
            float sum = 0.f;
            FOR_UNROLL (short j = 0; j < 8; j++) {
                const uint bytev = (q32 >> (8*(j >> 1))) & 0xFF;
                const uint code  = (j & 1) ? (bytev >> 4) : (bytev & 0xF);
                const uint b     = (s8 >> j) & 1;
                const float4 yy = j < 4
                    ? float4(yv0a[j], yv1a[j], yv2a[j], yv3a[j])
                    : float4(yv0b[j-4], yv1b[j-4], yv2b[j-4], yv3b[j-4]);
                if (DECODE == 2) {
                    sum += dot(float4(stq_cb4_full[(b << 4) | code]), yy);
                    continue;
                }
                const uint qpack = stq_cb[(b << 4) | code];
                const float l0 = (float)((int)(qpack        & 3) - 1);
                const float l1 = (float)((int)((qpack >> 2) & 3) - 1);
                const float l2 = (float)((int)((qpack >> 4) & 3) - 1);
                const float l3 = (float)((int)((qpack >> 6) & 3) - 1);
                if (j < 4) {
                    sum += l0*yv0a[j] + l1*yv1a[j] + l2*yv2a[j] + l3*yv3a[j];
                } else {
                    sum += l0*yv0b[j-4] + l1*yv1b[j-4] + l2*yv2b[j-4] + l3*yv3b[j-4];
                }
            }
            sumf[row] += d * sum;
        }
    }

    for (int row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.ne01) {
            dst[(uint64_t)r1*args.ne0 + first_row + row] = tot;
        }
    }
}

kernel void kernel_mv_v0_8x2(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v0<8,2>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v2_8x2(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_ga<8,2,0>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v3_8x2(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_ga<8,2,1>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v2_4x4(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_ga<4,4,0>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v3_4x4(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_ga<4,4,1>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v2_16x1(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_ga<16,1,0>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v3_16x1(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_ga<16,1,1>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v4_4x2(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v4<4,2>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v4_8x2(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v4<8,2>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v4_4x4(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v4<4,4>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v4_8x4(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v4<8,4>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v4_2x8(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v4<2,8>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v4_4x8(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v4<4,8>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v2_4x8(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_ga<4,8,0>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v2_8x4(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_ga<8,4,0>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v2_2x8(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_ga<2,8,0>(args, src0, y, dst, tgpig, tiisg, sgitg);
}

kernel void kernel_mv_v2_2x16(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                              uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_ga<2,16,0>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v7_charfull_2x16(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                                      uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_ga<2,16,2>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v8_fourblock_charfull_2x16(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                                                 uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v4<2,16,2>(args, src0, y, dst, tgpig, tiisg, sgitg);
}

// ---- diagnostics: isolate y-load cost (no decode) and decode cost (no y) ----
// diag_yonly: same y float4 loads + dot as v9/nr2, but no weight decode (fixed lv).
// diag_decodeonly: decodes weights but does trivial accumulate (no y loads / dots).
template<int NR0, int NSG, int MODE>
void mv_impl_diag(constant mvargs & args,
                  device const char * src0,
                  device const float * y,
                  device float * dst,
                  uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int nb = args.ne00 / 256;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    device const block_stq * ax[NR0];
    for (int row = 0; row < NR0; ++row) {
        ax[row] = (device const block_stq *)(src0 + (first_row + row) * args.nb01);
    }

    float sumf[NR0] = {0.f};
    const short ix = tiisg / 8;
    const short t8 = tiisg % 8;
    const int chunk = t8 >> 1;
    const int gbase = (t8 & 1) << 3;

    for (int ib = ix; ib < nb; ib += 4) {
        device const float * yb = y + r1*args.ne00 + ib*256 + chunk*64 + gbase;
        const float4 yv0a = *(device const float4 *)(yb);
        const float4 yv0b = *(device const float4 *)(yb + 4);
        const float4 yv1a = *(device const float4 *)(yb + 16);
        const float4 yv1b = *(device const float4 *)(yb + 20);
        const float4 yv2a = *(device const float4 *)(yb + 32);
        const float4 yv2b = *(device const float4 *)(yb + 36);
        const float4 yv3a = *(device const float4 *)(yb + 48);
        const float4 yv3b = *(device const float4 *)(yb + 52);

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const block_stq * bx = ax[row] + ib;
            const ushort qlo = *(device const ushort *)(bx->qs + 4*t8);
            const ushort qhi = *(device const ushort *)(bx->qs + 4*t8 + 2);
            const uint  q32 = (uint)qlo | ((uint)qhi << 16);
            const uchar s8  = bx->sign[t8];
            const float d = bx->d;
            float sum = 0.f;
            FOR_UNROLL (short j = 0; j < 8; j++) {
                if (MODE == 0) {
                    // no decode: fixed value, still touches y (memory traffic identical)
                    const float4 yy = j < 4
                        ? float4(yv0a[j], yv1a[j], yv2a[j], yv3a[j])
                        : float4(yv0b[j-4], yv1b[j-4], yv2b[j-4], yv3b[j-4]);
                    sum += dot(float4(1.f, -1.f, 1.f, -1.f), yy);
                } else {
                    // decode but cheap accumulate: no y loads, no gather needed value
                    const uint bytev = (q32 >> (8*(j >> 1))) & 0xFF;
                    const uint code  = (j & 1) ? (bytev >> 4) : (bytev & 0xF);
                    const uint b     = (s8 >> j) & 1;
                    const float4 lv  = float4(stq_cb4_full[(b << 4) | code]);
                    sum += lv.x + lv.y + lv.z + lv.w;
                }
            }
            sumf[row] += (MODE == 0) ? sum : d * sum;
        }
    }

    for (int row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.ne01) {
            dst[(uint64_t)r1*args.ne0 + first_row + row] = tot;
        }
    }
}
kernel void kernel_mv_diag_yonly_2x16(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                                      uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_diag<2,16,0>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_diag_decodeonly_2x16(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                                           uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_diag<2,16,1>(args, src0, y, dst, tgpig, tiisg, sgitg);
}

// ---- v18: v8 four-block charfull decode, y-column hoisted + 2-way accumulator split.
// Hoists the 8 float4 y-columns out of the NR0 row loop (they only depend on the
// block, not the row) and splits the per-thread accumulator to shorten the FMA
// dependency chain. Same group-aligned mapping and 42B-safe ushort loads as v8.
template<int NR0, int NSG>
void mv_impl_v18(constant mvargs & args,
                 device const char * src0,
                 device const float * y,
                 device float * dst,
                 uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int nb = args.ne00 / 256;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    device const block_stq * ax[NR0];
    for (int row = 0; row < NR0; ++row) {
        ax[row] = (device const block_stq *)(src0 + (first_row + row) * args.nb01);
    }

    float sumf[NR0] = {0.f};
    const short ix = tiisg / 8;
    const short t8 = tiisg % 8;
    const int chunk = t8 >> 1;
    const int gbase = (t8 & 1) << 3;

    for (int ib = ix; ib < nb; ib += 4) {
        device const float * yb = y + r1*args.ne00 + ib*256 + chunk*64 + gbase;
        const float4 yv0a = *(device const float4 *)(yb);
        const float4 yv0b = *(device const float4 *)(yb + 4);
        const float4 yv1a = *(device const float4 *)(yb + 16);
        const float4 yv1b = *(device const float4 *)(yb + 20);
        const float4 yv2a = *(device const float4 *)(yb + 32);
        const float4 yv2b = *(device const float4 *)(yb + 36);
        const float4 yv3a = *(device const float4 *)(yb + 48);
        const float4 yv3b = *(device const float4 *)(yb + 52);
        // 8 y-columns (one per group), shared by all NR0 rows
        const float4 ycol0 = float4(yv0a.x, yv1a.x, yv2a.x, yv3a.x);
        const float4 ycol1 = float4(yv0a.y, yv1a.y, yv2a.y, yv3a.y);
        const float4 ycol2 = float4(yv0a.z, yv1a.z, yv2a.z, yv3a.z);
        const float4 ycol3 = float4(yv0a.w, yv1a.w, yv2a.w, yv3a.w);
        const float4 ycol4 = float4(yv0b.x, yv1b.x, yv2b.x, yv3b.x);
        const float4 ycol5 = float4(yv0b.y, yv1b.y, yv2b.y, yv3b.y);
        const float4 ycol6 = float4(yv0b.z, yv1b.z, yv2b.z, yv3b.z);
        const float4 ycol7 = float4(yv0b.w, yv1b.w, yv2b.w, yv3b.w);
        const float4 ycol[8] = { ycol0, ycol1, ycol2, ycol3, ycol4, ycol5, ycol6, ycol7 };

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const block_stq * bx = ax[row] + ib;
            const ushort qlo = *(device const ushort *)(bx->qs + 4*t8);
            const ushort qhi = *(device const ushort *)(bx->qs + 4*t8 + 2);
            const uint  q32 = (uint)qlo | ((uint)qhi << 16);
            const uchar s8  = bx->sign[t8];
            const float d = bx->d;
            float sum0 = 0.f, sum1 = 0.f;
            FOR_UNROLL (short j = 0; j < 8; j++) {
                const uint bytev = (q32 >> (8*(j >> 1))) & 0xFF;
                const uint code  = (j & 1) ? (bytev >> 4) : (bytev & 0xF);
                const uint b     = (s8 >> j) & 1;
                const float4 lv  = float4(stq_cb4_full[(b << 4) | code]);
                if (j & 1) sum1 += dot(lv, ycol[j]);
                else       sum0 += dot(lv, ycol[j]);
            }
            sumf[row] += d * (sum0 + sum1);
        }
    }

    for (int row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.ne01) {
            dst[(uint64_t)r1*args.ne0 + first_row + row] = tot;
        }
    }
}
kernel void kernel_mv_v18_hoist2x16(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                                    uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v18<2,16>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v18_hoist4x16(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                                    uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v18<4,16>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v18_hoist4x8(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                                   uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v18<4,8>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v18_hoist8x8(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                                   uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v18<8,8>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v18_hoist8x4(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                                   uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v18<8,4>(args, src0, y, dst, tgpig, tiisg, sgitg);
}

// ---- v19: v18 + all 8 y-columns hoisted and a 4-way accumulator (full ILP).
// The y-columns are shared across NR0 rows; weight codes and their char4 table
// gathers remain row-specific.
template<int NR0, int NSG, bool BCAST_SCALE = false, bool SPARSE_FMA = false, bool EXACT_K6144 = false>
void mv_impl_v19(constant mvargs & args,
                 device const char * src0,
                 device const float * y,
                 device float * dst,
                 uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int nb = args.ne00 / 256;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    device const block_stq * ax[NR0];
    for (int row = 0; row < NR0; ++row) {
        ax[row] = (device const block_stq *)(src0 + (first_row + row) * args.nb01);
    }

    float sumf[NR0] = {0.f};
    const short ix = tiisg / 8;
    const short t8 = tiisg % 8;
    const int chunk = t8 >> 1;
    const int gbase = (t8 & 1) << 3;

    const int loop_nb = EXACT_K6144 ? 24 : nb;
    for (int ib = ix; ib < loop_nb; ib += 4) {
        device const float * yb = y + r1*args.ne00 + ib*256 + chunk*64 + gbase;
        const float4 yv0a = *(device const float4 *)(yb);
        const float4 yv0b = *(device const float4 *)(yb + 4);
        const float4 yv1a = *(device const float4 *)(yb + 16);
        const float4 yv1b = *(device const float4 *)(yb + 20);
        const float4 yv2a = *(device const float4 *)(yb + 32);
        const float4 yv2b = *(device const float4 *)(yb + 36);
        const float4 yv3a = *(device const float4 *)(yb + 48);
        const float4 yv3b = *(device const float4 *)(yb + 52);
        const float4 ycol[8] = {
            float4(yv0a.x, yv1a.x, yv2a.x, yv3a.x), float4(yv0a.y, yv1a.y, yv2a.y, yv3a.y),
            float4(yv0a.z, yv1a.z, yv2a.z, yv3a.z), float4(yv0a.w, yv1a.w, yv2a.w, yv3a.w),
            float4(yv0b.x, yv1b.x, yv2b.x, yv3b.x), float4(yv0b.y, yv1b.y, yv2b.y, yv3b.y),
            float4(yv0b.z, yv1b.z, yv2b.z, yv3b.z), float4(yv0b.w, yv1b.w, yv2b.w, yv3b.w)
        };

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const block_stq * bx = ax[row] + ib;
            const ushort qlo = *(device const ushort *)(bx->qs + 4*t8);
            const ushort qhi = *(device const ushort *)(bx->qs + 4*t8 + 2);
            const uint  q32 = (uint)qlo | ((uint)qhi << 16);
            const uchar s8  = bx->sign[t8];
            const float d = BCAST_SCALE
                ? simd_shuffle(t8 == 0 ? (float)bx->d : 0.f, ix*8)
                : (float)bx->d;
            float a0=0.f,a1=0.f,a2=0.f,a3=0.f;
            FOR_UNROLL (short j = 0; j < 8; j++) {
                const uint bytev = (q32 >> (8*(j >> 1))) & 0xFF;
                const uint code  = (j & 1) ? (bytev >> 4) : (bytev & 0xF);
                const uint b     = (s8 >> j) & 1;
                const float4 lv  = float4(stq_cb4_full[(b << 4) | code]);
                const float4 yj  = ycol[j];
                if (SPARSE_FMA) {
                    const uint zero_lane = code >> 2;
                    if (zero_lane != 0) a0 += lv.x*yj.x;
                    if (zero_lane != 1) a1 += lv.y*yj.y;
                    if (zero_lane != 2) a2 += lv.z*yj.z;
                    if (zero_lane != 3) a3 += lv.w*yj.w;
                } else {
                    a0 += lv.x*yj.x; a1 += lv.y*yj.y; a2 += lv.z*yj.z; a3 += lv.w*yj.w;
                }
            }
            sumf[row] += d * ((a0+a1)+(a2+a3));
        }
    }

    for (int row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.ne01) {
            dst[(uint64_t)r1*args.ne0 + first_row + row] = tot;
        }
    }
}
kernel void kernel_mv_v19_4acc_2x16(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                                    uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v19<2,16>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v19_4acc_2x8(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                                   uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v19<2,8>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v21_scale_bcast_2x16(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                                           uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v19<2,16,true>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v22_sparse_fma_2x16(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                                          uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v19<2,16,false,true>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v23_exact_k6144_2x16(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                                           uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v19<2,16,false,false,true>(args, src0, y, dst, tgpig, tiisg, sgitg);
}

// ---- v20: v18 y-column-hoisted structure but arithmetic codebook decode (no
// table gather). The notes rejected arithmetic under the old mapping (58-64 vs
// 85-94 GB/s) because Apple GPUs like small hot LUTs; re-tested here because the
// hoisted y-columns change the instruction mix and the gather is the suspected
// cost. Uses the algebraically-derived decode (slot = zero-pos + sign bits).
template<int NR0, int NSG>
void mv_impl_v20(constant mvargs & args,
                 device const char * src0,
                 device const float * y,
                 device float * dst,
                 uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int nb = args.ne00 / 256;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    device const block_stq * ax[NR0];
    for (int row = 0; row < NR0; ++row) {
        ax[row] = (device const block_stq *)(src0 + (first_row + row) * args.nb01);
    }

    float sumf[NR0] = {0.f};
    const short ix = tiisg / 8;
    const short t8 = tiisg % 8;
    const int chunk = t8 >> 1;
    const int gbase = (t8 & 1) << 3;

    for (int ib = ix; ib < nb; ib += 4) {
        device const float * yb = y + r1*args.ne00 + ib*256 + chunk*64 + gbase;
        const float4 yv0a = *(device const float4 *)(yb);
        const float4 yv0b = *(device const float4 *)(yb + 4);
        const float4 yv1a = *(device const float4 *)(yb + 16);
        const float4 yv1b = *(device const float4 *)(yb + 20);
        const float4 yv2a = *(device const float4 *)(yb + 32);
        const float4 yv2b = *(device const float4 *)(yb + 36);
        const float4 yv3a = *(device const float4 *)(yb + 48);
        const float4 yv3b = *(device const float4 *)(yb + 52);
        const float4 ycol[8] = {
            float4(yv0a.x, yv1a.x, yv2a.x, yv3a.x), float4(yv0a.y, yv1a.y, yv2a.y, yv3a.y),
            float4(yv0a.z, yv1a.z, yv2a.z, yv3a.z), float4(yv0a.w, yv1a.w, yv2a.w, yv3a.w),
            float4(yv0b.x, yv1b.x, yv2b.x, yv3b.x), float4(yv0b.y, yv1b.y, yv2b.y, yv3b.y),
            float4(yv0b.z, yv1b.z, yv2b.z, yv3b.z), float4(yv0b.w, yv1b.w, yv2b.w, yv3b.w)
        };

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const block_stq * bx = ax[row] + ib;
            const ushort qlo = *(device const ushort *)(bx->qs + 4*t8);
            const ushort qhi = *(device const ushort *)(bx->qs + 4*t8 + 2);
            const uint  q32 = (uint)qlo | ((uint)qhi << 16);
            const uchar s8  = bx->sign[t8];
            const float d = bx->d;
            float sum = 0.f;
            FOR_UNROLL (short j = 0; j < 8; j++) {
                const uint bytev = (q32 >> (8*(j >> 1))) & 0xFF;
                const uint code  = (j & 1) ? (bytev >> 4) : (bytev & 0xF);
                const uint b     = (s8 >> j) & 1;
                const uchar qpack = stq_decode_arith(code, b);
                const float l0 = (float)((int)(qpack        & 3) - 1);
                const float l1 = (float)((int)((qpack >> 2) & 3) - 1);
                const float l2 = (float)((int)((qpack >> 4) & 3) - 1);
                const float l3 = (float)((int)((qpack >> 6) & 3) - 1);
                const float4 yj = ycol[j];
                sum += l0*yj.x + l1*yj.y + l2*yj.z + l3*yj.w;
            }
            sumf[row] += d * sum;
        }
    }

    for (int row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.ne01) {
            dst[(uint64_t)r1*args.ne0 + first_row + row] = tot;
        }
    }
}
kernel void kernel_mv_v20_arith_2x16(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                                     uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v20<2,16>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
// then all threads/rows read y from fast shared memory. Keeps NR0 small (no
// register blowup) while cutting device y re-reads to once per threadgroup.
template<int NR0, int NSG>
void mv_impl_v11(constant mvargs & args,
                 device const char * src0,
                 device const float * y,
                 device float * dst,
                 threadgroup float * ysh,
                 uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int nb = args.ne00 / 256;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int first_row = (r0 * NSG + sgitg) * NR0;
    const int nthreads = 32 * NSG;

    // cooperative load of y into shared memory (threadgroup is per (r0) group of rows)
    for (int i = tiisg + sgitg*32; i < args.ne00; i += nthreads) {
        ysh[i] = y[r1*args.ne00 + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const block_stq * ax[NR0];
    for (int row = 0; row < NR0; ++row) {
        ax[row] = (device const block_stq *)(src0 + (first_row + row) * args.nb01);
    }

    float sumf[NR0] = {0.f};
    const short ix = tiisg / 8;
    const short t8 = tiisg % 8;
    const int chunk = t8 >> 1;
    const int gbase = (t8 & 1) << 3;

    for (int ib = ix; ib < nb; ib += 4) {
        const float4 yv0a = *(threadgroup const float4 *)(ysh + ib*256 + chunk*64 + gbase);
        const float4 yv0b = *(threadgroup const float4 *)(ysh + ib*256 + chunk*64 + gbase + 4);
        const float4 yv1a = *(threadgroup const float4 *)(ysh + ib*256 + chunk*64 + gbase + 16);
        const float4 yv1b = *(threadgroup const float4 *)(ysh + ib*256 + chunk*64 + gbase + 20);
        const float4 yv2a = *(threadgroup const float4 *)(ysh + ib*256 + chunk*64 + gbase + 32);
        const float4 yv2b = *(threadgroup const float4 *)(ysh + ib*256 + chunk*64 + gbase + 36);
        const float4 yv3a = *(threadgroup const float4 *)(ysh + ib*256 + chunk*64 + gbase + 48);
        const float4 yv3b = *(threadgroup const float4 *)(ysh + ib*256 + chunk*64 + gbase + 52);

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const block_stq * bx = ax[row] + ib;
            const ushort qlo = *(device const ushort *)(bx->qs + 4*t8);
            const ushort qhi = *(device const ushort *)(bx->qs + 4*t8 + 2);
            const uint  q32 = (uint)qlo | ((uint)qhi << 16);
            const uchar s8  = bx->sign[t8];
            const float d = bx->d;
            float sum = 0.f;
            FOR_UNROLL (short j = 0; j < 8; j++) {
                const uint bytev = (q32 >> (8*(j >> 1))) & 0xFF;
                const uint code  = (j & 1) ? (bytev >> 4) : (bytev & 0xF);
                const uint b     = (s8 >> j) & 1;
                const float4 lv  = float4(stq_cb4_full[(b << 4) | code]);
                const float4 yy = j < 4
                    ? float4(yv0a[j], yv1a[j], yv2a[j], yv3a[j])
                    : float4(yv0b[j-4], yv1b[j-4], yv2b[j-4], yv3b[j-4]);
                sum += dot(lv, yy);
            }
            sumf[row] += d * sum;
        }
    }

    for (int row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.ne01) {
            dst[(uint64_t)r1*args.ne0 + first_row + row] = tot;
        }
    }
}
kernel void kernel_mv_v11_shy_2x16(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                                   threadgroup float * ysh [[threadgroup(0)]],
                                   uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v11<2,16>(args, src0, y, dst, ysh, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v11_shy_2x8(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                                  threadgroup float * ysh [[threadgroup(0)]],
                                  uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v11<2,8>(args, src0, y, dst, ysh, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v11_shy_2x4(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                                  threadgroup float * ysh [[threadgroup(0)]],
                                  uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v11<2,4>(args, src0, y, dst, ysh, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v11_shy_4x8(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                                  threadgroup float * ysh [[threadgroup(0)]],
                                  uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v11<4,8>(args, src0, y, dst, ysh, tgpig, tiisg, sgitg);
}
template<int NR0, int NSG>
void mv_impl_v4b(constant mvargs & args,
                device const char * src0,
                device const float * y,
                device float * dst,
                uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int nb = args.ne00 / 256;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    device const block_stq * ax[NR0];
    for (int row = 0; row < NR0; ++row) {
        ax[row] = (device const block_stq *)(src0 + (first_row + row) * args.nb01);
    }

    float sumf[NR0] = {0.f};

    const short ix = tiisg / 8;
    const short t8 = tiisg % 8;
    const int chunk = t8 >> 1;
    const int gbase = (t8 & 1) << 3;

    for (int ib = ix; ib < nb; ib += 4) {
        device const float * yb = y + r1*args.ne00 + ib*256 + chunk*64 + gbase;

        for (short row = 0; row < NR0; row++) {
            device const block_stq * bx = ax[row] + ib;
            // 42-byte blocks: uint loads here would be misaligned for odd ib —
            // read two 2-byte-aligned ushorts and combine instead
            const ushort qlo = *(device const ushort *)(bx->qs + 4*t8);
            const ushort qhi = *(device const ushort *)(bx->qs + 4*t8 + 2);
            const uint  q32 = (uint)qlo | ((uint)qhi << 16);
            const uchar s8  = bx->sign[t8];
            const float d = bx->d;
            float sum = 0.f;
            for (short j = 0; j < 8; j++) {
                const uint bytev = (q32 >> (8*(j >> 1))) & 0xFF;
                const uint code  = (j & 1) ? (bytev >> 4) : (bytev & 0xF);
                const uint b     = (s8 >> j) & 1;
                const uint qpack = stq_cb[(b << 4) | code];
                const device const float * yj = yb + j;
                const float l0 = (float)((int)(qpack        & 3) - 1);
                const float l1 = (float)((int)((qpack >> 2) & 3) - 1);
                const float l2 = (float)((int)((qpack >> 4) & 3) - 1);
                const float l3 = (float)((int)((qpack >> 6) & 3) - 1);
                sum += l0*yj[0] + l1*yj[16] + l2*yj[32] + l3*yj[48];
            }
            sumf[row] += d * sum;
        }
    }

    for (int row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.ne01) {
            dst[(uint64_t)r1*args.ne0 + first_row + row] = tot;
        }
    }
}
kernel void kernel_mv_v4b_2x8(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_v4b<2,8>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
