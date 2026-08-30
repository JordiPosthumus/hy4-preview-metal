// Standalone STQ1_0 matvec kernel variants for Apple Metal.
// v0: element-offset mapping, 16 table gathers / 16 weights / row  (current ggml kernel)
// v2: group-aligned mapping, 4 table gathers / 16 weights / row, float4 y loads
// v3: group-aligned mapping, arithmetic codebook decode (zero gathers), float4 y loads
#include <metal_stdlib>
using namespace metal;

constant uchar stq_cb[32] = {
    0xA9, 0x89, 0x29, 0x09, 0xA6, 0x86, 0x26, 0x06,
    0x9A, 0x92, 0x1A, 0x12, 0x6A, 0x62, 0x4A, 0x42,
    0x01, 0x21, 0x81, 0xA1, 0x04, 0x24, 0x84, 0xA4,
    0x10, 0x18, 0x90, 0x98, 0x40, 0x48, 0x60, 0x68,
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

// ---------------------------------------------------------------- v0: current ggml kernel
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

        for (short row = 0; row < NR0; row++) {
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
template<int NR0, int NSG, bool ARITH>
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

        for (short row = 0; row < NR0; row++) {
            device const block_stq * bx = ax[row] + ib;
            const ushort qp = *(device const ushort *)(bx->qs + 2*t16);
            const uchar  sb = bx->sign[t16 >> 1] >> sbsh;
            const float d = bx->d;
            float sum = 0.f;
            for (short j = 0; j < 4; j++) {
                const uint code = (qp >> (8*(j >> 1) + 4*(j & 1))) & 0xF;
                const uint b    = (sb >> j) & 1;
                const uint qpack = ARITH ? (uint)stq_decode_arith(code, b)
                                         : (uint)stq_cb[(b << 4) | code];
                const float l0 = (float)((int)(qpack        & 3) - 1);
                const float l1 = (float)((int)((qpack >> 2) & 3) - 1);
                const float l2 = (float)((int)((qpack >> 4) & 3) - 1);
                const float l3 = (float)((int)((qpack >> 6) & 3) - 1);
                sum += l0*yv0[j] + l1*yv1[j] + l2*yv2[j] + l3*yv3[j];
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
    mv_impl_ga<8,2,false>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v3_8x2(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_ga<8,2,true>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v2_4x4(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_ga<4,4,false>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v3_4x4(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_ga<4,4,true>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v2_16x1(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_ga<16,1,false>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
kernel void kernel_mv_v3_16x1(constant mvargs & args, device const char * src0, device const float * y, device float * dst,
                             uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mv_impl_ga<16,1,true>(args, src0, y, dst, tgpig, tiisg, sgitg);
}
