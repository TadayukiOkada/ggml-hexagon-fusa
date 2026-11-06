#ifndef HMX_Q8_PREP_H
#define HMX_Q8_PREP_H

#include "hmx-utils.h"
#include "hvx-utils.h"
#include "hvx-quant.h"

// Q8_0 K/V rows (as the DMA left them in VTCM) straight into the F16 HMX tiles of flash attention.
//
// Two blocks (64 values) per vector, from one unaligned vector load at the first block: d0 at bytes
// 0..1, qs0 at 2..33, d1 at 34..35, qs1 at 36..67. vrdelta with hmx_q8_codes_ctrl packs qs0 and
// qs1 into bytes 0..63, vdelta with hmx_q8_scale_ctrl puts d0 into halfwords 0..31 and d1 into
// 32..63 (no scalar loads from VTCM). The controls come from the stage rule of the HVX PRM
// (vdelta: offsets 64 .. 1, vrdelta: 1 .. 64; Vd.ub[k] = (Vv.ub[k] & offset) ? Vu.ub[k ^ offset] :
// Vu.ub[k]): a byte bound for k from s sits at (k & done) | (s & ~done) after the stages in done,
// and the control bit of the stage there is (s ^ k) & offset.
//
// The product of an int8 code and an F16 scale is exact in qf32 (8 x 11 bits), so converting the
// code to hf, multiplying in qf32 and rounding once with the qf32 -> hf conversion gives the same
// bits as hvx_dequantize_row_flat_q8_0_f16 (which forms the same exact product in F32 and rounds it
// with the same conversion); zeros are made +0 as there.
//
// The load reads up to 128 bytes from the start of a pair (at most 94 past its last block); the
// caller keeps that inside VTCM. Rows go innermost, two pairs per iteration with the four decodes
// before the stores, so the loop overlaps the decode latency.

static const uint8_t hmx_q8_codes_ctrl[128] __attribute__((aligned(128))) = {
    0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x0a, 0x0a, 0x02, 0x02, 0x06, 0x06, 0x02, 0x02, 0x16, 0x16,
    0x02, 0x02, 0x06, 0x06, 0x02, 0x02, 0x0e, 0x0e, 0x02, 0x02, 0x06, 0x06, 0x02, 0x02, 0x2e, 0x2e,
    0x04, 0x04, 0x06, 0x06, 0x08, 0x08, 0x0c, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x14, 0x14, 0x1c, 0x1c,
    0x04, 0x04, 0x04, 0x04, 0x0c, 0x0c, 0x0c, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x4c, 0x4c, 0x5c, 0x5c,
    0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x00, 0x00, 0x00, 0x00, 0x08, 0x08, 0x08, 0x08,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x10, 0x10, 0x10,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x20, 0x20, 0x20,
};

static const uint8_t hmx_q8_scale_ctrl[128] __attribute__((aligned(128))) = {
    0x00, 0x00, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
    0x10, 0x10, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
    0x20, 0x20, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
    0x10, 0x10, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
    0x02, 0x02, 0x20, 0x20, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04,
    0x02, 0x02, 0x10, 0x10, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04,
    0x02, 0x02, 0x40, 0x40, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04,
    0x02, 0x02, 0x10, 0x10, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04,
};

static inline HVX_Vector hmx_q8x2_f16(const uint8_t * blk, HVX_Vector cc, HVX_Vector cs) {
    const HVX_Vector v       = *(const HVX_UVector *) blk;
    const HVX_Vector codes   = Q6_V_vrdelta_VV(v, cc);
    const HVX_Vector scale   = Q6_V_vdelta_VV(v, cs);
    const HVX_Vector code_hf = Q6_Vhf_equals_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(codes)));
    const HVX_Vector r       = Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(code_hf, scale));
    const HVX_VectorPred z   = Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(r, Q6_Vh_vsplat_R(0x7fff)), Q6_V_vzero());
    return Q6_V_vmux_QVV(z, Q6_V_vzero(), r);
}

static inline HVX_Vector hmx_q8x1_f16(const uint8_t * blk, HVX_Vector cc, HVX_Vector cs, HVX_VectorPred lo) {
    return Q6_V_vmux_QVV(lo, hmx_q8x2_f16(blk, cc, cs), Q6_V_vzero());
}

// Columns c .. c + 63 (two whole blocks) of the row pairs in [r0, r1); tcol is the first tile of the
// column pair, tstride the elements of one row of tiles. The staging and the tiles do not overlap.
static inline void hmx_q8_k_pairs(__fp16 * restrict tcol, const uint8_t * restrict col, size_t stride,
                                  size_t tstride, uint32_t r0, uint32_t r1, HVX_Vector cc, HVX_Vector cs,
                                  HVX_Vector offsets) {
    uint32_t r = r0;
    for (; r + 4 <= r1; r += 4) {
        const uint8_t *  s0 = col + (size_t) r * stride;
        const HVX_Vector a0 = hmx_q8x2_f16(s0, cc, cs);
        const HVX_Vector a1 = hmx_q8x2_f16(s0 + stride, cc, cs);
        const HVX_Vector b0 = hmx_q8x2_f16(s0 + 2 * stride, cc, cs);
        const HVX_Vector b1 = hmx_q8x2_f16(s0 + 3 * stride, cc, cs);
        const HVX_Vector oa0 = Q6_Vw_vadd_VwVw(offsets, Q6_V_vsplat_R((r % 32) * 4));
        const HVX_Vector oa1 = Q6_Vw_vadd_VwVw(oa0, Q6_V_vsplat_R(4));
        const HVX_Vector ob0 = Q6_Vw_vadd_VwVw(offsets, Q6_V_vsplat_R(((r + 2) % 32) * 4));
        const HVX_Vector ob1 = Q6_Vw_vadd_VwVw(ob0, Q6_V_vsplat_R(4));
        const size_t ta = (size_t) (tcol + (size_t) (r / 32) * tstride);
        const size_t tb = (size_t) (tcol + (size_t) ((r + 2) / 32) * tstride);
        Q6_vscatter_RMVwV(ta, 4095, oa0, a0);
        Q6_vscatter_RMVwV(ta, 4095, oa1, a1);
        Q6_vscatter_RMVwV(tb, 4095, ob0, b0);
        Q6_vscatter_RMVwV(tb, 4095, ob1, b1);
    }
    for (; r < r1; r += 2) {
        const uint8_t *  s0 = col + (size_t) r * stride;
        const HVX_Vector o0 = Q6_Vw_vadd_VwVw(offsets, Q6_V_vsplat_R((r % 32) * 4));
        const HVX_Vector o1 = Q6_Vw_vadd_VwVw(o0, Q6_V_vsplat_R(4));
        const size_t     t  = (size_t) (tcol + (size_t) (r / 32) * tstride);
        Q6_vscatter_RMVwV(t, 4095, o0, hmx_q8x2_f16(s0, cc, cs));
        Q6_vscatter_RMVwV(t, 4095, o1, hmx_q8x2_f16(s0 + stride, cc, cs));
    }
}

// The same for V; t1o is the element offset of the second column tile of the pair.
static inline void hmx_q8_v_pairs(__fp16 * restrict tcol, const uint8_t * restrict col, size_t stride,
                                  size_t t1o, uint32_t r0, uint32_t r1, HVX_Vector cc, HVX_Vector cs) {
    uint32_t r = r0;
    for (; r + 4 <= r1; r += 4) {
        const uint8_t *  s0 = col + (size_t) r * stride;
        const HVX_Vector a0 = hmx_q8x2_f16(s0, cc, cs);
        const HVX_Vector a1 = hmx_q8x2_f16(s0 + stride, cc, cs);
        const HVX_Vector b0 = hmx_q8x2_f16(s0 + 2 * stride, cc, cs);
        const HVX_Vector b1 = hmx_q8x2_f16(s0 + 3 * stride, cc, cs);
        const HVX_VectorPair pa = Q6_W_vshuff_VVR(a1, a0, -2);
        const HVX_VectorPair pb = Q6_W_vshuff_VVR(b1, b0, -2);
        HVX_Vector * ta = (HVX_Vector *) (tcol + (size_t) (r / 32) * 1024) + (r % 32) / 2;
        HVX_Vector * tb = (HVX_Vector *) (tcol + (size_t) ((r + 2) / 32) * 1024) + ((r + 2) % 32) / 2;
        ta[0] = Q6_V_lo_W(pa);
        *(HVX_Vector *) ((__fp16 *) ta + t1o) = Q6_V_hi_W(pa);
        tb[0] = Q6_V_lo_W(pb);
        *(HVX_Vector *) ((__fp16 *) tb + t1o) = Q6_V_hi_W(pb);
    }
    for (; r < r1; r += 2) {
        const uint8_t *      s0 = col + (size_t) r * stride;
        const HVX_VectorPair pa = Q6_W_vshuff_VVR(hmx_q8x2_f16(s0 + stride, cc, cs), hmx_q8x2_f16(s0, cc, cs), -2);
        HVX_Vector * ta = (HVX_Vector *) (tcol + (size_t) (r / 32) * 1024) + (r % 32) / 2;
        ta[0] = Q6_V_lo_W(pa);
        *(HVX_Vector *) ((__fp16 *) ta + t1o) = Q6_V_hi_W(pa);
    }
}

// start is even; callers partition work in row pairs. K uses word scatters: the caller completes
// them before HMX reads the tiles.
static inline void hmx_q8_k_to_tiles(__fp16 * dst, const uint8_t * src, size_t stride,
                                     uint32_t dim, uint32_t padded, uint32_t start, uint32_t end) {
    assert(dim % 32 == 0 && padded % 64 == 0 && start % 2 == 0);
    const HVX_Vector     cc      = hvx_vmem(hmx_q8_codes_ctrl);
    const HVX_Vector     cs      = hvx_vmem(hmx_q8_scale_ctrl);
    const HVX_Vector     offsets = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_VectorPred lo      = Q6_Q_vsetq_R(64);
    const uint32_t       end2    = start + ((end - start) & ~1u);
    const size_t         tstride = (size_t) (padded / 32) * 1024;
    for (uint32_t c = 0; c < padded; c += 64) {
        const uint8_t * col  = src + (size_t) (c / 32) * sizeof(block_q8_0);
        __fp16 *        tcol = dst + (size_t) (c / 32) * 1024;
        const int       kind = c + 64 <= dim ? 2 : c < dim ? 1 : 0;
        if (kind == 2) {
            hmx_q8_k_pairs(tcol, col, stride, tstride, start, end2, cc, cs, offsets);
        } else {
            for (uint32_t r = start; r < end2; r += 2) {
                const HVX_Vector off0 = Q6_Vw_vadd_VwVw(offsets, Q6_V_vsplat_R((r % 32) * 4));
                const HVX_Vector off1 = Q6_Vw_vadd_VwVw(off0, Q6_V_vsplat_R(4));
                const uint8_t *  s0   = col + (size_t) r * stride;
                const HVX_Vector v0   = kind ? hmx_q8x1_f16(s0, cc, cs, lo) : Q6_V_vzero();
                const HVX_Vector v1   = kind ? hmx_q8x1_f16(s0 + stride, cc, cs, lo) : Q6_V_vzero();
                __fp16 *         tile = tcol + (size_t) (r / 32) * tstride;
                Q6_vscatter_RMVwV((size_t) tile, 4095, off0, v0);
                Q6_vscatter_RMVwV((size_t) tile, 4095, off1, v1);
            }
        }
        if (end2 < end) {
            const uint32_t   r    = end2;
            const HVX_Vector off0 = Q6_Vw_vadd_VwVw(offsets, Q6_V_vsplat_R((r % 32) * 4));
            const HVX_Vector off1 = Q6_Vw_vadd_VwVw(off0, Q6_V_vsplat_R(4));
            const uint8_t *  s0   = col + (size_t) r * stride;
            const HVX_Vector v0   = kind == 2 ? hmx_q8x2_f16(s0, cc, cs) : kind == 1 ? hmx_q8x1_f16(s0, cc, cs, lo) : Q6_V_vzero();
            __fp16 *         tile = tcol + (size_t) (r / 32) * tstride;
            Q6_vscatter_RMVwV((size_t) tile, 4095, off0, v0);
            Q6_vscatter_RMVwV((size_t) tile, 4095, off1, Q6_V_vzero());
        }
    }
}

// V interleaves adjacent token rows. row_tiles is the allocated Bc stride, including at the final,
// partial KV block.
static inline void hmx_q8_v_to_tiles(__fp16 * dst, const uint8_t * src, size_t stride,
                                     uint32_t dim, uint32_t padded, uint32_t row_tiles,
                                     uint32_t start, uint32_t end) {
    assert(dim % 32 == 0 && padded % 64 == 0 && start % 2 == 0);
    const HVX_Vector     cc   = hvx_vmem(hmx_q8_codes_ctrl);
    const HVX_Vector     cs   = hvx_vmem(hmx_q8_scale_ctrl);
    const HVX_VectorPred lo   = Q6_Q_vsetq_R(64);
    const uint32_t       end2 = start + ((end - start) & ~1u);
    const size_t         t1o  = (size_t) row_tiles * 1024;
    for (uint32_t c = 0; c < padded; c += 64) {
        const uint8_t * col  = src + (size_t) (c / 32) * sizeof(block_q8_0);
        __fp16 *        tcol = dst + (size_t) (c / 32) * row_tiles * 1024;
        const int       kind = c + 64 <= dim ? 2 : c < dim ? 1 : 0;
        if (kind == 2) {
            hmx_q8_v_pairs(tcol, col, stride, t1o, start, end2, cc, cs);
        } else {
            for (uint32_t r = start; r < end2; r += 2) {
                const uint8_t *      s0   = col + (size_t) r * stride;
                const HVX_Vector     v0   = kind ? hmx_q8x1_f16(s0, cc, cs, lo) : Q6_V_vzero();
                const HVX_Vector     v1   = kind ? hmx_q8x1_f16(s0 + stride, cc, cs, lo) : Q6_V_vzero();
                const HVX_VectorPair pair = Q6_W_vshuff_VVR(v1, v0, -2);
                HVX_Vector * t0 = (HVX_Vector *) (tcol + (size_t) (r / 32) * 1024) + (r % 32) / 2;
                t0[0] = Q6_V_lo_W(pair);
                *(HVX_Vector *) ((__fp16 *) t0 + t1o) = Q6_V_hi_W(pair);
            }
        }
        if (end2 < end) {
            const uint32_t       r    = end2;
            const uint8_t *      s0   = col + (size_t) r * stride;
            const HVX_Vector     v0   = kind == 2 ? hmx_q8x2_f16(s0, cc, cs) : kind == 1 ? hmx_q8x1_f16(s0, cc, cs, lo) : Q6_V_vzero();
            const HVX_VectorPair pair = Q6_W_vshuff_VVR(Q6_V_vzero(), v0, -2);
            HVX_Vector * t0 = (HVX_Vector *) (tcol + (size_t) (r / 32) * 1024) + (r % 32) / 2;
            t0[0] = Q6_V_lo_W(pair);
            *(HVX_Vector *) ((__fp16 *) t0 + t1o) = Q6_V_hi_W(pair);
        }
    }
}

#endif
