#ifndef HMX_Q8_PREP_H
#define HMX_Q8_PREP_H

#include "hmx-utils.h"
#include "hvx-utils.h"
#include "hvx-quant.h"

// The F16 values of two Q8_0 blocks, columns c .. c + 63 of a row (64 lanes, zero past dim).
// The product of an int8 code and an F16 scale is exact in qf32 (8 x 11 bits), so converting the
// code to hf, multiplying in qf32 and rounding once with the qf32 -> hf conversion gives the same
// bits as hvx_dequantize_row_flat_q8_0_f16 (zeros as +0), which forms the same exact product in F32 and rounds
// it with the same conversion; it works on 64 values per vector instead of 32. The codes are read
// with one unaligned vector load as there (up to 96 bytes past the block, inside the VTCM staging,
// whose rows are laid out for F16): copying only 32 bytes compiled to a serial vinsert chain.
static inline HVX_Vector hmx_q8_pair_f16(const uint8_t * row, uint32_t c, uint32_t dim) {
    if (c >= dim) {
        return Q6_V_vzero();
    }
    const block_q8_0 * b0  = (const block_q8_0 *) (row + (c / 32) * sizeof(block_q8_0));
    const bool         two = c + 32 < dim;
    const HVX_Vector   v   = *(const HVX_UVector *) b0->qs; // b0 codes at 0..31, b1 codes at 34..65
    const HVX_VectorPred lo32 = Q6_Q_vsetq_R(32);
    const HVX_Vector codes = Q6_V_vmux_QVV(lo32, v, two ? Q6_V_valign_VVR(v, v, 2) : Q6_V_vzero());
    const HVX_Vector code_hf = Q6_Vhf_equals_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(codes)));
    const int16_t s0 = *(const int16_t *) &b0->d;
    const int16_t s1 = two ? *(const int16_t *) &b0[1].d : 0;
    const HVX_Vector scale = Q6_V_vmux_QVV(Q6_Q_vsetq_R(64), Q6_Vh_vsplat_R(s0), Q6_Vh_vsplat_R(s1));
    const HVX_Vector r = Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(code_hf, scale));
    // A zero code with a negative scale gives -0 here; the old path gives +0.
    const HVX_VectorPred zero = Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(r, Q6_Vh_vsplat_R(0x7fff)), Q6_V_vzero());
    return Q6_V_vmux_QVV(zero, Q6_V_vzero(), r);
}

// start is even; callers partition work in row pairs. K uses word scatters.
// Caller must complete scatters before HMX consumes the tiles.
static inline void hmx_q8_k_to_tiles(__fp16 * dst, const uint8_t * src, size_t stride,
                                    uint32_t dim, uint32_t padded, uint32_t start, uint32_t end) {
    assert(dim % 32 == 0 && padded % 64 == 0 && start % 2 == 0);
    const HVX_Vector offsets = hvx_vmem(hmx_transpose_scatter_offsets);
    for (uint32_t r = start; r < end; r += 2) {
        HVX_Vector off0 = Q6_Vw_vadd_VwVw(offsets, Q6_V_vsplat_R((r % 32) * 4));
        HVX_Vector off1 = Q6_Vw_vadd_VwVw(off0, Q6_V_vsplat_R(4));
        __fp16 * tile = dst + (size_t) (r / 32) * (padded / 32) * 1024;
        for (uint32_t c = 0; c < padded; c += 64) {
            HVX_Vector v0 = hmx_q8_pair_f16(src + (size_t) r * stride, c, dim);
            HVX_Vector v1 = r + 1 < end ? hmx_q8_pair_f16(src + (size_t) (r + 1) * stride, c, dim) : Q6_V_vzero();
            Q6_vscatter_RMVwV((size_t) tile, 4095, off0, v0);
            Q6_vscatter_RMVwV((size_t) tile, 4095, off1, v1);
            tile += 2048;
        }
    }
}

// V interleaves adjacent token rows. row_tiles is the allocated Bc stride,
// including at the final, partial KV block.
static inline void hmx_q8_v_to_tiles(__fp16 * dst, const uint8_t * src, size_t stride,
                                    uint32_t dim, uint32_t padded, uint32_t row_tiles,
                                    uint32_t start, uint32_t end) {
    assert(dim % 32 == 0 && padded % 64 == 0 && start % 2 == 0);
    for (uint32_t r = start; r < end; r += 2) {
        for (uint32_t c = 0; c < padded; c += 64) {
            HVX_Vector v0 = hmx_q8_pair_f16(src + (size_t) r * stride, c, dim);
            HVX_Vector v1 = r + 1 < end ? hmx_q8_pair_f16(src + (size_t) (r + 1) * stride, c, dim) : Q6_V_vzero();
            HVX_VectorPair pair = Q6_W_vshuff_VVR(v1, v0, -2);
            __fp16 * tile = dst + ((size_t) (c / 32) * row_tiles + r / 32) * 1024;
            ((HVX_Vector *) tile)[(r % 32) / 2] = Q6_V_lo_W(pair);
            ((HVX_Vector *) (tile + (size_t) row_tiles * 1024))[(r % 32) / 2] = Q6_V_hi_W(pair);
        }
    }
}

#endif
