#ifndef HMX_Q8_PREP_H
#define HMX_Q8_PREP_H

#include "hmx-utils.h"
#include "hvx-utils.h"
#include "hvx-quant.h"

// Same arithmetic as hvx_dequantize_row_flat_q8_0_f16, without a flat F16 store.
// Copy only the 32 valid code bytes: a full unaligned vector load can overread
// the last DMA staging row. Keep the staging stride/layout unchanged.
static inline HVX_Vector hmx_q8_block_f32(const block_q8_0 * src) {
    HVX_Vector code = Q6_V_vzero();
    memcpy(&code, src->qs, QK8_0);
    uint16_t scale;
    memcpy(&scale, &src->d, sizeof(scale));
    HVX_VectorPair ds = hvx_vec_f16_to_f32(Q6_Vh_vsplat_R(scale));
    HVX_VectorPair i16 = Q6_Wh_vunpack_Vb(code);
    HVX_VectorPair i32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(i16));
    return hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(i32)), Q6_V_lo_W(ds));
}

static inline HVX_Vector hmx_q8_pair_f16(const uint8_t * row, uint32_t c, uint32_t dim) {
    HVX_Vector lo = c < dim ? hmx_q8_block_f32((const block_q8_0 *) (row + (c / 32) * sizeof(block_q8_0))) : Q6_V_vzero();
    HVX_Vector hi = c + 32 < dim ? hmx_q8_block_f32((const block_q8_0 *) (row + (c / 32 + 1) * sizeof(block_q8_0))) : Q6_V_vzero();
    return hvx_vec_f32_to_f16(lo, hi);
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
