#ifndef KV_CACHE_QUANT_H
#define KV_CACHE_QUANT_H

#include "htp-tensor.h"
#include "hvx-base.h"
#include "ggml-impl.h"

// Specialization of upstream's IQ4_NL cache quantizer (ntry=-1, one 32-value
// block, no imatrix). Keep the initial codes and fit their F32 weighted scale.
static inline void kv_quantize_iq4_nl(uint8_t * dst, const float * src, uint32_t n) {
    #pragma clang fp contract(off)
    static const int values[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    for (uint32_t b = 0; b < n / 32; ++b) {
        const float * x = src + b * 32;
        uint8_t * y = dst + b * 18;
        float amax = 0, signed_max = 0;
        for (int j = 0; j < 32; ++j) {
            if (fabsf(x[j]) > amax) { amax = fabsf(x[j]); signed_max = x[j]; }
        }
        if (amax < 1e-15f) { memset(y, 0, 18); continue; }
        const float initial = signed_max / -127.0f;
        const float inv = 1.0f / initial;
        // Word lanes also avoid an unsupported v8i8 shift selected by toolv87's SLP pass.
        uint32_t codes[32];
        float sqx = 0, sq2 = 0;
        for (int j = 0; j < 32; ++j) {
            const float scaled = inv * x[j];
            int code = 15;
            for (int k = 0; k < 15; ++k) code -= scaled < 0.5f * (values[k] + values[k + 1]);
            codes[j] = code;
            const float w = x[j] * x[j], q = values[code];
            sqx += w * q * x[j];
            sq2 += w * q * q;
        }
        const uint16_t scale = GGML_FP32_TO_FP16(sq2 > 0 ? sqx / sq2 : 0);
        memcpy(y, &scale, 2);
        for (int j = 0; j < 16; ++j) y[2 + j] = codes[j] | (codes[16 + j] << 4);
    }
}

// Reference block rule for Q4_0/Q4_1/Q5_0/Q5_1: choose the scale from F32, store it as F16, and round
// the codes with the original F32 scale (not with an F16 input or reciprocal).
// Scalar F32 only: with -fvectorize the compiler may otherwise turn these loops into HVX qf32
// arithmetic, which does not round like IEEE F32.
static inline void kv_quantize_block_ref(uint8_t * y, const float * x, int type) {
    #pragma clang fp contract(off)
    const bool affine = type == HTP_TYPE_Q4_1 || type == HTP_TYPE_Q5_1;
    const bool five = type == HTP_TYPE_Q5_0 || type == HTP_TYPE_Q5_1;
    const int limit = five ? 31 : 15;
    float lo = x[0], hi = x[0], amax = 0, signed_max = 0;
    #pragma clang loop vectorize(disable) interleave(disable)
    for (uint32_t j = 0; j < 32; ++j) {
        lo = fminf(lo, x[j]); hi = fmaxf(hi, x[j]);
        if (fabsf(x[j]) > amax) { amax = fabsf(x[j]); signed_max = x[j]; }
    }
    const float scale = affine ? (hi - lo) / limit : signed_max / (five ? -16.0f : -8.0f);
    const float inv = scale ? 1.0f / scale : 0.0f;
    const uint16_t hscale = GGML_FP32_TO_FP16(scale);
    memcpy(y, &hscale, 2);
    if (affine) { const uint16_t m = GGML_FP32_TO_FP16(lo); memcpy(y + 2, &m, 2); }
    const uint32_t off = 2 + (affine ? 2 : 0) + (five ? 4 : 0);
    uint32_t high = 0;
    #pragma clang loop vectorize(disable) interleave(disable)
    for (uint32_t j = 0; j < 16; ++j) {
        const float bias = affine ? 0.5f : five ? 16.5f : 8.5f;
        const int q0 = MIN(limit, (int) ((affine ? x[j] - lo : x[j]) * inv + bias));
        const int q1 = MIN(limit, (int) ((affine ? x[j + 16] - lo : x[j + 16]) * inv + bias));
        y[off + j] = (q0 & 15) | ((q1 & 15) << 4);
        high |= (uint32_t) ((q0 >> 4) & 1) << j;
        high |= (uint32_t) ((q1 >> 4) & 1) << (j + 16);
    }
    if (five) memcpy(y + off - 4, &high, 4);
}

// ---- HVX writer for Q4_0/Q4_1/Q5_0/Q5_1 -------------------------------------------------------------
//
// Four blocks (128 values) per group, one block per vector, and no scalar F32 work per block. The block
// min/max come from a 4-way tree reduction (word lane l ends up with block l & 3). The stored fields are
// exact: Q4_0/Q5_0 scale = max / -8 (-16) by a sign flip and an exponent decrement, the F16 bits by
// integer round to nearest even (kv_f16_bits); the Q4_1/Q5_1 scale (max - min) / limit is formed in qf32,
// and a block whose scale lies near an F16 rounding point goes to the reference. The codes need not be
// exact on the way: with an approximate reciprocal and qf32 products, a lane whose value before
// truncation lies within 2^-12 of an integer is marked (the error is below 2^-15 there) and its block is
// written by kv_quantize_block_ref. Ties of opposite sign, zero minimum (the sign of the stored min),
// tiny or huge values also go to the reference. So the output equals kv_quantize_block_ref (and upstream's
// quantize_row_*_ref) for finite inputs.
//
// Staging slot k (32 bytes): codes 0..15, qh word at byte 16 + 4k (the reduction lanes of block k),
// d/m word at byte 16 + 4((k + 1) & 3); one vrdelta packs the slots into consecutive blocks
// (controls: tests/htp/gen-kv-ctrl.py pack/splat, stage rule of the v73 HVX PRM).
static const uint8_t kv_pack_q4_0[128] __attribute__((aligned(128))) = {
    0x1e, 0x1e, 0x02, 0x02, 0x06, 0x06, 0x02, 0x02, 0x0e, 0x0e, 0x02, 0x02, 0x06, 0x06, 0x02, 0x02,
    0x14, 0x14, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x0c, 0x0c, 0x0c, 0x0c, 0x44, 0x44, 0x44, 0x44, 0x4c, 0x4c, 0x4c, 0x4c, 0x44, 0x44, 0x44, 0x44,
    0x40, 0x40, 0x48, 0x48, 0x50, 0x50, 0x50, 0x50, 0x50, 0x50, 0x52, 0x52, 0x50, 0x50, 0x50, 0x50,
    0x2a, 0x2a, 0x2e, 0x2e, 0x3a, 0x3a, 0x26, 0x26, 0x0a, 0x0a, 0x0e, 0x0e, 0x0a, 0x0a, 0x06, 0x06,
    0x10, 0x10, 0x10, 0x10, 0x18, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x08, 0x08, 0x08, 0x08, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28,
    0x20, 0x20, 0x22, 0x22, 0x20, 0x20, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
};
static const uint8_t kv_pack_q4_1[128] __attribute__((aligned(128))) = {
    0x1c, 0x1c, 0x1c, 0x1c, 0x04, 0x04, 0x04, 0x04, 0x0c, 0x0c, 0x0c, 0x0c, 0x04, 0x04, 0x04, 0x04,
    0x14, 0x14, 0x14, 0x14, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48,
    0x40, 0x40, 0x40, 0x40, 0x48, 0x48, 0x48, 0x48, 0x50, 0x50, 0x50, 0x50, 0x54, 0x54, 0x54, 0x54,
    0x24, 0x24, 0x24, 0x24, 0x2c, 0x2c, 0x2c, 0x2c, 0x34, 0x34, 0x34, 0x34, 0x2c, 0x2c, 0x2c, 0x2c,
    0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x14, 0x14, 0x14, 0x14, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x24, 0x24, 0x24, 0x24, 0x20, 0x20, 0x20, 0x20, 0x08, 0x08, 0x08, 0x08,
};
static const uint8_t kv_pack_q5_0[128] __attribute__((aligned(128))) = {
    0x1a, 0x1a, 0x1e, 0x1e, 0x1a, 0x1a, 0x06, 0x06, 0x0a, 0x0a, 0x0e, 0x0e, 0x0a, 0x0a, 0x06, 0x06,
    0x16, 0x16, 0x12, 0x12, 0x14, 0x14, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x04, 0x04, 0x04, 0x04, 0x0c, 0x0c, 0x0c, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x4c, 0x4c, 0x4c, 0x4c,
    0x44, 0x44, 0x44, 0x44, 0x40, 0x40, 0x48, 0x48, 0x48, 0x48, 0x4a, 0x4a, 0x50, 0x50, 0x54, 0x54,
    0x0e, 0x0e, 0x22, 0x22, 0x26, 0x26, 0x22, 0x22, 0x2e, 0x2e, 0x22, 0x22, 0x36, 0x36, 0x32, 0x32,
    0x28, 0x28, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x12, 0x12, 0x12, 0x12, 0x10, 0x10, 0x14, 0x14,
    0x08, 0x08, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x08, 0x08, 0x08, 0x08, 0x28, 0x28, 0x28, 0x28,
    0x30, 0x30, 0x32, 0x32, 0x38, 0x38, 0x38, 0x38, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
};
static const uint8_t kv_pack_q5_1[128] __attribute__((aligned(128))) = {
    0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x48, 0x48, 0x48, 0x48,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28,
    0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x38, 0x38, 0x38, 0x38, 0x30, 0x30, 0x30, 0x30,
};

static const int32_t kv_lane_index[32] __attribute__((aligned(128))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
};

static inline HVX_Vector kv_splat_f32(float f) {
    union { float f; int32_t i; } u = { f };
    return Q6_V_vsplat_R(u.i);
}

static inline float kv_lane_f32(HVX_Vector v, int lane) {
    union { int32_t i; float f; } u;
    u.i = Q6_R_vextract_VR(v, lane * 4);
    return u.f;
}

#define KV_REDUCE4(NAME, OP)                                                                       \
static inline HVX_Vector NAME(HVX_Vector v0, HVX_Vector v1, HVX_Vector v2, HVX_Vector v3) {        \
    HVX_VectorPair p = Q6_W_vshuff_VVR(v1, v0, -4);                                                \
    const HVX_Vector a = OP(Q6_V_lo_W(p), Q6_V_hi_W(p));                                           \
    p = Q6_W_vshuff_VVR(v3, v2, -4);                                                               \
    const HVX_Vector b = OP(Q6_V_lo_W(p), Q6_V_hi_W(p));                                           \
    p = Q6_W_vshuff_VVR(b, a, -8);                                                                 \
    HVX_Vector r = OP(Q6_V_lo_W(p), Q6_V_hi_W(p));                                                 \
    r = OP(r, Q6_V_vror_VR(r, 16));                                                                \
    r = OP(r, Q6_V_vror_VR(r, 32));                                                                \
    return OP(r, Q6_V_vror_VR(r, 64));                                                             \
}
KV_REDUCE4(kv_reduce4_max, Q6_Vsf_vmax_VsfVsf)
KV_REDUCE4(kv_reduce4_min, Q6_Vsf_vmin_VsfVsf)
KV_REDUCE4(kv_reduce4_or, Q6_V_vor_VV)
KV_REDUCE4(kv_reduce4_minw, Q6_Vw_vmin_VwVw)
#undef KV_REDUCE4

// Codes of one block, clamped to limit; *sus |= lanes near a rounding boundary (bit of the block).
static inline __attribute__((always_inline)) HVX_Vector kv_block_codes(HVX_Vector x, HVX_Vector lo, HVX_Vector inv,
                                                                       HVX_Vector bias, bool affine, int limit,
                                                                       HVX_Vector * sus, int bit) {
    const HVX_Vector y = affine ? Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(x, lo)) : x;
    const HVX_Vector t = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(y, inv), bias));
    // t > 0: integer and fraction bits of the significand
    const HVX_Vector e    = Q6_Vw_vmax_VwVw(Q6_Vw_vsub_VwVw(Q6_Vuw_vlsr_VuwR(t, 23), Q6_V_vsplat_R(127)), Q6_V_vsplat_R(-8));
    const HVX_Vector mant = Q6_V_vor_VV(Q6_V_vand_VV(t, Q6_V_vsplat_R(0x7fffff)), Q6_V_vsplat_R(0x800000));
    const HVX_Vector n    = Q6_Vw_vasr_VwVw(mant, Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(23), e));
    const HVX_Vector frac = Q6_Vw_vasl_VwVw(mant, Q6_Vw_vadd_VwVw(e, Q6_V_vsplat_R(9)));
    const HVX_Vector top  = Q6_V_vand_VV(Q6_Vw_vadd_VwVw(Q6_Vuw_vlsr_VuwR(frac, 20), Q6_V_vsplat_R(1)), Q6_V_vsplat_R(4095));
    *sus = Q6_V_vor_VV(*sus, Q6_V_vand_QR(Q6_Q_vcmp_gt_VuwVuw(Q6_V_vsplat_R(2), top), 1u << bit));
    return Q6_Vw_vmin_VwVw(n, Q6_V_vsplat_R(limit));
}

// Bytes of v selected by keep (in v's own byte order) go to dst .. dst + 127, unaligned.
static inline void kv_store_masked(uint8_t * dst, HVX_Vector v, HVX_VectorPred keep) {
    const HVX_Vector     m   = Q6_V_vlalign_VVR(Q6_V_vand_QR(keep, -1), Q6_V_vand_QR(keep, -1), (size_t) dst);
    const HVX_VectorPred q   = Q6_Q_vand_VR(m, -1);
    const HVX_VectorPred low = Q6_Q_vsetq_R((size_t) dst);  // bytes below dst in its vector
    v = Q6_V_vlalign_VVR(v, v, (size_t) dst);
    Q6_vmem_QRIV(Q6_Q_and_QQn(q, low), (HVX_Vector *) dst, v);
    if (((size_t) dst & 127) != 0) {
        Q6_vmem_QRIV(Q6_Q_and_QQ(q, low), (HVX_Vector *) dst + 1, v);
    }
}

// vdelta: every word <- word 0
static const uint8_t kv_splat_w0[128] __attribute__((aligned(128))) = {
    0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
};
// vdelta: every word <- word 1
static const uint8_t kv_splat_w1[128] __attribute__((aligned(128))) = {
    0x04, 0x04, 0x04, 0x04, 0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08,
    0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08,
    0x04, 0x04, 0x04, 0x04, 0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08,
    0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08,
    0x04, 0x04, 0x04, 0x04, 0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08,
    0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08,
    0x04, 0x04, 0x04, 0x04, 0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08,
    0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08,
};
// vdelta: every word <- word 2
static const uint8_t kv_splat_w2[128] __attribute__((aligned(128))) = {
    0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04,
    0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04,
    0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04,
    0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04,
    0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04,
    0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04,
    0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04,
    0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04,
};
// vdelta: every word <- word 3
static const uint8_t kv_splat_w3[128] __attribute__((aligned(128))) = {
    0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x00, 0x00, 0x00, 0x00,
    0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10,
    0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x20, 0x20, 0x20, 0x20,
    0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10,
    0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x40, 0x40, 0x40, 0x40,
    0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10,
    0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x20, 0x20, 0x20, 0x20,
    0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04, 0x10, 0x10, 0x10, 0x10,
};

static const int32_t kv_class_bits[32] __attribute__((aligned(128))) = {
    1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8,
};

// F32 -> F16 bits, round to nearest even, of every word (as ggml_compute_fp32_to_fp16: subnormals,
// overflow to inf, NaN -> sign | 0x7e00); the result is in the low halfword.
static inline HVX_Vector kv_f16_bits(HVX_Vector w) {
    const HVX_Vector one  = Q6_V_vsplat_R(1);
    const HVX_Vector a    = Q6_V_vand_VV(w, Q6_V_vsplat_R(0x7fffffff));
    const HVX_Vector sign = Q6_Vuw_vlsr_VuwR(Q6_V_vand_VV(w, Q6_V_vsplat_R(0x80000000)), 16);
    // normal F16: drop 13 bits with RNE, rebias the exponent 127 -> 15
    const HVX_Vector odd  = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(a, 13), one);
    const HVX_Vector nrm  = Q6_Vw_vsub_VwVw(Q6_Vuw_vlsr_VuwR(Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(a, Q6_V_vsplat_R(0xfff)), odd), 13),
                                            Q6_V_vsplat_R(112 << 10));
    // subnormal F16: significand * 2^(e - 126), RNE (e = biased exponent; F32 subnormals give 0)
    const HVX_Vector e    = Q6_Vuw_vlsr_VuwR(a, 23);
    const HVX_Vector sh   = Q6_Vw_vmin_VwVw(Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(126), e), Q6_V_vsplat_R(31));
    const HVX_Vector m    = Q6_V_vor_VV(Q6_V_vand_VV(a, Q6_V_vsplat_R(0x7fffff)), Q6_V_vsplat_R(0x800000));
    const HVX_Vector q    = Q6_Vw_vasr_VwVw(m, sh);
    const HVX_Vector half = Q6_Vw_vasl_VwVw(one, Q6_Vw_vsub_VwVw(sh, one));
    const HVX_Vector rem  = Q6_V_vand_VV(m, Q6_Vw_vsub_VwVw(Q6_Vw_vadd_VwVw(half, half), one));
    const HVX_VectorPred up = Q6_Q_or_QQ(Q6_Q_vcmp_gt_VuwVuw(rem, half),
                                         Q6_Q_vcmp_eqand_QVwVw(Q6_Q_vcmp_eq_VwVw(rem, half), Q6_V_vand_VV(q, one), one));
    const HVX_Vector sub  = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(e, Q6_V_vzero()), Q6_V_vzero(), Q6_Vw_vadd_VwVw(q, Q6_V_vand_QR(up, 1)));
    HVX_Vector r = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(Q6_V_vsplat_R(0x38800000), a), sub, nrm);
    r = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(Q6_V_vsplat_R(0x477ff000), a), r, Q6_V_vsplat_R(0x7c00));
    r = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(a, Q6_V_vsplat_R(0x7f800000)), Q6_V_vsplat_R(0x7e00), r);
    return Q6_V_vor_VV(r, sign);
}

// 1/d to about 2^-22 (magic seed, three qf32 Newton steps); 0 for d = +-0. Only used for codes, whose
// boundary margin covers it.
static inline HVX_Vector kv_recip_f32(HVX_Vector d) {
    const HVX_Vector a  = Q6_V_vand_VV(d, Q6_V_vsplat_R(0x7fffffff));
    const HVX_Vector two = kv_splat_f32(2.0f);
    HVX_Vector r = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7ef311c3), a);
    for (int i = 0; i < 3; ++i) {
        const HVX_Vector t = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, r));
        r = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two, t))));
    }
    r = Q6_V_vor_VV(r, Q6_V_vand_VV(d, Q6_V_vsplat_R(0x80000000)));
    return Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(a, Q6_V_vzero()), Q6_V_vzero(), r);
}

static inline __attribute__((always_inline)) void kv_quantize_row_hvx(uint8_t * dst, const float * src, uint32_t n,
                                                                      const int type) {
    const bool     affine = type == HTP_TYPE_Q4_1 || type == HTP_TYPE_Q5_1;
    const bool     five   = type == HTP_TYPE_Q5_0 || type == HTP_TYPE_Q5_1;
    const int      limit  = five ? 31 : 15;
    const uint32_t bsize  = htp_tensor_get_row_size(type, 32);
    const HVX_Vector ctrl = hvx_vmem(type == HTP_TYPE_Q4_0 ? kv_pack_q4_0 : type == HTP_TYPE_Q4_1 ? kv_pack_q4_1 :
                                     type == HTP_TYPE_Q5_0 ? kv_pack_q5_0 : kv_pack_q5_1);
    const HVX_Vector bias  = kv_splat_f32(affine ? 0.5f : five ? 16.5f : 8.5f);
    const HVX_Vector lanes = hvx_vmem(kv_lane_index);
    const HVX_Vector cbits = hvx_vmem(kv_class_bits);
    const HVX_Vector splat[4] = { hvx_vmem(kv_splat_w0), hvx_vmem(kv_splat_w1), hvx_vmem(kv_splat_w2), hvx_vmem(kv_splat_w3) };
    const HVX_Vector sgn   = Q6_V_vsplat_R(0x80000000);
    const HVX_Vector absm  = Q6_V_vsplat_R(0x7fffffff);
    const HVX_Vector zero  = Q6_V_vzero();
    // d/m words at byte 16 + 4((k + 1) & 3) of slot k, qh words at 16 + 4k, codes at 0..15
    HVX_VectorPred q_dm = Q6_Q_vsetq_R(0), q_qh = Q6_Q_vsetq_R(0), q_qs = Q6_Q_vsetq_R(0);
    for (int k = 0; k < 4; ++k) {
        const int dm = 32 * k + 16 + 4 * ((k + 1) & 3), qh = 32 * k + 16 + 4 * k;
        q_qs = Q6_Q_or_QQ(q_qs, Q6_Q_and_QQn(Q6_Q_vsetq_R(32 * k + 16), Q6_Q_vsetq_R(32 * k)));
        q_dm = Q6_Q_or_QQ(q_dm, Q6_Q_and_QQn(Q6_Q_vsetq2_R(dm + 4), Q6_Q_vsetq_R(dm)));
        q_qh = Q6_Q_or_QQ(q_qh, Q6_Q_and_QQn(Q6_Q_vsetq2_R(qh + 4), Q6_Q_vsetq_R(qh)));
    }
    const uint32_t nb = n / 32;
    for (uint32_t g = 0; g < nb; g += 4) {
        const uint32_t   nblk = MIN(4u, nb - g);
        const float *    x    = src + g * 32;
        const HVX_Vector x0   = *(const HVX_UVector *) x;
        const HVX_Vector x1   = nblk > 1 ? *(const HVX_UVector *) (x + 32) : zero;
        const HVX_Vector x2   = nblk > 2 ? *(const HVX_UVector *) (x + 64) : zero;
        const HVX_Vector x3   = nblk > 3 ? *(const HVX_UVector *) (x + 96) : zero;
        // per block (word lane l: block l & 3)
        const HVX_Vector hi = kv_reduce4_max(x0, x1, x2, x3);
        const HVX_Vector lo = kv_reduce4_min(x0, x1, x2, x3);
        HVX_VectorPred   ref;
        HVX_Vector       d, hm = zero;
        if (affine) {
            // the stored min keeps the sign of its first occurrence: zero minima go to the reference
            const HVX_Vector ehi = Q6_Vuw_vlsr_VuwR(Q6_V_vand_VV(hi, absm), 23), elo = Q6_Vuw_vlsr_VuwR(Q6_V_vand_VV(lo, absm), 23);
            ref = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(lo, absm), zero);
            ref = Q6_Q_or_QQ(ref, Q6_Q_vcmp_gt_VwVw(Q6_Vw_vmax_VwVw(ehi, elo), Q6_V_vsplat_R(127 + 100)));
            ref = Q6_Q_or_QQ(ref, Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(127 - 100), Q6_Vw_vmax_VwVw(ehi, elo)));
            // (hi - lo) / limit in qf32: exact unless it lies near an F16 rounding point (then reference)
            d  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(hi, lo)),
                                                         kv_splat_f32(1.0f / limit)));
            const HVX_Vector low13 = Q6_V_vand_VV(d, Q6_V_vsplat_R(0x1fff));
            const HVX_Vector dist  = Q6_Vw_vabs_Vw(Q6_Vw_vsub_VwVw(low13, Q6_V_vsplat_R(0x1000)));
            ref = Q6_Q_or_QQ(ref, Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(64), dist));
            ref = Q6_Q_or_QQ(ref, Q6_Q_vcmp_gt_VuwVuw(Q6_V_vsplat_R(0x38800000), Q6_V_vand_VV(d, absm)));
            ref = Q6_Q_and_QQn(ref, Q6_Q_vcmp_eq_VwVw(hi, lo));  // constant block: d = 0 exactly
            ref = Q6_Q_or_QQ(ref, Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(lo, absm), zero));
            hm  = Q6_Vw_vasl_VwR(kv_f16_bits(lo), 16);
        } else {
            const HVX_Vector     nlo  = Q6_V_vxor_VV(lo, sgn);
            const HVX_VectorPred p_hi = Q6_Q_vcmp_gt_VsfVsf(hi, nlo), p_lo = Q6_Q_vcmp_gt_VsfVsf(nlo, hi);
            const HVX_Vector     sm   = Q6_V_vmux_QVV(p_hi, hi, Q6_V_vmux_QVV(p_lo, lo, zero));
            const HVX_Vector     esm  = Q6_Vuw_vlsr_VuwR(Q6_V_vand_VV(sm, absm), 23);
            // ties of opposite sign (and NaN) take the first occurrence; tiny or huge scales: reference
            ref = Q6_Q_and_QQn(Q6_Q_vcmp_gt_VuwVuw(Q6_V_vand_VV(hi, absm), zero), Q6_Q_or_QQ(p_hi, p_lo));
            ref = Q6_Q_or_QQ(ref, Q6_Q_vcmp_gt_VwVw(esm, Q6_V_vsplat_R(127 + 100)));
            ref = Q6_Q_or_QQ(ref, Q6_Q_and_QQ(Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(127 - 96), esm),
                                              Q6_Q_vcmp_gt_VuwVuw(Q6_V_vand_VV(sm, absm), zero)));
            // d = sm / -8 (-16): sign flip and exponent - 3 (- 4), exact for these normal scales; sm = 0 -> -0
            d = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(sm, zero), sgn,
                              Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(sm, sgn), Q6_V_vsplat_R((five ? 4 : 3) << 23)));
        }
        const HVX_Vector inv = kv_recip_f32(d);
        HVX_Vector       sus = Q6_V_vmux_QVV(ref, cbits, zero);
        HVX_Vector       dmv = Q6_V_vor_VV(Q6_V_vand_VV(kv_f16_bits(d), Q6_V_vsplat_R(0xffff)), hm);
        HVX_Vector codes[4];
        const HVX_Vector xs[4] = { x0, x1, x2, x3 };
        for (int k = 0; k < 4; ++k) {
            codes[k] = kv_block_codes(xs[k], affine ? Q6_V_vdelta_VV(lo, splat[k]) : zero, Q6_V_vdelta_VV(inv, splat[k]),
                                      bias, affine, limit, &sus, k);
        }
        // the d/m word of block k sits one word above a lane of class k
        HVX_Vector packed = Q6_V_vmux_QVV(q_dm, Q6_V_vlalign_VVR(dmv, dmv, 4), zero);
        if (five) {
            HVX_Vector hb[4];
            for (int k = 0; k < 4; ++k) {
                hb[k] = Q6_Vw_vasl_VwVw(Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(codes[k], 4), Q6_V_vsplat_R(1)), lanes);
            }
            packed = Q6_V_vmux_QVV(q_qh, kv_reduce4_or(hb[0], hb[1], hb[2], hb[3]), packed);
        }
        HVX_Vector nib[4];
        for (int k = 0; k < 4; ++k) {
            const HVX_Vector c = Q6_V_vand_VV(codes[k], Q6_V_vsplat_R(15));
            nib[k] = Q6_V_vor_VV(c, Q6_Vw_vasl_VwR(Q6_V_vror_VR(c, 64), 4));
        }
        const HVX_Vector qs = Q6_Vb_vpacke_VhVh(Q6_Vh_vpacke_VwVw(nib[3], nib[2]), Q6_Vh_vpacke_VwVw(nib[1], nib[0]));
        packed = Q6_V_vmux_QVV(q_qs, qs, packed);
        uint8_t * y        = dst + (size_t) g * bsize;
        uint32_t  fallback = (uint32_t) Q6_R_vextract_VR(kv_reduce4_or(sus, zero, zero, zero), 0) & ((1u << nblk) - 1);
        // Blocks that need the reference are written by scalar stores and left out of the vector store,
        // so no byte is written by both (the order of scalar and HVX stores to VTCM is not relied on).
        HVX_VectorPred keep = Q6_Q_vsetq2_R(nblk * bsize);
        for (uint32_t f = fallback; f;) {
            const int k = __builtin_ctz(f);
            f &= f - 1;
            kv_quantize_block_ref(y + k * bsize, x + k * 32, type);
            keep = Q6_Q_and_QQn(keep, Q6_Q_and_QQn(Q6_Q_vsetq2_R((k + 1) * bsize), Q6_Q_vsetq_R(k * bsize)));
        }
        kv_store_masked(y, Q6_V_vrdelta_VV(packed, ctrl), keep);
    }
}

// ---- HVX writer for IQ4_NL -------------------------------------------------------------------------
// Codes from the vector product inv * x (qf32) against the 15 decision points of kv_quantize_iq4_nl,
// widened by 2^-9 on both sides: a block with a value inside a widened interval (where qf32 and F32 could
// choose differently) is written by the scalar quantizer. The weighted F32 scale fit keeps the scalar
// reference's sequential sums, from a register-staged copy of the block (not scalar VTCM reads).
static inline void kv_quantize_row_iq4_hvx(uint8_t * dst, const float * src, uint32_t n) {
    #pragma clang fp contract(off)
    static const float mids[15] = { -115.5f, -93.5f, -74.0f, -57.0f, -42.0f, -28.5f, -16.0f, -4.5f,
                                    7.0f, 19.0f, 31.5f, 45.5f, 61.0f, 79.0f, 101.0f };
    static const int values[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
    const HVX_Vector ctrl = hvx_vmem(kv_pack_q4_0);
    HVX_VectorPred   q_dm[4], q_qs = Q6_Q_vsetq_R(0);
    for (int k = 0; k < 4; ++k) {
        const int dm = 32 * k + 16 + 4 * ((k + 1) & 3);
        q_dm[k] = Q6_Q_and_QQn(Q6_Q_vsetq2_R(dm + 4), Q6_Q_vsetq_R(dm));
        q_qs    = Q6_Q_or_QQ(q_qs, Q6_Q_and_QQn(Q6_Q_vsetq_R(32 * k + 16), Q6_Q_vsetq_R(32 * k)));
    }
    const uint32_t nb = n / 32;
    for (uint32_t g = 0; g < nb; g += 4) {
        const uint32_t   nblk = MIN(4u, nb - g);
        const float *    x    = src + g * 32;
        const HVX_Vector zero = Q6_V_vzero();
        union { HVX_Vector v[4]; float f[128]; } xb;
        union { HVX_Vector v[4]; int32_t i[128]; } cb;
        xb.v[0] = *(const HVX_UVector *) x;
        xb.v[1] = nblk > 1 ? *(const HVX_UVector *) (x + 32) : zero;
        xb.v[2] = nblk > 2 ? *(const HVX_UVector *) (x + 64) : zero;
        xb.v[3] = nblk > 3 ? *(const HVX_UVector *) (x + 96) : zero;
        const HVX_Vector hi = kv_reduce4_max(xb.v[0], xb.v[1], xb.v[2], xb.v[3]);
        const HVX_Vector lo = kv_reduce4_min(xb.v[0], xb.v[1], xb.v[2], xb.v[3]);
        uint32_t fallback = 0, zeroed = 0;
        for (uint32_t k = 0; k < 4; ++k) {
            const float mx = kv_lane_f32(hi, k), mn = kv_lane_f32(lo, k);
            float sm = 0.0f;
            if (mx > -mn) sm = mx; else if (-mn > mx) sm = mn;
            bool ref = (sm == 0.0f && mx != 0.0f) || !(fabsf(sm) <= 0x1p100f);
            if (fabsf(sm) < 1e-15f && !ref) zeroed |= 1u << k;
            const float inv = (zeroed >> k) & 1 ? 0.0f : 1.0f / (sm / -127.0f);
            ref = ref || !(fabsf(inv) <= 0x1p100f);
            const HVX_Vector sc = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(xb.v[k], kv_splat_f32(inv)));
            HVX_Vector nlo = zero, nhi = zero;
            for (int t = 0; t < 15; ++t) {
                nlo = Q6_Vw_vadd_VwVw(nlo, Q6_V_vand_QR(Q6_Q_vcmp_gt_VsfVsf(kv_splat_f32(mids[t] - 0x1p-9f), sc), 1));
                nhi = Q6_Vw_vadd_VwVw(nhi, Q6_V_vand_QR(Q6_Q_vcmp_gt_VsfVsf(kv_splat_f32(mids[t] + 0x1p-9f), sc), 1));
            }
            const HVX_Vector diff = Q6_V_vand_QR(Q6_Q_vcmp_eq_VwVw(nlo, nhi), 1);
            ref = ref || ((uint32_t) Q6_R_vextract_VR(kv_reduce4_minw(diff, diff, diff, diff), 0) == 0);
            cb.v[k] = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(15), nhi);
            if (ref && k < nblk) fallback |= 1u << k;
        }
        HVX_Vector packed = zero;
        for (uint32_t k = 0; k < 4; ++k) {
            uint16_t hd = 0;
            if (!((zeroed | fallback) >> k & 1) && k < nblk) {
                float sqx = 0, sq2 = 0;
                #pragma clang loop vectorize(disable) interleave(disable)
                for (int j = 0; j < 32; ++j) {
                    const float xv = xb.f[32 * k + j], w = xv * xv, q = values[cb.i[32 * k + j]];
                    sqx += w * q * xv;
                    sq2 += w * q * q;
                }
                hd = GGML_FP32_TO_FP16(sq2 > 0 ? sqx / sq2 : 0);
            }
            if ((zeroed >> k) & 1) cb.v[k] = zero;
            packed = Q6_V_vmux_QVV(q_dm[k], Q6_V_vsplat_R(hd), packed);
        }
        HVX_Vector nib[4];
        for (int k = 0; k < 4; ++k) {
            nib[k] = Q6_V_vor_VV(cb.v[k], Q6_Vw_vasl_VwR(Q6_V_vror_VR(cb.v[k], 64), 4));
        }
        const HVX_Vector qs = Q6_Vb_vpacke_VhVh(Q6_Vh_vpacke_VwVw(nib[3], nib[2]), Q6_Vh_vpacke_VwVw(nib[1], nib[0]));
        packed = Q6_V_vmux_QVV(q_qs, qs, packed);
        uint8_t *      y    = dst + (size_t) g * 18;
        HVX_VectorPred keep = Q6_Q_vsetq2_R(nblk * 18);
        for (uint32_t f = fallback; f;) {
            const int k = __builtin_ctz(f);
            f &= f - 1;
            kv_quantize_iq4_nl(y + k * 18, x + k * 32, 32);
            keep = Q6_Q_and_QQn(keep, Q6_Q_and_QQn(Q6_Q_vsetq2_R((k + 1) * 18), Q6_Q_vsetq_R(k * 18)));
        }
        kv_store_masked(y, Q6_V_vrdelta_VV(packed, ctrl), keep);
    }
}

// ---- HVX writer for F8_E4M3 --------------------------------------------------------------------------
// Integer form of upstream ggml_fp32_to_f8_e4m3 (round to nearest even on the F32 bits), exact for all
// inputs: NaN -> 0x7f, |x| >= 448 or inf -> sign | 0x7e, |x| < 2^-6 -> sign | RNE(|x| * 512),
// otherwise the significand rounded to 3 bits (a carry raises the exponent), exponent bias 7.
static inline HVX_Vector kv_f8_word(HVX_Vector x) {
    const HVX_Vector u    = Q6_V_vand_VV(x, Q6_V_vsplat_R(0x7fffffff));
    const HVX_Vector sign = Q6_Vuw_vlsr_VuwR(Q6_V_vand_VV(x, Q6_V_vsplat_R(0x80000000)), 24);
    // normal: (u + 0x7ffff + bit 20) >> 20, rebias exponent 127 -> 7
    const HVX_Vector odd  = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(u, 20), Q6_V_vsplat_R(1));
    const HVX_Vector nrm  = Q6_Vw_vsub_VwVw(Q6_Vuw_vlsr_VuwR(Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(u, Q6_V_vsplat_R(0x7ffff)), odd), 20),
                                            Q6_V_vsplat_R(120 << 3));
    // subnormal: m * 2^(e - 23 + 9), e = exponent - 127 <= -7: shift s = 14 - e (clamped to 31), RNE
    const HVX_Vector e    = Q6_Vw_vsub_VwVw(Q6_Vuw_vlsr_VuwR(u, 23), Q6_V_vsplat_R(127));
    const HVX_Vector sh   = Q6_Vw_vmin_VwVw(Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(14), e), Q6_V_vsplat_R(31));
    const HVX_Vector m    = Q6_V_vor_VV(Q6_V_vand_VV(u, Q6_V_vsplat_R(0x7fffff)), Q6_V_vsplat_R(0x800000));
    const HVX_Vector q    = Q6_Vw_vasr_VwVw(m, sh);
    const HVX_Vector one  = Q6_V_vsplat_R(1);
    const HVX_Vector half = Q6_Vw_vasl_VwVw(one, Q6_Vw_vsub_VwVw(sh, one));
    const HVX_Vector rem  = Q6_V_vand_VV(m, Q6_Vw_vsub_VwVw(Q6_Vw_vadd_VwVw(half, half), one));
    const HVX_VectorPred up = Q6_Q_or_QQ(Q6_Q_vcmp_gt_VuwVuw(rem, half),
                                         Q6_Q_vcmp_eqand_QVwVw(Q6_Q_vcmp_eq_VwVw(rem, half), Q6_V_vand_VV(q, one), one));
    const HVX_Vector sub  = Q6_Vw_vadd_VwVw(q, Q6_V_vand_QR(up, 1));
    HVX_Vector r = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(Q6_V_vsplat_R(0x3c800000), u), sub, nrm);
    r = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(Q6_V_vsplat_R(0x43e00000), u), r, Q6_V_vsplat_R(0x7e));
    r = Q6_V_vor_VV(r, sign);
    return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(u, Q6_V_vsplat_R(0x7f800000)), Q6_V_vsplat_R(0x7f), r);
}

// FP8 has no blocks (any row length): the last n % 32 values are converted by scalar code into bytes the
// vector stores leave alone.
static inline void kv_quantize_row_f8(uint8_t * dst, const float * src, uint32_t n) {
    const uint32_t n32 = n / 32 * 32;
    for (uint32_t i = n32; i < n; ++i) {
        dst[i] = ggml_fp32_to_f8_e4m3(src[i]);
    }
    for (uint32_t i = 0; i < n32; i += 128) {
        const uint32_t   nv = MIN(4u, (n32 - i) / 32);
        const HVX_Vector z  = Q6_V_vzero();
        const HVX_Vector w0 = kv_f8_word(*(const HVX_UVector *) (src + i));
        const HVX_Vector w1 = nv > 1 ? kv_f8_word(*(const HVX_UVector *) (src + i + 32)) : z;
        const HVX_Vector w2 = nv > 2 ? kv_f8_word(*(const HVX_UVector *) (src + i + 64)) : z;
        const HVX_Vector w3 = nv > 3 ? kv_f8_word(*(const HVX_UVector *) (src + i + 96)) : z;
        const HVX_Vector b  = Q6_Vb_vpacke_VhVh(Q6_Vh_vpacke_VwVw(w3, w2), Q6_Vh_vpacke_VwVw(w1, w0));
        kv_store_masked(dst + i, b, Q6_Q_vsetq2_R(nv * 32));
    }
}

// Cache writes follow the reference block rule (upstream quantize_row_*_ref): bit-identical for finite
// inputs. The existing SET_ROWS DMA pipeline parallelizes rows and keeps DDR access out of these loops.
static inline void kv_quantize_row(void * dst, const float * src, uint32_t n, int type) {
    switch (type) {
        case HTP_TYPE_Q4_0: kv_quantize_row_hvx(dst, src, n, HTP_TYPE_Q4_0); return;
        case HTP_TYPE_Q4_1: kv_quantize_row_hvx(dst, src, n, HTP_TYPE_Q4_1); return;
        case HTP_TYPE_Q5_0: kv_quantize_row_hvx(dst, src, n, HTP_TYPE_Q5_0); return;
        case HTP_TYPE_Q5_1: kv_quantize_row_hvx(dst, src, n, HTP_TYPE_Q5_1); return;
        case HTP_TYPE_IQ4_NL: kv_quantize_row_iq4_hvx(dst, src, n); return;
        case HTP_TYPE_F8_E4M3: kv_quantize_row_f8(dst, src, n); return;
        default: assert(0);
    }
}

#endif
