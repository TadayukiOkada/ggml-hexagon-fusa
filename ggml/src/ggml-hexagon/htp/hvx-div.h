#ifndef HVX_DIV_H
#define HVX_DIV_H

#include <HAP_farf.h>

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "hvx-base.h"
#include "hex-utils.h"
#include "hvx-inverse.h"
#include "hvx-arith.h"

#if __HVX_ARCH__ < 79
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#define HVX_OP_MUL_F16(a, b) Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(a, b))
#else
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#define HVX_OP_MUL_F16(a, b) Q6_Vhf_vmpy_VhfVhf(a, b)
#endif

// Restore IEEE half special values after the reciprocal and qf32 conversion,
// which may otherwise flush infinities or lose signed zero.
static inline HVX_Vector hvx_div_f16_special(HVX_Vector result, HVX_Vector a, HVX_Vector b) {
    const HVX_Vector abs_mask = Q6_Vh_vsplat_R(0x7fff);
    const HVX_Vector inf = Q6_Vh_vsplat_R(0x7c00);
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector aa = Q6_V_vand_VV(a, abs_mask);
    const HVX_Vector ab = Q6_V_vand_VV(b, abs_mask);
    const HVX_Vector sign = Q6_V_vand_VV(Q6_V_vxor_VV(a, b), Q6_Vh_vsplat_R(0x8000));
    const HVX_VectorPred az = Q6_Q_vcmp_eq_VhVh(aa, zero);
    const HVX_VectorPred bz = Q6_Q_vcmp_eq_VhVh(ab, zero);
    const HVX_VectorPred ai = Q6_Q_vcmp_eq_VhVh(aa, inf);
    const HVX_VectorPred bi = Q6_Q_vcmp_eq_VhVh(ab, inf);
    HVX_VectorPred invalid = Q6_Q_or_QQ(Q6_Q_vcmp_gt_VuhVuh(aa, inf), Q6_Q_vcmp_gt_VuhVuh(ab, inf));
    invalid = Q6_Q_or_QQ(invalid, Q6_Q_or_QQ(Q6_Q_and_QQ(az, bz), Q6_Q_and_QQ(ai, bi)));
    HVX_Vector magnitude = Q6_V_vand_VV(result, abs_mask);
    // qf32-to-half overflow can encode NaN rather than infinity.
    magnitude = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuhVuh(magnitude, Q6_Vh_vsplat_R(0x7bff)), inf, magnitude);
    magnitude = Q6_V_vmux_QVV(Q6_Q_or_QQ(az, bi), zero, magnitude);
    magnitude = Q6_V_vmux_QVV(Q6_Q_or_QQ(bz, ai), inf, magnitude);
    result = Q6_V_vor_VV(magnitude, sign);
    return Q6_V_vmux_QVV(invalid, Q6_Vh_vsplat_R(0x7e00), result);
}

// |a| / |b| for finite nonzero half magnitudes: widen to f32, divide with an f32 reciprocal and
// round to half once. Special operands give garbage here; hvx_div_f16_special replaces those lanes,
// so the reciprocal needs no inf/NaN guard.
static inline HVX_Vector hvx_vec_div_f16_mag(HVX_Vector aa, HVX_Vector ab, HVX_Vector hf_one) {
#if __HVX_ARCH__ < 79
    const HVX_VectorPair a_qf = Q6_Wqf32_vmpy_VhfVhf(aa, hf_one);  // *1.0, exact
    const HVX_VectorPair b_qf = Q6_Wqf32_vmpy_VhfVhf(ab, hf_one);
    const HVX_Vector     r0   = hvx_vec_inverse_f32(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(b_qf)));
    const HVX_Vector     r1   = hvx_vec_inverse_f32(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(b_qf)));
    const HVX_Vector     q0   = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(a_qf)), r0);
    const HVX_Vector     q1   = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(a_qf)), r1);
    return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q1, q0));
#else
    const HVX_VectorPair a_sf = Q6_Wsf_vmpy_VhfVhf(aa, hf_one);  // *1.0
    const HVX_VectorPair b_sf = Q6_Wsf_vmpy_VhfVhf(ab, hf_one);
    const HVX_Vector     q0   = HVX_OP_MUL_F32(Q6_V_lo_W(a_sf), hvx_vec_inverse_f32(Q6_V_lo_W(b_sf)));
    const HVX_Vector     q1   = HVX_OP_MUL_F32(Q6_V_hi_W(a_sf), hvx_vec_inverse_f32(Q6_V_hi_W(b_sf)));
    return Q6_Vhf_vcvt_VsfVsf(q0, q1);
#endif
}

static inline HVX_Vector hvx_vec_div_f16(HVX_Vector a, HVX_Vector b, HVX_Vector hf_one) {
    const HVX_Vector abs_mask = Q6_Vh_vsplat_R(0x7fff);
    HVX_Vector result = hvx_vec_div_f16_mag(Q6_V_vand_VV(a, abs_mask), Q6_V_vand_VV(b, abs_mask), hf_one);
    return hvx_div_f16_special(result, a, b);
}

// 1/|b| for hvx_div_f16_by_const, in the form its multiply takes (qf32 below v79).
static inline HVX_Vector hvx_div_f16_const_recip(float b) {
#if __HVX_ARCH__ < 79
    return Q6_Vqf32_vadd_VsfVsf(hvx_vec_splat_f32(1.0f / fabsf(b)), Q6_V_vzero());
#else
    return hvx_vec_splat_f32(1.0f / fabsf(b));
#endif
}

// a / b for a finite nonzero constant b. a is widened exactly, multiplied by the f32 reciprocal of
// |b| and rounded to half once. With such a b only an infinite or NaN a (passed through) and an
// overflow (to infinity) are special; the sign is the xor of the operand signs.
static inline HVX_Vector hvx_div_f16_by_const(HVX_Vector a, HVX_Vector recip, HVX_Vector b, HVX_Vector hf_one) {
    const HVX_Vector abs_mask = Q6_Vh_vsplat_R(0x7fff);
    const HVX_Vector max_hf   = Q6_Vh_vsplat_R(0x7bff);
    const HVX_Vector aa       = Q6_V_vand_VV(a, abs_mask);
#if __HVX_ARCH__ < 79
    const HVX_VectorPair a_qf = Q6_Wqf32_vmpy_VhfVhf(aa, hf_one);  // *1.0, exact
    HVX_Vector q = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(a_qf), recip),
                                                        Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(a_qf), recip)));
#else
    const HVX_VectorPair a_sf = Q6_Wsf_vmpy_VhfVhf(aa, hf_one);  // *1.0
    HVX_Vector q = Q6_Vhf_vcvt_VsfVsf(HVX_OP_MUL_F32(Q6_V_lo_W(a_sf), recip), HVX_OP_MUL_F32(Q6_V_hi_W(a_sf), recip));
#endif
    q = Q6_V_vand_VV(q, abs_mask);
    // qf32-to-half overflow can encode NaN rather than infinity.
    q = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuhVuh(q, max_hf), Q6_Vh_vsplat_R(0x7c00), q);
    q = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuhVuh(aa, max_hf), aa, q);
    return Q6_V_vor_VV(q, Q6_V_vand_VV(Q6_V_vxor_VV(a, b), Q6_Vh_vsplat_R(0x8000)));
}

#define hvx_div_scaler_f16_loop_body(dst_type, src_type, vec_store)                                    \
    do {                                                                                               \
        dst_type * restrict vdst = (dst_type *) dst;                                                   \
        src_type * restrict vsrc = (src_type *) src;                                                   \
                                                                                                       \
        const HVX_Vector hf_one = Q6_Vh_vsplat_R(0x3C00);                                              \
        const HVX_Vector vb     = hvx_vec_splat_f16(val);                                              \
        const float      bf     = (float) val;                                                         \
                                                                                                       \
        const uint32_t nvec = n / VLEN_FP16;                                                           \
        const uint32_t nloe = n % VLEN_FP16;                                                           \
                                                                                                       \
        uint32_t i = 0;                                                                                \
                                                                                                       \
        if (bf != 0.0f && isfinite(bf)) {                                                              \
            const HVX_Vector recip = hvx_div_f16_const_recip(bf);                                      \
            _Pragma("unroll(4)")                                                                       \
            for (; i < nvec; i++) {                                                                    \
                vdst[i] = hvx_div_f16_by_const(vsrc[i], recip, vb, hf_one);                            \
            }                                                                                          \
            if (nloe) {                                                                                \
                vec_store((void *) &vdst[i], nloe * SIZEOF_FP16, hvx_div_f16_by_const(vsrc[i], recip, vb, hf_one)); \
            }                                                                                          \
        } else {                                                                                       \
            for (; i < nvec; i++) {                                                                    \
                vdst[i] = hvx_vec_div_f16(vsrc[i], vb, hf_one);                                        \
            }                                                                                          \
            if (nloe) {                                                                                \
                vec_store((void *) &vdst[i], nloe * SIZEOF_FP16, hvx_vec_div_f16(vsrc[i], vb, hf_one)); \
            }                                                                                          \
        }                                                                                              \
    } while(0)

static inline void hvx_div_scalar_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, const _Float16 val, uint32_t n) {
    assert((uintptr_t) dst % 128 == 0);
    assert((uintptr_t) src % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}
static inline void hvx_div_scalar_f16_au(uint8_t * restrict dst, const uint8_t * restrict src, const _Float16 val, uint32_t n) {
    assert((uintptr_t) dst % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}
static inline void hvx_div_scalar_f16_ua(uint8_t * restrict dst, const uint8_t * restrict src, const _Float16 val, uint32_t n) {
    assert((uintptr_t) src % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}
static inline void hvx_div_scalar_f16_uu(uint8_t * restrict dst, const uint8_t * restrict src, const _Float16 val, uint32_t n) {
    hvx_div_scaler_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

#define hvx_div_f16_loop_body(dst_type, src0_type, src1_type, vec_store)                  \
    do {                                                                                  \
        dst_type * restrict vdst = (dst_type *) dst;                                      \
        src0_type * restrict vsrc0 = (src0_type *) src0;                                  \
        src1_type * restrict vsrc1 = (src1_type *) src1;                                  \
                                                                                          \
        const HVX_Vector hf_one = Q6_Vh_vsplat_R(0x3C00);                                 \
                                                                                          \
        const uint32_t nvec = n / VLEN_FP16;                                              \
        const uint32_t nloe = n % VLEN_FP16;                                              \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        for (; i < nvec; i++) {                                                           \
            vdst[i] = hvx_vec_div_f16(vsrc0[i], vsrc1[i], hf_one);                        \
        }                                                                                 \
        if (nloe) {                                                                       \
            HVX_Vector res = hvx_vec_div_f16(vsrc0[i], vsrc1[i], hf_one);                 \
            vec_store((void *) &vdst[i], nloe * SIZEOF_FP16, res);                        \
        }                                                                                 \
    } while(0)

#define hvx_div_f32_loop_body(dst_type, src0_type, src1_type, vec_store)             \
    do {                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                 \
        src0_type * restrict vsrc0 = (src0_type *) src0;                             \
        src1_type * restrict vsrc1 = (src1_type *) src1;                             \
                                                                                     \
        const HVX_Vector nan_inf_mask = Q6_V_vsplat_R(0x7f800000);                   \
                                                                                     \
        const uint32_t nvec = n / VLEN_FP32;                                         \
        const uint32_t nloe = n % VLEN_FP32;                                         \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        for (; i < nvec; i++) {                                                      \
            HVX_Vector inv_src1 = hvx_vec_inverse_f32_guard(vsrc1[i], nan_inf_mask); \
            HVX_Vector res = HVX_OP_MUL_F32(vsrc0[i], inv_src1);                     \
            vdst[i] = res;                                                           \
        }                                                                            \
        if (nloe) {                                                                  \
            HVX_Vector inv_src1 = hvx_vec_inverse_f32_guard(vsrc1[i], nan_inf_mask); \
            HVX_Vector res = HVX_OP_MUL_F32(vsrc0[i], inv_src1);                     \
            vec_store((void *) &vdst[i], nloe * SIZEOF_FP32, res);                   \
        }                                                                            \
    } while(0)

// Generic macro to define alignment permutations for an op
#define DEFINE_HVX_DIV_OP_VARIANTS(OP_NAME, OP_LOOP_BODY)                                                                            \
static inline void OP_NAME##_aaa(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) dst % 128 == 0);                                                                                              \
    assert((uintptr_t) src0 % 128 == 0);                                                                                             \
    assert((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_Vector, HVX_Vector, HVX_Vector, hvx_vec_store_a);                                                               \
}                                                                                                                                    \
static inline void OP_NAME##_aau(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) dst % 128 == 0);                                                                                              \
    assert((uintptr_t) src0 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_Vector, HVX_Vector, HVX_UVector, hvx_vec_store_a);                                                              \
}                                                                                                                                    \
static inline void OP_NAME##_aua(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) dst % 128 == 0);                                                                                              \
    assert((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_Vector, HVX_UVector, HVX_Vector, hvx_vec_store_a);                                                              \
}                                                                                                                                    \
static inline void OP_NAME##_auu(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) dst % 128 == 0);                                                                                              \
    OP_LOOP_BODY(HVX_Vector, HVX_UVector, HVX_UVector, hvx_vec_store_a);                                                             \
}                                                                                                                                    \
static inline void OP_NAME##_uaa(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) src0 % 128 == 0);                                                                                             \
    assert((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_Vector, HVX_Vector, hvx_vec_store_u);                                                              \
}                                                                                                                                    \
static inline void OP_NAME##_uau(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) src0 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_Vector, HVX_UVector, hvx_vec_store_u);                                                             \
}                                                                                                                                    \
static inline void OP_NAME##_uua(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) src1 % 128 == 0);                                                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_UVector, HVX_Vector, hvx_vec_store_u);                                                             \
}                                                                                                                                    \
static inline void OP_NAME##_uuu(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    OP_LOOP_BODY(HVX_UVector, HVX_UVector, HVX_UVector, hvx_vec_store_u);                                                            \
}                                                                                                                                    \

// Dispatcher logic
#define HVX_DIV_DISPATCHER(OP_NAME)                                                                                                          \
static inline void OP_NAME(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, const uint32_t num_elems) { \
    if (hex_is_aligned((void *) dst, 128)) {                                                                                                 \
        if (hex_is_aligned((void *) src0, 128)) {                                                                                            \
            if (hex_is_aligned((void *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, num_elems);                                               \
            else                                    OP_NAME##_aau(dst, src0, src1, num_elems);                                               \
        } else {                                                                                                                             \
            if (hex_is_aligned((void *) src1, 128)) OP_NAME##_aua(dst, src0, src1, num_elems);                                               \
            else                                    OP_NAME##_auu(dst, src0, src1, num_elems);                                               \
        }                                                                                                                                    \
    } else {                                                                                                                                 \
        if (hex_is_aligned((void *) src0, 128)) {                                                                                            \
            if (hex_is_aligned((void *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, num_elems);                                               \
            else                                    OP_NAME##_uau(dst, src0, src1, num_elems);                                               \
        } else {                                                                                                                             \
            if (hex_is_aligned((void *) src1, 128)) OP_NAME##_uua(dst, src0, src1, num_elems);                                               \
            else                                    OP_NAME##_uuu(dst, src0, src1, num_elems);                                               \
        }                                                                                                                                    \
    }                                                                                                                                        \
}

DEFINE_HVX_DIV_OP_VARIANTS(hvx_div_f32, hvx_div_f32_loop_body)
DEFINE_HVX_DIV_OP_VARIANTS(hvx_div_f16, hvx_div_f16_loop_body)

HVX_DIV_DISPATCHER(hvx_div_f32)
HVX_DIV_DISPATCHER(hvx_div_f16)

#undef HVX_OP_MUL_F32
#undef HVX_OP_MUL_F16

#endif // HVX_DIV_H
