#ifndef HVX_SCALE_H
#define HVX_SCALE_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "hvx-base.h"

#define hvx_scale_f32_loop_body(dst_type, src_type, vec_store)                       \
    do {                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                 \
        src_type * restrict vsrc = (src_type *) src;                                 \
                                                                                     \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                    \
                                                                                     \
        const uint32_t elem_size = sizeof(float);                                    \
        const uint32_t epv = 128 / elem_size;                                        \
        const uint32_t nvec = n / epv;                                               \
        const uint32_t nloe = n % epv;                                               \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        for (; i < nvec; ++i) {                                                      \
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vdst[i]      = Q6_Vsf_equals_Vqf32(v);                                   \
        }                                                                            \
        if (nloe) {                                                                  \
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vec_store((void *) &vdst[i], nloe * elem_size, Q6_Vsf_equals_Vqf32(v));  \
        }                                                                            \
    } while(0)

static inline void hvx_scale_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    assert((size_t) dst % 128 == 0);
    assert((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_scale_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    assert((size_t) dst % 128 == 0);
    hvx_scale_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_scale_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    assert((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_scale_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    hvx_scale_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_scale_f32(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    if (((size_t) dst & 127) == 0) {
        if (((size_t) src & 127) == 0) {
            hvx_scale_f32_aa(dst, src, n, scale);
        } else {
            hvx_scale_f32_au(dst, src, n, scale);
        }
    } else {
        if (((size_t) src & 127) == 0) {
            hvx_scale_f32_ua(dst, src, n, scale);
        } else {
            hvx_scale_f32_uu(dst, src, n, scale);
        }
    }
}

#define hvx_scale_offset_f32_loop_body(dst_type, src_type, vec_store)                     \
    do {                                                                                  \
        dst_type * restrict vdst = (dst_type *) dst;                                      \
        src_type * restrict vsrc = (src_type *) src;                                      \
                                                                                          \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                         \
        HVX_Vector vo = hvx_vec_splat_f32(offset);                                        \
                                                                                          \
        const uint32_t elem_size = sizeof(float);                                         \
        const uint32_t epv = 128 / elem_size;                                             \
        const uint32_t nvec = n / epv;                                                    \
        const uint32_t nloe = n % epv;                                                    \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        for (; i < nvec; ++i) {                                                           \
            HVX_Vector v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vdst[i] = Q6_Vsf_equals_Vqf32(v);                                             \
        }                                                                                 \
        if (nloe) {                                                                       \
            HVX_Vector v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vec_store((void *) &vdst[i], nloe * elem_size, Q6_Vsf_equals_Vqf32(v));       \
        }                                                                                 \
    } while(0)

static inline void hvx_scale_offset_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    assert((size_t) dst % 128 == 0);
    assert((size_t) src % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_scale_offset_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    assert((size_t) dst % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_scale_offset_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    assert((size_t) src % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_scale_offset_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    hvx_scale_offset_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_scale_offset_f32(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    if (((size_t) dst & 127) == 0) {
        if (((size_t) src & 127) == 0) {
            hvx_scale_offset_f32_aa(dst, src, n, scale, offset);
        } else {
            hvx_scale_offset_f32_au(dst, src, n, scale, offset);
        }
    } else {
        if (((size_t) src & 127) == 0) {
            hvx_scale_offset_f32_ua(dst, src, n, scale, offset);
        } else {
            hvx_scale_offset_f32_uu(dst, src, n, scale, offset);
        }
    }
}

#if __HVX_ARCH__ < 79
// On __HVX_ARCH__ < 79, HVX qf32 arithmetic/conversion can introduce a +/-1 F32 ULP bias
// and Q6_Vhf_equals_Wqf32 rounds halfway ties away from zero (RNA). Use scalar IEEE-754
// float hardware with pure-integer bit-exact F16 <-> F32 RNE conversion (no libcalls).
static inline float hvx_scale_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {
            int e = -14;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x03FFu;
            u = sign | ((uint32_t) (e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static inline uint16_t hvx_scale_f32_to_f16_rne(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t sign  = (u >> 16) & 0x8000u;
    uint32_t abs_u = u & 0x7FFFFFFFu;
    uint32_t h;
    if (abs_u > 0x7F800000u) {
        h = 0x7E00u | ((abs_u >> 13) & 0x01FFu);
    } else if (abs_u >= 0x477FF000u) {
        h = 0x7C00u;
    } else if (abs_u < 0x38800000u) {
        if (abs_u <= 0x33000000u) {
            h = 0u;
        } else {
            uint32_t shift = 126u - (abs_u >> 23);
            uint32_t m     = 0x00800000u | (abs_u & 0x007FFFFFu);
            uint32_t lsb   = (m >> shift) & 1u;
            uint32_t round = (1u << (shift - 1u)) - 1u;
            h = (m + round + lsb) >> shift;
        }
    } else {
        uint32_t lsb = (abs_u >> 13) & 1u;
        h = ((abs_u - 0x38000000u) + 0x00000FFFu + lsb) >> 13;
    }
    return (uint16_t) (sign | h);
}

#define hvx_scale_offset_f16_loop_body(dst_type, src_type, vec_store)            \
    do {                                                                         \
        uint16_t * restrict       vdst = (uint16_t *) dst;                       \
        const uint16_t * restrict vsrc = (const uint16_t *) src;                 \
        for (int i = 0; i < n; ++i) {                                            \
            vdst[i] = hvx_scale_f32_to_f16_rne(                                  \
                scale * hvx_scale_f16_to_f32(vsrc[i]) + offset);                 \
        }                                                                        \
    } while (0)
#else
#define hvx_scale_offset_f16_loop_body(dst_type, src_type, vec_store)                                                \
    do {                                                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                                                 \
        src_type * restrict vsrc = (src_type *) src;                                                                 \
                                                                                                                     \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                                                    \
        HVX_Vector vo = hvx_vec_splat_f32(offset);                                                                   \
                                                                                                                     \
        const uint32_t nvec = n / VLEN_FP16;                                                                         \
        const uint32_t nloe = n % VLEN_FP16;                                                                         \
                                                                                                                     \
        uint32_t i = 0;                                                                                              \
                                                                                                                     \
        _Pragma("unroll(4)")                                                                                         \
        for (; i < nvec; ++i) {                                                                                      \
            HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);                                                          \
            HVX_Vector r0 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo);                            \
            HVX_Vector r1 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo);                            \
            vdst[i] = hvx_vec_f32_to_f16(r0, r1);                                                                    \
        }                                                                                                            \
        if (nloe) {                                                                                                  \
            HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);                                                          \
            HVX_Vector r0 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(p), vs), vo);                            \
            HVX_Vector r1 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(p), vs), vo);                            \
            HVX_Vector v = hvx_vec_f32_to_f16(r0, r1);                                                               \
            vec_store((void *) &vdst[i], nloe * SIZEOF_FP16, v);                                                     \
        }                                                                                                            \
    } while(0)
#endif

static inline void hvx_scale_offset_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    assert((size_t) dst % 128 == 0);
    assert((size_t) src % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_scale_offset_f16_au(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    assert((size_t) dst % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_scale_offset_f16_ua(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    assert((size_t) src % 128 == 0);
    hvx_scale_offset_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_scale_offset_f16_uu(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    hvx_scale_offset_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_scale_offset_f16(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    if (((size_t) dst & 127) == 0) {
        if (((size_t) src & 127) == 0) {
            hvx_scale_offset_f16_aa(dst, src, n, scale, offset);
        } else {
            hvx_scale_offset_f16_au(dst, src, n, scale, offset);
        }
    } else {
        if (((size_t) src & 127) == 0) {
            hvx_scale_offset_f16_ua(dst, src, n, scale, offset);
        } else {
            hvx_scale_offset_f16_uu(dst, src, n, scale, offset);
        }
    }
}

#endif // HVX_SCALE_H
