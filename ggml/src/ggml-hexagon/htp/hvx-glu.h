#ifndef HVX_GLU_H
#define HVX_GLU_H

// Per-vector GLU activations. The GLU ops and the matmul store that fuses them into the gate/up
// matmul both use these operations, so fused and separate results are the same on every HVX
// generation (qf32 on v73/v75, IEEE single precision from v79).

#include "hvx-utils.h"
#include "hvx-sigmoid.h"

static const float GELU_COEF_A     = 0.044715f;
static const float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

// geglu(x, g) = gelu(x) * g with the tanh approximation, the operations of hvx_geglu_f32_aa()
static inline HVX_Vector hvx_vec_geglu_f32(HVX_Vector x, HVX_Vector g) {
    const HVX_Vector v_coef_a_times_sqrt = hvx_vec_splat_f32(GELU_COEF_A * SQRT_2_OVER_PI);
    const HVX_Vector v_sqrt_2_pi         = hvx_vec_splat_f32(SQRT_2_OVER_PI);
    const HVX_Vector v_one               = hvx_vec_splat_f32(1.0f);
    const HVX_Vector v_max_exp           = hvx_vec_splat_f32(87.0f);
    const HVX_Vector v_min_exp           = hvx_vec_splat_f32(-87.0f);

    HVX_Vector x2    = hvx_vec_mul_f32_f32(x, x);
    HVX_Vector coef  = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt), v_sqrt_2_pi);
    HVX_Vector inner = hvx_vec_mul_f32_f32(x, coef);
    HVX_Vector y2    = hvx_vec_add_f32_f32(inner, inner);
    HVX_Vector sig2y = hvx_vec_fast_sigmoid_f32_guard_2it(y2, v_one, v_max_exp, v_min_exp);

    return hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, sig2y), g);
}

// swiglu(x, g) = x * sigmoid(x) * g, the operations of the SWIGLU op (sigmoid pass, then mul-mul)
static inline HVX_Vector hvx_vec_swiglu_f32(HVX_Vector x, HVX_Vector g) {
    const HVX_Vector one     = hvx_vec_splat_f32(1.f);
    const HVX_Vector max_exp = hvx_vec_splat_f32(87.f);
    const HVX_Vector min_exp = hvx_vec_splat_f32(-87.f);

    HVX_Vector sig = hvx_vec_fast_sigmoid_f32_guard(x, one, max_exp, min_exp);
    return hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, sig), g);  // HVX_OP_MUL_F32 of the mul-mul pass
}

#endif /* HVX_GLU_H */
