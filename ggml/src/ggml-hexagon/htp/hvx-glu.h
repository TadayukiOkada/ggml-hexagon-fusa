#ifndef HVX_GLU_H
#define HVX_GLU_H

// Per-vector GLU activations. The GLU ops and the matmul store that fuses them into the gate/up
// matmul both use these operations, so fused and separate results are the same on every HVX
// generation (qf32 on v73/v75, IEEE single precision from v79).

#include "hvx-utils.h"
#include "hvx-sigmoid.h"

static const float GELU_COEF_A     = 0.044715f;
static const float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

static inline HVX_Vector hvx_vec_fast_sigmoid_f32_2it(HVX_Vector v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_Vector in_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_Vector x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(in_int));
    HVX_Vector xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_Vector v3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    v3            = Q6_Vw_vaslacc_VwVwR(v3, in_int, 24);

    HVX_Vector v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_Vector v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    // Newton-Raphson with 2 iterations
    HVX_Vector two_sf = hvx_vec_splat_f32(2.0f);
    HVX_Vector i_sf   = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EEEEBB3), v5);
    HVX_Vector r_qf   = Q6_Vqf32_vmpy_VsfVsf(
        i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v5)))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v5))));
    HVX_Vector res = Q6_Vsf_equals_Vqf32(r_qf);

    res = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    return Q6_Vsf_equals_Vqf32(res);
}

static inline HVX_Vector hvx_vec_fast_sigmoid_f32_guard_2it(HVX_Vector v,
                                                            HVX_Vector one,
                                                            HVX_Vector max_exp,
                                                            HVX_Vector min_exp) {
    const HVX_VectorPred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    const HVX_VectorPred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_Vector out = hvx_vec_fast_sigmoid_f32_2it(v);
    out            = Q6_V_vmux_QVV(pred_max, out, one);
    return Q6_V_vmux_QVV(pred_min, out, Q6_V_vzero());
}

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
