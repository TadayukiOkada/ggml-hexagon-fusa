#ifndef HVX_UTILS_H
#define HVX_UTILS_H

#include "hex-utils.h"

#include "hvx-types.h"
#include "hvx-copy.h"
#include "hvx-repl.h"
#include "hvx-scale.h"
#include "hvx-exp.h"
#include "hvx-erf.h"
#include "hvx-inverse.h"
#include "hvx-reduce.h"
#include "hvx-sigmoid.h"
#include "hvx-sqrt.h"
#include "hvx-arith.h"
#include "hvx-div.h"
#include "hvx-floor.h"
#include "hvx-sin-cos.h"
#include "hvx-base.h"
#include "hvx-pow.h"
#include "hvx-log.h"
#include "hvx-norm.h"

// softplus(x) = x for x > 20, else max(x, 0) + log1p(exp(-|x|))
static inline HVX_Vector hvx_vec_softplus_f32_inline(HVX_Vector x) {
    const HVX_Vector zero   = Q6_V_vzero();
    const HVX_Vector twenty = hvx_vec_splat_f32(20.0f);
    const HVX_Vector one    = hvx_vec_splat_f32(1.0f);
    const HVX_Vector thresh = hvx_vec_splat_f32(0.25f);

    const HVX_Vector u = hvx_vec_neg_f32(hvx_vec_abs_f32(x));
    const HVX_Vector e = hvx_vec_exp_f32(u);

    // Degree-6 Taylor log1p(e) for e < 0.25 avoids catastrophic cancellation in log(1 + e)
    HVX_Vector lp_small = hvx_vec_mul_f32_f32(e, hvx_vec_splat_f32(-1.0f / 6.0f));
    lp_small = hvx_vec_mul_f32_f32(e, hvx_vec_add_f32_f32(lp_small, hvx_vec_splat_f32( 1.0f / 5.0f)));
    lp_small = hvx_vec_mul_f32_f32(e, hvx_vec_add_f32_f32(lp_small, hvx_vec_splat_f32(-1.0f / 4.0f)));
    lp_small = hvx_vec_mul_f32_f32(e, hvx_vec_add_f32_f32(lp_small, hvx_vec_splat_f32( 1.0f / 3.0f)));
    lp_small = hvx_vec_mul_f32_f32(e, hvx_vec_add_f32_f32(lp_small, hvx_vec_splat_f32(-1.0f / 2.0f)));
    lp_small = hvx_vec_mul_f32_f32(e, hvx_vec_add_f32_f32(lp_small, one));

    const HVX_Vector lp_big = hvx_vec_log_f32(hvx_vec_add_f32_f32(one, e));
    const HVX_Vector lp     = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(thresh, e), lp_small, lp_big);
    const HVX_Vector pos_x  = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(x, zero), x, zero);
    const HVX_Vector sp     = hvx_vec_add_f32_f32(pos_x, lp);

    return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(x, twenty), x, sp);
}

#endif /* HVX_UTILS_H */
