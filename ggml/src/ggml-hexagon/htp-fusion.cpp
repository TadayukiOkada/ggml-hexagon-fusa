#include "htp-fusion.h"
#include "htp-kparams.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstring>

static inline bool htp_same_tensor_data(const ggml_tensor * a, const ggml_tensor * b) {
    return a && b && (a == b || a->data == b->data);
}

static inline bool htp_same_shape_and_contiguity(const ggml_tensor * a, const ggml_tensor * b) {
    return ggml_are_same_shape(a, b) && (ggml_is_contiguous(a) == ggml_is_contiguous(b));
}

static inline bool htp_inplace_overlap_unsafe(const ggml_tensor * dst, const ggml_tensor * in) {
    return htp_tensors_overlap(dst, in) && (dst->data != in->data || !ggml_are_same_stride(dst, in));
}

static inline bool htp_inputs_overlap_unsafe(const htp_opnode & prev, const ggml_tensor * dst, const ggml_tensor * extra = nullptr) {
    if (extra && htp_inplace_overlap_unsafe(dst, extra)) return true;
    for (const auto * in : prev.inputs) {
        if (htp_inplace_overlap_unsafe(dst, in)) return true;
    }
    return false;
}

static inline bool htp_match_consumer(
    const htp_opnode & prev,
    const htp_opnode & cur,
    bool commutative,
    const ggml_tensor ** other_src = nullptr
) {
    if (prev.outputs.size() != 1) {
        return false;
    }
    const ggml_tensor * prev_out = prev.dst();
    if (!ggml_hexagon_tensor_is_fuseable(prev_out)) {
        return false;
    }
    if (htp_same_tensor_data(cur.src0(), prev_out)) {
        if (other_src) *other_src = cur.src1();
        return true;
    }
    if (commutative && htp_same_tensor_data(cur.src1(), prev_out)) {
        if (other_src) *other_src = cur.src0();
        return true;
    }
    return false;
}

static bool is_supported_mul_mat_nx_kernel(const ggml_tensor * src0, const struct htp_mm_kernel_params * kparams) {
    if (kparams->n_hmx) {
        return htp_mm_kernel_is_hmx_2d(kparams->kernel_type);
    }
    return ggml_hexagon_tensor_is_tiled(src0) &&
           (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK);
}

static bool is_supported_mul_mat_id_nx_kernel(const ggml_tensor * src0, const struct htp_mm_kernel_params * kparams) {
    if (kparams->n_hmx) {
        return htp_mm_kernel_is_hmx_2d(kparams->kernel_type);
    }
    return ggml_hexagon_tensor_is_tiled(src0) &&
           (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK);
}

// --- Fusion rules ---

static bool fuse_allreduce_add(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (opt_ar_select != 2 || prev.opcode != HTP_OP_ALLREDUCE) return false;

    const auto * ar_kparams = prev.kparams<htp_allreduce_kernel_params>();
    const uint32_t rank    = (uint32_t) ar_kparams->rank;
    const uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;

    const ggml_tensor * ar_local = prev.inputs[rank];
    if (!ggml_hexagon_tensor_is_fuseable(ar_local)) return false;

    if (cur.inputs.size() < 2) return false;
    const ggml_tensor * cur_src0   = cur.inputs[0];
    const ggml_tensor * cur_src1   = cur.inputs[1];
    const ggml_tensor * add_dst    = cur.dst();
    const ggml_tensor * res_tensor = nullptr;
    if (htp_same_tensor_data(cur_src0, ar_local)) {
        res_tensor = cur_src1;
    } else if (htp_same_tensor_data(cur_src1, ar_local)) {
        res_tensor = cur_src0;
    } else {
        return false;
    }

    if (ar_local->type != res_tensor->type || ar_local->type != add_dst->type) return false;

    const bool is_same_shape = ggml_are_same_shape(ar_local, res_tensor);
    const bool is_row_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] && ggml_is_vector(res_tensor));
    if (!is_same_shape && !is_row_bcast) return false;

    if (is_same_shape && !ggml_are_same_stride(ar_local, res_tensor)) {
        return false;
    }
    if (ggml_is_contiguous(ar_local) != ggml_is_contiguous(add_dst)) return false;

    const bool is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS && add_dst->data == ar_local->data;

    struct htp_allreduce_kernel_params kparams;
    if (!htp_kparams_solve_allreduce(sess, add_dst, rank, n_ranks, true, is_row_bcast, is_shard_ok, &kparams)) {
        HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: solver failed\n", htp_session_name(sess));
        return false;
    }

    const bool scatter_ok = kparams.n_dsts > 1;
    kparams.mode = scatter_ok ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

    for (uint32_t r = 0; r < n_ranks; r++) {
        const ggml_tensor * ar_src = prev.inputs[r];
        if (!htp_tensors_overlap(add_dst, ar_src)) continue;
        if (scatter_ok && r == rank) continue;
        HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: dst overlaps allreduce src %u\n", htp_session_name(sess), r);
        return false;
    }

    prev.opcode = HTP_OP_ALLREDUCE_ADD;
    prev.inputs.push_back(res_tensor);
    if (kparams.mode != HTP_ALLREDUCE_SHARDED_FANOUT) {
        prev.outputs = { add_dst };
    }
    prev.fused.push_back(cur.node);
    prev.set_kparams(kparams);
    return true;
}

static bool fuse_rms_norm_mul(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_RMS_NORM) return false;

    const ggml_tensor * weight = nullptr;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ true, &weight)) return false;

    const ggml_tensor * src0 = prev.src0();
    const ggml_tensor * dst  = cur.dst();
    if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
        (weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_F16) ||
        src0->ne[0] != weight->ne[0] || !htp_same_shape_and_contiguity(prev.dst(), dst)) return false;
    if (htp_inputs_overlap_unsafe(prev, dst, weight)) return false;

    const bool is_row_bcast  = ggml_is_vector(weight);
    const bool is_dim1_bcast = (weight->ne[1] == src0->ne[1] && ggml_is_matrix(weight));
    const bool is_same_shape = ggml_are_same_shape(src0, weight);
    if (!is_row_bcast && !is_dim1_bcast && !is_same_shape) return false;

    struct htp_unary_kernel_params kparams;
    if (!htp_kparams_solve_rms_norm_mul(sess, src0, weight, dst, &kparams)) {
        HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_MUL fusion: VTCM needed (%d) > budget (%zu)\n",
                    htp_session_name(sess), kparams.vtcm_size, htp_session_vtcm_size(sess));
        return false;
    }

    prev.opcode  = HTP_OP_RMS_NORM_MUL;
    prev.inputs  = { src0, weight };
    prev.outputs = { dst };
    prev.fused.push_back(cur.node);
    prev.set_kparams(kparams);
    return true;
}

// post-norm residual: dst = rms_norm(x) * w + r
static bool fuse_rms_norm_mul_add(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_RMS_NORM_MUL) return false;

    const ggml_tensor * r = nullptr;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ true, &r)) return false;

    const ggml_tensor * src0   = prev.src0();
    const ggml_tensor * weight = prev.inputs[1];
    const ggml_tensor * dst    = cur.dst();
    if ((r->type != GGML_TYPE_F32 && r->type != GGML_TYPE_F16) ||
        (weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_F16)) return false;
    if (!ggml_are_same_shape(src0, r) || (ggml_is_contiguous(src0) != ggml_is_contiguous(r))) return false;
    if (htp_inputs_overlap_unsafe(prev, dst, r)) return false;

    struct htp_unary_kernel_params kparams;
    if (!htp_kparams_solve_rms_norm_mul_add(sess, src0, weight, r, dst, &kparams)) return false;

    prev.opcode = HTP_OP_RMS_NORM_MUL_ADD;
    prev.inputs.push_back(r);
    prev.outputs[0] = dst;
    prev.fused.push_back(cur.node);
    prev.set_kparams(kparams);
    return true;
}

static bool fuse_rms_norm_scale(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    GGML_UNUSED(sess);
    if (prev.opcode != HTP_OP_RMS_NORM && prev.opcode != HTP_OP_RMS_NORM_MUL) return false;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ false)) return false;

    const ggml_tensor * src0 = prev.src0();
    const ggml_tensor * dst  = cur.dst();
    if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) || !htp_same_shape_and_contiguity(prev.dst(), dst)) return false;
    if (htp_inputs_overlap_unsafe(prev, dst)) return false;

    float scale = 0.0f;
    float bias  = 0.0f;
    memcpy(&scale, (const float *) cur.params + 0, sizeof(float));
    memcpy(&bias,  (const float *) cur.params + 1, sizeof(float));
    if (bias != 0.0f || scale == 0.0f) return false;

    float prev_scale = 0.0f;
    memcpy(&prev_scale, &prev.params[1], sizeof(float));
    if (prev_scale != 0.0f) {
        scale *= prev_scale;
    }
    memcpy(&prev.params[1], &scale, sizeof(float));

    prev.outputs[0] = dst;
    prev.fused.push_back(cur.node);
    return true;
}

static bool fuse_mul_mat_add(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_MUL_MAT) return false;

    const ggml_tensor * src2 = nullptr;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ true, &src2)) return false;
    if ((src2->type != GGML_TYPE_F32 && src2->type != GGML_TYPE_F16) ||
        !ggml_is_contiguous_to_2(src2) ||
        src2->ne[0] != cur.dst()->ne[0]) {
        return false;
    }

    const ggml_tensor * src0 = prev.node->src[0];
    const ggml_tensor * src1 = prev.node->src[1];
    const ggml_tensor * dst  = cur.node;
    if (!ggml_are_same_shape(prev.node, dst)) return false;

    // Require matching ne[0], valid broadcast dims (1 or equal), and no unsafe overlap.
    if (src2->ne[0] != dst->ne[0] ||
        (src2->ne[1] != 1 && src2->ne[1] != dst->ne[1]) ||
        (src2->ne[2] != 1 && src2->ne[2] != dst->ne[2]) ||
        (src2->ne[3] != 1 && src2->ne[3] != dst->ne[3])) {
        return false;
    }
    if (htp_tensors_overlap(dst, src0) || htp_tensors_overlap(dst, src1) || htp_inplace_overlap_unsafe(dst, src2)) {
        return false;
    }

    const bool prev_collapsed = htp_matmul_can_collapse(src0, src1, prev.node);
    const bool cur_collapsed  = htp_matmul_add_can_collapse(src0, src1, src2, dst);
    if (prev_collapsed != cur_collapsed) {
        return false;
    }

    const auto * orig_kparams = prev.kparams<htp_mm_kernel_params>();
    struct htp_mm_kernel_params kparams;
    if (!htp_kparams_solve_matmul_add(sess, src0, src1, src2, dst, &kparams)) {
        return false;
    }

    const int  src1_nrows = (int) ggml_nrows(src1);
    const bool is_batched = (!ggml_is_matrix(src0) || !ggml_is_matrix(src1));
    if (kparams.n_hmx == 0 && (src1_nrows != 1 || is_batched)) return false;

    if (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
        if (kparams.m_chunk < orig_kparams->m_chunk ||
            kparams.n_act_hvx < orig_kparams->n_act_hvx) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, hvx %d->%d)\n",
                        htp_session_name(sess), orig_kparams->m_chunk, kparams.m_chunk,
                        orig_kparams->n_chunk, kparams.n_chunk,
                        orig_kparams->n_act_hvx, kparams.n_act_hvx);
            return false;
        }
    }

    prev.opcode  = HTP_OP_MUL_MAT_ADD;
    prev.inputs  = { src0, src1, src2 };
    prev.outputs = { dst };
    prev.fused.push_back(cur.node);
    prev.set_kparams(kparams);
    return true;
}

static bool fuse_mul_mat_nx(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_MUL_MAT && prev.opcode != HTP_OP_MUL_MAT_NX) return false;

    const uint32_t curr_n = (uint32_t) prev.outputs.size();
    if (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 1 >= HTP_OP_MAX_INPUTS) return false;

    const ggml_tensor * w0     = prev.inputs[0];
    const ggml_tensor * x_orig = prev.node->src[1];
    const ggml_tensor * w_in   = cur.node->src[0];
    const ggml_tensor * x_in   = cur.node->src[1];
    const ggml_tensor * d_in   = cur.node;

    auto * orig_kparams = prev.kparams<htp_mm_kernel_params>();
    const auto * cur_kparams = cur.kparams<htp_mm_kernel_params>();

    if (x_in != x_orig || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] ||
        !ggml_is_matrix(w0) || !ggml_is_matrix(w_in) ||
        ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0) ||
        orig_kparams->n_hmx != cur_kparams->n_hmx ||
        orig_kparams->kernel_type != cur_kparams->kernel_type) {
        return false;
    }

    const bool collapse = htp_matmul_can_collapse(w0, x_orig, prev.node) && htp_matmul_can_collapse(w_in, x_in, d_in);
    if (!ggml_is_matrix(x_orig) && !collapse) return false;

    const ggml_tensor * prev_w        = prev.inputs[0];
    ggml_type           prev_dst_type = prev.outputs[0]->type;
    for (size_t i = 1; i < curr_n; ++i) {
        if (prev.inputs[i]->ne[1] > prev_w->ne[1]) prev_w = prev.inputs[i];
        if (ggml_type_size(prev.outputs[i]->type) > ggml_type_size(prev_dst_type)) {
            prev_dst_type = prev.outputs[i]->type;
        }
    }
    const ggml_tensor * w_for_solve  = (w_in->ne[1] > prev_w->ne[1]) ? w_in : prev_w;
    const ggml_type     max_dst_type = (ggml_type_size(d_in->type) > ggml_type_size(prev_dst_type)) ? d_in->type : prev_dst_type;

    if (prev.opcode == HTP_OP_MUL_MAT_NX && w_for_solve == prev_w &&
        ggml_type_size(max_dst_type) == ggml_type_size(prev_dst_type)) {
        orig_kparams->n_weights = curr_n + 1;
    } else {
        struct htp_mm_kernel_params kparams;
        const ggml_tensor x_collapsed = collapse ? htp_tensor_collapse_rows(x_orig) : *x_orig;
        ggml_tensor       d_collapsed = collapse ? htp_tensor_collapse_rows(prev.outputs[0]) : *prev.outputs[0];
        d_collapsed.type = max_dst_type;
        if (!htp_kparams_solve_mmnx(sess, w_for_solve, &x_collapsed, &d_collapsed, curr_n + 1, &kparams)) return false;
        if (!is_supported_mul_mat_nx_kernel(w0, &kparams)) return false;

        prev.set_kparams(kparams);
        prev.opcode = HTP_OP_MUL_MAT_NX;
    }

    prev.inputs.insert(prev.inputs.begin() + curr_n, w_in);
    prev.outputs.push_back(d_in);
    prev.fused.push_back(cur.node);
    return true;
}

static bool htp_is_fuseable_mmnx_glu_op(uint32_t op) {
    return op == HTP_OP_GLU_SWIGLU || op == HTP_OP_GLU_GEGLU ||
           op == HTP_OP_GLU_GEGLU_QUICK || op == HTP_OP_GLU_REGLU;
}

static bool fuse_mul_mat_nx_glu(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_MUL_MAT_NX || prev.outputs.size() != 2 || prev.inputs.size() != 3) {
        return false;
    }
    if (!htp_is_fuseable_mmnx_glu_op(cur.opcode) || cur.inputs.size() != 2 || cur.outputs.size() != 1 || !cur.src1()) {
        return false;
    }
    if (prev.fused.size() + 1 + cur.fused.size() > HTP_OP_MAX_FUSED) {
        return false;
    }

    const auto * orig_kparams = prev.kparams<htp_mm_kernel_params>();
    if (!orig_kparams->n_hmx || !htp_mm_kernel_is_hmx_2d(orig_kparams->kernel_type)) {
        return false;
    }

    const ggml_tensor * out0 = prev.outputs[0];
    const ggml_tensor * out1 = prev.outputs[1];
    if (!ggml_hexagon_tensor_is_fuseable(out0) || !ggml_hexagon_tensor_is_fuseable(out1) ||
        (out0->flags & GGML_TENSOR_FLAG_OUTPUT) || (out1->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    if (ggml_prec_get_acc(out0) == GGML_PREC_F32 || ggml_prec_get_acc(out1) == GGML_PREC_F32) {
        return false;
    }

    const ggml_tensor * w_gate   = nullptr;
    const ggml_tensor * w_up     = nullptr;
    const ggml_tensor * out_gate = nullptr;
    const ggml_tensor * out_up   = nullptr;
    if (htp_same_tensor_data(cur.src0(), out0) && htp_same_tensor_data(cur.src1(), out1)) {
        w_gate   = prev.inputs[0];
        w_up     = prev.inputs[1];
        out_gate = out0;
        out_up   = out1;
    } else if (htp_same_tensor_data(cur.src0(), out1) && htp_same_tensor_data(cur.src1(), out0)) {
        w_gate   = prev.inputs[1];
        w_up     = prev.inputs[0];
        out_gate = out1;
        out_up   = out0;
    } else {
        return false;
    }

    const ggml_tensor * act = prev.inputs[2];
    const ggml_tensor * dst = cur.dst();
    if (w_gate->ne[1] != w_up->ne[1] || w_gate->ne[0] != w_up->ne[0] ||
        !ggml_is_contiguous(out_gate) || !ggml_is_contiguous(out_up) || !ggml_is_contiguous(dst) ||
        out_gate->type != cur.src0()->type || out_up->type != cur.src1()->type ||
        !ggml_are_same_shape(out_gate, cur.src0()) || !ggml_are_same_stride(out_gate, cur.src0()) ||
        !ggml_are_same_shape(out_up,   cur.src1()) || !ggml_are_same_stride(out_up,   cur.src1()) ||
        !ggml_are_same_shape(out_gate, dst)) {
        return false;
    }
    if (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (htp_tensors_overlap(dst, w_gate) || htp_tensors_overlap(dst, w_up) || htp_tensors_overlap(dst, act)) {
        return false;
    }

    const bool collapse = htp_matmul_can_collapse(w_gate, act, dst) && htp_matmul_can_collapse(w_up, act, dst);
    if (!ggml_is_matrix(act) && !collapse) {
        return false;
    }

    const ggml_tensor x_collapsed = collapse ? htp_tensor_collapse_rows(act) : *act;
    const ggml_tensor d_collapsed = collapse ? htp_tensor_collapse_rows(dst) : *dst;

    struct htp_mm_kernel_params kparams;
    if (!htp_kparams_solve_mmnx_glu(sess, w_gate, w_up, &x_collapsed, &d_collapsed, (uint8_t) cur.opcode, &kparams)) {
        return false;
    }

    prev.opcode  = HTP_OP_MUL_MAT_NX_GLU;
    prev.inputs  = { w_gate, w_up, act };
    prev.outputs = { dst };
    prev.fused.push_back(cur.node);
    for (const ggml_tensor * f : cur.fused) {
        prev.fused.push_back(f);
    }
    prev.set_kparams(kparams);
    return true;
}

static bool fuse_mul_mat_id_nx(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_MUL_MAT_ID && prev.opcode != HTP_OP_MUL_MAT_ID_NX) return false;

    const uint32_t curr_n = (uint32_t) prev.outputs.size();
    if (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 2 >= HTP_OP_MAX_INPUTS) return false;

    const ggml_tensor * w0     = prev.inputs[0];
    const ggml_tensor * x      = prev.inputs[curr_n];
    const ggml_tensor * ids    = prev.inputs[curr_n + 1];
    const ggml_tensor * w_in   = cur.src0();
    const ggml_tensor * x_in   = cur.src1();
    const ggml_tensor * ids_in = cur.node->src[2];
    const ggml_tensor * d_in   = cur.dst();

    auto * orig_kparams = prev.kparams<htp_mm_kernel_params>();
    const auto * cur_kparams = cur.kparams<htp_mm_kernel_params>();

    if (x_in != x || ids_in != ids || w_in->type != w0->type ||
        w_in->ne[0] != w0->ne[0] || w_in->ne[2] != w0->ne[2] ||
        ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0) ||
        orig_kparams->n_hmx != cur_kparams->n_hmx) {
        return false;
    }

    struct htp_mm_kernel_params kparams;
    if (d_in->nb[1] == prev.node->nb[1]) {
        kparams = *orig_kparams;
        kparams.n_weights = curr_n + 1;
    } else if (!htp_kparams_solve_mmidnx(sess, w0, x, d_in, curr_n + 1, &kparams)) {
        return false;
    }
    if (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) return false;

    prev.opcode = HTP_OP_MUL_MAT_ID_NX;
    prev.inputs.insert(prev.inputs.begin() + curr_n, w_in);
    prev.outputs.push_back(d_in);
    prev.fused.push_back(cur.node);
    prev.set_kparams(kparams);
    return true;
}

static bool fuse_gdn_cpy(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    GGML_UNUSED(sess);
    if (prev.opcode != HTP_OP_GATED_DELTA_NET || prev.outputs.size() != 1) return false;

    const ggml_tensor * gdn_out  = prev.dst();
    const ggml_tensor * cpy_node = cur.node;
    const ggml_tensor * cpy_src  = cur.src0();
    const ggml_tensor * cpy_dst  = cur.dst();

    if (gdn_out->type != GGML_TYPE_F32 || cpy_src->type != GGML_TYPE_F32 || cpy_dst->type != GGML_TYPE_F32) return false;
    if ((gdn_out->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy_node->flags & GGML_TENSOR_FLAG_OUTPUT)) return false;

    const ggml_tensor * v = prev.node->src[2];
    if (!v) return false;

    const int64_t S_v       = v->ne[0];
    const int64_t H         = v->ne[1];
    const int64_t n_tokens  = v->ne[2];
    const int64_t n_seqs    = v->ne[3];
    const int64_t K         = ggml_get_op_params_i32(prev.node, 0);
    const size_t  tail_off  = (size_t) S_v * H * n_tokens * n_seqs * sizeof(float);
    const int64_t D         = S_v * S_v * H;
    const int64_t n_written = std::min<int64_t>(n_tokens, K);

    if (cpy_src->op != GGML_OP_VIEW || !htp_same_tensor_data(cpy_src->view_src, gdn_out) ||
        cpy_src->view_offs != tail_off || !ggml_is_contiguous(cpy_src)) {
        return false;
    }

    if (cpy_dst->ne[0] != D || cpy_dst->ne[1] != n_seqs || cpy_dst->ne[2] != n_written || !ggml_is_contiguous(cpy_dst)) {
        return false;
    }

    prev.outputs.push_back(cpy_dst);
    prev.fused.push_back(cur.node);
    return true;
}

static bool fuse_ssm_conv_silu(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_SSM_CONV) return false;

    auto * kparams = prev.kparams<htp_ssm_conv_kernel_params>();
    if (kparams->silu) return false;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ false)) return false;

    const ggml_tensor * conv_dst = prev.dst();
    const ggml_tensor * dst      = cur.dst();
    if (!ggml_are_same_shape(conv_dst, cur.src0()) || !ggml_are_same_stride(conv_dst, cur.src0())) return false;
    if (!ggml_are_same_shape(conv_dst, dst)) return false;
    if (htp_tensors_overlap(dst, prev.src0()) || htp_tensors_overlap(dst, prev.src1())) return false;

    htp_opnode cand = prev;
    cand.outputs[0] = dst;
    cand.fused.push_back(cur.node);
    cand.kparams<htp_ssm_conv_kernel_params>()->silu = 1;
    if (!htp_kparams_solve(sess, cand)) return false;

    prev = std::move(cand);
    return true;
}

static bool fuse_repeat_get_rows(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_REPEAT) return false;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ false)) return false;

    const ggml_tensor * repeat_out  = prev.dst();
    const ggml_tensor * repeat_src0 = prev.src0();
    const ggml_tensor * gr_src1     = cur.src1();
    const ggml_tensor * gr_dst      = cur.dst();

    if (repeat_src0->ne[0] != repeat_out->ne[0] || repeat_src0->ne[1] != repeat_out->ne[1]) return false;
    if (repeat_src0->type != repeat_out->type) return false;

    struct htp_get_rows_kernel_params kparams;
    if (!htp_kparams_solve_repeat_get_rows(sess, repeat_src0, gr_src1, gr_dst, &kparams) || kparams.vtcm_size == 0) return false;

    prev.opcode  = HTP_OP_GET_ROWS;
    prev.inputs  = { repeat_src0, gr_src1 };
    prev.outputs = { gr_dst };
    prev.fused.push_back(cur.node);
    prev.set_kparams(kparams);
    return true;
}

static inline bool htp_is_scale_bias_unary(uint32_t opcode) {
    switch (opcode) {
        case HTP_OP_UNARY_EXP:
        case HTP_OP_UNARY_SIGMOID:
        case HTP_OP_UNARY_SILU:
        case HTP_OP_UNARY_GELU:
        case HTP_OP_UNARY_GELU_QUICK:
        case HTP_OP_UNARY_GELU_ERF:
        case HTP_OP_UNARY_SOFTPLUS:
        case HTP_OP_UNARY_TANH:
            return true;
        default:
            return false;
    }
}

static bool fuse_scale_unary(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_SCALE || !prev.fused.empty() || !htp_is_scale_bias_unary(cur.opcode)) return false;
    if (cur.kparams<htp_unary_kernel_params>()->has_scale_bias) return false;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ false)) return false;

    const ggml_tensor * src0 = prev.src0();
    const ggml_tensor * dst  = cur.dst();
    if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
        dst->type != src0->type || !htp_same_shape_and_contiguity(prev.dst(), dst)) {
        return false;
    }
    if (htp_inputs_overlap_unsafe(prev, dst)) return false;

    float in_scale = 0.0f;
    float in_bias  = 0.0f;
    memcpy(&in_scale, (const float *) prev.params + 0, sizeof(float));
    memcpy(&in_bias,  (const float *) prev.params + 1, sizeof(float));

    htp_opnode cand = prev;
    cand.opcode  = cur.opcode;
    cand.outputs = { dst };
    memcpy(cand.params, cur.params, sizeof(cand.params));

    auto * kp          = cand.kparams<htp_unary_kernel_params>();
    kp->has_scale_bias = 1;
    kp->in_scale       = in_scale;
    kp->in_bias        = in_bias;
    kp->out_scale      = 1.0f;
    kp->out_bias       = 0.0f;
    if (!htp_kparams_solve(sess, cand)) return false;

    prev = std::move(cand);
    prev.fused.push_back(cur.node);
    return true;
}

static bool fuse_unary_scale(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (!htp_is_scale_bias_unary(prev.opcode) || prev.fused.size() >= HTP_OP_MAX_FUSED) return false;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ false)) return false;

    const ggml_tensor * src0 = prev.src0();
    const ggml_tensor * dst  = cur.dst();
    if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
        dst->type != src0->type || !htp_same_shape_and_contiguity(prev.dst(), dst)) {
        return false;
    }
    if (htp_inputs_overlap_unsafe(prev, dst)) return false;

    float s_out = 0.0f;
    float b_out = 0.0f;
    memcpy(&s_out, (const float *) cur.params + 0, sizeof(float));
    memcpy(&b_out, (const float *) cur.params + 1, sizeof(float));

    const auto * prev_kp  = prev.kparams<htp_unary_kernel_params>();
    const float in_scale  = prev_kp->has_scale_bias ? prev_kp->in_scale  : 1.0f;
    const float in_bias   = prev_kp->has_scale_bias ? prev_kp->in_bias   : 0.0f;
    const float out_scale = (prev_kp->has_scale_bias ? prev_kp->out_scale : 1.0f) * s_out;
    const float out_bias  = (prev_kp->has_scale_bias ? prev_kp->out_bias  : 0.0f) * s_out + b_out;

    htp_opnode cand = prev;
    cand.outputs = { dst };

    auto * kp          = cand.kparams<htp_unary_kernel_params>();
    kp->has_scale_bias = 1;
    kp->in_scale       = in_scale;
    kp->in_bias        = in_bias;
    kp->out_scale      = out_scale;
    kp->out_bias       = out_bias;
    if (!htp_kparams_solve(sess, cand)) return false;

    prev = std::move(cand);
    prev.fused.push_back(cur.node);
    return true;
}

static bool fuse_unary_mul_glu(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    htp_op_code glu_op = HTP_OP_INVALID;
    switch (prev.opcode) {
        case HTP_OP_UNARY_SILU:       glu_op = HTP_OP_GLU_SWIGLU;      break;
        case HTP_OP_UNARY_GELU:       glu_op = HTP_OP_GLU_GEGLU;       break;
        case HTP_OP_UNARY_GELU_QUICK: glu_op = HTP_OP_GLU_GEGLU_QUICK; break;
        case HTP_OP_UNARY_GELU_ERF:   glu_op = HTP_OP_GLU_GEGLU_ERF;   break;
        case HTP_OP_UNARY_RELU:       glu_op = HTP_OP_GLU_REGLU;       break;
        case HTP_OP_UNARY_SIGMOID:    glu_op = HTP_OP_GLU_SIGMOID;     break;
        default: return false;
    }
    if (prev.kparams<htp_unary_kernel_params>()->has_scale_bias) return false;

    const ggml_tensor * up = nullptr;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ true, &up)) return false;

    const ggml_tensor * gate = prev.src0();
    const ggml_tensor * dst  = cur.dst();
    if (htp_inputs_overlap_unsafe(prev, dst, up)) return false;

    htp_opnode cand = prev;
    cand.opcode  = glu_op;
    cand.inputs  = { gate, up };
    cand.outputs = { dst };
    if (!htp_kparams_solve(sess, cand)) return false;

    prev = std::move(cand);
    prev.fused.push_back(cur.node);
    memset(prev.params, 0, sizeof(prev.params));
    return true;
}

static bool fuse_mul_mul_scalar(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_MUL || prev.inputs.size() != 2) return false;

    const auto * k0 = prev.kparams<htp_binary_kernel_params>();
    const auto * k1 = cur.kparams<htp_binary_kernel_params>();
    if (k0->kernel_type != HTP_BINARY_KERNEL_SCALAR || k1->kernel_type != HTP_BINARY_KERNEL_SCALAR) return false;
    if (k0->src1_strided || k1->src1_strided) return false;

    const ggml_tensor * src2 = nullptr;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ false, &src2)) return false;

    const ggml_tensor * src1 = prev.src1();
    const ggml_tensor * dst  = cur.dst();
    if (prev.src0()->type != src1->type || prev.dst()->type != src1->type || src2->type != src1->type || dst->type != src1->type) return false;
    if (!ggml_are_same_shape(src1, src2) || !htp_same_shape_and_contiguity(prev.dst(), dst)) return false;
    if (htp_inputs_overlap_unsafe(prev, dst, src2)) return false;

    const size_t vtcm_size = htp_session_vtcm_size(sess);
    struct htp_binary_kernel_params kparams = *k0;
    kparams.src2_size       = kparams.src1_size;
    kparams.rows_per_buffer = 0;
    if (kparams.src1_size > 65536 || kparams.src1_size + kparams.src2_size > vtcm_size) return false;

    struct htp_binary_vtcm_layout L;
    htp_binary_vtcm_layout_build(&L, &kparams, vtcm_size);
    if (L.rows_per_buffer == 0 || L.total_bytes > vtcm_size) return false;

    kparams.rows_per_buffer = L.rows_per_buffer;
    kparams.vtcm_size       = L.total_bytes;

    prev.inputs.push_back(src2);
    prev.outputs[0] = dst;
    prev.fused.push_back(cur.node);
    prev.set_kparams(kparams);
    return true;
}

static bool fuse_binary_nx(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != cur.opcode) return false;

    const uint32_t curr_n = (uint32_t) prev.inputs.size();
    if (curr_n < 2 || curr_n >= HTP_OP_MAX_INPUTS) return false;

    const auto * k0 = prev.kparams<htp_binary_kernel_params>();
    const auto * k1 = cur.kparams<htp_binary_kernel_params>();
    const bool k0_ok = (k0->kernel_type == HTP_BINARY_KERNEL_SAME_SHAPE) ||
                       (k0->kernel_type == HTP_BINARY_KERNEL_CHUNKED && !k0->is_scalar);
    const bool k1_ok = (k1->kernel_type == HTP_BINARY_KERNEL_SAME_SHAPE) ||
                       (k1->kernel_type == HTP_BINARY_KERNEL_CHUNKED && !k1->is_scalar);
    if (!k0_ok || !k1_ok) return false;

    const bool is_comm = (cur.opcode == HTP_OP_ADD || cur.opcode == HTP_OP_MUL);
    const ggml_tensor * new_src = nullptr;
    if (!htp_match_consumer(prev, cur, is_comm, &new_src)) return false;

    const ggml_tensor * src0 = prev.inputs[0];
    const ggml_tensor * dst  = cur.dst();
    if (prev.dst()->type != src0->type || new_src->type != src0->type || dst->type != src0->type) return false;
    for (const auto * in : prev.inputs) {
        if (in->type != src0->type) return false;
    }
    if (!ggml_are_same_shape(src0, prev.inputs[1]) ||
        !ggml_are_same_shape(src0, new_src) ||
        !htp_same_shape_and_contiguity(prev.dst(), dst)) {
        return false;
    }
    if (htp_inputs_overlap_unsafe(prev, dst, new_src)) return false;

    const size_t vtcm_size = htp_session_vtcm_size(sess);
    struct htp_binary_kernel_params kparams = *k0;
    struct htp_binary_vtcm_layout L;
    memset(&L, 0, sizeof(L));

    const uint32_t n_srcs = curr_n + 1;
    if (k0->kernel_type == HTP_BINARY_KERNEL_SAME_SHAPE || k1->kernel_type == HTP_BINARY_KERNEL_SAME_SHAPE) {
        if (k0->kernel_type != HTP_BINARY_KERNEL_SAME_SHAPE) {
            kparams = *k1;
        }
        kparams.n_srcs          = (uint8_t) n_srcs;
        kparams.rows_per_buffer = 0;

        auto check_split = [&](const ggml_tensor * t) {
            if (src0->ne[2] > 1 && t->nb[2] != t->ne[1] * t->nb[1]) kparams.split_at_ne01 = 1;
            if (src0->ne[3] > 1 && t->nb[3] != t->ne[2] * t->nb[2]) kparams.split_at_ne02 = 1;
        };
        for (const auto * in : prev.inputs) check_split(in);
        check_split(new_src);
        check_split(dst);

        htp_binary_vtcm_layout_build(&L, &kparams, vtcm_size);
    }

    if (L.rows_per_buffer == 0 || L.total_bytes == 0 || L.total_bytes > vtcm_size) {
        bool all_contig = ggml_is_contiguous(new_src) && ggml_is_contiguous(dst);
        for (const auto * in : prev.inputs) {
            all_contig = all_contig && ggml_is_contiguous(in);
        }
        if (!all_contig) return false;

        const uint32_t total_elems = (uint32_t) ggml_nelements(src0);
        const uint32_t elem_size   = (uint32_t) ggml_type_size(src0->type);
        const uint32_t max_hvx     = htp_session_n_hvx(sess);
        if (!htp_binary_solve_chunked(&L, &kparams, total_elems, elem_size, elem_size, elem_size, max_hvx, n_srcs, false, vtcm_size)) {
            return false;
        }
    }

    kparams.rows_per_buffer = L.rows_per_buffer;
    kparams.vtcm_size       = (uint32_t) L.total_bytes;

    prev.inputs.push_back(new_src);
    prev.outputs[0] = dst;
    prev.fused.push_back(cur.node);
    prev.set_kparams(kparams);
    return true;
}

bool htp_fusion_can_stack(const struct ggml_tensor * a) {
    if (!a || !a->src[0] || !a->src[1]) return false;
    switch (a->op) {
        case GGML_OP_MUL_MAT:
            if (!htp_fusion_enabled(HTP_FUSE_MUL_MAT_NX)) return false;
            if (htp_is_hadamard(a)) return false;
            if (a->src[0]->ne[2] != 1 || a->src[0]->ne[3] != 1) return false;
            return ggml_is_quantized(a->src[0]->type) ||
                   (a->src[0]->type == GGML_TYPE_F16 && a->src[0]->ne[0] % 32 == 0);
        case GGML_OP_MUL_MAT_ID:
            if (!htp_fusion_enabled(HTP_FUSE_MUL_MAT_ID_NX)) return false;
            return a->src[2] != nullptr && ggml_is_quantized(a->src[0]->type);
        default:
            return false;
    }
}

bool htp_fusion_can_stack(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    if (!htp_fusion_can_stack(b) || a->op != b->op || a->src[1] != b->src[1]) {
        return false;
    }
    if (a->op == GGML_OP_MUL_MAT_ID && a->src[2] != b->src[2]) {
        return false;
    }
    const ggml_tensor * w0 = a->src[0];
    const ggml_tensor * w1 = b->src[0];
    return w0->type == w1->type && w0->ne[0] == w1->ne[0] && w0->ne[2] == w1->ne[2];
}

static bool htp_lora_is_float_weight_ok(const ggml_tensor * t) {
    if (!t || (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_BF16)) {
        return false;
    }
    if (ggml_hexagon_tensor_is_tiled(t) || ggml_is_permuted(t) || !ggml_is_contiguous_rows(t)) {
        return false;
    }
    if (t->ne[2] != 1 || t->ne[3] != 1 || t->nb[0] != ggml_type_size(t->type)) {
        return false;
    }
    return true;
}

static int htp_lora_match_chain_graph(
    const struct ggml_cgraph * graph,
    int                        idx,
    const ggml_tensor *        expected_x,
    const ggml_tensor *        expected_prev,
    const ggml_tensor **       prev_out,
    const ggml_tensor **       add_out,
    int *                      mm_a_idx_out,
    int *                      mm_b_idx_out
) {
    const int n = graph->n_nodes;
    if (idx + 2 >= n) {
        return 0;
    }
    const ggml_tensor * mm_a = graph->nodes[idx];
    const ggml_tensor * mm_b = graph->nodes[idx + 1];
    if (!mm_a || !mm_b || mm_a->op != GGML_OP_MUL_MAT || mm_b->op != GGML_OP_MUL_MAT) {
        return 0;
    }
    if (htp_is_hadamard(mm_a) || htp_is_hadamard(mm_b) || mm_b->src[1] != mm_a) {
        return 0;
    }
    const ggml_tensor * a = mm_a->src[0];
    const ggml_tensor * x = mm_a->src[1];
    const ggml_tensor * b = mm_b->src[0];
    if (!a || !b || !x || (expected_x && x != expected_x)) {
        return 0;
    }
    if (!htp_lora_is_float_weight_ok(a) || !htp_lora_is_float_weight_ok(b)) {
        return 0;
    }
    if (a->ne[0] != x->ne[0] || a->ne[1] < 1 || a->ne[1] > 256 || b->ne[0] != a->ne[1]) {
        return 0;
    }
    if (mm_a->view_src || mm_b->view_src || (mm_a->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        (mm_b->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(graph, idx) != 1 ||
        ggml_node_get_use_count(graph, idx + 1) != 1) {
        return 0;
    }

    int                 chain_len = 0;
    const ggml_tensor * ba_scaled = mm_b;
    int                 add_idx   = idx + 2;

    if (graph->nodes[idx + 2]->op == GGML_OP_SCALE) {
        if (idx + 3 >= n) {
            return 0;
        }
        const ggml_tensor * sc = graph->nodes[idx + 2];
        if (sc->src[0] != mm_b || sc->view_src || (sc->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            ggml_node_get_use_count(graph, idx + 2) != 1) {
            return 0;
        }
        float bias = 0.0f;
        memcpy(&bias, (const float *) sc->op_params + 1, sizeof(float));
        if (bias != 0.0f) {
            return 0;
        }
        ba_scaled = sc;
        add_idx   = idx + 3;
        chain_len = 4;
    } else {
        chain_len = 3;
    }

    const ggml_tensor * add = graph->nodes[add_idx];
    if (!add || add->op != GGML_OP_ADD || add->view_src) {
        return 0;
    }
    const ggml_tensor * other = nullptr;
    if (add->src[0] == ba_scaled) {
        other = add->src[1];
    } else if (add->src[1] == ba_scaled) {
        other = add->src[0];
    } else {
        return 0;
    }
    if (!other || (expected_prev && other != expected_prev) || b->ne[1] != other->ne[0]) {
        return 0;
    }
    if (prev_out)     *prev_out     = other;
    if (add_out)      *add_out      = add;
    if (mm_a_idx_out) *mm_a_idx_out = idx;
    if (mm_b_idx_out) *mm_b_idx_out = idx + 1;
    return chain_len;
}

int htp_fusion_max(const struct ggml_cgraph * graph, int idx) {
    const int n = graph->n_nodes;
    const ggml_op op0 = graph->nodes[idx]->op;
    if (htp_fusion_enabled(HTP_FUSE_LORA) && op0 == GGML_OP_MUL_MAT && htp_fusion_can_stack(graph->nodes[idx])) {
        const ggml_tensor * base_mm  = graph->nodes[idx];
        const ggml_tensor * cur_prev = base_mm;
        const ggml_tensor * x        = base_mm->src[1];
        int                 f        = 1;
        int                 pos      = idx + 1;
        int                 prev_idx = idx;
        while (pos < n && !(cur_prev->flags & GGML_TENSOR_FLAG_OUTPUT) &&
               ggml_node_get_use_count(graph, prev_idx) == 1) {
            const ggml_tensor * add_out = nullptr;
            const int cl = htp_lora_match_chain_graph(graph, pos, x, cur_prev, nullptr, &add_out, nullptr, nullptr);
            if (cl == 0 || f + cl > HTP_OP_MAX_FUSED) {
                break;
            }
            cur_prev = add_out;
            prev_idx = pos + cl - 1;
            pos += cl;
            f += cl;
        }
        if (f > 1) {
            return f;
        }
    }
    if (op0 != GGML_OP_ADD && op0 != GGML_OP_NORM && op0 != GGML_OP_RMS_NORM) {
        return 1;
    }

    enum ggml_op ops[HTP_OP_MAX_FUSED];
    ops[0] = op0;

    int f = idx + 1;
    while (f < n && f < idx + HTP_OP_MAX_FUSED) {
        const ggml_op op = graph->nodes[f]->op;
        if (op != GGML_OP_ADD && op != GGML_OP_MUL && op != GGML_OP_SCALE &&
            op != GGML_OP_NORM && op != GGML_OP_RMS_NORM) {
            break;
        }
        ops[f - idx] = op;
        f++;
    }

    f -= idx;
    for (; f > 1; f--) {
        if (ggml_can_fuse(graph, idx, ops, f)) {
            break;
        }
    }
    return f;
}

// Number of matmuls taking node i as their activation when those are all its uses, else 0.
static uint32_t htp_count_mm_uses(const struct ggml_cgraph * graph, int i) {
    const ggml_tensor * t = graph->nodes[i];
    if (t->view_src || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    uint32_t n = 0;
    for (int j = i + 1; j < graph->n_nodes; j++) {
        const ggml_tensor * c = graph->nodes[j];
        if (c->view_src == t) {
            return 0;
        }
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (c->src[s] != t) continue;
            if (c->op != GGML_OP_MUL_MAT || s != 1) {
                return 0;
            }
            n++;
        }
    }
    return n == (uint32_t) ggml_node_get_use_count(graph, i) ? n : 0;
}

void htp_fusion_tag_graph(struct ggml_cgraph * graph) {
    for (int i = 0; i < graph->n_nodes; i++) {
        ggml_tensor * node = graph->nodes[i];
        ggml_hexagon_tensor_clear_flags(node, HTP_TENSOR_FUSEABLE);

        const ggml_op op = node->op;
        if ((op == GGML_OP_RMS_NORM || op == GGML_OP_SCALE || op == GGML_OP_REPEAT ||
             op == GGML_OP_UNARY    || op == GGML_OP_ADD   || op == GGML_OP_SUB    ||
             op == GGML_OP_MUL      || op == GGML_OP_DIV   || op == GGML_OP_MUL_MAT_ID ||
             op == GGML_OP_SSM_CONV ||
             (op == GGML_OP_MUL_MAT && !htp_is_hadamard(node))) &&
            ggml_node_has_n_uses(graph, i, 1)) {
            ggml_hexagon_tensor_set_flags(node, HTP_TENSOR_FUSEABLE);
        }

        const bool norm_mul = op == GGML_OP_MUL && i > 0 && graph->nodes[i - 1]->op == GGML_OP_RMS_NORM;
        ggml_hexagon_tensor_set_mm_uses(node, norm_mul ? htp_count_mm_uses(graph, i) : 0);
    }
}

void htp_fusion_collect_lora_weights(const struct ggml_cgraph * graph,
                                     std::unordered_set<const struct ggml_tensor *> & lora_weights) {
    lora_weights.clear();
    if (!graph || !opt_opfusion || !htp_fusion_enabled(HTP_FUSE_LORA)) {
        return;
    }

    std::vector<uint8_t> matched_mm_nodes((size_t) graph->n_nodes, 0);
    for (int i = 0; i < graph->n_nodes; ++i) {
        int mm_a_idx = -1, mm_b_idx = -1;
        const int cl = htp_lora_match_chain_graph(graph, i, nullptr, nullptr, nullptr, nullptr, &mm_a_idx, &mm_b_idx);
        if (cl > 0) {
            lora_weights.insert(graph->nodes[mm_a_idx]->src[0]);
            lora_weights.insert(graph->nodes[mm_b_idx]->src[0]);
            matched_mm_nodes[mm_a_idx] = 1;
            matched_mm_nodes[mm_b_idx] = 1;
            i += cl - 1;
        }
    }
    if (lora_weights.empty()) {
        return;
    }

    // Never skip repacking any weight also consumed by a non-LoRA MUL_MAT / MUL_MAT_ID
    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * n = graph->nodes[i];
        if (!n || (n->op != GGML_OP_MUL_MAT && n->op != GGML_OP_MUL_MAT_ID)) {
            continue;
        }
        if (!matched_mm_nodes[i] && n->src[0]) {
            lora_weights.erase(n->src[0]);
        }
    }
}

static inline void htp_lora_push_fused_safe(htp_opnode & dst, const ggml_tensor * t) {
    if (t && dst.fused.size() < HTP_OP_MAX_FUSED) {
        dst.fused.push_back(t);
    }
}

static int htp_lora_match_chain_nodes(
    const std::vector<struct htp_opnode> & nodes,
    size_t                                 idx,
    const ggml_tensor *                    expected_x,
    const ggml_tensor *                    expected_base,
    struct htp_lora_adapter_desc *         ad_out,
    const ggml_tensor **                   base_out,
    const ggml_tensor **                   add_out
) {
    if (idx + 2 >= nodes.size()) {
        return 0;
    }
    const htp_opnode & n_a = nodes[idx];
    const htp_opnode & n_b = nodes[idx + 1];
    if (n_a.opcode != HTP_OP_MUL_MAT || n_b.opcode != HTP_OP_MUL_MAT || !n_a.node || !n_b.node) {
        return 0;
    }
    if (!n_a.fused.empty() || !n_b.fused.empty() || htp_is_hadamard(n_a.node) || htp_is_hadamard(n_b.node)) {
        return 0;
    }
    const ggml_tensor * a = n_a.node->src[0];
    const ggml_tensor * x = n_a.node->src[1];
    const ggml_tensor * b = n_b.node->src[0];
    if (!a || !b || !x || n_b.node->src[1] != n_a.node || (expected_x && x != expected_x)) {
        return 0;
    }
    if (!htp_lora_is_float_weight_ok(a) || !htp_lora_is_float_weight_ok(b) ||
        !ggml_hexagon_tensor_is_fuseable(n_a.node) || (n_a.node->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        n_a.node->view_src || !ggml_hexagon_tensor_is_fuseable(n_b.node) ||
        (n_b.node->flags & GGML_TENSOR_FLAG_OUTPUT) || n_b.node->view_src) {
        return 0;
    }

    int                 cl        = 3;
    float               scale     = 1.0f;
    const ggml_tensor * ba_scaled = n_b.node;
    size_t              add_idx   = idx + 2;

    if (nodes[idx + 2].opcode == HTP_OP_SCALE) {
        if (idx + 3 >= nodes.size()) {
            return 0;
        }
        const htp_opnode & n_s = nodes[idx + 2];
        if (!n_s.node || !n_s.fused.empty() || n_s.node->src[0] != n_b.node || n_s.node->view_src ||
            !ggml_hexagon_tensor_is_fuseable(n_s.node) || (n_s.node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        float bias = 0.0f;
        memcpy(&scale, (const float *) n_s.params + 0, sizeof(float));
        memcpy(&bias,  (const float *) n_s.params + 1, sizeof(float));
        if (bias != 0.0f) {
            return 0;
        }
        ba_scaled = n_s.node;
        add_idx   = idx + 3;
        cl        = 4;
    }

    const htp_opnode & n_add = nodes[add_idx];
    if (n_add.opcode != HTP_OP_ADD || !n_add.node || !n_add.fused.empty() || n_add.node->view_src) {
        return 0;
    }
    const ggml_tensor * other = nullptr;
    if (n_add.node->src[0] == ba_scaled) {
        other = n_add.node->src[1];
    } else if (n_add.node->src[1] == ba_scaled) {
        other = n_add.node->src[0];
    } else {
        return 0;
    }
    if (!other || (expected_base && other != expected_base)) {
        return 0;
    }
    if (ad_out) {
        ad_out->a      = a;
        ad_out->b      = b;
        ad_out->scale  = scale;
        ad_out->target = 0;
    }
    if (base_out) *base_out = other;
    if (add_out)  *add_out  = n_add.node;
    return cl;
}

static bool htp_lora_rows_collapsible(const ggml_tensor * t) {
    size_t s = 0;
    return t && htp_rows_stride(t->ne, t->nb, &s);
}

void htp_fusion_lora(const struct ggml_hexagon_session * sess, const struct ggml_cgraph * graph,
                     std::vector<struct htp_opnode> & nodes) {
    GGML_UNUSED(graph);
    if (!opt_opfusion || !htp_fusion_enabled(HTP_FUSE_LORA) || htp_session_mdev_count(sess) > 1 || opt_lora_unfused) {
        return;
    }

    // Pass 1: Collapse LoRA chains into HTP_OP_MUL_MAT_LORA (with base MM) or HTP_OP_LORA_ADD (fallback)
    size_t n_matmul_lora = 0;
    {
        size_t out_n = 0;
        for (size_t i = 0; i < nodes.size(); ++i) {
            struct htp_lora_adapter_desc adapters[HTP_LORA_MAX_ADAPTERS] = {};
            const ggml_tensor *          base_t                          = nullptr;
            const ggml_tensor *          cur_out                         = nullptr;
            const int cl0 = htp_lora_match_chain_nodes(nodes, i, nullptr, nullptr, &adapters[0], &base_t, &cur_out);
            if (cl0 == 0) {
                if (out_n != i) {
                    nodes[out_n] = std::move(nodes[i]);
                }
                ++out_n;
                continue;
            }

            const ggml_tensor * x_orig  = nodes[i].node->src[1];
            int                 n_loras = 1;
            size_t              cursor  = i + (size_t) cl0;

            while (n_loras < HTP_LORA_MAX_ADAPTERS && cursor < nodes.size()) {
                if (!ggml_hexagon_tensor_is_fuseable(cur_out) || (cur_out->flags & GGML_TENSOR_FLAG_OUTPUT) ||
                    cur_out->view_src) {
                    break;
                }
                const ggml_tensor * next_out = nullptr;
                const int cl = htp_lora_match_chain_nodes(nodes, cursor, x_orig, cur_out, &adapters[n_loras], nullptr,
                                                          &next_out);
                if (cl == 0) {
                    break;
                }
                cur_out = next_out;
                cursor += (size_t) cl;
                n_loras++;
            }

            const size_t        last_idx  = cursor - 1;
            const ggml_tensor * final_out = cur_out;

            // Try folding directly into immediately preceding base MUL_MAT
            bool folded_into_base = false;
            if (out_n > 0 && nodes[out_n - 1].opcode == HTP_OP_MUL_MAT && nodes[out_n - 1].fused.empty() &&
                nodes[out_n - 1].node == base_t && !base_t->view_src &&
                ggml_hexagon_tensor_is_fuseable(base_t) && !(base_t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                const ggml_tensor * base_w = base_t->src[0];
                if (base_t->src[1] == x_orig && base_w &&
                    !htp_tensors_overlap(final_out, base_w) && !htp_tensors_overlap(final_out, x_orig)) {
                    bool overlap = false;
                    for (int l = 0; l < n_loras; ++l) {
                        if (htp_tensors_overlap(final_out, adapters[l].a) ||
                            htp_tensors_overlap(final_out, adapters[l].b)) {
                            overlap = true;
                            break;
                        }
                    }
                    const bool do_collapse = (x_orig->ne[2] > 1 || x_orig->ne[3] > 1);
                    if (!overlap && (!do_collapse || htp_matmul_can_collapse(base_w, x_orig, final_out))) {
                        htp_opnode merged(HTP_OP_MUL_MAT_LORA, base_t);
                        const ggml_tensor * x_eff =
                            do_collapse ? merged.add_dummy(htp_tensor_collapse_rows(x_orig)) : x_orig;
                        const ggml_tensor * o_eff =
                            do_collapse ? merged.add_dummy(htp_tensor_collapse_rows(final_out)) : final_out;
                        struct htp_lora_kernel_params kp;
                        if (htp_kparams_solve_matmul_lora(sess, &base_w, 1, x_eff, nullptr, false, 0.0f, 0.0f, 0,
                                                          adapters, n_loras, &kp, nullptr)) {
                            merged.inputs = { base_w, x_eff };
                            for (int l = 0; l < n_loras; ++l) {
                                merged.inputs.push_back(adapters[l].a);
                                merged.inputs.push_back(adapters[l].b);
                            }
                            merged.outputs = { o_eff };
                            merged.set_kparams(kp);
                            for (size_t k = i; k <= last_idx; ++k) {
                                htp_lora_push_fused_safe(merged, nodes[k].node);
                            }
                            nodes[out_n - 1] = std::move(merged);
                            ++n_matmul_lora;
                            i                = last_idx;
                            folded_into_base = true;
                        }
                    }
                }
            }
            if (folded_into_base) {
                continue;
            }

            // Fallback: collapse into standalone HTP_OP_LORA_ADD
            bool collapsed_add = false;
            const bool do_collapse = (x_orig->ne[2] > 1 || x_orig->ne[3] > 1 ||
                                      base_t->ne[2] > 1 || base_t->ne[3] > 1);
            const bool can_col     = !do_collapse ||
                                     (htp_lora_rows_collapsible(x_orig) && htp_lora_rows_collapsible(base_t) &&
                                      htp_lora_rows_collapsible(final_out) &&
                                      x_orig->ne[1] == final_out->ne[1] && x_orig->ne[2] == final_out->ne[2] &&
                                      x_orig->ne[3] == final_out->ne[3] && ggml_are_same_shape(base_t, final_out));
            if (can_col) {
                htp_opnode merged(HTP_OP_LORA_ADD, nodes[last_idx].node);
                const ggml_tensor * b_eff = do_collapse ? merged.add_dummy(htp_tensor_collapse_rows(base_t))    : base_t;
                const ggml_tensor * x_eff = do_collapse ? merged.add_dummy(htp_tensor_collapse_rows(x_orig))    : x_orig;
                const ggml_tensor * o_eff = do_collapse ? merged.add_dummy(htp_tensor_collapse_rows(final_out)) : final_out;
                struct htp_lora_kernel_params kp;
                if (htp_kparams_solve_lora_add(sess, b_eff, o_eff, x_eff, adapters, n_loras, &kp)) {
                    merged.inputs = { b_eff, x_eff };
                    for (int l = 0; l < n_loras; ++l) {
                        merged.inputs.push_back(adapters[l].a);
                        merged.inputs.push_back(adapters[l].b);
                    }
                    merged.outputs = { o_eff };
                    merged.set_kparams(kp);
                    for (size_t k = i; k < last_idx; ++k) {
                        htp_lora_push_fused_safe(merged, nodes[k].node);
                    }
                    nodes[out_n++] = std::move(merged);
                    i              = last_idx;
                    collapsed_add  = true;
                }
            }
            if (!collapsed_add) {
                if (out_n != i) {
                    nodes[out_n] = std::move(nodes[i]);
                }
                ++out_n;
            }
        }
        nodes.resize(out_n);
    }

    if (n_matmul_lora == 0) {
        return;
    }

    // Pass 2: Stack adjacent HTP_OP_MUL_MAT_LORA / HTP_OP_MUL_MAT on same x into multi-weight MUL_MAT_LORA
    if (htp_fusion_enabled(HTP_FUSE_MUL_MAT_NX)) {
        auto collect_node = [](const htp_opnode & n, const ggml_tensor ** ws, uint32_t & nw,
                               htp_lora_adapter_desc * ads, uint32_t & nl, const ggml_tensor ** x_orig_out,
                               const ggml_tensor ** x_node_out, bool & has_lora) -> bool {
            if (n.opcode == HTP_OP_MUL_MAT_LORA) {
                const auto * kp = n.kparams<htp_lora_kernel_params>();
                if (kp->norm || kp->glu || !n.node) {
                    return false;
                }
                nw = kp->n_weights;
                nl = kp->n_loras;
                if (nw == 0 || nw > HTP_LORA_MAX_WEIGHTS || nl > HTP_LORA_MAX_ADAPTERS ||
                    n.inputs.size() != nw + 1 + 2 * nl || n.outputs.size() != nw) {
                    return false;
                }
                for (uint32_t p = 0; p < nw; ++p) {
                    ws[p] = n.inputs[p];
                }
                for (uint32_t l = 0; l < nl; ++l) {
                    ads[l] = { n.inputs[kp->lora_a_slot[l]], n.inputs[kp->lora_b_slot[l]],
                               kp->lora_scale[l], kp->lora_target[l] };
                }
                *x_orig_out = n.node->src[1];
                *x_node_out = n.inputs[nw];
                has_lora    = true;
                return true;
            }
            if (n.opcode == HTP_OP_MUL_MAT && n.fused.empty() && n.node && !htp_is_hadamard(n.node)) {
                nw          = 1;
                nl          = 0;
                ws[0]       = n.inputs[0];
                *x_orig_out = n.node->src[1];
                *x_node_out = n.inputs[1];
                return true;
            }
            return false;
        };

        size_t out_n = 0;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].opcode != HTP_OP_MUL_MAT_LORA &&
                (nodes[i].opcode != HTP_OP_MUL_MAT || !nodes[i].fused.empty() || !nodes[i].node ||
                 htp_is_hadamard(nodes[i].node))) {
                if (out_n != i) {
                    nodes[out_n] = std::move(nodes[i]);
                }
                ++out_n;
                continue;
            }

            const ggml_tensor *   w_acc[HTP_LORA_MAX_WEIGHTS]   = {};
            htp_lora_adapter_desc ad_acc[HTP_LORA_MAX_ADAPTERS] = {};
            const ggml_tensor *   out_acc[HTP_LORA_MAX_WEIGHTS] = {};
            uint32_t              nw_acc = 0, nl_acc = 0;
            const ggml_tensor *   x_orig0 = nullptr, * x_eff0 = nullptr;
            bool                  any_lora = false;
            if (!collect_node(nodes[i], w_acc, nw_acc, ad_acc, nl_acc, &x_orig0, &x_eff0, any_lora)) {
                if (out_n != i) {
                    nodes[out_n] = std::move(nodes[i]);
                }
                ++out_n;
                continue;
            }
            for (uint32_t p = 0; p < nw_acc; ++p) {
                out_acc[p] = nodes[i].outputs[p];
            }

            size_t consume_end = i + 1;
            struct htp_lora_kernel_params best_kp = {};
            bool   merged_any = false;

            while (consume_end < nodes.size()) {
                const ggml_tensor *   w_j[HTP_LORA_MAX_WEIGHTS]   = {};
                htp_lora_adapter_desc ad_j[HTP_LORA_MAX_ADAPTERS] = {};
                uint32_t              nw_j = 0, nl_j = 0;
                const ggml_tensor *   x_orig_j = nullptr, * x_eff_j = nullptr;
                bool                  j_lora   = false;
                if (!collect_node(nodes[consume_end], w_j, nw_j, ad_j, nl_j, &x_orig_j, &x_eff_j, j_lora)) {
                    break;
                }
                const bool comb_lora = any_lora || j_lora;
                if (!comb_lora || x_orig_j != x_orig0 || nw_acc + nw_j > HTP_LORA_MAX_WEIGHTS ||
                    nl_acc + nl_j == 0 || nl_acc + nl_j > HTP_LORA_MAX_ADAPTERS ||
                    nw_acc + nw_j + 1 + 2 * (nl_acc + nl_j) > HTP_OP_MAX_INPUTS) {
                    break;
                }
                if (w_j[0]->type != w_acc[0]->type || w_j[0]->ne[0] != w_acc[0]->ne[0] ||
                    ggml_hexagon_tensor_will_tile(w_j[0]) != ggml_hexagon_tensor_will_tile(w_acc[0])) {
                    break;
                }
                if (!ggml_are_same_shape(x_eff0, x_eff_j)) {
                    break;
                }

                bool overlap = false;
                for (uint32_t p = 0; p < nw_j && !overlap; ++p) {
                    const ggml_tensor * oj = nodes[consume_end].outputs[p];
                    if (htp_tensors_overlap(oj, x_eff0) || htp_tensors_overlap(oj, w_j[p])) {
                        overlap = true;
                    }
                    for (uint32_t q = 0; q < nw_acc; ++q) {
                        if (htp_tensors_overlap(out_acc[q], oj) || htp_tensors_overlap(out_acc[q], w_j[p]) ||
                            htp_tensors_overlap(oj, w_acc[q])) {
                            overlap = true;
                        }
                    }
                    for (uint32_t l = 0; l < nl_acc; ++l) {
                        if (htp_tensors_overlap(oj, ad_acc[l].a) || htp_tensors_overlap(oj, ad_acc[l].b)) {
                            overlap = true;
                        }
                    }
                }
                for (uint32_t l = 0; l < nl_j && !overlap; ++l) {
                    for (uint32_t q = 0; q < nw_acc; ++q) {
                        if (htp_tensors_overlap(out_acc[q], ad_j[l].a) || htp_tensors_overlap(out_acc[q], ad_j[l].b)) {
                            overlap = true;
                        }
                    }
                }
                if (overlap) {
                    break;
                }

                const ggml_tensor *   w_try[HTP_LORA_MAX_WEIGHTS];
                htp_lora_adapter_desc ad_try[HTP_LORA_MAX_ADAPTERS];
                for (uint32_t p = 0; p < nw_acc; ++p) w_try[p] = w_acc[p];
                for (uint32_t p = 0; p < nw_j;   ++p) w_try[nw_acc + p] = w_j[p];
                for (uint32_t l = 0; l < nl_acc; ++l) ad_try[l] = ad_acc[l];
                for (uint32_t l = 0; l < nl_j; ++l) {
                    ad_try[nl_acc + l]        = ad_j[l];
                    ad_try[nl_acc + l].target += (uint8_t) nw_acc;
                }

                struct htp_lora_kernel_params try_kp;
                if (!htp_kparams_solve_matmul_lora(sess, w_try, (int32_t) (nw_acc + nw_j), x_eff0, nullptr, false,
                                                   0.0f, 0.0f, 0, ad_try, (int32_t) (nl_acc + nl_j), &try_kp,
                                                   nullptr)) {
                    break;
                }

                for (uint32_t p = 0; p < nw_j; ++p) {
                    w_acc[nw_acc + p]   = w_j[p];
                    out_acc[nw_acc + p] = nodes[consume_end].outputs[p];
                }
                for (uint32_t l = 0; l < nl_j; ++l) {
                    ad_acc[nl_acc + l] = ad_try[nl_acc + l];
                }
                nw_acc    += nw_j;
                nl_acc    += nl_j;
                any_lora   = true;
                best_kp    = try_kp;
                merged_any = true;
                consume_end++;
            }

            if (merged_any) {
                htp_opnode merged(HTP_OP_MUL_MAT_LORA, nodes[i].node);
                const ggml_tensor * x_final = (x_orig0->ne[2] > 1 || x_orig0->ne[3] > 1)
                    ? merged.add_dummy(*x_eff0) : x_eff0;
                merged.inputs.clear();
                for (uint32_t p = 0; p < nw_acc; ++p) {
                    merged.inputs.push_back(w_acc[p]);
                }
                merged.inputs.push_back(x_final);
                for (uint32_t l = 0; l < nl_acc; ++l) {
                    merged.inputs.push_back(ad_acc[l].a);
                    merged.inputs.push_back(ad_acc[l].b);
                }
                merged.outputs.clear();
                for (uint32_t p = 0; p < nw_acc; ++p) {
                    const ggml_tensor * o = out_acc[p];
                    merged.outputs.push_back((o->ne[2] != nodes[i].node->ne[2] || x_orig0->ne[2] > 1 || x_orig0->ne[3] > 1)
                        ? merged.add_dummy(*o) : o);
                }
                for (size_t k = i; k < consume_end; ++k) {
                    if (k > i) {
                        htp_lora_push_fused_safe(merged, nodes[k].node);
                    }
                    for (const ggml_tensor * f : nodes[k].fused) {
                        htp_lora_push_fused_safe(merged, f);
                    }
                }
                merged.set_kparams(best_kp);
                nodes[out_n++] = std::move(merged);
                i              = consume_end - 1;
            } else {
                if (out_n != i) {
                    nodes[out_n] = std::move(nodes[i]);
                }
                ++out_n;
            }
        }
        nodes.resize(out_n);
    }

    // Pass 3: Absorb preceding RMS_NORM + MUL into HTP_OP_MUL_MAT_LORA
    if (htp_fusion_enabled(HTP_FUSE_LORA)) {
        size_t out_n = 0;
        for (size_t idx = 0; idx < nodes.size(); ++idx) {
            bool absorbed = false;
            do {
                if (out_n < 2) break;
                htp_opnode & cur = nodes[idx];
                if (cur.opcode != HTP_OP_MUL_MAT_LORA || !cur.node) break;
                const auto * kp = cur.kparams<htp_lora_kernel_params>();
                const uint32_t nw = kp->n_weights;
                const uint32_t nl = kp->n_loras;
                if (kp->norm || kp->glu || cur.inputs.size() != nw + 1 + 2 * nl) break;
                const htp_opnode & norm = nodes[out_n - 2];
                const htp_opnode & mul  = nodes[out_n - 1];
                if (norm.opcode != HTP_OP_RMS_NORM || mul.opcode != HTP_OP_MUL || !mul.fused.empty() ||
                    norm.inputs.size() != 1 || norm.outputs.size() != 1 || !norm.node || !mul.node) break;
                const ggml_tensor * norm_out = norm.node;
                if (!ggml_hexagon_tensor_is_fuseable(norm_out) || (norm_out->flags & GGML_TENSOR_FLAG_OUTPUT) ||
                    norm_out->view_src) break;
                const ggml_tensor * norm_w = nullptr;
                if (mul.node->src[0] == norm_out) {
                    norm_w = mul.node->src[1];
                } else if (mul.node->src[1] == norm_out) {
                    norm_w = mul.node->src[0];
                } else {
                    break;
                }
                const ggml_tensor * norm_src = norm.node->src[0];
                const ggml_tensor * x_orig   = cur.node->src[1];
                if (!norm_src || !norm_w || mul.node != x_orig) break;
                if (ggml_hexagon_tensor_get_mm_uses(x_orig) != nw + nl) break;
                if (nw + 2 + 2 * nl > HTP_OP_MAX_INPUTS) break;
                if (norm_src->type != GGML_TYPE_F32 || x_orig->type != GGML_TYPE_F32 || norm_w->type != GGML_TYPE_F32) break;
                if (ggml_nrows(norm_w) != 1 || norm_w->ne[0] != x_orig->ne[0] || !ggml_are_same_shape(norm_src, x_orig)) break;
                if (!ggml_is_contiguous(norm_src) || !ggml_is_contiguous(x_orig) || !ggml_is_contiguous(norm_w)) break;
                bool overlap = false;
                for (const auto * o : cur.outputs) {
                    if (htp_tensors_overlap(o, norm_src) || htp_tensors_overlap(o, norm_w)) {
                        overlap = true;
                        break;
                    }
                }
                if (overlap) break;

                float eps = 0.0f, scale = 0.0f;
                memcpy(&eps,   &norm.params[0], sizeof(float));
                memcpy(&scale, &norm.params[1], sizeof(float));

                const ggml_tensor * weights[HTP_LORA_MAX_WEIGHTS]    = {};
                htp_lora_adapter_desc ads[HTP_LORA_MAX_ADAPTERS]     = {};
                for (uint32_t p = 0; p < nw; ++p) weights[p] = cur.inputs[p];
                for (uint32_t l = 0; l < nl; ++l) {
                    ads[l] = { cur.inputs[kp->lora_a_slot[l]], cur.inputs[kp->lora_b_slot[l]],
                               kp->lora_scale[l], kp->lora_target[l] };
                }
                const bool do_collapse = (norm_src->ne[2] > 1 || norm_src->ne[3] > 1);
                const ggml_tensor * src_eff =
                    do_collapse ? cur.add_dummy(htp_tensor_collapse_rows(norm_src)) : norm_src;

                struct htp_lora_kernel_params new_kp;
                uint8_t norm_slot = 0xff;
                if (!htp_kparams_solve_matmul_lora(sess, weights, (int32_t) nw, src_eff, norm_w, true, eps, scale, 0,
                                                   ads, (int32_t) nl, &new_kp, &norm_slot)) {
                    break;
                }

                cur.inputs.clear();
                for (uint32_t p = 0; p < nw; ++p) cur.inputs.push_back(weights[p]);
                cur.inputs.push_back(src_eff);
                cur.inputs.push_back(norm_w);
                for (uint32_t l = 0; l < nl; ++l) {
                    cur.inputs.push_back(ads[l].a);
                    cur.inputs.push_back(ads[l].b);
                }
                htp_lora_push_fused_safe(cur, norm.node);
                for (const ggml_tensor * f : norm.fused) htp_lora_push_fused_safe(cur, f);
                htp_lora_push_fused_safe(cur, mul.node);
                cur.set_kparams(new_kp);

                nodes[out_n - 2] = std::move(cur);
                out_n -= 1;
                absorbed = true;
            } while (0);

            if (!absorbed) {
                if (out_n != idx) {
                    nodes[out_n] = std::move(nodes[idx]);
                }
                ++out_n;
            }
        }
        nodes.resize(out_n);
    }

    // Pass 4: Absorb following GEGLU/SWIGLU or SILU/GELU+MUL into 2-weight HTP_OP_MUL_MAT_LORA
    if (htp_fusion_enabled(HTP_FUSE_MUL_MAT_NX_GLU)) {
        size_t out_n = 0;
        for (size_t idx = 0; idx < nodes.size(); ++idx) {
            size_t absorbed_consume = 0;
            do {
                if (idx + 1 >= nodes.size()) break;
                htp_opnode & cur = nodes[idx];
                if (cur.opcode != HTP_OP_MUL_MAT_LORA || cur.outputs.size() != 2) break;
                const auto * kp = cur.kparams<htp_lora_kernel_params>();
                if (kp->n_weights != 2 || kp->glu != 0) break;

                uint8_t             glu_kind = 0;
                size_t              consume  = 0;
                const ggml_tensor * a        = nullptr;
                const ggml_tensor * b        = nullptr;
                const ggml_tensor * glu_dst  = nullptr;

                const htp_opnode & n1 = nodes[idx + 1];
                if ((n1.opcode == HTP_OP_GLU_GEGLU || n1.opcode == HTP_OP_GLU_SWIGLU) && n1.node && n1.node->src[1]) {
                    glu_kind = (n1.opcode == HTP_OP_GLU_GEGLU) ? HTP_MM_GLU_GEGLU : HTP_MM_GLU_SWIGLU;
                    a        = n1.node->src[0];
                    b        = n1.node->src[1];
                    glu_dst  = n1.node;
                    consume  = 1;
                } else if (idx + 2 < nodes.size() &&
                           (n1.opcode == HTP_OP_UNARY_GELU || n1.opcode == HTP_OP_UNARY_SILU) &&
                           n1.fused.empty() && n1.node && nodes[idx + 2].opcode == HTP_OP_MUL && nodes[idx + 2].node) {
                    const ggml_tensor * u = n1.node;
                    const ggml_tensor * m = nodes[idx + 2].node;
                    if (!ggml_hexagon_tensor_is_fuseable(u) || (u->flags & GGML_TENSOR_FLAG_OUTPUT) || u->view_src) {
                        break;
                    }
                    if (m->src[0] == u) {
                        b = m->src[1];
                    } else if (m->src[1] == u) {
                        b = m->src[0];
                    } else {
                        break;
                    }
                    glu_kind = (n1.opcode == HTP_OP_UNARY_GELU) ? HTP_MM_GLU_GEGLU : HTP_MM_GLU_SWIGLU;
                    a        = u->src[0];
                    glu_dst  = m;
                    consume  = 2;
                } else {
                    break;
                }

                if (!a || !b || !glu_dst) break;
                const ggml_tensor * d0 = cur.outputs[0];
                const ggml_tensor * d1 = cur.outputs[1];
                int gate_idx = -1;
                if (htp_same_tensor_data(a, d0) && htp_same_tensor_data(b, d1)) {
                    gate_idx = 0;
                } else if (htp_same_tensor_data(a, d1) && htp_same_tensor_data(b, d0)) {
                    gate_idx = 1;
                } else {
                    break;
                }

                if (!ggml_hexagon_tensor_is_fuseable(a) || !ggml_hexagon_tensor_is_fuseable(b) ||
                    ((a->flags | b->flags) & GGML_TENSOR_FLAG_OUTPUT) || a->view_src || b->view_src) break;
                if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || glu_dst->type != GGML_TYPE_F32) break;
                if (!ggml_are_same_shape(a, glu_dst) || !ggml_are_same_shape(b, glu_dst) ||
                    !ggml_is_contiguous_rows(glu_dst) || glu_dst->nb[1] != a->nb[1]) break;
                if (htp_inputs_overlap_unsafe(cur, glu_dst)) break;

                const ggml_tensor *   w_reordered[2] = { cur.inputs[gate_idx], cur.inputs[1 - gate_idx] };
                const ggml_tensor *   act_ptr        = cur.inputs[2];
                const ggml_tensor *   norm_w_ptr     = kp->norm ? cur.inputs[kp->norm_w_slot] : nullptr;
                htp_lora_adapter_desc ads[HTP_LORA_MAX_ADAPTERS] = {};
                const uint32_t        nl             = kp->n_loras;
                for (uint32_t l = 0; l < nl; ++l) {
                    ads[l] = { cur.inputs[kp->lora_a_slot[l]], cur.inputs[kp->lora_b_slot[l]],
                               kp->lora_scale[l], (uint8_t) (kp->lora_target[l] ^ (uint8_t) gate_idx) };
                }

                struct htp_lora_kernel_params new_kp;
                uint8_t norm_slot = 0xff;
                if (!htp_kparams_solve_matmul_lora(sess, w_reordered, 2, act_ptr, norm_w_ptr, kp->norm != 0,
                                                   kp->norm_eps, kp->norm_scale, glu_kind, ads, (int32_t) nl, &new_kp,
                                                   &norm_slot)) {
                    break;
                }

                const bool collapsed = (d0 != a && d0 != b);
                const ggml_tensor * out_eff = collapsed ? cur.add_dummy(htp_tensor_collapse_rows(glu_dst)) : glu_dst;

                cur.inputs.clear();
                cur.inputs.push_back(w_reordered[0]);
                cur.inputs.push_back(w_reordered[1]);
                cur.inputs.push_back(act_ptr);
                if (norm_w_ptr) {
                    cur.inputs.push_back(norm_w_ptr);
                }
                for (uint32_t l = 0; l < nl; ++l) {
                    cur.inputs.push_back(ads[l].a);
                    cur.inputs.push_back(ads[l].b);
                }
                cur.outputs = { out_eff };
                for (size_t c = 1; c <= consume; ++c) {
                    htp_lora_push_fused_safe(cur, nodes[idx + c].node);
                }
                cur.set_kparams(new_kp);
                absorbed_consume = consume;
            } while (0);

            if (out_n != idx) {
                nodes[out_n] = std::move(nodes[idx]);
            }
            ++out_n;
            idx += absorbed_consume;
        }
        nodes.resize(out_n);
    }
}

bool htp_fusion_try_fuse(const struct ggml_hexagon_session * sess, struct htp_opnode & prev, const struct htp_opnode & cur) {
    if (!opt_opfusion) return false;
    if (!cur.fused.empty()) return false;

    switch (cur.opcode) {
        case HTP_OP_ADD:
            return (htp_fusion_enabled(HTP_FUSE_ALLREDUCE_ADD) && fuse_allreduce_add(sess, prev, cur)) ||
                   (htp_fusion_enabled(HTP_FUSE_MUL_MAT_ADD)   && fuse_mul_mat_add(sess, prev, cur))   ||
                   (htp_fusion_enabled(HTP_FUSE_RMS_NORM_MUL_ADD) && fuse_rms_norm_mul_add(sess, prev, cur)) ||
                   (htp_fusion_enabled(HTP_FUSE_BINARY_NX)     && fuse_binary_nx(sess, prev, cur));
        case HTP_OP_MUL:
            return (htp_fusion_enabled(HTP_FUSE_RMS_NORM_MUL)   && fuse_rms_norm_mul(sess, prev, cur))   ||
                   (htp_fusion_enabled(HTP_FUSE_UNARY_MUL_GLU)  && fuse_unary_mul_glu(sess, prev, cur))  ||
                   (htp_fusion_enabled(HTP_FUSE_MUL_MUL_SCALAR) && fuse_mul_mul_scalar(sess, prev, cur)) ||
                   (htp_fusion_enabled(HTP_FUSE_BINARY_NX)      && fuse_binary_nx(sess, prev, cur));
        case HTP_OP_SUB:
        case HTP_OP_DIV:
            return htp_fusion_enabled(HTP_FUSE_BINARY_NX) && fuse_binary_nx(sess, prev, cur);
        case HTP_OP_SCALE:
            return (htp_fusion_enabled(HTP_FUSE_RMS_NORM_SCALE) && fuse_rms_norm_scale(sess, prev, cur)) ||
                   (htp_fusion_enabled(HTP_FUSE_UNARY_SCALE)    && fuse_unary_scale(sess, prev, cur));
        case HTP_OP_UNARY_SILU:
            return (htp_fusion_enabled(HTP_FUSE_SSM_CONV_SILU) && fuse_ssm_conv_silu(sess, prev, cur)) ||
                   (htp_fusion_enabled(HTP_FUSE_UNARY_SCALE)   && fuse_scale_unary(sess, prev, cur));
        case HTP_OP_UNARY_EXP:
        case HTP_OP_UNARY_SIGMOID:
        case HTP_OP_UNARY_GELU:
        case HTP_OP_UNARY_GELU_QUICK:
        case HTP_OP_UNARY_GELU_ERF:
        case HTP_OP_UNARY_SOFTPLUS:
        case HTP_OP_UNARY_TANH:
            return htp_fusion_enabled(HTP_FUSE_UNARY_SCALE) && fuse_scale_unary(sess, prev, cur);
        case HTP_OP_GET_ROWS:
            return htp_fusion_enabled(HTP_FUSE_REPEAT_GET_ROWS) && fuse_repeat_get_rows(sess, prev, cur);
        case HTP_OP_CPY:
            return htp_fusion_enabled(HTP_FUSE_GDN_CPY) && fuse_gdn_cpy(sess, prev, cur);
        case HTP_OP_MUL_MAT:
            return htp_fusion_enabled(HTP_FUSE_MUL_MAT_NX) && fuse_mul_mat_nx(sess, prev, cur);
        case HTP_OP_MUL_MAT_ID:
            return htp_fusion_enabled(HTP_FUSE_MUL_MAT_ID_NX) && fuse_mul_mat_id_nx(sess, prev, cur);
        case HTP_OP_GLU_SWIGLU:
        case HTP_OP_GLU_GEGLU:
        case HTP_OP_GLU_GEGLU_QUICK:
        case HTP_OP_GLU_REGLU:
            return htp_fusion_enabled(HTP_FUSE_MUL_MAT_NX_GLU) && fuse_mul_mat_nx_glu(sess, prev, cur);
        default:
            return false;
    }
}

static void htp_fuse_gdn_state(const struct ggml_hexagon_session * sess, std::vector<struct htp_opnode> & nodes) {
    for (size_t c = 0; c < nodes.size(); ++c) {
        if (nodes[c].opcode != HTP_OP_GATED_DELTA_NET || nodes[c].inputs.size() != 6 ||
            nodes[c].fused.size() >= HTP_OP_MAX_FUSED) {
            continue;
        }
        const ggml_tensor * v = nodes[c].inputs[2];
        const ggml_tensor * s = nodes[c].inputs[5];

        const ggml_tensor * base_s = s;
        while (base_s->view_src &&
               (base_s->op == GGML_OP_RESHAPE || (base_s->op == GGML_OP_VIEW && base_s->view_offs == 0)) &&
               ggml_is_contiguous(base_s)) {
            base_s = base_s->view_src;
        }
        if (base_s->flags & GGML_TENSOR_FLAG_OUTPUT) continue;

        size_t p = c;
        for (size_t i = 0; i < c; ++i) {
            if (nodes[i].opcode == HTP_OP_GET_ROWS && nodes[i].fused.empty() &&
                nodes[i].outputs.size() == 1 && nodes[i].dst() == base_s) {
                p = i;
                break;
            }
        }
        if (p == c) continue;

        const ggml_tensor * gr_src0 = nodes[p].src0();
        const ggml_tensor * gr_src1 = nodes[p].src1();
        const int64_t n_seqs = v->ne[3];
        if (htp_tensors_overlap(nodes[c].outputs[0], gr_src0) || htp_tensors_overlap(nodes[c].outputs[0], gr_src1)) {
            continue;
        }
        if (nodes[c].outputs.size() > 1 && nodes[c].outputs[1]) {
            const ggml_tensor * dst_cache = nodes[c].outputs[1];
            if (htp_tensors_overlap(dst_cache, gr_src1)) continue;
            if (htp_tensors_overlap(gr_src0, dst_cache)) {
                const uintptr_t src0_ptr = (uintptr_t) htp_tensor_data_ptr(gr_src0);
                const uintptr_t dst_ptr  = (uintptr_t) htp_tensor_data_ptr(dst_cache);
                const bool slot_aligned  = (n_seqs == 1) &&
                                           (dst_cache->type == gr_src0->type) &&
                                           (dst_ptr >= src0_ptr) &&
                                           (gr_src0->nb[1] > 0) &&
                                           ((dst_ptr - src0_ptr) % gr_src0->nb[1] == 0) &&
                                           (dst_cache->ne[2] == 1 || gr_src0->nb[1] == dst_cache->nb[2]);
                if (!slot_aligned) continue;
            }
        }

        bool safe = true;
        for (size_t k = 0; k < nodes.size() && safe; ++k) {
            if (k == p) continue;
            if (k != c) {
                for (const ggml_tensor * in : nodes[k].inputs) {
                    for (const ggml_tensor * t = in; t != nullptr; t = t->view_src) {
                        if (t == base_s) { safe = false; break; }
                    }
                    if (!safe) break;
                }
            }
            if (k > p && k < c) {
                for (const ggml_tensor * out : nodes[k].outputs) {
                    if (out && (htp_tensors_overlap(out, gr_src0) || htp_tensors_overlap(out, gr_src1) ||
                                htp_tensors_overlap(out, base_s))) {
                        safe = false;
                        break;
                    }
                }
            }
        }
        if (!safe) continue;

        htp_opnode cand = nodes[c];
        cand.inputs[5] = gr_src0;
        cand.inputs.push_back(gr_src1);
        cand.fused.insert(cand.fused.begin(), nodes[p].node);
        if (!htp_kparams_solve(sess, cand)) continue;

        nodes[c] = std::move(cand);
        nodes.erase(nodes.begin() + (ptrdiff_t) p);
        --c;
    }
}

static bool htp_unwrap_transpose_01(const ggml_tensor * t, const ggml_tensor ** out_src) {
    if (!t || !t->view_src || t->view_offs != 0) {
        return false;
    }
    if (t->op == GGML_OP_TRANSPOSE) {
        *out_src = t->view_src;
        return true;
    }
    if (t->op == GGML_OP_PERMUTE) {
        if (ggml_get_op_params_i32(t, 0) == 1 &&
            ggml_get_op_params_i32(t, 1) == 0 &&
            ggml_get_op_params_i32(t, 2) == 2 &&
            ggml_get_op_params_i32(t, 3) == 3) {
            *out_src = t->view_src;
            return true;
        }
    }
    return false;
}

static bool htp_tensor_is_view_of(const ggml_tensor * in, const ggml_tensor * target) {
    if (!in || !target) {
        return false;
    }
    for (const ggml_tensor * t = in; t != nullptr; t = t->view_src) {
        if (t == target) {
            return true;
        }
    }
    return false;
}

static bool htp_tensor_reads_from(const ggml_tensor * in, const ggml_tensor * target) {
    if (!in || !target) {
        return false;
    }
    if (htp_tensor_is_view_of(in, target)) {
        return true;
    }
    return htp_tensors_overlap(in, target);
}

static void htp_fuse_concat_ssm_conv(const struct ggml_hexagon_session * sess, std::vector<struct htp_opnode> & nodes) {
    for (size_t c = 0; c < nodes.size(); ++c) {
        if (nodes[c].opcode != HTP_OP_SSM_CONV || nodes[c].inputs.size() != 2 ||
            nodes[c].outputs.size() != 1 || nodes[c].fused.size() >= HTP_OP_MAX_FUSED) {
            continue;
        }

        const ggml_tensor * conv_x      = nodes[c].src0();
        const ggml_tensor * conv_weight = nodes[c].src1();
        const ggml_tensor * conv_dst    = nodes[c].dst();
        if (conv_x->flags & GGML_TENSOR_FLAG_OUTPUT) {
            continue;
        }

        size_t p = c;
        for (size_t i = 0; i < c; ++i) {
            if (nodes[i].opcode == HTP_OP_CONCAT && nodes[i].fused.empty() &&
                nodes[i].outputs.size() == 1 && nodes[i].dst() == conv_x &&
                ggml_get_op_params_i32(nodes[i].node, 0) == 0) {
                p = i;
                break;
            }
        }
        if (p == c) {
            continue;
        }

        const ggml_tensor * cat_state = nodes[p].src0();
        const ggml_tensor * cat_xT    = nodes[p].src1();
        const ggml_tensor * cat_x     = nullptr;
        if (!htp_unwrap_transpose_01(cat_xT, &cat_x)) {
            continue;
        }

        const int64_t d_conv  = conv_weight->ne[0];
        const int64_t d_inner = conv_weight->ne[1];
        const int64_t n_t     = conv_dst->ne[1];
        const int64_t n_s     = conv_dst->ne[2];
        const int64_t S       = d_conv - 1;

        size_t q          = nodes.size();
        size_t n_other    = 0;
        for (size_t k = 0; k < nodes.size(); ++k) {
            if (k == p || k == c) {
                continue;
            }
            bool uses_conv_x = false;
            for (const ggml_tensor * in : nodes[k].inputs) {
                if (htp_tensor_is_view_of(in, conv_x)) {
                    uses_conv_x = true;
                    break;
                }
            }
            if (uses_conv_x) {
                q = k;
                ++n_other;
            }
        }
        if (n_other > 1) {
            continue;
        }

        const bool has_cpy = (n_other == 1);
        const ggml_tensor * cpy_dst = nullptr;
        if (has_cpy) {
            if (q <= p || nodes[q].opcode != HTP_OP_CPY || !nodes[q].fused.empty() ||
                nodes[q].outputs.size() != 1 || (nodes[q].node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                continue;
            }
            if (nodes[c].fused.size() + 2 > HTP_OP_MAX_FUSED) {
                continue;
            }
            const ggml_tensor * cpy_src = nodes[q].src0();
            cpy_dst = nodes[q].dst();
            const size_t conv_x_es = ggml_type_size(conv_x->type);
            if (cpy_src->op != GGML_OP_VIEW ||
                !htp_same_tensor_data(cpy_src->view_src, conv_x) ||
                cpy_src->ne[0] != S || cpy_src->ne[1] != d_inner || cpy_src->ne[2] != n_s || cpy_src->ne[3] != 1 ||
                cpy_src->nb[0] != conv_x_es || cpy_src->nb[1] != conv_x->nb[1] ||
                (n_s > 1 && cpy_src->nb[2] != conv_x->nb[2]) ||
                cpy_src->view_offs != (size_t) n_t * conv_x_es) {
                continue;
            }
        }

        const bool inplace_x_dst = htp_same_tensor_data(conv_dst, cat_x) &&
                                   conv_dst->type == cat_x->type &&
                                   ggml_are_same_shape(conv_dst, cat_x) &&
                                   ggml_are_same_stride(conv_dst, cat_x);
        if (htp_tensors_overlap(conv_dst, cat_state) ||
            (htp_tensors_overlap(conv_dst, cat_x) && !inplace_x_dst) ||
            htp_tensors_overlap(conv_dst, conv_weight)) {
            continue;
        }
        if (has_cpy) {
            if (htp_tensors_overlap(conv_dst, cpy_dst) ||
                htp_tensors_overlap(cpy_dst, cat_x) ||
                htp_tensors_overlap(cpy_dst, conv_weight)) {
                continue;
            }
            if (htp_tensors_overlap(cpy_dst, cat_state)) {
                const bool inplace_state = (n_s == 1) &&
                                           htp_same_tensor_data(cpy_dst, cat_state) &&
                                           cpy_dst->type == cat_state->type;
                if (!inplace_state) {
                    continue;
                }
            }
        }

        bool safe = true;
        for (size_t k = p + 1; k < c && safe; ++k) {
            if (has_cpy && k == q) {
                continue;
            }
            for (const ggml_tensor * out : nodes[k].outputs) {
                if (!out) {
                    continue;
                }
                if (htp_tensors_overlap(out, cat_state) ||
                    htp_tensors_overlap(out, cat_x) ||
                    htp_tensors_overlap(out, conv_x) ||
                    htp_tensors_overlap(out, conv_weight) ||
                    (has_cpy && htp_tensors_overlap(out, cpy_dst))) {
                    safe = false;
                    break;
                }
            }
        }
        if (has_cpy && safe) {
            const size_t lo = (q < c) ? (q + 1) : (c + 1);
            const size_t hi = (q < c) ? c       : q;
            for (size_t k = lo; k < hi && safe; ++k) {
                for (const ggml_tensor * in : nodes[k].inputs) {
                    if (htp_tensor_reads_from(in, cpy_dst)) {
                        safe = false;
                        break;
                    }
                }
                for (const ggml_tensor * out : nodes[k].outputs) {
                    if (out && (htp_tensors_overlap(out, cpy_dst) || (q > c && htp_tensors_overlap(out, conv_x)))) {
                        safe = false;
                        break;
                    }
                }
            }
        }
        if (!safe) {
            continue;
        }

        htp_opnode cand = nodes[c];
        cand.inputs = { cat_state, conv_weight, cat_x };
        if (has_cpy) {
            cand.outputs = { conv_dst, cpy_dst };
            cand.fused.insert(cand.fused.begin(), nodes[q].node);
        }
        cand.fused.insert(cand.fused.begin(), nodes[p].node);
        if (!htp_kparams_solve(sess, cand)) {
            continue;
        }

        nodes[c] = std::move(cand);
        if (has_cpy) {
            nodes.erase(nodes.begin() + (ptrdiff_t) q);
            if (q < c) {
                --c;
            }
        }
        nodes.erase(nodes.begin() + (ptrdiff_t) p);
        --c;
    }
}

static void htp_fuse_mul_mat_nx_glu(const struct ggml_hexagon_session * sess, std::vector<struct htp_opnode> & nodes) {
    for (size_t c = 0; c < nodes.size(); ++c) {
        if (!htp_is_fuseable_mmnx_glu_op(nodes[c].opcode) || nodes[c].inputs.size() != 2 ||
            nodes[c].outputs.size() != 1 || !nodes[c].src1()) {
            continue;
        }

        const ggml_tensor * s0 = nodes[c].src0();
        const ggml_tensor * s1 = nodes[c].src1();

        size_t p = c;
        for (size_t i = 0; i < c; ++i) {
            if (nodes[i].opcode != HTP_OP_MUL_MAT_NX || nodes[i].outputs.size() != 2 || nodes[i].inputs.size() != 3) {
                continue;
            }
            const ggml_tensor * o0 = nodes[i].outputs[0];
            const ggml_tensor * o1 = nodes[i].outputs[1];
            if ((htp_same_tensor_data(s0, o0) && htp_same_tensor_data(s1, o1)) ||
                (htp_same_tensor_data(s0, o1) && htp_same_tensor_data(s1, o0))) {
                p = i;
                break;
            }
        }
        if (p == c) {
            continue;
        }

        const ggml_tensor * w0  = nodes[p].inputs[0];
        const ggml_tensor * w1  = nodes[p].inputs[1];
        const ggml_tensor * act = nodes[p].inputs[2];
        const ggml_tensor * o0  = nodes[p].outputs[0];
        const ggml_tensor * o1  = nodes[p].outputs[1];

        bool safe = true;
        for (size_t k = 0; k < nodes.size() && safe; ++k) {
            if (k == p || k == c) {
                continue;
            }
            for (const ggml_tensor * in : nodes[k].inputs) {
                if (htp_tensor_reads_from(in, o0) || htp_tensor_reads_from(in, o1)) {
                    safe = false;
                    break;
                }
            }
            if (safe && k > p && k < c) {
                for (const ggml_tensor * out : nodes[k].outputs) {
                    if (out && (htp_tensors_overlap(out, w0) || htp_tensors_overlap(out, w1) ||
                                htp_tensors_overlap(out, act) || htp_tensors_overlap(out, o0) ||
                                htp_tensors_overlap(out, o1))) {
                        safe = false;
                        break;
                    }
                }
            }
        }
        if (!safe) {
            continue;
        }

        htp_opnode cand = nodes[p];
        if (!fuse_mul_mat_nx_glu(sess, cand, nodes[c])) {
            continue;
        }

        nodes[c] = std::move(cand);
        nodes.erase(nodes.begin() + (ptrdiff_t) p);
        --c;
    }
}

void htp_fusion_optimize_nodes(const struct ggml_hexagon_session * sess, const struct ggml_cgraph * graph, std::vector<struct htp_opnode> & nodes) {
    GGML_UNUSED(graph);

    if (htp_fusion_enabled(HTP_FUSE_MUL_MAT_NX_GLU)) {
        htp_fuse_mul_mat_nx_glu(sess, nodes);
    }
    if (htp_fusion_enabled(HTP_FUSE_GDN_STATE)) {
        htp_fuse_gdn_state(sess, nodes);
    }
    if (htp_fusion_enabled(HTP_FUSE_CONCAT_SSM_CONV)) {
        htp_fuse_concat_ssm_conv(sess, nodes);
    }
}
