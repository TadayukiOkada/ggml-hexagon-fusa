#include "htp-budget.h"
#include "htp-kparams.h"
#include "ggml-impl.h"

#include <algorithm>

double opt_op_budget_ms = GGML_HEXAGON_OP_BUDGET_MS_DEFAULT;

static bool htp_budget_check(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op, double est) {
    if (est > opt_op_budget_ms) {
        HEX_VERBOSE("ggml-hex: %s %s %s estimated %.1f ms > budget %.1f ms, left to the CPU\n", htp_session_name(sess),
                    ggml_op_desc(op), op->name, est, opt_op_budget_ms);
        return false;
    }
    return true;
}

// Conservative op time estimates (ms) for sorts. A single row cannot release VTCM; several rows
// release it between rounds when the skel is built with yield inside ops.
// Fitted to v73 (QCS9075, 4 HVX threads): a single row is sorted by all threads, several rows
// one per thread; ARGSORT rows above 512 and TOP_K rows that are not a multiple of 128 or take
// a large k go through much slower paths.
static double htp_estimate_sort_ms(const struct ggml_tensor * src0, int64_t k, uint32_t n_threads) {
    const double n      = (double) src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);
#ifdef GGML_HEXAGON_HTP_YIELD_IN_OPS
    // Several rows run in rounds with a yield point between them: only one row per thread counts.
    const double rounds = 1.0;
#else
    const double rounds = (double) ((nrows + n_threads - 1) / n_threads);
#endif

    if (k < 0) { // ARGSORT
        if (nrows == 1) {
            double npow2 = 1;
            while (npow2 < n) {
                npow2 *= 2;
            }
            return 0.4e-3 * npow2;
        }
        return (src0->ne[0] <= 512 ? 0.05e-3 : 1.1e-3) * n * rounds;
    }

    if (src0->ne[0] % 128 == 0 && k <= 128) {
        return nrows == 1 ? 0.1e-3 * n / n_threads : 0.1e-3 * n * rounds;
    }
    return 0.8e-3 * n * rounds;
}

// The long matmuls are output projections over many tokens (perplexity, embeddings). The HMX
// 2D kernel releases VTCM between weight chunks when the skel is built with yield inside ops;
// the other kernels have no such point. Estimate from v73 (QCS9075): about 5 TFLOP/s on HMX
// and 45 GB/s of weights, whichever bounds it (lm_head 262144x1536 x 512 tokens: 71 ms).
// Only for weights with data: llama.cpp places weights by asking about a 512-token matmul
// with a zero-size placeholder buffer, and a refusal there would keep the weight, and every
// decode step, off the HTP. In a real graph a refused matmul runs on the CPU with the weight
// copied over. test-backend-ops asks before allocating, so it does not see this check.
static bool htp_mul_mat_within_budget(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (src0->data == nullptr) {
        return true;
    }

#ifdef GGML_HEXAGON_HTP_YIELD_IN_OPS
    // The kernel the op runs with: as at graph compute, after its rows are collapsed. A Hadamard
    // MUL_MAT is another op.
    if (htp_op_remap(op) == HTP_OP_MUL_MAT) {
        htp_opnode node(HTP_OP_MUL_MAT, op);
        if (htp_kparams_solve(sess, node) && node.kparams<htp_mm_kernel_params>()->kernel_type == HTP_MM_KERNEL_HMX_2D) {
            return true;
        }
    }
#endif

    const double flops = 2.0 * src0->ne[0] * src0->ne[1] * ggml_nrows(src1);
    const double est   = (std::max)(flops / 5.0e9, (double) ggml_nbytes(src0) / 45.0e6);
    return htp_budget_check(sess, op, est);
}

bool htp_op_within_budget(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    if (opt_op_budget_ms <= 0) {
        return true;
    }

    const uint32_t n_threads = htp_session_n_threads(sess) > 0 ? htp_session_n_threads(sess) : 4;

    switch (op->op) {
        case GGML_OP_ARGSORT:
            return htp_budget_check(sess, op, htp_estimate_sort_ms(op->src[0], -1, n_threads));
        case GGML_OP_TOP_K:
            return htp_budget_check(sess, op, htp_estimate_sort_ms(op->src[0], op->ne[0], n_threads));
        case GGML_OP_MUL_MAT:
            return htp_mul_mat_within_budget(sess, op);
        default:
            return true;
    }
}
