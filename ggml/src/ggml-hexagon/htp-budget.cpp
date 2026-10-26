#include "htp-budget.h"
#include "ggml-impl.h"

double opt_op_budget_ms = GGML_HEXAGON_OP_BUDGET_MS_DEFAULT;

static bool htp_budget_check(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op, double est) {
    if (est > opt_op_budget_ms) {
        HEX_VERBOSE("ggml-hex: %s %s %s estimated %.1f ms > budget %.1f ms, left to the CPU\n", htp_session_name(sess),
                    ggml_op_desc(op), op->name, est, opt_op_budget_ms);
        return false;
    }
    return true;
}

// Conservative op time estimates (ms) for sorts, which have no point where VTCM could be released.
// Fitted to v73 (QCS9075, 4 HVX threads): a single row is sorted by all threads, several rows
// one per thread; ARGSORT rows above 512 and TOP_K rows that are not a multiple of 128 or take
// a large k go through much slower paths.
static double htp_estimate_sort_ms(const struct ggml_tensor * src0, int64_t k, uint32_t n_threads) {
    const double n      = (double) src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);
    const double rounds = (double) ((nrows + n_threads - 1) / n_threads);

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
        default:
            return true;
    }
}
