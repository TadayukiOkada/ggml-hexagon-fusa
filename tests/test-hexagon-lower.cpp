// Graph lowering (htp_lower_graph) must lower the same F32 roots to F16, with the same kernel params,
// whether its transactions run as upstream's loops or as the fork's faster version
// (ggml/src/ggml-hexagon/htp-lower-fast.inc). GGML_HEXAGON_LOWER_VERIFY=1 runs both on every graph and
// aborts on any difference; this test sets it and computes small random F32 graphs on the HTP: mat-muls
// with F16 weights, element-wise and broadcast binary ops, scale, RMS norm, unary ops, split GLUs, repeat,
// sum_rows, softmax and transposes. Three graphs come first: a case where
// an earlier fast version gave up a transaction that upstream completes (S = REPEAT(x), T = ADD(S, y),
// U = SCALE(T), F = SWIGLU(S, U), G = SWIGLU(S, T), SUM_ROWS of F and G), a root with 65 users (more
// touched nodes than one word of the fast version's bitset) and two branches on offset views of one root.
//
// usage: test-hexagon-lower [device, default HTP0] [random graphs, default 300] [fast mode, default 2]
//   fast mode: GGML_HEXAGON_LOWER_FAST (1 without the connectivity check, 2 with it)
//   TEST_LOWER_TRACE=1 prints the ops of each graph before it is computed
// exit: 0 pass, 1 a graph failed to compute or a pinned graph did not reach the HTP, abort on a lowering
// difference, 77 no such device (skip)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int D = 64;  // rows and columns of every activation, so transposes keep the shape

struct graph_case {
    ggml_backend_t be;
    std::mt19937 & rng;

    ggml_context *              cw = nullptr;  // weights
    ggml_context *              cc = nullptr;  // activations
    std::vector<ggml_tensor *>  weights, inputs, pool, outs;

    graph_case(ggml_backend_t be_, std::mt19937 & rng_) : be(be_), rng(rng_) {
        ggml_init_params ip = { 512 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        cw = ggml_init(ip);
        cc = ggml_init(ip);
        for (int i = 0; i < 3; ++i) {
            weights.push_back(ggml_new_tensor_2d(cw, GGML_TYPE_F16, D, D));
        }
        for (int i = 0; i < 3; ++i) {
            inputs.push_back(ggml_new_tensor_2d(cc, GGML_TYPE_F32, D, D));
            pool.push_back(inputs.back());
        }
        inputs.push_back(ggml_new_tensor_2d(cc, GGML_TYPE_F32, D, 1));  // broadcast vector
    }
    ~graph_case() {
        ggml_free(cc);
        ggml_free(cw);
    }

    ggml_tensor * vec() const { return inputs.back(); }

    ggml_tensor * pick() {
        // mostly recent tensors, so the graphs get deep
        const size_t n = pool.size();
        std::uniform_int_distribution<size_t> recent(n > 4 ? n - 4 : 0, n - 1), any(0, n - 1);
        return pool[(rng() % 3) ? recent(rng) : any(rng)];
    }

    void add(ggml_tensor * t) {
        if (t && t->ne[0] == D && t->ne[1] == D && ggml_backend_supports_op(be, t)) {
            pool.push_back(t);
        }
    }

    void random_op() {
        ggml_tensor * a = pick();
        ggml_tensor * b = pick();
        switch (rng() % 15) {
            case 0:  add(ggml_mul_mat(cc, weights[rng() % weights.size()], a)); break;
            case 1:  add(ggml_add(cc, a, b)); break;
            case 2:  add(ggml_mul(cc, a, b)); break;
            case 3:  add(ggml_sub(cc, a, b)); break;
            case 4:  add(ggml_scale(cc, a, 0.5f)); break;
            case 5:  add(ggml_mul(cc, ggml_rms_norm(cc, a, 1e-6f), vec())); break;
            case 6:  add(ggml_silu(cc, a)); break;
            case 7:  add(ggml_gelu(cc, a)); break;
            case 8:  add(ggml_swiglu_split(cc, a, b)); break;
            case 9:  add(ggml_geglu_split(cc, a, b)); break;
            case 10: add(ggml_repeat(cc, vec(), a)); break;
            case 11: add(ggml_add(cc, a, vec())); break;
            case 12: add(ggml_cont(cc, ggml_transpose(cc, a))); break;
            case 13: add(ggml_repeat(cc, ggml_sum_rows(cc, a), a)); break;
            // Mat-muls between activations are left out: the DSP fails (dspqueue_read 0x2e, also before
            // the v1.1.0 import) on some graphs that have one and a split GLU, e.g. MUL_MAT(x1, x0) and
            // SWIGLU(x2, x1) as two outputs. Not a lowering problem.
            case 14: add(ggml_soft_max(cc, a)); break;
        }
    }

    // the counterexample from the header
    void pinned() {
        ggml_tensor * s = ggml_repeat(cc, vec(), inputs[0]);
        ggml_tensor * t = ggml_add(cc, s, inputs[1]);
        ggml_tensor * u = ggml_scale(cc, t, 0.5f);
        outs.push_back(ggml_sum_rows(cc, ggml_swiglu_split(cc, s, u)));
        outs.push_back(ggml_sum_rows(cc, ggml_swiglu_split(cc, s, t)));
    }

    // a root with 65 users
    void pinned_wide() {
        ggml_tensor * s   = ggml_repeat(cc, vec(), inputs[0]);
        ggml_tensor * acc = nullptr;
        for (int k = 0; k < 64; ++k) {
            ggml_tensor * c = ggml_scale(cc, s, 0.01f * (float) (k + 1));
            acc             = acc ? ggml_add(cc, acc, c) : c;
        }
        outs.push_back(ggml_sum_rows(cc, ggml_mul(cc, acc, s)));
    }

    // two branches on offset views of one root
    void pinned_views() {
        ggml_tensor * big = ggml_add(cc, ggml_mul_mat(cc, weights[0], inputs[0]), inputs[1]);
        ggml_tensor * v0  = ggml_view_2d(cc, big, D, D / 2, big->nb[1], 0);
        ggml_tensor * v1  = ggml_view_2d(cc, big, D, D / 2, big->nb[1], big->nb[1] * (D / 2));
        outs.push_back(ggml_sum_rows(cc, ggml_mul_mat(cc, weights[1], ggml_scale(cc, v0, 0.5f))));
        outs.push_back(ggml_sum_rows(cc, ggml_mul(cc, ggml_rms_norm(cc, v1, 1e-6f), vec())));
    }

    enum result { COMPUTED, FAILED, SKIPPED };

    result run() {
        ggml_cgraph * gf = ggml_new_graph(cc);
        for (ggml_tensor * o : outs) {
            ggml_set_output(o);
            ggml_build_forward_expand(gf, o);
        }
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            if (!ggml_backend_supports_op(be, ggml_graph_node(gf, i))) {
                return SKIPPED;  // not an HTP graph (pool members are checked, the rest here)
            }
        }
        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors(cw, be);
        ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_t bc = ggml_backend_alloc_ctx_tensors(cc, be);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (ggml_tensor * w : weights) {
            std::vector<ggml_fp16_t> v(ggml_nelements(w));
            for (auto & x : v) {
                x = ggml_fp32_to_fp16(u(rng) * 0.2f);
            }
            ggml_backend_tensor_set(w, v.data(), 0, ggml_nbytes(w));
        }
        for (ggml_tensor * in : inputs) {
            std::vector<float> v(ggml_nelements(in));
            for (auto & x : v) {
                x = u(rng);
            }
            ggml_backend_tensor_set(in, v.data(), 0, ggml_nbytes(in));
        }
        if (getenv("TEST_LOWER_TRACE")) {
            auto desc = [](const ggml_tensor * t) {
                static char buf[4][96];
                static int  k = 0;
                char *      b = buf[k++ % 4];
                snprintf(b, sizeof(buf[0]), "%s:%s:%lldx%lld", t->name, ggml_type_name(t->type), (long long) t->ne[0],
                         (long long) t->ne[1]);
                return b;
            };
            fprintf(stderr, "test-hexagon-lower: graph:");
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                const ggml_tensor * n = ggml_graph_node(gf, i);
                fprintf(stderr, " %s=%s(%s%s%s)%s", desc(n), ggml_op_desc(n), n->src[0] ? desc(n->src[0]) : "",
                        n->src[1] ? "," : "", n->src[1] ? desc(n->src[1]) : "", i + 1 < ggml_graph_n_nodes(gf) ? ";" : "\n");
            }
        }
        const bool ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
        ggml_backend_buffer_free(bc);
        ggml_backend_buffer_free(bw);
        return ok ? COMPUTED : FAILED;
    }
};

}  // namespace

int main(int argc, char ** argv) {
    const std::string dev_name = argc > 1 ? argv[1] : "HTP0";
    const int         n_graphs = argc > 2 ? atoi(argv[2]) : 300;
    const char *      mode     = argc > 3 ? argv[3] : "2";

    setenv("GGML_HEXAGON_LOWER_VERIFY", "1", 1);
    setenv("GGML_HEXAGON_LOWER_FAST", mode, 1);
    setenv("GGML_HEXAGON_OPLOWER", "1", 0);  // GGML_HEXAGON_OPLOWER=0 from outside: compute only

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name.c_str());
    if (!dev) {
        fprintf(stderr, "test-hexagon-lower: no device %s, skipped\n", dev_name.c_str());
        return 77;
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);

    std::mt19937 rng(4321);
    int          n_fail = 0, n_computed = 0, n_skipped = 0;
    auto count = [&](graph_case::result r, const char * what, int i) {
        if (r == graph_case::COMPUTED) {
            ++n_computed;
        } else if (r == graph_case::SKIPPED) {
            ++n_skipped;
        } else {
            fprintf(stderr, "test-hexagon-lower: %s %d failed to compute\n", what, i);
            ++n_fail;
        }
        return r;
    };
    void (graph_case::*pinned[])() = { &graph_case::pinned, &graph_case::pinned_wide, &graph_case::pinned_views };
    for (int i = 0; i < 3; ++i) {
        graph_case g(be, rng);
        (g.*pinned[i])();
        if (count(g.run(), "pinned graph", i) == graph_case::SKIPPED) {
            fprintf(stderr, "test-hexagon-lower: pinned graph %d has an op the device does not support\n", i);
            ++n_fail;
        }
    }
    for (int i = 0; i < n_graphs; ++i) {
        graph_case g(be, rng);
        const int n_ops = 4 + (int) (rng() % 28);
        for (int k = 0; k < n_ops; ++k) {
            g.random_op();
        }
        // one to three outputs among the last tensors
        const int n_out = 1 + (int) (rng() % 3);
        for (int k = 0; k < n_out && k < (int) g.pool.size() - 3; ++k) {
            g.outs.push_back(g.pool[g.pool.size() - 1 - k]);
        }
        if (g.outs.empty()) {
            ++n_skipped;
            continue;
        }
        count(g.run(), "graph", i);
    }

    ggml_backend_free(be);
    printf("test-hexagon-lower: mode %s, %d graphs computed with lowering verified, %d skipped, %d failed\n", mode,
           n_computed, n_skipped, n_fail);
    return n_fail ? 1 : 0;
}
