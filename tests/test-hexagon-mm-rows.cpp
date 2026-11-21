// A quantized mat-mul with one to four activation rows must give every row the same bits whatever the
// number of rows: a decode step computes one row, the verify batch of speculative decoding (MTP) the
// same row among two to four, and a difference there makes the generated text depend on whether
// speculative decoding is on. The HVX kernels pick different code by the row count and the weight shape
// (vec_dot_32x1 / _32x2, and from two rows on large weights the kernels on a compact activation copy),
// so each row of an R-row product is compared bit for bit with the same row computed alone.
//
// Cases: Q4_0 and Q6_K weights (as llama.cpp places them, so the HTP repacks them into tiles), K 256 and
// 1536, 2048 to 8192 weight rows with a partial last tile, a plain mat-mul and a gate/up pair with GEGLU,
// activation rows of very different magnitudes (fp16-subnormal activation scales).
//
// usage: test-hexagon-mm-rows [device, default HTP0]
// exit: 0 pass, 1 fail, 77 no such device (skip)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct mm_case {
    ggml_type type;
    int       k;
    int       n;      // weight rows
    bool      glu;
};

struct runner {
    ggml_backend_t       be;
    const mm_case &      c;
    std::vector<uint8_t> w0, w1;   // quantized weights
    const float *        x;        // activation rows of length k

    // rows [r0, r0 + rows) of x through the case, output [rows][n]
    std::vector<float> run(int rows, int r0) const {
        ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context *   cw = ggml_init(ip);
        ggml_tensor *    a0 = ggml_new_tensor_2d(cw, c.type, c.k, c.n);
        ggml_tensor *    a1 = c.glu ? ggml_new_tensor_2d(cw, c.type, c.k, c.n) : nullptr;
        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors(cw, be);
        ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(a0, w0.data(), 0, ggml_nbytes(a0));
        if (a1) {
            ggml_backend_tensor_set(a1, w1.data(), 0, ggml_nbytes(a1));
        }

        ggml_context * cc = ggml_init(ip);
        ggml_tensor *  b  = ggml_new_tensor_2d(cc, GGML_TYPE_F32, c.k, rows);
        ggml_tensor *  o  = ggml_mul_mat(cc, a0, b);
        if (c.glu) {
            o = ggml_geglu_split(cc, o, ggml_mul_mat(cc, a1, b));
        }
        ggml_cgraph * gf = ggml_new_graph(cc);
        ggml_build_forward_expand(gf, o);
        ggml_backend_buffer_t bc = ggml_backend_alloc_ctx_tensors(cc, be);
        ggml_backend_tensor_set(b, x + (size_t) r0 * c.k, 0, ggml_nbytes(b));

        std::vector<float> out((size_t) rows * c.n);
        if (ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS) {
            ggml_backend_tensor_get(o, out.data(), 0, ggml_nbytes(o));
        } else {
            out.clear();
        }
        ggml_backend_buffer_free(bc);
        ggml_free(cc);
        ggml_backend_buffer_free(bw);
        ggml_free(cw);
        return out;
    }
};

}  // namespace

int main(int argc, char ** argv) {
    const std::string dev_name = argc > 1 ? argv[1] : "HTP0";

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name.c_str());
    if (!dev) {
        fprintf(stderr, "test-hexagon-mm-rows: no device %s, skipped\n", dev_name.c_str());
        return 77;
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);

    constexpr int MAX_ROWS = 4;
    const mm_case cases[] = {
        { GGML_TYPE_Q4_0, 1536, 8192, false },
        { GGML_TYPE_Q4_0, 1536, 4100, true  },
        { GGML_TYPE_Q4_0,  256, 8192, false },
        { GGML_TYPE_Q4_0, 1536, 2048, false },
        { GGML_TYPE_Q6_K, 1536, 8200, false },
        { GGML_TYPE_Q6_K, 1536, 4100, true  },
        { GGML_TYPE_Q6_K,  256, 8192, false },
    };

    std::mt19937                    rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    int n_fail = 0;
    for (const mm_case & c : cases) {
        // activation rows from 1 down to 1e-6 in magnitude: the smallest have fp16-subnormal scales
        std::vector<float> x((size_t) MAX_ROWS * 2 * c.k);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = nd(rng) * std::pow(1e-6f, (float) (i / c.k) / (2 * MAX_ROWS - 1));
        }
        runner rn = { be, c, {}, {}, x.data() };
        std::vector<float> wf((size_t) c.k * c.n);
        for (int i = 0; i < (c.glu ? 2 : 1); ++i) {
            for (float & v : wf) {
                v = nd(rng) * 0.05f;
            }
            std::vector<uint8_t> & q = i ? rn.w1 : rn.w0;
            q.resize(ggml_row_size(c.type, c.k) * c.n);
            ggml_quantize_chunk(c.type, wf.data(), q.data(), 0, c.n, c.k, nullptr);
        }

        std::vector<std::vector<float>> one(2 * MAX_ROWS);
        for (int r = 0; r < 2 * MAX_ROWS; ++r) {
            one[r] = rn.run(1, r);
        }
        std::string res;
        int         bad_case = 0;
        for (int rows = 2; rows <= MAX_ROWS; ++rows) {
            for (int r0 : { 0, 2 * MAX_ROWS - rows }) {
                const std::vector<float> out = rn.run(rows, r0);
                int bad = 0;
                for (int r = 0; r < rows; ++r) {
                    const bool ok = !out.empty() && !one[r0 + r].empty() &&
                                    memcmp(&out[(size_t) r * c.n], one[r0 + r].data(), sizeof(float) * c.n) == 0;
                    bad += !ok;
                }
                if (bad) {
                    res += " rows " + std::to_string(rows) + "@" + std::to_string(r0) + ": " + std::to_string(bad) + " differ;";
                    bad_case += bad;
                }
            }
        }
        printf("%-5s k %4d n %4d %-4s %s%s\n", ggml_type_name(c.type), c.k, c.n, c.glu ? "glu" : "mm",
               bad_case ? "FAIL" : "ok", res.c_str());
        n_fail += bad_case != 0;
    }

    ggml_backend_free(be);
    printf("%s\n", n_fail ? "FAIL" : "PASS");
    return n_fail ? 1 : 0;
}
