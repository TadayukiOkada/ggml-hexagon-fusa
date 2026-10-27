// Tensor metadata of the Hexagon backend must not grow while graphs are allocated again and
// again: each init_tensor used to add an extra to the pool of the buffer it initialized in, and a
// buffer the graph allocator does not reset (weights, KV cache) kept them until it was freed.
// A SenseVoice server grew by 71 KiB per request this way (views of its FSMN tap weights), and
// llama.cpp by the KV cache views of each graph rebuild.
//
// On HTP0, a reused scheduler allocates graphs of changing shape that use
//   - views of a weight (usage WEIGHTS), as SenseVoice's FSMN taps,
//   - SET_ROWS into a cache tensor and a view of it (usage ANY), as llama.cpp's KV cache,
//   - scratch views made outside the allocator, as llama.cpp's state copies,
// checks the results against the host and fails if the metadata count of a buffer grows.
//
// usage: test-hexagon-metadata [device, default HTP0] [iterations, default 64]
// exit: 0 pass, 1 fail, 77 no such device (skip)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static constexpr int D = 64;      // row length
static constexpr int K = 11;      // weight views per graph (SenseVoice's FSMN kernel)
static constexpr int ROWS = 8;    // cache rows

using metadata_count_fn = size_t (*)(ggml_backend_buffer_t);

static int fail(const char * what) {
    fprintf(stderr, "test-hexagon-metadata: FAIL: %s\n", what);
    return 1;
}

static bool close_to(float a, float b) {
    return std::fabs(a - b) <= 1e-2f * (1.0f + std::fabs(b));
}

int main(int argc, char ** argv) {
    const std::string dev_name = argc > 1 ? argv[1] : "HTP0";
    const int iterations = argc > 2 ? atoi(argv[2]) : 64;

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name.c_str());
    if (!dev) {
        fprintf(stderr, "test-hexagon-metadata: no device %s, skipped\n", dev_name.c_str());
        return 77;
    }
    auto metadata_count = (metadata_count_fn) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev), "ggml_backend_buffer_metadata_count");
    if (!metadata_count) return fail("the backend has no ggml_backend_buffer_metadata_count");

    ggml_backend_t htp = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!htp || !cpu) return fail("cannot initialize the backends");

    // Persistent tensors: a weight [D, K] and a cache [D, ROWS], each in its own buffer.
    ggml_init_params pp = { 2 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_w = ggml_init(pp);
    ggml_context * ctx_c = ggml_init(pp);
    ggml_tensor * w = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, D, K);
    ggml_tensor * c = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, D, ROWS);
    ggml_set_name(w, "weight");
    ggml_set_name(c, "cache");
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, htp);
    ggml_backend_buffer_t buf_c = ggml_backend_alloc_ctx_tensors(ctx_c, htp);
    if (!buf_w || !buf_c) return fail("cannot allocate the persistent buffers");
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<float> wh((size_t) D * K), ch((size_t) D * ROWS, 0.0f);
    for (size_t i = 0; i < wh.size(); i++) wh[i] = 0.01f * (float) ((i * 7) % 23) - 0.1f;
    ggml_backend_tensor_set(w, wh.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(c, ch.data(), 0, ggml_nbytes(c));

    ggml_backend_t backends[] = { htp, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 4096, false, true);
    if (!sched) return fail("cannot create the scheduler");

    size_t count_w = 0, count_c = 0, count_compute = 0;
    int rc = 0;
    for (int it = 0; it < iterations && rc == 0; it++) {
        const int n = 1 + it % 3;  // the shape changes, so the scheduler allocates again

        ggml_init_params gp = { ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(gp);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n);
        ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n);
        ggml_set_input(x);
        ggml_set_input(idx);

        // y = x * (1 + sum_j w[:, j]), one view of the weight per tap
        ggml_tensor * y = x;
        for (int j = 0; j < K; j++) {
            ggml_tensor * wj = ggml_view_1d(ctx, w, D, (size_t) j * w->nb[1]);
            y = ggml_add(ctx, y, ggml_mul(ctx, x, wj));
        }
        // the columns of x go to cache rows idx, then m = cache^T x through a view of the cache
        ggml_tensor * s = ggml_set_rows(ctx, c, x, idx);
        ggml_tensor * cv = ggml_view_2d(ctx, s, D, ROWS, s->nb[1], 0);
        ggml_tensor * m = ggml_mul_mat(ctx, cv, x);
        ggml_set_output(y);
        ggml_set_output(m);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, s);
        ggml_build_forward_expand(gf, y);
        ggml_build_forward_expand(gf, m);

        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) { rc = fail("cannot allocate the graph"); break; }

        std::vector<float> xh((size_t) D * n);
        std::vector<int64_t> ih(n);
        for (size_t i = 0; i < xh.size(); i++) xh[i] = 0.05f * (float) ((i + 3 * it) % 17) - 0.3f;
        for (int r = 0; r < n; r++) ih[r] = (it + 3 * r) % ROWS;
        ggml_backend_tensor_set(x, xh.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(idx, ih.data(), 0, ggml_nbytes(idx));
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) { rc = fail("compute failed"); break; }

        for (int r = 0; r < n; r++)
            for (int i = 0; i < D; i++) ch[(size_t) ih[r] * D + i] = xh[(size_t) r * D + i];

        std::vector<float> yh((size_t) D * n), mh((size_t) ROWS * n);
        ggml_backend_tensor_get(y, yh.data(), 0, ggml_nbytes(y));
        ggml_backend_tensor_get(m, mh.data(), 0, ggml_nbytes(m));
        for (int r = 0; r < n && rc == 0; r++) {
            for (int i = 0; i < D; i++) {
                float sw = 1.0f;
                for (int j = 0; j < K; j++) sw += wh[(size_t) j * D + i];
                if (!close_to(yh[(size_t) r * D + i], xh[(size_t) r * D + i] * sw)) { rc = fail("wrong result through the weight views"); break; }
            }
            for (int q = 0; q < ROWS && rc == 0; q++) {
                float e = 0.0f;
                for (int i = 0; i < D; i++) e += ch[(size_t) q * D + i] * xh[(size_t) r * D + i];
                if (!close_to(mh[(size_t) r * ROWS + q], e)) rc = fail("wrong result through the cache view");
            }
        }

        // a scratch view outside the allocator, read and dropped, as llama.cpp's state copies
        ggml_init_params sp = { ggml_tensor_overhead(), nullptr, true };
        ggml_context * ctx_s = ggml_init(sp);
        ggml_tensor * row = ggml_view_1d(ctx_s, c, D, (size_t) ih[0] * c->nb[1]);
        if (ggml_backend_view_init(row) != GGML_STATUS_SUCCESS) { rc = fail("cannot initialize a scratch view"); }
        std::vector<float> rh(D);
        ggml_backend_tensor_get(row, rh.data(), 0, ggml_nbytes(row));
        for (int i = 0; i < D && rc == 0; i++)
            if (rh[i] != ch[(size_t) ih[0] * D + i]) rc = fail("wrong data through a scratch view");
        ggml_free(ctx_s);

        // The compute buffer's pool stays at the largest graph (recycled on reset); the
        // persistent buffers must not grow at all after the first graph.
        const size_t nw = metadata_count(buf_w), nc = metadata_count(buf_c);
        const size_t ny = metadata_count(y->buffer);
        if (it == 0) {
            count_w = nw; count_c = nc;
        } else if (nw != count_w || nc != count_c) {
            fprintf(stderr, "test-hexagon-metadata: graph %d: weight buffer %zu (first %zu), cache buffer %zu (first %zu)\n",
                    it, nw, count_w, nc, count_c);
            rc = fail("metadata of a persistent buffer grew");
        }
        if (ny != SIZE_MAX) {
            if (it >= 3 && ny > count_compute) {
                fprintf(stderr, "test-hexagon-metadata: graph %d: compute buffer %zu (largest of the first 3 graphs %zu)\n",
                        it, ny, count_compute);
                rc = fail("metadata of the compute buffer grew");
            }
            if (it < 3 && ny > count_compute) count_compute = ny;
        }
        ggml_free(ctx);
    }
    if (rc == 0)
        printf("test-hexagon-metadata: OK, %d graphs on %s: weight buffer %zu, cache buffer %zu, compute buffer %zu extras\n",
               iterations, dev_name.c_str(), count_w, count_c, count_compute);

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buf_w);
    ggml_backend_buffer_free(buf_c);
    ggml_free(ctx_w);
    ggml_free(ctx_c);
    ggml_backend_free(htp);
    ggml_backend_free(cpu);
    return rc;
}
