#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "dma-queue.h"
#include "work-queue.h"
#include "hvx-utils.h"
#include "hex-utils.h"
#include "hvx-copy.h"
#include "hvx-quant.h"
#include "kv-cache-quant.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "hex-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp/set-rows-ops.h"

#define set_rows_preamble                      \
    const uint32_t ne00 = octx->src[0]->ne[0]; \
    const uint32_t ne01 = octx->src[0]->ne[1]; \
    const uint32_t ne02 = octx->src[0]->ne[2]; \
    const uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    const uint32_t ne10 = octx->src[1]->ne[0]; \
    const uint32_t ne11 = octx->src[1]->ne[1]; \
    const uint32_t ne12 = octx->src[1]->ne[2]; \
    const uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    const uint32_t nb01 = octx->src[0]->nb[1]; \
    const uint32_t nb02 = octx->src[0]->nb[2]; \
    const uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    const uint32_t nb10 = octx->src[1]->nb[0]; \
    const uint32_t nb11 = octx->src[1]->nb[1]; \
    const uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    const uint32_t nb1 = octx->dst->nb[1];     \
    const uint32_t nb2 = octx->dst->nb[2];     \
    const uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    const uint32_t ne0 = octx->dst->ne[0];     \
    const uint32_t ne1 = octx->dst->ne[1];     \
    const uint32_t ne2 = octx->dst->ne[2];     \
    const uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    const uint32_t nr  = ne01;

struct set_rows_context {
    struct htp_ops_context * octx;
    const struct htp_set_rows_kernel_params * kparams;
    struct htp_set_rows_vtcm_layout vtcm_layout;
    uint8_t * vtcm_base;
    uint32_t task_start;
    uint32_t tasks;
    uint32_t tasks_per_thread;
};

#define SET_ROWS_THREAD_DMA_FN(TYPE_NAME, IDX_TYPE, COMPUTE_EXPR)                                                \
static void set_rows_thread_dma_##TYPE_NAME##_##IDX_TYPE(unsigned int nth, unsigned int ith, void *data) {       \
    struct set_rows_context * srctx = (struct set_rows_context *)data;                                           \
    struct htp_ops_context * octx = srctx->octx;                                                                 \
    const struct htp_set_rows_kernel_params * kparams = srctx->kparams;                                          \
    set_rows_preamble;                                                                                           \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                       \
    const uint32_t dr  = srctx->tasks_per_thread;                                                                \
    const uint32_t ir0 = srctx->task_start + dr * ith;                                                           \
    if (ir0 >= srctx->task_start + srctx->tasks) {                                                               \
        return;                                                                                                  \
    }                                                                                                            \
    const uint32_t ir1 = MIN(ir0 + dr, srctx->task_start + srctx->tasks);                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                     \
    const struct htp_set_rows_vtcm_layout * vtcm_layout = &srctx->vtcm_layout;                                   \
    uint8_t * vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, srctx->vtcm_base, vtcm_layout->off_src0) +                    \
                          ith * vtcm_layout->src0_bytes_per_thread;                                              \
    uint8_t * vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, srctx->vtcm_base, vtcm_layout->off_dst) +                     \
                          ith * vtcm_layout->dst_bytes_per_thread;                                               \
    const uint32_t src0_row_size = ne00 * sizeof(float);                                                         \
    const uint32_t dst_row_size  = htp_tensor_get_row_size(octx->dst->type, ne00);                               \
    const uint32_t nrows_per_thread = ir1 - ir0;                                                                 \
    const uint32_t total_steps = ne03 * ne02 * nrows_per_thread;                                                 \
    uint32_t pi_step = 0;                                                                                        \
    uint32_t pi02 = 0;                                                                                           \
    uint32_t pi03 = 0;                                                                                           \
    for (uint32_t step = 0, spad_idx = 0; step < total_steps && spad_idx < 2; ++step, spad_idx++) {              \
        uint32_t i = ir0 + pi_step;                                                                              \
        const dma_addr_t src0_data = octx->src[0]->data + i*nb01 + pi02*nb02 + pi03*nb03;                        \
        dma_queue_push(dma_q,                                                                                    \
                       dma_make_data(octx->dst->data,                                                            \
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_size),                     \
                       dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 0);                          \
        dma_queue_push(dma_q,                                                                                    \
                       dma_make_data(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_size,                    \
                                     src0_data),                                                                 \
                       vtcm_layout->src0_spad_half_size, src0_row_size, src0_row_size, 1);                       \
        pi_step++;                                                                                               \
        if (pi_step == nrows_per_thread) {                                                                       \
            pi_step = 0;                                                                                         \
            pi02++;                                                                                              \
            if (pi02 == ne02) {                                                                                  \
                pi02 = 0;                                                                                        \
                pi03++;                                                                                          \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    uint32_t ci_step = 0;                                                                                        \
    uint32_t ci02 = 0;                                                                                           \
    uint32_t ci03 = 0;                                                                                           \
    uint32_t ci11_base = 0;                                                                                      \
    uint32_t ci12_base = 0;                                                                                      \
    for (uint32_t step = 0; step < total_steps; ++step) {                                                        \
        void * dst_spad = (void *) dma_queue_pop(dma_q).src;                                                     \
        void * src_spad = (void *) dma_queue_pop(dma_q).dst;                                                     \
        uint32_t i = ir0 + ci_step;                                                                              \
        const uintptr_t src1_addr = octx->src[1]->data + i*nb10 + ci11_base*nb11 + ci12_base*nb12;               \
        const IDX_TYPE i1 = *(const IDX_TYPE *)src1_addr;                                                        \
        const bool valid_i1 = ((uint64_t)i1 < (uint64_t)ne1);                                                    \
        const uint32_t target_i1 = (uint32_t)i1;                                                                 \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, step);                                                 \
        if (valid_i1) {                                                                                          \
            COMPUTE_EXPR;                                                                                        \
        }                                                                                                        \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, step);                                                  \
        if (valid_i1) {                                                                                          \
            const dma_addr_t dst_data = octx->dst->data + target_i1*nb1 + ci02*nb2 + ci03*nb3;                   \
            dma_queue_push(dma_q,                                                                                \
                           dma_make_data(dst_data, dst_spad),                                                    \
                           dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 1);                      \
        } else {                                                                                                 \
            dma_queue_push(dma_q,                                                                                \
                           dma_make_data(octx->dst->data, dst_spad),                                             \
                           dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 0);                      \
        }                                                                                                        \
        const uint32_t next_step = step + 2;                                                                     \
        if (next_step < total_steps) {                                                                           \
            uint32_t ni = ir0 + pi_step;                                                                         \
            const dma_addr_t psrc0_data = octx->src[0]->data + ni*nb01 + pi02*nb02 + pi03*nb03;                  \
            dma_queue_push(dma_q,                                                                                \
                           dma_make_data(src_spad, psrc0_data),                                                  \
                           vtcm_layout->src0_spad_half_size, src0_row_size, src0_row_size, 1);                   \
            pi_step++;                                                                                           \
            if (pi_step == nrows_per_thread) {                                                                   \
                pi_step = 0;                                                                                     \
                pi02++;                                                                                          \
                if (pi02 == ne02) {                                                                              \
                    pi02 = 0;                                                                                    \
                    pi03++;                                                                                      \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
        ci_step++;                                                                                               \
        if (ci_step == nrows_per_thread) {                                                                       \
            ci_step = 0;                                                                                         \
            ci02++;                                                                                              \
            ci11_base++;                                                                                         \
            if (ci11_base == ne11) {                                                                             \
                ci11_base = 0;                                                                                   \
            }                                                                                                    \
            if (ci02 == ne02) {                                                                                  \
                ci02 = 0;                                                                                        \
                ci03++;                                                                                          \
                ci12_base++;                                                                                     \
                if (ci12_base == ne12) {                                                                         \
                    ci12_base = 0;                                                                               \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    dma_queue_flush(dma_q);                                                                                      \
}

SET_ROWS_THREAD_DMA_FN(f32,  int32_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (const uint8_t *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(f32,  int64_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (const uint8_t *)src_spad, ne00); })

SET_ROWS_THREAD_DMA_FN(f16,  int32_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (const uint8_t *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(f16,  int64_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (const uint8_t *)src_spad, ne00); })

SET_ROWS_THREAD_DMA_FN(q8_0, int32_t, { hvx_quantize_row_q8_0_f32(dst_spad, (const float *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(q8_0, int64_t, { hvx_quantize_row_q8_0_f32(dst_spad, (const float *)src_spad, ne00); })

SET_ROWS_THREAD_DMA_FN(q4_0, int32_t, { kv_quantize_row(dst_spad, (const float *)src_spad, ne00, HTP_TYPE_Q4_0); })
SET_ROWS_THREAD_DMA_FN(q4_0, int64_t, { kv_quantize_row(dst_spad, (const float *)src_spad, ne00, HTP_TYPE_Q4_0); })
SET_ROWS_THREAD_DMA_FN(q4_1, int32_t, { kv_quantize_row(dst_spad, (const float *)src_spad, ne00, HTP_TYPE_Q4_1); })
SET_ROWS_THREAD_DMA_FN(q4_1, int64_t, { kv_quantize_row(dst_spad, (const float *)src_spad, ne00, HTP_TYPE_Q4_1); })
SET_ROWS_THREAD_DMA_FN(q5_0, int32_t, { kv_quantize_row(dst_spad, (const float *)src_spad, ne00, HTP_TYPE_Q5_0); })
SET_ROWS_THREAD_DMA_FN(q5_0, int64_t, { kv_quantize_row(dst_spad, (const float *)src_spad, ne00, HTP_TYPE_Q5_0); })
SET_ROWS_THREAD_DMA_FN(q5_1, int32_t, { kv_quantize_row(dst_spad, (const float *)src_spad, ne00, HTP_TYPE_Q5_1); })
SET_ROWS_THREAD_DMA_FN(q5_1, int64_t, { kv_quantize_row(dst_spad, (const float *)src_spad, ne00, HTP_TYPE_Q5_1); })
SET_ROWS_THREAD_DMA_FN(f8_e4m3, int32_t, { kv_quantize_row(dst_spad, (const float *)src_spad, ne00, HTP_TYPE_F8_E4M3); })
SET_ROWS_THREAD_DMA_FN(f8_e4m3, int64_t, { kv_quantize_row(dst_spad, (const float *)src_spad, ne00, HTP_TYPE_F8_E4M3); })
SET_ROWS_THREAD_DMA_FN(iq4_nl, int32_t, { kv_quantize_row(dst_spad, (const float *)src_spad, ne00, HTP_TYPE_IQ4_NL); })
SET_ROWS_THREAD_DMA_FN(iq4_nl, int64_t, { kv_quantize_row(dst_spad, (const float *)src_spad, ne00, HTP_TYPE_IQ4_NL); })

// The rows that the op writes, as one address range. The output is the whole K or V cache while a
// decode writes one row of it, and the next op to read the cache flushes what is marked dirty.
#define SET_ROWS_RANGE_MAX_INDICES 4096

bool op_set_rows_written_range(const struct htp_ops_context * octx, uint32_t * start, uint32_t * end) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * idx  = octx->src[1];
    const struct htp_tensor * dst  = octx->dsts[0];
    if (!src0 || !idx || !dst || htp_tensor_is_extended(idx) || htp_tensor_is_extended(dst)) {
        return false;
    }
    // Small caches do not amortize the index scan and partial-range bookkeeping.
    // Keep whole-output tracking there, including the short-context decode path.
    if (dst->size < HTP_DCACHE_PARTIAL_MIN_BYTES) {
        return false;
    }
    if (idx->type != HTP_TYPE_I32 && idx->type != HTP_TYPE_I64) {
        return false;
    }
    const uint64_t n_idx = (uint64_t) idx->ne[0] * idx->ne[1] * idx->ne[2];
    if (n_idx == 0 || n_idx > SET_ROWS_RANGE_MAX_INDICES || src0->ne[2] == 0 || src0->ne[3] == 0) {
        return false;
    }

    const uint32_t ne1 = dst->ne[1];
    uint32_t lo = UINT32_MAX, hi = 0;
    for (uint32_t i2 = 0; i2 < idx->ne[2]; i2++) {
        for (uint32_t i1 = 0; i1 < idx->ne[1]; i1++) {
            const uint8_t * p = (const uint8_t *) (uintptr_t) idx->data + (size_t) i1 * idx->nb[1] + (size_t) i2 * idx->nb[2];
            for (uint32_t i0 = 0; i0 < idx->ne[0]; i0++, p += idx->nb[0]) {
                const uint64_t r = idx->type == HTP_TYPE_I32 ? (uint64_t) (int64_t) * (const int32_t *) p
                                                             : (uint64_t) * (const int64_t *) p;
                if (r < ne1) {  // the op skips the other rows
                    lo = MIN(lo, (uint32_t) r);
                    hi = MAX(hi, (uint32_t) r);
                }
            }
        }
    }
    if (lo > hi) {  // nothing is written
        *start = *end = 0;
        return true;
    }

    const uint64_t row_size = htp_tensor_get_row_size(dst->type, src0->ne[0]);
    const uint64_t first    = dst->data + (uint64_t) lo * dst->nb[1];
    const uint64_t last     = dst->data + (uint64_t) hi * dst->nb[1] + (uint64_t) (src0->ne[2] - 1) * dst->nb[2] +
                              (uint64_t) (src0->ne[3] - 1) * dst->nb[3] + row_size;
    if (last > dst->data + dst->size || last > UINT32_MAX) {
        return false;
    }
    *start = (uint32_t) first;
    *end   = (uint32_t) last;
    return true;
}

int op_set_rows(struct htp_ops_context * octx) {
    const struct htp_set_rows_kernel_params * kparams = (const struct htp_set_rows_kernel_params *)octx->kernel_params;
    set_rows_preamble;

    if (htp_tensor_is_extended(octx->src[1])) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_tensor * dst = octx->dst;
    const uint32_t total_tasks    = kparams->total_tasks;

    uint32_t task_start = 0;
    uint32_t tasks      = total_tasks;

    if (octx->ctx->mdev.count > 1) {
        const bool can_split = htp_tensor_mdev_data_aligned(dst) && (dst->nb[1] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0 && !htp_tensor_is_permuted(dst);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_tasks, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        task_start = range.start;
        tasks      = range.count;
    }

    if (tasks == 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_threads(octx, (uint32_t) kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t n_threads = octx->n_threads;

    // l2fetch the src1 (indices) tensor in the main thread
    hex_l2fetch_block((const void *)octx->src[1]->data, octx->src[1]->ne[3] * octx->src[1]->nb[3]);

    struct set_rows_context srctx;
    srctx.octx = octx;
    srctx.kparams = kparams;
    srctx.task_start = task_start;
    srctx.tasks = tasks;
    srctx.tasks_per_thread = fastdiv(tasks + n_threads - 1, &octx->n_threads_div);

    htp_set_rows_vtcm_layout_build(&srctx.vtcm_layout, octx->dst->type, ne00, n_threads);
    srctx.vtcm_base = (uint8_t *)octx->ctx->vtcm_base;

    work_queue_func_t q_func = NULL;
    const bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    switch (octx->dst->type) {
        case HTP_TYPE_F32:  q_func = is_i32 ? set_rows_thread_dma_f32_int32_t  : set_rows_thread_dma_f32_int64_t;  break;
        case HTP_TYPE_F16:  q_func = is_i32 ? set_rows_thread_dma_f16_int32_t  : set_rows_thread_dma_f16_int64_t;  break;
        case HTP_TYPE_Q8_0: q_func = is_i32 ? set_rows_thread_dma_q8_0_int32_t : set_rows_thread_dma_q8_0_int64_t; break;
        case HTP_TYPE_Q4_0: q_func = is_i32 ? set_rows_thread_dma_q4_0_int32_t : set_rows_thread_dma_q4_0_int64_t; break;
        case HTP_TYPE_Q4_1: q_func = is_i32 ? set_rows_thread_dma_q4_1_int32_t : set_rows_thread_dma_q4_1_int64_t; break;
        case HTP_TYPE_Q5_0: q_func = is_i32 ? set_rows_thread_dma_q5_0_int32_t : set_rows_thread_dma_q5_0_int64_t; break;
        case HTP_TYPE_Q5_1: q_func = is_i32 ? set_rows_thread_dma_q5_1_int32_t : set_rows_thread_dma_q5_1_int64_t; break;
        case HTP_TYPE_F8_E4M3: q_func = is_i32 ? set_rows_thread_dma_f8_e4m3_int32_t : set_rows_thread_dma_f8_e4m3_int64_t; break;
        case HTP_TYPE_IQ4_NL: q_func = is_i32 ? set_rows_thread_dma_iq4_nl_int32_t : set_rows_thread_dma_iq4_nl_int64_t; break;
        default:            return HTP_STATUS_NO_SUPPORT;
    }

    FARF(HIGH, "set-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         srctx.vtcm_layout.src0_bytes_per_thread * n_threads,
         srctx.vtcm_layout.dst_bytes_per_thread  * n_threads,
         n_threads);

#ifdef HTP_YIELD_IN_OPS
    // The transposed V cache without flash attention takes one-element rows (a 512-token ubatch
    // of 1024 values is 524288 rows, 42 ms on v73), so rows run in rounds of at most 32768 rows
    // and about 8 MiB of output, and VTCM can be given away between rounds when a release is
    // requested. Each round stages its own rows.
    const size_t   row_bytes = MAX((size_t) htp_tensor_get_row_size(octx->dst->type, ne00) * ne02 * ne03, (size_t) 1);
    size_t         max_rows  = MAX((size_t) 1, MIN((size_t) 32768 / ((size_t) ne02 * ne03), (8u << 20) / row_bytes));
    // The Q4/Q5/IQ4_NL/FP8 writers do more work per output byte than Q8_0, and a block that needs the scalar
    // reference costs about 4000 cycles: limit each round to 1 MiB of input as well (about 5 ms if every
    // block took the reference, well under 1 ms otherwise), before joining workers and yielding.
    if (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16 && dst->type != HTP_TYPE_Q8_0) {
        max_rows = MIN(max_rows, MAX((size_t) 1, (1u << 20) / ((size_t) ne00 * 4 * ne02 * ne03)));
    }
    const uint32_t round     = MAX(n_threads, (uint32_t) MIN((size_t) tasks, max_rows) / n_threads * n_threads);
    if (round < tasks) {
        for (uint32_t done = 0; done < tasks;) {
            const uint32_t n = MIN(round, tasks - done);
            srctx.task_start       = task_start + done;
            srctx.tasks            = n;
            srctx.tasks_per_thread = fastdiv(n + n_threads - 1, &octx->n_threads_div);
            work_queue_run(octx->ctx->work_queue, q_func, &srctx, n_threads);
            done += n;
            if (done < tasks && atomic_load(&octx->ctx->vtcm_needs_release)) {
                htp_vtcm_yield(octx->ctx);
            }
        }
        return HTP_STATUS_OK;
    }
#endif
    work_queue_run(octx->ctx->work_queue, q_func, &srctx, n_threads);

    return HTP_STATUS_OK;
}
