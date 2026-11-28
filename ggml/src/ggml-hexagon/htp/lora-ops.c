#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <HAP_compute_res.h>
#include <qurt_memory.h>

#include <assert.h>
#include <math.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "dma-queue.h"
#include "hvx-utils.h"
#include "hvx-dump.h"
#include "hvx-arith.h"
#include "hvx-reduce.h"
#include "hvx-glu.h"
#include "hvx-copy.h"
#include "hvx-norm.h"
#include "hvx-scale.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "lora-ops.h"

// LUT tables required by hvx-mm-kernels-tiled.h (kept file-local so matmul-ops.c stays untouched)
static const uint8_t __attribute__((aligned(128))) expand_x32_e8m0[128] = {
    0x00, 0x00, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x02, 0x00, 0x08, 0x08, 0x01, 0x02, 0x00, 0x04, 0x04, 0x00, 0x00,
    0x00, 0x11, 0x10, 0x10, 0x10, 0x02, 0x00, 0x04, 0x00, 0x01, 0x02, 0x08, 0x08, 0x08, 0x08, 0x00, 0x00, 0x01, 0x04,
    0x00, 0x00, 0x22, 0x20, 0x20, 0x20, 0x21, 0x22, 0x20, 0x24, 0x04, 0x00, 0x00, 0x00, 0x09, 0x08, 0x00, 0x00, 0x02,
    0x00, 0x04, 0x00, 0x11, 0x12, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x01, 0x04, 0x00, 0x00, 0x02, 0x00, 0x08, 0x08,
    0x01, 0x02, 0x00, 0x04, 0x44, 0x40, 0x40, 0x40, 0x41, 0x40, 0x40, 0x40, 0x42, 0x40, 0x44, 0x40, 0x41, 0x42, 0x48,
    0x48, 0x08, 0x08, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x12, 0x10, 0x10, 0x10, 0x01, 0x02, 0x00, 0x04, 0x04, 0x00,
    0x00, 0x00, 0x09, 0x08, 0x00, 0x00, 0x22, 0x20, 0x24, 0x20, 0x21, 0x22, 0x20, 0x20,
};

static const uint8_t __attribute__((aligned(VLEN))) kvalues_iq4nl_lut[] = {
    0x81, 0, 0x98, 0, 0xAD, 0, 0xBF, 0, 0xCF, 0, 0xDD, 0, 0xEA, 0, 0xF6, 0, 0x01, 0, 0x0D, 0, 0x19, 0, 0x26, 0,
    0x35, 0, 0x45, 0, 0x59, 0, 0x71, 0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
};

static const uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[] = {
    0,    0, 1,    0, 2,    0, 3, 0, 4, 0, 6, 0, 8, 0, 12, 0, 0, 0, 0xff, 0, 0xfe, 0, 0xfd, 0, 0xfc, 0,
    0xfa, 0, 0xf8, 0, 0xf4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0,
};

#include "iq2xxs-tables.h"

static const uint8_t * iq2xxs_vtcm_lut;

static inline void iq2xxs_lut_copy(uint8_t * restrict lut) {
    for (uint32_t i = 0; i < HTP_IQ2XXS_LUT_BYTES / VLEN; i++) {
        hvx_vmem(lut + i * VLEN) = hvx_vmem(htp_iq2xxs_lut + i * VLEN);
    }
    iq2xxs_vtcm_lut = lut;
}

#include "hvx-mm-kernels-tiled.h"
#include "hvx-mm-kernels-float.h"
#include "hmx-mm-kernels-tiled-fp.h"

// ============================================================================
// Shared Kernel Dispatch & HMX Helpers (Replicated locally to leave matmul-ops.c untouched)
// ============================================================================

typedef void (*lora_vec_dot_32x1_t)(const uint32_t n, void * restrict s, const void * restrict vx,
                                    const void * restrict vy, uint32_t valid_rows, const void * restrict sz,
                                    hvx_tiled_store_row_fn_t store_row);
typedef void (*lora_vec_dot_32x2_t)(const uint32_t n, void * restrict s0, void * restrict s1,
                                    const void * restrict vx, const void * restrict vy0, const void * restrict vy1,
                                    uint32_t valid_rows, const void * restrict sz0, const void * restrict sz1,
                                    hvx_tiled_store_row_fn_t store_row);

// LoRA ops write outputs (and in HMX mode read base_out/norm_w) via direct cached HVX loads/stores
// while other ops in the same opbatch access DDR via bypass UDMA. Flush written outputs and
// invalidate directly-read inputs to keep DDR coherent across ops within a batch.
static inline void lora_dcache_flush_tensor(const struct htp_tensor * t) {
    if (t && t->data && t->size > 0) {
        qurt_mem_cache_clean((qurt_addr_t) t->data, t->size, QURT_MEM_CACHE_FLUSH_INVALIDATE, QURT_MEM_DCACHE);
    }
}

static inline void lora_dcache_invalidate_tensor(const struct htp_tensor * t) {
    if (t && t->data && t->size > 0) {
        qurt_mem_cache_clean((qurt_addr_t) t->data, t->size, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    }
}

static int lora_init_tiled_vec_dot(enum htp_data_type type, lora_vec_dot_32x1_t * out1, lora_vec_dot_32x2_t * out2) {
    switch (type) {
        case HTP_TYPE_Q4_0:    *out1 = tiled_vec_dot_q4_0_32x1;    *out2 = tiled_vec_dot_q4_0_32x2;    return 0;
        case HTP_TYPE_Q4_1:
        case HTP_TYPE_Q4_K:    *out1 = tiled_vec_dot_q4_1_32x1;    *out2 = tiled_vec_dot_q4_1_32x2;    return 0;
        case HTP_TYPE_Q8_0:    *out1 = tiled_vec_dot_q8_0_32x1;    *out2 = tiled_vec_dot_q8_0_32x2;    return 0;
        case HTP_TYPE_Q5_0:    *out1 = tiled_vec_dot_q5_0_32x1;    *out2 = tiled_vec_dot_q5_0_32x2;    return 0;
        case HTP_TYPE_Q5_1:
        case HTP_TYPE_Q5_K:    *out1 = tiled_vec_dot_q5_k_32x1;    *out2 = tiled_vec_dot_q5_k_32x2;    return 0;
        case HTP_TYPE_Q6_K:    *out1 = tiled_vec_dot_q6_k_32x1;    *out2 = tiled_vec_dot_q6_k_32x2;    return 0;
        case HTP_TYPE_Q3_K:    *out1 = tiled_vec_dot_q3_k_32x1;    *out2 = tiled_vec_dot_q3_k_32x2;    return 0;
        case HTP_TYPE_Q2_K:    *out1 = tiled_vec_dot_q2_k_32x1;    *out2 = tiled_vec_dot_q2_k_32x2;    return 0;
        case HTP_TYPE_Q2_0:    *out1 = tiled_vec_dot_q2_0_32x1;    *out2 = tiled_vec_dot_q2_0_32x2;    return 0;
        case HTP_TYPE_IQ4_NL:  *out1 = tiled_vec_dot_iq4nl_32x1;   *out2 = tiled_vec_dot_iq4nl_32x2;   return 0;
        case HTP_TYPE_IQ2_XXS: *out1 = tiled_vec_dot_iq2xxs_32x1;  *out2 = tiled_vec_dot_iq2xxs_32x2;  return 0;
        case HTP_TYPE_MXFP4:   *out1 = tiled_vec_dot_mxfp4_32x1;   *out2 = tiled_vec_dot_mxfp4_32x2;   return 0;
        case HTP_TYPE_NVFP4:   *out1 = tiled_vec_dot_nvfp4_32x1;   *out2 = tiled_vec_dot_nvfp4_32x2;   return 0;
        case HTP_TYPE_Q1_0:    *out1 = tiled_vec_dot_q1_0_32x1;    *out2 = tiled_vec_dot_q1_0_32x2;    return 0;
        case HTP_TYPE_F16:     *out1 = tiled_vec_dot_f16_32x1;     *out2 = tiled_vec_dot_f16_32x2;     return 0;
        default:               return -1;
    }
}

static inline tiled_dequant_task_fn_t lora_hmx_get_dequant_worker_fn(int weight_type, bool is_tiled) {
    switch (weight_type) {
        case HTP_TYPE_Q4_0:    return dequantize_tiled_weight_to_fp16_task_q4_0;
        case HTP_TYPE_IQ4_NL:  return dequantize_tiled_weight_to_fp16_task_iq4_nl;
        case HTP_TYPE_Q4_1:
        case HTP_TYPE_Q4_K:    return dequantize_tiled_weight_to_fp16_task_q4_1;
        case HTP_TYPE_MXFP4:   return dequantize_tiled_weight_to_fp16_task_mxfp4;
        case HTP_TYPE_NVFP4:   return dequantize_tiled_weight_to_fp16_task_nvfp4;
        case HTP_TYPE_IQ2_XXS: return dequantize_tiled_weight_to_fp16_task_iq2_xxs;
        case HTP_TYPE_Q8_0:    return dequantize_tiled_weight_to_fp16_task_q8_0;
        case HTP_TYPE_Q5_0:    return dequantize_tiled_weight_to_fp16_task_q5_0;
        case HTP_TYPE_Q5_1:
        case HTP_TYPE_Q5_K:    return dequantize_tiled_weight_to_fp16_task_q5_k;
        case HTP_TYPE_Q6_K:    return dequantize_tiled_weight_to_fp16_task_q6_k;
        case HTP_TYPE_Q3_K:    return dequantize_tiled_weight_to_fp16_task_q3_k;
        case HTP_TYPE_Q2_K:    return dequantize_tiled_weight_to_fp16_task_q2_k;
        case HTP_TYPE_Q1_0:    return dequantize_tiled_weight_to_fp16_task_q1_0;
        case HTP_TYPE_Q2_0:    return dequantize_tiled_weight_to_fp16_task_q2_0;
        case HTP_TYPE_F16:     return is_tiled ? NULL : convert_f16_weight_to_fp16_tiles_task;
        case HTP_TYPE_BF16:    return convert_bf16_weight_to_fp16_tiles_task;
        case HTP_TYPE_F32:     return quantize_f32_weight_to_fp16_tiles_task;
        default:               return NULL;
    }
}

static void lora_dequant_worker_loop(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    tiled_dequantize_state_t * state = (tiled_dequantize_state_t *) data;
    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    for (unsigned int task_id = i; task_id < (unsigned int) state->n_tasks; task_id += n) {
        int start = task_id * state->n_tiles_per_task;
        int end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);
        state->task_fn(state, start, end);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

static void lora_dequantize_tiled_weight_chunk_to_fp16_tiles(
    struct htp_context * ctx, __fp16 * vtcm_dst, const void * weight_src_ddr,
    int n_cols, int k_block, size_t row_stride, int weight_type,
    int n_k_tiles, struct fastdiv_values n_k_tiles_div,
    tiled_dequant_task_fn_t dequant_worker_fn, int n_hvx
) {
    if (!dequant_worker_fn) {
        return;
    }
    assert(k_block % HTP_MM_HMX_TILE_N_COLS == 0);
    const size_t   n_col_tiles    = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);
    const uint32_t n_weight_tiles = k_block / htp_tensor_get_tile_k(weight_type);
    const size_t   n_tot_tiles    = n_col_tiles * n_weight_tiles;
    const size_t   n_tiles_per_task = (n_hvx == 1) ? n_tot_tiles : hmx_ceil_div(n_tot_tiles, n_hvx);

    tiled_dequantize_state_t state;
    state.n_tasks           = (n_tot_tiles + n_tiles_per_task - 1) / n_tiles_per_task;
    state.n_tot_tiles       = n_tot_tiles;
    state.n_tiles_per_task  = n_tiles_per_task;
    state.dst               = vtcm_dst;
    state.src               = (const uint8_t *) weight_src_ddr;
    state.task_fn           = dequant_worker_fn;
    state.n_cols            = n_cols;
    state.k_block           = k_block;
    state.row_stride        = row_stride;
    state.weight_type       = weight_type;
    state.n_k_tiles         = n_k_tiles;
    state.n_weight_tiles    = n_weight_tiles;
    state.n_k_tiles_div     = n_k_tiles_div;
    state.tile_size         = htp_tensor_get_tile_size(weight_type);
    state.aligned_tile_size = htp_tensor_get_tile_padded_size(weight_type);

    work_queue_run(ctx->hvx_queue, lora_dequant_worker_loop, &state, hex_smin((int) state.n_tasks, n_hvx));
}

// --- HMX Activation Staging (F32/F16 + Optional Fused RMS_NORM+MUL) ---

typedef struct {
    __fp16 *               dst;
    dma_addr_t             act_dma_addr;
    float *                vtcm_f32_act;
    uint32_t               n_tasks;
    uint32_t               n_tot_chunks;
    uint32_t               n_chunks_per_task;
    uint32_t               k_block;
    uint32_t               k_stride;
    uint32_t               k_valid;
    size_t                 vtcm_f32_act_bytes_per_hvx;
    uint32_t               dma_step_rows;
    uint32_t               dma_step_rows_shift;
    uint32_t               act_elem_size;
    const float *          norm_w;
    float                  norm_eps;
    float                  norm_scale;
} lora_act_task_state_t;

typedef struct {
    __fp16 *              dst;
    dma_addr_t            act_dma_addr;
    float *               vtcm_f32_act;
    uint32_t              n_rows;
    uint32_t              k_block;
    uint32_t              k_stride;
    uint32_t              k_valid;
    struct fastdiv_values n_hvx_div;
    size_t                vtcm_f32_act_bytes;
    uint32_t              dma_step_rows;
    uint32_t              dma_step_rows_shift;
} lora_act_col_state_t;

static void lora_transfer_act_col_worker(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    (void) n;
    lora_act_col_state_t * st = (lora_act_col_state_t *) data;
    struct htp_thread_trace * tr = th->trace;

    const uint32_t n_blocks = st->k_block / 32;
    const uint32_t c_first  = fastdiv(n_blocks * i, &st->n_hvx_div) * 32;
    const uint32_t c_last   = fastdiv(n_blocks * (i + 1), &st->n_hvx_div) * 32;
    const uint32_t c_len    = c_last - c_first;
    if (c_len == 0) {
        return;
    }

    uint32_t k_chunk_valid = 0;
    if (st->k_valid > c_first) {
        k_chunk_valid = hex_smin(st->k_valid, c_last) - c_first;
    }

    const size_t thr_bytes = hex_align_down(fastdiv(st->vtcm_f32_act_bytes, &st->n_hvx_div), 128);
    float * thr_act        = (float *) ((char *) st->vtcm_f32_act + i * thr_bytes);
    const uint32_t R       = st->dma_step_rows;
    const uint32_t n_steps = hex_align_up(st->n_rows, HTP_MM_HMX_TILE_N_ROWS) >> st->dma_step_rows_shift;
    dma_queue * dma_q      = th->dma_q;

    if (n_steps > 0 && st->n_rows > 0) {
        dma_queue_push(dma_q, dma_make_data(thr_act, st->act_dma_addr + (size_t) c_first * sizeof(float)),
                       c_len * sizeof(float), st->k_stride * sizeof(float), k_chunk_valid * sizeof(float),
                       hex_smin(st->n_rows, R));
    }
    if (n_steps > 1 && R < st->n_rows) {
        dma_queue_push(dma_q,
                       dma_make_data(thr_act + R * c_len,
                                     st->act_dma_addr + ((size_t) R * st->k_stride + c_first) * sizeof(float)),
                       c_len * sizeof(float), st->k_stride * sizeof(float), k_chunk_valid * sizeof(float),
                       hex_smin(st->n_rows - R, R));
    }
    for (uint32_t s = 0; s < n_steps; ++s) {
        const uint32_t r = s << st->dma_step_rows_shift;
        float * curr_buf = thr_act;
        if (r < st->n_rows) {
            curr_buf = (float *) dma_queue_pop(dma_q).dst;
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        for (uint32_t p = 0; p < (R >> 1); ++p) {
            const uint32_t row_idx  = r + (p << 1);
            float *        pair_buf = curr_buf + (p << 1) * c_len;
            transfer_activation_row_pair_fp32_to_fp16_col_chunk(
                st->dst, pair_buf, pair_buf + c_len, row_idx, st->k_block, c_first, c_len, k_chunk_valid,
                row_idx < st->n_rows, (row_idx + 1) < st->n_rows);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        const uint32_t next_r = (s + 2) << st->dma_step_rows_shift;
        if (next_r < st->n_rows) {
            dma_queue_push(dma_q,
                           dma_make_data(curr_buf,
                                         st->act_dma_addr + ((size_t) next_r * st->k_stride + c_first) * sizeof(float)),
                           c_len * sizeof(float), st->k_stride * sizeof(float), k_chunk_valid * sizeof(float),
                           hex_smin(st->n_rows - next_r, R));
        }
    }
}

static void lora_transfer_act_worker(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    lora_act_task_state_t * st = (lora_act_task_state_t *) data;
    struct htp_thread_trace * tr = th->trace;

    for (unsigned int task_id = i; task_id < st->n_tasks; task_id += n) {
        const uint32_t chunk_idx    = task_id * st->n_chunks_per_task;
        const uint32_t n_rows       = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);
        __fp16 *       dst          = st->dst + chunk_idx * st->k_block;
        const dma_addr_t act_addr   = st->act_dma_addr + (size_t) chunk_idx * st->k_stride * st->act_elem_size;
        uint8_t *      thread_act   = (uint8_t *) st->vtcm_f32_act + i * st->vtcm_f32_act_bytes_per_hvx;
        const uint32_t R            = st->dma_step_rows;
        const uint32_t n_steps      = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS) >> st->dma_step_rows_shift;
        const bool     act_is_f16   = (st->act_elem_size == sizeof(__fp16));
        const size_t   row_bytes    = (size_t) st->k_block * st->act_elem_size;
        const size_t   src_stride   = (size_t) st->k_stride * st->act_elem_size;
        const size_t   width_bytes  = (size_t) st->k_valid * st->act_elem_size;
        dma_queue *    dma_q        = th->dma_q;

        if (n_steps > 0 && n_rows > 0) {
            dma_queue_push(dma_q, dma_make_data(thread_act, act_addr), row_bytes, src_stride, width_bytes,
                           hex_smin(n_rows, R));
        }
        if (n_steps > 1 && R < n_rows) {
            dma_queue_push(dma_q, dma_make_data(thread_act + R * row_bytes, act_addr + R * src_stride), row_bytes,
                           src_stride, width_bytes, hex_smin(n_rows - R, R));
        }
        for (uint32_t s = 0; s < n_steps; ++s) {
            const uint32_t r = s << st->dma_step_rows_shift;
            uint8_t * curr_buf = thread_act;
            if (r < n_rows) {
                curr_buf = (uint8_t *) dma_queue_pop(dma_q).dst;
            }
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
            if (act_is_f16) {
                for (uint32_t p = 0; p < (R >> 1); ++p) {
                    const uint32_t row_idx  = r + (p << 1);
                    const __fp16 * pair_buf = (const __fp16 *) (curr_buf + (p << 1) * row_bytes);
                    transfer_activation_row_pair_f16_to_f16(dst, pair_buf, pair_buf + st->k_block, row_idx, st->k_block,
                                                            st->k_valid, row_idx < n_rows, (row_idx + 1) < n_rows);
                }
            } else {
                for (uint32_t p = 0; p < (R >> 1); ++p) {
                    const uint32_t row_idx  = r + (p << 1);
                    const float *  pair_buf = (const float *) (curr_buf + (p << 1) * row_bytes);
                    const bool     r0_valid = row_idx < n_rows;
                    const bool     r1_valid = (row_idx + 1) < n_rows;
                    if (st->norm_w) {
                        uint8_t * row0 = curr_buf + (p << 1) * row_bytes;
                        if (r0_valid) {
                            hvx_fast_rms_norm_mul_f32(row0, (const uint8_t *) st->norm_w, row0, st->k_valid,
                                                      st->norm_eps, st->norm_scale);
                        }
                        if (r1_valid) {
                            hvx_fast_rms_norm_mul_f32(row0 + row_bytes, (const uint8_t *) st->norm_w, row0 + row_bytes,
                                                      st->k_valid, st->norm_eps, st->norm_scale);
                        }
                    }
                    transfer_activation_row_pair_fp32_to_fp16(dst, pair_buf, pair_buf + st->k_block, row_idx, st->k_block,
                                                             st->k_valid, r0_valid, r1_valid);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

            const uint32_t next_r = (s + 2) << st->dma_step_rows_shift;
            if (next_r < n_rows) {
                dma_queue_push(dma_q, dma_make_data(curr_buf, act_addr + next_r * src_stride), row_bytes, src_stride,
                               width_bytes, hex_smin(n_rows - next_r, R));
            }
        }
    }
}

static void lora_transfer_activation_chunk_threaded(
    struct htp_context * ctx, __fp16 * dst, dma_addr_t act_dma_addr, int n_rows, int k_block, int k_stride,
    int n_hvx, const struct fastdiv_values * act_hvx_div, const struct fastdiv_values * k_div, int k_valid,
    float * vtcm_f32_act, size_t vtcm_f32_act_bytes, uint32_t act_elem_size,
    const float * norm_w, float norm_eps, float norm_scale
) {
    if (n_rows <= 0) {
        return;
    }
    const size_t n_tasks = (n_rows + 31) >> 5;
    if (act_elem_size == sizeof(float) && n_hvx > 1 && k_block > 32 && n_tasks < (size_t) n_hvx && !norm_w) {
        uint32_t dma_step_rows = 2, dma_step_rows_shift = 1;
        if (vtcm_f32_act_bytes > 0) {
            const size_t elems = hex_align_down(fastdiv(vtcm_f32_act_bytes, act_hvx_div), 128) / sizeof(float);
            if (fastdiv(elems / 2, k_div) >= 4) {
                dma_step_rows = 4;
                dma_step_rows_shift = 2;
            }
        }
        lora_act_col_state_t col_state = {
            .dst = dst, .act_dma_addr = act_dma_addr, .vtcm_f32_act = vtcm_f32_act,
            .n_rows = (uint32_t) n_rows, .k_block = (uint32_t) k_block, .k_stride = (uint32_t) k_stride,
            .k_valid = (uint32_t) k_valid, .n_hvx_div = *act_hvx_div,
            .vtcm_f32_act_bytes = vtcm_f32_act_bytes,
            .dma_step_rows = dma_step_rows, .dma_step_rows_shift = dma_step_rows_shift,
        };
        work_queue_run(ctx->hvx_queue, lora_transfer_act_col_worker, &col_state, n_hvx);
        return;
    }

    const size_t   n_chunks_per_task = (n_hvx == 1) ? (size_t) n_rows : 32;
    lora_act_task_state_t state;
    state.n_tasks                    = (n_hvx == 1) ? 1u : hmx_ceil_div((uint32_t) n_rows, 32u);
    state.n_tot_chunks               = (uint32_t) n_rows;
    state.n_chunks_per_task          = (uint32_t) n_chunks_per_task;
    state.dst                        = dst;
    state.act_dma_addr               = act_dma_addr;
    state.k_block                    = (uint32_t) k_block;
    state.k_stride                   = (uint32_t) k_stride;
    state.k_valid                    = (uint32_t) k_valid;
    state.vtcm_f32_act               = vtcm_f32_act;
    state.act_elem_size              = act_elem_size;
    state.norm_w                     = norm_w;
    state.norm_eps                   = norm_eps;
    state.norm_scale                 = norm_scale;
    state.vtcm_f32_act_bytes_per_hvx = hex_align_down(fastdiv(vtcm_f32_act_bytes, act_hvx_div), 128);

    uint32_t dma_step_rows = 2, dma_step_rows_shift = 1;
    if (state.vtcm_f32_act_bytes_per_hvx > 0) {
        const size_t elems = state.vtcm_f32_act_bytes_per_hvx / sizeof(float);
        if (fastdiv(elems / 2, k_div) >= 4) {
            dma_step_rows = 4;
            dma_step_rows_shift = 2;
        }
    }
    state.dma_step_rows       = dma_step_rows;
    state.dma_step_rows_shift = dma_step_rows_shift;

    work_queue_run(ctx->hvx_queue, lora_transfer_act_worker, &state, hex_smin(n_hvx, (int) state.n_tasks));
}

// --- HMX Output Transfer & GLU LUT Store ---

typedef struct {
    float *               dst;
    const __fp16 *        vtcm_src;
    const float *         src2;
    uint32_t              n_rows;
    uint32_t              n_cols;
    uint32_t              dst_stride;
    uint32_t              src2_stride;
    uint32_t              dst_cols;
    struct fastdiv_values n_hvx_div;
    bool                  writeback;
} lora_out_col_state_t;

typedef struct {
    float *        dst;
    const __fp16 * vtcm_src;
    const float *  src2;
    uint32_t       n_tasks;
    uint32_t       n_tot_chunks;
    uint32_t       n_chunks_per_task;
    uint32_t       n_cols;
    uint32_t       dst_stride;
    uint32_t       src2_stride;
    uint32_t       dst_cols;
    bool           writeback;
} lora_out_row_state_t;

static void lora_out_col_worker(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    (void) n;
    const lora_out_col_state_t * st = (const lora_out_col_state_t *) data;
    struct htp_thread_trace *    tr = th->trace;

    const uint32_t n_blocks = hmx_ceil_div(st->n_cols, 32);
    const uint32_t b_first  = fastdiv(n_blocks * i, &st->n_hvx_div);
    const uint32_t b_last   = fastdiv(n_blocks * (i + 1), &st->n_hvx_div);
    const uint32_t c_first  = b_first * 32;
    const uint32_t c_len    = b_last * 32 - c_first;
    if (c_len == 0) {
        return;
    }

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);
    const int chunk_dst_cols = (int) st->dst_cols - (int) c_first;
    if (chunk_dst_cols > 0) {
        hmx_fp16_output_col_chunk_fn_t fn = resolve_fp16_output_col_chunk_fn(sizeof(float), st->src2, sizeof(float), false);
        fn(st->dst + c_first, st->src2 ? (st->src2 + c_first) : NULL,
           st->vtcm_src + b_first * HTP_MM_HMX_TILE_N_ELMS,
           0, st->n_rows, c_len, st->n_cols, st->dst_stride, st->src2_stride,
           (uint32_t) chunk_dst_cols, sizeof(float));
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);
}

static void lora_out_row_worker(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    const lora_out_row_state_t * st = (const lora_out_row_state_t *) data;
    struct htp_thread_trace *    tr = th->trace;
    const uint32_t start_chunk_idx  = i * st->n_chunks_per_task;
    hmx_fp16_output_col_chunk_fn_t fn = resolve_fp16_output_col_chunk_fn(sizeof(float), st->src2, sizeof(float), false);

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);
    for (unsigned int task_id = i; task_id < st->n_tasks; task_id += n) {
        const uint32_t chunk_idx  = task_id * st->n_chunks_per_task;
        const uint32_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);
        fn(st->dst + chunk_idx * st->dst_stride,
           st->src2 ? (st->src2 + chunk_idx * st->src2_stride) : NULL,
           st->vtcm_src, chunk_idx, chunk_size, st->n_cols, st->n_cols, st->dst_stride, st->src2_stride,
           st->dst_cols, sizeof(float));
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);
}

static void lora_transfer_output_chunk_threaded(
    struct htp_context * ctx, float * dst, const float * src2, const __fp16 * vtcm_src,
    int n_rows, int n_cols, int dst_stride, uint32_t src2_stride, int dst_cols, int n_hvx
) {
    if (n_rows <= 0 || dst_cols <= 0) {
        return;
    }
    const uint32_t n_blocks = hmx_ceil_div((uint32_t) n_cols, 32);
    if (n_hvx > 1 && n_blocks >= (uint32_t) n_hvx) {
        lora_out_col_state_t col_state = {
            .dst = dst, .vtcm_src = vtcm_src, .src2 = src2,
            .n_rows = (uint32_t) n_rows, .n_cols = (uint32_t) n_cols,
            .dst_stride = (uint32_t) dst_stride, .src2_stride = src2_stride,
            .dst_cols = (uint32_t) dst_cols,
            .n_hvx_div = (n_hvx == (int) ctx->n_hvx) ? ctx->n_hvx_div : init_fastdiv_values(n_hvx),
            .writeback = true,
        };
        work_queue_run(ctx->hvx_queue, lora_out_col_worker, &col_state, n_hvx);
        return;
    }
    const uint32_t n_chunks_per_task = hex_align_up((n_hvx == 1) ? (uint32_t) n_rows : hmx_ceil_div(n_rows, n_hvx), 2);
    const int      actual_hvx        = (int) hmx_ceil_div((uint32_t) n_rows, n_chunks_per_task);
    lora_out_row_state_t state = {
        .dst = dst, .vtcm_src = vtcm_src, .src2 = src2,
        .n_tasks = (uint32_t) actual_hvx, .n_tot_chunks = (uint32_t) n_rows,
        .n_chunks_per_task = n_chunks_per_task, .n_cols = (uint32_t) n_cols,
        .dst_stride = (uint32_t) dst_stride, .src2_stride = src2_stride,
        .dst_cols = (uint32_t) dst_cols, .writeback = true,
    };
    work_queue_run(ctx->hvx_queue, lora_out_row_worker, &state, actual_hvx);
}

// --- GLU Store + Shared LUT Cache ---

#define LORA_GLU_LUT_BYTES     (65536 * sizeof(float))
#define LORA_GLU_LUT_TMP_BYTES (HTP_MAX_NTHREADS * 4 * VLEN)

typedef struct {
    float *               dst;
    const __fp16 *        vtcm_src;
    uint32_t              n_rows;
    uint32_t              n_cols;
    uint32_t              n_cols_total;
    uint32_t              up_tile_off;
    uint32_t              dst_stride;
    uint32_t              dst_cols;
    struct fastdiv_values n_hvx_div;
    bool                  writeback;
} lora_out_glu_state_t;

typedef struct {
    lora_out_glu_state_t st;
    const float *        lut;
    HVX_Vector *         tmp;
} lora_out_glu_lut_state_t;

static float * lora_glu_lut_master(struct htp_context * ctx, uint32_t glu) {
    const uint32_t k = glu == HTP_MM_GLU_GEGLU ? 0 : 1;
    if (!ctx->glu_lut[k]) {
        float * t = (float *) memalign(VLEN, LORA_GLU_LUT_BYTES);
        if (!t) {
            return NULL;
        }
        const HVX_Vector one = hvx_vec_splat_f16(1.0);
        uint16_t codes[64] __attribute__((aligned(VLEN)));
        for (uint32_t c = 0; c < 65536; c += 32) {
            for (uint32_t i = 0; i < 32; i++) {
                codes[2 * i]     = (uint16_t) (c + i);
                codes[2 * i + 1] = 0;
            }
            const HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(hvx_vmem(codes), one);
            const HVX_Vector     x = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
            hvx_vmem(t + c)        = k == 0 ? hvx_vec_geglu_gate_f32(x) : hvx_vec_swiglu_gate_f32(x);
        }
        ctx->glu_lut[k] = t;
    }
    return ctx->glu_lut[k];
}

static const float * lora_glu_lut_get(struct htp_context * ctx, uint32_t glu) {
    const struct htp_lora_kernel_params * kparams =
        (const struct htp_lora_kernel_params *) ctx->octx.kernel_params;
    const size_t top = hex_align_down(ctx->vtcm_size, VLEN);
    if ((size_t) kparams->vtcm_size + LORA_GLU_LUT_BYTES + LORA_GLU_LUT_TMP_BYTES > top) {
        return NULL;
    }
    float * const  lut  = (float *) (ctx->vtcm_base + top - LORA_GLU_LUT_BYTES);
    const uint32_t kind = glu == HTP_MM_GLU_GEGLU ? 1 : 2;
    if (ctx->glu_lut_kind != kind || ctx->glu_lut_epoch != ctx->vtcm_epoch ||
        ctx->glu_lut_batch != ctx->diag.batch_seq || ctx->glu_lut_op != ctx->diag.op_idx) {
        const float * master = lora_glu_lut_master(ctx, glu);
        if (!master) {
            return NULL;
        }
        hvx_copy_aa((uint8_t *) lut, (const uint8_t *) master, 65536, sizeof(float));
        ctx->glu_lut_kind  = kind;
        ctx->glu_lut_epoch = ctx->vtcm_epoch;
        ctx->glu_lut_batch = ctx->diag.batch_seq;
        ctx->glu_lut_op    = ctx->diag.op_idx;
    }
    return lut;
}

static void lora_out_glu_lut_worker(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    (void) n;
    const lora_out_glu_lut_state_t * ls = (const lora_out_glu_lut_state_t *) data;
    const lora_out_glu_state_t *     st = &ls->st;

    const uint32_t n_pairs = (st->n_rows + 1) / 2;
    const uint32_t p_first = fastdiv(n_pairs * i, &st->n_hvx_div);
    const uint32_t p_last  = fastdiv(n_pairs * (i + 1), &st->n_hvx_div);
    if (p_first >= p_last) {
        return;
    }

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, p_first);

    const size_t     tile_row_stride = hmx_ceil_div(st->n_cols_total, HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;
    const size_t     limit_c         = hex_smin(st->n_cols, st->dst_cols);
    const uint32_t   n_vec           = (uint32_t) ((limit_c + 31) / 32);
    const size_t     up_off          = (size_t) st->up_tile_off * HTP_MM_HMX_TILE_N_ELMS;
    const HVX_Vector one             = hvx_vec_splat_f16(1.0);
    const HVX_Vector even            = Q6_V_vsplat_R(0x0000ffff);
    const size_t     lut             = (size_t) ls->lut;
    HVX_Vector *     tmp             = ls->tmp + 4 * i;
    const uint32_t   vstep           = HTP_MM_HMX_TILE_N_ELMS / 64;

    for (uint32_t p = p_first; p < p_last; p++) {
        const size_t       r      = (size_t) p * 2;
        const bool         has_r1 = r + 1 < st->n_rows;
        const HVX_Vector * gp = (const HVX_Vector *) (st->vtcm_src + (r / HTP_MM_HMX_TILE_N_ROWS) * tile_row_stride)
                                + (r % HTP_MM_HMX_TILE_N_ROWS) / 2;
        const HVX_Vector * up   = gp + up_off / 64;
        float *            out0 = st->dst + r * st->dst_stride;
        float *            out1 = out0 + st->dst_stride;

        for (uint32_t b = 0; b < n_vec; b++) {
            const HVX_Vector g  = gp[b * vstep];
            const HVX_Vector o0 = Q6_Vw_vasl_VwR(Q6_V_vand_VV(g, even), 2);
            const HVX_Vector o1 = Q6_Vw_vasl_VwR(Q6_Vuw_vlsr_VuwR(g, 16), 2);
            Q6_vgather_ARMVw(&tmp[0], lut, LORA_GLU_LUT_BYTES - 1, o0);
            Q6_vgather_ARMVw(&tmp[1], lut, LORA_GLU_LUT_BYTES - 1, o1);
            const HVX_VectorPair pu = Q6_Wqf32_vmpy_VhfVhf(up[b * vstep], one);
            const HVX_Vector     v0 = hvx_vec_mul_f32_f32(tmp[0], Q6_Vsf_equals_Vqf32(Q6_V_lo_W(pu)));
            const HVX_Vector     v1 = hvx_vec_mul_f32_f32(tmp[1], Q6_Vsf_equals_Vqf32(Q6_V_hi_W(pu)));
            const size_t         nb = hex_smin(limit_c - (size_t) b * 32, 32) * sizeof(float);
            if (nb == VLEN) {
                hvx_vmemu(out0 + b * 32) = v0;
                if (has_r1) {
                    hvx_vmemu(out1 + b * 32) = v1;
                }
            } else {
                hvx_vec_store_u(out0 + b * 32, nb, v0);
                if (has_r1) {
                    hvx_vec_store_u(out1 + b * 32, nb, v1);
                }
            }
        }
        if (st->writeback) {
            hex_l2clean(out0, limit_c * sizeof(float));
            if (has_r1) {
                hex_l2clean(out1, limit_c * sizeof(float));
            }
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, p_first);
}

#define LORA_DEFINE_GLU_FALLBACK(fn_name, glu_expr)                                                               \
static void fn_name(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {                        \
    (void) n;                                                                                                     \
    const lora_out_glu_state_t * st = (const lora_out_glu_state_t *) data;                                        \
    const uint32_t n_pairs = (st->n_rows + 1) / 2;                                                                \
    const uint32_t p_first = fastdiv(n_pairs * i, &st->n_hvx_div);                                                \
    const uint32_t p_last  = fastdiv(n_pairs * (i + 1), &st->n_hvx_div);                                          \
    if (p_first >= p_last) {                                                                                      \
        return;                                                                                                   \
    }                                                                                                             \
    struct htp_thread_trace * tr = th->trace;                                                                     \
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, p_first);                                                 \
    const size_t tile_row_stride = hmx_ceil_div(st->n_cols_total, HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS; \
    const size_t limit_c = hex_smin(st->n_cols, st->dst_cols);                                                    \
    const uint32_t n_vec = (uint32_t) ((limit_c + 31) / 32);                                                      \
    const size_t up_off = (size_t) st->up_tile_off * HTP_MM_HMX_TILE_N_ELMS;                                      \
    const HVX_Vector one = hvx_vec_splat_f16(1.0);                                                                \
    const uint32_t vstep = HTP_MM_HMX_TILE_N_ELMS / 64;                                                           \
    for (uint32_t p = p_first; p < p_last; p++) {                                                                 \
        const size_t r = (size_t) p * 2;                                                                          \
        const bool has_r1 = r + 1 < st->n_rows;                                                                   \
        const HVX_Vector * gp = (const HVX_Vector *) (st->vtcm_src + (r / HTP_MM_HMX_TILE_N_ROWS) * tile_row_stride) \
                                + (r % HTP_MM_HMX_TILE_N_ROWS) / 2;                                               \
        const HVX_Vector * up = gp + up_off / 64;                                                                 \
        float * out0 = st->dst + r * st->dst_stride;                                                              \
        float * out1 = out0 + st->dst_stride;                                                                     \
        for (uint32_t b = 0; b < n_vec; b++) {                                                                    \
            const HVX_VectorPair pg = Q6_Wqf32_vmpy_VhfVhf(gp[b * vstep], one);                                   \
            const HVX_VectorPair pu = Q6_Wqf32_vmpy_VhfVhf(up[b * vstep], one);                                   \
            const HVX_Vector g0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(pg));                                             \
            const HVX_Vector g1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(pg));                                             \
            const HVX_Vector u0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(pu));                                             \
            const HVX_Vector u1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(pu));                                             \
            const HVX_Vector v0 = glu_expr(g0, u0);                                                               \
            const HVX_Vector v1 = glu_expr(g1, u1);                                                               \
            const size_t nb = hex_smin(limit_c - (size_t) b * 32, 32) * sizeof(float);                            \
            if (nb == VLEN) {                                                                                     \
                hvx_vmemu(out0 + b * 32) = v0;                                                                    \
                if (has_r1) { hvx_vmemu(out1 + b * 32) = v1; }                                                    \
            } else {                                                                                              \
                hvx_vec_store_u(out0 + b * 32, nb, v0);                                                           \
                if (has_r1) { hvx_vec_store_u(out1 + b * 32, nb, v1); }                                           \
            }                                                                                                     \
        }                                                                                                         \
        if (st->writeback) {                                                                                      \
            hex_l2clean(out0, limit_c * sizeof(float));                                                           \
            if (has_r1) { hex_l2clean(out1, limit_c * sizeof(float)); }                                           \
        }                                                                                                         \
    }                                                                                                             \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, p_first);                                                  \
}

LORA_DEFINE_GLU_FALLBACK(lora_out_geglu_worker, hvx_vec_geglu_f32)
LORA_DEFINE_GLU_FALLBACK(lora_out_swiglu_worker, hvx_vec_swiglu_f32)

static void lora_transfer_output_chunk_glu_threaded(
    struct htp_context * ctx, float * dst, const __fp16 * vtcm_src, int n_rows, int n_cols, int n_cols_total,
    int up_tile_off, int dst_stride, int dst_cols, uint32_t glu, int n_hvx
) {
    if (n_rows <= 0 || n_cols <= 0 || dst_cols <= 0) {
        return;
    }
    const uint32_t n_pairs = ((uint32_t) n_rows + 1) / 2;
    const int      n_use   = (int) hex_smin((size_t) n_hvx, (size_t) n_pairs);

    lora_out_glu_state_t st = {
        .dst          = dst,
        .vtcm_src     = vtcm_src,
        .n_rows       = (uint32_t) n_rows,
        .n_cols       = (uint32_t) n_cols,
        .n_cols_total = (uint32_t) n_cols_total,
        .up_tile_off  = (uint32_t) up_tile_off,
        .dst_stride   = (uint32_t) dst_stride,
        .dst_cols     = (uint32_t) dst_cols,
        .n_hvx_div    = (n_use == (int) ctx->n_hvx) ? ctx->n_hvx_div : init_fastdiv_values(n_use),
        .writeback    = true,
    };

    const float * lut = lora_glu_lut_get(ctx, glu);
    if (lut) {
        lora_out_glu_lut_state_t ls = { st, lut, (HVX_Vector *) ((const uint8_t *) lut - LORA_GLU_LUT_TMP_BYTES) };
        work_queue_run(ctx->hvx_queue, lora_out_glu_lut_worker, &ls, n_use);
        return;
    }
    const work_queue_func_t fn = (glu == HTP_MM_GLU_GEGLU) ? lora_out_geglu_worker : lora_out_swiglu_worker;
    work_queue_run(ctx->hvx_queue, fn, &st, n_use);
}

static inline void hvx_mm_glu_store(float * dst, const float * gate, const float * up, uint32_t n, uint32_t glu) {
    const HVX_Vector g = hvx_vmem(gate);
    const HVX_Vector u = hvx_vmem(up);
    const HVX_Vector v = (glu == HTP_MM_GLU_GEGLU) ? hvx_vec_geglu_f32(g, u) : hvx_vec_swiglu_f32(g, u);
    if (n == 32) {
        hvx_vmemu(dst) = v;
    } else {
        hvx_vec_store_u(dst, n * sizeof(float), v);
    }
}

// ============================================================================
// Section 1: Architecture-Safe HVX Building Blocks (< v79 and >= v79)
// ============================================================================

static inline void lora_convert_f32_to_f16_row(__fp16 * restrict dst_f16, const float * restrict src_f32, uint32_t k) {
    const HVX_Vector * restrict vs = (const HVX_Vector *) src_f32;
    HVX_Vector * restrict       vd = (HVX_Vector *) dst_f16;
    const uint32_t nvec64          = k / 64;
    const uint32_t rem             = k % 64;

    for (uint32_t i = 0; i < nvec64; ++i) {
        vd[i] = hvx_vec_f32_to_f16(vs[2 * i + 0], vs[2 * i + 1]);
    }
    if (rem > 0) {
        const HVX_Vector v0    = vs[2 * nvec64 + 0];
        const HVX_Vector v1    = (rem > 32) ? vs[2 * nvec64 + 1] : Q6_V_vzero();
        const HVX_Vector v_hf  = hvx_vec_f32_to_f16(v0, v1);
        vd[nvec64]             = Q6_V_vand_QV(Q6_Q_vsetq_R(rem * sizeof(__fp16)), v_hf);
    }
}

static inline void lora_copy_f16_pad_row(__fp16 * restrict dst_f16, const __fp16 * restrict src_f16, uint32_t k) {
    const uint32_t nvec64 = k / 64;
    const uint32_t rem    = k % 64;
    for (uint32_t i = 0; i < nvec64; ++i) {
        hvx_vmem(dst_f16 + i * 64) = hvx_vmemu(src_f16 + i * 64);
    }
    if (rem > 0) {
        const HVX_Vector v         = hvx_vmemu(src_f16 + nvec64 * 64);
        hvx_vmem(dst_f16 + nvec64 * 64) = Q6_V_vand_QV(Q6_Q_vsetq_R(rem * sizeof(__fp16)), v);
    }
}

static inline void lora_a_dot_f16_4x1(
    uint32_t k, float out4[4],
    const __fp16 * restrict a0, const __fp16 * restrict a1,
    const __fp16 * restrict a2, const __fp16 * restrict a3,
    const __fp16 * restrict x_f16
) {
    const uint32_t nvec = k / 64;
    const uint32_t rem  = k % 64;

    HVX_VectorPair acc0 = Q6_W_vzero();
    HVX_VectorPair acc1 = Q6_W_vzero();
    HVX_VectorPair acc2 = Q6_W_vzero();
    HVX_VectorPair acc3 = Q6_W_vzero();

    const HVX_Vector * restrict vx  = (const HVX_Vector *) x_f16;
    const HVX_Vector * restrict va0 = (const HVX_Vector *) a0;
    const HVX_Vector * restrict va1 = (const HVX_Vector *) a1;
    const HVX_Vector * restrict va2 = (const HVX_Vector *) a2;
    const HVX_Vector * restrict va3 = (const HVX_Vector *) a3;

    #pragma unroll(2)
    for (uint32_t i = 0; i < nvec; ++i) {
        const HVX_Vector x = vx[i];
        acc0 = hvx_vec_mpyacc_f32_f16(acc0, va0[i], x);
        acc1 = hvx_vec_mpyacc_f32_f16(acc1, va1[i], x);
        acc2 = hvx_vec_mpyacc_f32_f16(acc2, va2[i], x);
        acc3 = hvx_vec_mpyacc_f32_f16(acc3, va3[i], x);
    }
    if (rem > 0) {
        const HVX_VectorPred q = Q6_Q_vsetq_R(rem * sizeof(__fp16));
        const HVX_Vector     x = Q6_V_vand_QV(q, vx[nvec]);
        acc0 = hvx_vec_mpyacc_f32_f16(acc0, Q6_V_vand_QV(q, va0[nvec]), x);
        acc1 = hvx_vec_mpyacc_f32_f16(acc1, Q6_V_vand_QV(q, va1[nvec]), x);
        acc2 = hvx_vec_mpyacc_f32_f16(acc2, Q6_V_vand_QV(q, va2[nvec]), x);
        acc3 = hvx_vec_mpyacc_f32_f16(acc3, Q6_V_vand_QV(q, va3[nvec]), x);
    }

    const HVX_Vector s0 = hvx_vec_add_f32_f32(Q6_V_lo_W(acc0), Q6_V_hi_W(acc0));
    const HVX_Vector s1 = hvx_vec_add_f32_f32(Q6_V_lo_W(acc1), Q6_V_hi_W(acc1));
    const HVX_Vector s2 = hvx_vec_add_f32_f32(Q6_V_lo_W(acc2), Q6_V_hi_W(acc2));
    const HVX_Vector s3 = hvx_vec_add_f32_f32(Q6_V_lo_W(acc3), Q6_V_hi_W(acc3));
    const HVX_Vector r  = hvx_vec_reduce_sum_f32x4((HVX_Vector_x4) {{ s0, s1, s2, s3 }});
    hvx_vec_store_u(out4, 4 * sizeof(float), r);
}

// Compute dot(B[0..31], h) scaled by `scale` -> 32 FP32 values in lane order 0..31.
static inline HVX_Vector lora_b_dot_32_f16(const __fp16 * restrict b_ptr, const __fp16 * restrict h_f16,
                                           uint32_t r, float scale) {
    const HVX_Vector v_scale = hvx_vec_splat_f32(scale);
    const uint8_t * restrict bp = (const uint8_t *) b_ptr;

    if (r == 16) {
        const HVX_Vector m0  = hvx_vmem(h_f16);
        const HVX_Vector h32 = Q6_V_vand_QV(Q6_Q_vsetq_R(32), m0);
        const HVX_Vector h64 = Q6_V_vor_VV(h32, Q6_V_vror_VR(h32, 96));
        const HVX_Vector vH  = Q6_V_vor_VV(h64, Q6_V_vror_VR(h64, 64));

        HVX_Vector s[8], t[4], u[2];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const HVX_Vector vB = hvx_vmemu(bp + k * VLEN);
#if __HVX_ARCH__ < 79
            const HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(vB, vH);
            s[k] = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
#else
            const HVX_VectorPair p = Q6_Wsf_vmpy_VhfVhf(vB, vH);
            s[k] = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(p), Q6_V_hi_W(p));
#endif
        }
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const HVX_VectorPair d = Q6_W_vdeal_VVR(s[2 * k + 1], s[2 * k + 0], -4);
#if __HVX_ARCH__ < 79
            t[k] = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(d), Q6_V_hi_W(d));
#else
            t[k] = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(d), Q6_V_hi_W(d));
#endif
        }
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            const HVX_VectorPair d = Q6_W_vdeal_VVR(t[2 * k + 1], t[2 * k + 0], -4);
#if __HVX_ARCH__ < 79
            u[k] = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(d), Q6_V_hi_W(d));
#else
            u[k] = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(d), Q6_V_hi_W(d));
#endif
        }
        const HVX_VectorPair df = Q6_W_vdeal_VVR(u[1], u[0], -4);
#if __HVX_ARCH__ < 79
        const HVX_Vector sum_sf = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(df), Q6_V_hi_W(df)));
#else
        const HVX_Vector sum_sf = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(df), Q6_V_hi_W(df));
#endif
        return hvx_vec_mul_f32_f32(sum_sf, v_scale);
    }

    if (r == 8) {
        const HVX_Vector m0  = hvx_vmem(h_f16);
        const HVX_Vector h16 = Q6_V_vand_QV(Q6_Q_vsetq_R(16), m0);
        const HVX_Vector h32 = Q6_V_vor_VV(h16, Q6_V_vror_VR(h16, 112));
        const HVX_Vector h64 = Q6_V_vor_VV(h32, Q6_V_vror_VR(h32, 96));
        const HVX_Vector vH  = Q6_V_vor_VV(h64, Q6_V_vror_VR(h64, 64));

        HVX_Vector s[4], t[2];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const HVX_Vector vB = hvx_vmemu(bp + k * VLEN);
#if __HVX_ARCH__ < 79
            const HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(vB, vH);
            s[k] = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
#else
            const HVX_VectorPair p = Q6_Wsf_vmpy_VhfVhf(vB, vH);
            s[k] = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(p), Q6_V_hi_W(p));
#endif
        }
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            const HVX_VectorPair d = Q6_W_vdeal_VVR(s[2 * k + 1], s[2 * k + 0], -4);
#if __HVX_ARCH__ < 79
            t[k] = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(d), Q6_V_hi_W(d));
#else
            t[k] = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(d), Q6_V_hi_W(d));
#endif
        }
        const HVX_VectorPair df = Q6_W_vdeal_VVR(t[1], t[0], -4);
#if __HVX_ARCH__ < 79
        const HVX_Vector sum_sf = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(df), Q6_V_hi_W(df)));
#else
        const HVX_Vector sum_sf = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(df), Q6_V_hi_W(df));
#endif
        return hvx_vec_mul_f32_f32(sum_sf, v_scale);
    }

    if (r == 32) {
        const HVX_Vector m0  = hvx_vmem(h_f16);
        const HVX_Vector h64 = Q6_V_vand_QV(Q6_Q_vsetq_R(64), m0);
        const HVX_Vector vH  = Q6_V_vor_VV(h64, Q6_V_vror_VR(h64, 64));

        HVX_Vector s[16], t[8], u[4], w[2];
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            const HVX_Vector vB = hvx_vmemu(bp + k * VLEN);
#if __HVX_ARCH__ < 79
            const HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(vB, vH);
            s[k] = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
#else
            const HVX_VectorPair p = Q6_Wsf_vmpy_VhfVhf(vB, vH);
            s[k] = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(p), Q6_V_hi_W(p));
#endif
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const HVX_VectorPair d = Q6_W_vdeal_VVR(s[2 * k + 1], s[2 * k + 0], -4);
#if __HVX_ARCH__ < 79
            t[k] = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(d), Q6_V_hi_W(d));
#else
            t[k] = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(d), Q6_V_hi_W(d));
#endif
        }
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const HVX_VectorPair d = Q6_W_vdeal_VVR(t[2 * k + 1], t[2 * k + 0], -4);
#if __HVX_ARCH__ < 79
            u[k] = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(d), Q6_V_hi_W(d));
#else
            u[k] = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(d), Q6_V_hi_W(d));
#endif
        }
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            const HVX_VectorPair d = Q6_W_vdeal_VVR(u[2 * k + 1], u[2 * k + 0], -4);
#if __HVX_ARCH__ < 79
            w[k] = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(d), Q6_V_hi_W(d));
#else
            w[k] = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(d), Q6_V_hi_W(d));
#endif
        }
        const HVX_VectorPair df = Q6_W_vdeal_VVR(w[1], w[0], -4);
#if __HVX_ARCH__ < 79
        const HVX_Vector sum_sf = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(df), Q6_V_hi_W(df)));
#else
        const HVX_Vector sum_sf = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(df), Q6_V_hi_W(df));
#endif
        return hvx_vec_mul_f32_f32(sum_sf, v_scale);
    }

    // r % 64 == 0 and general r >= 1: 4 rows at a time via hvx_vec_reduce_sum_f32x4
    float __attribute__((aligned(128))) tmp[32];
    const uint32_t           nvec = r / 64;
    const uint32_t           rem  = r % 64;
    const HVX_Vector * restrict vh = (const HVX_Vector *) h_f16;

    for (uint32_t g = 0; g < 8; ++g) {
        const __fp16 * b0 = b_ptr + (g * 4 + 0) * r;
        const __fp16 * b1 = b_ptr + (g * 4 + 1) * r;
        const __fp16 * b2 = b_ptr + (g * 4 + 2) * r;
        const __fp16 * b3 = b_ptr + (g * 4 + 3) * r;

        HVX_VectorPair acc0 = Q6_W_vzero(), acc1 = Q6_W_vzero(), acc2 = Q6_W_vzero(), acc3 = Q6_W_vzero();
        for (uint32_t j = 0; j < nvec; ++j) {
            const HVX_Vector hv = vh[j];
            acc0 = hvx_vec_mpyacc_f32_f16(acc0, hvx_vmemu(b0 + j * 64), hv);
            acc1 = hvx_vec_mpyacc_f32_f16(acc1, hvx_vmemu(b1 + j * 64), hv);
            acc2 = hvx_vec_mpyacc_f32_f16(acc2, hvx_vmemu(b2 + j * 64), hv);
            acc3 = hvx_vec_mpyacc_f32_f16(acc3, hvx_vmemu(b3 + j * 64), hv);
        }
        if (rem > 0) {
            const HVX_VectorPred q  = Q6_Q_vsetq_R(rem * sizeof(__fp16));
            const HVX_Vector     hv = Q6_V_vand_QV(q, vh[nvec]);
            acc0 = hvx_vec_mpyacc_f32_f16(acc0, Q6_V_vand_QV(q, hvx_vmemu(b0 + nvec * 64)), hv);
            acc1 = hvx_vec_mpyacc_f32_f16(acc1, Q6_V_vand_QV(q, hvx_vmemu(b1 + nvec * 64)), hv);
            acc2 = hvx_vec_mpyacc_f32_f16(acc2, Q6_V_vand_QV(q, hvx_vmemu(b2 + nvec * 64)), hv);
            acc3 = hvx_vec_mpyacc_f32_f16(acc3, Q6_V_vand_QV(q, hvx_vmemu(b3 + nvec * 64)), hv);
        }
        const HVX_Vector s0 = hvx_vec_add_f32_f32(Q6_V_lo_W(acc0), Q6_V_hi_W(acc0));
        const HVX_Vector s1 = hvx_vec_add_f32_f32(Q6_V_lo_W(acc1), Q6_V_hi_W(acc1));
        const HVX_Vector s2 = hvx_vec_add_f32_f32(Q6_V_lo_W(acc2), Q6_V_hi_W(acc2));
        const HVX_Vector s3 = hvx_vec_add_f32_f32(Q6_V_lo_W(acc3), Q6_V_hi_W(acc3));
        const HVX_Vector r4 = hvx_vec_reduce_sum_f32x4((HVX_Vector_x4) {{ s0, s1, s2, s3 }});
        hvx_vec_store_u(tmp + g * 4, 4 * sizeof(float), r4);
    }
    return hvx_vec_mul_f32_f32(*(const HVX_Vector *) tmp, v_scale);
}

static inline HVX_Vector lora_b_dot_32_f32(const float * restrict b_ptr, const float * restrict h_f32,
                                           uint32_t r, float scale) {
    float __attribute__((aligned(128))) tmp[32];
    const uint32_t           nvec = r / 32;
    const uint32_t           rem  = r % 32;
    const HVX_Vector * restrict vh = (const HVX_Vector *) h_f32;

    for (uint32_t g = 0; g < 8; ++g) {
        const float * b0 = b_ptr + (g * 4 + 0) * r;
        const float * b1 = b_ptr + (g * 4 + 1) * r;
        const float * b2 = b_ptr + (g * 4 + 2) * r;
        const float * b3 = b_ptr + (g * 4 + 3) * r;

        HVX_Vector s0 = Q6_V_vzero(), s1 = Q6_V_vzero(), s2 = Q6_V_vzero(), s3 = Q6_V_vzero();
        for (uint32_t j = 0; j < nvec; ++j) {
            const HVX_Vector hv = vh[j];
            s0 = hvx_vec_add_f32_f32(s0, hvx_vec_mul_f32_f32(hvx_vmemu(b0 + j * 32), hv));
            s1 = hvx_vec_add_f32_f32(s1, hvx_vec_mul_f32_f32(hvx_vmemu(b1 + j * 32), hv));
            s2 = hvx_vec_add_f32_f32(s2, hvx_vec_mul_f32_f32(hvx_vmemu(b2 + j * 32), hv));
            s3 = hvx_vec_add_f32_f32(s3, hvx_vec_mul_f32_f32(hvx_vmemu(b3 + j * 32), hv));
        }
        if (rem > 0) {
            const HVX_VectorPred q  = Q6_Q_vsetq_R(rem * sizeof(float));
            const HVX_Vector     hv = Q6_V_vand_QV(q, vh[nvec]);
            s0 = hvx_vec_add_f32_f32(s0, hvx_vec_mul_f32_f32(Q6_V_vand_QV(q, hvx_vmemu(b0 + nvec * 32)), hv));
            s1 = hvx_vec_add_f32_f32(s1, hvx_vec_mul_f32_f32(Q6_V_vand_QV(q, hvx_vmemu(b1 + nvec * 32)), hv));
            s2 = hvx_vec_add_f32_f32(s2, hvx_vec_mul_f32_f32(Q6_V_vand_QV(q, hvx_vmemu(b2 + nvec * 32)), hv));
            s3 = hvx_vec_add_f32_f32(s3, hvx_vec_mul_f32_f32(Q6_V_vand_QV(q, hvx_vmemu(b3 + nvec * 32)), hv));
        }
        const HVX_Vector r4 = hvx_vec_reduce_sum_f32x4((HVX_Vector_x4) {{ s0, s1, s2, s3 }});
        hvx_vec_store_u(tmp + g * 4, 4 * sizeof(float), r4);
    }
    return hvx_vec_mul_f32_f32(*(const HVX_Vector *) tmp, hvx_vec_splat_f32(scale));
}

static inline HVX_Vector lora_b_dot_32_bf16(const uint16_t * restrict b_ptr, const float * restrict h_f32,
                                            uint32_t r, float scale) {
    float __attribute__((aligned(128))) tmp[32];
    const uint32_t           nvec = r / 64;
    const uint32_t           rem  = r % 64;
    const HVX_Vector * restrict vh = (const HVX_Vector *) h_f32;

    for (uint32_t g = 0; g < 8; ++g) {
        const uint16_t * b0 = b_ptr + (g * 4 + 0) * r;
        const uint16_t * b1 = b_ptr + (g * 4 + 1) * r;
        const uint16_t * b2 = b_ptr + (g * 4 + 2) * r;
        const uint16_t * b3 = b_ptr + (g * 4 + 3) * r;

        HVX_Vector s0 = Q6_V_vzero(), s1 = Q6_V_vzero(), s2 = Q6_V_vzero(), s3 = Q6_V_vzero();
        for (uint32_t j = 0; j < nvec; ++j) {
            const HVX_Vector     h0 = vh[2 * j + 0];
            const HVX_Vector     h1 = vh[2 * j + 1];
            const HVX_VectorPair p0 = Q6_Ww_vunpack_Vh(hvx_vmemu(b0 + j * 64));
            const HVX_VectorPair p1 = Q6_Ww_vunpack_Vh(hvx_vmemu(b1 + j * 64));
            const HVX_VectorPair p2 = Q6_Ww_vunpack_Vh(hvx_vmemu(b2 + j * 64));
            const HVX_VectorPair p3 = Q6_Ww_vunpack_Vh(hvx_vmemu(b3 + j * 64));
            s0 = hvx_vec_add_f32_f32(s0, hvx_vec_mul_f32_f32(Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16), h0));
            s0 = hvx_vec_add_f32_f32(s0, hvx_vec_mul_f32_f32(Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16), h1));
            s1 = hvx_vec_add_f32_f32(s1, hvx_vec_mul_f32_f32(Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16), h0));
            s1 = hvx_vec_add_f32_f32(s1, hvx_vec_mul_f32_f32(Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16), h1));
            s2 = hvx_vec_add_f32_f32(s2, hvx_vec_mul_f32_f32(Q6_Vw_vasl_VwR(Q6_V_lo_W(p2), 16), h0));
            s2 = hvx_vec_add_f32_f32(s2, hvx_vec_mul_f32_f32(Q6_Vw_vasl_VwR(Q6_V_hi_W(p2), 16), h1));
            s3 = hvx_vec_add_f32_f32(s3, hvx_vec_mul_f32_f32(Q6_Vw_vasl_VwR(Q6_V_lo_W(p3), 16), h0));
            s3 = hvx_vec_add_f32_f32(s3, hvx_vec_mul_f32_f32(Q6_Vw_vasl_VwR(Q6_V_hi_W(p3), 16), h1));
        }
        if (rem > 0) {
            const HVX_VectorPred qb = Q6_Q_vsetq2_R(rem * sizeof(uint16_t));
            const HVX_VectorPair p0 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(qb, hvx_vmemu(b0 + nvec * 64)));
            const HVX_VectorPair p1 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(qb, hvx_vmemu(b1 + nvec * 64)));
            const HVX_VectorPair p2 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(qb, hvx_vmemu(b2 + nvec * 64)));
            const HVX_VectorPair p3 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(qb, hvx_vmemu(b3 + nvec * 64)));

            if (rem <= 32) {
                const HVX_VectorPred qa = Q6_Q_vsetq2_R(rem * sizeof(float));
                const HVX_Vector     h0 = Q6_V_vand_QV(qa, vh[2 * nvec + 0]);
                s0 = hvx_vec_add_f32_f32(s0, hvx_vec_mul_f32_f32(Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16), h0));
                s1 = hvx_vec_add_f32_f32(s1, hvx_vec_mul_f32_f32(Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16), h0));
                s2 = hvx_vec_add_f32_f32(s2, hvx_vec_mul_f32_f32(Q6_Vw_vasl_VwR(Q6_V_lo_W(p2), 16), h0));
                s3 = hvx_vec_add_f32_f32(s3, hvx_vec_mul_f32_f32(Q6_Vw_vasl_VwR(Q6_V_lo_W(p3), 16), h0));
            } else {
                const HVX_Vector     h0 = vh[2 * nvec + 0];
                const HVX_VectorPred qa = Q6_Q_vsetq2_R((rem - 32) * sizeof(float));
                const HVX_Vector     h1 = Q6_V_vand_QV(qa, vh[2 * nvec + 1]);
                s0 = hvx_vec_add_f32_f32(s0, hvx_vec_mul_f32_f32(Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16), h0));
                s0 = hvx_vec_add_f32_f32(s0, hvx_vec_mul_f32_f32(Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16), h1));
                s1 = hvx_vec_add_f32_f32(s1, hvx_vec_mul_f32_f32(Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16), h0));
                s1 = hvx_vec_add_f32_f32(s1, hvx_vec_mul_f32_f32(Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16), h1));
                s2 = hvx_vec_add_f32_f32(s2, hvx_vec_mul_f32_f32(Q6_Vw_vasl_VwR(Q6_V_lo_W(p2), 16), h0));
                s2 = hvx_vec_add_f32_f32(s2, hvx_vec_mul_f32_f32(Q6_Vw_vasl_VwR(Q6_V_hi_W(p2), 16), h1));
                s3 = hvx_vec_add_f32_f32(s3, hvx_vec_mul_f32_f32(Q6_Vw_vasl_VwR(Q6_V_lo_W(p3), 16), h0));
                s3 = hvx_vec_add_f32_f32(s3, hvx_vec_mul_f32_f32(Q6_Vw_vasl_VwR(Q6_V_hi_W(p3), 16), h1));
            }
        }
        const HVX_Vector r4 = hvx_vec_reduce_sum_f32x4((HVX_Vector_x4) {{ s0, s1, s2, s3 }});
        hvx_vec_store_u(tmp + g * 4, 4 * sizeof(float), r4);
    }
    return hvx_vec_mul_f32_f32(*(const HVX_Vector *) tmp, hvx_vec_splat_f32(scale));
}

static inline void lora_apply_tile_32(
    float * restrict tile_f32, uint32_t n_active,
    const uint8_t * restrict active_b_types, const uint8_t * const * restrict b_ptrs,
    const float * const * restrict h_f32_ptrs, const __fp16 * const * restrict h_f16_ptrs,
    const uint32_t * restrict r_arr, const float * restrict scales
) {
    HVX_Vector acc = hvx_vmem(tile_f32);
    for (uint32_t a = 0; a < n_active; ++a) {
        HVX_Vector d = Q6_V_vzero();
        switch (active_b_types[a]) {
            case HTP_TYPE_F16:
                d = lora_b_dot_32_f16((const __fp16 *) b_ptrs[a], h_f16_ptrs[a], r_arr[a], scales[a]);
                break;
            case HTP_TYPE_F32:
                d = lora_b_dot_32_f32((const float *) b_ptrs[a], h_f32_ptrs[a], r_arr[a], scales[a]);
                break;
            case HTP_TYPE_BF16:
                d = lora_b_dot_32_bf16((const uint16_t *) b_ptrs[a], h_f32_ptrs[a], r_arr[a], scales[a]);
                break;
            default:
                break;
        }
        acc = hvx_vec_add_f32_f32(acc, d);
    }
    hvx_vmem(tile_f32) = acc;
}

// ============================================================================
// Section 2: HVX LoRA_A Worker + Decode op_matmul_lora (M = 1..4)
// ============================================================================

static inline void lora_convert_f16_to_f32_row(float * restrict dst_f32, const __fp16 * restrict src_f16, uint32_t k) {
    const uint32_t nvec64 = k / 64;
    const uint32_t rem    = k % 64;
    for (uint32_t i = 0; i < nvec64; ++i) {
        const HVX_VectorPair p = hvx_vec_f16_to_f32(hvx_vmemu(src_f16 + i * 64));
        hvx_vmem(dst_f32 + i * 64 + 0)  = Q6_V_lo_W(p);
        hvx_vmem(dst_f32 + i * 64 + 32) = Q6_V_hi_W(p);
    }
    if (rem > 0) {
        const HVX_Vector v_hf  = Q6_V_vand_QV(Q6_Q_vsetq_R(rem * sizeof(__fp16)), hvx_vmemu(src_f16 + nvec64 * 64));
        const HVX_VectorPair p = hvx_vec_f16_to_f32(v_hf);
        hvx_vmem(dst_f32 + nvec64 * 64 + 0) = Q6_V_lo_W(p);
        if (rem > 32) {
            hvx_vmem(dst_f32 + nvec64 * 64 + 32) = Q6_V_hi_W(p);
        }
    }
}

#define HTP_LORA_MAX_HVX_M 4

typedef struct {
    struct htp_ops_context *               octx;
    const struct htp_lora_kernel_params *  kparams;
    const struct htp_tensor *              act;
    const struct htp_tensor *              a_tensors[HTP_LORA_MAX_ADAPTERS];
    const struct htp_tensor *              b_tensors[HTP_LORA_MAX_ADAPTERS];

    uint8_t *                              vtcm_act;
    uint8_t *                              vtcm_act_raw;
    uint8_t *                              vtcm_src0;
    __fp16 *                               vtcm_x_f16;
    float *                                vtcm_h_f32;
    __fp16 *                               vtcm_h_f16;
    float *                                vtcm_tile_out;
    float *                                vtcm_glu_gate;
    uint8_t *                              vtcm_scratch;

    size_t                                 vtcm_act_raw_stride;
    size_t                                 vtcm_act_stride;
    size_t                                 vtcm_src0_size_per_hvx;
    size_t                                 x_f16_row_stride;
    size_t                                 h_f32_row_stride;
    size_t                                 h_f16_row_stride;
    size_t                                 scratch_per_hvx;
    size_t                                 per_lora_b_bytes;
    size_t                                 phase1_row_bytes;
    uint32_t                               phase1_batch_rows;
    size_t                                 glu_stride;

    uint32_t                               m_offset;  // row offset in act (0 for MUL_MAT_LORA, mr for LORA_ADD)
    uint32_t                               m_rows;
    uint32_t                               k;

    // Combined quantize task state (MUL_MAT_LORA HVX only)
    bool                                   do_quant;
    bool                                   quant_is_block;
    int                                    base_wtype;
    uint32_t                               n_quant_tasks;
    uint32_t                               n_quant_rows_per_hvx;
    uint32_t                               quant_ib_first[HTP_MAX_NTHREADS];
    uint32_t                               quant_ib_last[HTP_MAX_NTHREADS];
    uint32_t                               quant_r[HTP_MAX_NTHREADS];
    uint32_t                               quant_c[HTP_MAX_NTHREADS];

    lora_vec_dot_32x1_t                    vec_dot_32x1;
    lora_vec_dot_32x2_t                    vec_dot_32x2;
} lora_hvx_ctx_t;

static void lora_run_quant_slice(const htp_thread_t * th, unsigned int ith, lora_hvx_ctx_t * lctx) {
    if (!lctx->do_quant || ith >= lctx->n_quant_tasks) {
        return;
    }
    struct htp_thread_trace * tr   = th->trace;
    const uint32_t            ne0  = lctx->k;
    const uint32_t            wtype = (uint32_t) lctx->base_wtype;
    const int                 atype = lctx->act->type;

    if (lctx->quant_is_block) {
        const uint32_t ib_first = lctx->quant_ib_first[ith];
        const uint32_t ib_last  = lctx->quant_ib_last[ith];
        if (ib_first >= ib_last) {
            return;
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ib_first);
        const uint32_t r0 = lctx->quant_r[ith];
        const uint32_t c0 = lctx->quant_c[ith];
        if (wtype == HTP_TYPE_F16) {
            const size_t d_stride = htp_mm_f16_tiled_row_size(ne0);
            if (atype == HTP_TYPE_F16) {
                quantize_f16_f16_tiled_block_kernel((const __fp16 *) lctx->vtcm_act_raw, lctx->vtcm_act, NULL, ne0,
                                                    ib_first, ib_last, lctx->vtcm_act_raw_stride, d_stride, r0, c0);
            } else {
                quantize_f32_f16_tiled_block_kernel((const float *) lctx->vtcm_act_raw, lctx->vtcm_act, NULL, ne0,
                                                    ib_first, ib_last, lctx->vtcm_act_raw_stride, d_stride, r0, c0);
            }
        } else if (wtype == HTP_TYPE_Q2_K) {
            quantize_f32_q8_1_s16_tiled_block_kernel((const float *) lctx->vtcm_act_raw, lctx->vtcm_act, NULL, ne0,
                                                     ib_first, ib_last, lctx->vtcm_act_raw_stride,
                                                     htp_mm_q8_1_tiled_row_size(ne0), r0, c0);
        } else if (htp_mm_weight_has_offset(wtype)) {
            quantize_f32_q8_1_tiled_block_kernel((const float *) lctx->vtcm_act_raw, lctx->vtcm_act, NULL, ne0,
                                                 ib_first, ib_last, lctx->vtcm_act_raw_stride,
                                                 htp_mm_q8_1_tiled_row_size(ne0), r0, c0);
        } else {
            quantize_f32_q8_0_tiled_block_kernel((const float *) lctx->vtcm_act_raw, lctx->vtcm_act, NULL, ne0,
                                                 ib_first, ib_last, lctx->vtcm_act_raw_stride,
                                                 htp_mm_q8_0_tiled_row_size(ne0), r0, c0);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ib_first);
        return;
    }

    const uint32_t ir_first = lctx->n_quant_rows_per_hvx * ith;
    if (ir_first >= lctx->m_rows) {
        return;
    }
    const uint32_t ir_last = MIN(ir_first + lctx->n_quant_rows_per_hvx, lctx->m_rows);
    const uint32_t nrows   = ir_last - ir_first;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);
    const uint8_t * src_p = lctx->vtcm_act_raw + ir_first * lctx->vtcm_act_raw_stride;
    if (wtype == HTP_TYPE_F16) {
        const size_t d_stride = htp_mm_f16_tiled_row_size(ne0);
        uint8_t *    dst_p    = lctx->vtcm_act + ir_first * d_stride;
        if (atype == HTP_TYPE_F16) {
            quantize_f16_f16_tiled_kernel(src_p, dst_p, NULL, ne0, nrows, lctx->vtcm_act_raw_stride, d_stride);
        } else {
            quantize_f32_f16_tiled_kernel(src_p, dst_p, NULL, ne0, nrows, lctx->vtcm_act_raw_stride, d_stride);
        }
    } else if (wtype == HTP_TYPE_Q2_K) {
        const size_t d_stride = htp_mm_q8_1_tiled_row_size(ne0);
        quantize_f32_q8_1_s16_tiled_kernel(src_p, lctx->vtcm_act + ir_first * d_stride, NULL, ne0, nrows,
                                           lctx->vtcm_act_raw_stride, d_stride);
    } else if (htp_mm_weight_has_offset(wtype)) {
        const size_t d_stride = htp_mm_q8_1_tiled_row_size(ne0);
        quantize_f32_q8_1_tiled_kernel(src_p, lctx->vtcm_act + ir_first * d_stride, NULL, ne0, nrows,
                                       lctx->vtcm_act_raw_stride, d_stride);
    } else {
        const size_t d_stride = htp_mm_q8_0_tiled_row_size(ne0);
        quantize_f32_q8_0_tiled_kernel(src_p, lctx->vtcm_act + ir_first * d_stride, NULL, ne0, nrows,
                                       lctx->vtcm_act_raw_stride, d_stride);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);
}

static inline size_t lora_elem_size(int type) {
    switch (type) {
        case HTP_TYPE_F32:  return sizeof(float);
        case HTP_TYPE_F16:  return sizeof(__fp16);
        case HTP_TYPE_BF16: return sizeof(uint16_t);
        default:            return 0;
    }
}

static void lora_a_compute_batch(
    const lora_hvx_ctx_t * lctx, uint32_t l, uint32_t j_start, uint32_t batch_n,
    const uint8_t * buf_a, const uint8_t * raw_f32_act
) {
    const struct htp_tensor * A       = lctx->a_tensors[l];
    const struct htp_tensor * B       = lctx->b_tensors[l];
    const uint32_t            k       = lctx->k;
    const size_t              p1_row  = lctx->phase1_row_bytes;
    const uint32_t            m_rows  = lctx->m_rows;
    const bool                need_hf = (B->type == HTP_TYPE_F16);

    for (uint32_t m = 0; m < m_rows; ++m) {
        float * restrict h_f32_row =
            lctx->vtcm_h_f32 + ((size_t) l * m_rows + m) * lctx->h_f32_row_stride + j_start;

        if (A->type == HTP_TYPE_F16) {
            const __fp16 * restrict x_hf = lctx->vtcm_x_f16 + m * lctx->x_f16_row_stride;
            uint32_t r = 0;
            for (; r + 4 <= batch_n; r += 4) {
                lora_a_dot_f16_4x1(
                    k, h_f32_row + r,
                    (const __fp16 *) (buf_a + (r + 0) * p1_row),
                    (const __fp16 *) (buf_a + (r + 1) * p1_row),
                    (const __fp16 *) (buf_a + (r + 2) * p1_row),
                    (const __fp16 *) (buf_a + (r + 3) * p1_row),
                    x_hf);
            }
            for (; r + 2 <= batch_n; r += 2) {
                flat_vec_dot_f16_f16_aa_2x1(k, h_f32_row + r, buf_a + (r + 0) * p1_row,
                                            buf_a + (r + 1) * p1_row, x_hf, hvx_flat_store_2x1_f32);
            }
            if (r < batch_n) {
                flat_vec_dot_f16_f16_aa_1x1(k, h_f32_row + r, buf_a + r * p1_row, x_hf, hvx_flat_store_1x1_f32);
            }
        } else {
            // A->type is F32 or BF16: convert VTCM row m of x_f32 if needed (for F16 x_raw fallback)
            const float * restrict x_f32 = (const float *) (raw_f32_act + m * lctx->vtcm_act_raw_stride);
            if (A->type == HTP_TYPE_F32) {
                uint32_t r = 0;
                for (; r + 2 <= batch_n; r += 2) {
                    flat_vec_dot_f32_f32_aa_2x1(k, h_f32_row + r, buf_a + (r + 0) * p1_row,
                                                buf_a + (r + 1) * p1_row, x_f32, hvx_flat_store_2x1_f32);
                }
                if (r < batch_n) {
                    flat_vec_dot_f32_f32_aa_1x1(k, h_f32_row + r, buf_a + r * p1_row, x_f32, hvx_flat_store_1x1_f32);
                }
            } else {
                uint32_t r = 0;
                for (; r + 2 <= batch_n; r += 2) {
                    flat_vec_dot_bf16_f32_aa_2x1(k, h_f32_row + r, buf_a + (r + 0) * p1_row,
                                                 buf_a + (r + 1) * p1_row, x_f32, hvx_flat_store_2x1_f32);
                }
                if (r < batch_n) {
                    flat_vec_dot_bf16_f32_aa_1x1(k, h_f32_row + r, buf_a + r * p1_row, x_f32, hvx_flat_store_1x1_f32);
                }
            }
        }

        if (need_hf) {
            __fp16 * restrict h_f16_row =
                lctx->vtcm_h_f16 + ((size_t) l * m_rows + m) * lctx->h_f16_row_stride + j_start;
            for (uint32_t r = 0; r < batch_n; ++r) {
                h_f16_row[r] = (__fp16) h_f32_row[r];
            }
        }
    }
}

static void lora_a_hvx_worker_slice(const htp_thread_t * th, unsigned int nth, unsigned int ith, lora_hvx_ctx_t * lctx) {
    const struct htp_lora_kernel_params * kp = lctx->kparams;
    const uint32_t n_loras = kp->n_loras;

    // Total rank across active adapters
    uint32_t r_prefix[HTP_LORA_MAX_ADAPTERS + 1];
    r_prefix[0] = 0;
    for (uint32_t l = 0; l < n_loras; ++l) {
        const uint32_t rl = (kp->lora_scale[l] != 0.0f) ? lctx->a_tensors[l]->ne[1] : 0u;
        r_prefix[l + 1]   = r_prefix[l] + rl;
    }
    const uint32_t r_tot = r_prefix[n_loras];
    if (r_tot == 0) {
        return;
    }

    const uint32_t g_start = (r_tot * ith) / nth;
    const uint32_t g_end   = (r_tot * (ith + 1)) / nth;
    if (g_start >= g_end) {
        return;
    }

    const uint32_t batch_rows   = lctx->phase1_batch_rows;
    const size_t   p1_row_bytes = lctx->phase1_row_bytes;
    uint8_t *      scratch_base = lctx->vtcm_scratch + ith * lctx->scratch_per_hvx;
    uint8_t *      bufs[2]      = { scratch_base, scratch_base + batch_rows * p1_row_bytes };
    dma_queue *    dma_q        = th->dma_q;

    // Both MUL_MAT_LORA and LORA_ADD ensure vtcm_act_raw holds F32 whenever any A_l is F32/BF16.
    const uint8_t * act_f32_ptr = lctx->vtcm_act_raw;

    typedef struct {
        uint32_t l;
        uint32_t j0;
        uint32_t n;
    } lora_a_batch_desc_t;

    // Pipeline batches across all adapters owned by this thread
    uint32_t cur_g = g_start;
    uint32_t cur_l = 0;
    while (cur_l < n_loras && r_prefix[cur_l + 1] <= cur_g) {
        cur_l++;
    }

    #define NEXT_BATCH(desc_out, has_out)                                                                       \
        do {                                                                                                    \
            while (cur_g < g_end && cur_l < n_loras && cur_g >= r_prefix[cur_l + 1]) {                          \
                cur_l++;                                                                                        \
            }                                                                                                   \
            if (cur_g < g_end && cur_l < n_loras) {                                                             \
                const uint32_t j0    = cur_g - r_prefix[cur_l];                                                 \
                const uint32_t avail = MIN(g_end, r_prefix[cur_l + 1]) - cur_g;                                 \
                const uint32_t n     = MIN(batch_rows, avail);                                                  \
                (desc_out).l         = cur_l;                                                                   \
                (desc_out).j0        = j0;                                                                      \
                (desc_out).n         = n;                                                                       \
                cur_g += n;                                                                                     \
                (has_out) = true;                                                                               \
            } else {                                                                                            \
                (has_out) = false;                                                                              \
            }                                                                                                   \
        } while (0)

    #define PUSH_A_BATCH(buf_idx, dsc)                                                                          \
        do {                                                                                                    \
            const struct htp_tensor * A_  = lctx->a_tensors[(dsc).l];                                           \
            const size_t              wb_ = lctx->k * lora_elem_size(A_->type);                                 \
            dma_queue_push(dma_q, dma_make_data(bufs[(buf_idx)], A_->data + (size_t) (dsc).j0 * A_->nb[1]),      \
                           p1_row_bytes, A_->nb[1], wb_, (dsc).n);                                              \
        } while (0)

    lora_a_batch_desc_t pipe[2];
    bool                has[2] = { false, false };
    uint32_t            slot   = 0;

    NEXT_BATCH(pipe[0], has[0]);
    if (has[0]) {
        PUSH_A_BATCH(0, pipe[0]);
    }
    NEXT_BATCH(pipe[1], has[1]);
    if (has[1]) {
        PUSH_A_BATCH(1, pipe[1]);
    }

    while (has[slot]) {
        const uint8_t * ready_buf = (const uint8_t *) dma_queue_pop(dma_q).dst;
        const lora_a_batch_desc_t cur = pipe[slot];
        const uint32_t next_slot = slot ^ 1u;

        lora_a_compute_batch(lctx, cur.l, cur.j0, cur.n, ready_buf, act_f32_ptr);

        NEXT_BATCH(pipe[slot], has[slot]);
        if (has[slot]) {
            PUSH_A_BATCH(slot, pipe[slot]);
        }
        slot = next_slot;
    }

    #undef NEXT_BATCH
    #undef PUSH_A_BATCH
}

static void lora_phase1_combined_worker(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    lora_hvx_ctx_t * lctx = (lora_hvx_ctx_t *) data;
    lora_run_quant_slice(th, ith, lctx);
    lora_a_hvx_worker_slice(th, nth, ith, lctx);
}

static void hvx_mm_lora_2d_tiled(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    lora_hvx_ctx_t *                       lctx    = (lora_hvx_ctx_t *) data;
    struct htp_ops_context *               octx    = lctx->octx;
    const struct htp_lora_kernel_params *  kp      = lctx->kparams;
    const uint32_t                         n_weights = kp->n_weights;
    const uint32_t                         n_prefetch = kp->n_prefetch;
    const uint32_t                         glu        = kp->glu;
    const uint32_t                         act_nrows  = lctx->m_rows;
    const uint32_t                         ne10       = lctx->k;
    struct htp_thread_trace *              tr         = th->trace;
    dma_queue *                            dma_q      = th->dma_q;

    uint8_t * restrict vtcm_w_ptr  = lctx->vtcm_src0 + lctx->vtcm_src0_size_per_hvx * ith;
    const uint8_t * restrict act_d = lctx->vtcm_act;
    float * restrict tile_out_base = lctx->vtcm_tile_out + ith * 64;
    uint8_t * restrict thr_scratch = lctx->vtcm_scratch + ith * lctx->scratch_per_hvx;

    for (uint32_t widx = 0; widx < n_weights; ++widx) {
        const struct htp_tensor * restrict src_w = octx->src[widx];
        const struct htp_tensor * restrict dst   = glu ? octx->dsts[0] : octx->dsts[widx];
        if (!src_w || !dst) {
            continue;
        }
        const bool glu_gate = (glu != 0 && widx == 0);
        const bool glu_out  = (glu != 0 && widx == 1);

        // Gather active LoRAs targeting this weight
        uint8_t                  active_l[HTP_LORA_MAX_ADAPTERS];
        uint8_t                  active_b_types[HTP_LORA_MAX_ADAPTERS];
        uint32_t                 active_r[HTP_LORA_MAX_ADAPTERS];
        size_t                   active_b_row_bytes[HTP_LORA_MAX_ADAPTERS];
        float                    active_scales[HTP_LORA_MAX_ADAPTERS];
        const struct htp_tensor * active_B[HTP_LORA_MAX_ADAPTERS];
        uint8_t *                active_b_vtcm[HTP_LORA_MAX_ADAPTERS];
        uint32_t                 n_active = 0;

        for (uint32_t l = 0; l < kp->n_loras; ++l) {
            if (kp->lora_target[l] == widx && kp->lora_scale[l] != 0.0f) {
                const struct htp_tensor * B = lctx->b_tensors[l];
                active_l[n_active]           = (uint8_t) l;
                active_B[n_active]           = B;
                active_b_types[n_active]     = (uint8_t) B->type;
                active_r[n_active]           = B->ne[0];
                active_b_row_bytes[n_active] = B->ne[0] * lora_elem_size(B->type);
                active_scales[n_active]      = kp->lora_scale[l];
                active_b_vtcm[n_active]      = thr_scratch + n_active * lctx->per_lora_b_bytes;
                n_active++;
            }
        }

        const uint32_t ne01            = src_w->ne[1];
        const uint32_t tile_size       = htp_tensor_get_tile_size(src_w->type);
        const uint32_t aligned_tile    = htp_tensor_get_tile_padded_size(src_w->type);
        const uint32_t n_k_tiles_w     = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
        const uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        const uint32_t tile_row_vtcm   = hex_align_up(n_k_tiles_w * aligned_tile, 128);
        const size_t   dst_row_size    = dst->nb[1];

        uint32_t nrows_per_hvx = fastdiv(ne01 + nth - 1, &octx->n_hvx_div);
        nrows_per_hvx          = hex_round_up(nrows_per_hvx, 32);
        const uint32_t start_r = nrows_per_hvx * ith;
        const uint32_t end_r   = MIN(start_r + nrows_per_hvx, ne01);
        if (start_r >= end_r) {
            continue;
        }

        const uint32_t ct_start     = start_r / 32;
        const uint32_t ct_end       = (end_r + 31) / 32;
        const uint32_t b_chunk_tl   = MAX(1u, (uint32_t) kp->b_chunk_tiles);

        for (uint32_t c0 = ct_start; c0 < ct_end; c0 += b_chunk_tl) {
            const uint32_t c1           = MIN(c0 + b_chunk_tl, ct_end);
            const uint32_t row_c0       = c0 * 32;
            const uint32_t chunk_valid  = MIN(c1 * 32, ne01) - row_c0;
            const uint32_t chunk_pad32  = (c1 - c0) * 32;

            // Push 1 contiguous DMA per active B_l chunk
            for (uint32_t a = 0; a < n_active; ++a) {
                const struct htp_tensor * B      = active_B[a];
                const size_t              brow   = active_b_row_bytes[a];
                const dma_addr_t          b_src  = B->data + (size_t) row_c0 * B->nb[1];
                if (B->nb[1] == brow) {
                    const size_t bytes = (size_t) chunk_valid * brow;
                    dma_queue_push(dma_q, dma_make_data(active_b_vtcm[a], b_src), bytes, bytes, bytes, 1);
                } else {
                    dma_queue_push(dma_q, dma_make_data(active_b_vtcm[a], b_src), brow, B->nb[1], brow, chunk_valid);
                }
            }

            // Push initial base weight tiles so DMA runs concurrently with LoRA B DMA + tail setup
            uint32_t push_ct = c0;
            for (uint32_t d = 0; d < n_prefetch && push_ct < c1; ++d, ++push_ct) {
                dma_queue_push(dma_q,
                               dma_make_data(vtcm_w_ptr + d * tile_row_vtcm, src_w->data + push_ct * tile_row_stride),
                               aligned_tile, tile_size, tile_size, n_k_tiles_w);
            }

            // Pop all B_l DMAs first (FIFO order) and zero-pad tail rows + 128 B guard
            for (uint32_t a = 0; a < n_active; ++a) {
                dma_queue_pop(dma_q);
                const size_t valid_bytes = (size_t) chunk_valid * active_b_row_bytes[a];
                const size_t total_zero  = (size_t) (chunk_pad32 - chunk_valid) * active_b_row_bytes[a] + 128;
                memset(active_b_vtcm[a] + valid_bytes, 0, total_zero);
            }

            // Hoist H pointers per activation row for each active adapter
            const float *  h_f32_m[HTP_LORA_MAX_HVX_M][HTP_LORA_MAX_ADAPTERS];
            const __fp16 * h_f16_m[HTP_LORA_MAX_HVX_M][HTP_LORA_MAX_ADAPTERS];
            for (uint32_t m = 0; m < act_nrows; ++m) {
                for (uint32_t a = 0; a < n_active; ++a) {
                    const uint32_t l = active_l[a];
                    h_f32_m[m][a] = lctx->vtcm_h_f32 + ((size_t) l * act_nrows + m) * lctx->h_f32_row_stride;
                    h_f16_m[m][a] = lctx->vtcm_h_f16 + ((size_t) l * act_nrows + m) * lctx->h_f16_row_stride;
                }
            }

            for (uint32_t ct = c0; ct < c1; ++ct) {
                const uint8_t * w_tile     = (const uint8_t *) dma_queue_pop(dma_q).dst;
                int             valid_rows = (int) ne01 - (int) (ct * 32);
                valid_rows                 = MIN(32, MAX(0, valid_rows));

                const uint8_t * b_tile_ptrs[HTP_LORA_MAX_ADAPTERS];
                for (uint32_t a = 0; a < n_active; ++a) {
                    b_tile_ptrs[a] = active_b_vtcm[a] + (size_t) (ct - c0) * 32 * active_b_row_bytes[a];
                }

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                uint32_t ir1 = 0;
                for (; ir1 + 1 < act_nrows; ir1 += 2) {
                    const uint8_t * restrict act_col0 = act_d + (ir1 + 0) * lctx->vtcm_act_stride;
                    const uint8_t * restrict act_col1 = act_d + (ir1 + 1) * lctx->vtcm_act_stride;
                    float *                  dst_ptr0 = (float *) (dst->data + (ir1 + 0) * dst_row_size) + ct * 32;
                    float *                  dst_ptr1 = (float *) (dst->data + (ir1 + 1) * dst_row_size) + ct * 32;
                    float *                  gate0    = lctx->vtcm_glu_gate + (ir1 + 0) * lctx->glu_stride + ct * 32;
                    float *                  gate1    = gate0 + lctx->glu_stride;

                    if (glu_gate) {
                        lctx->vec_dot_32x2(ne10, gate0, gate1, w_tile, act_col0, act_col1, valid_rows, NULL, NULL,
                                           hvx_tiled_store_row_f32_none);
                        if (n_active > 0) {
                            lora_apply_tile_32(gate0, n_active, active_b_types, b_tile_ptrs, h_f32_m[ir1 + 0],
                                               h_f16_m[ir1 + 0], active_r, active_scales);
                            lora_apply_tile_32(gate1, n_active, active_b_types, b_tile_ptrs, h_f32_m[ir1 + 1],
                                               h_f16_m[ir1 + 1], active_r, active_scales);
                        }
                    } else if (glu_out) {
                        lctx->vec_dot_32x2(ne10, tile_out_base, tile_out_base + 32, w_tile, act_col0, act_col1,
                                           valid_rows, NULL, NULL, hvx_tiled_store_row_f32_none);
                        if (n_active > 0) {
                            lora_apply_tile_32(tile_out_base, n_active, active_b_types, b_tile_ptrs, h_f32_m[ir1 + 0],
                                               h_f16_m[ir1 + 0], active_r, active_scales);
                            lora_apply_tile_32(tile_out_base + 32, n_active, active_b_types, b_tile_ptrs,
                                               h_f32_m[ir1 + 1], h_f16_m[ir1 + 1], active_r, active_scales);
                        }
                        hvx_mm_glu_store(dst_ptr0, gate0, tile_out_base, (uint32_t) valid_rows, glu);
                        hvx_mm_glu_store(dst_ptr1, gate1, tile_out_base + 32, (uint32_t) valid_rows, glu);
                    } else if (n_active > 0) {
                        lctx->vec_dot_32x2(ne10, tile_out_base, tile_out_base + 32, w_tile, act_col0, act_col1,
                                           valid_rows, NULL, NULL, hvx_tiled_store_row_f32_none);
                        lora_apply_tile_32(tile_out_base, n_active, active_b_types, b_tile_ptrs, h_f32_m[ir1 + 0],
                                           h_f16_m[ir1 + 0], active_r, active_scales);
                        lora_apply_tile_32(tile_out_base + 32, n_active, active_b_types, b_tile_ptrs,
                                           h_f32_m[ir1 + 1], h_f16_m[ir1 + 1], active_r, active_scales);
                        if (valid_rows == 32) {
                            hvx_vmemu(dst_ptr0) = hvx_vmem(tile_out_base);
                            hvx_vmemu(dst_ptr1) = hvx_vmem(tile_out_base + 32);
                        } else if (valid_rows > 0) {
                            hvx_vec_store_u(dst_ptr0, (uint32_t) valid_rows * sizeof(float), hvx_vmem(tile_out_base));
                            hvx_vec_store_u(dst_ptr1, (uint32_t) valid_rows * sizeof(float),
                                            hvx_vmem(tile_out_base + 32));
                        }
                    } else {
                        lctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, NULL,
                                           NULL, hvx_tiled_store_row_f32_none);
                    }
                }
                for (; ir1 < act_nrows; ++ir1) {
                    const uint8_t * restrict act_col = act_d + ir1 * lctx->vtcm_act_stride;
                    float *                  dst_ptr = (float *) (dst->data + ir1 * dst_row_size) + ct * 32;
                    float *                  gate    = lctx->vtcm_glu_gate + ir1 * lctx->glu_stride + ct * 32;

                    if (glu_gate) {
                        lctx->vec_dot_32x1(ne10, gate, w_tile, act_col, valid_rows, NULL,
                                           hvx_tiled_store_row_f32_none);
                        if (n_active > 0) {
                            lora_apply_tile_32(gate, n_active, active_b_types, b_tile_ptrs, h_f32_m[ir1], h_f16_m[ir1],
                                               active_r, active_scales);
                        }
                    } else if (glu_out) {
                        lctx->vec_dot_32x1(ne10, tile_out_base, w_tile, act_col, valid_rows, NULL,
                                           hvx_tiled_store_row_f32_none);
                        if (n_active > 0) {
                            lora_apply_tile_32(tile_out_base, n_active, active_b_types, b_tile_ptrs, h_f32_m[ir1],
                                               h_f16_m[ir1], active_r, active_scales);
                        }
                        hvx_mm_glu_store(dst_ptr, gate, tile_out_base, (uint32_t) valid_rows, glu);
                    } else if (n_active > 0) {
                        lctx->vec_dot_32x1(ne10, tile_out_base, w_tile, act_col, valid_rows, NULL,
                                           hvx_tiled_store_row_f32_none);
                        lora_apply_tile_32(tile_out_base, n_active, active_b_types, b_tile_ptrs, h_f32_m[ir1],
                                           h_f16_m[ir1], active_r, active_scales);
                        if (valid_rows == 32) {
                            hvx_vmemu(dst_ptr) = hvx_vmem(tile_out_base);
                        } else if (valid_rows > 0) {
                            hvx_vec_store_u(dst_ptr, (uint32_t) valid_rows * sizeof(float), hvx_vmem(tile_out_base));
                        }
                    } else {
                        lctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, NULL,
                                           hvx_tiled_store_row_f32_none);
                    }
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                if (push_ct < c1) {
                    dma_queue_push(dma_q,
                                   dma_make_data((void *) w_tile, src_w->data + push_ct * tile_row_stride),
                                   aligned_tile, tile_size, tile_size, n_k_tiles_w);
                    push_ct++;
                }
            }
        }
    }
}

static int hvx_mm_lora_op(struct htp_ops_context * octx, const struct htp_lora_kernel_params * kp) {
    struct htp_thread_trace * tr        = &octx->ctx->trace[0];
    const uint32_t            n_weights = kp->n_weights;
    const struct htp_tensor * w0        = octx->src[0];
    const struct htp_tensor * act       = octx->src[n_weights];

    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const uint32_t k         = act->ne[0];
    const uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    if (act_nrows == 0) {
        return HTP_STATUS_OK;
    }
    if (act_nrows > HTP_LORA_MAX_HVX_M || octx->ctx->mdev.count > 1) {
        return HTP_STATUS_NO_SUPPORT;
    }

    lora_hvx_ctx_t lctx_s;
    memset(&lctx_s, 0, sizeof(lctx_s));
    lora_hvx_ctx_t * lctx = &lctx_s;
    lctx->octx       = octx;
    lctx->kparams    = kp;
    lctx->act        = act;
    lctx->m_offset   = 0;
    lctx->m_rows     = act_nrows;
    lctx->k          = k;
    lctx->base_wtype = w0->type;

    if (lora_init_tiled_vec_dot((enum htp_data_type) w0->type, &lctx->vec_dot_32x1, &lctx->vec_dot_32x2) != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    uint32_t max_r             = 1;
    size_t   max_a_elem_bytes  = 2;
    size_t   max_b_row_bytes   = 0;
    uint32_t per_w_cnt[HTP_LORA_MAX_WEIGHTS] = { 0 };
    uint32_t max_loras_per_w   = 1;
    for (uint32_t l = 0; l < kp->n_loras; ++l) {
        const struct htp_tensor * A = octx->src[kp->lora_a_slot[l]];
        const struct htp_tensor * B = octx->src[kp->lora_b_slot[l]];
        lctx->a_tensors[l] = A;
        lctx->b_tensors[l] = B;
        const size_t ae = lora_elem_size(A->type);
        const size_t be = lora_elem_size(B->type);
        if (ae == 0 || be == 0) {
            return HTP_STATUS_NO_SUPPORT;
        }
        max_r            = MAX(max_r, (uint32_t) A->ne[1]);
        max_a_elem_bytes = MAX(max_a_elem_bytes, ae);
        max_b_row_bytes  = MAX(max_b_row_bytes, (size_t) B->ne[0] * be);
        const uint8_t tgt = kp->lora_target[l];
        if (tgt < n_weights) {
            per_w_cnt[tgt]++;
            max_loras_per_w = MAX(max_loras_per_w, per_w_cnt[tgt]);
        }
    }

    uint32_t max_n = 0;
    for (uint32_t p = 0; p < n_weights; ++p) {
        max_n = MAX(max_n, octx->src[p]->ne[1]);
    }

    struct htp_mm_hvx_vtcm_layout L_base;
    htp_mm_hvx_vtcm_layout_build(&L_base, kp->kernel_type, w0->type, k, act_nrows, octx->n_hvx, 0, 0,
                                 sizeof(float), 0, kp->n_prefetch, false, true);

    struct htp_lora_hvx_vtcm_layout L_lora;
    htp_lora_hvx_layout_build(&L_lora, L_base.total_bytes, k, act_nrows, octx->n_hvx, kp->n_loras, max_r,
                              max_a_elem_bytes, max_b_row_bytes, max_loras_per_w, kp->b_chunk_tiles, kp->glu, max_n);

    if (L_lora.total_bytes > octx->ctx->vtcm_size) {
        FARF(ERROR, "hvx-mm-lora: VTCM overflow: need %zu have %zu", L_lora.total_bytes, octx->ctx->vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    if (L_base.lut_bytes > 0) {
        iq2xxs_lut_copy(base + L_base.off_lut);
    }

    lctx->vtcm_src0              = VTCM_LAYOUT_PTR(uint8_t, base, L_base.off_src0);
    lctx->vtcm_act               = VTCM_LAYOUT_PTR(uint8_t, base, L_base.off_act);
    lctx->vtcm_act_raw           = VTCM_LAYOUT_PTR(uint8_t, base, L_base.off_act_raw);
    lctx->vtcm_src0_size_per_hvx = fastdiv(L_base.src0_bytes, &octx->n_hvx_div);
    lctx->vtcm_act_stride        = (size_t) kp->act_row_size;
    lctx->vtcm_act_raw_stride    = htp_mm_act_raw_row_size(k, sizeof(float));

    lctx->vtcm_x_f16        = VTCM_LAYOUT_PTR(__fp16, base, L_lora.off_x_f16);
    lctx->vtcm_h_f32        = VTCM_LAYOUT_PTR(float, base, L_lora.off_h_f32);
    lctx->vtcm_h_f16        = VTCM_LAYOUT_PTR(__fp16, base, L_lora.off_h_f16);
    lctx->vtcm_tile_out     = VTCM_LAYOUT_PTR(float, base, L_lora.off_tile_out);
    lctx->vtcm_glu_gate     = L_lora.off_glu_gate ? VTCM_LAYOUT_PTR(float, base, L_lora.off_glu_gate) : NULL;
    lctx->vtcm_scratch      = VTCM_LAYOUT_PTR(uint8_t, base, L_lora.off_scratch);
    lctx->x_f16_row_stride  = hex_align_up(k, 64);
    lctx->h_f32_row_stride  = L_lora.h_f32_row_stride;
    lctx->h_f16_row_stride  = L_lora.h_f16_row_stride;
    lctx->scratch_per_hvx   = L_lora.scratch_per_hvx;
    lctx->per_lora_b_bytes  = L_lora.per_lora_b_bytes;
    lctx->phase1_row_bytes  = L_lora.phase1_row_bytes;
    lctx->phase1_batch_rows = L_lora.phase1_batch_rows;
    lctx->glu_stride        = hex_round_up(max_n, 32);

    // Prepare quantize split (identical strategy to op_matmul_nx)
    lctx->do_quant       = true;
    const uint32_t qk    = HTP_MM_ACT_BLOCK_K;
    const uint32_t nb    = (k + qk - 1) / qk;
    const uint32_t tot_nb = act_nrows * nb;
    if (act_nrows < octx->n_hvx) {
        lctx->quant_is_block = true;
        lctx->n_quant_tasks  = MIN(tot_nb, octx->n_hvx);
        for (uint32_t ith = 0; ith < lctx->n_quant_tasks; ++ith) {
            const uint32_t ib0       = (tot_nb * ith) / lctx->n_quant_tasks;
            const uint32_t ib1       = (tot_nb * (ith + 1)) / lctx->n_quant_tasks;
            lctx->quant_ib_first[ith] = ib0;
            lctx->quant_ib_last[ith]  = ib1;
            lctx->quant_r[ith]        = ib0 / nb;
            lctx->quant_c[ith]        = ib0 % nb;
        }
    } else {
        lctx->quant_is_block       = false;
        lctx->n_quant_tasks        = MIN(act_nrows, octx->n_hvx);
        lctx->n_quant_rows_per_hvx = (act_nrows + lctx->n_quant_tasks - 1) / lctx->n_quant_tasks;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    // Stage activation into vtcm_act_raw and optionally fuse RMS_NORM+MUL
    if (kp->norm) {
        lora_dcache_invalidate_tensor(octx->src[kp->norm_w_slot]);
        hex_l2fetch((const void *) octx->src[kp->norm_w_slot]->data, k * sizeof(float), k * sizeof(float), 1);
    }
    const size_t act_elem_sz = lora_elem_size(act->type);
    const size_t act_row_b   = (size_t) k * act_elem_sz;
    dma_queue *  act_dma     = octx->ctx->dma[0];
    if (act->nb[1] == act_row_b && lctx->vtcm_act_raw_stride == act_row_b) {
        dma_queue_push(act_dma, dma_make_data(lctx->vtcm_act_raw, act->data), act_nrows * act_row_b,
                       act_nrows * act_row_b, act_nrows * act_row_b, 1);
    } else {
        dma_queue_push(act_dma, dma_make_data(lctx->vtcm_act_raw, act->data), lctx->vtcm_act_raw_stride, act->nb[1],
                       act_row_b, act_nrows);
    }
    dma_queue_pop(act_dma);

    const uint32_t k_pad32 = hex_round_up(k, 32);
    if (k_pad32 > k) {
        for (uint32_t ir = 0; ir < act_nrows; ++ir) {
            uint8_t * dst_pad = lctx->vtcm_act_raw + ir * lctx->vtcm_act_raw_stride + act_row_b;
            if (act->type == HTP_TYPE_F32) {
                hvx_splat_f32_u(dst_pad, 0.0f, k_pad32 - k);
            } else {
                hvx_splat_f16_u(dst_pad, (_Float16) 0.0f, k_pad32 - k);
            }
        }
    }

    if (kp->norm) {
        if (act->type != HTP_TYPE_F32) {
            return HTP_STATUS_NO_SUPPORT;
        }
        const uint8_t * norm_w = (const uint8_t *) octx->src[kp->norm_w_slot]->data;
        for (uint32_t ir = 0; ir < act_nrows; ++ir) {
            uint8_t * row = lctx->vtcm_act_raw + ir * lctx->vtcm_act_raw_stride;
            hvx_fast_rms_norm_mul_f32(row, norm_w, row, k, kp->norm_eps, kp->norm_scale);
        }
    }

    // Convert normalized activation to contiguous flat FP16 row(s) for LoRA A
    for (uint32_t ir = 0; ir < act_nrows; ++ir) {
        __fp16 * dst_f16 = lctx->vtcm_x_f16 + ir * lctx->x_f16_row_stride;
        if (act->type == HTP_TYPE_F32) {
            lora_convert_f32_to_f16_row(dst_f16, (const float *) (lctx->vtcm_act_raw + ir * lctx->vtcm_act_raw_stride),
                                        k);
        } else {
            lora_copy_f16_pad_row(dst_f16, (const __fp16 *) (lctx->vtcm_act_raw + ir * lctx->vtcm_act_raw_stride), k);
        }
    }

    // Barrier 1: base activation quantization + LoRA A (H = A @ x) simultaneously on n_hvx threads
    work_queue_run(octx->ctx->hvx_queue, lora_phase1_combined_worker, lctx, octx->n_hvx);

    // Barrier 2: base tiled matmul + fused LoRA B addition (+ optional GLU)
    work_queue_run(octx->ctx->hvx_queue, hvx_mm_lora_2d_tiled, lctx, octx->n_hvx);

    const uint32_t n_dsts = kp->glu ? 1u : n_weights;
    for (uint32_t p = 0; p < n_dsts; ++p) {
        lora_dcache_flush_tensor(octx->dsts[p]);
    }

    return HTP_STATUS_OK;
}

// ============================================================================
// Section 3: HMX Building Blocks & op_matmul_lora (Prefill M > 4)
// ============================================================================

static void hmx_init_eye_tile(__fp16 * vtcm_eye) {
    __fp16 eye_rm[32 * 32] __attribute__((aligned(VLEN)));
    memset(eye_rm, 0, sizeof(eye_rm));
    for (int i = 0; i < 32; ++i) {
        eye_rm[i * 32 + i] = (__fp16) 1.0f;
    }
    hmx_interleave_rows_to_tiles(vtcm_eye, eye_rm, 32, 32, 32, 0, 32);
}

static void core_mma_chunk_fp16_strided(
    __fp16 * restrict c, const __fp16 * restrict a, const __fp16 * restrict b,
    const __fp16 * restrict col_scales, const __fp16 * restrict eye_tile,
    uint32_t n_row_tiles, uint32_t n_col_tiles, uint32_t n_dot_tiles,
    uint32_t c_row_tile_stride, bool zero_init
) {
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);

    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int) col_scales));
    const size_t dot_stride = (size_t) n_dot_tiles * HTP_MM_HMX_TILE_N_ELMS;

    for (size_t i = 0; i < n_row_tiles; ++i) {
        const __fp16 * row_base   = a + i * dot_stride;
        const __fp16 * col_base   = b;
        __fp16 *       accum_tile = c + i * c_row_tile_stride * HTP_MM_HMX_TILE_N_ELMS;

        for (size_t j = 0; j < n_col_tiles; ++j) {
            const __fp16 * row_tiles = row_base;
            const __fp16 * col_tiles = col_base;

            asm volatile(HMX_CLRACC_F16());
            if (!zero_init) {
                asm volatile(HMX_LOAD_MPY_F16("%1", "%2", "%0") : : "r"(2047), "r"(accum_tile), "r"(eye_tile));
            }
            const uint32_t n_loops = n_dot_tiles / 32;
            const uint32_t rem     = n_dot_tiles % 32;
            for (uint32_t l = 0; l < n_loops; ++l) {
                asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(65535), "r"(row_tiles), "r"(col_tiles));
                row_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
                col_tiles += 32 * HTP_MM_HMX_TILE_N_ELMS;
            }
            if (rem > 0) {
                asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0")
                             : : "r"(2048u * rem - 1), "r"(row_tiles), "r"(col_tiles));
            }
            asm volatile(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(accum_tile), "r"(0) : "memory");

            col_base   += dot_stride;
            accum_tile += HTP_MM_HMX_TILE_N_ELMS;
        }
    }
}

typedef struct {
    __fp16 *       output;
    const __fp16 * activation;
    const __fp16 * weight;
    const __fp16 * scales;
    uint32_t       n_row_tiles;
    uint32_t       n_col_tiles;
    uint32_t       n_dot_tiles;
} lora_hmx_dot_job_t;

static void lora_hmx_dot_worker_fn(const htp_thread_t * th, void * data) {
    (void) th;
    const lora_hmx_dot_job_t * j = (const lora_hmx_dot_job_t *) data;
    core_dot_chunk_fp16(j->output, j->activation, j->weight, j->scales, j->n_row_tiles, j->n_col_tiles, j->n_dot_tiles);
}

typedef struct {
    __fp16 *       dst;
    const __fp16 * H;
    const __fp16 * B_tiles;
    uint32_t       n_row_tiles;
    uint32_t       n_col_tiles;
    uint32_t       r_tiles;
    uint32_t       c_row_tile_stride;
    bool           zero_init;
} lora_hmx_mma_subjob_t;

typedef struct {
    const __fp16 *        scales;
    const __fp16 *        eye_tile;
    uint32_t              n_subjobs;
    lora_hmx_mma_subjob_t sub[HTP_LORA_MAX_ADAPTERS];
} lora_hmx_mma_job_t;

static void lora_hmx_mma_worker_fn(const htp_thread_t * th, void * data) {
    (void) th;
    const lora_hmx_mma_job_t * job = (const lora_hmx_mma_job_t *) data;
    for (uint32_t s = 0; s < job->n_subjobs; ++s) {
        const lora_hmx_mma_subjob_t * sub = &job->sub[s];
        core_mma_chunk_fp16_strided(sub->dst, sub->H, sub->B_tiles, job->scales, job->eye_tile, sub->n_row_tiles,
                                    sub->n_col_tiles, sub->r_tiles, sub->c_row_tile_stride, sub->zero_init);
    }
}

typedef struct {
    __fp16 *        a_tiles;
    const uint8_t * a_ddr;
    size_t          a_stride;
    int             a_type;
    uint32_t        k_valid;
    uint32_t        k_tiles;
    uint32_t        strip_rows;
} lora_a_tile_prep_state_t;

static void lora_a_strip_to_fp16_tiles_worker(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    (void) th;
    const lora_a_tile_prep_state_t * st = (const lora_a_tile_prep_state_t *) data;

    const HVX_Vector     v_scat_base = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector     v_scat_step = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64    = Q6_Q_vsetq_R(64);
    const size_t         elem_sz     = lora_elem_size(st->a_type);
    const uint8_t *      a_ddr_end   = st->a_ddr + st->strip_rows * st->a_stride;

    for (uint32_t kt = i; kt < st->k_tiles; kt += n) {
        __fp16 *       tile_base = st->a_tiles + kt * HTP_MM_HMX_TILE_N_ELMS;
        const uint32_t k_off     = kt * 32;
        const int      rem_k     = (int) st->k_valid - (int) k_off;
        const uint32_t valid_k   = (rem_k <= 0) ? 0u : (uint32_t) hex_smin(32, rem_k);
        const HVX_VectorPred qk  = Q6_Q_vsetq_R(valid_k * sizeof(__fp16));
        HVX_Vector     v_off     = v_scat_base;

        for (uint32_t r = 0; r < 32; r += 2) {
            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();

            if (valid_k > 0) {
                const uint8_t * r0 = (r < st->strip_rows) ? (st->a_ddr + r * st->a_stride + k_off * elem_sz) : NULL;
                const uint8_t * r1 =
                    (r + 1 < st->strip_rows) ? (st->a_ddr + (r + 1) * st->a_stride + k_off * elem_sz) : NULL;

                if (st->a_type == HTP_TYPE_F16) {
                    if (r0) {
                        if (r0 + VLEN <= a_ddr_end) {
                            v0 = hvx_vmemu(r0);
                        } else {
                            __fp16 tmp[64] __attribute__((aligned(VLEN)));
                            memcpy(tmp, r0, valid_k * sizeof(__fp16));
                            v0 = hvx_vmem(tmp);
                        }
                    }
                    if (r1) {
                        if (r1 + VLEN <= a_ddr_end) {
                            v1 = hvx_vmemu(r1);
                        } else {
                            __fp16 tmp[64] __attribute__((aligned(VLEN)));
                            memcpy(tmp, r1, valid_k * sizeof(__fp16));
                            v1 = hvx_vmem(tmp);
                        }
                    }
                    v0 = Q6_V_vand_QV(qk, v0);
                    v1 = Q6_V_vand_QV(qk, v1);
                } else {
                    HVX_Vector f0 = Q6_V_vzero();
                    HVX_Vector f1 = Q6_V_vzero();
                    if (st->a_type == HTP_TYPE_F32) {
                        if (r0) {
                            if (r0 + VLEN <= a_ddr_end) {
                                f0 = hvx_vmemu(r0);
                            } else {
                                float tmp[32] __attribute__((aligned(VLEN)));
                                memcpy(tmp, r0, valid_k * sizeof(float));
                                f0 = hvx_vmem(tmp);
                            }
                        }
                        if (r1) {
                            if (r1 + VLEN <= a_ddr_end) {
                                f1 = hvx_vmemu(r1);
                            } else {
                                float tmp[32] __attribute__((aligned(VLEN)));
                                memcpy(tmp, r1, valid_k * sizeof(float));
                                f1 = hvx_vmem(tmp);
                            }
                        }
                    } else {
                        HVX_Vector bf0 = Q6_V_vzero(), bf1 = Q6_V_vzero();
                        if (r0) {
                            if (r0 + VLEN <= a_ddr_end) {
                                bf0 = hvx_vmemu(r0);
                            } else {
                                uint16_t tmp[64] __attribute__((aligned(VLEN)));
                                memcpy(tmp, r0, valid_k * sizeof(uint16_t));
                                bf0 = hvx_vmem(tmp);
                            }
                        }
                        if (r1) {
                            if (r1 + VLEN <= a_ddr_end) {
                                bf1 = hvx_vmemu(r1);
                            } else {
                                uint16_t tmp[64] __attribute__((aligned(VLEN)));
                                memcpy(tmp, r1, valid_k * sizeof(uint16_t));
                                bf1 = hvx_vmem(tmp);
                            }
                        }
                        f0 = Q6_Vw_vasl_VwR(Q6_V_lo_W(Q6_Ww_vunpack_Vh(bf0)), 16);
                        f1 = Q6_Vw_vasl_VwR(Q6_V_lo_W(Q6_Ww_vunpack_Vh(bf1)), 16);
                    }
                    const HVX_Vector v_packed = hvx_vec_f32_to_f16(f0, f1);
                    v0 = Q6_V_vand_QV(qk, v_packed);
                    v1 = Q6_V_vand_QV(qk, Q6_V_vror_VR(v_packed, 64));
                }
            }
            Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_base, HTP_MM_HMX_TILE_SIZE - 1, v_off, v0);
            v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
            Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_base, HTP_MM_HMX_TILE_SIZE - 1, v_off, v1);
            v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
        }
    }
}

typedef struct {
    __fp16 * ptr;
    size_t   n_vecs;
    float    scale;
} lora_h_scale_state_t;

static void lora_h_scale_worker(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    (void) th;
    const lora_h_scale_state_t * st    = (const lora_h_scale_state_t *) data;
    const size_t                 v_beg = (st->n_vecs * i) / n;
    const size_t                 v_end = (st->n_vecs * (i + 1)) / n;
    if (v_beg >= v_end) {
        return;
    }
    const size_t n_elems = (v_end - v_beg) * 64;
    __fp16 *     p       = st->ptr + v_beg * 64;
    hvx_mul_scalar_f16_aa((uint8_t *) p, (const uint8_t *) p, (_Float16) st->scale, (uint32_t) n_elems);
}

static void hmx_compute_lora_H(
    struct htp_context * ctx, struct htp_ops_context * octx, const struct htp_lora_kernel_params * kp,
    const __fp16 * vtcm_f16_act, const __fp16 * vtcm_scales,
    __fp16 * vtcm_H[HTP_LORA_MAX_ADAPTERS], __fp16 * a_tiles_scratch, __fp16 * h_tmp_scratch,
    uint32_t n_rows, uint32_t k_padded, uint32_t k_valid, int n_hvx, float lora_s_b[HTP_LORA_MAX_ADAPTERS]
) {
    const uint32_t m_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);
    const uint32_t k_tiles     = k_padded / HTP_MM_HMX_TILE_N_COLS;

    for (uint32_t l = 0; l < kp->n_loras; ++l) {
        const float sc = kp->lora_scale[l];
        if (sc == 0.0f) {
            lora_s_b[l] = 1.0f;
            continue;
        }
        const struct htp_tensor * A        = octx->src[kp->lora_a_slot[l]];
        const uint32_t            r_l      = A->ne[1];
        const uint32_t            r_tiles  = hmx_ceil_div(r_l, 32);
        const size_t              elem_sz  = lora_elem_size(A->type);

        for (uint32_t rt = 0; rt < r_tiles; ++rt) {
            const uint32_t strip_rows = hex_smin(32u, r_l - rt * 32);
            const uint8_t * strip_src = (const uint8_t *) A->data + (size_t) rt * 32 * A->nb[1];
            hex_l2fetch(strip_src, (size_t) k_valid * elem_sz, A->nb[1], strip_rows);

            lora_a_tile_prep_state_t prep = {
                .a_tiles    = a_tiles_scratch,
                .a_ddr      = strip_src,
                .a_stride   = A->nb[1],
                .a_type     = A->type,
                .k_valid    = k_valid,
                .k_tiles    = k_tiles,
                .strip_rows = strip_rows,
            };
            work_queue_run(ctx->hvx_queue, lora_a_strip_to_fp16_tiles_worker, &prep,
                           hex_smin(n_hvx, (int) k_tiles));

            __fp16 *           dot_dst = (r_tiles == 1) ? vtcm_H[l] : h_tmp_scratch;
            lora_hmx_dot_job_t djob    = { dot_dst, vtcm_f16_act, a_tiles_scratch, vtcm_scales, m_row_tiles, 1,
                                           k_tiles };
            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(lora_hmx_dot_worker_fn, &djob));
            hmx_queue_pop(ctx->hmx_queue);

            if (r_tiles > 1) {
                for (uint32_t m_t = 0; m_t < m_row_tiles; ++m_t) {
                    hvx_copy_aa((uint8_t *) (vtcm_H[l] + (m_t * r_tiles + rt) * HTP_MM_HMX_TILE_N_ELMS),
                                (const uint8_t *) (h_tmp_scratch + m_t * HTP_MM_HMX_TILE_N_ELMS),
                                HTP_MM_HMX_TILE_N_ELMS, sizeof(__fp16));
                }
            }
        }

        // Factor scale_l = s_h * s_b[l]: keep s_h cleanly inside normal FP16 range
        const float asc = fabsf(sc);
        float       s_h = sc;
        float       s_b = 1.0f;
        if (asc > 16.0f || (asc > 0.0f && asc < 0.0625f)) {
            const float root = sqrtf(asc);
            s_h              = (sc < 0.0f) ? -root : root;
            s_b              = root;
        }
        lora_s_b[l] = s_b;

        if (s_h != 1.0f) {
            const size_t total_tiles = (size_t) m_row_tiles * r_tiles;
            const size_t n_vecs      = total_tiles * (HTP_MM_HMX_TILE_N_ELMS / 64);
            if (total_tiles <= 4) {
                hvx_mul_scalar_f16_aa((uint8_t *) vtcm_H[l], (const uint8_t *) vtcm_H[l], (_Float16) s_h,
                                      (uint32_t) (total_tiles * HTP_MM_HMX_TILE_N_ELMS));
            } else {
                lora_h_scale_state_t sst = { vtcm_H[l], n_vecs, s_h };
                work_queue_run(ctx->hvx_queue, lora_h_scale_worker, &sst, hex_smin(n_hvx, (int) total_tiles));
            }
        }
    }
}

typedef struct {
    const struct htp_tensor * B;
    __fp16 *                  b_tiles_out;
    uint32_t                  nc;
    uint32_t                  n_cols;
    float                     s_b;
} hmx_b_prep_entry_t;

typedef struct {
    uint32_t           n_entries;
    hmx_b_prep_entry_t entries[HTP_LORA_MAX_ADAPTERS];
} hmx_b_prep_state_t;

// Never touches ctx->dma[0]; reads B_l from L2/DDR via hex_l2fetch + hvx_vmemu (with end-of-buffer guard).
static void hmx_prep_lora_B_worker(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    (void) th;
    const hmx_b_prep_state_t * st          = (const hmx_b_prep_state_t *) data;
    const HVX_Vector           v_scat_base = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector           v_scat_step = Q6_V_vsplat_R(4);
    const HVX_VectorPred       q_mask64    = Q6_Q_vsetq_R(64);

    // Map global column-tile index across entries
    uint32_t total_ct = 0;
    uint32_t cum[HTP_LORA_MAX_ADAPTERS + 1];
    cum[0] = 0;
    for (uint32_t e = 0; e < st->n_entries; ++e) {
        total_ct += hmx_ceil_div(st->entries[e].n_cols, 32);
        cum[e + 1] = total_ct;
    }

    for (uint32_t g_ct = i; g_ct < total_ct; g_ct += n) {
        uint32_t e = 0;
        while (cum[e + 1] <= g_ct) {
            e++;
        }
        const hmx_b_prep_entry_t * ent      = &st->entries[e];
        const struct htp_tensor *  B        = ent->B;
        const uint32_t             ct       = g_ct - cum[e];
        const uint32_t             r_l      = B->ne[0];
        const uint32_t             r_tiles  = hmx_ceil_div(r_l, 32);
        const size_t               elem_sz  = lora_elem_size(B->type);
        const size_t               b_stride = B->nb[1];
        const uint8_t *            b_base   = (const uint8_t *) B->data + (size_t) ent->nc * b_stride;
        const uint8_t *            b_end    = (const uint8_t *) B->data + (size_t) B->ne[1] * b_stride;

        // Fast-path: F16, unit B scale, r_l <= 32 -> scatter directly into HMX tile
        if (B->type == HTP_TYPE_F16 && ent->s_b == 1.0f && r_l <= 32) {
            __fp16 *             tile_dst = ent->b_tiles_out + ct * HTP_MM_HMX_TILE_N_ELMS;
            const HVX_VectorPred q_r      = Q6_Q_vsetq_R(r_l * sizeof(__fp16));
            HVX_Vector           v_off    = v_scat_base;
            for (uint32_t row = 0; row < 32; row += 2) {
                const uint32_t r0 = ct * 32 + row;
                const uint32_t r1 = r0 + 1;
                HVX_Vector     v0 = Q6_V_vzero();
                HVX_Vector     v1 = Q6_V_vzero();
                if (r0 < ent->n_cols) {
                    const uint8_t * p0 = b_base + r0 * b_stride;
                    if (p0 + VLEN <= b_end) {
                        v0 = hvx_vmemu(p0);
                    } else {
                        __fp16 tmp[64] __attribute__((aligned(VLEN)));
                        memcpy(tmp, p0, r_l * sizeof(__fp16));
                        v0 = hvx_vmem(tmp);
                    }
                    v0 = Q6_V_vand_QV(q_r, v0);
                }
                if (r1 < ent->n_cols) {
                    const uint8_t * p1 = b_base + r1 * b_stride;
                    if (p1 + VLEN <= b_end) {
                        v1 = hvx_vmemu(p1);
                    } else {
                        __fp16 tmp[64] __attribute__((aligned(VLEN)));
                        memcpy(tmp, p1, r_l * sizeof(__fp16));
                        v1 = hvx_vmem(tmp);
                    }
                    v1 = Q6_V_vand_QV(q_r, v1);
                }
                Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_dst, HTP_MM_HMX_TILE_SIZE - 1, v_off, v0);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
                Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_dst, HTP_MM_HMX_TILE_SIZE - 1, v_off, v1);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
            }
            continue;
        }

        // General path: r_l > 32, F32, BF16, or s_b != 1.0f
        const HVX_Vector v_sb_f32 = hvx_vec_splat_f32(ent->s_b);
        const HVX_Vector v_sb_f16 = hvx_vec_splat_f16((_Float16) ent->s_b);
        const bool       do_sb    = (ent->s_b != 1.0f);

        for (uint32_t rt = 0; rt < r_tiles; ++rt) {
            __fp16 *             tile_dst = ent->b_tiles_out + (ct * r_tiles + rt) * HTP_MM_HMX_TILE_N_ELMS;
            const uint32_t       r_off    = rt * 32;
            const uint32_t       valid_r  = hex_smin(32u, r_l - r_off);
            const HVX_VectorPred q_vr     = Q6_Q_vsetq_R(valid_r * sizeof(__fp16));
            HVX_Vector           v_off    = v_scat_base;

            for (uint32_t row = 0; row < 32; row += 2) {
                const uint32_t r0 = ct * 32 + row;
                const uint32_t r1 = r0 + 1;
                const uint8_t * p0 = (r0 < ent->n_cols) ? (b_base + r0 * b_stride + r_off * elem_sz) : NULL;
                const uint8_t * p1 = (r1 < ent->n_cols) ? (b_base + r1 * b_stride + r_off * elem_sz) : NULL;

                HVX_Vector v0 = Q6_V_vzero();
                HVX_Vector v1 = Q6_V_vzero();

                if (B->type == HTP_TYPE_F16) {
                    if (p0) {
                        if (p0 + VLEN <= b_end) {
                            v0 = hvx_vmemu(p0);
                        } else {
                            __fp16 tmp[64] __attribute__((aligned(VLEN)));
                            memcpy(tmp, p0, valid_r * sizeof(__fp16));
                            v0 = hvx_vmem(tmp);
                        }
                    }
                    if (p1) {
                        if (p1 + VLEN <= b_end) {
                            v1 = hvx_vmemu(p1);
                        } else {
                            __fp16 tmp[64] __attribute__((aligned(VLEN)));
                            memcpy(tmp, p1, valid_r * sizeof(__fp16));
                            v1 = hvx_vmem(tmp);
                        }
                    }
                    if (do_sb) {
                        v0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v0, v_sb_f16));
                        v1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v1, v_sb_f16));
                    }
                    v0 = Q6_V_vand_QV(q_vr, v0);
                    v1 = Q6_V_vand_QV(q_vr, v1);
                } else {
                    HVX_Vector f0 = Q6_V_vzero(), f1 = Q6_V_vzero();
                    if (B->type == HTP_TYPE_F32) {
                        if (p0) {
                            if (p0 + VLEN <= b_end) {
                                f0 = hvx_vmemu(p0);
                            } else {
                                float tmp[32] __attribute__((aligned(VLEN)));
                                memcpy(tmp, p0, valid_r * sizeof(float));
                                f0 = hvx_vmem(tmp);
                            }
                        }
                        if (p1) {
                            if (p1 + VLEN <= b_end) {
                                f1 = hvx_vmemu(p1);
                            } else {
                                float tmp[32] __attribute__((aligned(VLEN)));
                                memcpy(tmp, p1, valid_r * sizeof(float));
                                f1 = hvx_vmem(tmp);
                            }
                        }
                    } else {
                        HVX_Vector bf0 = Q6_V_vzero(), bf1 = Q6_V_vzero();
                        if (p0) {
                            if (p0 + VLEN <= b_end) {
                                bf0 = hvx_vmemu(p0);
                            } else {
                                uint16_t tmp[64] __attribute__((aligned(VLEN)));
                                memcpy(tmp, p0, valid_r * sizeof(uint16_t));
                                bf0 = hvx_vmem(tmp);
                            }
                        }
                        if (p1) {
                            if (p1 + VLEN <= b_end) {
                                bf1 = hvx_vmemu(p1);
                            } else {
                                uint16_t tmp[64] __attribute__((aligned(VLEN)));
                                memcpy(tmp, p1, valid_r * sizeof(uint16_t));
                                bf1 = hvx_vmem(tmp);
                            }
                        }
                        f0 = Q6_Vw_vasl_VwR(Q6_V_lo_W(Q6_Ww_vunpack_Vh(bf0)), 16);
                        f1 = Q6_Vw_vasl_VwR(Q6_V_lo_W(Q6_Ww_vunpack_Vh(bf1)), 16);
                    }
                    if (do_sb) {
                        f0 = hvx_vec_mul_f32_f32(f0, v_sb_f32);
                        f1 = hvx_vec_mul_f32_f32(f1, v_sb_f32);
                    }
                    const HVX_Vector vp = hvx_vec_f32_to_f16(f0, f1);
                    v0 = Q6_V_vand_QV(q_vr, vp);
                    v1 = Q6_V_vand_QV(q_vr, Q6_V_vror_VR(vp, 64));
                }

                Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_dst, HTP_MM_HMX_TILE_SIZE - 1, v_off, v0);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
                Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_dst, HTP_MM_HMX_TILE_SIZE - 1, v_off, v1);
                v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
            }
        }
    }
}

static void hmx_prep_lora_B_tiles_threaded(
    struct htp_context * ctx, struct htp_ops_context * octx, const struct htp_lora_kernel_params * kp,
    __fp16 * b_tiles_buf, uint32_t widx_filter, bool is_glu, uint32_t nc, uint32_t n_cols, uint32_t half_tiles,
    const float lora_s_b[HTP_LORA_MAX_ADAPTERS], __fp16 * dst_chunk, uint32_t m_row_tiles,
    uint32_t c_row_tile_stride, int n_hvx, __fp16 * vtcm_H[HTP_LORA_MAX_ADAPTERS],
    const __fp16 * vtcm_scales, const __fp16 * vtcm_eye, lora_hmx_mma_job_t * mma_job_out
) {
    hmx_b_prep_state_t prep;
    prep.n_entries          = 0;
    mma_job_out->scales     = vtcm_scales;
    mma_job_out->eye_tile   = vtcm_eye;
    mma_job_out->n_subjobs  = 0;

    __fp16 * cur_b_buf = b_tiles_buf;
    const uint32_t col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

    for (uint32_t pass = 0; pass < (is_glu ? 2u : 1u); ++pass) {
        const uint32_t tgt_w    = is_glu ? pass : widx_filter;
        __fp16 *       tgt_dst  = is_glu ? (dst_chunk + pass * half_tiles * HTP_MM_HMX_TILE_N_ELMS) : dst_chunk;
        const uint32_t nt_sub   = is_glu ? half_tiles : col_tiles;

        for (uint32_t l = 0; l < kp->n_loras; ++l) {
            if (kp->lora_target[l] != tgt_w || kp->lora_scale[l] == 0.0f) {
                continue;
            }
            const struct htp_tensor * B       = octx->src[kp->lora_b_slot[l]];
            const uint32_t            r_tiles = hmx_ceil_div(B->ne[0], 32);

            prep.entries[prep.n_entries++] = (hmx_b_prep_entry_t) {
                .B = B, .b_tiles_out = cur_b_buf, .nc = nc, .n_cols = n_cols, .s_b = lora_s_b[l],
            };
            mma_job_out->sub[mma_job_out->n_subjobs++] = (lora_hmx_mma_subjob_t) {
                .dst               = tgt_dst,
                .H                 = vtcm_H[l],
                .B_tiles           = cur_b_buf,
                .n_row_tiles       = m_row_tiles,
                .n_col_tiles       = col_tiles,
                .r_tiles           = r_tiles,
                .c_row_tile_stride = c_row_tile_stride,
                .zero_init         = false,
            };
            cur_b_buf += (size_t) col_tiles * r_tiles * HTP_MM_HMX_TILE_N_ELMS;
        }
        (void) nt_sub;
    }

    if (prep.n_entries > 0) {
        uint32_t total_ct = 0;
        for (uint32_t e = 0; e < prep.n_entries; ++e) {
            total_ct += hmx_ceil_div(prep.entries[e].n_cols, 32);
        }
        work_queue_run(ctx->hvx_queue, hmx_prep_lora_B_worker, &prep, hex_smin(n_hvx, (int) total_ct));
    }
}

static void hmx_l2fetch_lora_B_chunk(
    struct htp_ops_context * octx, const struct htp_lora_kernel_params * kp,
    uint32_t widx_filter, bool is_glu, uint32_t nc, uint32_t n_cols
) {
    if (n_cols == 0) {
        return;
    }
    for (uint32_t l = 0; l < kp->n_loras; ++l) {
        if (kp->lora_scale[l] == 0.0f) {
            continue;
        }
        if (!is_glu && kp->lora_target[l] != widx_filter) {
            continue;
        }
        const struct htp_tensor * B      = octx->src[kp->lora_b_slot[l]];
        if (nc >= B->ne[1]) {
            continue;
        }
        const uint32_t avail_rows = hex_smin(n_cols, B->ne[1] - nc);
        const size_t   row_bytes  = (size_t) B->ne[0] * lora_elem_size(B->type);
        hex_l2fetch((const void *) (B->data + (size_t) nc * B->nb[1]), row_bytes, B->nb[1], avail_rows);
    }
}

static void hmx_mm_lora_vtcm_setup(
    __fp16 * vtcm_scales, __fp16 * vtcm_eye, uint8_t * vtcm_lut, size_t lut_bytes
) {
    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));
    hmx_init_eye_tile(vtcm_eye);
    if (vtcm_lut && lut_bytes > 0) {
        iq2xxs_lut_copy(vtcm_lut);
    }
}

static int hmx_mm_lora_op(struct htp_ops_context * octx, const struct htp_lora_kernel_params * kp) {
    struct htp_context *      ctx       = octx->ctx;
    struct htp_thread_trace * tr        = &ctx->trace[0];
    const uint32_t            n_weights = kp->n_weights;
    const struct htp_tensor * w0        = octx->src[0];
    const struct htp_tensor * act       = octx->src[n_weights];

    if (ctx->mdev.count > 1) {
        return HTP_STATUS_NO_SUPPORT;
    }

    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const uint32_t m            = act->ne[1] * act->ne[2] * act->ne[3];
    const uint32_t k_valid      = act->ne[0];
    const uint32_t k            = hex_align_up(k_valid, 32);
    const size_t   row_stride   = w0->nb[1];
    const int      weight_type  = w0->type;
    const uint32_t act_elem_sz  = (act->type == HTP_TYPE_F16) ? (uint32_t) sizeof(__fp16) : (uint32_t) sizeof(float);
    const int      tile_size    = kp->tile_size;
    const int      aligned_ts   = kp->aligned_tile_size;
    const bool     is_tiled     = (tile_size > 0);
    const int      mc           = kp->m_chunk;
    const int      nc_cols      = kp->n_chunk;
    const bool     pipeline     = (kp->pipeline != 0);
    const int      n_hvx        = octx->n_hvx;
    const int      act_hvx      = kp->n_act_hvx;
    const uint32_t n_k_tiles    = k / HTP_MM_HMX_TILE_N_COLS;
    const uint32_t n_wt_tiles   = k / htp_tensor_get_tile_k(weight_type);
    const uint32_t dma_dst_str  = is_tiled ? (uint32_t) aligned_ts : (uint32_t) row_stride;
    const uint32_t dma_w_bytes  = is_tiled ? (uint32_t) tile_size  : (uint32_t) row_stride;

    tiled_dequant_task_fn_t dequant_fn = lora_hmx_get_dequant_worker_fn(weight_type, is_tiled);
    if (kp->glu && (!pipeline || !dequant_fn || n_weights != 2)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    uint32_t lora_r_tiles[HTP_LORA_MAX_ADAPTERS] = { 0 };
    uint32_t max_r                               = 1;
    for (uint32_t l = 0; l < kp->n_loras; ++l) {
        const uint32_t r_l = octx->src[kp->lora_a_slot[l]]->ne[1];
        lora_r_tiles[l]    = hmx_ceil_div(r_l, 32);
        max_r              = MAX(max_r, r_l);
    }

    struct htp_lora_hmx_layout L_lora;
    htp_lora_hmx_layout_build(&L_lora, mc, nc_cols, pipeline, kp->glu, n_hvx, n_weights, kp->n_loras,
                              kp->lora_target, lora_r_tiles, max_r);

    struct htp_mm_hmx_vtcm_layout L_hmx;
    htp_mm_hmx_vtcm_layout_build(&L_hmx, HTP_MM_KERNEL_HMX_2D, weight_type, k, mc, nc_cols, 1, pipeline, act_hvx,
                                 aligned_ts, L_lora.total_group_a_bytes);
    if (L_hmx.total_bytes > ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base         = (uint8_t *) ctx->vtcm_base;
    __fp16 *        vtcm_scales  = VTCM_LAYOUT_PTR(__fp16, base, L_hmx.off_scales);
    __fp16 *        vtcm_f16_act = VTCM_LAYOUT_PTR(__fp16, base, L_hmx.off_act);
    float *         vtcm_f32_act = VTCM_LAYOUT_PTR(float, base, L_hmx.off_act_f32);
    __fp16 *        vtcm_output  = VTCM_LAYOUT_PTR(__fp16, base, L_hmx.off_dst[0]);
    void *          vtcm_s0      = VTCM_LAYOUT_PTR(void, base, L_hmx.off_scratch[0]);
    void *          vtcm_s1      = VTCM_LAYOUT_PTR_OPTIONAL(void, base, L_hmx.off_scratch[1], pipeline);
    void *          vtcm_dst1    = VTCM_LAYOUT_PTR_OPTIONAL(void, base, L_hmx.off_dst[1], pipeline);
    __fp16 *        vtcm_w_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, L_hmx.off_weight[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L_hmx.off_weight[1], pipeline),
    };
    uint8_t * vtcm_lut = L_hmx.lut_bytes ? (base + L_hmx.off_lut) : NULL;

    uint8_t * const lora_ga      = base + L_hmx.off_bias;
    __fp16 *        vtcm_eye     = VTCM_LAYOUT_PTR(__fp16, lora_ga, L_lora.off_eye_tile);
    __fp16 *        vtcm_H[HTP_LORA_MAX_ADAPTERS];
    for (uint32_t l = 0; l < kp->n_loras; ++l) {
        vtcm_H[l] = VTCM_LAYOUT_PTR(__fp16, lora_ga, L_lora.off_H[l]);
    }
    __fp16 * vtcm_lora_b[2] = {
        VTCM_LAYOUT_PTR(__fp16, lora_ga, L_lora.off_b_tiles[0]),
        pipeline ? VTCM_LAYOUT_PTR(__fp16, lora_ga, L_lora.off_b_tiles[1]) : NULL,
    };

    const float * norm_w = kp->norm ? (const float *) octx->src[kp->norm_w_slot]->data : NULL;
    if (norm_w) {
        if (act->type != HTP_TYPE_F32) {
            return HTP_STATUS_NO_SUPPORT;
        }
        lora_dcache_invalidate_tensor(octx->src[kp->norm_w_slot]);
        hex_l2fetch((const void *) norm_w, k_valid * sizeof(float), k_valid * sizeof(float), 1);
    }

    hmx_mm_lora_vtcm_setup(vtcm_scales, vtcm_eye, vtcm_lut, L_hmx.lut_bytes);

    const struct fastdiv_values k_div       = kp->div_ne00_padded;
    const struct fastdiv_values act_hvx_div = kp->div_n_act_hvx;
    const struct fastdiv_values k_tiles_div = init_fastdiv_values((int) n_k_tiles);
    const size_t                act_stride  = act->nb[1] / act_elem_sz;
    const dma_addr_t            act_addr    = act->data;
    dma_queue *                 weight_dma  = ctx->dma[0];

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    if (pipeline && kp->glu) {
        const struct htp_tensor * w_gate       = octx->src[0];
        const struct htp_tensor * w_up         = octx->src[1];
        const struct htp_tensor * dst          = octx->dsts[0];
        const size_t              n            = w_gate->ne[1];
        const int                 half         = (nc_cols / 2) & ~31;
        const size_t              w_stride     = w_gate->nb[1];
        const size_t              dst_stride   = dst->nb[1] / sizeof(float);
        const int                 dst_cols     = (int) dst->ne[0];
        const uint32_t            dma_src_str  = is_tiled ? (uint32_t) tile_size : (uint32_t) w_stride;
        if (half <= 0 || w_up->ne[1] != n || w_up->type != w_gate->type) {
            return HTP_STATUS_INVAL_PARAMS;
        }
        const int                 n_chunk_cnt  = (int) hmx_ceil_div(n, (size_t) half);
        const size_t              up_off       = is_tiled ? (size_t) (half / 32) * n_wt_tiles * aligned_ts
                                                          : (size_t) half * row_stride;
        void *                    vtcm_w_b[2]  = { vtcm_s0, vtcm_s1 };
        void *                    vtcm_o_b[2]  = { vtcm_output, vtcm_dst1 };

        lora_hmx_dot_job_t  base_jobs[2];
        lora_hmx_mma_job_t  lora_jobs[2];

        for (size_t mr = 0; mr < m; mr += mc) {
            const size_t   n_rows      = hex_smin(m - mr, (size_t) mc);
            const uint32_t m_row_tiles = hmx_ceil_div((uint32_t) n_rows, HTP_MM_HMX_TILE_N_ROWS);
            float          lora_s_b[HTP_LORA_MAX_ADAPTERS];
            int            i_start = 0;

            #define GLU_PUSH_W_CHUNK(c, buf)                                                                         \
                do {                                                                                                 \
                    const size_t   nc_ = (size_t) (c) * half;                                                        \
                    const size_t   nn_ = hex_smin(n - nc_, (size_t) half);                                           \
                    const uint32_t h_  = is_tiled ? hmx_ceil_div((uint32_t) nn_, 32) * n_wt_tiles : (uint32_t) nn_;  \
                    dma_queue_push(weight_dma, dma_make_data((buf), w_gate->data + nc_ * w_stride), dma_dst_str,     \
                                   dma_src_str, dma_w_bytes, h_);                                                    \
                    dma_queue_push(weight_dma, dma_make_data((uint8_t *) (buf) + up_off, w_up->data + nc_ * w_stride),\
                                   dma_dst_str, dma_src_str, dma_w_bytes, h_);                                       \
                } while (0)

#ifdef HTP_YIELD_IN_OPS
        glu_pipelined_pass:;
#endif
            lora_transfer_activation_chunk_threaded(
                ctx, vtcm_f16_act, act_addr + mr * act_stride * act_elem_sz, (int) n_rows, (int) k,
                (int) act_stride, act_hvx, &act_hvx_div, &k_div, (int) k_valid, vtcm_f32_act, L_hmx.act_f32_bytes,
                act_elem_sz, norm_w, kp->norm_eps, kp->norm_scale);
            hmx_compute_lora_H(ctx, octx, kp, vtcm_f16_act, vtcm_scales, vtcm_H, (__fp16 *) vtcm_s0, vtcm_output,
                               (uint32_t) n_rows, k, k_valid, n_hvx, lora_s_b);

            if (i_start < n_chunk_cnt) {
                GLU_PUSH_W_CHUNK(i_start, vtcm_w_raw[0]);
                hmx_l2fetch_lora_B_chunk(octx, kp, 0, true, (uint32_t) i_start * half,
                                         (uint32_t) hex_smin(n - (size_t) i_start * half, (size_t) half));
            }
            if (i_start + 1 < n_chunk_cnt) {
                GLU_PUSH_W_CHUNK(i_start + 1, vtcm_w_raw[1]);
                hmx_l2fetch_lora_B_chunk(octx, kp, 0, true, (uint32_t) (i_start + 1) * half,
                                         (uint32_t) hex_smin(n - (size_t) (i_start + 1) * half, (size_t) half));
            }

            int prev_jobs = 0;
            for (int i = i_start; i < n_chunk_cnt; ++i) {
                const size_t nc     = (size_t) i * half;
                const size_t n_cols = hex_smin(n - nc, (size_t) half);

                void * curr_raw = (void *) dma_queue_pop(weight_dma).dst;
                dma_queue_pop(weight_dma);

                lora_dequantize_tiled_weight_chunk_to_fp16_tiles(ctx, (__fp16 *) vtcm_w_b[i % 2], curr_raw, 2 * half,
                                                                 (int) k, row_stride, weight_type, (int) n_k_tiles,
                                                                 k_tiles_div, dequant_fn, n_hvx);

                if (i + 2 < n_chunk_cnt) {
                    GLU_PUSH_W_CHUNK(i + 2, curr_raw);
                    const size_t nc_p2 = (size_t) (i + 2) * half;
                    hmx_l2fetch_lora_B_chunk(octx, kp, 0, true, (uint32_t) nc_p2,
                                             (uint32_t) hex_smin(n - nc_p2, (size_t) half));
                }

                base_jobs[i % 2] = (lora_hmx_dot_job_t) {
                    (__fp16 *) vtcm_o_b[i % 2], vtcm_f16_act, (__fp16 *) vtcm_w_b[i % 2], vtcm_scales,
                    m_row_tiles, (uint32_t) (2 * half) / HTP_MM_HMX_TILE_N_COLS, k / HTP_MM_HMX_TILE_N_ROWS,
                };
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(lora_hmx_dot_worker_fn, &base_jobs[i % 2]));

                if (i > i_start) {
                    for (int j = 0; j < prev_jobs; ++j) {
                        hmx_queue_pop(ctx->hmx_queue);
                    }
                    const size_t nc_prev = (size_t) (i - 1) * half;
                    lora_transfer_output_chunk_glu_threaded(
                        ctx, (float *) dst->data + mr * dst_stride + nc_prev, vtcm_o_b[(i - 1) % 2], (int) n_rows,
                        (int) hex_smin(n - nc_prev, (size_t) half), 2 * half, half / 32, (int) dst_stride,
                        dst_cols - (int) nc_prev, kp->glu, n_hvx);
                }

                // While HMX computes base_jobs[i % 2], prepare B_l tiles via HVX into vtcm_lora_b[i % 2]
                hmx_prep_lora_B_tiles_threaded(ctx, octx, kp, vtcm_lora_b[i % 2], 0, true, (uint32_t) nc,
                                               (uint32_t) n_cols, (uint32_t) (half / 32), lora_s_b,
                                               (__fp16 *) vtcm_o_b[i % 2], m_row_tiles, (uint32_t) (2 * (half / 32)),
                                               n_hvx, vtcm_H, vtcm_scales, vtcm_eye, &lora_jobs[i % 2]);
                int cur_jobs = 1;
                if (lora_jobs[i % 2].n_subjobs > 0) {
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(lora_hmx_mma_worker_fn, &lora_jobs[i % 2]));
                    cur_jobs++;
                }
                prev_jobs = cur_jobs;

#ifdef HTP_YIELD_IN_OPS
                if (i + 1 < n_chunk_cnt && atomic_load(&ctx->vtcm_needs_release)) {
                    for (int j = 0; j < cur_jobs; ++j) {
                        hmx_queue_pop(ctx->hmx_queue);
                    }
                    lora_transfer_output_chunk_glu_threaded(ctx, (float *) dst->data + mr * dst_stride + nc,
                                                            vtcm_o_b[i % 2], (int) n_rows, (int) n_cols, 2 * half,
                                                            half / 32, (int) dst_stride, dst_cols - (int) nc, kp->glu,
                                                            n_hvx);
                    dma_queue_pop(weight_dma);
                    dma_queue_pop(weight_dma);
                    if (i + 2 < n_chunk_cnt) {
                        dma_queue_pop(weight_dma);
                        dma_queue_pop(weight_dma);
                    }
                    htp_vtcm_yield(ctx);
                    hmx_mm_lora_vtcm_setup(vtcm_scales, vtcm_eye, vtcm_lut, L_hmx.lut_bytes);
                    i_start = i + 1;
                    goto glu_pipelined_pass;
                }
#endif
            }
            #undef GLU_PUSH_W_CHUNK

            for (int j = 0; j < prev_jobs; ++j) {
                hmx_queue_pop(ctx->hmx_queue);
            }
            const size_t nc_last = (size_t) (n_chunk_cnt - 1) * half;
            lora_transfer_output_chunk_glu_threaded(
                ctx, (float *) dst->data + mr * dst_stride + nc_last, vtcm_o_b[(n_chunk_cnt - 1) % 2], (int) n_rows,
                (int) hex_smin(n - nc_last, (size_t) half), 2 * half, half / 32, (int) dst_stride,
                dst_cols - (int) nc_last, kp->glu, n_hvx);

#ifdef HTP_YIELD_IN_OPS
            if (mr + (size_t) mc < m && atomic_load(&ctx->vtcm_needs_release)) {
                htp_vtcm_yield(ctx);
                hmx_mm_lora_vtcm_setup(vtcm_scales, vtcm_eye, vtcm_lut, L_hmx.lut_bytes);
            }
#endif
        }
    } else if (pipeline) {
        void * vtcm_raw_b[4] = {
            vtcm_w_raw[0], vtcm_w_raw[1],
            dequant_fn ? vtcm_w_raw[0] : vtcm_s0, dequant_fn ? vtcm_w_raw[1] : vtcm_s1,
        };
        void * vtcm_w_b[4] = {
            dequant_fn ? vtcm_s0 : vtcm_w_raw[0], dequant_fn ? vtcm_s1 : vtcm_w_raw[1],
            vtcm_s0, vtcm_s1,
        };
        void *             vtcm_o_b[2] = { vtcm_output, vtcm_dst1 };
        lora_hmx_dot_job_t base_jobs[2];
        lora_hmx_mma_job_t lora_jobs[2];

        for (size_t mr = 0; mr < m; mr += mc) {
            const size_t   n_rows      = hex_smin(m - mr, (size_t) mc);
            const uint32_t m_row_tiles = hmx_ceil_div((uint32_t) n_rows, HTP_MM_HMX_TILE_N_ROWS);
            float          lora_s_b[HTP_LORA_MAX_ADAPTERS];
            uint32_t       p_start     = 0;
            int            i_start     = 0;

#ifdef HTP_YIELD_IN_OPS
        nx_pipelined_pass:;
#endif
            lora_transfer_activation_chunk_threaded(
                ctx, vtcm_f16_act, act_addr + mr * act_stride * act_elem_sz, (int) n_rows, (int) k,
                (int) act_stride, act_hvx, &act_hvx_div, &k_div, (int) k_valid, vtcm_f32_act, L_hmx.act_f32_bytes,
                act_elem_sz, norm_w, kp->norm_eps, kp->norm_scale);
            hmx_compute_lora_H(ctx, octx, kp, vtcm_f16_act, vtcm_scales, vtcm_H, (__fp16 *) vtcm_s0, vtcm_output,
                               (uint32_t) n_rows, k, k_valid, n_hvx, lora_s_b);

            for (uint32_t p = p_start; p < n_weights; ++p) {
                const struct htp_tensor * restrict src_w = octx->src[p];
                const struct htp_tensor * restrict dst   = octx->dsts[p];
                if (!src_w || !dst || src_w->ne[1] == 0) {
                    i_start = 0;
                    continue;
                }
                const size_t   n           = src_w->ne[1];
                const size_t   w_stride    = src_w->nb[1];
                const size_t   dst_stride  = dst->nb[1] / sizeof(float);
                const int      dst_cols    = (int) dst->ne[0];
                float *        dst_ptr     = (float *) dst->data;
                const int      n_chunk_cnt = (int) hmx_ceil_div(n, (size_t) nc_cols);
                const uint32_t dma_src_str = is_tiled ? (uint32_t) tile_size : (uint32_t) w_stride;

                for (int pre = 0; pre < 2 && i_start + pre < n_chunk_cnt; ++pre) {
                    const int      c   = i_start + pre;
                    const size_t   nc  = (size_t) c * nc_cols;
                    const size_t   nn  = hex_smin(n - nc, (size_t) nc_cols);
                    const uint32_t ht  = is_tiled ? hmx_ceil_div((uint32_t) nn, 32) * n_wt_tiles : (uint32_t) nn;
                    dma_queue_push(weight_dma, dma_make_data(vtcm_raw_b[c % 4], src_w->data + nc * w_stride),
                                   dma_dst_str, dma_src_str, dma_w_bytes, ht);
                    hmx_l2fetch_lora_B_chunk(octx, kp, p, false, (uint32_t) nc, (uint32_t) nn);
                }

                int prev_jobs = 0;
                for (int i = i_start; i < n_chunk_cnt; ++i) {
                    const size_t   nc     = (size_t) i * nc_cols;
                    const size_t   n_cols = hex_smin(n - nc, (size_t) nc_cols);
                    const uint32_t nt_c   = hmx_ceil_div((uint32_t) n_cols, HTP_MM_HMX_TILE_N_COLS);

                    void * curr_raw = (void *) dma_queue_pop(weight_dma).dst;
                    lora_dequantize_tiled_weight_chunk_to_fp16_tiles(
                        ctx, (__fp16 *) vtcm_w_b[i % 4], curr_raw, (int) n_cols, (int) k, row_stride, weight_type,
                        (int) n_k_tiles, k_tiles_div, dequant_fn, n_hvx);

                    if (i + 2 < n_chunk_cnt) {
                        const size_t   nc_p2 = (size_t) (i + 2) * nc_cols;
                        const size_t   nn_p2 = hex_smin(n - nc_p2, (size_t) nc_cols);
                        const uint32_t ht_p2 = is_tiled ? hmx_ceil_div((uint32_t) nn_p2, 32) * n_wt_tiles
                                                        : (uint32_t) nn_p2;
                        dma_queue_push(weight_dma,
                                       dma_make_data(vtcm_raw_b[(i + 2) % 4], src_w->data + nc_p2 * w_stride),
                                       dma_dst_str, dma_src_str, dma_w_bytes, ht_p2);
                        hmx_l2fetch_lora_B_chunk(octx, kp, p, false, (uint32_t) nc_p2, (uint32_t) nn_p2);
                    }

                    base_jobs[i % 2] = (lora_hmx_dot_job_t) {
                        (__fp16 *) vtcm_o_b[i % 2], vtcm_f16_act, (__fp16 *) vtcm_w_b[i % 4], vtcm_scales,
                        m_row_tiles, nt_c, k / HTP_MM_HMX_TILE_N_ROWS,
                    };
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(lora_hmx_dot_worker_fn, &base_jobs[i % 2]));

                    if (i > i_start) {
                        for (int j = 0; j < prev_jobs; ++j) {
                            hmx_queue_pop(ctx->hmx_queue);
                        }
                        const size_t nc_prev = (size_t) (i - 1) * nc_cols;
                        const size_t nn_prev = hex_smin(n - nc_prev, (size_t) nc_cols);
                        lora_transfer_output_chunk_threaded(ctx, dst_ptr + mr * dst_stride + nc_prev, NULL,
                                                            vtcm_o_b[(i - 1) % 2], (int) n_rows, (int) nn_prev,
                                                            (int) dst_stride, 0, dst_cols - (int) nc_prev, n_hvx);
                    }

                    hmx_prep_lora_B_tiles_threaded(ctx, octx, kp, vtcm_lora_b[i % 2], p, false, (uint32_t) nc,
                                                   (uint32_t) n_cols, 0, lora_s_b, (__fp16 *) vtcm_o_b[i % 2],
                                                   m_row_tiles, nt_c, n_hvx, vtcm_H, vtcm_scales, vtcm_eye,
                                                   &lora_jobs[i % 2]);
                    int cur_jobs = 1;
                    if (lora_jobs[i % 2].n_subjobs > 0) {
                        hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(lora_hmx_mma_worker_fn, &lora_jobs[i % 2]));
                        cur_jobs++;
                    }
                    prev_jobs = cur_jobs;

#ifdef HTP_YIELD_IN_OPS
                    if (i + 1 < n_chunk_cnt && atomic_load(&ctx->vtcm_needs_release)) {
                        for (int j = 0; j < cur_jobs; ++j) {
                            hmx_queue_pop(ctx->hmx_queue);
                        }
                        lora_transfer_output_chunk_threaded(ctx, dst_ptr + mr * dst_stride + nc, NULL, vtcm_o_b[i % 2],
                                                            (int) n_rows, (int) n_cols, (int) dst_stride, 0,
                                                            dst_cols - (int) nc, n_hvx);
                        dma_queue_pop(weight_dma);
                        if (i + 2 < n_chunk_cnt) {
                            dma_queue_pop(weight_dma);
                        }
                        htp_vtcm_yield(ctx);
                        hmx_mm_lora_vtcm_setup(vtcm_scales, vtcm_eye, vtcm_lut, L_hmx.lut_bytes);
                        p_start = p;
                        i_start = i + 1;
                        goto nx_pipelined_pass;
                    }
#endif
                }

                for (int j = 0; j < prev_jobs; ++j) {
                    hmx_queue_pop(ctx->hmx_queue);
                }
                const size_t nc_last = (size_t) (n_chunk_cnt - 1) * nc_cols;
                const size_t nn_last = hex_smin(n - nc_last, (size_t) nc_cols);
                lora_transfer_output_chunk_threaded(ctx, dst_ptr + mr * dst_stride + nc_last, NULL,
                                                    vtcm_o_b[(n_chunk_cnt - 1) % 2], (int) n_rows, (int) nn_last,
                                                    (int) dst_stride, 0, dst_cols - (int) nc_last, n_hvx);
                i_start = 0;
            }
#ifdef HTP_YIELD_IN_OPS
            if (mr + (size_t) mc < m && atomic_load(&ctx->vtcm_needs_release)) {
                htp_vtcm_yield(ctx);
                hmx_mm_lora_vtcm_setup(vtcm_scales, vtcm_eye, vtcm_lut, L_hmx.lut_bytes);
            }
#endif
        }
    } else {
        // Synchronous non-pipelined loop
        void *             vtcm_w_b = dequant_fn ? vtcm_s0 : vtcm_w_raw[0];
        lora_hmx_dot_job_t base_job;
        lora_hmx_mma_job_t lora_job;

        for (size_t mr = 0; mr < m; mr += mc) {
            const size_t   n_rows      = hex_smin(m - mr, (size_t) mc);
            const uint32_t m_row_tiles = hmx_ceil_div((uint32_t) n_rows, HTP_MM_HMX_TILE_N_ROWS);
            float          lora_s_b[HTP_LORA_MAX_ADAPTERS];
            uint32_t       p_start     = 0;
            int            i_start     = 0;

#ifdef HTP_YIELD_IN_OPS
        nx_sync_pass:;
#endif
            lora_transfer_activation_chunk_threaded(
                ctx, vtcm_f16_act, act_addr + mr * act_stride * act_elem_sz, (int) n_rows, (int) k,
                (int) act_stride, act_hvx, &act_hvx_div, &k_div, (int) k_valid, vtcm_f32_act, L_hmx.act_f32_bytes,
                act_elem_sz, norm_w, kp->norm_eps, kp->norm_scale);
            hmx_compute_lora_H(ctx, octx, kp, vtcm_f16_act, vtcm_scales, vtcm_H, (__fp16 *) vtcm_s0, vtcm_output,
                               (uint32_t) n_rows, k, k_valid, n_hvx, lora_s_b);

            for (uint32_t p = p_start; p < n_weights; ++p) {
                const struct htp_tensor * restrict src_w = octx->src[p];
                const struct htp_tensor * restrict dst   = octx->dsts[p];
                if (!src_w || !dst || src_w->ne[1] == 0) {
                    i_start = 0;
                    continue;
                }
                const size_t   n           = src_w->ne[1];
                const size_t   w_stride    = src_w->nb[1];
                const size_t   dst_stride  = dst->nb[1] / sizeof(float);
                const int      dst_cols    = (int) dst->ne[0];
                float *        dst_ptr     = (float *) dst->data;
                const int      n_chunk_cnt = (int) hmx_ceil_div(n, (size_t) nc_cols);
                const uint32_t dma_src_str = is_tiled ? (uint32_t) tile_size : (uint32_t) w_stride;

                for (int i = i_start; i < n_chunk_cnt; ++i) {
                    const size_t   nc     = (size_t) i * nc_cols;
                    const size_t   n_cols = hex_smin(n - nc, (size_t) nc_cols);
                    const uint32_t nt_c   = hmx_ceil_div((uint32_t) n_cols, HTP_MM_HMX_TILE_N_COLS);
                    const uint32_t ht     = is_tiled ? hmx_ceil_div((uint32_t) n_cols, 32) * n_wt_tiles
                                                     : (uint32_t) n_cols;

                    hmx_l2fetch_lora_B_chunk(octx, kp, p, false, (uint32_t) nc, (uint32_t) n_cols);
                    dma_queue_push(weight_dma, dma_make_data(vtcm_w_raw[0], src_w->data + nc * w_stride), dma_dst_str,
                                   dma_src_str, dma_w_bytes, ht);
                    void * curr_raw = (void *) dma_queue_pop(weight_dma).dst;

                    lora_dequantize_tiled_weight_chunk_to_fp16_tiles(
                        ctx, (__fp16 *) vtcm_w_b, curr_raw, (int) n_cols, (int) k, row_stride, weight_type,
                        (int) n_k_tiles, k_tiles_div, dequant_fn, n_hvx);

                    base_job = (lora_hmx_dot_job_t) {
                        vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_w_b, vtcm_scales, m_row_tiles, nt_c,
                        k / HTP_MM_HMX_TILE_N_ROWS,
                    };
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(lora_hmx_dot_worker_fn, &base_job));

                    hmx_prep_lora_B_tiles_threaded(ctx, octx, kp, vtcm_lora_b[0], p, false, (uint32_t) nc,
                                                   (uint32_t) n_cols, 0, lora_s_b, vtcm_output, m_row_tiles, nt_c,
                                                   n_hvx, vtcm_H, vtcm_scales, vtcm_eye, &lora_job);
                    if (lora_job.n_subjobs > 0) {
                        hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(lora_hmx_mma_worker_fn, &lora_job));
                        hmx_queue_pop(ctx->hmx_queue);
                    }
                    hmx_queue_pop(ctx->hmx_queue);

                    lora_transfer_output_chunk_threaded(ctx, dst_ptr + mr * dst_stride + nc, NULL, vtcm_output,
                                                        (int) n_rows, (int) n_cols, (int) dst_stride, 0,
                                                        dst_cols - (int) nc, n_hvx);

#ifdef HTP_YIELD_IN_OPS
                    if (i + 1 < n_chunk_cnt && atomic_load(&ctx->vtcm_needs_release)) {
                        htp_vtcm_yield(ctx);
                        hmx_mm_lora_vtcm_setup(vtcm_scales, vtcm_eye, vtcm_lut, L_hmx.lut_bytes);
                        p_start = p;
                        i_start = i + 1;
                        goto nx_sync_pass;
                    }
#endif
                }
                i_start = 0;
            }
#ifdef HTP_YIELD_IN_OPS
            if (mr + (size_t) mc < m && atomic_load(&ctx->vtcm_needs_release)) {
                htp_vtcm_yield(ctx);
                hmx_mm_lora_vtcm_setup(vtcm_scales, vtcm_eye, vtcm_lut, L_hmx.lut_bytes);
            }
#endif
        }
    }

    const uint32_t n_dsts = kp->glu ? 1u : n_weights;
    for (uint32_t p = 0; p < n_dsts; ++p) {
        lora_dcache_flush_tensor(octx->dsts[p]);
    }

    return HTP_STATUS_OK;
}

// ============================================================================
// Section 4: Standalone op_lora_add (HVX & HMX Fallback)
// ============================================================================

typedef struct {
    lora_hvx_ctx_t                     base;
    const struct htp_tensor *          base_out;
    const struct htp_tensor *          dst;
    const struct htp_lora_add_hvx_layout * add_layout;
} lora_add_hvx_ctx_t;

static void lora_add_b_hvx_worker(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    lora_add_hvx_ctx_t *                  actx   = (lora_add_hvx_ctx_t *) data;
    lora_hvx_ctx_t *                      lctx   = &actx->base;
    const struct htp_lora_kernel_params * kp     = lctx->kparams;
    const struct htp_lora_add_hvx_layout * L     = actx->add_layout;
    const struct htp_tensor *             base_t = actx->base_out;
    const struct htp_tensor *             dst_t  = actx->dst;
    dma_queue *                           dma_q  = th->dma_q;

    const uint32_t n_total    = dst_t->ne[0];
    uint32_t       per_hvx    = (n_total + nth - 1) / nth;
    per_hvx                   = hex_round_up(per_hvx, 32);
    const uint32_t c_start    = per_hvx * ith;
    const uint32_t c_end      = MIN(c_start + per_hvx, n_total);
    if (c_start >= c_end) {
        return;
    }

    uint8_t                  active_types[HTP_LORA_MAX_ADAPTERS];
    uint32_t                 active_r[HTP_LORA_MAX_ADAPTERS];
    size_t                   active_b_row_b[HTP_LORA_MAX_ADAPTERS];
    float                    active_scales[HTP_LORA_MAX_ADAPTERS];
    const struct htp_tensor * active_B[HTP_LORA_MAX_ADAPTERS];
    uint8_t *                b_vtcm[HTP_LORA_MAX_ADAPTERS];
    uint8_t *                thr_scratch = lctx->vtcm_scratch + ith * L->scratch_per_hvx;
    uint32_t                 n_active    = 0;

    for (uint32_t l = 0; l < kp->n_loras; ++l) {
        if (kp->lora_scale[l] == 0.0f) {
            continue;
        }
        const struct htp_tensor * B = lctx->b_tensors[l];
        active_types[n_active]      = (uint8_t) B->type;
        active_r[n_active]          = B->ne[0];
        active_b_row_b[n_active]    = (size_t) B->ne[0] * lora_elem_size(B->type);
        active_scales[n_active]     = kp->lora_scale[l];
        active_B[n_active]          = B;
        b_vtcm[n_active]            = thr_scratch + n_active * L->per_lora_b_bytes;
        n_active++;
    }

    float *        base_buf = (float *) (thr_scratch + n_active * L->per_lora_b_bytes);
    const uint32_t n_chunk  = (uint32_t) kp->n_chunk;
    const uint32_t m_rows   = lctx->m_rows;

    for (uint32_t nc = c_start; nc < c_end; nc += n_chunk) {
        const uint32_t chunk_cols = MIN(n_chunk, c_end - nc);
        const uint32_t chunk_pad  = hex_round_up(chunk_cols, 32);

        for (uint32_t a = 0; a < n_active; ++a) {
            const struct htp_tensor * B     = active_B[a];
            const size_t              brow  = active_b_row_b[a];
            const dma_addr_t          b_src = B->data + (size_t) nc * B->nb[1];
            if (B->nb[1] == brow) {
                const size_t bytes = (size_t) chunk_cols * brow;
                dma_queue_push(dma_q, dma_make_data(b_vtcm[a], b_src), bytes, bytes, bytes, 1);
            } else {
                dma_queue_push(dma_q, dma_make_data(b_vtcm[a], b_src), brow, B->nb[1], brow, chunk_cols);
            }
        }
        for (uint32_t a = 0; a < n_active; ++a) {
            dma_queue_pop(dma_q);
            const size_t valid_b = (size_t) chunk_cols * active_b_row_b[a];
            memset(b_vtcm[a] + valid_b, 0, (size_t) (chunk_pad - chunk_cols) * active_b_row_b[a] + 128);
        }

        for (uint32_t m = 0; m < m_rows; ++m) {
            const uint32_t gm         = lctx->m_offset + m;
            const dma_addr_t b_row    = base_t->data + (size_t) gm * base_t->nb[1] + nc * sizeof(float);
            const size_t     cbytes   = (size_t) chunk_cols * sizeof(float);
            dma_queue_push(dma_q, dma_make_data(base_buf, b_row), cbytes, cbytes, cbytes, 1);
            dma_queue_pop(dma_q);

            const float *  h_f32_m[HTP_LORA_MAX_ADAPTERS];
            const __fp16 * h_f16_m[HTP_LORA_MAX_ADAPTERS];
            uint32_t       ai = 0;
            for (uint32_t l = 0; l < kp->n_loras; ++l) {
                if (kp->lora_scale[l] == 0.0f) {
                    continue;
                }
                h_f32_m[ai] = lctx->vtcm_h_f32 + ((size_t) l * m_rows + m) * lctx->h_f32_row_stride;
                h_f16_m[ai] = lctx->vtcm_h_f16 + ((size_t) l * m_rows + m) * lctx->h_f16_row_stride;
                ai++;
            }

            float * dst_row = (float *) (dst_t->data + (size_t) gm * dst_t->nb[1]) + nc;
            for (uint32_t c = 0; c < chunk_cols; c += 32) {
                const uint32_t rem = MIN(32u, chunk_cols - c);
                if (n_active > 0) {
                    const uint8_t * b_tile_ptrs[HTP_LORA_MAX_ADAPTERS];
                    for (uint32_t a = 0; a < n_active; ++a) {
                        b_tile_ptrs[a] = b_vtcm[a] + (size_t) c * active_b_row_b[a];
                    }
                    lora_apply_tile_32(base_buf + c, n_active, active_types, b_tile_ptrs, h_f32_m, h_f16_m,
                                       active_r, active_scales);
                }
                if (rem == 32) {
                    hvx_vmemu(dst_row + c) = hvx_vmem(base_buf + c);
                } else {
                    hvx_vec_store_u(dst_row + c, rem * sizeof(float), hvx_vmem(base_buf + c));
                }
            }
        }
    }
}

static void lora_a_only_worker(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    lora_a_hvx_worker_slice(th, nth, ith, (lora_hvx_ctx_t *) data);
}

static int hvx_lora_add_op(struct htp_ops_context * octx, const struct htp_lora_kernel_params * kp) {
    const struct htp_tensor * base_out = octx->src[0];
    const struct htp_tensor * act      = octx->src[1];
    const struct htp_tensor * dst      = octx->dst;
    struct htp_context *      ctx      = octx->ctx;

    const uint32_t k       = act->ne[0];
    const uint32_t m_total = act->ne[1] * act->ne[2] * act->ne[3];
    if (m_total == 0) {
        return HTP_STATUS_OK;
    }

    uint32_t max_r        = 1;
    size_t   max_a        = 2;
    size_t   max_b        = 0;
    bool     need_f32_act = (act->type == HTP_TYPE_F32);
    for (uint32_t l = 0; l < kp->n_loras; ++l) {
        const struct htp_tensor * A = octx->src[kp->lora_a_slot[l]];
        const struct htp_tensor * B = octx->src[kp->lora_b_slot[l]];
        max_r = MAX(max_r, (uint32_t) A->ne[1]);
        max_a = MAX(max_a, lora_elem_size(A->type));
        max_b = MAX(max_b, (size_t) B->ne[0] * lora_elem_size(B->type));
        if (A->type != HTP_TYPE_F16) {
            need_f32_act = true;
        }
    }

    const size_t raw_elem_sz = need_f32_act ? sizeof(float) : sizeof(__fp16);
    struct htp_lora_add_hvx_layout L;
    htp_lora_add_hvx_layout_build(&L, k, (uint32_t) kp->m_chunk, (uint32_t) kp->n_chunk, octx->n_hvx, kp->n_loras,
                                  max_r, raw_elem_sz, max_a, max_b);
    if (L.total_bytes > ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) ctx->vtcm_base;
    lora_add_hvx_ctx_t actx;
    memset(&actx, 0, sizeof(actx));
    actx.base_out             = base_out;
    actx.dst                  = dst;
    actx.add_layout           = &L;
    actx.base.octx            = octx;
    actx.base.kparams         = kp;
    actx.base.act             = act;
    actx.base.k               = k;
    actx.base.do_quant        = false;
    actx.base.vtcm_act_raw    = VTCM_LAYOUT_PTR(uint8_t, base, L.off_x_raw);
    actx.base.vtcm_x_f16      = VTCM_LAYOUT_PTR(__fp16, base, L.off_x_f16);
    actx.base.vtcm_h_f32      = VTCM_LAYOUT_PTR(float, base, L.off_h_f32);
    actx.base.vtcm_h_f16      = VTCM_LAYOUT_PTR(__fp16, base, L.off_h_f16);
    actx.base.vtcm_scratch    = VTCM_LAYOUT_PTR(uint8_t, base, L.off_scratch);
    actx.base.vtcm_act_raw_stride = L.x_raw_row_stride;
    actx.base.x_f16_row_stride = hex_align_up(k, 64);
    actx.base.h_f32_row_stride = L.h_f32_row_stride;
    actx.base.h_f16_row_stride = L.h_f16_row_stride;
    actx.base.scratch_per_hvx  = L.scratch_per_hvx;
    actx.base.phase1_row_bytes  = L.phase1_row_bytes;
    actx.base.phase1_batch_rows = L.phase1_batch_rows;
    for (uint32_t l = 0; l < kp->n_loras; ++l) {
        actx.base.a_tensors[l] = octx->src[kp->lora_a_slot[l]];
        actx.base.b_tensors[l] = octx->src[kp->lora_b_slot[l]];
    }

    const size_t act_elem_sz = lora_elem_size(act->type);
    const size_t act_row_b   = (size_t) k * act_elem_sz;
    const size_t f16_row_b   = (size_t) k * sizeof(__fp16);
    const size_t f16_dst_str = actx.base.x_f16_row_stride * sizeof(__fp16);
    dma_queue *  act_dma     = ctx->dma[0];

    for (uint32_t mr = 0; mr < m_total; mr += (uint32_t) kp->m_chunk) {
        const uint32_t m_rows = MIN((uint32_t) kp->m_chunk, m_total - mr);
        actx.base.m_offset    = mr;
        actx.base.m_rows      = m_rows;

        if (act->type == HTP_TYPE_F16 && need_f32_act) {
            // DMA F16 activation straight into vtcm_x_f16, pad tail, and widen to F32 into vtcm_act_raw
            if (act->nb[1] == f16_row_b && f16_dst_str == f16_row_b) {
                dma_queue_push(act_dma, dma_make_data(actx.base.vtcm_x_f16, act->data + (size_t) mr * act->nb[1]),
                               m_rows * f16_row_b, m_rows * f16_row_b, m_rows * f16_row_b, 1);
            } else {
                dma_queue_push(act_dma, dma_make_data(actx.base.vtcm_x_f16, act->data + (size_t) mr * act->nb[1]),
                               f16_dst_str, act->nb[1], f16_row_b, m_rows);
            }
            dma_queue_pop(act_dma);
            const uint32_t k_pad64 = hex_align_up(k, 64);
            for (uint32_t ir = 0; ir < m_rows; ++ir) {
                __fp16 * dst_r = actx.base.vtcm_x_f16 + ir * actx.base.x_f16_row_stride;
                if (k_pad64 > k) {
                    hvx_splat_f16_u((uint8_t *) (dst_r + k), (_Float16) 0.0f, k_pad64 - k);
                }
                lora_convert_f16_to_f32_row((float *) (actx.base.vtcm_act_raw + ir * L.x_raw_row_stride), dst_r, k);
            }
        } else {
            if (act->nb[1] == act_row_b && L.x_raw_row_stride == act_row_b) {
                dma_queue_push(act_dma, dma_make_data(actx.base.vtcm_act_raw, act->data + (size_t) mr * act->nb[1]),
                               m_rows * act_row_b, m_rows * act_row_b, m_rows * act_row_b, 1);
            } else {
                dma_queue_push(act_dma, dma_make_data(actx.base.vtcm_act_raw, act->data + (size_t) mr * act->nb[1]),
                               L.x_raw_row_stride, act->nb[1], act_row_b, m_rows);
            }
            dma_queue_pop(act_dma);

            for (uint32_t ir = 0; ir < m_rows; ++ir) {
                const uint8_t * raw_r = actx.base.vtcm_act_raw + ir * L.x_raw_row_stride;
                __fp16 *        dst_r = actx.base.vtcm_x_f16 + ir * actx.base.x_f16_row_stride;
                if (act->type == HTP_TYPE_F32) {
                    lora_convert_f32_to_f16_row(dst_r, (const float *) raw_r, k);
                } else {
                    lora_copy_f16_pad_row(dst_r, (const __fp16 *) raw_r, k);
                }
            }
        }

        work_queue_run(ctx->hvx_queue, lora_a_only_worker, &actx.base, octx->n_hvx);
        work_queue_run(ctx->hvx_queue, lora_add_b_hvx_worker, &actx, octx->n_hvx);

#ifdef HTP_YIELD_IN_OPS
        if (mr + (uint32_t) kp->m_chunk < m_total && atomic_load(&ctx->vtcm_needs_release)) {
            htp_vtcm_yield(ctx);
        }
#endif
    }

    lora_dcache_flush_tensor(dst);

    return HTP_STATUS_OK;
}

static int hmx_lora_add_op(struct htp_ops_context * octx, const struct htp_lora_kernel_params * kp) {
    struct htp_context *      ctx      = octx->ctx;
    const struct htp_tensor * base_out = octx->src[0];
    const struct htp_tensor * act      = octx->src[1];
    const struct htp_tensor * dst      = octx->dst;

    const uint32_t m           = act->ne[1] * act->ne[2] * act->ne[3];
    const uint32_t k_valid     = act->ne[0];
    const uint32_t k           = hex_align_up(k_valid, 32);
    const uint32_t n           = dst->ne[0];
    const uint32_t act_elem_sz = (act->type == HTP_TYPE_F16) ? (uint32_t) sizeof(__fp16) : (uint32_t) sizeof(float);
    const size_t   act_stride  = act->nb[1] / act_elem_sz;
    const size_t   base_stride = base_out->nb[1] / sizeof(float);
    const size_t   dst_stride  = dst->nb[1] / sizeof(float);
    const int      n_hvx       = octx->n_hvx;
    const int      act_hvx     = kp->n_act_hvx;
    const uint32_t mc          = (uint32_t) kp->m_chunk;
    const uint32_t nc_cols     = (uint32_t) kp->n_chunk;

    uint32_t r_tiles[HTP_LORA_MAX_ADAPTERS] = { 0 };
    uint32_t max_r                          = 1;
    for (uint32_t l = 0; l < kp->n_loras; ++l) {
        const uint32_t rl = octx->src[kp->lora_a_slot[l]]->ne[1];
        r_tiles[l]        = hmx_ceil_div(rl, 32);
        max_r             = MAX(max_r, rl);
    }

    struct htp_lora_add_hmx_layout L;
    htp_lora_add_hmx_layout_build(&L, k, mc, nc_cols, act_hvx, n_hvx, kp->n_loras, r_tiles, max_r);
    if (L.total_bytes > ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base         = (uint8_t *) ctx->vtcm_base;
    __fp16 *        vtcm_scales  = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);
    __fp16 *        vtcm_eye     = VTCM_LAYOUT_PTR(__fp16, base, L.off_eye_tile);
    __fp16 *        vtcm_f16_act = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    __fp16 *        vtcm_H[HTP_LORA_MAX_ADAPTERS];
    for (uint32_t l = 0; l < kp->n_loras; ++l) {
        vtcm_H[l] = VTCM_LAYOUT_PTR(__fp16, base, L.off_H[l]);
    }
    __fp16 * vtcm_b_tiles = VTCM_LAYOUT_PTR(__fp16, base, L.off_b_tiles);
    __fp16 * vtcm_dst     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst);
    float *  vtcm_act_f32 = VTCM_LAYOUT_PTR(float, base, L.off_act_f32);
    __fp16 * vtcm_a_tiles = VTCM_LAYOUT_PTR(__fp16, base, L.off_a_tiles);
    __fp16 * vtcm_h_tmp   = VTCM_LAYOUT_PTR(__fp16, base, L.off_h_tmp);

    const struct fastdiv_values k_div       = kp->div_ne00_padded;
    const struct fastdiv_values act_hvx_div = kp->div_n_act_hvx;

    hmx_mm_lora_vtcm_setup(vtcm_scales, vtcm_eye, NULL, 0);
    lora_dcache_invalidate_tensor(base_out);

    for (uint32_t mr = 0; mr < m; mr += mc) {
        const uint32_t n_rows      = hex_smin(m - mr, mc);
        const uint32_t m_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);
        float          lora_s_b[HTP_LORA_MAX_ADAPTERS];
        uint32_t       nc_start = 0;

#ifdef HTP_YIELD_IN_OPS
    add_hmx_pass:;
#endif
        lora_transfer_activation_chunk_threaded(
            ctx, vtcm_f16_act, act->data + (size_t) mr * act_stride * act_elem_sz, (int) n_rows, (int) k,
            (int) act_stride, act_hvx, &act_hvx_div, &k_div, (int) k_valid, vtcm_act_f32, L.act_f32_bytes, act_elem_sz,
            NULL, 0.0f, 0.0f);
        hmx_compute_lora_H(ctx, octx, kp, vtcm_f16_act, vtcm_scales, vtcm_H, vtcm_a_tiles, vtcm_h_tmp, n_rows, k,
                           k_valid, n_hvx, lora_s_b);

        for (uint32_t nc = nc_start; nc < n; nc += nc_cols) {
            const uint32_t n_cols = hex_smin(n - nc, nc_cols);
            const uint32_t nt_c   = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

            hmx_l2fetch_lora_B_chunk(octx, kp, 0, false, nc, n_cols);
            lora_hmx_mma_job_t mma_job;
            hmx_prep_lora_B_tiles_threaded(ctx, octx, kp, vtcm_b_tiles, 0, false, nc, n_cols, 0, lora_s_b, vtcm_dst,
                                           m_row_tiles, nt_c, n_hvx, vtcm_H, vtcm_scales, vtcm_eye, &mma_job);

            if (mma_job.n_subjobs == 0) {
                memset(vtcm_dst, 0, (size_t) m_row_tiles * nt_c * HTP_MM_HMX_TILE_SIZE);
            } else {
                const lora_hmx_mma_subjob_t * s0 = &mma_job.sub[0];
                lora_hmx_dot_job_t djob = {
                    vtcm_dst, s0->H, s0->B_tiles, vtcm_scales, m_row_tiles, nt_c, s0->r_tiles,
                };
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(lora_hmx_dot_worker_fn, &djob));
                if (mma_job.n_subjobs > 1) {
                    lora_hmx_mma_job_t rest = mma_job;
                    rest.n_subjobs          = mma_job.n_subjobs - 1;
                    for (uint32_t j = 0; j < rest.n_subjobs; ++j) {
                        rest.sub[j] = mma_job.sub[j + 1];
                    }
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(lora_hmx_mma_worker_fn, &rest));
                    hmx_queue_pop(ctx->hmx_queue);
                }
                hmx_queue_pop(ctx->hmx_queue);
            }

            hex_l2fetch((const void *) (base_out->data + (size_t) mr * base_out->nb[1] + nc * sizeof(float)),
                        n_cols * sizeof(float), base_out->nb[1], n_rows);
            lora_transfer_output_chunk_threaded(
                ctx, (float *) dst->data + mr * dst_stride + nc,
                (const float *) base_out->data + mr * base_stride + nc, vtcm_dst,
                (int) n_rows, (int) n_cols, (int) dst_stride, (uint32_t) base_stride, (int) (n - nc), n_hvx);

#ifdef HTP_YIELD_IN_OPS
            if (nc + nc_cols < n && atomic_load(&ctx->vtcm_needs_release)) {
                htp_vtcm_yield(ctx);
                hmx_mm_lora_vtcm_setup(vtcm_scales, vtcm_eye, NULL, 0);
                nc_start = nc + nc_cols;
                goto add_hmx_pass;
            }
#endif
        }
#ifdef HTP_YIELD_IN_OPS
        if (mr + mc < m && atomic_load(&ctx->vtcm_needs_release)) {
            htp_vtcm_yield(ctx);
            hmx_mm_lora_vtcm_setup(vtcm_scales, vtcm_eye, NULL, 0);
        }
#endif
    }

    lora_dcache_flush_tensor(dst);

    return HTP_STATUS_OK;
}

int op_matmul_lora(struct htp_ops_context * octx) {
    const struct htp_lora_kernel_params * kp = (const struct htp_lora_kernel_params *) octx->kernel_params;
    if (kp->n_weights == 0 || kp->n_weights > HTP_LORA_MAX_WEIGHTS || kp->n_loras == 0 ||
        kp->n_loras > HTP_LORA_MAX_ADAPTERS) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    for (uint32_t l = 0; l < kp->n_loras; ++l) {
        const struct htp_tensor * A = octx->src[kp->lora_a_slot[l]];
        const struct htp_tensor * B = octx->src[kp->lora_b_slot[l]];
        if (!A || !B || htp_tensor_is_extended(A) || htp_tensor_is_extended(B) ||
            htp_tensor_is_tiled(A) || htp_tensor_is_tiled(B)) {
            return HTP_STATUS_NO_SUPPORT;
        }
    }
    if (kp->n_hvx > 0) {
        htp_ops_context_set_n_hvx(octx, kp->n_hvx);
    }
    if (kp->lora_mode == HTP_LORA_MODE_HMX) {
        return hmx_mm_lora_op(octx, kp);
    }
    return hvx_mm_lora_op(octx, kp);
}

int op_lora_add(struct htp_ops_context * octx) {
    const struct htp_lora_kernel_params * kp = (const struct htp_lora_kernel_params *) octx->kernel_params;
    if (kp->n_loras == 0 || kp->n_loras > HTP_LORA_MAX_ADAPTERS) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    for (uint32_t l = 0; l < kp->n_loras; ++l) {
        const struct htp_tensor * A = octx->src[kp->lora_a_slot[l]];
        const struct htp_tensor * B = octx->src[kp->lora_b_slot[l]];
        if (!A || !B || htp_tensor_is_extended(A) || htp_tensor_is_extended(B) ||
            htp_tensor_is_tiled(A) || htp_tensor_is_tiled(B)) {
            return HTP_STATUS_NO_SUPPORT;
        }
    }
    if (kp->n_hvx > 0) {
        htp_ops_context_set_n_hvx(octx, kp->n_hvx);
    }
    if (kp->lora_mode == HTP_LORA_MODE_HMX) {
        return hmx_lora_add_op(octx, kp);
    }
    return hvx_lora_add_op(octx, kp);
}

