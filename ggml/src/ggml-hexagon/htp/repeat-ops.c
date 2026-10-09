#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <string.h>

#include "hvx-utils.h"
#include "hvx-copy.h"
#include "dma-copy.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "hex-common.h"
#include "hex-profile.h"
#include "htp-ops.h"
#include "htp-tensor.h"

struct htp_repeat_context {
    struct htp_ops_context * octx;

    uint32_t nr0;
    uint32_t nr1;
    uint32_t nr2;
    uint32_t nr3;

    uint32_t nrows_per_thread;
    uint32_t total_dst_rows;  // ne1 * ne2 * ne3
    uint32_t row_start;

    size_t   type_size;
    uint32_t size_per_thread;
    bool     copy_direct;
};

static void repeat_row(uint8_t * dst, uint8_t * src, uint32_t row_bytes, uint32_t dst_bytes, uint32_t type_size) {
    if (row_bytes == type_size) {
        uint32_t bits = 0;
        memcpy(&bits, src, type_size);
        const HVX_Vector value = type_size == 4 ? Q6_V_vsplat_R(bits) : Q6_Vh_vsplat_R(bits);
        hvx_splat_u(dst, value, dst_bytes / type_size, type_size);
        return;
    }

    uint32_t block_bytes = row_bytes;
    if (row_bytes < 128) {
        block_bytes = MIN(dst_bytes, (128 / row_bytes) * row_bytes);
        for (uint32_t filled = row_bytes; filled < block_bytes;) {
            const uint32_t bytes = MIN(filled, block_bytes - filled);
            memcpy(src + filled, src, bytes);
            filled += bytes;
        }
        if (block_bytes == 128) {
            hvx_splat_u(dst, *(const HVX_Vector *) src, dst_bytes, 1);
            return;
        }
    }

    for (uint32_t offset = 0; offset < dst_bytes;) {
        const uint32_t bytes = MIN(block_bytes, dst_bytes - offset);
        hvx_copy_ua(dst + offset, src, bytes, 1);
        offset += bytes;
    }
}

static void repeat_copy_job_per_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_repeat_context * rctx = (const struct htp_repeat_context *) data;
    struct htp_ops_context * octx = rctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;

    const uint32_t ne00 = src->ne[0];
    const uint32_t ne01 = src->ne[1];
    const uint32_t ne02 = src->ne[2];
    const uint32_t ne03 = src->ne[3];

    const uint32_t nb00 = src->nb[0];
    const uint32_t nb01 = src->nb[1];
    const uint32_t nb02 = src->nb[2];
    const uint32_t nb03 = src->nb[3];

    const uint32_t ne0 = dst->ne[0];
    const uint32_t ne1 = dst->ne[1];
    const uint32_t ne2 = dst->ne[2];
    const uint32_t ne3 = dst->ne[3];

    const uint32_t nb0 = dst->nb[0];
    const uint32_t nb1 = dst->nb[1];
    const uint32_t nb2 = dst->nb[2];
    const uint32_t nb3 = dst->nb[3];

    const uint32_t nr0 = rctx->nr0;
    const uint32_t nr1 = rctx->nr1;
    const uint32_t nr2 = rctx->nr2;
    const uint32_t nr3 = rctx->nr3;

    const size_t row_bytes = ne00 * rctx->type_size;

    const uint32_t row_start = rctx->row_start + MIN((uint64_t) rctx->nrows_per_thread * ith, rctx->total_dst_rows);
    const uint32_t row_end   = row_start + MIN(rctx->nrows_per_thread, rctx->row_start + rctx->total_dst_rows - row_start);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    for (uint32_t dst_row = row_start; dst_row < row_end; dst_row++) {
        // Decompose flat dst row index into (i1, i2, i3)
        const uint32_t i1 = dst_row % ne1;
        const uint32_t i2 = (dst_row / ne1) % ne2;
        const uint32_t i3 = dst_row / (ne1 * ne2);

        // Map to source indices (tiling)
        const uint32_t k1 = i1 % ne01;
        const uint32_t k2 = i2 % ne02;
        const uint32_t k3 = i3 % ne03;

        const uint8_t * src_row = (const uint8_t *) (uintptr_t) src->data + k1 * nb01 + k2 * nb02 + k3 * nb03;
        uint8_t * dst_base      = (uint8_t *) (uintptr_t) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3;

        // Tile along dimension 0
        for (uint32_t i0 = 0; i0 < nr0; i0++) {
            uint8_t * dst_ptr = dst_base + i0 * ne00 * nb0;
            memcpy(dst_ptr, src_row, row_bytes);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(HIGH, "repeat %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

static void repeat_job_per_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_repeat_context * rctx = (const struct htp_repeat_context *) data;
    struct htp_ops_context * octx = rctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;

    const uint32_t ne00 = src->ne[0];
    const uint32_t ne01 = src->ne[1];
    const uint32_t ne02 = src->ne[2];
    const uint32_t ne03 = src->ne[3];

    const uint32_t nb00 = src->nb[0];
    const uint32_t nb01 = src->nb[1];
    const uint32_t nb02 = src->nb[2];
    const uint32_t nb03 = src->nb[3];

    const uint32_t ne0 = dst->ne[0];
    const uint32_t ne1 = dst->ne[1];
    const uint32_t ne2 = dst->ne[2];
    const uint32_t ne3 = dst->ne[3];

    const uint32_t nb0 = dst->nb[0];
    const uint32_t nb1 = dst->nb[1];
    const uint32_t nb2 = dst->nb[2];
    const uint32_t nb3 = dst->nb[3];

    const uint32_t nr0 = rctx->nr0;
    const uint32_t nr1 = rctx->nr1;
    const uint32_t nr2 = rctx->nr2;
    const uint32_t nr3 = rctx->nr3;

    const size_t row_bytes = ne00 * rctx->type_size;

    const uint32_t row_start = rctx->row_start + MIN((uint64_t) rctx->nrows_per_thread * ith, rctx->total_dst_rows);
    const uint32_t row_end   = row_start + MIN(rctx->nrows_per_thread, rctx->row_start + rctx->total_dst_rows - row_start);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint8_t * spad = octx->ctx->vtcm_base + ith * rctx->size_per_thread;
    const uint32_t dst_bytes = ne0 * rctx->type_size;
    const uint32_t pitch = row_bytes > rctx->size_per_thread ? 0 :
        (row_bytes == rctx->type_size ? row_bytes : hex_align_up(row_bytes, 128));
    const uint32_t staged_rows = pitch == 0 ? 0 : rctx->size_per_thread / pitch;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    for (uint32_t dst_row = row_start; dst_row < row_end;) {
        // Decompose flat dst row index into (i1, i2, i3)
        const uint32_t i1 = dst_row % ne1;
        const uint32_t i2 = (dst_row / ne1) % ne2;
        const uint32_t i3 = dst_row / (ne1 * ne2);

        // Map to source indices (tiling)
        const uint32_t k1 = i1 % ne01;
        const uint32_t k2 = i2 % ne02;
        const uint32_t k3 = i3 % ne03;

        const dma_addr_t src_row = src->data + (dma_addr_t) k1 * nb01 + (dma_addr_t) k2 * nb02 + (dma_addr_t) k3 * nb03;
        uint8_t * dst_base = (uint8_t *) (uintptr_t) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3;

        if (pitch != 0) {
            const uint32_t nrows = MIN(MIN(row_end - dst_row, ne1 - i1),
                                       MIN(ne01 - k1, staged_rows));
            dma_cpy_push_2d_chunked(dma_q, (dma_addr_t) (uintptr_t) spad, src_row, pitch, nb01, row_bytes, nrows);
            dma_queue_flush(dma_q);
            for (uint32_t row = 0; row < nrows; row++) {
                repeat_row(dst_base + row * nb1, spad + row * pitch, row_bytes, dst_bytes, rctx->type_size);
            }
            dst_row += nrows;
        } else {
            for (uint32_t offset = 0; offset < row_bytes;) {
                const uint32_t bytes = MIN(row_bytes - offset, rctx->size_per_thread);
                dma_cpy_sametype_reshape_contig(dma_q, (dma_addr_t) (uintptr_t) spad, src_row + offset, bytes);
                dma_queue_flush(dma_q);
                for (uint32_t i0 = 0; i0 < nr0; i0++) {
                    hvx_copy_ua(dst_base + i0 * row_bytes + offset, spad, bytes, 1);
                }
                offset += bytes;
            }
            dst_row++;
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(HIGH, "repeat %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

int op_repeat(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    size_t type_size;
    switch (src0->type) {
        case HTP_TYPE_F32: type_size = 4; break;
        case HTP_TYPE_F16: type_size = 2; break;
        default:
            FARF(ERROR, "repeat: unsupported type %u\n", src0->type);
            return HTP_STATUS_NO_SUPPORT;
    }

    if (src0->type != dst->type || src0->nb[0] != type_size || dst->nb[0] != type_size) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }
#if __HVX_ARCH__ <= 79
    if (htp_tensor_is_extended(src0)) {
        return HTP_STATUS_NO_SUPPORT;
    }
#endif

    // Software division is costly on Hexagon: divide once per dim and check overflow with 64-bit products.
    uint64_t src_size = type_size;
    uint64_t dst_size = type_size;
    uint32_t total_dst_rows = 1;
    uint32_t nr[HTP_OP_MAX_DIMS];
    for (uint32_t i = 0; i < HTP_OP_MAX_DIMS; i++) {
        if (src0->ne[i] == 0 || dst->ne[i] == 0) {
            return HTP_STATUS_INVAL_PARAMS;
        }
        nr[i] = dst->ne[i] / src0->ne[i];
        if (nr[i] * src0->ne[i] != dst->ne[i]) {
            return HTP_STATUS_INVAL_PARAMS;
        }
        const uint64_t src_offset = (uint64_t) (src0->ne[i] - 1) * src0->nb[i];
        const uint64_t dst_offset = (uint64_t) (dst->ne[i] - 1) * dst->nb[i];
        if (src_offset > UINT32_MAX - src_size || dst_offset > UINT32_MAX - dst_size) {
            return HTP_STATUS_INVAL_PARAMS;
        }
        src_size += src_offset;
        dst_size += dst_offset;
        if (i > 0) {
            const uint64_t rows = (uint64_t) total_dst_rows * dst->ne[i];
            if (rows > UINT32_MAX) {
                return HTP_STATUS_INVAL_PARAMS;
            }
            total_dst_rows = (uint32_t) rows;
        }
    }
    if (dst_size - 1 > UINT32_MAX - dst->data || src0->data >= (UINT64_C(1) << 40) ||
        src_size > (UINT64_C(1) << 40) - src0->data) {
        return HTP_STATUS_INVAL_PARAMS;
    }
#if __HVX_ARCH__ <= 79
    if (src_size - 1 > UINT32_MAX - src0->data) {
        return HTP_STATUS_INVAL_PARAMS;
    }
#endif

    const size_t dst_row_size = dst->ne[0] * type_size;

    uint32_t row_start = 0;
    uint32_t nrows     = total_dst_rows;

    if (octx->ctx->mdev.count > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, type_size, (uint32_t) dst_row_size, &rows_per_chunk);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_dst_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads = MIN(octx->n_threads, octx->ctx->vtcm_size / 128);
    if (!htp_ops_context_set_n_threads(octx, n_threads)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    struct htp_repeat_context rctx = {
        .octx             = octx,
        .nr0              = nr[0],
        .nr1              = nr[1],
        .nr2              = nr[2],
        .nr3              = nr[3],
        .nrows_per_thread = fastdiv(nrows - 1, &octx->n_threads_div) + 1,
        .total_dst_rows   = nrows,
        .row_start        = row_start,
        .type_size        = type_size,
        .size_per_thread  = MIN(16 * 1024, (octx->ctx->vtcm_size / n_threads) & ~(size_t) 127),
        // DMA setup and masked HVX stores cost more than memcpy for short rows.
        .copy_direct      = (dst_size <= 8192 || dst_row_size <= 512) && !htp_tensor_is_extended(src0) && src_size - 1 <= UINT32_MAX - src0->data,
    };

    FARF(HIGH, "repeat: (%ux%ux%ux%u) -> (%ux%ux%ux%u) nr=(%u,%u,%u,%u)\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         rctx.nr0, rctx.nr1, rctx.nr2, rctx.nr3);

    work_queue_run(octx->ctx->work_queue, rctx.copy_direct ? repeat_copy_job_per_thread : repeat_job_per_thread, &rctx, n_threads);

    return HTP_STATUS_OK;
}
