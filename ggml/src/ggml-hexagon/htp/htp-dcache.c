#include "htp-dcache.h"

#include <qurt.h>
#include <qurt_memory.h>
#include <HAP_farf.h>

#include "hex-common.h"
#include "hex-utils.h"
#include "hex-fastdiv.h"
#include "htp-trace.h"
#include "htp-ctx.h"
#include "work-queue.h"

struct l2flush_range {
    uint32_t start;       // line-aligned start address
    uint32_t end;         // line-aligned end address
    uint32_t block_first; // global block index of this range's first block
    uint32_t n_blocks;    // number of HTP_DCACHE_BLOCK_SIZE chunks (last may be partial)
};

struct l2flush_multi_task {
    struct l2flush_range      ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t                  n_ranges;
    uint32_t                  total_blocks;
    uint32_t                  blocks_per_hvx;
};

static void flush_all_dcache(struct htp_context * ctx) {
    struct htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    hex_l2fetch_block(ctx, ctx->footprint);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    ctx->n_dirty_ranges = 0;
}

static void l2flush_multi_worker(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    struct l2flush_multi_task * task = (struct l2flush_multi_task *) data;
    (void) n;

    const uint32_t gb_first = i * task->blocks_per_hvx;
    uint32_t       gb_last  = gb_first + task->blocks_per_hvx;
    if (gb_last > task->total_blocks) {
        gb_last = task->total_blocks;
    }
    if (gb_first >= gb_last) {
        return;
    }

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);

    for (uint32_t r = 0; r < task->n_ranges; r++) {
        const struct l2flush_range * rg = &task->ranges[r];
        const uint32_t rb_first = rg->block_first;
        const uint32_t rb_last  = rg->block_first + rg->n_blocks;

        const uint32_t lo = gb_first > rb_first ? gb_first : rb_first;
        const uint32_t hi = gb_last  < rb_last  ? gb_last  : rb_last;
        if (lo >= hi) {
            continue;
        }

        const uint32_t s = rg->start + (lo - rb_first) * HTP_DCACHE_BLOCK_SIZE;
        uint32_t       e = rg->start + (hi - rb_first) * HTP_DCACHE_BLOCK_SIZE;
        if (e > rg->end) {
            e = rg->end;
        }
        hex_l2flush((void *) (uintptr_t) s, e - s);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, gb_first);
}

static void flush_dirty_ranges(struct htp_context * ctx, const struct htp_dirty_range * ranges, uint32_t n_ranges, uint64_t total_dirty) {
    if (total_dirty >= HTP_DCACHE_FLUSH_WQ_THRESHOLD && ctx->n_hvx > 1) {
        struct l2flush_multi_task task;
        task.n_ranges = n_ranges;

        uint32_t block_acc = 0;
        for (uint32_t i = 0; i < n_ranges; i++) {
            const struct htp_dirty_range * r = &ranges[i];
            struct l2flush_range * rg = &task.ranges[i];
            rg->start = hex_align_down((size_t) r->start, HEX_L2_LINE_SIZE);
            rg->end   = hex_align_up((size_t) r->end, HEX_L2_LINE_SIZE);
            rg->block_first = block_acc;
            rg->n_blocks = (rg->end - rg->start + HTP_DCACHE_BLOCK_SIZE - 1) / HTP_DCACHE_BLOCK_SIZE;
            block_acc += rg->n_blocks;
        }

        task.total_blocks      = block_acc;
        task.blocks_per_hvx = fastdiv(block_acc + ctx->n_hvx - 1, &ctx->n_hvx_div);

        work_queue_run(ctx->hvx_queue, l2flush_multi_worker, &task, ctx->n_hvx);
    } else {
        struct htp_thread_trace * tr = &ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_L2FLUSH, 0);
        for (uint32_t i = 0; i < n_ranges; i++) {
            const struct htp_dirty_range * r = &ranges[i];
            hex_l2flush((void *) (uintptr_t) r->start, r->end - r->start);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_L2FLUSH, 0);
    }
}

// The dirty ranges are ctx->dirty_ranges[0 .. n_dirty_ranges): in no order, and no two of them overlap
// or are adjacent. An op costs a pass over the ranges in use (a decode batch runs hundreds of ops of
// a few microseconds each, and its table is full most of the time).

// Add [start, end). The ranges that touch it are absorbed into it; they cannot touch one another or
// any other range, so one pass leaves the table disjoint. With a full table, one range is flushed.
static void dirty_add(struct htp_context * ctx, uint32_t start, uint32_t end) {
    struct htp_dirty_range * const d = ctx->dirty_ranges;
    uint32_t n = ctx->n_dirty_ranges;

    for (uint32_t i = 0; i < n;) {
        if (d[i].start <= end && start <= d[i].end) {
            start = MIN(start, d[i].start);
            end   = MAX(end, d[i].end);
            d[i]  = d[--n];
        } else {
            i++;
        }
    }

    if (n == HTP_MAX_DIRTY_RANGES) {
        const struct htp_dirty_range r    = d[0];
        const uint32_t               size = r.end - r.start;
        if (size > HTP_DCACHE_FLUSH_ALL_THRESHOLD) {
            flush_all_dcache(ctx);
            n = 0;
        } else {
            flush_dirty_ranges(ctx, &r, 1, size);
            d[0] = d[--n];
        }
    }

    d[n].start = start;
    d[n].end   = end;
    ctx->n_dirty_ranges = n + 1;
}

void htp_dcache_reset(struct htp_context * ctx) {
    ctx->n_dirty_ranges = 0;
}

void htp_dcache_mark_dirty_tensors(struct htp_context * ctx, const struct htp_tensor * const * tensors, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const struct htp_tensor * t = tensors[i];
        if (!t || !t->size || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
            continue;
        }
        dirty_add(ctx, t->data, t->data + t->size);
    }
}

void htp_dcache_mark_dirty_range(struct htp_context * ctx, uint32_t start, uint32_t end) {
    if (end <= start) {
        return;
    }
    dirty_add(ctx, start, end);
}

void htp_dcache_mark_range_clean(struct htp_context * ctx, uint32_t start, uint32_t end) {
    struct htp_dirty_range * const d = ctx->dirty_ranges;
    uint32_t n = ctx->n_dirty_ranges;

    for (uint32_t i = 0; i < n;) {
        struct htp_dirty_range * r = &d[i];
        if (r->end <= start || end <= r->start) {
            i++;
            continue;
        }
        if (start <= r->start && r->end <= end) {
            d[i] = d[--n];
            continue;
        }
        if (start <= r->start) {
            r->start = end;
        } else if (r->end <= end) {
            r->end = start;
        } else if (n < HTP_MAX_DIRTY_RANGES) {
            // inside a larger range (outputs next to each other merge): split it. With a full table
            // the range stays dirty.
            d[n].start = end;
            d[n].end   = r->end;
            r->end     = start;
            n++;
        }
        i++;
    }
    ctx->n_dirty_ranges = n;
}

void htp_dcache_flush_dirty_ranges(struct htp_context * ctx) {
    const uint32_t n = ctx->n_dirty_ranges;
    uint64_t total_dirty = 0;

    for (uint32_t i = 0; i < n; i++) {
        total_dirty += ctx->dirty_ranges[i].end - ctx->dirty_ranges[i].start;
    }
    if (total_dirty == 0) {
        return;
    }
    if (total_dirty > HTP_DCACHE_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        return;
    }

    flush_dirty_ranges(ctx, ctx->dirty_ranges, n, total_dirty);
    ctx->n_dirty_ranges = 0;
}

// The dirty parts of the tensors are flushed, not the whole tensors: an input may be much larger than
// what was written to it (the K/V caches after a SET_ROWS of one token).
void htp_dcache_flush_dirty_tensors(struct htp_context * ctx, const struct htp_tensor * const * tensors, uint32_t n) {
    const uint32_t n_dirty = ctx->n_dirty_ranges;
    if (n_dirty == 0) {
        return;
    }

    struct htp_dirty_range ranges[HTP_MAX_DIRTY_RANGES];
    uint32_t n_ranges = 0;
    uint64_t total_dirty = 0;

    for (uint32_t i = 0; i < n; i++) {
        const struct htp_tensor * t = tensors[i];
        if (!t || (t->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_FENCE))) {
            continue;
        }
        const uint32_t t_start = t->data;
        const uint32_t t_end   = t_start + t->size;
        const bool whole_tensor = t->size < HTP_DCACHE_PARTIAL_MIN_BYTES;

        for (uint32_t j = 0; j < n_dirty; j++) {
            const struct htp_dirty_range * r = &ctx->dirty_ranges[j];
            if (r->end <= t_start || t_end <= r->start) {
                continue;
            }
            if (n_ranges == HTP_MAX_DIRTY_RANGES) {
                htp_dcache_flush_dirty_ranges(ctx);  // more parts than fit: everything that is dirty
                return;
            }
            // Small inputs are flushed whole on the first overlap, which is cheaper than collecting
            // their dirty parts.
            ranges[n_ranges].start = whole_tensor ? t_start : MAX(r->start, t_start);
            ranges[n_ranges].end   = whole_tensor ? t_end : MIN(r->end, t_end);
            total_dirty += ranges[n_ranges].end - ranges[n_ranges].start;
            n_ranges++;
            // The dirty ranges are disjoint: when one covers the input, no other one intersects it
            if (whole_tensor || (r->start <= t_start && t_end <= r->end)) {
                break;
            }
        }
    }

    if (total_dirty == 0) {
        return;
    }

    if (total_dirty > HTP_DCACHE_FLUSH_ALL_THRESHOLD) {
        flush_all_dcache(ctx);
        return;
    }

    flush_dirty_ranges(ctx, ranges, n_ranges, total_dirty);
    for (uint32_t i = 0; i < n_ranges; i++) {
        htp_dcache_mark_range_clean(ctx, ranges[i].start, ranges[i].end);
    }
}
