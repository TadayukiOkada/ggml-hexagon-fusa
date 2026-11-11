#ifndef HTP_DCACHE_H
#define HTP_DCACHE_H

#include <stdint.h>
#include "htp-ops.h"

#define HTP_MAX_DIRTY_RANGES 64
// Partial tracking pays off for large cache tensors; keep small tensors coarse.
#define HTP_DCACHE_PARTIAL_MIN_BYTES (1024 * 1024)

struct htp_dirty_range {
    uint32_t start;
    uint32_t end;
};

struct htp_context;

// no range is dirty (the whole D-cache was flushed)
void htp_dcache_reset(struct htp_context * ctx);
void htp_dcache_flush_dirty_ranges(struct htp_context * ctx);
void htp_dcache_flush_dirty_tensors(struct htp_context * ctx, const struct htp_tensor * const * tensors, uint32_t n);
void htp_dcache_mark_dirty_tensors(struct htp_context * ctx, const struct htp_tensor * const * tensors, uint32_t n);
// [start, end) of an output was written (an op that knows which part of its output it writes)
void htp_dcache_mark_dirty_range(struct htp_context * ctx, uint32_t start, uint32_t end);
// [start, end) was stored by HVX and written back (dccleana) by the threads that stored it
void htp_dcache_mark_range_clean(struct htp_context * ctx, uint32_t start, uint32_t end);

#endif // HTP_DCACHE_H
