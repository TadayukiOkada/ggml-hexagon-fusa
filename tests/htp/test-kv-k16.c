// hmx_f16_k_to_tiles (four-row scatters) == hmx_interleave_rows_to_tiles, all slices.
#include <stdio.h>
#include <string.h>
#include <HAP_vtcm_mgr.h>
#include <qurt_hvx.h>
#include "hmx-q8-prep.h"
int main(void) {
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != 0) return 2;
    uint8_t * base = HAP_request_VTCM(2*1024*1024, 1);
    unsigned cases = 0;
    const unsigned rows_all[] = {1, 2, 3, 5, 31, 32, 33, 63, 64, 65, 127, 128};
    const unsigned pads[] = {64, 128, 192, 256, 576};
    for (unsigned pi = 0; pi < 5; ++pi) for (unsigned ri = 0; ri < 12; ++ri) for (unsigned th = 1; th <= 4; th++) for (unsigned al = 2; al <= 4; al += 2) {
        const unsigned rows = rows_all[ri], padded = pads[pi], stride = padded;
        __fp16 * src = (void *) base, * a = (void *) (base + 512*1024), * b = (void *) (base + 1024*1024);
        for (unsigned i = 0; i < rows * stride; ++i) src[i] = (__fp16) (float) ((i * 37) % 2001 - 1000);
        const size_t bytes = (size_t) ((rows + 31) / 32) * (padded / 32) * 2048;
        memset(a, 0x55, bytes); memset(b, 0x55, bytes);
        const unsigned span = (((rows + th - 1) / th) + al - 1) / al * al;
        for (unsigned t = 0; t < th; ++t) {
            const unsigned s = t * span, e = s + span < rows ? s + span : rows;
            if (s >= e) continue;
            hmx_interleave_rows_to_tiles(a, src, rows, padded, stride, s, e); hvx_gather_sync(a);
            hmx_f16_k_to_tiles(b, src, rows, padded, stride, s, e); hvx_gather_sync(b);
        }
        if (memcmp(a, b, bytes)) { printf("FAIL rows=%u padded=%u th=%u al=%u\n", rows, padded, th, al); fflush(stdout); return 1; }
        ++cases;
    }
    printf("PASS %u F16 K tile comparisons\n", cases);
    return 0;
}
