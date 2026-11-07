// HVX cache writers vs the scalar reference (kv_quantize_block_ref / kv_quantize_iq4_nl /
// ggml_fp32_to_f8_e4m3), which test-kv-write checks against upstream: many rows of varied data.
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <HAP_vtcm_mgr.h>
#include <qurt_hvx.h>
#include "kv-cache-quant.h"

static uint32_t rs = 1;
static float rnd(void) { rs = rs * 1664525u + 1013904223u; return (float) (rs >> 8) / 16777216.0f; }

int main(void) {
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != 0) return 2;
    uint8_t * base = HAP_request_VTCM(1024 * 1024, 1);
    if (!base) return 3;
    float * x = (float *) base;
    uint8_t * got = base + 256 * 1024, * ref = base + 512 * 1024;
    const int types[] = { HTP_TYPE_Q4_0, HTP_TYPE_Q4_1, HTP_TYPE_Q5_0, HTP_TYPE_Q5_1, HTP_TYPE_IQ4_NL, HTP_TYPE_F8_E4M3 };
    const unsigned dims[] = { 32, 64, 96, 128, 160, 256, 576, 33, 1, 47 };
    unsigned cases = 0;
    for (int dist = 0; dist < 10; ++dist)
    for (unsigned di = 0; di < 10; ++di)
    for (int rep = 0; rep < 6; ++rep) {
        const unsigned n = dims[di];
        for (unsigned i = 0; i < n; ++i) {
            float v;
            switch (dist) {
                case 0: v = (rnd() - 0.5f) * 4.0f; break;                       // uniform
                case 1: v = (rnd() + rnd() + rnd() - 1.5f) * 30.0f; break;       // bell
                case 2: v = (float) ((int) (rnd() * 33) - 16) / 8.0f; break;     // grid: ties and exact halves
                case 3: v = -rnd() * 5.0f; break;                                // negative only
                case 4: v = rnd() * 1e-30f; break;                               // tiny
                case 5: v = (rnd() - 0.5f) * 1e4f; break;                        // large
                case 6: v = (i % 7 == 0) ? 0.0f : (i % 5 == 0 ? -0.0f : (rnd() - 0.5f)); break;  // signed zeros
                case 7: v = (i & 1) ? 3.0f : -3.0f; break;                       // opposite-sign ties
                case 8: v = (rnd() - 0.5f) * 1e-6f; break;                       // small
                default: v = (rnd() < 0.05f) ? (rnd() - 0.5f) * 600.0f : (rnd() - 0.5f) * 0.1f; break;  // outliers
            }
            x[i] = v;
        }
        for (unsigned ti = 0; ti < 6; ++ti) {
            const int type = types[ti];
            if (n % 32 && type != HTP_TYPE_F8_E4M3) continue;
            const uint32_t bytes = htp_tensor_get_row_size(type, n);
            memset(got, 0x55, bytes + 128); memset(ref, 0x55, bytes + 128);
            kv_quantize_row(got, x, n, type);
            if (type == HTP_TYPE_IQ4_NL) kv_quantize_iq4_nl(ref, x, n);
            else if (type == HTP_TYPE_F8_E4M3) for (unsigned i = 0; i < n; ++i) ref[i] = ggml_fp32_to_f8_e4m3(x[i]);
            else for (unsigned b = 0; b < n / 32; ++b) kv_quantize_block_ref(ref + b * htp_tensor_get_row_size(type, 32), x + b * 32, type);
            for (unsigned j = 0; j < bytes + 128; ++j) if (got[j] != ref[j]) {
                uint16_t sc; const unsigned bs = htp_tensor_get_row_size(type, 32);
                if (type == HTP_TYPE_IQ4_NL && j < bytes && j % bs >= 2) { memcpy(&sc, ref + (j / bs) * bs, 2); if (!(sc & 0x7fff)) continue; }
                printf("FAIL dist=%d n=%u rep=%d type=%d byte=%u ref=%02x got=%02x\n", dist, n, rep, type, j, ref[j], got[j]);
                fflush(stdout); return 1;
            }
            ++cases;
        }
    }
    printf("PASS %u rows: HVX writers equal the scalar reference\n", cases);
    return 0;
}
