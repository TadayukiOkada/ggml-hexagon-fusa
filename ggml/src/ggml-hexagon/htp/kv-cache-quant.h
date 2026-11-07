#ifndef KV_CACHE_QUANT_H
#define KV_CACHE_QUANT_H

#include "htp-tensor.h"
#include "ggml-impl.h"

// Specialization of upstream's IQ4_NL cache quantizer (ntry=-1, one 32-value
// block, no imatrix). Keep the initial codes and fit their F32 weighted scale.
static inline void kv_quantize_iq4_nl(uint8_t * dst, const float * src, uint32_t n) {
    #pragma clang fp contract(off)
    static const int values[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    for (uint32_t b = 0; b < n / 32; ++b) {
        const float * x = src + b * 32;
        uint8_t * y = dst + b * 18;
        float amax = 0, signed_max = 0;
        for (int j = 0; j < 32; ++j) {
            if (fabsf(x[j]) > amax) { amax = fabsf(x[j]); signed_max = x[j]; }
        }
        if (amax < 1e-15f) { memset(y, 0, 18); continue; }
        const float initial = signed_max / -127.0f;
        const float inv = 1.0f / initial;
        // Word lanes also avoid an unsupported v8i8 shift selected by toolv87's SLP pass.
        uint32_t codes[32];
        float sqx = 0, sq2 = 0;
        for (int j = 0; j < 32; ++j) {
            const float scaled = inv * x[j];
            int code = 15;
            for (int k = 0; k < 15; ++k) code -= scaled < 0.5f * (values[k] + values[k + 1]);
            codes[j] = code;
            const float w = x[j] * x[j], q = values[code];
            sqx += w * q * x[j];
            sq2 += w * q * q;
        }
        const uint16_t scale = GGML_FP32_TO_FP16(sq2 > 0 ? sqx / sq2 : 0);
        memcpy(y, &scale, 2);
        for (int j = 0; j < 16; ++j) y[2 + j] = codes[j] | (codes[16 + j] << 4);
    }
}

// Cache writes use the reference block rule: choose scale from F32, store it as
// F16, and round codes with the original F32 scale. In particular do not first
// round the input or the reciprocal to F16. The existing SET_ROWS DMA pipeline
// parallelizes rows and keeps DDR access out of these conversion loops.
static inline void kv_quantize_row(void * dst, const float * src, uint32_t n, int type) {
    if (type == HTP_TYPE_IQ4_NL) {
        kv_quantize_iq4_nl(dst, src, n);
        return;
    }
    if (type == HTP_TYPE_F8_E4M3) {
        for (uint32_t i = 0; i < n; ++i) ((uint8_t *) dst)[i] = ggml_fp32_to_f8_e4m3(src[i]);
        return;
    }
    const bool affine = type == HTP_TYPE_Q4_1 || type == HTP_TYPE_Q5_1;
    const bool five = type == HTP_TYPE_Q5_0 || type == HTP_TYPE_Q5_1;
    const int limit = five ? 31 : 15;
    const uint32_t size = htp_tensor_get_row_size(type, 32);
    for (uint32_t b = 0; b < n / 32; ++b) {
        const float * x = src + b * 32;
        uint8_t * y = (uint8_t *) dst + b * size;
        float lo = x[0], hi = x[0], amax = 0, signed_max = 0;
        for (uint32_t j = 0; j < 32; ++j) {
            lo = fminf(lo, x[j]); hi = fmaxf(hi, x[j]);
            if (fabsf(x[j]) > amax) { amax = fabsf(x[j]); signed_max = x[j]; }
        }
        const float scale = affine ? (hi - lo) / limit : signed_max / (five ? -16.0f : -8.0f);
        const float inv = scale ? 1.0f / scale : 0.0f;
        const uint16_t hscale = GGML_FP32_TO_FP16(scale);
        memcpy(y, &hscale, 2);
        if (affine) { const uint16_t m = GGML_FP32_TO_FP16(lo); memcpy(y + 2, &m, 2); }
        const uint32_t off = 2 + (affine ? 2 : 0) + (five ? 4 : 0);
        uint32_t high = 0;
        for (uint32_t j = 0; j < 16; ++j) {
            const float bias = affine ? 0.5f : five ? 16.5f : 8.5f;
            const int q0 = MIN(limit, (int) ((affine ? x[j] - lo : x[j]) * inv + bias));
            const int q1 = MIN(limit, (int) ((affine ? x[j + 16] - lo : x[j + 16]) * inv + bias));
            y[off + j] = (q0 & 15) | ((q1 & 15) << 4);
            high |= (uint32_t) ((q0 >> 4) & 1) << j;
            high |= (uint32_t) ((q1 >> 4) & 1) << (j + 16);
        }
        if (five) memcpy(y + off - 4, &high, 4);
    }
}

#endif
