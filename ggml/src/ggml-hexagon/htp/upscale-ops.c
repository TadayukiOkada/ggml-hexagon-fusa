#include <float.h>
#include <math.h>
#include <string.h>
#include <HAP_farf.h>

#include "hex-common.h"
#include "dma-queue.h"
#include "dma-pipeline.h"
#include "htp-trace.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hvx-types.h"
#include "hvx-base.h"
#include "hvx-utils.h"
#include "upscale-ops.h"

static inline int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float clamp_f32(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float bicubic_weight1(float x) {
    const float a = -0.75f;
    return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
}

static inline float bicubic_weight2(float x) {
    const float a = -0.75f;
    return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
}

static inline void bicubic_weights(float x, float w[4]) {
    w[0] = bicubic_weight2(x + 1.0f);
    w[1] = bicubic_weight1(x);
    w[2] = bicubic_weight1(1.0f - x);
    w[3] = bicubic_weight2(2.0f - x);
}

static void upscale_init_luts_nearest(
    uint32_t * lut_x,
    uint32_t * lut_y,
    const struct htp_tensor * src0,
    const struct htp_tensor * dst,
    const struct htp_upscale_kernel_params * p
) {
    const uint32_t src_x = src0->ne[0];
    const uint32_t src_y = src0->ne[1];
    const uint32_t dst_x = dst->ne[0];
    const uint32_t dst_y = dst->ne[1];
    const uint32_t dst_x_aligned = (dst_x + VLEN_FP32 - 1) & ~(VLEN_FP32 - 1);

    for (uint32_t ox = 0; ox < dst_x_aligned; ++ox) {
        if (ox < dst_x) {
            uint32_t x0 = (uint32_t) ((float) ox * p->inv_sf0);
            lut_x[ox] = MIN(x0, src_x - 1) * sizeof(float);
        } else {
            lut_x[ox] = 0;
        }
    }
    for (uint32_t oy = 0; oy < dst_y; ++oy) {
        uint32_t y0 = (uint32_t) ((float) oy * p->inv_sf1);
        lut_y[oy] = MIN(y0, src_y - 1);
    }
}

static void upscale_init_luts_bilinear(
    struct upscale_bilinear_x_chunk * lut_x,
    struct upscale_bilinear_y * lut_y,
    const struct htp_tensor * src0,
    const struct htp_tensor * dst,
    const struct htp_upscale_kernel_params * p
) {
    const uint32_t src_x = src0->ne[0];
    const uint32_t src_y = src0->ne[1];
    const uint32_t dst_x = dst->ne[0];
    const uint32_t dst_y = dst->ne[1];
    const uint32_t n_chunks = (dst_x + VLEN_FP32 - 1) / VLEN_FP32;

    for (uint32_t c = 0; c < n_chunks; ++c) {
        struct upscale_bilinear_x_chunk * chunk = &lut_x[c];
        const uint32_t ox_start = c * VLEN_FP32;
        for (uint32_t l = 0; l < VLEN_FP32; ++l) {
            const uint32_t ox = ox_start + l;
            if (ox < dst_x) {
                float xf = ((float) ox + p->pixel_offset) * p->inv_sf0 - p->pixel_offset;
                int32_t x0 = (int32_t) floorf(xf);
                int32_t x1 = x0 + 1;
                x0 = clamp_i32(x0, 0, (int32_t) src_x - 1);
                x1 = clamp_i32(x1, 0, (int32_t) src_x - 1);
                float dx = clamp_f32(xf - (float) x0, 0.0f, 1.0f);

                chunk->off0[l] = x0 * (int32_t) sizeof(float);
                chunk->off1[l] = x1 * (int32_t) sizeof(float);
                chunk->w0[l]   = 1.0f - dx;
                chunk->dx[l]   = dx;
            } else {
                chunk->off0[l] = 0;
                chunk->off1[l] = 0;
                chunk->w0[l]   = 0.0f;
                chunk->dx[l]   = 0.0f;
            }
        }
    }
    for (uint32_t oy = 0; oy < dst_y; ++oy) {
        float yf = ((float) oy + p->pixel_offset) * p->inv_sf1 - p->pixel_offset;
        int32_t y0 = (int32_t) floorf(yf);
        int32_t y1 = y0 + 1;
        y0 = clamp_i32(y0, 0, (int32_t) src_y - 1);
        y1 = clamp_i32(y1, 0, (int32_t) src_y - 1);
        float dy = clamp_f32(yf - (float) y0, 0.0f, 1.0f);

        lut_y[oy].y0  = (uint32_t) y0;
        lut_y[oy].y1  = (uint32_t) y1;
        lut_y[oy].wy1 = dy;
        lut_y[oy].wy0 = 1.0f - dy;
    }
}

static inline float upscale_triangle(float x) {
    return MAX(1.0f - fabsf(x), 0.0f);
}

// Taps and normalised weights of the antialiased bilinear filter for one output along one axis, as
// the CPU computes them: source pixels [trunc(c - r + po), trunc(c + r + po)) around the centre
// c = (o + po) / sf with radius r, weighted by the triangle filter. w gets taps weights, those past
// the support 0. Returns the number of source pixels in the support.
static uint32_t upscale_antialias_axis(float o, float inv_sf, float pixel_offset, float radius, float invscale,
                                       uint32_t n, uint32_t taps, int32_t * first, float * w) {
    const float    c  = (o + pixel_offset) * inv_sf;
    const int32_t  lo = clamp_i32((int32_t) MAX(c - radius + pixel_offset, 0.0f), 0, (int32_t) n);
    const int32_t  hi = clamp_i32((int32_t) MIN(c + radius + pixel_offset, (float) n), 0, (int32_t) n);
    const uint32_t nt = hi > lo ? MIN((uint32_t) (hi - lo), taps) : 0;

    float sum = 0.0f;
    for (uint32_t k = 0; k < nt; ++k) {
        w[k] = upscale_triangle(((float) (lo + (int32_t) k) - c + pixel_offset) * invscale);
        sum += w[k];
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint32_t k = 0; k < taps; ++k) {
        w[k] = k < nt ? w[k] * inv : 0.0f;
    }
    *first = lo;
    return nt;
}

// Bicubic taps for one output along one axis, as the CPU computes them: the source pixels x0 - 1 ..
// x0 + 2 around c = (o + po) / sf - po, x0 = floor(c), clamped to the row. Clamped pixels repeat
// at the edges; their weights are added up so that the taps cover a contiguous range.
static uint32_t upscale_bicubic_axis(float o, float inv_sf, float pixel_offset, uint32_t n, uint32_t taps, int32_t * first,
                                     float * w) {
    const float   c  = (o + pixel_offset) * inv_sf - pixel_offset;
    const int32_t x0 = (int32_t) floorf(c);
    float         wk[4];
    bicubic_weights(c - (float) x0, wk);

    const int32_t lo = clamp_i32(x0 - 1, 0, (int32_t) n - 1);
    const int32_t hi = clamp_i32(x0 + 2, 0, (int32_t) n - 1);
    for (uint32_t k = 0; k < taps; ++k) {
        w[k] = 0.0f;
    }
    for (int32_t k = 0; k < 4; ++k) {
        w[clamp_i32(x0 - 1 + k, 0, (int32_t) n - 1) - lo] += wk[k];
    }
    *first = lo;
    return (uint32_t) (hi - lo + 1);
}

static inline uint32_t upscale_sep_axis(const struct htp_upscale_kernel_params * p, int axis, float o, uint32_t n,
                                        uint32_t taps, int32_t * first, float * w) {
    const float inv_sf = axis ? p->inv_sf1 : p->inv_sf0;
    if (p->mode == HTP_UPSCALE_MODE_BICUBIC) {
        return upscale_bicubic_axis(o, inv_sf, p->pixel_offset, n, taps, first, w);
    }
    return upscale_antialias_axis(o, inv_sf, p->pixel_offset, axis ? p->filter_radius1 : p->filter_radius0,
                                  axis ? p->invscale1 : p->invscale0, n, taps, first, w);
}

static void upscale_init_luts_separable(
    struct upscale_sep_x_tap * lut_x,
    struct upscale_sep_y * lut_y,
    float * lut_wy,
    const struct htp_tensor * src0,
    const struct htp_tensor * dst,
    const struct htp_upscale_kernel_params * p
) {
    const uint32_t src_x    = src0->ne[0];
    const uint32_t src_y    = src0->ne[1];
    const uint32_t dst_x    = dst->ne[0];
    const uint32_t dst_y    = dst->ne[1];
    const uint32_t taps0    = p->taps0;
    const uint32_t taps1    = p->taps1;
    const uint32_t n_chunks = (dst_x + VLEN_FP32 - 1) / VLEN_FP32;

    float w[taps0];

    // Taps past the support (and lanes past the row) read the first pixel with weight 0, so every
    // gather offset is inside the row
    for (uint32_t c = 0; c < n_chunks; ++c) {
        struct upscale_sep_x_tap * t = &lut_x[c * taps0];
        for (uint32_t l = 0; l < VLEN_FP32; ++l) {
            const uint32_t ox = c * VLEN_FP32 + l;
            int32_t  x0 = 0;
            uint32_t nt = 0;
            if (ox < dst_x) {
                nt = upscale_sep_axis(p, 0, (float) ox, src_x, taps0, &x0, w);
            }
            for (uint32_t k = 0; k < taps0; ++k) {
                const int32_t x = k < nt ? x0 + (int32_t) k : clamp_i32(x0, 0, (int32_t) src_x - 1);
                t[k].off[l] = x * (int32_t) sizeof(float);
                t[k].w[l]   = k < nt ? w[k] : 0.0f;
            }
        }
    }
    for (uint32_t oy = 0; oy < dst_y; ++oy) {
        int32_t        y0 = 0;
        const uint32_t nt = upscale_sep_axis(p, 1, (float) oy, src_y, taps1, &y0, lut_wy + oy * taps1);
        lut_y[oy].y_min = y0;
        lut_y[oy].y_max = y0 + (int32_t) nt;
    }
}

static inline void get_src_y_window(
    const struct htp_tensor * src0,
    const struct htp_tensor * dst,
    const struct htp_upscale_kernel_params * p,
    const void * lut_y,
    uint32_t dst_y_start,
    uint32_t dst_y_count,
    uint32_t * out_src_y_start,
    uint32_t * out_src_y_count
) {
    const uint32_t src_y = src0->ne[1];
    const uint32_t dst_y = dst->ne[1];

    if (dst_y_count == dst_y) {
        *out_src_y_start = 0;
        *out_src_y_count = src_y;
        return;
    }

    const uint32_t last_oy = dst_y_start + dst_y_count - 1;

    if (p->mode == HTP_UPSCALE_MODE_NEAREST) {
        const uint32_t * ly = (const uint32_t *) lut_y;
        const uint32_t y0 = ly[dst_y_start];
        const uint32_t y1 = ly[last_oy];
        *out_src_y_start = y0;
        *out_src_y_count = y1 - y0 + 1;
    } else if (htp_upscale_is_separable(p->mode, p->antialias)) {
        const struct upscale_sep_y * ly = (const struct upscale_sep_y *) lut_y;
        const int32_t y0 = ly[dst_y_start].y_min;
        const int32_t y1 = ly[last_oy].y_max;
        *out_src_y_start = (uint32_t) y0;
        *out_src_y_count = y1 > y0 ? (uint32_t) (y1 - y0) : 0;
    } else {
        const struct upscale_bilinear_y * ly = (const struct upscale_bilinear_y *) lut_y;
        const uint32_t y0 = ly[dst_y_start].y0;
        const uint32_t y1 = ly[last_oy].y1;
        *out_src_y_start = y0;
        *out_src_y_count = y1 - y0 + 1;
    }
}

static inline void resolve_dst_slice(
    const struct htp_tensor * dst,
    const struct htp_upscale_kernel_params * p,
    uint32_t chunk_idx,
    dma_addr_t * out_slice_dst_off,
    uint32_t * out_dst_ys,
    uint32_t * out_dst_yc
) {
    const uint32_t s_idx = fastdiv(chunk_idx, &p->div_chunks_per_slice);
    const uint32_t c_idx = chunk_idx - s_idx * p->chunks_per_slice;
    const uint32_t d_ys  = c_idx * p->dst_nrows_per_chunk;
    const uint32_t d_yc  = MIN(p->dst_nrows_per_chunk, dst->ne[1] - d_ys);
    const uint32_t i3    = fastdiv(s_idx, &p->div_dst_z);
    const uint32_t i2    = s_idx - i3 * dst->ne[2];

    *out_slice_dst_off = (dma_addr_t) i2 * dst->nb[2] + (dma_addr_t) i3 * dst->nb[3];
    *out_dst_ys        = d_ys;
    *out_dst_yc        = d_yc;
}

static inline void resolve_src_slice(
    const struct htp_tensor * src0,
    const struct htp_tensor * dst,
    const struct htp_upscale_kernel_params * p,
    const void * lut_y,
    uint32_t chunk_idx,
    dma_addr_t * out_slice_src_off,
    uint32_t * out_src_ys,
    uint32_t * out_src_yc
) {
    const uint32_t s_idx = fastdiv(chunk_idx, &p->div_chunks_per_slice);
    const uint32_t c_idx = chunk_idx - s_idx * p->chunks_per_slice;
    const uint32_t d_ys  = c_idx * p->dst_nrows_per_chunk;
    const uint32_t d_yc  = MIN(p->dst_nrows_per_chunk, dst->ne[1] - d_ys);
    uint32_t s_ys, s_yc;
    get_src_y_window(src0, dst, p, lut_y, d_ys, d_yc, &s_ys, &s_yc);

    const uint32_t i3  = fastdiv(s_idx, &p->div_dst_z);
    const uint32_t i2  = s_idx - i3 * dst->ne[2];
    const uint32_t i03 = (uint32_t) ((float) i3 * p->inv_sf3);
    const uint32_t i02 = (uint32_t) ((float) i2 * p->inv_sf2);

    *out_slice_src_off = (dma_addr_t) i02 * src0->nb[2] + (dma_addr_t) i03 * src0->nb[3];
    *out_src_ys        = s_ys;
    *out_src_yc        = s_yc;
}

struct upscale_context;

typedef void (*upscale_chunk_fn_t)(
    const struct upscale_context * ctx,
    const float * srcb,
    float * dstb,
    float * R_buf,
    float * gather_tmp,
    uint32_t dst_y_start,
    uint32_t dst_y_count,
    uint32_t src_y_start
);

struct upscale_context {
    struct htp_ops_context * octx;
    const struct htp_upscale_kernel_params * kparams;
    struct htp_upscale_vtcm_layout layout;
    upscale_chunk_fn_t compute_chunk;
    const void * lut_x;
    const void * lut_y;
    const float * lut_wy;
    uint32_t chunk_start;
    uint32_t chunk_count;
    uint32_t chunks_per_hvx;
};

static void compute_nearest_chunk(
    const struct upscale_context * ctx,
    const float * srcb,
    float * dstb,
    float * R_buf,
    float * gather_tmp,
    uint32_t dst_y_start,
    uint32_t dst_y_count,
    uint32_t src_y_start
) {
    const uint32_t src_x = ctx->octx->src[0]->ne[0];
    const uint32_t dst_x = ctx->octx->dst->ne[0];
    const uint32_t * lut_x = (const uint32_t *) ctx->lut_x;
    const uint32_t * lut_y = (const uint32_t *) ctx->lut_y;
    const uint32_t region_size = hex_round_up(src_x * sizeof(float), 128);

    for (uint32_t oy = dst_y_start; oy < dst_y_start + dst_y_count; ++oy) {
        const uint32_t y0 = lut_y[oy];
        const float * r0  = srcb + (y0 - src_y_start) * src_x;
        float * dst_row   = dstb + (oy - dst_y_start) * dst_x;

        for (uint32_t ix = 0; ix < src_x; ix += VLEN_FP32) {
            *(HVX_Vector *) (R_buf + ix) = hvx_vmemu(r0 + ix);
        }

        for (uint32_t ox = 0; ox < dst_x; ox += VLEN_FP32) {
            const uint32_t rem = dst_x - ox;
            const uint32_t nbytes = (rem < VLEN_FP32) ? (rem * sizeof(float)) : VLEN;
            const HVX_Vector voff = hvx_vmem(lut_x + ox);
            Q6_vgather_ARMVw((HVX_Vector *) gather_tmp, (size_t) R_buf, region_size, voff);
            hvx_gather_sync(gather_tmp);
            hvx_vec_store_u(dst_row + ox, nbytes, hvx_vmem(gather_tmp));
        }
    }
}

static void compute_nearest_2x_chunk(
    const struct upscale_context * ctx,
    const float * srcb,
    float * dstb,
    float * R_buf,
    float * gather_tmp,
    uint32_t dst_y_start,
    uint32_t dst_y_count,
    uint32_t src_y_start
) {
    (void) R_buf;
    (void) gather_tmp;
    const uint32_t src_x = ctx->octx->src[0]->ne[0];
    const uint32_t dst_x = ctx->octx->dst->ne[0];

    for (uint32_t oy = dst_y_start; oy < dst_y_start + dst_y_count; ++oy) {
        const uint32_t sy = oy >> 1;
        const float * r0  = srcb + (sy - src_y_start) * src_x;
        float * dst_row   = dstb + (oy - dst_y_start) * dst_x;

        for (uint32_t ix = 0; ix < src_x; ix += VLEN_FP32) {
            const HVX_Vector vin = hvx_vmemu(r0 + ix);

            const HVX_VectorPair p = Q6_W_vshuff_VVR(vin, vin, -4);
            const HVX_Vector v_lo = Q6_V_lo_W(p);
            const HVX_Vector v_hi = Q6_V_hi_W(p);

            const uint32_t ox0 = 2 * ix;
            const uint32_t rem_out0 = (dst_x > ox0) ? (dst_x - ox0) : 0;
            const uint32_t nbytes_out0 = (rem_out0 < VLEN_FP32) ? (rem_out0 * sizeof(float)) : VLEN;
            hvx_vec_store_u(dst_row + ox0, nbytes_out0, v_lo);

            const uint32_t ox1 = ox0 + VLEN_FP32;
            if (dst_x > ox1) {
                const uint32_t rem_out1 = dst_x - ox1;
                const uint32_t nbytes_out1 = (rem_out1 < VLEN_FP32) ? (rem_out1 * sizeof(float)) : VLEN;
                hvx_vec_store_u(dst_row + ox1, nbytes_out1, v_hi);
            }
        }
    }
}

static void compute_nearest_4x_chunk(
    const struct upscale_context * ctx,
    const float * srcb,
    float * dstb,
    float * R_buf,
    float * gather_tmp,
    uint32_t dst_y_start,
    uint32_t dst_y_count,
    uint32_t src_y_start
) {
    (void) R_buf;
    (void) gather_tmp;
    const uint32_t src_x = ctx->octx->src[0]->ne[0];
    const uint32_t dst_x = ctx->octx->dst->ne[0];

    for (uint32_t oy = dst_y_start; oy < dst_y_start + dst_y_count; ++oy) {
        const uint32_t sy = oy >> 2;
        const float * r0  = srcb + (sy - src_y_start) * src_x;
        float * dst_row   = dstb + (oy - dst_y_start) * dst_x;

        for (uint32_t ix = 0; ix < src_x; ix += VLEN_FP32) {
            const HVX_Vector vin = hvx_vmemu(r0 + ix);

            const HVX_VectorPair p = Q6_W_vshuff_VVR(vin, vin, -4);
            const HVX_Vector v_lo = Q6_V_lo_W(p);
            const HVX_Vector v_hi = Q6_V_hi_W(p);

            const HVX_VectorPair p0 = Q6_W_vshuff_VVR(v_lo, v_lo, -4);
            const HVX_VectorPair p1 = Q6_W_vshuff_VVR(v_hi, v_hi, -4);

            const HVX_Vector out0 = Q6_V_lo_W(p0);
            const HVX_Vector out1 = Q6_V_hi_W(p0);
            const HVX_Vector out2 = Q6_V_lo_W(p1);
            const HVX_Vector out3 = Q6_V_hi_W(p1);

            const uint32_t ox = 4 * ix;
            if (dst_x > ox) {
                const uint32_t rem = dst_x - ox;
                hvx_vec_store_u(dst_row + ox, (rem < VLEN_FP32) ? (rem * sizeof(float)) : VLEN, out0);
            }
            if (dst_x > ox + VLEN_FP32) {
                const uint32_t rem = dst_x - (ox + VLEN_FP32);
                hvx_vec_store_u(dst_row + ox + VLEN_FP32, (rem < VLEN_FP32) ? (rem * sizeof(float)) : VLEN, out1);
            }
            if (dst_x > ox + 2 * VLEN_FP32) {
                const uint32_t rem = dst_x - (ox + 2 * VLEN_FP32);
                hvx_vec_store_u(dst_row + ox + 2 * VLEN_FP32, (rem < VLEN_FP32) ? (rem * sizeof(float)) : VLEN, out2);
            }
            if (dst_x > ox + 3 * VLEN_FP32) {
                const uint32_t rem = dst_x - (ox + 3 * VLEN_FP32);
                hvx_vec_store_u(dst_row + ox + 3 * VLEN_FP32, (rem < VLEN_FP32) ? (rem * sizeof(float)) : VLEN, out3);
            }
        }
    }
}

static inline void compute_bilinear_chunk_32(
    float * restrict dst_chunk,
    const float * restrict R_buf,
    float * restrict tmp_a,
    float * restrict tmp_b,
    uint32_t region_size,
    const struct upscale_bilinear_x_chunk * restrict chunk,
    uint32_t nbytes
) {
    const HVX_Vector voff0 = hvx_vmem(chunk->off0);
    const HVX_Vector voff1 = hvx_vmem(chunk->off1);
    const HVX_Vector vw0   = hvx_vmem(chunk->w0);
    const HVX_Vector vdx   = hvx_vmem(chunk->dx);

    Q6_vgather_ARMVw((HVX_Vector *) tmp_a, (size_t) R_buf, region_size, voff0);
    Q6_vgather_ARMVw((HVX_Vector *) tmp_b, (size_t) R_buf, region_size, voff1);
    hvx_gather_sync(tmp_b);

    const HVX_Vector va = hvx_vmem(tmp_a);
    const HVX_Vector vb = hvx_vmem(tmp_b);

    const HVX_Vector val = hvx_vec_add_f32_f32(
        hvx_vec_mul_f32_f32(vw0, va),
        hvx_vec_mul_f32_f32(vdx, vb)
    );

    hvx_vec_store_u(dst_chunk, nbytes, val);
}

static void compute_bilinear_chunk(
    const struct upscale_context * ctx,
    const float * srcb,
    float * dstb,
    float * R_buf,
    float * gather_tmp,
    uint32_t dst_y_start,
    uint32_t dst_y_count,
    uint32_t src_y_start
) {
    const uint32_t src_x = ctx->octx->src[0]->ne[0];
    const uint32_t dst_x = ctx->octx->dst->ne[0];
    const struct upscale_bilinear_x_chunk * lut_x = (const struct upscale_bilinear_x_chunk *) ctx->lut_x;
    const struct upscale_bilinear_y * lut_y = (const struct upscale_bilinear_y *) ctx->lut_y;
    const uint32_t n_chunks = (dst_x + VLEN_FP32 - 1) / VLEN_FP32;
    const uint32_t region_size = hex_round_up(src_x * sizeof(float), 128);
    float * tmp_a = gather_tmp;
    float * tmp_b = gather_tmp + VLEN_FP32;

    for (uint32_t oy = dst_y_start; oy < dst_y_start + dst_y_count; ++oy) {
        const struct upscale_bilinear_y * ey = &lut_y[oy];
        const float * r0 = srcb + (ey->y0 - src_y_start) * src_x;
        const float * r1 = srcb + (ey->y1 - src_y_start) * src_x;
        float * dst_row  = dstb + (oy - dst_y_start) * dst_x;

        if (ey->y0 == ey->y1) {
            for (uint32_t ix = 0; ix < src_x; ix += VLEN_FP32) {
                *(HVX_Vector *) (R_buf + ix) = hvx_vmemu(r0 + ix);
            }
        } else {
            const HVX_Vector vwy0 = hvx_vec_splat_f32(ey->wy0);
            const HVX_Vector vwy1 = hvx_vec_splat_f32(ey->wy1);
            for (uint32_t ix = 0; ix < src_x; ix += VLEN_FP32) {
                const HVX_Vector vr0 = hvx_vmemu(r0 + ix);
                const HVX_Vector vr1 = hvx_vmemu(r1 + ix);
                *(HVX_Vector *) (R_buf + ix) = hvx_vec_add_f32_f32(
                    hvx_vec_mul_f32_f32(vwy0, vr0),
                    hvx_vec_mul_f32_f32(vwy1, vr1)
                );
            }
        }

        for (uint32_t c = 0; c < n_chunks; ++c) {
            const uint32_t ox = c * VLEN_FP32;
            const uint32_t rem = dst_x - ox;
            const uint32_t nbytes = (rem < VLEN_FP32) ? (rem * sizeof(float)) : VLEN;
            compute_bilinear_chunk_32(dst_row + ox, R_buf, tmp_a, tmp_b, region_size, &lut_x[c], nbytes);
        }
    }
}

static void compute_bilinear_2x_chunk(
    const struct upscale_context * ctx,
    const float * srcb,
    float * dstb,
    float * R_buf,
    float * gather_tmp,
    uint32_t dst_y_start,
    uint32_t dst_y_count,
    uint32_t src_y_start
) {
    (void) R_buf;
    (void) gather_tmp;
    const uint32_t src_x = ctx->octx->src[0]->ne[0];
    const uint32_t dst_x = ctx->octx->dst->ne[0];
    const struct upscale_bilinear_y * lut_y = (const struct upscale_bilinear_y *) ctx->lut_y;
    const HVX_Vector c_075 = hvx_vec_splat_f32(0.75f);
    const HVX_Vector c_025 = hvx_vec_splat_f32(0.25f);

    for (uint32_t oy = dst_y_start; oy < dst_y_start + dst_y_count; ++oy) {
        const struct upscale_bilinear_y * ey = &lut_y[oy];
        const float * r0 = srcb + (ey->y0 - src_y_start) * src_x;
        const float * r1 = srcb + (ey->y1 - src_y_start) * src_x;
        float * dst_row  = dstb + (oy - dst_y_start) * dst_x;

        const bool same_row = (ey->y0 == ey->y1);
        const HVX_Vector vwy0 = hvx_vec_splat_f32(ey->wy0);
        const HVX_Vector vwy1 = hvx_vec_splat_f32(ey->wy1);

        HVX_Vector v_curr = hvx_vmem(r0);
        if (!same_row) {
            const HVX_Vector v1 = hvx_vmem(r1);
            v_curr = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(vwy0, v_curr), hvx_vec_mul_f32_f32(vwy1, v1));
        }

        HVX_Vector v_prev = Q6_V_vror_VR(v_curr, 4);

        for (uint32_t ix = 0; ix < src_x; ix += VLEN_FP32) {
            HVX_Vector v_next_chunk;
            const uint32_t next_ix = ix + VLEN_FP32;
            if (next_ix < src_x) {
                v_next_chunk = hvx_vmem(r0 + next_ix);
                if (!same_row) {
                    const HVX_Vector v1 = hvx_vmem(r1 + next_ix);
                    v_next_chunk = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(vwy0, v_next_chunk), hvx_vec_mul_f32_f32(vwy1, v1));
                }
            } else {
                v_next_chunk = Q6_V_vror_VR(v_curr, 124);
            }

            const HVX_Vector v_shl = Q6_V_valign_VVR(v_curr, v_prev, 124);
            const HVX_Vector v_shr = Q6_V_valign_VVR(v_next_chunk, v_curr, 4);

            const HVX_Vector ve = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(c_075, v_curr), hvx_vec_mul_f32_f32(c_025, v_shl));
            const HVX_Vector vo = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(c_075, v_curr), hvx_vec_mul_f32_f32(c_025, v_shr));

            const HVX_VectorPair p = Q6_W_vshuff_VVR(vo, ve, -4);
            const HVX_Vector out0 = Q6_V_lo_W(p);
            const HVX_Vector out1 = Q6_V_hi_W(p);

            const uint32_t ox0 = 2 * ix;
            hvx_vmem(dst_row + ox0)             = out0;
            hvx_vmem(dst_row + ox0 + VLEN_FP32) = out1;

            v_prev = v_curr;
            v_curr = v_next_chunk;
        }
    }
}

// Separable: the source rows of an output row are summed with their weights into R_buf (HVX), then
// every 32 outputs gather their taps from R_buf and sum them with the weights along x.
static void compute_separable_chunk(
    const struct upscale_context * ctx,
    const float * srcb,
    float * dstb,
    float * R_buf,
    float * gather_tmp,
    uint32_t dst_y_start,
    uint32_t dst_y_count,
    uint32_t src_y_start
) {
    const uint32_t src_x = ctx->octx->src[0]->ne[0];
    const uint32_t dst_x = ctx->octx->dst->ne[0];
    const uint32_t taps0 = ctx->kparams->taps0;
    const uint32_t taps1 = ctx->kparams->taps1;
    const struct upscale_sep_x_tap * lut_x = (const struct upscale_sep_x_tap *) ctx->lut_x;
    const struct upscale_sep_y * lut_y = (const struct upscale_sep_y *) ctx->lut_y;
    const uint32_t n_chunks    = (dst_x + VLEN_FP32 - 1) / VLEN_FP32;
    const uint32_t region_size = hex_round_up(src_x * sizeof(float), 128);
    float * tmp_a = gather_tmp;
    float * tmp_b = gather_tmp + VLEN_FP32;

    for (uint32_t oy = dst_y_start; oy < dst_y_start + dst_y_count; ++oy) {
        const struct upscale_sep_y * ey = &lut_y[oy];
        const float * wy      = ctx->lut_wy + oy * taps1;
        const int32_t ny      = ey->y_max - ey->y_min;
        float *       dst_row = dstb + (oy - dst_y_start) * dst_x;

        if (ny <= 0) { // no source row in the antialiasing support: the CPU writes 0
            for (uint32_t ox = 0; ox < dst_x; ox += VLEN_FP32) {
                const uint32_t rem = dst_x - ox;
                hvx_vec_store_u(dst_row + ox, rem < VLEN_FP32 ? rem * sizeof(float) : VLEN, Q6_V_vzero());
            }
            continue;
        }

        const float * r0 = srcb + (uint32_t) (ey->y_min - (int32_t) src_y_start) * src_x;
        const HVX_Vector vw0 = hvx_vec_splat_f32(wy[0]);
        for (uint32_t ix = 0; ix < src_x; ix += VLEN_FP32) {
            *(HVX_Vector *) (R_buf + ix) = hvx_vec_mul_f32_f32(vw0, hvx_vmemu(r0 + ix));
        }
        for (int32_t k = 1; k < ny; ++k) {
            const HVX_Vector vw = hvx_vec_splat_f32(wy[k]);
            const float *    r  = r0 + (uint32_t) k * src_x;
            for (uint32_t ix = 0; ix < src_x; ix += VLEN_FP32) {
                HVX_Vector * acc = (HVX_Vector *) (R_buf + ix);
                *acc = hvx_vec_add_f32_f32(*acc, hvx_vec_mul_f32_f32(vw, hvx_vmemu(r + ix)));
            }
        }

        for (uint32_t c = 0; c < n_chunks; ++c) {
            const struct upscale_sep_x_tap * t = &lut_x[c * taps0];
            const uint32_t ox  = c * VLEN_FP32;
            const uint32_t rem = dst_x - ox;
            HVX_Vector     acc = Q6_V_vzero();

            // two gathers in flight per sync
            for (uint32_t k = 0; k < taps0; k += 2) {
                const bool pair = k + 1 < taps0;
                Q6_vgather_ARMVw((HVX_Vector *) tmp_a, (size_t) R_buf, region_size, hvx_vmem(t[k].off));
                if (pair) {
                    Q6_vgather_ARMVw((HVX_Vector *) tmp_b, (size_t) R_buf, region_size, hvx_vmem(t[k + 1].off));
                    hvx_gather_sync(tmp_b);
                } else {
                    hvx_gather_sync(tmp_a);
                }
                const HVX_Vector va = hvx_vec_mul_f32_f32(hvx_vmem(t[k].w), hvx_vmem(tmp_a));
                acc = k == 0 ? va : hvx_vec_add_f32_f32(acc, va);
                if (pair) {
                    acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(hvx_vmem(t[k + 1].w), hvx_vmem(tmp_b)));
                }
            }

            hvx_vec_store_u(dst_row + ox, rem < VLEN_FP32 ? rem * sizeof(float) : VLEN, acc);
        }
    }
}

static inline void dma_push_src_chunk(
    dma_queue * dma_q,
    float * spad,
    const struct htp_tensor * src0,
    dma_addr_t slice_src_off,
    uint32_t src_ys,
    uint32_t src_yc
) {
    const uint32_t src_x = src0->ne[0];
    const dma_addr_t src_addr = src0->data + slice_src_off + (dma_addr_t) src_ys * src0->nb[1];

    if (src0->nb[0] == sizeof(float)) {
        const uint32_t row_size   = src_x * sizeof(float);
        const uint32_t dst_stride = row_size;
        const uint32_t src_stride = src0->nb[1];
        dma_queue_push(dma_q, dma_make_data(spad, src_addr), dst_stride, src_stride, row_size, src_yc);
    } else {
        const uint32_t row_size   = sizeof(float);
        const uint32_t dst_stride = sizeof(float);
        const uint32_t src_stride = src0->nb[0];
        const uint32_t nrows      = src_x * src_yc;
        dma_queue_push(dma_q, dma_make_data(spad, src_addr), dst_stride, src_stride, row_size, nrows);
    }
}

struct upscale_pipe_state {
    struct upscale_context *                   ctx;
    const struct htp_upscale_kernel_params *   p;
    const struct htp_tensor *                  src0;
    const struct htp_tensor *                  dst;
    const void *                               lut_y;
    float *                                    R_buf;
    float *                                    gather_tmp;
    uint32_t                                   dst_row_size;
    uint32_t                                   dst_stride;
    uint32_t                                   chunk;
    uint32_t                                   chunk_end;
};

static bool upscale_next_task(void * ctx, struct dma_pipe_task * task) {
    struct upscale_pipe_state * s = (struct upscale_pipe_state *) ctx;
    if (s->chunk >= s->chunk_end) {
        return false;
    }
    task->row = s->chunk++;
    task->off = 0;
    task->len = 0;
    task->aux = 0;
    return true;
}

static void upscale_push_in(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * in_spad) {
    struct upscale_pipe_state * s = (struct upscale_pipe_state *) ctx;
    dma_addr_t slice_src_off;
    uint32_t src_ys, src_yc;
    resolve_src_slice(s->src0, s->dst, s->p, s->lut_y, task->row, &slice_src_off, &src_ys, &src_yc);
    dma_push_src_chunk(dma_q, (float *) in_spad, s->src0, slice_src_off, src_ys, src_yc);
}

static void upscale_compute(void * ctx, const struct dma_pipe_task * task, void * out_spad, void * in_spad) {
    struct upscale_pipe_state * s = (struct upscale_pipe_state *) ctx;
    dma_addr_t slice_dst_off;
    uint32_t dst_ys, dst_yc;
    resolve_dst_slice(s->dst, s->p, task->row, &slice_dst_off, &dst_ys, &dst_yc);

    uint32_t src_ys, src_yc;
    get_src_y_window(s->src0, s->dst, s->p, s->lut_y, dst_ys, dst_yc, &src_ys, &src_yc);

    s->ctx->compute_chunk(s->ctx, (const float *) in_spad, (float *) out_spad, s->R_buf, s->gather_tmp, dst_ys, dst_yc, src_ys);
}

static void upscale_push_out(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * out_spad) {
    struct upscale_pipe_state * s = (struct upscale_pipe_state *) ctx;
    dma_addr_t slice_dst_off;
    uint32_t dst_ys, dst_yc;
    resolve_dst_slice(s->dst, s->p, task->row, &slice_dst_off, &dst_ys, &dst_yc);

    const dma_addr_t dst_addr = s->dst->data + slice_dst_off + (dma_addr_t) dst_ys * s->dst_stride;
    dma_queue_push(dma_q, dma_make_data(dst_addr, out_spad), s->dst_stride, s->dst_row_size, s->dst_row_size, dst_yc);
}

static void upscale_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct upscale_context * ctx = (struct upscale_context *) data;
    const struct htp_upscale_kernel_params * p = ctx->kparams;
    const struct htp_upscale_vtcm_layout * L = &ctx->layout;
    const struct htp_tensor * src0 = ctx->octx->src[0];
    const struct htp_tensor * dst  = ctx->octx->dst;

    const uint32_t first = ctx->chunk_start + ith * ctx->chunks_per_hvx;
    const uint32_t last  = MIN(first + ctx->chunks_per_hvx, ctx->chunk_start + ctx->chunk_count);

    if (first >= last) {
        return;
    }

    const uint32_t src_spad_half = (uint32_t) L->src_spad_half_size;
    const uint32_t dst_spad_half = (uint32_t) L->dst_spad_half_size;

    uint8_t * vtcm_base  = (uint8_t *) ctx->octx->ctx->vtcm_base;
    uint8_t * src_spad   = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, L->off_src)    + ith * L->src_bytes_per_hvx;
    uint8_t * dst_spad   = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, L->off_dst)    + ith * L->dst_bytes_per_hvx;
    void *    in_buf[2]  = { src_spad, src_spad + src_spad_half };
    void *    out_buf[2] = { dst_spad, dst_spad + dst_spad_half };

    struct upscale_pipe_state s = {
        .ctx          = ctx,
        .p            = p,
        .src0         = src0,
        .dst          = dst,
        .lut_y        = ctx->lut_y,
        .R_buf        = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_r)      + ith * (L->r_bytes_per_hvx / sizeof(float)),
        .gather_tmp   = VTCM_LAYOUT_PTR(float, vtcm_base, L->off_gather) + ith * (L->gather_bytes_per_hvx / sizeof(float)),
        .dst_row_size = dst->ne[0] * sizeof(float),
        .dst_stride   = dst->nb[1],
        .chunk        = first,
        .chunk_end    = last,
    };

    dma_pipe_io_run(th->dma_q, th->trace, &s, in_buf, out_buf,
                    upscale_next_task, upscale_push_in, upscale_compute, upscale_push_out);
}

int op_upscale(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;
    const struct htp_upscale_kernel_params * p =
        (const struct htp_upscale_kernel_params *) octx->kernel_params;

    const uint32_t src_x = src0->ne[0];
    const uint32_t src_y = src0->ne[1];
    const uint32_t dst_x = dst->ne[0];
    const uint32_t dst_y = dst->ne[1];

    uint32_t chunk_start = 0;
    uint32_t chunk_count = p->total_chunks;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            chunk_count, 1,
            octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        chunk_start = range.start;
        chunk_count = range.count;
    }

    if (chunk_count == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_hvx = MIN(p->n_hvx, chunk_count);
    if (!htp_ops_context_set_n_hvx(octx, n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    size_t lut_x_size, lut_y_size, lut_wy_size;
    htp_upscale_lut_sizes(p->mode, p->antialias, dst_x, dst_y, p->taps0, p->taps1, &lut_x_size, &lut_y_size, &lut_wy_size);
    if (lut_x_size + lut_y_size + lut_wy_size > octx->ctx->ddr_spad_size) {
        return HTP_STATUS_NO_SUPPORT;
    }

    // The tables are in DDR: they stay valid when VTCM is released between rounds
    uint8_t * base_scratch = (uint8_t *) octx->ctx->ddr_spad_base;
    void *    lut_x        = base_scratch;
    void *    lut_y        = base_scratch + lut_x_size;
    float *   lut_wy       = (float *) (base_scratch + lut_x_size + lut_y_size);

    if (p->mode == HTP_UPSCALE_MODE_NEAREST) {
        upscale_init_luts_nearest((uint32_t *) lut_x, (uint32_t *) lut_y, src0, dst, p);
    } else if (htp_upscale_is_separable(p->mode, p->antialias)) {
        upscale_init_luts_separable((struct upscale_sep_x_tap *) lut_x, (struct upscale_sep_y *) lut_y, lut_wy, src0, dst, p);
    } else {
        upscale_init_luts_bilinear((struct upscale_bilinear_x_chunk *) lut_x, (struct upscale_bilinear_y *) lut_y, src0, dst, p);
    }

    upscale_chunk_fn_t compute_chunk;
    switch (p->kernel_type) {
        case HTP_UPSCALE_KERNEL_NEAREST:
            compute_chunk = compute_nearest_chunk;
            break;
        case HTP_UPSCALE_KERNEL_NEAREST_2X:
            compute_chunk = compute_nearest_2x_chunk;
            break;
        case HTP_UPSCALE_KERNEL_NEAREST_4X:
            compute_chunk = compute_nearest_4x_chunk;
            break;
        case HTP_UPSCALE_KERNEL_BILINEAR:
            compute_chunk = compute_bilinear_chunk;
            break;
        case HTP_UPSCALE_KERNEL_BILINEAR_2X:
            compute_chunk = compute_bilinear_2x_chunk;
            break;
        case HTP_UPSCALE_KERNEL_BILINEAR_AA:
        case HTP_UPSCALE_KERNEL_BICUBIC:
            compute_chunk = compute_separable_chunk;
            break;
        default:
            return HTP_STATUS_INVAL_PARAMS;
    }

    struct upscale_context ctx = {
        .octx               = octx,
        .kparams            = p,
        .compute_chunk      = compute_chunk,
        .lut_x              = lut_x,
        .lut_y              = lut_y,
        .lut_wy             = lut_wy,
    };

    htp_upscale_vtcm_layout_build(&ctx.layout, src_x, src_y, dst_x, dst_y,
                                  p->dst_nrows_per_chunk, p->inv_sf1, p->filter_radius1, n_hvx);

#ifdef HTP_YIELD_IN_OPS
    // Rounds of chunks, sized on the host from the kernel's time per output, so VTCM/HMX can be
    // given away between them when a release is requested. Every chunk stages its own rows.
    const uint32_t round = p->chunks_per_round ? MAX(p->chunks_per_round, n_hvx) : chunk_count;
#else
    const uint32_t round = chunk_count;
#endif
    for (uint32_t done = 0; done < chunk_count;) {
        const uint32_t n   = MIN(round, chunk_count - done);
        ctx.chunk_start    = chunk_start + done;
        ctx.chunk_count    = n;
        ctx.chunks_per_hvx = fastdiv(n + n_hvx - 1, &octx->n_hvx_div);
        work_queue_run(octx->ctx->hvx_queue, upscale_thread, &ctx, n_hvx);
        done += n;
#ifdef HTP_YIELD_IN_OPS
        if (done < chunk_count && atomic_load(&octx->ctx->vtcm_needs_release)) {
            htp_vtcm_yield(octx->ctx);
        }
#endif
    }

    return HTP_STATUS_OK;
}
