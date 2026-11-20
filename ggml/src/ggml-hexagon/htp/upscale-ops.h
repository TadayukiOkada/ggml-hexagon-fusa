#ifndef HTP_UPSCALE_OPS_H
#define HTP_UPSCALE_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>

#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-vtcm.h"

enum htp_upscale_mode {
    HTP_UPSCALE_MODE_NEAREST  = 0,
    HTP_UPSCALE_MODE_BILINEAR = 1,
    HTP_UPSCALE_MODE_BICUBIC  = 2,
};

enum htp_upscale_kernel_type {
    HTP_UPSCALE_KERNEL_NEAREST     = 0,
    HTP_UPSCALE_KERNEL_NEAREST_2X  = 1,
    HTP_UPSCALE_KERNEL_NEAREST_4X  = 2,
    HTP_UPSCALE_KERNEL_BILINEAR    = 3,
    HTP_UPSCALE_KERNEL_BILINEAR_2X = 4,
    HTP_UPSCALE_KERNEL_BILINEAR_AA = 5,
    HTP_UPSCALE_KERNEL_BICUBIC     = 6,
};

struct htp_upscale_kernel_params {
    uint8_t  kernel_type;
    uint8_t  mode;
    uint8_t  align_corners;
    uint8_t  antialias;

    float    inv_sf0;
    float    inv_sf1;
    float    inv_sf2;
    float    inv_sf3;
    float    pixel_offset;

    float    filter_radius0;
    float    filter_radius1;
    float    invscale0;
    float    invscale1;

    uint32_t n_hvx;
    uint32_t chunks_per_group;

    uint32_t dst_nrows_per_chunk;
    uint32_t chunks_per_slice;
    uint32_t total_chunks;
    uint32_t vtcm_size;

    // Chunks per round: the op releases VTCM/HMX between rounds on request (skel built with
    // yield inside ops). Chosen on the host from the kernel's time per output; 0 = one round.
    uint32_t chunks_per_round;
    // Separable filters (bicubic, antialiased bilinear): taps per output along x and y
    uint16_t taps0;
    uint16_t taps1;

    struct fastdiv_values div_chunks_per_slice;
    struct fastdiv_values div_dst_z;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_upscale_kernel_params) <= 128, "htp_upscale_kernel_params is too large");
#else
_Static_assert(sizeof(struct htp_upscale_kernel_params) <= 128, "htp_upscale_kernel_params is too large");
#endif

struct __attribute__((aligned(128))) upscale_bilinear_x_chunk {
    int32_t  off0[32];
    int32_t  off1[32];
    float    w0[32];
    float    dx[32];
};

struct upscale_bilinear_y {
    uint32_t y0;
    uint32_t y1;
    float    wy0;
    float    wy1;
};

// Antialiased bilinear and bicubic run as separable filters. Bicubic: the CPU sums each of the 4
// source rows along x with the cubic weights and then the 4 row sums along y. Antialiased bilinear
// (triangle filter): the weight of a source pixel is wx * wy and the normalisation
// sum(wx * wy) = sum(wx) * sum(wy), so weights normalised per axis give the same sums. An output is
// a weighted sum along y of contiguous source rows, each a weighted sum along x.
// Along x, 32 outputs keep taps0 pairs of vectors: byte offsets into the row and weights (0 for
// taps past the support).
struct __attribute__((aligned(128))) upscale_sep_x_tap {
    int32_t off[32];
    float   w[32];
};

// Along y: the source rows [y_min, y_max) of one output row; their weights are taps1 floats per
// output row in a separate table.
struct upscale_sep_y {
    int32_t y_min;
    int32_t y_max;
};

#define HTP_UPSCALE_BICUBIC_TAPS 4

static inline bool htp_upscale_is_separable(uint32_t mode, bool antialias) {
    return mode == HTP_UPSCALE_MODE_BICUBIC || (mode == HTP_UPSCALE_MODE_BILINEAR && antialias);
}

// Taps of the antialiased filter along one axis: the source range [trunc(c - r), trunc(c + r)) of a
// support radius r holds at most ceil(2r) + 1 pixels.
static inline uint32_t htp_upscale_antialias_taps(float filter_radius) {
    return (uint32_t) ceilf(2.0f * filter_radius) + 1;
}

// Sizes of the lookup tables in the DDR scratch: along x, along y and, for the separable filters,
// the weights along y. The host checks them against HTP_DDR_SPAD_SIZE, the DSP lays them out.
static inline void htp_upscale_lut_sizes(uint32_t mode, bool antialias, uint32_t dst_x, uint32_t dst_y, uint32_t taps0,
                                         uint32_t taps1, size_t * lut_x_size, size_t * lut_y_size, size_t * lut_wy_size) {
    const uint32_t n_chunks = (dst_x + 31) / 32;
    size_t x = 0, elem_y = sizeof(uint32_t), wy = 0;
    if (htp_upscale_is_separable(mode, antialias)) {
        x      = (size_t) n_chunks * taps0 * sizeof(struct upscale_sep_x_tap);
        elem_y = sizeof(struct upscale_sep_y);
        wy     = (size_t) dst_y * taps1 * sizeof(float);
    } else if (mode == HTP_UPSCALE_MODE_BILINEAR) {
        x      = n_chunks * sizeof(struct upscale_bilinear_x_chunk);
        elem_y = sizeof(struct upscale_bilinear_y);
    } else {
        x = dst_x * sizeof(uint32_t);
    }
    *lut_x_size  = hex_round_up(x, 128);
    *lut_y_size  = hex_round_up(dst_y * elem_y, 128);
    *lut_wy_size = hex_round_up(wy, 128);
}

struct htp_upscale_vtcm_layout {
    size_t   total_bytes;
    size_t   off_src;
    size_t   off_dst;
    size_t   off_r;
    size_t   off_gather;
    size_t   src_bytes_per_hvx;
    size_t   dst_bytes_per_hvx;
    size_t   r_bytes_per_hvx;
    size_t   gather_bytes_per_hvx;
    size_t   src_spad_half_size;
    size_t   dst_spad_half_size;
    uint32_t max_src_chunk_rows;
    uint32_t dst_chunk_rows;
};

static inline void htp_upscale_vtcm_layout_build(
    struct htp_upscale_vtcm_layout * L,
    uint32_t src_x,
    uint32_t src_y,
    uint32_t dst_x,
    uint32_t dst_y,
    uint32_t dst_chunk_rows,
    float    inv_sf1,
    float    filter_radius1,
    uint32_t n_hvx
) {
    uint32_t max_src_rows;
    if (dst_chunk_rows >= dst_y) {
        max_src_rows = src_y;
    } else {
        const float est = (float) dst_chunk_rows * inv_sf1 + 2.0f * filter_radius1;
        max_src_rows = (uint32_t) ceilf(est) + 4;
        if (max_src_rows > src_y) {
            max_src_rows = src_y;
        }
    }

    const uint32_t src_chunk_bytes = src_x * max_src_rows * sizeof(float);
    const uint32_t dst_chunk_bytes = dst_x * dst_chunk_rows * sizeof(float);

    L->max_src_chunk_rows      = max_src_rows;
    L->dst_chunk_rows          = dst_chunk_rows;
    L->src_spad_half_size      = hex_round_up(src_chunk_bytes + 256, 128);
    L->dst_spad_half_size      = hex_round_up(dst_chunk_bytes, 128);
    L->src_bytes_per_hvx       = 2 * L->src_spad_half_size;
    L->dst_bytes_per_hvx       = 2 * L->dst_spad_half_size;
    L->r_bytes_per_hvx         = hex_round_up(src_x * sizeof(float), 128);
    L->gather_bytes_per_hvx    = 256;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src,    L->src_bytes_per_hvx * n_hvx);
    VTCM_LAYOUT_ALLOC(off, off_dst,    L->dst_bytes_per_hvx * n_hvx);
    VTCM_LAYOUT_ALLOC(off, off_r,      L->r_bytes_per_hvx   * n_hvx);
    VTCM_LAYOUT_ALLOC(off, off_gather, L->gather_bytes_per_hvx * n_hvx);
    L->total_bytes = off;
}

static inline bool htp_upscale_solve_layout(
    struct htp_upscale_vtcm_layout * layout,
    uint32_t src_x,
    uint32_t src_y,
    uint32_t dst_x,
    uint32_t dst_y,
    float    inv_sf1,
    float    filter_radius1,
    uint32_t n_hvx,
    size_t   vtcm_budget,
    uint32_t * out_dst_chunk_rows
) {
    // 1. Try full plane
    uint32_t chunk_rows = dst_y;
    htp_upscale_vtcm_layout_build(layout, src_x, src_y, dst_x, dst_y, chunk_rows, inv_sf1, filter_radius1, n_hvx);
    if (layout->total_bytes <= vtcm_budget) {
        *out_dst_chunk_rows = chunk_rows;
        return true;
    }

    // 2. Try smaller chunk bands: 64, 32, 16, 8, 4, 2, 1
    chunk_rows = 64;
    while (chunk_rows > 1) {
        if (chunk_rows < dst_y) {
            htp_upscale_vtcm_layout_build(layout, src_x, src_y, dst_x, dst_y, chunk_rows, inv_sf1, filter_radius1, n_hvx);
            if (layout->total_bytes <= vtcm_budget) {
                *out_dst_chunk_rows = chunk_rows;
                return true;
            }
        }
        chunk_rows /= 2;
    }

    // 3. Fallback to 1 row
    chunk_rows = 1;
    htp_upscale_vtcm_layout_build(layout, src_x, src_y, dst_x, dst_y, chunk_rows, inv_sf1, filter_radius1, n_hvx);
    if (layout->total_bytes <= vtcm_budget) {
        *out_dst_chunk_rows = chunk_rows;
        return true;
    }

    return false;
}

int op_upscale(struct htp_ops_context * octx);

#endif // HTP_UPSCALE_OPS_H
