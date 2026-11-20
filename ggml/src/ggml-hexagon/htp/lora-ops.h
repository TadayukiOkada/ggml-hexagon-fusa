#ifndef LORA_OPS_H
#define LORA_OPS_H

#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "hex-common.h"
#include "hex-fastdiv.h"
#include "matmul-ops.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HTP_LORA_MAX_ADAPTERS 4
#define HTP_LORA_MAX_WEIGHTS  3

enum htp_lora_mode {
    HTP_LORA_MODE_HVX = 0,
    HTP_LORA_MODE_HMX = 1,
};

struct htp_lora_kernel_params {
    uint8_t  kernel_type;        // enum htp_mm_kernel_type (base matmul for MUL_MAT_LORA; unused for LORA_ADD)
    uint8_t  pipeline;           // HMX pipelined (1) or sync (0)
    uint8_t  glu;                // enum htp_mm_glu (0=none, 1=geglu, 2=swiglu)
    uint8_t  n_hmx;              // 1=HMX, 0=HVX

    uint8_t  n_hvx;              // HVX threads
    uint8_t  n_act_hvx;          // HMX activation prep threads
    uint8_t  n_prefetch;         // HVX base weight tile prefetch count (power of 2, 2..16)
    uint8_t  n_weights;          // 0 (LORA_ADD), 1..3 (MUL_MAT_LORA)

    uint8_t  norm;               // 1 = fused RMS_NORM+MUL on x (weight at src[norm_w_slot])
    uint8_t  norm_w_slot;        // octx->src index for norm_w (0xff if none)
    uint8_t  n_loras;            // 1..HTP_LORA_MAX_ADAPTERS
    uint8_t  lora_mode;          // enum htp_lora_mode

    uint16_t b_chunk_tiles;      // HVX outer chunk size in 32-row tiles for B_l
    uint16_t pad0;

    int32_t  m_chunk;            // HMX M chunk (or LORA_ADD HVX M chunk)
    int32_t  n_chunk;            // HMX N chunk (or LORA_ADD HVX N chunk rows per thread)
    int32_t  tile_size;          // Base weight tile size
    int32_t  aligned_tile_size;  // Base weight aligned tile size
    int32_t  act_row_size;       // Base HVX tiled act row size
    int32_t  vtcm_size;          // Total VTCM bytes required

    float    norm_eps;
    float    norm_scale;

    uint8_t  lora_target[HTP_LORA_MAX_ADAPTERS]; // output weight index 0..n_weights-1 (0 for LORA_ADD)
    uint8_t  lora_a_slot[HTP_LORA_MAX_ADAPTERS]; // octx->src index for A_l
    uint8_t  lora_b_slot[HTP_LORA_MAX_ADAPTERS]; // octx->src index for B_l
    float    lora_scale[HTP_LORA_MAX_ADAPTERS];  // scalar multiplier per adapter

    struct fastdiv_values div_n_act_hvx;
    struct fastdiv_values div_ne00_padded;
};

#ifdef __cplusplus
static_assert(sizeof(struct htp_lora_kernel_params) <= 128,
              "htp_lora_kernel_params must fit in 128 bytes");
#else
_Static_assert(sizeof(struct htp_lora_kernel_params) <= 128,
               "htp_lora_kernel_params must fit in 128 bytes");
#endif

// --- HVX MUL_MAT_LORA VTCM Layout (appended after htp_mm_hvx_vtcm_layout) ---

struct htp_lora_hvx_vtcm_layout {
    size_t off_lora_base;
    size_t off_x_f16;
    size_t off_h_f32;
    size_t off_h_f16;
    size_t off_tile_out;
    size_t off_glu_gate;
    size_t off_scratch;

    size_t h_f32_row_stride;     // floats per (l, m) row (padded to 32 floats)
    size_t h_f16_row_stride;     // fp16s per (l, m) row (padded to 64 fp16s)
    size_t scratch_per_hvx;
    size_t per_lora_b_bytes;
    size_t phase1_row_bytes;
    uint32_t phase1_batch_rows;
    size_t extra_bytes;
    size_t total_bytes;
};

static inline void htp_lora_hvx_layout_build(
    struct htp_lora_hvx_vtcm_layout * L,
    size_t base_total_bytes,
    uint32_t k,
    uint32_t m,
    uint32_t n_hvx,
    uint32_t n_loras,
    uint32_t max_r,
    size_t max_a_elem_bytes,
    size_t max_b_row_bytes,
    uint32_t max_loras_per_w,
    uint32_t b_chunk_tiles,
    uint32_t glu,
    uint32_t max_n
) {
    const size_t base_aligned = hex_align_up(base_total_bytes, 128);
    size_t off = base_aligned;

    const size_t k_pad64 = hex_align_up((size_t) k, 64);
    const size_t x_f16_sz = hex_align_up((size_t) m * k_pad64 * sizeof(uint16_t), 128) + 128;

    const size_t h_f32_stride = hex_align_up((size_t) max_r, 32);
    const size_t h_f16_stride = hex_align_up((size_t) max_r, 64);
    const size_t h_f32_sz = hex_align_up((size_t) n_loras * m * h_f32_stride * sizeof(float), 128);
    const size_t h_f16_sz = hex_align_up((size_t) n_loras * m * h_f16_stride * sizeof(uint16_t), 128);

    // Per-thread 2 x 32-float output tile buffer (reused for GLU up tiles)
    const size_t tile_out_sz = hex_align_up((size_t) n_hvx * 64 * sizeof(float), 128);
    const size_t glu_gate_sz = (glu != 0) ? hex_align_up((size_t) m * hex_round_up(max_n, 32) * sizeof(float), 128) : 0;

    // Shared per-thread DMA scratch (Phase 1: A rows; Phase 2: B_l chunk rows)
    const size_t p1_row_bytes = hex_align_up((size_t) k * max_a_elem_bytes, 128);
    const uint32_t p1_batch_rows = (p1_row_bytes > 32768) ? 2u : 4u;
    const size_t p1_per_hvx = hex_align_up(2u * p1_batch_rows * p1_row_bytes, 128);

    const size_t per_lora_b = hex_align_up((size_t) b_chunk_tiles * 32 * max_b_row_bytes, 128) + 128;
    const size_t p2_per_hvx = hex_align_up((size_t) max_loras_per_w * per_lora_b, 128);

    const size_t scratch_per_hvx = hex_smax(p1_per_hvx, p2_per_hvx);
    const size_t scratch_sz = (size_t) n_hvx * scratch_per_hvx;

    L->off_lora_base = base_aligned;
    VTCM_LAYOUT_ALLOC(off, off_x_f16, x_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_h_f32, h_f32_sz);
    VTCM_LAYOUT_ALLOC(off, off_h_f16, h_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_tile_out, tile_out_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_glu_gate, glu_gate_sz, glu_gate_sz > 0);
    VTCM_LAYOUT_ALLOC(off, off_scratch, scratch_sz);

    L->h_f32_row_stride = h_f32_stride;
    L->h_f16_row_stride = h_f16_stride;
    L->scratch_per_hvx  = scratch_per_hvx;
    L->per_lora_b_bytes = per_lora_b;
    L->phase1_row_bytes = p1_row_bytes;
    L->phase1_batch_rows = p1_batch_rows;
    L->extra_bytes = off - base_aligned;
    L->total_bytes = off;
}

static inline size_t htp_lora_hvx_extra_bytes(
    uint32_t k,
    uint32_t m,
    uint32_t n_hvx,
    uint32_t n_loras,
    uint32_t max_r,
    size_t max_a_elem_bytes,
    size_t max_b_row_bytes,
    uint32_t max_loras_per_w,
    uint32_t b_chunk_tiles,
    uint32_t glu,
    uint32_t max_n
) {
    struct htp_lora_hvx_vtcm_layout L;
    htp_lora_hvx_layout_build(&L, 0, k, m, n_hvx, n_loras, max_r, max_a_elem_bytes, max_b_row_bytes, max_loras_per_w,
                              b_chunk_tiles, glu, max_n);
    return L.extra_bytes;
}

// --- HMX MUL_MAT_LORA Group A Sub-Layout (inside HMX layout's off_bias) ---

struct htp_lora_hmx_layout {
    size_t off_eye_tile;
    size_t off_H[HTP_LORA_MAX_ADAPTERS];
    size_t H_bytes[HTP_LORA_MAX_ADAPTERS];
    size_t off_b_tiles[2];
    size_t b_tiles_bytes;
    size_t off_b_stage;
    size_t b_stage_per_hvx;
    size_t b_stage_bytes;
    size_t total_group_a_bytes;
};

static inline void htp_lora_hmx_layout_build(
    struct htp_lora_hmx_layout * L,
    size_t m_chunk,
    size_t n_chunk,
    bool pipeline,
    uint32_t glu,
    uint32_t n_hvx,
    uint32_t n_weights,
    uint32_t n_loras,
    const uint8_t * lora_target,
    const uint32_t * lora_r_tiles,
    uint32_t max_r
) {
    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_eye_tile, HTP_MM_HMX_TILE_SIZE);

    const size_t m_tiles = (m_chunk + HTP_MM_HMX_TILE_N_ROWS - 1) / HTP_MM_HMX_TILE_N_ROWS;
    for (uint32_t l = 0; l < HTP_LORA_MAX_ADAPTERS; ++l) {
        if (l < n_loras) {
            const size_t sz = hex_align_up(m_tiles * lora_r_tiles[l] * HTP_MM_HMX_TILE_SIZE, HTP_MM_HMX_TILE_SIZE);
            L->off_H[l]   = off;
            L->H_bytes[l] = sz;
            off += sz;
        } else {
            L->off_H[l]   = 0;
            L->H_bytes[l] = 0;
        }
    }

    size_t max_b_tiles = 0;
    if (glu != 0) {
        const size_t half_cols  = (n_chunk / 2) & ~31u;
        const size_t half_tiles = half_cols / HTP_MM_HMX_TILE_N_COLS;
        for (uint32_t l = 0; l < n_loras; ++l) {
            max_b_tiles += half_tiles * lora_r_tiles[l];
        }
    } else {
        const size_t col_tiles = (n_chunk + HTP_MM_HMX_TILE_N_COLS - 1) / HTP_MM_HMX_TILE_N_COLS;
        for (uint32_t w = 0; w < n_weights; ++w) {
            size_t w_tiles = 0;
            for (uint32_t l = 0; l < n_loras; ++l) {
                if (lora_target[l] == w) {
                    w_tiles += col_tiles * lora_r_tiles[l];
                }
            }
            if (w_tiles > max_b_tiles) {
                max_b_tiles = w_tiles;
            }
        }
    }

    const size_t b_bytes = hex_align_up(max_b_tiles * HTP_MM_HMX_TILE_SIZE, HTP_MM_HMX_TILE_SIZE);
    L->off_b_tiles[0] = off;
    off += b_bytes;
    if (pipeline) {
        L->off_b_tiles[1] = off;
        off += b_bytes;
    } else {
        L->off_b_tiles[1] = 0;
    }
    L->b_tiles_bytes = b_bytes;

    const size_t max_r_pad32 = hex_align_up((size_t) max_r, 32);
    const size_t b_stage_per_hvx = hex_align_up(2u * max_r_pad32 * sizeof(uint16_t) + 128, 128);
    const size_t b_stage_bytes = hex_align_up((size_t) n_hvx * b_stage_per_hvx, HTP_MM_HMX_TILE_SIZE);
    L->b_stage_per_hvx = b_stage_per_hvx;
    L->b_stage_bytes   = b_stage_bytes;
    VTCM_LAYOUT_ALLOC(off, off_b_stage, b_stage_bytes);

    L->total_group_a_bytes = hex_align_up(off, HTP_MM_HMX_TILE_SIZE);
}

static inline size_t htp_lora_hmx_group_a_bytes(
    size_t m_chunk,
    size_t n_chunk,
    bool pipeline,
    uint32_t glu,
    uint32_t n_hvx,
    uint32_t n_weights,
    uint32_t n_loras,
    const uint8_t * lora_target,
    const uint32_t * lora_r_tiles,
    uint32_t max_r
) {
    struct htp_lora_hmx_layout L;
    htp_lora_hmx_layout_build(&L, m_chunk, n_chunk, pipeline, glu, n_hvx, n_weights, n_loras, lora_target, lora_r_tiles, max_r);
    return L.total_group_a_bytes;
}

// --- Standalone LORA_ADD VTCM Layouts ---

struct htp_lora_add_hvx_layout {
    size_t off_x_raw;
    size_t off_x_f16;
    size_t off_h_f32;
    size_t off_h_f16;
    size_t off_scratch;

    size_t x_raw_row_stride; // bytes per x row
    size_t h_f32_row_stride; // floats per (l, m) row (padded to 32)
    size_t h_f16_row_stride; // fp16s per (l, m) row (padded to 64)
    size_t scratch_per_hvx;
    size_t per_lora_b_bytes;
    size_t base_buf_bytes;
    size_t phase1_row_bytes;
    uint32_t phase1_batch_rows;
    size_t total_bytes;
};

static inline void htp_lora_add_hvx_layout_build(
    struct htp_lora_add_hvx_layout * L,
    uint32_t k,
    uint32_t m_chunk,
    uint32_t n_chunk,
    uint32_t n_hvx,
    uint32_t n_loras,
    uint32_t max_r,
    size_t act_elem_bytes,
    size_t max_a_elem_bytes,
    size_t max_b_row_bytes
) {
    size_t off = 0;
    const size_t x_raw_stride = hex_align_up((size_t) k * act_elem_bytes, 128);
    const size_t x_raw_sz = hex_align_up((size_t) m_chunk * x_raw_stride, 128) + 128;

    const size_t k_pad64 = hex_align_up((size_t) k, 64);
    const size_t x_f16_sz = hex_align_up((size_t) m_chunk * k_pad64 * sizeof(uint16_t), 128) + 128;

    const size_t h_f32_stride = hex_align_up((size_t) max_r, 32);
    const size_t h_f16_stride = hex_align_up((size_t) max_r, 64);
    const size_t h_f32_sz = hex_align_up((size_t) n_loras * m_chunk * h_f32_stride * sizeof(float), 128);
    const size_t h_f16_sz = hex_align_up((size_t) n_loras * m_chunk * h_f16_stride * sizeof(uint16_t), 128);

    const size_t p1_row_bytes = hex_align_up((size_t) k * max_a_elem_bytes, 128);
    const uint32_t p1_batch_rows = (p1_row_bytes > 32768) ? 2u : 4u;
    const size_t p1_per_hvx = hex_align_up(2u * p1_batch_rows * p1_row_bytes, 128);

    const size_t per_lora_b = hex_align_up((size_t) n_chunk * max_b_row_bytes, 128) + 128;
    const size_t base_buf   = hex_align_up((size_t) n_chunk * sizeof(float), 128) + 128;
    const size_t p2_per_hvx = hex_align_up((size_t) n_loras * per_lora_b + 2u * base_buf, 128);

    const size_t scratch_per_hvx = hex_smax(p1_per_hvx, p2_per_hvx);
    const size_t scratch_sz = (size_t) n_hvx * scratch_per_hvx;

    VTCM_LAYOUT_ALLOC(off, off_x_raw, x_raw_sz);
    VTCM_LAYOUT_ALLOC(off, off_x_f16, x_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_h_f32, h_f32_sz);
    VTCM_LAYOUT_ALLOC(off, off_h_f16, h_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_scratch, scratch_sz);

    L->x_raw_row_stride = x_raw_stride;
    L->h_f32_row_stride = h_f32_stride;
    L->h_f16_row_stride = h_f16_stride;
    L->scratch_per_hvx  = scratch_per_hvx;
    L->per_lora_b_bytes = per_lora_b;
    L->base_buf_bytes   = base_buf;
    L->phase1_row_bytes = p1_row_bytes;
    L->phase1_batch_rows = p1_batch_rows;
    L->total_bytes = off;
}

struct htp_lora_add_hmx_layout {
    // Group A (persistent across mr chunk)
    size_t off_scales;
    size_t off_eye_tile;
    size_t off_act;
    size_t off_H[HTP_LORA_MAX_ADAPTERS];
    size_t H_bytes[HTP_LORA_MAX_ADAPTERS];

    // Group B (nc compute loop)
    size_t off_b_tiles;
    size_t off_dst;
    size_t off_b_stage;
    size_t b_stage_per_hvx;

    // Group C (overlaps Group B during transfer_activation_chunk_threaded)
    size_t off_act_f32;
    size_t act_f32_bytes;

    // Group D (overlaps Group B during hmx_compute_lora_H)
    size_t off_a_tiles;
    size_t off_h_tmp;

    size_t total_bytes;
};

static inline void htp_lora_add_hmx_layout_build(
    struct htp_lora_add_hmx_layout * L,
    uint32_t k,
    size_t m_chunk,
    size_t n_chunk,
    uint32_t act_hvx,
    uint32_t n_hvx,
    uint32_t n_loras,
    const uint32_t * lora_r_tiles,
    uint32_t max_r
) {
    const size_t k_pad32 = hex_align_up((size_t) k, 32);
    const size_t m_pad32 = hex_align_up(m_chunk, 32);
    const size_t n_pad32 = hex_align_up(n_chunk, 32);
    const size_t m_tiles = m_pad32 / 32;
    const size_t n_tiles = n_pad32 / 32;
    const size_t k_tiles = k_pad32 / 32;

    size_t off_a = 0;
    VTCM_LAYOUT_ALLOC(off_a, off_scales, HTP_MM_HMX_TILE_SIZE);
    VTCM_LAYOUT_ALLOC(off_a, off_eye_tile, HTP_MM_HMX_TILE_SIZE);
    VTCM_LAYOUT_ALLOC(off_a, off_act, hex_align_up(m_pad32 * k_pad32 * sizeof(uint16_t), HTP_MM_HMX_TILE_SIZE));

    size_t sum_r_tiles = 0;
    for (uint32_t l = 0; l < HTP_LORA_MAX_ADAPTERS; ++l) {
        if (l < n_loras) {
            const size_t sz = hex_align_up(m_tiles * lora_r_tiles[l] * HTP_MM_HMX_TILE_SIZE, HTP_MM_HMX_TILE_SIZE);
            L->off_H[l]   = off_a;
            L->H_bytes[l] = sz;
            off_a += sz;
            sum_r_tiles += lora_r_tiles[l];
        } else {
            L->off_H[l]   = 0;
            L->H_bytes[l] = 0;
        }
    }

    // Group B
    size_t off_b = off_a;
    const size_t b_tiles_sz = hex_align_up(sum_r_tiles * n_tiles * HTP_MM_HMX_TILE_SIZE, HTP_MM_HMX_TILE_SIZE);
    const size_t dst_sz = hex_align_up(m_pad32 * n_pad32 * sizeof(uint16_t), HTP_MM_HMX_TILE_SIZE);
    const size_t max_r_pad32 = hex_align_up((size_t) max_r, 32);
    const size_t b_stage_per_hvx = hex_align_up(2u * max_r_pad32 * sizeof(uint16_t) + 128, 128);
    const size_t b_stage_sz = hex_align_up((size_t) n_hvx * b_stage_per_hvx, HTP_MM_HMX_TILE_SIZE);
    VTCM_LAYOUT_ALLOC(off_b, off_b_tiles, b_tiles_sz);
    VTCM_LAYOUT_ALLOC(off_b, off_dst, dst_sz);
    VTCM_LAYOUT_ALLOC(off_b, off_b_stage, b_stage_sz);
    L->b_stage_per_hvx = b_stage_per_hvx;
    const size_t group_b_sz = off_b - off_a;

    // Group C (for activation DMA staging: act_hvx * 4 rows of K floats)
    size_t off_c = off_a;
    const size_t min_f32_sz = hex_align_up((size_t) act_hvx * HTP_MM_DMA_ACT_MULTIPLIER * k_pad32 * sizeof(float), 128);
    const size_t max_f32_sz = hex_align_up((size_t) act_hvx * 64u * k_pad32 * sizeof(float), 128);
    const size_t act_f32_sz = hex_align_up(hex_smin(max_f32_sz, hex_smax(min_f32_sz, group_b_sz)), 128);
    VTCM_LAYOUT_ALLOC(off_c, off_act_f32, act_f32_sz);
    L->act_f32_bytes = act_f32_sz;
    const size_t group_c_sz = off_c - off_a;

    // Group D (for hmx_compute_lora_H: k_tiles of interleaved HMX tiles + m_tiles scratch)
    size_t off_d = off_a;
    const size_t a_tiles_sz = hex_align_up(k_tiles * HTP_MM_HMX_TILE_SIZE, HTP_MM_HMX_TILE_SIZE);
    const size_t h_tmp_sz   = hex_align_up(m_tiles * HTP_MM_HMX_TILE_SIZE, HTP_MM_HMX_TILE_SIZE);
    VTCM_LAYOUT_ALLOC(off_d, off_a_tiles, a_tiles_sz);
    VTCM_LAYOUT_ALLOC(off_d, off_h_tmp, h_tmp_sz);
    const size_t group_d_sz = off_d - off_a;

    L->total_bytes = off_a + hex_smax(group_b_sz, hex_smax(group_c_sz, group_d_sz));
}

#ifdef __cplusplus
}
#endif

#endif // LORA_OPS_H
