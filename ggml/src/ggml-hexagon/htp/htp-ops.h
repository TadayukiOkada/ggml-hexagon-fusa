#ifndef HTP_OPS_H
#define HTP_OPS_H

#include <assert.h>

// ggml-common.h must be included prio to this header

enum htp_status {
    HTP_STATUS_OK             = 1,
    HTP_STATUS_INTERNAL_ERR   = 2,
    HTP_STATUS_NO_SUPPORT     = 3,
    HTP_STATUS_INVAL_PARAMS   = 4,
    HTP_STATUS_VTCM_TOO_SMALL = 5,
};

// First set of values must match the ggml_type.
// Duplicated here because we can't include full ggml.h in the htp build.
// We have some static_asserts in the cpp code to ensure things are in sync.
enum htp_data_type {
    HTP_TYPE_F32    = 0,
    HTP_TYPE_F16    = 1,
    HTP_TYPE_Q4_0   = 2,
    HTP_TYPE_Q4_1   = 3,
    HTP_TYPE_Q8_0   = 8,
    HTP_TYPE_Q2_K   = 10,
    HTP_TYPE_Q3_K   = 11,
    HTP_TYPE_Q4_K   = 12,
    HTP_TYPE_Q5_K   = 13,
    HTP_TYPE_Q6_K   = 14,
    HTP_TYPE_IQ4_NL = 20,
    HTP_TYPE_I32    = 26,
    HTP_TYPE_I64    = 27,
    HTP_TYPE_BF16   = 30,
    HTP_TYPE_MXFP4  = 39,
    HTP_TYPE_Q1_0   = 41,
    HTP_TYPE_Q2_0   = 42,

    HTP_TYPE_INVALID
};

// Do not reorder first 4 (used as an index)
enum htp_op_code {
    HTP_OP_MUL = 0,
    HTP_OP_ADD = 1,
    HTP_OP_SUB = 2,
    HTP_OP_DIV = 3,
    HTP_OP_MUL_MAT,
    HTP_OP_MUL_MAT_ID,
    HTP_OP_MUL_MAT_NX,
    HTP_OP_MUL_MAT_ID_NX,
    HTP_OP_MUL_MAT_ADD,
    HTP_OP_RMS_NORM,
    HTP_OP_RMS_NORM_MUL,
    HTP_OP_UNARY_SILU,
    HTP_OP_UNARY_GELU,
    HTP_OP_UNARY_SIGMOID,
    HTP_OP_UNARY_EXP,
    HTP_OP_UNARY_NEG,
    HTP_OP_UNARY_SOFTPLUS,
    HTP_OP_UNARY_TANH,
    HTP_OP_UNARY_ABS,
    HTP_OP_UNARY_LOG,
    HTP_OP_UNARY_RELU,
    HTP_OP_UNARY_STEP,
    HTP_OP_GLU_SWIGLU,
    HTP_OP_GLU_SWIGLU_OAI,
    HTP_OP_GLU_GEGLU,
    HTP_OP_GLU_GEGLU_QUICK,
    HTP_OP_SOFTMAX,
    HTP_OP_ADD_ID,
    HTP_OP_ROPE,
    HTP_OP_FLASH_ATTN_EXT,
    HTP_OP_SET_ROWS,
    HTP_OP_GET_ROWS,
    HTP_OP_SCALE,
    HTP_OP_CPY,
    HTP_OP_CPY_FENCE,
    HTP_OP_ARGSORT,
    HTP_OP_TOP_K,
    HTP_OP_SQR,
    HTP_OP_SQRT,
    HTP_OP_SUM,
    HTP_OP_SUM_ROWS,
    HTP_OP_SSM_CONV,
    HTP_OP_REPEAT,
    HTP_OP_CUMSUM,
    HTP_OP_FILL,
    HTP_OP_DIAG,
    HTP_OP_SOLVE_TRI,
    HTP_OP_L2_NORM,
    HTP_OP_GATED_DELTA_NET,
    HTP_OP_TRI,
    HTP_OP_PAD,
    HTP_OP_NORM,
    HTP_OP_CONCAT,
    HTP_OP_CLAMP,
    HTP_OP_LEAKY_RELU,
    HTP_OP_IM2COL,
    HTP_OP_FENCE,
    HTP_OP_ALLREDUCE,
    HTP_OP_ALLREDUCE_ADD,
    HTP_OP_GLU_SWIGLU_CLAMP,
    HTP_OP_MDEV_GROUP,
    HTP_OP_ROLL,
    HTP_OP_ARGMAX,
    HTP_OP_UNARY_GELU_ERF,
    HTP_OP_GLU_GEGLU_ERF,
    HTP_OP_POOL_2D,
    HTP_OP_POOL_1D,
    HTP_OP_UNARY_GELU_QUICK,
    HTP_OP_IM2COL_3D,
    HTP_OP_HADAMARD,
    HTP_OP_GLU_REGLU,
    HTP_OP_GLU_SIGMOID,
    HTP_OP_UNARY_CEIL,
    HTP_OP_UNARY_SGN,

    HTP_OP_INVALID
};

#define HTP_OP_MAX_DIMS    4    // aka GGML_MAX_DIMS
#define HTP_OP_MAX_INPUTS  10   // aka GGML_MAX_SRCS
#define HTP_OP_MAX_OUTPUTS 4
#define HTP_OP_MAX_PARAMS  16   // aka GGML_MAX_OP_PARAMS
#define HTP_OP_MAX_KERN_PARAMS 32

#define HTP_OP_MAX_BUFS    64
#define HTP_OP_MAX_TENSORS 65534 // must stay under 64K (uint16)

#define HTP_FENCE_TIMEOUT  (1000000000ULL)
#define HTP_FENCE_SLOT_SIZE 128

#define HTP_OP_MAX_VMEM_DEFAULT (3355443200u)

#define HTP_MMAP_MAX_VMEM  (2147483648u)

enum htp_tensor_flags {
    HTP_TENSOR_WEIGHT   = (1U << 0), // Tensor buffer model weight data (not compute)
    HTP_TENSOR_TILED    = (1U << 1), // Tensor is in tiled format
    HTP_TENSOR_FENCE    = (1U << 2), // Tensor is synchronization fence (explicitly managed)
    HTP_TENSOR_FUSEABLE = (1U << 3), // Tensor can be fused (host-only)
};

enum htp_buf_flags {
    HTP_BUF_EXTENDED = (1U << 0),
};

// Tensor descriptor
struct htp_tensor {
    uint64_t data;                 // Buffer offset in the messages, and data pointer on the NPU
    uint32_t size;                 // Data size in bytes
    uint32_t flags;                // Buffer / tensor flags
    uint32_t type;                 // Data type
    uint16_t bi;                   // Buffer index
    uint16_t ti;                   // Tensor index
    uint32_t ne[HTP_OP_MAX_DIMS];  // Number of elements
    uint32_t nb[HTP_OP_MAX_DIMS];  // Stride in bytes (see ggml.h ggml_tensor)
};

// Buffer descriptor
struct htp_buf_desc {
    uint64_t base;     // base address
    uint64_t size;     // total size
    uint32_t flags;    // HTP_BUF_*
    uint32_t fd;       // file descriptor
};

enum htp_op_flags {
    HTP_OPFLAGS_STUB  = (1U << 0),
};

// Op descriptor
struct htp_op_desc {
    uint32_t opcode;                    // GGML/HTP Op
    uint32_t flags;                     // Op flags
    int32_t  params[HTP_OP_MAX_PARAMS]; // Params for the op, e.g. epsilon of RMS norm
    int32_t  kernel_params[HTP_OP_MAX_KERN_PARAMS]; // generic blob for host-precomputed parameters
    uint16_t src[HTP_OP_MAX_INPUTS];    // Input tensors indices
    uint16_t dst[HTP_OP_MAX_OUTPUTS];   // Output tensor indices
    uint16_t pad[2];                    // padding to align to 64 bits
};

#ifndef HTP_MAX_NTHREADS
#define HTP_MAX_NTHREADS 10
#endif

#define HTP_TRACE_MAX_EVENTS 256

enum htp_profiler_mode {
    HTP_PROF_DISABLED = 0,
    HTP_PROF_BASIC    = 1,
    HTP_PROF_PMU      = 2,
    HTP_PROF_TRACE    = 3,
};

// DCVS/HMX power profile requested for the session.
enum htp_power_mode {
    HTP_POWER_MODE_LOW_POWER_SAVER            = 0,
    HTP_POWER_MODE_POWER_SAVER                = 1,
    HTP_POWER_MODE_HIGH_POWER_SAVER           = 2,
    HTP_POWER_MODE_LOW_BALANCED               = 3,
    HTP_POWER_MODE_BALANCED                   = 4,
    HTP_POWER_MODE_HIGH_PERFORMANCE           = 5,
    HTP_POWER_MODE_SUSTAINED_HIGH_PERFORMANCE = 6,
    HTP_POWER_MODE_BURST                      = 7,

    HTP_POWER_MODE_COUNT
};

enum htp_trace_event_id {
    HTP_TRACE_EVT_DMA                 = 0,
    HTP_TRACE_EVT_L2FLUSH             = 1,
    HTP_TRACE_EVT_INIT                = 2,
    HTP_TRACE_EVT_BUFF                = 3,
    HTP_TRACE_EVT_FENCE               = 4,

    HTP_TRACE_EVT_HVX_COMP            = 20,
    HTP_TRACE_EVT_HVX_A_QUANT         = 21,
    HTP_TRACE_EVT_HVX_A_PREP          = 22,
    HTP_TRACE_EVT_HVX_W_DEQUANT       = 23,
    HTP_TRACE_EVT_HVX_W_PREP          = 24,
    HTP_TRACE_EVT_HVX_O_PROC          = 25,
    HTP_TRACE_EVT_HVX_FA_QK           = 26,
    HTP_TRACE_EVT_HVX_FA_SFM          = 27,
    HTP_TRACE_EVT_HVX_FA_Q_PREP       = 28,
    HTP_TRACE_EVT_HVX_FA_K_PREP       = 29,
    HTP_TRACE_EVT_HVX_FA_V_PREP       = 30,

    HTP_TRACE_EVT_HVX_GDN_PREP        = 31,
    HTP_TRACE_EVT_HVX_GDN_SOLVE       = 32,
    HTP_TRACE_EVT_HVX_GDN_V_PREP      = 33,
    HTP_TRACE_EVT_HVX_GDN_D_PREP      = 34,
    HTP_TRACE_EVT_HVX_GDN_OUT         = 35,
    HTP_TRACE_EVT_HVX_GDN_STATE       = 36,
    HTP_TRACE_EVT_HVX_GDN_REM         = 37,

    HTP_TRACE_EVT_HMX_COMP            = 40,
};

struct htp_trace_desc {
    uint32_t cycles;  // lower 32-bits of cycle counter
    uint16_t id;      // Event ID
    uint16_t info;    // bit 15: is_stop. bits 14-0: tile/chunk index or other metadata.
};

#define HTP_PROF_PMU_NCNT 8

// Profile descriptor
struct htp_prof_desc {
    uint32_t opcode;                 // GGML/HTP Op
    uint32_t usecs;                  // Number of usec
    uint32_t cycles_start;           // Start cycle counter
    uint32_t cycles_stop;            // Stop cycle counter
    uint32_t pmu[HTP_PROF_PMU_NCNT]; // PMU counters
};

struct htp_opbatch_req {
    uint64_t seq;         // Sequence number
    uint32_t n_bufs;      // Number of buffers
    uint32_t n_tensors;   // Number of tensors
    uint32_t n_ops;       // Number of ops
    uint32_t n_traces;    // Number of trace descriptors per thread
    // struct htp_buf_desc  bufs[];    -- dspqueue buf 0
    // struct htp_tensor    tensors[]; -- dspqueue buf 0
    // struct htp_op_desc   ops[];     -- dspqueue buf 0
};

// Where the DSP main thread is (htp_iface_diag(), layout in htp/main.c). The host asks for it when
// a batch does not complete; the main thread writes it, a FastRPC thread reads it. Times are the
// low 32 bits of HAP_perf_get_time_us().
enum htp_diag_phase {
    HTP_DIAG_IDLE = 0,       // waiting for a packet (dspqueue_peek)
    HTP_DIAG_POLL,           // polling the queue between batches
    HTP_DIAG_BATCH_PREP,     // cache clean, buffers, tensors
    HTP_DIAG_OP,             // running op `op_idx`
    HTP_DIAG_YIELD_SUSPEND,  // suspending the HMX and work queues to give VTCM away
    HTP_DIAG_YIELD_WAKEUP,   // waking them after taking VTCM back
    HTP_DIAG_BATCH_END,      // suspending the queues, cache clean
    HTP_DIAG_BARRIER,        // mdev barrier at the end of the batch
    HTP_DIAG_RESPONSE,       // writing the response
};

#define HTP_DIAG_WORDS 34  // htp_iface_diag() output

struct htp_opbatch_rsp {
    uint64_t seq;            // Sequence number
    uint64_t cycles_start;   // Start cycle counter
    uint64_t cycles_stop;    // Stop cycle counter
    uint32_t status;         // HTP_STATUS_...
    uint32_t n_bufs;         // Number of buffers
    uint32_t n_tensors;      // Number of tensors
    uint32_t n_ops;          // Number of op profile descriptors
    uint32_t usecs;          // Number of usec
    uint32_t n_traces[HTP_MAX_NTHREADS + 1];
    // VTCM/HMX profiler counters, cumulative since session open (profiler only)
    uint32_t resource_hold_max_us;      // longest acquire -> release, requested or not
    uint32_t resource_response_max_us;  // longest release request -> release
    uint32_t resource_requests_served;
    uint32_t resource_releases;
    uint32_t resource_pending_us;       // age of a request still unserved at response time
    // Queue errors the DSP main thread recovered from since session open, and the last of each
    uint32_t peek_errors;
    uint32_t peek_err;
    uint32_t read_errors;
    uint32_t read_err;
    // struct htp_prof_desc profs[];  -- dspqueue buf 0
};

#endif /* HTP_OPS_H */
