#ifndef HTP_BUDGET_H
#define HTP_BUDGET_H

#include "htp-common.h"

#ifndef GGML_HEXAGON_OP_BUDGET_MS_DEFAULT
#define GGML_HEXAGON_OP_BUDGET_MS_DEFAULT 0
#endif

// Ops estimated to hold the DSP longer than this (ms) without a yield point are left to the CPU.
// Functional-safety firmware stops the DSP when its self-test cannot get VTCM/HMX in time. 0 = off.
// GGML_HEXAGON_OP_BUDGET_MS overrides the default at run time.
extern double opt_op_budget_ms;

// False when the op is estimated to run past opt_op_budget_ms; call for ops the backend supports.
bool htp_op_within_budget(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op);

#endif // HTP_BUDGET_H
