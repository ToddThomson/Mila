#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace Mila::Dnn::Compute::Cuda::DeltaNet
{
    /**
     * @brief Gated delta rule, recurrent form, over a chunk of @p T steps.
     *
     * Per value head, the carried state S is [head_k_dim, head_v_dim] and each step is
     *
     *   S     <- S * exp(g_t)
     *   kv    <- k_t^T S                       (a [head_v_dim] row)
     *   delta <- (v_t - kv) * beta_t
     *   S     <- S + k_t (x) delta             (outer product)
     *   out_t <- q_t^T S
     *
     * with q and k L2-normalized (eps 1e-6) and q additionally scaled by
     * 1/sqrt(head_k_dim). `g` and `beta` are derived here rather than by the caller:
     *
     *   beta_t = sigmoid(b_t)
     *   g_t    = -exp(A_log) * softplus(a_t + dt_bias)
     *
     * A_log and dt_bias are the rule's own parameters, so deriving them here keeps a
     * softplus off the public activation enum and saves two launches over tiny tensors.
     *
     * PARALLELISM. One block per (batch, value head); one thread per head_v_dim column.
     * Thread j owns S[:, j] -- and every step of the recurrence touches only that column,
     * so the state stays in REGISTERS for the whole chunk with no cross-thread exchange.
     * That is what makes the sequential recurrence affordable: the [128, 128] state would
     * not fit in shared memory, and streaming it through global memory each step would
     * cost more bandwidth per layer than reading the whole model.
     *
     * q and k are the only shared values (they are per k-head, not per column), so they
     * pass through shared memory and their norms are recomputed redundantly by every
     * thread -- uniform control flow, no block reduction, no partial-warp sync.
     *
     * GROUPED HEADS. Value head h reads k-head h / group, so the caller passes q and k at
     * num_k_heads width and no repeat_interleave is materialized.
     *
     * The state is read and written in place: pass it zeroed to start a sequence, and pass
     * it back unchanged to continue one. Accumulation is float regardless of storage type,
     * matching the reference's float32 recurrence (`mamba_ssm_dtype`).
     *
     * TWO KERNELS SIT BEHIND THIS ENTRY POINT and the call chooses on @p steps alone: the
     * sequential recurrence above, and a chunked (UT-transform) form that regroups the same
     * arithmetic into one triangular solve per 32 steps. The chunked form runs from 32
     * steps up, so prefill always takes it and decode never does. Semantics are identical
     * and the recurrent form is the oracle -- the difference is float summation order, not
     * definition. See GatedDeltaRule.cu for the derivation and the parallelism argument.
     */
    void cuda_gated_delta_rule_fp32(
        float* out,
        const float* q,
        const float* k,
        const float* v,
        const float* a,
        const float* b,
        const float* A_log,
        const float* dt_bias,
        float* state,
        int batch, int steps,
        int num_k_heads, int num_v_heads,
        int head_k_dim, int head_v_dim,
        cudaStream_t stream );

    void cuda_gated_delta_rule_bf16(
        __nv_bfloat16* out,
        const __nv_bfloat16* q,
        const __nv_bfloat16* k,
        const __nv_bfloat16* v,
        const __nv_bfloat16* a,
        const __nv_bfloat16* b,
        const __nv_bfloat16* A_log,
        const __nv_bfloat16* dt_bias,
        float* state,
        int batch, int steps,
        int num_k_heads, int num_v_heads,
        int head_k_dim, int head_v_dim,
        cudaStream_t stream );
}
