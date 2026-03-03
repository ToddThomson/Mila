/**
 * @file CudaGqa.Permute.cu
 * @brief CUDA kernels for Grouped-Query Attention permute operations.
 *
 * Provides the QKV split, KV expansion, KV gradient reduction, and backward
 * gradient pack kernels that are specific to GQA's asymmetric head layout.
 *
 * The softmax and unpermute kernels shared with MHA live in:
 *   Common/Kernels/CudaAttentionCommon.Softmax.cu
 *   Common/Kernels/CudaAttentionCommon.Unpermute.cu
 *
 * Layout convention (row-major throughout)
 * ─────────────────────────────────────────
 *   Input X   : [B, T, (NH + 2*NKV) * HS]
 *                 offset 0           → Q  (NH  heads)
 *                 offset NH*HS       → K  (NKV heads)
 *                 offset (NH+NKV)*HS → V  (NKV heads)
 *   Q output  : [B, NH,  T, HS]
 *   K, V      : [B, NKV, T, HS]   — compact KV cache layout
 *   k_exp,v_exp:[B, NH,  T, HS]   — expanded for cuBLASLt (GS = NH/NKV)
 *
 * Kernel inventory
 * ────────────────
 *   permute_qkv_*_kernel           — training / full-sequence split
 *   permute_qkv_padded_*_kernel    — prefill split with T-padding
 *   permute_qkv_decode_*_kernel    — single-token decode, appends to KV cache
 *   expand_kv_*_kernel             — [B,NKV,T,HS] → [B,NH,T,HS] broadcast
 *   reduce_kv_grad_*_kernel        — [B,NH,T,HS]  → [B,NKV,T,HS] sum (backward)
 *   permute_backward_*_kernel      — pack dQ/dK/dV → dX [B,T,(NH+2*NKV)*HS]
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "CudaGqa.cuh"

namespace Mila::Dnn::Compute::Cuda::GroupedQueryAttention
{
    // ========================================================================
    // QKV permute — training / full sequence
    // ========================================================================

    /**
     * @brief Split packed QKV input into separate Q, K, V buffers (FP32).
     *
     * Input trailing dim: (NH + 2*NKV) * HS
     *   [0,          NH*HS)        → Q  mapped to Q[B, NH,  T, HS]
     *   [NH*HS,      (NH+NKV)*HS)  → K  mapped to K[B, NKV, T, HS]
     *   [(NH+NKV)*HS,(NH+2*NKV)*HS)→ V  mapped to V[B, NKV, T, HS]
     *
     * Total threads: B * max(NH, NKV) * T * HS.
     * Each thread writes one element of either Q (when its head index < NH)
     * or K/V (when its head index < NKV), using separate index spaces.
     *
     * Implementation note: we launch over B * NH * T * HS threads for Q and
     * a separate pass over B * NKV * T * HS for K/V.  Combining both into a
     * single launch is possible but complicates the index arithmetic; the two-
     * launch approach mirrors the MHA pattern and keeps each kernel simple.
     */
    __global__ void permute_q_fp32_kernel(
        float* Q,
        const float* X,
        int B, int T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * T * HS )
        {
            const int b = idx / (NH * T * HS);
            int rest = idx % (NH * T * HS);
            const int nh = rest / (T * HS);
            rest = rest % (T * HS);
            const int t = rest / HS;
            const int hs = rest % HS;

            // Q block starts at offset 0 in the trailing dim.
            const int q_src = b * T * (NH + 2 * NKV) * HS
                + t * (NH + 2 * NKV) * HS
                + nh * HS + hs;

            const int out_idx = b * (NH * T * HS) + nh * (T * HS) + t * HS + hs;

            Q[ out_idx ] = __ldcs( &X[ q_src ] );
        }
    }

    __global__ void permute_kv_fp32_kernel(
        float* K, float* V,
        const float* X,
        int B, int T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NKV * T * HS )
        {
            const int b = idx / (NKV * T * HS);
            int rest = idx % (NKV * T * HS);
            const int nkv = rest / (T * HS);
            rest = rest % (T * HS);
            const int t = rest / HS;
            const int hs = rest % HS;

            // K block starts at offset NH*HS; V at (NH+NKV)*HS.
            const int base = b * T * (NH + 2 * NKV) * HS + t * (NH + 2 * NKV) * HS;
            const int k_src = base + NH * HS + nkv * HS + hs;
            const int v_src = base + (NH + NKV) * HS + nkv * HS + hs;

            const int out_idx = b * (NKV * T * HS) + nkv * (T * HS) + t * HS + hs;

            K[ out_idx ] = __ldcs( &X[ k_src ] );
            V[ out_idx ] = __ldcs( &X[ v_src ] );
        }
    }

    // ========================================================================
    // QKV permute — padded prefill
    // ========================================================================

    /**
     * @brief Split packed QKV into Q for a padded prefill batch (FP32).
     *
     * Threads beyond input_T zero-fill Q so that the full output_T buffer is
     * initialised (cuBLASLt reads the full padded length).
     */
    __global__ void permute_q_padded_fp32_kernel(
        float* Q,
        const float* X,
        int B, int input_T, int output_T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * output_T * HS )
        {
            const int b = idx / (NH * output_T * HS);
            int rest = idx % (NH * output_T * HS);
            const int nh = rest / (output_T * HS);
            rest = rest % (output_T * HS);
            const int t = rest / HS;
            const int hs = rest % HS;

            const int out_idx = b * (NH * output_T * HS) + nh * (output_T * HS) + t * HS + hs;

            if ( t >= input_T )
            {
                Q[ out_idx ] = 0.0f;
                return;
            }

            const int q_src = b * input_T * (NH + 2 * NKV) * HS
                + t * (NH + 2 * NKV) * HS
                + nh * HS + hs;

            Q[ out_idx ] = __ldcs( &X[ q_src ] );
        }
    }

    __global__ void permute_kv_padded_fp32_kernel(
        float* K, float* V,
        const float* X,
        int B, int input_T, int output_T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NKV * output_T * HS )
        {
            const int b = idx / (NKV * output_T * HS);
            int rest = idx % (NKV * output_T * HS);
            const int nkv = rest / (output_T * HS);
            rest = rest % (output_T * HS);
            const int t = rest / HS;
            const int hs = rest % HS;

            const int out_idx = b * (NKV * output_T * HS) + nkv * (output_T * HS) + t * HS + hs;

            if ( t >= input_T )
            {
                K[ out_idx ] = 0.0f;
                V[ out_idx ] = 0.0f;
                return;
            }

            const int base = b * input_T * (NH + 2 * NKV) * HS + t * (NH + 2 * NKV) * HS;
            const int k_src = base + NH * HS + nkv * HS + hs;
            const int v_src = base + (NH + NKV) * HS + nkv * HS + hs;

            K[ out_idx ] = __ldcs( &X[ k_src ] );
            V[ out_idx ] = __ldcs( &X[ v_src ] );
        }
    }

    // ========================================================================
    // QKV permute — single-token decode (appends to KV cache)
    // ========================================================================

    /**
     * @brief Write single Q token into Q cache at position (FP32).
     *
     * Input X is a single token: [B, 1, (NH+2*NKV)*HS].
     * Writes Q[b, nh, position, hs].
     */
    __global__ void permute_q_decode_fp32_kernel(
        float* Q,
        const float* X,
        int B, int position, int cache_T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * HS )
        {
            const int b = idx / (NH * HS);
            int rest = idx % (NH * HS);
            const int nh = rest / HS;
            const int hs = rest % HS;

            const int q_src = b * (NH + 2 * NKV) * HS + nh * HS + hs;
            const int out_idx = b * (NH * cache_T * HS) + nh * (cache_T * HS) + position * HS + hs;

            Q[ out_idx ] = __ldcs( &X[ q_src ] );
        }
    }

    /**
     * @brief Append single K/V token to KV cache at position (FP32).
     *
     * Writes K[b, nkv, position, hs] and V[b, nkv, position, hs].
     */
    __global__ void permute_kv_decode_fp32_kernel(
        float* K, float* V,
        const float* X,
        int B, int position, int cache_T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NKV * HS )
        {
            const int b = idx / (NKV * HS);
            int rest = idx % (NKV * HS);
            const int nkv = rest / HS;
            const int hs = rest % HS;

            const int base = b * (NH + 2 * NKV) * HS;
            const int k_src = base + NH * HS + nkv * HS + hs;
            const int v_src = base + (NH + NKV) * HS + nkv * HS + hs;

            const int out_idx = b * (NKV * cache_T * HS) + nkv * (cache_T * HS) + position * HS + hs;

            K[ out_idx ] = __ldcs( &X[ k_src ] );
            V[ out_idx ] = __ldcs( &X[ v_src ] );
        }
    }

    // ========================================================================
    // KV expansion: [B, NKV, T, HS] → [B, NH, T, HS]
    // ========================================================================

    /**
     * @brief Broadcast each KV head to its group of Q heads (FP32).
     *
     * For Q head nh: kv_head = nh / GS  where GS = NH / NKV.
     * Reads k_compact/v_compact[b, kv_head, t, hs] and writes
     * k_exp/v_exp[b, nh, t, hs].
     *
     * Total threads: B * NH * T * HS (indexed over the expanded layout).
     */
    __global__ void expand_kv_fp32_kernel(
        float* k_exp, float* v_exp,
        const float* k_compact, const float* v_compact,
        int B, int T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * T * HS )
        {
            const int b = idx / (NH * T * HS);
            int rest = idx % (NH * T * HS);
            const int nh = rest / (T * HS);
            rest = rest % (T * HS);
            const int t = rest / HS;
            const int hs = rest % HS;

            // Map Q head → KV head (integer division, GS = NH/NKV).
            const int nkv = nh / (NH / NKV);

            const int src_idx = b * (NKV * T * HS) + nkv * (T * HS) + t * HS + hs;

            k_exp[ idx ] = k_compact[ src_idx ];
            v_exp[ idx ] = v_compact[ src_idx ];
        }
    }

    __global__ void expand_kv_fp16_kernel(
        half* k_exp, half* v_exp,
        const half* k_compact, const half* v_compact,
        int B, int T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * T * HS )
        {
            const int b = idx / (NH * T * HS);
            int rest = idx % (NH * T * HS);
            const int nh = rest / (T * HS);
            rest = rest % (T * HS);
            const int t = rest / HS;
            const int hs = rest % HS;

            const int nkv = nh / (NH / NKV);
            const int src_idx = b * (NKV * T * HS) + nkv * (T * HS) + t * HS + hs;

            k_exp[ idx ] = k_compact[ src_idx ];
            v_exp[ idx ] = v_compact[ src_idx ];
        }
    }

    // ========================================================================
    // KV gradient reduction: [B, NH, T, HS] → [B, NKV, T, HS]
    // ========================================================================

    /**
     * @brief Sum expanded KV gradients back to compact KV layout (FP32).
     *
     * For each (b, nkv, t, hs), sums over the GS = NH/NKV Q-heads that share
     * this KV head:
     *
     *   dK[b, nkv, t, hs] = Σ_{g=0}^{GS-1}  dK_exp[b, nkv*GS+g, t, hs]
     *
     * Total threads: B * NKV * T * HS.
     *
     * Note: atomic operations are not required because each output element is
     * written by exactly one thread.
     */
    __global__ void reduce_kv_grad_fp32_kernel(
        float* dk_compact, float* dv_compact,
        const float* dk_exp, const float* dv_exp,
        int B, int T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NKV * T * HS )
        {
            const int b = idx / (NKV * T * HS);
            int rest = idx % (NKV * T * HS);
            const int nkv = rest / (T * HS);
            rest = rest % (T * HS);
            const int t = rest / HS;
            const int hs = rest % HS;

            const int GS = NH / NKV;

            float sum_k = 0.0f;
            float sum_v = 0.0f;

            for ( int g = 0; g < GS; ++g )
            {
                const int nh = nkv * GS + g;
                const int exp_idx = b * (NH * T * HS) + nh * (T * HS) + t * HS + hs;

                sum_k += dk_exp[ exp_idx ];
                sum_v += dv_exp[ exp_idx ];
            }

            dk_compact[ idx ] = sum_k;
            dv_compact[ idx ] = sum_v;
        }
    }

    __global__ void reduce_kv_grad_fp16_kernel(
        half* dk_compact, half* dv_compact,
        const half* dk_exp, const half* dv_exp,
        int B, int T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NKV * T * HS )
        {
            const int b = idx / (NKV * T * HS);
            int rest = idx % (NKV * T * HS);
            const int nkv = rest / (T * HS);
            rest = rest % (T * HS);
            const int t = rest / HS;
            const int hs = rest % HS;

            const int GS = NH / NKV;

            // Accumulate in FP32 to avoid half precision rounding during summation.
            float sum_k = 0.0f;
            float sum_v = 0.0f;

            for ( int g = 0; g < GS; ++g )
            {
                const int nh = nkv * GS + g;
                const int exp_idx = b * (NH * T * HS) + nh * (T * HS) + t * HS + hs;

                sum_k += __half2float( dk_exp[ exp_idx ] );
                sum_v += __half2float( dv_exp[ exp_idx ] );
            }

            dk_compact[ idx ] = __float2half( sum_k );
            dv_compact[ idx ] = __float2half( sum_v );
        }
    }

    // ========================================================================
    // Permute backward: pack dQ/dK/dV → dX [B, T, (NH+2*NKV)*HS]
    // ========================================================================

    /**
     * @brief Pack dQ gradient into dX (FP32).
     *
     * Reads dQ[b, nh, t, hs] and writes to
     * dX[b, t, nh*HS + hs]  (Q block, offset 0).
     */
    __global__ void permute_backward_q_fp32_kernel(
        float* dX,
        const float* dQ,
        int B, int T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * T * HS )
        {
            const int b = idx / (NH * T * HS);
            int rest = idx % (NH * T * HS);
            const int nh = rest / (T * HS);
            rest = rest % (T * HS);
            const int t = rest / HS;
            const int hs = rest % HS;

            const int dx_idx = b * T * (NH + 2 * NKV) * HS
                + t * (NH + 2 * NKV) * HS
                + nh * HS + hs;

            dX[ dx_idx ] = dQ[ idx ];
        }
    }

    /**
     * @brief Pack dK and dV gradients into dX (FP32).
     *
     * Reads dK[b, nkv, t, hs] → dX[b, t, (NH + nkv)*HS + hs]
     *       dV[b, nkv, t, hs] → dX[b, t, (NH+NKV+nkv)*HS + hs]
     */
    __global__ void permute_backward_kv_fp32_kernel(
        float* dX,
        const float* dK, const float* dV,
        int B, int T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NKV * T * HS )
        {
            const int b = idx / (NKV * T * HS);
            int rest = idx % (NKV * T * HS);
            const int nkv = rest / (T * HS);
            rest = rest % (T * HS);
            const int t = rest / HS;
            const int hs = rest % HS;

            const int base = b * T * (NH + 2 * NKV) * HS + t * (NH + 2 * NKV) * HS;
            const int dk_idx = base + NH * HS + nkv * HS + hs;
            const int dv_idx = base + (NH + NKV) * HS + nkv * HS + hs;

            dX[ dk_idx ] = dK[ idx ];
            dX[ dv_idx ] = dV[ idx ];
        }
    }

    // ---- FP16 backward kernels ----

    __global__ void permute_backward_q_fp16_kernel(
        half* dX,
        const half* dQ,
        int B, int T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * T * HS )
        {
            const int b = idx / (NH * T * HS);
            int rest = idx % (NH * T * HS);
            const int nh = rest / (T * HS);
            rest = rest % (T * HS);
            const int t = rest / HS;
            const int hs = rest % HS;

            const int dx_idx = b * T * (NH + 2 * NKV) * HS
                + t * (NH + 2 * NKV) * HS
                + nh * HS + hs;

            dX[ dx_idx ] = dQ[ idx ];
        }
    }

    __global__ void permute_backward_kv_fp16_kernel(
        half* dX,
        const half* dK, const half* dV,
        int B, int T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NKV * T * HS )
        {
            const int b = idx / (NKV * T * HS);
            int rest = idx % (NKV * T * HS);
            const int nkv = rest / (T * HS);
            rest = rest % (T * HS);
            const int t = rest / HS;
            const int hs = rest % HS;

            const int base = b * T * (NH + 2 * NKV) * HS + t * (NH + 2 * NKV) * HS;
            const int dk_idx = base + NH * HS + nkv * HS + hs;
            const int dv_idx = base + (NH + NKV) * HS + nkv * HS + hs;

            dX[ dk_idx ] = dK[ idx ];
            dX[ dv_idx ] = dV[ idx ];
        }
    }

    // ---- FP16 QKV permute kernels ----

    __global__ void permute_q_fp16_kernel(
        half* Q,
        const half* X,
        int B, int T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * T * HS )
        {
            const int b = idx / (NH * T * HS);
            int rest = idx % (NH * T * HS);
            const int nh = rest / (T * HS);
            rest = rest % (T * HS);
            const int t = rest / HS;
            const int hs = rest % HS;

            const int q_src = b * T * (NH + 2 * NKV) * HS
                + t * (NH + 2 * NKV) * HS
                + nh * HS + hs;

            Q[ idx ] = __ldcs( &X[ q_src ] );
        }
    }

    __global__ void permute_kv_fp16_kernel(
        half* K, half* V,
        const half* X,
        int B, int T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NKV * T * HS )
        {
            const int b = idx / (NKV * T * HS);
            int rest = idx % (NKV * T * HS);
            const int nkv = rest / (T * HS);
            rest = rest % (T * HS);
            const int t = rest / HS;
            const int hs = rest % HS;

            const int base = b * T * (NH + 2 * NKV) * HS + t * (NH + 2 * NKV) * HS;
            const int k_src = base + NH * HS + nkv * HS + hs;
            const int v_src = base + (NH + NKV) * HS + nkv * HS + hs;

            const int out_idx = b * (NKV * T * HS) + nkv * (T * HS) + t * HS + hs;

            K[ out_idx ] = __ldcs( &X[ k_src ] );
            V[ out_idx ] = __ldcs( &X[ v_src ] );
        }
    }

    __global__ void permute_q_padded_fp16_kernel(
        half* Q,
        const half* X,
        int B, int input_T, int output_T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * output_T * HS )
        {
            const int b = idx / (NH * output_T * HS);
            int rest = idx % (NH * output_T * HS);
            const int nh = rest / (output_T * HS);
            rest = rest % (output_T * HS);
            const int t = rest / HS;
            const int hs = rest % HS;

            const int out_idx = b * (NH * output_T * HS) + nh * (output_T * HS) + t * HS + hs;

            if ( t >= input_T )
            {
                Q[ out_idx ] = __float2half( 0.0f );
                return;
            }

            const int q_src = b * input_T * (NH + 2 * NKV) * HS
                + t * (NH + 2 * NKV) * HS
                + nh * HS + hs;

            Q[ out_idx ] = __ldcs( &X[ q_src ] );
        }
    }

    __global__ void permute_kv_padded_fp16_kernel(
        half* K, half* V,
        const half* X,
        int B, int input_T, int output_T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NKV * output_T * HS )
        {
            const int b = idx / (NKV * output_T * HS);
            int rest = idx % (NKV * output_T * HS);
            const int nkv = rest / (output_T * HS);
            rest = rest % (output_T * HS);
            const int t = rest / HS;
            const int hs = rest % HS;

            const int out_idx = b * (NKV * output_T * HS) + nkv * (output_T * HS) + t * HS + hs;

            if ( t >= input_T )
            {
                K[ out_idx ] = __float2half( 0.0f );
                V[ out_idx ] = __float2half( 0.0f );
                return;
            }

            const int base = b * input_T * (NH + 2 * NKV) * HS + t * (NH + 2 * NKV) * HS;
            const int k_src = base + NH * HS + nkv * HS + hs;
            const int v_src = base + (NH + NKV) * HS + nkv * HS + hs;

            K[ out_idx ] = __ldcs( &X[ k_src ] );
            V[ out_idx ] = __ldcs( &X[ v_src ] );
        }
    }

    __global__ void permute_q_decode_fp16_kernel(
        half* Q,
        const half* X,
        int B, int position, int cache_T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * HS )
        {
            const int b = idx / (NH * HS);
            int rest = idx % (NH * HS);
            const int nh = rest / HS;
            const int hs = rest % HS;

            const int q_src = b * (NH + 2 * NKV) * HS + nh * HS + hs;
            const int out_idx = b * (NH * cache_T * HS) + nh * (cache_T * HS) + position * HS + hs;

            Q[ out_idx ] = __ldcs( &X[ q_src ] );
        }
    }

    __global__ void permute_kv_decode_fp16_kernel(
        half* K, half* V,
        const half* X,
        int B, int position, int cache_T, int NH, int NKV, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NKV * HS )
        {
            const int b = idx / (NKV * HS);
            int rest = idx % (NKV * HS);
            const int nkv = rest / HS;
            const int hs = rest % HS;

            const int base = b * (NH + 2 * NKV) * HS;
            const int k_src = base + NH * HS + nkv * HS + hs;
            const int v_src = base + (NH + NKV) * HS + nkv * HS + hs;
            const int out_idx = b * (NKV * cache_T * HS) + nkv * (cache_T * HS) + position * HS + hs;

            K[ out_idx ] = __ldcs( &X[ k_src ] );
            V[ out_idx ] = __ldcs( &X[ v_src ] );
        }
    }

    // ========================================================================
    // Host launchers — FP32
    // ========================================================================

    void cuda_gqa_permute_qkv_fp32(
        float* Q, float* K, float* V,
        const float* X,
        int B, int T, int NH, int NKV, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;

        permute_q_fp32_kernel << <ceil_div( B * NH * T * HS, block_size ), block_size, 0, stream >> > (
            Q, X, B, T, NH, NKV, HS);
        cudaCheck( cudaGetLastError() );

        permute_kv_fp32_kernel << <ceil_div( B * NKV * T * HS, block_size ), block_size, 0, stream >> > (
            K, V, X, B, T, NH, NKV, HS);
        cudaCheck( cudaGetLastError() );
    }

    void cuda_gqa_permute_qkv_padded_fp32(
        float* Q, float* K, float* V,
        const float* X,
        int B, int input_T, int output_T, int NH, int NKV, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;

        permute_q_padded_fp32_kernel << <ceil_div( B * NH * output_T * HS, block_size ), block_size, 0, stream >> > (
            Q, X, B, input_T, output_T, NH, NKV, HS);
        cudaCheck( cudaGetLastError() );

        permute_kv_padded_fp32_kernel << <ceil_div( B * NKV * output_T * HS, block_size ), block_size, 0, stream >> > (
            K, V, X, B, input_T, output_T, NH, NKV, HS);
        cudaCheck( cudaGetLastError() );
    }

    void cuda_gqa_permute_qkv_decode_fp32(
        float* Q, float* K, float* V,
        const float* X,
        int B, int position, int cache_T, int NH, int NKV, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;

        permute_q_decode_fp32_kernel << <ceil_div( B * NH * HS, block_size ), block_size, 0, stream >> > (
            Q, X, B, position, cache_T, NH, NKV, HS);
        cudaCheck( cudaGetLastError() );

        permute_kv_decode_fp32_kernel << <ceil_div( B * NKV * HS, block_size ), block_size, 0, stream >> > (
            K, V, X, B, position, cache_T, NH, NKV, HS);
        cudaCheck( cudaGetLastError() );
    }

    void cuda_gqa_expand_kv_fp32(
        float* k_exp, float* v_exp,
        const float* k_compact, const float* v_compact,
        int B, int T, int NH, int NKV, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int num_blocks = ceil_div( B * NH * T * HS, block_size );

        expand_kv_fp32_kernel << <num_blocks, block_size, 0, stream >> > (
            k_exp, v_exp, k_compact, v_compact, B, T, NH, NKV, HS);
        cudaCheck( cudaGetLastError() );
    }

    void cuda_gqa_reduce_kv_grad_fp32(
        float* dk_compact, float* dv_compact,
        const float* dk_exp, const float* dv_exp,
        int B, int T, int NH, int NKV, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int num_blocks = ceil_div( B * NKV * T * HS, block_size );

        reduce_kv_grad_fp32_kernel << <num_blocks, block_size, 0, stream >> > (
            dk_compact, dv_compact, dk_exp, dv_exp, B, T, NH, NKV, HS);
        cudaCheck( cudaGetLastError() );
    }

    void cuda_gqa_permute_backward_fp32(
        float* dX,
        const float* dQ, const float* dK, const float* dV,
        int B, int T, int NH, int NKV, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;

        permute_backward_q_fp32_kernel << <ceil_div( B * NH * T * HS, block_size ), block_size, 0, stream >> > (
            dX, dQ, B, T, NH, NKV, HS);
        cudaCheck( cudaGetLastError() );

        permute_backward_kv_fp32_kernel << <ceil_div( B * NKV * T * HS, block_size ), block_size, 0, stream >> > (
            dX, dK, dV, B, T, NH, NKV, HS);
        cudaCheck( cudaGetLastError() );
    }

    // ========================================================================
    // Host launchers — FP16
    // ========================================================================

    void cuda_gqa_permute_qkv_fp16(
        half* Q, half* K, half* V,
        const half* X,
        int B, int T, int NH, int NKV, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;

        permute_q_fp16_kernel << <ceil_div( B * NH * T * HS, block_size ), block_size, 0, stream >> > (
            Q, X, B, T, NH, NKV, HS);
        cudaCheck( cudaGetLastError() );

        permute_kv_fp16_kernel << <ceil_div( B * NKV * T * HS, block_size ), block_size, 0, stream >> > (
            K, V, X, B, T, NH, NKV, HS);
        cudaCheck( cudaGetLastError() );
    }

    void cuda_gqa_permute_qkv_padded_fp16(
        half* Q, half* K, half* V,
        const half* X,
        int B, int input_T, int output_T, int NH, int NKV, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;

        permute_q_padded_fp16_kernel << <ceil_div( B * NH * output_T * HS, block_size ), block_size, 0, stream >> > (
            Q, X, B, input_T, output_T, NH, NKV, HS);
        cudaCheck( cudaGetLastError() );

        permute_kv_padded_fp16_kernel << <ceil_div( B * NKV * output_T * HS, block_size ), block_size, 0, stream >> > (
            K, V, X, B, input_T, output_T, NH, NKV, HS);
        cudaCheck( cudaGetLastError() );
    }

    void cuda_gqa_permute_qkv_decode_fp16(
        half* Q, half* K, half* V,
        const half* X,
        int B, int position, int cache_T, int NH, int NKV, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;

        permute_q_decode_fp16_kernel << <ceil_div( B * NH * HS, block_size ), block_size, 0, stream >> > (
            Q, X, B, position, cache_T, NH, NKV, HS);
        cudaCheck( cudaGetLastError() );

        permute_kv_decode_fp16_kernel << <ceil_div( B * NKV * HS, block_size ), block_size, 0, stream >> > (
            K, V, X, B, position, cache_T, NH, NKV, HS);
        cudaCheck( cudaGetLastError() );
    }

    void cuda_gqa_expand_kv_fp16(
        half* k_exp, half* v_exp,
        const half* k_compact, const half* v_compact,
        int B, int T, int NH, int NKV, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int num_blocks = ceil_div( B * NH * T * HS, block_size );

        expand_kv_fp16_kernel << <num_blocks, block_size, 0, stream >> > (
            k_exp, v_exp, k_compact, v_compact, B, T, NH, NKV, HS);
        cudaCheck( cudaGetLastError() );
    }

    void cuda_gqa_reduce_kv_grad_fp16(
        half* dk_compact, half* dv_compact,
        const half* dk_exp, const half* dv_exp,
        int B, int T, int NH, int NKV, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int num_blocks = ceil_div( B * NKV * T * HS, block_size );

        reduce_kv_grad_fp16_kernel << <num_blocks, block_size, 0, stream >> > (
            dk_compact, dv_compact, dk_exp, dv_exp, B, T, NH, NKV, HS);
        cudaCheck( cudaGetLastError() );
    }

    void cuda_gqa_permute_backward_fp16(
        half* dX,
        const half* dQ, const half* dK, const half* dV,
        int B, int T, int NH, int NKV, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;

        permute_backward_q_fp16_kernel << <ceil_div( B * NH * T * HS, block_size ), block_size, 0, stream >> > (
            dX, dQ, B, T, NH, NKV, HS);
        cudaCheck( cudaGetLastError() );

        permute_backward_kv_fp16_kernel << <ceil_div( B * NKV * T * HS, block_size ), block_size, 0, stream >> > (
            dX, dK, dV, B, T, NH, NKV, HS);
        cudaCheck( cudaGetLastError() );
    }

}
