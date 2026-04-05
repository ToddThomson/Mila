/**
 * @file Swiglu.Bf16.cu
 * @brief BF16 CUDA kernels for SwiGLU activation forward and backward passes.
 *
 * Implements the BF16 precision variant of SwiGLU using uint4 vectorized
 * loads and stores (8 BF16 elements per load) with FP32 promotion for all
 * arithmetic.
 *
 * Memory layout (contiguous halves per token):
 * @code
 *   X: [ gate_0 ... gate_(hw-1) | up_0 ... up_(hw-1) ]  per token row  (BF16)
 *   Y: [ y_0   ... y_(hw-1)    ]                         per token row  (BF16)
 * @endcode
 *
 * where hw = half_width = last input dimension / 2.
 * N  = total output elements = T * half_width.
 * half_width is the per-token stride to the up half and must be passed
 * to the kernel. N and half_width are equal only when T=1.
 *
 * Vectorization:
 *   Each thread processes kSwigluBf16VectorWidth (8) output elements.
 *   uint4 loads 16 bytes = 8 x BF16 elements per instruction.
 *   Thread i maps to token (i / vec_half_width), column (i % vec_half_width)
 *   where vec_half_width = half_width / kSwigluBf16VectorWidth.
 *   This indexing is correct for any T >= 1.
 *   Loads are unconditional — no scalar fallback. The op layer enforces that
 *   N and half_width are multiples of kSwigluBf16VectorWidth.
 *   Buffer pointers are guaranteed 64-byte aligned by the Mila tensor allocator
 *   (CUDA_WARP_SIZE * sizeof(__nv_bfloat16) = 64 bytes), satisfying uint4
 *   alignment (16 bytes).
 *
 * Arithmetic precision (FP32 promotion):
 *   All arithmetic is performed in FP32. BF16 values are promoted before any
 *   computation and demoted only at the store. This ensures training stability —
 *   BF16's 7 mantissa bits are insufficient for sigmoid, exp, and gradient
 *   chain products. Matches PyTorch's internal BF16 kernel strategy.
 *
 * Backward tensor types (mixed input/output):
 *   X  (saved activations): BF16 — promoted to FP32 for arithmetic
 *   dY (upstream gradient): FP32 — canonical optimizer boundary format
 *   dX (gradient output):   FP32 — consumed by CUDA Adam or CPU Adam offload
 *
 * See MilaComputeSpec.md for full design rationale.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute::Cuda::Swiglu
{
    /**
     * @brief Vector width for BF16 SwiGLU kernels.
     *
     * Each thread processes this many output elements per invocation.
     * uint4 = 16 bytes = 8 x BF16 elements.
     * The op layer must validate that N % kSwigluBf16VectorWidth == 0
     * and half_width % kSwigluBf16VectorWidth == 0 before launching.
     */
    constexpr int kSwigluBf16VectorWidth = 8;

    // =========================================================================
    // Device helpers
    // =========================================================================

    /**
     * @brief Computes SiLU for a float2 pair in FP32.
     *
     * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
     *
     * Both elements computed independently using fast FP32 intrinsics.
     * Caller promotes __nv_bfloat162 to float2 via __bfloat1622float2 before calling.
     *
     * @param v float2 gate values in FP32
     * @return float2 { SiLU(v.x), SiLU(v.y) }
     */
    __device__ __forceinline__ float2 silu_float2( float2 v )
    {
        float2 result;
        result.x = v.x * __frcp_rn( 1.0f + __expf( -v.x ) );
        result.y = v.y * __frcp_rn( 1.0f + __expf( -v.y ) );
        return result;
    }

    /**
     * @brief Computes sigmoid and d(SiLU)/d(gate) for a float2 pair in FP32.
     *
     * @code
     *   sigmoid(gate)       = 1 / (1 + exp(-gate))
     *   d(SiLU)/d(gate)     = sigmoid * (1 + gate * (1 - sigmoid))
     * @endcode
     *
     * Both sigmoid and dswish are returned as outputs because the backward
     * kernel needs sigmoid to compute swish = gate * sigmoid (for dup),
     * and dswish for the gate gradient. Computing both in one call avoids
     * a redundant __expf.
     *
     * @param gate   float2 gate values in FP32
     * @param sig    Output: float2 sigmoid(gate)
     * @param dswish Output: float2 d(SiLU)/d(gate)
     */
    __device__ __forceinline__ void silu_grad_float2(
        float2 gate,
        float2& sig,
        float2& dswish )
    {
        sig.x = __frcp_rn( 1.0f + __expf( -gate.x ) );
        sig.y = __frcp_rn( 1.0f + __expf( -gate.y ) );
        dswish.x = sig.x * (1.0f + gate.x * (1.0f - sig.x));
        dswish.y = sig.y * (1.0f + gate.y * (1.0f - sig.y));
    }

    // =========================================================================
    // Forward kernel
    // =========================================================================

    /**
     * @brief BF16 SwiGLU forward kernel — uint4 vectorized, FP32 arithmetic.
     *
     * Computes SwiGLU(gate, up) = SiLU(gate) * up for 8 BF16 elements per thread.
     * All arithmetic is in FP32. Input and output are BF16.
     *
     * Thread i maps to:
     *   token        = i / vec_half_width
     *   col          = i % vec_half_width
     *   gate_vec     = token * vec_half_width * 2 + col
     *   up_vec       = token * vec_half_width * 2 + vec_half_width + col
     *
     * @param Y          BF16 output buffer [N], 64-byte aligned
     * @param X          BF16 input buffer [T * 2 * half_width], 64-byte aligned
     * @param vec_N      Total uint4 output chunks (N / kSwigluBf16VectorWidth)
     * @param half_width Per-token output element count (last input dim / 2)
     */
    __global__ void swiglu_forward_bf16_kernel(
        __nv_bfloat16* __restrict__ Y,
        const __nv_bfloat16* __restrict__ X,
        int vec_N,
        int half_width )
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if ( i < vec_N )
        {
            int vec_half_width = half_width / kSwigluBf16VectorWidth;
            int token = i / vec_half_width;
            int col = i % vec_half_width;
            int gate_vec = token * vec_half_width * 2 + col;
            int up_vec = token * vec_half_width * 2 + vec_half_width + col;

            // Load 8 BF16 gate and up values as uint4
            uint4 gate_raw = reinterpret_cast<const uint4*>( X )[ gate_vec ];
            uint4 up_raw = reinterpret_cast<const uint4*>( X )[ up_vec ];

            // Reinterpret as four __nv_bfloat162 pairs (2 BF16 per pair)
            __nv_bfloat162 gate01 = reinterpret_cast<const __nv_bfloat162*>(&gate_raw)[ 0 ];
            __nv_bfloat162 gate23 = reinterpret_cast<const __nv_bfloat162*>(&gate_raw)[ 1 ];
            __nv_bfloat162 gate45 = reinterpret_cast<const __nv_bfloat162*>(&gate_raw)[ 2 ];
            __nv_bfloat162 gate67 = reinterpret_cast<const __nv_bfloat162*>(&gate_raw)[ 3 ];

            __nv_bfloat162 up01 = reinterpret_cast<const __nv_bfloat162*>(&up_raw)[ 0 ];
            __nv_bfloat162 up23 = reinterpret_cast<const __nv_bfloat162*>(&up_raw)[ 1 ];
            __nv_bfloat162 up45 = reinterpret_cast<const __nv_bfloat162*>(&up_raw)[ 2 ];
            __nv_bfloat162 up67 = reinterpret_cast<const __nv_bfloat162*>(&up_raw)[ 3 ];

            // Promote to FP32 for arithmetic
            float2 g01f = __bfloat1622float2( gate01 );
            float2 g23f = __bfloat1622float2( gate23 );
            float2 g45f = __bfloat1622float2( gate45 );
            float2 g67f = __bfloat1622float2( gate67 );

            float2 u01f = __bfloat1622float2( up01 );
            float2 u23f = __bfloat1622float2( up23 );
            float2 u45f = __bfloat1622float2( up45 );
            float2 u67f = __bfloat1622float2( up67 );

            // SiLU(gate) * up in FP32
            float2 s01f = silu_float2( g01f );
            float2 s23f = silu_float2( g23f );
            float2 s45f = silu_float2( g45f );
            float2 s67f = silu_float2( g67f );

            float2 y01f = { s01f.x * u01f.x, s01f.y * u01f.y };
            float2 y23f = { s23f.x * u23f.x, s23f.y * u23f.y };
            float2 y45f = { s45f.x * u45f.x, s45f.y * u45f.y };
            float2 y67f = { s67f.x * u67f.x, s67f.y * u67f.y };

            // Demote to BF16 and pack as uint4 for store
            __nv_bfloat162 y01 = __float22bfloat162_rn( y01f );
            __nv_bfloat162 y23 = __float22bfloat162_rn( y23f );
            __nv_bfloat162 y45 = __float22bfloat162_rn( y45f );
            __nv_bfloat162 y67 = __float22bfloat162_rn( y67f );

            uint4 out_raw;
            reinterpret_cast<__nv_bfloat162*>(&out_raw)[ 0 ] = y01;
            reinterpret_cast<__nv_bfloat162*>(&out_raw)[ 1 ] = y23;
            reinterpret_cast<__nv_bfloat162*>(&out_raw)[ 2 ] = y45;
            reinterpret_cast<__nv_bfloat162*>(&out_raw)[ 3 ] = y67;

            reinterpret_cast<uint4*>(Y)[ i ] = out_raw;
        }
    }

    // =========================================================================
    // Backward kernel
    // =========================================================================

    /**
     * @brief BF16 SwiGLU backward kernel — uint4 BF16 input, float4 FP32 gradient output.
     *
     * Computes gradients with respect to gate and up inputs:
     * @code
     *   sigmoid  = 1 / (1 + exp(-gate))          [FP32]
     *   swish    = gate * sigmoid                  [FP32]
     *   dL/d_up  = dY * swish                     [FP32]
     *   dL/dgate = dY * up * d(SiLU)/d(gate)      [FP32]
     *            where d(SiLU)/d(gate) = sigmoid * (1 + gate * (1 - sigmoid))
     * @endcode
     *
     * Tensor types:
     *   X  (saved activations): BF16 — promoted to FP32 for arithmetic
     *   dY (upstream gradient): FP32 — canonical optimizer boundary format
     *   dX (gradient output):   FP32 — consumed by CUDA Adam or CPU Adam offload
     *
     * dX layout mirrors X layout — contiguous halves per token:
     * @code
     *   dX: [ dgate_0 ... dgate_(hw-1) | dup_0 ... dup_(hw-1) ]  per token  (FP32)
     * @endcode
     *
     * The gate_vec / up_vec indexing is identical to the forward kernel.
     * dY is indexed linearly over output elements (i), matching Y layout.
     * dX is written at gate_vec / up_vec in float4 units, converting from
     * the BF16 vec_half_width space to the FP32 vec_half_width space.
     *
     * @param dX         FP32 gradient output [T * 2 * half_width], 128-byte aligned
     * @param X          BF16 saved forward input [T * 2 * half_width], 64-byte aligned
     * @param dY         FP32 upstream gradient [N], 128-byte aligned
     * @param vec_N      Total uint4 BF16 output chunks (N / kSwigluBf16VectorWidth)
     * @param half_width Per-token output element count (last input dim / 2)
     */
    __global__ void swiglu_backward_bf16_kernel(
        float* __restrict__ dX,
        const __nv_bfloat16* __restrict__ X,
        const float* __restrict__ dY,
        int vec_N,
        int half_width )
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if ( i < vec_N )
        {
            // BF16 vector indexing (8 elements per chunk)
            int vec_half_width_bf16 = half_width / kSwigluBf16VectorWidth;
            int token = i / vec_half_width_bf16;
            int col_bf16 = i % vec_half_width_bf16;
            int gate_vec_bf16 = token * vec_half_width_bf16 * 2 + col_bf16;
            int up_vec_bf16 = token * vec_half_width_bf16 * 2 + vec_half_width_bf16 + col_bf16;

            // FP32 vector indexing (4 elements per float4, 2 float4 per 8 elements)
            // Each BF16 uint4 chunk (8 elements) maps to 2 float4 chunks
            constexpr int kFp32VectorWidth = 4;
            int vec_half_width_fp32 = half_width / kFp32VectorWidth;
            int col_fp32 = col_bf16 * 2; // 2 float4 per uint4
            int gate_vec_fp32_0 = token * vec_half_width_fp32 * 2 + col_fp32;
            int gate_vec_fp32_1 = gate_vec_fp32_0 + 1;
            int up_vec_fp32_0 = token * vec_half_width_fp32 * 2 + vec_half_width_fp32 + col_fp32;
            int up_vec_fp32_1 = up_vec_fp32_0 + 1;

            // Load 8 BF16 activations
            uint4 gate_raw = reinterpret_cast<const uint4*>(X)[ gate_vec_bf16 ];
            uint4 up_raw = reinterpret_cast<const uint4*>(X)[ up_vec_bf16 ];

            // Load 8 FP32 upstream gradients as 2 x float4
            float4 dy0f = reinterpret_cast<const float4*>(dY)[ i * 2 ];
            float4 dy1f = reinterpret_cast<const float4*>(dY)[ i * 2 + 1 ];

            // Reinterpret BF16 raw loads as __nv_bfloat162 pairs
            __nv_bfloat162 gate01 = reinterpret_cast<const __nv_bfloat162*>(&gate_raw)[ 0 ];
            __nv_bfloat162 gate23 = reinterpret_cast<const __nv_bfloat162*>(&gate_raw)[ 1 ];
            __nv_bfloat162 gate45 = reinterpret_cast<const __nv_bfloat162*>(&gate_raw)[ 2 ];
            __nv_bfloat162 gate67 = reinterpret_cast<const __nv_bfloat162*>(&gate_raw)[ 3 ];

            __nv_bfloat162 up01 = reinterpret_cast<const __nv_bfloat162*>(&up_raw)[ 0 ];
            __nv_bfloat162 up23 = reinterpret_cast<const __nv_bfloat162*>(&up_raw)[ 1 ];
            __nv_bfloat162 up45 = reinterpret_cast<const __nv_bfloat162*>(&up_raw)[ 2 ];
            __nv_bfloat162 up67 = reinterpret_cast<const __nv_bfloat162*>(&up_raw)[ 3 ];

            // Promote activations to FP32
            float2 g01f = __bfloat1622float2( gate01 );
            float2 g23f = __bfloat1622float2( gate23 );
            float2 g45f = __bfloat1622float2( gate45 );
            float2 g67f = __bfloat1622float2( gate67 );

            float2 u01f = __bfloat1622float2( up01 );
            float2 u23f = __bfloat1622float2( up23 );
            float2 u45f = __bfloat1622float2( up45 );
            float2 u67f = __bfloat1622float2( up67 );

            // Compute sigmoid and d(SiLU)/d(gate) for each pair
            float2 sig01, dswish01;
            float2 sig23, dswish23;
            float2 sig45, dswish45;
            float2 sig67, dswish67;

            silu_grad_float2( g01f, sig01, dswish01 );
            silu_grad_float2( g23f, sig23, dswish23 );
            silu_grad_float2( g45f, sig45, dswish45 );
            silu_grad_float2( g67f, sig67, dswish67 );

            // swish = gate * sigmoid (needed for dup gradient)
            float2 swish01 = { g01f.x * sig01.x, g01f.y * sig01.y };
            float2 swish23 = { g23f.x * sig23.x, g23f.y * sig23.y };
            float2 swish45 = { g45f.x * sig45.x, g45f.y * sig45.y };
            float2 swish67 = { g67f.x * sig67.x, g67f.y * sig67.y };

            // Pack into float4 pairs for FP32 output
            // First float4: elements 0-3
            float4 dgate0f;
            dgate0f.x = dy0f.x * u01f.x * dswish01.x;
            dgate0f.y = dy0f.y * u01f.y * dswish01.y;
            dgate0f.z = dy0f.z * u23f.x * dswish23.x;
            dgate0f.w = dy0f.w * u23f.y * dswish23.y;

            // Second float4: elements 4-7
            float4 dgate1f;
            dgate1f.x = dy1f.x * u45f.x * dswish45.x;
            dgate1f.y = dy1f.y * u45f.y * dswish45.y;
            dgate1f.z = dy1f.z * u67f.x * dswish67.x;
            dgate1f.w = dy1f.w * u67f.y * dswish67.y;

            float4 dup0f;
            dup0f.x = dy0f.x * swish01.x;
            dup0f.y = dy0f.y * swish01.y;
            dup0f.z = dy0f.z * swish23.x;
            dup0f.w = dy0f.w * swish23.y;

            float4 dup1f;
            dup1f.x = dy1f.x * swish45.x;
            dup1f.y = dy1f.y * swish45.y;
            dup1f.z = dy1f.z * swish67.x;
            dup1f.w = dy1f.w * swish67.y;

            // Store FP32 gradients at correct token-aware positions
            reinterpret_cast<float4*>(dX)[ gate_vec_fp32_0 ] = dgate0f;
            reinterpret_cast<float4*>(dX)[ gate_vec_fp32_1 ] = dgate1f;
            reinterpret_cast<float4*>(dX)[ up_vec_fp32_0 ] = dup0f;
            reinterpret_cast<float4*>(dX)[ up_vec_fp32_1 ] = dup1f;
        }
    }

    // =========================================================================
    // Host launchers
    // =========================================================================

    /**
     * @brief Launches the BF16 SwiGLU forward kernel.
     *
     * @pre N % kSwigluBf16VectorWidth == 0           (enforced by CudaSwigluOp::forward)
     * @pre half_width % kSwigluBf16VectorWidth == 0  (enforced by CudaSwigluOp::forward)
     * @pre Y, X are 64-byte aligned                  (guaranteed by Mila tensor allocator)
     *
     * @param Y          BF16 output buffer [N]
     * @param X          BF16 input buffer [T * 2 * half_width]
     * @param N          Total output elements (T * half_width)
     * @param half_width Per-token output element count (last input dim / 2)
     * @param stream     CUDA stream
     */
    void cuda_swiglu_forward_bf16(
        __nv_bfloat16* Y,
        const __nv_bfloat16* X,
        int N, int half_width,
        cudaStream_t stream )
    {
        int vec_N = N / kSwigluBf16VectorWidth;
        int block_size = 256;
        int grid_size = (vec_N + block_size - 1) / block_size;

        swiglu_forward_bf16_kernel << <grid_size, block_size, 0, stream >> > (Y, X, vec_N, half_width);

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Launches the BF16 SwiGLU backward kernel.
     *
     * @pre N % kSwigluBf16VectorWidth == 0           (enforced by CudaSwigluOp::backward)
     * @pre half_width % kSwigluBf16VectorWidth == 0  (enforced by CudaSwigluOp::backward)
     * @pre X is 64-byte aligned                      (guaranteed by Mila tensor allocator)
     * @pre dX, dY are 128-byte aligned               (FP32, guaranteed by Mila tensor allocator)
     *
     * @param dX         FP32 gradient output [T * 2 * half_width]
     * @param X          BF16 saved forward input [T * 2 * half_width]
     * @param dY         FP32 upstream gradient [N]
     * @param N          Total output elements (T * half_width)
     * @param half_width Per-token output element count (last input dim / 2)
     * @param stream     CUDA stream
     */
    void cuda_swiglu_backward_bf16(
        float* dX,
        const __nv_bfloat16* X,
        const float* dY,
        int N, int half_width,
        cudaStream_t stream )
    {
        int vec_N = N / kSwigluBf16VectorWidth;
        int block_size = 256;
        int grid_size = (vec_N + block_size - 1) / block_size;

        swiglu_backward_bf16_kernel << <grid_size, block_size, 0, stream >> > (dX, X, dY, vec_N, half_width);

        cudaCheck( cudaGetLastError() );
    }
}
