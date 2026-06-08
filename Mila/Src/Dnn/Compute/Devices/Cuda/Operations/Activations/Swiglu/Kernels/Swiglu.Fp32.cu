/**
 * @file Swiglu.Fp32.cu
 * @brief FP32 CUDA kernels for SwiGLU activation forward and backward passes.
 *
 * Implements the FP32 precision variant of SwiGLU using float4 vectorized
 * loads and stores for maximum memory bandwidth utilization.
 *
 * Memory layout (contiguous halves per token):
 * @code
 *   X: [ gate_0 ... gate_(hw-1) | up_0 ... up_(hw-1) ]  per token row
 *   Y: [ y_0   ... y_(hw-1)    ]                         per token row
 * @endcode
 *
 * where hw = half_width = last input dimension / 2.
 * N  = total output elements = T * half_width.
 * half_width is the per-token stride to the up half and must be passed
 * to the kernel. N and half_width are equal only when T=1.
 *
 * Vectorization:
 *   Each thread processes kSwigluFp32VectorWidth (4) output elements.
 *   Thread i maps to token (i / vec_half_width), column (i % vec_half_width)
 *   where vec_half_width = half_width / kSwigluFp32VectorWidth.
 *   This indexing is correct for any T >= 1.
 *   Loads are unconditional float4 — no scalar fallback. The op layer enforces
 *   that N and half_width are multiples of kSwigluFp32VectorWidth.
 *   Buffer pointers are guaranteed 128-byte aligned by the Mila tensor allocator
 *   (CUDA_WARP_SIZE * sizeof(float) = 128 bytes), satisfying float4 alignment.
 *
 * Backward tensor types:
 *   dY and dX are FP32 — the canonical gradient format at the optimizer boundary,
 *   consumed by both CUDA Adam and CPU Adam offload. X is FP32 (saved activations).
 *
 * See MilaComputeSpec.md for full design rationale.
 */

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute::Cuda::Swiglu
{
    /**
     * @brief Vector width for FP32 SwiGLU kernels.
     *
     * Each thread processes this many output elements per invocation.
     * The op layer must validate that N % kSwigluFp32VectorWidth == 0
     * and half_width % kSwigluFp32VectorWidth == 0 before launching.
     */
    constexpr int kSwigluFp32VectorWidth = 4;

    // =========================================================================
    // Device helpers
    // =========================================================================

    /**
     * @brief Computes SiLU (Sigmoid Linear Unit) for a single float.
     *
     * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
     *
     * Uses fast reciprocal and fast exp intrinsics.
     *
     * @param x Input value
     * @return SiLU(x)
     */
    __device__ __forceinline__ float silu_fp32( float x )
    {
        float sigmoid = __frcp_rn( 1.0f + __expf( -x ) );
        return x * sigmoid;
    }

    // =========================================================================
    // Forward kernel
    // =========================================================================

    /**
     * @brief FP32 SwiGLU forward kernel — float4 vectorized.
     *
     * Computes SwiGLU(gate, up) = SiLU(gate) * up for 4 elements per thread.
     *
     * Thread i maps to:
     *   token    = i / vec_half_width
     *   col      = i % vec_half_width
     *   gate_vec = token * vec_half_width * 2 + col
     *   up_vec   = token * vec_half_width * 2 + vec_half_width + col
     *
     * @param Y          Output buffer [N], 128-byte aligned
     * @param X          Input buffer [T * 2 * half_width], 128-byte aligned
     * @param vec_N      Total float4 output chunks (N / kSwigluFp32VectorWidth)
     * @param half_width Per-token output element count (last input dim / 2)
     */
    __global__ void swiglu_forward_fp32_kernel(
        float* __restrict__ Y,
        const float* __restrict__ X,
        int vec_N,
        int half_width )
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if ( i < vec_N )
        {
            int vec_half_width = half_width / kSwigluFp32VectorWidth;
            int token = i / vec_half_width;
            int col = i % vec_half_width;
            int gate_vec = token * vec_half_width * 2 + col;
            int up_vec = token * vec_half_width * 2 + vec_half_width + col;

            float4 gate4 = reinterpret_cast<const float4*>( X )[ gate_vec ];
            float4 up4 = reinterpret_cast<const float4*>( X )[ up_vec ];

            float4 y4;
            y4.x = silu_fp32( gate4.x ) * up4.x;
            y4.y = silu_fp32( gate4.y ) * up4.y;
            y4.z = silu_fp32( gate4.z ) * up4.z;
            y4.w = silu_fp32( gate4.w ) * up4.w;

            reinterpret_cast<float4*>(Y)[ i ] = y4;
        }
    }

    // =========================================================================
    // Backward kernel
    // =========================================================================

    /**
     * @brief FP32 SwiGLU backward kernel — float4 vectorized.
     *
     * Computes gradients with respect to gate and up inputs:
     * @code
     *   sigmoid  = 1 / (1 + exp(-gate))
     *   swish    = gate * sigmoid                            (= SiLU(gate))
     *   dL/d_up  = dY * swish
     *   dL/dgate = dY * up * sigmoid * (1 + gate * (1 - sigmoid))
     * @endcode
     *
     * dX layout mirrors X layout — contiguous halves per token:
     * @code
     *   dX: [ dgate_0 ... dgate_(hw-1) | dup_0 ... dup_(hw-1) ]  per token
     * @endcode
     *
     * The gate_vec / up_vec indexing is identical to the forward kernel,
     * ensuring dX writes land in the correct positions for any T >= 1.
     *
     * @param dX         FP32 gradient output [T * 2 * half_width], 128-byte aligned
     * @param X          FP32 saved forward input [T * 2 * half_width], 128-byte aligned
     * @param dY         FP32 upstream gradient [N], 128-byte aligned
     * @param vec_N      Total float4 output chunks (N / kSwigluFp32VectorWidth)
     * @param half_width Per-token output element count (last input dim / 2)
     */
    __global__ void swiglu_backward_fp32_kernel(
        float* __restrict__ dX,
        const float* __restrict__ X,
        const float* __restrict__ dY,
        int vec_N,
        int half_width )
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if ( i < vec_N )
        {
            int vec_half_width = half_width / kSwigluFp32VectorWidth;
            int token = i / vec_half_width;
            int col = i % vec_half_width;
            int gate_vec = token * vec_half_width * 2 + col;
            int up_vec = token * vec_half_width * 2 + vec_half_width + col;

            float4 gate4 = reinterpret_cast<const float4*>( X )[ gate_vec ];
            float4 up4 = reinterpret_cast<const float4*>( X )[ up_vec ];
            float4 dy4 = reinterpret_cast<const float4*>( dY )[ i ];

            float4 dgate4;
            float4 dup4;

            // Element x
            {
                float sigmoid = __frcp_rn( 1.0f + __expf( -gate4.x ) );
                float swish = gate4.x * sigmoid;
                dup4.x = dy4.x * swish;
                dgate4.x = dy4.x * up4.x * sigmoid * (1.0f + gate4.x * (1.0f - sigmoid));
            }

            // Element y
            {
                float sigmoid = __frcp_rn( 1.0f + __expf( -gate4.y ) );
                float swish = gate4.y * sigmoid;
                dup4.y = dy4.y * swish;
                dgate4.y = dy4.y * up4.y * sigmoid * (1.0f + gate4.y * (1.0f - sigmoid));
            }

            // Element z
            {
                float sigmoid = __frcp_rn( 1.0f + __expf( -gate4.z ) );
                float swish = gate4.z * sigmoid;
                dup4.z = dy4.z * swish;
                dgate4.z = dy4.z * up4.z * sigmoid * (1.0f + gate4.z * (1.0f - sigmoid));
            }

            // Element w
            {
                float sigmoid = __frcp_rn( 1.0f + __expf( -gate4.w ) );
                float swish = gate4.w * sigmoid;
                dup4.w = dy4.w * swish;
                dgate4.w = dy4.w * up4.w * sigmoid * (1.0f + gate4.w * (1.0f - sigmoid));
            }

            reinterpret_cast<float4*>(dX)[ gate_vec ] = dgate4;
            reinterpret_cast<float4*>(dX)[ up_vec ] = dup4;
        }
    }

    // =========================================================================
    // Host launchers
    // =========================================================================

    /**
     * @brief Launches the FP32 SwiGLU forward kernel.
     *
     * @pre N % kSwigluFp32VectorWidth == 0           (enforced by CudaSwigluOp::forward)
     * @pre half_width % kSwigluFp32VectorWidth == 0  (enforced by CudaSwigluOp::forward)
     * @pre Y, X are 128-byte aligned                 (guaranteed by Mila tensor allocator)
     *
     * @param Y          FP32 output buffer [N]
     * @param X          FP32 input buffer [T * 2 * half_width]
     * @param N          Total output elements (T * half_width)
     * @param half_width Per-token output element count (last input dim / 2)
     * @param stream     CUDA stream
     */
    void cuda_swiglu_forward_fp32(
        float* Y,
        const float* X,
        int N,
        int half_width,
        cudaStream_t stream )
    {
        int vec_N = N / kSwigluFp32VectorWidth;
        int block_size = 256;
        int grid_size = (vec_N + block_size - 1) / block_size;

        swiglu_forward_fp32_kernel << <grid_size, block_size, 0, stream >> > (Y, X, vec_N, half_width);

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Launches the FP32 SwiGLU backward kernel.
     *
     * @pre N % kSwigluFp32VectorWidth == 0           (enforced by CudaSwigluOp::backward)
     * @pre half_width % kSwigluFp32VectorWidth == 0  (enforced by CudaSwigluOp::backward)
     * @pre dX, X, dY are 128-byte aligned            (guaranteed by Mila tensor allocator)
     *
     * @param dX         FP32 gradient output [T * 2 * half_width]
     * @param X          FP32 saved forward input [T * 2 * half_width]
     * @param dY         FP32 upstream gradient [N]
     * @param N          Total output elements (T * half_width)
     * @param half_width Per-token output element count (last input dim / 2)
     * @param stream     CUDA stream
     */
    void cuda_swiglu_backward_fp32(
        float* dX,
        const float* X,
        const float* dY,
        int N,
        int half_width,
        cudaStream_t stream )
    {
        int vec_N = N / kSwigluFp32VectorWidth;
        int block_size = 256;
        int grid_size = (vec_N + block_size - 1) / block_size;

        swiglu_backward_fp32_kernel << <grid_size, block_size, 0, stream >> > (dX, X, dY, vec_N, half_width);

        cudaCheck( cudaGetLastError() );
    }
}
