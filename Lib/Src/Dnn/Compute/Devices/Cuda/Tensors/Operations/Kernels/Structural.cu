/**
 * @file Structural.cu
 * @brief CUDA kernels and host launchers for structural tensor operations.
 *
 * Structural operations reshape or partition tensor data without performing
 * arithmetic. All kernels operate on the last dimension only, which is the
 * only concrete use case in Mila — fused projections always concatenate along
 * the output feature dimension.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdexcept>
#include <type_traits>
#include <cstdint>

#include "Structural.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute::Cuda
{
    // =========================================================================
    // Kernels
    // =========================================================================

    /**
     * @brief Vectorized 3-way last-dimension split kernel (float32).
     *
     * Splits a row-major input tensor of shape [B, T, D] — where D = D0+D1+D2
     * — into three output tensors of shapes [B, T, D0], [B, T, D1], [B, T, D2].
     *
     * Each thread processes one float4 (4 contiguous floats = 16 bytes). The
     * global thread index maps to a (batch*token, vec_col) pair. Within each
     * token row the thread determines which output slice owns vec_col and writes
     * the float4 to the corresponding output buffer.
     *
     * Because D0, D1, D2 are all multiples of 4, a float4 never straddles a
     * slice boundary — the branch taken by all threads in a warp is uniform
     * within each slice region, so there is no warp divergence penalty.
     *
     * @param src   Input buffer [B * T * D floats], read-only.
     * @param out0  Output buffer 0 [B * T * D0 floats].
     * @param out1  Output buffer 1 [B * T * D1 floats].
     * @param out2  Output buffer 2 [B * T * D2 floats].
     * @param B     Batch size.
     * @param T     Sequence length (token count).
     * @param D0    Last-dim size of output 0. Must be a multiple of 4.
     * @param D1    Last-dim size of output 1. Must be a multiple of 4.
     * @param D2    Last-dim size of output 2. Must be a multiple of 4.
     *
     * @pre D0 + D1 + D2 == input last dim.
     * @pre D0, D1, D2 are all multiples of 4.
     * @pre All pointers are 16-byte aligned.
     */
    __global__ void split3_fp32_vectorized_kernel(
        const float* __restrict__ src,
        float* __restrict__ out0,
        float* __restrict__ out1,
        float* __restrict__ out2,
        int N,
        int D0, int D1, int D2 )
    {
        const int D = D0 + D1 + D2;
        const int vecs_per_row = D / 4;
        const int total_vecs = N * vecs_per_row;

        const int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( vec_idx >= total_vecs )
        {
            return;
        }

        // Decompose flat vector index into (batch*token row, column vec)
        const int bt = vec_idx / vecs_per_row;
        const int vec_col = vec_idx % vecs_per_row;

        // First float index within this token row — used for slice boundary test
        const int col = vec_col * 4;

        // Vectorized load from source row
        const float4 val = reinterpret_cast<const float4*>(src + bt * D)[ vec_col ];

        // Scatter to the owning output slice.
        // Branches are warp-uniform within each slice region — no divergence.
        if ( col < D0 )
        {
            const int out_vec = vec_col;
            reinterpret_cast<float4*>( out0 + bt * D0 )[ out_vec ] = val;
        }
        else if ( col < D0 + D1 )
        {
            const int out_vec = vec_col - (D0 / 4);
            reinterpret_cast<float4*>( out1 + bt * D1 )[ out_vec ] = val;
        }
        else
        {
            const int out_vec = vec_col - (D0 / 4) - (D1 / 4);
            reinterpret_cast<float4*>( out2 + bt * D2 )[ out_vec ] = val;
        }
    }

    __global__ void split3_bf16_vectorized_kernel(
        const __nv_bfloat16* __restrict__ src,
        __nv_bfloat16* __restrict__ out0,
        __nv_bfloat16* __restrict__ out1,
        __nv_bfloat16* __restrict__ out2,
        int N,
        int D0, int D1, int D2 )
    {
        const int D = D0 + D1 + D2;
        const int vecs_per_row = D / 8;
        const int total_vecs = N * vecs_per_row;

        const int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( vec_idx >= total_vecs )
        {
            return;
        }

        const int bt = vec_idx / vecs_per_row;
        const int vec_col = vec_idx % vecs_per_row;
        const int col = vec_col * 8;

        const uint4 val = reinterpret_cast<const uint4*>(src + bt * D)[ vec_col ];

        if ( col < D0 )
        {
            const int out_vec = vec_col;
            reinterpret_cast<uint4*>( out0 + bt * D0 )[ out_vec ] = val;
        }
        else if ( col < D0 + D1 )
        {
            const int out_vec = vec_col - (D0 / 8);
            reinterpret_cast<uint4*>( out1 + bt * D1 )[ out_vec ] = val;
        }
        else
        {
            const int out_vec = vec_col - (D0 / 8) - (D1 / 8);
            reinterpret_cast<uint4*>( out2 + bt * D2 )[ out_vec ] = val;
        }
    }

    // =========================================================================
    // Host launchers
    // =========================================================================

    /**
     * @brief Host launcher for the vectorized 3-way last-dimension split (float32).
     *
     * Splits input [B, T, D0+D1+D2] into three contiguous output tensors along
     * the last dimension. Uses float4 vectorized loads and stores for maximum
     * memory throughput.
     *
     * Grid dimensions are computed to cover all B*T*(D/4) float4 elements with
     * a fixed block size of 256 threads.
     *
     * @param src    Input buffer [B * T * (D0+D1+D2) floats], device memory.
     * @param out0   Output buffer 0 [B * T * D0 floats], device memory.
     * @param out1   Output buffer 1 [B * T * D1 floats], device memory.
     * @param out2   Output buffer 2 [B * T * D2 floats], device memory.
     * @param B      Batch size.
     * @param T      Sequence length (token count).
     * @param D0     Last-dim size of output 0. Must be a multiple of 4.
     * @param D1     Last-dim size of output 1. Must be a multiple of 4.
     * @param D2     Last-dim size of output 2. Must be a multiple of 4.
     * @param stream CUDA stream on which to schedule the kernel.
     *
     * @throws std::runtime_error if the kernel launch fails.
     */
    void cuda_split3_fp32(
        const float* __restrict__ src,
        float* __restrict__ out0,
        float* __restrict__ out1,
        float* __restrict__ out2,
        int rows,
        int D0, int D1, int D2,
        cudaStream_t stream )
    {
        const int D = D0 + D1 + D2;
        const int total_vecs = rows * (D / 4);
        const int block_size = 256;
        const int grid_size = ceil_div( total_vecs, block_size );

        split3_fp32_vectorized_kernel <<< grid_size, block_size, 0, stream >>> (
            src,
            out0, out1, out2,
            rows, D0, D1, D2);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_split3_bf16(
        const __nv_bfloat16* __restrict__ src,
        __nv_bfloat16* __restrict__ out0,
        __nv_bfloat16* __restrict__ out1,
        __nv_bfloat16* __restrict__ out2,
        int rows,
        int D0, int D1, int D2,
        cudaStream_t stream )
    {
        const int D = D0 + D1 + D2;
        const int total_vecs = rows * (D / 8);
        const int block_size = 256;
        const int grid_size = ceil_div( total_vecs, block_size );

        split3_bf16_vectorized_kernel << < grid_size, block_size, 0, stream >> > (
            src,
            out0, out1, out2,
            rows, D0, D1, D2);

        cudaCheck( cudaGetLastError() );
    }
}
