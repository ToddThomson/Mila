/**
 * @file CudaMatVecBiasFp32.cu
 * 
 */

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cassert>

namespace Mila::Dnn::Compute::Cuda::Linear
{
    __device__ inline float4 ld_vec( const float* ptr )
    {
        return *reinterpret_cast<const float4*>(ptr);
    }

    /**
     * @brief Optimized CUDA kernel for matrix-vector multiply (M=1 decode path).
     *
     * Computes y[oc] = sum(x[c] * weight[oc, c]) + bias[oc]
     * for each output channel oc independently.
     *
     * Grid:  (OC / BLOCK_OC, 1)         — each block computes 8 output elements
     * Block: (THREADS_PER_OC, BLOCK_OC) — 32 threads collaborate per output element
     *
     * Each output element is a dot product of x[C] with weight row[C].
     * Uses warp shuffle reduction for the final sum.
     *
     * Requirements:
     *   - C must be divisible by 4 (float4 loads)
     *   - Warp shuffle requires THREADS_PER_OC to be a power of 2 <= 32
     */
    static constexpr int kMatvecThreadsPerOC = 32;
    static constexpr int kMatvecBlockOC = 8;

    __global__ void __launch_bounds__( kMatvecThreadsPerOC* kMatvecBlockOC )
        matvec_decode_fp32_kernel(
            float* __restrict__       y,
            const float* __restrict__ x,
            const float* __restrict__ weight,
            const float* __restrict__ bias,
            int C,
            int OC )
    {
        const int oc_base = blockIdx.x * kMatvecBlockOC;
        const int oc = oc_base + threadIdx.y;

        if ( oc >= OC ) return;

        const float* w_row = weight + oc * C;

        float acc = 0.0f;
        const int c_start = threadIdx.x * 4;
        const int c_step = kMatvecThreadsPerOC * 4;

        // Vectorized float4 loads for coalesced memory access
        for ( int c = c_start; c < C; c += c_step )
        {
            float4 x_vec = ld_vec( x + c );
            float4 w_vec = ld_vec( w_row + c );
            acc += x_vec.x * w_vec.x;
            acc += x_vec.y * w_vec.y;
            acc += x_vec.z * w_vec.z;
            acc += x_vec.w * w_vec.w;
        }

        // Warp shuffle reduction across kMatvecThreadsPerOC threads
    #pragma unroll
        for ( int offset = kMatvecThreadsPerOC / 2; offset > 0; offset >>= 1 )
        {
            acc += __shfl_down_sync( 0xffffffff, acc, offset );
        }

        // Thread 0 of each output channel writes result
        if ( threadIdx.x == 0 )
        {
            y[ oc ] = acc + (bias != nullptr ? bias[ oc ] : 0.0f);
        }
    }

    void cuda_matvec_decode_fp32(
        float* y,
        const float* x,
        const float* weight,
        const float* bias,
        int C,
        int OC,
        cudaStream_t stream )
    {
        assert( C % 4 == 0 && "cuda_matvec_decode_fp32: C must be divisible by 4 for float4 loads" );
        assert( OC % kMatvecBlockOC == 0 && "cuda_matvec_decode_fp32: OC must be divisible by kMatvecBlockOC" );

        const dim3 block( kMatvecThreadsPerOC, kMatvecBlockOC );
        const dim3 grid( (OC + kMatvecBlockOC - 1) / kMatvecBlockOC );

        matvec_decode_fp32_kernel<<<grid, block, 0, stream>>>(y, x, weight, bias, C, OC);
    }
}