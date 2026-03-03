#define _USE_MATH_DEFINES
#include <math.h>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "TokenEmbedding.cuh"

namespace Mila::Dnn::Compute::Cuda::Embeddings
{
    // ========================================================================
    // Kernels
    // ========================================================================

    /**
     * @brief Full-sequence token embedding forward kernel (FP32).
     *
     * Y[b,t,:] = wte[X[b,t],:].
     * float4 loads/stores emit 128-bit LDG/STG for maximum memory throughput.
     */
    __global__ void token_embedding_forward_fp32_kernel(
        float4* __restrict__       Y,
        const int* __restrict__    X,
        const float4* __restrict__ Wte,
        int B, int T, int C )
    {
        int C4 = C / 4;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * T * C4 )
        {
            int bt = idx / C4;
            int c4 = idx % C4;
            int ix = X[ bt ];
            Y[ bt * C4 + c4 ] = Wte[ ix * C4 + c4 ];
        }
    }

    /**
     * @brief Full-sequence token embedding backward kernel (FP32).
     *
     * Accumulates gradients into dwte via atomicAdd. atomicAdd is required
     * because multiple (b,t) pairs may reference the same vocabulary row.
     * dwpe scatter is gone — positional encoding is now a separate concern.
     */
    __global__ void token_embedding_backward_fp32_kernel(
        float4* __restrict__       dWte,
        const float4* __restrict__ dY,
        const int* __restrict__    X,
        int B, int T, int C )
    {
        int C4 = C / 4;
        int bt = blockIdx.x;
        int c4 = threadIdx.x;

        if ( bt < B * T )
        {
            int ix = X[ bt ];
            float4 grad = dY[ bt * C4 + c4 ];

            atomicAdd( &dWte[ ix * C4 + c4 ].x, grad.x );
            atomicAdd( &dWte[ ix * C4 + c4 ].y, grad.y );
            atomicAdd( &dWte[ ix * C4 + c4 ].z, grad.z );
            atomicAdd( &dWte[ ix * C4 + c4 ].w, grad.w );
        }
    }

    /**
     * @brief Single-token decode kernel (FP32).
     *
     * Y[b,:] = wte[X[b],:] for each batch element.
     * No positional offset — position is handled downstream by RoPE or ALiBi.
     */
    __global__ void token_embedding_decode_fp32_kernel(
        float4* __restrict__       Y,
        const int* __restrict__    X,
        const float4* __restrict__ Wte,
        int B, int C )
    {
        int C4 = C / 4;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * C4 )
        {
            int b = idx / C4;
            int c4 = idx % C4;
            int ix = X[ b ];
            Y[ b * C4 + c4 ] = Wte[ ix * C4 + c4 ];
        }
    }

    // ========================================================================
    // Host launchers — FP32
    // ========================================================================

    void cuda_token_embedding_forward_fp32(
        float* Y, const int* X, const float* wte,
        int B, int T, int C, cudaStream_t stream )
    {
        assert( C % 4 == 0 );

        constexpr int BLOCK_SIZE = 256;
        int grid = (B * T * (C / 4) + BLOCK_SIZE - 1) / BLOCK_SIZE;

        token_embedding_forward_fp32_kernel << <grid, BLOCK_SIZE, 0, stream >> > (
            reinterpret_cast<float4*>(Y),
            X,
            reinterpret_cast<const float4*>(wte),
            B, T, C);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_token_embedding_backward_fp32(
        float* dwte, const float* dY, const int* X,
        int B, int T, int C, cudaStream_t stream )
    {
        assert( C % 4 == 0 );

        int C4 = C / 4;
        dim3 grid( B * T );
        dim3 block( C4 );

        token_embedding_backward_fp32_kernel << <grid, block, 0, stream >> > (
            reinterpret_cast<float4*>(dwte),
            reinterpret_cast<const float4*>(dY),
            X,
            B, T, C);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_token_embedding_decode_fp32(
        float* Y, const int* X, const float* wte,
        int B, int C, cudaStream_t stream )
    {
        assert( C % 4 == 0 );

        constexpr int BLOCK_SIZE = 256;
        int grid = (B * (C / 4) + BLOCK_SIZE - 1) / BLOCK_SIZE;

        token_embedding_decode_fp32_kernel << <grid, BLOCK_SIZE, 0, stream >> > (
            reinterpret_cast<float4*>(Y),
            X,
            reinterpret_cast<const float4*>(wte),
            B, C);

        cudaCheck( cudaGetLastError() );
    }

    // ========================================================================
    // Host launchers — FP16 (stubs)
    // ========================================================================

    void cuda_token_embedding_forward_fp16(
        half* Y, const int* X, const half* wte,
        int B, int T, int C, cudaStream_t stream )
    {
        // TODO: cuda_token_embedding_forward_fp16 kernel
    }

    void cuda_token_embedding_backward_fp16(
        half* dwte, const half* dY, const int* X,
        int B, int T, int C, cudaStream_t stream )
    {
        // TODO: cuda_token_embedding_backward_fp16 kernel
    }

    void cuda_token_embedding_decode_fp16(
        half* Y, const int* X, const half* wte,
        int B, int C, cudaStream_t stream )
    {
        // TODO: cuda_token_embedding_decode_fp16 kernel
    }
}