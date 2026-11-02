#define _USE_MATH_DEFINES
#include <math.h>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "TokenEmbedding.cuh"

namespace Mila::Dnn::Compute::Cuda::TokenEmbedding
{
    // ========================================================================
    // Kernels
    // ========================================================================

    /**
     * @brief Full-sequence token embedding forward kernel (BF16).
     *
     * Y[b,t,:] = wte[X[b,t],:].
     * int4 reinterpret casts emit 128-bit LDG/STG; no arithmetic is performed
     * so treating the 8 packed BF16 values as opaque 128-bit words is safe.
     */
    __global__ void token_embedding_forward_bf16_kernel(
        int4* __restrict__       Y,
        const int* __restrict__  X,
        const int4* __restrict__ Wte,
        int B, int T, int C )
    {
        int C8 = C / 8;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * T * C8 )
        {
            int bt = idx / C8;
            int c8 = idx % C8;
            int ix = X[ bt ];

            Y[ bt * C8 + c8 ] = Wte[ ix * C8 + c8 ];
        }
    }

    /**
     * @brief Full-sequence token embedding backward kernel (BF16).
     *
     * Accumulates gradients into dwte via __nv_bfloat162 atomicAdd (sm_80+).
     * Each thread covers 4 BF16 elements through two packed-pair atomic updates,
     * matching the per-thread work of the FP32 float4 backward kernel.
     * Multiple (b,t) pairs may alias the same vocabulary row, requiring atomics.
     */
    __global__ void token_embedding_backward_bf16_kernel(
        __nv_bfloat162* __restrict__       dWte,
        const __nv_bfloat162* __restrict__ dY,
        const int* __restrict__            X,
        int B, int T, int C )
    {
        int C2 = C / 2;
        int bt = blockIdx.x;
        int c4 = threadIdx.x;

        if ( bt < B * T )
        {
            int ix = X[ bt ];
            int src = bt * C2 + c4 * 2;
            int dst = ix * C2 + c4 * 2;

            __nv_bfloat162 g0 = dY[ src ];
            __nv_bfloat162 g1 = dY[ src + 1 ];

            atomicAdd( &dWte[ dst ], g0 );
            atomicAdd( &dWte[ dst + 1 ], g1 );
        }
    }

    /**
     * @brief Single-token decode kernel (BF16).
     *
     * Y[b,:] = wte[X[b],:] for each batch element.
     * int4 loads/stores emit 128-bit LDG/STG; no positional offset applied.
     */
    __global__ void token_embedding_decode_bf16_kernel(
        int4* __restrict__       Y,
        const int* __restrict__  X,
        const int4* __restrict__ Wte,
        int B, int C )
    {
        int C8 = C / 8;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * C8 )
        {
            int b = idx / C8;
            int c8 = idx % C8;
            int ix = X[ b ];

            Y[ b * C8 + c8 ] = Wte[ ix * C8 + c8 ];
        }
    }

    // ========================================================================
    // Host launchers — BF16
    // ========================================================================

    void cuda_token_embedding_forward_bf16(
        __nv_bfloat16* Y, const int* X, const __nv_bfloat16* wte,
        int B, int T, int C, cudaStream_t stream )
    {
        assert( C % 8 == 0 );

        constexpr int BLOCK_SIZE = 256;
        int grid = (B * T * (C / 8) + BLOCK_SIZE - 1) / BLOCK_SIZE;

        token_embedding_forward_bf16_kernel << <grid, BLOCK_SIZE, 0, stream >> > (
            reinterpret_cast<int4*>(Y),
            X,
            reinterpret_cast<const int4*>(wte),
            B, T, C);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_token_embedding_backward_bf16(
        __nv_bfloat16* dwte, const __nv_bfloat16* dY, const int* X,
        int B, int T, int C, cudaStream_t stream )
    {
        assert( C % 8 == 0 );

        dim3 grid( B * T );
        dim3 block( C / 4 );

        token_embedding_backward_bf16_kernel << <grid, block, 0, stream >> > (
            reinterpret_cast<__nv_bfloat162*>(dwte),
            reinterpret_cast<const __nv_bfloat162*>(dY),
            X,
            B, T, C);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_token_embedding_decode_bf16(
        __nv_bfloat16* Y, const int* X, const __nv_bfloat16* wte,
        int B, int C, cudaStream_t stream )
    {
        assert( C % 8 == 0 );

        constexpr int BLOCK_SIZE = 256;
        int grid = (B * (C / 8) + BLOCK_SIZE - 1) / BLOCK_SIZE;

        token_embedding_decode_bf16_kernel << <grid, BLOCK_SIZE, 0, stream >> > (
            reinterpret_cast<int4*>(Y),
            X,
            reinterpret_cast<const int4*>(wte),
            B, C);

        cudaCheck( cudaGetLastError() );
    }
}