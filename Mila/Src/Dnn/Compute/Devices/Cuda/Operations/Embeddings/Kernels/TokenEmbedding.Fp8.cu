/**
 * @file TokenEmbedding.Fp8.cu
 * @brief FP8_E4M3 table gather-dequant kernels for the TokenEmbedding operation.
 *
 * D4 Design B: the tied embedding/lm_head table is stored once as FP8_E4M3 with
 * one float32 absmax scale per vocabulary row. The gather dequantizes inline:
 * Y[bt,:] = float(wte_fp8[X[bt],:]) * scales[X[bt]].
 */

#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "TokenEmbedding.cuh"

namespace Mila::Dnn::Compute::Cuda::TokenEmbedding
{
    // ========================================================================
    // Kernels
    // ========================================================================

    /**
     * @brief Full-sequence FP8-table gather-dequant kernel (BF16 output).
     *
     * Each thread covers 8 elements: one int2 (8-byte) FP8 load, eight
     * fp8->float conversions scaled by the row scale, one int4 (16-byte)
     * BF16 store. Byte offsets use int64 arithmetic -- a 262K x 3840 table
     * has ~1e9 elements, too close to INT_MAX for int indexing.
     */
    __global__ void token_embedding_forward_bf16_qfp8_kernel(
        __nv_bfloat16* __restrict__       Y,
        const int* __restrict__           X,
        const __nv_fp8_e4m3* __restrict__ Wte,
        const float* __restrict__         Scales,
        int B, int T, int C )
    {
        int C8 = C / 8;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * T * C8 )
        {
            int bt = idx / C8;
            int c8 = idx % C8;
            int ix = X[ bt ];

            const float scale = Scales[ ix ];

            int2 raw = *reinterpret_cast<const int2*>(
                Wte + static_cast<int64_t>( ix ) * C + c8 * 8 );
            const __nv_fp8_e4m3* elems = reinterpret_cast<const __nv_fp8_e4m3*>( &raw );

            __nv_bfloat16 out[ 8 ];

#pragma unroll
            for ( int i = 0; i < 8; ++i )
                out[ i ] = __float2bfloat16( static_cast<float>( elems[ i ] ) * scale );

            *reinterpret_cast<int4*>( Y + static_cast<int64_t>( bt ) * C + c8 * 8 ) =
                *reinterpret_cast<const int4*>( out );
        }
    }

    /**
     * @brief Single-token decode FP8-table gather-dequant kernel (BF16 output).
     */
    __global__ void token_embedding_decode_bf16_qfp8_kernel(
        __nv_bfloat16* __restrict__       Y,
        const int* __restrict__           X,
        const __nv_fp8_e4m3* __restrict__ Wte,
        const float* __restrict__         Scales,
        int B, int C )
    {
        int C8 = C / 8;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * C8 )
        {
            int b = idx / C8;
            int c8 = idx % C8;
            int ix = X[ b ];

            const float scale = Scales[ ix ];

            int2 raw = *reinterpret_cast<const int2*>(
                Wte + static_cast<int64_t>( ix ) * C + c8 * 8 );
            const __nv_fp8_e4m3* elems = reinterpret_cast<const __nv_fp8_e4m3*>( &raw );

            __nv_bfloat16 out[ 8 ];

#pragma unroll
            for ( int i = 0; i < 8; ++i )
                out[ i ] = __float2bfloat16( static_cast<float>( elems[ i ] ) * scale );

            *reinterpret_cast<int4*>( Y + static_cast<int64_t>( b ) * C + c8 * 8 ) =
                *reinterpret_cast<const int4*>( out );
        }
    }

    // ========================================================================
    // Host launchers -- BF16 output, FP8 table
    // ========================================================================

    void cuda_token_embedding_forward_bf16_qfp8(
        __nv_bfloat16* Y, const int* X, const void* wte_fp8, const float* scales,
        int B, int T, int C, cudaStream_t stream )
    {
        assert( C % 8 == 0 );

        constexpr int BLOCK_SIZE = 256;
        int grid = (B * T * (C / 8) + BLOCK_SIZE - 1) / BLOCK_SIZE;

        token_embedding_forward_bf16_qfp8_kernel << <grid, BLOCK_SIZE, 0, stream >> > (
            Y,
            X,
            static_cast<const __nv_fp8_e4m3*>(wte_fp8),
            scales,
            B, T, C);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_token_embedding_decode_bf16_qfp8(
        __nv_bfloat16* Y, const int* X, const void* wte_fp8, const float* scales,
        int B, int C, cudaStream_t stream )
    {
        assert( C % 8 == 0 );

        constexpr int BLOCK_SIZE = 256;
        int grid = (B * (C / 8) + BLOCK_SIZE - 1) / BLOCK_SIZE;

        token_embedding_decode_bf16_qfp8_kernel << <grid, BLOCK_SIZE, 0, stream >> > (
            Y,
            X,
            static_cast<const __nv_fp8_e4m3*>(wte_fp8),
            scales,
            B, C);

        cudaCheck( cudaGetLastError() );
    }
}
