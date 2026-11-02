/**
 * @file TokenEmbedding.Int6.cu
 * @brief INT6 table gather-dequant kernels for the TokenEmbedding operation.
 *
 * The table is the layout of Int6Packing.ixx: each row its codes' low nibbles, then their high two bits, with one
 * IEEE half scale per 32 elements. Y[bt,c] = bf16( ( code[X[bt],c] - 32 ) * scales[X[bt], c / 32] ).
 */

#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "TokenEmbedding.cuh"

namespace Mila::Dnn::Compute::Cuda::TokenEmbedding
{
    namespace
    {
        constexpr int kGroupSize = 32;

        /**
         * Eight elements of one table row, starting at column c (a multiple of 8): a 4-byte low-nibble load, a
         * 2-byte high-bit load and one scale. ( code - 32 ) * scale is exact in FP32, so each output rounds once.
         */
        __device__ __forceinline__ void gatherEight(
            __nv_bfloat16* __restrict__  destination,
            const uint8_t* __restrict__  codes,
            const __half* __restrict__   scales,
            int64_t                      row,
            int                          c,
            int                          C )
        {
            const uint8_t* code_row = codes + row * ( C / 2 + C / 4 );
            const uint32_t low = *reinterpret_cast<const uint32_t*>( code_row + c / 2 );
            const uint32_t high = *reinterpret_cast<const uint16_t*>( code_row + C / 2 + c / 4 );
            const float scale = __half2float( scales[ row * ( C / kGroupSize ) + c / kGroupSize ] );

            __nv_bfloat16 out[ 8 ];

#pragma unroll
            for ( int i = 0; i < 8; ++i )
            {
                const int code = static_cast<int>( ( ( low >> ( 4 * i ) ) & 0xFu ) | ( ( ( high >> ( 2 * i ) ) & 0x3u ) << 4 ) );
                out[ i ] = __float2bfloat16( static_cast<float>( code - 32 ) * scale );
            }

            *reinterpret_cast<int4*>( destination ) = *reinterpret_cast<const int4*>( out );
        }

        __global__ void token_embedding_forward_bf16_qint6_kernel(
            __nv_bfloat16* __restrict__ Y,
            const int* __restrict__     X,
            const uint8_t* __restrict__ codes,
            const __half* __restrict__  scales,
            int B, int T, int C )
        {
            const int C8 = C / 8;
            const int64_t idx = static_cast<int64_t>( blockIdx.x ) * blockDim.x + threadIdx.x;

            if ( idx < static_cast<int64_t>( B ) * T * C8 )
            {
                const int64_t bt = idx / C8;
                const int c = static_cast<int>( idx % C8 ) * 8;

                gatherEight( Y + bt * C + c, codes, scales, X[ bt ], c, C );
            }
        }

        __global__ void token_embedding_decode_bf16_qint6_kernel(
            __nv_bfloat16* __restrict__ Y,
            const int* __restrict__     X,
            const uint8_t* __restrict__ codes,
            const __half* __restrict__  scales,
            int B, int C )
        {
            const int C8 = C / 8;
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if ( idx < B * C8 )
            {
                const int b = idx / C8;
                const int c = ( idx % C8 ) * 8;

                gatherEight( Y + static_cast<int64_t>( b ) * C + c, codes, scales, X[ b ], c, C );
            }
        }
    }

    void cuda_token_embedding_forward_bf16_qint6(
        __nv_bfloat16* Y, const int* X, const unsigned char* wte_codes, const void* scales,
        int B, int T, int C, cudaStream_t stream )
    {
        assert( C % 64 == 0 );

        constexpr int BLOCK_SIZE = 256;
        const int64_t threads = static_cast<int64_t>( B ) * T * ( C / 8 );
        const auto grid = static_cast<unsigned>( ( threads + BLOCK_SIZE - 1 ) / BLOCK_SIZE );

        token_embedding_forward_bf16_qint6_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            Y, X, wte_codes, static_cast<const __half*>( scales ), B, T, C );

        cudaCheck( cudaGetLastError() );
    }

    void cuda_token_embedding_decode_bf16_qint6(
        __nv_bfloat16* Y, const int* X, const unsigned char* wte_codes, const void* scales,
        int B, int C, cudaStream_t stream )
    {
        assert( C % 64 == 0 );

        constexpr int BLOCK_SIZE = 256;
        const int grid = ( B * ( C / 8 ) + BLOCK_SIZE - 1 ) / BLOCK_SIZE;

        token_embedding_decode_bf16_qint6_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            Y, X, wte_codes, static_cast<const __half*>( scales ), B, C );

        cudaCheck( cudaGetLastError() );
    }
}
