/**
 * @file NgramEmbedding.cu
 * @brief Kernels and host launchers for the hashed n-gram embedding: the hash with its history, and the gather.
 */

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "NgramEmbedding.cuh"

namespace Mila::Dnn::Compute::Cuda::Embedding
{
    namespace
    {
        constexpr int kHashBlockSize = 128;
        constexpr int kGatherBlockSize = 256;

        // Position p of [history ; chunk] for batch row b.
        __device__ __forceinline__ std::int64_t sequenceAt(
            const std::int32_t* old_history, const std::int32_t* row_ids, int history_length, int p )
        {
            return p < history_length ? old_history[ p ] : row_ids[ p - history_length ];
        }

        __global__ void ngram_hash_kernel( const std::int32_t* __restrict__ token_ids, std::int32_t* __restrict__ history,
            std::int32_t* __restrict__ ngram_ids, int length, NgramHashConstants constants )
        {
            __shared__ std::int32_t old_history[ kMaxNgramSize ];

            const int b = blockIdx.x;
            const int history_length = constants.history_length;
            const std::int32_t* row_ids = token_ids + static_cast<std::size_t>( b ) * length;

            if ( threadIdx.x < history_length )
            {
                old_history[ threadIdx.x ] = history[ b * history_length + threadIdx.x ];
            }

            __syncthreads();

            for ( int t = threadIdx.x; t < length; t += blockDim.x )
            {
                const int p = history_length + t;
                std::int64_t shifted[ kMaxNgramSize ];

                for ( int s = 0; s < constants.ngram_size; ++s )
                {
                    // An EOS among the s ids before this one cuts the n-gram; the EOS itself belongs to the segment it ends.
                    std::int64_t value = sequenceAt( old_history, row_ids, history_length, p - s );

                    for ( int back = 1; back <= s; ++back )
                    {
                        if ( sequenceAt( old_history, row_ids, history_length, p - back ) == constants.eos_token_id )
                        {
                            value = constants.eos_token_id;
                            break;
                        }
                    }

                    shifted[ s ] = value;
                }

                std::int32_t* out = ngram_ids + ( static_cast<std::size_t>( b ) * length + t ) * constants.heads;

                for ( int order = 2; order <= constants.ngram_size; ++order )
                {
                    std::uint64_t mixed = static_cast<std::uint64_t>( shifted[ 0 ] ) * static_cast<std::uint64_t>( constants.multipliers[ 0 ] );

                    for ( int s = 1; s < order; ++s )
                    {
                        mixed ^= static_cast<std::uint64_t>( shifted[ s ] ) * static_cast<std::uint64_t>( constants.multipliers[ s ] );
                    }

                    const std::int64_t signed_mixed = static_cast<std::int64_t>( mixed );

                    for ( int head = 0; head < constants.heads_per_ngram; ++head )
                    {
                        const int global_head = ( order - 2 ) * constants.heads_per_ngram + head;
                        const std::int64_t prime = constants.head_vocab_sizes[ global_head ];

                        // A floor mod: % keeps a negative dividend's sign, torch.remainder does not.
                        std::int64_t row = signed_mixed % prime;

                        if ( row < 0 )
                        {
                            row += prime;
                        }

                        out[ global_head ] = static_cast<std::int32_t>( row + constants.head_offsets[ global_head ] );
                    }
                }
            }

            __syncthreads();

            if ( threadIdx.x == 0 )
            {
                for ( int j = 0; j < history_length; ++j )
                {
                    history[ b * history_length + j ] = static_cast<std::int32_t>(
                        sequenceAt( old_history, row_ids, history_length, length + j ) );
                }
            }
        }

        template<typename T>
        __device__ __forceinline__ T quietNan();

        template<>
        __device__ __forceinline__ float quietNan<float>() { return __int_as_float( 0x7fc00000 ); }

        template<>
        __device__ __forceinline__ __nv_bfloat16 quietNan<__nv_bfloat16>() { return __float2bfloat16( __int_as_float( 0x7fc00000 ) ); }

        template<typename T>
        __global__ void ngram_gather_kernel( T* __restrict__ output, const T* __restrict__ table, const std::int32_t* __restrict__ ngram_ids,
            int count, int row_width, int table_rows )
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;

            if ( i >= count * row_width )
            {
                return;
            }

            const int entry = i / row_width;
            const int column = i % row_width;
            const std::int32_t row = ngram_ids[ entry ];

            output[ i ] = ( row < 0 || row >= table_rows )
                ? quietNan<T>()
                : table[ static_cast<std::size_t>( row ) * row_width + column ];
        }

        template<typename T>
        void launchGather( T* output, const T* table, const std::int32_t* ngram_ids, int count, int row_width, int table_rows, cudaStream_t stream )
        {
            const int total = count * row_width;

            if ( total > 0 )
            {
                ngram_gather_kernel<T><<<ceil_div( total, kGatherBlockSize ), kGatherBlockSize, 0, stream>>>(
                    output, table, ngram_ids, count, row_width, table_rows );
            }

            cudaCheck( cudaGetLastError() );
        }
    }

    void cuda_ngram_hash( const std::int32_t* token_ids, std::int32_t* history, std::int32_t* ngram_ids,
        int batch, int length, const NgramHashConstants& constants, cudaStream_t stream )
    {
        if ( batch > 0 )
        {
            ngram_hash_kernel<<<batch, kHashBlockSize, 0, stream>>>( token_ids, history, ngram_ids, length, constants );
        }

        cudaCheck( cudaGetLastError() );
    }

    void cuda_ngram_gather( float* output, const float* table, const std::int32_t* ngram_ids,
        int count, int row_width, int table_rows, cudaStream_t stream )
    {
        launchGather( output, table, ngram_ids, count, row_width, table_rows, stream );
    }

    void cuda_ngram_gather( __nv_bfloat16* output, const __nv_bfloat16* table, const std::int32_t* ngram_ids,
        int count, int row_width, int table_rows, cudaStream_t stream )
    {
        launchGather( output, table, ngram_ids, count, row_width, table_rows, stream );
    }
}
