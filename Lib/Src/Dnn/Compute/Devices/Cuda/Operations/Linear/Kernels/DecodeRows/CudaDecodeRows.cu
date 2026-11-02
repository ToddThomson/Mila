/**
 * @file CudaDecodeRows.cu
 * @brief A few rows of decode through one tensor-core product per Linear, for every weight format (Gemma4Mtp.md 4.7).
 *
 * The rows are the narrow n = 8 operand and the weights are widened to BF16 in registers with their scales on the
 * accumulator -- the idea of Frantar et al., "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large
 * Language Models", arXiv 2408.11743 -- so the work per weight is the same at every row count up to 8 and the product
 * reads weights at decode's bandwidth (Profiling/Microbenchmarks/VerifyRows.cu). Marlin's offline weight reorder and
 * shared-memory pipeline are not used: the mma's k order is permuted instead, so each lane reads Mila's packed layout
 * as stored, prefetching into registers.
 */

#include "CudaDecodeRows.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <format>
#include <stdexcept>
#include "Fp4E2M1.h"

namespace Mila::Dnn::Compute::Cuda::Linear
{
    namespace
    {
        constexpr int kBlockColumns = 32;      // one scale block: two mma k-steps
        constexpr int kWarpsPerBlock = 8;      // split-K across a CUDA block's warps; measured best in VerifyRows.cu

        template<DecodeRowsFormat kFormat>
        constexpr int kWordsPerBlock =
            ( kFormat == DecodeRowsFormat::Bf16 ) ? 4 :
            ( kFormat == DecodeRowsFormat::Fp8PerChannel || kFormat == DecodeRowsFormat::Int6
                || kFormat == DecodeRowsFormat::Codebook3 ) ? 2 : 1;

        template<DecodeRowsFormat kFormat>
        constexpr bool kScalePerBlock =
            kFormat != DecodeRowsFormat::Bf16 && kFormat != DecodeRowsFormat::Fp8PerChannel;

        template<DecodeRowsFormat kFormat>
        constexpr bool kIsCodebook = kFormat == DecodeRowsFormat::Codebook2 || kFormat == DecodeRowsFormat::Codebook3;

        template<DecodeRowsFormat kFormat>
        constexpr int kCodebookEntries = ( kFormat == DecodeRowsFormat::Codebook3 ) ? 8 : 4;

        __device__ __forceinline__ void mma_bf16_16816( float ( &d )[ 4 ], const uint32_t ( &a )[ 4 ], uint32_t b0, uint32_t b1 )
        {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"( d[ 0 ] ), "+f"( d[ 1 ] ), "+f"( d[ 2 ] ), "+f"( d[ 3 ] )
                : "r"( a[ 0 ] ), "r"( a[ 1 ] ), "r"( a[ 2 ] ), "r"( a[ 3 ] ), "r"( b0 ), "r"( b1 ) );
        }

        // Two integer steps as BF16, exactly: 0x43 0x0n is 128 + n for n < 128, so subtracting 128 + offset leaves
        // code - offset.
        __device__ __forceinline__ uint32_t steps_bf16x2( uint32_t code0, uint32_t code1, float offset )
        {
            const uint32_t bits = code0 | ( code1 << 16 ) | 0x43004300u;
            const __nv_bfloat162 biased = *reinterpret_cast<const __nv_bfloat162*>( &bits );
            const __nv_bfloat162 step = __hsub2( biased, __float2bfloat162_rn( 128.0f + offset ) );

            return *reinterpret_cast<const uint32_t*>( &step );
        }

        /**
         * The lane's 8 weights of one row and one 32-column block -- columns 8t to 8t + 7 -- as the raw words of the
         * format. Lane t of a quad owns those columns in both operands, so the mma's k order is a permutation of the
         * columns and every load is one contiguous word per row.
         */
        template<DecodeRowsFormat kFormat>
        __device__ __forceinline__ void load_words(
            const DecodeRowsWeights& weights, int64_t row, int block, int t, int C, uint32_t ( &words )[ kWordsPerBlock<kFormat> ] )
        {
            const auto* codes = static_cast<const std::uint8_t*>( weights.codes );

            if constexpr ( kFormat == DecodeRowsFormat::Int4 || kFormat == DecodeRowsFormat::Fp4 )
            {
                words[ 0 ] = __ldg( reinterpret_cast<const uint32_t*>( codes + row * ( C / 2 ) + block * 16 + 4 * t ) );
            }
            else if constexpr ( kFormat == DecodeRowsFormat::Int6 )
            {
                const std::uint8_t* base = codes + row * ( 3 * C / 4 );
                words[ 0 ] = __ldg( reinterpret_cast<const uint32_t*>( base + block * 16 + 4 * t ) );
                words[ 1 ] = __ldg( reinterpret_cast<const unsigned short*>( base + C / 2 + block * 8 + 2 * t ) );
            }
            else if constexpr ( kFormat == DecodeRowsFormat::Fp8PerChannel )
            {
                const uint2 raw = __ldg( reinterpret_cast<const uint2*>( codes + row * C + block * 32 + 8 * t ) );
                words[ 0 ] = raw.x;
                words[ 1 ] = raw.y;
            }
            else if constexpr ( kFormat == DecodeRowsFormat::Bf16 )
            {
                const uint4 raw = __ldg( reinterpret_cast<const uint4*>(
                    static_cast<const __nv_bfloat16*>( weights.codes ) + row * C + block * 32 + 8 * t ) );
                words[ 0 ] = raw.x;
                words[ 1 ] = raw.y;
                words[ 2 ] = raw.z;
                words[ 3 ] = raw.w;
            }
            else
            {
                words[ 0 ] = __ldg( reinterpret_cast<const unsigned short*>( codes + row * ( C / 4 ) + block * 8 + 2 * t ) );

                if constexpr ( kFormat == DecodeRowsFormat::Codebook3 )
                    words[ 1 ] = __ldg( weights.high_plane + row * ( C / 8 ) + block * 4 + t );
            }
        }

        template<DecodeRowsFormat kFormat>
        __device__ __forceinline__ float load_block_scale( const DecodeRowsWeights& weights, int64_t row, int block, int C )
        {
            if constexpr ( kFormat == DecodeRowsFormat::Fp4 )
            {
                const int groups = C / weights.group_size;

                return __ldg( static_cast<const float*>( weights.scales ) + row * groups + block * kBlockColumns / weights.group_size );
            }
            else
            {
                const int groups = C / weights.group_size;

                return __half2float( static_cast<const __half*>( weights.scales )[ row * groups + block * kBlockColumns / weights.group_size ] );
            }
        }

        /**
         * The lane's 8 weights as four BF16 pairs, pair p holding columns 8t + 2p and 8t + 2p + 1. A codebook entry is
         * carried as two BF16 values whose sum is the FP32 entry to about 2^-16; `residual` takes the second.
         */
        template<DecodeRowsFormat kFormat>
        __device__ __forceinline__ void widen(
            const uint32_t ( &words )[ kWordsPerBlock<kFormat> ], uint32_t table, uint32_t ( &pairs )[ 4 ], uint32_t ( &residual )[ 4 ] )
        {
            if constexpr ( kFormat == DecodeRowsFormat::Int4 )
            {
#pragma unroll
                for ( int p = 0; p < 4; ++p )
                {
                    const uint32_t byte = ( words[ 0 ] >> ( 8 * p ) ) & 0xFFu;
                    pairs[ p ] = steps_bf16x2( byte & 0xFu, byte >> 4, 8.0f );
                }
            }
            else if constexpr ( kFormat == DecodeRowsFormat::Int6 )
            {
#pragma unroll
                for ( int p = 0; p < 4; ++p )
                {
                    const uint32_t low = ( words[ 0 ] >> ( 8 * p ) ) & 0xFFu;
                    const uint32_t high = ( words[ 1 ] >> ( 4 * p ) ) & 0xFu;
                    pairs[ p ] = steps_bf16x2( ( low & 0xFu ) | ( ( high & 0x3u ) << 4 ), ( low >> 4 ) | ( ( high >> 2 ) << 4 ), 32.0f );
                }
            }
            else if constexpr ( kFormat == DecodeRowsFormat::Fp4 )
            {
                __nv_bfloat162 decoded[ 4 ];
                fp4x8_decode_bf16x2( words[ 0 ], decoded );

#pragma unroll
                for ( int p = 0; p < 4; ++p )
                    pairs[ p ] = *reinterpret_cast<const uint32_t*>( &decoded[ p ] );
            }
            else if constexpr ( kFormat == DecodeRowsFormat::Fp8PerChannel )
            {
#pragma unroll
                for ( int p = 0; p < 4; ++p )
                {
                    const auto codes = static_cast<__nv_fp8x2_storage_t>( words[ p / 2 ] >> ( 16 * ( p % 2 ) ) );
                    const float2 values = __half22float2( __half2( __nv_cvt_fp8x2_to_halfraw2( codes, __NV_E4M3 ) ) );
                    const __nv_bfloat162 pair = __floats2bfloat162_rn( values.x, values.y );
                    pairs[ p ] = *reinterpret_cast<const uint32_t*>( &pair );
                }
            }
            else if constexpr ( kFormat == DecodeRowsFormat::Bf16 )
            {
#pragma unroll
                for ( int p = 0; p < 4; ++p )
                    pairs[ p ] = words[ p ];
            }
            else
            {
#pragma unroll
                for ( int p = 0; p < 4; ++p )
                {
                    uint32_t code0 = ( words[ 0 ] >> ( 4 * p ) ) & 0x3u;
                    uint32_t code1 = ( words[ 0 ] >> ( 4 * p + 2 ) ) & 0x3u;

                    if constexpr ( kFormat == DecodeRowsFormat::Codebook3 )
                    {
                        code0 |= ( ( words[ 1 ] >> ( 2 * p ) ) & 0x1u ) << 2;
                        code1 |= ( ( words[ 1 ] >> ( 2 * p + 1 ) ) & 0x1u ) << 2;
                    }

                    // The table is one entry per lane (CodebookGemv.cu: a shuffle, not a runtime-indexed array).
                    const uint32_t entry0 = __shfl_sync( 0xffffffffu, table, static_cast<int>( code0 ) );
                    const uint32_t entry1 = __shfl_sync( 0xffffffffu, table, static_cast<int>( code1 ) );

                    pairs[ p ] = ( entry0 & 0xFFFFu ) | ( entry1 << 16 );
                    residual[ p ] = ( entry0 >> 16 ) | ( entry1 & 0xFFFF0000u );
                }
            }
        }

        template<DecodeRowsFormat kFormat>
        struct Stage
        {
            uint32_t first[ 2 ][ kWordsPerBlock<kFormat> ];     // channel g of the tile, the chunk's two blocks
            uint32_t second[ 2 ][ kWordsPerBlock<kFormat> ];    // channel g + 8
            float scale_first[ 2 ];
            float scale_second[ 2 ];
            uint4 x[ 2 ];                                       // activation row g, columns 8t..8t+7 of each block
        };

        /**
         * A CUDA block owns 16 output channels; its warps split the reduction by chunks of two scale blocks (64
         * columns) and meet in shared memory, summed in warp order. Lane (g, t) holds channels g and g + 8 and, as the
         * B operand, activation row g; rows past `rows` are zero and are not stored.
         */
        template<DecodeRowsFormat kFormat>
        __global__ void __launch_bounds__( kWarpsPerBlock * 32 )
            decode_rows_kernel(
                __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ x, const DecodeRowsWeights weights,
                const __nv_bfloat16* __restrict__ bias, int rows, int C, int OC )
        {
            __shared__ float partial[ kWarpsPerBlock ][ 16 * kDecodeRowsMaximum ];

            const int lane = threadIdx.x & 31;
            const int warp = threadIdx.x >> 5;
            const int g = lane >> 2;
            const int t = lane & 3;

            const int oc_base = blockIdx.x * 16;

            // Channels past OC read the last channel; their sums are never stored.
            const int64_t first = min( oc_base + g, OC - 1 );
            const int64_t second = min( oc_base + g + 8, OC - 1 );

            const int blocks = C / kBlockColumns;
            const int chunks = ( blocks + 1 ) / 2;

            const bool has_row = g < rows;
            const __nv_bfloat16* x_row = x + static_cast<int64_t>( has_row ? g : 0 ) * C;

            uint32_t table = 0;

            if constexpr ( kIsCodebook<kFormat> )
            {
                const float entry = weights.codebook[ lane % kCodebookEntries<kFormat> ];
                const __nv_bfloat16 head = __float2bfloat16_rn( entry );
                const __nv_bfloat16 tail = __float2bfloat16_rn( entry - __bfloat162float( head ) );
                table = static_cast<uint32_t>( __bfloat16_as_ushort( head ) )
                    | ( static_cast<uint32_t>( __bfloat16_as_ushort( tail ) ) << 16 );
            }

            const auto load = [&]( int chunk, Stage<kFormat>& stage )
            {
#pragma unroll
                for ( int b = 0; b < 2; ++b )
                {
                    const int block = 2 * chunk + b;

                    // Warp-uniform: an odd block count leaves the last chunk one block short.
                    if ( block >= blocks )
                        break;

                    load_words<kFormat>( weights, first, block, t, C, stage.first[ b ] );
                    load_words<kFormat>( weights, second, block, t, C, stage.second[ b ] );

                    if constexpr ( kScalePerBlock<kFormat> )
                    {
                        stage.scale_first[ b ] = load_block_scale<kFormat>( weights, first, block, C );
                        stage.scale_second[ b ] = load_block_scale<kFormat>( weights, second, block, C );
                    }

                    stage.x[ b ] = has_row
                        ? __ldg( reinterpret_cast<const uint4*>( x_row + block * kBlockColumns + 8 * t ) )
                        : make_uint4( 0u, 0u, 0u, 0u );
                }
            };

            float acc[ 4 ] = { 0.0f, 0.0f, 0.0f, 0.0f };
            Stage<kFormat> next;

            if ( warp < chunks )
                load( warp, next );

            for ( int chunk = warp; chunk < chunks; chunk += kWarpsPerBlock )
            {
                const Stage<kFormat> current = next;

                if ( chunk + kWarpsPerBlock < chunks )
                    load( chunk + kWarpsPerBlock, next );

#pragma unroll
                for ( int b = 0; b < 2; ++b )
                {
                    if ( 2 * chunk + b >= blocks )
                        break;

                    uint32_t first_pairs[ 4 ], second_pairs[ 4 ];
                    uint32_t first_residual[ 4 ], second_residual[ 4 ];
                    widen<kFormat>( current.first[ b ], table, first_pairs, first_residual );
                    widen<kFormat>( current.second[ b ], table, second_pairs, second_residual );

                    // Logical k 2t + e of step s is column 8t + 4s + e; logical k 2t + 8 + e is column 8t + 4s + 2 + e.
                    // The activations follow the same map, so the B operand is the lane's own 8 columns.
                    const uint4 activations = current.x[ b ];
                    float raw[ 4 ] = { 0.0f, 0.0f, 0.0f, 0.0f };

                    {
                        const uint32_t step0[ 4 ] = { first_pairs[ 0 ], second_pairs[ 0 ], first_pairs[ 1 ], second_pairs[ 1 ] };
                        const uint32_t step1[ 4 ] = { first_pairs[ 2 ], second_pairs[ 2 ], first_pairs[ 3 ], second_pairs[ 3 ] };
                        mma_bf16_16816( raw, step0, activations.x, activations.y );
                        mma_bf16_16816( raw, step1, activations.z, activations.w );
                    }

                    if constexpr ( kIsCodebook<kFormat> )
                    {
                        const uint32_t step0[ 4 ] = { first_residual[ 0 ], second_residual[ 0 ], first_residual[ 1 ], second_residual[ 1 ] };
                        const uint32_t step1[ 4 ] = { first_residual[ 2 ], second_residual[ 2 ], first_residual[ 3 ], second_residual[ 3 ] };
                        mma_bf16_16816( raw, step0, activations.x, activations.y );
                        mma_bf16_16816( raw, step1, activations.z, activations.w );
                    }

                    if constexpr ( kScalePerBlock<kFormat> )
                    {
                        acc[ 0 ] = fmaf( current.scale_first[ b ], raw[ 0 ], acc[ 0 ] );
                        acc[ 1 ] = fmaf( current.scale_first[ b ], raw[ 1 ], acc[ 1 ] );
                        acc[ 2 ] = fmaf( current.scale_second[ b ], raw[ 2 ], acc[ 2 ] );
                        acc[ 3 ] = fmaf( current.scale_second[ b ], raw[ 3 ], acc[ 3 ] );
                    }
                    else
                    {
#pragma unroll
                        for ( int i = 0; i < 4; ++i )
                            acc[ i ] += raw[ i ];
                    }
                }
            }

            // Accumulator element (channel m, row n) at m * 8 + n: c0, c1 are channel g rows 2t, 2t + 1; c2, c3 channel g + 8.
            float* tile = partial[ warp ];
            tile[ g * 8 + 2 * t ] = acc[ 0 ];
            tile[ g * 8 + 2 * t + 1 ] = acc[ 1 ];
            tile[ ( g + 8 ) * 8 + 2 * t ] = acc[ 2 ];
            tile[ ( g + 8 ) * 8 + 2 * t + 1 ] = acc[ 3 ];

            __syncthreads();

            for ( int o = threadIdx.x; o < 16 * kDecodeRowsMaximum; o += kWarpsPerBlock * 32 )
            {
                const int m = o % 16;
                const int n = o / 16;
                const int oc = oc_base + m;

                if ( n >= rows || oc >= OC )
                    continue;

                float sum = 0.0f;

#pragma unroll
                for ( int w = 0; w < kWarpsPerBlock; ++w )
                    sum += partial[ w ][ m * 8 + n ];

                // The per-channel scale applies to the whole sum, as the FP8 matvec applies it.
                if constexpr ( kFormat == DecodeRowsFormat::Fp8PerChannel )
                    sum *= __ldg( static_cast<const float*>( weights.scales ) + oc );

                if ( bias != nullptr )
                    sum += __bfloat162float( bias[ oc ] );

                y[ static_cast<int64_t>( n ) * OC + oc ] = __float2bfloat16( sum );
            }
        }

        template<DecodeRowsFormat kFormat>
        void launch( __nv_bfloat16* y, const __nv_bfloat16* x, const DecodeRowsWeights& weights,
            const __nv_bfloat16* bias, int rows, int C, int OC, cudaStream_t stream )
        {
            const dim3 grid( ( OC + 15 ) / 16 );
            decode_rows_kernel<kFormat><<<grid, kWarpsPerBlock * 32, 0, stream>>>( y, x, weights, bias, rows, C, OC );
        }

        bool groupSizeServes( const DecodeRowsWeights& weights, int C )
        {
            switch ( weights.format )
            {
                case DecodeRowsFormat::Bf16:
                case DecodeRowsFormat::Fp8PerChannel:
                    return true;

                case DecodeRowsFormat::Int4:
                case DecodeRowsFormat::Int6:
                    return weights.group_size == 32;

                default:
                    // A scale block of 32 columns must lie inside one group.
                    return weights.group_size >= kBlockColumns && weights.group_size % kBlockColumns == 0
                        && C % weights.group_size == 0;
            }
        }
    }

    void cuda_decode_rows_bf16(
        __nv_bfloat16* y,
        const __nv_bfloat16* x,
        const DecodeRowsWeights& weights,
        const __nv_bfloat16* bias,
        int rows,
        int C,
        int OC,
        cudaStream_t stream )
    {
        if ( rows < 1 || rows > kDecodeRowsMaximum )
        {
            throw std::invalid_argument( std::format(
                "cuda_decode_rows_bf16: {} rows; one call takes 1 to {}", rows, kDecodeRowsMaximum ) );
        }

        if ( C % kBlockColumns != 0 || !groupSizeServes( weights, C ) )
        {
            throw std::invalid_argument( std::format(
                "cuda_decode_rows_bf16: C {} with group size {} is unsupported; C must be a multiple of {} and of the "
                "group", C, weights.group_size, kBlockColumns ) );
        }

        switch ( weights.format )
        {
            case DecodeRowsFormat::Bf16: launch<DecodeRowsFormat::Bf16>( y, x, weights, bias, rows, C, OC, stream ); break;
            case DecodeRowsFormat::Fp8PerChannel: launch<DecodeRowsFormat::Fp8PerChannel>( y, x, weights, bias, rows, C, OC, stream ); break;
            case DecodeRowsFormat::Fp4: launch<DecodeRowsFormat::Fp4>( y, x, weights, bias, rows, C, OC, stream ); break;
            case DecodeRowsFormat::Int4: launch<DecodeRowsFormat::Int4>( y, x, weights, bias, rows, C, OC, stream ); break;
            case DecodeRowsFormat::Int6: launch<DecodeRowsFormat::Int6>( y, x, weights, bias, rows, C, OC, stream ); break;
            case DecodeRowsFormat::Codebook2: launch<DecodeRowsFormat::Codebook2>( y, x, weights, bias, rows, C, OC, stream ); break;
            case DecodeRowsFormat::Codebook3: launch<DecodeRowsFormat::Codebook3>( y, x, weights, bias, rows, C, OC, stream ); break;
        }
    }
}
