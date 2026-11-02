// Q4_0 expert bank prefill: the routing into expert segments and the two grouped INT8 GEMMs. See Moe.cuh.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "Moe.cuh"
#include "../../Linear/Kernels/Int4/CudaInt4Gemm.cuh"
#include "../../Linear/Kernels/Int4/Int4Int8Mma.cuh"
// Shared activation math source, at the depth-relative path the elementwise kernel uses.
#include "../../../../../../Components/Activations/Activation/Kernels/ElementwiseActivation.h"

namespace Mila::Dnn::Compute::Cuda::Moe
{
    namespace
    {
        using Int4Mma::copyAsync;
        using Int4Mma::commitAsync;
        using Int4Mma::waitAsync;
        using Int4Mma::unpackWeightFragment;
        using Int4Mma::mmaInt8Biased;
        using Int4Mma::kFloatBiasValue;

        // One Q4_0 block: the elements that share an activation scale and a weight scale.
        constexpr int kBlock = Linear::kInt4GemmBlockSize;

        // An expert's segment averages tokens x top_k / experts rows -- 64 for the Gemma 4 26B-A4B at a 1024-row
        // chunk -- so the M tile is half Linear's.
        constexpr int kTileM = 64;
        constexpr int kTileN = 128;
        constexpr int kStages = 2;
        constexpr int kThreads = 256;

        // Warps tile the block 2 (rows) x 4 (columns); each warp owns kTileM / 2 x 32 outputs as m16n8 tiles.
        constexpr int kWarpTilesM = kTileM / 32;
        constexpr int kWarpTilesN = 4;

        constexpr int kRoutingThreads = 1024;
        constexpr int kRoutingWarps = kRoutingThreads / 32;

        template <int kTileK>
        struct TileGeometry
        {
            static constexpr int kBlocksPerTileK = kTileK / kBlock;

            // Linear's padded strides: the fragment loads below are free of bank conflicts at the 128-deep tile.
            static constexpr int kActivationRowBytes = kTileK + 32;
            static constexpr int kWeightRowBytes = kTileK / 2 + 16;

            static constexpr int kActivationTileBytes = kTileM * kActivationRowBytes;
            static constexpr int kWeightTileBytes = kTileN * kWeightRowBytes;
            static constexpr int kActivationScaleBytes = kTileM * kBlocksPerTileK * static_cast<int>( sizeof( float ) );
            static constexpr int kWeightScaleBytes = kTileN * kBlocksPerTileK * static_cast<int>( sizeof( __half ) );
            static constexpr int kStageBytes = kActivationTileBytes + kWeightTileBytes + kActivationScaleBytes + kWeightScaleBytes;
            static constexpr int kSharedBytes = kStages * kStageBytes;

            static_assert( kStageBytes % 16 == 0 );
            static_assert( kTileM * ( kTileK / 16 ) % kThreads == 0 && kTileN * ( kTileK / 32 ) % kThreads == 0,
                "every thread copies the same number of 16-byte chunks" );
            static_assert( kTileM + kTileN <= kThreads, "one thread per scale row" );
        };

        enum class Pass
        {
            // A rows are tokens, B rows an expert's gate and up rows interleaved, the output FP32 gated values.
            Gated,
            // A rows are (token, slot) rows of gated values, B rows an expert's down rows, the output FP32 rows.
            Combine
        };

        /**
         * One M tile of one expert segment against 128 columns. The tile table holds (expert, first row, end row)
         * triples over the permuted rows; blocks past the table's count exit. The main loop is Linear's
         * int4_int8_gemm_kernel with the A rows gathered.
         */
        template <Pass kPass, int kTileK, typename TFunctor>
        __global__ void __launch_bounds__( kThreads ) moe_grouped_int4_int8_gemm_kernel(
            const int8_t* __restrict__   activation_codes,
            const float* __restrict__    activation_scales,
            const uint8_t* __restrict__  weight_codes,
            const __half* __restrict__   weight_scales,
            const int32_t* __restrict__  permuted_rows,
            const int32_t* __restrict__  tile_table,
            const int32_t* __restrict__  tile_count,
            float* __restrict__          output,
            int                          in_features,
            int                          out_features,
            int                          top_k,
            int                          first_flat_row,
            TFunctor                     functor )
        {
            using Geometry = TileGeometry<kTileK>;

            constexpr int kBlocksPerTileK = Geometry::kBlocksPerTileK;
            constexpr int kActivationRowBytes = Geometry::kActivationRowBytes;
            constexpr int kWeightRowBytes = Geometry::kWeightRowBytes;
            constexpr int kActivationTileBytes = Geometry::kActivationTileBytes;
            constexpr int kWeightTileBytes = Geometry::kWeightTileBytes;
            constexpr int kActivationScaleBytes = Geometry::kActivationScaleBytes;
            constexpr int kStageBytes = Geometry::kStageBytes;

            extern __shared__ __align__( 16 ) unsigned char shared[];
            __shared__ int source_rows[ kTileM ];
            __shared__ int flat_rows[ kTileM ];

            const int tile = static_cast<int>( blockIdx.x );

            if ( tile >= *tile_count )
            {
                return;
            }

            const int expert = tile_table[ 3 * tile ];
            const int first_row = tile_table[ 3 * tile + 1 ];
            const int end_row = tile_table[ 3 * tile + 2 ];

            const int thread = static_cast<int>( threadIdx.x );
            const int warp = thread / 32;
            const int lane = thread % 32;
            const int group = lane / 4;
            const int thread_in_group = lane % 4;
            const int warp_row = ( warp / 4 ) * ( kWarpTilesM * 16 );
            const int warp_column = ( warp % 4 ) * ( kWarpTilesN * 8 );

            // Gated: tile columns are interleaved gate and up rows, two per unit.
            const int tile_column = static_cast<int>( blockIdx.y ) * kTileN;

            if ( thread < kTileM )
            {
                const int row = first_row + thread;
                const int flat = row < end_row ? permuted_rows[ row ] : -1;

                flat_rows[ thread ] = flat;
                source_rows[ thread ] = flat < 0 ? -1 : ( kPass == Pass::Gated ? flat / top_k : flat );
            }

            __syncthreads();

            const int64_t expert_first_weight_row =
                static_cast<int64_t>( expert ) * ( kPass == Pass::Gated ? 2 * out_features : out_features );

            // The expert weight row behind a tile column, or -1 past the edge.
            const auto weightRow = [&]( int column ) -> int
            {
                if constexpr ( kPass == Pass::Gated )
                {
                    const int unit = ( tile_column + column ) / 2;

                    return unit < out_features ? ( ( column & 1 ) ? out_features + unit : unit ) : -1;
                }
                else
                {
                    return tile_column + column < out_features ? tile_column + column : -1;
                }
            };

            const int blocks_per_row = in_features / kBlock;
            const int k_tiles = in_features / kTileK;

            const auto loadStage = [&]( int stage, int k_tile )
            {
                unsigned char* base = shared + stage * kStageBytes;
                const int k = k_tile * kTileK;

                constexpr int kActivationChunksPerRow = kTileK / 16;
                constexpr int kWeightChunksPerRow = kTileK / 32;

#pragma unroll
                for ( int part = 0; part < kTileM * kActivationChunksPerRow / kThreads; ++part )
                {
                    const int chunk = thread + part * kThreads;
                    const int row = chunk / kActivationChunksPerRow;
                    const int column = ( chunk % kActivationChunksPerRow ) * 16;
                    const int source_row = source_rows[ row ];
                    const bool valid = source_row >= 0;

                    copyAsync<16>( base + row * kActivationRowBytes + column,
                        activation_codes + static_cast<int64_t>( valid ? source_row : 0 ) * in_features + k + column,
                        valid );
                }

#pragma unroll
                for ( int part = 0; part < kTileN * kWeightChunksPerRow / kThreads; ++part )
                {
                    const int chunk = thread + part * kThreads;
                    const int row = chunk / kWeightChunksPerRow;
                    const int column = ( chunk % kWeightChunksPerRow ) * 16;
                    const int weight_row = weightRow( row );
                    const bool valid = weight_row >= 0;

                    copyAsync<16>( base + kActivationTileBytes + row * kWeightRowBytes + column,
                        weight_codes + ( expert_first_weight_row + ( valid ? weight_row : 0 ) ) * ( in_features / 2 ) + k / 2 + column,
                        valid );
                }

                if ( thread < kTileM )
                {
                    const int source_row = source_rows[ thread ];
                    const bool valid = source_row >= 0;

                    copyAsync<kBlocksPerTileK * sizeof( float )>(
                        base + kActivationTileBytes + kWeightTileBytes + thread * kBlocksPerTileK * sizeof( float ),
                        activation_scales + static_cast<int64_t>( valid ? source_row : 0 ) * blocks_per_row + k / kBlock,
                        valid );
                }
                else if ( thread < kTileM + kTileN )
                {
                    const int row = thread - kTileM;
                    const int weight_row = weightRow( row );
                    const bool valid = weight_row >= 0;

                    copyAsync<kBlocksPerTileK * sizeof( __half )>(
                        base + kActivationTileBytes + kWeightTileBytes + kActivationScaleBytes + row * kBlocksPerTileK * sizeof( __half ),
                        weight_scales + ( expert_first_weight_row + ( valid ? weight_row : 0 ) ) * blocks_per_row + k / kBlock,
                        valid );
                }
            };

            float accumulators[ kWarpTilesM ][ kWarpTilesN ][ 4 ];

#pragma unroll
            for ( int m = 0; m < kWarpTilesM; ++m )
#pragma unroll
                for ( int n = 0; n < kWarpTilesN; ++n )
#pragma unroll
                    for ( int i = 0; i < 4; ++i )
                        accumulators[ m ][ n ][ i ] = 0.0f;

#pragma unroll
            for ( int stage = 0; stage < kStages - 1; ++stage )
            {
                if ( stage < k_tiles )
                    loadStage( stage, stage );

                commitAsync();
            }

            for ( int k_tile = 0; k_tile < k_tiles; ++k_tile )
            {
                waitAsync<kStages - 2>();
                __syncthreads();

                const int next = k_tile + kStages - 1;

                if ( next < k_tiles )
                    loadStage( next % kStages, next );

                commitAsync();

                const unsigned char* base = shared + ( k_tile % kStages ) * kStageBytes;
                const unsigned char* activation_tile = base;
                const unsigned char* weight_tile = base + kActivationTileBytes;
                const float* activation_scale_tile = reinterpret_cast<const float*>( base + kActivationTileBytes + kWeightTileBytes );
                const __half* weight_scale_tile = reinterpret_cast<const __half*>(
                    base + kActivationTileBytes + kWeightTileBytes + kActivationScaleBytes );

#pragma unroll
                for ( int block = 0; block < kBlocksPerTileK; ++block )
                {
                    uint32_t b[ kWarpTilesN ][ 2 ];
                    float weight_scale[ kWarpTilesN ][ 2 ];
                    float bias_times_scale[ kWarpTilesN ][ 2 ];

#pragma unroll
                    for ( int n = 0; n < kWarpTilesN; ++n )
                    {
                        const int column = warp_column + n * 8;
                        const uint32_t word = *reinterpret_cast<const uint32_t*>(
                            weight_tile + ( column + group ) * kWeightRowBytes + block * 16 + 4 * thread_in_group );
                        unpackWeightFragment( word, b[ n ][ 0 ], b[ n ][ 1 ] );

                        const int output_column = column + 2 * thread_in_group;
                        weight_scale[ n ][ 0 ] = __half2float( weight_scale_tile[ output_column * kBlocksPerTileK + block ] );
                        weight_scale[ n ][ 1 ] = __half2float( weight_scale_tile[ ( output_column + 1 ) * kBlocksPerTileK + block ] );
                        bias_times_scale[ n ][ 0 ] = -kFloatBiasValue * weight_scale[ n ][ 0 ];
                        bias_times_scale[ n ][ 1 ] = -kFloatBiasValue * weight_scale[ n ][ 1 ];
                    }

#pragma unroll
                    for ( int m = 0; m < kWarpTilesM; ++m )
                    {
                        const int row = warp_row + m * 16 + group;
                        const uint2 upper = *reinterpret_cast<const uint2*>(
                            activation_tile + row * kActivationRowBytes + block * 32 + 8 * thread_in_group );
                        const uint2 lower = *reinterpret_cast<const uint2*>(
                            activation_tile + ( row + 8 ) * kActivationRowBytes + block * 32 + 8 * thread_in_group );

                        const float upper_scale = activation_scale_tile[ row * kBlocksPerTileK + block ];
                        const float lower_scale = activation_scale_tile[ ( row + 8 ) * kBlocksPerTileK + block ];

#pragma unroll
                        for ( int n = 0; n < kWarpTilesN; ++n )
                        {
                            float biased[ 4 ];
                            mmaInt8Biased( biased, upper.x, lower.x, upper.y, lower.y, b[ n ][ 0 ], b[ n ][ 1 ] );

                            // fmaf( biased, w, -bias * w ) is exactly dot * w rounded once, as in Linear.
                            float* accumulator = accumulators[ m ][ n ];
                            accumulator[ 0 ] = fmaf( fmaf( biased[ 0 ], weight_scale[ n ][ 0 ], bias_times_scale[ n ][ 0 ] ), upper_scale, accumulator[ 0 ] );
                            accumulator[ 1 ] = fmaf( fmaf( biased[ 1 ], weight_scale[ n ][ 1 ], bias_times_scale[ n ][ 1 ] ), upper_scale, accumulator[ 1 ] );
                            accumulator[ 2 ] = fmaf( fmaf( biased[ 2 ], weight_scale[ n ][ 0 ], bias_times_scale[ n ][ 0 ] ), lower_scale, accumulator[ 2 ] );
                            accumulator[ 3 ] = fmaf( fmaf( biased[ 3 ], weight_scale[ n ][ 1 ], bias_times_scale[ n ][ 1 ] ), lower_scale, accumulator[ 3 ] );
                        }
                    }
                }
            }

            waitAsync<0>();

            // A thread's two columns are adjacent: for the gated pass one unit's gate and up.
#pragma unroll
            for ( int m = 0; m < kWarpTilesM; ++m )
            {
#pragma unroll
                for ( int n = 0; n < kWarpTilesN; ++n )
                {
                    const int column = tile_column + warp_column + n * 8 + 2 * thread_in_group;

#pragma unroll
                    for ( int half_tile = 0; half_tile < 2; ++half_tile )
                    {
                        const int flat = flat_rows[ warp_row + m * 16 + group + half_tile * 8 ];
                        const float first = accumulators[ m ][ n ][ 2 * half_tile ];
                        const float second = accumulators[ m ][ n ][ 2 * half_tile + 1 ];

                        if ( flat < 0 )
                        {
                            continue;
                        }

                        if constexpr ( kPass == Pass::Gated )
                        {
                            const int unit = column / 2;

                            if ( unit < out_features )
                            {
                                output[ static_cast<int64_t>( flat ) * out_features + unit ] = functor.fwd( first ) * second;
                            }
                        }
                        else
                        {
                            if ( column < out_features )
                            {
                                *reinterpret_cast<float2*>( output + static_cast<int64_t>( flat - first_flat_row ) * out_features + column ) =
                                    make_float2( first, second );
                            }
                        }
                    }
                }
            }
        }

        /**
         * Counts rows per expert, then places each row at its expert's offset plus the number of earlier rows
         * routed there, so a segment lists its rows in ascending (token, slot) order. One block.
         */
        __global__ void __launch_bounds__( kRoutingThreads ) moe_route_kernel(
            const int32_t* __restrict__ indices, int rows, int experts,
            int32_t* __restrict__ expert_offsets, int32_t* __restrict__ permuted_rows )
        {
            __shared__ int counts[ kGroupedMaximumExperts ];
            __shared__ int running[ kGroupedMaximumExperts ];
            __shared__ int warp_counts[ kRoutingWarps * kGroupedMaximumExperts ];

            const int thread = static_cast<int>( threadIdx.x );
            const int warp = thread / 32;
            const int lane = thread % 32;

            for ( int expert = thread; expert < experts; expert += kRoutingThreads )
            {
                counts[ expert ] = 0;
            }

            __syncthreads();

            for ( int row = thread; row < rows; row += kRoutingThreads )
            {
                const int expert = indices[ row ];

                if ( expert >= 0 && expert < experts )
                {
                    atomicAdd( &counts[ expert ], 1 );
                }
            }

            __syncthreads();

            if ( thread == 0 )
            {
                int offset = 0;

                for ( int expert = 0; expert < experts; ++expert )
                {
                    expert_offsets[ expert ] = offset;
                    running[ expert ] = offset;
                    offset += counts[ expert ];
                }

                expert_offsets[ experts ] = offset;
            }

            for ( int chunk = 0; chunk < rows; chunk += kRoutingThreads )
            {
                for ( int entry = thread; entry < kRoutingWarps * experts; entry += kRoutingThreads )
                {
                    warp_counts[ entry ] = 0;
                }

                __syncthreads();

                const int row = chunk + thread;
                const int expert = row < rows ? indices[ row ] : -1;
                const bool valid = expert >= 0 && expert < experts;
                const unsigned peers = __match_any_sync( 0xffffffffu, valid ? expert : -1 );
                const int rank = __popc( peers & ( ( 1u << lane ) - 1u ) );

                if ( valid && rank == 0 )
                {
                    warp_counts[ warp * experts + expert ] = __popc( peers );
                }

                __syncthreads();

                if ( valid )
                {
                    int position = running[ expert ] + rank;

                    for ( int earlier = 0; earlier < warp; ++earlier )
                    {
                        position += warp_counts[ earlier * experts + expert ];
                    }

                    permuted_rows[ position ] = row;
                }

                __syncthreads();

                for ( int counted = thread; counted < experts; counted += kRoutingThreads )
                {
                    int sum = 0;

                    for ( int each = 0; each < kRoutingWarps; ++each )
                    {
                        sum += warp_counts[ each * experts + counted ];
                    }

                    running[ counted ] += sum;
                }

                // The next chunk clears warp_counts, which the sums above are still reading.
                __syncthreads();
            }
        }

        // First position in [first, last) whose row is at least `bound`; the rows ascend within a segment.
        __device__ int lowerBound( const int32_t* rows, int first, int last, int bound )
        {
            while ( first < last )
            {
                const int middle = first + ( last - first ) / 2;

                if ( rows[ middle ] < bound )
                {
                    first = middle + 1;
                }
                else
                {
                    last = middle;
                }
            }

            return first;
        }

        // The M tiles of every segment's rows whose token lies in [first_token, end_token). One thread per expert.
        __global__ void __launch_bounds__( kGroupedMaximumExperts ) moe_tile_kernel(
            const int32_t* __restrict__ expert_offsets, const int32_t* __restrict__ permuted_rows,
            int experts, int top_k, int first_token, int end_token,
            int32_t* __restrict__ tile_table, int32_t* __restrict__ tile_count )
        {
            __shared__ int tile_offsets[ kGroupedMaximumExperts ];

            const int expert = static_cast<int>( threadIdx.x );
            int first_row = 0;
            int end_row = 0;
            int tiles = 0;

            if ( expert < experts )
            {
                const int segment_end = expert_offsets[ expert + 1 ];

                first_row = lowerBound( permuted_rows, expert_offsets[ expert ], segment_end, first_token * top_k );
                end_row = lowerBound( permuted_rows, first_row, segment_end, end_token * top_k );
                tiles = ( end_row - first_row + kTileM - 1 ) / kTileM;
            }

            tile_offsets[ expert ] = tiles;
            __syncthreads();

            if ( expert == 0 )
            {
                int offset = 0;

                for ( int each = 0; each < experts; ++each )
                {
                    const int count = tile_offsets[ each ];

                    tile_offsets[ each ] = offset;
                    offset += count;
                }

                *tile_count = offset;
            }

            __syncthreads();

            for ( int index = 0; index < tiles; ++index )
            {
                const int entry = tile_offsets[ expert ] + index;

                tile_table[ 3 * entry ] = expert;
                tile_table[ 3 * entry + 1 ] = first_row + index * kTileM;
                tile_table[ 3 * entry + 2 ] = min( first_row + ( index + 1 ) * kTileM, end_row );
            }
        }

        // Linear's per-block activation quantizer over FP32 input: scale = largest / 127, code = round-half-even(
        // x * ( 127 / largest ) ), four adjacent lanes to a block.
        constexpr int kElementsPerThread = 8;
        constexpr int kThreadsPerBlock32 = kBlock / kElementsPerThread;

        __global__ void quantize_fp32_to_int8_per_block_kernel(
            int8_t* __restrict__ codes, float* __restrict__ scales, const float* __restrict__ input, int64_t thread_count )
        {
            const int64_t index = static_cast<int64_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
            const bool valid = index < thread_count;

            float values[ kElementsPerThread ];
            float largest = 0.0f;

            if ( valid )
            {
                const float4 low = reinterpret_cast<const float4*>( input )[ 2 * index ];
                const float4 high = reinterpret_cast<const float4*>( input )[ 2 * index + 1 ];

                values[ 0 ] = low.x;
                values[ 1 ] = low.y;
                values[ 2 ] = low.z;
                values[ 3 ] = low.w;
                values[ 4 ] = high.x;
                values[ 5 ] = high.y;
                values[ 6 ] = high.z;
                values[ 7 ] = high.w;

#pragma unroll
                for ( int element = 0; element < kElementsPerThread; ++element )
                    largest = fmaxf( largest, fabsf( values[ element ] ) );
            }

#pragma unroll
            for ( int offset = 1; offset < kThreadsPerBlock32; offset <<= 1 )
                largest = fmaxf( largest, __shfl_xor_sync( 0xFFFFFFFFu, largest, offset ) );

            if ( !valid )
                return;

            const float inverse = largest > 0.0f ? 127.0f / largest : 0.0f;
            uint32_t words[ 2 ] = { 0u, 0u };

#pragma unroll
            for ( int element = 0; element < kElementsPerThread; ++element )
            {
                const int code = __float2int_rn( values[ element ] * inverse );
                words[ element / 4 ] |= ( static_cast<uint32_t>( code ) & 0xFFu ) << ( 8 * ( element % 4 ) );
            }

            reinterpret_cast<uint2*>( codes )[ index ] = make_uint2( words[ 0 ], words[ 1 ] );

            if ( ( threadIdx.x % kThreadsPerBlock32 ) == 0 )
                scales[ index / kThreadsPerBlock32 ] = largest / 127.0f;
        }

        // A token's top_k combine rows summed in slot order, weighted, to BF16; two columns a thread.
        __global__ void moe_grouped_reduce_kernel(
            const float* __restrict__ rows, const __nv_bfloat16* __restrict__ weights, const int32_t* __restrict__ indices,
            __nv_bfloat16* __restrict__ output, int first_token, int end_token, int hidden, int experts, int top_k )
        {
            const int pairs = hidden / 2;
            const int64_t index = static_cast<int64_t>( blockIdx.x ) * blockDim.x + threadIdx.x;

            if ( index >= static_cast<int64_t>( end_token - first_token ) * pairs )
            {
                return;
            }

            const int local_token = static_cast<int>( index / pairs );
            const int column = 2 * static_cast<int>( index % pairs );
            const int64_t token = first_token + local_token;

            float2 sum = make_float2( 0.0f, 0.0f );
            bool poisoned = false;

            for ( int slot = 0; slot < top_k; ++slot )
            {
                const int32_t expert = indices[ token * top_k + slot ];

                poisoned |= expert < 0 || expert >= experts;

                const float weight = __bfloat162float( weights[ token * top_k + slot ] );
                const float2 value = *reinterpret_cast<const float2*>(
                    rows + ( static_cast<int64_t>( local_token ) * top_k + slot ) * hidden + column );

                sum.x = fmaf( weight, value.x, sum.x );
                sum.y = fmaf( weight, value.y, sum.y );
            }

            if ( poisoned )
            {
                sum = make_float2( NAN, NAN );
            }

            *reinterpret_cast<__nv_bfloat162*>( output + token * hidden + column ) = __floats2bfloat162_rn( sum.x, sum.y );
        }

        constexpr std::size_t aligned16( std::size_t bytes )
        {
            return ( bytes + 15u ) & ~static_cast<std::size_t>( 15u );
        }

        int maximumTiles( int tokens, int top_k, int experts )
        {
            return ( tokens * top_k + kTileM - 1 ) / kTileM + experts;
        }

        struct ScratchLayout
        {
            std::size_t expert_offsets;
            std::size_t permuted_rows;
            std::size_t tile_table;
            std::size_t tile_count;
            std::size_t activations;
            std::size_t combine_rows;
            std::size_t total;
        };

        ScratchLayout scratchLayout( int tokens, int hidden, int intermediate, int experts, int top_k, int64_t gated_capacity )
        {
            const auto rows = static_cast<std::size_t>( tokens ) * top_k;

            ScratchLayout layout{};
            std::size_t offset = 0;

            layout.expert_offsets = offset;
            offset += aligned16( static_cast<std::size_t>( experts + 1 ) * sizeof( int32_t ) );
            layout.permuted_rows = offset;
            offset += aligned16( rows * sizeof( int32_t ) );
            layout.tile_table = offset;
            offset += aligned16( 3u * static_cast<std::size_t>( maximumTiles( tokens, top_k, experts ) ) * sizeof( int32_t ) );
            layout.tile_count = offset;
            offset += 16;
            layout.activations = offset;
            offset += aligned16( std::max(
                Linear::int4GemmScratchBytes( static_cast<std::size_t>( tokens ), static_cast<std::size_t>( hidden ) ),
                Linear::int4GemmScratchBytes( rows, static_cast<std::size_t>( intermediate ) ) ) );

            // The gated buffer holds the combine rows once dead, unless it cannot hold one token's.
            const int64_t token_combine_elements = static_cast<int64_t>( top_k ) * hidden;

            layout.combine_rows = offset;

            if ( gated_capacity < token_combine_elements )
            {
                offset += aligned16( static_cast<std::size_t>( token_combine_elements ) * sizeof( float ) );
            }

            layout.total = offset;

            return layout;
        }

        template <Pass kPass, int kTileK, typename TFunctor>
        void launchGroupedGemmTiles(
            const int8_t* activation_codes, const float* activation_scales,
            const uint8_t* weight_codes, const __half* weight_scales,
            const int32_t* permuted_rows, const int32_t* tile_table, const int32_t* tile_count,
            float* output, int in_features, int out_features, int top_k, int first_flat_row,
            dim3 grid, TFunctor functor, cudaStream_t stream )
        {
            constexpr int kSharedBytes = TileGeometry<kTileK>::kSharedBytes;

            // Past the default 48 KB the kernel must ask, per device; idempotent.
            if constexpr ( kSharedBytes > 48 * 1024 )
            {
                cudaCheck( cudaFuncSetAttribute( moe_grouped_int4_int8_gemm_kernel<kPass, kTileK, TFunctor>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes ) );
            }

            moe_grouped_int4_int8_gemm_kernel<kPass, kTileK, TFunctor><<<grid, kThreads, kSharedBytes, stream>>>(
                activation_codes, activation_scales, weight_codes, weight_scales, permuted_rows, tile_table,
                tile_count, output, in_features, out_features, top_k, first_flat_row, functor );
        }

        template <Pass kPass, typename TFunctor>
        void launchGroupedGemm(
            const int8_t* activation_codes, const float* activation_scales,
            const uint8_t* weight_codes, const __half* weight_scales,
            const int32_t* permuted_rows, const int32_t* tile_table, const int32_t* tile_count,
            float* output, int in_features, int out_features, int top_k, int first_flat_row,
            int grid_tiles, TFunctor functor, cudaStream_t stream )
        {
            const int columns = kPass == Pass::Gated ? 2 * out_features : out_features;
            const dim3 grid( static_cast<unsigned>( grid_tiles ), static_cast<unsigned>( ( columns + kTileN - 1 ) / kTileN ) );

            // Linear's rule: 128 deep wherever it divides the width, measured faster; 64 otherwise.
            if ( in_features % 128 == 0 )
            {
                launchGroupedGemmTiles<kPass, 128>( activation_codes, activation_scales, weight_codes, weight_scales,
                    permuted_rows, tile_table, tile_count, output, in_features, out_features, top_k, first_flat_row,
                    grid, functor, stream );
            }
            else
            {
                launchGroupedGemmTiles<kPass, 64>( activation_codes, activation_scales, weight_codes, weight_scales,
                    permuted_rows, tile_table, tile_count, output, in_features, out_features, top_k, first_flat_row,
                    grid, functor, stream );
            }

            cudaCheck( cudaGetLastError() );
        }
    }

    size_t moe_grouped_scratch_bytes(
        int tokens, int hidden, int intermediate, int experts, int top_k, int64_t gated_capacity )
    {
        return scratchLayout( tokens, hidden, intermediate, experts, top_k, gated_capacity ).total;
    }

    template<typename TFunctor>
    void launch_moe_grouped_prefill_int4(
        const __nv_bfloat16* input, const uint8_t* gate_up, const __half* gate_up_scales,
        const uint8_t* down, const __half* down_scales, const __nv_bfloat16* weights, const int32_t* indices,
        float* gated, int64_t gated_capacity, __nv_bfloat16* output, void* scratch,
        int tokens, int hidden, int intermediate, int experts, int top_k,
        TFunctor functor, cudaStream_t stream )
    {
        if ( experts > kGroupedMaximumExperts || hidden % 64 != 0 || intermediate % 64 != 0 )
        {
            throw std::invalid_argument( std::format(
                "launch_moe_grouped_prefill_int4: {} experts (at most {}), hidden {} and intermediate {} (multiples of 64)",
                experts, kGroupedMaximumExperts, hidden, intermediate ) );
        }

        if ( tokens == 0 )
        {
            return;
        }

        const ScratchLayout layout = scratchLayout( tokens, hidden, intermediate, experts, top_k, gated_capacity );
        auto* base = static_cast<char*>( scratch );
        auto* expert_offsets = reinterpret_cast<int32_t*>( base + layout.expert_offsets );
        auto* permuted_rows = reinterpret_cast<int32_t*>( base + layout.permuted_rows );
        auto* tile_table = reinterpret_cast<int32_t*>( base + layout.tile_table );
        auto* tile_count = reinterpret_cast<int32_t*>( base + layout.tile_count );
        auto* activation_codes = reinterpret_cast<int8_t*>( base + layout.activations );

        const int rows = tokens * top_k;

        moe_route_kernel<<<1, kRoutingThreads, 0, stream>>>( indices, rows, experts, expert_offsets, permuted_rows );
        cudaCheck( cudaGetLastError() );

        // Gated pass: the input's INT8 codes, then every segment against its gate and up rows.
        auto* input_scales = reinterpret_cast<float*>(
            activation_codes + Linear::int4GemmScaleOffset( static_cast<std::size_t>( tokens ), static_cast<std::size_t>( hidden ) ) );

        Linear::cuda_quantize_bf16_to_int8_per_block( activation_codes, input_scales, input, tokens, hidden, stream );

        moe_tile_kernel<<<1, kGroupedMaximumExperts, 0, stream>>>(
            expert_offsets, permuted_rows, experts, top_k, 0, tokens, tile_table, tile_count );
        cudaCheck( cudaGetLastError() );

        launchGroupedGemm<Pass::Gated>( activation_codes, input_scales, gate_up, gate_up_scales, permuted_rows,
            tile_table, tile_count, gated, hidden, intermediate, top_k, 0,
            maximumTiles( tokens, top_k, experts ), functor, stream );

        // The gated values' INT8 codes over the input's, which the gated pass no longer reads.
        auto* gated_scales = reinterpret_cast<float*>(
            activation_codes + Linear::int4GemmScaleOffset( static_cast<std::size_t>( rows ), static_cast<std::size_t>( intermediate ) ) );
        const int64_t quantize_threads = static_cast<int64_t>( rows ) * intermediate / kElementsPerThread;

        quantize_fp32_to_int8_per_block_kernel<<<static_cast<unsigned>( ( quantize_threads + 255 ) / 256 ), 256, 0, stream>>>(
            activation_codes, gated_scales, gated, quantize_threads );
        cudaCheck( cudaGetLastError() );

        // Combine pass, as many tokens at a time as the dead gated buffer holds.
        const int64_t token_combine_elements = static_cast<int64_t>( top_k ) * hidden;
        const bool rows_in_gated = gated_capacity >= token_combine_elements;
        float* combine_rows = rows_in_gated ? gated : reinterpret_cast<float*>( base + layout.combine_rows );
        const int tokens_per_pass = rows_in_gated
            ? static_cast<int>( std::min<int64_t>( tokens, gated_capacity / token_combine_elements ) )
            : 1;

        for ( int first_token = 0; first_token < tokens; first_token += tokens_per_pass )
        {
            const int end_token = std::min( tokens, first_token + tokens_per_pass );

            moe_tile_kernel<<<1, kGroupedMaximumExperts, 0, stream>>>(
                expert_offsets, permuted_rows, experts, top_k, first_token, end_token, tile_table, tile_count );
            cudaCheck( cudaGetLastError() );

            launchGroupedGemm<Pass::Combine>( activation_codes, gated_scales, down, down_scales, permuted_rows,
                tile_table, tile_count, combine_rows, intermediate, hidden, top_k, first_token * top_k,
                maximumTiles( end_token - first_token, top_k, experts ), functor, stream );

            const int64_t reduce_threads = static_cast<int64_t>( end_token - first_token ) * ( hidden / 2 );

            moe_grouped_reduce_kernel<<<static_cast<unsigned>( ( reduce_threads + 255 ) / 256 ), 256, 0, stream>>>(
                combine_rows, weights, indices, output, first_token, end_token, hidden, experts, top_k );
            cudaCheck( cudaGetLastError() );
        }
    }

    template void launch_moe_grouped_prefill_int4<Mila::Dnn::Activations::GeluTanh>(
        const __nv_bfloat16*, const uint8_t*, const __half*, const uint8_t*, const __half*, const __nv_bfloat16*,
        const int32_t*, float*, int64_t, __nv_bfloat16*, void*, int, int, int, int, int,
        Mila::Dnn::Activations::GeluTanh, cudaStream_t );
    template void launch_moe_grouped_prefill_int4<Mila::Dnn::Activations::Silu>(
        const __nv_bfloat16*, const uint8_t*, const __half*, const uint8_t*, const __half*, const __nv_bfloat16*,
        const int32_t*, float*, int64_t, __nv_bfloat16*, void*, int, int, int, int, int,
        Mila::Dnn::Activations::Silu, cudaStream_t );
}
