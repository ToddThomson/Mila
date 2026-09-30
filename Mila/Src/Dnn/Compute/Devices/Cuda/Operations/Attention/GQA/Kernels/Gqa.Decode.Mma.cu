/**
 * @file Gqa.Decode.Mma.cu
 * @brief Fused single-token decode attention over the compact KV cache, on the tensor cores.
 *
 * A block packs a KV head's whole query-head group into the rows of one m16n8k16 tile and walks one split of the
 * live band; a fixup launch merges the splits. Supersedes the CUDA-core kernel of Gqa.Decode.Bf16.cu (retired).
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <mutex>
#include <numeric>
#include "CudaUtils.h"
#include "CudaGqa.cuh"

namespace Mila::Dnn::Compute::Cuda::Gqa
{
    namespace
    {
        // Split-K partial rows are [head_size + 2] floats: unnormalized O, then m, l.
        constexpr int kMaxDecodeSplits = 128;

        // The query-head group fills the rows of one 16-row MMA tile.
        constexpr int kMaxDecodeGroupSize = 16;

        /// The live band at `position`: [window_start, position + 1).
        __device__ __forceinline__ int decodeBandStart( int actual_len, int window )
        {
            return ( window > 0 ) ? max( 0, actual_len - window ) : 0;
        }

        constexpr int kMmaRows = 16;
        constexpr int kMmaK = 16;
        constexpr int kMmaN = 8;
        constexpr int kSliceDims = 128;                        // head dims per warp
        constexpr int kSliceNTiles = kSliceDims / kMmaN;       // O n-tiles per warp
        constexpr int kSliceKSteps = kSliceDims / kMmaK;       // QK k-steps per warp
        constexpr int kGroupKeys = 16;                         // keys per key group per tile: one PV k-step
        constexpr int kCopyElements = 8;                       // bf16 per 16-byte cp.async
        constexpr int kSkew = 8;                               // row padding: ldmatrix rows 4 banks apart
        constexpr int kMinKeysPerSplit = 64;

        /**
         * A block's warps split the head dimension into 128-wide slices (kSplit) and the key tile into groups of
         * 16 keys (kKeyGroups). Each key group carries its own softmax state; the groups merge in shared memory at
         * the end of the block.
         */
        template<int kHeadSize, bool kFp8, int kKeyGroupCount>
        struct MmaDecodeGeometry
        {
            static_assert( kHeadSize % kSliceDims == 0 && kKeyGroupCount >= 1 );

            static constexpr int kSplit = kHeadSize / kSliceDims;
            static constexpr int kKeyGroups = kKeyGroupCount;
            static constexpr int kWarps = kSplit * kKeyGroups;
            static constexpr int kThreads = kWarps * 32;
            static constexpr int kTileKeys = kKeyGroups * kGroupKeys;
            static constexpr int kPad = kHeadSize + kSkew;
            static constexpr int kExchangeStride = kGroupKeys + 2;  // even (float2 stores), rows off-bank
            static constexpr int kMergeStride = kHeadSize + 8;

            static constexpr std::size_t kStageElements = static_cast<std::size_t>( kTileKeys ) * kPad;
            static constexpr int kBf16Stages = kFp8 ? 1 : 2;

            static constexpr std::size_t kExchangeBytes = kSplit > 1
                ? static_cast<std::size_t>( kKeyGroups ) * kSplit * kMmaRows * kExchangeStride * sizeof( float )
                : 0;

            static constexpr std::size_t kBf16Bytes = 2 * kBf16Stages * kStageElements * sizeof( __nv_bfloat16 );
            static constexpr std::size_t kCodeStageBytes = static_cast<std::size_t>( kTileKeys ) * kHeadSize;
            static constexpr std::size_t kCodeBytes = kFp8 ? 4 * kCodeStageBytes : 0;
            static constexpr std::size_t kScaleBytes = kFp8 ? 4 * static_cast<std::size_t>( kTileKeys ) * sizeof( float ) : 0;

            static constexpr std::size_t kSharedBytes = kExchangeBytes + kBf16Bytes + kCodeBytes + kScaleBytes;

            // The key groups' (O, m, l) merge reuses the stages once the last tile is done.
            static constexpr std::size_t kMergeBytes = kKeyGroups > 1
                ? static_cast<std::size_t>( kKeyGroups ) * kMmaRows * ( kMergeStride + 2 ) * sizeof( float )
                : 0;

            static_assert( kMergeBytes <= kSharedBytes );
            static_assert( kSharedBytes <= 99 * 1024, "a block holds at most 99 KB of shared memory" );
        };

        struct DecodeSplits
        {
            int count;
            int chunk;
        };

        /**
         * @brief The splits of a band: at most `target_splits`, none shorter than kMinKeysPerSplit keys, each a
         *        whole number of key tiles, and none empty.
         *
         * One function for both sides (DecodeGraph.md section 4.2): the host sizes the grid from the largest band
         * the op can hold, and the attention and fixup kernels choose the live count from the band at the
         * device-resident position, so the three cannot disagree.
         */
        __host__ __device__ inline DecodeSplits decodeSplits( int band_len, int target_splits, int tile_keys )
        {
            const int minimum = tile_keys > kMinKeysPerSplit ? tile_keys : kMinKeysPerSplit;
            const int by_band = ( band_len + minimum - 1 ) / minimum;
            const int bounded = by_band < target_splits ? by_band : target_splits;
            const int desired = bounded > 1 ? bounded : 1;
            const int per_split = ( band_len + desired - 1 ) / desired;
            const int chunk = ( ( per_split + tile_keys - 1 ) / tile_keys ) * tile_keys;

            return { ( band_len + chunk - 1 ) / chunk, chunk };
        }

        /// The most splits a band up to max_band can use: the grid's split axis.
        inline int decodeGridSplits( int max_band, int target_splits, int tile_keys )
        {
            const int minimum = tile_keys > kMinKeysPerSplit ? tile_keys : kMinKeysPerSplit;
            const int by_band = ( max_band + minimum - 1 ) / minimum;

            return std::max( 1, std::min( by_band, target_splits ) );
        }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        __device__ __forceinline__ void mma_m16n8k16_bf16(
            float& c0, float& c1, float& c2, float& c3,
            uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
            uint32_t b0, uint32_t b1 )
        {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"( c0 ), "+f"( c1 ), "+f"( c2 ), "+f"( c3 )
                : "r"( a0 ), "r"( a1 ), "r"( a2 ), "r"( a3 ), "r"( b0 ), "r"( b1 ) );
        }

        // Explicit cvta + the .shared qualifier on ldmatrix: see Gqa.Flash.Wmma.cu for why both are load-bearing.
        __device__ __forceinline__ uint32_t shared_address( const void* pointer )
        {
            uint32_t address;
            asm volatile(
                "{ .reg .u64 address64; cvta.to.shared.u64 address64, %1; cvt.u32.u64 %0, address64; }\n"
                : "=r"( address ) : "l"( pointer ) );

            return address;
        }

        __device__ __forceinline__ void ldmatrix_x4(
            uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t address )
        {
            asm volatile( "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"( r0 ), "=r"( r1 ), "=r"( r2 ), "=r"( r3 ) : "r"( address ) );
        }

        __device__ __forceinline__ void ldmatrix_x4_trans(
            uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t address )
        {
            asm volatile( "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"( r0 ), "=r"( r1 ), "=r"( r2 ), "=r"( r3 ) : "r"( address ) );
        }

        // Synchronizes only the `threads` threads that name `barrier` (ids above 0; 0 is __syncthreads), with the
        // shared-memory ordering __syncthreads gives.
        __device__ __forceinline__ void named_barrier( int barrier, int threads )
        {
            asm volatile( "bar.sync %0, %1;\n" :: "r"( barrier ), "r"( threads ) : "memory" );
        }

        __device__ __forceinline__ uint32_t pack_bf16x2( float low, float high )
        {
            const __nv_bfloat162 packed = __floats2bfloat162_rn( low, high );

            return *reinterpret_cast<const uint32_t*>( &packed );
        }
#endif

        /**
         * @brief Tensor-core decode attention: one (kv_head, split, batch) block.
         *
         * The block's query-head group fills the rows of one m16n8k16 tile (rows past the group are zero), so a
         * KV head is read once for its whole group and QK and PV run on the tensor cores, as the packed prefill
         * does (Gqa.Flash.Packed.cu, whose fragment layout this follows). The split walks absolute positions
         * [chunk_begin, chunk_end) with cache row = position % capacity; a key outside the split is staged as
         * zeros and scored -inf, so no unwritten row is read.
         *
         * With one split the block normalizes and writes Y; otherwise it writes the unnormalized (O, m, l) partial
         * the fixup below merges.
         *
         * With kFp8 the cache holds E4M3 codes and one scale per row (PerTokenKvFp8): the codes widen unscaled into
         * the BF16 stage, each key's K scale multiplies its score, and each key's V scale its probability as P is
         * packed, while l sums the unscaled probabilities (Quantization.md, Part III).
         */
        template<int kHeadSize, bool kFp8, int kKeyGroupCount>
        __global__ void __launch_bounds__( MmaDecodeGeometry<kHeadSize, kFp8, kKeyGroupCount>::kThreads )
            gqa_decode_attention_mma_kernel(
                const __nv_bfloat16* __restrict__ q,
                const void* __restrict__ k_cache_raw,
                const void* __restrict__ v_cache_raw,
                const float* __restrict__ k_scales,
                const float* __restrict__ v_scales,
                __nv_bfloat16* __restrict__ y,
                float* __restrict__ split_partials,
                int num_kv_heads,
                int group_size,
                int capacity,
                const int* __restrict__ position,
                int window,
                int target_splits,
                float scale )
        {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            using Geometry = MmaDecodeGeometry<kHeadSize, kFp8, kKeyGroupCount>;

            const int tid = threadIdx.x;
            const int warp = tid >> 5;
            const int lane = tid & 31;
            const int g = lane >> 2;
            const int tg = lane & 3;

            const int key_group = warp / Geometry::kSplit;
            const int slice = warp % Geometry::kSplit;
            const int slice0 = slice * kSliceDims;

            const int kv = blockIdx.x;
            const int split = blockIdx.y;
            const int batch = blockIdx.z;
            const int num_heads = num_kv_heads * group_size;

            const int actual_len = *position + 1;
            const int window_start = decodeBandStart( actual_len, window );
            const DecodeSplits splits = decodeSplits( actual_len - window_start, target_splits, Geometry::kTileKeys );

            if ( split >= splits.count )
                return;

            const int chunk_begin = window_start + split * splits.chunk;
            const int chunk_end = min( chunk_begin + splits.chunk, actual_len );
            const int tile_count = ( chunk_end - chunk_begin + Geometry::kTileKeys - 1 ) / Geometry::kTileKeys;

            const std::size_t kv_row_base = ( static_cast<std::size_t>( batch ) * num_kv_heads + kv ) * capacity;

            extern __shared__ __align__( 16 ) char smem_raw[];
            float* s_exchange = reinterpret_cast<float*>( smem_raw );
            __nv_bfloat16* s_k = reinterpret_cast<__nv_bfloat16*>( smem_raw + Geometry::kExchangeBytes );
            __nv_bfloat16* s_v = s_k + Geometry::kBf16Stages * Geometry::kStageElements;

            uint8_t* s_k_codes = reinterpret_cast<uint8_t*>( smem_raw + Geometry::kExchangeBytes + Geometry::kBf16Bytes );
            uint8_t* s_v_codes = s_k_codes + 2 * Geometry::kCodeStageBytes;
            float* s_k_scale = reinterpret_cast<float*>( s_v_codes + 2 * Geometry::kCodeStageBytes );
            float* s_v_scale = s_k_scale + 2 * Geometry::kTileKeys;

            const auto bf16Stage = [&]( __nv_bfloat16* base, int tile ) -> __nv_bfloat16*
            {
                return base + ( kFp8 ? 0 : ( tile & 1 ) ) * Geometry::kStageElements;
            };

            // Keys past the split are zeros, so the -inf mask meets only finite values.
            const auto loadTile = [&]( int tile )
            {
                const int tile_start = chunk_begin + tile * Geometry::kTileKeys;
                const int stage = tile & 1;

                if constexpr ( kFp8 )
                {
                    constexpr int kChunksPerRow = kHeadSize / 16;
                    uint8_t* k_codes = s_k_codes + stage * Geometry::kCodeStageBytes;
                    uint8_t* v_codes = s_v_codes + stage * Geometry::kCodeStageBytes;
                    const uint8_t* K = static_cast<const uint8_t*>( k_cache_raw );
                    const uint8_t* V = static_cast<const uint8_t*>( v_cache_raw );

                    for ( int chunk = tid; chunk < Geometry::kTileKeys * kChunksPerRow; chunk += Geometry::kThreads )
                    {
                        const int key = chunk / kChunksPerRow;
                        const int column = ( chunk % kChunksPerRow ) * 16;
                        const int key_position = tile_start + key;
                        const int destination = key * kHeadSize + column;

                        if ( key_position < chunk_end )
                        {
                            const int row = key_position < capacity ? key_position : key_position % capacity;
                            const std::size_t source = ( kv_row_base + row ) * kHeadSize + column;

                            __pipeline_memcpy_async( k_codes + destination, K + source, 16 );
                            __pipeline_memcpy_async( v_codes + destination, V + source, 16 );
                        }
                        else
                        {
                            *reinterpret_cast<int4*>( k_codes + destination ) = int4{ 0, 0, 0, 0 };
                            *reinterpret_cast<int4*>( v_codes + destination ) = int4{ 0, 0, 0, 0 };
                        }
                    }

                    for ( int key = tid; key < Geometry::kTileKeys; key += Geometry::kThreads )
                    {
                        const int key_position = tile_start + key;
                        float* k_scale = s_k_scale + stage * Geometry::kTileKeys + key;
                        float* v_scale = s_v_scale + stage * Geometry::kTileKeys + key;

                        if ( key_position < chunk_end )
                        {
                            const int row = key_position < capacity ? key_position : key_position % capacity;

                            __pipeline_memcpy_async( k_scale, k_scales + kv_row_base + row, 4 );
                            __pipeline_memcpy_async( v_scale, v_scales + kv_row_base + row, 4 );
                        }
                        else
                        {
                            *k_scale = 0.0f;
                            *v_scale = 0.0f;
                        }
                    }
                }
                else
                {
                    constexpr int kChunksPerRow = kHeadSize / kCopyElements;
                    __nv_bfloat16* k_stage = s_k + stage * Geometry::kStageElements;
                    __nv_bfloat16* v_stage = s_v + stage * Geometry::kStageElements;
                    const __nv_bfloat16* K = static_cast<const __nv_bfloat16*>( k_cache_raw );
                    const __nv_bfloat16* V = static_cast<const __nv_bfloat16*>( v_cache_raw );

                    for ( int chunk = tid; chunk < Geometry::kTileKeys * kChunksPerRow; chunk += Geometry::kThreads )
                    {
                        const int key = chunk / kChunksPerRow;
                        const int column = ( chunk % kChunksPerRow ) * kCopyElements;
                        const int key_position = tile_start + key;
                        const int destination = key * Geometry::kPad + column;

                        if ( key_position < chunk_end )
                        {
                            const int row = key_position < capacity ? key_position : key_position % capacity;
                            const std::size_t source = ( kv_row_base + row ) * kHeadSize + column;

                            __pipeline_memcpy_async( k_stage + destination, K + source, 16 );
                            __pipeline_memcpy_async( v_stage + destination, V + source, 16 );
                        }
                        else
                        {
                            *reinterpret_cast<int4*>( k_stage + destination ) = int4{ 0, 0, 0, 0 };
                            *reinterpret_cast<int4*>( v_stage + destination ) = int4{ 0, 0, 0, 0 };
                        }
                    }
                }

                __pipeline_commit();
            };

            // Q for this warp's slice, straight into A fragments; rows past the group are zero.
            const bool active_low = g < group_size;
            const bool active_high = g + 8 < group_size;
            const std::size_t head0 = static_cast<std::size_t>( batch ) * num_heads + kv * group_size;
            const __nv_bfloat16* q_low = q + ( head0 + g ) * kHeadSize;
            const __nv_bfloat16* q_high = q + ( head0 + g + 8 ) * kHeadSize;

            uint32_t q_fragment[ kSliceKSteps ][ 4 ];

#pragma unroll
            for ( int step = 0; step < kSliceKSteps; ++step )
            {
                const int column = slice0 + step * kMmaK + 2 * tg;

                q_fragment[ step ][ 0 ] = active_low ? *reinterpret_cast<const uint32_t*>( q_low + column ) : 0u;
                q_fragment[ step ][ 1 ] = active_high ? *reinterpret_cast<const uint32_t*>( q_high + column ) : 0u;
                q_fragment[ step ][ 2 ] = active_low ? *reinterpret_cast<const uint32_t*>( q_low + column + 8 ) : 0u;
                q_fragment[ step ][ 3 ] = active_high ? *reinterpret_cast<const uint32_t*>( q_high + column + 8 ) : 0u;
            }

            float o[ kSliceNTiles ][ 4 ];

#pragma unroll
            for ( int nt = 0; nt < kSliceNTiles; ++nt )
                o[ nt ][ 0 ] = o[ nt ][ 1 ] = o[ nt ][ 2 ] = o[ nt ][ 3 ] = 0.0f;

            float m_low = -CUDART_INF_F, m_high = -CUDART_INF_F;
            float l_low = 0.0f, l_high = 0.0f;

            loadTile( 0 );

            for ( int tile = 0; tile < tile_count; ++tile )
            {
                const int group_start = chunk_begin + tile * Geometry::kTileKeys + key_group * kGroupKeys;
                const __nv_bfloat16* k_tile = bf16Stage( s_k, tile ) + key_group * kGroupKeys * Geometry::kPad;
                const __nv_bfloat16* v_tile = bf16Stage( s_v, tile ) + key_group * kGroupKeys * Geometry::kPad;
                const float* k_scale_tile = s_k_scale + ( tile & 1 ) * Geometry::kTileKeys + key_group * kGroupKeys;
                const float* v_scale_tile = s_v_scale + ( tile & 1 ) * Geometry::kTileKeys + key_group * kGroupKeys;

                __pipeline_wait_prior( 0 );

                // This tile visible to every warp; and, every thread having finished the previous tile, the stage
                // the next prefetch writes and the exchange buffer are free. For FP8 that stage's codes were
                // widened in the previous iteration, so the prefetch goes before this tile's widening and the
                // load stays in flight through it.
                __syncthreads();

                if ( tile + 1 < tile_count )
                    loadTile( tile + 1 );

                if constexpr ( kFp8 )
                {
                    // Sixteen codes a thread a step: one 16-byte read, two 16-byte writes.
                    constexpr int kChunksPerRow = kHeadSize / 16;
                    const uint8_t* k_codes = s_k_codes + ( tile & 1 ) * Geometry::kCodeStageBytes;
                    const uint8_t* v_codes = s_v_codes + ( tile & 1 ) * Geometry::kCodeStageBytes;

                    const auto widen = []( const uint4 codes, __nv_bfloat16* destination )
                    {
                        const uint32_t words[ 4 ] = { codes.x, codes.y, codes.z, codes.w };
                        uint32_t widened[ 8 ];

#pragma unroll
                        for ( int word = 0; word < 4; ++word )
                        {
#pragma unroll
                            for ( int half = 0; half < 2; ++half )
                            {
                                const __nv_fp8x2_storage_t pair = static_cast<__nv_fp8x2_storage_t>( words[ word ] >> ( 16 * half ) );
                                const __nv_bfloat162 value = __float22bfloat162_rn(
                                    __half22float2( __half2( __nv_cvt_fp8x2_to_halfraw2( pair, __NV_E4M3 ) ) ) );
                                widened[ 2 * word + half ] = *reinterpret_cast<const uint32_t*>( &value );
                            }
                        }

                        reinterpret_cast<uint4*>( destination )[ 0 ] = make_uint4( widened[ 0 ], widened[ 1 ], widened[ 2 ], widened[ 3 ] );
                        reinterpret_cast<uint4*>( destination )[ 1 ] = make_uint4( widened[ 4 ], widened[ 5 ], widened[ 6 ], widened[ 7 ] );
                    };

                    for ( int chunk = tid; chunk < Geometry::kTileKeys * kChunksPerRow; chunk += Geometry::kThreads )
                    {
                        const int key = chunk / kChunksPerRow;
                        const int column = ( chunk % kChunksPerRow ) * 16;

                        widen( *reinterpret_cast<const uint4*>( k_codes + key * kHeadSize + column ), s_k + key * Geometry::kPad + column );
                        widen( *reinterpret_cast<const uint4*>( v_codes + key * kHeadSize + column ), s_v + key * Geometry::kPad + column );
                    }

                    // The widened tile visible to every warp before the MMAs read it.
                    __syncthreads();
                }

                // --- QK over this warp's slice and key group ---
                float s[ 2 ][ 4 ] = {};

#pragma unroll
                for ( int step = 0; step < kSliceKSteps; ++step )
                {
                    uint32_t b00, b01, b10, b11;
                    const int key = ( lane >> 4 ) * kMmaN + ( lane & 7 );
                    const int column = slice0 + step * kMmaK + ( ( lane >> 3 ) & 1 ) * kMmaN;
                    ldmatrix_x4( b00, b01, b10, b11, shared_address( &k_tile[ key * Geometry::kPad + column ] ) );

                    mma_m16n8k16_bf16( s[ 0 ][ 0 ], s[ 0 ][ 1 ], s[ 0 ][ 2 ], s[ 0 ][ 3 ],
                        q_fragment[ step ][ 0 ], q_fragment[ step ][ 1 ], q_fragment[ step ][ 2 ], q_fragment[ step ][ 3 ],
                        b00, b01 );
                    mma_m16n8k16_bf16( s[ 1 ][ 0 ], s[ 1 ][ 1 ], s[ 1 ][ 2 ], s[ 1 ][ 3 ],
                        q_fragment[ step ][ 0 ], q_fragment[ step ][ 1 ], q_fragment[ step ][ 2 ], q_fragment[ step ][ 3 ],
                        b10, b11 );
                }

                // --- the key group's partial scores meet: every warp sums all slices in slice order ---
                if constexpr ( Geometry::kSplit > 1 )
                {
                    float* mine = s_exchange
                        + ( key_group * Geometry::kSplit + slice ) * kMmaRows * Geometry::kExchangeStride;

#pragma unroll
                    for ( int nt = 0; nt < 2; ++nt )
                    {
                        const int column = nt * kMmaN + 2 * tg;
                        *reinterpret_cast<float2*>( &mine[ g * Geometry::kExchangeStride + column ] ) = make_float2( s[ nt ][ 0 ], s[ nt ][ 1 ] );
                        *reinterpret_cast<float2*>( &mine[ ( g + 8 ) * Geometry::kExchangeStride + column ] ) = make_float2( s[ nt ][ 2 ], s[ nt ][ 3 ] );
                    }

                    named_barrier( 1 + key_group, Geometry::kSplit * 32 );

#pragma unroll
                    for ( int nt = 0; nt < 2; ++nt )
                        s[ nt ][ 0 ] = s[ nt ][ 1 ] = s[ nt ][ 2 ] = s[ nt ][ 3 ] = 0.0f;

#pragma unroll
                    for ( int part = 0; part < Geometry::kSplit; ++part )
                    {
                        const float* partial = s_exchange
                            + ( key_group * Geometry::kSplit + part ) * kMmaRows * Geometry::kExchangeStride;

#pragma unroll
                        for ( int nt = 0; nt < 2; ++nt )
                        {
                            const int column = nt * kMmaN + 2 * tg;
                            const float2 low = *reinterpret_cast<const float2*>( &partial[ g * Geometry::kExchangeStride + column ] );
                            const float2 high = *reinterpret_cast<const float2*>( &partial[ ( g + 8 ) * Geometry::kExchangeStride + column ] );

                            s[ nt ][ 0 ] += low.x;
                            s[ nt ][ 1 ] += low.y;
                            s[ nt ][ 2 ] += high.x;
                            s[ nt ][ 3 ] += high.y;
                        }
                    }
                }

                // --- online softmax for rows g and g + 8; a lane holds keys nt*8 + 2tg + {0,1}, shared by both ---
                float max_low = -CUDART_INF_F, max_high = -CUDART_INF_F;

#pragma unroll
                for ( int nt = 0; nt < 2; ++nt )
                {
#pragma unroll
                    for ( int j = 0; j < 2; ++j )
                    {
                        const int key = nt * kMmaN + 2 * tg + j;
                        const bool live = group_start + key < chunk_end;
                        const float key_scale = kFp8 ? scale * k_scale_tile[ key ] : scale;

                        const float low = live ? s[ nt ][ j ] * key_scale : -CUDART_INF_F;
                        const float high = live ? s[ nt ][ 2 + j ] * key_scale : -CUDART_INF_F;

                        s[ nt ][ j ] = low;
                        s[ nt ][ 2 + j ] = high;
                        max_low = fmaxf( max_low, low );
                        max_high = fmaxf( max_high, high );
                    }
                }

#pragma unroll
                for ( int offset = 2; offset >= 1; offset >>= 1 )
                {
                    max_low = fmaxf( max_low, __shfl_xor_sync( 0xffffffffu, max_low, offset ) );
                    max_high = fmaxf( max_high, __shfl_xor_sync( 0xffffffffu, max_high, offset ) );
                }

                // A key group whose keys all lie past the split keeps its state; the exps below then see only -inf.
                const bool scored = group_start < chunk_end;
                const float new_m_low = scored ? fmaxf( m_low, max_low ) : m_low;
                const float new_m_high = scored ? fmaxf( m_high, max_high ) : m_high;

                float sum_low = 0.0f, sum_high = 0.0f;

#pragma unroll
                for ( int nt = 0; nt < 2; ++nt )
                {
#pragma unroll
                    for ( int j = 0; j < 2; ++j )
                    {
                        s[ nt ][ j ] = s[ nt ][ j ] == -CUDART_INF_F ? 0.0f : __expf( s[ nt ][ j ] - new_m_low );
                        s[ nt ][ 2 + j ] = s[ nt ][ 2 + j ] == -CUDART_INF_F ? 0.0f : __expf( s[ nt ][ 2 + j ] - new_m_high );
                        sum_low += s[ nt ][ j ];
                        sum_high += s[ nt ][ 2 + j ];
                    }
                }

#pragma unroll
                for ( int offset = 2; offset >= 1; offset >>= 1 )
                {
                    sum_low += __shfl_xor_sync( 0xffffffffu, sum_low, offset );
                    sum_high += __shfl_xor_sync( 0xffffffffu, sum_high, offset );
                }

                if ( scored )
                {
                    const float alpha_low = m_low == -CUDART_INF_F ? 0.0f : __expf( m_low - new_m_low );
                    const float alpha_high = m_high == -CUDART_INF_F ? 0.0f : __expf( m_high - new_m_high );

                    l_low = l_low * alpha_low + sum_low;
                    l_high = l_high * alpha_high + sum_high;
                    m_low = new_m_low;
                    m_high = new_m_high;

#pragma unroll
                    for ( int nt = 0; nt < kSliceNTiles; ++nt )
                    {
                        o[ nt ][ 0 ] *= alpha_low;
                        o[ nt ][ 1 ] *= alpha_low;
                        o[ nt ][ 2 ] *= alpha_high;
                        o[ nt ][ 3 ] *= alpha_high;
                    }
                }

                // --- PV over this warp's 128 output columns, P straight from the score registers ---
                float v0 = 1.0f, v1 = 1.0f, v2 = 1.0f, v3 = 1.0f;

                if constexpr ( kFp8 )
                {
                    v0 = v_scale_tile[ 2 * tg ];
                    v1 = v_scale_tile[ 2 * tg + 1 ];
                    v2 = v_scale_tile[ kMmaN + 2 * tg ];
                    v3 = v_scale_tile[ kMmaN + 2 * tg + 1 ];
                }

                const uint32_t p0 = pack_bf16x2( s[ 0 ][ 0 ] * v0, s[ 0 ][ 1 ] * v1 );
                const uint32_t p1 = pack_bf16x2( s[ 0 ][ 2 ] * v0, s[ 0 ][ 3 ] * v1 );
                const uint32_t p2 = pack_bf16x2( s[ 1 ][ 0 ] * v2, s[ 1 ][ 1 ] * v3 );
                const uint32_t p3 = pack_bf16x2( s[ 1 ][ 2 ] * v2, s[ 1 ][ 3 ] * v3 );

#pragma unroll
                for ( int pair = 0; pair < kSliceNTiles / 2; ++pair )
                {
                    uint32_t b00, b01, b10, b11;
                    ldmatrix_x4_trans( b00, b01, b10, b11, shared_address(
                        &v_tile[ ( lane & 15 ) * Geometry::kPad + slice0 + pair * 16 + ( lane >> 4 ) * kMmaN ] ) );

                    mma_m16n8k16_bf16( o[ 2 * pair ][ 0 ], o[ 2 * pair ][ 1 ], o[ 2 * pair ][ 2 ], o[ 2 * pair ][ 3 ],
                        p0, p1, p2, p3, b00, b01 );
                    mma_m16n8k16_bf16( o[ 2 * pair + 1 ][ 0 ], o[ 2 * pair + 1 ][ 1 ], o[ 2 * pair + 1 ][ 2 ], o[ 2 * pair + 1 ][ 3 ],
                        p0, p1, p2, p3, b10, b11 );
                }
            }

            const auto partialRow = [&]( int row ) -> float*
            {
                return split_partials +
                    ( ( ( static_cast<std::size_t>( batch ) * num_kv_heads + kv ) * splits.count + split )
                        * group_size + row ) * ( kHeadSize + 2 );
            };

            const auto headRow = [&]( int row ) -> __nv_bfloat16*
            {
                return y + ( head0 + row ) * kHeadSize;
            };

            if constexpr ( Geometry::kKeyGroups == 1 )
            {
                if ( splits.count == 1 )
                {
                    const float inverse_low = active_low ? 1.0f / l_low : 0.0f;
                    const float inverse_high = active_high ? 1.0f / l_high : 0.0f;

#pragma unroll
                    for ( int nt = 0; nt < kSliceNTiles; ++nt )
                    {
                        const int column = slice0 + nt * kMmaN + 2 * tg;

                        if ( active_low )
                            *reinterpret_cast<__nv_bfloat162*>( headRow( g ) + column ) =
                                __floats2bfloat162_rn( o[ nt ][ 0 ] * inverse_low, o[ nt ][ 1 ] * inverse_low );

                        if ( active_high )
                            *reinterpret_cast<__nv_bfloat162*>( headRow( g + 8 ) + column ) =
                                __floats2bfloat162_rn( o[ nt ][ 2 ] * inverse_high, o[ nt ][ 3 ] * inverse_high );
                    }
                }
                else
                {
#pragma unroll
                    for ( int nt = 0; nt < kSliceNTiles; ++nt )
                    {
                        const int column = slice0 + nt * kMmaN + 2 * tg;

                        if ( active_low )
                            *reinterpret_cast<float2*>( partialRow( g ) + column ) = make_float2( o[ nt ][ 0 ], o[ nt ][ 1 ] );

                        if ( active_high )
                            *reinterpret_cast<float2*>( partialRow( g + 8 ) + column ) = make_float2( o[ nt ][ 2 ], o[ nt ][ 3 ] );
                    }

                    if ( slice == 0 && tg == 0 )
                    {
                        if ( active_low )
                        {
                            partialRow( g )[ kHeadSize ] = m_low;
                            partialRow( g )[ kHeadSize + 1 ] = l_low;
                        }

                        if ( active_high )
                        {
                            partialRow( g + 8 )[ kHeadSize ] = m_high;
                            partialRow( g + 8 )[ kHeadSize + 1 ] = l_high;
                        }
                    }
                }
            }
            else
            {
                // The key groups merge through shared memory: every stage is consumed, no copy is in flight.
                __syncthreads();

                float* s_output = reinterpret_cast<float*>( smem_raw );
                float* s_state = s_output + Geometry::kKeyGroups * kMmaRows * Geometry::kMergeStride;
                float* group_output = s_output + key_group * kMmaRows * Geometry::kMergeStride;

#pragma unroll
                for ( int nt = 0; nt < kSliceNTiles; ++nt )
                {
                    const int column = slice0 + nt * kMmaN + 2 * tg;
                    *reinterpret_cast<float2*>( &group_output[ g * Geometry::kMergeStride + column ] ) = make_float2( o[ nt ][ 0 ], o[ nt ][ 1 ] );
                    *reinterpret_cast<float2*>( &group_output[ ( g + 8 ) * Geometry::kMergeStride + column ] ) = make_float2( o[ nt ][ 2 ], o[ nt ][ 3 ] );
                }

                if ( slice == 0 && tg == 0 )
                {
                    float* state = s_state + key_group * kMmaRows * 2;
                    state[ 2 * g ] = m_low;
                    state[ 2 * g + 1 ] = l_low;
                    state[ 2 * ( g + 8 ) ] = m_high;
                    state[ 2 * ( g + 8 ) + 1 ] = l_high;
                }

                __syncthreads();

                for ( int index = tid; index < group_size * kHeadSize; index += Geometry::kThreads )
                {
                    const int row = index / kHeadSize;
                    const int column = index - row * kHeadSize;

                    float m_max = -CUDART_INF_F;

#pragma unroll
                    for ( int group = 0; group < Geometry::kKeyGroups; ++group )
                        m_max = fmaxf( m_max, s_state[ ( group * kMmaRows + row ) * 2 ] );

                    float l_total = 0.0f;
                    float accumulated = 0.0f;

#pragma unroll
                    for ( int group = 0; group < Geometry::kKeyGroups; ++group )
                    {
                        const float m_group = s_state[ ( group * kMmaRows + row ) * 2 ];
                        const float weight = m_group == -CUDART_INF_F ? 0.0f : __expf( m_group - m_max );

                        l_total += s_state[ ( group * kMmaRows + row ) * 2 + 1 ] * weight;
                        accumulated += s_output[ ( group * kMmaRows + row ) * Geometry::kMergeStride + column ] * weight;
                    }

                    if ( splits.count == 1 )
                    {
                        headRow( row )[ column ] = __float2bfloat16( accumulated / l_total );
                    }
                    else
                    {
                        float* partial = partialRow( row );
                        partial[ column ] = accumulated;

                        if ( column == 0 )
                        {
                            partial[ kHeadSize ] = m_max;
                            partial[ kHeadSize + 1 ] = l_total;
                        }
                    }
                }
            }
#endif // __CUDA_ARCH__ >= 800
        }

        /**
         * @brief Split-K merge for the tensor-core kernel: combines its per-split (O, m, l) partials into Y.
         *
         * The split count is the attention kernel's, from the same device position and target; at one split that
         * kernel wrote Y itself and this one exits.
         */
        __global__ void gqa_decode_attention_mma_fixup_kernel(
            __nv_bfloat16* __restrict__ y,
            const float* __restrict__ split_partials,
            int num_kv_heads,
            int group_size,
            int head_size,
            const int* __restrict__ position,
            int window,
            int target_splits,
            int tile_keys )
        {
            __shared__ float s_weight[ kMaxDecodeSplits ];
            __shared__ float s_inv_l;

            const int actual_len = *position + 1;
            const int num_splits = decodeSplits(
                actual_len - decodeBandStart( actual_len, window ), target_splits, tile_keys ).count;

            if ( num_splits == 1 )
                return;

            const int h = blockIdx.x;
            const int batch = blockIdx.y;
            const int kv = h / group_size;
            const int g = h - kv * group_size;
            const int num_heads = gridDim.x;

            const size_t partial_stride = static_cast<size_t>( head_size + 2 );
            const size_t split_stride = static_cast<size_t>( group_size ) * partial_stride;
            const float* base = split_partials +
                ( ( static_cast<size_t>( batch ) * num_kv_heads + kv ) * num_splits * group_size + g )
                * partial_stride;

            if ( threadIdx.x < 32 )
            {
                float m_max = -CUDART_INF_F;

                for ( int s = threadIdx.x; s < num_splits; s += 32 )
                    m_max = fmaxf( m_max, base[ s * split_stride + head_size ] );

#pragma unroll
                for ( int offset = 16; offset > 0; offset >>= 1 )
                    m_max = fmaxf( m_max, __shfl_xor_sync( 0xffffffffu, m_max, offset ) );

                float l_total = 0.0f;

                for ( int s = threadIdx.x; s < num_splits; s += 32 )
                {
                    const float weight = __expf( base[ s * split_stride + head_size ] - m_max );
                    s_weight[ s ] = weight;
                    l_total += base[ s * split_stride + head_size + 1 ] * weight;
                }

#pragma unroll
                for ( int offset = 16; offset > 0; offset >>= 1 )
                    l_total += __shfl_xor_sync( 0xffffffffu, l_total, offset );

                if ( threadIdx.x == 0 )
                    s_inv_l = 1.0f / l_total;
            }

            __syncthreads();

            for ( int dim = threadIdx.x; dim < head_size; dim += blockDim.x )
            {
                float acc = 0.0f;

                for ( int s = 0; s < num_splits; ++s )
                    acc += base[ s * split_stride + dim ] * s_weight[ s ];

                y[ ( static_cast<size_t>( batch ) * num_heads + h ) * head_size + dim ] =
                    __float2bfloat16( acc * s_inv_l );
            }
        }

        /**
         * The blocks one wave of the tensor-core kernel holds on the current device, measured once per device and
         * kernel; the shared-memory opt-in the launch needs is set on the same first call.
         */
        template<int kHeadSize, bool kFp8, int kKeyGroupCount>
        int mmaDecodeSlots()
        {
            using Geometry = MmaDecodeGeometry<kHeadSize, kFp8, kKeyGroupCount>;

            constexpr int kMaxDevices = 64;
            static std::mutex mutex;
            static std::array<int, kMaxDevices> slots{};

            int device = 0;
            cudaCheck( cudaGetDevice( &device ) );
            assert( device < kMaxDevices );

            std::lock_guard lock( mutex );
            int& cached = slots[ device ];

            if ( cached == 0 )
            {
                const auto kernel = gqa_decode_attention_mma_kernel<kHeadSize, kFp8, kKeyGroupCount>;
                const int shared_bytes = static_cast<int>( Geometry::kSharedBytes );

                cudaCheck( cudaFuncSetAttribute( kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes ) );

                int per_multiprocessor = 0;
                int multiprocessors = 0;
                cudaCheck( cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &per_multiprocessor, kernel, Geometry::kThreads, shared_bytes ) );
                cudaCheck( cudaDeviceGetAttribute( &multiprocessors, cudaDevAttrMultiProcessorCount, device ) );

                cached = std::max( 1, per_multiprocessor ) * multiprocessors;
            }

            return cached;
        }

        /**
         * The split count that fills whole waves: the least one for which (KV heads x batch x splits) is a multiple of
         * the device's slots, so no wave runs part-empty.
         */
        inline int waveFillingSplits( int slots, int kv_head_blocks )
        {
            const int splits = slots / std::gcd( slots, kv_head_blocks );

            return std::clamp( splits, 1, kMaxDecodeSplits );
        }

        template<int kHeadSize, bool kFp8, int kKeyGroupCount>
        void launchMmaDecode(
            const __nv_bfloat16* Q, const void* K, const void* V, const float* k_scales, const float* v_scales,
            __nv_bfloat16* Y, float* split_scratch,
            int B, int NH, int NKV, int cache_capacity,
            const int* position, int max_band, int window, float scale,
            cudaStream_t stream )
        {
            using Geometry = MmaDecodeGeometry<kHeadSize, kFp8, kKeyGroupCount>;

            const int group_size = NH / NKV;
            const int target_splits = waveFillingSplits( mmaDecodeSlots<kHeadSize, kFp8, kKeyGroupCount>(), NKV * B );
            const int grid_splits = decodeGridSplits( max_band, target_splits, Geometry::kTileKeys );

            const dim3 grid( NKV, grid_splits, B );

            gqa_decode_attention_mma_kernel<kHeadSize, kFp8, kKeyGroupCount><<<grid, Geometry::kThreads, Geometry::kSharedBytes, stream>>>(
                Q, K, V, k_scales, v_scales, Y, split_scratch, NKV, group_size, cache_capacity,
                position, window, target_splits, scale );

            cudaCheck( cudaGetLastError() );

            if ( grid_splits > 1 )
            {
                const dim3 fixup_grid( NH, B );

                gqa_decode_attention_mma_fixup_kernel<<<fixup_grid, 128, 0, stream>>>(
                    Y, split_scratch, NKV, group_size, kHeadSize, position, window, target_splits, Geometry::kTileKeys );

                cudaCheck( cudaGetLastError() );
            }
        }

        /// Key groups per head size: four warps a block at every head size. Fewer measured slower at depth and
        /// faster only on short bands, by a few percent either way (GqaDecodeAttention.md).
        template<int kHeadSize>
        constexpr int kKeyGroupsFor = 4 / ( kHeadSize / kSliceDims );

        /**
         * Launch shared by the BF16 and FP8 caches. The grid holds the most splits the band can ever need
         * (max_band), so the launch is the same at every position; the blocks choose the live count.
         */
        template<bool kFp8>
        void launchDecodeAttention(
            const __nv_bfloat16* Q, const void* K, const void* V, const float* k_scales, const float* v_scales,
            __nv_bfloat16* Y, float* split_scratch,
            int B, int NH, int NKV, int HS, int cache_capacity,
            const int* position, int max_band, int window, float scale,
            cudaStream_t stream )
        {
            switch ( HS )
            {
                case 128:
                    launchMmaDecode<128, kFp8, kKeyGroupsFor<128>>( Q, K, V, k_scales, v_scales, Y, split_scratch,
                        B, NH, NKV, cache_capacity, position, max_band, window, scale, stream );
                    break;

                case 256:
                    launchMmaDecode<256, kFp8, kKeyGroupsFor<256>>( Q, K, V, k_scales, v_scales, Y, split_scratch,
                        B, NH, NKV, cache_capacity, position, max_band, window, scale, stream );
                    break;

                case 512:
                    launchMmaDecode<512, kFp8, kKeyGroupsFor<512>>( Q, K, V, k_scales, v_scales, Y, split_scratch,
                        B, NH, NKV, cache_capacity, position, max_band, window, scale, stream );
                    break;

                default:
                    assert( false && "cuda_gqa_decode_attention: unsupported head size" );
                    return;
            }
        }
    } // anonymous namespace


    bool cuda_gqa_decode_attention_supported( int head_size, int group_size )
    {
        return ( head_size == 128 || head_size == 256 || head_size == 512 )
            && group_size >= 1 && group_size <= kMaxDecodeGroupSize;
    }

    size_t cuda_gqa_decode_attention_scratch_bytes( int B, int NH, int HS )
    {
        return static_cast<size_t>( B ) * NH * kMaxDecodeSplits
            * ( static_cast<size_t>( HS ) + 2 ) * sizeof( float );
    }

    void cuda_gqa_decode_attention_bf16(
        const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
        __nv_bfloat16* Y, float* split_scratch,
        int B, int NH, int NKV, int HS, int cache_capacity,
        const int* position, int max_band, int window, float scale,
        cudaStream_t stream )
    {
        assert( NH % NKV == 0 );
        assert( cuda_gqa_decode_attention_supported( HS, NH / NKV ) );
        assert( max_band >= 1 );

        launchDecodeAttention<false>( Q, K, V, nullptr, nullptr, Y, split_scratch,
            B, NH, NKV, HS, cache_capacity, position, max_band, window, scale, stream );
    }

    void cuda_gqa_decode_attention_fp8(
        const __nv_bfloat16* Q, const __nv_fp8_e4m3* K, const __nv_fp8_e4m3* V,
        const float* k_scales, const float* v_scales,
        __nv_bfloat16* Y, float* split_scratch,
        int B, int NH, int NKV, int HS, int cache_capacity,
        const int* position, int max_band, int window, float scale,
        cudaStream_t stream )
    {
        assert( NH % NKV == 0 );
        assert( cuda_gqa_decode_attention_supported( HS, NH / NKV ) );
        assert( max_band >= 1 );

        launchDecodeAttention<true>( Q, K, V, k_scales, v_scales, Y, split_scratch,
            B, NH, NKV, HS, cache_capacity, position, max_band, window, scale, stream );
    }
}
