// Gqa.Flash.Packed.cu
//
// FlashAttention prefill over the unbounded BF16 GQA KV cache (Stage 3, GqaFlashAttention.md 5.7):
// a block owns one KV head and packs its whole query-head group into the MMA rows, position-major.
//
// A block's rows are flat over (query position, head within the group): row f is position f / G and
// head f % G, so any group size packs without padding and every row carries its own position for
// the causal mask. The head dimension splits into 128-wide slices, one warp per slice; a row tile
// is the kSplit warps sharing 16 rows. QK is split-K across those warps and their partial scores
// meet in shared memory behind a named barrier of the row tile only. Every warp sums the partials
// in slice order, so all of a row tile's warps hold bitwise-identical scores and softmax state.
// P is repacked from the score accumulators in registers; O stays in registers. One block barrier
// per key tile guards the double-buffered K/V load.
//
// The same kernel serves the bounded sliding-window ring: key position p lives in cache row
// p % cache_capacity, and the key loop starts at the window's band, which it already does.
//
// m16n8k16 .f32.bf16.bf16.f32 fragment layout (PTX ISA), g = lane/4, tg = lane%4:
//   A[16x16] a0..a3 (row-major): a0={A[g][2t],A[g][2t+1]} a1={A[g+8][2t],..}
//                                a2={A[g][2t+8],..}       a3={A[g+8][2t+8],..}
//   B[16x8]  b0,b1  (col-major): b0={B[2t][g],B[2t+1][g]} b1={B[2t+8][g],B[2t+9][g]}
//   C[16x8]  c0..c3 (row-major): c0=C[g][2t] c1=C[g][2t+1] c2=C[g+8][2t] c3=C[g+8][2t+1]
//
// Two score n-tiles (C) make one A fragment of PV: a0 = C0{c0,c1}, a1 = C0{c2,c3}, a2 = C1{c0,c1},
// a3 = C1{c2,c3} -- which is why P needs no shared memory.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cuda_pipeline.h>
#include "CudaUtils.h"
#include "CudaGqa.cuh"

namespace Mila::Dnn::Compute::Cuda::Gqa
{
    namespace
    {
        constexpr int kTileRows = 16;                          // MMA M: rows per row tile
        constexpr int kMmaK = 16;
        constexpr int kMmaN = 8;
        constexpr int kSliceDims = 128;                        // head dims per warp
        constexpr int kSliceNTiles = kSliceDims / kMmaN;       // O n-tiles per warp (16)
        constexpr int kSliceKSteps = kSliceDims / kMmaK;       // QK k-steps per warp (8)
        constexpr int kWarps = 8;
        constexpr int kCopyElements = 8;                       // bf16 per 16-byte cp.async
        constexpr int kSkew = 8;                               // row padding: rows 4 banks apart
        constexpr int kTileElements = 8192;                    // keys x head dims per K (or V) tile

        // kFp8: the cache holds E4M3 codes and one scale per row (PerTokenKvFp8). The FP8 staging carries the double
        // buffer and each tile widens into ONE BF16 stage, since a second BF16 stage on top would pass the 99 KB a
        // block may hold.
        template<int kHeadSize, bool kFp8 = false>
        struct PackedGeometry
        {
            static_assert( kHeadSize % kSliceDims == 0 && kWarps % ( kHeadSize / kSliceDims ) == 0 );

            static constexpr int kSplit = kHeadSize / kSliceDims;
            static constexpr int kRowTiles = kWarps / kSplit;
            static constexpr int kRows = kRowTiles * kTileRows;
            static constexpr int kKeys = kTileElements / kHeadSize;
            static constexpr int kKeyNTiles = kKeys / kMmaN;
            static constexpr int kKeySteps = kKeys / kMmaK;
            static constexpr int kPad = kHeadSize + kSkew;
            static constexpr int kExchangeStride = kKeys + 2;  // even (float2 stores), rows off-bank

            static constexpr std::size_t kStageElements = static_cast<std::size_t>( kKeys ) * kPad;
            static constexpr int kBf16Stages = kFp8 ? 1 : 2;

            static constexpr std::size_t kExchangeBytes = kSplit > 1
                ? static_cast<std::size_t>( kRowTiles ) * kSplit * kTileRows * kExchangeStride * sizeof( float )
                : 0;

            static constexpr std::size_t kBf16Bytes = 2 * kBf16Stages * kStageElements * sizeof( __nv_bfloat16 );

            // FP8: K and V codes, two stages each, unpadded; then K and V scales, two stages each.
            static constexpr std::size_t kCodeStageBytes = static_cast<std::size_t>( kTileElements );
            static constexpr std::size_t kCodeBytes = kFp8 ? 4 * kCodeStageBytes : 0;
            static constexpr std::size_t kScaleBytes = kFp8 ? 4 * static_cast<std::size_t>( kKeys ) * sizeof( float ) : 0;

            // Exchange floats first (a multiple of 16 bytes), then the BF16 K stages and V stages, then FP8 staging.
            static constexpr std::size_t kSharedBytes = kExchangeBytes + kBf16Bytes + kCodeBytes + kScaleBytes;

            static_assert( kSharedBytes <= 99 * 1024, "a block holds at most 99 KB of shared memory" );
        };

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

        // Explicit cvta + the .shared qualifier on ldmatrix: see Gqa.Flash.Wmma.cu for why both
        // are load-bearing.
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

        // Synchronizes only the `threads` threads that name `barrier` (ids above 0; 0 is
        // __syncthreads), with the shared-memory ordering __syncthreads gives.
        __device__ __forceinline__ void named_barrier( int barrier, int threads )
        {
            asm volatile( "bar.sync %0, %1;\n" :: "r"( barrier ), "r"( threads ) : "memory" );
        }

        __device__ __forceinline__ uint32_t pack_bf16x2( float low, float high )
        {
            const __nv_bfloat162 packed = __floats2bfloat162_rn( low, high );

            return *reinterpret_cast<const uint32_t*>( &packed );
        }

        // One key tile of K and V ([keys x HS] each) into a stage. Unbounded, keys past the cache clamp to its
        // last row; in a ring, position p lives in row p % cache_capacity. Either way a row that is not the key's
        // own lands only on a column the causal or window mask sets to -inf.
        template<int kHeadSize, int kKeys>
        __device__ __forceinline__ void load_kv_tile(
            __nv_bfloat16* k_stage, __nv_bfloat16* v_stage,
            const __nv_bfloat16* K, const __nv_bfloat16* V,
            std::size_t kv_base, int tile_start, int cache_capacity, bool ring, int tid )
        {
            constexpr int kChunksPerRow = kHeadSize / kCopyElements;
            constexpr int kPad = kHeadSize + kSkew;

            for ( int chunk = tid; chunk < kKeys * kChunksPerRow; chunk += kWarps * 32 )
            {
                const int key = chunk / kChunksPerRow;
                const int column = ( chunk % kChunksPerRow ) * kCopyElements;
                const int position = tile_start + key;
                const int row = ring ? position % cache_capacity
                    : ( position < cache_capacity ? position : cache_capacity - 1 );
                const std::size_t source = kv_base + static_cast<std::size_t>( row ) * kHeadSize + column;

                __pipeline_memcpy_async( k_stage + key * kPad + column, K + source, 16 );
                __pipeline_memcpy_async( v_stage + key * kPad + column, V + source, 16 );
            }
        }

        // One key tile of FP8 K and V codes and their row scales into a staging stage, unpadded. Keys past the cache
        // clamp to its last row, as load_kv_tile does.
        template<int kHeadSize, int kKeys>
        __device__ __forceinline__ void load_kv_tile_fp8(
            uint8_t* k_codes, uint8_t* v_codes, float* k_scale_stage, float* v_scale_stage,
            const uint8_t* K, const uint8_t* V, const float* k_scales, const float* v_scales,
            std::size_t kv_base, std::size_t scale_base, int tile_start, int cache_capacity, int tid )
        {
            constexpr int kChunksPerRow = kHeadSize / 16;

            for ( int chunk = tid; chunk < kKeys * kChunksPerRow; chunk += kWarps * 32 )
            {
                const int key = chunk / kChunksPerRow;
                const int column = ( chunk % kChunksPerRow ) * 16;
                const int position = tile_start + key;
                const int row = position < cache_capacity ? position : cache_capacity - 1;
                const std::size_t source = kv_base + static_cast<std::size_t>( row ) * kHeadSize + column;

                __pipeline_memcpy_async( k_codes + key * kHeadSize + column, K + source, 16 );
                __pipeline_memcpy_async( v_codes + key * kHeadSize + column, V + source, 16 );
            }

            for ( int key = tid; key < kKeys; key += kWarps * 32 )
            {
                const int position = tile_start + key;
                const int row = position < cache_capacity ? position : cache_capacity - 1;

                __pipeline_memcpy_async( k_scale_stage + key, k_scales + scale_base + row, 4 );
                __pipeline_memcpy_async( v_scale_stage + key, v_scales + scale_base + row, 4 );
            }
        }

        // Widen a staged tile of E4M3 codes into the padded BF16 stage the MMA reads, unscaled: every E4M3 value is
        // exact in BF16.
        template<int kHeadSize, int kKeys>
        __device__ __forceinline__ void widen_kv_tile(
            __nv_bfloat16* k_stage, __nv_bfloat16* v_stage, const uint8_t* k_codes, const uint8_t* v_codes, int tid )
        {
            constexpr int kPairsPerRow = kHeadSize / 2;
            constexpr int kPad = kHeadSize + kSkew;

            for ( int pair = tid; pair < kKeys * kPairsPerRow; pair += kWarps * 32 )
            {
                const int key = pair / kPairsPerRow;
                const int column = ( pair % kPairsPerRow ) * 2;

                const __nv_fp8x2_storage_t k_pair = *reinterpret_cast<const __nv_fp8x2_storage_t*>( k_codes + key * kHeadSize + column );
                const __nv_fp8x2_storage_t v_pair = *reinterpret_cast<const __nv_fp8x2_storage_t*>( v_codes + key * kHeadSize + column );

                *reinterpret_cast<__nv_bfloat162*>( k_stage + key * kPad + column ) =
                    __float22bfloat162_rn( __half22float2( __half2( __nv_cvt_fp8x2_to_halfraw2( k_pair, __NV_E4M3 ) ) ) );
                *reinterpret_cast<__nv_bfloat162*>( v_stage + key * kPad + column ) =
                    __float22bfloat162_rn( __half22float2( __half2( __nv_cvt_fp8x2_to_halfraw2( v_pair, __NV_E4M3 ) ) ) );
            }
        }
#endif
    }

    // kFp8: K and V hold E4M3 codes with one scale per row (PerTokenKvFp8). Codes widen unscaled into the BF16 stage;
    // each key's K scale multiplies its score after the split-K sum, and its V scale its probability as P is packed
    // for PV, while l sums the unscaled probabilities (Quantization.md, Part III).
    template<int kHeadSize, bool kFp8>
    __global__ void __launch_bounds__( kWarps * 32, 1 )
    gqa_flash_prefill_packed_bf16_kernel(
        const __nv_bfloat16* __restrict__ Q,   // [B, chunk_len, NH * HS]
        const void* __restrict__ K_raw,        // [B, NKV, cache_capacity, HS]
        const void* __restrict__ V_raw,        // [B, NKV, cache_capacity, HS]
        const float* __restrict__ k_scales,    // [B, NKV, cache_capacity], kFp8 only
        const float* __restrict__ v_scales,    // [B, NKV, cache_capacity], kFp8 only
        __nv_bfloat16* __restrict__ Y,         // [B, chunk_len, NH * HS]
        int chunk_len, int NH, int NKV, int cache_capacity,
        int position_offset, int window, float scale, bool ring )
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        using Geometry = PackedGeometry<kHeadSize, kFp8>;

        const __nv_bfloat16* __restrict__ K = static_cast<const __nv_bfloat16*>( K_raw );
        const __nv_bfloat16* __restrict__ V = static_cast<const __nv_bfloat16*>( V_raw );

        const int tid = threadIdx.x;
        const int warp = tid >> 5;
        const int lane = tid & 31;
        const int g = lane >> 2;
        const int tg = lane & 3;

        const int row_tile = warp / Geometry::kSplit;
        const int slice = warp % Geometry::kSplit;
        const int slice0 = slice * kSliceDims;

        const int group = NH / NKV;
        const int kv_head = blockIdx.y;
        const int b = blockIdx.z;
        const int total_rows = chunk_len * group;
        const int block_row0 = blockIdx.x * Geometry::kRows;

        const std::size_t kv_base = ( static_cast<std::size_t>( b ) * NKV + kv_head ) * cache_capacity * kHeadSize;

        extern __shared__ __align__( 16 ) char smem_raw[];
        float* s_exchange = reinterpret_cast<float*>( smem_raw );
        __nv_bfloat16* s_k = reinterpret_cast<__nv_bfloat16*>( smem_raw + Geometry::kExchangeBytes );
        __nv_bfloat16* s_v = s_k + Geometry::kBf16Stages * Geometry::kStageElements;

        // FP8 staging, after the BF16 stages: K codes (two stages), V codes, then K scales and V scales.
        uint8_t* s_k_codes = reinterpret_cast<uint8_t*>( smem_raw + Geometry::kExchangeBytes + Geometry::kBf16Bytes );
        uint8_t* s_v_codes = s_k_codes + 2 * Geometry::kCodeStageBytes;
        float* s_k_scale = reinterpret_cast<float*>( s_v_codes + 2 * Geometry::kCodeStageBytes );
        float* s_v_scale = s_k_scale + 2 * Geometry::kKeys;
        const std::size_t scale_base = ( static_cast<std::size_t>( blockIdx.z ) * NKV + blockIdx.y ) * cache_capacity;

        // The tile's BF16 stage: alternating for BF16, the one widened stage for FP8.
        const auto bf16Stage = [&]( __nv_bfloat16* base, int tile ) -> __nv_bfloat16*
        {
            return base + ( kFp8 ? 0 : ( tile & 1 ) ) * Geometry::kStageElements;
        };

        const auto loadTile = [&]( int tile )
        {
            if constexpr ( kFp8 )
            {
                const int stage = tile & 1;

                load_kv_tile_fp8<kHeadSize, Geometry::kKeys>(
                    s_k_codes + stage * Geometry::kCodeStageBytes, s_v_codes + stage * Geometry::kCodeStageBytes,
                    s_k_scale + stage * Geometry::kKeys, s_v_scale + stage * Geometry::kKeys,
                    static_cast<const uint8_t*>( K_raw ), static_cast<const uint8_t*>( V_raw ), k_scales, v_scales,
                    kv_base, scale_base,
                    tile * Geometry::kKeys, cache_capacity, threadIdx.x );
            }
            else
            {
                load_kv_tile<kHeadSize, Geometry::kKeys>(
                    bf16Stage( s_k, tile ), bf16Stage( s_v, tile ),
                    K, V, kv_base,
                    tile * Geometry::kKeys, cache_capacity, ring, threadIdx.x );
            }
        };

        // This lane's two rows: g and g + 8 of its row tile.
        const int row_low = block_row0 + row_tile * kTileRows + g;
        const int row_high = row_low + 8;
        const bool active_low = row_low < total_rows;
        const bool active_high = row_high < total_rows;
        const int position_low = active_low ? row_low / group : 0;
        const int position_high = active_high ? row_high / group : 0;
        const int absolute_low = position_offset + position_low;
        const int absolute_high = position_offset + position_high;
        const int window_low = window > 0 ? max( 0, absolute_low - window + 1 ) : 0;
        const int window_high = window > 0 ? max( 0, absolute_high - window + 1 ) : 0;

        const auto rowOffset = [&]( int row ) -> std::size_t
        {
            const int position = row / group;
            const int head = kv_head * group + row % group;

            return ( ( static_cast<std::size_t>( b ) * chunk_len + position ) * NH + head ) * kHeadSize;
        };

        const std::size_t offset_low = active_low ? rowOffset( row_low ) : 0;
        const std::size_t offset_high = active_high ? rowOffset( row_high ) : 0;

        // Q for this warp's slice, straight into A fragments. Rows past the chunk are zero.
        uint32_t q[ kSliceKSteps ][ 4 ];
#pragma unroll
        for ( int step = 0; step < kSliceKSteps; ++step )
        {
            const int column = slice0 + step * kMmaK + 2 * tg;
            const uint32_t* low = reinterpret_cast<const uint32_t*>( Q + offset_low + column );
            const uint32_t* high = reinterpret_cast<const uint32_t*>( Q + offset_high + column );

            q[ step ][ 0 ] = active_low ? low[ 0 ] : 0u;
            q[ step ][ 1 ] = active_high ? high[ 0 ] : 0u;
            q[ step ][ 2 ] = active_low ? low[ 4 ] : 0u;
            q[ step ][ 3 ] = active_high ? high[ 4 ] : 0u;
        }

        // The block's key range: every position its rows hold. Row order is position-major.
        const int last_row = min( block_row0 + Geometry::kRows, total_rows ) - 1;
        const int first_position = block_row0 / group;
        const int last_position = last_row / group;
        const int band_start = window > 0 ? max( 0, position_offset + first_position - window + 1 ) : 0;
        const int first_tile = band_start / Geometry::kKeys;
        const int tile_count = ( position_offset + last_position ) / Geometry::kKeys + 1;

        float o[ kSliceNTiles ][ 4 ];
#pragma unroll
        for ( int nt = 0; nt < kSliceNTiles; ++nt )
            o[ nt ][ 0 ] = o[ nt ][ 1 ] = o[ nt ][ 2 ] = o[ nt ][ 3 ] = 0.0f;

        float m_low = -CUDART_INF_F, m_high = -CUDART_INF_F;
        float l_low = 0.0f, l_high = 0.0f;

        loadTile( first_tile );
        __pipeline_commit();

        for ( int tile = first_tile; tile < tile_count; ++tile )
        {
            const int tile_start = tile * Geometry::kKeys;
            const __nv_bfloat16* k_tile = bf16Stage( s_k, tile );
            const __nv_bfloat16* v_tile = bf16Stage( s_v, tile );
            const float* k_scale_tile = s_k_scale + ( tile & 1 ) * Geometry::kKeys;
            const float* v_scale_tile = s_v_scale + ( tile & 1 ) * Geometry::kKeys;

            __pipeline_wait_prior( 0 );

            // This tile visible to every warp; and, every thread having finished the previous
            // tile, the stage the next prefetch writes and the exchange buffer are free.
            __syncthreads();

            if constexpr ( kFp8 )
            {
                // The single BF16 stage is free: every thread finished the previous tile above.
                widen_kv_tile<kHeadSize, Geometry::kKeys>( bf16Stage( s_k, tile ), bf16Stage( s_v, tile ),
                    s_k_codes + ( tile & 1 ) * Geometry::kCodeStageBytes,
                    s_v_codes + ( tile & 1 ) * Geometry::kCodeStageBytes, tid );
            }

            if ( tile + 1 < tile_count )
            {
                loadTile( tile + 1 );
                __pipeline_commit();
            }

            if constexpr ( kFp8 )
            {
                // The widened tile visible to every warp before the MMAs read it.
                __syncthreads();
            }

            // --- QK over this warp's slice ---
            float s[ Geometry::kKeyNTiles ][ 4 ];
#pragma unroll
            for ( int nt = 0; nt < Geometry::kKeyNTiles; ++nt )
                s[ nt ][ 0 ] = s[ nt ][ 1 ] = s[ nt ][ 2 ] = s[ nt ][ 3 ] = 0.0f;

#pragma unroll
            for ( int step = 0; step < kSliceKSteps; ++step )
            {
#pragma unroll
                for ( int pair = 0; pair < Geometry::kKeyNTiles / 2; ++pair )
                {
                    uint32_t b00, b01, b10, b11;
                    const int key = pair * 16 + ( lane >> 4 ) * kMmaN + ( lane & 7 );
                    const int column = slice0 + step * kMmaK + ( ( lane >> 3 ) & 1 ) * kMmaN;
                    ldmatrix_x4( b00, b01, b10, b11, shared_address( &k_tile[ key * Geometry::kPad + column ] ) );

                    mma_m16n8k16_bf16( s[ 2 * pair ][ 0 ], s[ 2 * pair ][ 1 ], s[ 2 * pair ][ 2 ], s[ 2 * pair ][ 3 ],
                        q[ step ][ 0 ], q[ step ][ 1 ], q[ step ][ 2 ], q[ step ][ 3 ], b00, b01 );
                    mma_m16n8k16_bf16( s[ 2 * pair + 1 ][ 0 ], s[ 2 * pair + 1 ][ 1 ], s[ 2 * pair + 1 ][ 2 ], s[ 2 * pair + 1 ][ 3 ],
                        q[ step ][ 0 ], q[ step ][ 1 ], q[ step ][ 2 ], q[ step ][ 3 ], b10, b11 );
                }
            }

            // --- the row tile's partial scores meet: every warp sums all slices in slice order ---
            if constexpr ( Geometry::kSplit > 1 )
            {
                float* mine = s_exchange + ( row_tile * Geometry::kSplit + slice ) * kTileRows * Geometry::kExchangeStride;

#pragma unroll
                for ( int nt = 0; nt < Geometry::kKeyNTiles; ++nt )
                {
                    const int column = nt * kMmaN + 2 * tg;
                    *reinterpret_cast<float2*>( &mine[ g * Geometry::kExchangeStride + column ] ) = make_float2( s[ nt ][ 0 ], s[ nt ][ 1 ] );
                    *reinterpret_cast<float2*>( &mine[ ( g + 8 ) * Geometry::kExchangeStride + column ] ) = make_float2( s[ nt ][ 2 ], s[ nt ][ 3 ] );
                }

                named_barrier( 1 + row_tile, Geometry::kSplit * 32 );

#pragma unroll
                for ( int nt = 0; nt < Geometry::kKeyNTiles; ++nt )
                    s[ nt ][ 0 ] = s[ nt ][ 1 ] = s[ nt ][ 2 ] = s[ nt ][ 3 ] = 0.0f;

#pragma unroll
                for ( int part = 0; part < Geometry::kSplit; ++part )
                {
                    const float* partial = s_exchange + ( row_tile * Geometry::kSplit + part ) * kTileRows * Geometry::kExchangeStride;

#pragma unroll
                    for ( int nt = 0; nt < Geometry::kKeyNTiles; ++nt )
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

            // --- online softmax for rows g and g + 8; a lane holds keys nt*8 + 2tg + {0,1} ---
            float max_low = -CUDART_INF_F, max_high = -CUDART_INF_F;

#pragma unroll
            for ( int nt = 0; nt < Geometry::kKeyNTiles; ++nt )
            {
#pragma unroll
                for ( int j = 0; j < 2; ++j )
                {
                    const int key = tile_start + nt * kMmaN + 2 * tg + j;
                    const float key_scale = kFp8 ? scale * k_scale_tile[ nt * kMmaN + 2 * tg + j ] : scale;

                    float low = s[ nt ][ j ] * key_scale;
                    if ( !active_low || key > absolute_low || key < window_low )
                        low = -CUDART_INF_F;

                    float high = s[ nt ][ 2 + j ] * key_scale;
                    if ( !active_high || key > absolute_high || key < window_high )
                        high = -CUDART_INF_F;

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

            // A fully masked tile keeps the row's state; the exps below then see only -inf.
            const bool scored_low = max_low > -CUDART_INF_F;
            const bool scored_high = max_high > -CUDART_INF_F;
            const float new_m_low = scored_low ? fmaxf( m_low, max_low ) : m_low;
            const float new_m_high = scored_high ? fmaxf( m_high, max_high ) : m_high;

            float sum_low = 0.0f, sum_high = 0.0f;

#pragma unroll
            for ( int nt = 0; nt < Geometry::kKeyNTiles; ++nt )
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

            float alpha_low = 1.0f, alpha_high = 1.0f;

            if ( scored_low )
            {
                alpha_low = m_low == -CUDART_INF_F ? 0.0f : __expf( m_low - new_m_low );
                l_low = l_low * alpha_low + sum_low;
                m_low = new_m_low;
            }

            if ( scored_high )
            {
                alpha_high = m_high == -CUDART_INF_F ? 0.0f : __expf( m_high - new_m_high );
                l_high = l_high * alpha_high + sum_high;
                m_high = new_m_high;
            }

#pragma unroll
            for ( int nt = 0; nt < kSliceNTiles; ++nt )
            {
                o[ nt ][ 0 ] *= alpha_low;
                o[ nt ][ 1 ] *= alpha_low;
                o[ nt ][ 2 ] *= alpha_high;
                o[ nt ][ 3 ] *= alpha_high;
            }

            // --- PV over this warp's 128 output columns, P straight from the score registers ---
#pragma unroll
            for ( int step = 0; step < Geometry::kKeySteps; ++step )
            {
                // Keys 2*step*8 + 2tg + {0,1} and (2*step + 1)*8 + 2tg + {0,1}; rows g and g + 8 share them.
                float v0 = 1.0f, v1 = 1.0f, v2 = 1.0f, v3 = 1.0f;

                if constexpr ( kFp8 )
                {
                    v0 = v_scale_tile[ 2 * step * kMmaN + 2 * tg ];
                    v1 = v_scale_tile[ 2 * step * kMmaN + 2 * tg + 1 ];
                    v2 = v_scale_tile[ ( 2 * step + 1 ) * kMmaN + 2 * tg ];
                    v3 = v_scale_tile[ ( 2 * step + 1 ) * kMmaN + 2 * tg + 1 ];
                }

                const uint32_t p0 = pack_bf16x2( s[ 2 * step ][ 0 ] * v0, s[ 2 * step ][ 1 ] * v1 );
                const uint32_t p1 = pack_bf16x2( s[ 2 * step ][ 2 ] * v0, s[ 2 * step ][ 3 ] * v1 );
                const uint32_t p2 = pack_bf16x2( s[ 2 * step + 1 ][ 0 ] * v2, s[ 2 * step + 1 ][ 1 ] * v3 );
                const uint32_t p3 = pack_bf16x2( s[ 2 * step + 1 ][ 2 ] * v2, s[ 2 * step + 1 ][ 3 ] * v3 );

#pragma unroll
                for ( int pair = 0; pair < kSliceNTiles / 2; ++pair )
                {
                    uint32_t b00, b01, b10, b11;
                    ldmatrix_x4_trans( b00, b01, b10, b11, shared_address( &v_tile[
                        ( step * kMmaK + ( lane & 15 ) ) * Geometry::kPad + slice0 + pair * 16 + ( lane >> 4 ) * kMmaN ] ) );

                    mma_m16n8k16_bf16( o[ 2 * pair ][ 0 ], o[ 2 * pair ][ 1 ], o[ 2 * pair ][ 2 ], o[ 2 * pair ][ 3 ],
                        p0, p1, p2, p3, b00, b01 );
                    mma_m16n8k16_bf16( o[ 2 * pair + 1 ][ 0 ], o[ 2 * pair + 1 ][ 1 ], o[ 2 * pair + 1 ][ 2 ], o[ 2 * pair + 1 ][ 3 ],
                        p0, p1, p2, p3, b10, b11 );
                }
            }
        }

        // --- normalize and write this warp's 128 columns of its rows ---
        const float inverse_low = active_low ? 1.0f / l_low : 0.0f;
        const float inverse_high = active_high ? 1.0f / l_high : 0.0f;

#pragma unroll
        for ( int nt = 0; nt < kSliceNTiles; ++nt )
        {
            const int column = slice0 + nt * kMmaN + 2 * tg;

            if ( active_low )
                *reinterpret_cast<__nv_bfloat162*>( Y + offset_low + column ) =
                    __floats2bfloat162_rn( o[ nt ][ 0 ] * inverse_low, o[ nt ][ 1 ] * inverse_low );

            if ( active_high )
                *reinterpret_cast<__nv_bfloat162*>( Y + offset_high + column ) =
                    __floats2bfloat162_rn( o[ nt ][ 2 ] * inverse_high, o[ nt ][ 3 ] * inverse_high );
        }
#endif // __CUDA_ARCH__ >= 800
    }

    namespace
    {
        template<int kHeadSize, bool kFp8>
        void launch_packed_prefill(
            const __nv_bfloat16* Q, const void* K, const void* V, const float* k_scales, const float* v_scales,
            __nv_bfloat16* Y, int B, int chunk_len, int NH, int NKV, int cache_capacity,
            int position_offset, int window, float scale, bool ring, cudaStream_t stream )
        {
            using Geometry = PackedGeometry<kHeadSize, kFp8>;

            cudaCheck( cudaFuncSetAttribute(
                gqa_flash_prefill_packed_bf16_kernel<kHeadSize, kFp8>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>( Geometry::kSharedBytes ) ) );

            const int total_rows = chunk_len * ( NH / NKV );
            const dim3 grid( ceil_div( total_rows, Geometry::kRows ), NKV, B );

            gqa_flash_prefill_packed_bf16_kernel<kHeadSize, kFp8> <<< grid, kWarps * 32, Geometry::kSharedBytes, stream >>> (
                Q, K, V, k_scales, v_scales, Y, chunk_len, NH, NKV, cache_capacity, position_offset, window, scale, ring );

            cudaCheck( cudaGetLastError() );
        }

        /// The geometry and device checks every flash launch shares.
        void requireLaunchable( const char* caller, int NH, int NKV )
        {
            if ( NKV <= 0 || NH % NKV != 0 )
                throw std::runtime_error( std::string( caller ) + ": query heads must be a multiple of KV heads" );

            int device = 0;
            int sm_major = 0;
            cudaCheck( cudaGetDevice( &device ) );
            cudaCheck( cudaDeviceGetAttribute( &sm_major, cudaDevAttrComputeCapabilityMajor, device ) );

            if ( sm_major < 8 )
                throw std::runtime_error( std::string( caller ) + ": requires compute capability 8.0 or later" );
        }

        /// Head size, geometry and device checks, then the packed launch for either cache.
        template<bool kFp8>
        void flashPrefill(
            const char* caller,
            const __nv_bfloat16* Q, const void* K, const void* V, const float* k_scales, const float* v_scales,
            __nv_bfloat16* Y, int B, int chunk_len, int NH, int NKV, int HS, int cache_capacity,
            int position_offset, int window, float scale, bool ring, cudaStream_t stream )
        {
            if ( !cuda_gqa_flash_prefill_supported( HS ) )
                throw std::runtime_error( std::string( caller ) + ": head size " + std::to_string( HS )
                    + " is not supported; the packed kernel serves 128, 256 and 512" );

            requireLaunchable( caller, NH, NKV );

            switch ( HS )
            {
                case 128:
                    launch_packed_prefill<128, kFp8>( Q, K, V, k_scales, v_scales, Y, B, chunk_len, NH, NKV, cache_capacity,
                        position_offset, window, scale, ring, stream );
                    break;

                case 256:
                    launch_packed_prefill<256, kFp8>( Q, K, V, k_scales, v_scales, Y, B, chunk_len, NH, NKV, cache_capacity,
                        position_offset, window, scale, ring, stream );
                    break;

                case 512:
                    launch_packed_prefill<512, kFp8>( Q, K, V, k_scales, v_scales, Y, B, chunk_len, NH, NKV, cache_capacity,
                        position_offset, window, scale, ring, stream );
                    break;
            }
        }
    }

    void cuda_gqa_flash_prefill_ring_bf16(
        const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
        __nv_bfloat16* Y,
        int B, int chunk_len, int NH, int NKV, int HS, int cache_capacity,
        int position_offset, int window, float scale,
        cudaStream_t stream )
    {
        // The packed kernel reads each KV head once for its whole query-head group; the FA-2 kernel, once per query
        // head, serves the head sizes the packed geometry does not.
        if ( !cuda_gqa_flash_prefill_supported( HS ) )
        {
            cuda_gqa_flash_prefill_ring_fa2_bf16( Q, K, V, Y, B, chunk_len, NH, NKV, HS, cache_capacity,
                position_offset, window, scale, stream );

            return;
        }

        if ( window <= 0 )
            throw std::runtime_error( "cuda_gqa_flash_prefill_ring_bf16: a bounded ring requires a positive window" );

        flashPrefill<false>( "cuda_gqa_flash_prefill_ring_bf16", Q, K, V, nullptr, nullptr, Y,
            B, chunk_len, NH, NKV, HS, cache_capacity, position_offset, window, scale, true, stream );
    }

    bool cuda_gqa_flash_prefill_supported( int head_size )
    {
        return head_size == 128 || head_size == 256 || head_size == 512;
    }

    void cuda_gqa_flash_prefill_bf16(
        const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
        __nv_bfloat16* Y,
        int B, int chunk_len, int NH, int NKV, int HS, int cache_capacity,
        int position_offset, int window, float scale,
        cudaStream_t stream )
    {
        // Head size 512 over the whole cache runs key-major (Gqa.Flash.WideHead.cu); the packed kernel serves the
        // other head sizes and a band-limited window.
        if ( HS == 512 && window <= 0 )
        {
            requireLaunchable( "cuda_gqa_flash_prefill_bf16", NH, NKV );
            cuda_gqa_flash_prefill_wide_head_bf16( Q, K, V, Y, B, chunk_len, NH, NKV, cache_capacity,
                position_offset, scale, stream );

            return;
        }

        flashPrefill<false>( "cuda_gqa_flash_prefill_bf16", Q, K, V, nullptr, nullptr, Y,
            B, chunk_len, NH, NKV, HS, cache_capacity, position_offset, window, scale, false, stream );
    }

    void cuda_gqa_flash_prefill_fp8(
        const __nv_bfloat16* Q, const __nv_fp8_e4m3* K, const __nv_fp8_e4m3* V,
        const float* k_scales, const float* v_scales,
        __nv_bfloat16* Y,
        int B, int chunk_len, int NH, int NKV, int HS, int cache_capacity,
        int position_offset, int window, float scale,
        cudaStream_t stream )
    {
        // The same split as cuda_gqa_flash_prefill_bf16, so a cache of lossless codes matches the BF16 cache bit for bit.
        if ( HS == 512 && window <= 0 )
        {
            requireLaunchable( "cuda_gqa_flash_prefill_fp8", NH, NKV );
            cuda_gqa_flash_prefill_wide_head_fp8( Q, K, V, k_scales, v_scales, Y, B, chunk_len, NH, NKV, cache_capacity,
                position_offset, scale, stream );

            return;
        }

        flashPrefill<true>( "cuda_gqa_flash_prefill_fp8", Q, K, V, k_scales, v_scales, Y,
            B, chunk_len, NH, NKV, HS, cache_capacity, position_offset, window, scale, false, stream );
    }
}