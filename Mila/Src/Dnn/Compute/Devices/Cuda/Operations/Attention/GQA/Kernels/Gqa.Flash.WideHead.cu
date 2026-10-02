// Gqa.Flash.WideHead.cu
//
// FlashAttention prefill over the unbounded GQA KV cache at head size 512 (GqaFlashAttention.md 5.8): keys are the
// MMA rows, so each warp owns eight query rows across the whole head and no warp shares a score.
//
// A block's 64 rows are flat over (query position, head within the group), as in Gqa.Flash.Packed.cu. S^T = K Q^T
// puts 16 keys on the MMA M dimension and a warp's 8 query rows on N, each warp summing all 512 dims itself. P^T
// comes out of the score accumulators by one movmatrix per 8 keys, and O^T = V^T P^T keeps [512 dims x 8 rows] in
// FP32 registers. K and V alternate through two single stages: V(t) loads during QK(t), K(t + 1) during PV(t).
// Over the BF16 cache, Q's first 256 dims live in registers and the rest in shared memory, which leaves registers
// to hide latency; the FP8 cache needs that shared memory for its code staging and holds all of Q in registers.
//
// QK is BF16 with FP32 accumulation. PV runs on FP16 V and P, each 32-key tile summed with FP16 accumulation --
// twice the FP32-accumulate rate on GeForce -- and folded into FP32 O (GqaFlashAttention.md 5.8). A tile's sum
// stays finite while |v| < 2047; head size 512 is Gemma's, whose weightless v_norm bounds |v| by sqrt(512).
//
// m16n8k16 .f32.bf16.bf16.f32 fragment layout (PTX ISA), g = lane/4, tg = lane%4:
//   A[16x16] a0..a3 (row-major): a0={A[g][2t],A[g][2t+1]} a1={A[g+8][2t],..}
//                                a2={A[g][2t+8],..}       a3={A[g+8][2t+8],..}
//   B[16x8]  b0,b1  (col-major): b0={B[2t][g],B[2t+1][g]} b1={B[2t+8][g],B[2t+9][g]}
//   C[16x8]  c0..c3 (row-major): c0=C[g][2t] c1=C[g][2t+1] c2=C[g+8][2t] c3=C[g+8][2t+1]
// The .f16 accumulator packs the same positions in pairs: d0={C[g][2t],C[g][2t+1]}, d1={C[g+8][2t],C[g+8][2t+1]}.
//
// A score C tile holds keys g, g + 8 of rows 2t, 2t + 1; the PV B fragment needs keys 2t, 2t + 1 of row g, so each
// 8-key half is the transpose of its C half -- movmatrix.trans of the packed pair.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>
#include <cstdint>
#include <stdexcept>
#include <cuda_pipeline.h>
#include "CudaUtils.h"
#include "CudaGqa.cuh"
#include "Gqa.Fp8Widen.cuh"

namespace Mila::Dnn::Compute::Cuda::Gqa
{
    namespace
    {
        constexpr int kHeadSize = 512;
        constexpr int kWarps = 8;
        constexpr int kWarpRows = 8;                               // query rows per warp: the MMA N
        constexpr int kRows = kWarps * kWarpRows;                  // 64 per block
        constexpr int kKeys = 32;                                  // keys per tile
        constexpr int kKeyTiles = kKeys / 16;                      // MMA M tiles per key tile
        constexpr int kSteps = kHeadSize / 16;                     // QK k-steps
        constexpr int kDimTiles = kHeadSize / 16;                  // PV M tiles
        constexpr int kPad = kHeadSize + 8;                        // rows 4 banks apart
        constexpr std::size_t kStageElements = static_cast<std::size_t>( kKeys ) * kPad;
        constexpr std::size_t kStageBytes = 2 * kStageElements * sizeof( __nv_bfloat16 );

        static_assert( kRows <= 2 * kKeys, "the output staging reuses the K and V stages" );

        // kFp8: the cache holds E4M3 codes and one scale per row (PerTokenKvFp8); each tile's codes stage unpadded
        // and widen into the stage the MMAs read, K as BF16 and V as FP16.
        template<bool kFp8>
        struct WideHeadGeometry
        {
            static constexpr int kRegisterSteps = kFp8 ? kSteps : kSteps / 2;   // QK k-steps of Q held in registers
            static constexpr int kSharedQueryDims = ( kSteps - kRegisterSteps ) * 16;
            static constexpr int kQueryPad = kSharedQueryDims + 8;

            static constexpr std::size_t kQueryBytes = kSharedQueryDims > 0
                ? static_cast<std::size_t>( kRows ) * kQueryPad * sizeof( __nv_bfloat16 )
                : 0;

            static constexpr std::size_t kCodeStageBytes = static_cast<std::size_t>( kKeys ) * kHeadSize;
            static constexpr std::size_t kCodeBytes = kFp8 ? 2 * kCodeStageBytes + 2 * kKeys * sizeof( float ) : 0;

            // K stage, V stage, then Q's shared dims (BF16) or the K and V codes and their scales (FP8).
            static constexpr std::size_t kSharedBytes = kStageBytes + kQueryBytes + kCodeBytes;

            static_assert( kSharedBytes <= 99 * 1024, "a block holds at most 99 KB of shared memory" );
        };

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        __device__ __forceinline__ void mma_m16n8k16_bf16( float* c,
            uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1 )
        {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"( c[ 0 ] ), "+f"( c[ 1 ] ), "+f"( c[ 2 ] ), "+f"( c[ 3 ] )
                : "r"( a0 ), "r"( a1 ), "r"( a2 ), "r"( a3 ), "r"( b0 ), "r"( b1 ) );
        }

        __device__ __forceinline__ void mma_m16n8k16_f16( uint32_t* d,
            uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1 )
        {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                : "+r"( d[ 0 ] ), "+r"( d[ 1 ] )
                : "r"( a0 ), "r"( a1 ), "r"( a2 ), "r"( a3 ), "r"( b0 ), "r"( b1 ) );
        }

        __device__ __forceinline__ uint32_t shared_address( const void* pointer )
        {
            return static_cast<uint32_t>( __cvta_generic_to_shared( pointer ) );
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

        __device__ __forceinline__ uint32_t movmatrix_trans( uint32_t fragment )
        {
            uint32_t transposed;
            asm volatile( "movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"( transposed ) : "r"( fragment ) );

            return transposed;
        }

        __device__ __forceinline__ uint32_t pack_half2( float low, float high )
        {
            const __half2 packed = __floats2half2_rn( low, high );

            return *reinterpret_cast<const uint32_t*>( &packed );
        }

        __device__ __forceinline__ int cache_row( int position, int cache_capacity )
        {
            return position < cache_capacity ? position : cache_capacity - 1;
        }

        // One tile of K or V ([kKeys x 512]) into a stage. Keys past the cache clamp to its last row, which lands only
        // on a column the causal mask sets to -inf.
        __device__ __forceinline__ void load_tile( __nv_bfloat16* stage, const __nv_bfloat16* source,
            std::size_t kv_base, int tile_start, int cache_capacity, int tid )
        {
            constexpr int kChunksPerRow = kHeadSize / 8;

#pragma unroll
            for ( int i = 0; i < kKeys * kChunksPerRow / ( kWarps * 32 ); ++i )
            {
                const int chunk = tid + i * kWarps * 32;
                const int key = chunk / kChunksPerRow;
                const int column = ( chunk % kChunksPerRow ) * 8;
                const int row = cache_row( tile_start + key, cache_capacity );

                __pipeline_memcpy_async( stage + key * kPad + column,
                    source + kv_base + static_cast<std::size_t>( row ) * kHeadSize + column, 16 );
            }
        }

        // One tile of E4M3 codes ([kKeys x 512], unpadded) and their row scales into a code stage.
        __device__ __forceinline__ void load_code_tile( uint8_t* codes, float* scales,
            const uint8_t* source, const float* source_scales,
            std::size_t kv_base, std::size_t scale_base, int tile_start, int cache_capacity, int tid )
        {
            constexpr int kChunksPerRow = kHeadSize / 16;

#pragma unroll
            for ( int i = 0; i < kKeys * kChunksPerRow / ( kWarps * 32 ); ++i )
            {
                const int chunk = tid + i * kWarps * 32;
                const int key = chunk / kChunksPerRow;
                const int column = ( chunk % kChunksPerRow ) * 16;
                const int row = cache_row( tile_start + key, cache_capacity );

                __pipeline_memcpy_async( codes + key * kHeadSize + column,
                    source + kv_base + static_cast<std::size_t>( row ) * kHeadSize + column, 16 );
            }

            if ( tid < kKeys )
                __pipeline_memcpy_async( scales + tid, source_scales + scale_base + cache_row( tile_start + tid, cache_capacity ), 4 );
        }

        // Widen a K code stage into the padded BF16 stage, unscaled, sixteen codes a thread a step.
        __device__ __forceinline__ void widen_tile( __nv_bfloat16* stage, const uint8_t* codes, int tid )
        {
            constexpr int kChunksPerRow = kHeadSize / 16;

#pragma unroll
            for ( int i = 0; i < kKeys * kChunksPerRow / ( kWarps * 32 ); ++i )
            {
                const int chunk = tid + i * kWarps * 32;
                const int key = chunk / kChunksPerRow;
                const int column = ( chunk % kChunksPerRow ) * 16;

                widen_sixteen_codes( *reinterpret_cast<const uint4*>( codes + key * kHeadSize + column ), stage + key * kPad + column );
            }
        }

        // Widen a V code stage into the padded stage as FP16, each key's values times its scale. Chunks are walked as
        // load_code_tile walks them, though the scales come from other threads, so a block barrier comes first.
        __device__ __forceinline__ void widen_value_tile( __half* stage, const uint8_t* codes, const float* scales, int tid )
        {
            constexpr int kChunksPerRow = kHeadSize / 16;

#pragma unroll
            for ( int i = 0; i < kKeys * kChunksPerRow / ( kWarps * 32 ); ++i )
            {
                const int chunk = tid + i * kWarps * 32;
                const int key = chunk / kChunksPerRow;
                const int column = ( chunk % kChunksPerRow ) * 16;

                widen_sixteen_codes_to_half( *reinterpret_cast<const uint4*>( codes + key * kHeadSize + column ),
                    scales[ key ], stage + key * kPad + column );
            }
        }

        // Rewrite a BF16 V stage as FP16 in place, eight values a thread a step. Chunks are walked as load_tile walks
        // them, so a thread rewrites only what its own cp.async wrote, and its own pipeline wait suffices.
        __device__ __forceinline__ void narrow_value_tile_to_half( __nv_bfloat16* stage, int tid )
        {
            constexpr int kChunksPerRow = kHeadSize / 8;

#pragma unroll
            for ( int i = 0; i < kKeys * kChunksPerRow / ( kWarps * 32 ); ++i )
            {
                const int chunk = tid + i * kWarps * 32;
                const int key = chunk / kChunksPerRow;
                const int column = ( chunk % kChunksPerRow ) * 8;

                uint4* values = reinterpret_cast<uint4*>( stage + key * kPad + column );
                const uint4 raw = *values;
                const uint32_t words[ 4 ] = { raw.x, raw.y, raw.z, raw.w };
                uint32_t narrowed[ 4 ];

#pragma unroll
                for ( int word = 0; word < 4; ++word )
                {
                    const float2 value = __bfloat1622float2( *reinterpret_cast<const __nv_bfloat162*>( &words[ word ] ) );
                    const __half2 converted = __floats2half2_rn( value.x, value.y );
                    narrowed[ word ] = *reinterpret_cast<const uint32_t*>( &converted );
                }

                *values = make_uint4( narrowed[ 0 ], narrowed[ 1 ], narrowed[ 2 ], narrowed[ 3 ] );
            }
        }
#endif
    }

    // kFp8: each key's K scale multiplies its score, and its V scale its values as they widen to FP16 for PV, while l sums
    // the probabilities; so a cache holding lossless values -- every scale a power of two -- matches the BF16 cache bit
    // for bit (Quantization.md, Part III).
    template<bool kFp8>
    __global__ void __launch_bounds__( kWarps * 32, 1 )
    gqa_flash_prefill_wide_head_kernel(
        const __nv_bfloat16* __restrict__ Q,   // [B, chunk_len, NH * 512]
        const void* __restrict__ K_raw,        // [B, NKV, cache_capacity, 512]
        const void* __restrict__ V_raw,        // [B, NKV, cache_capacity, 512]
        const float* __restrict__ k_scales,    // [B, NKV, cache_capacity], kFp8 only
        const float* __restrict__ v_scales,    // [B, NKV, cache_capacity], kFp8 only
        __nv_bfloat16* __restrict__ Y,         // [B, chunk_len, NH * 512]
        int chunk_len, int NH, int NKV, int cache_capacity, int position_offset, float scale )
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        using Geometry = WideHeadGeometry<kFp8>;

        const int tid = threadIdx.x;
        const int warp = tid >> 5;
        const int lane = tid & 31;
        const int g = lane >> 2;
        const int tg = lane & 3;

        const int group = NH / NKV;
        const int kv_head = blockIdx.y;
        const int b = blockIdx.z;
        const int total_rows = chunk_len * group;
        const int block_row0 = blockIdx.x * kRows;
        const int warp_row0 = block_row0 + warp * kWarpRows;

        const std::size_t kv_base = ( static_cast<std::size_t>( b ) * NKV + kv_head ) * cache_capacity * kHeadSize;
        const std::size_t scale_base = ( static_cast<std::size_t>( b ) * NKV + kv_head ) * cache_capacity;

        extern __shared__ __align__( 16 ) char smem_raw[];
        __nv_bfloat16* s_k = reinterpret_cast<__nv_bfloat16*>( smem_raw );
        __nv_bfloat16* s_v = s_k + kStageElements;
        __nv_bfloat16* s_q = s_v + kStageElements;

        // FP8, after the K and V stages: K codes, V codes, K scales, V scales.
        uint8_t* s_k_codes = reinterpret_cast<uint8_t*>( smem_raw + kStageBytes );
        uint8_t* s_v_codes = s_k_codes + Geometry::kCodeStageBytes;
        float* s_k_scale = reinterpret_cast<float*>( s_v_codes + Geometry::kCodeStageBytes );
        float* s_v_scale = s_k_scale + kKeys;

        const auto loadK = [&]( int tile_start )
        {
            if constexpr ( kFp8 )
                load_code_tile( s_k_codes, s_k_scale, static_cast<const uint8_t*>( K_raw ), k_scales,
                    kv_base, scale_base, tile_start, cache_capacity, tid );
            else
                load_tile( s_k, static_cast<const __nv_bfloat16*>( K_raw ), kv_base, tile_start, cache_capacity, tid );
        };

        const auto loadV = [&]( int tile_start )
        {
            if constexpr ( kFp8 )
                load_code_tile( s_v_codes, s_v_scale, static_cast<const uint8_t*>( V_raw ), v_scales,
                    kv_base, scale_base, tile_start, cache_capacity, tid );
            else
                load_tile( s_v, static_cast<const __nv_bfloat16*>( V_raw ), kv_base, tile_start, cache_capacity, tid );
        };

        const auto rowOffset = [&]( int row ) -> std::size_t
        {
            const int position = row / group;
            const int head = kv_head * group + row % group;

            return ( ( static_cast<std::size_t>( b ) * chunk_len + position ) * NH + head ) * kHeadSize;
        };

        // Q^T B fragments of the register steps: this lane's query row g, dims 2tg and 2tg + 8 of each k-step.
        const int query_row = warp_row0 + g;
        const bool query_active = query_row < total_rows;
        const std::size_t query_offset = query_active ? rowOffset( query_row ) : 0;

        uint32_t q[ Geometry::kRegisterSteps ][ 2 ];
#pragma unroll
        for ( int step = 0; step < Geometry::kRegisterSteps; ++step )
        {
            const uint32_t* pair = reinterpret_cast<const uint32_t*>( Q + query_offset + step * 16 + 2 * tg );

            q[ step ][ 0 ] = query_active ? pair[ 0 ] : 0u;
            q[ step ][ 1 ] = query_active ? pair[ 4 ] : 0u;
        }

        if constexpr ( Geometry::kSharedQueryDims > 0 )
        {
            // The warp's rows, dims past the register steps; only this warp reads them. Rows past the chunk are zero.
#pragma unroll
            for ( int i = 0; i < kWarpRows * Geometry::kSharedQueryDims / 8 / 32; ++i )
            {
                const int chunk = lane + i * 32;
                const int r = chunk / ( Geometry::kSharedQueryDims / 8 );
                const int column = ( chunk % ( Geometry::kSharedQueryDims / 8 ) ) * 8;
                const int row = warp_row0 + r;
                uint4 value = make_uint4( 0, 0, 0, 0 );

                if ( row < total_rows )
                    value = *reinterpret_cast<const uint4*>( Q + rowOffset( row ) + Geometry::kRegisterSteps * 16 + column );

                *reinterpret_cast<uint4*>( s_q + ( warp * kWarpRows + r ) * Geometry::kQueryPad + column ) = value;
            }

            __syncwarp();
        }

        // The score columns this lane holds: rows 2tg and 2tg + 1 of the warp.
        int absolute[ 2 ];
        bool active[ 2 ];

#pragma unroll
        for ( int j = 0; j < 2; ++j )
        {
            const int row = warp_row0 + 2 * tg + j;
            active[ j ] = row < total_rows;
            absolute[ j ] = position_offset + ( active[ j ] ? row / group : 0 );
        }

        // The block's keys run to its last position; tiles wholly at or before its first position need no mask.
        const int last_row = min( block_row0 + kRows, total_rows ) - 1;
        const int first_position = position_offset + block_row0 / group;
        const int last_position = position_offset + last_row / group;
        const int tile_count = last_position / kKeys + 1;
        const int unmasked_tiles = ( first_position + 1 ) / kKeys;

        float o[ kDimTiles ][ 4 ];
#pragma unroll
        for ( int d = 0; d < kDimTiles; ++d )
            o[ d ][ 0 ] = o[ d ][ 1 ] = o[ d ][ 2 ] = o[ d ][ 3 ] = 0.0f;

        float m[ 2 ] = { -CUDART_INF_F, -CUDART_INF_F };
        float l[ 2 ] = { 0.0f, 0.0f };

        // ldmatrix lane addresses. K (A, row-major): matrices (keys 0-7 | 8-15) x (dims 0-7 | 8-15). V (A of V^T,
        // transposed on load): matrices (dims 0-7 | 8-15) x (keys 0-7 | 8-15). Q (B): two k-steps per load.
        const int k_key = ( lane & 7 ) + ( ( lane >> 3 ) & 1 ) * 8;
        const int k_dim = ( lane >> 4 ) * 8;
        const int v_key = ( lane & 7 ) + ( lane >> 4 ) * 8;
        const int v_dim = ( ( lane >> 3 ) & 1 ) * 8;
        const uint32_t q_lane = shared_address( s_q + ( warp * kWarpRows + ( lane & 7 ) ) * Geometry::kQueryPad + ( lane >> 3 ) * 8 );

        loadK( 0 );
        __pipeline_commit();

        for ( int tile = 0; tile < tile_count; ++tile )
        {
            const int tile_start = tile * kKeys;

            __pipeline_wait_prior( 0 );

            // K(tile) visible to every warp; and, every warp having finished PV(tile - 1), the V stage is free. For FP8
            // the V codes and scales are free too, so V's load is in flight through K's widening.
            __syncthreads();

            loadV( tile_start );
            __pipeline_commit();

            if constexpr ( kFp8 )
            {
                // The BF16 K stage is free too: every warp finished QK(tile - 1) before the previous PV barrier.
                widen_tile( s_k, s_k_codes, tid );

                // The widened K visible before the MMAs read it.
                __syncthreads();
            }

            // --- S^T = K Q^T over the whole head ---
            float s[ kKeyTiles ][ 4 ];
#pragma unroll
            for ( int kt = 0; kt < kKeyTiles; ++kt )
                s[ kt ][ 0 ] = s[ kt ][ 1 ] = s[ kt ][ 2 ] = s[ kt ][ 3 ] = 0.0f;

#pragma unroll
            for ( int step = 0; step < Geometry::kRegisterSteps; ++step )
            {
#pragma unroll
                for ( int kt = 0; kt < kKeyTiles; ++kt )
                {
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4( a0, a1, a2, a3, shared_address( &s_k[ ( kt * 16 + k_key ) * kPad + step * 16 + k_dim ] ) );
                    mma_m16n8k16_bf16( s[ kt ], a0, a1, a2, a3, q[ step ][ 0 ], q[ step ][ 1 ] );
                }
            }

#pragma unroll
            for ( int step = Geometry::kRegisterSteps; step < kSteps; step += 2 )
            {
                uint32_t b0, b1, b2, b3;
                ldmatrix_x4( b0, b1, b2, b3, q_lane + ( step - Geometry::kRegisterSteps ) * 16 * sizeof( __nv_bfloat16 ) );

#pragma unroll
                for ( int half = 0; half < 2; ++half )
                {
#pragma unroll
                    for ( int kt = 0; kt < kKeyTiles; ++kt )
                    {
                        uint32_t a0, a1, a2, a3;
                        ldmatrix_x4( a0, a1, a2, a3,
                            shared_address( &s_k[ ( kt * 16 + k_key ) * kPad + ( step + half ) * 16 + k_dim ] ) );
                        mma_m16n8k16_bf16( s[ kt ], a0, a1, a2, a3, half ? b2 : b0, half ? b3 : b1 );
                    }
                }
            }

            // --- online softmax per row; this lane holds keys kt*16 + g and kt*16 + g + 8 of rows 2tg, 2tg + 1 ---
            const bool masked = tile >= unmasked_tiles;
            float tile_max[ 2 ] = { -CUDART_INF_F, -CUDART_INF_F };

#pragma unroll
            for ( int kt = 0; kt < kKeyTiles; ++kt )
            {
#pragma unroll
                for ( int e = 0; e < 4; ++e )
                {
                    const int j = e & 1;
                    const int key_in_tile = kt * 16 + g + ( e >> 1 ) * 8;
                    const float key_scale = kFp8 ? scale * s_k_scale[ key_in_tile ] : scale;
                    float score = s[ kt ][ e ] * key_scale;

                    if ( masked && ( !active[ j ] || tile_start + key_in_tile > absolute[ j ] ) )
                        score = -CUDART_INF_F;

                    s[ kt ][ e ] = score;
                    tile_max[ j ] = fmaxf( tile_max[ j ], score );
                }
            }

            // A row's keys span the eight g lanes.
#pragma unroll
            for ( int offset = 4; offset <= 16; offset <<= 1 )
            {
                tile_max[ 0 ] = fmaxf( tile_max[ 0 ], __shfl_xor_sync( 0xffffffffu, tile_max[ 0 ], offset ) );
                tile_max[ 1 ] = fmaxf( tile_max[ 1 ], __shfl_xor_sync( 0xffffffffu, tile_max[ 1 ], offset ) );
            }

            // A fully masked tile keeps the row's state: alpha 1, and the exps below see only -inf.
            float alpha[ 2 ], new_m[ 2 ];

#pragma unroll
            for ( int j = 0; j < 2; ++j )
            {
                const bool scored = tile_max[ j ] > -CUDART_INF_F;
                new_m[ j ] = scored ? fmaxf( m[ j ], tile_max[ j ] ) : m[ j ];
                alpha[ j ] = !scored ? 1.0f : ( m[ j ] == -CUDART_INF_F ? 0.0f : __expf( m[ j ] - new_m[ j ] ) );
            }

            float sum[ 2 ] = { 0.0f, 0.0f };

#pragma unroll
            for ( int kt = 0; kt < kKeyTiles; ++kt )
            {
#pragma unroll
                for ( int e = 0; e < 4; ++e )
                {
                    const int j = e & 1;
                    s[ kt ][ e ] = s[ kt ][ e ] == -CUDART_INF_F ? 0.0f : __expf( s[ kt ][ e ] - new_m[ j ] );
                    sum[ j ] += s[ kt ][ e ];
                }
            }

#pragma unroll
            for ( int offset = 4; offset <= 16; offset <<= 1 )
            {
                sum[ 0 ] += __shfl_xor_sync( 0xffffffffu, sum[ 0 ], offset );
                sum[ 1 ] += __shfl_xor_sync( 0xffffffffu, sum[ 1 ], offset );
            }

#pragma unroll
            for ( int j = 0; j < 2; ++j )
            {
                l[ j ] = l[ j ] * alpha[ j ] + sum[ j ];
                m[ j ] = new_m[ j ];
            }

#pragma unroll
            for ( int d = 0; d < kDimTiles; ++d )
            {
                o[ d ][ 0 ] *= alpha[ 0 ];
                o[ d ][ 1 ] *= alpha[ 1 ];
                o[ d ][ 2 ] *= alpha[ 0 ];
                o[ d ][ 3 ] *= alpha[ 1 ];
            }

            __pipeline_wait_prior( 0 );

            if constexpr ( !kFp8 )
            {
                // This thread's own chunks of V(tile) landed; the barrier below publishes them as FP16.
                narrow_value_tile_to_half( s_v, tid );
            }

            // V(tile) visible to every warp; and, every warp having finished QK(tile), the K stage is free. For FP8
            // the K codes were widened before QK and its scales read in the softmax, so the next K's load is in flight
            // through V's widening.
            __syncthreads();

            if ( tile + 1 < tile_count )
            {
                loadK( tile_start + kKeys );
                __pipeline_commit();
            }

            // P^T as FP16 B fragments.
            uint32_t p[ kKeyTiles ][ 2 ];
#pragma unroll
            for ( int kt = 0; kt < kKeyTiles; ++kt )
            {
                p[ kt ][ 0 ] = movmatrix_trans( pack_half2( s[ kt ][ 0 ], s[ kt ][ 1 ] ) );
                p[ kt ][ 1 ] = movmatrix_trans( pack_half2( s[ kt ][ 2 ], s[ kt ][ 3 ] ) );
            }

            if constexpr ( kFp8 )
            {
                // The V stage is free: every warp finished PV(tile - 1) before this tile's first barrier. V's scales
                // came from other threads, published by the barrier above.
                widen_value_tile( reinterpret_cast<__half*>( s_v ), s_v_codes, s_v_scale, tid );

                // The widened V visible before the MMAs read it.
                __syncthreads();
            }

            // --- O^T += V^T P^T: the tile's 32 keys summed in FP16, then added to FP32 O ---
#pragma unroll
            for ( int d = 0; d < kDimTiles; ++d )
            {
                uint32_t tile_sum[ 2 ] = { 0u, 0u };

#pragma unroll
                for ( int kt = 0; kt < kKeyTiles; ++kt )
                {
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4_trans( a0, a1, a2, a3, shared_address( &s_v[ ( kt * 16 + v_key ) * kPad + d * 16 + v_dim ] ) );
                    mma_m16n8k16_f16( tile_sum, a0, a1, a2, a3, p[ kt ][ 0 ], p[ kt ][ 1 ] );
                }

                const float2 low = __half22float2( *reinterpret_cast<const __half2*>( &tile_sum[ 0 ] ) );
                const float2 high = __half22float2( *reinterpret_cast<const __half2*>( &tile_sum[ 1 ] ) );

                o[ d ][ 0 ] += low.x;
                o[ d ][ 1 ] += low.y;
                o[ d ][ 2 ] += high.x;
                o[ d ][ 3 ] += high.y;
            }
        }

        // --- normalize, then stage the warp's rows through the free K and V stages for 16-byte stores ---
        // Every warp has finished its last PV before any overwrites a stage.
        __syncthreads();

        __nv_bfloat16* s_out = s_k + warp * kWarpRows * kPad;
        const float inverse[ 2 ] = { active[ 0 ] ? 1.0f / l[ 0 ] : 0.0f, active[ 1 ] ? 1.0f / l[ 1 ] : 0.0f };

#pragma unroll
        for ( int d = 0; d < kDimTiles; ++d )
        {
#pragma unroll
            for ( int e = 0; e < 4; ++e )
            {
                const int j = e & 1;
                const int dim = d * 16 + g + ( e >> 1 ) * 8;
                s_out[ ( 2 * tg + j ) * kPad + dim ] = __float2bfloat16_rn( o[ d ][ e ] * inverse[ j ] );
            }
        }

        __syncwarp();

#pragma unroll
        for ( int i = 0; i < kWarpRows * kHeadSize / 8 / 32; ++i )
        {
            const int chunk = lane + i * 32;
            const int r = chunk / ( kHeadSize / 8 );
            const int column = ( chunk % ( kHeadSize / 8 ) ) * 8;
            const int row = warp_row0 + r;

            if ( row < total_rows )
                *reinterpret_cast<uint4*>( Y + rowOffset( row ) + column ) = *reinterpret_cast<const uint4*>( s_out + r * kPad + column );
        }
#endif // __CUDA_ARCH__ >= 800
    }

    namespace
    {
        template<bool kFp8>
        void launch_wide_head_prefill(
            const __nv_bfloat16* Q, const void* K, const void* V, const float* k_scales, const float* v_scales,
            __nv_bfloat16* Y, int B, int chunk_len, int NH, int NKV, int cache_capacity,
            int position_offset, float scale, cudaStream_t stream )
        {
            constexpr std::size_t kSharedBytes = WideHeadGeometry<kFp8>::kSharedBytes;

            cudaCheck( cudaFuncSetAttribute(
                gqa_flash_prefill_wide_head_kernel<kFp8>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>( kSharedBytes ) ) );

            const int total_rows = chunk_len * ( NH / NKV );
            const dim3 grid( ceil_div( total_rows, kRows ), NKV, B );

            gqa_flash_prefill_wide_head_kernel<kFp8> <<< grid, kWarps * 32, kSharedBytes, stream >>> (
                Q, K, V, k_scales, v_scales, Y, chunk_len, NH, NKV, cache_capacity, position_offset, scale );

            cudaCheck( cudaGetLastError() );
        }
    }

    void cuda_gqa_flash_prefill_wide_head_bf16(
        const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
        __nv_bfloat16* Y,
        int B, int chunk_len, int NH, int NKV, int cache_capacity,
        int position_offset, float scale,
        cudaStream_t stream )
    {
        launch_wide_head_prefill<false>( Q, K, V, nullptr, nullptr, Y, B, chunk_len, NH, NKV, cache_capacity,
            position_offset, scale, stream );
    }

    void cuda_gqa_flash_prefill_wide_head_fp8(
        const __nv_bfloat16* Q, const __nv_fp8_e4m3* K, const __nv_fp8_e4m3* V,
        const float* k_scales, const float* v_scales,
        __nv_bfloat16* Y,
        int B, int chunk_len, int NH, int NKV, int cache_capacity,
        int position_offset, float scale,
        cudaStream_t stream )
    {
        launch_wide_head_prefill<true>( Q, K, V, k_scales, v_scales, Y, B, chunk_len, NH, NKV, cache_capacity,
            position_offset, scale, stream );
    }
}
