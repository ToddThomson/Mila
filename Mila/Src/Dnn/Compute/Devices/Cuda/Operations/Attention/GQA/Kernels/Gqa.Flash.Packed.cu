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

        template<int kHeadSize>
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

            static constexpr std::size_t kExchangeBytes = kSplit > 1
                ? static_cast<std::size_t>( kRowTiles ) * kSplit * kTileRows * kExchangeStride * sizeof( float )
                : 0;

            // Exchange floats first (a multiple of 16 bytes), then K stages 0-1 and V stages 0-1.
            static constexpr std::size_t kSharedBytes = kExchangeBytes + 4 * kStageElements * sizeof( __nv_bfloat16 );
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

        // One key tile of K and V ([keys x HS] each) into a stage. Keys past the cache clamp to its
        // last row; every such column is causally masked.
        template<int kHeadSize, int kKeys>
        __device__ __forceinline__ void load_kv_tile(
            __nv_bfloat16* k_stage, __nv_bfloat16* v_stage,
            const __nv_bfloat16* K, const __nv_bfloat16* V,
            std::size_t kv_base, int tile_start, int cache_capacity, int tid )
        {
            constexpr int kChunksPerRow = kHeadSize / kCopyElements;
            constexpr int kPad = kHeadSize + kSkew;

            for ( int chunk = tid; chunk < kKeys * kChunksPerRow; chunk += kWarps * 32 )
            {
                const int key = chunk / kChunksPerRow;
                const int column = ( chunk % kChunksPerRow ) * kCopyElements;
                const int position = tile_start + key;
                const int row = position < cache_capacity ? position : cache_capacity - 1;
                const std::size_t source = kv_base + static_cast<std::size_t>( row ) * kHeadSize + column;

                __pipeline_memcpy_async( k_stage + key * kPad + column, K + source, 16 );
                __pipeline_memcpy_async( v_stage + key * kPad + column, V + source, 16 );
            }
        }
#endif
    }

    template<int kHeadSize>
    __global__ void __launch_bounds__( kWarps * 32, 1 )
    gqa_flash_prefill_packed_bf16_kernel(
        const __nv_bfloat16* __restrict__ Q,   // [B, chunk_len, NH * HS]
        const __nv_bfloat16* __restrict__ K,   // [B, NKV, cache_capacity, HS]
        const __nv_bfloat16* __restrict__ V,   // [B, NKV, cache_capacity, HS]
        __nv_bfloat16* __restrict__ Y,         // [B, chunk_len, NH * HS]
        int chunk_len, int NH, int NKV, int cache_capacity,
        int position_offset, int window, float scale )
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        using Geometry = PackedGeometry<kHeadSize>;

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
        __nv_bfloat16* s_v = s_k + 2 * Geometry::kStageElements;

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

        load_kv_tile<kHeadSize, Geometry::kKeys>(
            s_k + ( first_tile & 1 ) * Geometry::kStageElements, s_v + ( first_tile & 1 ) * Geometry::kStageElements,
            K, V, kv_base, first_tile * Geometry::kKeys, cache_capacity, tid );
        __pipeline_commit();

        for ( int tile = first_tile; tile < tile_count; ++tile )
        {
            const int tile_start = tile * Geometry::kKeys;
            const __nv_bfloat16* k_tile = s_k + ( tile & 1 ) * Geometry::kStageElements;
            const __nv_bfloat16* v_tile = s_v + ( tile & 1 ) * Geometry::kStageElements;

            __pipeline_wait_prior( 0 );

            // This tile visible to every warp; and, every thread having finished the previous
            // tile, the stage the next prefetch writes and the exchange buffer are free.
            __syncthreads();

            if ( tile + 1 < tile_count )
            {
                const int next = ( tile + 1 ) & 1;
                load_kv_tile<kHeadSize, Geometry::kKeys>(
                    s_k + next * Geometry::kStageElements, s_v + next * Geometry::kStageElements,
                    K, V, kv_base, ( tile + 1 ) * Geometry::kKeys, cache_capacity, tid );
                __pipeline_commit();
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

                    float low = s[ nt ][ j ] * scale;
                    if ( !active_low || key > absolute_low || key < window_low )
                        low = -CUDART_INF_F;

                    float high = s[ nt ][ 2 + j ] * scale;
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
                const uint32_t p0 = pack_bf16x2( s[ 2 * step ][ 0 ], s[ 2 * step ][ 1 ] );
                const uint32_t p1 = pack_bf16x2( s[ 2 * step ][ 2 ], s[ 2 * step ][ 3 ] );
                const uint32_t p2 = pack_bf16x2( s[ 2 * step + 1 ][ 0 ], s[ 2 * step + 1 ][ 1 ] );
                const uint32_t p3 = pack_bf16x2( s[ 2 * step + 1 ][ 2 ], s[ 2 * step + 1 ][ 3 ] );

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
        template<int kHeadSize>
        void launch_packed_prefill(
            const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, __nv_bfloat16* Y,
            int B, int chunk_len, int NH, int NKV, int cache_capacity,
            int position_offset, int window, float scale, cudaStream_t stream )
        {
            using Geometry = PackedGeometry<kHeadSize>;

            cudaCheck( cudaFuncSetAttribute(
                gqa_flash_prefill_packed_bf16_kernel<kHeadSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>( Geometry::kSharedBytes ) ) );

            const int total_rows = chunk_len * ( NH / NKV );
            const dim3 grid( ceil_div( total_rows, Geometry::kRows ), NKV, B );

            gqa_flash_prefill_packed_bf16_kernel<kHeadSize> <<< grid, kWarps * 32, Geometry::kSharedBytes, stream >>> (
                Q, K, V, Y, chunk_len, NH, NKV, cache_capacity, position_offset, window, scale );

            cudaCheck( cudaGetLastError() );
        }
    }

    void cuda_gqa_flash_prefill_bf16(
        const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
        __nv_bfloat16* Y,
        int B, int chunk_len, int NH, int NKV, int HS, int cache_capacity,
        int position_offset, int window, float scale,
        cudaStream_t stream )
    {
        if ( NKV <= 0 || NH % NKV != 0 )
            throw std::runtime_error( "cuda_gqa_flash_prefill_bf16: query heads must be a multiple of KV heads" );

        int device = 0;
        int sm_major = 0;
        cudaCheck( cudaGetDevice( &device ) );
        cudaCheck( cudaDeviceGetAttribute( &sm_major, cudaDevAttrComputeCapabilityMajor, device ) );

        if ( sm_major < 8 )
            throw std::runtime_error( "cuda_gqa_flash_prefill_bf16: requires compute capability 8.0 or later" );

        switch ( HS )
        {
            case 128:
                launch_packed_prefill<128>( Q, K, V, Y, B, chunk_len, NH, NKV, cache_capacity, position_offset, window, scale, stream );
                break;

            case 256:
                launch_packed_prefill<256>( Q, K, V, Y, B, chunk_len, NH, NKV, cache_capacity, position_offset, window, scale, stream );
                break;

            case 512:
                launch_packed_prefill<512>( Q, K, V, Y, B, chunk_len, NH, NKV, cache_capacity, position_offset, window, scale, stream );
                break;

            default:
                throw std::runtime_error( "cuda_gqa_flash_prefill_bf16: head size " + std::to_string( HS )
                    + " is not supported; the packed kernel serves 128, 256 and 512" );
        }
    }
}
