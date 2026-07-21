// Gqa.Flash.Fa2.cu
//
// FlashAttention-2 (row-split) prefill for the Gemma 4 LOCAL sliding-window layers.
// Head dim 256 -- the ONE regime where classic FA-2 is expressible on Ada. The global
// layers (head dim 512) stay on the HS-split kernel (Gqa.Flash.Wmma.cu): at 512 the
// register-resident O accumulator is 256 f32/thread and spills, which is precisely why
// that kernel splits the head dimension across warps instead of the query rows.
//
// WHY A SEPARATE KERNEL. The HS-split kernel is latency-bound at ~2 warps/scheduler:
// per key tile it crosses three block-wide __syncthreads, and its P (softmax output)
// crosses warps through shared memory because every warp owns HS-columns of ALL rows.
// ncu on the +103 ring kernel: No-Eligible 66.8%, top stall fixed-latency execution
// dependency 40.5% -- the warps lock-step through the barriers. FA-2 attacks the
// dependency graph directly: each warp owns its OWN 16-row query tile, keeps O and the
// softmax state in registers, and needs only its own P -- so P is per-warp (a tiny
// shared scratch read back with ldmatrix under __syncwarp, no block barrier, no split-K
// reduction). Only the cooperative K/V tile load is block-wide: ONE __syncthreads per
// tile instead of three, and long independent per-warp stretches between them.
//
// LAYOUT. A block owns kFa2WarpsPerBlock consecutive query tiles (one per warp) for one
// (batch, query-head). All warps march the same key-tile loop over the UNION band of
// their rows (window >> block rows, so the bands overlap almost entirely) and share the
// double-buffered K/V smem tiles; each warp masks keys outside its own causal/window
// range. Q for every tile is resident in smem (register-resident Q would join the
// 128-reg O and spill), reloaded per k-step with ldmatrix.
//
// m16n8k16 .f32.bf16.bf16.f32 fragment layout (PTX ISA), g = lane/4, tg = lane%4:
//   A[16x16] a0..a3 (row-major): a0={A[g][2t],A[g][2t+1]} a1={A[g+8][2t],..}
//                                a2={A[g][2t+8],..}       a3={A[g+8][2t+8],..}
//   B[16x8]  b0,b1  (col-major): b0={B[2t][g],B[2t+1][g]} b1={B[2t+8][g],B[2t+9][g]}
//   C[16x8]  c0..c3 (row-major): c0=C[g][2t] c1=C[g][2t+1] c2=C[g+8][2t] c3=C[g+8][2t+1]
//
// Math, masking, and the online-softmax state are equivalent to the HS-split kernel, so
// CudaGqaFlashRingPrefillParity (HS=256, NKV=8, ragged 83-token context, window wrap +
// no-wrap) gates this rewrite -- it takes over the cuda_gqa_flash_prefill_ring_bf16
// symbol, so no cuh/Dispatch/Op change and the oracle exercises it unchanged.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>
#include <stdexcept>
#include <cuda_pipeline.h>
#include "CudaUtils.h"
#include "CudaGqa.cuh"

namespace Mila::Dnn::Compute::Cuda::Gqa
{
    namespace
    {
        constexpr int kFa2Rows = 16;          // Br: query rows per warp tile (MMA M)
        constexpr int kFa2Keys = 16;          // Bc: key positions per tile
        constexpr int kMmaK = 16;             // contraction step (m16n8k16 K)
        constexpr int kMmaN = 8;              // mma.sync N tile (m16n8k16)
        constexpr int kCopyElems = 8;         // bf16 elements per 16-byte cp.async copy
        constexpr int kFa2KeyNTiles = kFa2Keys / kMmaN;   // QK score n-tiles (Bc == 16 -> 2)

        // The QK block issues exactly two mma calls (one per key n-tile); a different Bc
        // would need it generalized. This also references kFa2KeyNTiles in the host
        // compilation pass, where the device-only kernel body is elided (else #177-D).
        static_assert( kFa2KeyNTiles == 2, "QK block assumes Bc == 16" );

        // Row padding (bf16) for the Q/K/V smem tiles: an unpadded HS=256 row strides
        // 512 B = a multiple of the 128-B bank cycle, mapping every row onto the same 32
        // banks. +8 shifts consecutive rows 4 banks apart while keeping each row 16-B
        // aligned for cp.async. Storage stride only; the math is unchanged.
        constexpr int kFa2Skew = 8;

        static_assert( kFa2Skew % kCopyElems == 0,
            "kFa2Skew must be a multiple of the cp.async copy width" );

        // Register O accumulator ceiling: HS / kMmaN n-tiles. HS = 256 -> 32 n-tiles ->
        // o_acc[32][4] = 128 f32/thread (spills past Ada's 255-reg cap, but the spill is
        // L1-resident and cheaper than the recompute a two-pass split costs -- measured).
        // The launcher rejects HS above this bound.
        constexpr int kFa2MaxNTiles = 32;

        // Query tiles processed per block (one per warp). The kernel has ONE block-wide
        // barrier per key tile (the cooperative K/V load) with fully independent per-warp
        // compute between them, so more warps per block raise warps/SM directly (the block
        // stays 1/SM, smem-limited) and improve latency hiding at ~2 warps/scheduler --
        // unlike the HS-split kernel, where extra warps just contend on its three barriers.
        // Six warps land the block at ~87 KB smem (still 1 block/SM) -> 6 warps/SM. Each
        // added warp also widens the union band a warp masks, so this is a tuning knob.
        constexpr int kFa2WarpsPerBlock = 6;

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

        // Generic -> shared-state-space conversion via explicit PTX (the CUTLASS
        // cast_smem_ptr_to_uint form). The .shared qualifier on ldmatrix below is
        // load-bearing for the same reason -- see Gqa.Flash.Wmma.cu.
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

        // Prefetch one key tile's K and V ([Bc x HS] BF16 each) into the given smem stage
        // via 16-byte cp.async copies, addressed by ABSOLUTE key position with the ring
        // cache-row mapping row = abs_pos % cache_capacity. Aliased slots land only on
        // columns the causal/window mask forces to -inf. Caller commits + waits.
        __device__ __forceinline__ void cp_async_kv_tile_ring(
            __nv_bfloat16* s_K_stage, __nv_bfloat16* s_V_stage,
            const __nv_bfloat16* K, const __nv_bfloat16* V,
            size_t kv_head_base, int tile_start, int HS, int hs_pad, int cache_capacity,
            int tid, int block_threads )
        {
            const int chunks_per_row = HS / kCopyElems;
            const int num_chunks = kFa2Keys * chunks_per_row;

            for ( int c = tid; c < num_chunks; c += block_threads )
            {
                const int p_local = c / chunks_per_row;
                const int d0 = ( c % chunks_per_row ) * kCopyElems;
                const int p = tile_start + p_local;
                const int p_src = p % cache_capacity;

                const size_t g_off = kv_head_base + static_cast<size_t>( p_src ) * HS + d0;
                const int s_off = p_local * hs_pad + d0;

                __pipeline_memcpy_async( s_K_stage + s_off, K + g_off, 16 );
                __pipeline_memcpy_async( s_V_stage + s_off, V + g_off, 16 );
            }
        }
#endif

        // Shared-memory footprint for one block, matching the kernel's carve exactly:
        // per-warp Q tiles, double-buffered K and V, and per-warp P scratch (no split-K
        // partials and no softmax state in smem -- both are register-resident).
        static size_t flash_fa2_smem_bytes( int HS, int warps )
        {
            const size_t hs_pad = static_cast<size_t>( HS + kFa2Skew );
            const size_t bf16_elems =
                static_cast<size_t>( warps * kFa2Rows ) * hs_pad          // Q (one tile/warp)
                + static_cast<size_t>( 2 * kFa2Keys ) * hs_pad            // K (2 stages)
                + static_cast<size_t>( 2 * kFa2Keys ) * hs_pad            // V (2 stages)
                + static_cast<size_t>( warps * kFa2Rows * kFa2Keys );     // P (one tile/warp)
            return bf16_elems * sizeof( __nv_bfloat16 );
        }
    }

    // Row-split FA-2: warp w of the block owns query tile (blockIdx.x * warps + w). Each
    // warp keeps its full O[16 x HS] accumulator and online-softmax state (m, l per row)
    // in registers and reads its own P back through a per-warp smem scratch. The bounded
    // sliding-window ring cache-row mapping lives only in the cp.async source math.
    // The block is smem-limited to one per SM, so pin minBlocks = 1: this lets the compiler
    // spend registers freely rather than cap them for an occupancy it cannot reach.
    __global__ void __launch_bounds__( kFa2WarpsPerBlock * 32, 1 )
    gqa_flash_prefill_fa2_bf16_kernel(
        const __nv_bfloat16* __restrict__ Q,   // [B, chunk_len, NH * HS]
        const __nv_bfloat16* __restrict__ K,   // [B, NKV, cache_capacity, HS]
        const __nv_bfloat16* __restrict__ V,   // [B, NKV, cache_capacity, HS]
        __nv_bfloat16* __restrict__ Y,         // [B, chunk_len, NH * HS]
        int B, int chunk_len, int NH, int NKV, int HS, int cache_capacity,
        int position_offset, int window, float scale )
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        const int tid = threadIdx.x;
        const int warp_id = tid >> 5;          // which query tile within the block
        const int lane = tid & 31;
        const int warps = blockDim.x >> 5;
        const int block_threads = blockDim.x;

        const int g = lane >> 2;               // mma row group (owns rows g, g+8)
        const int tg = lane & 3;               // mma thread-in-group (owns cols 2*tg, +1)

        const int hs_pad = HS + kFa2Skew;
        const int num_ntiles = HS / kMmaN;     // 32 at HS = 256

        const int nh = blockIdx.y;
        const int b = blockIdx.z;

        const int block_row0 = blockIdx.x * ( warps * kFa2Rows );
        if ( block_row0 >= chunk_len )
            return;

        const int qrow0 = block_row0 + warp_id * kFa2Rows;   // this warp's tile first row
        const bool warp_active = ( qrow0 < chunk_len );

        const int group_size = NH / NKV;
        const int head_kv = nh / group_size;
        const size_t kv_head_base =
            static_cast<size_t>( b * NKV + head_kv ) * cache_capacity * HS;

        // --- shared-memory layout (must match flash_fa2_smem_bytes) ---
        extern __shared__ char smem_raw[];
        __nv_bfloat16* s_Q = reinterpret_cast<__nv_bfloat16*>( smem_raw );  // [warps][Br * hs_pad]
        __nv_bfloat16* s_K = s_Q + warps * kFa2Rows * hs_pad;              // [2][Bc * hs_pad]
        __nv_bfloat16* s_V = s_K + 2 * kFa2Keys * hs_pad;                  // [2][Bc * hs_pad]
        __nv_bfloat16* s_P = s_V + 2 * kFa2Keys * hs_pad;                  // [warps][Br * Bc]

        __nv_bfloat16* s_Q_w = s_Q + warp_id * kFa2Rows * hs_pad;
        __nv_bfloat16* s_P_w = s_P + warp_id * kFa2Rows * kFa2Keys;

        const int stage_elems = kFa2Keys * hs_pad;

        // --- load every warp's Q tile into smem once (shared by both output passes);
        //     zero rows past chunk_len ---
        for ( int idx = tid; idx < warps * kFa2Rows * HS; idx += block_threads )
        {
            const int w = idx / ( kFa2Rows * HS );
            const int rem = idx - w * ( kFa2Rows * HS );
            const int m = rem / HS;
            const int d = rem - m * HS;
            const int t = block_row0 + w * kFa2Rows + m;

            __nv_bfloat16* dst = s_Q + w * kFa2Rows * hs_pad + m * hs_pad + d;

            if ( t < chunk_len )
            {
                const size_t row_base = ( static_cast<size_t>( b * chunk_len + t ) * NH + nh ) * HS;
                *dst = Q[ row_base + d ];
            }
            else
                *dst = __float2bfloat16( 0.0f );
        }

        // --- union band over the block's active rows: all warps march the same key loop ---
        const int block_row_last = min( chunk_len - 1, block_row0 + warps * kFa2Rows - 1 );
        const int band_start =
            ( window > 0 ) ? max( 0, position_offset + block_row0 - window + 1 ) : 0;
        const int first_tile = band_start / kFa2Keys;
        const int block_max_key = position_offset + block_row_last;
        const int num_tiles = block_max_key / kFa2Keys + 1;

        // this warp's own causal/window limits for its two accumulator rows
        const int abs_g = position_offset + qrow0 + g;
        const int abs_h = position_offset + qrow0 + g + 8;
        const int wstart_g = ( window > 0 ) ? max( 0, abs_g - window + 1 ) : 0;
        const int wstart_h = ( window > 0 ) ? max( 0, abs_h - window + 1 ) : 0;
        const bool row_g_active = warp_active && ( qrow0 + g < chunk_len );
        const bool row_h_active = warp_active && ( qrow0 + g + 8 < chunk_len );

        // --- register O accumulator and online-softmax state (rows g, g+8) ---
        float o_acc[ kFa2MaxNTiles ][ 4 ];
#pragma unroll
        for ( int nt = 0; nt < kFa2MaxNTiles; ++nt )
            o_acc[ nt ][ 0 ] = o_acc[ nt ][ 1 ] = o_acc[ nt ][ 2 ] = o_acc[ nt ][ 3 ] = 0.0f;

        float m_g = -CUDART_INF_F, m_h = -CUDART_INF_F;
        float l_g = 0.0f, l_h = 0.0f;

        // --- prologue: prefetch the first band tile into its parity stage ---
        cp_async_kv_tile_ring( s_K + ( first_tile & 1 ) * stage_elems,
            s_V + ( first_tile & 1 ) * stage_elems,
            K, V, kv_head_base, first_tile * kFa2Keys, HS, hs_pad, cache_capacity,
            tid, block_threads );
        __pipeline_commit();

        for ( int t = first_tile; t < num_tiles; ++t )
        {
            const int tile_start = t * kFa2Keys;
            const int stage = t & 1;
            __nv_bfloat16* Kc = s_K + stage * stage_elems;
            __nv_bfloat16* Vc = s_V + stage * stage_elems;

            __pipeline_wait_prior( 0 );

            // The single block-wide barrier per tile: this tile's K/V visible to every
            // warp, and (every thread having drained iteration t-1) the stage the next
            // prefetch overwrites has no remaining readers.
            __syncthreads();

            if ( t + 1 < num_tiles )
            {
                const int next_stage = ( t + 1 ) & 1;
                cp_async_kv_tile_ring( s_K + next_stage * stage_elems, s_V + next_stage * stage_elems,
                    K, V, kv_head_base, ( t + 1 ) * kFa2Keys, HS, hs_pad, cache_capacity,
                    tid, block_threads );
                __pipeline_commit();
            }

            if ( warp_active )
            {
                // --- QK^T over the FULL head dim: this warp's S[16 x 16] in registers ---
                float s_acc[ kFa2KeyNTiles ][ 4 ];
#pragma unroll
                for ( int nt = 0; nt < kFa2KeyNTiles; ++nt )
                    s_acc[ nt ][ 0 ] = s_acc[ nt ][ 1 ] = s_acc[ nt ][ 2 ] = s_acc[ nt ][ 3 ] = 0.0f;

                for ( int k_step = 0; k_step < HS; k_step += kMmaK )
                {
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4( a0, a1, a2, a3, shared_address(
                        &s_Q_w[ ( lane & 15 ) * hs_pad + k_step + ( lane >> 4 ) * kMmaN ] ) );

                    uint32_t b00, b01, b10, b11;
                    {
                        const int key_row = ( lane >> 4 ) * kMmaN + ( lane & 7 );
                        const int dim_half = ( ( lane >> 3 ) & 1 ) * kMmaN;
                        ldmatrix_x4( b00, b01, b10, b11, shared_address(
                            &Kc[ key_row * hs_pad + k_step + dim_half ] ) );
                    }

                    mma_m16n8k16_bf16(
                        s_acc[ 0 ][ 0 ], s_acc[ 0 ][ 1 ], s_acc[ 0 ][ 2 ], s_acc[ 0 ][ 3 ],
                        a0, a1, a2, a3, b00, b01 );
                    mma_m16n8k16_bf16(
                        s_acc[ 1 ][ 0 ], s_acc[ 1 ][ 1 ], s_acc[ 1 ][ 2 ], s_acc[ 1 ][ 3 ],
                        a0, a1, a2, a3, b10, b11 );
                }

                // --- online softmax over this warp's rows g and g+8. The four scores each
                // lane holds for a row span key columns {2tg, 2tg+1, 8+2tg, 8+2tg+1}; the
                // four tg-lanes of a group cover all 16 keys, so the row max/sum reduce
                // with __shfl_xor over the group (offsets 1, 2). m_g/l_g stay replicated
                // and consistent across the group. ---
                const int cols[ 4 ] = { 2 * tg, 2 * tg + 1, 8 + 2 * tg, 8 + 2 * tg + 1 };
                float scr_g[ 4 ] = { s_acc[ 0 ][ 0 ], s_acc[ 0 ][ 1 ], s_acc[ 1 ][ 0 ], s_acc[ 1 ][ 1 ] };
                float scr_h[ 4 ] = { s_acc[ 0 ][ 2 ], s_acc[ 0 ][ 3 ], s_acc[ 1 ][ 2 ], s_acc[ 1 ][ 3 ] };

                float mt_g = -CUDART_INF_F, mt_h = -CUDART_INF_F;
#pragma unroll
                for ( int i = 0; i < 4; ++i )
                {
                    const int key = tile_start + cols[ i ];

                    float sg = scr_g[ i ] * scale;
                    if ( !row_g_active || key > abs_g || key < wstart_g )
                        sg = -CUDART_INF_F;

                    float sh = scr_h[ i ] * scale;
                    if ( !row_h_active || key > abs_h || key < wstart_h )
                        sh = -CUDART_INF_F;

                    scr_g[ i ] = sg; scr_h[ i ] = sh;
                    mt_g = fmaxf( mt_g, sg );
                    mt_h = fmaxf( mt_h, sh );
                }

#pragma unroll
                for ( int off = 2; off >= 1; off >>= 1 )
                {
                    mt_g = fmaxf( mt_g, __shfl_xor_sync( 0xffffffffu, mt_g, off ) );
                    mt_h = fmaxf( mt_h, __shfl_xor_sync( 0xffffffffu, mt_h, off ) );
                }

                const bool hs_g = ( mt_g > -CUDART_INF_F );
                const bool hs_h = ( mt_h > -CUDART_INF_F );
                const float mn_g = hs_g ? fmaxf( m_g, mt_g ) : m_g;
                const float mn_h = hs_h ? fmaxf( m_h, mt_h ) : m_h;

                float ls_g = 0.0f, ls_h = 0.0f;
#pragma unroll
                for ( int i = 0; i < 4; ++i )
                {
                    const float pg = ( scr_g[ i ] == -CUDART_INF_F ) ? 0.0f : __expf( scr_g[ i ] - mn_g );
                    const float ph = ( scr_h[ i ] == -CUDART_INF_F ) ? 0.0f : __expf( scr_h[ i ] - mn_h );
                    s_P_w[ g * kFa2Keys + cols[ i ] ] = __float2bfloat16( pg );
                    s_P_w[ ( g + 8 ) * kFa2Keys + cols[ i ] ] = __float2bfloat16( ph );
                    ls_g += pg;
                    ls_h += ph;
                }

#pragma unroll
                for ( int off = 2; off >= 1; off >>= 1 )
                {
                    ls_g += __shfl_xor_sync( 0xffffffffu, ls_g, off );
                    ls_h += __shfl_xor_sync( 0xffffffffu, ls_h, off );
                }

                // alpha rescales the running O and denominator; on a fully-masked tile the
                // row keeps its state (alpha = 1). m_g/l_g update identically on all four
                // group lanes (mt/ls are reduced; m/l are replicated).
                float alpha_g = 1.0f, alpha_h = 1.0f;
                if ( hs_g )
                {
                    alpha_g = ( m_g == -CUDART_INF_F ) ? 0.0f : __expf( m_g - mn_g );
                    l_g = l_g * alpha_g + ls_g;
                    m_g = mn_g;
                }
                if ( hs_h )
                {
                    alpha_h = ( m_h == -CUDART_INF_F ) ? 0.0f : __expf( m_h - mn_h );
                    l_h = l_h * alpha_h + ls_h;
                    m_h = mn_h;
                }

                // s_P_w writes must be visible before this warp's ldmatrix reads them.
                __syncwarp();

#pragma unroll
                for ( int nt = 0; nt < kFa2MaxNTiles; ++nt )
                {
                    o_acc[ nt ][ 0 ] *= alpha_g;
                    o_acc[ nt ][ 1 ] *= alpha_g;
                    o_acc[ nt ][ 2 ] *= alpha_h;
                    o_acc[ nt ][ 3 ] *= alpha_h;
                }

                // --- load A = P (this warp's own tile) into the mma fragment ---
                uint32_t pa0, pa1, pa2, pa3;
                ldmatrix_x4( pa0, pa1, pa2, pa3, shared_address(
                    &s_P_w[ ( lane & 15 ) * kFa2Keys + ( lane >> 4 ) * kMmaN ] ) );

                // --- PV: O[16 x HS] += P . V over all n-tiles, accumulated in registers.
                // One ldmatrix.x4.trans loads V's B fragments for TWO n-tiles. ---
#pragma unroll
                for ( int nt = 0; nt < kFa2MaxNTiles; nt += 2 )
                {
                    if ( nt >= num_ntiles )
                        continue;

                    uint32_t b00, b01, b10, b11;
                    ldmatrix_x4_trans( b00, b01, b10, b11, shared_address(
                        &Vc[ ( lane & 15 ) * hs_pad + nt * kMmaN + ( lane >> 4 ) * kMmaN ] ) );

                    mma_m16n8k16_bf16(
                        o_acc[ nt ][ 0 ], o_acc[ nt ][ 1 ], o_acc[ nt ][ 2 ], o_acc[ nt ][ 3 ],
                        pa0, pa1, pa2, pa3, b00, b01 );
                    mma_m16n8k16_bf16(
                        o_acc[ nt + 1 ][ 0 ], o_acc[ nt + 1 ][ 1 ], o_acc[ nt + 1 ][ 2 ], o_acc[ nt + 1 ][ 3 ],
                        pa0, pa1, pa2, pa3, b10, b11 );
                }

                // The next iteration overwrites s_P_w only after its QK + softmax, which
                // follows the top-of-loop __syncthreads -- so this tile's ldmatrix read is
                // ordered before that write without an extra barrier here.
            }
        }

        // --- normalize the register accumulator by 1/l and write out (active rows only) ---
        if ( warp_active )
        {
            const int t_lo = qrow0 + g;
            const int t_hi = qrow0 + g + 8;
            const float inv_lo = ( t_lo < chunk_len ) ? ( 1.0f / l_g ) : 0.0f;
            const float inv_hi = ( t_hi < chunk_len ) ? ( 1.0f / l_h ) : 0.0f;

#pragma unroll
            for ( int nt = 0; nt < kFa2MaxNTiles; ++nt )
            {
                if ( nt >= num_ntiles )
                    continue;

                const int c_base = nt * kMmaN + 2 * tg;

                if ( t_lo < chunk_len )
                {
                    const size_t rb = ( static_cast<size_t>( b * chunk_len + t_lo ) * NH + nh ) * HS;
                    Y[ rb + c_base     ] = __float2bfloat16( o_acc[ nt ][ 0 ] * inv_lo );
                    Y[ rb + c_base + 1 ] = __float2bfloat16( o_acc[ nt ][ 1 ] * inv_lo );
                }

                if ( t_hi < chunk_len )
                {
                    const size_t rb = ( static_cast<size_t>( b * chunk_len + t_hi ) * NH + nh ) * HS;
                    Y[ rb + c_base     ] = __float2bfloat16( o_acc[ nt ][ 2 ] * inv_hi );
                    Y[ rb + c_base + 1 ] = __float2bfloat16( o_acc[ nt ][ 3 ] * inv_hi );
                }
            }
        }
#endif // __CUDA_ARCH__ >= 800
    }

    void cuda_gqa_flash_prefill_ring_bf16(
        const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
        __nv_bfloat16* Y,
        int B, int chunk_len, int NH, int NKV, int HS, int cache_capacity,
        int position_offset, int window, float scale,
        cudaStream_t stream )
    {
        if ( HS % kMmaK != 0 )
            throw std::runtime_error(
                "cuda_gqa_flash_prefill_ring_bf16 (FA-2): head_size must be a multiple of 16" );

        if ( HS % kCopyElems != 0 )
            throw std::runtime_error(
                "cuda_gqa_flash_prefill_ring_bf16 (FA-2): head_size must be a multiple of 8 for cp.async" );

        if ( HS > kFa2MaxNTiles * kMmaN )
            throw std::runtime_error(
                "cuda_gqa_flash_prefill_ring_bf16 (FA-2): head_size exceeds the register accumulator bound (256)" );

        if ( window <= 0 )
            throw std::runtime_error(
                "cuda_gqa_flash_prefill_ring_bf16 (FA-2): bounded ring requires a positive window" );

        int device = 0;
        int sm_major = 0;
        cudaCheck( cudaGetDevice( &device ) );
        cudaCheck( cudaDeviceGetAttribute( &sm_major, cudaDevAttrComputeCapabilityMajor, device ) );

        if ( sm_major < 8 )
            throw std::runtime_error(
                "cuda_gqa_flash_prefill_ring_bf16 (FA-2): requires compute capability >= 8.0" );

        const int warps = kFa2WarpsPerBlock;
        const size_t smem_bytes = flash_fa2_smem_bytes( HS, warps );

        cudaCheck( cudaFuncSetAttribute(
            gqa_flash_prefill_fa2_bf16_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>( smem_bytes ) ) );

        const int rows_per_block = warps * kFa2Rows;
        const int num_blocks_x = ceil_div( chunk_len, rows_per_block );
        dim3 grid( num_blocks_x, NH, B );
        dim3 block( warps * 32, 1, 1 );

        gqa_flash_prefill_fa2_bf16_kernel <<< grid, block, smem_bytes, stream >>> (
            Q, K, V, Y,
            B, chunk_len, NH, NKV, HS, cache_capacity,
            position_offset, window, scale );

        cudaCheck( cudaGetLastError() );
    }
}
