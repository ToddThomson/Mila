/**
 * @file CudaW4A16Gemm.Wmma.cu
 * @brief WMMA-accelerated FP4 E2M1 x BF16 W4A16 GEMM for SM80+.
 *
 * Computes C[M,N] = A[M,K] . dequant(W)[N,K]^T + bias, where A is BF16 activations,
 * W is per-group FP4 E2M1 packed weights (2 nibbles/byte along K) with per-(channel,
 * K-group) FP32 scales, and C is BF16. M = tokens (prefill chunk), N = out_features,
 * K = in_features.
 *
 * STAGE 2 (cp.async double-buffered software pipeline). Stage 1 tiled a 64 x 64 output
 * block across 4 warps with register-resident wmma accumulators, but loaded each K-tile
 * synchronously: the global reads blocked the MMAs, so the kernel profiled Stall Long
 * Scoreboard bound (global-load latency) at ~33% occupancy, Compute (SM) ~37% -- latency
 * starved, not throughput starved. Stage 2 keeps the exact tile geometry, warp layout,
 * MMA, and epilogue, and replaces the synchronous loads with a two-stage cp.async
 * pipeline: while the current K-tile's MMAs run, the next tile's activations and packed
 * weights stream global->smem asynchronously, hiding the latency without needing more
 * occupancy (software pipelining substitutes for occupancy-based latency hiding).
 *
 * W4A16 wrinkle: cp.async copies raw bytes and cannot decode FP4 inline, so the packed
 * weight tile is double-buffered as raw nibbles (rawW) and a single BF16 scratch (bf16W)
 * is filled by a dequant pass each iteration -- that pass reads smem (short scoreboard),
 * not global, so the expensive latency is the one cp.async now overlaps. Activations are
 * already BF16, so sA is double-buffered and fed to wmma directly.
 *
 * The FP4 path guarantees K % group_size == 0 (per-group scales), and group_size is 64 or
 * 128, so K is a multiple of 64: BK=32 divides K with no tail and every cp.async address
 * is 16-byte aligned. Bounds handling therefore reduces to the M (token) and N (channel)
 * row edges; OOB rows are zero-filled in smem so the MMA/dequant math is unchanged.
 *
 * NEXT (Stage 2 ladder): XOR swizzle to kill smem bank conflicts, ldmatrix, and larger
 * tiles (128x64 / 128x128). See the fa-5090 optimization ladder in GqaFlashAttention.md.
 *
 * Fragment B orientation (unchanged): bf16W holds the dequantized tile as [n_local,
 * k_local] row-major. Loading it as wmma matrix_b col_major with ldm = kBlockK yields
 * B[k][n] = bf16W[n][k] = W^T, the transposed weight required for C = A . W^T.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cstdint>
#include "CudaW4A16Gemm.Wmma.cuh"

namespace Mila::Dnn::Compute::Cuda::Linear
{
    namespace
    {
        using namespace nvcuda;

        static constexpr int kWmmaM = 16;
        static constexpr int kWmmaN = 16;
        static constexpr int kWmmaK = 16;

        // Block tile: 64 x 64 output, contracted in BK=32 steps by 4 warps arranged 2 x 2.
        static constexpr int kBlockM = 64;
        static constexpr int kBlockN = 64;
        static constexpr int kBlockK = 32;
        static constexpr int kWarpsM = 2;                 // warp rows
        static constexpr int kWarpsN = 2;                 // warp cols
        static constexpr int kNumWarps = kWarpsM * kWarpsN;
        static constexpr int kBlockThreads = kNumWarps * 32;   // 128
        static constexpr int kWarpM = kBlockM / kWarpsM;  // 32 output rows per warp
        static constexpr int kWarpN = kBlockN / kWarpsN;  // 32 output cols per warp
        // The next three and the two helpers below are referenced only inside the kernel's
        // __CUDA_ARCH__ >= 800 guard, so every sub-SM80 compilation pass sees them
        // unreferenced. The constants the host-side launcher also reads (kBlockM,
        // kBlockThreads) need no such marking.
        [[maybe_unused]] static constexpr int kWarpTilesM = kWarpM / kWmmaM;    // 2
        [[maybe_unused]] static constexpr int kWarpTilesN = kWarpN / kWmmaN;    // 2
        [[maybe_unused]] static constexpr int kKSubTiles = kBlockK / kWmmaK;    // 2
        static constexpr int kPackedBytesPerRow = kBlockK / 2; // 16 packed FP4 bytes per weight row

        /**
         * @brief Decode a 4-bit FP4 E2M1 nibble to float32.
         */
        [[maybe_unused]] __device__ __forceinline__ float fp4_e2m1_decode( uint8_t nibble )
        {
            static constexpr float kLut[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
            const float magnitude = kLut[ nibble & 0x7u ];
            return ( nibble & 0x8u ) ? -magnitude : magnitude;
        }

        /**
         * @brief Issue the cp.async global->smem copies for one K-tile.
         *
         * Activations land in sA_buf ready for wmma; packed FP4 weights land in rawW_buf and
         * are decoded later by the dequant pass. OOB M/N rows are zero-filled synchronously so
         * the async group only covers valid data. The caller commits the group and waits.
         *
         * K is a multiple of kBlockK (FP4 group_size constraint), so k0 + kBlockK <= K always
         * and every 16-byte copy is in-bounds along K and 16-byte aligned.
         */
        [[maybe_unused]] __device__ __forceinline__ void loadTileAsync(
            __nv_bfloat16 (&sA_buf)[ kBlockM ][ kBlockK ],
            uint8_t (&rawW_buf)[ kBlockN ][ kPackedBytesPerRow ],
            const __nv_bfloat16* __restrict__ activations,
            const uint8_t* __restrict__       weights_packed,
            int m_block, int n_block, int k0,
            int M, int N, int K, int half_K, int tid )
        {
            // Activation tile: 16-byte (8 bf16) chunks, kBlockK/8 chunks per row.
            constexpr int kAChunksPerRow = kBlockK / 8;
#pragma unroll
            for ( int c = tid; c < kBlockM * kAChunksPerRow; c += kBlockThreads )
            {
                const int row = c / kAChunksPerRow;
                const int col = ( c % kAChunksPerRow ) * 8;
                const int gm  = m_block + row;
                void* dst = &sA_buf[ row ][ col ];

                if ( gm < M )
                    __pipeline_memcpy_async(
                        dst, &activations[ static_cast<int64_t>( gm ) * K + k0 + col ], 16 );
                else
                    *reinterpret_cast<float4*>( dst ) = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
            }

            // Weight tile: one 16-byte (kBlockK/2 packed) copy per row.
#pragma unroll
            for ( int c = tid; c < kBlockN; c += kBlockThreads )
            {
                const int gn = n_block + c;
                void* dst = &rawW_buf[ c ][ 0 ];

                if ( gn < N )
                    __pipeline_memcpy_async(
                        dst, &weights_packed[ static_cast<int64_t>( gn ) * half_K + k0 / 2 ], 16 );
                else
                    *reinterpret_cast<float4*>( dst ) = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
            }
        }

        /**
         * @brief Multi-warp cp.async double-buffered WMMA FP4 E2M1 x BF16 GEMM kernel.
         *
         * @tparam kGroupSize Quantization group size along K (64 or 128).
         *
         * Grid  (blockIdx.x, blockIdx.y): (ceil(N/kBlockN), ceil(M/kBlockM))
         * Block: kBlockThreads (kNumWarps warps).
         */
        template <int kGroupSize>
        __global__ __launch_bounds__(kBlockThreads)
        void fp4a16_wmma_gemm_kernel(
            __nv_bfloat16* __restrict__       output,
            const __nv_bfloat16* __restrict__ activations,
            const uint8_t* __restrict__       weights_packed,
            const float* __restrict__         scales,
            const __nv_bfloat16* __restrict__ bias,
            int M, int N, int K )
        {
// BF16 WMMA fragments require SM >= 8.0. The host-side SM gate (use_wmma_fp4_gemm_)
// ensures this kernel is never launched on SM < 8.0; the guard suppresses the
// incomplete-type error when PTX is compiled for SM 7.5 targets.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            const int tid     = static_cast<int>( threadIdx.x );
            const int warp_id = tid >> 5;
            const int lane    = tid & 31;
            const int warp_row = warp_id / kWarpsN;   // 0..kWarpsM-1
            const int warp_col = warp_id % kWarpsN;   // 0..kWarpsN-1

            const int m_block = static_cast<int>( blockIdx.y ) * kBlockM;
            const int n_block = static_cast<int>( blockIdx.x ) * kBlockN;

            const int half_K     = K / 2;
            const int num_groups = K / kGroupSize;

            // Double-buffered raw inputs (cp.async targets) + single-buffered dequantized
            // weight scratch (produced fresh each iteration). 16-byte aligned so the cp.async
            // and float4 zero-fill stores are legal.
            __shared__ __align__(16) __nv_bfloat16 sA[ 2 ][ kBlockM ][ kBlockK ];
            __shared__ __align__(16) uint8_t       rawW[ 2 ][ kBlockN ][ kPackedBytesPerRow ];
            __shared__ __align__(16) __nv_bfloat16 bf16W[ kBlockN ][ kBlockK ];
            // Epilogue staging: one [16 x 16] tile per warp, reused across its sub-tiles.
            __shared__ float sOut[ kNumWarps ][ kWmmaM ][ kWmmaN ];

            // Register-resident accumulators: this warp's kWarpTilesM x kWarpTilesN grid of
            // [16 x 16] fragments, persistent across the whole K loop.
            wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag[ kWarpTilesM ][ kWarpTilesN ];
#pragma unroll
            for ( int i = 0; i < kWarpTilesM; ++i )
#pragma unroll
                for ( int j = 0; j < kWarpTilesN; ++j )
                    wmma::fill_fragment( c_frag[ i ][ j ], 0.0f );

            const int num_k_tiles = K / kBlockK;

            // Prologue: kick off the first tile's async loads.
            loadTileAsync( sA[ 0 ], rawW[ 0 ], activations, weights_packed,
                m_block, n_block, 0, M, N, K, half_K, tid );
            __pipeline_commit();

            for ( int k = 0; k < num_k_tiles; ++k )
            {
                const int cur = k & 1;

                // Prefetch the next tile so its global latency overlaps this tile's dequant + MMA.
                if ( k + 1 < num_k_tiles )
                {
                    loadTileAsync( sA[ ( k + 1 ) & 1 ], rawW[ ( k + 1 ) & 1 ],
                        activations, weights_packed,
                        m_block, n_block, ( k + 1 ) * kBlockK, M, N, K, half_K, tid );
                    __pipeline_commit();
                    __pipeline_wait_prior( 1 );   // this tile (1 group behind) is now resident
                }
                else
                    __pipeline_wait_prior( 0 );   // drain the final tile

                __syncthreads();

                // --- dequant this tile's weights sm(raw) -> smem(bf16); reads smem, not global ---
#pragma unroll
                for ( int e = tid; e < kBlockN * kBlockK; e += kBlockThreads )
                {
                    const int row = e / kBlockK;   // n_local
                    const int col = e % kBlockK;   // k_local
                    const int gn  = n_block + row;
                    const int gk  = k * kBlockK + col;

                    const uint8_t byte   = rawW[ cur ][ row ][ col >> 1 ];
                    const uint8_t nibble = ( col & 1 ) ? ( byte >> 4 ) : ( byte & 0xFu );
                    const float   scale  = ( gn < N )
                        ? scales[ static_cast<int64_t>( gn ) * num_groups + gk / kGroupSize ]
                        : 0.0f;
                    bf16W[ row ][ col ] = __float2bfloat16( fp4_e2m1_decode( nibble ) * scale );
                }

                __syncthreads();

                // --- MMA: accumulate this warp's sub-tiles over the BK slice ---
#pragma unroll
                for ( int kk = 0; kk < kKSubTiles; ++kk )
                {
                    const int k_sub = kk * kWmmaK;

                    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::row_major> a_frag[ kWarpTilesM ];
                    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::col_major> b_frag[ kWarpTilesN ];

#pragma unroll
                    for ( int i = 0; i < kWarpTilesM; ++i )
                        wmma::load_matrix_sync( a_frag[ i ], &sA[ cur ][ warp_row * kWarpM + i * kWmmaM ][ k_sub ], kBlockK );

#pragma unroll
                    for ( int j = 0; j < kWarpTilesN; ++j )
                        wmma::load_matrix_sync( b_frag[ j ], &bf16W[ warp_col * kWarpN + j * kWmmaN ][ k_sub ], kBlockK );

#pragma unroll
                    for ( int i = 0; i < kWarpTilesM; ++i )
#pragma unroll
                        for ( int j = 0; j < kWarpTilesN; ++j )
                            wmma::mma_sync( c_frag[ i ][ j ], a_frag[ i ], b_frag[ j ], c_frag[ i ][ j ] );
                }

                // Barrier so this tile's MMA finishes reading bf16W (single-buffered) and
                // sA[cur]/rawW[cur] before the next iteration's dequant overwrites bf16W and
                // its prefetch reissues cp.async into buf[cur].
                __syncthreads();
            }

            // --- epilogue: stage each sub-tile through smem, add bias, write with bounds ---
#pragma unroll
            for ( int i = 0; i < kWarpTilesM; ++i )
            {
#pragma unroll
                for ( int j = 0; j < kWarpTilesN; ++j )
                {
                    wmma::store_matrix_sync( &sOut[ warp_id ][ 0 ][ 0 ], c_frag[ i ][ j ], kWmmaN, wmma::mem_row_major );
                    __syncwarp();

                    const int tile_m = m_block + warp_row * kWarpM + i * kWmmaM;
                    const int tile_n = n_block + warp_col * kWarpN + j * kWmmaN;

                    for ( int e = lane; e < kWmmaM * kWmmaN; e += 32 )
                    {
                        const int r  = e / kWmmaN;
                        const int c  = e % kWmmaN;
                        const int gm = tile_m + r;
                        const int gn = tile_n + c;

                        if ( gm < M && gn < N )
                        {
                            const float bias_val = ( bias != nullptr ) ? __bfloat162float( bias[ gn ] ) : 0.0f;
                            output[ static_cast<int64_t>( gm ) * N + gn ] =
                                __float2bfloat16( sOut[ warp_id ][ r ][ c ] + bias_val );
                        }
                    }

                    __syncwarp();
                }
            }
#endif // __CUDA_ARCH__ >= 800
        }

    } // anonymous namespace

    void cuda_fp4a16_gemm_wmma(
        __nv_bfloat16*       output,
        const __nv_bfloat16* activations,
        const uint8_t*       weights_packed,
        const float*         scales,
        const __nv_bfloat16* bias,
        int                  outer_size,
        int                  in_features,
        int                  out_features,
        int                  group_size,
        cudaStream_t         stream )
    {
        const dim3 block( kBlockThreads );
        const dim3 grid(
            ( static_cast<unsigned>( out_features ) + kBlockN - 1u ) / kBlockN,
            ( static_cast<unsigned>( outer_size    ) + kBlockM - 1u ) / kBlockM );

        switch ( group_size )
        {
            case 64:
                fp4a16_wmma_gemm_kernel<64><<<grid, block, 0, stream>>>(
                    output, activations, weights_packed, scales, bias,
                    outer_size, out_features, in_features );
                break;

            case 128:
                fp4a16_wmma_gemm_kernel<128><<<grid, block, 0, stream>>>(
                    output, activations, weights_packed, scales, bias,
                    outer_size, out_features, in_features );
                break;

            default:
                break;
        }
    }

} // namespace Mila::Dnn::Compute::Cuda::Linear
