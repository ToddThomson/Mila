/**
 * @file CudaW4A16Gemm.Wmma.cu
 * @brief WMMA-accelerated FP4 E2M1 x BF16 W4A16 GEMM for SM80+.
 *
 * Computes C[M,N] = A[M,K] . dequant(W)[N,K]^T + bias, where A is BF16 activations,
 * W is per-group FP4 E2M1 packed weights (2 nibbles/byte along K) with per-(channel,
 * K-group) FP32 scales, and C is BF16. M = tokens (prefill chunk), N = out_features,
 * K = in_features.
 *
 * STAGE 1 (multi-warp register-accumulator tiling). The prior kernel was one warp per
 * block computing a single [16 x 16] tile -- no A/W reuse, ~1 block-per-16x16, measured
 * ~2.5 TFLOPS (Gemma4InferenceReview.md 10.2), which is why the 2-phase dequant ->
 * cuBLASLt path was faster and the fused kernel stayed toggled off. This rewrite tiles
 * a 64 x 64 output block across 4 warps (2 x 2), each owning a 32 x 32 sub-tile as
 * register-resident wmma accumulators that persist across the whole K loop, so each A
 * row-strip feeds all BN columns and each W column-strip feeds all BM rows -- the reuse
 * the naive kernel lacked. A GEMM has no online-softmax rescale, so the safe nvcuda::wmma
 * API suffices (no mma.sync PTX). The per-tile FP4 decode, scale indexing, and fragment
 * orientations are reused verbatim from the proven naive kernel.
 *
 * NEXT (Stage 2): cp.async double-buffered global->smem loads (overlap load with MMA --
 * the "Stall Long Scoreboard" fix), XOR swizzle to kill smem bank conflicts, ldmatrix,
 * and larger tiles. See the fa-5090 optimization ladder in GqaFlashAttention.md notes.
 *
 * Fragment B orientation (unchanged): sW holds the dequantized tile as [n_local, k_local]
 * row-major. Loading it as wmma matrix_b col_major with ldm = BK yields B[k][n] =
 * sW[n][k] = W^T, the transposed weight required for C = A . W^T.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
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
        static constexpr int kWarpTilesM = kWarpM / kWmmaM;    // 2
        static constexpr int kWarpTilesN = kWarpN / kWmmaN;    // 2
        static constexpr int kKSubTiles = kBlockK / kWmmaK;    // 2

        /**
         * @brief Decode a 4-bit FP4 E2M1 nibble to float32.
         */
        __device__ __forceinline__ float fp4_e2m1_decode( uint8_t nibble )
        {
            static constexpr float kLut[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
            const float magnitude = kLut[ nibble & 0x7u ];
            return ( nibble & 0x8u ) ? -magnitude : magnitude;
        }

        /**
         * @brief Multi-warp WMMA FP4 E2M1 x BF16 GEMM kernel.
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

            __shared__ __nv_bfloat16 sA[ kBlockM ][ kBlockK ];
            __shared__ __nv_bfloat16 sW[ kBlockN ][ kBlockK ];
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

            for ( int k0 = 0; k0 < K; k0 += kBlockK )
            {
                // --- cooperative load of the activation tile sA[kBlockM][kBlockK] (BF16) ---
#pragma unroll
                for ( int e = tid; e < kBlockM * kBlockK; e += kBlockThreads )
                {
                    const int row = e / kBlockK;
                    const int col = e % kBlockK;
                    const int gm  = m_block + row;
                    const int gk  = k0 + col;

                    sA[ row ][ col ] = ( gm < M && gk < K )
                        ? activations[ static_cast<int64_t>( gm ) * K + gk ]
                        : __float2bfloat16( 0.0f );
                }

                // --- cooperative dequant of the weight tile sW[kBlockN][kBlockK] (FP4 -> BF16) ---
#pragma unroll
                for ( int e = tid; e < kBlockN * kBlockK; e += kBlockThreads )
                {
                    const int row = e / kBlockK;   // n_local
                    const int col = e % kBlockK;   // k_local
                    const int gn  = n_block + row;
                    const int gk  = k0 + col;

                    if ( gn < N && gk < K )
                    {
                        const uint8_t byte   = weights_packed[ static_cast<int64_t>( gn ) * half_K + gk / 2 ];
                        const uint8_t nibble = ( gk % 2 == 0 ) ? ( byte & 0xFu ) : ( byte >> 4 );
                        const float   scale  = scales[ static_cast<int64_t>( gn ) * num_groups + gk / kGroupSize ];
                        sW[ row ][ col ] = __float2bfloat16( fp4_e2m1_decode( nibble ) * scale );
                    }
                    else
                        sW[ row ][ col ] = __float2bfloat16( 0.0f );
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
                        wmma::load_matrix_sync( a_frag[ i ], &sA[ warp_row * kWarpM + i * kWmmaM ][ k_sub ], kBlockK );

#pragma unroll
                    for ( int j = 0; j < kWarpTilesN; ++j )
                        wmma::load_matrix_sync( b_frag[ j ], &sW[ warp_col * kWarpN + j * kWmmaN ][ k_sub ], kBlockK );

#pragma unroll
                    for ( int i = 0; i < kWarpTilesM; ++i )
#pragma unroll
                        for ( int j = 0; j < kWarpTilesN; ++j )
                            wmma::mma_sync( c_frag[ i ][ j ], a_frag[ i ], b_frag[ j ], c_frag[ i ][ j ] );
                }

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
