/**
 * @file CudaW4A16Gemm.Wmma.cu
 * @brief WMMA-accelerated FP4 E2M1 x BF16 W4A16 GEMM for SM80+.
 *
 * Uses nvcuda::wmma m16n16k16 BF16 tensor core MMA. One warp (32 threads)
 * per block computes one [16 x 16] output tile. Dequantization of packed FP4
 * E2M1 nibbles happens in shared memory before each WMMA B fragment load.
 *
 * Algorithm:
 *   Grid:  (ceil(N / 16), ceil(M / 16))
 *   Block: 32 threads (1 warp)
 *
 *   For each K-tile (16 columns):
 *     1. All 32 threads cooperatively load tile_A[16][16] from activations.
 *     2. All 32 threads unpack FP4 E2M1 nibbles, apply per-group scale, and
 *        store dequantized BF16 values into tile_W[16][16].
 *     3. Load tile_A as WMMA matrix_a (row_major [M, K]).
 *        Load tile_W as WMMA matrix_b (col_major: [N, K] row_major = [K, N] col_major).
 *     4. wmma::mma_sync accumulates into a float32 accumulator fragment.
 *
 *   Post-loop: store accumulator to smem_out, add optional bias, write BF16 output.
 *
 * Shared memory per block (2560 bytes):
 *   tile_A[16][16]   BF16  (512 bytes)
 *   tile_W[16][16]   BF16  (512 bytes)
 *   smem_out[16][16] float (1024 bytes, written after K-loop)
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

        static constexpr int kWmmaM        = 16;
        static constexpr int kWmmaN        = 16;
        static constexpr int kWmmaK        = 16;
        static constexpr int kBlockThreads = 32;   // 1 warp per block

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
         * @brief Single-warp WMMA FP4 E2M1 x BF16 GEMM kernel.
         *
         * @tparam kGroupSize Quantization group size along K (64 or 128).
         *
         * Grid  (blockIdx.x, blockIdx.y): (ceil(N/16), ceil(M/16))
         * Block: 32 threads (one warp).
         *
         * Fragment B orientation:
         *   tile_W is stored [kWmmaN, kWmmaK] = [N, K] row-major.
         *   Treating this as [K, N] col-major gives B[k][n] = tile_W[n][k],
         *   which is the transposed weight required for C = A * W^T.
         *   Leading dimension for col_major [K, N] load = kWmmaK (rows in K).
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
            const int m_block = static_cast<int>( blockIdx.y ) * kWmmaM;
            const int n_block = static_cast<int>( blockIdx.x ) * kWmmaN;
            const int tid     = static_cast<int>( threadIdx.x );

            __shared__ __nv_bfloat16 tile_A[kWmmaM][kWmmaK];
            __shared__ __nv_bfloat16 tile_W[kWmmaN][kWmmaK];
            __shared__ float smem_out[kWmmaM][kWmmaN];

            wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
            wmma::fill_fragment( c_frag, 0.0f );

            const int half_K    = K / 2;
            const int num_groups = K / kGroupSize;

            for ( int k = 0; k < K; k += kWmmaK )
            {
                // --- Load activation tile tile_A[16][16] ---
                // 256 elements, 32 threads, 8 per thread.
                // flat = i*32 + tid: consecutive tids access consecutive K columns
                // within the same M row — coalesced within the warp.
#pragma unroll
                for ( int i = 0; i < 8; ++i )
                {
                    const int flat     = i * kBlockThreads + tid;
                    const int a_row    = flat / kWmmaK;
                    const int a_col    = flat % kWmmaK;
                    const int global_m = m_block + a_row;
                    const int global_k = k + a_col;

                    tile_A[a_row][a_col] = ( global_m < M && global_k < K )
                        ? activations[ static_cast<int64_t>( global_m ) * K + global_k ]
                        : __float2bfloat16( 0.0f );
                }

                // --- Dequantize weight tile tile_W[16][16] ---
                // 256 elements, 32 threads, 8 per thread.
                // All K elements in a 16-wide chunk share one group_index for
                // group_size in {64, 128} (both are multiples of 16).
                {
                    const int group_index = k / kGroupSize;

#pragma unroll
                    for ( int i = 0; i < 8; ++i )
                    {
                        const int flat     = i * kBlockThreads + tid;
                        const int w_row    = flat / kWmmaK;
                        const int w_col    = flat % kWmmaK;
                        const int global_n = n_block + w_row;
                        const int global_k = k + w_col;

                        if ( global_n < N && global_k < K )
                        {
                            const uint8_t byte   = weights_packed[ static_cast<int64_t>( global_n ) * half_K + global_k / 2 ];
                            const uint8_t nibble = ( global_k % 2 == 0 ) ? ( byte & 0xFu ) : ( byte >> 4 );
                            const float scale    = scales[ static_cast<int64_t>( global_n ) * num_groups + group_index ];
                            tile_W[w_row][w_col] = __float2bfloat16( fp4_e2m1_decode( nibble ) * scale );
                        }
                        else
                        {
                            tile_W[w_row][w_col] = __float2bfloat16( 0.0f );
                        }
                    }
                }

                __syncthreads();

                // --- WMMA fragment loads and MMA ---
                wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::col_major> b_frag;

                wmma::load_matrix_sync( a_frag, &tile_A[0][0], kWmmaK );
                wmma::load_matrix_sync( b_frag, &tile_W[0][0], kWmmaK );
                wmma::mma_sync( c_frag, a_frag, b_frag, c_frag );

                __syncthreads();
            }

            // --- Store accumulator, apply bias, write output ---
            wmma::store_matrix_sync( &smem_out[0][0], c_frag, kWmmaN, wmma::mem_row_major );
            __syncthreads();

#pragma unroll
            for ( int i = 0; i < 8; ++i )
            {
                const int flat     = i * kBlockThreads + tid;
                const int out_row  = flat / kWmmaN;
                const int out_col  = flat % kWmmaN;
                const int global_m = m_block + out_row;
                const int global_n = n_block + out_col;

                if ( global_m < M && global_n < N )
                {
                    const float bias_val = ( bias != nullptr )
                        ? __bfloat162float( bias[ global_n ] )
                        : 0.0f;

                    output[ static_cast<int64_t>( global_m ) * N + global_n ] =
                        __float2bfloat16( smem_out[out_row][out_col] + bias_val );
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
            ( static_cast<unsigned>( out_features ) + kWmmaN - 1u ) / kWmmaN,
            ( static_cast<unsigned>( outer_size    ) + kWmmaM - 1u ) / kWmmaM );

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
