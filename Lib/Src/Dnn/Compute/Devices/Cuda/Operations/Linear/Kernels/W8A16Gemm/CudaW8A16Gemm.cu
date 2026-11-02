/**
 * @file CudaW8A16Gemm.cu
 * @brief Fused W8A16 tiled GEMM: FP8_E4M3 weights x BF16 activations -> BF16 output.
 *
 * Single-pass shared-memory tiled GEMM with inline per-channel FP8 dequantization.
 * Replaces the 2-phase path (cuda_fp8_dequantize_to_bf16 + cuBLASLt BF16 GEMM) for
 * the quantized batch forward pass. Weights are read once from VRAM as FP8; the BF16
 * staging buffer (dequant_weight_buffer_) is eliminated entirely.
 *
 * Algorithm:
 *   Grid:  (ceil(N / kTileSize), ceil(M / kTileSize))
 *   Block: (kTileSize, kTileSize) = 256 threads
 *
 *   Each block computes one [kTileSize x kTileSize] tile of the output matrix.
 *   For each K-tile:
 *     1. Thread (ty, tx) loads smem_A[ty][tx] = A[m_base+ty, k+tx]  (BF16 -> float).
 *     2. Thread (ty, tx) loads smem_W[ty][tx] = W[n_base+ty, k+tx]  (FP8 -> float * scale).
 *        Per-channel scale (scales[n_base+ty]) is applied inline — zero extra memory traffic.
 *     3. __syncthreads().
 *     4. Thread (ty, tx) accumulates: acc += sum_kk( smem_A[ty][kk] * smem_W[tx][kk] )
 *        producing the partial dot product for output element C[m_base+ty, n_base+tx].
 *     5. __syncthreads().
 *   Post-loop: store acc (+ optional bias) as BF16 to output.
 *
 * Shared memory note:
 *   The inner loop accesses smem_W[tx][kk] — tx varies across the warp, kk is the
 *   loop index. With kTileSize=16 floats per row this causes a 2-way bank conflict.
 *   Future optimisation: store smem_W transposed with +1 column padding to eliminate
 *   the conflict. The bandwidth savings (3-4x fewer global memory bytes vs 2-phase)
 *   dominate in the target memory-bound regime, making the conflict a second-order effect.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdexcept>
#include <format>
#include "CudaW8A16Gemm.cuh"

namespace Mila::Dnn::Compute::Cuda::Linear
{
    namespace
    {
        static constexpr int kTileSize = 16;  // TILE_M = TILE_N = TILE_K

        /**
         * @brief Fused per-channel W8A16 tiled GEMM kernel.
         *
         * Grid  (blockIdx.x, blockIdx.y): (ceil(N/kTileSize), ceil(M/kTileSize))
         * Block (threadIdx.x, threadIdx.y): (kTileSize, kTileSize) = 256 threads
         *
         * Thread (ty, tx) in block (bx, by) computes output[by*T+ty, bx*T+tx] where T = kTileSize.
         *
         * Shared memory:
         *   smem_A[kTileSize][kTileSize]: activation tile — smem_A[ty][tx] = A[row, k+tx]
         *   smem_W[kTileSize][kTileSize]: weight tile (dequantized) — smem_W[ty][tx] = W[n_base+ty, k+tx]*scale
         *
         * Inner loop: acc += smem_A[ty][kk] * smem_W[tx][kk]
         *   smem_A[ty][kk] — broadcast (all warp threads share ty) — no bank conflict.
         *   smem_W[tx][kk] — tx varies across warp — 2-way bank conflict (see file comment).
         */
        __global__ void fused_w8a16_gemm_kernel(
            __nv_bfloat16* __restrict__       output,
            const __nv_bfloat16* __restrict__ activations,
            const __nv_fp8_e4m3* __restrict__ weights,
            const float* __restrict__         scales,
            const __nv_bfloat16* __restrict__ bias,
            int M, int N, int K )
        {
            const int ty = static_cast<int>( threadIdx.y );
            const int tx = static_cast<int>( threadIdx.x );

            const int row    = static_cast<int>( blockIdx.y ) * kTileSize + ty;  // m
            const int col    = static_cast<int>( blockIdx.x ) * kTileSize + tx;  // n
            const int n_base = static_cast<int>( blockIdx.x ) * kTileSize;       // output-channel base for this block

            __shared__ float smem_A[kTileSize][kTileSize];  // activation tile [TILE_M, TILE_K]
            __shared__ float smem_W[kTileSize][kTileSize];  // dequantized weight tile [TILE_N, TILE_K]

            float acc = 0.0f;

            for ( int k = 0; k < K; k += kTileSize )
            {
                // --- Load activation tile ---
                // Thread (ty, tx) loads A[m_base+ty, k+tx] into smem_A[ty][tx].
                // Access is coalesced: consecutive tx in a warp maps to consecutive K indices.
                const int k_a = k + tx;
                smem_A[ty][tx] = ( row < M && k_a < K )
                    ? __bfloat162float( activations[ static_cast<int64_t>( row ) * K + k_a ] )
                    : 0.0f;

                // --- Load and dequantize weight tile ---
                // Thread (ty, tx) loads W[n_base+ty, k+tx] into smem_W[ty][tx].
                // Per-channel scale scales[n_base+ty] is applied inline — no extra memory traffic.
                // Access is coalesced: consecutive tx in a warp maps to consecutive K indices.
                //
                // Use __nv_cvt_fp8_to_halfraw + __half2float rather than operator float().
                // This is the same path used by cuda_fp8_dequantize_to_bf16 (the proven baseline).
                const int w_row = n_base + ty;
                const int k_w   = k + tx;
                const float scale = ( w_row < N ) ? scales[ w_row ] : 1.0f;
                smem_W[ty][tx] = ( w_row < N && k_w < K )
                    ? __half2float( __nv_cvt_fp8_to_halfraw(
                          weights[ static_cast<int64_t>( w_row ) * K + k_w ].__x, __NV_E4M3 ) )
                      * scale
                    : 0.0f;

                __syncthreads();

                // --- Accumulate partial dot product ---
                // Thread (ty, tx) computes the contribution of this K-tile to C[row, col]:
                //   acc += sum_kk( smem_A[ty][kk] * smem_W[tx][kk] )
                //        = sum_kk( A[row, k+kk] * W[col, k+kk] * scales[col] )
#pragma unroll
                for ( int kk = 0; kk < kTileSize; ++kk )
                    acc += smem_A[ty][kk] * smem_W[tx][kk];

                __syncthreads();
            }

            // --- Write output ---
            if ( row < M && col < N )
            {
                const float bias_val = ( bias != nullptr )
                    ? __bfloat162float( bias[ col ] )
                    : 0.0f;

                output[ static_cast<int64_t>( row ) * N + col ] = __float2bfloat16( acc + bias_val );
            }
        }

    } // anonymous namespace

    void cuda_w8a16_gemm(
        __nv_bfloat16*       output,
        const __nv_bfloat16* activations,
        const __nv_fp8_e4m3* weights,
        const float*         scales,
        const __nv_bfloat16* bias,
        int                  outer_size,
        int                  in_features,
        int                  out_features,
        cudaStream_t         stream )
    {
        const dim3 block( kTileSize, kTileSize );
        const dim3 grid(
            ( static_cast<unsigned>( out_features ) + kTileSize - 1u ) / kTileSize,
            ( static_cast<unsigned>( outer_size    ) + kTileSize - 1u ) / kTileSize );

        fused_w8a16_gemm_kernel<<<grid, block, 0, stream>>>(
            output, activations, weights, scales, bias,
            outer_size, out_features, in_features );

        const cudaError_t err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            throw std::runtime_error( std::format(
                "cuda_w8a16_gemm: kernel launch failed (M={}, N={}, K={}): {}",
                outer_size, out_features, in_features, cudaGetErrorString( err ) ) );
        }
    }
}
