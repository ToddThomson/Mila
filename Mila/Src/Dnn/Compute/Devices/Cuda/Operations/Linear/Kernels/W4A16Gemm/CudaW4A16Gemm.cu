/**
 * @file CudaW4A16Gemm.cu
 * @brief Fused W4A16 tiled GEMM: packed INT4 weights x BF16 activations -> BF16 output.
 *
 * Single-pass shared-memory tiled GEMM with inline per-group INT4 dequantization.
 * Weights are stored as packed INT4 (2 nibbles per byte); each tile load unpacks the
 * relevant nibble, fetches the group scale and zero point, and writes a dequantized
 * float32 value into shared memory. No intermediate float weight buffer is required.
 *
 * Algorithm:
 *   Grid:  (ceil(N / kTileSize), ceil(M / kTileSize))
 *   Block: (kTileSize, kTileSize) = 256 threads
 *
 *   Each block computes one [kTileSize x kTileSize] tile of the output matrix.
 *   For each K-tile:
 *     1. Thread (ty, tx) loads smem_A[ty][tx]  = A[m_base+ty, k+tx]  (BF16 -> float).
 *     2. Thread (ty, tx) loads smem_W[ty][tx]:
 *          a. Fetch packed byte: weights_packed[n_base+ty, (k+tx)/2]
 *          b. Extract nibble:    (k+tx) even -> low nibble & 0xF; odd -> high nibble >> 4
 *          c. Fetch group scale: scales[n_base+ty, (k+tx)/group_size]
 *          d. Fetch group zero:  zero_points[n_base+ty, (k+tx)/group_size / 2]  (or 8.0)
 *          e. Dequantize:        smem_W[ty][tx] = (nibble - zero) * scale
 *     3. __syncthreads().
 *     4. Thread (ty, tx) accumulates: acc += sum_kk( smem_A[ty][kk] * smem_W[tx][kk] )
 *     5. __syncthreads().
 *   Post-loop: store acc (+ optional bias) as BF16 to output.
 *
 * Template parameter kGroupSize controls the quantization group size along K.
 * The host function cuda_w4a16_gemm() dispatches to the correct instantiation
 * (kGroupSize = 64 or 128) at runtime.
 *
 * Shared memory note:
 *   The inner loop accesses smem_W[tx][kk] — tx varies across the warp, kk is the
 *   loop index. With kTileSize=16 floats per row this causes a 2-way bank conflict.
 *   Future optimisation: store smem_W transposed with +1 column padding to eliminate
 *   the conflict. For the INT4 memory-bound regime the bandwidth savings dominate.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include "CudaW4A16Gemm.cuh"

namespace Mila::Dnn::Compute::Cuda::Linear
{
    namespace
    {
        static constexpr int kTileSize = 16;  // TILE_M = TILE_N = TILE_K

        /**
         * @brief Fused per-group W4A16 tiled GEMM kernel.
         *
         * @tparam kGroupSize  Quantization group size along K (64 or 128).
         *
         * Grid  (blockIdx.x, blockIdx.y): (ceil(N/kTileSize), ceil(M/kTileSize))
         * Block (threadIdx.x, threadIdx.y): (kTileSize, kTileSize) = 256 threads
         *
         * Thread (ty, tx) in block (bx, by) computes output[by*T+ty, bx*T+tx] where T = kTileSize.
         *
         * Shared memory:
         *   smem_A[kTileSize][kTileSize]: activation tile — smem_A[ty][tx] = A[row, k+tx]
         *   smem_W[kTileSize][kTileSize]: dequantized weight tile — smem_W[ty][tx] = dequant(W[n_base+ty, k+tx])
         *
         * Dequantization per element:
         *   nibble = low nibble if k_w even, high nibble if k_w odd
         *   zero   = nibble from zero_points (or 8.0 if nullptr)
         *   result = (float(nibble) - zero) * scales[n_base+ty, k_w/kGroupSize]
         *
         * Inner loop: acc += smem_A[ty][kk] * smem_W[tx][kk]
         *   smem_W[tx][kk] — tx varies across warp — 2-way bank conflict (see file comment).
         */
        /**
         * @brief Decode a 4-bit FP4 E2M1 nibble to float32.
         *
         * Nibble layout: bit3=sign, bits[2:1]=exponent, bit0=mantissa.
         * Positive representable values (nibbles 0-7): {0, 0.5, 1, 1.5, 2, 3, 4, 6}
         * Negative values (nibbles 8-15): sign-magnitude mirror.
         */
        __device__ __forceinline__ float fp4_e2m1_decode( uint8_t nibble )
        {
            static constexpr float kLut[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
            const float mag  = kLut[ nibble & 0x7u ];
            return ( nibble & 0x8u ) ? -mag : mag;
        }

        template <int kGroupSize, bool kIsFp4E2M1 = false>
        __global__ void fused_w4a16_gemm_kernel(
            __nv_bfloat16* __restrict__       output,
            const __nv_bfloat16* __restrict__ activations,
            const uint8_t* __restrict__       weights_packed,
            const float* __restrict__         scales,
            const uint8_t* __restrict__       zero_points,
            const __nv_bfloat16* __restrict__ bias,
            int M, int N, int K )
        {
            const int ty = static_cast<int>( threadIdx.y );
            const int tx = static_cast<int>( threadIdx.x );

            const int row    = static_cast<int>( blockIdx.y ) * kTileSize + ty;  // m
            const int col    = static_cast<int>( blockIdx.x ) * kTileSize + tx;  // n
            const int n_base = static_cast<int>( blockIdx.x ) * kTileSize;       // output-channel base for this block

            __shared__ float smem_A[kTileSize][kTileSize];  // activation tile   [TILE_M, TILE_K]
            __shared__ float smem_W[kTileSize][kTileSize];  // dequantized weight tile [TILE_N, TILE_K]

            // Precomputed strides — derived from runtime K, constant across all K-tiles.
            const int num_groups     = K / kGroupSize;
            const int half_K         = K / 2;  // stride of weights_packed in columns
            const int half_num_groups = num_groups / 2;  // stride of zero_points in groups

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
                // The packed uint8 row is accessed coalesced (two consecutive tx share one byte,
                // but the byte load itself is still efficient at this granularity).
                const int w_row = n_base + ty;
                const int k_w   = k + tx;

                if ( w_row < N && k_w < K )
                {
                    // Unpack the nibble for column k_w.
                    //   Even k_w: low  nibble (bits [3:0])
                    //   Odd  k_w: high nibble (bits [7:4])
                    const uint8_t w_byte   = weights_packed[ static_cast<int64_t>( w_row ) * half_K + k_w / 2 ];
                    const uint8_t nibble   = ( k_w % 2 == 0 ) ? ( w_byte & 0xFu ) : ( w_byte >> 4 );

                    const int   group_index = k_w / kGroupSize;
                    const float scale       = scales[ static_cast<int64_t>( w_row ) * num_groups + group_index ];

                    if constexpr ( kIsFp4E2M1 )
                    {
                        // FP4 E2M1: sign encoded in nibble bit 3, look up float value directly.
                        // No zero-point — symmetric format.
                        smem_W[ty][tx] = fp4_e2m1_decode( nibble ) * scale;
                    }
                    else
                    {
                        // INT4: unsigned nibble in [0, 15], de-bias by zero point.
                        const float w_int4 = static_cast<float>( nibble );

                        float zero;
                        if ( zero_points != nullptr )
                        {
                            const uint8_t zp_byte = zero_points[
                                static_cast<int64_t>( w_row ) * half_num_groups + group_index / 2 ];
                            zero = ( group_index % 2 == 0 )
                                ? static_cast<float>( zp_byte & 0xFu )
                                : static_cast<float>( zp_byte >> 4 );
                        }
                        else
                        {
                            zero = 8.0f;  // symmetric: mid-point of [0, 15]
                        }

                        smem_W[ty][tx] = ( w_int4 - zero ) * scale;
                    }
                }
                else
                {
                    smem_W[ty][tx] = 0.0f;
                }

                __syncthreads();

                // --- Accumulate partial dot product ---
                // Thread (ty, tx) computes the contribution of this K-tile to C[row, col]:
                //   acc += sum_kk( smem_A[ty][kk] * smem_W[tx][kk] )
                //        = sum_kk( A[row, k+kk] * dequant(W[col, k+kk]) )
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

        /**
         * @brief Per-group FP4 E2M1 -> BF16 weight dequantization (prefill staging).
         *
         * One block per output channel; threads stride over the packed bytes and
         * write two BF16 values per byte. Both nibbles of a byte share one group
         * scale (group_size is a multiple of 64, so a byte never straddles groups).
         * BF16 rounding matches the fused GEMM kernels' __float2bfloat16 weight
         * treatment, so the staged weights are bit-identical to what the fused
         * kernels multiply against.
         */
        template <int kGroupSize>
        __global__ void dequantize_fp4_to_bf16_kernel(
            __nv_bfloat16* __restrict__ output,
            const uint8_t* __restrict__ weights_packed,
            const float* __restrict__ scales,
            int in_features )
        {
            const int output_channel = static_cast<int>( blockIdx.x );
            const int half_k = in_features / 2;
            const int num_groups = in_features / kGroupSize;

            const uint8_t* row_source = weights_packed + static_cast<ptrdiff_t>( output_channel ) * half_k;
            const float* row_scales = scales + static_cast<ptrdiff_t>( output_channel ) * num_groups;
            __nv_bfloat162* row_dest = reinterpret_cast<__nv_bfloat162*>(
                output + static_cast<ptrdiff_t>( output_channel ) * in_features );

            for ( int b = static_cast<int>( threadIdx.x ); b < half_k; b += static_cast<int>( blockDim.x ) )
            {
                const uint8_t byte = row_source[ b ];
                const float scale = row_scales[ b / ( kGroupSize / 2 ) ];

                // Packing contract: low nibble = element 2b, high nibble = element 2b + 1.
                row_dest[ b ] = __floats2bfloat162_rn(
                    fp4_e2m1_decode( byte & 0xFu ) * scale,
                    fp4_e2m1_decode( byte >> 4 ) * scale );
            }
        }

        /**
         * @brief Max-reduce the FP4 per-group scales into a raw absmax scalar.
         *
         * Grid-stride reduction; atomicMax on the int reinterpretation is valid because
         * the scales are non-negative (their IEEE-754 patterns order as signed ints).
         * out must be pre-zeroed. Finalized by fp8_weight_scale_finalize_kernel.
         */
        __global__ void fp8_weight_scale_reduce_kernel(
            const float* __restrict__ fp4_group_scales,
            int64_t num_scales,
            float* __restrict__ out )
        {
            __shared__ float shared_max[ 256 ];

            float local = 0.0f;

            for ( int64_t i = static_cast<int64_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
                  i < num_scales;
                  i += static_cast<int64_t>( gridDim.x ) * blockDim.x )
            {
                local = fmaxf( local, fp4_group_scales[ i ] );
            }

            shared_max[ threadIdx.x ] = local;
            __syncthreads();

            for ( int offset = blockDim.x / 2; offset > 0; offset >>= 1 )
            {
                if ( threadIdx.x < offset )
                {
                    shared_max[ threadIdx.x ] = fmaxf( shared_max[ threadIdx.x ], shared_max[ threadIdx.x + offset ] );
                }

                __syncthreads();
            }

            if ( threadIdx.x == 0 )
            {
                atomicMax( reinterpret_cast<int*>( out ), __float_as_int( shared_max[ 0 ] ) );
            }
        }

        /**
         * @brief Fold the raw max group scale into the E4M3 weight scale.
         *
         * weight_fp8_scale = (6 / 448) * max(scale). The 6 lifts the group scale back to
         * the weight absmax (peak FP4 magnitude); the /448 maps that to E4M3 range.
         */
        __global__ void fp8_weight_scale_finalize_kernel( float* __restrict__ out )
        {
            *out = fmaxf( *out, 1e-12f ) * ( 6.0f / 448.0f );
        }

        /**
         * @brief Per-group FP4 E2M1 -> FP8_E4M3 weight upcast (prefill staging).
         *
         * One block per output channel; threads stride over the packed bytes and write
         * two FP8 values per byte. Both nibbles of a byte share one group scale (group_size
         * is a multiple of 64, so a byte never straddles groups). The per-group scale is
         * folded in and the values divided by the per-tensor weight scale so they land in
         * E4M3 range; cuBLASLt's A_SCALE recovers the per-tensor factor at GEMM time.
         */
        template <int kGroupSize>
        __global__ void dequantize_fp4_to_fp8_kernel(
            __nv_fp8_e4m3* __restrict__ output,
            const uint8_t* __restrict__ weights_packed,
            const float* __restrict__ scales,
            const float* __restrict__ weight_fp8_scale,
            int in_features )
        {
            const int output_channel = static_cast<int>( blockIdx.x );
            const int half_k = in_features / 2;
            const int num_groups = in_features / kGroupSize;
            const float inv_weight_scale = 1.0f / ( *weight_fp8_scale );

            const uint8_t* row_source = weights_packed + static_cast<ptrdiff_t>( output_channel ) * half_k;
            const float* row_scales = scales + static_cast<ptrdiff_t>( output_channel ) * num_groups;
            __nv_fp8_e4m3* row_dest = output + static_cast<ptrdiff_t>( output_channel ) * in_features;

            for ( int b = static_cast<int>( threadIdx.x ); b < half_k; b += static_cast<int>( blockDim.x ) )
            {
                const uint8_t byte = row_source[ b ];
                const float scale = row_scales[ b / ( kGroupSize / 2 ) ] * inv_weight_scale;

                // Packing contract: low nibble = element 2b, high nibble = element 2b + 1.
                row_dest[ 2 * b ]     = __nv_fp8_e4m3( fp4_e2m1_decode( byte & 0xFu ) * scale );
                row_dest[ 2 * b + 1 ] = __nv_fp8_e4m3( fp4_e2m1_decode( byte >> 4 ) * scale );
            }
        }

    } // anonymous namespace

    void cuda_w4a16_gemm(
        __nv_bfloat16*       output,
        const __nv_bfloat16* activations,
        const uint8_t*       weights_packed,
        const float*         scales,
        const uint8_t*       zero_points,
        const __nv_bfloat16* bias,
        int                  outer_size,
        int                  in_features,
        int                  out_features,
        int                  group_size,
        cudaStream_t         stream )
    {
        const dim3 block( kTileSize, kTileSize );
        const dim3 grid(
            ( static_cast<unsigned>( out_features ) + kTileSize - 1u ) / kTileSize,
            ( static_cast<unsigned>( outer_size    ) + kTileSize - 1u ) / kTileSize );

        switch ( group_size )
        {
            case 64:
                fused_w4a16_gemm_kernel<64><<<grid, block, 0, stream>>>(
                    output, activations, weights_packed, scales, zero_points, bias,
                    outer_size, out_features, in_features );
                break;

            case 128:
                fused_w4a16_gemm_kernel<128><<<grid, block, 0, stream>>>(
                    output, activations, weights_packed, scales, zero_points, bias,
                    outer_size, out_features, in_features );
                break;

            default:
                // Unsupported group_size — caller must validate before dispatch.
                break;
        }
    }

    void cuda_fp4a16_gemm(
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
        const dim3 block( kTileSize, kTileSize );
        const dim3 grid(
            ( static_cast<unsigned>( out_features ) + kTileSize - 1u ) / kTileSize,
            ( static_cast<unsigned>( outer_size    ) + kTileSize - 1u ) / kTileSize );

        switch ( group_size )
        {
            case 64:
                fused_w4a16_gemm_kernel<64, true><<<grid, block, 0, stream>>>(
                    output, activations, weights_packed, scales, nullptr, bias,
                    outer_size, out_features, in_features );
                break;

            case 128:
                fused_w4a16_gemm_kernel<128, true><<<grid, block, 0, stream>>>(
                    output, activations, weights_packed, scales, nullptr, bias,
                    outer_size, out_features, in_features );
                break;

            default:
                break;
        }
    }

    void cuda_fp4_dequantize_to_bf16(
        __nv_bfloat16* output,
        const uint8_t* weights_packed,
        const float*   scales,
        int            out_features,
        int            in_features,
        int            group_size,
        cudaStream_t   stream )
    {
        constexpr int kBlockSize = 256;
        const auto grid = static_cast<unsigned int>( out_features );

        switch ( group_size )
        {
            case 64:
                dequantize_fp4_to_bf16_kernel<64><<<grid, kBlockSize, 0, stream>>>(
                    output, weights_packed, scales, in_features );
                break;

            case 128:
                dequantize_fp4_to_bf16_kernel<128><<<grid, kBlockSize, 0, stream>>>(
                    output, weights_packed, scales, in_features );
                break;

            default:
                // Unsupported group_size -- caller must validate before dispatch.
                break;
        }
    }

    void cuda_compute_fp8_weight_scale(
        float*       weight_fp8_scale_out,
        const float* fp4_group_scales,
        int64_t      num_scales,
        cudaStream_t stream )
    {
        constexpr int kBlockSize = 256;
        int64_t blocks = ( num_scales + kBlockSize - 1 ) / kBlockSize;
        if ( blocks < 1 ) blocks = 1;
        if ( blocks > 1024 ) blocks = 1024;
        const auto grid = static_cast<unsigned int>( blocks );

        cudaMemsetAsync( weight_fp8_scale_out, 0, sizeof( float ), stream );
        fp8_weight_scale_reduce_kernel<<<grid, kBlockSize, 0, stream>>>(
            fp4_group_scales, num_scales, weight_fp8_scale_out );
        fp8_weight_scale_finalize_kernel<<<1, 1, 0, stream>>>( weight_fp8_scale_out );
    }

    void cuda_fp4_dequantize_to_fp8(
        __nv_fp8_e4m3* output,
        const uint8_t* weights_packed,
        const float*   scales,
        const float*   weight_fp8_scale,
        int            out_features,
        int            in_features,
        int            group_size,
        cudaStream_t   stream )
    {
        constexpr int kBlockSize = 256;
        const auto grid = static_cast<unsigned int>( out_features );

        switch ( group_size )
        {
            case 64:
                dequantize_fp4_to_fp8_kernel<64><<<grid, kBlockSize, 0, stream>>>(
                    output, weights_packed, scales, weight_fp8_scale, in_features );
                break;

            case 128:
                dequantize_fp4_to_fp8_kernel<128><<<grid, kBlockSize, 0, stream>>>(
                    output, weights_packed, scales, weight_fp8_scale, in_features );
                break;

            default:
                // Unsupported group_size -- caller must validate before dispatch.
                break;
        }
    }

} // namespace Mila::Dnn::Compute::Cuda::Linear
