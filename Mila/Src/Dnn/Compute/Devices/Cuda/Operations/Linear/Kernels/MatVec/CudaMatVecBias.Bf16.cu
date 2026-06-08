/**
 * @file CudaMatVecBias.Bf16.cu
 * @brief BF16 matrix-vector multiply for the M=1 decode path. Includes BF16-weight,
 *        FP8-E4M3-weight (per-channel), and FP4-E2M1-weight (per-group) variants.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "device_launch_parameters.h"
#include <cassert>
#include <cstdint>

namespace Mila::Dnn::Compute::Cuda::Linear
{
    namespace
    {
        // FP4 E2M1 nibble decode: bit3=sign, bits[2:1]=exponent, bit0=mantissa.
        // Positive nibbles 0-7: {0, 0.5, 1, 1.5, 2, 3, 4, 6}; negatives are sign-magnitude.
        __device__ __forceinline__ float fp4_e2m1_decode( uint8_t nibble )
        {
            static constexpr float kLut[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
            const float mag = kLut[ nibble & 0x7u ];
            return ( nibble & 0x8u ) ? -mag : mag;
        }
    } // anonymous namespace
    // Loads 4 BF16 elements as two __nv_bfloat162 pairs via an int2 (8-byte) load.
    // Requires ptr to be 8-byte aligned, guaranteed when C % 4 == 0.
    __device__ inline void ld_bf16x4(
        __nv_bfloat162& lo,
        __nv_bfloat162& hi,
        const __nv_bfloat16* ptr )
    {
        int2 raw = *reinterpret_cast<const int2*>(ptr);
        lo = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
        hi = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
    }

    // Loads 8 FP8-E4M3 elements via a single int2 (8-byte) load and converts to float4x2.
    // Requires ptr to be 8-byte aligned, guaranteed when C % 8 == 0.
    __device__ inline void ld_fp8x8_to_float(
        float (&out)[8],
        const __nv_fp8_e4m3* ptr )
    {
        int2 raw = *reinterpret_cast<const int2*>(ptr);
        const __nv_fp8_e4m3* elems = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);

#pragma unroll
        for ( int i = 0; i < 8; ++i )
            out[i] = (float)elems[i];
    }

    /**
     * @brief Optimized CUDA kernel for BF16 matrix-vector multiply (M=1 decode path).
     *
     * Computes y[oc] = sum(x[c] * weight[oc, c]) + bias[oc] for each oc independently.
     * Dot-product accumulation is performed in float32 to preserve precision.
     *
     * Grid:  (ceil(OC / kMatvecBlockOC), 1)
     * Block: (kMatvecThreadsPerOC, kMatvecBlockOC) — 32 threads collaborate per output element
     *
     * OC need not be a multiple of kMatvecBlockOC; threads with oc >= OC exit early.
     *
     * Requirements:
     *   - C must be divisible by 4 (int2 loads require 8-byte alignment on the weight row)
     *   - kMatvecThreadsPerOC must be a power of 2 <= 32 (warp shuffle)
     */
    static constexpr int kMatvecThreadsPerOC = 32;
    static constexpr int kMatvecBlockOC = 8;

    __global__ void __launch_bounds__( kMatvecThreadsPerOC* kMatvecBlockOC )
        matvec_decode_bf16_kernel(
            __nv_bfloat16* __restrict__       y,
            const __nv_bfloat16* __restrict__ x,
            const __nv_bfloat16* __restrict__ weight,
            const __nv_bfloat16* __restrict__ bias,
            int C,
            int OC )
    {
        const int oc_base = blockIdx.x * kMatvecBlockOC;
        const int oc = oc_base + threadIdx.y;

        if ( oc >= OC )
            return;

        const __nv_bfloat16* w_row = weight + oc * C;

        float acc = 0.0f;
        const int c_start = threadIdx.x * 4;
        const int c_step = kMatvecThreadsPerOC * 4;

        for ( int c = c_start; c < C; c += c_step )
        {
            __nv_bfloat162 x_lo, x_hi, w_lo, w_hi;
            ld_bf16x4( x_lo, x_hi, x + c );
            ld_bf16x4( w_lo, w_hi, w_row + c );

            float2 x_lo_f = __bfloat1622float2( x_lo );
            float2 w_lo_f = __bfloat1622float2( w_lo );
            float2 x_hi_f = __bfloat1622float2( x_hi );
            float2 w_hi_f = __bfloat1622float2( w_hi );

            acc += x_lo_f.x * w_lo_f.x + x_lo_f.y * w_lo_f.y
                + x_hi_f.x * w_hi_f.x + x_hi_f.y * w_hi_f.y;
        }

#pragma unroll
        for ( int offset = kMatvecThreadsPerOC / 2; offset > 0; offset >>= 1 )
        {
            acc += __shfl_down_sync( 0xffffffff, acc, offset );
        }

        if ( threadIdx.x == 0 )
        {
            float bias_val = (bias != nullptr) ? __bfloat162float( bias[ oc ] ) : 0.0f;
            y[ oc ] = __float2bfloat16( acc + bias_val );
        }
    }

    /**
     * @brief Decode-path matvec with BF16 activations and FP8-E4M3 weights.
     *
     * Computes y[oc] = scale[oc] * sum(x[c] * (float)weight[oc,c]) + bias[oc].
     * Weights are stored as FP8-E4M3 with one float32 scale per output channel.
     * Accumulation is performed in float32; the per-channel scale is applied after
     * the warp reduction to minimize the number of multiply instructions per thread.
     *
     * Grid:  (ceil(OC / kMatvecBlockOC), 1)
     * Block: (kMatvecThreadsPerOC, kMatvecBlockOC)
     *
     * Requirements:
     *   - C must be divisible by 8 (int2 loads pack 8 FP8 bytes)
     *   - kMatvecThreadsPerOC must be a power of 2 <= 32
     */
    __global__ void __launch_bounds__( kMatvecThreadsPerOC* kMatvecBlockOC )
        matvec_decode_bf16_qfp8_kernel(
            __nv_bfloat16* __restrict__        y,
            const __nv_bfloat16* __restrict__  x,
            const __nv_fp8_e4m3* __restrict__  weight,
            const float* __restrict__          scales,
            const __nv_bfloat16* __restrict__  bias,
            int C,
            int OC )
    {
        const int oc_base = blockIdx.x * kMatvecBlockOC;
        const int oc = oc_base + threadIdx.y;

        if ( oc >= OC )
            return;

        const __nv_fp8_e4m3* w_row = weight + oc * C;

        float acc = 0.0f;
        const int c_start = threadIdx.x * 8;
        const int c_step = kMatvecThreadsPerOC * 8;

        for ( int c = c_start; c < C; c += c_step )
        {
            __nv_bfloat162 x_lo, x_hi_0, x_hi_1, x_hi_2;
            ld_bf16x4( x_lo, x_hi_0, x + c );
            ld_bf16x4( x_hi_1, x_hi_2, x + c + 4 );

            float w_f[8];
            ld_fp8x8_to_float( w_f, w_row + c );

            float2 x0 = __bfloat1622float2( x_lo );
            float2 x1 = __bfloat1622float2( x_hi_0 );
            float2 x2 = __bfloat1622float2( x_hi_1 );
            float2 x3 = __bfloat1622float2( x_hi_2 );

            acc += x0.x * w_f[0] + x0.y * w_f[1]
                + x1.x * w_f[2] + x1.y * w_f[3]
                + x2.x * w_f[4] + x2.y * w_f[5]
                + x3.x * w_f[6] + x3.y * w_f[7];
        }

#pragma unroll
        for ( int offset = kMatvecThreadsPerOC / 2; offset > 0; offset >>= 1 )
        {
            acc += __shfl_down_sync( 0xffffffff, acc, offset );
        }

        if ( threadIdx.x == 0 )
        {
            float bias_val = (bias != nullptr) ? __bfloat162float( bias[ oc ] ) : 0.0f;
            y[ oc ] = __float2bfloat16( scales[ oc ] * acc + bias_val );
        }
    }

    /**
     * @brief Decode-path matvec with BF16 activations and FP4-E2M1 weights (per-group scale).
     *
     * Computes y[oc] = sum_c( x[c] * fp4_decode(W[oc,c]) * scale[oc, c/kGroupSize] ) + bias[oc].
     *
     * Weights are stored as packed FP4-E2M1 nibbles (2 per byte, low=even col, high=odd col).
     * Each thread processes 8 nibbles (4 packed bytes) per inner loop iteration.
     * Because kGroupSize ∈ {64, 128} is a multiple of 8 and c_start = tid*8, all 8 nibbles in
     * a chunk are guaranteed to belong to the same quantization group — only one scale load
     * is needed per 8-element chunk.
     *
     * Grid:  (ceil(OC / kMatvecBlockOC), 1)
     * Block: (kMatvecThreadsPerOC, kMatvecBlockOC)
     *
     * Requirements:
     *   - C must be divisible by 8 (int2/uint32 alignment and nibble packing)
     *   - kGroupSize must be a multiple of 8 (64 or 128 satisfy this)
     */
    template<int kGroupSize>
    __global__ void __launch_bounds__( kMatvecThreadsPerOC* kMatvecBlockOC )
        matvec_decode_bf16_qfp4_kernel(
            __nv_bfloat16* __restrict__       y,
            const __nv_bfloat16* __restrict__ x,
            const uint8_t* __restrict__       weights_packed,
            const float* __restrict__         scales,
            const __nv_bfloat16* __restrict__ bias,
            int C,
            int OC )
    {
        const int oc_base = blockIdx.x * kMatvecBlockOC;
        const int oc      = oc_base + threadIdx.y;

        if ( oc >= OC ) return;

        const uint8_t* w_row     = weights_packed + oc * ( C / 2 );
        const int      num_groups = C / kGroupSize;

        float acc = 0.0f;

        // Each thread processes 8 nibbles (4 packed bytes) per iteration.
        // c indexes the BF16 activation / FP4 nibble dimension.
        const int c_start = threadIdx.x * 8;
        const int c_step  = kMatvecThreadsPerOC * 8;  // 32 * 8 = 256

        for ( int c = c_start; c < C; c += c_step )
        {
            // Load 4 packed weight bytes (8 nibbles) — coalesced: thread k accesses byte offset k*4.
            const uint32_t w_packed = *reinterpret_cast<const uint32_t*>( w_row + c / 2 );

            // Load 8 BF16 activations via two int2 loads (same pattern as qfp8 kernel).
            __nv_bfloat162 x_lo, x_hi, x_lo2, x_hi2;
            ld_bf16x4( x_lo,  x_hi,  x + c );
            ld_bf16x4( x_lo2, x_hi2, x + c + 4 );

            // All 8 nibbles in [c, c+7] share the same group (guaranteed by kGroupSize % 8 == 0).
            const float scale = scales[ oc * num_groups + c / kGroupSize ];

            // Unpack 4 bytes into 8 decoded weight floats.
            // byte[i] layout: low nibble = W[c + 2i], high nibble = W[c + 2i + 1]
            float w_f[8];
#pragma unroll
            for ( int b = 0; b < 4; ++b )
            {
                const uint8_t byte  = ( w_packed >> ( b * 8 ) ) & 0xFFu;
                w_f[ 2 * b     ] = fp4_e2m1_decode( byte & 0xFu ) * scale;
                w_f[ 2 * b + 1 ] = fp4_e2m1_decode( byte >> 4   ) * scale;
            }

            const float2 x0 = __bfloat1622float2( x_lo  );
            const float2 x1 = __bfloat1622float2( x_hi  );
            const float2 x2 = __bfloat1622float2( x_lo2 );
            const float2 x3 = __bfloat1622float2( x_hi2 );

            acc += x0.x * w_f[0] + x0.y * w_f[1]
                 + x1.x * w_f[2] + x1.y * w_f[3]
                 + x2.x * w_f[4] + x2.y * w_f[5]
                 + x3.x * w_f[6] + x3.y * w_f[7];
        }

#pragma unroll
        for ( int offset = kMatvecThreadsPerOC / 2; offset > 0; offset >>= 1 )
        {
            acc += __shfl_down_sync( 0xffffffff, acc, offset );
        }

        if ( threadIdx.x == 0 )
        {
            const float bias_val = ( bias != nullptr ) ? __bfloat162float( bias[ oc ] ) : 0.0f;
            y[ oc ] = __float2bfloat16( acc + bias_val );
        }
    }

    void cuda_matvec_decode_bf16(
        __nv_bfloat16* y,
        const __nv_bfloat16* x,
        const __nv_bfloat16* weight,
        const __nv_bfloat16* bias,
        int C,
        int OC,
        cudaStream_t stream )
    {
        assert( C % 4 == 0 && "cuda_matvec_decode_bf16: C must be divisible by 4 for int2/bfloat162 loads" );

        const dim3 block( kMatvecThreadsPerOC, kMatvecBlockOC );
        const dim3 grid( (OC + kMatvecBlockOC - 1) / kMatvecBlockOC );

        matvec_decode_bf16_kernel<<<grid, block, 0, stream>>>( y, x, weight, bias, C, OC );
    }

    void cuda_matvec_decode_bf16_qfp8(
        __nv_bfloat16* y,
        const __nv_bfloat16* x,
        const __nv_fp8_e4m3* weight,
        const float* scales,
        const __nv_bfloat16* bias,
        int C,
        int OC,
        cudaStream_t stream )
    {
        assert( C % 8 == 0 && "cuda_matvec_decode_bf16_qfp8: C must be divisible by 8 for int2 FP8 loads" );

        const dim3 block( kMatvecThreadsPerOC, kMatvecBlockOC );
        const dim3 grid( (OC + kMatvecBlockOC - 1) / kMatvecBlockOC );

        matvec_decode_bf16_qfp8_kernel<<<grid, block, 0, stream>>>( y, x, weight, scales, bias, C, OC );
    }

    void cuda_matvec_decode_bf16_qfp4(
        __nv_bfloat16*       y,
        const __nv_bfloat16* x,
        const uint8_t*       weights_packed,
        const float*         scales,
        const __nv_bfloat16* bias,
        int                  C,
        int                  OC,
        int                  group_size,
        cudaStream_t         stream )
    {
        assert( C % 8 == 0 && "cuda_matvec_decode_bf16_qfp4: C must be divisible by 8" );

        const dim3 block( kMatvecThreadsPerOC, kMatvecBlockOC );
        const dim3 grid( ( OC + kMatvecBlockOC - 1 ) / kMatvecBlockOC );

        switch ( group_size )
        {
            case 64:
                matvec_decode_bf16_qfp4_kernel<64><<<grid, block, 0, stream>>>(
                    y, x, weights_packed, scales, bias, C, OC );
                break;

            case 128:
                matvec_decode_bf16_qfp4_kernel<128><<<grid, block, 0, stream>>>(
                    y, x, weights_packed, scales, bias, C, OC );
                break;

            default:
                assert( false && "cuda_matvec_decode_bf16_qfp4: unsupported group_size (must be 64 or 128)" );
                break;
        }
    }
}