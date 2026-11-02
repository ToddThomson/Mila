/**
 * @file CudaFp8Prefill.cu
 * @brief FP8 prefill CUDA kernels: scale application, dequantization, and the
 * per-token BF16 -> FP8 activation quantizer for the W4A8-FP8 prefill GEMM.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "CudaFp8Prefill.cuh"

namespace Mila::Dnn::Compute::Cuda::Linear
{
    // =========================================================================
    // cuda_fp8_apply_per_channel_scales
    // =========================================================================

    /**
     * One thread per output feature; iterates over all outer_size tokens.
     * Grid: ceil(out_features / kBlockSize) blocks of kBlockSize threads.
     */
    __global__ void apply_per_channel_scales_kernel(
        __nv_bfloat16* __restrict__ output,
        const float* __restrict__ scales,
        int outer_size,
        int out_features )
    {
        const int out = static_cast<int>( blockIdx.x ) * blockDim.x + static_cast<int>( threadIdx.x );
        if ( out >= out_features )
            return;

        const float scale = scales[ out ];

        for ( int t = 0; t < outer_size; ++t )
        {
            const int index = t * out_features + out;
            output[ index ] = __float2bfloat16( __bfloat162float( output[ index ] ) * scale );
        }
    }

    void cuda_fp8_apply_per_channel_scales(
        __nv_bfloat16* output,
        const float* scales,
        int outer_size,
        int out_features,
        cudaStream_t stream )
    {
        constexpr int kBlockSize = 256;
        const int grid_size = ( out_features + kBlockSize - 1 ) / kBlockSize;

        apply_per_channel_scales_kernel<<<grid_size, kBlockSize, 0, stream>>>(
            output, scales, outer_size, out_features );
    }

    // =========================================================================
    // cuda_fp8_dequantize_to_bf16
    // =========================================================================

    /**
     * One block per output channel (row of weight matrix).
     * Threads stride over in_features with step blockDim.x.
     * Grid: out_features blocks of kBlockSize threads.
     */
    __global__ void dequantize_fp8_to_bf16_kernel(
        __nv_bfloat16* __restrict__ output,
        const __nv_fp8_e4m3* __restrict__ input,
        const float* __restrict__ scales,
        int in_features )
    {
        const int output_channel = static_cast<int>( blockIdx.x );
        const float scale = scales[ output_channel ];

        const __nv_fp8_e4m3* row_source = input  + static_cast<ptrdiff_t>( output_channel ) * in_features;
        __nv_bfloat16*        row_dest   = output + static_cast<ptrdiff_t>( output_channel ) * in_features;

        for ( int k = static_cast<int>( threadIdx.x ); k < in_features; k += static_cast<int>( blockDim.x ) )
        {
            // FP8-E4M3 → half → float, apply per-channel scale, store as BF16.
            const float value = __half2float(
                __nv_cvt_fp8_to_halfraw( row_source[ k ].__x, __NV_E4M3 ) ) * scale;

            row_dest[ k ] = __float2bfloat16( value );
        }
    }

    void cuda_fp8_dequantize_to_bf16(
        __nv_bfloat16* output,
        const __nv_fp8_e4m3* input,
        const float* scales,
        int out_features,
        int in_features,
        cudaStream_t stream )
    {
        constexpr int kBlockSize = 256;

        dequantize_fp8_to_bf16_kernel<<<
            static_cast<unsigned int>( out_features ),
            kBlockSize,
            0,
            stream>>>(
                output, input, scales, in_features );
    }

    // =========================================================================
    // cuda_quantize_bf16_to_fp8_per_token
    // =========================================================================

    /**
     * One block per token (row). Phase 1: shared-memory absmax reduction over the
     * row. Phase 2 (same launch, after the reduction converges): every thread reads
     * the row scale from shared memory and quantizes its strided slice. Epsilon
     * guards an all-zero row from producing a zero scale (and a division by zero).
     * The E4M3 constructor saturates, but inputs are already scaled into
     * [-448, 448] so no clamping occurs in practice.
     */
    __global__ void fp8_activation_quantize_per_token_kernel(
        const __nv_bfloat16* __restrict__ input,
        __nv_fp8_e4m3* __restrict__ output,
        float* __restrict__ scales,
        int in_features )
    {
        __shared__ float shared_max[ 256 ];

        const int64_t row_offset = static_cast<int64_t>( blockIdx.x ) * in_features;
        const __nv_bfloat16* row_source = input + row_offset;
        __nv_fp8_e4m3* row_dest = output + row_offset;

        float local = 0.0f;

        for ( int k = static_cast<int>( threadIdx.x ); k < in_features; k += static_cast<int>( blockDim.x ) )
        {
            local = fmaxf( local, fabsf( __bfloat162float( row_source[ k ] ) ) );
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

        // The reduction loop ends on a __syncthreads(), so shared_max[0] is visible
        // to every thread here.
        const float scale = fmaxf( shared_max[ 0 ], 1e-12f ) / 448.0f;

        if ( threadIdx.x == 0 )
        {
            scales[ blockIdx.x ] = scale;
        }

        const float inv_scale = 1.0f / scale;

        for ( int k = static_cast<int>( threadIdx.x ); k < in_features; k += static_cast<int>( blockDim.x ) )
        {
            row_dest[ k ] = __nv_fp8_e4m3( __bfloat162float( row_source[ k ] ) * inv_scale );
        }
    }

    void cuda_quantize_bf16_to_fp8_per_token(
        __nv_fp8_e4m3* fp8_out,
        float*         scales_out,
        const __nv_bfloat16* input,
        int            outer_size,
        int            in_features,
        cudaStream_t   stream )
    {
        constexpr int kBlockSize = 256;

        fp8_activation_quantize_per_token_kernel<<<
            static_cast<unsigned int>( outer_size ),
            kBlockSize,
            0,
            stream>>>(
                input, fp8_out, scales_out, in_features );
    }

    // =========================================================================
    // cuda_fp8_apply_per_token_scales
    // =========================================================================

    /**
     * One block per token (row); threads stride over out_features. The row scale is
     * loaded once per block. Bias is folded in when present (nullptr otherwise).
     */
    __global__ void apply_per_token_scales_kernel(
        __nv_bfloat16* __restrict__ output,
        const float* __restrict__ scales,
        const __nv_bfloat16* __restrict__ bias,
        int out_features )
    {
        const float scale = scales[ blockIdx.x ];
        __nv_bfloat16* row = output + static_cast<int64_t>( blockIdx.x ) * out_features;

        for ( int out = static_cast<int>( threadIdx.x ); out < out_features; out += static_cast<int>( blockDim.x ) )
        {
            float value = __bfloat162float( row[ out ] ) * scale;

            if ( bias != nullptr )
            {
                value += __bfloat162float( bias[ out ] );
            }

            row[ out ] = __float2bfloat16( value );
        }
    }

    void cuda_fp8_apply_per_token_scales(
        __nv_bfloat16* output,
        const float* scales,
        const __nv_bfloat16* bias,
        int outer_size,
        int out_features,
        cudaStream_t stream )
    {
        constexpr int kBlockSize = 256;

        apply_per_token_scales_kernel<<<
            static_cast<unsigned int>( outer_size ),
            kBlockSize,
            0,
            stream>>>(
                output, scales, bias, out_features );
    }

    // =========================================================================
    // cuda_add_bias
    // =========================================================================

    /**
     * One thread per output feature; iterates over all outer_size tokens.
     * Grid: ceil(out_features / kBlockSize) blocks of kBlockSize threads.
     */
    __global__ void add_bias_kernel(
        __nv_bfloat16* __restrict__ output,
        const __nv_bfloat16* __restrict__ bias,
        int outer_size,
        int out_features )
    {
        const int out = static_cast<int>( blockIdx.x ) * blockDim.x + static_cast<int>( threadIdx.x );
        if ( out >= out_features )
            return;

        const float bias_val = __bfloat162float( bias[ out ] );

        for ( int t = 0; t < outer_size; ++t )
        {
            const int index = t * out_features + out;
            output[ index ] = __float2bfloat16( __bfloat162float( output[ index ] ) + bias_val );
        }
    }

    void cuda_add_bias(
        __nv_bfloat16* output,
        const __nv_bfloat16* bias,
        int outer_size,
        int out_features,
        cudaStream_t stream )
    {
        constexpr int kBlockSize = 256;
        const int grid_size = ( out_features + kBlockSize - 1 ) / kBlockSize;

        add_bias_kernel<<<grid_size, kBlockSize, 0, stream>>>(
            output, bias, outer_size, out_features );
    }

    // FP32 counterpart for the non-quantized prefill path (see header note).
    __global__ void add_bias_kernel_f32(
        float* __restrict__ output,
        const float* __restrict__ bias,
        int outer_size,
        int out_features )
    {
        const int out = static_cast<int>( blockIdx.x ) * blockDim.x + static_cast<int>( threadIdx.x );
        if ( out >= out_features )
            return;

        const float bias_val = bias[ out ];

        for ( int t = 0; t < outer_size; ++t )
        {
            const int index = t * out_features + out;
            output[ index ] += bias_val;
        }
    }

    void cuda_add_bias(
        float* output,
        const float* bias,
        int outer_size,
        int out_features,
        cudaStream_t stream )
    {
        constexpr int kBlockSize = 256;
        const int grid_size = ( out_features + kBlockSize - 1 ) / kBlockSize;

        add_bias_kernel_f32<<<grid_size, kBlockSize, 0, stream>>>(
            output, bias, outer_size, out_features );
    }
}
