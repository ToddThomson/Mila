/**
 * @file CudaFp8Prefill.cu
 * @brief FP8 prefill CUDA kernels: per-channel scale application and FP8→BF16 dequantization.
 *
 * Two kernels support the FP8 prefill GEMM paths in CudaLinearOp:
 *
 *  - cuda_fp8_apply_per_channel_scales: post-GEMM correction for the native FP8
 *    cuBLASLt path. cuBLASLt accepts only a per-tensor scale at execute time; we
 *    set that to 1.0f and apply the true per-channel scales here.
 *
 *  - cuda_fp8_dequantize_to_bf16: FP8-E4M3 weight matrix → BF16 conversion for
 *    the fallback path, producing a temporary BF16 copy that feeds the standard
 *    BF16 cuBLASLt GEMM plan.
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
}
