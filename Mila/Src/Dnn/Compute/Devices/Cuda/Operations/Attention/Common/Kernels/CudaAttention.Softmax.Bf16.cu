/**
 * @file CudaAttention.Softmax.Bf16.cu
 * @brief BF16 softmax kernels for attention operations.
 *
 * Implements the BF16 softmax family declared in CudaAttention.cuh. All
 * arithmetic is promoted to float32; BF16 values are widened on load and
 * narrowed on store. These implementations are shared across MHA and GQA.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "CudaAttention.cuh"

namespace Mila::Dnn::Compute::Cuda::Attention::Common
{
    // ========================================================================
    // BF16 Softmax Kernels
    // ========================================================================

    __global__ void softmax_forward_bf16_kernel(
        __nv_bfloat16* att, float scale, const __nv_bfloat16* preatt,
        int B_NH, int T )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_rows = B_NH * T;

        if ( idx < total_rows )
        {
            int b_nh = idx / T;
            int t = idx % T;

            const __nv_bfloat16* preatt_row = preatt + b_nh * (T * T) + t * T;
            __nv_bfloat16* att_row = att + b_nh * (T * T) + t * T;

            float max_val = -INFINITY;

            for ( int t2 = 0; t2 <= t; ++t2 )
                max_val = fmaxf( max_val, __bfloat162float( preatt_row[ t2 ] ) );

            float sum = 0.0f;

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                float val = expf( (__bfloat162float( preatt_row[ t2 ] ) - max_val) * scale );
                sum += val;
                att_row[ t2 ] = __float2bfloat16( val );
            }

            float inv_sum = 1.0f / sum;

            for ( int t2 = 0; t2 <= t; ++t2 )
                att_row[ t2 ] = __float2bfloat16( __bfloat162float( att_row[ t2 ] ) * inv_sum );

            for ( int t2 = t + 1; t2 < T; ++t2 )
                att_row[ t2 ] = __float2bfloat16( 0.0f );
        }
    }

    __global__ void softmax_padded_forward_bf16_kernel(
        __nv_bfloat16* att, float scale, const __nv_bfloat16* preatt,
        int B_NH, int T, int actual_T )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_rows = B_NH * T;

        if ( idx < total_rows )
        {
            int b_nh = idx / T;
            int t = idx % T;

            __nv_bfloat16* att_row = att + b_nh * (T * T) + t * T;

            if ( t >= actual_T )
            {
                for ( int t2 = 0; t2 < T; ++t2 )
                    att_row[ t2 ] = __float2bfloat16( 0.0f );

                return;
            }

            const __nv_bfloat16* preatt_row = preatt + b_nh * (T * T) + t * T;

            int max_t2 = min( t, actual_T - 1 );
            float max_val = -INFINITY;

            for ( int t2 = 0; t2 <= max_t2; ++t2 )
                max_val = fmaxf( max_val, __bfloat162float( preatt_row[ t2 ] ) );

            float sum = 0.0f;

            for ( int t2 = 0; t2 <= max_t2; ++t2 )
            {
                float val = expf( (__bfloat162float( preatt_row[ t2 ] ) - max_val) * scale );
                sum += val;
                att_row[ t2 ] = __float2bfloat16( val );
            }

            float inv_sum = 1.0f / sum;

            for ( int t2 = 0; t2 <= max_t2; ++t2 )
                att_row[ t2 ] = __float2bfloat16( __bfloat162float( att_row[ t2 ] ) * inv_sum );

            for ( int t2 = max_t2 + 1; t2 < T; ++t2 )
                att_row[ t2 ] = __float2bfloat16( 0.0f );
        }
    }

    __global__ void softmax_decode_forward_bf16_kernel(
        __nv_bfloat16* att, float scale, const __nv_bfloat16* preatt,
        int B_NH, int max_len, int actual_len )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B_NH )
        {
            const __nv_bfloat16* preatt_row = preatt + idx * max_len;
            __nv_bfloat16* att_row = att + idx * max_len;

            float max_val = -INFINITY;

            for ( int t2 = 0; t2 < actual_len; ++t2 )
                max_val = fmaxf( max_val, __bfloat162float( preatt_row[ t2 ] ) );

            float sum = 0.0f;

            for ( int t2 = 0; t2 < actual_len; ++t2 )
            {
                float val = expf( (__bfloat162float( preatt_row[ t2 ] ) - max_val) * scale );
                sum += val;
                att_row[ t2 ] = __float2bfloat16( val );
            }

            float inv_sum = 1.0f / sum;

            for ( int t2 = 0; t2 < actual_len; ++t2 )
                att_row[ t2 ] = __float2bfloat16( __bfloat162float( att_row[ t2 ] ) * inv_sum );

            for ( int t2 = actual_len; t2 < max_len; ++t2 )
                att_row[ t2 ] = __float2bfloat16( 0.0f );
        }
    }

    __global__ void softmax_backward_bf16_kernel(
        __nv_bfloat16* dpreatt, const __nv_bfloat16* datt, const __nv_bfloat16* att,
        float scale,
        int B_NH, int T )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_rows = B_NH * T;

        if ( idx < total_rows )
        {
            int b_nh = idx / T;
            int t = idx % T;

            const __nv_bfloat16* att_row = att + b_nh * (T * T) + t * T;
            const __nv_bfloat16* datt_row = datt + b_nh * (T * T) + t * T;
            __nv_bfloat16* dpreatt_row = dpreatt + b_nh * (T * T) + t * T;

            float sum = 0.0f;

            for ( int t2 = 0; t2 <= t; ++t2 )
                sum += __bfloat162float( datt_row[ t2 ] ) * __bfloat162float( att_row[ t2 ] );

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                float grad = scale * __bfloat162float( att_row[ t2 ] ) *
                    (__bfloat162float( datt_row[ t2 ] ) - sum);
                dpreatt_row[ t2 ] = __float2bfloat16( grad );
            }

            for ( int t2 = t + 1; t2 < T; ++t2 )
                dpreatt_row[ t2 ] = __float2bfloat16( 0.0f );
        }
    }

    // ========================================================================
    // Host launchers
    // ========================================================================

    void cuda_attention_softmax_forward_bf16(
        __nv_bfloat16* att, float scale, const __nv_bfloat16* preatt,
        int B, int NH, int T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int B_NH = B * NH;
        const int num_blocks = ceil_div( B_NH * T, block_size );

        softmax_forward_bf16_kernel << < num_blocks, block_size, 0, stream >> > (
            att, scale, preatt, B_NH, T);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_attention_softmax_padded_forward_bf16(
        __nv_bfloat16* att, float scale, const __nv_bfloat16* preatt,
        int B, int NH, int max_T, int actual_T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int B_NH = B * NH;
        const int num_blocks = ceil_div( B_NH * max_T, block_size );

        softmax_padded_forward_bf16_kernel << < num_blocks, block_size, 0, stream >> > (
            att, scale, preatt, B_NH, max_T, actual_T);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_attention_softmax_decode_forward_bf16(
        __nv_bfloat16* att, float scale, const __nv_bfloat16* preatt,
        int B, int NH, int max_len, int actual_len,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int B_NH = B * NH;
        const int num_blocks = ceil_div( B_NH, block_size );

        softmax_decode_forward_bf16_kernel << < num_blocks, block_size, 0, stream >> > (
            att, scale, preatt, B_NH, max_len, actual_len);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_attention_softmax_backward_bf16(
        __nv_bfloat16* dpreatt, const __nv_bfloat16* datt, const __nv_bfloat16* att,
        float scale,
        int B, int NH, int T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int B_NH = B * NH;
        const int num_blocks = ceil_div( B_NH * T, block_size );

        softmax_backward_bf16_kernel << < num_blocks, block_size, 0, stream >> > (
            dpreatt, datt, att, scale, B_NH, T);

        cudaCheck( cudaGetLastError() );
    }
}