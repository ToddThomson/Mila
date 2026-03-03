/**
 * @file CudaAttention.Softmax.cu
 * @brief Shared CUDA softmax kernels for attention operations.
 *
 * Implements the softmax family declared in CudaAttention.cuh. These
 * implementations are identical across MHA and GQA and are compiled once.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "CudaAttention.cuh"

namespace Mila::Dnn::Compute::Cuda::Attention::Common
{
    // ========================================================================
    // FP32 Kernels
    // ========================================================================

    __global__ void softmax_forward_fp32_kernel(
        float* att, float scale, const float* preatt,
        int B_NH, int T )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_rows = B_NH * T;

        if ( idx < total_rows )
        {
            int b_nh = idx / T;
            int t = idx % T;

            const float* preatt_row = preatt + b_nh * (T * T) + t * T;
            float* att_row = att + b_nh * (T * T) + t * T;

            float max_val = -INFINITY;

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                max_val = fmaxf( max_val, preatt_row[ t2 ] );
            }

            float sum = 0.0f;

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                float val = expf( (preatt_row[ t2 ] - max_val) * scale );
                sum += val;
                att_row[ t2 ] = val;
            }

            float inv_sum = 1.0f / sum;

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                att_row[ t2 ] *= inv_sum;
            }

            for ( int t2 = t + 1; t2 < T; ++t2 )
            {
                att_row[ t2 ] = 0.0f;
            }
        }
    }

    __global__ void softmax_padded_forward_fp32_kernel(
        float* att, float scale, const float* preatt,
        int B_NH, int T, int actual_T )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_rows = B_NH * T;

        if ( idx < total_rows )
        {
            int b_nh = idx / T;
            int t = idx % T;

            float* att_row = att + b_nh * (T * T) + t * T;

            if ( t >= actual_T )
            {
                for ( int t2 = 0; t2 < T; ++t2 )
                    att_row[ t2 ] = 0.0f;

                return;
            }

            const float* preatt_row = preatt + b_nh * (T * T) + t * T;

            int   max_t2 = min( t, actual_T - 1 );
            float max_val = -INFINITY;

            for ( int t2 = 0; t2 <= max_t2; ++t2 )
            {
                max_val = fmaxf( max_val, preatt_row[ t2 ] );
            }

            float sum = 0.0f;

            for ( int t2 = 0; t2 <= max_t2; ++t2 )
            {
                float val = expf( (preatt_row[ t2 ] - max_val) * scale );
                sum += val;
                att_row[ t2 ] = val;
            }

            float inv_sum = 1.0f / sum;

            for ( int t2 = 0; t2 <= max_t2; ++t2 )
            {
                att_row[ t2 ] *= inv_sum;
            }

            for ( int t2 = max_t2 + 1; t2 < T; ++t2 )
            {
                att_row[ t2 ] = 0.0f;
            }
        }
    }

    __global__ void softmax_decode_forward_fp32_kernel(
        float* att, float scale, const float* preatt,
        int B_NH, int max_len, int actual_len )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B_NH )
        {
            const float* preatt_row = preatt + idx * max_len;
            float* att_row = att + idx * max_len;

            float max_val = -INFINITY;

            for ( int t2 = 0; t2 < actual_len; ++t2 )
            {
                max_val = fmaxf( max_val, preatt_row[ t2 ] );
            }

            float sum = 0.0f;

            for ( int t2 = 0; t2 < actual_len; ++t2 )
            {
                float val = expf( (preatt_row[ t2 ] - max_val) * scale );
                sum += val;
                att_row[ t2 ] = val;
            }

            float inv_sum = 1.0f / sum;

            for ( int t2 = 0; t2 < actual_len; ++t2 )
            {
                att_row[ t2 ] *= inv_sum;
            }

            for ( int t2 = actual_len; t2 < max_len; ++t2 )
            {
                att_row[ t2 ] = 0.0f;
            }
        }
    }

    __global__ void softmax_backward_fp32_kernel(
        float* dpreatt, const float* datt, const float* att,
        float scale,
        int B_NH, int T )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_rows = B_NH * T;

        if ( idx < total_rows )
        {
            int b_nh = idx / T;
            int t = idx % T;

            const float* att_row = att + b_nh * (T * T) + t * T;
            const float* datt_row = datt + b_nh * (T * T) + t * T;
            float* dpreatt_row = dpreatt + b_nh * (T * T) + t * T;

            float sum = 0.0f;

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                sum += datt_row[ t2 ] * att_row[ t2 ];
            }

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                dpreatt_row[ t2 ] = scale * att_row[ t2 ] * (datt_row[ t2 ] - sum);
            }

            for ( int t2 = t + 1; t2 < T; ++t2 )
            {
                dpreatt_row[ t2 ] = 0.0f;
            }
        }
    }

    // ========================================================================
    // FP16 Kernels
    // ========================================================================

    __global__ void softmax_forward_fp16_kernel(
        half* att, float scale, const half* preatt,
        int B_NH, int T )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_rows = B_NH * T;

        if ( idx < total_rows )
        {
            int b_nh = idx / T;
            int t = idx % T;

            const half* preatt_row = preatt + b_nh * (T * T) + t * T;
            half* att_row = att + b_nh * (T * T) + t * T;

            float max_val = -INFINITY;

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                max_val = fmaxf( max_val, __half2float( preatt_row[ t2 ] ) );
            }

            float sum = 0.0f;

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                float val = expf( (__half2float( preatt_row[ t2 ] ) - max_val) * scale );
                sum += val;
                att_row[ t2 ] = __float2half( val );
            }

            float inv_sum = 1.0f / sum;

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                att_row[ t2 ] = __float2half( __half2float( att_row[ t2 ] ) * inv_sum );
            }

            for ( int t2 = t + 1; t2 < T; ++t2 )
            {
                att_row[ t2 ] = __float2half( 0.0f );
            }
        }
    }

    __global__ void softmax_padded_forward_fp16_kernel(
        half* att, float scale, const half* preatt,
        int B_NH, int T, int actual_T )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_rows = B_NH * T;

        if ( idx < total_rows )
        {
            int b_nh = idx / T;
            int t = idx % T;

            half* att_row = att + b_nh * (T * T) + t * T;

            if ( t >= actual_T )
            {
                for ( int t2 = 0; t2 < T; ++t2 )
                    att_row[ t2 ] = __float2half( 0.0f );

                return;
            }

            const half* preatt_row = preatt + b_nh * (T * T) + t * T;

            int   max_t2 = min( t, actual_T - 1 );
            float max_val = -INFINITY;

            for ( int t2 = 0; t2 <= max_t2; ++t2 )
            {
                max_val = fmaxf( max_val, __half2float( preatt_row[ t2 ] ) );
            }

            float sum = 0.0f;

            for ( int t2 = 0; t2 <= max_t2; ++t2 )
            {
                float val = expf( (__half2float( preatt_row[ t2 ] ) - max_val) * scale );
                sum += val;
                att_row[ t2 ] = __float2half( val );
            }

            float inv_sum = 1.0f / sum;

            for ( int t2 = 0; t2 <= max_t2; ++t2 )
            {
                att_row[ t2 ] = __float2half( __half2float( att_row[ t2 ] ) * inv_sum );
            }

            for ( int t2 = max_t2 + 1; t2 < T; ++t2 )
            {
                att_row[ t2 ] = __float2half( 0.0f );
            }
        }
    }

    __global__ void softmax_decode_forward_fp16_kernel(
        half* att, float scale, const half* preatt,
        int B_NH, int max_len, int actual_len )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B_NH )
        {
            const half* preatt_row = preatt + idx * max_len;
            half* att_row = att + idx * max_len;

            float max_val = -INFINITY;

            for ( int t2 = 0; t2 < actual_len; ++t2 )
            {
                max_val = fmaxf( max_val, __half2float( preatt_row[ t2 ] ) );
            }

            float sum = 0.0f;

            for ( int t2 = 0; t2 < actual_len; ++t2 )
            {
                float val = expf( (__half2float( preatt_row[ t2 ] ) - max_val) * scale );
                sum += val;
                att_row[ t2 ] = __float2half( val );
            }

            float inv_sum = 1.0f / sum;

            for ( int t2 = 0; t2 < actual_len; ++t2 )
            {
                att_row[ t2 ] = __float2half( __half2float( att_row[ t2 ] ) * inv_sum );
            }

            for ( int t2 = actual_len; t2 < max_len; ++t2 )
            {
                att_row[ t2 ] = __float2half( 0.0f );
            }
        }
    }

    __global__ void softmax_backward_fp16_kernel(
        half* dpreatt, const half* datt, const half* att,
        float scale,
        int B_NH, int T )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_rows = B_NH * T;

        if ( idx < total_rows )
        {
            int b_nh = idx / T;
            int t = idx % T;

            const half* att_row = att + b_nh * (T * T) + t * T;
            const half* datt_row = datt + b_nh * (T * T) + t * T;
            half* dpreatt_row = dpreatt + b_nh * (T * T) + t * T;

            float sum = 0.0f;

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                sum += __half2float( datt_row[ t2 ] ) * __half2float( att_row[ t2 ] );
            }

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                float grad = scale * __half2float( att_row[ t2 ] ) * (__half2float( datt_row[ t2 ] ) - sum);
                dpreatt_row[ t2 ] = __float2half( grad );
            }

            for ( int t2 = t + 1; t2 < T; ++t2 )
            {
                dpreatt_row[ t2 ] = __float2half( 0.0f );
            }
        }
    }

    // ========================================================================
    // Host launch functions — FP32
    // ========================================================================

    void cuda_attention_softmax_forward_fp32(
        float* att, float scale, const float* preatt,
        int B, int NH, int T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int B_NH = B * NH;
        const int num_blocks = ceil_div( B_NH * T, block_size );

        softmax_forward_fp32_kernel << <num_blocks, block_size, 0, stream >> > (
            att, scale, preatt, B_NH, T);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_attention_softmax_padded_forward_fp32(
        float* att, float scale, const float* preatt,
        int B, int NH, int max_T, int actual_T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int B_NH = B * NH;
        const int num_blocks = ceil_div( B_NH * max_T, block_size );

        softmax_padded_forward_fp32_kernel << <num_blocks, block_size, 0, stream >> > (
            att, scale, preatt, B_NH, max_T, actual_T);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_attention_softmax_decode_forward_fp32(
        float* att, float scale, const float* preatt,
        int B, int NH, int max_len, int actual_len,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int B_NH = B * NH;
        const int num_blocks = ceil_div( B_NH, block_size );

        softmax_decode_forward_fp32_kernel << <num_blocks, block_size, 0, stream >> > (
            att, scale, preatt, B_NH, max_len, actual_len);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_attention_softmax_backward_fp32(
        float* dpreatt, const float* datt, const float* att,
        float scale,
        int B, int NH, int T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int B_NH = B * NH;
        const int num_blocks = ceil_div( B_NH * T, block_size );

        softmax_backward_fp32_kernel << <num_blocks, block_size, 0, stream >> > (
            dpreatt, datt, att, scale, B_NH, T);

        cudaCheck( cudaGetLastError() );
    }

    // ========================================================================
    // Host launch functions — FP16
    // ========================================================================

    void cuda_attention_softmax_forward_fp16(
        half* att, float scale, const half* preatt,
        int B, int NH, int T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int B_NH = B * NH;
        const int num_blocks = ceil_div( B_NH * T, block_size );

        softmax_forward_fp16_kernel << <num_blocks, block_size, 0, stream >> > (
            att, scale, preatt, B_NH, T);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_attention_softmax_padded_forward_fp16(
        half* att, float scale, const half* preatt,
        int B, int NH, int max_T, int actual_T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int B_NH = B * NH;
        const int num_blocks = ceil_div( B_NH * max_T, block_size );

        softmax_padded_forward_fp16_kernel << <num_blocks, block_size, 0, stream >> > (
            att, scale, preatt, B_NH, max_T, actual_T);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_attention_softmax_decode_forward_fp16(
        half* att, float scale, const half* preatt,
        int B, int NH, int max_len, int actual_len,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int B_NH = B * NH;
        const int num_blocks = ceil_div( B_NH, block_size );

        softmax_decode_forward_fp16_kernel << <num_blocks, block_size, 0, stream >> > (
            att, scale, preatt, B_NH, max_len, actual_len);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_attention_softmax_backward_fp16(
        half* dpreatt, const half* datt, const half* att,
        float scale,
        int B, int NH, int T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int B_NH = B * NH;
        const int num_blocks = ceil_div( B_NH * T, block_size );

        softmax_backward_fp16_kernel << <num_blocks, block_size, 0, stream >> > (
            dpreatt, datt, att, scale, B_NH, T);

        cudaCheck( cudaGetLastError() );
    }
}