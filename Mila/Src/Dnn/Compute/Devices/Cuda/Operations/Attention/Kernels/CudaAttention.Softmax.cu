/**
 * @file CudaAttention.Softmax.cu
 * @brief CUDA kernels for Multi - Head Attention auxiliary operations.
 *
 *Provides kernels for operations that cannot be handled by cuBLASLt :
 *
 * -Softmax with causal masking
 * -Backward pass for softmax
*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute::Cuda::Attention
{
    // ========================================================================
    // FP32 Softmax with Causal Masking and Scaling
    // ========================================================================

    /**
     * @brief Softmax forward with causal masking and scaling.
     *
     * Applies softmax to pre-attention scores with causal mask (t2 <= t).
     * Input: preatt [B*NH, T, T] (scaled attention scores)
     * Output: att [B*NH, T, T] (attention probabilities)
     */
    __global__ void softmax_forward_fp32_kernel(
        float* att, float scale, const float* preatt,
        int B_NH, int T )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_rows = B_NH * T;

        if ( idx < total_rows )
        {
            int t = idx % T;
            const float* preatt_row = preatt + idx * T;
            float* att_row = att + idx * T;

            // Pass 1: Find max (unscaled)
            float max_val = -INFINITY;
            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                max_val = fmaxf( max_val, preatt_row[ t2 ] );
            }

            // Pass 2: Compute exp((x - max) * scale) and sum
            float sum = 0.0f;
            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                float val = expf( (preatt_row[ t2 ] - max_val) * scale );
                sum += val;
                att_row[ t2 ] = val;
            }

            // Pass 3: Normalize
            float inv_sum = 1.0f / sum;
            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                att_row[ t2 ] *= inv_sum;
            }

            // Pass 4: Apply causal mask
            for ( int t2 = t + 1; t2 < T; ++t2 )
            {
                att_row[ t2 ] = 0.0f;
            }
        }
    }

    /**
    * @brief Softmax backward pass.
    *
    * Computes gradient of pre-softmax scores from gradient of softmax output.
    * Input: datt [B*NH, T, T], att [B*NH, T, T]
    * Output: dpreatt [B*NH, T, T]
    */
    __global__ void softmax_backward_fp32_kernel(
        float* dpreatt, const float* datt, const float* att,
        float scale,
        int B_NH, int T )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_rows = B_NH * T;

        if ( idx < total_rows )
        {
            int t = idx % T;
            const float* att_row = att + idx * T;
            const float* datt_row = datt + idx * T;
            float* dpreatt_row = dpreatt + idx * T;

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
    // FP16 Softmax with Causal Masking and Scaling
    // ========================================================================

    __global__ void softmax_forward_fp16_kernel(
        half* att, float scale, const half* preatt,
        int B_NH, int T )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_rows = B_NH * T;

        if ( idx < total_rows )
        {
            int t = idx % T;
            const half* preatt_row = preatt + idx * T;
            half* att_row = att + idx * T;

            // Find max (in float precision for numerical stability)
            float max_val = -INFINITY;
            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                max_val = fmaxf( max_val, __half2float( preatt_row[ t2 ] ) );
            }

            // Compute exp and sum
            float sum = 0.0f;
            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                float val = expf( (__half2float( preatt_row[ t2 ] ) - max_val) * scale );
                sum += val;
                att_row[ t2 ] = __float2half( val );
            }

            // Normalize
            float inv_sum = 1.0f / sum;
            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                att_row[ t2 ] = __float2half( __half2float( att_row[ t2 ] ) * inv_sum );
            }

            // Causal mask
            for ( int t2 = t + 1; t2 < T; ++t2 )
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
            int t = idx % T;
            const half* att_row = att + idx * T;
            const half* datt_row = datt + idx * T;
            half* dpreatt_row = dpreatt + idx * T;

            float sum = 0.0f;

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                sum += __half2float( datt_row[ t2 ] ) * __half2float( att_row[ t2 ] );
            }

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                float grad = __half2float( att_row[ t2 ] ) * (__half2float( datt_row[ t2 ] ) - sum);
                dpreatt_row[ t2 ] = __float2half( grad );
            }

            for ( int t2 = t + 1; t2 < T; ++t2 )
            {
                dpreatt_row[ t2 ] = __float2half( 0.0f );
            }
        }
    }

    // ========================================================================
    // Host Functions
    // ========================================================================

    void cuda_softmax_forward_fp32(
        float* att, float scale, const float* preatt,
        int B, int NH, int T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int B_NH = B * NH;
        int total_rows = B_NH * T;
        int num_blocks = ceil_div( total_rows, block_size );

        softmax_forward_fp32_kernel << <num_blocks, block_size, 0, stream >> > (att, scale, preatt, B_NH, T);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_softmax_backward_fp32(
        float* dpreatt, const float* datt, const float* att,
        float scale,
        int B, int NH, int T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int B_NH = B * NH;
        int total_rows = B_NH * T;
        int num_blocks = ceil_div( total_rows, block_size );

        softmax_backward_fp32_kernel << <num_blocks, block_size, 0, stream >> > (dpreatt, datt, att, scale, B_NH, T);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_softmax_forward_fp16(
        half* att, float scale, const half* preatt,
        int B, int NH, int T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int B_NH = B * NH;
        int total_rows = B_NH * T;
        int num_blocks = ceil_div( total_rows, block_size );

        softmax_forward_fp16_kernel << <num_blocks, block_size, 0, stream >> > (att, scale, preatt, B_NH, T);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_softmax_backward_fp16(
        half* dpreatt, const half* datt, const half* att,
        float scale,
        int B, int NH, int T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int B_NH = B * NH;
        int total_rows = B_NH * T;
        int num_blocks = ceil_div( total_rows, block_size );

        softmax_backward_fp16_kernel << <num_blocks, block_size, 0, stream >> > (dpreatt, datt, att, scale, B_NH, T);

        cudaCheck( cudaGetLastError() );
    }
}