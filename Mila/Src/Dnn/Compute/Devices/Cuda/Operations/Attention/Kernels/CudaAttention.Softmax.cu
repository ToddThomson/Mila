/**
 * @file CudaAttention.Softmax.cu
 * @brief CUDA kernels for Multi-Head Attention auxiliary operations.
 *
 * Provides kernels for operations that cannot be handled by cuBLASLt:
 * - Softmax with causal masking
 * - Backward pass for softmax
 *
 * All operations use row-major layout exclusively:
 * - Attention matrices: [B*NH, T, T] where each row is contiguous
 * - Indexing: element at (row=t, col=t2) is at index t*T + t2
 * - Each query position t attends to key positions 0..t (causal masking)
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute::Cuda::Attention
{
    // DEBUG: FP32 Softmax with Causal Masking and Scaling, with Padding Support
    __global__ void softmax_padded_forward_fp32_kernel(
        float* att, float scale, const float* preatt,
        int B_NH, int T, int actual_T )  // Add actual_T parameter
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_rows = B_NH * T;
        if ( idx < total_rows )
        {
            int b_nh = idx / T;
            int t = idx % T;

            // Only process rows for real tokens
            if ( t >= actual_T )
            {
                // This is a padding position - zero out attention
                float* att_row = att + b_nh * (T * T) + t * T;
                for ( int t2 = 0; t2 < T; ++t2 )
                {
                    att_row[ t2 ] = 0.0f;
                }
                return;
            }

            const float* preatt_matrix = preatt + b_nh * (T * T);
            float* att_matrix = att + b_nh * (T * T);
            const float* preatt_row = preatt_matrix + t * T;
            float* att_row = att_matrix + t * T;

            float max_val = -INFINITY;
            // Only attend to real tokens (up to actual_T) and causal positions
            int max_t2 = min( t, actual_T - 1 );
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

            // Zero out everything after real tokens
            for ( int t2 = max_t2 + 1; t2 < T; ++t2 )
            {
                att_row[ t2 ] = 0.0f;
            }
        }
    }

    // ========================================================================
    // FP32 Softmax with Causal Masking and Scaling
    // ========================================================================

    /**
     * @brief Softmax forward pass with causal masking (FP32).
     *
     * Applies scaled softmax to preatt and writes to att, with causal masking.
     * Each thread processes one row (query position) of the attention matrix.
     *
     * Preconditions:
     * - preatt size >= B*NH * T * T
     * - att size >= B*NH * T * T
     *
     * Side-effects:
     * - Writes to device buffer att.
     *
     * @param att Output attention weights [B*NH, T, T].
     * @param scale Scaling factor (typically 1/sqrt(head_size)).
     * @param preatt Input pre-softmax scores [B*NH, T, T].
     * @param B_NH Combined batch and head count (B * NH).
     * @param T Sequence length.
     */
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

            const float* preatt_matrix = preatt + b_nh * (T * T);
            float* att_matrix = att + b_nh * (T * T);

            const float* preatt_row = preatt_matrix + t * T;
            float* att_row = att_matrix + t * T;

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

    /**
     * @brief Softmax backward pass (FP32).
     *
     * Computes gradient of pre-softmax scores from gradient of softmax output.
     * Each thread processes one row of the gradient matrices.
     *
     * Preconditions:
     * - datt size >= B*NH * T * T
     * - att size >= B*NH * T * T
     * - dpreatt size >= B*NH * T * T
     *
     * Side-effects:
     * - Writes to device buffer dpreatt.
     *
     * @param dpreatt Output gradient of pre-softmax scores [B*NH, T, T].
     * @param datt Input gradient of attention weights [B*NH, T, T].
     * @param att Input attention weights [B*NH, T, T].
     * @param scale Scaling factor (typically 1/sqrt(head_size)).
     * @param B_NH Combined batch and head count (B * NH).
     * @param T Sequence length.
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
            int b_nh = idx / T;
            int t = idx % T;

            const float* att_matrix = att + b_nh * (T * T);
            const float* datt_matrix = datt + b_nh * (T * T);
            float* dpreatt_matrix = dpreatt + b_nh * (T * T);

            const float* att_row = att_matrix + t * T;
            const float* datt_row = datt_matrix + t * T;
            float* dpreatt_row = dpreatt_matrix + t * T;

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

    /**
     * @brief Softmax forward pass with causal masking (FP16).
     *
     * FP16 variant of softmax_forward_fp32_kernel.
     * Computations performed in FP32 for numerical stability.
     *
     * Preconditions:
     * - preatt size >= B*NH * T * T
     * - att size >= B*NH * T * T
     *
     * Side-effects:
     * - Writes to device buffer att.
     *
     * @param att Output attention weights [B*NH, T, T].
     * @param scale Scaling factor (typically 1/sqrt(head_size)).
     * @param preatt Input pre-softmax scores [B*NH, T, T].
     * @param B_NH Combined batch and head count (B * NH).
     * @param T Sequence length.
     */
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

            const half* preatt_matrix = preatt + b_nh * (T * T);
            half* att_matrix = att + b_nh * (T * T);

            const half* preatt_row = preatt_matrix + t * T;
            half* att_row = att_matrix + t * T;

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

    /**
     * @brief Softmax backward pass (FP16).
     *
     * FP16 variant of softmax_backward_fp32_kernel.
     * Computations performed in FP32 for numerical stability.
     *
     * Preconditions:
     * - datt size >= B*NH * T * T
     * - att size >= B*NH * T * T
     * - dpreatt size >= B*NH * T * T
     *
     * Side-effects:
     * - Writes to device buffer dpreatt.
     *
     * @param dpreatt Output gradient of pre-softmax scores [B*NH, T, T].
     * @param datt Input gradient of attention weights [B*NH, T, T].
     * @param att Input attention weights [B*NH, T, T].
     * @param scale Scaling factor (typically 1/sqrt(head_size)).
     * @param B_NH Combined batch and head count (B * NH).
     * @param T Sequence length.
     */
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

            const half* att_matrix = att + b_nh * (T * T);
            const half* datt_matrix = datt + b_nh * (T * T);
            half* dpreatt_matrix = dpreatt + b_nh * (T * T);

            const half* att_row = att_matrix + t * T;
            const half* datt_row = datt_matrix + t * T;
            half* dpreatt_row = dpreatt_matrix + t * T;

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

            if ( t >= actual_T )
            {
                half* att_row = att + b_nh * (T * T) + t * T;

                for ( int t2 = 0; t2 < T; ++t2 )
                {
                    att_row[ t2 ] = __float2half( 0.0f );
                }

                return;
            }

            const half* preatt_matrix = preatt + b_nh * (T * T);
            half* att_matrix = att + b_nh * (T * T);
            const half* preatt_row = preatt_matrix + t * T;
            half* att_row = att_matrix + t * T;

            float max_val = -INFINITY;
            int max_t2 = min( t, actual_T - 1 );

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

    // ========================================================================
    // Host Functions
    // ========================================================================

    /**
     * @brief Launch softmax_forward_fp32_kernel with causal masking.
     *
     * @param att Output attention weights [B*NH, T, T].
     * @param scale Scaling factor (typically 1/sqrt(head_size)).
     * @param preatt Input pre-softmax scores [B*NH, T, T].
     * @param B Batch size.
     * @param NH Number of heads.
     * @param T Sequence length.
     * @param stream CUDA stream to launch the kernel on.
     */
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

    /**
     * @brief Launch softmax_backward_fp32_kernel.
     *
     * @param dpreatt Output gradient of pre-softmax scores [B*NH, T, T].
     * @param datt Input gradient of attention weights [B*NH, T, T].
     * @param att Input attention weights [B*NH, T, T].
     * @param scale Scaling factor (typically 1/sqrt(head_size)).
     * @param B Batch size.
     * @param NH Number of heads.
     * @param T Sequence length.
     * @param stream CUDA stream to launch the kernel on.
     */
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

    /**
     * @brief Launch softmax_forward_fp16_kernel with causal masking.
     *
     * @param att Output attention weights [B*NH, T, T].
     * @param scale Scaling factor (typically 1/sqrt(head_size)).
     * @param preatt Input pre-softmax scores [B*NH, T, T].
     * @param B Batch size.
     * @param NH Number of heads.
     * @param T Sequence length.
     * @param stream CUDA stream to launch the kernel on.
     */
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

    /**
     * @brief Launch softmax_backward_fp16_kernel.
     *
     * @param dpreatt Output gradient of pre-softmax scores [B*NH, T, T].
     * @param datt Input gradient of attention weights [B*NH, T, T].
     * @param att Input attention weights [B*NH, T, T].
     * @param scale Scaling factor (typically 1/sqrt(head_size)).
     * @param B Batch size.
     * @param NH Number of heads.
     * @param T Sequence length.
     * @param stream CUDA stream to launch the kernel on.
     */
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

    // ========================================================================
    // Padded and Decoder Softmax Functions
    // ========================================================================

    /**
     * @brief Launch softmax_padded_forward_fp32_kernel with causal masking.
     *
     * @param att Output attention weights [B*NH, T, T].
     * @param scale Scaling factor (typically 1/sqrt(head_size)).
     * @param preatt Input pre-softmax scores [B*NH, T, T].
     * @param B Batch size.
     * @param NH Number of heads.
     * @param max_T Maximum sequence length (padded).
     * @param actual_T Actual sequence length (non-padded).
     * @param stream CUDA stream to launch the kernel on.
     */
    void cuda_softmax_padded_forward_fp32(
        float* att, float scale, const float* preatt,
        int B, int NH, int max_T, int actual_T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int B_NH = B * NH;
        int total_rows = B_NH * max_T;
        int num_blocks = ceil_div( total_rows, block_size );

        softmax_padded_forward_fp32_kernel << <num_blocks, block_size, 0, stream >> > (att, scale, preatt, B_NH, max_T, actual_T);

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Launch softmax_padded_forward_fp16_kernel with causal masking.
     *
     * @param att Output attention weights [B*NH, T, T].
     * @param scale Scaling factor (typically 1/sqrt(head_size)).
     * @param preatt Input pre-softmax scores [B*NH, T, T].
     * @param B Batch size.
     * @param NH Number of heads.
     * @param max_T Maximum sequence length (padded).
     * @param actual_T Actual sequence length (non-padded).
     * @param stream CUDA stream to launch the kernel on.
     */
    void cuda_softmax_padded_forward_fp16(
        half* att, float scale, const half* preatt,
        int B, int NH, int max_T, int actual_T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int B_NH = B * NH;
        int total_rows = B_NH * max_T;
        int num_blocks = ceil_div( total_rows, block_size );

        softmax_padded_forward_fp16_kernel << <num_blocks, block_size, 0, stream >> > (att, scale, preatt, B_NH, max_T, actual_T);

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Launch softmax_decode_forward_fp32_kernel.
     *
     * @param att Output attention weights [B*NH, T, T].
     * @param scale Scaling factor (typically 1/sqrt(head_size)).
     * @param preatt Input pre-softmax scores [B*NH, T, T].
     * @param B Batch size.
     * @param NH Number of heads.
     * @param max_len Maximum sequence length (for decoding).
     * @param actual_len Actual sequence length (non-padded, for decoding).
     * @param stream CUDA stream to launch the kernel on.
     */
    void cuda_softmax_decode_forward_fp32(
        float* att, float scale, const float* preatt,
        int B, int NH, int max_len, int actual_len,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int B_NH = B * NH;
        int num_blocks = ceil_div( B_NH, block_size );

        softmax_decode_forward_fp32_kernel << <num_blocks, block_size, 0, stream >> > (att, scale, preatt, B_NH, max_len, actual_len);

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Launch softmax_decode_forward_fp16_kernel.
     *
     * @param att Output attention weights [B*NH, T, T].
     * @param scale Scaling factor (typically 1/sqrt(head_size)).
     * @param preatt Input pre-softmax scores [B*NH, T, T].
     * @param B Batch size.
     * @param NH Number of heads.
     * @param max_len Maximum sequence length (for decoding).
     * @param actual_len Actual sequence length (non-padded, for decoding).
     * @param stream CUDA stream to launch the kernel on.
     */
    void cuda_softmax_decode_forward_fp16(
        half* att, float scale, const half* preatt,
        int B, int NH, int max_len, int actual_len,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int B_NH = B * NH;
        int num_blocks = ceil_div( B_NH, block_size );

        softmax_decode_forward_fp16_kernel << <num_blocks, block_size, 0, stream >> > (att, scale, preatt, B_NH, max_len, actual_len);

        cudaCheck( cudaGetLastError() );
    }
}