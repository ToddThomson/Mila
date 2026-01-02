/**
 * @file CudaMHA.cu
 * @brief CUDA kernels for Multi-Head Attention auxiliary operations.
 *
 * Provides kernels for operations that cannot be handled by cuBLASLt:
 * - QKV permutation (split and reshape)
 * - Output unpermutation (reshape and concatenate)
 * - Softmax with causal masking
 * - Backward pass for softmax and permutations
 *
 * The matmul operations are delegated to cuBLASLt plans built in CudaAttentionOp.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute
{
    // ========================================================================
    // Forward Pass Kernels
    // ========================================================================

    /**
     * @brief Permute and split concatenated QKV input into separate Q, K, V tensors.
     *
     * Input: [B, T, 3, NH, HS] (concatenated QKV)
     * Output: Q, K, V each [B, NH, T, HS]
     */
    __global__ void permute_qkv_fp32_kernel(
        float* q, float* k, float* v,
        const float* inp,
        int B, int T, int NH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * T * HS )
        {
            int b = idx / (NH * T * HS);
            int rest = idx % (NH * T * HS);
            int nh = rest / (T * HS);
            rest = rest % (T * HS);
            int t = rest / HS;
            int hs = rest % HS;

            int inp_idx = (b * T * 3 * NH * HS) + (t * 3 * NH * HS) + (0 * NH * HS) + (nh * HS) + hs;

            q[ idx ] = __ldcs( &inp[ inp_idx ] );
            k[ idx ] = __ldcs( &inp[ inp_idx + NH * HS ] );
            v[ idx ] = __ldcs( &inp[ inp_idx + 2 * NH * HS ] );
        }
    }

    __global__ void permute_qkv_fp16_kernel(
        half* q, half* k, half* v,
        const half* inp,
        int B, int T, int NH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * T * HS )
        {
            int b = idx / (NH * T * HS);
            int rest = idx % (NH * T * HS);
            int nh = rest / (T * HS);
            rest = rest % (T * HS);
            int t = rest / HS;
            int hs = rest % HS;

            int inp_idx = (b * T * 3 * NH * HS) + (t * 3 * NH * HS) + (0 * NH * HS) + (nh * HS) + hs;

            q[ idx ] = __ldcs( &inp[ inp_idx ] );
            k[ idx ] = __ldcs( &inp[ inp_idx + NH * HS ] );
            v[ idx ] = __ldcs( &inp[ inp_idx + 2 * NH * HS ] );
        }
    }

    /**
     * @brief Unpermute attention output back to concatenated format.
     *
     * Input: [B, NH, T, HS]
     * Output: [B, T, C] where C = NH * HS
     */
    __global__ void unpermute_output_fp32_kernel(
        const float* vaccum, float* out,
        int B, int T, int NH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int C = NH * HS;

        if ( idx < B * T * C )
        {
            int b = idx / (T * C);
            int rest = idx % (T * C);
            int t = rest / C;
            int c = rest % C;

            int nh = c / HS;
            int hs = c % HS;

            int vaccum_idx = (b * NH * T * HS) + (nh * T * HS) + (t * HS) + hs;

            out[ idx ] = vaccum[ vaccum_idx ];
        }
    }

    __global__ void unpermute_output_fp16_kernel(
        const half* vaccum, half* out,
        int B, int T, int NH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int C = NH * HS;

        if ( idx < B * T * C )
        {
            int b = idx / (T * C);
            int rest = idx % (T * C);
            int t = rest / C;
            int c = rest % C;

            int nh = c / HS;
            int hs = c % HS;

            int vaccum_idx = (b * NH * T * HS) + (nh * T * HS) + (t * HS) + hs;

            out[ idx ] = vaccum[ vaccum_idx ];
        }
    }

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

            float max_val = -INFINITY;

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                float val = preatt_row[ t2 ] * scale;
                if ( val > max_val )
                {
                    max_val = val;
                }
            }

            float sum = 0.0f;

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                float val = expf( preatt_row[ t2 ] * scale - max_val );
                sum += val;
                att_row[ t2 ] = val;
            }

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                att_row[ t2 ] /= sum;
            }

            for ( int t2 = t + 1; t2 < T; ++t2 )
            {
                att_row[ t2 ] = 0.0f;
            }
        }
    }

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

            float max_val = -INFINITY;

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                float val = __half2float( preatt_row[ t2 ] ) * scale;
                if ( val > max_val )
                {
                    max_val = val;
                }
            }

            float sum = 0.0f;

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                float val = expf( __half2float( preatt_row[ t2 ] ) * scale - max_val );
                sum += val;
                att_row[ t2 ] = __float2half( val );
            }

            for ( int t2 = 0; t2 <= t; ++t2 )
            {
                float normalized = __half2float( att_row[ t2 ] ) / sum;
                att_row[ t2 ] = __float2half( normalized );
            }

            for ( int t2 = t + 1; t2 < T; ++t2 )
            {
                att_row[ t2 ] = __float2half( 0.0f );
            }
        }
    }

    // ========================================================================
    // Backward Pass Kernels
    // ========================================================================

    /**
     * @brief Softmax backward pass.
     *
     * Computes gradient of pre-softmax scores from gradient of softmax output.
     * Input: datt [B*NH, T, T], att [B*NH, T, T]
     * Output: dpreatt [B*NH, T, T]
     */
    __global__ void softmax_backward_fp32_kernel(
        float* dpreatt, const float* datt, const float* att,
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
                dpreatt_row[ t2 ] = att_row[ t2 ] * (datt_row[ t2 ] - sum);
            }

            for ( int t2 = t + 1; t2 < T; ++t2 )
            {
                dpreatt_row[ t2 ] = 0.0f;
            }
        }
    }

    __global__ void softmax_backward_fp16_kernel(
        half* dpreatt, const half* datt, const half* att,
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

    /**
     * @brief Backward pass for output unpermutation.
     *
     * Input: dout [B, T, C]
     * Output: dvaccum [B, NH, T, HS]
     */
    __global__ void unpermute_backward_fp32_kernel(
        float* dvaccum, const float* dout,
        int B, int T, int NH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int C = NH * HS;

        if ( idx < B * T * C )
        {
            int b = idx / (T * C);
            int rest = idx % (T * C);
            int t = rest / C;
            int c = rest % C;

            int nh = c / HS;
            int hs = c % HS;

            int dvaccum_idx = (b * NH * T * HS) + (nh * T * HS) + (t * HS) + hs;

            dvaccum[ dvaccum_idx ] = dout[ idx ];
        }
    }

    __global__ void unpermute_backward_fp16_kernel(
        half* dvaccum, const half* dout,
        int B, int T, int NH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int C = NH * HS;

        if ( idx < B * T * C )
        {
            int b = idx / (T * C);
            int rest = idx % (T * C);
            int t = rest / C;
            int c = rest % C;

            int nh = c / HS;
            int hs = c % HS;

            int dvaccum_idx = (b * NH * T * HS) + (nh * T * HS) + (t * HS) + hs;

            dvaccum[ dvaccum_idx ] = dout[ idx ];
        }
    }

    /**
     * @brief Backward pass for QKV permutation (recombine gradients).
     *
     * Input: dq, dk, dv each [B, NH, T, HS]
     * Output: dinp [B, T, 3, NH, HS] (concatenated gradient)
     */
    __global__ void permute_backward_fp32_kernel(
        float* dinp,
        const float* dq, const float* dk, const float* dv,
        int B, int T, int NH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * T * HS )
        {
            int b = idx / (NH * T * HS);
            int rest = idx % (NH * T * HS);
            int nh = rest / (T * HS);
            rest = rest % (T * HS);
            int t = rest / HS;
            int hs = rest % HS;

            int dinp_q_idx = (b * T * 3 * NH * HS) + (t * 3 * NH * HS) + (0 * NH * HS) + (nh * HS) + hs;
            int dinp_k_idx = dinp_q_idx + NH * HS;
            int dinp_v_idx = dinp_q_idx + 2 * NH * HS;

            dinp[ dinp_q_idx ] = dq[ idx ];
            dinp[ dinp_k_idx ] = dk[ idx ];
            dinp[ dinp_v_idx ] = dv[ idx ];
        }
    }

    __global__ void permute_backward_fp16_kernel(
        half* dinp,
        const half* dq, const half* dk, const half* dv,
        int B, int T, int NH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * T * HS )
        {
            int b = idx / (NH * T * HS);
            int rest = idx % (NH * T * HS);
            int nh = rest / (T * HS);
            rest = rest % (T * HS);
            int t = rest / HS;
            int hs = rest % HS;

            int dinp_q_idx = (b * T * 3 * NH * HS) + (t * 3 * NH * HS) + (0 * NH * HS) + (nh * HS) + hs;
            int dinp_k_idx = dinp_q_idx + NH * HS;
            int dinp_v_idx = dinp_q_idx + 2 * NH * HS;

            dinp[ dinp_q_idx ] = dq[ idx ];
            dinp[ dinp_k_idx ] = dk[ idx ];
            dinp[ dinp_v_idx ] = dv[ idx ];
        }
    }

    // ========================================================================
    // Host Functions - FP32
    // ========================================================================

    void cuda_permute_qkv_fp32(
        float* q, float* k, float* v,
        const float* inp,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * NH * T * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        permute_qkv_fp32_kernel<<<num_blocks, block_size, 0, stream>>> (q, k, v, inp, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_unpermute_output_fp32(
        const float* vaccum, float* out,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int C = NH * HS;
        int total_threads = B * T * C;
        int num_blocks = ceil_div( total_threads, block_size );

        unpermute_output_fp32_kernel << <num_blocks, block_size, 0, stream >> > (vaccum, out, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_softmax_forward_fp32(
        float* att, float scale, const float* preatt,
        int B, int NH, int T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int B_NH = B * NH;
        int total_rows = B_NH * T;
        int num_blocks = ceil_div( total_rows, block_size );

        softmax_forward_fp32_kernel <<<num_blocks, block_size, 0, stream >> > (att, scale, preatt, B_NH, T);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_softmax_backward_fp32(
        float* dpreatt, const float* datt, const float* att,
        int B, int NH, int T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int B_NH = B * NH;
        int total_rows = B_NH * T;
        int num_blocks = ceil_div( total_rows, block_size );

        softmax_backward_fp32_kernel << <num_blocks, block_size, 0, stream >> > (dpreatt, datt, att, B_NH, T);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_unpermute_backward_fp32(
        float* dvaccum, const float* dout,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * T * NH * HS;
        int num_blocks = ceil_div( total_threads, block_size );
        // Launch kernel with: num_blocks=6144, block_size=256

        unpermute_backward_fp32_kernel <<<num_blocks, block_size, 0, stream >>>( dvaccum, dout, B, T, NH, HS );

        cudaCheck( cudaGetLastError() );
    }

    void cuda_permute_backward_fp32(
        float* dinp,
        const float* dq, const float* dk, const float* dv,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * T * NH * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        permute_backward_fp32_kernel <<<num_blocks, block_size, 0, stream >> > (dinp, dq, dk, dv, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    // ========================================================================
    // Host Functions - FP16
    // ========================================================================

    void cuda_permute_qkv_fp16(
        half* q, half* k, half* v,
        const half* inp,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * NH * T * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        permute_qkv_fp16_kernel << <num_blocks, block_size, 0, stream >> > (q, k, v, inp, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_unpermute_output_fp16(
        const half* vaccum, half* out,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int C = NH * HS;
        int total_threads = B * T * C;
        int num_blocks = ceil_div( total_threads, block_size );

        unpermute_output_fp16_kernel << <num_blocks, block_size, 0, stream >> > (vaccum, out, B, T, NH, HS);

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
        int B, int NH, int T,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int B_NH = B * NH;
        int total_rows = B_NH * T;
        int num_blocks = ceil_div( total_rows, block_size );

        softmax_backward_fp16_kernel << <num_blocks, block_size, 0, stream >> > (dpreatt, datt, att, B_NH, T);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_unpermute_backward_fp16(
        half* dvaccum, const half* dout,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int C = NH * HS;
        int total_threads = B * T * C;
        int num_blocks = ceil_div( total_threads, block_size );

        unpermute_backward_fp16_kernel << <num_blocks, block_size, 0, stream >> > (dvaccum, dout, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_permute_backward_fp16(
        half* dinp,
        const half* dq, const half* dk, const half* dv,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * NH * T * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        permute_backward_fp16_kernel << <num_blocks, block_size, 0, stream >> > (dinp, dq, dk, dv, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }
}