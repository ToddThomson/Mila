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

namespace Mila::Dnn::Compute::Cuda::Attention
{
    // ========================================================================
    // Forward Pass Kernels
    // ========================================================================

    /**
     * @brief Permute and split concatenated QKV input into separate Q, K, V tensors.
     *
     * Input: [B, T, 3*C] where C = NH*HS (flat concatenated QKV)
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

            int C = NH * HS;
            int base_idx = b * T * 3 * C + t * 3 * C;
            int head_offset = nh * HS + hs;

            int q_idx = base_idx + head_offset;
            int k_idx = base_idx + C + head_offset;
            int v_idx = base_idx + 2 * C + head_offset;
            
            q[ idx ] = __ldcs( &inp[ q_idx ] );
            k[ idx ] = __ldcs( &inp[ k_idx ] );
            v[ idx ] = __ldcs( &inp[ v_idx ] );
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

            int C = NH * HS;
            int base_idx = b * T * 3 * C + t * 3 * C;
            int head_offset = nh * HS + hs;

            int q_idx = base_idx + head_offset;
            int k_idx = base_idx + C + head_offset;
            int v_idx = base_idx + 2 * C + head_offset;

            q[ idx ] = __ldcs( &inp[ q_idx ] );
            k[ idx ] = __ldcs( &inp[ k_idx ] );
            v[ idx ] = __ldcs( &inp[ v_idx ] );
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
     * Output: dinp [B, T, 3*C] where C = NH*HS (flat concatenated gradient)
     * Within each token: [dQ_h0, dQ_h1, ..., dK_h0, dK_h1, ..., dV_h0, dV_h1, ...]
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

            int C = NH * HS;
            int base_idx = b * T * 3 * C + t * 3 * C;
            int head_offset = nh * HS + hs;

            int dinp_q_idx = base_idx + head_offset;
            int dinp_k_idx = base_idx + C + head_offset;
            int dinp_v_idx = base_idx + 2 * C + head_offset;

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

            int C = NH * HS;
            int base_idx = b * T * 3 * C + t * 3 * C;
            int head_offset = nh * HS + hs;

            int dinp_q_idx = base_idx + head_offset;
            int dinp_k_idx = base_idx + C + head_offset;
            int dinp_v_idx = base_idx + 2 * C + head_offset;

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

        permute_qkv_fp32_kernel << <num_blocks, block_size, 0, stream >> > (q, k, v, inp, B, T, NH, HS);

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

    void cuda_unpermute_backward_fp32(
        float* dvaccum, const float* dout,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * T * NH * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        unpermute_backward_fp32_kernel << <num_blocks, block_size, 0, stream >> > (dvaccum, dout, B, T, NH, HS);

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

        permute_backward_fp32_kernel << <num_blocks, block_size, 0, stream >> > (dinp, dq, dk, dv, B, T, NH, HS);

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

    void cuda_unpermute_backward_fp16(
        half* dvaccum, const half* dout,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int C = NH * HS;
        int total_threads = B * T * C;
        int num_blocks = ceil_div( total_threads, block_size );

        unpermute_backward_fp16_kernel <<<num_blocks, block_size, 0, stream >>> (dvaccum, dout, B, T, NH, HS);

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