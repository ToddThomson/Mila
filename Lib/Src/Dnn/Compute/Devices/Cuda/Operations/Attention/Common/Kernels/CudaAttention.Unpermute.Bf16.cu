/**
 * @file CudaAttention.Unpermute.Bf16.cu
 * @brief BF16 output-unpermute CUDA kernels for attention operations.
 *
 * Implements the BF16 unpermute family declared in CudaAttention.cuh.
 * These kernels are shared between MHA and GQA; no QKV-layout knowledge
 * is required. QKV-split and gradient-pack kernels remain in their
 * respective op files (CudaMha.Permute.cu / CudaGqa.Permute.cu).
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "CudaAttention.cuh"

namespace Mila::Dnn::Compute::Cuda::Attention::Common
{
    // ========================================================================
    // BF16 device kernels
    // ========================================================================

    /**
     * @brief Reorder [B, NH, T, HS] → [B, T, C] (BF16).
     */
    __global__ void unpermute_output_bf16_kernel(
        const __nv_bfloat16* vaccum, __nv_bfloat16* out,
        int B, int T, int NH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int C = NH * HS;

        if ( idx < B * T * C )
        {
            const int b = idx / (T * C);
            int rest = idx % (T * C);
            const int t = rest / C;
            const int c = rest % C;
            const int nh = c / HS;
            const int hs = c % HS;

            out[ idx ] = vaccum[ b * (NH * T * HS) + nh * (T * HS) + t * HS + hs ];
        }
    }

    /**
     * @brief Reorder [B, NH, padded_T, HS] → [B, actual_T, C] (BF16).
     */
    __global__ void unpermute_output_padded_bf16_kernel(
        const __nv_bfloat16* vaccum, __nv_bfloat16* out,
        int B, int actual_T, int padded_T, int NH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int C = NH * HS;

        if ( idx < B * actual_T * C )
        {
            const int b = idx / (actual_T * C);
            int rest = idx % (actual_T * C);
            const int t = rest / C;
            const int c = rest % C;
            const int nh = c / HS;
            const int hs = c % HS;

            out[ idx ] = vaccum[ b * (NH * padded_T * HS) + nh * (padded_T * HS) + t * HS + hs ];
        }
    }

    /**
     * @brief Scatter [B, T, C] → [B, NH, T, HS] (BF16, backward).
     */
    __global__ void unpermute_backward_bf16_kernel(
        __nv_bfloat16* dvaccum, const __nv_bfloat16* dout,
        int B, int T, int NH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int C = NH * HS;

        if ( idx < B * T * C )
        {
            const int b = idx / (T * C);
            int rest = idx % (T * C);
            const int t = rest / C;
            const int c = rest % C;
            const int nh = c / HS;
            const int hs = c % HS;

            dvaccum[ b * (NH * T * HS) + nh * (T * HS) + t * HS + hs ] = dout[ idx ];
        }
    }

    // ========================================================================
    // Host launchers
    // ========================================================================

    void cuda_attention_unpermute_output_bf16(
        const __nv_bfloat16* vaccum, __nv_bfloat16* out,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int num_blocks = ceil_div( B * T * NH * HS, block_size );

        unpermute_output_bf16_kernel << < num_blocks, block_size, 0, stream >> > (
            vaccum, out, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_attention_unpermute_output_padded_bf16(
        const __nv_bfloat16* vaccum, __nv_bfloat16* out,
        int B, int actual_T, int padded_T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int num_blocks = ceil_div( B * actual_T * NH * HS, block_size );

        unpermute_output_padded_bf16_kernel << < num_blocks, block_size, 0, stream >> > (
            vaccum, out, B, actual_T, padded_T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_attention_unpermute_backward_bf16(
        __nv_bfloat16* dvaccum, const __nv_bfloat16* dout,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int num_blocks = ceil_div( B * T * NH * HS, block_size );

        unpermute_backward_bf16_kernel << < num_blocks, block_size, 0, stream >> > (
            dvaccum, dout, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }
}