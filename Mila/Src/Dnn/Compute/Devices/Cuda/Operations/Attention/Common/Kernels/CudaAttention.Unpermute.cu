/**
 * @file CudaAttention.Permute.cu
 * @brief Shared output-unpermute CUDA kernels for attention operations.
 *
 * Implements the unpermute family declared in CudaAttention.cuh.
 * These kernels are shared between MHA and GQA because their only inputs
 * are the attention accumulator vaccum [B, NH, T, HS] and the output
 * buffer [B, T, C=NH*HS] — no QKV-layout knowledge is required.
 *
 * The permute (QKV split) and permute_backward (gradient pack) kernels are
 * NOT included here because they depend on the QKV trailing dimension:
 *   MHA : 3 * NH * HS
 *   GQA : (NH + 2 * NKV) * HS
 * Those remain in CudaMha.Permute.cu and CudaGqa.Permute.cu respectively.
 *
 * Layout convention
 * ─────────────────
 *   vaccum / dvaccum : [B, NH, T, HS]   row-major
 *   out    / dout    : [B, T,  C]        C = NH * HS
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "CudaAttention.cuh"

namespace Mila::Dnn::Compute::Cuda::Attention::Common
{
    // ========================================================================
    // FP32 device kernels
    // ========================================================================

    /**
     * @brief Reorder [B, NH, T, HS] → [B, T, C] (FP32).
     *
     * One thread per output element identified by the flat index
     * (b, t, c) where c = nh * HS + hs.
     */
    __global__ void unpermute_output_fp32_kernel(
        const float* vaccum, float* out,
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
     * @brief Reorder [B, NH, padded_T, HS] → [B, actual_T, C] (FP32).
     *
     * Source buffer uses padded_T as its inner stride; only the first
     * actual_T token rows are emitted.  No data is written for positions
     * >= actual_T.
     */
    __global__ void unpermute_output_padded_fp32_kernel(
        const float* vaccum, float* out,
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
     * @brief Scatter [B, T, C] → [B, NH, T, HS] (FP32, backward).
     *
     * Inverse of unpermute_output: routes each gradient element from the
     * flat output-gradient layout back into the per-head layout expected
     * by the cuBLASLt backward plans.
     */
    __global__ void unpermute_backward_fp32_kernel(
        float* dvaccum, const float* dout,
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
    // FP16 device kernels
    // ========================================================================

    /**
     * @brief Reorder [B, NH, T, HS] → [B, T, C] (FP16).
     */
    __global__ void unpermute_output_fp16_kernel(
        const half* vaccum, half* out,
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
     * @brief Reorder [B, NH, padded_T, HS] → [B, actual_T, C] (FP16).
     */
    __global__ void unpermute_output_padded_fp16_kernel(
        const half* vaccum, half* out,
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
     * @brief Scatter [B, T, C] → [B, NH, T, HS] (FP16, backward).
     */
    __global__ void unpermute_backward_fp16_kernel(
        half* dvaccum, const half* dout,
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
    // Host launchers — FP32
    // ========================================================================

    void cuda_attention_unpermute_output_fp32(
        const float* vaccum, float* out,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int num_blocks = ceil_div( B * T * NH * HS, block_size );

        unpermute_output_fp32_kernel << <num_blocks, block_size, 0, stream >> > (
            vaccum, out, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_attention_unpermute_output_padded_fp32(
        const float* vaccum, float* out,
        int B, int actual_T, int padded_T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int num_blocks = ceil_div( B * actual_T * NH * HS, block_size );

        unpermute_output_padded_fp32_kernel << <num_blocks, block_size, 0, stream >> > (
            vaccum, out, B, actual_T, padded_T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_attention_unpermute_backward_fp32(
        float* dvaccum, const float* dout,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int num_blocks = ceil_div( B * T * NH * HS, block_size );

        unpermute_backward_fp32_kernel << <num_blocks, block_size, 0, stream >> > (
            dvaccum, dout, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    // ========================================================================
    // Host launchers — FP16
    // ========================================================================

    void cuda_attention_unpermute_output_fp16(
        const half* vaccum, half* out,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int num_blocks = ceil_div( B * T * NH * HS, block_size );

        unpermute_output_fp16_kernel << <num_blocks, block_size, 0, stream >> > (
            vaccum, out, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_attention_unpermute_output_padded_fp16(
        const half* vaccum, half* out,
        int B, int actual_T, int padded_T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int num_blocks = ceil_div( B * actual_T * NH * HS, block_size );

        unpermute_output_padded_fp16_kernel << <num_blocks, block_size, 0, stream >> > (
            vaccum, out, B, actual_T, padded_T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_attention_unpermute_backward_fp16(
        half* dvaccum, const half* dout,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int num_blocks = ceil_div( B * T * NH * HS, block_size );

        unpermute_backward_fp16_kernel << <num_blocks, block_size, 0, stream >> > (
            dvaccum, dout, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }
}
