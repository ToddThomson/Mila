/**
 * @file CudaMHA.cu
 * @brief CUDA kernels for Multi-Head Attention auxiliary operations.
 *
 * Provides kernels for operations that cannot be handled by cuBLASLt:
 * - QKV permutation (split and reshape) - row-major and column-major variants
 * - Output unpermutation (reshape and concatenate) - row-major and column-major variants
 * - Softmax with causal masking
 * - Backward pass for softmax and permutations
 *
 * Row-major layout: [B, NH, T, HS] with strides [NH*T*HS, T*HS, HS, 1]
 * Column-major layout: [B, NH, HS, T] with strides [NH*HS*T, HS*T, 1, HS]
 *   - Appears as [B, NH, T, HS] to cuBLAS (which reads column-major)
 *   - Simplifies cuBLAS matmul operations by eliminating transpose flags
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
    // Forward Pass Kernels - Row Major
    // ========================================================================

    /**
     * @brief Permute and split concatenated QKV input into separate Q, K, V tensors (row-major).
     *
     * Input: [B, T, 3*C] where C = NH*HS (flat concatenated QKV)
     * Output: Q, K, V each [B, NH, T, HS] in row-major layout
     */
    __global__ void permute_qkv_rowmajor_fp32_kernel(
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

    __global__ void permute_qkv_rowmajor_fp16_kernel(
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

    // ========================================================================
    // Forward Pass Kernels - Column Major
    // ========================================================================

    /**
     * @brief Permute and split concatenated QKV input into separate Q, K, V tensors (column-major).
     *
     * Input: [B, T, 3*C] where C = NH*HS (flat concatenated QKV)
     * Output: Q, K, V each [B, NH, HS, T] in column-major layout
     *   - Memory layout has HS as fastest-changing dimension
     *   - cuBLAS will interpret this as [B, NH, T, HS] in column-major
     *   - This eliminates the need for transpose flags in matmuls
     */
    __global__ void permute_qkv_colmajor_fp32_kernel(
        float* q, float* k, float* v,
        const float* inp,
        int B, int T, int NH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * T * HS )
        {
            // Decompose linear index for column-major output [B, NH, HS, T]
            int b = idx / (NH * HS * T);
            int rest = idx % (NH * HS * T);
            int nh = rest / (HS * T);
            rest = rest % (HS * T);
            int hs = rest / T;
            int t = rest % T;

            // Input is row-major [B, T, 3*C]
            int C = NH * HS;
            int base_idx = b * T * 3 * C + t * 3 * C;
            int head_offset = nh * HS + hs;

            int q_idx = base_idx + head_offset;
            int k_idx = base_idx + C + head_offset;
            int v_idx = base_idx + 2 * C + head_offset;

            // Output index in column-major layout
            int out_idx = b * (NH * HS * T) + nh * (HS * T) + hs * T + t;

            q[ out_idx ] = __ldcs( &inp[ q_idx ] );
            k[ out_idx ] = __ldcs( &inp[ k_idx ] );
            v[ out_idx ] = __ldcs( &inp[ v_idx ] );
        }
    }

    __global__ void permute_qkv_colmajor_fp16_kernel(
        half* q, half* k, half* v,
        const half* inp,
        int B, int T, int NH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * T * HS )
        {
            // Decompose linear index for column-major output [B, NH, HS, T]
            int b = idx / (NH * HS * T);
            int rest = idx % (NH * HS * T);
            int nh = rest / (HS * T);
            rest = rest % (HS * T);
            int hs = rest / T;
            int t = rest % T;

            // Input is row-major [B, T, 3*C]
            int C = NH * HS;
            int base_idx = b * T * 3 * C + t * 3 * C;
            int head_offset = nh * HS + hs;

            int q_idx = base_idx + head_offset;
            int k_idx = base_idx + C + head_offset;
            int v_idx = base_idx + 2 * C + head_offset;

            // Output index in column-major layout
            int out_idx = b * (NH * HS * T) + nh * (HS * T) + hs * T + t;

            q[ out_idx ] = __ldcs( &inp[ q_idx ] );
            k[ out_idx ] = __ldcs( &inp[ k_idx ] );
            v[ out_idx ] = __ldcs( &inp[ v_idx ] );
        }
    }

    // ========================================================================
    // Output Unpermute - Row Major
    // ========================================================================

    /**
     * @brief Unpermute attention output back to concatenated format (from row-major).
     *
     * Input: [B, NH, T, HS] in row-major layout
     * Output: [B, T, C] where C = NH * HS
     */
    __global__ void unpermute_output_rowmajor_fp32_kernel(
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

    __global__ void unpermute_output_rowmajor_fp16_kernel(
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

    // ========================================================================
    // Output Unpermute - Column Major
    // ========================================================================

    /**
     * @brief Unpermute attention output back to concatenated format (from column-major).
     *
     * Input: [B, NH, HS, T] in column-major layout (cuBLAS sees as [B, NH, T, HS])
     * Output: [B, T, C] where C = NH * HS
     */
    __global__ void unpermute_output_colmajor_fp32_kernel(
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

            // vaccum is in column-major: [B, NH, HS, T]
            int vaccum_idx = b * (NH * HS * T) + nh * (HS * T) + hs * T + t;

            out[ idx ] = vaccum[ vaccum_idx ];
        }
    }

    __global__ void unpermute_output_colmajor_fp16_kernel(
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

            // vaccum is in column-major: [B, NH, HS, T]
            int vaccum_idx = b * (NH * HS * T) + nh * (HS * T) + hs * T + t;

            out[ idx ] = vaccum[ vaccum_idx ];
        }
    }

    // ========================================================================
    // Backward Pass Kernels
    // ========================================================================

    /**
     * @brief Backward pass for output unpermutation (row-major).
     *
     * Input: dout [B, T, C]
     * Output: dvaccum [B, NH, T, HS]
     */
    __global__ void unpermute_backward_rowmajor_fp32_kernel(
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

    __global__ void unpermute_backward_rowmajor_fp16_kernel(
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
     * @brief Backward pass for QKV permutation (recombine gradients, row-major).
     *
     * Input: dq, dk, dv each [B, NH, T, HS]
     * Output: dinp [B, T, 3*C] where C = NH*HS (flat concatenated gradient)
     */
    __global__ void permute_backward_rowmajor_fp32_kernel(
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

    __global__ void permute_backward_rowmajor_fp16_kernel(
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
    // Backward - Column Major
    // ========================================================================

    /**
     * @brief Backward pass for output unpermutation (column-major).
     *
     * Input: dout [B, T, C]
     * Output: dvaccum [B, NH, HS, T] (column-major)
     */
    __global__ void unpermute_backward_colmajor_fp32_kernel(
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

            // dvaccum is column-major: [B, NH, HS, T]
            int dvaccum_idx = b * (NH * HS * T) + nh * (HS * T) + hs * T + t;

            dvaccum[ dvaccum_idx ] = dout[ idx ];
        }
    }

    __global__ void unpermute_backward_colmajor_fp16_kernel(
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

            // dvaccum is column-major: [B, NH, HS, T]
            int dvaccum_idx = b * (NH * HS * T) + nh * (HS * T) + hs * T + t;

            dvaccum[ dvaccum_idx ] = dout[ idx ];
        }
    }

    /**
     * @brief Backward pass for QKV permutation (recombine gradients, column-major).
     *
     * Input: dq, dk, dv each [B, NH, HS, T] (column-major)
     * Output: dinp [B, T, 3*C] where C = NH*HS (flat concatenated gradient)
     */
    __global__ void permute_backward_colmajor_fp32_kernel(
        float* dinp,
        const float* dq, const float* dk, const float* dv,
        int B, int T, int NH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * T * HS )
        {
            // Decompose for column-major input [B, NH, HS, T]
            int b = idx / (NH * HS * T);
            int rest = idx % (NH * HS * T);
            int nh = rest / (HS * T);
            rest = rest % (HS * T);
            int hs = rest / T;
            int t = rest % T;

            int C = NH * HS;
            int base_idx = b * T * 3 * C + t * 3 * C;
            int head_offset = nh * HS + hs;

            int dinp_q_idx = base_idx + head_offset;
            int dinp_k_idx = base_idx + C + head_offset;
            int dinp_v_idx = base_idx + 2 * C + head_offset;

            // Input index in column-major layout
            int in_idx = b * (NH * HS * T) + nh * (HS * T) + hs * T + t;

            dinp[ dinp_q_idx ] = dq[ in_idx ];
            dinp[ dinp_k_idx ] = dk[ in_idx ];
            dinp[ dinp_v_idx ] = dv[ in_idx ];
        }
    }

    __global__ void permute_backward_colmajor_fp16_kernel(
        half* dinp,
        const half* dq, const half* dk, const half* dv,
        int B, int T, int NH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * T * HS )
        {
            // Decompose for column-major input [B, NH, HS, T]
            int b = idx / (NH * HS * T);
            int rest = idx % (NH * HS * T);
            int nh = rest / (HS * T);
            rest = rest % (HS * T);
            int hs = rest / T;
            int t = rest % T;

            int C = NH * HS;
            int base_idx = b * T * 3 * C + t * 3 * C;
            int head_offset = nh * HS + hs;

            int dinp_q_idx = base_idx + head_offset;
            int dinp_k_idx = base_idx + C + head_offset;
            int dinp_v_idx = base_idx + 2 * C + head_offset;

            // Input index in column-major layout
            int in_idx = b * (NH * HS * T) + nh * (HS * T) + hs * T + t;

            dinp[ dinp_q_idx ] = dq[ in_idx ];
            dinp[ dinp_k_idx ] = dk[ in_idx ];
            dinp[ dinp_v_idx ] = dv[ in_idx ];
        }
    }

    // ========================================================================
    // Host Functions - FP32 - Row Major
    // ========================================================================

    void cuda_permute_qkv_rowmajor_fp32(
        float* q, float* k, float* v,
        const float* inp,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * NH * T * HS;
        int num_blocks = ceil_div( total_threads, block_size );
        permute_qkv_rowmajor_fp32_kernel << <num_blocks, block_size, 0, stream >> > (q, k, v, inp, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_unpermute_output_rowmajor_fp32(
        const float* vaccum, float* out,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int C = NH * HS;
        int total_threads = B * T * C;
        int num_blocks = ceil_div( total_threads, block_size );

        unpermute_output_rowmajor_fp32_kernel << <num_blocks, block_size, 0, stream >> > (vaccum, out, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_unpermute_backward_rowmajor_fp32(
        float* dvaccum, const float* dout,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * T * NH * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        unpermute_backward_rowmajor_fp32_kernel << <num_blocks, block_size, 0, stream >> > (dvaccum, dout, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_permute_backward_rowmajor_fp32(
        float* dinp,
        const float* dq, const float* dk, const float* dv,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * T * NH * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        permute_backward_rowmajor_fp32_kernel << <num_blocks, block_size, 0, stream >> > (dinp, dq, dk, dv, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    // ========================================================================
    // Host Functions - FP32 - Column Major
    // ========================================================================

    void cuda_permute_qkv_colmajor_fp32(
        float* q, float* k, float* v,
        const float* inp,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * NH * T * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        permute_qkv_colmajor_fp32_kernel << <num_blocks, block_size, 0, stream >> > (q, k, v, inp, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_unpermute_output_colmajor_fp32(
        const float* vaccum, float* out,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int C = NH * HS;
        int total_threads = B * T * C;
        int num_blocks = ceil_div( total_threads, block_size );

        unpermute_output_colmajor_fp32_kernel << <num_blocks, block_size, 0, stream >> > (vaccum, out, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_unpermute_backward_colmajor_fp32(
        float* dvaccum, const float* dout,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * T * NH * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        unpermute_backward_colmajor_fp32_kernel << <num_blocks, block_size, 0, stream >> > (dvaccum, dout, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_permute_backward_colmajor_fp32(
        float* dinp,
        const float* dq, const float* dk, const float* dv,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * NH * T * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        permute_backward_colmajor_fp32_kernel << <num_blocks, block_size, 0, stream >> > (dinp, dq, dk, dv, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    // ========================================================================
    // Host Functions - FP16 versions (similar pattern)
    // ========================================================================

    // Row-major FP16
    void cuda_permute_qkv_rowmajor_fp16(
        half* q, half* k, half* v,
        const half* inp,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * NH * T * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        permute_qkv_rowmajor_fp16_kernel << <num_blocks, block_size, 0, stream >> > (q, k, v, inp, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_unpermute_output_rowmajor_fp16(
        const half* vaccum, half* out,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int C = NH * HS;
        int total_threads = B * T * C;
        int num_blocks = ceil_div( total_threads, block_size );

        unpermute_output_rowmajor_fp16_kernel << <num_blocks, block_size, 0, stream >> > (vaccum, out, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_unpermute_backward_rowmajor_fp16(
        half* dvaccum, const half* dout,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int C = NH * HS;
        int total_threads = B * T * C;
        int num_blocks = ceil_div( total_threads, block_size );

        unpermute_backward_rowmajor_fp16_kernel << <num_blocks, block_size, 0, stream >> > (dvaccum, dout, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_permute_backward_rowmajor_fp16(
        half* dinp,
        const half* dq, const half* dk, const half* dv,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * NH * T * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        permute_backward_rowmajor_fp16_kernel << <num_blocks, block_size, 0, stream >> > (dinp, dq, dk, dv, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    // Column-major FP16
    void cuda_permute_qkv_colmajor_fp16(
        half* q, half* k, half* v,
        const half* inp,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * NH * T * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        permute_qkv_colmajor_fp16_kernel << <num_blocks, block_size, 0, stream >> > (q, k, v, inp, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_unpermute_output_colmajor_fp16(
        const half* vaccum, half* out,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int C = NH * HS;
        int total_threads = B * T * C;
        int num_blocks = ceil_div( total_threads, block_size );

        unpermute_output_colmajor_fp16_kernel << <num_blocks, block_size, 0, stream >> > (vaccum, out, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_unpermute_backward_colmajor_fp16(
        half* dvaccum, const half* dout,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * T * NH * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        unpermute_backward_colmajor_fp16_kernel << <num_blocks, block_size, 0, stream >> > (dvaccum, dout, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_permute_backward_colmajor_fp16(
        half* dinp,
        const half* dq, const half* dk, const half* dv,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * NH * T * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        permute_backward_colmajor_fp16_kernel << <num_blocks, block_size, 0, stream >> > (dinp, dq, dk, dv, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }
}