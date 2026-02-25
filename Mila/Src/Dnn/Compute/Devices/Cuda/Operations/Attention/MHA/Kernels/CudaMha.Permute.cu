/**
 * @file CudaAttention.Permute.cu
 * @brief CUDA kernels for Multi-Head Attention auxiliary operations.
 *
 * Provides kernels for operations that cannot be handled by cuBLASLt:
 * - QKV permutation (split and reshape)
 * - Output unpermutation (reshape and concatenate)
 * - Backward pass for softmax and permutations
 *
 * All operations use row-major layout exclusively:
 * - Input X: [B, T, 3*C] where C = NH * HS
 * - Per-head tensors (Q, K, V, vaccum): [B, NH, T, HS]
 * - Output: [B, T, C]
 * - Memory layout: innermost dimension is contiguous (HS values per sequence position)
 *
 * The matmul operations are delegated to cuBLASLt plans built in CudaAttentionOp.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"

namespace Mila::Dnn::Compute::Cuda::MultiHeadAttention
{
    // ========================================================================
    // Forward Pass Kernels
    // ========================================================================

    /**
     * @brief Split input X into Q, K, V (FP32).
     *
     * Each thread handles one element identified by (b, nh, t, hs).
     * Reads concatenated Q/K/V from X[b, t, 3*C] where C = NH*HS
     * and writes into per-head outputs Q, K, V with layout [B, NH, T, HS].
     *
     * Preconditions:
     * - X size >= B * T * 3 * (NH * HS)
     * - Q, K, V size >= B * NH * T * HS
     *
     * Side-effects:
     * - Writes to device buffers Q, K, V.
     *
     * @param Q Output queries [B, NH, T, HS].
     * @param K Output keys [B, NH, T, HS].
     * @param V Output values [B, NH, T, HS].
     * @param X Input concatenated QKV [B, T, 3 * NH * HS].
     * @param B Batch size.
     * @param T Sequence length.
     * @param NH Number of heads.
     * @param HS Head size.
     */
    __global__ void permute_qkv_fp32_kernel(
        float* Q, float* K, float* V,
        const float* X,
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

            int out_idx = b * (NH * T * HS) + nh * (T * HS) + t * HS + hs;

            Q[ out_idx ] = __ldcs( &X[ q_idx ] );
            K[ out_idx ] = __ldcs( &X[ k_idx ] );
            V[ out_idx ] = __ldcs( &X[ v_idx ] );
        }
    }

    /**
     * @brief Split input X into Q, K, V (FP16).
     *
     * FP16 variant of permute_qkv_fp32_kernel.
     *
     * Preconditions:
     * - X size >= B * T * 3 * (NH * HS)
     * - Q, K, V size >= B * NH * T * HS
     *
     * Side-effects:
     * - Writes to device buffers Q, K, V.
     *
     * @param Q Output queries [B, NH, T, HS].
     * @param K Output keys [B, NH, T, HS].
     * @param V Output values [B, NH, T, HS].
     * @param X Input concatenated QKV [B, T, 3 * NH * HS].
     * @param B Batch size.
     * @param T Sequence length.
     * @param NH Number of heads.
     * @param HS Head size.
     */
    __global__ void permute_qkv_fp16_kernel(
        half* Q, half* K, half* V,
        const half* X,
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

            int out_idx = b * (NH * T * HS) + nh * (T * HS) + t * HS + hs;

            Q[ out_idx ] = __ldcs( &X[ q_idx ] );
            K[ out_idx ] = __ldcs( &X[ k_idx ] );
            V[ out_idx ] = __ldcs( &X[ v_idx ] );
        }
    }

    // ========================================================================
    // Output Unpermute
    // ========================================================================

    /**
     * @brief Reorder per-head vaccum into concatenated output (FP32).
     *
     * Reads vaccum organized as [B, NH, T, HS] and writes out as [B, T, C] where C = NH * HS.
     *
     * Preconditions:
     * - vaccum size >= B * NH * T * HS
     * - out size >= B * T * C
     *
     * Side-effects:
     * - Writes to device buffer out.
     *
     * @param vaccum Input buffer [B, NH, T, HS].
     * @param out Output buffer [B, T, C].
     * @param B Batch size.
     * @param T Sequence length.
     * @param NH Number of heads.
     * @param HS Head size.
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

            const int vaccum_idx = b * (NH * T * HS) + nh * (T * HS) + t * HS + hs;

            out[ idx ] = vaccum[ vaccum_idx ];
        }
    }

    /**
     * @brief Reorder per-head vaccum into concatenated output (FP16).
     *
     * FP16 variant of unpermute_output_fp32_kernel.
     *
     * Preconditions:
     * - vaccum size >= B * NH * T * HS
     * - out size >= B * T * C
     *
     * Side-effects:
     * - Writes to device buffer out.
     *
     * @param vaccum Input buffer [B, NH, T, HS].
     * @param out Output buffer [B, T, C].
     * @param B Batch size.
     * @param T Sequence length.
     * @param NH Number of heads.
     * @param HS Head size.
     */
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

            int vaccum_idx = b * (NH * T * HS) + nh * (T * HS) + t * HS + hs;

            out[ idx ] = vaccum[ vaccum_idx ];
        }
    }

    // ========================================================================
    // Backward
    // ========================================================================

    /**
     * @brief Scatter concatenated gradient into per-head gradients (FP32).
     *
     * Reads dout [B, T, C] and writes to dvaccum [B, NH, T, HS].
     *
     * Preconditions:
     * - dout size >= B * T * C
     * - dvaccum size >= B * NH * T * HS
     *
     * Side-effects:
     * - Writes to device buffer dvaccum.
     *
     * @param dvaccum Output gradient buffer [B, NH, T, HS].
     * @param dout Input gradient buffer [B, T, C].
     * @param B Batch size.
     * @param T Sequence length.
     * @param NH Number of heads.
     * @param HS Head size.
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

            int dvaccum_idx = b * (NH * T * HS) + nh * (T * HS) + t * HS + hs;

            dvaccum[ dvaccum_idx ] = dout[ idx ];
        }
    }

    /**
     * @brief Scatter concatenated gradient into per-head gradients (FP16).
     *
     * FP16 variant of unpermute_backward_fp32_kernel.
     *
     * Preconditions:
     * - dout size >= B * T * C
     * - dvaccum size >= B * NH * T * HS
     *
     * Side-effects:
     * - Writes to device buffer dvaccum.
     *
     * @param dvaccum Output gradient buffer [B, NH, T, HS].
     * @param dout Input gradient buffer [B, T, C].
     * @param B Batch size.
     * @param T Sequence length.
     * @param NH Number of heads.
     * @param HS Head size.
     */
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

            int dvaccum_idx = b * (NH * T * HS) + nh * (T * HS) + t * HS + hs;

            dvaccum[ dvaccum_idx ] = dout[ idx ];
        }
    }

    /**
     * @brief Pack per-head gradients dq/dk/dv into concatenated dinp (FP32).
     *
     * Reads per-head gradients dq/dk/dv ([B, NH, T, HS]) and writes
     * them back into the concatenated layout dinp [B, T, 3*C].
     *
     * Preconditions:
     * - dq/dk/dv size >= B * NH * T * HS
     * - dinp size >= B * T * 3 * C
     *
     * Side-effects:
     * - Writes to device buffer dinp.
     *
     * @param dinp Output concatenated gradient buffer [B, T, 3*C].
     * @param dq Input gradient buffer for Q [B, NH, T, HS].
     * @param dk Input gradient buffer for K [B, NH, T, HS].
     * @param dv Input gradient buffer for V [B, NH, T, HS].
     * @param B Batch size.
     * @param T Sequence length.
     * @param NH Number of heads.
     * @param HS Head size.
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

            int in_idx = b * (NH * T * HS) + nh * (T * HS) + t * HS + hs;

            dinp[ dinp_q_idx ] = dq[ in_idx ];
            dinp[ dinp_k_idx ] = dk[ in_idx ];
            dinp[ dinp_v_idx ] = dv[ in_idx ];
        }
    }

    /**
     * @brief Pack per-head gradients dq/dk/dv into concatenated dinp (FP16).
     *
     * FP16 variant of permute_backward_fp32_kernel.
     *
     * Preconditions:
     * - dq/dk/dv size >= B * NH * T * HS
     * - dinp size >= B * T * 3 * C
     *
     * Side-effects:
     * - Writes to device buffer dinp.
     *
     * @param dinp Output concatenated gradient buffer [B, T, 3*C].
     * @param dq Input gradient buffer for Q [B, NH, T, HS].
     * @param dk Input gradient buffer for K [B, NH, T, HS].
     * @param dv Input gradient buffer for V [B, NH, T, HS].
     * @param B Batch size.
     * @param T Sequence length.
     * @param NH Number of heads.
     * @param HS Head size.
     */
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

            int in_idx = b * (NH * T * HS) + nh * (T * HS) + t * HS + hs;

            dinp[ dinp_q_idx ] = dq[ in_idx ];
            dinp[ dinp_k_idx ] = dk[ in_idx ];
            dinp[ dinp_v_idx ] = dv[ in_idx ];
        }
    }

    __global__ void permute_qkv_padded_fp32_kernel(
        float* Q, float* K, float* V,
        const float* X,
        int B, int input_T, int output_T, int NH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * output_T * HS )
        {
            int b = idx / (NH * output_T * HS);
            int rest = idx % (NH * output_T * HS);
            int nh = rest / (output_T * HS);
            rest = rest % (output_T * HS);
            int t = rest / HS;
            int hs = rest % HS;

            int out_idx = b * (NH * output_T * HS) + nh * (output_T * HS) + t * HS + hs;

            if ( t >= input_T )
            {
                Q[ out_idx ] = 0.0f;
                K[ out_idx ] = 0.0f;
                V[ out_idx ] = 0.0f;

                return;
            }

            int C = NH * HS;
            int base_idx = b * input_T * 3 * C + t * 3 * C;
            int head_offset = nh * HS + hs;

            int q_idx = base_idx + head_offset;
            int k_idx = base_idx + C + head_offset;
            int v_idx = base_idx + 2 * C + head_offset;

            Q[ out_idx ] = __ldcs( &X[ q_idx ] );
            K[ out_idx ] = __ldcs( &X[ k_idx ] );
            V[ out_idx ] = __ldcs( &X[ v_idx ] );
        }
    }

    __global__ void permute_qkv_padded_fp16_kernel(
        half* Q, half* K, half* V,
        const half* X,
        int B, int input_T, int output_T, int NH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * output_T * HS )
        {
            int b = idx / (NH * output_T * HS);
            int rest = idx % (NH * output_T * HS);
            int nh = rest / (output_T * HS);
            rest = rest % (output_T * HS);
            int t = rest / HS;
            int hs = rest % HS;

            int out_idx = b * (NH * output_T * HS) + nh * (output_T * HS) + t * HS + hs;

            if ( t >= input_T )
            {
                Q[ out_idx ] = __float2half( 0.0f );
                K[ out_idx ] = __float2half( 0.0f );
                V[ out_idx ] = __float2half( 0.0f );

                return;
            }

            int C = NH * HS;
            int base_idx = b * input_T * 3 * C + t * 3 * C;
            int head_offset = nh * HS + hs;

            int q_idx = base_idx + head_offset;
            int k_idx = base_idx + C + head_offset;
            int v_idx = base_idx + 2 * C + head_offset;

            Q[ out_idx ] = __ldcs( &X[ q_idx ] );
            K[ out_idx ] = __ldcs( &X[ k_idx ] );
            V[ out_idx ] = __ldcs( &X[ v_idx ] );
        }
    }

    __global__ void permute_qkv_decode_fp32_kernel(
        float* Q, float* K, float* V,
        const float* X,
        int B, int position, int cache_T, int NH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * HS )
        {
            int b = idx / (NH * HS);
            int rest = idx % (NH * HS);
            int nh = rest / HS;
            int hs = rest % HS;

            int C = NH * HS;
            int base_idx = b * 3 * C;
            int head_offset = nh * HS + hs;

            int q_idx = base_idx + head_offset;
            int k_idx = base_idx + C + head_offset;
            int v_idx = base_idx + 2 * C + head_offset;

            int out_idx = b * (NH * cache_T * HS) + nh * (cache_T * HS) + position * HS + hs;

            Q[ out_idx ] = __ldcs( &X[ q_idx ] );
            K[ out_idx ] = __ldcs( &X[ k_idx ] );
            V[ out_idx ] = __ldcs( &X[ v_idx ] );
        }
    }

    __global__ void permute_qkv_decode_fp16_kernel(
        half* Q, half* K, half* V,
        const half* X,
        int B, int position, int cache_T, int NH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx < B * NH * HS )
        {
            int b = idx / (NH * HS);
            int rest = idx % (NH * HS);
            int nh = rest / HS;
            int hs = rest % HS;

            int C = NH * HS;
            int base_idx = b * 3 * C;
            int head_offset = nh * HS + hs;

            int q_idx = base_idx + head_offset;
            int k_idx = base_idx + C + head_offset;
            int v_idx = base_idx + 2 * C + head_offset;

            int out_idx = b * (NH * cache_T * HS) + nh * (cache_T * HS) + position * HS + hs;

            Q[ out_idx ] = __ldcs( &X[ q_idx ] );
            K[ out_idx ] = __ldcs( &X[ k_idx ] );
            V[ out_idx ] = __ldcs( &X[ v_idx ] );
        }
    }

    void cuda_permute_qkv_padded_fp32(
        float* Q, float* K, float* V,
        const float* X,
        int B, int input_T, int output_T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * NH * output_T * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        permute_qkv_padded_fp32_kernel << <num_blocks, block_size, 0, stream >> > (Q, K, V, X, B, input_T, output_T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_permute_qkv_decode_fp32(
        float* Q, float* K, float* V,
        const float* X,
        int B, int position, int cache_T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * NH * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        permute_qkv_decode_fp32_kernel << <num_blocks, block_size, 0, stream >> > (Q, K, V, X, B, position, cache_T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_permute_qkv_padded_fp16(
        half* Q, half* K, half* V,
        const half* X,
        int B, int input_T, int output_T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * NH * output_T * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        permute_qkv_padded_fp16_kernel << <num_blocks, block_size, 0, stream >> > (Q, K, V, X, B, input_T, output_T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_permute_qkv_decode_fp16(
        half* Q, half* K, half* V,
        const half* X,
        int B, int position, int cache_T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * NH * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        permute_qkv_decode_fp16_kernel << <num_blocks, block_size, 0, stream >> > (Q, K, V, X, B, position, cache_T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    // ========================================================================
    // Host Functions - FP32
    // ========================================================================

    /**
     * @brief Launch permute_qkv_fp32_kernel to split X into Q/K/V.
     *
     * Preconditions:
     * - Device pointers must be valid and sized as documented in the kernel Doxygen.
     *
     * Side-effects:
     * - Launches a CUDA kernel and checks for launch errors.
     *
     * @param Q Output pointer for queries (device) [B, NH, T, HS].
     * @param K Output pointer for keys (device) [B, NH, T, HS].
     * @param V Output pointer for values (device) [B, NH, T, HS].
     * @param X Input pointer (device) [B, T, 3 * NH * HS].
     * @param B Batch size.
     * @param T Sequence length.
     * @param NH Number of heads.
     * @param HS Head size.
     * @param stream CUDA stream to launch the kernel on.
     */
    void cuda_permute_qkv_fp32(
        float* Q, float* K, float* V,
        const float* X,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * NH * T * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        permute_qkv_fp32_kernel << <num_blocks, block_size, 0, stream >> > (Q, K, V, X, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Launch unpermute_output_fp32_kernel to reorder vaccum into out.
     *
     * @param vaccum Input buffer [B, NH, T, HS].
     * @param out Output buffer [B, T, C].
     * @param B Batch size.
     * @param T Sequence length.
     * @param NH Number of heads.
     * @param HS Head size.
     * @param stream CUDA stream to launch the kernel on.
     */
    void cuda_unpermute_output_fp32(
        const float* vaccum, float* out,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * T * NH * HS;
        int num_blocks = ceil_div( total_threads, block_size );

        unpermute_output_fp32_kernel << <num_blocks, block_size, 0, stream >> > (vaccum, out, B, T, NH, HS);

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Launch unpermute_backward_fp32_kernel to scatter dout into dvaccum.
     *
     * @param dvaccum Output gradient buffer [B, NH, T, HS].
     * @param dout Input gradient buffer [B, T, C].
     * @param B Batch size.
     * @param T Sequence length.
     * @param NH Number of heads.
     * @param HS Head size.
     * @param stream CUDA stream to launch the kernel on.
     */
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

    /**
     * @brief Launch permute_backward_fp32_kernel to pack dq/dk/dv into dinp.
     *
     * @param dinp Output concatenated gradient buffer [B, T, 3*C].
     * @param dq Input gradient buffer for Q [B, NH, T, HS].
     * @param dk Input gradient buffer for K [B, NH, T, HS].
     * @param dv Input gradient buffer for V [B, NH, T, HS].
     * @param B Batch size.
     * @param T Sequence length.
     * @param NH Number of heads.
     * @param HS Head size.
     * @param stream CUDA stream to launch the kernel on.
     */
    void cuda_permute_backward_fp32(
        float* dinp,
        const float* dq, const float* dk, const float* dv,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * NH * T * HS;
        int num_blocks = ceil_div( total_threads, block_size );
        permute_backward_fp32_kernel << <num_blocks, block_size, 0, stream >> > (dinp, dq, dk, dv, B, T, NH, HS);
        cudaCheck( cudaGetLastError() );
    }

    // ========================================================================
    // Host Functions - FP16
    // ========================================================================

    /**
     * @brief Launch permute_qkv_fp16_kernel to split X into Q/K/V.
     *
     * @param Q Output pointer for queries (device) [B, NH, T, HS].
     * @param K Output pointer for keys (device) [B, NH, T, HS].
     * @param V Output pointer for values (device) [B, NH, T, HS].
     * @param X Input pointer (device) [B, T, 3 * NH * HS].
     * @param B Batch size.
     * @param T Sequence length.
     * @param NH Number of heads.
     * @param HS Head size.
     * @param stream CUDA stream to launch the kernel on.
     */
    void cuda_permute_qkv_fp16(
        half* Q, half* K, half* V,
        const half* X,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * NH * T * HS;
        int num_blocks = ceil_div( total_threads, block_size );
        permute_qkv_fp16_kernel << <num_blocks, block_size, 0, stream >> > (Q, K, V, X, B, T, NH, HS);
        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Launch unpermute_output_fp16_kernel to reorder vaccum into out.
     *
     * @param vaccum Input buffer [B, NH, T, HS].
     * @param out Output buffer [B, T, C].
     * @param B Batch size.
     * @param T Sequence length.
     * @param NH Number of heads.
     * @param HS Head size.
     * @param stream CUDA stream to launch the kernel on.
     */
    void cuda_unpermute_output_fp16(
        const half* vaccum, half* out,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * T * NH * HS;
        int num_blocks = ceil_div( total_threads, block_size );
        unpermute_output_fp16_kernel << <num_blocks, block_size, 0, stream >> > (vaccum, out, B, T, NH, HS);
        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Launch unpermute_backward_fp16_kernel to scatter dout into dvaccum.
     *
     * @param dvaccum Output gradient buffer [B, NH, T, HS].
     * @param dout Input gradient buffer [B, T, C].
     * @param B Batch size.
     * @param T Sequence length.
     * @param NH Number of heads.
     * @param HS Head size.
     * @param stream CUDA stream to launch the kernel on.
     */
    void cuda_unpermute_backward_fp16(
        half* dvaccum, const half* dout,
        int B, int T, int NH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        int total_threads = B * T * NH * HS;
        int num_blocks = ceil_div( total_threads, block_size );
        unpermute_backward_fp16_kernel << <num_blocks, block_size, 0, stream >> > (dvaccum, dout, B, T, NH, HS);
        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Launch permute_backward_fp16_kernel to pack dq/dk/dv into dinp.
     *
     * @param dinp Output concatenated gradient buffer [B, T, 3*C].
     * @param dq Input gradient buffer for Q [B, NH, T, HS].
     * @param dk Input gradient buffer for K [B, NH, T, HS].
     * @param dv Input gradient buffer for V [B, NH, T, HS].
     * @param B Batch size.
     * @param T Sequence length.
     * @param NH Number of heads.
     * @param HS Head size.
     * @param stream CUDA stream to launch the kernel on.
     */
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