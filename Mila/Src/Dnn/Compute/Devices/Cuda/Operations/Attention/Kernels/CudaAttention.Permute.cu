/**
 * @file CudaAttention.Permute.cu
 * @brief CUDA kernels for Multi-Head Attention auxiliary operations.
 *
 * Provides kernels for operations that cannot be handled by cuBLASLt:
 * - QKV permutation (split and reshape) - column-major variants
 * - Output unpermutation (reshape and concatenate) - column-major variants
 * - Backward pass for softmax and permutations
 *
 * Column-major layout: [B, NH, HS, T] - each head is [HS rows, T cols] in column-major
 *   - Columns are contiguous: index = t * HS + hs
 *   - cuBLAS column-major sees this as [HS, T] matrix per head
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
    // Forward Pass Kernels - Column Major
    // ========================================================================

    /**
     * @brief Split row-major input X into column-major Q, K, V (FP32).
     *
     * Each thread handles one element identified by (b, nh, t, hs).
     * Reads concatenated Q/K/V from row-major X[b, t, 3*C] where C = NH*HS
     * and writes into column-major per-head outputs Q, K, V with layout
     * [B, NH, HS, T].
     *
     * Preconditions:
     * - X size >= B * T * 3 * (NH * HS)
     * - Q, K, V size >= B * NH * HS * T
     *
     * Side-effects:
     * - Writes to device buffers Q, K, V.
     *
     * @param Q Output queries in column-major [B, NH, HS, T].
     * @param K Output keys in column-major [B, NH, HS, T].
     * @param V Output values in column-major [B, NH, HS, T].
     * @param X Input concatenated QKV in row-major [B, T, 3 * NH * HS].
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

            // Column-major [HS, T]: column t starts at t*HS, row hs is offset hs
            int out_idx = b * (NH * HS * T) + nh * (HS * T) + t * HS + hs;

            Q[ out_idx ] = __ldcs( &X[ q_idx ] );
            K[ out_idx ] = __ldcs( &X[ k_idx ] );
            V[ out_idx ] = __ldcs( &X[ v_idx ] );
        }
    }

    /**
     * @brief Split row-major input X into column-major Q, K, V (FP16).
     *
     * Same behavior as the FP32 kernel but for half precision buffers.
     *
     * Preconditions:
     * - X size >= B * T * 3 * (NH * HS)
     * - Q, K, V size >= B * NH * HS * T
     *
     * Side-effects:
     * - Writes to device buffers Q, K, V.
     *
     * @param Q Output queries in column-major [B, NH, HS, T].
     * @param K Output keys in column-major [B, NH, HS, T].
     * @param V Output values in column-major [B, NH, HS, T].
     * @param X Input concatenated QKV in row-major [B, T, 3 * NH * HS].
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

            // Column-major [HS, T]: column t starts at t*HS, row hs is offset hs
            int out_idx = b * (NH * HS * T) + nh * (HS * T) + t * HS + hs;

            Q[ out_idx ] = __ldcs( &X[ q_idx ] );
            K[ out_idx ] = __ldcs( &X[ k_idx ] );
            V[ out_idx ] = __ldcs( &X[ v_idx ] );
        }
    }

    // ========================================================================
    // Output Unpermute - Column Major
    // ========================================================================

    /**
     * @brief Reorder column-major vaccum into row-major out (FP32).
     *
     * Reads vaccum organized as column-major per-head [B, NH, HS, T] and
     * writes out as row-major [B, T, C] where C = NH * HS.
     *
     * Preconditions:
     * - vaccum size >= B * NH * HS * T
     * - out size >= B * T * C
     *
     * Side-effects:
     * - Writes to device buffer out.
     *
     * @param vaccum Input column-major buffer [B, NH, HS, T].
     * @param out Output row-major buffer [B, T, C].
     * @param B Batch size.
     * @param T Sequence length.
     * @param NH Number of heads.
     * @param HS Head size.
     */
    __global__ void unpermute_output_fp32_kernel(
        const float* vaccum, float* out,
        int B, int T, int NH, int HS )
    {
        // vaccum: [B, NH, HS, T] in column-major (output from permute kernel)
        // out: [B, T, C] in row-major where C = NH * HS

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int C = NH * HS;

        if ( idx < B * T * C ) {
            const int b = idx / (T * C);
            int rest = idx % (T * C);
            const int t = rest / C;
            const int c = rest % C;
            const int nh = c / HS;
            const int hs = c % HS;

            // Note: Despite dimension parameters [HS, T], cuBLASLt outputs the natural
            // result of att[T,T] @ V^T[T,HS] which is column-major [T, HS].
            // Memory layout: T is contiguous (stride 1), HS is strided (stride T)
            const int vaccum_idx = b * (NH * HS * T) + nh * (HS * T) + hs * T + t;

            out[ idx ] = vaccum[ vaccum_idx ];
        }
    }

    /**
     * @brief Reorder column-major vaccum into row-major out (FP16).
     *
     * FP16 variant of unpermute_output_fp32_kernel.
     *
     * Preconditions:
     * - vaccum size >= B * NH * HS * T
     * - out size >= B * T * C
     *
     * Side-effects:
     * - Writes to device buffer out.
     *
     * @param vaccum Input column-major buffer [B, NH, HS, T].
     * @param out Output row-major buffer [B, T, C].
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

            // Column-major [HS, T]: column t starts at t*HS, row hs is offset hs
            int vaccum_idx = (b * (NH * HS * T)) + (nh * (HS * T)) + (t * HS) + hs;

            out[ idx ] = vaccum[ vaccum_idx ];
        }
    }

    // ========================================================================
    // Backward - Column Major
    // ========================================================================

    /**
     * @brief Scatter row-major dout into column-major dvaccum (FP32).
     *
     * Reads dout [B, T, C] and writes to dvaccum in column-major per-head layout
     * [B, NH, HS, T].
     *
     * Preconditions:
     * - dout size >= B * T * C
     * - dvaccum size >= B * NH * HS * T
     *
     * Side-effects:
     * - Writes to device buffer dvaccum.
     *
     * @param dvaccum Output column-major gradient buffer [B, NH, HS, T].
     * @param dout Input row-major gradient buffer [B, T, C].
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

            // Column-major [HS, T]: column t starts at t*HS, row hs is offset hs
            int dvaccum_idx = (b * (NH * HS * T)) + (nh * (HS * T)) + (t * HS) + hs;

            dvaccum[ dvaccum_idx ] = dout[ idx ];
        }
    }

    /**
     * @brief Scatter row-major dout into column-major dvaccum (FP16).
     *
     * FP16 variant of unpermute_backward_fp32_kernel.
     *
     * Preconditions:
     * - dout size >= B * T * C
     * - dvaccum size >= B * NH * HS * T
     *
     * Side-effects:
     * - Writes to device buffer dvaccum.
     *
     * @param dvaccum Output column-major gradient buffer [B, NH, HS, T].
     * @param dout Input row-major gradient buffer [B, T, C].
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

            // Column-major [HS, T]: column t starts at t*HS, row hs is offset hs
            int dvaccum_idx = (b * (NH * HS * T)) + (nh * (HS * T)) + (t * HS) + hs;

            dvaccum[ dvaccum_idx ] = dout[ idx ];
        }
    }

    /**
     * @brief Pack column-major dq/dk/dv into row-major dinp (FP32) for upstream gradient.
     *
     * Reads column-major per-head gradients dq/dk/dv ([B, NH, HS, T]) and writes
     * them back into the original row-major concatenated layout dinp [B, T, 3*C].
     *
     * Preconditions:
     * - dq/dk/dv size >= B * NH * HS * T
     * - dinp size >= B * T * 3 * C
     *
     * Side-effects:
     * - Writes to device buffer dinp.
     *
     * @param dinp Output concatenated gradient buffer in row-major [B, T, 3*C].
     * @param dq Input column-major gradient buffer for Q [B, NH, HS, T].
     * @param dk Input column-major gradient buffer for K [B, NH, HS, T].
     * @param dv Input column-major gradient buffer for V [B, NH, HS, T].
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

            // Column-major [HS, T]: column t starts at t*HS, row hs is offset hs
            int in_idx = b * (NH * HS * T) + nh * (HS * T) + t * HS + hs;

            dinp[ dinp_q_idx ] = dq[ in_idx ];
            dinp[ dinp_k_idx ] = dk[ in_idx ];
            dinp[ dinp_v_idx ] = dv[ in_idx ];
        }
    }

    /**
     * @brief Pack column-major dq/dk/dv into row-major dinp (FP16).
     *
     * FP16 variant of permute_backward_fp32_kernel.
     *
     * Preconditions:
     * - dq/dk/dv size >= B * NH * HS * T
     * - dinp size >= B * T * 3 * C
     *
     * Side-effects:
     * - Writes to device buffer dinp.
     *
     * @param dinp Output concatenated gradient buffer in row-major [B, T, 3*C].
     * @param dq Input column-major gradient buffer for Q [B, NH, HS, T].
     * @param dk Input column-major gradient buffer for K [B, NH, HS, T].
     * @param dv Input column-major gradient buffer for V [B, NH, HS, T].
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

            // Column-major [HS, T]: column t starts at t*HS, row hs is offset hs
            int in_idx = b * (NH * HS * T) + nh * (HS * T) + t * HS + hs;

            dinp[ dinp_q_idx ] = dq[ in_idx ];
            dinp[ dinp_k_idx ] = dk[ in_idx ];
            dinp[ dinp_v_idx ] = dv[ in_idx ];
        }
    }

    // ========================================================================
    // Host Functions - FP32 - Column Major
    // ========================================================================

    /**
     * @brief Launch permute_qkv_fp32_kernel to split row-major X into column-major Q/K/V.
     *
     * This launcher computes thread grid size and dispatches the FP32 kernel on the
     * provided CUDA stream.
     *
     * Preconditions:
     * - Device pointers must be valid and sized as documented in the kernel Doxygen.
     *
     * Side-effects:
     * - Launches a CUDA kernel and checks for launch errors.
     *
     * @param Q Output pointer for queries (device), column-major [B, NH, HS, T].
     * @param K Output pointer for keys (device), column-major [B, NH, HS, T].
     * @param V Output pointer for values (device), column-major [B, NH, HS, T].
     * @param X Input pointer (device) in row-major [B, T, 3 * NH * HS].
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
     * Computes grid size and launches the FP32 unpermute kernel on the provided stream.
     *
     * @param vaccum Input column-major buffer [B, NH, HS, T].
     * @param out Output row-major buffer [B, T, C].
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
     * @param dvaccum Output column-major gradient buffer [B, NH, HS, T].
     * @param dout Input row-major gradient buffer [B, T, C].
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
     * @brief Launch permute_backward_fp32_kernel to pack column-major dq/dk/dv into dinp.
     *
     * @param dinp Output concatenated gradient buffer in row-major [B, T, 3*C].
     * @param dq Input column-major gradient buffer for Q [B, NH, HS, T].
     * @param dk Input column-major gradient buffer for K [B, NH, HS, T].
     * @param dv Input column-major gradient buffer for V [B, NH, HS, T].
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
    // Host Functions - FP16 - Column Major
    // ========================================================================

    /**
     * @brief Launch permute_qkv_fp16_kernel to split row-major X into column-major Q/K/V.
     *
     * FP16 launcher of cuda_permute_qkv_fp32.
     *
     * @param Q Output pointer for queries (device), column-major [B, NH, HS, T].
     * @param K Output pointer for keys (device), column-major [B, NH, HS, T].
     * @param V Output pointer for values (device), column-major [B, NH, HS, T].
     * @param X Input pointer (device) in row-major [B, T, 3 * NH * HS].
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
     * @brief Launch unpermute_output_fp16_kernel to reorder vaccum into out (FP16).
     *
     * @param vaccum Input column-major buffer [B, NH, HS, T].
     * @param out Output row-major buffer [B, T, C].
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
     * @brief Launch unpermute_backward_fp16_kernel to scatter dout into dvaccum (FP16).
     *
     * @param dvaccum Output column-major gradient buffer [B, NH, HS, T].
     * @param dout Input row-major gradient buffer [B, T, C].
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
     * @brief Launch permute_backward_fp16_kernel to pack column-major dq/dk/dv into dinp (FP16).
     *
     * @param dinp Output concatenated gradient buffer in row-major [B, T, 3*C].
     * @param dq Input column-major gradient buffer for Q [B, NH, HS, T].
     * @param dk Input column-major gradient buffer for K [B, NH, HS, T].
     * @param dv Input column-major gradient buffer for V [B, NH, HS, T].
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