// CudaGqa.Prefill.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include "CudaUtils.h"
#include "CudaGqa.cuh"

namespace Mila::Dnn::Compute::Cuda::GroupedQueryAttention
{
    /**
     * @brief Permute Q from [B, chunk_len, NH*HS] to [B, NH, T_max, HS].
     *
     * Each thread handles one scalar element. The source is a contiguous
     * row-major Q tensor produced by TensorOps::split. The destination is
     * the head-major KV-cache buffer, written at absolute position
     * kv_pos = start_pos + t.
     *
     * @param Q         Output buffer [B * NH * T_max * HS], device memory.
     * @param X         Input buffer  [B * chunk_len * NH * HS], device memory.
     * @param B         Batch size.
     * @param chunk_len Number of tokens in this prefill chunk.
     * @param NH        Number of query heads.
     * @param HS        Head dimension (elements per head).
     * @param start_pos Absolute token position of the first token in this chunk.
     * @param T_max     Maximum sequence length (KV cache capacity).
     */
    __global__ void prefill_permute_q_fp32_kernel(
        float* Q,
        const float* X,
        int B, int chunk_len,
        int NH, int HS,
        int start_pos,
        int T_max )
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx >= B * NH * chunk_len * HS )
        {
            return;
        }

        const int b = idx / (NH * chunk_len * HS);
        int rest = idx % (NH * chunk_len * HS);
        const int nh = rest / (chunk_len * HS);
        rest %= (chunk_len * HS);
        const int t = rest / HS;
        const int hs = rest % HS;

        const int kv_pos = start_pos + t;

        const int out_idx =
            b * (NH * T_max * HS)
            + nh * (T_max * HS)
            + kv_pos * HS
            + hs;

        const int src_idx =
            b * (chunk_len * NH * HS)
            + t * (NH * HS)
            + nh * HS
            + hs;

        Q[ out_idx ] = __ldcs( &X[ src_idx ] );
    }

    /**
     * @brief Permute K and V from [B, chunk_len, NKV*HS] to [B, NKV, T_max, HS].
     *
     * Each thread handles one scalar element for both K and V simultaneously,
     * since they share identical source and destination stride arithmetic.
     * The destination is the head-major KV-cache buffer, written at absolute
     * position kv_pos = start_pos + t.
     *
     * @param K         Output K buffer [B * NKV * T_max * HS], device memory.
     * @param V         Output V buffer [B * NKV * T_max * HS], device memory.
     * @param Xk        Input K buffer  [B * chunk_len * NKV * HS], device memory.
     * @param Xv        Input V buffer  [B * chunk_len * NKV * HS], device memory.
     * @param B         Batch size.
     * @param chunk_len Number of tokens in this prefill chunk.
     * @param NKV       Number of key/value heads.
     * @param HS        Head dimension (elements per head).
     * @param start_pos Absolute token position of the first token in this chunk.
     * @param T_max     Maximum sequence length (KV cache capacity).
     */
    __global__ void prefill_permute_kv_fp32_kernel(
        float* K, float* V,
        const float* Xk, const float* Xv,
        int B, int chunk_len,
        int NKV, int HS,
        int start_pos,
        int T_max )
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx >= B * NKV * chunk_len * HS )
        {
            return;
        }

        const int b = idx / (NKV * chunk_len * HS);
        int rest = idx % (NKV * chunk_len * HS);
        const int nkv = rest / (chunk_len * HS);
        rest %= (chunk_len * HS);
        const int t = rest / HS;
        const int hs = rest % HS;

        const int kv_pos = start_pos + t;

        const int out_idx =
            b * (NKV * T_max * HS)
            + nkv * (T_max * HS)
            + kv_pos * HS
            + hs;

        const int src_idx =
            b * (chunk_len * NKV * HS)
            + t * (NKV * HS)
            + nkv * HS
            + hs;

        K[ out_idx ] = __ldcs( &Xk[ src_idx ] );
        V[ out_idx ] = __ldcs( &Xv[ src_idx ] );
    }

    __global__ void prefill_softmax_fp32_kernel_v2(
        float* att,
        const float* preatt,
        int B,
        int NH,
        int T_stride,
        int chunk_stride,
        int chunk_len,
        int position_offset )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_rows = B * NH * chunk_len;

        if ( idx >= total_rows ) 
            return;

        // Decode batch, head, and query index within chunk
        int b_nh = idx / chunk_len;   // flattened batch * heads
        int t = idx % chunk_len;  // query index in this chunk

        int b = b_nh / NH;
        int nh = b_nh % NH;

        // Compute row start for this query
        int row_offset = ((b * NH + nh) * chunk_stride + t) * T_stride;

        const float* preatt_row = preatt + row_offset;
        float* att_row = att + row_offset;

        // Compute causal mask limit: cannot attend to future tokens
        int abs_t = position_offset + t;        // global query index
        int max_t2 = min( abs_t, T_stride - 1 );  // last key index to attend

        // Step 1: find max for numerical stability
        float max_val = -CUDART_INF_F;
        for ( int t2 = 0; t2 <= max_t2; ++t2 )
            max_val = fmaxf( max_val, preatt_row[ t2 ] );

        // Step 2: exponentiate & sum
        float sum = 0.0f;
        for ( int t2 = 0; t2 <= max_t2; ++t2 )
        {
            float val = expf( preatt_row[ t2 ] - max_val );
            sum += val;
            att_row[ t2 ] = val;
        }

        // Step 3: normalize
        float inv_sum = 1.0f / sum;
        for ( int t2 = 0; t2 <= max_t2; ++t2 )
            att_row[ t2 ] *= inv_sum;

        // Step 4: zero out future tokens
        for ( int t2 = max_t2 + 1; t2 < T_stride; ++t2 )
            att_row[ t2 ] = 0.0f;
    }

    // -----------------------------------------------------------------------
    // prefill_expand_kv_fp32_kernel
    //
    // Broadcasts NKV heads → NH heads and writes into the correct
    // position slot within the full-context k_exp / v_exp buffers.
    //
    // k_compact / v_compact : [B, NKV, chunk_len, HS]  (packed)
    // k_exp     / v_exp     : [B, NH,  T_stride,  HS]  (full context)
    // -----------------------------------------------------------------------

    __global__ void prefill_expand_kv_fp32_kernel(
        float* k_exp, float* v_exp,
        const float* k_compact, const float* v_compact,
        int B, int chunk_len,
        int T_stride, int NH, int NKV, int HS,
        int position_offset )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if ( idx >= B * NH * chunk_len * HS )
        {
            return;
        }

        const int b = idx / (NH * chunk_len * HS);
        int rest = idx % (NH * chunk_len * HS);
        const int nh = rest / (chunk_len * HS);
        rest = rest % (chunk_len * HS);
        const int t = rest / HS;
        const int hs = rest % HS;

        const int nkv = nh / (NH / NKV);

        int src_idx = b * (NKV * T_stride * HS)
            + nkv * (T_stride * HS)
            + (position_offset + t) * HS
            + hs;

        int dst_idx = b * (NH * T_stride * HS)
            + nh * (T_stride * HS)
            + (position_offset + t) * HS
            + hs;

        k_exp[ dst_idx ] = k_compact[ src_idx ];
        v_exp[ dst_idx ] = v_compact[ src_idx ];
    }

    __global__ void gqa_prefill_unpermute_output_padded_fp32_kernel(
        const float* __restrict__ vaccum,
        float* __restrict__ out,
        int B, int actual_T, int padded_T,
        int NQH, int HS )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int C = NQH * HS; // channels per token

        // Total number of output elements: B * actual_T * NQH * HS
        int total = B * actual_T * C;

        if ( idx >= total ) return;

        // Decode flat index → (b, t, nh, hs)
        const int b = idx / (actual_T * C);
        int rest = idx % (actual_T * C);
        const int t = rest / C;
        const int c = rest % C;
        const int nh = c / HS;
        const int hs = c % HS;

        // Read from vaccum: [B][NQH][padded_T][HS]
        const int in_idx =
            b * (NQH * padded_T * HS)
            + nh * (padded_T * HS)
            + t * HS
            + hs;

        // Write to out: [B][actual_T][NQH][HS]
        out[ idx ] = vaccum[ in_idx ];
    }

    // -----------------------------------------------------------------------
    // Prefill launchers
    // -----------------------------------------------------------------------

    void cuda_gqa_prefill_softmax_fp32(
        float* att, const float* preatt,
        int B, int NH, int T_stride, int chunk_stride,
        int chunk_len, int position_offset,
        cudaStream_t stream )
    {
        int total_rows = B * NH * chunk_len;
        int block_size = 256;
        int grid_size = (total_rows + block_size - 1) / block_size;

        prefill_softmax_fp32_kernel_v2 <<<grid_size, block_size, 0, stream >>> (
            att, preatt,
            B, NH, T_stride, chunk_stride,
            chunk_len, position_offset);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_gqa_prefill_expand_kv_fp32(
        float* k_exp, float* v_exp,
        const float* k_compact, const float* v_compact,
        int B, int chunk_len, int T_stride, int NH, int NKV, int HS,
        int position_offset,
        cudaStream_t stream )
    {
        int block_size = 256;
        int total = B * NH * chunk_len * HS;
        int num_blocks = ceil_div( total, block_size );

        prefill_expand_kv_fp32_kernel <<<num_blocks, block_size, 0, stream>>> (
            k_exp, v_exp,
            k_compact, v_compact,
            B, chunk_len, T_stride, NH, NKV, HS,
            position_offset);

        cudaCheck( cudaGetLastError() );
    }

    // =========================================================================
    // Host launchers
    // =========================================================================

    /**
     * @brief Host launcher for Q permute: [B, chunk_len, NH*HS] → [B, NH, T_max, HS].
     *
     * @param Q           Output Q buffer [B * NH * T_max * HS], device memory.
     * @param X           Input Q buffer  [B * chunk_len * NH * HS], device memory.
     * @param batch       Batch size.
     * @param chunk_len   Number of tokens in this prefill chunk.
     * @param NH          Number of query heads.
     * @param HS          Head dimension.
     * @param start_pos   Absolute token position of the first chunk token.
     * @param max_seq_len KV cache capacity.
     * @param stream      CUDA stream for kernel scheduling.
     */
    void cuda_gqa_prefill_permute_q_fp32(
        float* Q,
        const float* X,
        int batch, int chunk_len,
        int NH, int HS,
        int start_pos, int max_seq_len,
        cudaStream_t stream )
    {
        const int total = batch * NH * chunk_len * HS;
        const int block_size = 256;
        const int grid_size = ceil_div( total, block_size );

        prefill_permute_q_fp32_kernel << <grid_size, block_size, 0, stream >> > (
            Q, X,
            batch, chunk_len,
            NH, HS,
            start_pos, max_seq_len);

        cudaCheck( cudaGetLastError() );
    }

    /**
     * @brief Host launcher for KV permute: [B, chunk_len, NKV*HS] → [B, NKV, T_max, HS].
     *
     * K and V are permuted in a single kernel launch since they share identical
     * stride arithmetic and are both needed for attention computation.
     *
     * @param K           Output K buffer [B * NKV * T_max * HS], device memory.
     * @param V           Output V buffer [B * NKV * T_max * HS], device memory.
     * @param Xk          Input K buffer  [B * chunk_len * NKV * HS], device memory.
     * @param Xv          Input V buffer  [B * chunk_len * NKV * HS], device memory.
     * @param batch       Batch size.
     * @param chunk_len   Number of tokens in this prefill chunk.
     * @param NKV         Number of key/value heads.
     * @param HS          Head dimension.
     * @param start_pos   Absolute token position of the first chunk token.
     * @param max_seq_len KV cache capacity.
     * @param stream      CUDA stream for kernel scheduling.
     */
    void cuda_gqa_prefill_permute_kv_fp32(
        float* K, float* V,
        const float* Xk, const float* Xv,
        int batch, int chunk_len,
        int NKV, int HS,
        int start_pos, int max_seq_len,
        cudaStream_t stream )
    {
        const int total = batch * NKV * chunk_len * HS;
        const int block_size = 256;
        const int grid_size = ceil_div( total, block_size );

        prefill_permute_kv_fp32_kernel <<< grid_size, block_size, 0, stream >>> (
            K, V, Xk, Xv,
            batch, chunk_len,
            NKV, HS,
            start_pos, max_seq_len);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_gqa_prefill_unpermute_output_padded_fp32(
        const float* vaccum, float* out,
        int B, int actual_T, int padded_T,
        int NQH, int HS,
        cudaStream_t stream )
    {
        const int block_size = 256;
        const int total = B * actual_T * NQH * HS;
        const int num_blocks = (total + block_size - 1) / block_size;

        gqa_prefill_unpermute_output_padded_fp32_kernel <<< num_blocks, block_size, 0, stream >>> (
                vaccum, out, B, actual_T, padded_T, NQH, HS );

        cudaCheck( cudaGetLastError() );
    }

}