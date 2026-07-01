// Gqa.Cache.Fp32.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include "CudaUtils.h"
#include "CudaGqa.cuh"

namespace Mila::Dnn::Compute::Cuda::Gqa
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
    __global__ void kvcache_write_q_fp32_kernel(
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
    __global__ void kvcache_write_kv_fp32_kernel(
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

        // Ring-buffer write: T_max is the cache row count (capacity). For the
        // unbounded cache T_max == context length and start_pos + t < T_max, so the
        // modulo is the identity and this is byte-identical to a linear write. For
        // the bounded sliding-window ring (SlidingWindowKvCache) T_max == capacity
        // and the wrap evicts the oldest key. See SlidingWindowKvCache.md D6.
        const int kv_pos = (start_pos + t) % T_max;

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

    __global__ void kvcache_expand_kv_fp32_kernel(
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
    void cuda_gqa_kvcache_write_q_fp32(
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

        kvcache_write_q_fp32_kernel << <grid_size, block_size, 0, stream >> > (
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
    void cuda_gqa_kvcache_write_kv_fp32(
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

        kvcache_write_kv_fp32_kernel <<< grid_size, block_size, 0, stream >>> (
            K, V, Xk, Xv,
            batch, chunk_len,
            NKV, HS,
            start_pos, max_seq_len);

        cudaCheck( cudaGetLastError() );
    }

    void cuda_gqa_kvcache_expand_kv_fp32(
        float* k_exp, float* v_exp,
        const float* k_compact, const float* v_compact,
        int B, int chunk_len, int T_stride, int NH, int NKV, int HS,
        int position_offset,
        cudaStream_t stream )
    {
        int block_size = 256;
        int total = B * NH * chunk_len * HS;
        int num_blocks = ceil_div( total, block_size );

        kvcache_expand_kv_fp32_kernel <<< num_blocks, block_size, 0, stream >>> (
            k_exp, v_exp,
            k_compact, v_compact,
            B, chunk_len, T_stride, NH, NKV, HS,
            position_offset);

        cudaCheck( cudaGetLastError() );
    }


}