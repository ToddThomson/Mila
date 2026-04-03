/**
 * @file IKVAttentionOp.ixx
 * @brief KV-cache compute interface for modern attention backends (GQA and beyond).
 */

module;

export module Compute.IKvInference;

import Dnn.ITensor;
import Compute.IKvCacheLifecycle;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Compute interface for attention operations that maintain a KV cache.
     *
     * Extends IKVCacheLifecycle with the two-phase inference compute contract
     * used by modern transformer architectures (GQA, MQA, and derived models):
     *
     *   prefill — populate the KV cache from a (possibly chunked) prompt sequence.
     *             Q, K, V are passed separately to support upstream in-place RoPE
     *             rotation before cache insertion. An explicit position_offset
     *             supports chunked prefill over sequences longer than a single pass.
     *
     *   decode  — process a single autoregressive token against the accumulated
     *             KV cache. Input is packed QKV [B, 1, (n_heads + 2*n_kv_heads)*head_dim].
     *
     */
    export struct IKvInference : IKvCacheLifecycle
    {
        /**
         * @brief Populate the KV cache and compute attention output for a token chunk.
         *
         * @param q               Query  [B, T_chunk, n_heads    * head_dim].
         * @param k               Key    [B, T_chunk, n_kv_heads * head_dim].
         * @param v               Value  [B, T_chunk, n_kv_heads * head_dim].
         * @param output          Pre-allocated output [B, T_chunk, model_dim].
         * @param position_offset Absolute position of the first token in this chunk.
         */
        virtual void prefill(
            const ITensor& q,
            const ITensor& k,
            const ITensor& v,
            ITensor& output,
            int position_offset ) = 0;

        /**
         * @brief Process a single token at an explicit KV cache position.
         *
         * @param input    Packed QKV [B, 1, (n_heads + 2*n_kv_heads) * head_dim].
         * @param output   Pre-allocated output [B, 1, model_dim].
         * @param position Zero-based absolute sequence position into the KV cache.
         */
        virtual void decode(
            const ITensor& input,
            ITensor& output,
            int position ) = 0;

        ~IKvInference() override = default;
    };
}