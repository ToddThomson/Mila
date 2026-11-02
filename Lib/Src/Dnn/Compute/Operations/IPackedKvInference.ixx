/**
 * @file IPackedKvInference.ixx
 * @brief KV-cache inference interface for GPT-style packed-QKV MHA backends.
 */

module;

export module Compute.IPackedKvInference;

import Dnn.ITensor;
import Dnn.TensorTypes;
import Compute.IKvCacheLifecycle;

namespace Mila::Dnn::Compute
{
    /**
     * @brief KV-cache inference interface for packed-QKV MHA backends.
     *
     * Implemented by GPT-style MHA backends (e.g. CudaMultiHeadAttentionOp).
     * Uses fused QKV input throughout -- Q, K, and V are concatenated along the
     * feature axis and split internally by the backend kernel.
     *
     * Position is implicit: GPT-style MHA always begins prefill at position 0.
     * Absolute positional encoding is handled upstream by Lpe, not inside attention.
     *
     * Two-phase inference protocol:
     *   prefill -- populate the KV cache from the full prompt sequence.
     *   decode  -- process one autoregressive token against the accumulated cache.
     */
    export struct IPackedKvInference : IKvCacheLifecycle
    {
        /**
         * @brief Populate the KV cache from a packed QKV sequence and compute output.
         *
         * @param qkv    Packed QKV input [B, T, 3 * embedding_dim].
         * @param output Pre-allocated attention output [B, T, embedding_dim].
         */
        virtual void prefill( const ITensor& qkv, ITensor& output ) = 0;

        /**
         * @brief Process a single autoregressive token against the KV cache.
         *
         * @param input    Packed QKV single-token input [B, 1, 3 * embedding_dim].
         * @param output   Pre-allocated output [B, 1, embedding_dim].
         * @param position Zero-based absolute sequence position into the KV cache.
         */
        virtual void decode( const ITensor& input, ITensor& output, dim_t position ) = 0;

        ~IPackedKvInference() override = default;
    };
}