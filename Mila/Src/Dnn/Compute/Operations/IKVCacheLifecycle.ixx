/**
 * @file IKVCacheLifecycle.ixx
 * @brief Interface for operations that own and manage a KV cache.
 */

module;

export module Compute.IKvCacheLifecycle;

import Dnn.TensorTypes;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Capability interface for KV-cache state management.
     *
     * Implemented by attention operations (GQA, MHA) that allocate and
     * maintain key/value caches across autoregressive decode steps.
     * This concern is orthogonal to positional dispatch -- an operation
     * may implement both IPositionalUnaryOp and IKVCacheLifecycle.
     */
    export struct IKvCacheLifecycle
    {
        /**
         * @brief Allocate the KV cache for a given batch size and maximum sequence length.
         *
         * @param batch_size          Number of sequences in the batch.
         * @param max_sequence_length Maximum number of tokens the cache must hold.
         */
        virtual void initializeKvCache( dim_t batch_size, dim_t max_sequence_length ) = 0;

        /**
         * @brief Reset the KV cache to an empty state, preserving the allocation.
         */
        virtual void resetKvCache() = 0;

        /**
         * @brief Rewind the logical cache fill position without touching device
         * K/V buffer contents, so positions [0, position) can be reused by a
         * subsequent partial prefill (PromptCaching.md).
         *
         * @return true when the rewind is valid and was applied. Implementations
         * must refuse (return false, cache state unchanged) when reuse would be
         * incorrect -- e.g. position exceeds the current fill, or a bounded
         * sliding-window ring has already overwritten the window a continuation
         * from `position` would attend to. On false the caller falls back to a
         * full prefill, which positionally overwrites regardless of cache state.
         */
        virtual bool rewindKvCache( dim_t position ) = 0;

        virtual ~IKvCacheLifecycle() = default;
    };
}