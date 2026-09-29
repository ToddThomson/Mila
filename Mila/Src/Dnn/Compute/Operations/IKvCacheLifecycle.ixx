/**
 * @file IKvCacheLifecycle.ixx
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
     * may implement both IPositionalUnaryOp and IKvCacheLifecycle.
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
         * @brief Whether positions [0, position) of this cache can be reused by a
         * subsequent partial prefill (PromptCaching.md), device K/V contents untouched.
         *
         * The number of positions the cache holds is the network's, passed as
         * `cached_length`: the op keeps no fill count of its own, so a decode step
         * leaves no host state behind (DecodeGraph.md section 4.3). The network
         * applies an accepted rewind by setting its cached length to `position`.
         *
         * @return false when reuse would be incorrect -- e.g. position exceeds
         * cached_length, or a bounded sliding-window ring has already overwritten
         * the window a continuation from `position` would attend to. On false the
         * caller falls back to a full prefill, which positionally overwrites
         * regardless of cache state.
         */
        virtual bool rewindKvCache( dim_t position, dim_t cached_length ) = 0;

        virtual ~IKvCacheLifecycle() = default;
    };
}