/**
 * @file IKVCacheLifecycle.ixx
 * @brief Interface for operations that own and manage a KV cache.
 */

module;

export module Compute.IKvCacheLifecycle;

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
        virtual void initializeKvCache( int batch_size, int max_sequence_length ) = 0;

        /**
         * @brief Reset the KV cache to an empty state, preserving the allocation.
         */
        virtual void resetKvCache() = 0;

        virtual ~IKvCacheLifecycle() = default;
    };
}