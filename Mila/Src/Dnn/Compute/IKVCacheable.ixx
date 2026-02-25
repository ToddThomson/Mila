/**
 * @file KVCacheable.ixx
 * @brief Optional capability interface for KV-cache-enabled operations.
 */

module;

export module Compute.KVCacheable;

import Dnn.ITensor;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Optional interface for KV-cache-capable Attention operations.
     *
     * Attention Operations that support cached decoding should implement this interface
     * and be accessed via dynamic_cast at call sites.
     */
    export struct IKVCacheable
    {
        virtual ~IKVCacheable() = default;

        virtual void initializeKVCache( int batch_size, int max_seq_length ) = 0;

        virtual void resetKVCache() = 0;

        virtual void forwardPrefill( const ITensor& input, ITensor& output ) = 0;

        virtual void forwardDecode( const ITensor& input, ITensor& output, int position ) = 0;
    };
}