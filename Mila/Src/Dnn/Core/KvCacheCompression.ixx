/**
 * @file KvCacheCompression.ixx
 * @brief The KV-cache storage format a deployment asks for.
 */

module;

export module Dnn.KvCacheCompression;

namespace Mila::Dnn
{
    /**
     * @brief KV cache storage and compression strategy for GroupedQueryAttention.
     *
     * Maps to the TKvCachePolicy template parameter on GroupedQueryAttention and
     * CudaGqaOp via the load() runtime->compile-time bridge. The mapping is:
     *
     *   None  -> NoKvCompression      (BF16 cache, no compression overhead)
     *   FP8   -> PerChannelKvFp8<>    (FP8_E4M3 cache, per-head per-token float32 scales)
     *
     * New compression algorithms add a value here and a corresponding policy struct in
     * KvCache.QuantPolicy -- no other changes are required at this level.
     */
    export enum class KvCacheCompression
    {
        None,   ///< No compression -- default; BF16 KV cache.
        FP8,    ///< FP8_E4M3 per-head per-token KV cache compression.
    };
}
