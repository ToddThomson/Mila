/**
 * @file KvCacheView.ixx
 * @brief A read-only view of one attention layer's KV cache, for attention another component runs over it.
 */

export module Compute.KvCacheView;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Where one layer's cached keys and values live and how they are laid out.
     *
     * Keys are stored after their norm and RoPE, values after theirs, [B, num_kv_heads, capacity, head_size]
     * each; a position p lives in row p % capacity, which is p itself when the cache holds the whole context.
     * An FP8 cache holds E4M3 codes with one FP32 scale per row of each, [B, num_kv_heads, capacity]; a BF16
     * cache has null scales.
     *
     * Non-owning: the layer that owns the cache must outlive every reader, and a reader writes nothing.
     * Gemma 4's draft model attends the target's caches through it (Gemma4Mtp.md 4.3).
     */
    export struct KvCacheView
    {
        const void* keys{ nullptr };
        const void* values{ nullptr };
        const float* key_scales{ nullptr };
        const float* value_scales{ nullptr };

        int capacity{ 0 };
        int num_kv_heads{ 0 };
        int head_size{ 0 };

        /// Sliding window the cache was written for; 0 for a cache holding the whole context.
        int window{ 0 };

        bool fp8{ false };
    };
}
