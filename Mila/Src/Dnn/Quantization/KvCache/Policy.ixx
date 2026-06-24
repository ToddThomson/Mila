/**
 * @file Policy.ixx
 * @brief KV cache compression policy concept and identity struct.
 *
 * Establishes the KvCachePolicy extension point consumed by GroupedQueryAttention
 * and CudaGqaOp. All active compression policies (PerChannelKvFp8, future SlidingWindow, future MLA)
 * satisfy this concept. GroupedQueryAttention constrains its TKvPolicy
 * parameter to KvCachePolicy.
 *
 * The concept is intentionally minimal: only kIsActive is required. Active
 * policy refinements (QuantKvPolicy) impose additional field requirements in
 * their own modules. A SlidingWindowPolicy satisfies KvCachePolicy without
 * carrying dtype fields at all.
 */

module;
#include <concepts>

export module Dnn.Quantization.KvCache.Policy;

namespace Mila::Dnn::Quant::KvCache
{
    /**
     * @brief Minimum contract for all KV cache management policies.
     *
     * Requires only kIsActive. Active policies set kIsActive = true and provide
     * algorithm-specific fields consumed by CudaGqaOp dispatch via if constexpr.
     * Active policy refinements such as QuantKvPolicy impose additional field
     * requirements in their own modules. A SlidingWindowPolicy satisfies
     * KvCachePolicy without carrying dtype fields.
     *
     * @tparam T The type to constrain against the policy contract.
     */
    export template<typename T>
        concept KvCachePolicy = requires
    {
        { T::kIsActive } -> std::convertible_to<bool>;
    };

    /**
     * @brief Identity policy - no compression.
     *
     * Zero storage overhead and zero runtime cost; all if constexpr branches
     * on kKvCompressed compile away entirely. Default for all
     * GroupedQueryAttention instantiations.
     *
     * When this policy is active, kCacheDtype on GroupedQueryAttention falls
     * back to TComputePrecision, so no kStorageDtype field is required or
     * present.
     */
    export struct NoKvCompression
    {
        static constexpr bool kIsActive = false;
    };

    // Verify the identity satisfies the concept at definition time.
    static_assert(KvCachePolicy<NoKvCompression>);
}
