// Mila - Dnn/Quantization/KvCache/Policy.ixx
// KV cache compression policy concept and identity struct.
//
// This module establishes the KvCachePolicy extension point. All active
// compression policies (PerChannelKvFp8, future SlidingWindow, future MLA)
// satisfy this concept. GroupedQueryAttention constrains its TKvPolicy
// parameter to KvCachePolicy.
//
// The concept is intentionally minimal: only kIsActive is required. Active
// policy refinements (QuantKvPolicy) impose additional field requirements in
// their own modules. A SlidingWindowPolicy satisfies KvCachePolicy without
// carrying dtype fields at all.
//
// Consumers: GroupedQueryAttention component and CudaGqaOp only.
// Neither WeightQuant nor any other subsystem imports this module.

module;
#include <concepts>

export module Dnn.Quantization.KvCache.Policy;

namespace Mila::Dnn::Quant::KvCache
{
    // -------------------------------------------------------------------------
    // KvCachePolicy concept
    //
    // Minimum contract for all KV cache management policies. Active policies
    // set kIsActive = true and provide algorithm-specific fields consumed by
    // CudaGqaOp dispatch via if constexpr.
    // -------------------------------------------------------------------------
    export template<typename T>
        concept KvCachePolicy = requires
    {
        { T::kIsActive } -> std::convertible_to<bool>;
    };

    // -------------------------------------------------------------------------
    // NoKvCompression
    //
    // Identity policy — no compression. Zero storage overhead, zero runtime
    // cost. All if constexpr branches on kKvCompressed compile away entirely.
    // Default for all GroupedQueryAttention instantiations.
    //
    // kCacheDtype on GroupedQueryAttention falls back to TComputePrecision when
    // this policy is active, so no kStorageDtype field is required or present.
    // -------------------------------------------------------------------------
    export struct NoKvCompression
    {
        static constexpr bool kIsActive = false;
    };

    // Verify the identity satisfies the concept at definition time.
    static_assert(KvCachePolicy<NoKvCompression>);

}
