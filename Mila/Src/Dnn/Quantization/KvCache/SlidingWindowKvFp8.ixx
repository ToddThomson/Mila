/**
 * @file SlidingWindowKvFp8.ixx
 * @brief The bounded sliding-window ring holding FP8 codes: SlidingWindowKvCache's rows in PerTokenKvFp8's format.
 */

module;
#include <concepts>

export module Dnn.Quantization.KvCache.SlidingWindowKvFp8;

import Dnn.Quantization.KvCache.Policy;
import Dnn.Quantization.KvCache.QuantPolicy;
import Dnn.TensorDataType;

namespace Mila::Dnn::Quant::KvCache
{
    /**
     * @brief A sliding layer's KV cache as a ring of min(T, window + prefill_chunk - 1) rows, each row -- one KV head
     *        at one token -- stored as E4M3 codes with one FP32 scale, max|x| / 448, as PerTokenKvFp8 stores it.
     *
     * Valid only on layers with a positive window. Read only by the fused flash prefill and fused decode, as the
     * FP8 cache is (Quantization.md, Part III; SlidingWindowKvCache.md).
     */
    export struct SlidingWindowKvFp8
    {
        static constexpr bool kIsActive = true; ///< Compresses the cache.
        static constexpr TensorDataType kStorageDtype = TensorDataType::FP8_E4M3; ///< Stored K and V values.
        static constexpr TensorDataType kScaleDtype = TensorDataType::FP32; ///< Per-row scales.
        static constexpr bool kPerHeadPerToken = true; ///< One scale per KV head per token.
        static constexpr bool kSymmetric = true; ///< K and V share the policy.
    };

    static_assert( KvCachePolicy<SlidingWindowKvFp8> );
    static_assert( QuantKvPolicy<SlidingWindowKvFp8> );
}
