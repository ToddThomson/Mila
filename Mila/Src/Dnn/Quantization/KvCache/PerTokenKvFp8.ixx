/**
 * @file PerTokenKvFp8.ixx
 * @brief FP8 KV-cache compression with one scale per KV head per token.
 */

module;
#include <concepts>

export module Dnn.Quantization.KvCache.PerTokenKvFp8;

import Dnn.Quantization.KvCache.Policy;
import Dnn.Quantization.KvCache.QuantPolicy;
import Dnn.TensorDataType;

namespace Mila::Dnn::Quant::KvCache
{
    /**
     * @brief Symmetric FP8 KV-cache compression, one FP32 scale per KV head per token.
     *
     * Each cached row -- one KV head at one token -- is stored as E4M3 with scale max|x| / 448, computed when the row
     * is written; K and V alike. Attention reads the codes and scales on chip: the K scale multiplies each key's
     * score and the V scale each key's probability, in FP32, after the products they factor out of
     * (Quantization.md, Part III). Scales are [B, NKV, capacity].
     *
     * @tparam TStorage Storage dtype of the cached values; FP8_E4M3.
     */
    export template<TensorDataType TStorage = TensorDataType::FP8_E4M3>
    struct PerTokenKvFp8
    {
        static constexpr bool kIsActive = true;                                ///< Compresses the cache.
        static constexpr TensorDataType kStorageDtype = TStorage;              ///< Stored K and V values.
        static constexpr TensorDataType kScaleDtype = TensorDataType::FP32;    ///< Per-row scales.
        static constexpr bool kPerHeadPerToken = true;                         ///< One scale per KV head per token.
        static constexpr bool kSymmetric = true;                               ///< K and V share the policy.
    };

    static_assert( KvCachePolicy<PerTokenKvFp8<>> );
    static_assert( QuantKvPolicy<PerTokenKvFp8<>> );
}
