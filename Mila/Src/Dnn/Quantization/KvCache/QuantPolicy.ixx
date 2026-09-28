/**
 * @file QuantPolicy.ixx
 * @brief The concept a quantizing KV-cache policy satisfies: KvCachePolicy plus its storage and scale fields.
 */

module;
#include <concepts>

export module Dnn.Quantization.KvCache.QuantPolicy;

import Dnn.Quantization.KvCache.Policy;
import Dnn.TensorDataType;

namespace Mila::Dnn::Quant::KvCache
{
    /**
     * @brief A KV-cache policy that stores quantized values: the storage and scale dtypes and the scale granularity.
     *
     * NoKvCompression satisfies KvCachePolicy and not this; code that reads these fields guards on kIsActive first.
     * PerTokenKvFp8 is the one policy that satisfies it.
     *
     * @tparam T Candidate policy type.
     */
    export template<typename T>
    concept QuantKvPolicy = KvCachePolicy<T> && requires
    {
        { T::kStorageDtype } -> std::convertible_to<TensorDataType>;
        { T::kScaleDtype } -> std::convertible_to<TensorDataType>;
        { T::kPerHeadPerToken } -> std::convertible_to<bool>;
        { T::kSymmetric } -> std::convertible_to<bool>;
    };
}
