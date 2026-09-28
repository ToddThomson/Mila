/**
 * @file WeightQuantPolicy.ixx
 * @brief The concept every weight quantization policy satisfies.
 */

module;
#include <concepts>

export module Dnn.Quantization.Weight.WeightQuantPolicy;

import Dnn.TensorDataType;

namespace Mila::Dnn::Quant::Weight
{
    /**
     * @brief Any type satisfying this concept may be the TWeightQuant parameter of Linear and CudaLinearOp.
     *
     * Only the fields Linear needs for compile-time decisions are required. NoWeightQuant satisfies it with
     * sentinel values that every consumer guards behind if constexpr ( kIsQuantized ).
     */
    export template<typename T>
        concept WeightQuantPolicy = requires
    {
        { T::kIsQuantized } -> std::convertible_to<bool>;
        { T::kStorageDtype } -> std::convertible_to<TensorDataType>;
        { T::kScaleDtype } -> std::convertible_to<TensorDataType>;
        { T::kPerChannel } -> std::convertible_to<bool>;

        // Bits of the PRIMARY weight tensor per logical element, which sizes its allocation. A format that
        // spills into a companion plane counts only the primary tensor; the companion carries its own extent.
        { T::kStorageBitsPerElement } -> std::convertible_to<int>;
    };
}
