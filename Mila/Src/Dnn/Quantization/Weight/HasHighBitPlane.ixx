/**
 * @file HasHighBitPlane.ixx
 * @brief Detects a weight format that stores one bit per element in a second plane.
 */

module;
#include <concepts>

export module Dnn.Quantization.Weight.HasHighBitPlane;

namespace Mila::Dnn::Quant::Weight
{
    /**
     * @brief The format spills one bit per element into a second byte-aligned plane, so the primary tensor's
     * kStorageBitsPerElement does not account for the whole code.
     */
    export template<typename T>
        concept HasHighBitPlane = requires
    {
        { T::kHasHighBitPlane } -> std::convertible_to<bool>;
    } && T::kHasHighBitPlane;
}
