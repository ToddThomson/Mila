/**
 * @file HasInt6Codes.ixx
 * @brief Detects a per-group weight format whose codes are unsigned 6-bit integers offset by 32.
 */

module;
#include <concepts>

export module Dnn.Quantization.Weight.HasInt6Codes;

namespace Mila::Dnn::Quant::Weight
{
    /**
     * @brief A row holds its codes' low nibbles, then their high two bits; a weight is ( code - 32 ) * its group's
     * scale.
     *
     * Detection is structural, so a policy defined outside the core satisfies it without the core naming it.
     */
    export template<typename T>
        concept HasInt6Codes = requires
    {
        { T::kIsInt6 } -> std::convertible_to<bool>;
        { T::kQuantizationGroupSize } -> std::convertible_to<int>;
    } && T::kIsQuantized && T::kIsInt6;
}
