/**
 * @file HasInt4Codes.ixx
 * @brief Detects a per-group weight format whose codes are unsigned 4-bit integers offset by 8.
 */

module;
#include <concepts>

export module Dnn.Quantization.Weight.HasInt4Codes;

namespace Mila::Dnn::Quant::Weight
{
    /**
     * @brief Two codes per byte, low nibble the even column; a weight is ( code - 8 ) * its group's scale.
     *
     * Detection is structural, so a policy defined outside the core satisfies it without the core naming it.
     */
    export template<typename T>
        concept HasInt4Codes = requires
    {
        { T::kIsInt4 } -> std::convertible_to<bool>;
        { T::kQuantizationGroupSize } -> std::convertible_to<int>;
    } && T::kIsQuantized && T::kIsInt4;
}
