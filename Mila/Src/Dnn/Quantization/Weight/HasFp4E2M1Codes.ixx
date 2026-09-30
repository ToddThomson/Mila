/**
 * @file HasFp4E2M1Codes.ixx
 * @brief Detects a per-group weight format whose codes are FP4 E2M1 nibbles.
 */

module;
#include <concepts>

export module Dnn.Quantization.Weight.HasFp4E2M1Codes;

namespace Mila::Dnn::Quant::Weight
{
    /**
     * @brief Two E2M1 nibbles per byte, low nibble the even column, one scale per group.
     *
     * Detection is structural, so a policy defined outside the core satisfies it without the core naming it.
     */
    export template<typename T>
        concept HasFp4E2M1Codes = requires
    {
        { T::kIsFp4E2M1 } -> std::convertible_to<bool>;
        { T::kQuantizationGroupSize } -> std::convertible_to<int>;
    } && T::kIsQuantized && T::kIsFp4E2M1;
}
