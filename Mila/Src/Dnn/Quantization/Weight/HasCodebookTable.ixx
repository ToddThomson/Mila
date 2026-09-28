/**
 * @file HasCodebookTable.ixx
 * @brief Detects a weight format that decodes through a per-tensor table.
 */

module;
#include <concepts>

export module Dnn.Quantization.Weight.HasCodebookTable;

namespace Mila::Dnn::Quant::Weight
{
    /**
     * @brief The format decodes through a per-tensor table, which travels as its own tensor.
     *
     * Detection is structural, so a policy defined outside the core satisfies it without the core naming it.
     */
    export template<typename T>
        concept HasCodebookTable = requires
    {
        { T::kIsCodebook } -> std::convertible_to<bool>;
        { T::kCodebookEntries } -> std::convertible_to<int>;
    } && T::kIsCodebook;
}
