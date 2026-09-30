/**
 * @file PerGroupInt4.ixx
 * @brief Symmetric per-group INT4 weight quantization; at a group of 32 it is Q4_0.
 */

export module Dnn.Quantization.Weight.PerGroupInt4;

import Dnn.TensorDataType;
import Dnn.Quantization.Weight.WeightQuantPolicy;

namespace Mila::Dnn::Quant::Weight
{
    /**
     * @brief Packed 4-bit codes with one FP16 scale per group of kGroupSize input channels.
     *
     * Dequantized value: ( code - 8 ) * scale. The layout and the reference rounding are normative in
     * Int4Packing.ixx; see Specifications/Quantization.md, "Q4_0".
     */
    export template<int kGroupSize = 32>
        struct PerGroupInt4
    {
        static constexpr bool kIsQuantized = true;
        static constexpr TensorDataType kStorageDtype = TensorDataType::UINT8;
        static constexpr TensorDataType kScaleDtype = TensorDataType::FP16;
        static constexpr bool kPerChannel = false;
        static constexpr int kQuantizationGroupSize = kGroupSize;
        static constexpr bool kIsFp4E2M1 = false;
        static constexpr bool kIsInt4 = true;
        static constexpr int kStorageBitsPerElement = 4;
    };

    static_assert( WeightQuantPolicy<PerGroupInt4<>> );
}
