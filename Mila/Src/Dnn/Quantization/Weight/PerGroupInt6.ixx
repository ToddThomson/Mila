/**
 * @file PerGroupInt6.ixx
 * @brief Symmetric per-group INT6 weight quantization: six-bit codes and one FP16 scale per 32 elements of a row.
 */

export module Dnn.Quantization.Weight.PerGroupInt6;

import Dnn.TensorDataType;
import Dnn.Quantization.Weight.WeightQuantPolicy;

namespace Mila::Dnn::Quant::Weight
{
    /**
     * @brief Six-bit codes with one FP16 scale per group of kGroupSize input channels, 6.5 bits a weight.
     *
     * Dequantized value: ( code - 32 ) * scale. Q4_0's rule widened to six bits; the layout and the reference
     * rounding are normative in Int6Packing.ixx. Gemma's tied embedding and head table is stored this way
     * whenever its body is quantized (Specifications/Quantization.md, "The tied table -- six bits per 32").
     */
    export template<int kGroupSize = 32>
        struct PerGroupInt6
    {
        static constexpr bool kIsQuantized = true;
        static constexpr TensorDataType kStorageDtype = TensorDataType::UINT8;
        static constexpr TensorDataType kScaleDtype = TensorDataType::FP16;
        static constexpr bool kPerChannel = false;
        static constexpr int kQuantizationGroupSize = kGroupSize;
        static constexpr bool kIsFp4E2M1 = false;
        static constexpr bool kIsInt4 = false;
        static constexpr bool kIsInt6 = true;
        static constexpr int kStorageBitsPerElement = 6;
    };

    static_assert( WeightQuantPolicy<PerGroupInt6<>> );
}
