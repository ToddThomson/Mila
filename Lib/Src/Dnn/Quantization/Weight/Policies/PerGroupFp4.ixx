/**
 * @file PerGroupFp4.ixx
 * @brief Per-group FP4 E2M1 weight quantization.
 */

export module Dnn.Quantization.Weight.PerGroupFp4;

import Dnn.TensorDataType;
import Dnn.Quantization.Weight.WeightQuantPolicy;

namespace Mila::Dnn::Quant::Weight
{
    /**
     * @brief Packed FP4 E2M1 nibbles with one FP32 scale per group, scale = max(|W[group]|) / 6.
     *
     * Two nibbles per byte, low nibble = even column. A nibble is sign-magnitude: bit 3 is the sign and
     * bits 0-2 index {0, 0.5, 1, 1.5, 2, 3, 4, 6}. Dequantized value: decode( nibble ) * scale.
     */
    export template<int kGroupSize = 128>
        struct PerGroupFp4
    {
        static constexpr bool kIsQuantized = true;
        static constexpr TensorDataType kStorageDtype = TensorDataType::UINT8;
        static constexpr TensorDataType kScaleDtype = TensorDataType::FP32;
        static constexpr bool kPerChannel = false;
        static constexpr int kQuantizationGroupSize = kGroupSize;
        static constexpr bool kIsFp4E2M1 = true;
        static constexpr int kStorageBitsPerElement = 4;
    };

    static_assert( WeightQuantPolicy<PerGroupFp4<>> );
    static_assert( WeightQuantPolicy<PerGroupFp4<64>> );
}
