/**
 * @file PerGroupCodebook3.ixx
 * @brief 3-bit codebook weight quantization.
 */

export module Dnn.Quantization.Weight.PerGroupCodebook3;

import Dnn.TensorDataType;
import Dnn.Quantization.Weight.WeightQuantPolicy;

namespace Mila::Dnn::Quant::Weight
{
    /**
     * @brief 3-bit codes into an 8-entry per-tensor table, FP16 scale per group: 3.25 bits per weight at 64.
     *
     * The low two bits of each code pack as the 2-bit format in the primary tensor and the third bit in a
     * separate byte-aligned plane, so kStorageBitsPerElement describes the primary tensor only. Codes are
     * fitted offline; the layout is normative in CodebookPacking.ixx.
     */
    export template<int kGroupSize = 64>
        struct PerGroupCodebook3
    {
        static constexpr bool kIsQuantized = true;
        static constexpr TensorDataType kStorageDtype = TensorDataType::UINT8;
        static constexpr TensorDataType kScaleDtype = TensorDataType::FP16;
        static constexpr bool kPerChannel = false;
        static constexpr int kQuantizationGroupSize = kGroupSize;
        static constexpr bool kIsFp4E2M1 = false;
        static constexpr int kStorageBitsPerElement = 2;
        static constexpr bool kIsCodebook = true;
        static constexpr int kCodeBits = 3;
        static constexpr int kCodebookEntries = 8;
        static constexpr bool kHasHighBitPlane = true;
    };

    static_assert( WeightQuantPolicy<PerGroupCodebook3<>> );
    static_assert( WeightQuantPolicy<PerGroupCodebook3<128>> );
}
