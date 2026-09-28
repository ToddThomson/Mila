/**
 * @file PerGroupCodebook2.ixx
 * @brief 2-bit codebook weight quantization.
 */

export module Dnn.Quantization.Weight.PerGroupCodebook2;

import Dnn.TensorDataType;
import Dnn.Quantization.Weight.WeightQuantPolicy;

namespace Mila::Dnn::Quant::Weight
{
    /**
     * @brief 2-bit codes into a 4-entry per-tensor table, FP16 scale per group: 2.5 bits per weight at 32.
     *
     * Codes are fitted offline and uploaded; there is no quantize-on-load path. The layout is normative in
     * CodebookPacking.ixx.
     */
    export template<int kGroupSize = 32>
        struct PerGroupCodebook2
    {
        static constexpr bool kIsQuantized = true;
        static constexpr TensorDataType kStorageDtype = TensorDataType::UINT8;
        static constexpr TensorDataType kScaleDtype = TensorDataType::FP16;
        static constexpr bool kPerChannel = false;
        static constexpr int kQuantizationGroupSize = kGroupSize;
        static constexpr bool kIsFp4E2M1 = false;
        static constexpr int kStorageBitsPerElement = 2;
        static constexpr bool kIsCodebook = true;
        static constexpr int kCodeBits = 2;
        static constexpr int kCodebookEntries = 4;
        static constexpr bool kHasHighBitPlane = false;
    };

    static_assert( WeightQuantPolicy<PerGroupCodebook2<>> );
    static_assert( WeightQuantPolicy<PerGroupCodebook2<64>> );
}
