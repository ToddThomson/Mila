/**
 * @file CodebookPolicies.ixx
 * @brief Sub-4-bit codebook weight quantization policies for the Qwen3.8 research track.
 * Experimental: outside the v0.20 surface. See Specifications/Qwen3.8.md sections 5 and 8.
 */

module;
#include <concepts>

export module Experimental.Quantization.CodebookPolicies;

import Dnn.TensorDataType;
import Dnn.Quantization.Weight.Policies;

namespace Mila::Dnn::Experimental::Quantization
{
    // -------------------------------------------------------------------------
    // PerGroupCodebook2<kGroupSize>
    //
    // 2-bit codes into a per-tensor 4-entry codebook, one FP16 absmax scale per
    // group of kGroupSize input channels: 2 + 16/32 = 2.5 bits per weight at the
    // default group size. There is no zero point anywhere in this format -- the
    // codebook carries the asymmetry, which the Phase 0 gate measured as both
    // cheaper and better than a per-group zero.
    //
    // Codes are produced offline by the converter (GPTQ-compensated, calibrated);
    // loadParameter() uploads codes, scales and codebook -- it never quantizes.
    // The packed layout is normative in Experimental.Quantization.CodebookPacking.
    // -------------------------------------------------------------------------
    export template<int kGroupSize = 32>
        struct PerGroupCodebook2
    {
        static constexpr bool kIsQuantized = true;
        static constexpr Dnn::TensorDataType kStorageDtype = Dnn::TensorDataType::UINT8;
        static constexpr Dnn::TensorDataType kScaleDtype = Dnn::TensorDataType::FP16;
        static constexpr bool kPerChannel = false;
        static constexpr int kQuantizationGroupSize = kGroupSize;
        static constexpr bool kIsFp4E2M1 = false;
        static constexpr bool kIsCodebook = true;
        static constexpr int kCodeBits = 2;
        static constexpr int kCodebookEntries = 4;
    };

    // -------------------------------------------------------------------------
    // PerGroupCodebook3<kGroupSize>
    //
    // 3-bit codes into a per-tensor 8-entry codebook, one FP16 absmax scale per
    // group of kGroupSize input channels: 3 + 16/64 = 3.25 bits per weight at
    // the default group size. Codes are stored as two byte-aligned planes (the
    // two low bits packed as in PerGroupCodebook2, the high bit one-per-code) so
    // both codebook policies share one kernel family and aligned tile loads.
    // -------------------------------------------------------------------------
    export template<int kGroupSize = 64>
        struct PerGroupCodebook3
    {
        static constexpr bool kIsQuantized = true;
        static constexpr Dnn::TensorDataType kStorageDtype = Dnn::TensorDataType::UINT8;
        static constexpr Dnn::TensorDataType kScaleDtype = Dnn::TensorDataType::FP16;
        static constexpr bool kPerChannel = false;
        static constexpr int kQuantizationGroupSize = kGroupSize;
        static constexpr bool kIsFp4E2M1 = false;
        static constexpr bool kIsCodebook = true;
        static constexpr int kCodeBits = 3;
        static constexpr int kCodebookEntries = 8;
    };

    // Both policies satisfy the same concept the production Linear dispatches on,
    // so a future precision plan can name them without a parallel concept.
    static_assert( Dnn::Quant::Weight::WeightQuantPolicy<PerGroupCodebook2<>> );
    static_assert( Dnn::Quant::Weight::WeightQuantPolicy<PerGroupCodebook3<>> );
    static_assert( Dnn::Quant::Weight::WeightQuantPolicy<PerGroupCodebook2<64>> );
    static_assert( Dnn::Quant::Weight::WeightQuantPolicy<PerGroupCodebook3<128>> );
}
