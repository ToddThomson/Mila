/**
 * @file PerChannelFp8.ixx
 * @brief Per-output-channel FP8 weight quantization.
 */

export module Dnn.Quantization.Weight.PerChannelFp8;

import Dnn.TensorDataType;
import Dnn.Quantization.Weight.WeightQuantPolicy;

namespace Mila::Dnn::Quant::Weight
{
    /**
     * @brief FP8 weights with one FP32 scale per output channel, scale[o] = max(|W[o,:]|) / 448.
     *
     * @tparam TStorage FP8_E4M3 for stored weights; FP8_E5M2 is a gradient format and not an inference target.
     */
    export template<TensorDataType TStorage = TensorDataType::FP8_E4M3>
        struct PerChannelFp8
    {
        static constexpr bool kIsQuantized = true;
        static constexpr TensorDataType kStorageDtype = TStorage;
        static constexpr TensorDataType kScaleDtype = TensorDataType::FP32;
        static constexpr bool kPerChannel = true;
        static constexpr int kStorageBitsPerElement = 8;
    };

    static_assert( WeightQuantPolicy<PerChannelFp8<>> );
}
