/**
 * @file Gemma.FeedForward.Traits.ixx
 * @brief Maps a GemmaFeedForward value to the sublayer component a Gemma 4 block builds for it.
 */

module;

export module Dnn.Components.GemmaFeedForwardTraits;

import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceType;
import Dnn.Components.GemmaFeedForward;
import Dnn.Components.GemmaDenseFeedForward;
import Dnn.Components.GemmaRoutedFeedForward;
import Dnn.Quantization.Weight.Policies;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;

    /**
     * @brief The sublayer type for one GemmaFeedForward value: `type< device, precision, weight policy >`.
     */
    export template<GemmaFeedForward kFeedForward>
    struct GemmaFeedForwardTraits;

    template<>
    struct GemmaFeedForwardTraits<GemmaFeedForward::Dense>
    {
        template<DeviceType TDeviceType, TensorDataType TPrecision, WeightQuantPolicy TWeightQuantization>
        using type = GemmaDenseFeedForward<TDeviceType, TPrecision, TWeightQuantization>;
    };

    template<>
    struct GemmaFeedForwardTraits<GemmaFeedForward::Routed>
    {
        template<DeviceType TDeviceType, TensorDataType TPrecision, WeightQuantPolicy TWeightQuantization>
        using type = GemmaRoutedFeedForward<TDeviceType, TPrecision, TWeightQuantization>;
    };
}
