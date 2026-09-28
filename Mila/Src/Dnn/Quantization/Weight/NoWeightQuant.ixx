/**
 * @file NoWeightQuant.ixx
 * @brief The identity weight policy: weights stored at compute precision.
 */

export module Dnn.Quantization.Weight.NoWeightQuant;

import Dnn.TensorDataType;
import Dnn.Quantization.Weight.WeightQuantPolicy;

namespace Mila::Dnn::Quant::Weight
{
    /**
     * @brief No quantization; the default for every Linear.
     *
     * The dtype fields are sentinels that no consumer reads: every use is guarded by
     * if constexpr ( kIsQuantized ).
     */
    export struct NoWeightQuant
    {
        static constexpr bool kIsQuantized = false;
        static constexpr TensorDataType kStorageDtype = TensorDataType::FP32;
        static constexpr TensorDataType kScaleDtype = TensorDataType::FP32;
        static constexpr bool kPerChannel = false;
        static constexpr int kStorageBitsPerElement = 32;
    };

    static_assert( WeightQuantPolicy<NoWeightQuant> );
}
