/**
 * @file QuantizationDispatch.cpp
 * @brief The family-blind dispatcher refuses every designed plan, so a family without one never builds a body
 *        that runs unquantized and reports success.
 */

#include <gtest/gtest.h>

#include <stdexcept>
#include <string_view>

import Dnn.WeightQuantization;
import Dnn.TensorDataType;
import Dnn.Models.QuantizationDispatch;

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;

    namespace
    {
        int dispatched( WeightQuantization quantization )
        {
            return dispatchWeightQuantization<TensorDataType::BF16, int>( quantization, "test",
                []<typename TPolicy>() -> int { return 1; } );
        }
    }

    TEST( QuantizationDispatch, RefusesEveryDesignedPlan )
    {
        EXPECT_THROW( dispatched( WeightQuantization::Plan ), std::runtime_error );
        EXPECT_THROW( dispatched( WeightQuantization::Fp4Codebook3 ), std::runtime_error );
    }

    TEST( QuantizationDispatch, BuildsEveryUniformFormat )
    {
        EXPECT_EQ( dispatched( WeightQuantization::None ), 1 );
        EXPECT_EQ( dispatched( WeightQuantization::FP4 ), 1 );
        EXPECT_EQ( dispatched( WeightQuantization::FP8 ), 1 );
        EXPECT_EQ( dispatched( WeightQuantization::Q4_0 ), 1 );
    }
}
