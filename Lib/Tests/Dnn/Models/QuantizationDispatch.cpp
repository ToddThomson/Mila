/**
 * @file QuantizationDispatch.cpp
 * @brief The family-blind dispatcher builds a weight format exactly where its Linear operation exists, and refuses
 *        every designed plan, so a family without one never builds a body that runs unquantized and reports success.
 */

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <string_view>

import Dnn.WeightQuantization;
import Dnn.TensorDataType;
import Dnn.Quantization.Weight.Policies;
import Dnn.Models.QuantizationDispatch;
import Compute.DeviceType;
import Compute.OperationTraits;
import Compute.OperationType;

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;

    namespace
    {
        template<DeviceType TDeviceType, TensorDataType TPrecision>
        int dispatched( WeightQuantization quantization )
        {
            return dispatchWeightQuantization<TDeviceType, TPrecision, int>( quantization, "test",
                []<typename TPolicy>() -> int { return 1; } );
        }

        /// Built when the format's Linear operation exists on the device at the precision, refused when it does not.
        template<DeviceType TDeviceType, TensorDataType TPrecision, typename TPolicy>
        void expectBuiltExactlyWhereSupported( WeightQuantization quantization, std::string_view format )
        {
            if constexpr ( OperationSupported<OperationType::LinearOp, TDeviceType, TPrecision, TPolicy> )
            {
                EXPECT_EQ( ( dispatched<TDeviceType, TPrecision>( quantization ) ), 1 ) << format;
            }
            else
            {
                EXPECT_THROW( ( dispatched<TDeviceType, TPrecision>( quantization ) ), std::runtime_error ) << format;
            }
        }

        template<DeviceType TDeviceType, TensorDataType TPrecision>
        void expectEveryUniformFormatBuiltExactlyWhereSupported()
        {
            expectBuiltExactlyWhereSupported<TDeviceType, TPrecision, PerGroupFp4<128>>( WeightQuantization::FP4, "FP4" );
            expectBuiltExactlyWhereSupported<TDeviceType, TPrecision, PerChannelFp8<>>( WeightQuantization::FP8, "FP8" );
            expectBuiltExactlyWhereSupported<TDeviceType, TPrecision, PerGroupInt4<32>>( WeightQuantization::Q4_0, "Q4_0" );
        }
    }

    TEST( QuantizationDispatch, RefusesEveryDesignedPlan )
    {
        EXPECT_THROW( ( dispatched<DeviceType::Cuda, TensorDataType::BF16>( WeightQuantization::Plan ) ), std::runtime_error );
        EXPECT_THROW( ( dispatched<DeviceType::Cuda, TensorDataType::BF16>( WeightQuantization::Fp4Codebook3 ) ), std::runtime_error );
        EXPECT_THROW( ( dispatched<DeviceType::Metal, TensorDataType::FP16>( WeightQuantization::Plan ) ), std::runtime_error );
    }

    TEST( QuantizationDispatch, BuildsUnquantizedWeightsOnEveryDevice )
    {
        EXPECT_EQ( ( dispatched<DeviceType::Cuda, TensorDataType::BF16>( WeightQuantization::None ) ), 1 );
        EXPECT_EQ( ( dispatched<DeviceType::Metal, TensorDataType::FP16>( WeightQuantization::None ) ), 1 );
        EXPECT_EQ( ( dispatched<DeviceType::Metal, TensorDataType::BF16>( WeightQuantization::None ) ), 1 );
    }

    // Holds in every build: which formats a build has is OperationTraits' answer, and the dispatch must give the same one.
    TEST( QuantizationDispatch, BuildsEachFormatExactlyWhereItsLinearOperationExists )
    {
        expectEveryUniformFormatBuiltExactlyWhereSupported<DeviceType::Cuda, TensorDataType::BF16>();
        expectEveryUniformFormatBuiltExactlyWhereSupported<DeviceType::Cuda, TensorDataType::FP32>();
        expectEveryUniformFormatBuiltExactlyWhereSupported<DeviceType::Metal, TensorDataType::FP16>();
        expectEveryUniformFormatBuiltExactlyWhereSupported<DeviceType::Metal, TensorDataType::BF16>();
    }

    TEST( QuantizationDispatch, TheRefusalNamesTheDeviceAndThePrecision )
    {
        try
        {
            (void)dispatched<DeviceType::Metal, TensorDataType::FP16>( WeightQuantization::FP8 );
            FAIL() << "FP8 weights were dispatched at FP16 on Metal, which has no such operation";
        }
        catch ( const std::runtime_error& error )
        {
            const std::string message = error.what();

            EXPECT_NE( message.find( "FP8" ), std::string::npos ) << message;
            EXPECT_NE( message.find( "Metal" ), std::string::npos ) << message;
            EXPECT_NE( message.find( "FP16" ), std::string::npos ) << message;
        }
    }
}
