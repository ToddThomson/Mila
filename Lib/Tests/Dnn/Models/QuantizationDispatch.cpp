/**
 * @file QuantizationDispatch.cpp
 * @brief The family-blind dispatcher builds a weight format exactly where its Linear operation exists and an FP8 KV cache
 *        exactly where its attention exists, and refuses every designed plan, so a family without one never builds a body
 *        that runs unquantized and reports success.
 */

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

import Dnn.KvCacheCompression;
import Dnn.WeightQuantization;
import Dnn.TensorDataType;
import Dnn.Quantization.Weight.Policies;
import Dnn.Quantization.KvCache.PerTokenKvFp8;
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
        // Q4_0 is BF16 only on CUDA, and absent from a build without CUDA.
        static_assert( !OperationSupported<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::FP32, PerGroupInt4<32>>,
            "This test needs a format a device does not carry at a precision; pick another" );

        try
        {
            (void)dispatched<DeviceType::Cuda, TensorDataType::FP32>( WeightQuantization::Q4_0 );
            FAIL() << "Q4_0 weights were dispatched at FP32 on CUDA, which has no such operation";
        }
        catch ( const std::runtime_error& error )
        {
            const std::string message = error.what();

            EXPECT_NE( message.find( "Q4_0" ), std::string::npos ) << message;
            EXPECT_NE( message.find( "CUDA" ), std::string::npos ) << message;
            EXPECT_NE( message.find( "FP32" ), std::string::npos ) << message;
        }
    }

    namespace
    {
        template<DeviceType TDeviceType, TensorDataType TPrecision>
        int dispatchedCache( KvCacheCompression compression )
        {
            return dispatchKvCacheCompression<TDeviceType, TPrecision, int>( compression, "test",
                []<typename TPolicy>() -> int { return std::is_same_v<TPolicy, Quant::KvCache::PerTokenKvFp8<>> ? 8 : 16; } );
        }

        /// An FP8 cache is built where attention has one on the device at the precision, and refused where it does not.
        template<DeviceType TDeviceType, TensorDataType TPrecision>
        void expectFp8CacheBuiltExactlyWhereSupported()
        {
            EXPECT_EQ( ( dispatchedCache<TDeviceType, TPrecision>( KvCacheCompression::None ) ), 16 );

            if constexpr ( OperationSupported<OperationType::GroupedQueryAttentionOp, TDeviceType, TPrecision, Quant::KvCache::PerTokenKvFp8<>> )
            {
                EXPECT_EQ( ( dispatchedCache<TDeviceType, TPrecision>( KvCacheCompression::FP8 ) ), 8 );
            }
            else
            {
                EXPECT_THROW( ( dispatchedCache<TDeviceType, TPrecision>( KvCacheCompression::FP8 ) ), std::runtime_error );
            }
        }
    }

    TEST( QuantizationDispatch, BuildsAnFp8CacheExactlyWhereItsAttentionExists )
    {
        expectFp8CacheBuiltExactlyWhereSupported<DeviceType::Cuda, TensorDataType::BF16>();
        expectFp8CacheBuiltExactlyWhereSupported<DeviceType::Cuda, TensorDataType::FP32>();
        expectFp8CacheBuiltExactlyWhereSupported<DeviceType::Metal, TensorDataType::FP16>();
        expectFp8CacheBuiltExactlyWhereSupported<DeviceType::Metal, TensorDataType::BF16>();
    }

    TEST( QuantizationDispatch, TheCacheRefusalNamesTheDeviceAndThePrecision )
    {
        static_assert( !OperationSupported<OperationType::GroupedQueryAttentionOp, DeviceType::Cuda, TensorDataType::FP32,
            Quant::KvCache::PerTokenKvFp8<>> );

        try
        {
            (void)dispatchedCache<DeviceType::Cuda, TensorDataType::FP32>( KvCacheCompression::FP8 );
            FAIL() << "An FP8 cache was dispatched at FP32 on CUDA, which has no such attention";
        }
        catch ( const std::runtime_error& error )
        {
            const std::string message = error.what();

            EXPECT_NE( message.find( "FP8" ), std::string::npos ) << message;
            EXPECT_NE( message.find( "CUDA" ), std::string::npos ) << message;
            EXPECT_NE( message.find( "FP32" ), std::string::npos ) << message;
        }
    }
}
