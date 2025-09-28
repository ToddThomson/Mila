#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <memory>

import Mila;

namespace Dnn::Tensors::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    TEST( TensorOpsFillCpu, FillFloatTensor_NoThrowAndSizePreserved ) {
        // Create a CPU host tensor of FP32 with shape 2x3
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( "CPU", { 2, 3 } );
        ASSERT_EQ( t.size(), 6u );

        // Calls must compile and not throw; forwarding + conversion should work
        ASSERT_NO_THROW( fill( t, 1.5f ) );

        {
            const float* data = static_cast<const float*>( t.rawData() );
            for ( size_t i = 0; i < t.size(); ++i ) {
                EXPECT_FLOAT_EQ( data[i], 1.5f );
            }
        }

        ASSERT_NO_THROW( fill( t, 2 ) ); // int -> float conversion
        {
            const float* data = static_cast<const float*>( t.rawData() );
            for ( size_t i = 0; i < t.size(); ++i ) {
                EXPECT_FLOAT_EQ( data[i], 2.0f );
            }
        }

        // Shape and size must remain unchanged by fill
        EXPECT_EQ( t.shape().size(), 2u );
        EXPECT_EQ( t.shape()[0], 2u );
        EXPECT_EQ( t.shape()[1], 3u );
        EXPECT_EQ( t.size(), 6u );
    }

    TEST( TensorOpsFillCpu, FillIntTensor_NoThrow_WithDifferentValueTypes ) {
        // Create an INT32 CPU tensor
        Tensor<TensorDataType::INT32, CpuMemoryResource> t( "CPU", { 4 } );
        ASSERT_EQ( t.size(), 4u );

        // Fill with matching and convertible types; should not throw
        ASSERT_NO_THROW( fill( t, int32_t{7} ) );

        // Verify integer values
        {
            const int32_t* data = static_cast<const int32_t*>( t.rawData() );
            for ( size_t i = 0; i < t.size(); ++i ) {
                EXPECT_EQ( data[i], int32_t{7} );
            }
        }

        // Also test conversion from smaller integer types (should be convertible)
        ASSERT_NO_THROW( fill( t, static_cast<int16_t>(-2) ) );
        {
            const int32_t* data = static_cast<const int32_t*>( t.rawData() );
            for ( size_t i = 0; i < t.size(); ++i ) {
                EXPECT_EQ( data[i], int32_t{-2} );
            }
        }
    }

    TEST( TensorOpsFillCpu, FillEmptyTensor_NoThrow ) {
        // Empty (scalar / zero-sized) tensor: constructor with empty shape produces size==0
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( "CPU", std::vector<size_t>{} );
        ASSERT_EQ( t.size(), 0u );

        // Filling an empty tensor should be a no-op and must not throw
        ASSERT_NO_THROW( fill( t, 0.0f ) );
        // rawData may be nullptr or valid; size==0 is the main contract
        EXPECT_EQ( t.size(), 0u );
    }
}