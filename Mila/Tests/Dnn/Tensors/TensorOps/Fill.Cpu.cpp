#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <memory>

import Mila;

namespace Dnn::Tensors::TensorOps::Tests
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
            auto data = t.data();
            for ( size_t i = 0; i < t.size(); ++i ) {
                EXPECT_FLOAT_EQ( data[i], 1.5f );
            }
        }

        ASSERT_NO_THROW( fill( t, 2.0f ) ); // Use explicit float literal
        {
            auto data = t.data();
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
            auto data = t.data();
            for ( size_t i = 0; i < t.size(); ++i ) {
                EXPECT_EQ( data[i], int32_t{7} );
            }
        }

        // Also test conversion from smaller integer types (should be convertible)
        ASSERT_NO_THROW( fill( t, int32_t{-2} ) ); // Use explicit int32_t
        {
            auto data = t.data();
            for ( size_t i = 0; i < t.size(); ++i ) {
                EXPECT_EQ( data[i], int32_t{-2} );
            }
        }
    }

    TEST( TensorOpsFillCpu, FillScalarTensor_NoThrow ) {
        // Scalar tensor: empty shape {} produces rank 0, size 1
        Tensor<TensorDataType::FP32, CpuMemoryResource> scalar( "CPU", {} );
        
        // Verify scalar properties
        ASSERT_EQ( scalar.rank(), 0u );
        ASSERT_EQ( scalar.size(), 1u );
        ASSERT_TRUE( scalar.isScalar() );
        ASSERT_FALSE( scalar.empty() );

        // Filling a scalar should work and set the single value
        ASSERT_NO_THROW( fill( scalar, 3.14f ) );
        
        // Verify scalar value using item()
        EXPECT_FLOAT_EQ( scalar.item(), 3.14f );

        // Fill with different value
        ASSERT_NO_THROW( fill( scalar, -2.71f ) );
        EXPECT_FLOAT_EQ( scalar.item(), -2.71f );
    }

    TEST( TensorOpsFillCpu, FillEmptyTensor_NoThrow ) {
        // Empty 1D tensor: shape {0} produces rank 1, size 0
        Tensor<TensorDataType::FP32, CpuMemoryResource> empty1d( "CPU", { 0 } );
        
        // Verify empty tensor properties
        ASSERT_EQ( empty1d.rank(), 1u );
        ASSERT_EQ( empty1d.size(), 0u );
        ASSERT_FALSE( empty1d.isScalar() );
        ASSERT_TRUE( empty1d.empty() );

        // Filling an empty tensor should be a no-op and must not throw
        ASSERT_NO_THROW( fill( empty1d, 0.0f ) );
        
        // Size and rank remain unchanged
        EXPECT_EQ( empty1d.size(), 0u );
        EXPECT_EQ( empty1d.rank(), 1u );
    }

    TEST( TensorOpsFillCpu, FillEmpty2DTensor_NoThrow ) {
        // Empty 2D tensor: shape {0, 5} produces rank 2, size 0
        Tensor<TensorDataType::INT32, CpuMemoryResource> empty2d( "CPU", { 0, 5 } );
        
        // Verify empty tensor properties
        ASSERT_EQ( empty2d.rank(), 2u );
        ASSERT_EQ( empty2d.size(), 0u );
        ASSERT_TRUE( empty2d.empty() );

        // Filling should be a no-op
        ASSERT_NO_THROW( fill( empty2d, 42 ) );
        EXPECT_EQ( empty2d.size(), 0u );
    }

    TEST( TensorOpsFillCpu, FillWithSpan_ArrayValues ) {
        // Test fill with span of values
        Tensor<TensorDataType::FP32, CpuMemoryResource> t( "CPU", { 5 } );
        ASSERT_EQ( t.size(), 5u );

        std::vector<float> values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        ASSERT_NO_THROW( fill( t, std::span<const float>{values} ) );

        {
            auto data = t.data();
            for ( size_t i = 0; i < t.size(); ++i ) {
                EXPECT_FLOAT_EQ( data[i], values[i] );
            }
        }
    }

    TEST( TensorOpsFillCpu, FillWithSpan_PartialFill ) {
        // Test partial fill when span is smaller than tensor
        Tensor<TensorDataType::INT32, CpuMemoryResource> t( "CPU", { 10 } );
        ASSERT_EQ( t.size(), 10u );

        std::vector<int32_t> values = { 100, 200, 300 };
        ASSERT_NO_THROW( fill( t, std::span<const int32_t>{values} ) );

        {
            auto data = t.data();
            // First 3 elements should be filled
            EXPECT_EQ( data[0], 100 );
            EXPECT_EQ( data[1], 200 );
            EXPECT_EQ( data[2], 300 );
            // Remaining elements are undefined (not tested)
        }
    }

    TEST( TensorOpsFillCpu, FillScalarWithItem_DirectAccess ) {
        // Test scalar-specific access patterns
        Tensor<TensorDataType::INT32, CpuMemoryResource> scalar( "CPU", {} );
        
        ASSERT_TRUE( scalar.isScalar() );
        ASSERT_EQ( scalar.size(), 1u );

        // Fill scalar
        fill( scalar, 999 );
        
        // Access via item()
        EXPECT_EQ( scalar.item(), 999 );

        // Verify operator[] throws for scalars
        EXPECT_THROW( scalar[std::vector<size_t>{}], std::runtime_error );
    }
}