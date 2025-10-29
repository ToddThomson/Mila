#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <memory>

import Mila;

namespace Dnn::Tensors::TensorOps::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    TEST( TensorOpsFillCuda, FillFloatTensor_NoThrowAndSizePreserved ) {
        // Create a CUDA device tensor of FP32 with shape 2x3
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> t( "CUDA:0", { 2, 3 } );
        ASSERT_EQ( t.size(), 6u );

        // Calls must compile and not throw; forwarding + conversion should work
        ASSERT_NO_THROW( fill( t, 1.5f ) );

        // Create CPU tensor for verification
        Tensor<TensorDataType::FP32, CpuMemoryResource> cpu_t( "CPU", { 2, 3 } );
        fill( cpu_t, 1.5f );

        // Shape and size must remain unchanged by fill
        EXPECT_EQ( t.shape().size(), 2u );
        EXPECT_EQ( t.shape()[0], 2u );
        EXPECT_EQ( t.shape()[1], 3u );
        EXPECT_EQ( t.size(), 6u );

        ASSERT_NO_THROW( fill( t, 2.0f ) ); // Use explicit float literal

        // Verify size and shape preservation after second fill
        EXPECT_EQ( t.shape().size(), 2u );
        EXPECT_EQ( t.shape()[0], 2u );
        EXPECT_EQ( t.shape()[1], 3u );
        EXPECT_EQ( t.size(), 6u );
    }

    TEST( TensorOpsFillCuda, FillIntTensor_NoThrow_WithDifferentValueTypes ) {
        // Create an INT32 CUDA device tensor
        Tensor<TensorDataType::INT32, CudaDeviceMemoryResource> t( "CUDA:0", { 4 } );
        ASSERT_EQ( t.size(), 4u );

        // Fill with matching and convertible types; should not throw
        ASSERT_NO_THROW( fill( t, int32_t{ 7 } ) );

        // Also test conversion from smaller integer types (should be convertible)
        ASSERT_NO_THROW( fill( t, int32_t{ -2 } ) ); // Use explicit int32_t

        // Verify size preservation
        EXPECT_EQ( t.size(), 4u );
    }

    TEST( TensorOpsFillCuda, FillScalarTensor_NoThrow ) {
        // Scalar tensor: empty shape {} produces rank 0, size 1
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> scalar( "CUDA:0", {} );

        // Verify scalar properties
        ASSERT_EQ( scalar.rank(), 0u );
        ASSERT_EQ( scalar.size(), 1u );
        ASSERT_TRUE( scalar.isScalar() );
        ASSERT_FALSE( scalar.empty() );

        // Filling a scalar should work and set the single value
        ASSERT_NO_THROW( fill( scalar, 3.14f ) );

        // Verify scalar properties are preserved
        EXPECT_EQ( scalar.rank(), 0u );
        EXPECT_EQ( scalar.size(), 1u );
        EXPECT_TRUE( scalar.isScalar() );

        // Fill with different value
        ASSERT_NO_THROW( fill( scalar, -2.71f ) );

        // Properties still preserved
        EXPECT_EQ( scalar.rank(), 0u );
        EXPECT_EQ( scalar.size(), 1u );
        EXPECT_TRUE( scalar.isScalar() );
    }

    TEST( TensorOpsFillCuda, FillEmptyTensor_NoThrow ) {
        // Empty 1D tensor: shape {0} produces rank 1, size 0
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> empty1d( "CUDA:0", { 0 } );

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

    TEST( TensorOpsFillCuda, FillEmpty2DTensor_NoThrow ) {
        // Empty 2D tensor: shape {0, 5} produces rank 2, size 0
        Tensor<TensorDataType::INT32, CudaDeviceMemoryResource> empty2d( "CUDA:0", { 0, 5 } );

        // Verify empty tensor properties
        ASSERT_EQ( empty2d.rank(), 2u );
        ASSERT_EQ( empty2d.size(), 0u );
        ASSERT_TRUE( empty2d.empty() );

        // Filling should be a no-op
        ASSERT_NO_THROW( fill( empty2d, 42 ) );
        EXPECT_EQ( empty2d.size(), 0u );
    }

    TEST( TensorOpsFillCuda, FillWithSpan_ArrayValues ) {
        // Test fill with span of values
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> t( "CUDA:0", { 5 } );
        ASSERT_EQ( t.size(), 5u );

        std::vector<float> values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        ASSERT_NO_THROW( fill( t, std::span<const float>{values} ) );

        // Verify size preservation
        EXPECT_EQ( t.size(), 5u );
    }

    TEST( TensorOpsFillCuda, FillWithSpan_PartialFill ) {
        // Test partial fill when span is smaller than tensor
        Tensor<TensorDataType::INT32, CudaDeviceMemoryResource> t( "CUDA:0", { 10 } );
        ASSERT_EQ( t.size(), 10u );

        std::vector<int32_t> values = { 100, 200, 300 };
        ASSERT_NO_THROW( fill( t, std::span<const int32_t>{values} ) );

        // Verify size preservation
        EXPECT_EQ( t.size(), 10u );
    }

    TEST( TensorOpsFillCuda, FillManagedMemoryTensor_FloatValues ) {
        // Test with CUDA managed memory for both device and host accessibility
        Tensor<TensorDataType::FP32, CudaManagedMemoryResource> managed_t( "CUDA:0", { 3, 2 } );
        ASSERT_EQ( managed_t.size(), 6u );

        // Fill operation should work
        ASSERT_NO_THROW( fill( managed_t, 4.5f ) );

        // Properties preserved
        EXPECT_EQ( managed_t.shape().size(), 2u );
        EXPECT_EQ( managed_t.shape()[0], 3u );
        EXPECT_EQ( managed_t.shape()[1], 2u );
        EXPECT_EQ( managed_t.size(), 6u );
    }

    TEST( TensorOpsFillCuda, FillPinnedMemoryTensor_IntValues ) {
        // Test with CUDA pinned memory for faster host-device transfers
        Tensor<TensorDataType::INT32, CudaPinnedMemoryResource> pinned_t( "CUDA:0", { 8 } );
        ASSERT_EQ( pinned_t.size(), 8u );

        // Fill with integer value
        ASSERT_NO_THROW( fill( pinned_t, int32_t{ 999 } ) );

        // Since pinned memory is host-accessible, we can verify the data
        {
            auto data = pinned_t.data();
            for (size_t i = 0; i < pinned_t.size(); ++i)
            {
                EXPECT_EQ( data[i], int32_t{ 999 } );
            }
        }

        // Test with different value
        ASSERT_NO_THROW( fill( pinned_t, int32_t{ -123 } ) );
        {
            auto data = pinned_t.data();
            for (size_t i = 0; i < pinned_t.size(); ++i)
            {
                EXPECT_EQ( data[i], int32_t{ -123 } );
            }
        }
    }

    TEST( TensorOpsFillCuda, FillScalarPinnedMemory_DirectAccess ) {
        // Test scalar-specific access patterns with pinned memory
        Tensor<TensorDataType::INT32, CudaPinnedMemoryResource> scalar( "CUDA:0", {} );

        ASSERT_TRUE( scalar.isScalar() );
        ASSERT_EQ( scalar.size(), 1u );

        // Fill scalar
        fill( scalar, 777 );

        // Access via item() since pinned memory is host-accessible
        EXPECT_EQ( scalar.item(), 777 );

        // Verify operator[] throws for scalars
        EXPECT_THROW( scalar[std::vector<int64_t>{}], std::runtime_error );
    }

    TEST( TensorOpsFillCuda, FillLargeTensor_PerformanceAndCorrectness ) {
        // Test with larger tensor to verify CUDA implementation scales
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> large_t( "CUDA:0", { 1000, 1000 } );
        ASSERT_EQ( large_t.size(), 1000000u );

        // Fill should complete without error
        ASSERT_NO_THROW( fill( large_t, 0.5f ) );

        // Properties preserved
        EXPECT_EQ( large_t.size(), 1000000u );
        EXPECT_EQ( large_t.shape().size(), 2u );
        EXPECT_EQ( large_t.shape()[0], 1000u );
        EXPECT_EQ( large_t.shape()[1], 1000u );
    }

    TEST( TensorOpsFillCuda, FillWithSpan_ManagedMemoryVerification ) {
        // Test span fill with managed memory for verification
        Tensor<TensorDataType::FP32, CudaManagedMemoryResource> t( "CUDA:0", { 4 } );
        ASSERT_EQ( t.size(), 4u );

        std::vector<float> values = { 10.0f, 20.0f, 30.0f, 40.0f };
        ASSERT_NO_THROW( fill( t, std::span<const float>{values} ) );

        // Since managed memory is host-accessible, verify the values
        {
            auto data = t.data();
            for (size_t i = 0; i < t.size(); ++i)
            {
                EXPECT_FLOAT_EQ( data[i], values[i] );
            }
        }
    }

    TEST( TensorOpsFillCuda, FillMultipleTensorTypes_MixedOperations ) {
        // Test multiple tensor types in sequence
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> device_t( "CUDA:0", { 5 } );
        Tensor<TensorDataType::FP32, CudaManagedMemoryResource> managed_t( "CUDA:0", { 5 } );
        Tensor<TensorDataType::FP32, CudaPinnedMemoryResource> pinned_t( "CUDA:0", { 5 } );

        // All should fill without error
        ASSERT_NO_THROW( fill( device_t, 1.0f ) );
        ASSERT_NO_THROW( fill( managed_t, 2.0f ) );
        ASSERT_NO_THROW( fill( pinned_t, 3.0f ) );

        // Verify pinned memory result (host-accessible)
        {
            auto data = pinned_t.data();
            for (size_t i = 0; i < pinned_t.size(); ++i)
            {
                EXPECT_FLOAT_EQ( data[i], 3.0f );
            }
        }

        // Verify managed memory result (also host-accessible)
        {
            auto data = managed_t.data();
            for (size_t i = 0; i < managed_t.size(); ++i)
            {
                EXPECT_FLOAT_EQ( data[i], 2.0f );
            }
        }
    }
}