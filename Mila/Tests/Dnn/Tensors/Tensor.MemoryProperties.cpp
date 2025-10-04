#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

import Mila;

namespace Tensors::Tests
{
    using namespace Mila::Dnn;

    class TensorMemoryPropertiesTest : public testing::Test {
    protected:
        TensorMemoryPropertiesTest() {}
    };

    // ====================================================================
    // Host-Compatible Types with All Memory Resources
    // ====================================================================

    TEST( TensorMemoryPropertiesTest, HostCompatibleTypes_AllMemoryResources ) {
        std::vector<size_t> shape = { 2, 3 };

        // float (FP32)
        {
            Tensor<float, Compute::HostMemoryResource> host_tensor( shape );
            Tensor<float, Compute::CudaDeviceMemoryResource> cuda_tensor( shape );
            Tensor<float, Compute::CudaPinnedMemoryResource> pinned_tensor( shape );
            Tensor<float, Compute::CudaManagedMemoryResource> managed_tensor( shape );

            EXPECT_TRUE( host_tensor.is_host_accessible() );
            EXPECT_FALSE( host_tensor.is_device_accessible() );
            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );
            EXPECT_TRUE( pinned_tensor.is_host_accessible() );
            EXPECT_TRUE( pinned_tensor.is_device_accessible() );
            EXPECT_TRUE( managed_tensor.is_host_accessible() );
            EXPECT_TRUE( managed_tensor.is_device_accessible() );

            EXPECT_EQ( host_tensor.getDataType(), TensorDataType::FP32 );
            EXPECT_EQ( cuda_tensor.getDataType(), TensorDataType::FP32 );
            EXPECT_EQ( pinned_tensor.getDataType(), TensorDataType::FP32 );
            EXPECT_EQ( managed_tensor.getDataType(), TensorDataType::FP32 );

            EXPECT_EQ( host_tensor.getDataTypeName(), "FP32" );
            EXPECT_EQ( cuda_tensor.getDataTypeName(), "FP32" );
            EXPECT_EQ( pinned_tensor.getDataTypeName(), "FP32" );
            EXPECT_EQ( managed_tensor.getDataTypeName(), "FP32" );
        }

        // int16_t (INT16)
        {
            Tensor<int16_t, Compute::HostMemoryResource> host_tensor( shape );
            Tensor<int16_t, Compute::CudaDeviceMemoryResource> cuda_tensor( shape );
            Tensor<int16_t, Compute::CudaPinnedMemoryResource> pinned_tensor( shape );
            Tensor<int16_t, Compute::CudaManagedMemoryResource> managed_tensor( shape );

            EXPECT_TRUE( host_tensor.is_host_accessible() );
            EXPECT_FALSE( host_tensor.is_device_accessible() );
            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );
            EXPECT_TRUE( pinned_tensor.is_host_accessible() );
            EXPECT_TRUE( pinned_tensor.is_device_accessible() );
            EXPECT_TRUE( managed_tensor.is_host_accessible() );
            EXPECT_TRUE( managed_tensor.is_device_accessible() );

            EXPECT_EQ( host_tensor.getDataType(), TensorDataType::INT16 );
            EXPECT_EQ( cuda_tensor.getDataType(), TensorDataType::INT16 );
            EXPECT_EQ( pinned_tensor.getDataType(), TensorDataType::INT16 );
            EXPECT_EQ( managed_tensor.getDataType(), TensorDataType::INT16 );

            EXPECT_EQ( host_tensor.getDataTypeName(), "INT16" );
        }

        // int (INT32)
        {
            Tensor<int, Compute::HostMemoryResource> host_tensor( shape );
            Tensor<int, Compute::CudaDeviceMemoryResource> cuda_tensor( shape );
            Tensor<int, Compute::CudaPinnedMemoryResource> pinned_tensor( shape );
            Tensor<int, Compute::CudaManagedMemoryResource> managed_tensor( shape );

            EXPECT_TRUE( host_tensor.is_host_accessible() );
            EXPECT_FALSE( host_tensor.is_device_accessible() );
            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );
            EXPECT_TRUE( pinned_tensor.is_host_accessible() );
            EXPECT_TRUE( pinned_tensor.is_device_accessible() );
            EXPECT_TRUE( managed_tensor.is_host_accessible() );
            EXPECT_TRUE( managed_tensor.is_device_accessible() );

            EXPECT_EQ( host_tensor.getDataType(), TensorDataType::INT32 );
            EXPECT_EQ( cuda_tensor.getDataType(), TensorDataType::INT32 );
            EXPECT_EQ( pinned_tensor.getDataType(), TensorDataType::INT32 );
            EXPECT_EQ( managed_tensor.getDataType(), TensorDataType::INT32 );

            EXPECT_EQ( host_tensor.getDataTypeName(), "INT32" );
        }

        // uint16_t (UINT16)
        {
            Tensor<uint16_t, Compute::HostMemoryResource> host_tensor( shape );
            Tensor<uint16_t, Compute::CudaDeviceMemoryResource> cuda_tensor( shape );
            Tensor<uint16_t, Compute::CudaPinnedMemoryResource> pinned_tensor( shape );
            Tensor<uint16_t, Compute::CudaManagedMemoryResource> managed_tensor( shape );

            EXPECT_TRUE( host_tensor.is_host_accessible() );
            EXPECT_FALSE( host_tensor.is_device_accessible() );
            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );
            EXPECT_TRUE( pinned_tensor.is_host_accessible() );
            EXPECT_TRUE( pinned_tensor.is_device_accessible() );
            EXPECT_TRUE( managed_tensor.is_host_accessible() );
            EXPECT_TRUE( managed_tensor.is_device_accessible() );

            EXPECT_EQ( host_tensor.getDataType(), TensorDataType::UINT16 );
            EXPECT_EQ( cuda_tensor.getDataType(), TensorDataType::UINT16 );
            EXPECT_EQ( pinned_tensor.getDataType(), TensorDataType::UINT16 );
            EXPECT_EQ( managed_tensor.getDataType(), TensorDataType::UINT16 );

            EXPECT_EQ( host_tensor.getDataTypeName(), "UINT16" );
        }

        // uint32_t (UINT32)
        {
            Tensor<uint32_t, Compute::HostMemoryResource> host_tensor( shape );
            Tensor<uint32_t, Compute::CudaDeviceMemoryResource> cuda_tensor( shape );
            Tensor<uint32_t, Compute::CudaPinnedMemoryResource> pinned_tensor( shape );
            Tensor<uint32_t, Compute::CudaManagedMemoryResource> managed_tensor( shape );

            EXPECT_TRUE( host_tensor.is_host_accessible() );
            EXPECT_FALSE( host_tensor.is_device_accessible() );
            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );
            EXPECT_TRUE( pinned_tensor.is_host_accessible() );
            EXPECT_TRUE( pinned_tensor.is_device_accessible() );
            EXPECT_TRUE( managed_tensor.is_host_accessible() );
            EXPECT_TRUE( managed_tensor.is_device_accessible() );

            EXPECT_EQ( host_tensor.getDataType(), TensorDataType::UINT32 );
            EXPECT_EQ( cuda_tensor.getDataType(), TensorDataType::UINT32 );
            EXPECT_EQ( pinned_tensor.getDataType(), TensorDataType::UINT32 );
            EXPECT_EQ( managed_tensor.getDataType(), TensorDataType::UINT32 );

            EXPECT_EQ( host_tensor.getDataTypeName(), "UINT32" );
        }
    }

    // ====================================================================
    // Static Class Method Tests for All Valid Types
    // ====================================================================

    TEST( TensorMemoryPropertiesTest, StaticMethods_HostCompatibleTypes ) {
        // Host-compatible types can be used with all memory resources

        // float
        EXPECT_TRUE( (Tensor<float, Compute::HostMemoryResource>::is_host_accessible()) );
        EXPECT_FALSE( (Tensor<float, Compute::HostMemoryResource>::is_device_accessible()) );
        EXPECT_FALSE( (Tensor<float, Compute::CudaDeviceMemoryResource>::is_host_accessible()) );
        EXPECT_TRUE( (Tensor<float, Compute::CudaDeviceMemoryResource>::is_device_accessible()) );
        EXPECT_TRUE( (Tensor<float, Compute::CudaPinnedMemoryResource>::is_host_accessible()) );
        EXPECT_TRUE( (Tensor<float, Compute::CudaPinnedMemoryResource>::is_device_accessible()) );
        EXPECT_TRUE( (Tensor<float, Compute::CudaManagedMemoryResource>::is_host_accessible()) );
        EXPECT_TRUE( (Tensor<float, Compute::CudaManagedMemoryResource>::is_device_accessible()) );

        // int16_t
        EXPECT_TRUE( (Tensor<int16_t, Compute::HostMemoryResource>::is_host_accessible()) );
        EXPECT_FALSE( (Tensor<int16_t, Compute::HostMemoryResource>::is_device_accessible()) );
        EXPECT_FALSE( (Tensor<int16_t, Compute::CudaDeviceMemoryResource>::is_host_accessible()) );
        EXPECT_TRUE( (Tensor<int16_t, Compute::CudaDeviceMemoryResource>::is_device_accessible()) );

        // int
        EXPECT_TRUE( (Tensor<int, Compute::HostMemoryResource>::is_host_accessible()) );
        EXPECT_FALSE( (Tensor<int, Compute::HostMemoryResource>::is_device_accessible()) );
        EXPECT_FALSE( (Tensor<int, Compute::CudaDeviceMemoryResource>::is_host_accessible()) );
        EXPECT_TRUE( (Tensor<int, Compute::CudaDeviceMemoryResource>::is_device_accessible()) );

        // uint16_t
        EXPECT_TRUE( (Tensor<uint16_t, Compute::HostMemoryResource>::is_host_accessible()) );
        EXPECT_FALSE( (Tensor<uint16_t, Compute::HostMemoryResource>::is_device_accessible()) );
        EXPECT_FALSE( (Tensor<uint16_t, Compute::CudaDeviceMemoryResource>::is_host_accessible()) );
        EXPECT_TRUE( (Tensor<uint16_t, Compute::CudaDeviceMemoryResource>::is_device_accessible()) );

        // uint32_t
        EXPECT_TRUE( (Tensor<uint32_t, Compute::HostMemoryResource>::is_host_accessible()) );
        EXPECT_FALSE( (Tensor<uint32_t, Compute::HostMemoryResource>::is_device_accessible()) );
        EXPECT_FALSE( (Tensor<uint32_t, Compute::CudaDeviceMemoryResource>::is_host_accessible()) );
        EXPECT_TRUE( (Tensor<uint32_t, Compute::CudaDeviceMemoryResource>::is_device_accessible()) );
    }

    // ====================================================================
    // Compile-time Property Verification
    // ====================================================================

    TEST( TensorMemoryPropertiesTest, CompileTimeProperties_AllTypes ) {
        // Host-compatible types with all memory resources
        static_assert(Tensor<float, Compute::HostMemoryResource>::is_host_accessible());
        static_assert(!Tensor<float, Compute::HostMemoryResource>::is_device_accessible());
        static_assert(!Tensor<float, Compute::CudaDeviceMemoryResource>::is_host_accessible());
        static_assert(Tensor<float, Compute::CudaDeviceMemoryResource>::is_device_accessible());
        static_assert(Tensor<float, Compute::CudaPinnedMemoryResource>::is_host_accessible());
        static_assert(Tensor<float, Compute::CudaPinnedMemoryResource>::is_device_accessible());
        static_assert(Tensor<float, Compute::CudaManagedMemoryResource>::is_host_accessible());
        static_assert(Tensor<float, Compute::CudaManagedMemoryResource>::is_device_accessible());

        static_assert(Tensor<int16_t, Compute::HostMemoryResource>::is_host_accessible());
        static_assert(!Tensor<int16_t, Compute::HostMemoryResource>::is_device_accessible());
        static_assert(!Tensor<int16_t, Compute::CudaDeviceMemoryResource>::is_host_accessible());
        static_assert(Tensor<int16_t, Compute::CudaDeviceMemoryResource>::is_device_accessible());

        static_assert(Tensor<int, Compute::HostMemoryResource>::is_host_accessible());
        static_assert(!Tensor<int, Compute::HostMemoryResource>::is_device_accessible());
        static_assert(!Tensor<int, Compute::CudaDeviceMemoryResource>::is_host_accessible());
        static_assert(Tensor<int, Compute::CudaDeviceMemoryResource>::is_device_accessible());

        static_assert(Tensor<uint16_t, Compute::HostMemoryResource>::is_host_accessible());
        static_assert(!Tensor<uint16_t, Compute::HostMemoryResource>::is_device_accessible());
        static_assert(!Tensor<uint16_t, Compute::CudaDeviceMemoryResource>::is_host_accessible());
        static_assert(Tensor<uint16_t, Compute::CudaDeviceMemoryResource>::is_device_accessible());

        static_assert(Tensor<uint32_t, Compute::HostMemoryResource>::is_host_accessible());
        static_assert(!Tensor<uint32_t, Compute::HostMemoryResource>::is_device_accessible());
        static_assert(!Tensor<uint32_t, Compute::CudaDeviceMemoryResource>::is_host_accessible());
        static_assert(Tensor<uint32_t, Compute::CudaDeviceMemoryResource>::is_device_accessible());

        SUCCEED();
    }

    // ====================================================================
    // Type Alias Property Tests
    // ====================================================================

    TEST( TensorMemoryPropertiesTest, TypeAliasProperties_AllTypes ) {
        std::vector<size_t> shape = { 2, 3 };

        // Test type aliases with different element types

        // float aliases
        HostTensor<float> host_float_tensor( shape );
        DeviceTensor<float> device_float_tensor( shape );
        PinnedTensor<float> pinned_float_tensor( shape );
        UniversalTensor<float> universal_float_tensor( shape );

        EXPECT_TRUE( host_float_tensor.is_host_accessible() );
        EXPECT_FALSE( host_float_tensor.is_device_accessible() );
        EXPECT_FALSE( device_float_tensor.is_host_accessible() );
        EXPECT_TRUE( device_float_tensor.is_device_accessible() );
        EXPECT_TRUE( pinned_float_tensor.is_host_accessible() );
        EXPECT_TRUE( pinned_float_tensor.is_device_accessible() );
        EXPECT_TRUE( universal_float_tensor.is_host_accessible() );
        EXPECT_TRUE( universal_float_tensor.is_device_accessible() );

        // int aliases
        HostTensor<int> host_int_tensor( shape );
        DeviceTensor<int> device_int_tensor( shape );
        PinnedTensor<int> pinned_int_tensor( shape );
        UniversalTensor<int> universal_int_tensor( shape );

        EXPECT_TRUE( host_int_tensor.is_host_accessible() );
        EXPECT_FALSE( host_int_tensor.is_device_accessible() );
        EXPECT_FALSE( device_int_tensor.is_host_accessible() );
        EXPECT_TRUE( device_int_tensor.is_device_accessible() );
        EXPECT_TRUE( pinned_int_tensor.is_host_accessible() );
        EXPECT_TRUE( pinned_int_tensor.is_device_accessible() );
        EXPECT_TRUE( universal_int_tensor.is_host_accessible() );
        EXPECT_TRUE( universal_int_tensor.is_device_accessible() );

        // uint16_t aliases
        HostTensor<uint16_t> host_uint16_tensor( shape );
        DeviceTensor<uint16_t> device_uint16_tensor( shape );
        PinnedTensor<uint16_t> pinned_uint16_tensor( shape );
        UniversalTensor<uint16_t> universal_uint16_tensor( shape );

        EXPECT_TRUE( host_uint16_tensor.is_host_accessible() );
        EXPECT_FALSE( host_uint16_tensor.is_device_accessible() );
        EXPECT_FALSE( device_uint16_tensor.is_host_accessible() );
        EXPECT_TRUE( device_uint16_tensor.is_device_accessible() );
        EXPECT_TRUE( pinned_uint16_tensor.is_host_accessible() );
        EXPECT_TRUE( pinned_uint16_tensor.is_device_accessible() );
        EXPECT_TRUE( universal_uint16_tensor.is_host_accessible() );
        EXPECT_TRUE( universal_uint16_tensor.is_device_accessible() );
    }

    // ====================================================================
    // Memory Property Consistency Tests
    // ====================================================================

    TEST( TensorMemoryPropertiesTest, PropertyConsistencyAcrossOperations_AllTypes ) {
        std::vector<size_t> shape = { 3, 3 };

        // Test with float
        {
            Tensor<float, Compute::HostMemoryResource> host_tensor( shape, 1.0f );

            EXPECT_TRUE( host_tensor.is_host_accessible() );
            EXPECT_FALSE( host_tensor.is_device_accessible() );

            host_tensor.reshape( { 9 } );
            EXPECT_TRUE( host_tensor.is_host_accessible() );
            EXPECT_FALSE( host_tensor.is_device_accessible() );

            host_tensor.fill( 2.0f );
            EXPECT_TRUE( host_tensor.is_host_accessible() );
            EXPECT_FALSE( host_tensor.is_device_accessible() );

            host_tensor.reshape( { 3, 3 } );
            host_tensor.flatten();
            EXPECT_TRUE( host_tensor.is_host_accessible() );
            EXPECT_FALSE( host_tensor.is_device_accessible() );
        }

        // Test with int
        {
            Tensor<int, Compute::CudaDeviceMemoryResource> cuda_tensor( shape, 42 );

            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );

            cuda_tensor.reshape( { 9 } );
            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );

            cuda_tensor.fill( 84 );
            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );
        }

        // Test with uint16_t
        {
            Tensor<uint16_t, Compute::CudaPinnedMemoryResource> pinned_tensor( shape, 100 );

            EXPECT_TRUE( pinned_tensor.is_host_accessible() );
            EXPECT_TRUE( pinned_tensor.is_device_accessible() );

            pinned_tensor.reshape( { 9 } );
            EXPECT_TRUE( pinned_tensor.is_host_accessible() );
            EXPECT_TRUE( pinned_tensor.is_device_accessible() );

            pinned_tensor.fill( 200 );
            EXPECT_TRUE( pinned_tensor.is_host_accessible() );
            EXPECT_TRUE( pinned_tensor.is_device_accessible() );
        }
    }

    // ====================================================================
    // Data Type Information Tests
    // ====================================================================

    TEST( TensorMemoryPropertiesTest, DataTypeInformation_AllTypes ) {
        std::vector<size_t> shape = { 2, 2 };

        // Test data type information for all supported types
        {
            Tensor<float, Compute::HostMemoryResource> tensor( shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::FP32 );
            EXPECT_EQ( tensor.getDataTypeName(), "FP32" );
        }

        {
            Tensor<int16_t, Compute::HostMemoryResource> tensor( shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::INT16 );
            EXPECT_EQ( tensor.getDataTypeName(), "INT16" );
        }

        {
            Tensor<int, Compute::HostMemoryResource> tensor( shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::INT32 );
            EXPECT_EQ( tensor.getDataTypeName(), "INT32" );
        }

        {
            Tensor<uint16_t, Compute::HostMemoryResource> tensor( shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::UINT16 );
            EXPECT_EQ( tensor.getDataTypeName(), "UINT16" );
        }

        {
            Tensor<uint32_t, Compute::HostMemoryResource> tensor( shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::UINT32 );
            EXPECT_EQ( tensor.getDataTypeName(), "UINT32" );
        }
    }

    // ====================================================================
    // TensorTrait Validation Tests
    // ====================================================================

    TEST( TensorMemoryPropertiesTest, TensorTraitValidation ) {
        // Verify TensorTrait properties for all supported types

        static_assert(TensorTrait<float>::is_float_type);
        static_assert(!TensorTrait<float>::is_integer_type);
        static_assert(!TensorTrait<float>::is_device_only);
        static_assert(TensorTrait<float>::data_type == TensorDataType::FP32);
        static_assert(TensorTrait<float>::size_in_bytes == sizeof( float ));

        static_assert(!TensorTrait<int16_t>::is_float_type);
        static_assert(TensorTrait<int16_t>::is_integer_type);
        static_assert(!TensorTrait<int16_t>::is_device_only);
        static_assert(TensorTrait<int16_t>::data_type == TensorDataType::INT16);
        static_assert(TensorTrait<int16_t>::size_in_bytes == sizeof( int16_t ));

        static_assert(!TensorTrait<int>::is_float_type);
        static_assert(TensorTrait<int>::is_integer_type);
        static_assert(!TensorTrait<int>::is_device_only);
        static_assert(TensorTrait<int>::data_type == TensorDataType::INT32);
        static_assert(TensorTrait<int>::size_in_bytes == sizeof( int ));

        static_assert(!TensorTrait<uint16_t>::is_float_type);
        static_assert(TensorTrait<uint16_t>::is_integer_type);
        static_assert(!TensorTrait<uint16_t>::is_device_only);
        static_assert(TensorTrait<uint16_t>::data_type == TensorDataType::UINT16);
        static_assert(TensorTrait<uint16_t>::size_in_bytes == sizeof( uint16_t ));

        static_assert(!TensorTrait<uint32_t>::is_float_type);
        static_assert(TensorTrait<uint32_t>::is_integer_type);
        static_assert(!TensorTrait<uint32_t>::is_device_only);
        static_assert(TensorTrait<uint32_t>::data_type == TensorDataType::UINT32);
        static_assert(TensorTrait<uint32_t>::size_in_bytes == sizeof( uint32_t ));

        SUCCEED();
    }

    // ====================================================================
    // Concept Validation Tests
    // ====================================================================

    TEST( TensorMemoryPropertiesTest, ConceptValidation ) {
        // Verify that all supported types satisfy the ValidTensorType concept
        static_assert(ValidTensorType<float>);
        static_assert(ValidTensorType<int16_t>);
        static_assert(ValidTensorType<int>);
        static_assert(ValidTensorType<uint16_t>);
        static_assert(ValidTensorType<uint32_t>);

        // Verify integer type concept
        static_assert(!ValidIntTensorType<float>);
        static_assert(ValidIntTensorType<int16_t>);
        static_assert(ValidIntTensorType<int>);
        static_assert(ValidIntTensorType<uint16_t>);
        static_assert(ValidIntTensorType<uint32_t>);

        // Verify floating-point type concept
        static_assert(ValidFloatTensorType<float>);
        static_assert(!ValidFloatTensorType<int16_t>);
        static_assert(!ValidFloatTensorType<int>);
        static_assert(!ValidFloatTensorType<uint16_t>);
        static_assert(!ValidFloatTensorType<uint32_t>);

        // Verify device-only type concept (currently none are device-only in TensorTraits.ixx)
        static_assert(!DeviceOnlyType<float>);
        static_assert(!DeviceOnlyType<int16_t>);
        static_assert(!DeviceOnlyType<int>);
        static_assert(!DeviceOnlyType<uint16_t>);
        static_assert(!DeviceOnlyType<uint32_t>);

        // Verify isValidTensor concept
        static_assert(isValidTensor<float, Compute::HostMemoryResource>);
        static_assert(isValidTensor<float, Compute::CudaDeviceMemoryResource>);
        static_assert(isValidTensor<int, Compute::HostMemoryResource>);
        static_assert(isValidTensor<int, Compute::CudaDeviceMemoryResource>);
        static_assert(isValidTensor<uint16_t, Compute::CudaPinnedMemoryResource>);
        static_assert(isValidTensor<uint32_t, Compute::CudaManagedMemoryResource>);

        SUCCEED();
    }
}