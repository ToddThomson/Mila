#include <gtest/gtest.h>

import Mila;

namespace Dnn::Tensors::Tests
{
    using namespace Mila::Dnn;

    class TensorMemoryPropertiesTest : public testing::Test {
    protected:
        TensorMemoryPropertiesTest() {}
    };

    // ====================================================================
    // Host-Compatible Abstract Data Types with All Memory Resources
    // ====================================================================

    TEST( TensorMemoryPropertiesTest, HostCompatibleDataTypes_AllMemoryResources ) {
        std::vector<size_t> shape = { 2, 3 };

        // FP32 (host-compatible floating-point)
        {
            Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> host_tensor( "CPU", shape );
            Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource> cuda_tensor( "CUDA:0", shape );
            Tensor<TensorDataType::FP32, Compute::CudaPinnedMemoryResource> pinned_tensor( "CUDA:0", shape );
            Tensor<TensorDataType::FP32, Compute::CudaManagedMemoryResource> managed_tensor( "CUDA:0", shape );

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

        // INT16 (host-compatible signed integer)
        {
            Tensor<TensorDataType::INT16, Compute::CpuMemoryResource> host_tensor( "CPU", shape );
            Tensor<TensorDataType::INT16, Compute::CudaDeviceMemoryResource> cuda_tensor( "CUDA:0", shape );
            Tensor<TensorDataType::INT16, Compute::CudaPinnedMemoryResource> pinned_tensor( "CUDA:0", shape );
            Tensor<TensorDataType::INT16, Compute::CudaManagedMemoryResource> managed_tensor( "CUDA:0", shape );

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

        // INT32 (host-compatible signed integer)
        {
            Tensor<TensorDataType::INT32, Compute::CpuMemoryResource> host_tensor( "CPU", shape );
            Tensor<TensorDataType::INT32, Compute::CudaDeviceMemoryResource> cuda_tensor( "CUDA:0", shape );
            Tensor<TensorDataType::INT32, Compute::CudaPinnedMemoryResource> pinned_tensor( "CUDA:0", shape );
            Tensor<TensorDataType::INT32, Compute::CudaManagedMemoryResource> managed_tensor( "CUDA:0", shape );

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

        // UINT16 (host-compatible unsigned integer)
        {
            Tensor<TensorDataType::UINT16, Compute::CpuMemoryResource> host_tensor( "CPU", shape );
            Tensor<TensorDataType::UINT16, Compute::CudaDeviceMemoryResource> cuda_tensor( "CUDA:0", shape );
            Tensor<TensorDataType::UINT16, Compute::CudaPinnedMemoryResource> pinned_tensor( "CUDA:0", shape );
            Tensor<TensorDataType::UINT16, Compute::CudaManagedMemoryResource> managed_tensor( "CUDA:0", shape );

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

        // UINT32 (host-compatible unsigned integer)
        {
            Tensor<TensorDataType::UINT32, Compute::CpuMemoryResource> host_tensor( "CPU", shape );
            Tensor<TensorDataType::UINT32, Compute::CudaDeviceMemoryResource> cuda_tensor( "CUDA:0", shape );
            Tensor<TensorDataType::UINT32, Compute::CudaPinnedMemoryResource> pinned_tensor( "CUDA:0", shape );
            Tensor<TensorDataType::UINT32, Compute::CudaManagedMemoryResource> managed_tensor( "CUDA:0", shape );

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
    // Device-Only Abstract Data Types (CUDA only)
    // ====================================================================

    TEST( TensorMemoryPropertiesTest, DeviceOnlyDataTypes_CudaMemoryResources ) {
        std::vector<size_t> shape = { 2, 3 };

        // FP16 (device-only half precision)
        {
            Tensor<TensorDataType::FP16, Compute::CudaDeviceMemoryResource> cuda_tensor( "CUDA:0", shape );
            Tensor<TensorDataType::FP16, Compute::CudaPinnedMemoryResource> pinned_tensor( "CUDA:0", shape );
            Tensor<TensorDataType::FP16, Compute::CudaManagedMemoryResource> managed_tensor( "CUDA:0", shape );

            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );
            EXPECT_TRUE( pinned_tensor.is_host_accessible() );
            EXPECT_TRUE( pinned_tensor.is_device_accessible() );
            EXPECT_TRUE( managed_tensor.is_host_accessible() );
            EXPECT_TRUE( managed_tensor.is_device_accessible() );

            EXPECT_EQ( cuda_tensor.getDataType(), TensorDataType::FP16 );
            EXPECT_EQ( pinned_tensor.getDataType(), TensorDataType::FP16 );
            EXPECT_EQ( managed_tensor.getDataType(), TensorDataType::FP16 );

            EXPECT_EQ( cuda_tensor.getDataTypeName(), "FP16" );
            EXPECT_EQ( pinned_tensor.getDataTypeName(), "FP16" );
            EXPECT_EQ( managed_tensor.getDataTypeName(), "FP16" );
        }

        // BF16 (device-only brain float)
        {
            Tensor<TensorDataType::BF16, Compute::CudaDeviceMemoryResource> cuda_tensor( "CUDA:0", shape );
            Tensor<TensorDataType::BF16, Compute::CudaPinnedMemoryResource> pinned_tensor( "CUDA:0", shape );
            Tensor<TensorDataType::BF16, Compute::CudaManagedMemoryResource> managed_tensor( "CUDA:0", shape );

            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );
            EXPECT_TRUE( pinned_tensor.is_host_accessible() );
            EXPECT_TRUE( pinned_tensor.is_device_accessible() );
            EXPECT_TRUE( managed_tensor.is_host_accessible() );
            EXPECT_TRUE( managed_tensor.is_device_accessible() );

            EXPECT_EQ( cuda_tensor.getDataType(), TensorDataType::BF16 );
            EXPECT_EQ( pinned_tensor.getDataType(), TensorDataType::BF16 );
            EXPECT_EQ( managed_tensor.getDataType(), TensorDataType::BF16 );

            EXPECT_EQ( cuda_tensor.getDataTypeName(), "BF16" );
        }

        // FP8_E4M3 (device-only 8-bit float)
        {
            Tensor<TensorDataType::FP8_E4M3, Compute::CudaDeviceMemoryResource> cuda_tensor( "CUDA:0", shape );
            Tensor<TensorDataType::FP8_E4M3, Compute::CudaPinnedMemoryResource> pinned_tensor( "CUDA:0", shape );
            Tensor<TensorDataType::FP8_E4M3, Compute::CudaManagedMemoryResource> managed_tensor( "CUDA:0", shape );

            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );
            EXPECT_TRUE( pinned_tensor.is_host_accessible() );
            EXPECT_TRUE( pinned_tensor.is_device_accessible() );
            EXPECT_TRUE( managed_tensor.is_host_accessible() );
            EXPECT_TRUE( managed_tensor.is_device_accessible() );

            EXPECT_EQ( cuda_tensor.getDataType(), TensorDataType::FP8_E4M3 );
            EXPECT_EQ( cuda_tensor.getDataTypeName(), "FP8_E4M3" );
        }

        // FP8_E5M2 (device-only 8-bit float alternative)
        {
            Tensor<TensorDataType::FP8_E5M2, Compute::CudaDeviceMemoryResource> cuda_tensor( "CUDA:0", shape );

            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );
            EXPECT_EQ( cuda_tensor.getDataType(), TensorDataType::FP8_E5M2 );
            EXPECT_EQ( cuda_tensor.getDataTypeName(), "FP8_E5M2" );
        }
    }

    // ====================================================================
    // Static Class Method Tests for Abstract Data Types
    // ====================================================================

    TEST( TensorMemoryPropertiesTest, StaticMethods_AbstractDataTypes ) {
        // Host-compatible types can be used with all memory resources

        // FP32
        EXPECT_TRUE( (Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>::is_host_accessible()) );
        EXPECT_FALSE( (Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>::is_device_accessible()) );
        EXPECT_FALSE( (Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>::is_host_accessible()) );
        EXPECT_TRUE( (Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>::is_device_accessible()) );
        EXPECT_TRUE( (Tensor<TensorDataType::FP32, Compute::CudaPinnedMemoryResource>::is_host_accessible()) );
        EXPECT_TRUE( (Tensor<TensorDataType::FP32, Compute::CudaPinnedMemoryResource>::is_device_accessible()) );
        EXPECT_TRUE( (Tensor<TensorDataType::FP32, Compute::CudaManagedMemoryResource>::is_host_accessible()) );
        EXPECT_TRUE( (Tensor<TensorDataType::FP32, Compute::CudaManagedMemoryResource>::is_device_accessible()) );

        // INT16
        EXPECT_TRUE( (Tensor<TensorDataType::INT16, Compute::CpuMemoryResource>::is_host_accessible()) );
        EXPECT_FALSE( (Tensor<TensorDataType::INT16, Compute::CpuMemoryResource>::is_device_accessible()) );
        EXPECT_FALSE( (Tensor<TensorDataType::INT16, Compute::CudaDeviceMemoryResource>::is_host_accessible()) );
        EXPECT_TRUE( (Tensor<TensorDataType::INT16, Compute::CudaDeviceMemoryResource>::is_device_accessible()) );

        // INT32
        EXPECT_TRUE( (Tensor<TensorDataType::INT32, Compute::CpuMemoryResource>::is_host_accessible()) );
        EXPECT_FALSE( (Tensor<TensorDataType::INT32, Compute::CpuMemoryResource>::is_device_accessible()) );
        EXPECT_FALSE( (Tensor<TensorDataType::INT32, Compute::CudaDeviceMemoryResource>::is_host_accessible()) );
        EXPECT_TRUE( (Tensor<TensorDataType::INT32, Compute::CudaDeviceMemoryResource>::is_device_accessible()) );

        // UINT16
        EXPECT_TRUE( (Tensor<TensorDataType::UINT16, Compute::CpuMemoryResource>::is_host_accessible()) );
        EXPECT_FALSE( (Tensor<TensorDataType::UINT16, Compute::CpuMemoryResource>::is_device_accessible()) );
        EXPECT_FALSE( (Tensor<TensorDataType::UINT16, Compute::CudaDeviceMemoryResource>::is_host_accessible()) );
        EXPECT_TRUE( (Tensor<TensorDataType::UINT16, Compute::CudaDeviceMemoryResource>::is_device_accessible()) );

        // UINT32
        EXPECT_TRUE( (Tensor<TensorDataType::UINT32, Compute::CpuMemoryResource>::is_host_accessible()) );
        EXPECT_FALSE( (Tensor<TensorDataType::UINT32, Compute::CpuMemoryResource>::is_device_accessible()) );
        EXPECT_FALSE( (Tensor<TensorDataType::UINT32, Compute::CudaDeviceMemoryResource>::is_host_accessible()) );
        EXPECT_TRUE( (Tensor<TensorDataType::UINT32, Compute::CudaDeviceMemoryResource>::is_device_accessible()) );

        // Device-only types
        EXPECT_FALSE( (Tensor<TensorDataType::FP16, Compute::CudaDeviceMemoryResource>::is_host_accessible()) );
        EXPECT_TRUE( (Tensor<TensorDataType::FP16, Compute::CudaDeviceMemoryResource>::is_device_accessible()) );
        EXPECT_FALSE( (Tensor<TensorDataType::BF16, Compute::CudaDeviceMemoryResource>::is_host_accessible()) );
        EXPECT_TRUE( (Tensor<TensorDataType::BF16, Compute::CudaDeviceMemoryResource>::is_device_accessible()) );
    }

    // ====================================================================
    // Compile-time Property Verification
    // ====================================================================

    TEST( TensorMemoryPropertiesTest, CompileTimeProperties_AbstractDataTypes ) {
        // Host-compatible types with all memory resources
        static_assert(Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>::is_host_accessible());
        static_assert(!Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>::is_device_accessible());
        static_assert(!Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>::is_host_accessible());
        static_assert(Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>::is_device_accessible());
        static_assert(Tensor<TensorDataType::FP32, Compute::CudaPinnedMemoryResource>::is_host_accessible());
        static_assert(Tensor<TensorDataType::FP32, Compute::CudaPinnedMemoryResource>::is_device_accessible());
        static_assert(Tensor<TensorDataType::FP32, Compute::CudaManagedMemoryResource>::is_host_accessible());
        static_assert(Tensor<TensorDataType::FP32, Compute::CudaManagedMemoryResource>::is_device_accessible());

        static_assert(Tensor<TensorDataType::INT16, Compute::CpuMemoryResource>::is_host_accessible());
        static_assert(!Tensor<TensorDataType::INT16, Compute::CpuMemoryResource>::is_device_accessible());
        static_assert(!Tensor<TensorDataType::INT16, Compute::CudaDeviceMemoryResource>::is_host_accessible());
        static_assert(Tensor<TensorDataType::INT16, Compute::CudaDeviceMemoryResource>::is_device_accessible());

        static_assert(Tensor<TensorDataType::INT32, Compute::CpuMemoryResource>::is_host_accessible());
        static_assert(!Tensor<TensorDataType::INT32, Compute::CpuMemoryResource>::is_device_accessible());
        static_assert(!Tensor<TensorDataType::INT32, Compute::CudaDeviceMemoryResource>::is_host_accessible());
        static_assert(Tensor<TensorDataType::INT32, Compute::CudaDeviceMemoryResource>::is_device_accessible());

        static_assert(Tensor<TensorDataType::UINT16, Compute::CpuMemoryResource>::is_host_accessible());
        static_assert(!Tensor<TensorDataType::UINT16, Compute::CpuMemoryResource>::is_device_accessible());
        static_assert(!Tensor<TensorDataType::UINT16, Compute::CudaDeviceMemoryResource>::is_host_accessible());
        static_assert(Tensor<TensorDataType::UINT16, Compute::CudaDeviceMemoryResource>::is_device_accessible());

        static_assert(Tensor<TensorDataType::UINT32, Compute::CpuMemoryResource>::is_host_accessible());
        static_assert(!Tensor<TensorDataType::UINT32, Compute::CpuMemoryResource>::is_device_accessible());
        static_assert(!Tensor<TensorDataType::UINT32, Compute::CudaDeviceMemoryResource>::is_host_accessible());
        static_assert(Tensor<TensorDataType::UINT32, Compute::CudaDeviceMemoryResource>::is_device_accessible());

        // Device-only types
        static_assert(!Tensor<TensorDataType::FP16, Compute::CudaDeviceMemoryResource>::is_host_accessible());
        static_assert(Tensor<TensorDataType::FP16, Compute::CudaDeviceMemoryResource>::is_device_accessible());
        static_assert(!Tensor<TensorDataType::BF16, Compute::CudaDeviceMemoryResource>::is_host_accessible());
        static_assert(Tensor<TensorDataType::BF16, Compute::CudaDeviceMemoryResource>::is_device_accessible());

        SUCCEED();
    }

    // ====================================================================
    // Type Alias Property Tests (Updated for Abstract Data Types)
    // ====================================================================

    TEST( TensorMemoryPropertiesTest, TypeAliasProperties_AbstractDataTypes ) {
        std::vector<size_t> shape = { 2, 3 };

        // Test type aliases with different abstract data types

        // FP32 aliases
        HostTensor<TensorDataType::FP32> host_fp32_tensor( "CPU", shape );
        DeviceTensor<TensorDataType::FP32> device_fp32_tensor( "CUDA:0", shape );
        PinnedTensor<TensorDataType::FP32> pinned_fp32_tensor( "CUDA:0", shape );
        UniversalTensor<TensorDataType::FP32> universal_fp32_tensor( "CUDA:0", shape );

        EXPECT_TRUE( host_fp32_tensor.is_host_accessible() );
        EXPECT_FALSE( host_fp32_tensor.is_device_accessible() );
        EXPECT_FALSE( device_fp32_tensor.is_host_accessible() );
        EXPECT_TRUE( device_fp32_tensor.is_device_accessible() );
        EXPECT_TRUE( pinned_fp32_tensor.is_host_accessible() );
        EXPECT_TRUE( pinned_fp32_tensor.is_device_accessible() );
        EXPECT_TRUE( universal_fp32_tensor.is_host_accessible() );
        EXPECT_TRUE( universal_fp32_tensor.is_device_accessible() );

        // INT32 aliases
        HostTensor<TensorDataType::INT32> host_int32_tensor( "CPU", shape );
        DeviceTensor<TensorDataType::INT32> device_int32_tensor( "CUDA:0", shape );
        PinnedTensor<TensorDataType::INT32> pinned_int32_tensor( "CUDA:0", shape );
        UniversalTensor<TensorDataType::INT32> universal_int32_tensor( "CUDA:0", shape );

        EXPECT_TRUE( host_int32_tensor.is_host_accessible() );
        EXPECT_FALSE( host_int32_tensor.is_device_accessible() );
        EXPECT_FALSE( device_int32_tensor.is_host_accessible() );
        EXPECT_TRUE( device_int32_tensor.is_device_accessible() );
        EXPECT_TRUE( pinned_int32_tensor.is_host_accessible() );
        EXPECT_TRUE( pinned_int32_tensor.is_device_accessible() );
        EXPECT_TRUE( universal_int32_tensor.is_host_accessible() );
        EXPECT_TRUE( universal_int32_tensor.is_device_accessible() );

        // UINT16 aliases
        HostTensor<TensorDataType::UINT16> host_uint16_tensor( "CPU", shape );
        DeviceTensor<TensorDataType::UINT16> device_uint16_tensor( "CUDA:0", shape );
        PinnedTensor<TensorDataType::UINT16> pinned_uint16_tensor( "CUDA:0", shape );
        UniversalTensor<TensorDataType::UINT16> universal_uint16_tensor( "CUDA:0", shape );

        EXPECT_TRUE( host_uint16_tensor.is_host_accessible() );
        EXPECT_FALSE( host_uint16_tensor.is_device_accessible() );
        EXPECT_FALSE( device_uint16_tensor.is_host_accessible() );
        EXPECT_TRUE( device_uint16_tensor.is_device_accessible() );
        EXPECT_TRUE( pinned_uint16_tensor.is_host_accessible() );
        EXPECT_TRUE( pinned_uint16_tensor.is_device_accessible() );
        EXPECT_TRUE( universal_uint16_tensor.is_host_accessible() );
        EXPECT_TRUE( universal_uint16_tensor.is_device_accessible() );

        // FP16 device-only aliases
        DeviceTensor<TensorDataType::FP16> device_fp16_tensor( "CUDA:0", shape );
        PinnedTensor<TensorDataType::FP16> pinned_fp16_tensor( "CUDA:0", shape );
        UniversalTensor<TensorDataType::FP16> universal_fp16_tensor( "CUDA:0", shape );

        EXPECT_FALSE( device_fp16_tensor.is_host_accessible() );
        EXPECT_TRUE( device_fp16_tensor.is_device_accessible() );
        EXPECT_TRUE( pinned_fp16_tensor.is_host_accessible() );
        EXPECT_TRUE( pinned_fp16_tensor.is_device_accessible() );
        EXPECT_TRUE( universal_fp16_tensor.is_host_accessible() );
        EXPECT_TRUE( universal_fp16_tensor.is_device_accessible() );
    }

    // ====================================================================
    // Memory Property Consistency Tests
    // ====================================================================

    TEST( TensorMemoryPropertiesTest, PropertyConsistencyAcrossOperations_AbstractDataTypes ) {
        std::vector<size_t> shape = { 3, 3 };

        // Test with FP32
        {
            Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> host_tensor( "CPU", shape );

            EXPECT_TRUE( host_tensor.is_host_accessible() );
            EXPECT_FALSE( host_tensor.is_device_accessible() );

            host_tensor.reshape( { 9 } );
            EXPECT_TRUE( host_tensor.is_host_accessible() );
            EXPECT_FALSE( host_tensor.is_device_accessible() );

            host_tensor.reshape( { 3, 3 } );
            host_tensor.flatten();
            EXPECT_TRUE( host_tensor.is_host_accessible() );
            EXPECT_FALSE( host_tensor.is_device_accessible() );
        }

        // Test with INT32
        {
            Tensor<TensorDataType::INT32, Compute::CudaDeviceMemoryResource> cuda_tensor( "CUDA:0", shape );

            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );

            cuda_tensor.reshape( { 9 } );
            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );
        }

        // Test with UINT16
        {
            Tensor<TensorDataType::UINT16, Compute::CudaPinnedMemoryResource> pinned_tensor( "CUDA:0", shape );

            EXPECT_TRUE( pinned_tensor.is_host_accessible() );
            EXPECT_TRUE( pinned_tensor.is_device_accessible() );

            pinned_tensor.reshape( { 9 } );
            EXPECT_TRUE( pinned_tensor.is_host_accessible() );
            EXPECT_TRUE( pinned_tensor.is_device_accessible() );
        }

        // Test with device-only FP16
        {
            Tensor<TensorDataType::FP16, Compute::CudaManagedMemoryResource> managed_tensor( "CUDA:0", shape );

            EXPECT_TRUE( managed_tensor.is_host_accessible() );
            EXPECT_TRUE( managed_tensor.is_device_accessible() );

            managed_tensor.reshape( { 9 } );
            EXPECT_TRUE( managed_tensor.is_host_accessible() );
            EXPECT_TRUE( managed_tensor.is_device_accessible() );
        }
    }

    // ====================================================================
    // Data Type Information Tests
    // ====================================================================

    TEST( TensorMemoryPropertiesTest, DataTypeInformation_AbstractDataTypes ) {
        std::vector<size_t> shape = { 2, 2 };

        // Test data type information for all supported abstract types
        {
            Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::FP32 );
            EXPECT_EQ( tensor.getDataTypeName(), "FP32" );
        }

        {
            Tensor<TensorDataType::FP16, Compute::CudaDeviceMemoryResource> tensor( "CUDA:0", shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::FP16 );
            EXPECT_EQ( tensor.getDataTypeName(), "FP16" );
        }

        {
            Tensor<TensorDataType::BF16, Compute::CudaDeviceMemoryResource> tensor( "CUDA:0", shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::BF16 );
            EXPECT_EQ( tensor.getDataTypeName(), "BF16" );
        }

        {
            Tensor<TensorDataType::FP8_E4M3, Compute::CudaDeviceMemoryResource> tensor( "CUDA:0", shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::FP8_E4M3 );
            EXPECT_EQ( tensor.getDataTypeName(), "FP8_E4M3" );
        }

        {
            Tensor<TensorDataType::FP8_E5M2, Compute::CudaDeviceMemoryResource> tensor( "CUDA:0", shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::FP8_E5M2 );
            EXPECT_EQ( tensor.getDataTypeName(), "FP8_E5M2" );
        }

        {
            Tensor<TensorDataType::INT8, Compute::CpuMemoryResource> tensor( "CPU", shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::INT8 );
            EXPECT_EQ( tensor.getDataTypeName(), "INT8" );
        }

        {
            Tensor<TensorDataType::INT16, Compute::CpuMemoryResource> tensor( "CPU", shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::INT16 );
            EXPECT_EQ( tensor.getDataTypeName(), "INT16" );
        }

        {
            Tensor<TensorDataType::INT32, Compute::CpuMemoryResource> tensor( "CPU", shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::INT32 );
            EXPECT_EQ( tensor.getDataTypeName(), "INT32" );
        }

        {
            Tensor<TensorDataType::UINT8, Compute::CpuMemoryResource> tensor( "CPU", shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::UINT8 );
            EXPECT_EQ( tensor.getDataTypeName(), "UINT8" );
        }

        {
            Tensor<TensorDataType::UINT16, Compute::CpuMemoryResource> tensor( "CPU", shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::UINT16 );
            EXPECT_EQ( tensor.getDataTypeName(), "UINT16" );
        }

        {
            Tensor<TensorDataType::UINT32, Compute::CpuMemoryResource> tensor( "CPU", shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::UINT32 );
            EXPECT_EQ( tensor.getDataTypeName(), "UINT32" );
        }
    }

    // ====================================================================
    // TensorDataTypeTraits Validation Tests
    // ====================================================================

    TEST( TensorMemoryPropertiesTest, TensorDataTypeTraitsValidation ) {
        // Verify TensorDataTypeTraits properties for all supported abstract types

        // FP32 traits
        static_assert(TensorDataTypeTraits<TensorDataType::FP32>::is_float_type);
        static_assert(!TensorDataTypeTraits<TensorDataType::FP32>::is_integer_type);
        static_assert(!TensorDataTypeTraits<TensorDataType::FP32>::is_device_only);
        static_assert(TensorDataTypeTraits<TensorDataType::FP32>::size_in_bytes == 4);

        // FP16 traits
        static_assert(TensorDataTypeTraits<TensorDataType::FP16>::is_float_type);
        static_assert(!TensorDataTypeTraits<TensorDataType::FP16>::is_integer_type);
        static_assert(TensorDataTypeTraits<TensorDataType::FP16>::is_device_only);
        static_assert(TensorDataTypeTraits<TensorDataType::FP16>::size_in_bytes == 2);

        // BF16 traits
        static_assert(TensorDataTypeTraits<TensorDataType::BF16>::is_float_type);
        static_assert(!TensorDataTypeTraits<TensorDataType::BF16>::is_integer_type);
        static_assert(TensorDataTypeTraits<TensorDataType::BF16>::is_device_only);
        static_assert(TensorDataTypeTraits<TensorDataType::BF16>::size_in_bytes == 2);

        // FP8 traits
        static_assert(TensorDataTypeTraits<TensorDataType::FP8_E4M3>::is_float_type);
        static_assert(!TensorDataTypeTraits<TensorDataType::FP8_E4M3>::is_integer_type);
        static_assert(TensorDataTypeTraits<TensorDataType::FP8_E4M3>::is_device_only);
        static_assert(TensorDataTypeTraits<TensorDataType::FP8_E4M3>::size_in_bytes == 1);

        static_assert(TensorDataTypeTraits<TensorDataType::FP8_E5M2>::is_float_type);
        static_assert(!TensorDataTypeTraits<TensorDataType::FP8_E5M2>::is_integer_type);
        static_assert(TensorDataTypeTraits<TensorDataType::FP8_E5M2>::is_device_only);
        static_assert(TensorDataTypeTraits<TensorDataType::FP8_E5M2>::size_in_bytes == 1);

        // Integer traits
        static_assert(!TensorDataTypeTraits<TensorDataType::INT8>::is_float_type);
        static_assert(TensorDataTypeTraits<TensorDataType::INT8>::is_integer_type);
        static_assert(!TensorDataTypeTraits<TensorDataType::INT8>::is_device_only);
        static_assert(TensorDataTypeTraits<TensorDataType::INT8>::size_in_bytes == 1);

        static_assert(!TensorDataTypeTraits<TensorDataType::INT16>::is_float_type);
        static_assert(TensorDataTypeTraits<TensorDataType::INT16>::is_integer_type);
        static_assert(!TensorDataTypeTraits<TensorDataType::INT16>::is_device_only);
        static_assert(TensorDataTypeTraits<TensorDataType::INT16>::size_in_bytes == 2);

        static_assert(!TensorDataTypeTraits<TensorDataType::INT32>::is_float_type);
        static_assert(TensorDataTypeTraits<TensorDataType::INT32>::is_integer_type);
        static_assert(!TensorDataTypeTraits<TensorDataType::INT32>::is_device_only);
        static_assert(TensorDataTypeTraits<TensorDataType::INT32>::size_in_bytes == 4);

        static_assert(!TensorDataTypeTraits<TensorDataType::UINT8>::is_float_type);
        static_assert(TensorDataTypeTraits<TensorDataType::UINT8>::is_integer_type);
        static_assert(!TensorDataTypeTraits<TensorDataType::UINT8>::is_device_only);
        static_assert(TensorDataTypeTraits<TensorDataType::UINT8>::size_in_bytes == 1);

        static_assert(!TensorDataTypeTraits<TensorDataType::UINT16>::is_float_type);
        static_assert(TensorDataTypeTraits<TensorDataType::UINT16>::is_integer_type);
        static_assert(!TensorDataTypeTraits<TensorDataType::UINT16>::is_device_only);
        static_assert(TensorDataTypeTraits<TensorDataType::UINT16>::size_in_bytes == 2);

        static_assert(!TensorDataTypeTraits<TensorDataType::UINT32>::is_float_type);
        static_assert(TensorDataTypeTraits<TensorDataType::UINT32>::is_integer_type);
        static_assert(!TensorDataTypeTraits<TensorDataType::UINT32>::is_device_only);
        static_assert(TensorDataTypeTraits<TensorDataType::UINT32>::size_in_bytes == 4);

        SUCCEED();
    }

    // ====================================================================
    // Concept Validation Tests
    // ====================================================================

    TEST( TensorMemoryPropertiesTest, ConceptValidation ) {
        // Verify that all supported abstract data types satisfy the validation concept
        static_assert(isValidTensor<TensorDataType::FP32, Compute::CpuMemoryResource>);
        static_assert(isValidTensor<TensorDataType::FP16, Compute::CudaDeviceMemoryResource>);
        static_assert(isValidTensor<TensorDataType::BF16, Compute::CudaDeviceMemoryResource>);
        static_assert(isValidTensor<TensorDataType::FP8_E4M3, Compute::CudaDeviceMemoryResource>);
        static_assert(isValidTensor<TensorDataType::FP8_E5M2, Compute::CudaDeviceMemoryResource>);
        static_assert(isValidTensor<TensorDataType::INT8, Compute::CpuMemoryResource>);
        static_assert(isValidTensor<TensorDataType::INT16, Compute::CpuMemoryResource>);
        static_assert(isValidTensor<TensorDataType::INT32, Compute::CpuMemoryResource>);
        static_assert(isValidTensor<TensorDataType::UINT8, Compute::CpuMemoryResource>);
        static_assert(isValidTensor<TensorDataType::UINT16, Compute::CpuMemoryResource>);
        static_assert(isValidTensor<TensorDataType::UINT32, Compute::CpuMemoryResource>);

        // Verify floating-point type concept
        static_assert(ValidFloatTensorDataType<TensorDataType::FP32>);
        static_assert(ValidFloatTensorDataType<TensorDataType::FP16>);
        static_assert(ValidFloatTensorDataType<TensorDataType::BF16>);
        static_assert(ValidFloatTensorDataType<TensorDataType::FP8_E4M3>);
        static_assert(ValidFloatTensorDataType<TensorDataType::FP8_E5M2>);
        static_assert(!ValidFloatTensorDataType<TensorDataType::INT16>);
        static_assert(!ValidFloatTensorDataType<TensorDataType::INT32>);
        static_assert(!ValidFloatTensorDataType<TensorDataType::UINT16>);
        static_assert(!ValidFloatTensorDataType<TensorDataType::UINT32>);

        // Verify integer type concept
        static_assert(!ValidIntegerTensorDataType<TensorDataType::FP32>);
        static_assert(!ValidIntegerTensorDataType<TensorDataType::FP16>);
        static_assert(!ValidIntegerTensorDataType<TensorDataType::BF16>);
        static_assert(ValidIntegerTensorDataType<TensorDataType::INT8>);
        static_assert(ValidIntegerTensorDataType<TensorDataType::INT16>);
        static_assert(ValidIntegerTensorDataType<TensorDataType::INT32>);
        static_assert(ValidIntegerTensorDataType<TensorDataType::UINT8>);
        static_assert(ValidIntegerTensorDataType<TensorDataType::UINT16>);
        static_assert(ValidIntegerTensorDataType<TensorDataType::UINT32>);

        // Verify device-only type concept
        static_assert(!DeviceOnlyTensorDataType<TensorDataType::FP32>);
        static_assert(DeviceOnlyTensorDataType<TensorDataType::FP16>);
        static_assert(DeviceOnlyTensorDataType<TensorDataType::BF16>);
        static_assert(DeviceOnlyTensorDataType<TensorDataType::FP8_E4M3>);
        static_assert(DeviceOnlyTensorDataType<TensorDataType::FP8_E5M2>);
        static_assert(!DeviceOnlyTensorDataType<TensorDataType::INT8>);
        static_assert(!DeviceOnlyTensorDataType<TensorDataType::INT16>);
        static_assert(!DeviceOnlyTensorDataType<TensorDataType::INT32>);
        static_assert(!DeviceOnlyTensorDataType<TensorDataType::UINT8>);
        static_assert(!DeviceOnlyTensorDataType<TensorDataType::UINT16>);
        static_assert(!DeviceOnlyTensorDataType<TensorDataType::UINT32>);

        // Verify host-compatible type concept
        static_assert(HostCompatibleTensorDataType<TensorDataType::FP32>);
        static_assert(!HostCompatibleTensorDataType<TensorDataType::FP16>);
        static_assert(!HostCompatibleTensorDataType<TensorDataType::BF16>);
        static_assert(!HostCompatibleTensorDataType<TensorDataType::FP8_E4M3>);
        static_assert(!HostCompatibleTensorDataType<TensorDataType::FP8_E5M2>);
        static_assert(HostCompatibleTensorDataType<TensorDataType::INT8>);
        static_assert(HostCompatibleTensorDataType<TensorDataType::INT16>);
        static_assert(HostCompatibleTensorDataType<TensorDataType::INT32>);
        static_assert(HostCompatibleTensorDataType<TensorDataType::UINT8>);
        static_assert(HostCompatibleTensorDataType<TensorDataType::UINT16>);
        static_assert(HostCompatibleTensorDataType<TensorDataType::UINT32>);

        // Verify mixed memory resource compatibility
        static_assert(isValidTensor<TensorDataType::FP32, Compute::CudaManagedMemoryResource>);
        static_assert(isValidTensor<TensorDataType::FP16, Compute::CudaPinnedMemoryResource>);
        static_assert(isValidTensor<TensorDataType::BF16, Compute::CudaManagedMemoryResource>);
        static_assert(isValidTensor<TensorDataType::FP8_E4M3, Compute::CudaPinnedMemoryResource>);
        static_assert(isValidTensor<TensorDataType::INT32, Compute::CudaManagedMemoryResource>);
        static_assert(isValidTensor<TensorDataType::UINT16, Compute::CudaPinnedMemoryResource>);

        SUCCEED();
    }

    // ====================================================================
    // Device Context Compatibility Tests
    // ====================================================================

    TEST( TensorMemoryPropertiesTest, DeviceContextCompatibility ) {
        std::vector<size_t> shape = { 2, 2 };

        // Test that tensors properly report their device types
        {
            Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> cpu_tensor( "CPU", shape );
            EXPECT_EQ( cpu_tensor.getDeviceType(), Compute::DeviceType::Cpu );
        }

        {
            Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource> cuda_tensor( "CUDA:0", shape );
            EXPECT_EQ( cuda_tensor.getDeviceType(), Compute::DeviceType::Cuda );
        }

        {
            Tensor<TensorDataType::FP16, Compute::CudaDeviceMemoryResource> cuda_fp16_tensor( "CUDA:0", shape );
            EXPECT_EQ( cuda_fp16_tensor.getDeviceType(), Compute::DeviceType::Cuda );
        }

        {
            Tensor<TensorDataType::INT32, Compute::CudaPinnedMemoryResource> pinned_tensor( "CUDA:0", shape );
            EXPECT_EQ( pinned_tensor.getDeviceType(), Compute::DeviceType::Cuda );
        }

        {
            Tensor<TensorDataType::BF16, Compute::CudaManagedMemoryResource> managed_tensor( "CUDA:0", shape );
            EXPECT_EQ( managed_tensor.getDeviceType(), Compute::DeviceType::Cuda );
        }
    }

    // ====================================================================
    // Memory Resource Properties Verification
    // ====================================================================

    TEST( TensorMemoryPropertiesTest, MemoryResourcePropertiesVerification ) {
        // Verify static memory resource properties
        static_assert(Compute::CpuMemoryResource::is_host_accessible == true);
        static_assert(Compute::CpuMemoryResource::is_device_accessible == false);
        static_assert(Compute::CpuMemoryResource::device_type == Compute::DeviceType::Cpu);

        static_assert(Compute::CudaDeviceMemoryResource::is_host_accessible == false);
        static_assert(Compute::CudaDeviceMemoryResource::is_device_accessible == true);
        static_assert(Compute::CudaDeviceMemoryResource::device_type == Compute::DeviceType::Cuda);

        static_assert(Compute::CudaPinnedMemoryResource::is_host_accessible == true);
        static_assert(Compute::CudaPinnedMemoryResource::is_device_accessible == true);
        static_assert(Compute::CudaPinnedMemoryResource::device_type == Compute::DeviceType::Cuda);

        static_assert(Compute::CudaManagedMemoryResource::is_host_accessible == true);
        static_assert(Compute::CudaManagedMemoryResource::is_device_accessible == true);
        static_assert(Compute::CudaManagedMemoryResource::device_type == Compute::DeviceType::Cuda);

        SUCCEED();
    }
}