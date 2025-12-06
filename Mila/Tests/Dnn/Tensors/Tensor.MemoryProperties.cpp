#include <gtest/gtest.h>

import Mila;

namespace Dnn::Tensors::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TensorMemoryPropertiesTest : public testing::Test {
    protected:
        void SetUp() override {
            has_cuda_device_ = DeviceRegistry::instance().hasDeviceType( DeviceType::Cuda );
        }

        TensorMemoryPropertiesTest() {}

        bool has_cuda_device_;
    };

    // ====================================================================
    // Host-only tests: verify CPU memory resource behavior and host-accessible types
    // ====================================================================

    TEST_F( TensorMemoryPropertiesTest, HostOnly_HostCompatibleDataTypes_CpuMemoryResource ) {
        std::vector<int64_t> shape = { 2, 3 };

        // FP32 (host-compatible floating-point) - CPU only
        {
            Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> host_tensor( Device::Cpu(), shape );

            EXPECT_TRUE( host_tensor.is_host_accessible() );
            EXPECT_FALSE( host_tensor.is_device_accessible() );

            EXPECT_EQ( host_tensor.getDataType(), TensorDataType::FP32 );
            EXPECT_EQ( host_tensor.getDataTypeName(), "FP32" );
        }

        // INT16 (host-compatible signed integer) - CPU only
        {
            Tensor<TensorDataType::INT16, Compute::CpuMemoryResource> host_tensor( Device::Cpu(), shape );

            EXPECT_TRUE( host_tensor.is_host_accessible() );
            EXPECT_FALSE( host_tensor.is_device_accessible() );

            EXPECT_EQ( host_tensor.getDataType(), TensorDataType::INT16 );
            EXPECT_EQ( host_tensor.getDataTypeName(), "INT16" );
        }

        // INT32 (host-compatible signed integer) - CPU only
        {
            Tensor<TensorDataType::INT32, Compute::CpuMemoryResource> host_tensor( Device::Cpu(), shape );

            EXPECT_TRUE( host_tensor.is_host_accessible() );
            EXPECT_FALSE( host_tensor.is_device_accessible() );

            EXPECT_EQ( host_tensor.getDataType(), TensorDataType::INT32 );
            EXPECT_EQ( host_tensor.getDataTypeName(), "INT32" );
        }

        // UINT16 (host-compatible unsigned integer) - CPU only
        {
            Tensor<TensorDataType::UINT16, Compute::CpuMemoryResource> host_tensor( Device::Cpu(), shape );

            EXPECT_TRUE( host_tensor.is_host_accessible() );
            EXPECT_FALSE( host_tensor.is_device_accessible() );

            EXPECT_EQ( host_tensor.getDataType(), TensorDataType::UINT16 );
            EXPECT_EQ( host_tensor.getDataTypeName(), "UINT16" );
        }

        // UINT32 (host-compatible unsigned integer) - CPU only
        {
            Tensor<TensorDataType::UINT32, Compute::CpuMemoryResource> host_tensor( Device::Cpu(), shape );

            EXPECT_TRUE( host_tensor.is_host_accessible() );
            EXPECT_FALSE( host_tensor.is_device_accessible() );

            EXPECT_EQ( host_tensor.getDataType(), TensorDataType::UINT32 );
            EXPECT_EQ( host_tensor.getDataTypeName(), "UINT32" );
        }
    }

    // ====================================================================
    // Device tests: verify CUDA memory resources and device-only types (skipped if no CUDA)
    // ====================================================================

    TEST_F( TensorMemoryPropertiesTest, Device_HostCompatibleDataTypes_CudaMemoryResources ) {
        if ( !has_cuda_device_ ) {
            GTEST_SKIP() << "CUDA device not available. Skipping CUDA memory-resource checks.";
        }

        std::vector<int64_t> shape = { 2, 3 };

        // FP32 across CUDA resources
        {
            Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource> cuda_tensor( Device::Cuda( 0 ), shape );
            Tensor<TensorDataType::FP32, Compute::CudaPinnedMemoryResource> pinned_tensor( Device::Cuda( 0 ), shape );
            Tensor<TensorDataType::FP32, Compute::CudaManagedMemoryResource> managed_tensor( Device::Cuda( 0 ), shape );

            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );

            EXPECT_TRUE( pinned_tensor.is_host_accessible() );
            EXPECT_TRUE( pinned_tensor.is_device_accessible() );

            EXPECT_TRUE( managed_tensor.is_host_accessible() );
            EXPECT_TRUE( managed_tensor.is_device_accessible() );

            EXPECT_EQ( cuda_tensor.getDataType(), TensorDataType::FP32 );
            EXPECT_EQ( pinned_tensor.getDataType(), TensorDataType::FP32 );
            EXPECT_EQ( managed_tensor.getDataType(), TensorDataType::FP32 );

            EXPECT_EQ( cuda_tensor.getDataTypeName(), "FP32" );
            EXPECT_EQ( pinned_tensor.getDataTypeName(), "FP32" );
            EXPECT_EQ( managed_tensor.getDataTypeName(), "FP32" );
        }

        // Host-compatible integer/unsigned types on CUDA resources: spot-check device resource behavior
        {
            Tensor<TensorDataType::INT16, Compute::CudaDeviceMemoryResource> cuda_tensor( Device::Cuda( 0 ), shape );
            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );
            EXPECT_EQ( cuda_tensor.getDataType(), TensorDataType::INT16 );
        }

        {
            Tensor<TensorDataType::INT32, Compute::CudaDeviceMemoryResource> cuda_tensor( Device::Cuda( 0 ), shape );
            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );
            EXPECT_EQ( cuda_tensor.getDataType(), TensorDataType::INT32 );
        }

        {
            Tensor<TensorDataType::UINT16, Compute::CudaDeviceMemoryResource> cuda_tensor( Device::Cuda( 0 ), shape );
            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );
            EXPECT_EQ( cuda_tensor.getDataType(), TensorDataType::UINT16 );
        }

        {
            Tensor<TensorDataType::UINT32, Compute::CudaDeviceMemoryResource> cuda_tensor( Device::Cuda( 0 ), shape );
            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );
            EXPECT_EQ( cuda_tensor.getDataType(), TensorDataType::UINT32 );
        }
    }

    TEST_F( TensorMemoryPropertiesTest, DeviceOnly_DataTypes_CudaMemoryResources ) {
        if ( !has_cuda_device_ ) {
            GTEST_SKIP() << "CUDA device not available. Skipping device-only type checks.";
        }

        std::vector<int64_t> shape = { 2, 3 };

        // FP16 (device-only half precision)
        {
            Tensor<TensorDataType::FP16, Compute::CudaDeviceMemoryResource> cuda_tensor( Device::Cuda( 0 ), shape );
            Tensor<TensorDataType::FP16, Compute::CudaPinnedMemoryResource> pinned_tensor( Device::Cuda( 0 ), shape );
            Tensor<TensorDataType::FP16, Compute::CudaManagedMemoryResource> managed_tensor( Device::Cuda( 0 ), shape );

            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );
            EXPECT_TRUE( pinned_tensor.is_host_accessible() );
            EXPECT_TRUE( pinned_tensor.is_device_accessible() );
            EXPECT_TRUE( managed_tensor.is_host_accessible() );
            EXPECT_TRUE( managed_tensor.is_device_accessible() );

            EXPECT_EQ( cuda_tensor.getDataType(), TensorDataType::FP16 );
            EXPECT_EQ( pinned_tensor.getDataType(), TensorDataType::FP16 );
            EXPECT_EQ( managed_tensor.getDataType(), TensorDataType::FP16 );
        }

        // BF16 (device-only brain float)
        {
            Tensor<TensorDataType::BF16, Compute::CudaDeviceMemoryResource> cuda_tensor( Device::Cuda( 0 ), shape );
            Tensor<TensorDataType::BF16, Compute::CudaPinnedMemoryResource> pinned_tensor( Device::Cuda( 0 ), shape );
            Tensor<TensorDataType::BF16, Compute::CudaManagedMemoryResource> managed_tensor( Device::Cuda( 0 ), shape );

            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );
            EXPECT_TRUE( pinned_tensor.is_host_accessible() );
            EXPECT_TRUE( pinned_tensor.is_device_accessible() );
            EXPECT_TRUE( managed_tensor.is_host_accessible() );
            EXPECT_TRUE( managed_tensor.is_device_accessible() );

            EXPECT_EQ( cuda_tensor.getDataType(), TensorDataType::BF16 );
            EXPECT_EQ( pinned_tensor.getDataType(), TensorDataType::BF16 );
            EXPECT_EQ( managed_tensor.getDataType(), TensorDataType::BF16 );
        }

        // FP8 variants
        {
            Tensor<TensorDataType::FP8_E4M3, Compute::CudaDeviceMemoryResource> cuda_tensor( Device::Cuda( 0 ), shape );
            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );
            EXPECT_EQ( cuda_tensor.getDataType(), TensorDataType::FP8_E4M3 );
            EXPECT_EQ( cuda_tensor.getDataTypeName(), "FP8_E4M3" );
        }

        {
            Tensor<TensorDataType::FP8_E5M2, Compute::CudaDeviceMemoryResource> cuda_tensor( Device::Cuda( 0 ), shape );
            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );
            EXPECT_EQ( cuda_tensor.getDataType(), TensorDataType::FP8_E5M2 );
            EXPECT_EQ( cuda_tensor.getDataTypeName(), "FP8_E5M2" );
        }
    }

    // ====================================================================
    // Static class-method checks and compile-time verifications (no CUDA runtime required)
    // ====================================================================

    TEST_F( TensorMemoryPropertiesTest, StaticMethods_AbstractDataTypes ) {
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

    TEST_F( TensorMemoryPropertiesTest, CompileTimeProperties_AbstractDataTypes ) {
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
    // Type alias tests: split host aliases and device aliases
    // ====================================================================

    TEST_F( TensorMemoryPropertiesTest, HostOnly_TypeAliasProperties ) {
        std::vector<int64_t> shape = { 2, 3 };

        // Host aliases for various types
        HostTensor<TensorDataType::FP32> host_fp32_tensor( Device::Cpu(), shape );
        EXPECT_TRUE( host_fp32_tensor.is_host_accessible() );
        EXPECT_FALSE( host_fp32_tensor.is_device_accessible() );
        EXPECT_EQ( host_fp32_tensor.getDataType(), TensorDataType::FP32 );

        HostTensor<TensorDataType::INT32> host_int32_tensor( Device::Cpu(), shape );
        EXPECT_TRUE( host_int32_tensor.is_host_accessible() );
        EXPECT_FALSE( host_int32_tensor.is_device_accessible() );
        EXPECT_EQ( host_int32_tensor.getDataType(), TensorDataType::INT32 );

        HostTensor<TensorDataType::UINT16> host_uint16_tensor( Device::Cpu(), shape );
        EXPECT_TRUE( host_uint16_tensor.is_host_accessible() );
        EXPECT_FALSE( host_uint16_tensor.is_device_accessible() );
        EXPECT_EQ( host_uint16_tensor.getDataType(), TensorDataType::UINT16 );
    }

    TEST_F( TensorMemoryPropertiesTest, Device_TypeAliasProperties ) {
        if ( !has_cuda_device_ ) {
            GTEST_SKIP() << "CUDA device not available. Skipping device alias checks.";
        }

        std::vector<int64_t> shape = { 2, 3 };

        DeviceTensor<TensorDataType::FP32> device_fp32_tensor( Device::Cuda( 0 ), shape );
        PinnedTensor<TensorDataType::FP32> pinned_fp32_tensor( Device::Cuda( 0 ), shape );
        UniversalTensor<TensorDataType::FP32> universal_fp32_tensor( Device::Cuda( 0 ), shape );

        EXPECT_FALSE( device_fp32_tensor.is_host_accessible() );
        EXPECT_TRUE( device_fp32_tensor.is_device_accessible() );
        EXPECT_TRUE( pinned_fp32_tensor.is_host_accessible() );
        EXPECT_TRUE( pinned_fp32_tensor.is_device_accessible() );
        EXPECT_TRUE( universal_fp32_tensor.is_host_accessible() );
        EXPECT_TRUE( universal_fp32_tensor.is_device_accessible() );

        DeviceTensor<TensorDataType::FP16> device_fp16_tensor( Device::Cuda( 0 ), shape );
        PinnedTensor<TensorDataType::FP16> pinned_fp16_tensor( Device::Cuda( 0 ), shape );
        UniversalTensor<TensorDataType::FP16> universal_fp16_tensor( Device::Cuda( 0 ), shape );

        EXPECT_FALSE( device_fp16_tensor.is_host_accessible() );
        EXPECT_TRUE( device_fp16_tensor.is_device_accessible() );
        EXPECT_TRUE( pinned_fp16_tensor.is_host_accessible() );
        EXPECT_TRUE( pinned_fp16_tensor.is_device_accessible() );
        EXPECT_TRUE( universal_fp16_tensor.is_host_accessible() );
        EXPECT_TRUE( universal_fp16_tensor.is_device_accessible() );
    }

    // ====================================================================
    // Property consistency across operations: split host and device parts
    // ====================================================================

    TEST_F( TensorMemoryPropertiesTest, Host_PropertyConsistencyAcrossOperations ) {
        std::vector<int64_t> shape = { 3, 3 };

        // Test with FP32 on CPU
        {
            Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> host_tensor( Device::Cpu(), shape );

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
    }

    TEST_F( TensorMemoryPropertiesTest, Device_PropertyConsistencyAcrossOperations ) {
        if ( !has_cuda_device_ ) {
            GTEST_SKIP() << "CUDA device not available. Skipping device operation consistency tests.";
        }

        std::vector<int64_t> shape = { 3, 3 };

        // Test with INT32 on CUDA device memory
        {
            Tensor<TensorDataType::INT32, Compute::CudaDeviceMemoryResource> cuda_tensor( Device::Cuda( 0 ), shape );

            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );

            cuda_tensor.reshape( { 9 } );
            EXPECT_FALSE( cuda_tensor.is_host_accessible() );
            EXPECT_TRUE( cuda_tensor.is_device_accessible() );
        }

        // Test with UINT16 pinned memory
        {
            Tensor<TensorDataType::UINT16, Compute::CudaPinnedMemoryResource> pinned_tensor( Device::Cuda( 0 ), shape );

            EXPECT_TRUE( pinned_tensor.is_host_accessible() );
            EXPECT_TRUE( pinned_tensor.is_device_accessible() );

            pinned_tensor.reshape( { 9 } );
            EXPECT_TRUE( pinned_tensor.is_host_accessible() );
            EXPECT_TRUE( pinned_tensor.is_device_accessible() );
        }

        // Device-only FP16 with managed memory
        {
            Tensor<TensorDataType::FP16, Compute::CudaManagedMemoryResource> managed_tensor( Device::Cuda( 0 ), shape );

            EXPECT_TRUE( managed_tensor.is_host_accessible() );
            EXPECT_TRUE( managed_tensor.is_device_accessible() );

            managed_tensor.reshape( { 9 } );
            EXPECT_TRUE( managed_tensor.is_host_accessible() );
            EXPECT_TRUE( managed_tensor.is_device_accessible() );
        }
    }

    // ====================================================================
    // Data type information: split host-only and device-only checks
    // ====================================================================

    TEST_F( TensorMemoryPropertiesTest, HostOnly_DataTypeInformation ) {
        std::vector<int64_t> shape = { 2, 2 };

        {
            Tensor<TensorDataType::INT8, Compute::CpuMemoryResource> tensor( Device::Cpu(), shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::INT8 );
            EXPECT_EQ( tensor.getDataTypeName(), "INT8" );
        }

        {
            Tensor<TensorDataType::INT16, Compute::CpuMemoryResource> tensor( Device::Cpu(), shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::INT16 );
            EXPECT_EQ( tensor.getDataTypeName(), "INT16" );
        }

        {
            Tensor<TensorDataType::INT32, Compute::CpuMemoryResource> tensor( Device::Cpu(), shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::INT32 );
            EXPECT_EQ( tensor.getDataTypeName(), "INT32" );
        }

        {
            Tensor<TensorDataType::UINT8, Compute::CpuMemoryResource> tensor( Device::Cpu(), shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::UINT8 );
            EXPECT_EQ( tensor.getDataTypeName(), "UINT8" );
        }

        {
            Tensor<TensorDataType::UINT16, Compute::CpuMemoryResource> tensor( Device::Cpu(), shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::UINT16 );
            EXPECT_EQ( tensor.getDataTypeName(), "UINT16" );
        }

        {
            Tensor<TensorDataType::UINT32, Compute::CpuMemoryResource> tensor( Device::Cpu(), shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::UINT32 );
            EXPECT_EQ( tensor.getDataTypeName(), "UINT32" );
        }

        // FP32 CPU check (host-compatible)
        {
            Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( Device::Cpu(), shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::FP32 );
            EXPECT_EQ( tensor.getDataTypeName(), "FP32" );
        }
    }

    TEST_F( TensorMemoryPropertiesTest, Device_DataTypeInformation ) {
        if ( !has_cuda_device_ ) {
            GTEST_SKIP() << "CUDA device not available. Skipping device data-type information tests.";
        }

        std::vector<int64_t> shape = { 2, 2 };

        {
            Tensor<TensorDataType::FP16, Compute::CudaDeviceMemoryResource> tensor( Device::Cuda( 0 ), shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::FP16 );
            EXPECT_EQ( tensor.getDataTypeName(), "FP16" );
        }

        {
            Tensor<TensorDataType::BF16, Compute::CudaDeviceMemoryResource> tensor( Device::Cuda( 0 ), shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::BF16 );
            EXPECT_EQ( tensor.getDataTypeName(), "BF16" );
        }

        {
            Tensor<TensorDataType::FP8_E4M3, Compute::CudaDeviceMemoryResource> tensor( Device::Cuda( 0 ), shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::FP8_E4M3 );
            EXPECT_EQ( tensor.getDataTypeName(), "FP8_E4M3" );
        }

        {
            Tensor<TensorDataType::FP8_E5M2, Compute::CudaDeviceMemoryResource> tensor( Device::Cuda( 0 ), shape );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::FP8_E5M2 );
            EXPECT_EQ( tensor.getDataTypeName(), "FP8_E5M2" );
        }
    }

    // ====================================================================
    // TensorDataTypeTraits and concept validations (compile-time)
    // ====================================================================

    TEST_F( TensorMemoryPropertiesTest, TensorDataTypeTraitsValidation ) {
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

    TEST_F( TensorMemoryPropertiesTest, ConceptValidation ) {
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
    // Device context compatibility: host part and device part
    // ====================================================================

    TEST_F( TensorMemoryPropertiesTest, Host_DeviceContextCompatibility ) {
        std::vector<int64_t> shape = { 2, 2 };

        {
            Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> cpu_tensor( Device::Cpu(), shape );
            EXPECT_EQ( cpu_tensor.getDeviceType(), Compute::DeviceType::Cpu );
        }
    }

    TEST_F( TensorMemoryPropertiesTest, Device_DeviceContextCompatibility ) {
        if ( !has_cuda_device_ ) {
            GTEST_SKIP() << "CUDA device not available. Skipping device context compatibility tests.";
        }

        std::vector<int64_t> shape = { 2, 2 };

        {
            Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource> cuda_tensor( Device::Cuda( 0 ), shape );
            EXPECT_EQ( cuda_tensor.getDeviceType(), Compute::DeviceType::Cuda );
        }

        {
            Tensor<TensorDataType::FP16, Compute::CudaDeviceMemoryResource> cuda_fp16_tensor( Device::Cuda( 0 ), shape );
            EXPECT_EQ( cuda_fp16_tensor.getDeviceType(), Compute::DeviceType::Cuda );
        }

        {
            Tensor<TensorDataType::INT32, Compute::CudaPinnedMemoryResource> pinned_tensor( Device::Cuda( 0 ), shape );
            EXPECT_EQ( pinned_tensor.getDeviceType(), Compute::DeviceType::Cuda );
        }

        {
            Tensor<TensorDataType::BF16, Compute::CudaManagedMemoryResource> managed_tensor( Device::Cuda( 0 ), shape );
            EXPECT_EQ( managed_tensor.getDeviceType(), Compute::DeviceType::Cuda );
        }
    }

    // ====================================================================
    // Memory resource property verification (compile-time)
    // ====================================================================

    TEST_F( TensorMemoryPropertiesTest, MemoryResourcePropertiesVerification ) {
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