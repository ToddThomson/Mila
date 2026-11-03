/**
 * @file Tensor.MemoryProperties.Cuda.cpp
 * @brief CUDA accessibility / storage tests for Tensor over device memory resources.
 *
 * Device companion to Tensor.MemoryProperties.cpp. Covers the host/device
 * accessibility matrix for CUDA device/pinned/managed resources, device-only data
 * types, getDeviceType == Cuda, getStorageSize for a device tensor, and the
 * isValidTensor / memory-resource compile-time rows that reference CUDA types.
 * GPU-local: compiled only under MILA_ENABLE_CUDA, skipped when no device present.
 */

#include <gtest/gtest.h>

import Mila;

namespace Mila::Tests::Dnn::Tensors
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TensorMemoryPropertiesCudaTests : public testing::Test {
    protected:
        void SetUp() override {
            has_cuda_ = DeviceRegistry::instance().hasDeviceType( DeviceType::Cuda );
        }

        bool has_cuda_ = false;
    };

    // ====================================================================
    // Accessibility matrix (compile-time + runtime)
    // ====================================================================

    TEST_F( TensorMemoryPropertiesCudaTests, Accessibility_ResourceMatrix ) {
        static_assert( !Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>::is_host_accessible() );
        static_assert( Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>::is_device_accessible() );
        static_assert( Tensor<TensorDataType::FP32, CudaPinnedMemoryResource>::is_host_accessible() );
        static_assert( Tensor<TensorDataType::FP32, CudaPinnedMemoryResource>::is_device_accessible() );
        static_assert( Tensor<TensorDataType::FP32, CudaManagedMemoryResource>::is_host_accessible() );
        static_assert( Tensor<TensorDataType::FP32, CudaManagedMemoryResource>::is_device_accessible() );

        static_assert( CudaDeviceMemoryResource::device_type == DeviceType::Cuda );
        static_assert( CudaPinnedMemoryResource::device_type == DeviceType::Cuda );
        static_assert( CudaManagedMemoryResource::device_type == DeviceType::Cuda );

        SUCCEED();
    }

    // ====================================================================
    // Runtime accessibility, device-only types, storage, device type
    // ====================================================================

    TEST_F( TensorMemoryPropertiesCudaTests, DeviceOnlyTypes_AcrossResources ) {
        if ( !has_cuda_ ) {
            GTEST_SKIP() << "CUDA device not available.";
        }

        shape_t shape = { 2, 3 };

        Tensor<TensorDataType::FP16, CudaDeviceMemoryResource> device( Device::Cuda( 0 ), shape );
        Tensor<TensorDataType::BF16, CudaManagedMemoryResource> managed( Device::Cuda( 0 ), shape );
        Tensor<TensorDataType::FP8_E4M3, CudaPinnedMemoryResource> pinned( Device::Cuda( 0 ), shape );

        EXPECT_FALSE( device.is_host_accessible() );
        EXPECT_TRUE( device.is_device_accessible() );
        EXPECT_EQ( device.getDataTypeName(), "FP16" );
        EXPECT_EQ( device.getDeviceType(), DeviceType::Cuda );

        EXPECT_TRUE( managed.is_host_accessible() );
        EXPECT_TRUE( managed.is_device_accessible() );

        EXPECT_TRUE( pinned.is_host_accessible() );
        EXPECT_EQ( pinned.getDataTypeName(), "FP8_E4M3" );
    }

    TEST_F( TensorMemoryPropertiesCudaTests, StorageSize_DeviceTensor ) {
        if ( !has_cuda_ ) {
            GTEST_SKIP() << "CUDA device not available.";
        }

        Tensor<TensorDataType::FP16, CudaDeviceMemoryResource> tensor( Device::Cuda( 0 ), shape_t{ 2, 3 } );
        EXPECT_EQ( tensor.elementSize(), 2u );
        EXPECT_EQ( tensor.getStorageSize(), 6u * 2u );
    }

    // ====================================================================
    // Concept rows referencing CUDA resources (compile-time)
    // ====================================================================

    TEST_F( TensorMemoryPropertiesCudaTests, ConceptValidity_CudaRows ) {
        static_assert( isValidTensor<TensorDataType::FP16, CudaDeviceMemoryResource> );
        static_assert( isValidTensor<TensorDataType::BF16, CudaManagedMemoryResource> );
        static_assert( isValidTensor<TensorDataType::FP8_E4M3, CudaPinnedMemoryResource> );

        SUCCEED();
    }
}
