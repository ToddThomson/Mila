/**
 * @file Tensor.Constructors.Cuda.cpp
 * @brief CUDA construction tests for Tensor over device memory resources.
 *
 * Device companion to Tensor.Constructors.cpp. Covers DeviceTensor / PinnedTensor /
 * UniversalTensor construction, the device/memory-resource mismatch throw (both
 * directions, each of which requires a CUDA type), the CUDA/managed/pinned
 * data-type surface, and the compile-time isValidTensor concept rows. GPU-local:
 * compiled only under MILA_ENABLE_CUDA and skipped at runtime when no device is
 * present.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <new>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Tensors
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TensorConstructorsCudaTests : public testing::Test {
    protected:
        void SetUp() override {
            has_cuda_ = DeviceRegistry::instance().hasDeviceType( DeviceType::Cuda );
        }

        bool has_cuda_ = false;
    };

    // ====================================================================
    // A. Device construction across memory resources
    // ====================================================================

    TEST_F( TensorConstructorsCudaTests, Construct_AcrossDeviceMemoryResources ) {
        if ( !has_cuda_ ) {
            GTEST_SKIP() << "CUDA device not available.";
        }

        shape_t shape = { 2, 3 };

        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> device_tensor( Device::Cuda( 0 ), shape );
        Tensor<TensorDataType::FP32, CudaPinnedMemoryResource> pinned_tensor( Device::Cuda( 0 ), shape );
        Tensor<TensorDataType::FP32, CudaManagedMemoryResource> managed_tensor( Device::Cuda( 0 ), shape );

        EXPECT_EQ( device_tensor.size(), 6 );
        EXPECT_EQ( pinned_tensor.size(), 6 );
        EXPECT_EQ( managed_tensor.size(), 6 );

        EXPECT_EQ( device_tensor.getDeviceType(), DeviceType::Cuda );

        EXPECT_FALSE( device_tensor.is_host_accessible() );
        EXPECT_TRUE( device_tensor.is_device_accessible() );
        EXPECT_TRUE( pinned_tensor.is_host_accessible() );
        EXPECT_TRUE( pinned_tensor.is_device_accessible() );
        EXPECT_TRUE( managed_tensor.is_host_accessible() );
        EXPECT_TRUE( managed_tensor.is_device_accessible() );

        EXPECT_NE( device_tensor.getUId(), pinned_tensor.getUId() );
        EXPECT_NE( device_tensor.getUId(), managed_tensor.getUId() );
    }

    TEST_F( TensorConstructorsCudaTests, Construct_DeviceAliases ) {
        if ( !has_cuda_ ) {
            GTEST_SKIP() << "CUDA device not available.";
        }

        shape_t shape = { 2, 3 };

        DeviceTensor<TensorDataType::FP32> device_tensor( Device::Cuda( 0 ), shape );
        PinnedTensor<TensorDataType::FP32> pinned_tensor( Device::Cuda( 0 ), shape );
        UniversalTensor<TensorDataType::FP32> universal_tensor( Device::Cuda( 0 ), shape );

        EXPECT_FALSE( device_tensor.is_host_accessible() );
        EXPECT_TRUE( pinned_tensor.is_host_accessible() );
        EXPECT_TRUE( universal_tensor.is_host_accessible() );
    }

    // ====================================================================
    // A(neg). validateDeviceId throws on device/memory-resource mismatch
    // ====================================================================

    TEST_F( TensorConstructorsCudaTests, Construct_ThrowsWhenCpuDeviceWithCudaResource ) {
        shape_t shape = { 2, 3 };

        EXPECT_THROW(
            (Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( Device::Cpu(), shape )),
            std::runtime_error );
        EXPECT_THROW(
            (Tensor<TensorDataType::FP32, CudaPinnedMemoryResource>( Device::Cpu(), shape )),
            std::runtime_error );
        EXPECT_THROW(
            (Tensor<TensorDataType::FP32, CudaManagedMemoryResource>( Device::Cpu(), shape )),
            std::runtime_error );
    }

    TEST_F( TensorConstructorsCudaTests, Construct_ThrowsWhenCudaDeviceWithCpuResource ) {
        if ( !has_cuda_ ) {
            GTEST_SKIP() << "CUDA device not available.";
        }

        EXPECT_THROW(
            (Tensor<TensorDataType::FP32, CpuMemoryResource>( Device::Cuda( 0 ), shape_t{ 2, 3 } )),
            std::runtime_error );
    }

    // ====================================================================
    // J. CUDA-supported data types (incl. device-only)
    // ====================================================================

    TEST_F( TensorConstructorsCudaTests, DataTypes_AllCudaSupported ) {
        if ( !has_cuda_ ) {
            GTEST_SKIP() << "CUDA device not available.";
        }

        shape_t shape = { 2, 3 };

        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> fp32( Device::Cuda( 0 ), shape );
        Tensor<TensorDataType::FP16, CudaDeviceMemoryResource> fp16( Device::Cuda( 0 ), shape );
        Tensor<TensorDataType::BF16, CudaDeviceMemoryResource> bf16( Device::Cuda( 0 ), shape );
        Tensor<TensorDataType::FP8_E4M3, CudaDeviceMemoryResource> fp8_e4m3( Device::Cuda( 0 ), shape );
        Tensor<TensorDataType::FP8_E5M2, CudaDeviceMemoryResource> fp8_e5m2( Device::Cuda( 0 ), shape );
        Tensor<TensorDataType::INT32, CudaDeviceMemoryResource> int32( Device::Cuda( 0 ), shape );

        EXPECT_EQ( fp32.getDataTypeName(), "FP32" );
        EXPECT_EQ( fp16.getDataTypeName(), "FP16" );
        EXPECT_EQ( bf16.getDataTypeName(), "BF16" );
        EXPECT_EQ( fp8_e4m3.getDataTypeName(), "FP8_E4M3" );
        EXPECT_EQ( fp8_e5m2.getDataTypeName(), "FP8_E5M2" );
        EXPECT_EQ( int32.getDataTypeName(), "INT32" );

        EXPECT_FALSE( fp16.is_host_accessible() );
        EXPECT_TRUE( fp16.is_device_accessible() );
    }

    // ====================================================================
    // K. A failed allocation fails cleanly
    // ====================================================================

    // Larger than any card and than the host memory a single-GPU process spills into, so the
    // allocation fails however many devices the process sees. Requested from the resource
    // directly: TensorBuffer refuses a tensor this large before it reaches cudaMalloc.
    constexpr std::size_t kUnallocatableBytes = std::size_t{ 1 } << 42;

    TEST_F( TensorConstructorsCudaTests, FailedDeviceAllocation_LeavesNoCudaErrorBehind ) {
        if ( !has_cuda_ ) {
            GTEST_SKIP() << "CUDA device not available.";
        }

        CudaDeviceMemoryResource resource( 0 );

        EXPECT_THROW( static_cast<void>( resource.allocate( kUnallocatableBytes ) ), std::bad_alloc );

        // Every later check of CUDA's last error -- a kernel launch, a copy -- would otherwise
        // report this allocation's failure as its own.
        EXPECT_EQ( cudaGetLastError(), cudaSuccess );
    }

    TEST_F( TensorConstructorsCudaTests, FailedPinnedAllocation_LeavesNoCudaErrorBehind ) {
        if ( !has_cuda_ ) {
            GTEST_SKIP() << "CUDA device not available.";
        }

        CudaPinnedMemoryResource resource( 0 );

        EXPECT_THROW( static_cast<void>( resource.allocate( kUnallocatableBytes ) ), std::bad_alloc );
        EXPECT_EQ( cudaGetLastError(), cudaSuccess );
    }

    // A model too large for its device fails this way: parameters already allocated are freed
    // while the allocation failure propagates, and a throw from that free ends the process.
    TEST_F( TensorConstructorsCudaTests, FailedDeviceAllocation_PropagatesWhileTensorsAreFreed ) {
        if ( !has_cuda_ ) {
            GTEST_SKIP() << "CUDA device not available.";
        }

        const auto allocateWhileATensorIsLive = []
        {
            Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> live( Device::Cuda( 0 ), shape_t{ 2, 3 } );
            CudaDeviceMemoryResource resource( 0 );
            static_cast<void>( resource.allocate( kUnallocatableBytes ) );
        };

        EXPECT_THROW( allocateWhileATensorIsLive(), std::bad_alloc );

        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> after( Device::Cuda( 0 ), shape_t{ 2, 3 } );

        EXPECT_EQ( after.size(), 6 );
    }

    // ====================================================================
    // Compile-time validity rows (isValidTensor concept)
    // ====================================================================

    TEST_F( TensorConstructorsCudaTests, ConceptValidity_DeviceCombinations ) {
        static_assert( isValidTensor<TensorDataType::FP32, CudaDeviceMemoryResource> );
        static_assert( isValidTensor<TensorDataType::FP16, CudaDeviceMemoryResource> );
        static_assert( isValidTensor<TensorDataType::FP8_E4M3, CudaManagedMemoryResource> );
        static_assert( isValidTensor<TensorDataType::FP8_E5M2, CudaPinnedMemoryResource> );

        SUCCEED();
    }
}
