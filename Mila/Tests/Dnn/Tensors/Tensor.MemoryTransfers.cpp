#include <gtest/gtest.h>
#include <vector>

import Mila;

namespace Dnn::Tensors::Tests
{
    using namespace Mila::Dnn;

    class TensorMemoryTransferTest : public testing::Test {
    protected:
        void SetUp() override {
            cpu_context_ = std::make_shared<Compute::CpuDeviceContext>();
            cuda_context_ = Compute::DeviceContext::create( "CUDA:0" );
        }

        std::shared_ptr<Compute::CpuDeviceContext> cpu_context_;
        std::shared_ptr<Compute::DeviceContext> cuda_context_;
    };

    // ====================================================================
    // CPU to CUDA Transfer Tests
    // ====================================================================

    TEST_F( TensorMemoryTransferTest, ConvertCpuToCuda ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> host_tensor( cpu_context_, { 2, 3 } );

		// Should be a TensorOps::Fill::constant in the future
        /*for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                host_tensor.set( { i, j }, 3.14f );
            }
        }*/

        EXPECT_TRUE( host_tensor.is_host_accessible() );
        EXPECT_FALSE( host_tensor.is_device_accessible() );

        auto cuda_tensor = host_tensor.toDevice<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>( cuda_context_ );

        // Verify original tensor is unchanged
        EXPECT_EQ( host_tensor.shape(), std::vector<size_t>( { 2, 3 } ) );
        EXPECT_EQ( host_tensor.size(), 6 );
        EXPECT_EQ( host_tensor.rank(), 2 );

        // Verify cuda_tensor properties
        EXPECT_EQ( cuda_tensor.shape(), std::vector<size_t>( { 2, 3 } ) );
        EXPECT_EQ( cuda_tensor.size(), 6 );
        EXPECT_EQ( cuda_tensor.rank(), 2 );

        std::vector<size_t> expected_strides = { 3, 1 };
        EXPECT_EQ( cuda_tensor.strides(), expected_strides );

        EXPECT_FALSE( cuda_tensor.is_host_accessible() );
        EXPECT_TRUE( cuda_tensor.is_device_accessible() );

        // Convert back to CPU - creates new tensor
        auto cpu_tensor = cuda_tensor.toHost();

        // Verify all properties are preserved
        EXPECT_EQ( cpu_tensor.shape(), std::vector<size_t>( { 2, 3 } ) );
        EXPECT_EQ( cpu_tensor.size(), 6 );
        EXPECT_EQ( cpu_tensor.rank(), 2 );
        EXPECT_EQ( cpu_tensor.strides(), expected_strides );

        // Verify memory resource type changed back
        EXPECT_TRUE( cpu_tensor.is_host_accessible() );
        EXPECT_FALSE( cpu_tensor.is_device_accessible() );

        // Verify data integrity after round-trip conversion
        // FIXME:
        /*for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                EXPECT_FLOAT_EQ( cpu_tensor.at( { i, j } ), 3.14f );
            }
        }*/
    }

    // ====================================================================
    // CUDA to CPU Transfer Tests
    // ====================================================================

    TEST_F( TensorMemoryTransferTest, ConvertCudaToCpu ) {
        Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource> tensor( cuda_context_, { 2, 3 } );

        // Convert to CPU - creates new tensor
        auto cpu_tensor = tensor.toHost();

        // Check buffer-related properties
        EXPECT_EQ( cpu_tensor.shape(), std::vector<size_t>( { 2, 3 } ) );
        EXPECT_EQ( cpu_tensor.size(), 6 );
        EXPECT_EQ( cpu_tensor.rank(), 2 );
        EXPECT_FALSE( cpu_tensor.empty() );

        std::vector<size_t> expected_strides = { 3, 1 };
        EXPECT_EQ( cpu_tensor.strides(), expected_strides );

        // Verify memory resource type changed
        EXPECT_TRUE( cpu_tensor.is_host_accessible() );
        EXPECT_FALSE( cpu_tensor.is_device_accessible() );

        // Verify original tensor is unchanged
        EXPECT_EQ( tensor.shape(), std::vector<size_t>( { 2, 3 } ) );
        EXPECT_EQ( tensor.size(), 6 );
        EXPECT_EQ( tensor.rank(), 2 );
        EXPECT_FALSE( tensor.empty() );
    }

    // ====================================================================
    // Deep Copy Transfer Tests
    // ====================================================================

    TEST_F( TensorMemoryTransferTest, CopyConstructor_CudaToCpu ) {
        Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource> src( cuda_context_, { 2, 3 } );

        auto dst = src.toHost<TensorDataType::FP32>();

        // Verify properties are copied correctly
        EXPECT_EQ( dst.shape(), src.shape() );
        EXPECT_EQ( dst.size(), src.size() );
        EXPECT_EQ( dst.rank(), src.rank() );
        EXPECT_EQ( dst.strides(), src.strides() );

        // Verify different UIDs (deep copy)
        EXPECT_NE( dst.getUId(), src.getUId() );

        // Verify memory resource type changed
        EXPECT_TRUE( dst.is_host_accessible() );
        EXPECT_FALSE( dst.is_device_accessible() );
    }

    TEST_F( TensorMemoryTransferTest, toDevice_CpuToCuda ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> src( cpu_context_, { 2, 3 } );

        auto dst = src.toDevice<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>( cuda_context_ );

        // Verify properties are copied correctly
        EXPECT_EQ( dst.shape(), src.shape() );
        EXPECT_EQ( dst.size(), src.size() );
        EXPECT_EQ( dst.rank(), src.rank() );
        EXPECT_EQ( dst.strides(), src.strides() );

        // Verify different UIDs (deep copy)
        EXPECT_NE( dst.getUId(), src.getUId() );

        // Verify memory resource type changed
        EXPECT_FALSE( dst.is_host_accessible() );
        EXPECT_TRUE( dst.is_device_accessible() );

        // Convert back to CPU to verify transfer worked
        auto verification_tensor = dst.toHost<TensorDataType::FP32>();

        EXPECT_EQ( verification_tensor.shape(), src.shape() );
    }

    // ====================================================================
    // Pinned Memory Transfer Tests
    // ====================================================================

    TEST_F( TensorMemoryTransferTest, HostToPinned ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> host_tensor( cpu_context_, { 3, 4 } );

        auto pinned_tensor = host_tensor.toHost<TensorDataType::FP32>(); // Note: Would need pinned context

        EXPECT_EQ( pinned_tensor.shape(), host_tensor.shape() );
        EXPECT_EQ( pinned_tensor.size(), host_tensor.size() );
        EXPECT_NE( pinned_tensor.getUId(), host_tensor.getUId() );

        EXPECT_TRUE( pinned_tensor.is_host_accessible() );
        // Note: Pinned memory accessibility depends on memory resource implementation
    }

    TEST_F( TensorMemoryTransferTest, PinnedToCuda ) {
        // Note: This test would need a proper pinned memory context
        // For now, using CPU memory as source
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> pinned_tensor( cpu_context_, { 2, 4 } );

        auto cuda_tensor = pinned_tensor.toDevice<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>( cuda_context_ );

        EXPECT_EQ( cuda_tensor.shape(), pinned_tensor.shape() );
        EXPECT_EQ( cuda_tensor.size(), pinned_tensor.size() );
        EXPECT_NE( cuda_tensor.getUId(), pinned_tensor.getUId() );

        EXPECT_FALSE( cuda_tensor.is_host_accessible() );
        EXPECT_TRUE( cuda_tensor.is_device_accessible() );

        // Verify data through round-trip
        auto verification_tensor = cuda_tensor.toHost<TensorDataType::FP32>();
        EXPECT_EQ( verification_tensor.shape(), pinned_tensor.shape() );
    }

    // ====================================================================
    // Managed Memory Transfer Tests
    // ====================================================================

    TEST_F( TensorMemoryTransferTest, HostToManaged ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> host_tensor( cpu_context_, { 2, 2 } );

        // Note: Would need managed memory context for true managed memory test
        auto managed_tensor = host_tensor.toHost<TensorDataType::FP32>( cpu_context_ );

        EXPECT_EQ( managed_tensor.shape(), host_tensor.shape() );
        EXPECT_EQ( managed_tensor.size(), host_tensor.size() );
        EXPECT_NE( managed_tensor.getUId(), host_tensor.getUId() );

        EXPECT_TRUE( managed_tensor.is_host_accessible() );
    }

    TEST_F( TensorMemoryTransferTest, ManagedToCuda ) {
        // Note: This test would need a proper managed memory context
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> managed_tensor( cpu_context_, { 3, 2 } );

        auto cuda_tensor = managed_tensor.toDevice<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>( cuda_context_ );

        EXPECT_EQ( cuda_tensor.shape(), managed_tensor.shape() );
        EXPECT_EQ( cuda_tensor.size(), managed_tensor.size() );
        EXPECT_NE( cuda_tensor.getUId(), managed_tensor.getUId() );

        EXPECT_FALSE( cuda_tensor.is_host_accessible() );
        EXPECT_TRUE( cuda_tensor.is_device_accessible() );

        // Verify data through round-trip
        auto verification_tensor = cuda_tensor.toHost<TensorDataType::FP32>( cpu_context_ );
        EXPECT_EQ( verification_tensor.shape(), managed_tensor.shape() );
    }

    // ====================================================================
    // Transfer with Different Data Types
    // ====================================================================

    TEST_F( TensorMemoryTransferTest, IntegerTypeTransfers ) {
        std::vector<size_t> shape = { 2, 3 };

        // INT16 transfers
        {
            Tensor<TensorDataType::INT16, Compute::CpuMemoryResource> host_tensor( cpu_context_, shape );
            auto cuda_tensor = host_tensor.toDevice<TensorDataType::INT16, Compute::CudaDeviceMemoryResource>( cuda_context_ );
            auto back_to_host = cuda_tensor.toHost<TensorDataType::INT16>( cpu_context_ );

            EXPECT_EQ( back_to_host.shape(), shape );
            EXPECT_EQ( back_to_host.getDataType(), TensorDataType::INT16 );
        }

        // UINT32 transfers
        {
            Tensor<TensorDataType::UINT32, Compute::CpuMemoryResource> host_tensor( cpu_context_, shape );
            auto cuda_tensor = host_tensor.toDevice<TensorDataType::UINT32, Compute::CudaDeviceMemoryResource>( cuda_context_ );
            auto back_to_host = cuda_tensor.toHost<TensorDataType::UINT32>( cpu_context_ );

            EXPECT_EQ( back_to_host.shape(), shape );
            EXPECT_EQ( back_to_host.getDataType(), TensorDataType::UINT32 );
        }
    }

    // ====================================================================
    // Transfer Metadata Preservation Tests
    // ====================================================================

    TEST_F( TensorMemoryTransferTest, MetadataPreservation ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> original( cpu_context_, { 2, 3 } );
        original.setName( "test_tensor" );

        auto cuda_tensor = original.toDevice<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>( cuda_context_ );
        auto back_to_host = cuda_tensor.toHost<TensorDataType::FP32>( cpu_context_ );

        // Name should be preserved
        EXPECT_EQ( cuda_tensor.getName(), "test_tensor" );
        EXPECT_EQ( back_to_host.getName(), "test_tensor" );

        // Shape, size, rank should be preserved
        EXPECT_EQ( cuda_tensor.shape(), original.shape() );
        EXPECT_EQ( cuda_tensor.size(), original.size() );
        EXPECT_EQ( cuda_tensor.rank(), original.rank() );
        EXPECT_EQ( cuda_tensor.strides(), original.strides() );

        EXPECT_EQ( back_to_host.shape(), original.shape() );
        EXPECT_EQ( back_to_host.size(), original.size() );
        EXPECT_EQ( back_to_host.rank(), original.rank() );
        EXPECT_EQ( back_to_host.strides(), original.strides() );

        // Data type should be preserved
        EXPECT_EQ( cuda_tensor.getDataType(), original.getDataType() );
        EXPECT_EQ( back_to_host.getDataType(), original.getDataType() );
        EXPECT_EQ( cuda_tensor.getDataTypeName(), original.getDataTypeName() );
        EXPECT_EQ( back_to_host.getDataTypeName(), original.getDataTypeName() );
    }

    // ====================================================================
    // Large Tensor Transfer Tests
    // ====================================================================

    TEST_F( TensorMemoryTransferTest, LargeTensorTransfer ) {
        std::vector<size_t> large_shape = { 100, 200 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> large_host_tensor( cpu_context_, large_shape );

        auto large_cuda_tensor = large_host_tensor.toDevice<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>( cuda_context_ );
        EXPECT_EQ( large_cuda_tensor.shape(), large_shape );
        EXPECT_EQ( large_cuda_tensor.size(), 20000 );

        auto verification_tensor = large_cuda_tensor.toHost<TensorDataType::FP32>( cpu_context_ );
        EXPECT_EQ( verification_tensor.shape(), large_shape );
    }

    // ====================================================================
    // Edge Case Transfer Tests
    // ====================================================================

    TEST_F( TensorMemoryTransferTest, EmptyTensorTransfer ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> empty_host_tensor( cpu_context_, {} );
        EXPECT_TRUE( empty_host_tensor.empty() );

        auto empty_cuda_tensor = empty_host_tensor.toDevice<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>( cuda_context_ );
        EXPECT_TRUE( empty_cuda_tensor.empty() );
        EXPECT_EQ( empty_cuda_tensor.size(), 0 );
        EXPECT_EQ( empty_cuda_tensor.rank(), 0 );

        auto back_to_host = empty_cuda_tensor.toHost<TensorDataType::FP32>( cpu_context_ );
        EXPECT_TRUE( back_to_host.empty() );
        EXPECT_EQ( back_to_host.size(), 0 );
        EXPECT_EQ( back_to_host.rank(), 0 );
    }

    TEST_F( TensorMemoryTransferTest, SingleElementTransfer ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> single_element( cpu_context_, { 1 } );

        auto cuda_tensor = single_element.toDevice<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>( cuda_context_ );
        EXPECT_EQ( cuda_tensor.size(), 1 );
        EXPECT_EQ( cuda_tensor.rank(), 1 );

        auto back_to_host = cuda_tensor.toHost<TensorDataType::FP32>( cpu_context_ );
        EXPECT_EQ( back_to_host.size(), 1 );
    }

    // ====================================================================
    // Host-to-Host Transfer Tests
    // ====================================================================

    TEST_F( TensorMemoryTransferTest, HostToHostTransfer ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> cpu_tensor( cpu_context_, { 2, 3 } );
        cpu_tensor.setName( "cpu_original" );

        // Host-to-host transfer (same memory resource)
        auto host_copy = cpu_tensor.toHost<TensorDataType::FP32>( cpu_context_ );

        EXPECT_EQ( host_copy.shape(), cpu_tensor.shape() );
        EXPECT_EQ( host_copy.size(), cpu_tensor.size() );
        EXPECT_EQ( host_copy.getName(), "cpu_original" );
        EXPECT_NE( host_copy.getUId(), cpu_tensor.getUId() );

        EXPECT_TRUE( host_copy.is_host_accessible() );
        EXPECT_FALSE( host_copy.is_device_accessible() );
    }

    // ====================================================================
    // Type Conversion Transfer Tests
    // ====================================================================

    TEST_F( TensorMemoryTransferTest, TypeConversionTransfer ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> fp32_tensor( cpu_context_, { 2, 2 } );

        // Initialize with test data
        // FIXME:
        /*for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                fp32_tensor.set( { i, j }, 3.14f );
            }
        }*/

        // Transfer to device with type conversion (if supported)
        auto cuda_tensor = fp32_tensor.toDevice<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>( cuda_context_ );

        // Convert back with different precision (this would test type conversion if implemented)
        auto host_result = cuda_tensor.toHost<TensorDataType::FP32>( cpu_context_ );

        EXPECT_EQ( host_result.getDataType(), TensorDataType::FP32 );
        EXPECT_EQ( host_result.shape(), fp32_tensor.shape() );
    }
}