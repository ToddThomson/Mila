#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cuda_runtime.h>

import Mila;

namespace Dnn::Tensors::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TensorConstructionTest : public testing::Test {
    protected:
        TensorConstructionTest() {}

        void SetUp() override {
            // Check if CUDA devices are available
            int device_count;
            cudaError_t error = cudaGetDeviceCount( &device_count );
            has_cuda_ = (error == cudaSuccess && device_count > 0);
        }

        bool has_cuda_ = false;
    };

    // ====================================================================
    // Device Name Constructor Tests
    // ====================================================================

    TEST_F( TensorConstructionTest, ConstructorWithDeviceName ) {
        std::vector<size_t> shape = { 2, 3 };

        Tensor<TensorDataType::FP32, CpuMemoryResource> cpu_tensor( "CPU", shape );

        EXPECT_FALSE( cpu_tensor.empty() );
        EXPECT_EQ( cpu_tensor.size(), 6 );
        EXPECT_EQ( cpu_tensor.rank(), 2 );
        EXPECT_EQ( cpu_tensor.shape(), shape );

        if (has_cuda_) {
            Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> cuda_tensor( "CUDA:0", shape );

            EXPECT_FALSE( cuda_tensor.empty() );
            EXPECT_EQ( cuda_tensor.size(), 6 );
            EXPECT_EQ( cuda_tensor.rank(), 2 );
            EXPECT_EQ( cuda_tensor.shape(), shape );
        }
        else {
            GTEST_SKIP() << "CUDA device not available for CUDA tensor test";
        }
    }

    TEST_F( TensorConstructionTest, ConstructorWithInvalidDeviceName ) {
        std::vector<size_t> shape = { 2, 3 };

        EXPECT_THROW(
            (Tensor<TensorDataType::FP32, CpuMemoryResource>( "INVALID_DEVICE", shape )),
            std::invalid_argument
        );
    }

    TEST_F( TensorConstructionTest, ConstructorWithEmptyDeviceName ) {
        std::vector<size_t> shape = { 2, 3 };

        EXPECT_THROW(
            (Tensor<TensorDataType::FP32, CpuMemoryResource>( "", shape )),
            std::invalid_argument
        );
    }

    TEST_F( TensorConstructionTest, ConstructorWithMismatchedDeviceAndMemoryResource ) {
        std::vector<size_t> shape = { 2, 3 };

        // CPU device name with CUDA memory resource should fail
        EXPECT_THROW(
            (Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( "CPU", shape )),
            std::runtime_error
        );

        if (has_cuda_) {
            // CUDA device name with CPU memory resource should fail
            EXPECT_THROW(
                (Tensor<TensorDataType::FP32, CpuMemoryResource>( "CUDA:0", shape )),
                std::runtime_error
            );
        }
    }

    // ====================================================================
    // Shape Constructor Tests
    // ====================================================================

    TEST_F( TensorConstructionTest, ConstructorWithShape ) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA device not available for this test";
        }

        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> tensor( "CUDA:0", shape );

        EXPECT_FALSE( tensor.empty() );
        EXPECT_EQ( tensor.size(), 6 );
        EXPECT_EQ( tensor.rank(), 2 );
        EXPECT_EQ( tensor.shape(), shape );
    }

    TEST_F( TensorConstructionTest, ConstructorWithEmptyShape ) {
        std::vector<size_t> shape = {};
        Tensor<TensorDataType::FP32, CpuMemoryResource> tensor( "CPU", shape );

        EXPECT_FALSE( tensor.empty() );  // Scalars are NOT empty
        EXPECT_EQ( tensor.size(), 1 );   // Scalars have size 1
        EXPECT_EQ( tensor.rank(), 0 );   // Scalars have rank 0
        EXPECT_EQ( tensor.strides().size(), 0 );
        EXPECT_EQ( tensor.shape(), shape );
        EXPECT_TRUE( tensor.isScalar() );
    }

    TEST_F( TensorConstructionTest, ConstructorWithZeroSizeShape ) {
        std::vector<size_t> shape = { 0 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> tensor( "CPU", shape );

        EXPECT_TRUE( tensor.empty() );   // Zero-size tensors ARE empty
        EXPECT_EQ( tensor.size(), 0 );
        EXPECT_EQ( tensor.rank(), 1 );   // Still has rank 1
        EXPECT_FALSE( tensor.isScalar() ); // Not a scalar
    }

    // ====================================================================
    // Move Constructor Tests
    // ====================================================================

    TEST_F( TensorConstructionTest, MoveConstructor ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> original( "CPU", shape );
        std::string original_uid = original.getUId();

        Tensor<TensorDataType::FP32, CpuMemoryResource> moved( std::move( original ) );

        EXPECT_EQ( moved.shape(), shape );
        EXPECT_EQ( moved.size(), 6 );
        EXPECT_EQ( moved.getUId(), original_uid );

        // Original should be in moved-from state
        EXPECT_TRUE( original.empty() );
        EXPECT_EQ( original.size(), 0 );
    }

    TEST_F( TensorConstructionTest, MoveConstructor_PreservesData ) {
        std::vector<size_t> shape = { 2, 2 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> original( "CPU", shape );

        Tensor<TensorDataType::FP32, CpuMemoryResource> moved( std::move( original ) );

        // Verify tensor structure is preserved in moved tensor
        EXPECT_EQ( moved.shape(), shape );
        EXPECT_EQ( moved.size(), 4 );
        EXPECT_FALSE( moved.empty() );
    }

    // ====================================================================
    // Move Assignment Tests
    // ====================================================================

    TEST_F( TensorConstructionTest, MoveAssignment ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> original( "CPU", shape );
        std::string original_uid = original.getUId();

        Tensor<TensorDataType::FP32, CpuMemoryResource> moved( "CPU", {} );
        moved = std::move( original );

        EXPECT_EQ( moved.shape(), shape );
        EXPECT_EQ( moved.size(), 6 );
        EXPECT_EQ( moved.getUId(), original_uid );

        // Original should be in moved-from state
        EXPECT_TRUE( original.empty() );
        EXPECT_EQ( original.size(), 0 );
    }

    TEST_F( TensorConstructionTest, MoveAssignment_SelfMove ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> tensor( "CPU", shape );
        std::string original_uid = tensor.getUId();

        tensor = std::move( tensor );

        // Self-move should leave tensor in valid state
        EXPECT_EQ( tensor.getUId(), original_uid );
        EXPECT_EQ( tensor.shape(), shape );
        EXPECT_EQ( tensor.size(), 6 );
    }

    // ====================================================================
    // Deleted Operations Tests
    // ====================================================================

    TEST_F( TensorConstructionTest, CopyOperationsAreDeleted ) {
        // Copy operations are deleted at compile time
        // These lines would not compile if uncommented:

        // Tensor<TensorDataType::FP32, CpuMemoryResource> tensor1("CPU", {2, 3});
        // Tensor<TensorDataType::FP32, CpuMemoryResource> tensor2(tensor1);  // Should not compile
        // Tensor<TensorDataType::FP32, CpuMemoryResource> tensor3("CPU", {});
        // tensor3 = tensor1;  // Should not compile

        SUCCEED();
    }

    // ====================================================================
    // Unique ID Generation Tests
    // ====================================================================

    TEST_F( TensorConstructionTest, UniqueIdGeneration ) {
        Tensor<TensorDataType::FP32, CpuMemoryResource> tensor1( "CPU", {} );
        Tensor<TensorDataType::FP32, CpuMemoryResource> tensor2( "CPU", {} );

        // Each tensor should have a unique ID
        EXPECT_NE( tensor1.getUId(), tensor2.getUId() );
    }

    TEST_F( TensorConstructionTest, UniqueIdGenerationWithShape ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> tensor1( "CPU", shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> tensor2( "CPU", shape );

        // Each tensor should have unique ID even with same shape
        EXPECT_NE( tensor1.getUId(), tensor2.getUId() );
    }

    TEST_F( TensorConstructionTest, UniqueIdGenerationAfterMove ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> original( "CPU", shape );
        std::string original_uid = original.getUId();

        Tensor<TensorDataType::FP32, CpuMemoryResource> moved( std::move( original ) );

        // UID should transfer with move
        EXPECT_EQ( moved.getUId(), original_uid );

        // Creating new tensor should get different UID
        Tensor<TensorDataType::FP32, CpuMemoryResource> new_tensor( "CPU", shape );
        EXPECT_NE( new_tensor.getUId(), original_uid );
    }

    // ====================================================================
    // Construction with Different Memory Resources
    // ====================================================================

    TEST_F( TensorConstructionTest, ConstructWithDifferentMemoryResources ) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA device not available for this test";
        }

        std::vector<size_t> shape = { 2, 3 };

        // Test construction with various memory resource types
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_tensor( "CPU", shape );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> cuda_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::FP32, CudaPinnedMemoryResource> pinned_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::FP32, CudaManagedMemoryResource> managed_tensor( "CUDA:0", shape );

        // Verify all tensors have correct properties
        EXPECT_EQ( host_tensor.shape(), shape );
        EXPECT_EQ( cuda_tensor.shape(), shape );
        EXPECT_EQ( pinned_tensor.shape(), shape );
        EXPECT_EQ( managed_tensor.shape(), shape );

        EXPECT_EQ( host_tensor.size(), 6 );
        EXPECT_EQ( cuda_tensor.size(), 6 );
        EXPECT_EQ( pinned_tensor.size(), 6 );
        EXPECT_EQ( managed_tensor.size(), 6 );

        // Verify memory accessibility properties
        EXPECT_TRUE( host_tensor.is_host_accessible() );
        EXPECT_FALSE( host_tensor.is_device_accessible() );

        EXPECT_FALSE( cuda_tensor.is_host_accessible() );
        EXPECT_TRUE( cuda_tensor.is_device_accessible() );

        EXPECT_TRUE( pinned_tensor.is_host_accessible() );
        EXPECT_TRUE( pinned_tensor.is_device_accessible() );

        EXPECT_TRUE( managed_tensor.is_host_accessible() );
        EXPECT_TRUE( managed_tensor.is_device_accessible() );

        // Verify all have unique UIDs
        EXPECT_NE( host_tensor.getUId(), cuda_tensor.getUId() );
        EXPECT_NE( host_tensor.getUId(), pinned_tensor.getUId() );
        EXPECT_NE( host_tensor.getUId(), managed_tensor.getUId() );
        EXPECT_NE( cuda_tensor.getUId(), pinned_tensor.getUId() );
    }

    // ====================================================================
    // Device Name Validation Tests
    // ====================================================================

    TEST_F( TensorConstructionTest, CudaMemoryResourceRequiresCudaDevice ) {
        std::vector<size_t> shape = { 2, 3 };

        // CPU device with CUDA memory resource should fail
        EXPECT_THROW(
            (Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>( "CPU", shape )),
            std::runtime_error
        );

        // CPU device with CUDA pinned memory resource should fail
        EXPECT_THROW(
            (Tensor<TensorDataType::FP32, CudaPinnedMemoryResource>( "CPU", shape )),
            std::runtime_error
        );

        // CPU device with CUDA managed memory resource should fail
        EXPECT_THROW(
            (Tensor<TensorDataType::FP32, CudaManagedMemoryResource>( "CPU", shape )),
            std::runtime_error
        );
    }

    TEST_F( TensorConstructionTest, CpuMemoryResourceRequiresCpuDevice ) {
        std::vector<size_t> shape = { 2, 3 };

        if (has_cuda_) {
            EXPECT_THROW(
                (Tensor<TensorDataType::FP32, CpuMemoryResource>( "CUDA:0", shape )),
                std::runtime_error
            );
        }
    }

    // ====================================================================
    // Constructor Edge Cases
    // ====================================================================

    TEST_F( TensorConstructionTest, ConstructorWithLargeShape ) {
        std::vector<size_t> large_shape = { 100, 200, 50 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> tensor( "CPU", large_shape );

        EXPECT_EQ( tensor.shape(), large_shape );
        EXPECT_EQ( tensor.size(), 1000000 );  // 100 * 200 * 50
        EXPECT_EQ( tensor.rank(), 3 );
        EXPECT_FALSE( tensor.empty() );
    }

    TEST_F( TensorConstructionTest, ConstructorWithSingleDimension ) {
        std::vector<size_t> single_dim = { 42 };
        Tensor<TensorDataType::INT32, CpuMemoryResource> tensor( "CPU", single_dim );

        EXPECT_EQ( tensor.shape(), single_dim );
        EXPECT_EQ( tensor.size(), 42 );
        EXPECT_EQ( tensor.rank(), 1 );
        EXPECT_FALSE( tensor.empty() );
    }

    TEST_F( TensorConstructionTest, ConstructorWithZeroInitialization ) {
        std::vector<size_t> shape = { 3, 4 };

        // Test tensor initialization (should allocate memory)
        Tensor<TensorDataType::FP32, CpuMemoryResource> float_tensor( "CPU", shape );
        Tensor<TensorDataType::INT32, CpuMemoryResource> int_tensor( "CPU", shape );

        EXPECT_EQ( float_tensor.shape(), shape );
        EXPECT_EQ( int_tensor.shape(), shape );
        EXPECT_EQ( float_tensor.size(), 12 );
        EXPECT_EQ( int_tensor.size(), 12 );
        EXPECT_FALSE( float_tensor.empty() );
        EXPECT_FALSE( int_tensor.empty() );
    }

    // ====================================================================
    // Constructor Tests with All CPU-Supported Data Types
    // ====================================================================

    TEST_F( TensorConstructionTest, ConstructorWithAllCpuSupportedDataTypes ) {
        std::vector<size_t> shape = { 2, 3 };

        // Test all CPU-supported data types
        Tensor<TensorDataType::FP32, CpuMemoryResource> fp32_tensor( "CPU", shape );
        Tensor<TensorDataType::INT8, CpuMemoryResource> int8_tensor( "CPU", shape );
        Tensor<TensorDataType::INT16, CpuMemoryResource> int16_tensor( "CPU", shape );
        Tensor<TensorDataType::INT32, CpuMemoryResource> int32_tensor( "CPU", shape );
        Tensor<TensorDataType::UINT8, CpuMemoryResource> uint8_tensor( "CPU", shape );
        Tensor<TensorDataType::UINT16, CpuMemoryResource> uint16_tensor( "CPU", shape );
        Tensor<TensorDataType::UINT32, CpuMemoryResource> uint32_tensor( "CPU", shape );

        // Verify all tensors have correct data types
        EXPECT_EQ( fp32_tensor.getDataType(), TensorDataType::FP32 );
        EXPECT_EQ( int8_tensor.getDataType(), TensorDataType::INT8 );
        EXPECT_EQ( int16_tensor.getDataType(), TensorDataType::INT16 );
        EXPECT_EQ( int32_tensor.getDataType(), TensorDataType::INT32 );
        EXPECT_EQ( uint8_tensor.getDataType(), TensorDataType::UINT8 );
        EXPECT_EQ( uint16_tensor.getDataType(), TensorDataType::UINT16 );
        EXPECT_EQ( uint32_tensor.getDataType(), TensorDataType::UINT32 );

        // Verify type names
        EXPECT_EQ( fp32_tensor.getDataTypeName(), "FP32" );
        EXPECT_EQ( int8_tensor.getDataTypeName(), "INT8" );
        EXPECT_EQ( int16_tensor.getDataTypeName(), "INT16" );
        EXPECT_EQ( int32_tensor.getDataTypeName(), "INT32" );
        EXPECT_EQ( uint8_tensor.getDataTypeName(), "UINT8" );
        EXPECT_EQ( uint16_tensor.getDataTypeName(), "UINT16" );
        EXPECT_EQ( uint32_tensor.getDataTypeName(), "UINT32" );

        // All should have same shape and size
        EXPECT_EQ( fp32_tensor.size(), 6 );
        EXPECT_EQ( int8_tensor.size(), 6 );
        EXPECT_EQ( int16_tensor.size(), 6 );
        EXPECT_EQ( int32_tensor.size(), 6 );
        EXPECT_EQ( uint8_tensor.size(), 6 );
        EXPECT_EQ( uint16_tensor.size(), 6 );
        EXPECT_EQ( uint32_tensor.size(), 6 );

        // Verify memory accessibility
        EXPECT_TRUE( fp32_tensor.is_host_accessible() );
        EXPECT_FALSE( fp32_tensor.is_device_accessible() );
        EXPECT_TRUE( int32_tensor.is_host_accessible() );
        EXPECT_FALSE( int32_tensor.is_device_accessible() );
    }

    // ====================================================================
    // Constructor Tests with All CUDA-Supported Data Types
    // ====================================================================

    TEST_F( TensorConstructionTest, ConstructorWithAllCudaSupportedDataTypes ) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA device not available for this test";
        }

        std::vector<size_t> shape = { 2, 3 };

        // Test all CUDA-supported data types
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> fp32_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::FP16, CudaDeviceMemoryResource> fp16_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::BF16, CudaDeviceMemoryResource> bf16_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::FP8_E4M3, CudaDeviceMemoryResource> fp8_e4m3_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::FP8_E5M2, CudaDeviceMemoryResource> fp8_e5m2_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::INT8, CudaDeviceMemoryResource> int8_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::INT16, CudaDeviceMemoryResource> int16_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::INT32, CudaDeviceMemoryResource> int32_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::UINT8, CudaDeviceMemoryResource> uint8_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::UINT16, CudaDeviceMemoryResource> uint16_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::UINT32, CudaDeviceMemoryResource> uint32_tensor( "CUDA:0", shape );

        // Verify all tensors have correct data types
        EXPECT_EQ( fp32_tensor.getDataType(), TensorDataType::FP32 );
        EXPECT_EQ( fp16_tensor.getDataType(), TensorDataType::FP16 );
        EXPECT_EQ( bf16_tensor.getDataType(), TensorDataType::BF16 );
        EXPECT_EQ( fp8_e4m3_tensor.getDataType(), TensorDataType::FP8_E4M3 );
        EXPECT_EQ( fp8_e5m2_tensor.getDataType(), TensorDataType::FP8_E5M2 );
        EXPECT_EQ( int8_tensor.getDataType(), TensorDataType::INT8 );
        EXPECT_EQ( int16_tensor.getDataType(), TensorDataType::INT16 );
        EXPECT_EQ( int32_tensor.getDataType(), TensorDataType::INT32 );
        EXPECT_EQ( uint8_tensor.getDataType(), TensorDataType::UINT8 );
        EXPECT_EQ( uint16_tensor.getDataType(), TensorDataType::UINT16 );
        EXPECT_EQ( uint32_tensor.getDataType(), TensorDataType::UINT32 );

        // Verify type names
        EXPECT_EQ( fp32_tensor.getDataTypeName(), "FP32" );
        EXPECT_EQ( fp16_tensor.getDataTypeName(), "FP16" );
        EXPECT_EQ( bf16_tensor.getDataTypeName(), "BF16" );
        EXPECT_EQ( fp8_e4m3_tensor.getDataTypeName(), "FP8_E4M3" );
        EXPECT_EQ( fp8_e5m2_tensor.getDataTypeName(), "FP8_E5M2" );
        EXPECT_EQ( int8_tensor.getDataTypeName(), "INT8" );
        EXPECT_EQ( int16_tensor.getDataTypeName(), "INT16" );
        EXPECT_EQ( int32_tensor.getDataTypeName(), "INT32" );
        EXPECT_EQ( uint8_tensor.getDataTypeName(), "UINT8" );
        EXPECT_EQ( uint16_tensor.getDataTypeName(), "UINT16" );
        EXPECT_EQ( uint32_tensor.getDataTypeName(), "UINT32" );

        // All should have same shape and size
        EXPECT_EQ( fp32_tensor.size(), 6 );
        EXPECT_EQ( fp16_tensor.size(), 6 );
        EXPECT_EQ( bf16_tensor.size(), 6 );
        EXPECT_EQ( fp8_e4m3_tensor.size(), 6 );
        EXPECT_EQ( fp8_e5m2_tensor.size(), 6 );
        EXPECT_EQ( int8_tensor.size(), 6 );
        EXPECT_EQ( int16_tensor.size(), 6 );
        EXPECT_EQ( int32_tensor.size(), 6 );
        EXPECT_EQ( uint8_tensor.size(), 6 );
        EXPECT_EQ( uint16_tensor.size(), 6 );
        EXPECT_EQ( uint32_tensor.size(), 6 );

        // Verify device accessibility
        EXPECT_FALSE( fp32_tensor.is_host_accessible() );
        EXPECT_TRUE( fp32_tensor.is_device_accessible() );
        EXPECT_FALSE( fp16_tensor.is_host_accessible() );
        EXPECT_TRUE( fp16_tensor.is_device_accessible() );

    }

    // ====================================================================
    // Constructor Tests with Managed Memory
    // ====================================================================

    TEST_F( TensorConstructionTest, ConstructorWithManagedMemoryAllDataTypes ) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA device not available for this test";
        }

        std::vector<size_t> shape = { 2, 3 };

        // Test selection of CUDA data types with managed memory
        Tensor<TensorDataType::FP32, CudaManagedMemoryResource> fp32_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::FP16, CudaManagedMemoryResource> fp16_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::BF16, CudaManagedMemoryResource> bf16_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::INT8, CudaManagedMemoryResource> int8_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::INT32, CudaManagedMemoryResource> int32_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::UINT32, CudaManagedMemoryResource> uint32_tensor( "CUDA:0", shape );

        // Verify managed memory accessibility
        EXPECT_TRUE( fp32_tensor.is_host_accessible() );
        EXPECT_TRUE( fp32_tensor.is_device_accessible() );
        EXPECT_TRUE( fp16_tensor.is_host_accessible() );
        EXPECT_TRUE( fp16_tensor.is_device_accessible() );
        EXPECT_TRUE( bf16_tensor.is_host_accessible() );
        EXPECT_TRUE( bf16_tensor.is_device_accessible() );
        EXPECT_TRUE( int8_tensor.is_host_accessible() );
        EXPECT_TRUE( int8_tensor.is_device_accessible() );
        EXPECT_TRUE( int32_tensor.is_host_accessible() );
        EXPECT_TRUE( int32_tensor.is_device_accessible() );
        EXPECT_TRUE( uint32_tensor.is_host_accessible() );
        EXPECT_TRUE( uint32_tensor.is_device_accessible() );

        // Verify data types
        EXPECT_EQ( fp32_tensor.getDataType(), TensorDataType::FP32 );
        EXPECT_EQ( fp16_tensor.getDataType(), TensorDataType::FP16 );
        EXPECT_EQ( bf16_tensor.getDataType(), TensorDataType::BF16 );
        EXPECT_EQ( int8_tensor.getDataType(), TensorDataType::INT8 );
        EXPECT_EQ( int32_tensor.getDataType(), TensorDataType::INT32 );
        EXPECT_EQ( uint32_tensor.getDataType(), TensorDataType::UINT32 );

    }

    // ====================================================================
    // Constructor Tests with Pinned Memory
    // ====================================================================

    TEST_F( TensorConstructionTest, ConstructorWithPinnedMemoryAllDataTypes ) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA device not available for this test";
        }

        std::vector<size_t> shape = { 2, 3 };

        // Test selection of CUDA data types with pinned memory
        Tensor<TensorDataType::FP32, CudaPinnedMemoryResource> fp32_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::FP16, CudaPinnedMemoryResource> fp16_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::BF16, CudaPinnedMemoryResource> bf16_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::FP8_E4M3, CudaPinnedMemoryResource> fp8_e4m3_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::INT8, CudaPinnedMemoryResource> int8_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::INT32, CudaPinnedMemoryResource> int32_tensor( "CUDA:0", shape );
        Tensor<TensorDataType::UINT16, CudaPinnedMemoryResource> uint16_tensor( "CUDA:0", shape );

        // Verify pinned memory accessibility
        EXPECT_TRUE( fp32_tensor.is_host_accessible() );
        EXPECT_TRUE( fp32_tensor.is_device_accessible() );
        EXPECT_TRUE( fp16_tensor.is_host_accessible() );
        EXPECT_TRUE( fp16_tensor.is_device_accessible() );
        EXPECT_TRUE( bf16_tensor.is_host_accessible() );
        EXPECT_TRUE( bf16_tensor.is_device_accessible() );
        EXPECT_TRUE( fp8_e4m3_tensor.is_host_accessible() );
        EXPECT_TRUE( fp8_e4m3_tensor.is_device_accessible() );
        EXPECT_TRUE( int8_tensor.is_host_accessible() );
        EXPECT_TRUE( int8_tensor.is_device_accessible() );
        EXPECT_TRUE( int32_tensor.is_host_accessible() );
        EXPECT_TRUE( int32_tensor.is_device_accessible() );
        EXPECT_TRUE( uint16_tensor.is_host_accessible() );
        EXPECT_TRUE( uint16_tensor.is_device_accessible() );

        // Verify data types
        EXPECT_EQ( fp32_tensor.getDataType(), TensorDataType::FP32 );
        EXPECT_EQ( fp16_tensor.getDataType(), TensorDataType::FP16 );
        EXPECT_EQ( bf16_tensor.getDataType(), TensorDataType::BF16 );
        EXPECT_EQ( fp8_e4m3_tensor.getDataType(), TensorDataType::FP8_E4M3 );
        EXPECT_EQ( int8_tensor.getDataType(), TensorDataType::INT8 );
        EXPECT_EQ( int32_tensor.getDataType(), TensorDataType::INT32 );
        EXPECT_EQ( uint16_tensor.getDataType(), TensorDataType::UINT16 );

    }

    

    // ====================================================================
    // Type Constraint Validation Tests
    // ====================================================================

    TEST_F( TensorConstructionTest, TypeConstraintValidation_ManagedMemoryCompatibility ) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA device not available for this test";
        }

        // Host-compatible types with managed memory
        {
            Tensor<TensorDataType::FP32, CudaManagedMemoryResource> tensor( "CUDA:0", { 2, 3 } );
            EXPECT_TRUE( tensor.is_host_accessible() );
            EXPECT_TRUE( tensor.is_device_accessible() );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::FP32 );
        }

        // Device-specific types with managed memory
        {
            Tensor<TensorDataType::FP16, CudaManagedMemoryResource> tensor( "CUDA:0", { 2, 3 } );
            EXPECT_TRUE( tensor.is_host_accessible() );
            EXPECT_TRUE( tensor.is_device_accessible() );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::FP16 );
        }

        // Device-only types with managed memory
        {
            Tensor<TensorDataType::FP8_E4M3, CudaManagedMemoryResource> tensor( "CUDA:0", { 2, 3 } );
            EXPECT_TRUE( tensor.is_host_accessible() );
            EXPECT_TRUE( tensor.is_device_accessible() );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::FP8_E4M3 );
        }
    }

    TEST_F( TensorConstructionTest, TypeConstraintValidation_PinnedMemoryCompatibility ) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA device not available for this test";
        }

        // Standard floating-point types
        {
            Tensor<TensorDataType::FP32, CudaPinnedMemoryResource> tensor( "CUDA:0", { 2, 3 } );
            EXPECT_TRUE( tensor.is_host_accessible() );
            EXPECT_TRUE( tensor.is_device_accessible() );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::FP32 );
        }

        // Half precision types
        {
            Tensor<TensorDataType::FP16, CudaPinnedMemoryResource> tensor( "CUDA:0", { 2, 3 } );
            EXPECT_TRUE( tensor.is_host_accessible() );
            EXPECT_TRUE( tensor.is_device_accessible() );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::FP16 );
        }

        // Integer types
        {
            Tensor<TensorDataType::INT32, CudaPinnedMemoryResource> tensor( "CUDA:0", { 2, 3 } );
            EXPECT_TRUE( tensor.is_host_accessible() );
            EXPECT_TRUE( tensor.is_device_accessible() );
            EXPECT_EQ( tensor.getDataType(), TensorDataType::INT32 );

        }
    }

    TEST_F( TensorConstructionTest, TypeConstraintValidation_DataTypeTraitsConsistency ) {
        // Verify data type names are consistent
        if (has_cuda_) {
            Tensor<TensorDataType::FP16, CudaDeviceMemoryResource> cuda_tensor( "CUDA:0", { 2, 3 } );
            Tensor<TensorDataType::FP16, CudaManagedMemoryResource> managed_tensor( "CUDA:0", { 2, 3 } );

            EXPECT_EQ( cuda_tensor.getDataTypeName(), managed_tensor.getDataTypeName() );
            EXPECT_EQ( cuda_tensor.getDataTypeName(), "FP16" );
        }

        // Verify device-only type characteristics
        if (has_cuda_) {
            Tensor<TensorDataType::FP8_E4M3, CudaDeviceMemoryResource> tensor( "CUDA:0", { 2, 3 } );
            EXPECT_EQ( tensor.getDataTypeName(), "FP8_E4M3" );
        }
    }

    TEST_F( TensorConstructionTest, TypeConstraintValidation_MemoryResourceInheritance ) {
        // Verify all memory resources inherit from base
        static_assert(std::is_base_of_v<MemoryResource, CpuMemoryResource>);
        static_assert(std::is_base_of_v<MemoryResource, CudaDeviceMemoryResource>);
        static_assert(std::is_base_of_v<MemoryResource, CudaManagedMemoryResource>);
        static_assert(std::is_base_of_v<MemoryResource, CudaPinnedMemoryResource>);

        // Verify memory accessibility properties
        static_assert(CpuMemoryResource::is_host_accessible == true);
        static_assert(CpuMemoryResource::is_device_accessible == false);

        static_assert(CudaDeviceMemoryResource::is_host_accessible == false);
        static_assert(CudaDeviceMemoryResource::is_device_accessible == true);

        static_assert(CudaManagedMemoryResource::is_host_accessible == true);
        static_assert(CudaManagedMemoryResource::is_device_accessible == true);

        static_assert(CudaPinnedMemoryResource::is_host_accessible == true);
        static_assert(CudaPinnedMemoryResource::is_device_accessible == true);
    }

    TEST_F( TensorConstructionTest, TypeConstraintValidation_ConceptValidation ) {
        // Valid combinations satisfy the concept
        static_assert(isValidTensor<TensorDataType::FP32, CpuMemoryResource>);
        static_assert(isValidTensor<TensorDataType::FP16, CudaDeviceMemoryResource>);
        static_assert(isValidTensor<TensorDataType::FP8_E4M3, CudaDeviceMemoryResource>);
        static_assert(isValidTensor<TensorDataType::INT32, CpuMemoryResource>);
        static_assert(isValidTensor<TensorDataType::FP32, CudaManagedMemoryResource>);
        static_assert(isValidTensor<TensorDataType::FP16, CudaPinnedMemoryResource>);

        // Device-only types require device-accessible memory
        static_assert(isValidTensor<TensorDataType::FP8_E4M3, CudaManagedMemoryResource>);
        static_assert(isValidTensor<TensorDataType::FP8_E5M2, CudaPinnedMemoryResource>);
    }
}