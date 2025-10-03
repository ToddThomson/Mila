/**
 * @file TensorOps.Transfer.cpp
 * @brief Unit tests for TensorOps Transfer operations
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <random>
#include <algorithm>

import Mila;

namespace Dnn::Tensors::TensorOps::Tests
{
    using namespace Mila::Dnn;

    class TensorOpsTransferTest : public ::testing::Test {
    protected:
        void SetUp() override {
            cpu_context_ = std::make_shared<Compute::CpuDeviceContext>();

            // Initialize CUDA context if available
            try {
                cuda_context_ = Compute::DeviceContext::create( "CUDA:0" );
            }
            catch (const std::exception&) {
                cuda_context_ = nullptr;
                // CUDA not available - tests will skip appropriately
            }

            // Initialize random number generator for test data
            rng_.seed( 12345 ); // Fixed seed for reproducible tests
        }

        void TearDown() override {
            // Cleanup handled by RAII
        }

        /**
         * @brief Initialize tensor with deterministic test data using TensorOps::Fill
         *
         * Uses the fill operation to populate tensors with known test values,
         * enabling proper validation of copy operations.
         */
        template<TensorDataType TDataType>
        void initializeTensorWithTestData( Tensor<TDataType, Compute::CpuMemoryResource>& tensor ) {
            // Ensure tensor is properly allocated
            ASSERT_FALSE( tensor.empty() );
            ASSERT_EQ( tensor.getDataType(), TDataType );

            // Use appropriate host value type for the tensor data type
            using HostValueType = std::conditional_t<TensorDataTypeTraits<TDataType>::is_integer_type, int32_t, float>;
            //using HostValueType = host_value_t<TDataType>;

            if constexpr (TensorDataTypeTraits<TDataType>::is_integer_type) {
                // Fill integer tensors with sequential values: 1, 2, 3, ...
                HostValueType value = static_cast<HostValueType>(1);
                fill( tensor, value );
            }
            else {
                // Fill floating-point tensors with a known test value
                HostValueType value = static_cast<HostValueType>(3.14159f);
                fill( tensor, value );
            }
        }

        /**
         * @brief Initialize tensor with random test data for performance tests
         *
         * Uses random values to better simulate real-world usage patterns
         * while maintaining deterministic results through fixed seed.
         */
        template<TensorDataType TDataType>
        void initializeTensorWithRandomData( Tensor<TDataType, Compute::CpuMemoryResource>& tensor ) {
            ASSERT_FALSE( tensor.empty() );

            using HostValueType = std::conditional_t<TensorDataTypeTraits<TDataType>::is_integer_type, int32_t, float>;

            if constexpr (TensorDataTypeTraits<TDataType>::is_integer_type) {
                // Random integers in range [1, 100]
                std::uniform_int_distribution<int> dist( 1, 100 );
                HostValueType value = static_cast<HostValueType>(dist( rng_ ));
                fill( tensor, value );
            }
            else {
                // Random floats in range [0.0, 1.0]
                std::uniform_real_distribution<float> dist( 0.0f, 1.0f );
                HostValueType value = static_cast<HostValueType>(dist( rng_ ));
                fill( tensor, value );
            }
        }

        /**
         * @brief Verify tensor contains expected test data after copy
         *
         * For now, just validates tensor properties. In the future, could
         * be extended to validate actual data content if data access methods
         * are available.
         */
        template<TensorDataType TDataType>
        void verifyTensorData( const Tensor<TDataType, Compute::CpuMemoryResource>& tensor ) {
            EXPECT_FALSE( tensor.empty() );
            EXPECT_EQ( tensor.getDataType(), TDataType );
            // Future: Add actual data validation when tensor data access is available
        }

        std::shared_ptr<Compute::CpuDeviceContext> cpu_context_;
        std::shared_ptr<Compute::DeviceContext> cuda_context_;
        std::mt19937 rng_; // Random number generator for test data
    };

    // ====================================================================
    // CPU to CPU Copy Tests (Host-accessible to Host-accessible)
    // ====================================================================

    TEST_F( TensorOpsTransferTest, CpuToCpu_SameType_FP32 ) {
        const std::vector<size_t> shape = { 2, 3, 4 };

        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, shape );
        auto dst = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, shape );

        initializeTensorWithTestData( src );

        EXPECT_NO_THROW( copy( src, dst ) );

        // Verify destination tensor properties
        EXPECT_EQ( dst.shape(), src.shape() );
        EXPECT_EQ( dst.size(), src.size() );
        EXPECT_EQ( dst.getDataType(), src.getDataType() );
        verifyTensorData( dst );
    }

    TEST_F( TensorOpsTransferTest, CpuToCpu_TypeConversion_FP32_to_INT32 ) {
        const std::vector<size_t> shape = { 2, 3 };

        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, shape );
        auto dst = Tensor<TensorDataType::INT32, Compute::CpuMemoryResource>( cpu_context_, shape );

        initializeTensorWithTestData( src );

        EXPECT_NO_THROW( copy( src, dst ) );

        // Verify type conversion occurred
        EXPECT_EQ( dst.getDataType(), TensorDataType::INT32 );
        EXPECT_EQ( dst.shape(), src.shape() );
    }

    TEST_F( TensorOpsTransferTest, CpuToCpu_DifferentShapes_ThrowsException ) {
        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, { 2, 3 } );
        auto dst = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, { 3, 2 } );

        initializeTensorWithTestData( src );

        EXPECT_THROW( copy( src, dst ), std::invalid_argument );
    }

    TEST_F( TensorOpsTransferTest, CpuToCpu_EmptyTensors_NoOperation ) {
        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, {} );
        auto dst = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, {} );

        EXPECT_NO_THROW( copy( src, dst ) );
        EXPECT_TRUE( src.empty() );
        EXPECT_TRUE( dst.empty() );
    }

    // ====================================================================
    // CPU to CUDA Copy Tests (Host-accessible to Device-only)
    // ====================================================================

    TEST_F( TensorOpsTransferTest, CpuToCuda_SameType_FP32 ) {
        if (!cuda_context_) {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 2, 3 };

        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, shape );
        auto dst = Tensor<TensorDataType::FP32, Compute::CudaMemoryResource>( cuda_context_, shape );

        initializeTensorWithTestData( src );

        EXPECT_NO_THROW( copy( src, dst ) );

        // Verify transfer properties
        EXPECT_EQ( dst.shape(), src.shape() );
        EXPECT_EQ( dst.size(), src.size() );
        EXPECT_EQ( dst.getDataType(), src.getDataType() );
        EXPECT_FALSE( dst.is_host_accessible() );
        EXPECT_TRUE( dst.is_device_accessible() );
    }

    TEST_F( TensorOpsTransferTest, CpuToCuda_TypeConversion_FP32_to_FP16 ) {
        if (!cuda_context_) {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 3, 3 };

        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, shape );
        auto dst = Tensor<TensorDataType::FP16, Compute::CudaMemoryResource>( cuda_context_, shape );

        initializeTensorWithTestData( src );

        EXPECT_NO_THROW( copy( src, dst ) );

        // Verify conversion occurred
        EXPECT_EQ( dst.getDataType(), TensorDataType::FP16 );
        EXPECT_EQ( dst.shape(), src.shape() );
    }

    // ====================================================================
    // CUDA to CPU Copy Tests (Device-only to Host-accessible)
    // ====================================================================

    TEST_F( TensorOpsTransferTest, CudaToCpu_SameType_FP32 ) {
        if (!cuda_context_) {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 2, 2 };

        auto src = Tensor<TensorDataType::FP32, Compute::CudaMemoryResource>( cuda_context_, shape );
        auto dst = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, shape );

        EXPECT_NO_THROW( copy( src, dst ) );

        // Verify transfer back to CPU
        EXPECT_EQ( dst.shape(), src.shape() );
        EXPECT_EQ( dst.size(), src.size() );
        EXPECT_EQ( dst.getDataType(), src.getDataType() );
        EXPECT_TRUE( dst.is_host_accessible() );
        EXPECT_FALSE( dst.is_device_accessible() );
    }

    TEST_F( TensorOpsTransferTest, CudaToCpu_TypeConversion_FP16_to_FP32 ) {
        if (!cuda_context_) {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 4, 2 };

        auto src = Tensor<TensorDataType::FP16, Compute::CudaMemoryResource>( cuda_context_, shape );
        auto dst = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, shape );

        EXPECT_NO_THROW( copy( src, dst ) );

        // Verify conversion from FP16 to FP32
        EXPECT_EQ( dst.getDataType(), TensorDataType::FP32 );
        EXPECT_EQ( dst.shape(), src.shape() );
    }

    // ====================================================================
    // CUDA to CUDA Copy Tests (Device-only to Device-only, same type)
    // ====================================================================

    TEST_F( TensorOpsTransferTest, CudaToCuda_SameType_FP32 ) {
        if (!cuda_context_) {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 3, 3 };

        auto src = Tensor<TensorDataType::FP32, Compute::CudaMemoryResource>( cuda_context_, shape );
        auto dst = Tensor<TensorDataType::FP32, Compute::CudaMemoryResource>( cuda_context_, shape );

        EXPECT_NO_THROW( copy( src, dst ) );

        // Verify device-to-device copy
        EXPECT_EQ( dst.shape(), src.shape() );
        EXPECT_EQ( dst.size(), src.size() );
        EXPECT_EQ( dst.getDataType(), src.getDataType() );
        EXPECT_FALSE( dst.is_host_accessible() );
        EXPECT_TRUE( dst.is_device_accessible() );
    }

    TEST_F( TensorOpsTransferTest, CudaToCuda_TypeConversion_FP32_to_FP16 ) {
        if (!cuda_context_) {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 2, 4 };

        auto src = Tensor<TensorDataType::FP32, Compute::CudaMemoryResource>( cuda_context_, shape );
        auto dst = Tensor<TensorDataType::FP16, Compute::CudaMemoryResource>( cuda_context_, shape );

        EXPECT_NO_THROW( copy( src, dst ) );

        // Verify type conversion on device
        EXPECT_EQ( dst.getDataType(), TensorDataType::FP16 );
        EXPECT_EQ( dst.shape(), src.shape() );
    }

    // ====================================================================
    // Mixed Memory Resource Tests
    // ====================================================================

    TEST_F( TensorOpsTransferTest, CpuToPinned_SameType ) {
        if (!cuda_context_) {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 2, 3 };

        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, shape );
        auto dst = Tensor<TensorDataType::FP32, Compute::CudaPinnedMemoryResource>( cuda_context_, shape );

        initializeTensorWithTestData( src );

        EXPECT_NO_THROW( copy( src, dst ) );

        // Verify pinned memory properties
        EXPECT_EQ( dst.shape(), src.shape() );
        EXPECT_EQ( dst.getDataType(), src.getDataType() );
        EXPECT_TRUE( dst.is_host_accessible() );  // Pinned memory is host-accessible
    }

    TEST_F( TensorOpsTransferTest, PinnedToCuda_SameType ) {
        if (!cuda_context_) {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 3, 2 };

        auto src = Tensor<TensorDataType::FP32, Compute::CudaPinnedMemoryResource>( cuda_context_, shape );
        auto dst = Tensor<TensorDataType::FP32, Compute::CudaMemoryResource>( cuda_context_, shape );

        EXPECT_NO_THROW( copy( src, dst ) );

        // Verify transfer from pinned to device memory
        EXPECT_EQ( dst.shape(), src.shape() );
        EXPECT_EQ( dst.getDataType(), src.getDataType() );
        EXPECT_FALSE( dst.is_host_accessible() );
        EXPECT_TRUE( dst.is_device_accessible() );
    }

    // ====================================================================
    // Edge Case Tests
    // ====================================================================

    TEST_F( TensorOpsTransferTest, ZeroSizedTensor_NoOperation ) {
        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, { 0 } );
        auto dst = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, { 0 } );

        EXPECT_NO_THROW( copy( src, dst ) );
        EXPECT_EQ( src.size(), 0 );
        EXPECT_EQ( dst.size(), 0 );
    }

    TEST_F( TensorOpsTransferTest, SingleElementTensor ) {
        const std::vector<size_t> shape = { 1 };

        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, shape );
        auto dst = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, shape );

        initializeTensorWithTestData( src );

        EXPECT_NO_THROW( copy( src, dst ) );
        EXPECT_EQ( dst.size(), 1 );
        EXPECT_EQ( dst.shape(), src.shape() );
        verifyTensorData( dst );
    }

    TEST_F( TensorOpsTransferTest, LargeTensor_Performance ) {
        const std::vector<size_t> shape = { 100, 100 };

        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, shape );
        auto dst = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, shape );

        initializeTensorWithRandomData( src );

        // This test verifies that large tensor copies don't cause issues
        EXPECT_NO_THROW( copy( src, dst ) );
        EXPECT_EQ( dst.size(), 10000 );
        EXPECT_EQ( dst.shape(), src.shape() );
    }

    // ====================================================================
    // Multiple Data Type Tests
    // ====================================================================

    TEST_F( TensorOpsTransferTest, IntegerTypes_INT8_to_INT32 ) {
        const std::vector<size_t> shape = { 2, 2 };

        auto src = Tensor<TensorDataType::INT8, Compute::CpuMemoryResource>( cpu_context_, shape );
        auto dst = Tensor<TensorDataType::INT32, Compute::CpuMemoryResource>( cpu_context_, shape );

        initializeTensorWithTestData( src );

        EXPECT_NO_THROW( copy( src, dst ) );

        // Verify integer type conversion
        EXPECT_EQ( dst.getDataType(), TensorDataType::INT32 );
        EXPECT_EQ( dst.shape(), src.shape() );
    }

    TEST_F( TensorOpsTransferTest, UnsignedTypes_UINT8_to_UINT16 ) {
        const std::vector<size_t> shape = { 3, 2 };

        auto src = Tensor<TensorDataType::UINT8, Compute::CpuMemoryResource>( cpu_context_, shape );
        auto dst = Tensor<TensorDataType::UINT16, Compute::CpuMemoryResource>( cpu_context_, shape );

        initializeTensorWithTestData( src );

        EXPECT_NO_THROW( copy( src, dst ) );

        // Verify unsigned type conversion
        EXPECT_EQ( dst.getDataType(), TensorDataType::UINT16 );
        EXPECT_EQ( dst.shape(), src.shape() );
    }

    // ====================================================================
    // Round-trip Transfer Tests
    // ====================================================================

    TEST_F( TensorOpsTransferTest, RoundTrip_CpuToCudaToCpu ) {
        if (!cuda_context_) {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 2, 3 };

        auto original = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, shape );
        auto gpu_copy = Tensor<TensorDataType::FP32, Compute::CudaMemoryResource>( cuda_context_, shape );
        auto final_copy = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, shape );

        initializeTensorWithTestData( original );

        // Round-trip: CPU -> GPU -> CPU
        EXPECT_NO_THROW( copy( original, gpu_copy ) );
        EXPECT_NO_THROW( copy( gpu_copy, final_copy ) );

        // Verify round-trip preserves properties
        EXPECT_EQ( final_copy.shape(), original.shape() );
        EXPECT_EQ( final_copy.size(), original.size() );
        EXPECT_EQ( final_copy.getDataType(), original.getDataType() );
        verifyTensorData( final_copy );
    }

    TEST_F( TensorOpsTransferTest, RoundTrip_WithTypeConversion ) {
        if (!cuda_context_) {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 2, 2 };

        auto original = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, shape );
        auto fp16_gpu = Tensor<TensorDataType::FP16, Compute::CudaMemoryResource>( cuda_context_, shape );
        auto final_copy = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_context_, shape );

        initializeTensorWithTestData( original );

        // Round-trip with type conversion: FP32 -> FP16 -> FP32
        EXPECT_NO_THROW( copy( original, fp16_gpu ) );
        EXPECT_NO_THROW( copy( fp16_gpu, final_copy ) );

        // Verify final tensor has correct properties
        EXPECT_EQ( final_copy.shape(), original.shape() );
        EXPECT_EQ( final_copy.getDataType(), TensorDataType::FP32 );
        // Note: Some precision may be lost in FP32->FP16->FP32 conversion
    }

    // ====================================================================
    // Stress Tests
    // ====================================================================

    TEST_F( TensorOpsTransferTest, MultipleSequentialCopies ) {
        const std::vector<size_t> shape = { 10, 10 };
        const int num_copies = 5;

        std::vector<Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>> tensors;
        tensors.reserve( num_copies );

        // Create multiple tensors
        for (int i = 0; i < num_copies; ++i) {
            tensors.emplace_back( cpu_context_, shape );
        }

        // Initialize first tensor with test data
        initializeTensorWithTestData( tensors[0] );

        // Perform sequential copies
        for (int i = 1; i < num_copies; ++i) {
            EXPECT_NO_THROW( copy( tensors[i - 1], tensors[i] ) );
            EXPECT_EQ( tensors[i].shape(), shape );
            verifyTensorData( tensors[i] );
        }
    }

} // namespace Dnn::TensorOps::Tests