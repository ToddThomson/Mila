#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

import Mila;

namespace Dnn::Tensors::TensorOps::Tests
{
    using namespace Mila::Dnn;

    class TensorOpsTransferTest : public ::testing::Test {
    protected:
        void SetUp() override {
            cpu_exec_context_ = std::make_shared<Compute::CpuExecutionContext>();

            // Initialize CUDA context if available
            try
            {
                cuda_exec_context_ = std::make_shared<Compute::CudaExecutionContext>( 0 );
            }
            catch (const std::exception&)
            {
                cuda_exec_context_ = nullptr;
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

            ASSERT_FALSE( tensor.empty() );
            ASSERT_EQ( tensor.getDataType(), TDataType );

            // Use appropriate host value type for the tensor data type
            using HostValueType = std::conditional_t<TensorDataTypeTraits<TDataType>::is_integer_type, int32_t, float>;

            if constexpr (TensorDataTypeTraits<TDataType>::is_integer_type)
            {
                HostValueType value = static_cast<HostValueType>(42);
                fill( tensor, value );
            }
            else
            {
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

            if constexpr (TensorDataTypeTraits<TDataType>::is_integer_type)
            {
                // Random integers in range [1, 100]
                std::uniform_int_distribution<int> dist( 1, 100 );
                HostValueType value = static_cast<HostValueType>(dist( rng_ ));
                fill( tensor, value );
            }
            else
            {
                // Random floats in range [0.0, 1.0]
                std::uniform_real_distribution<float> dist( 0.0f, 1.0f );
                HostValueType value = static_cast<HostValueType>(dist( rng_ ));
                fill( tensor, value );
            }
        }

        /**
         * @brief Per-data-type tolerances factory.
         *
         * Returns (relative_tolerance, absolute_tolerance) for the given abstract type.
         */
        template<TensorDataType TDataType>
        static constexpr std::pair<double, double> getTolerances() noexcept
        {
            if constexpr (TDataType == TensorDataType::FP32)
            {
                return { 1e-5, 1e-8 };
            }
            else if constexpr (TDataType == TensorDataType::FP16)
            {
                return { 1e-2, 1e-6 };
            }
            else if constexpr (TDataType == TensorDataType::BF16)
            {
                return { 1e-2, 1e-6 };
            }
            else if constexpr (TDataType == TensorDataType::FP8_E4M3)
            {
                return { 5e-2, 1e-3 };
            }
            else if constexpr (TDataType == TensorDataType::FP8_E5M2)
            {
                return { 5e-2, 1e-3 };
            }
            else
            {
                return { 1e-5, 1e-8 };
            }
        }

        // Helper to deduce tolerances from a tensor (convenience)
        template<TensorDataType TDataType, typename TM>
        static constexpr std::pair<double, double> getTolerancesFromTensor( const Tensor<TDataType, TM>& ) noexcept {
            return getTolerances<TDataType>();
        }

        /**
         * @brief Verify tensor contains expected test data after copy
         *
         * Accepts explicit tolerances (rtol, atol) so callers may pass
         * appropriate values for FP16/BF16/FP8.
         */
        template<TensorDataType TDataType>
        void verifyTensorData( const Tensor<TDataType, Compute::CpuMemoryResource>& tensor,
            std::pair<double, double> tolerances )
        {
            EXPECT_FALSE( tensor.empty() );
            EXPECT_EQ( tensor.getDataType(), TDataType );

            const size_t n = tensor.size();
            if (n == 0) return;

            using HostType = typename TensorHostTypeMap<TDataType>::host_type;
            auto* data = tensor.data();

            const double rtol = tolerances.first;
            const double atol = tolerances.second;

            if constexpr (TensorDataTypeTraits<TDataType>::is_integer_type)
            {
                const HostType expected = static_cast<HostType>(42);
                for (size_t i = 0; i < n; ++i)
                {
                    EXPECT_EQ( data[i], expected ) << "Mismatch at index " << i;
                }
            }
            else
            {
                // Floating-point: use combined absolute + relative tolerance:
                // pass if |actual - expected| <= atol + rtol * |expected|
                const double expected = 3.14159;
                for (size_t i = 0; i < n; ++i)
                {
                    const double actual = static_cast<double>( data[i] );
                    const double diff = std::fabs( actual - expected );
                    const double threshold = atol + rtol * std::fabs( expected );
                    EXPECT_LE( diff, threshold ) << "Mismatch at index " << i
                        << " actual=" << actual << " expected=" << expected
                        << " diff=" << diff << " thresh=" << threshold;
                }
            }
        }

        std::shared_ptr<Compute::CpuExecutionContext> cpu_exec_context_;
        std::shared_ptr<Compute::CudaExecutionContext> cuda_exec_context_;
        std::mt19937 rng_; // Random number generator for test data
    };

    // ====================================================================
    // CPU to CPU Copy Tests (Host-accessible to Host-accessible)
    // ====================================================================

    TEST_F( TensorOpsTransferTest, CpuToCpu_SameType_FP32 ) {
        const std::vector<size_t> shape = { 2, 3, 4 };

        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );
        auto dst = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );

        initializeTensorWithTestData( src );

        EXPECT_NO_THROW( copy( src, dst ) );

        EXPECT_EQ( dst.shape(), src.shape() );
        EXPECT_EQ( dst.size(), src.size() );
        EXPECT_EQ( dst.getDataType(), src.getDataType() );

        verifyTensorData( dst, TensorOpsTransferTest::getTolerancesFromTensor( dst ) );
    }

    TEST_F( TensorOpsTransferTest, CpuToCpu_TypeConversion_FP32_to_INT32 ) {
        const std::vector<size_t> shape = { 2, 3 };

        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );
        auto dst = Tensor<TensorDataType::INT32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );

        initializeTensorWithTestData( src );

        EXPECT_NO_THROW( copy( src, dst ) );

        // Verify type conversion occurred
        EXPECT_EQ( dst.getDataType(), TensorDataType::INT32 );
        EXPECT_EQ( dst.shape(), src.shape() );
    }

    TEST_F( TensorOpsTransferTest, CpuToCpu_DifferentShapes_ThrowsException ) {
        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), { 2, 3 } );
        auto dst = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), { 3, 2 } );

        initializeTensorWithTestData( src );

        EXPECT_THROW( copy( src, dst ), std::invalid_argument );
    }

    TEST_F( TensorOpsTransferTest, CpuToCpu_EmptyTensors_NoOperation ) {
        // Align with new Tensor semantics: empty shape {} == scalar (size 1).
        // Zero-sized tensor must use a shape with a zero dimension, e.g., {0}.
        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), { 0 } );
        auto dst = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), { 0 } );

        EXPECT_NO_THROW( copy( src, dst ) );
        EXPECT_TRUE( src.empty() );
        EXPECT_TRUE( dst.empty() );
    }

    // ====================================================================
    // CPU to CUDA Copy Tests (Host-accessible to Device-only)
    // ====================================================================

    TEST_F( TensorOpsTransferTest, CpuToCuda_SameType_FP32 ) {
        if (!cuda_exec_context_)
        {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 2, 3 };

        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );
        auto dst = Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>( cuda_exec_context_->getDevice(), shape );

        initializeTensorWithTestData( src );

        EXPECT_NO_THROW( copy( src, dst ) );

        EXPECT_EQ( dst.shape(), src.shape() );
        EXPECT_EQ( dst.size(), src.size() );
        EXPECT_EQ( dst.getDataType(), src.getDataType() );
        EXPECT_FALSE( dst.is_host_accessible() );
        EXPECT_TRUE( dst.is_device_accessible() );

        // Verify data by copying back to CPU
        auto verify_host = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );
        EXPECT_NO_THROW( copy( dst, verify_host ) );

        verifyTensorData( verify_host, TensorOpsTransferTest::getTolerancesFromTensor( verify_host ) );
    }

    TEST_F( TensorOpsTransferTest, CpuToCuda_TypeConversion_FP32_to_FP16 ) {
        if (!cuda_exec_context_)
        {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 3, 3 };

        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );
        auto dst = Tensor<TensorDataType::FP16, Compute::CudaDeviceMemoryResource>( cuda_exec_context_->getDevice(), shape );

        initializeTensorWithTestData( src );

        EXPECT_NO_THROW( copy( src, dst ) );

        EXPECT_EQ( dst.getDataType(), TensorDataType::FP16 );
        EXPECT_EQ( dst.shape(), src.shape() );

        auto verify_host = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );
        EXPECT_NO_THROW( copy( dst, verify_host ) );

        verifyTensorData( verify_host, TensorOpsTransferTest::getTolerancesFromTensor( dst ) );
    }

    // ====================================================================
    // toHost helper validation (device -> host)
    // ====================================================================

    TEST_F( TensorOpsTransferTest, DeviceTensor_ToHost_VerifyData ) {
        if (!cuda_exec_context_)
        {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 2, 2 };

        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );
        initializeTensorWithTestData( src );

        // Create device tensor and copy from host to device
        auto gpu_tensor = Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>( cuda_exec_context_->getDevice(), shape );
        EXPECT_NO_THROW( copy( src, gpu_tensor, cuda_exec_context_.get() ) );

        // Use toHost() to bring device tensor back to host and validate
        auto host_back = toHost<TensorDataType::FP32>( gpu_tensor, cuda_exec_context_.get() );
        
        EXPECT_TRUE( host_back.is_host_accessible() );
        EXPECT_EQ( host_back.shape(), src.shape() );
        
        verifyTensorData( host_back, TensorOpsTransferTest::getTolerancesFromTensor( host_back ) );
    }

    TEST_F( TensorOpsTransferTest, DeviceTensor_TypeConversion_ToHost ) {
        if (!cuda_exec_context_)
        {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 3, 1 };

        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );
        initializeTensorWithTestData( src );

        // Create FP16 device tensor, copy FP32->FP16 on device
        auto gpu_fp16 = Tensor<TensorDataType::FP16, Compute::CudaDeviceMemoryResource>( cuda_exec_context_->getDevice(), shape );
        EXPECT_NO_THROW( copy( src, gpu_fp16, cuda_exec_context_.get() ) );

        // toHost should return a host tensor in default host type (FP16->FP32 conversion)
        auto host_fp32 = toHost<TensorDataType::FP32>( gpu_fp16, cuda_exec_context_.get() );
        EXPECT_EQ( host_fp32.getDataType(), TensorDataType::FP32 );
        verifyTensorData( host_fp32, TensorOpsTransferTest::getTolerancesFromTensor( host_fp32 ) );
    }

    // ====================================================================
    // CUDA to CPU Copy Tests (Device-only to Host-accessible)
    // ====================================================================

    TEST_F( TensorOpsTransferTest, CudaToCpu_SameType_FP32 ) {
        if (!cuda_exec_context_)
        {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 2, 2 };

        auto src = Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>( cuda_exec_context_->getDevice(), shape );
        auto dst = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );

        EXPECT_NO_THROW( copy( src, dst ) );

        // Verify transfer back to CPU
        EXPECT_EQ( dst.shape(), src.shape() );
        EXPECT_EQ( dst.size(), src.size() );
        EXPECT_EQ( dst.getDataType(), src.getDataType() );
        EXPECT_TRUE( dst.is_host_accessible() );
        EXPECT_FALSE( dst.is_device_accessible() );
    }

    TEST_F( TensorOpsTransferTest, CudaToCpu_TypeConversion_FP16_to_FP32 ) {
        if (!cuda_exec_context_)
        {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 4, 2 };

        auto src = Tensor<TensorDataType::FP16, Compute::CudaDeviceMemoryResource>( cuda_exec_context_->getDevice(), shape );
        auto dst = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );

        EXPECT_NO_THROW( copy( src, dst ) );

        // Verify conversion from FP16 to FP32
        EXPECT_EQ( dst.getDataType(), TensorDataType::FP32 );
        EXPECT_EQ( dst.shape(), src.shape() );
    }

    // ====================================================================
    // Remaining tests unchanged...
    // ====================================================================

    TEST_F( TensorOpsTransferTest, CudaToCuda_SameType_FP32 ) {
        if (!cuda_exec_context_)
        {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 3, 3 };

        auto src = Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>( cuda_exec_context_->getDevice(), shape );
        auto dst = Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>( cuda_exec_context_->getDevice(), shape );

        EXPECT_NO_THROW( copy( src, dst ) );

        EXPECT_EQ( dst.shape(), src.shape() );
        EXPECT_EQ( dst.size(), src.size() );
        EXPECT_EQ( dst.getDataType(), src.getDataType() );
        EXPECT_FALSE( dst.is_host_accessible() );
        EXPECT_TRUE( dst.is_device_accessible() );
    }

    TEST_F( TensorOpsTransferTest, CudaToCuda_TypeConversion_FP32_to_FP16 ) {
        if (!cuda_exec_context_)
        {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 2, 4 };

        auto src = Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>( cuda_exec_context_->getDevice(), shape );
        auto dst = Tensor<TensorDataType::FP16, Compute::CudaDeviceMemoryResource>( cuda_exec_context_->getDevice(), shape );

        EXPECT_NO_THROW( copy( src, dst ) );

        EXPECT_EQ( dst.getDataType(), TensorDataType::FP16 );
        EXPECT_EQ( dst.shape(), src.shape() );
    }

    TEST_F( TensorOpsTransferTest, CpuToPinned_SameType ) {
        if (!cuda_exec_context_)
        {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 2, 3 };

        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );
        auto dst = Tensor<TensorDataType::FP32, Compute::CudaPinnedMemoryResource>( cuda_exec_context_->getDevice(), shape );

        initializeTensorWithTestData( src );

        EXPECT_NO_THROW( copy( src, dst ) );

        EXPECT_EQ( dst.shape(), src.shape() );
        EXPECT_EQ( dst.getDataType(), src.getDataType() );
        EXPECT_TRUE( dst.is_host_accessible() );
    }

    TEST_F( TensorOpsTransferTest, PinnedToCuda_SameType ) {
        if (!cuda_exec_context_)
        {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 3, 2 };

        auto src = Tensor<TensorDataType::FP32, Compute::CudaPinnedMemoryResource>( cuda_exec_context_->getDevice(), shape );
        auto dst = Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>( cuda_exec_context_->getDevice(), shape );

        EXPECT_NO_THROW( copy( src, dst ) );

        EXPECT_EQ( dst.shape(), src.shape() );
        EXPECT_EQ( dst.getDataType(), src.getDataType() );
        EXPECT_FALSE( dst.is_host_accessible() );
        EXPECT_TRUE( dst.is_device_accessible() );
    }

    TEST_F( TensorOpsTransferTest, ZeroSizedTensor_NoOperation ) {
        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), { 0 } );
        auto dst = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), { 0 } );

        EXPECT_NO_THROW( copy( src, dst ) );
        EXPECT_EQ( src.size(), 0 );
        EXPECT_EQ( dst.size(), 0 );
    }

    TEST_F( TensorOpsTransferTest, SingleElementTensor ) {
        const std::vector<size_t> shape = { 1 };

        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );
        auto dst = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );

        initializeTensorWithTestData( src );

        EXPECT_NO_THROW( copy( src, dst ) );
        EXPECT_EQ( dst.size(), 1 );
        EXPECT_EQ( dst.shape(), src.shape() );
        verifyTensorData( dst, TensorOpsTransferTest::getTolerancesFromTensor( dst ) );
    }

    TEST_F( TensorOpsTransferTest, ScalarTensor_CopyBetweenScalarAndVector1 ) {
        auto scalar = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), std::vector<size_t>{} );
        auto vec1 = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), std::vector<size_t>{1} );

        initializeTensorWithTestData( scalar );
        EXPECT_NO_THROW( copy( scalar, vec1 ) );
        EXPECT_EQ( vec1.size(), 1 );
        verifyTensorData( vec1, TensorOpsTransferTest::getTolerancesFromTensor( vec1 ) );

        initializeTensorWithTestData( vec1 );
        EXPECT_NO_THROW( copy( vec1, scalar ) );
        EXPECT_EQ( scalar.size(), 1 );
        verifyTensorData( scalar, TensorOpsTransferTest::getTolerancesFromTensor( scalar ) );
    }

    TEST_F( TensorOpsTransferTest, LargeTensor_Performance ) {
        const std::vector<size_t> shape = { 100, 100 };

        auto src = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );
        auto dst = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );

        initializeTensorWithRandomData( src );

        EXPECT_NO_THROW( copy( src, dst ) );

        EXPECT_EQ( dst.size(), 10000 );
        EXPECT_EQ( dst.shape(), src.shape() );
    }

    TEST_F( TensorOpsTransferTest, IntegerTypes_INT8_to_INT32 ) {
        const std::vector<size_t> shape = { 2, 2 };

        auto src = Tensor<TensorDataType::INT8, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );
        auto dst = Tensor<TensorDataType::INT32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );

        initializeTensorWithTestData( src );

        EXPECT_NO_THROW( copy( src, dst ) );

        EXPECT_EQ( dst.getDataType(), TensorDataType::INT32 );
        EXPECT_EQ( dst.shape(), src.shape() );
    }

    TEST_F( TensorOpsTransferTest, UnsignedTypes_UINT8_to_UINT16 ) {
        const std::vector<size_t> shape = { 3, 2 };

        auto src = Tensor<TensorDataType::UINT8, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );
        auto dst = Tensor<TensorDataType::UINT16, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );

        initializeTensorWithTestData( src );

        EXPECT_NO_THROW( copy( src, dst ) );

        EXPECT_EQ( dst.getDataType(), TensorDataType::UINT16 );
        EXPECT_EQ( dst.shape(), src.shape() );
    }

    TEST_F( TensorOpsTransferTest, RoundTrip_CpuToCudaToCpu ) {
        if (!cuda_exec_context_)
        {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 2, 3 };

        auto original = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );
        auto gpu_copy = Tensor<TensorDataType::FP32, Compute::CudaDeviceMemoryResource>( cuda_exec_context_->getDevice(), shape );
        auto final_copy = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );

        initializeTensorWithTestData( original );

        EXPECT_NO_THROW( copy( original, gpu_copy ) );
        EXPECT_NO_THROW( copy( gpu_copy, final_copy ) );

        EXPECT_EQ( final_copy.shape(), original.shape() );
        EXPECT_EQ( final_copy.size(), original.size() );
        EXPECT_EQ( final_copy.getDataType(), original.getDataType() );
        verifyTensorData( final_copy, TensorOpsTransferTest::getTolerancesFromTensor( final_copy ) );
    }

    TEST_F( TensorOpsTransferTest, RoundTrip_WithTypeConversion ) {
        if (!cuda_exec_context_)
        {
            GTEST_SKIP() << "CUDA not available for testing";
        }

        const std::vector<size_t> shape = { 2, 2 };

        auto original = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );
        auto fp16_gpu = Tensor<TensorDataType::FP16, Compute::CudaDeviceMemoryResource>( cuda_exec_context_->getDevice(), shape );
        auto final_copy = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( cpu_exec_context_->getDevice(), shape );

        initializeTensorWithTestData( original );

        EXPECT_NO_THROW( copy( original, fp16_gpu ) );
        EXPECT_NO_THROW( copy( fp16_gpu, final_copy ) );

        EXPECT_EQ( final_copy.shape(), original.shape() );
        EXPECT_EQ( final_copy.getDataType(), TensorDataType::FP32 );
    }

    TEST_F( TensorOpsTransferTest, MultipleSequentialCopies ) {
        const std::vector<size_t> shape = { 10, 10 };
        const int num_copies = 5;

        std::vector<Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>> tensors;
        tensors.reserve( num_copies );

        for (int i = 0; i < num_copies; ++i)
        {
            tensors.emplace_back( cpu_exec_context_->getDevice(), shape );
        }

        initializeTensorWithTestData( tensors[0] );

        for (int i = 1; i < num_copies; ++i)
        {
            EXPECT_NO_THROW( copy( tensors[i - 1], tensors[i] ) );
            EXPECT_EQ( tensors[i].shape(), shape );
            verifyTensorData( tensors[i], TensorOpsTransferTest::getTolerancesFromTensor( tensors[i] ) );
        }
    }

} // namespace Dnn::TensorOps::Tests