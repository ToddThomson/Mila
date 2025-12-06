#include <gtest/gtest.h>
#include <vector>
#include <cstdint>

import Mila;

namespace Dnn::Tensors::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TensorElementAccessTest : public testing::Test {
    protected:
        void SetUp() override {
            has_cuda_device_ = DeviceRegistry::instance().hasDeviceType( DeviceType::Cuda );
        }

        TensorElementAccessTest() {}

        bool has_cuda_device_;
    };

    // ====================================================================
    // Host-only element access tests (CPU memory resource)
    // ====================================================================

    TEST_F( TensorElementAccessTest, Host_OperatorIndex1D ) {
        std::vector<int64_t> shape = { 6 };
        auto tensor = Tensor<TensorDataType::FP32, CpuMemoryResource>( Device::Cpu(), shape );

        tensor[ {0} ] = 1.0f;
        EXPECT_FLOAT_EQ( (tensor[ {0} ]), 1.0f );
    }

    TEST_F( TensorElementAccessTest, Host_OperatorIndex2D ) {
        std::vector<int64_t> shape = { 2, 3 };
        auto tensor = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( Device::Cpu(), shape );

        tensor[ 0, 1 ] = 1.0f;
        EXPECT_FLOAT_EQ( (tensor[ 0, 1 ]), 1.0f );
    }

    TEST_F( TensorElementAccessTest, Host_OperatorIndex3D ) {
        std::vector<int64_t> shape = { 2, 3, 4 };
        auto tensor = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( Device::Cpu(), shape );

        tensor[ 1, 2, 3 ] = 5.5f;
        EXPECT_FLOAT_EQ( (tensor[ 1, 2, 3 ]), 5.5f );
    }

    TEST_F( TensorElementAccessTest, Host_OperatorIndex4D ) {
        std::vector<int64_t> shape = { 2, 2, 2, 2 };
        auto tensor = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( Device::Cpu(), shape );

        tensor[ 1, 0, 1, 0 ] = 7.7f;
        EXPECT_FLOAT_EQ( (tensor[ 1, 0, 1, 0 ]), 7.7f );
    }

    TEST_F( TensorElementAccessTest, Host_OperatorIndexOutOfBounds ) {
        std::vector<int64_t> shape = { 2, 3 };
        auto tensor = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( Device::Cpu(), shape );

        EXPECT_THROW( (tensor[ 2, 0 ]), std::out_of_range );
        EXPECT_THROW( (tensor[ 0, 3 ]), std::out_of_range );
        EXPECT_THROW( (tensor[ 2, 3 ]), std::out_of_range );
    }

    TEST_F( TensorElementAccessTest, Host_OperatorIndexWrongDimensions ) {
        std::vector<int64_t> shape = { 2, 3 };
        auto tensor = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( Device::Cpu(), shape );

        EXPECT_THROW( (tensor[ {0} ]), std::runtime_error );
        EXPECT_THROW( (tensor[ 0, 1, 2 ]), std::runtime_error );
    }

    TEST_F( TensorElementAccessTest, Host_VectorOperatorIndex ) {
        std::vector<int64_t> shape = { 2, 3, 4 };
        auto tensor = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( Device::Cpu(), shape );

        std::vector<int64_t> indices = { 1, 2, 3 };
        tensor[ indices ] = 9.9f;
        EXPECT_FLOAT_EQ( (tensor[ indices ]), 9.9f );
    }

    TEST_F( TensorElementAccessTest, Host_VectorOperatorIndexConst ) {
        shape_t shape = { 2, 3 };
        auto tensor = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( Device::Cpu(), shape );

        // Initialize all elements to 4.4f
        for ( int64_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( int64_t j = 0; j < shape[ 1 ]; ++j ) {
                tensor[ i, j ] = 4.4f;
            }
        }

        const auto& const_tensor = tensor;
        index_t indices = { 1, 2 };
        EXPECT_FLOAT_EQ( (const_tensor[ indices ]), 4.4f );
    }

    TEST_F( TensorElementAccessTest, Host_DeviceMemoryDirectAccessThrows_CompileTimeBehavior ) {
        std::vector<int64_t> shape = { 2, 3 };

        // This test validates that CPU tensor operator[] works as expected.
        auto cpu_tensor = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( Device::Cpu(), shape );

        cpu_tensor[ 0, 0 ] = 1.0f;
        EXPECT_FLOAT_EQ( (cpu_tensor[ 0, 0 ]), 1.0f );
    }

    TEST_F( TensorElementAccessTest, Host_IntegerTypes ) {
        std::vector<int64_t> shape = { 2, 2 };

        {
            auto int32_tensor = Tensor<TensorDataType::INT32, Compute::CpuMemoryResource>( Device::Cpu(), shape );
            int32_tensor[ 0, 0 ] = 42;
            EXPECT_EQ( (int32_tensor[ 0, 0 ]), 42 );
        }

        {
            auto int8_tensor = Tensor<TensorDataType::INT8, Compute::CpuMemoryResource>( Device::Cpu(), shape );
            int8_tensor[ 0, 1 ] = -100;
            EXPECT_EQ( (int8_tensor[ 0, 1 ]), -100 );
        }

        {
            auto uint32_tensor = Tensor<TensorDataType::UINT32, Compute::CpuMemoryResource>( Device::Cpu(), shape );
            uint32_tensor[ 1, 0 ] = 4000000000u;
            EXPECT_EQ( (uint32_tensor[ 1, 0 ]), 4000000000u );
        }
    }

    TEST_F( TensorElementAccessTest, Host_SingleElementTensor ) {
        std::vector<int64_t> shape = { 1 };
        auto single_tensor = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( Device::Cpu(), shape );

        single_tensor[ {0} ] = 99.0f;
        EXPECT_FLOAT_EQ( (single_tensor[ {0} ]), 99.0f );
    }

    TEST_F( TensorElementAccessTest, Host_MultiDimensionalSingleElement ) {
        std::vector<int64_t> shape = { 1, 1, 1 };
        auto tensor = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( Device::Cpu(), shape );

        tensor[ 0, 0, 0 ] = 77.0f;
        EXPECT_FLOAT_EQ( (tensor[ 0, 0, 0 ]), 77.0f );
    }

    TEST_F( TensorElementAccessTest, Host_LargeTensorAccess ) {
        std::vector<int64_t> shape = { 100, 200 };
        auto large_tensor = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( Device::Cpu(), shape );

        // Initialize all to 1.0f
        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                large_tensor[ i, j ] = 1.0f;
            }
        }

        large_tensor[ 50, 100 ] = 2.0f;
        large_tensor[ 99, 199 ] = 3.0f;
        large_tensor[ {0, 0} ] = 4.0f;

        EXPECT_FLOAT_EQ( (large_tensor[ 50, 100 ]), 2.0f );
        EXPECT_FLOAT_EQ( (large_tensor[ 99, 199 ]), 3.0f );
        EXPECT_FLOAT_EQ( (large_tensor[ {0, 0} ]), 4.0f );
        EXPECT_FLOAT_EQ( (large_tensor[ 25, 75 ]), 1.0f );
    }

    TEST_F( TensorElementAccessTest, Host_SequentialAccess ) {
        shape_t shape = { 3, 4 };
        auto tensor = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( Device::Cpu(), shape );

        float value = 1.0f;
        for ( int64_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( int64_t j = 0; j < shape[ 1 ]; ++j ) {
                tensor[ i, j ] = value;
                value += 1.0f;
            }
        }

        value = 1.0f;
        for ( int64_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( int64_t j = 0; j < shape[ 1 ]; ++j ) {
                EXPECT_FLOAT_EQ( (tensor[ i, j ]), value );
                EXPECT_FLOAT_EQ( (tensor[ {i, j} ]), value );
                value += 1.0f;
            }
        }
    }

    TEST_F( TensorElementAccessTest, Host_RandomAccess ) {
        std::vector<int64_t> shape = { 5, 5 };
        auto tensor = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( Device::Cpu(), shape );

        // Initialize all to 0.0f
        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                tensor[ i, j ] = 0.0f;
            }
        }

        tensor[ 2, 3 ] = 1.0f;
        tensor[ 0, 4 ] = 2.0f;
        tensor[ 4, 0 ] = 3.0f;
        tensor[ 1, 1 ] = 4.0f;
        tensor[ {3, 2} ] = 5.0f;

        EXPECT_FLOAT_EQ( (tensor[ 2, 3 ]), 1.0f );
        EXPECT_FLOAT_EQ( (tensor[ 0, 4 ]), 2.0f );
        EXPECT_FLOAT_EQ( (tensor[ 4, 0 ]), 3.0f );
        EXPECT_FLOAT_EQ( (tensor[ 1, 1 ]), 4.0f );
        EXPECT_FLOAT_EQ( (tensor[ {3, 2} ]), 5.0f );

        EXPECT_FLOAT_EQ( (tensor[ 0, 0 ]), 0.0f );
        EXPECT_FLOAT_EQ( (tensor[ 4, 4 ]), 0.0f );
    }

    TEST_F( TensorElementAccessTest, Host_ConstTensorAccess ) {
        shape_t shape = { 2, 3 };
        auto tensor = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( Device::Cpu(), shape );

        // Initialize all to 7.0f
        for ( size_t i = 0; i < shape[ 0 ]; ++i ) {
            for ( size_t j = 0; j < shape[ 1 ]; ++j ) {
                tensor[ i, j ] = 7.0f;
            }
        }

        const auto& const_tensor = tensor;

        EXPECT_FLOAT_EQ( (const_tensor[ 0, 0 ]), 7.0f );
        EXPECT_FLOAT_EQ( (const_tensor[ 1, 2 ]), 7.0f );

        index_t indices = { 1, 1 };
        EXPECT_FLOAT_EQ( (const_tensor[ indices ]), 7.0f );
    }

    TEST_F( TensorElementAccessTest, Host_ErrorMessageValidation ) {
        std::vector<int64_t> shape = { 2, 3 };
        auto tensor = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( Device::Cpu(), shape );

        try {
            (tensor[ 2, 0 ]);
            FAIL() << "Expected std::out_of_range exception";
        }
        catch ( const std::out_of_range& e ) {
            std::string error_msg( e.what() );
            EXPECT_NE( error_msg.find( "operator[]" ), std::string::npos );
            EXPECT_NE( error_msg.find( "out of range" ), std::string::npos );
        }

        try {
            (tensor[ {0, 3} ]);
            FAIL() << "Expected std::out_of_range exception";
        }
        catch ( const std::out_of_range& e ) {
            std::string error_msg( e.what() );
            EXPECT_NE( error_msg.find( "operator[]" ), std::string::npos );
            EXPECT_NE( error_msg.find( "out of range" ), std::string::npos );
        }

        try {
            (tensor[ {0} ]); // Wrong number of dimensions
            FAIL() << "Expected std::runtime_error exception";
        }
        catch ( const std::runtime_error& e ) {
            std::string error_msg( e.what() );
            EXPECT_NE( error_msg.find( "operator[]" ), std::string::npos );
            EXPECT_NE( error_msg.find( "match" ), std::string::npos );
        }
    }

    TEST_F( TensorElementAccessTest, Host_AccessPerformance ) {
        std::vector<int64_t> shape = { 100, 100 };
        auto tensor = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( Device::Cpu(), shape );

        for ( size_t i = 0; i < 100; ++i ) {
            for ( size_t j = 0; j < 100; ++j ) {
                tensor[ i, j ] = static_cast<float>( i * 100 + j );
            }
        }

        for ( size_t i = 0; i < 100; ++i ) {
            for ( size_t j = 0; j < 100; ++j ) {
                float expected = static_cast<float>( i * 100 + j );
                EXPECT_FLOAT_EQ( (tensor[ i, j ]), expected );
            }
        }
    }

    TEST_F( TensorElementAccessTest, Host_HostTypeMappingInt8 ) {
        std::vector<int64_t> shape = { 2, 2 };
        auto int8_tensor = Tensor<TensorDataType::INT8, Compute::CpuMemoryResource>( Device::Cpu(), shape );

        // INT8 should map to std::int8_t for host access
        int8_tensor[ 0, 0 ] = 127;
        int8_tensor[ 1, 1 ] = -128;

        EXPECT_EQ( (int8_tensor[ 0, 0 ]), 127 );
        EXPECT_EQ( (int8_tensor[ 1, 1 ]), -128 );
    }

    TEST_F( TensorElementAccessTest, Host_TypeSafetyCompileTime ) {
        std::vector<int64_t> shape = { 2, 2 };

        // Test that different data types work correctly
        auto fp32_tensor = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>( Device::Cpu(), shape );
        auto int32_tensor = Tensor<TensorDataType::INT32, Compute::CpuMemoryResource>( Device::Cpu(), shape );

        fp32_tensor[ 0, 0 ] = 3.14f;
        int32_tensor[ 0, 0 ] = 42;

        EXPECT_FLOAT_EQ( (fp32_tensor[ 0, 0 ]), 3.14f );
        EXPECT_EQ( (int32_tensor[ 0, 0 ]), 42 );

        // Verify types are correctly handled
        static_assert(std::is_same_v<decltype(fp32_tensor[ 0, 0 ]), float&>);
        static_assert(std::is_same_v<decltype(int32_tensor[ 0, 0 ]), std::int32_t&>);
    }

    // ====================================================================
    // Device-related element access tests (pinned / managed memory)
    // ====================================================================

    TEST_F( TensorElementAccessTest, Device_PinnedMemoryAccess ) {
        if ( !has_cuda_device_ ) {
            GTEST_SKIP() << "CUDA device not available. Skipping pinned memory access test.";
        }

        std::vector<int64_t> shape = { 2, 3 };
        auto pinned_tensor = Tensor<TensorDataType::FP32, Compute::CudaPinnedMemoryResource>( Device::Cuda( 0 ), shape );

        pinned_tensor[ 0, 0 ] = 1.1f;
        pinned_tensor[ 1, 2 ] = 2.2f;

        EXPECT_FLOAT_EQ( (pinned_tensor[ 0, 0 ]), 1.1f );
        EXPECT_FLOAT_EQ( (pinned_tensor[ 1, 2 ]), 2.2f );
    }

    TEST_F( TensorElementAccessTest, Device_ManagedMemoryAccess ) {
        if ( !has_cuda_device_ ) {
            GTEST_SKIP() << "CUDA device not available. Skipping managed memory access test.";
        }

        std::vector<int64_t> shape = { 2, 2 };
        auto managed_tensor = Tensor<TensorDataType::FP32, Compute::CudaManagedMemoryResource>( Device::Cuda( 0 ), shape );

        managed_tensor[ 0, 0 ] = 4.4f;
        managed_tensor[ 1, 1 ] = 5.5f;

        EXPECT_FLOAT_EQ( (managed_tensor[ 0, 0 ]), 4.4f );
        EXPECT_FLOAT_EQ( (managed_tensor[ 1, 1 ]), 5.5f );
    }
}