#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <cuda_runtime.h>

import Mila;

namespace Core::TensorBuffers::Tests
{
    using namespace Mila::Dnn;

    class TensorBufferTest : public ::testing::Test {
    protected:
        void SetUp() override {}
        void TearDown() override {}
    };

    TEST_F( TensorBufferTest, CpuBufferConstruction ) {
        // Default construction with size
        Mila::Dnn::TensorBuffer<float, Mila::Dnn::Compute::HostMemoryResource> buffer( 100 );
        EXPECT_EQ( buffer.size(), 100 );
        EXPECT_NE( buffer.data(), nullptr );

        // Construction with size and value
        Mila::Dnn::TensorBuffer<float, Mila::Dnn::Compute::HostMemoryResource> buffer2( 100, 1.0f );
        EXPECT_EQ( buffer2.size(), 100 );
        for ( size_t i = 0; i < buffer2.size(); ++i ) {
            EXPECT_FLOAT_EQ( buffer2.data()[ i ], 1.0f );
        }
    }

    TEST_F( TensorBufferTest, CpuBufferResize ) {
        Mila::Dnn::TensorBuffer<int, Mila::Dnn::Compute::HostMemoryResource> buffer( 50 );

        // Resize larger
        buffer.resize( 100 );
        EXPECT_EQ( buffer.size(), 100 );

        // Resize smaller
        buffer.resize( 25 );
        EXPECT_EQ( buffer.size(), 25 );

        // Resize to zero
        buffer.resize( 0 );
        EXPECT_EQ( buffer.size(), 0 );
    }

    TEST_F( TensorBufferTest, CpuBufferExternalMemory ) {
        std::vector<double> external_data( 100, 3.14 );
        Mila::Dnn::TensorBuffer<double, Mila::Dnn::Compute::HostMemoryResource> buffer(
            external_data.size(), external_data.data() );

        EXPECT_EQ( buffer.size(), external_data.size() );
        EXPECT_EQ( buffer.data(), external_data.data() );

        // Verify resize throws for external buffer
        EXPECT_THROW( buffer.resize( 50 ), std::runtime_error );
    }

    TEST_F( TensorBufferTest, VerifyAlignment ) {
        TensorBuffer<float, Compute::HostMemoryResource> cpu_buffer( 100 );
        EXPECT_TRUE( cpu_buffer.is_aligned() );

        TensorBuffer<float, Compute::DeviceMemoryResource> cuda_buffer( 100 );
        EXPECT_TRUE( cuda_buffer.is_aligned() );
    }

    // CUDA Buffer Tests
    TEST_F( TensorBufferTest, CudaBufferConstruction ) {
        // Basic construction
        Mila::Dnn::TensorBuffer<float, Mila::Dnn::Compute::DeviceMemoryResource> buffer( 100 );
        EXPECT_EQ( buffer.size(), 100 );
        EXPECT_NE( buffer.data(), nullptr );

        // Construction with initialization
        Mila::Dnn::TensorBuffer<float, Mila::Dnn::Compute::DeviceMemoryResource> buffer2( 100, 1.0f );
        std::vector<float> host_data( 100 );
        cudaMemcpy( host_data.data(), buffer2.data(), 100 * sizeof( float ),
            cudaMemcpyDeviceToHost );

        for ( float val : host_data ) {
            EXPECT_FLOAT_EQ( val, 1.0f );
        }
    }

    TEST_F( TensorBufferTest, CudaBufferResize ) {
        Mila::Dnn::TensorBuffer<int, Mila::Dnn::Compute::DeviceMemoryResource> buffer( 50 );
        auto original_ptr = buffer.data();

        // Resize larger
        buffer.resize( 100 );
        EXPECT_EQ( buffer.size(), 100 );
        EXPECT_NE( buffer.data(), original_ptr );

        // Resize smaller
        buffer.resize( 25 );
        EXPECT_EQ( buffer.size(), 25 );
    }

    TEST_F( TensorBufferTest, ExceptionHandling ) {
        // Null external pointer
        EXPECT_THROW( (Mila::Dnn::TensorBuffer<int, Mila::Dnn::Compute::HostMemoryResource>( 100, nullptr )), std::invalid_argument );

        // Very large allocation (should throw bad_alloc)
        EXPECT_THROW( (Mila::Dnn::TensorBuffer<int, Mila::Dnn::Compute::HostMemoryResource>( std::numeric_limits<size_t>::max() )), std::overflow_error );

        // Overflow in aligned_size calculation in constructor
        EXPECT_THROW( (Mila::Dnn::TensorBuffer<int, Mila::Dnn::Compute::HostMemoryResource>( (std::numeric_limits<size_t>::max() / sizeof( int )) + 1 )), std::overflow_error );

        // Overflow in aligned_size calculation in resize
        Mila::Dnn::TensorBuffer<int, Mila::Dnn::Compute::HostMemoryResource> buffer( 10 );
        EXPECT_THROW( buffer.resize( (std::numeric_limits<size_t>::max() / sizeof( int )) + 1 ), std::overflow_error );
    }

    // Type Traits Tests
    TEST_F( TensorBufferTest, DifferentTypes ) {
        // Test with different types
        Mila::Dnn::TensorBuffer<int, Mila::Dnn::Compute::HostMemoryResource> int_buffer( 10 );
        Mila::Dnn::TensorBuffer<double, Mila::Dnn::Compute::HostMemoryResource> double_buffer( 10 );
        Mila::Dnn::TensorBuffer<char, Mila::Dnn::Compute::HostMemoryResource> char_buffer( 10 );

        EXPECT_EQ( int_buffer.size(), 10 );
        EXPECT_EQ( double_buffer.size(), 10 );
        EXPECT_EQ( char_buffer.size(), 10 );
    }

    // Performance Test (optional)
    TEST_F( TensorBufferTest, LargeBufferPerformance ) {
        constexpr size_t large_size = 10000000;

        auto start = std::chrono::high_resolution_clock::now();
        Mila::Dnn::TensorBuffer<float, Mila::Dnn::Compute::HostMemoryResource> buffer( large_size );
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        EXPECT_LT( duration.count(), 1000 ); // Should allocate within 1 second
    }

    TEST_F( TensorBufferTest, CpuMemoryResource_DefaultInitialization ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        EXPECT_EQ( tensor.size(), 6 );
        for ( size_t i = 0; i < 2; ++i ) {
            for ( size_t j = 0; j < 3; ++j ) {
                auto val = tensor[ i, j ];
                EXPECT_EQ( val, 0.0f );
            }
        }
    }

    TEST_F( TensorBufferTest, CudaMemoryResource_DefaultInitialization ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::DeviceMemoryResource> tensor( shape );
        EXPECT_EQ( tensor.size(), 6 );

		auto host_tensor = tensor.to<Compute::HostMemoryResource>();

        for ( size_t i = 0; i < 2; ++i ) {
            for ( size_t j = 0; j < 3; ++j ) {
                auto val = host_tensor[ i, j ];
                EXPECT_EQ( val, 0.0f );
            }
        }
    }

    TEST_F( TensorBufferTest, CpuMemoryResource_ValueInitialization ) {
        std::vector<size_t> shape = { 5, 723 };
        Tensor<int, Compute::HostMemoryResource> tensor(shape, 5);
        for ( size_t i = 0; i < 5; ++i ) {
            for ( size_t j = 0; j < 723; ++j ) {
                auto val = tensor[ i, j ];
                EXPECT_EQ( val,5 );
            }
        }
    }

    TEST_F( TensorBufferTest, CudaMemoryResource_ValueInitialization ) {
        std::vector<size_t> shape = { 5, 723 };
        Tensor<float, Compute::DeviceMemoryResource> tensor( shape, 3.1415f );

        auto host_tensor = tensor.to<Compute::HostMemoryResource>();

        for ( size_t i = 0; i < 5; ++i ) {
            for ( size_t j = 0; j < 723; ++j ) {
                auto val = host_tensor[ i, j ];
                EXPECT_EQ( val, 3.1415f );
            }
        }
    }
}
