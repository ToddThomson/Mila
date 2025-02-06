#include <gtest/gtest.h>
#include <vector>

import Mila;

namespace Dnn::TensorBuffers::Tests
{
    using namespace Mila::Dnn;

    TEST( TensorBufferTest, CpuMemoryResource_DefaultInitialization ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::CpuMemoryResource> tensor( shape );
        EXPECT_EQ( tensor.size(), 6 );
        for ( size_t i = 0; i < 2; ++i ) {
            for ( size_t j = 0; j < 3; ++j ) {
                auto val = tensor[ i, j ];
                EXPECT_EQ( val, 0.0f );
            }
        }
    }

    TEST( TensorBufferTest, DeviceMemoryResource_DefaultInitialization ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::DeviceMemoryResource> tensor( shape );
        EXPECT_EQ( tensor.size(), 6 );

		auto host_tensor = tensor.to<Compute::CpuMemoryResource>();

        for ( size_t i = 0; i < 2; ++i ) {
            for ( size_t j = 0; j < 3; ++j ) {
                auto val = host_tensor[ i, j ];
                EXPECT_EQ( val, 0.0f );
            }
        }
    }

    TEST( TensorBufferTest, CpuMemoryResource_ValueInitialization ) {
        std::vector<size_t> shape = { 5, 723 };
        Tensor<int, Compute::CpuMemoryResource> tensor(shape, 5);
        for ( size_t i = 0; i < 5; ++i ) {
            for ( size_t j = 0; j < 723; ++j ) {
                auto val = tensor[ i, j ];
                EXPECT_EQ( val,5 );
            }
        }
    }

    TEST( TensorBufferTest, DeviceMemoryResource_ValueInitialization ) {
        std::vector<size_t> shape = { 5, 723 };
        Tensor<float, Compute::DeviceMemoryResource> tensor( shape, 3.1415f );

        auto host_tensor = tensor.to<Compute::CpuMemoryResource>();

        for ( size_t i = 0; i < 5; ++i ) {
            for ( size_t j = 0; j < 723; ++j ) {
                auto val = host_tensor[ i, j ];
                EXPECT_EQ( val, 3.1415f );
            }
        }
    }
}
