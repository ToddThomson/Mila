#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <memory>

import Mila;

namespace Dnn::Compute::Tests
{
    using namespace Mila::Dnn::Compute;

    class DeviceContextTest : public ::testing::Test {
    protected:
        void SetUp() override {
            // No need to reset singleton instance as DeviceContext is no longer a singleton
        }
    };

    TEST_F( DeviceContextTest, ConstructWithCudaOrCpuDevice ) {
        // Try to construct with CUDA:0, falling back to CPU if CUDA is not available
        try {
            DeviceContext context( "CUDA:0" );
            auto device = context.getDevice();
            EXPECT_NE( device, nullptr );
            EXPECT_EQ( device->getName(), "CUDA:0" );
        }
        catch ( const std::exception& ) {
            // CUDA:0 is not available, try with CPU
            DeviceContext context( "CPU" );
            auto device = context.getDevice();
            EXPECT_NE( device, nullptr );
            EXPECT_EQ( device->getName(), "CPU" );
        }
    }

    TEST_F( DeviceContextTest, ConstructWithSpecificDevice ) {
        DeviceContext context( "CPU" );
        auto device = context.getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getName(), "CPU" );
    }

    TEST_F( DeviceContextTest, ConstructWithInvalidDeviceThrows ) {
        EXPECT_THROW( DeviceContext( "InvalidDevice" ), std::runtime_error );
    }

    TEST_F( DeviceContextTest, MoveConstructor ) {
        DeviceContext context1( "CPU" );
        DeviceContext context2( std::move( context1 ) );

        auto device = context2.getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getName(), "CPU" );
    }

    TEST_F( DeviceContextTest, MoveAssignmentOperator ) {
        DeviceContext context1( "CPU" );

        // Try to create with CUDA:0 if available, otherwise use CPU
        DeviceContext* context2_ptr = nullptr;
        try {
            context2_ptr = new DeviceContext( "CUDA:0" );
        }
        catch ( const std::exception& ) {
            // If CUDA:0 is not available, use CPU instead for the second context
            context2_ptr = new DeviceContext( "CPU" );
        }

        DeviceContext& context2 = *context2_ptr;

        // Test the move assignment
        context2 = std::move( context1 );
        auto device = context2.getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getName(), "CPU" );

        delete context2_ptr;
    }

    TEST_F( DeviceContextTest, IsDeviceType ) {
        DeviceContext context( "CPU" );
        EXPECT_TRUE( context.isDeviceType( DeviceType::Cpu ) );
        EXPECT_FALSE( context.isDeviceType( DeviceType::Cuda ) );
    }

    TEST_F( DeviceContextTest, IsCudaDevice ) {
        DeviceContext cpuContext( "CPU" );
        EXPECT_FALSE( cpuContext.isCudaDevice() );

        // Optional test for CUDA device, if available
        try {
            DeviceContext cudaContext( "CUDA:0" );
            EXPECT_TRUE( cudaContext.isCudaDevice() );
        }
        catch ( const std::exception& ) {
            // CUDA:0 is not available, so skip this assertion
            std::cout << "CUDA device not available, skipping isCudaDevice() test" << std::endl;
        }
    }

    TEST_F( DeviceContextTest, GetStreamForCpuReturnsNull ) {
        DeviceContext context( "CPU" );
        EXPECT_EQ( context.getStream(), nullptr );
    }

    // This test only runs if CUDA is available
    TEST_F( DeviceContextTest, GetStreamForCudaReturnsStream ) {
        try {
            DeviceContext context( "CUDA:0" );
            if ( context.isCudaDevice() ) {
                EXPECT_NE( context.getStream(), nullptr );
            }
        }
        catch ( const std::exception& ) {
            // CUDA:0 is not available, so skip this test
            std::cout << "CUDA device not available, skipping getStream() test" << std::endl;
        }
    }

    // Test synchronize() doesn't throw for CPU
    TEST_F( DeviceContextTest, SynchronizeCpuDoesNotThrow ) {
        DeviceContext context( "CPU" );
        EXPECT_NO_THROW( context.synchronize() );
    }

    // New tests for cuBLASLt handle functionality
    TEST_F( DeviceContextTest, GetCublasLtHandleForCpu ) {
        DeviceContext context( "CPU" );
        // For CPU devices, we expect this to return a null handle
        EXPECT_EQ( context.getCublasLtHandle(), nullptr );
    }

    TEST_F( DeviceContextTest, GetCublasLtHandleForCuda ) {
        try {
            DeviceContext context( "CUDA:0" );
            if ( context.isCudaDevice() ) {
                // For CUDA devices, we expect a non-null handle
                EXPECT_NE( context.getCublasLtHandle(), nullptr );
            }
        }
        catch ( const std::exception& ) {
            // CUDA:0 is not available, so skip this test
            std::cout << "CUDA device not available, skipping getCublasLtHandle() test" << std::endl;
        }
    }

#ifdef USE_CUDNN
    // Test for CUDNN handle (only when USE_CUDNN is defined)
    TEST_F( DeviceContextTest, GetCudnnHandleForCpu ) {
        DeviceContext context( "CPU" );
        // For CPU devices, we expect this to return a null handle
        EXPECT_EQ( context.getCudnnHandle(), nullptr );
    }

    TEST_F( DeviceContextTest, GetCudnnHandleForCuda ) {
        try {
            DeviceContext context( "CUDA:0" );
            if ( context.isCudaDevice() ) {
                // For CUDA devices, we expect a non-null handle
                EXPECT_NE( context.getCudnnHandle(), nullptr );
            }
        }
        catch ( const std::exception& ) {
            // CUDA:0 is not available, so skip this test
            std::cout << "CUDA device not available, skipping getCudnnHandle() test" << std::endl;
        }
    }
#endif

    TEST_F( DeviceContextTest, CublasLtHandleCaching ) {
        try {
            DeviceContext context( "CUDA:0" );
            if ( context.isCudaDevice() ) {
                auto handle1 = context.getCublasLtHandle();
                auto handle2 = context.getCublasLtHandle();
                EXPECT_EQ( handle1, handle2 );
            }
        }
        catch ( const std::exception& ) {
            // CUDA:0 is not available, so skip this test
            std::cout << "CUDA device not available, skipping CublasLtHandleCaching test" << std::endl;
        }
    }

#ifdef USE_CUDNN
    // Test that multiple calls to getCudnnHandle return the same handle
    TEST_F( DeviceContextTest, CudnnHandleCaching ) {
        try {
            DeviceContext context( "CUDA:0" );
            if ( context.isCudaDevice() ) {
                auto handle1 = context.getCudnnHandle();
                auto handle2 = context.getCudnnHandle();
                EXPECT_EQ( handle1, handle2 );
            }
        }
        catch ( const std::exception& ) {
            // CUDA:0 is not available, so skip this test
            std::cout << "CUDA device not available, skipping CudnnHandleCaching test" << std::endl;
        }
    }
#endif
}