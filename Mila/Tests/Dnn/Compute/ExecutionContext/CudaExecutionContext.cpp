/**
 * @file CudaExecutionContext.cpp
 * @brief Unit tests for CUDA ExecutionContext specialization.
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>

import Mila;

namespace Dnn::Compute::ExecutionContexts::Tests
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Test fixture for CUDA ExecutionContext tests.
     */
    class CudaExecutionContextTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int device_count = 0;
            cudaError_t error = cudaGetDeviceCount( &device_count );
            cuda_available_ = (error == cudaSuccess && device_count > 0);

            if (cuda_available_) {
                cudaGetDeviceCount( &device_count_ );
            }
        }

        void TearDown() override {
            if (cuda_available_) {
                cudaDeviceReset();
            }
        }

        bool cuda_available_{ false };
        int device_count_{ 0 };
    };

    // ============================================================================
    // Construction Tests
    // ============================================================================

    TEST_F( CudaExecutionContextTest, ConstructionWithValidDeviceId ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        EXPECT_NO_THROW( {
            ExecutionContext<DeviceType::Cuda> exec_ctx( 0 );
            } );
    }

    TEST_F( CudaExecutionContextTest, ConstructionWithInvalidDeviceId ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        EXPECT_THROW( {
            ExecutionContext<DeviceType::Cuda> exec_ctx( -1 );
            }, std::invalid_argument );
    }

    TEST_F( CudaExecutionContextTest, ConstructionWithNonExistentDevice ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        EXPECT_THROW( {
            ExecutionContext<DeviceType::Cuda> exec_ctx( 999 );
            }, std::runtime_error );
    }

    TEST_F( CudaExecutionContextTest, ConstructionFromDevice ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        auto device = DeviceRegistry::instance().getDevice( "CUDA:0" );
        //auto device = std::make_shared<CudaDevice>( 0 );

        EXPECT_NO_THROW( { ExecutionContext<DeviceType::Cuda> exec_ctx( device ); } );
    }

    TEST_F( CudaExecutionContextTest, ConstructionFromNullDevice ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        std::shared_ptr<ComputeDevice> null_device;

        EXPECT_THROW( {
            ExecutionContext<DeviceType::Cuda> exec_ctx( null_device );
            }, std::invalid_argument );
    }

    TEST_F( CudaExecutionContextTest, ConstructionFromWrongDeviceType ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        auto cpu_device = std::make_shared<CpuDevice>();

        EXPECT_THROW( {
            ExecutionContext<DeviceType::Cuda> exec_ctx( cpu_device );
            }, std::invalid_argument );
    }

    // ============================================================================
    // Device Properties Tests
    // ============================================================================

    TEST_F( CudaExecutionContextTest, GetDeviceType ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        ExecutionContext<DeviceType::Cuda> exec_ctx( 0 );

        EXPECT_EQ( exec_ctx.getDeviceType(), DeviceType::Cuda );
    }

    TEST_F( CudaExecutionContextTest, GetDeviceName ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        ExecutionContext<DeviceType::Cuda> exec_ctx( 0 );

        EXPECT_EQ( exec_ctx.getDeviceName(), "CUDA:0" );
    }

    TEST_F( CudaExecutionContextTest, GetDeviceId ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        ExecutionContext<DeviceType::Cuda> exec_ctx( 0 );

        EXPECT_EQ( exec_ctx.getDeviceId(), 0 );
    }

    TEST_F( CudaExecutionContextTest, GetDevice ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        ExecutionContext<DeviceType::Cuda> exec_ctx( 0 );
        auto device = exec_ctx.getDevice();

        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cuda );
        EXPECT_EQ( device->getDeviceName(), "CUDA:0" );
        EXPECT_EQ( device->getDeviceId(), 0 );
    }

    // ============================================================================
    // Compile-Time Properties Tests
    // ============================================================================

    TEST_F( CudaExecutionContextTest, ConstexprDeviceType ) {
        using CudaContext = ExecutionContext<DeviceType::Cuda>;

        // Verify compile-time constants
        static_assert(CudaContext::getDeviceType() == DeviceType::Cuda);
        static_assert(CudaContext::isCudaDevice());
        static_assert(!CudaContext::isCpuDevice());
    }

    TEST_F( CudaExecutionContextTest, DeviceTypeChecks ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        ExecutionContext<DeviceType::Cuda> exec_ctx( 0 );

        EXPECT_TRUE( exec_ctx.isCudaDevice() );
        EXPECT_FALSE( exec_ctx.isCpuDevice() );
    }

    // ============================================================================
    // CUDA Stream Tests
    // ============================================================================

    TEST_F( CudaExecutionContextTest, GetStream ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        ExecutionContext<DeviceType::Cuda> exec_ctx( 0 );
        cudaStream_t stream = exec_ctx.getStream();

        EXPECT_NE( stream, nullptr );
    }

    TEST_F( CudaExecutionContextTest, IndependentStreams ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        ExecutionContext<DeviceType::Cuda> exec_ctx1( 0 );
        ExecutionContext<DeviceType::Cuda> exec_ctx2( 0 );

        // Each context should have independent stream
        EXPECT_NE( exec_ctx1.getStream(), exec_ctx2.getStream() );
    }

    // ============================================================================
    // Synchronization Tests
    // ============================================================================

    TEST_F( CudaExecutionContextTest, Synchronize ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        ExecutionContext<DeviceType::Cuda> exec_ctx( 0 );

        EXPECT_NO_THROW( exec_ctx.synchronize() );
    }

    TEST_F( CudaExecutionContextTest, MultipleSynchronizeCalls ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        ExecutionContext<DeviceType::Cuda> exec_ctx( 0 );

        for (int i = 0; i < 100; ++i) {
            EXPECT_NO_THROW( exec_ctx.synchronize() );
        }
    }

    // ============================================================================
    // Multiple Context Tests
    // ============================================================================

    TEST_F( CudaExecutionContextTest, MultipleContextsSameDevice ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        ExecutionContext<DeviceType::Cuda> exec_ctx1( 0 );
        ExecutionContext<DeviceType::Cuda> exec_ctx2( 0 );

        // Both should be valid and independent
        EXPECT_NO_THROW( exec_ctx1.synchronize() );
        EXPECT_NO_THROW( exec_ctx2.synchronize() );

        // Should have same device ID but different streams
        EXPECT_EQ( exec_ctx1.getDeviceId(), exec_ctx2.getDeviceId() );
        EXPECT_NE( exec_ctx1.getStream(), exec_ctx2.getStream() );
    }

    TEST_F( CudaExecutionContextTest, MultipleDevices ) {
        if (device_count_ < 2) {
            GTEST_SUCCEED() << "Multi-GPU tests require at least 2 CUDA devices";
            return;
        }

        ExecutionContext<DeviceType::Cuda> exec_ctx0( 0 );
        ExecutionContext<DeviceType::Cuda> exec_ctx1( 1 );

        EXPECT_EQ( exec_ctx0.getDeviceId(), 0 );
        EXPECT_EQ( exec_ctx1.getDeviceId(), 1 );

        EXPECT_NO_THROW( exec_ctx0.synchronize() );
        EXPECT_NO_THROW( exec_ctx1.synchronize() );
    }

    // ============================================================================
    // Shared Pointer Tests
    // ============================================================================

    TEST_F( CudaExecutionContextTest, SharedPointerConstruction ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        auto exec_ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        ASSERT_NE( exec_ctx, nullptr );
        EXPECT_EQ( exec_ctx->getDeviceType(), DeviceType::Cuda );
        EXPECT_NO_THROW( exec_ctx->synchronize() );
        EXPECT_NE( exec_ctx->getStream(), nullptr );
    }

    TEST_F( CudaExecutionContextTest, SharedPointerVector ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        std::vector<std::shared_ptr<ExecutionContext<DeviceType::Cuda>>> contexts;

        for (int i = 0; i < 10; ++i) {
            contexts.push_back( std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 ) );
        }

        EXPECT_EQ( contexts.size(), 10 );

        for (auto& ctx : contexts) {
            ASSERT_NE( ctx, nullptr );
            EXPECT_EQ( ctx->getDeviceType(), DeviceType::Cuda );
            EXPECT_NO_THROW( ctx->synchronize() );
            EXPECT_NE( ctx->getStream(), nullptr );
        }
    }

    // ============================================================================
    // Type Alias Tests
    // ============================================================================

    TEST_F( CudaExecutionContextTest, TypeAlias ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        CudaExecutionContext cuda_ctx( 0 );

        EXPECT_EQ( cuda_ctx.getDeviceType(), DeviceType::Cuda );
        EXPECT_TRUE( cuda_ctx.isCudaDevice() );
        EXPECT_FALSE( cuda_ctx.isCpuDevice() );
    }

    // ============================================================================
    // Resource Management Tests
    // ============================================================================

    TEST_F( CudaExecutionContextTest, DeviceOutlivesContext ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        std::shared_ptr<ComputeDevice> device;

        {
            ExecutionContext<DeviceType::Cuda> exec_ctx( 0 );
            device = exec_ctx.getDevice();

            ASSERT_NE( device, nullptr );
        } // exec_ctx destroyed here

        // Device should still be valid
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cuda );
        EXPECT_EQ( device->getDeviceName(), "CUDA:0" );
    }

    TEST_F( CudaExecutionContextTest, MultipleScopedContexts ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        for (int i = 0; i < 100; ++i) {
            ExecutionContext<DeviceType::Cuda> exec_ctx( 0 );
            EXPECT_NO_THROW( exec_ctx.synchronize() );
        }
    }

    // ============================================================================
    // Module Integration Pattern Tests
    // ============================================================================

    TEST_F( CudaExecutionContextTest, ModuleLikeUsage ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        // Simulate how Module<DeviceType::Cuda> would use ExecutionContext
        auto exec_ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        // Module would pass this to TensorOps
        EXPECT_NO_THROW( exec_ctx->synchronize() );
        EXPECT_EQ( exec_ctx->getDeviceType(), DeviceType::Cuda );
        EXPECT_TRUE( exec_ctx->isCudaDevice() );
        EXPECT_NE( exec_ctx->getStream(), nullptr );
    }

    TEST_F( CudaExecutionContextTest, SharedBetweenModules ) {
        if (!cuda_available_) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        // Multiple modules sharing same execution context (same stream)
        auto shared_ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        cudaStream_t stream = shared_ctx->getStream();

        // Module 1 uses context
        EXPECT_NO_THROW( shared_ctx->synchronize() );
        EXPECT_EQ( shared_ctx->getStream(), stream );

        // Module 2 uses same context (same stream)
        EXPECT_NO_THROW( shared_ctx->synchronize() );
        EXPECT_EQ( shared_ctx->getStream(), stream );
    }
}