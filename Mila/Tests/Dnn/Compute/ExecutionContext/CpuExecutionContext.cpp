/**
 * @file CpuExecutionContext.cpp
 * @brief Unit tests for CPU ExecutionContext specialization.
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

import Mila;

namespace Dnn::Compute::ExecutionContexts::Tests
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Test fixture for CPU ExecutionContext tests.
     */
    class CpuExecutionContextTest : public ::testing::Test {
    protected:
        void SetUp() override {}
        void TearDown() override {}
    };

    // ============================================================================
    // Construction Tests
    // ============================================================================

    TEST_F( CpuExecutionContextTest, DefaultConstruction ) {
        EXPECT_NO_THROW( {
            ExecutionContext<DeviceType::Cpu> exec_ctx;
            } );
    }

    // ============================================================================
    // Device Properties Tests
    // ============================================================================

    TEST_F( CpuExecutionContextTest, GetDeviceType ) {
        ExecutionContext<DeviceType::Cpu> exec_ctx;

        EXPECT_EQ( exec_ctx.getDeviceType(), DeviceType::Cpu );
    }

    TEST_F( CpuExecutionContextTest, GetDeviceName ) {
        ExecutionContext<DeviceType::Cpu> exec_ctx;

        EXPECT_EQ( exec_ctx.getDeviceName(), "CPU" );
    }

    TEST_F( CpuExecutionContextTest, GetDeviceId ) {
        ExecutionContext<DeviceType::Cpu> exec_ctx;

        // CPU always returns -1 for device ID
        EXPECT_EQ( exec_ctx.getDeviceId(), -1 );
    }

    TEST_F( CpuExecutionContextTest, GetDevice ) {
        ExecutionContext<DeviceType::Cpu> exec_ctx;
        auto device = exec_ctx.getDevice();

        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( device->getDeviceName(), "CPU" );
    }

    // ============================================================================
    // Compile-Time Properties Tests
    // ============================================================================

    TEST_F( CpuExecutionContextTest, ConstexprDeviceType ) {
        //using CpuContext = ExecutionContext<DeviceType::Cpu>;

        // Verify compile-time constants
        /*static_assert(CpuContext::getDeviceType() == DeviceType::Cpu);
        static_assert(CpuContext::isCpuDevice());
        static_assert(!CpuContext::isCudaDevice());*/
    }

    TEST_F( CpuExecutionContextTest, DeviceTypeChecks ) {
        ExecutionContext<DeviceType::Cpu> exec_ctx;

        EXPECT_TRUE( exec_ctx.getDeviceType() == DeviceType::Cpu );
    }

    // ============================================================================
    // Synchronization Tests
    // ============================================================================

    TEST_F( CpuExecutionContextTest, SynchronizeNoOp ) {
        ExecutionContext<DeviceType::Cpu> exec_ctx;

        // CPU synchronize is no-op, should not throw
        EXPECT_NO_THROW( exec_ctx.synchronize() );
    }

    TEST_F( CpuExecutionContextTest, MultipleSynchronizeCalls ) {
        ExecutionContext<DeviceType::Cpu> exec_ctx;

        // Multiple calls should work without issue
        for (int i = 0; i < 100; ++i) {
            EXPECT_NO_THROW( exec_ctx.synchronize() );
        }
    }

    // ============================================================================
    // Multiple Context Tests
    // ============================================================================

    TEST_F( CpuExecutionContextTest, MultipleIndependentContexts ) {
        ExecutionContext<DeviceType::Cpu> exec_ctx1;
        ExecutionContext<DeviceType::Cpu> exec_ctx2;

        // Both should be valid and independent
        EXPECT_NO_THROW( exec_ctx1.synchronize() );
        EXPECT_NO_THROW( exec_ctx2.synchronize() );

        // Both should have same device characteristics
        EXPECT_EQ( exec_ctx1.getDeviceType(), exec_ctx2.getDeviceType() );
        EXPECT_EQ( exec_ctx1.getDeviceName(), exec_ctx2.getDeviceName() );
        EXPECT_EQ( exec_ctx1.getDeviceId(), exec_ctx2.getDeviceId() );
    }

    /*TEST_F( CpuExecutionContextTest, VectorOfContexts ) {
        std::vector<ExecutionContext<DeviceType::Cpu>> contexts;

        for (int i = 0; i < 10; ++i) {
            contexts.emplace_back();
        }

        EXPECT_EQ( contexts.size(), 10 );

        for (auto& ctx : contexts) {
            EXPECT_EQ( ctx.getDeviceType(), DeviceType::Cpu );
            EXPECT_NO_THROW( ctx.synchronize() );
        }
    }*/

    // ============================================================================
    // Shared Pointer Tests
    // ============================================================================

    TEST_F( CpuExecutionContextTest, SharedPointerConstruction ) {
        auto exec_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        ASSERT_NE( exec_ctx, nullptr );
        EXPECT_EQ( exec_ctx->getDeviceType(), DeviceType::Cpu );
        EXPECT_NO_THROW( exec_ctx->synchronize() );
    }

    TEST_F( CpuExecutionContextTest, SharedPointerVector ) {
        std::vector<std::shared_ptr<ExecutionContext<DeviceType::Cpu>>> contexts;

        for (int i = 0; i < 10; ++i) {
            contexts.push_back( std::make_shared<ExecutionContext<DeviceType::Cpu>>() );
        }

        EXPECT_EQ( contexts.size(), 10 );

        for (auto& ctx : contexts) {
            ASSERT_NE( ctx, nullptr );
            EXPECT_EQ( ctx->getDeviceType(), DeviceType::Cpu );
            EXPECT_NO_THROW( ctx->synchronize() );
        }
    }

    // ============================================================================
    // Type Alias Tests
    // ============================================================================

    TEST_F( CpuExecutionContextTest, TypeAlias ) {
        CpuExecutionContext cpu_ctx;

        EXPECT_EQ( cpu_ctx.getDeviceType(), DeviceType::Cpu );
    }

    // ============================================================================
    // Resource Management Tests
    // ============================================================================

    TEST_F( CpuExecutionContextTest, DeviceOutlivesContext ) {
        std::shared_ptr<ComputeDevice> device;

        {
            ExecutionContext<DeviceType::Cpu> exec_ctx;
            device = exec_ctx.getDevice();

            ASSERT_NE( device, nullptr );
        } // exec_ctx destroyed here

        // Device should still be valid due to shared ownership
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( device->getDeviceName(), "CPU" );
    }

    TEST_F( CpuExecutionContextTest, MultipleScopedContexts ) {
        for (int i = 0; i < 100; ++i) {
            ExecutionContext<DeviceType::Cpu> exec_ctx;
            EXPECT_NO_THROW( exec_ctx.synchronize() );
        }
    }

    // ============================================================================
    // Module Integration Pattern Tests
    // ============================================================================

    TEST_F( CpuExecutionContextTest, ModuleLikeUsage ) {
        // Simulate how Module<DeviceType::Cpu> would use ExecutionContext
        auto exec_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        // Module would pass this to TensorOps
        EXPECT_NO_THROW( exec_ctx->synchronize() );
        EXPECT_EQ( exec_ctx->getDeviceType(), DeviceType::Cpu );

    }

    TEST_F( CpuExecutionContextTest, SharedBetweenModules ) {
        // Multiple modules sharing same execution context
        auto shared_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        // Module 1 uses context
        EXPECT_NO_THROW( shared_ctx->synchronize() );

        // Module 2 uses same context
        EXPECT_NO_THROW( shared_ctx->synchronize() );
    }

    
}