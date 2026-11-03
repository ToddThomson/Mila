/**
 * @file CpuExecutionContext.cpp
 * @brief Unit tests for CPU ExecutionContext specialization aligned with ExecutionContext interface.
 */

#include <gtest/gtest.h>
#include <memory>
#include <type_traits>

import Mila;

namespace Dnn::Compute::ExecutionContexts::Tests
{
    using namespace Mila::Dnn::Compute;

    class CpuExecutionContextTest : public ::testing::Test {
    protected:
        void SetUp() override {}
        void TearDown() override {}
    };

    // ============================================================================
    // Compile-time backend availability
    // ============================================================================

    //static_assert( hasBackend( DeviceType::Cpu ), "CPU backend must be available" );

    /*TEST_F( CpuExecutionContextTest, HasBackendRuntimeCheck ) {
        EXPECT_TRUE( hasBackend( DeviceType::Cpu ) );
    }*/

    // ============================================================================
    // Factory-based Construction Tests (must use createExecutionContext)
    // ============================================================================

    TEST_F( CpuExecutionContextTest, CreateExecutionContextSucceeds ) {
        EXPECT_NO_THROW( {
            auto ctx = createExecutionContext( Device::Cpu() );
            ASSERT_NE( ctx, nullptr );
        } );
    }

    TEST_F( CpuExecutionContextTest, CreateExecutionContextReturnsCorrectDeviceId ) {
        auto ctx = createExecutionContext( Device::Cpu() );

        ASSERT_NE( ctx, nullptr );
        EXPECT_EQ( ctx->getDeviceId(), Device::Cpu() );
    }

    // ============================================================================
    // DeviceId and Synchronization Tests (use type-erased interface)
    // ============================================================================

    TEST_F( CpuExecutionContextTest, GetDeviceIdHasCpuTypeAndIndexZero ) {
        auto ctx = createExecutionContext( Device::Cpu() );

        ASSERT_NE( ctx, nullptr );

        DeviceId id = ctx->getDeviceId();

        EXPECT_EQ( id.type, DeviceType::Cpu );
        EXPECT_EQ( id.index, 0 );
        EXPECT_EQ( id, Device::Cpu() );
    }

    TEST_F( CpuExecutionContextTest, SynchronizeNoOp ) {
        auto ctx = createExecutionContext( Device::Cpu() );

        ASSERT_NE( ctx, nullptr );
        EXPECT_NO_THROW( ctx->synchronize() );
    }

    TEST_F( CpuExecutionContextTest, MultipleSynchronizeCalls ) {
        auto ctx = createExecutionContext( Device::Cpu() );

        ASSERT_NE( ctx, nullptr );
        for ( int i = 0; i < 100; ++i ) {
            EXPECT_NO_THROW( ctx->synchronize() );
        }
    }

    // ============================================================================
    // Multiple context usage / shared pointer tests (created via factory)
    // ============================================================================

    TEST_F( CpuExecutionContextTest, MultipleIndependentContexts ) {
        auto ctx1 = createExecutionContext( Device::Cpu() );
        auto ctx2 = createExecutionContext( Device::Cpu() );

        ASSERT_NE( ctx1, nullptr );
        ASSERT_NE( ctx2, nullptr );

        EXPECT_NO_THROW( ctx1->synchronize() );
        EXPECT_NO_THROW( ctx2->synchronize() );

        EXPECT_EQ( ctx1->getDeviceId(), ctx2->getDeviceId() );
    }

    TEST_F( CpuExecutionContextTest, SharedPointerConstruction ) {
        auto ctx = createExecutionContext( Device::Cpu() );

        ASSERT_NE( ctx, nullptr );
        EXPECT_EQ( ctx->getDeviceId(), Device::Cpu() );
        EXPECT_NO_THROW( ctx->synchronize() );
    }

    TEST_F( CpuExecutionContextTest, SharedPointerVector ) {
        std::vector<std::shared_ptr<IExecutionContext>> contexts;
        for ( int i = 0; i < 10; ++i ) {
            contexts.push_back( createExecutionContext( Device::Cpu() ) );
        }

        EXPECT_EQ( contexts.size(), 10 );

        for ( auto& ctx : contexts ) {
            ASSERT_NE( ctx, nullptr );
            EXPECT_EQ( ctx->getDeviceId(), Device::Cpu() );
            EXPECT_NO_THROW( ctx->synchronize() );
        }
    }

    // ============================================================================
    // Resource Management Tests (adapted to IExecutionContext)
    // ============================================================================

    TEST_F( CpuExecutionContextTest, DeviceIdOutlivesContext ) {
        DeviceId saved_id;

        {
            auto ctx = createExecutionContext( Device::Cpu() );
            ASSERT_NE( ctx, nullptr );
            saved_id = ctx->getDeviceId();
        } // ctx destroyed here

        // DeviceId is a value type; it should remain valid after context destruction
        EXPECT_EQ( saved_id.type, DeviceType::Cpu );
        EXPECT_EQ( saved_id.index, 0 );
    }

    TEST_F( CpuExecutionContextTest, MultipleScopedContexts ) {
        for ( int i = 0; i < 100; ++i ) {
            auto ctx = createExecutionContext( Device::Cpu() );
            ASSERT_NE( ctx, nullptr );
            EXPECT_NO_THROW( ctx->synchronize() );
        }
    }

    // ============================================================================
    // Module Integration Pattern Tests (use createExecutionContext)
    // ============================================================================

    TEST_F( CpuExecutionContextTest, ModuleLikeUsage ) {
        std::shared_ptr<IExecutionContext> exec_ctx = createExecutionContext( Device::Cpu() );

        ASSERT_NE( exec_ctx, nullptr );
        EXPECT_NO_THROW( exec_ctx->synchronize() );
        EXPECT_EQ( exec_ctx->getDeviceId(), Device::Cpu() );
    }

    TEST_F( CpuExecutionContextTest, SharedBetweenModules ) {
        auto shared_ctx = createExecutionContext( Device::Cpu() );

        ASSERT_NE( shared_ctx, nullptr );
        EXPECT_NO_THROW( shared_ctx->synchronize() );
        EXPECT_EQ( shared_ctx->getDeviceId(), Device::Cpu() );
    }
}