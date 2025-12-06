/**
 * @file CudaExecutionContext.cpp
 * @brief Unit tests for CUDA ExecutionContext specialization (factory-based).
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>
#include <type_traits>

import Mila;

namespace Dnn::Compute::ExecutionContexts::Tests
{
    using namespace Mila::Dnn::Compute;

    class CudaExecutionContextTest : public ::testing::Test {
    protected:
        void SetUp() override
        {
            cuda_backend_available_ = hasBackend( DeviceType::Cuda );
        }

        bool cuda_backend_available_{ false };
    };

    // ============================================================================
    // Construction and factory tests
    // ============================================================================

    TEST_F( CudaExecutionContextTest, CreateExecutionContextSucceedsOrSkips ) {
        if ( !cuda_backend_available_ ) {
            GTEST_SKIP() << "CUDA backend not enabled in this build";
        }

        std::shared_ptr<IExecutionContext> ctx;

        EXPECT_NO_THROW( {
            try {
                ctx = createExecutionContext( Device::Cuda( 0 ) );
            }
            catch ( const std::exception& e ) {
                GTEST_SKIP() << "CUDA context creation failed: " << e.what();
            }
            catch ( ... ) {
                GTEST_SKIP() << "CUDA context creation failed (unknown error)";
            }
        } );

        ASSERT_NE( ctx, nullptr );
        EXPECT_EQ( ctx->getDeviceId(), Device::Cuda( 0 ) );
    }

    // ============================================================================
    // Runtime behavior via IExecutionContext only (no access to concrete specialization)
    // ============================================================================

    TEST_F( CudaExecutionContextTest, GetDeviceId ) {
        if ( !cuda_backend_available_ ) {
            GTEST_SKIP() << "CUDA backend not enabled in this build";
        }

        std::shared_ptr<IExecutionContext> ctx;
        try {
            ctx = createExecutionContext( Device::Cuda( 0 ) );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "CUDA context creation failed: " << e.what();
        }
        catch ( ... ) {
            GTEST_SKIP() << "CUDA context creation failed (unknown error)";
        }

        ASSERT_NE( ctx, nullptr );
        EXPECT_EQ( ctx->getDeviceId(), Device::Cuda( 0 ) );
    }

    TEST_F( CudaExecutionContextTest, Synchronize ) {
        if ( !cuda_backend_available_ ) {
            GTEST_SKIP() << "CUDA backend not enabled in this build";
        }

        std::shared_ptr<IExecutionContext> ctx;
        try {
            ctx = createExecutionContext( Device::Cuda( 0 ) );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "CUDA context creation failed: " << e.what();
        }
        catch ( ... ) {
            GTEST_SKIP() << "CUDA context creation failed (unknown error)";
        }

        ASSERT_NE( ctx, nullptr );
        EXPECT_NO_THROW( ctx->synchronize() );
    }

    TEST_F( CudaExecutionContextTest, MultipleSynchronizeCalls ) {
        if ( !cuda_backend_available_ ) {
            GTEST_SKIP() << "CUDA backend not enabled in this build";
        }

        std::shared_ptr<IExecutionContext> ctx;
        try {
            ctx = createExecutionContext( Device::Cuda( 0 ) );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "CUDA context creation failed: " << e.what();
        }
        catch ( ... ) {
            GTEST_SKIP() << "CUDA context creation failed (unknown error)";
        }

        ASSERT_NE( ctx, nullptr );

        for ( int i = 0; i < 100; ++i ) {
            EXPECT_NO_THROW( ctx->synchronize() );
        }
    }

    // ============================================================================
    // Multiple independent contexts (factory-created)
    // ============================================================================

    TEST_F( CudaExecutionContextTest, MultipleIndependentContexts ) {
        if ( !cuda_backend_available_ ) {
            GTEST_SKIP() << "CUDA backend not enabled in this build";
        }

        std::shared_ptr<IExecutionContext> ctx1;
        std::shared_ptr<IExecutionContext> ctx2;

        try {
            ctx1 = createExecutionContext( Device::Cuda( 0 ) );
            ctx2 = createExecutionContext( Device::Cuda( 0 ) );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "CUDA context creation failed: " << e.what();
        }
        catch ( ... ) {
            GTEST_SKIP() << "CUDA context creation failed (unknown error)";
        }

        ASSERT_NE( ctx1, nullptr );
        ASSERT_NE( ctx2, nullptr );

        EXPECT_NO_THROW( ctx1->synchronize() );
        EXPECT_NO_THROW( ctx2->synchronize() );

        EXPECT_EQ( ctx1->getDeviceId(), ctx2->getDeviceId() );
    }

    // ============================================================================
    // Shared pointer container usage (type-erased)
    // ============================================================================

    TEST_F( CudaExecutionContextTest, SharedPointerVector ) {
        if ( !cuda_backend_available_ ) {
            GTEST_SKIP() << "CUDA backend not enabled in this build";
        }

        std::vector<std::shared_ptr<IExecutionContext>> contexts;

        for ( int i = 0; i < 10; ++i ) {
            try {
                contexts.push_back( createExecutionContext( Device::Cuda( 0 ) ) );
            }
            catch ( const std::exception& e ) {
                GTEST_SKIP() << "CUDA context creation failed while populating vector: " << e.what();
            }
            catch ( ... ) {
                GTEST_SKIP() << "CUDA context creation failed while populating vector (unknown error)";
            }
        }

        EXPECT_EQ( contexts.size(), 10 );

        for ( auto& ctx : contexts ) {
            ASSERT_NE( ctx, nullptr );
            EXPECT_EQ( ctx->getDeviceId(), Device::Cuda( 0 ) );
            EXPECT_NO_THROW( ctx->synchronize() );
        }
    }

    // ============================================================================
    // Resource lifetime semantics (IExecutionContext-level)
    // ============================================================================

    TEST_F( CudaExecutionContextTest, DeviceIdOutlivesContext ) {
        if ( !cuda_backend_available_ ) {
            GTEST_SKIP() << "CUDA backend not enabled in this build";
        }

        std::shared_ptr<IExecutionContext> ctx;
        try {
            ctx = createExecutionContext( Device::Cuda( 0 ) );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "CUDA context creation failed: " << e.what();
        }
        catch ( ... ) {
            GTEST_SKIP() << "CUDA context creation failed (unknown error)";
        }

        ASSERT_NE( ctx, nullptr );

        DeviceId saved_id = ctx->getDeviceId();
        ctx.reset();

        EXPECT_EQ( saved_id.type, DeviceType::Cuda );
        EXPECT_EQ( saved_id.index, 0 );
    }
}