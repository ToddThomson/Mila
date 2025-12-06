/**
 * @file ExecutionContext.cpp
 * @brief Common integration tests for ExecutionContext template.
 */

#include <gtest/gtest.h>
#include <memory>

import Mila;

namespace Dnn::Compute::ExecutionContexts::Tests
{
    using namespace Mila::Dnn::Compute;

    class ExecutionContextIntegrationTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            int device_count = getDeviceCount( DeviceType::Cuda );
            cuda_available_ = (device_count > 0);
        }

        bool cuda_available_{ false };
    };

    // ============================================================================
    // Type Safety Integration Tests
    // ============================================================================

    TEST_F( ExecutionContextIntegrationTest, TypeSafetyCompileTime )
    {
        // CPU context
        auto cpu_ctx = createExecutionContext( Device::Cpu() );
        EXPECT_TRUE( cpu_ctx->getDeviceId().type == DeviceType::Cpu );

        // CUDA context (if available)
        if ( cuda_available_ )
        {
            auto cuda_ctx = createExecutionContext( Device::Cuda( 0 ) );
            EXPECT_TRUE( cuda_ctx->getDeviceId().type == DeviceType::Cuda );

            // Type mismatch would be caught at compile time:
            // ExecutionContext<DeviceType::Cpu> wrong = cuda_ctx; // Compile error!
        }

        SUCCEED();
    }

    TEST_F( ExecutionContextIntegrationTest, MixedDeviceTypes )
    {
        // Can create contexts for different devices
		auto cpu_ctx = createExecutionContext( Device::Cpu() );

        if ( cuda_available_ )
        {
            auto cuda_ctx = createExecutionContext( Device::Cuda( 0 ) );

            // Both should work independently
            EXPECT_NO_THROW( cpu_ctx->synchronize() );
            EXPECT_NO_THROW( cuda_ctx->synchronize() );
        }
    }
}