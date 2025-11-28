/**
 * @file ExecutionContext.cpp
 * @brief Common integration tests for ExecutionContext template.
 */

#include <gtest/gtest.h>
#include <memory>
#include <cuda_runtime.h>

import Mila;

namespace Dnn::Compute::ExecutionContexts::Tests
{
    using namespace Mila::Dnn::Compute;

    class ExecutionContextIntegrationTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int device_count = 0;
            cudaError_t error = cudaGetDeviceCount( &device_count );
            cuda_available_ = (error == cudaSuccess && device_count > 0);
        }

        void TearDown() override {
            if (cuda_available_) {
                cudaDeviceReset();
            }
        }

        bool cuda_available_{ false };
    };

    // ============================================================================
    // Type Safety Integration Tests
    // ============================================================================

    TEST_F( ExecutionContextIntegrationTest, TypeSafetyCompileTime ) {
        // CPU context
        ExecutionContext<DeviceType::Cpu> cpu_ctx;
        EXPECT_TRUE( cpu_ctx.getDeviceType() == DeviceType::Cpu);

        // CUDA context (if available)
        if (cuda_available_) {
            ExecutionContext<DeviceType::Cuda> cuda_ctx( 0 );
            EXPECT_TRUE( cuda_ctx.getDeviceType() == DeviceType::Cuda );

            // Type mismatch would be caught at compile time:
            // ExecutionContext<DeviceType::Cpu> wrong = cuda_ctx; // Compile error!
        }

        SUCCEED();
    }

    TEST_F( ExecutionContextIntegrationTest, MixedDeviceTypes ) {
        // Can create contexts for different devices
        ExecutionContext<DeviceType::Cpu> cpu_ctx;

        if (cuda_available_) {
            ExecutionContext<DeviceType::Cuda> cuda_ctx( 0 );

            // Both should work independently
            EXPECT_NO_THROW( cpu_ctx.synchronize() );
            EXPECT_NO_THROW( cuda_ctx.synchronize() );
        }
    }
}