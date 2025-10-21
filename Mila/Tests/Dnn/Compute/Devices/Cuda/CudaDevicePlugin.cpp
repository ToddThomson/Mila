
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <string>
#include <optional>

import Mila;

import Compute.CudaDevicePlugin;

namespace Dnn::Compute::Devices::Tests
{
    using namespace Mila::Dnn::Compute;

    class CudaDevicePluginTests : public ::testing::Test {
    protected:
        void SetUp() override {
            int deviceCount = 0;
            cudaError_t error = cudaGetDeviceCount( &deviceCount );
            runtime_has_cuda_ = (error == cudaSuccess && deviceCount > 0);
            runtime_device_count_ = runtime_has_cuda_ ? deviceCount : 0;
        }

        // No TearDown actions required; tests exercise only the plugin API.

        bool runtime_has_cuda_ = false;
        int runtime_device_count_ = 0;
    };

    // ============================================================================
    // Plugin static API tests (only test Compute.CudaDevicePlugin:: API)
    // ============================================================================

    TEST_F( CudaDevicePluginTests, PluginNameIsConstant ) {
        EXPECT_EQ( CudaDevicePlugin::getPluginName(), "CUDA" );
    }

    TEST_F( CudaDevicePluginTests, AvailabilityMatchesCudaRuntime ) {
        bool pluginAvailable = CudaDevicePlugin::isAvailable();

        // Plugin availability should match the CUDA runtime query (device count > 0).
        if (runtime_has_cuda_)
        {
            EXPECT_TRUE( pluginAvailable );
        }
        else
        {
            EXPECT_FALSE( pluginAvailable );
        }

        // Idempotency
        EXPECT_EQ( pluginAvailable, CudaDevicePlugin::isAvailable() );
        EXPECT_EQ( pluginAvailable, CudaDevicePlugin::isCudaAvailable() );
    }

    TEST_F( CudaDevicePluginTests, DeviceCountReflectsRuntime ) {
        int pluginCount = CudaDevicePlugin::getDeviceCount();

        if (runtime_has_cuda_)
        {
            EXPECT_EQ( pluginCount, runtime_device_count_ );
            EXPECT_GE( pluginCount, 1 );
        }
        else
        {
            EXPECT_EQ( pluginCount, 0 );
        }

        // Idempotency
        EXPECT_EQ( pluginCount, CudaDevicePlugin::getDeviceCount() );
    }

    TEST_F( CudaDevicePluginTests, GetDeviceFactoryConsistency ) {
        auto factoryOpt = CudaDevicePlugin::getDeviceFactory();

        // If the plugin returned a factory, CUDA must be available.
        if (factoryOpt.has_value())
        {
            EXPECT_TRUE( CudaDevicePlugin::isAvailable() );
            // The returned function object must be non-empty
            EXPECT_TRUE( static_cast<bool>(factoryOpt.value()) );
        }
        else
        {
            // If no factory was returned, the plugin may be unavailable
            // or no usable devices were found. In the case runtime reports
            // no devices, plugin must not provide a factory.
            if (!runtime_has_cuda_)
            {
                EXPECT_FALSE( factoryOpt.has_value() );
            }
        }
    }

    TEST_F( CudaDevicePluginTests, RepeatedCallsAreNoThrow ) {
        EXPECT_NO_THROW( {
            for (int i = 0; i < 100; ++i)
 {
(void)CudaDevicePlugin::getPluginName();
(void)CudaDevicePlugin::isAvailable();
(void)CudaDevicePlugin::getDeviceCount();
(void)CudaDevicePlugin::isCudaAvailable();
(void)CudaDevicePlugin::getDeviceFactory();
}
            } );
    }
}