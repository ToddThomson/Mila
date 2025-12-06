#include <gtest/gtest.h>
#include <string>
#include <optional>

import Mila;

namespace Dnn::Compute::Devices::Tests
{
    using namespace Mila::Dnn::Compute;

    class CudaDeviceRegistrarTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Query the runtime-backed count from the registrar (calls CUDA runtime).
            runtime_device_count_ = CudaDeviceRegistrar::getDeviceCount();
            runtime_has_cuda_ = (runtime_device_count_ > 0);
        }

        // No TearDown actions required; tests exercise only the registrar API.

        bool runtime_has_cuda_;
        int runtime_device_count_ = 0;
    };

    // ============================================================================
    // Registrar static API tests (aligns with Compute.CudaDeviceRegistrar)
    // ============================================================================

    TEST_F( CudaDeviceRegistrarTests, AvailabilityMatchesCudaRuntime )
    {
        bool registrarAvailable = (CudaDeviceRegistrar::getDeviceCount() > 0);

        // Registrar availability should match the CUDA runtime query (device count > 0).
        if ( runtime_has_cuda_ )
        {
            EXPECT_TRUE( registrarAvailable );
        }
        else
        {
            EXPECT_FALSE( registrarAvailable );
        }

        // Idempotency: repeated queries should be consistent
        EXPECT_EQ( registrarAvailable, (CudaDeviceRegistrar::getDeviceCount() > 0) );
    }

    TEST_F( CudaDeviceRegistrarTests, DeviceCountReflectsRuntime )
    {
        int registrarCount = CudaDeviceRegistrar::getDeviceCount();

        // Registrar queries the CUDA runtime directly; they must match.
        EXPECT_EQ( registrarCount, runtime_device_count_ );

        if ( runtime_has_cuda_ )
        {
            EXPECT_GE( registrarCount, 1 );
        }
        else
        {
            EXPECT_EQ( registrarCount, 0 );
        }

        // Idempotency
        EXPECT_EQ( registrarCount, CudaDeviceRegistrar::getDeviceCount() );
    }

    TEST_F( CudaDeviceRegistrarTests, DeviceRegistrationRegistersUsableDevices )
    {
        auto& registry = DeviceRegistry::instance();

        // Register devices with the global registry. This may register fewer devices
        // than the raw runtime count if some devices are deemed unusable.
        EXPECT_NO_THROW( CudaDeviceRegistrar::registerDevices() );

        std::size_t registeredCudaDevices = registry.getDeviceCount( DeviceType::Cuda );

        // If runtime reports zero devices, nothing should be registered.
        if ( runtime_device_count_ == 0 )
        {
            EXPECT_EQ( registeredCudaDevices, 0u );
        }
        else
        {
            // Registered count must be <= runtime count (registrar filters unusable devices).
            EXPECT_LE( registeredCudaDevices, static_cast<std::size_t>( runtime_device_count_ ) );
        }
    }

    TEST_F( CudaDeviceRegistrarTests, RepeatedQueriesAreNoThrow )
    {
        EXPECT_NO_THROW( {
            for ( int i = 0; i < 100; ++i )
            {
                (void)CudaDeviceRegistrar::getDeviceCount();
                (void)DeviceRegistry::instance().getDeviceCount( DeviceType::Cuda );
            }
        } );
    }
}