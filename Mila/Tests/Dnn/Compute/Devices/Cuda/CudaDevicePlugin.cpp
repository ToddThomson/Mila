#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <chrono>
#include <functional>

import Mila;
import Compute.DeviceRegistry;
import Compute.DeviceRegistrar;
import Compute.CudaDevicePlugin;
import Compute.CudaDevice;

namespace Dnn::Compute::Devices::Tests
{
    using namespace Mila::Dnn::Compute;

    class CudaDevicePluginTests : public ::testing::Test {
    protected:
        void SetUp() override {
            DeviceRegistrar::instance();

            int deviceCount;
            cudaError_t error = cudaGetDeviceCount( &deviceCount );
            has_cuda_ = (error == cudaSuccess && deviceCount > 0);
            cuda_device_count_ = has_cuda_ ? deviceCount : 0;
        }

        void TearDown() override {
            if (has_cuda_) {
                cudaDeviceReset();
            }
        }

        bool has_cuda_ = false;
        int cuda_device_count_ = 0;
    };

    // ============================================================================
    // Static Interface Tests
    // ============================================================================

    TEST_F( CudaDevicePluginTests, StaticInterfaceAvailability ) {
        bool isAvailable = CudaDevicePlugin::isAvailable();

        if (has_cuda_) {
            EXPECT_TRUE( isAvailable );
        }
        else {
            EXPECT_FALSE( isAvailable );
        }

        EXPECT_EQ( isAvailable, CudaDevicePlugin::isAvailable() );
        EXPECT_EQ( isAvailable, CudaDevicePlugin::isAvailable() );
    }

    TEST_F( CudaDevicePluginTests, StaticInterfaceDeviceCount ) {
        int deviceCount = CudaDevicePlugin::getDeviceCount();

        if (has_cuda_) {
            EXPECT_EQ( deviceCount, cuda_device_count_ );
            EXPECT_GE( deviceCount, 1 );
        }
        else {
            EXPECT_EQ( deviceCount, 0 );
        }

        EXPECT_EQ( deviceCount, CudaDevicePlugin::getDeviceCount() );
        EXPECT_EQ( deviceCount, CudaDevicePlugin::getDeviceCount() );
    }

    TEST_F( CudaDevicePluginTests, StaticInterfacePluginName ) {
        EXPECT_EQ( CudaDevicePlugin::getPluginName(), "CUDA" );
    }

    TEST_F( CudaDevicePluginTests, LegacyMethodCompatibility ) {
        EXPECT_EQ( CudaDevicePlugin::isCudaAvailable(), CudaDevicePlugin::isAvailable() );
    }

    // ============================================================================
    // Device Registration Tests
    // ============================================================================

    TEST_F( CudaDevicePluginTests, DeviceRegistration ) {
        auto& registry = DeviceRegistry::instance();

        if (has_cuda_) {
            for (int i = 0; i < cuda_device_count_; ++i) {
                std::string deviceName = "CUDA:" + std::to_string( i );
                EXPECT_TRUE( registry.hasDevice( deviceName ) );

                auto device = registry.createDevice( deviceName );
                EXPECT_NE( device, nullptr );
                EXPECT_EQ( device->getDeviceType(), DeviceType::Cuda );
                EXPECT_EQ( device->getName(), deviceName );
            }
        }
        else {
            EXPECT_FALSE( registry.hasDevice( "CUDA:0" ) );
        }
    }

    TEST_F( CudaDevicePluginTests, DeviceRegistrationCallback ) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA not available for callback testing";
        }

        int callbackCount = 0;
        std::vector<std::string> registeredNames;

        auto testCallback = [&callbackCount, &registeredNames](
            const std::string& name,
            DeviceRegistry::DeviceFactory factory ) {
                callbackCount++;
                registeredNames.push_back( name );

                auto device = factory();
                EXPECT_NE( device, nullptr );
            };

        CudaDevicePlugin::registerDevices( testCallback );

        EXPECT_EQ( callbackCount, cuda_device_count_ );
        EXPECT_EQ( registeredNames.size(), static_cast<size_t>(cuda_device_count_) );

        for (int i = 0; i < cuda_device_count_; ++i) {
            std::string expectedName = "CUDA:" + std::to_string( i );
            EXPECT_TRUE( std::find( registeredNames.begin(), registeredNames.end(), expectedName ) != registeredNames.end() );
        }
    }

    TEST_F( CudaDevicePluginTests, DeviceFactoryFunction ) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA not available for device factory testing";
        }

        auto& registry = DeviceRegistry::instance();
        std::string deviceName = "CUDA:0";

        auto device1 = registry.createDevice( deviceName );
        auto device2 = registry.createDevice( deviceName );

        EXPECT_NE( device1, nullptr );
        EXPECT_NE( device2, nullptr );
        EXPECT_NE( device1.get(), device2.get() );

        EXPECT_EQ( device1->getDeviceType(), device2->getDeviceType() );
        EXPECT_EQ( device1->getName(), device2->getName() );

        auto cudaDevice1 = std::dynamic_pointer_cast<CudaDevice>(device1);
        auto cudaDevice2 = std::dynamic_pointer_cast<CudaDevice>(device2);

        EXPECT_NE( cudaDevice1, nullptr );
        EXPECT_NE( cudaDevice2, nullptr );
        EXPECT_EQ( cudaDevice1->getDeviceId(), 0 );
        EXPECT_EQ( cudaDevice2->getDeviceId(), 0 );
    }

    // ============================================================================
    // CUDA-Specific Tests
    // ============================================================================

    TEST_F( CudaDevicePluginTests, CudaRuntimeIntegration ) {
        bool pluginAvailable = CudaDevicePlugin::isAvailable();
        int pluginDeviceCount = CudaDevicePlugin::getDeviceCount();

        int runtimeDeviceCount = 0;
        cudaError_t error = cudaGetDeviceCount( &runtimeDeviceCount );
        bool runtimeAvailable = (error == cudaSuccess && runtimeDeviceCount > 0);

        EXPECT_EQ( pluginAvailable, runtimeAvailable );
        if (runtimeAvailable) {
            EXPECT_EQ( pluginDeviceCount, runtimeDeviceCount );
        }
        else {
            EXPECT_EQ( pluginDeviceCount, 0 );
        }
    }

    TEST_F( CudaDevicePluginTests, DeviceCapabilityFiltering ) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA not available for capability testing";
        }

        auto& registry = DeviceRegistry::instance();

        for (int i = 0; i < cuda_device_count_; ++i) {
            std::string deviceName = "CUDA:" + std::to_string( i );

            if (registry.hasDevice( deviceName )) {
                auto device = registry.createDevice( deviceName );
                EXPECT_NE( device, nullptr );

                auto cudaDevice = std::dynamic_pointer_cast<CudaDevice>( device );
                EXPECT_NE( cudaDevice, nullptr );
            }
        }
    }

    TEST_F( CudaDevicePluginTests, MultipleDeviceSupport ) {
        if (cuda_device_count_ < 2) {
            GTEST_SUCCEED() << "Multiple CUDA devices not available for testing";
        }

        auto& registry = DeviceRegistry::instance();

        std::vector<std::shared_ptr<CudaDevice>> devices;
        for (int i = 0; i < cuda_device_count_; ++i) {
            std::string deviceName = "CUDA:" + std::to_string( i );
            auto device = registry.createDevice( deviceName );
            auto cudaDevice = std::dynamic_pointer_cast<CudaDevice>( device );

            EXPECT_NE( cudaDevice, nullptr );
            EXPECT_EQ( cudaDevice->getDeviceId(), i );
            devices.push_back( cudaDevice );
        }

        for (size_t i = 0; i < devices.size(); ++i) {
            for (size_t j = i + 1; j < devices.size(); ++j) {
                EXPECT_NE( devices[i]->getDeviceId(), devices[j]->getDeviceId() );
            }
        }
    }

    // ============================================================================
    // Exception Safety Tests
    // ============================================================================

    TEST_F( CudaDevicePluginTests, ExceptionSafetyRegistration ) {
        auto noopCallback = []( const std::string&, DeviceRegistry::DeviceFactory ) {};

        EXPECT_NO_THROW( {
            for (int i = 0; i < 50; ++i) {
                CudaDevicePlugin::registerDevices( noopCallback );
            }
            } );
    }

    TEST_F( CudaDevicePluginTests, ExceptionSafetyAvailabilityCheck ) {
        EXPECT_NO_THROW( {
            for (int i = 0; i < 100; ++i) {
                bool available = CudaDevicePlugin::isAvailable();
                bool legacyAvailable = CudaDevicePlugin::isCudaAvailable();
                EXPECT_EQ( available, legacyAvailable );
            }
            } );
    }

    TEST_F( CudaDevicePluginTests, ExceptionSafetySystemQueries ) {
        EXPECT_NO_THROW( {
            for (int i = 0; i < 10; ++i) {
                auto available = CudaDevicePlugin::isAvailable();
                auto deviceCount = CudaDevicePlugin::getDeviceCount();
                auto name = CudaDevicePlugin::getPluginName();

                EXPECT_EQ( name, "CUDA" );
                EXPECT_GE( deviceCount, 0 );
                EXPECT_EQ( available, (deviceCount > 0) );
            }
            } );
    }

    // ============================================================================
    // Edge Cases and Error Conditions
    // ============================================================================

    TEST_F( CudaDevicePluginTests, CudaUnavailableGracefulDegradation ) {
        EXPECT_NO_THROW( {
            auto noopCallback = []( const std::string&, DeviceRegistry::DeviceFactory ) {};
            CudaDevicePlugin::registerDevices( noopCallback );

            bool available = CudaDevicePlugin::isAvailable();
            int deviceCount = CudaDevicePlugin::getDeviceCount();
            std::string name = CudaDevicePlugin::getPluginName();

            if (!available) {
                EXPECT_EQ( deviceCount, 0 );
            }
            EXPECT_EQ( name, "CUDA" );
            } );
    }

    TEST_F( CudaDevicePluginTests, DeviceNamingConvention ) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA not available for naming convention testing";
        }

        auto& registry = DeviceRegistry::instance();

        for (int i = 0; i < cuda_device_count_; ++i) {
            std::string expectedName = "CUDA:" + std::to_string( i );
            EXPECT_TRUE( registry.hasDevice( expectedName ) );

            auto device = registry.createDevice( expectedName );
            EXPECT_NE( device, nullptr );
            EXPECT_EQ( device->getName(), expectedName );
        }
    }

    // ============================================================================
    // Plugin Interface Consistency Tests
    // ============================================================================

    TEST_F( CudaDevicePluginTests, InterfaceConsistency ) {
        std::string pluginName = CudaDevicePlugin::getPluginName();
        EXPECT_FALSE( pluginName.empty() );
        EXPECT_EQ( pluginName, "CUDA" );

        bool available1 = CudaDevicePlugin::isAvailable();
        bool available2 = CudaDevicePlugin::isAvailable();
        EXPECT_EQ( available1, available2 );

        int count1 = CudaDevicePlugin::getDeviceCount();
        int count2 = CudaDevicePlugin::getDeviceCount();
        EXPECT_EQ( count1, count2 );

        EXPECT_EQ( available1, (count1 > 0) );
    }

    TEST_F( CudaDevicePluginTests, PluginNameImmutable ) {
        std::string name1 = CudaDevicePlugin::getPluginName();
        std::string name2 = CudaDevicePlugin::getPluginName();
        std::string name3 = CudaDevicePlugin::getPluginName();

        EXPECT_EQ( name1, name2 );
        EXPECT_EQ( name2, name3 );
        EXPECT_EQ( name1, "CUDA" );
    }

    // ============================================================================
    // Integration Tests
    // ============================================================================

    TEST_F( CudaDevicePluginTests, RegistryIntegration ) {
        auto& registry = DeviceRegistry::instance();

        if (has_cuda_) {
            auto availableDevices = registry.listDevices();
            for (int i = 0; i < cuda_device_count_; ++i) {
                std::string deviceName = "CUDA:" + std::to_string( i );
                EXPECT_TRUE( std::find( availableDevices.begin(), availableDevices.end(), deviceName ) != availableDevices.end() );
            }

            std::string firstDevice = "CUDA:0";
            EXPECT_TRUE( registry.hasDevice( firstDevice ) );
            auto device = registry.createDevice( firstDevice );
            EXPECT_NE( device, nullptr );

            auto cudaDevice = std::dynamic_pointer_cast<CudaDevice>( device );
            EXPECT_NE( cudaDevice, nullptr );
            EXPECT_EQ( cudaDevice->getDeviceId(), 0 );
        }
    }

    TEST_F( CudaDevicePluginTests, PluginStaticInterface ) {
        auto noopCallback = []( const std::string&, DeviceRegistry::DeviceFactory ) {};

        EXPECT_NO_THROW( CudaDevicePlugin::registerDevices( noopCallback ) );
        EXPECT_NO_THROW( CudaDevicePlugin::isAvailable() );
        EXPECT_NO_THROW( CudaDevicePlugin::getDeviceCount() );
        EXPECT_NO_THROW( CudaDevicePlugin::getPluginName() );
        EXPECT_NO_THROW( CudaDevicePlugin::isCudaAvailable() );

        EXPECT_FALSE( CudaDevicePlugin::getPluginName().empty() );
        EXPECT_GE( CudaDevicePlugin::getDeviceCount(), 0 );
        EXPECT_EQ( CudaDevicePlugin::isAvailable(), CudaDevicePlugin::isCudaAvailable() );
    }

    TEST_F( CudaDevicePluginTests, CallbackNullSafety ) {
        auto nullCallback = CudaDevicePlugin::RegistrationCallback();

        EXPECT_NO_THROW( CudaDevicePlugin::registerDevices( nullCallback ) );
    }

    // ============================================================================
    // Performance and Stress Tests
    // ============================================================================

    TEST_F( CudaDevicePluginTests, PerformanceConsistency ) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < 1000; ++i) {
            CudaDevicePlugin::isAvailable();
            CudaDevicePlugin::getDeviceCount();
            CudaDevicePlugin::getPluginName();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( end - start );

        EXPECT_LT( duration.count(), 1000 );
    }
}