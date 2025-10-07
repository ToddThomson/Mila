#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <thread>
#include <functional>

import Mila;
import Compute.DeviceRegistry;
import Compute.DeviceRegistrar;
import Compute.CpuDevicePlugin;
import Compute.CpuDevice;

namespace Dnn::Compute::Devices::Tests
{
    using namespace Mila::Dnn::Compute;

    class CpuDevicePluginTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Ensure devices are registered before tests
            DeviceRegistrar::instance();
        }

        void TearDown() override {
            // Cleanup if needed
        }
    };

    // ============================================================================
    // Static Interface Tests
    // ============================================================================

    TEST_F( CpuDevicePluginTests, StaticInterfaceAvailability ) {
        EXPECT_TRUE( CpuDevicePlugin::isAvailable() );
    }

    TEST_F( CpuDevicePluginTests, StaticInterfaceDeviceCount ) {
        EXPECT_EQ( CpuDevicePlugin::getDeviceCount(), 1 );
    }

    TEST_F( CpuDevicePluginTests, StaticInterfacePluginName ) {
        EXPECT_EQ( CpuDevicePlugin::getPluginName(), "CPU" );
    }

    // ============================================================================
    // Device Registration Tests
    // ============================================================================

    TEST_F( CpuDevicePluginTests, DeviceRegistration ) {
        auto& registry = DeviceRegistry::instance();

        EXPECT_TRUE( registry.hasDevice( "CPU" ) );

        auto device = registry.createDevice( "CPU" );
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( device->getName(), "CPU" );
    }

    TEST_F( CpuDevicePluginTests, DeviceRegistrationCallback ) {
        bool callbackInvoked = false;
        std::string registeredName;

        auto testCallback = [&callbackInvoked, &registeredName](
            const std::string& name,
            DeviceRegistry::DeviceFactory factory ) {
                callbackInvoked = true;
                registeredName = name;

                auto device = factory();
                EXPECT_NE( device, nullptr );
            };

        CpuDevicePlugin::registerDevices( testCallback );

        EXPECT_TRUE( callbackInvoked );
        EXPECT_EQ( registeredName, "CPU" );
    }

    TEST_F( CpuDevicePluginTests, DeviceFactoryFunction ) {
        auto& registry = DeviceRegistry::instance();

        auto device1 = registry.createDevice( "CPU" );
        auto device2 = registry.createDevice( "CPU" );

        EXPECT_NE( device1, nullptr );
        EXPECT_NE( device2, nullptr );
        EXPECT_NE( device1.get(), device2.get() );

        EXPECT_EQ( device1->getDeviceType(), device2->getDeviceType() );
        EXPECT_EQ( device1->getName(), device2->getName() );
    }

    // ============================================================================
    // System Information Tests
    // ============================================================================

    TEST_F( CpuDevicePluginTests, LogicalCoreCount ) {
        unsigned int coreCount = CpuDevicePlugin::getLogicalCoreCount();

        EXPECT_GE( coreCount, 1u );
        EXPECT_LE( coreCount, 1024u );

        unsigned int stdCoreCount = std::thread::hardware_concurrency();
        if (stdCoreCount > 0) {
            EXPECT_EQ( coreCount, stdCoreCount );
        }
        else {
            EXPECT_EQ( coreCount, 1u );
        }
    }

    TEST_F( CpuDevicePluginTests, AvailableMemory ) {
        size_t availableMemory = CpuDevicePlugin::getAvailableMemory();
        EXPECT_EQ( availableMemory, 0u );
    }

    // ============================================================================
    // Exception Safety Tests
    // ============================================================================

    TEST_F( CpuDevicePluginTests, ExceptionSafetyRegistration ) {
        auto noopCallback = []( const std::string&, DeviceRegistry::DeviceFactory ) {};

        EXPECT_NO_THROW( {
            for (int i = 0; i < 100; ++i) {
                CpuDevicePlugin::registerDevices( noopCallback );
            }
            } );
    }

    TEST_F( CpuDevicePluginTests, ExceptionSafetyAvailabilityCheck ) {
        EXPECT_NO_THROW( {
            for (int i = 0; i < 100; ++i) {
                bool available = CpuDevicePlugin::isAvailable();
                EXPECT_TRUE( available );
            }
            } );
    }

    TEST_F( CpuDevicePluginTests, ExceptionSafetySystemQueries ) {
        EXPECT_NO_THROW( {
            for (int i = 0; i < 10; ++i) {
                auto coreCount = CpuDevicePlugin::getLogicalCoreCount();
                auto memory = CpuDevicePlugin::getAvailableMemory();
                auto name = CpuDevicePlugin::getPluginName();
                auto deviceCount = CpuDevicePlugin::getDeviceCount();

                EXPECT_GE( coreCount, 1u );
                EXPECT_EQ( name, "CPU" );
                EXPECT_EQ( deviceCount, 1 );
            }
            } );
    }

    // ============================================================================
    // Plugin Interface Consistency Tests
    // ============================================================================

    TEST_F( CpuDevicePluginTests, InterfaceConsistency ) {
        std::string pluginName = CpuDevicePlugin::getPluginName();
        EXPECT_FALSE( pluginName.empty() );
        EXPECT_EQ( pluginName, "CPU" );

        EXPECT_TRUE( CpuDevicePlugin::isAvailable() );
        EXPECT_TRUE( CpuDevicePlugin::isAvailable() );

        EXPECT_EQ( CpuDevicePlugin::getDeviceCount(), 1 );
        EXPECT_EQ( CpuDevicePlugin::getDeviceCount(), 1 );
    }

    TEST_F( CpuDevicePluginTests, PluginNameImmutable ) {
        std::string name1 = CpuDevicePlugin::getPluginName();
        std::string name2 = CpuDevicePlugin::getPluginName();
        std::string name3 = CpuDevicePlugin::getPluginName();

        EXPECT_EQ( name1, name2 );
        EXPECT_EQ( name2, name3 );
        EXPECT_EQ( name1, "CPU" );
    }

    // ============================================================================
    // CPU-Specific Behavior Tests
    // ============================================================================

    TEST_F( CpuDevicePluginTests, CpuAlwaysAvailable ) {
        EXPECT_TRUE( CpuDevicePlugin::isAvailable() );

        for (int i = 0; i < 50; ++i) {
            EXPECT_TRUE( CpuDevicePlugin::isAvailable() );
        }
    }

    TEST_F( CpuDevicePluginTests, SingleLogicalDevice ) {
        EXPECT_EQ( CpuDevicePlugin::getDeviceCount(), 1 );

        unsigned int coreCount = CpuDevicePlugin::getLogicalCoreCount();
        EXPECT_GE( coreCount, 1u );
        EXPECT_EQ( CpuDevicePlugin::getDeviceCount(), 1 );
    }

    TEST_F( CpuDevicePluginTests, DeviceCreationConsistency ) {
        auto& registry = DeviceRegistry::instance();

        std::vector<std::shared_ptr<ComputeDevice>> devices;
        for (int i = 0; i < 10; ++i) {
            auto device = registry.createDevice( "CPU" );
            EXPECT_NE( device, nullptr );
            devices.push_back( device );
        }

        for (const auto& device : devices) {
            EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
            EXPECT_EQ( device->getName(), "CPU" );
        }

        for (size_t i = 0; i < devices.size(); ++i) {
            for (size_t j = i + 1; j < devices.size(); ++j) {
                EXPECT_NE( devices[i].get(), devices[j].get() );
            }
        }
    }

    // ============================================================================
    // Integration Tests
    // ============================================================================

    TEST_F( CpuDevicePluginTests, RegistryIntegration ) {
        auto& registry = DeviceRegistry::instance();

        auto availableDevices = registry.listDevices();
        EXPECT_TRUE( std::find( availableDevices.begin(), availableDevices.end(), "CPU" ) != availableDevices.end() );

        EXPECT_TRUE( registry.hasDevice( "CPU" ) );
        auto device = registry.createDevice( "CPU" );
        EXPECT_NE( device, nullptr );

        auto cpuDevice = std::dynamic_pointer_cast<CpuDevice>(device);
        EXPECT_NE( cpuDevice, nullptr );
    }

    TEST_F( CpuDevicePluginTests, PluginStaticInterface ) {
        auto noopCallback = []( const std::string&, DeviceRegistry::DeviceFactory ) {};

        EXPECT_NO_THROW( CpuDevicePlugin::registerDevices( noopCallback ) );
        EXPECT_NO_THROW( CpuDevicePlugin::isAvailable() );
        EXPECT_NO_THROW( CpuDevicePlugin::getDeviceCount() );
        EXPECT_NO_THROW( CpuDevicePlugin::getPluginName() );
        EXPECT_NO_THROW( CpuDevicePlugin::getLogicalCoreCount() );
        EXPECT_NO_THROW( CpuDevicePlugin::getAvailableMemory() );

        EXPECT_TRUE( CpuDevicePlugin::isAvailable() );
        EXPECT_EQ( CpuDevicePlugin::getDeviceCount(), 1 );
        EXPECT_FALSE( CpuDevicePlugin::getPluginName().empty() );
        EXPECT_GE( CpuDevicePlugin::getLogicalCoreCount(), 1u );
    }

    TEST_F( CpuDevicePluginTests, CallbackNullSafety ) {
        auto nullCallback = CpuDevicePlugin::RegistrationCallback();

        EXPECT_NO_THROW( CpuDevicePlugin::registerDevices( nullCallback ) );
    }
}