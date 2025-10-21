#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <thread>
#include <functional>

import Mila;

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

        EXPECT_TRUE( registry.hasDeviceType( "CPU" ) );

        auto device = registry.getDevice( "CPU" );
        
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( device->getDeviceName(), "CPU" );
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

    

    // ============================================================================
    // Integration Tests
    // ============================================================================

    

    TEST_F( CpuDevicePluginTests, PluginStaticInterface ) {
        
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

    
}