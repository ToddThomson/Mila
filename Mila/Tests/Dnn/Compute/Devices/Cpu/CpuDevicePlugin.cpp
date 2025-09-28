#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <thread>

import Mila;
import Compute.DeviceRegistry;
import Compute.CpuDevicePlugin;
import Compute.CpuDevice;

namespace Dnn::Compute::Devices::Tests
{
    using namespace Mila::Dnn::Compute;

    class CpuDevicePluginTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Clear any existing device registrations for clean test state
            auto& registry = DeviceRegistry::instance();
            // Note: DeviceRegistry would need a clear() method for proper test isolation
            // For now, tests assume registry starts clean or handles duplicates gracefully
        }

        void TearDown() override {
            // Cleanup if needed
        }
    };

    // ============================================================================
    // Static Interface Tests
    // ============================================================================

    TEST_F(CpuDevicePluginTests, StaticInterfaceAvailability) {
        // CPU should always be available as fallback device
        EXPECT_TRUE(CpuDevicePlugin::isAvailable());
    }

    TEST_F(CpuDevicePluginTests, StaticInterfaceDeviceCount) {
        // CPU plugin should always report exactly one logical device
        EXPECT_EQ(CpuDevicePlugin::getDeviceCount(), 1);
    }

    TEST_F(CpuDevicePluginTests, StaticInterfacePluginName) {
        // Plugin name should be consistent and identifiable
        EXPECT_EQ(CpuDevicePlugin::getPluginName(), "CPU");
    }

    // ============================================================================
    // Device Registration Tests
    // ============================================================================

    TEST_F(CpuDevicePluginTests, DeviceRegistration) {
        auto& registry = DeviceRegistry::instance();

        // Register CPU devices through plugin
        EXPECT_NO_THROW(CpuDevicePlugin::registerDevices());

        // Verify CPU device was registered
        EXPECT_TRUE(registry.hasDevice("CPU"));

        // Verify device can be created through registry
        auto device = registry.createDevice("CPU");
        EXPECT_NE(device, nullptr);
        EXPECT_EQ(device->getDeviceType(), DeviceType::Cpu);
        EXPECT_EQ(device->getName(), "CPU");
    }

    TEST_F(CpuDevicePluginTests, DeviceRegistrationIdempotent) {
        auto& registry = DeviceRegistry::instance();

        // Register devices multiple times - should be safe
        EXPECT_NO_THROW(CpuDevicePlugin::registerDevices());
        EXPECT_NO_THROW(CpuDevicePlugin::registerDevices());
        EXPECT_NO_THROW(CpuDevicePlugin::registerDevices());

        // Should still have exactly one CPU device
        EXPECT_TRUE(registry.hasDevice("CPU"));

        // Device creation should still work
        auto device = registry.createDevice("CPU");
        EXPECT_NE(device, nullptr);
        EXPECT_EQ(device->getDeviceType(), DeviceType::Cpu);
    }

    TEST_F(CpuDevicePluginTests, DeviceFactoryFunction) {
        auto& registry = DeviceRegistry::instance();

        // Register devices
        CpuDevicePlugin::registerDevices();

        // Create multiple device instances - should be independent
        auto device1 = registry.createDevice("CPU");
        auto device2 = registry.createDevice("CPU");

        EXPECT_NE(device1, nullptr);
        EXPECT_NE(device2, nullptr);
        EXPECT_NE(device1.get(), device2.get()); // Different instances

        // Both should have same properties
        EXPECT_EQ(device1->getDeviceType(), device2->getDeviceType());
        EXPECT_EQ(device1->getName(), device2->getName());
    }

    // ============================================================================
    // System Information Tests
    // ============================================================================

    TEST_F(CpuDevicePluginTests, LogicalCoreCount) {
        unsigned int coreCount = CpuDevicePlugin::getLogicalCoreCount();

        // Should return at least 1 core (fallback behavior)
        EXPECT_GE(coreCount, 1u);

        // Should be reasonable for modern systems (not wildly high)
        EXPECT_LE(coreCount, 1024u); // Generous upper bound for validation

        // Should match std::thread::hardware_concurrency() behavior
        unsigned int stdCoreCount = std::thread::hardware_concurrency();
        if (stdCoreCount > 0) {
            EXPECT_EQ(coreCount, stdCoreCount);
        }
        else {
            // If std::thread::hardware_concurrency() fails, plugin should return 1
            EXPECT_EQ(coreCount, 1u);
        }
    }

    TEST_F(CpuDevicePluginTests, AvailableMemory) {
        size_t availableMemory = CpuDevicePlugin::getAvailableMemory();

        // Current implementation returns 0 (not implemented)
        // This test validates the current behavior and can be updated when implemented
        EXPECT_EQ(availableMemory, 0u);

        // Future implementation should return reasonable values:
        // EXPECT_GE(availableMemory, 1ULL << 20); // At least 1MB
        // EXPECT_LE(availableMemory, 1ULL << 40); // Less than 1TB (sanity check)
    }

    // ============================================================================
    // Exception Safety Tests
    // ============================================================================

    TEST_F(CpuDevicePluginTests, ExceptionSafetyRegistration) {
        // Device registration should never throw exceptions
        // This is critical for system stability during initialization
        EXPECT_NO_THROW({
            for (int i = 0; i < 100; ++i) {
                CpuDevicePlugin::registerDevices();
            }
            });
    }

    TEST_F(CpuDevicePluginTests, ExceptionSafetyAvailabilityCheck) {
        // Availability checks should never throw
        EXPECT_NO_THROW({
            for (int i = 0; i < 100; ++i) {
                bool available = CpuDevicePlugin::isAvailable();
                EXPECT_TRUE(available); // Should always be true for CPU
            }
            });
    }

    TEST_F(CpuDevicePluginTests, ExceptionSafetySystemQueries) {
        // System information queries should never throw
        EXPECT_NO_THROW({
            for (int i = 0; i < 10; ++i) {
                auto coreCount = CpuDevicePlugin::getLogicalCoreCount();
                auto memory = CpuDevicePlugin::getAvailableMemory();
                auto name = CpuDevicePlugin::getPluginName();
                auto deviceCount = CpuDevicePlugin::getDeviceCount();

                // Validate consistency across calls
                EXPECT_GE(coreCount, 1u);
                EXPECT_EQ(name, "CPU");
                EXPECT_EQ(deviceCount, 1);
            }
            });
    }

    // ============================================================================
    // Plugin Interface Consistency Tests
    // ============================================================================

    TEST_F(CpuDevicePluginTests, InterfaceConsistency) {
        // Verify plugin follows expected interface contract

        // Plugin name should be descriptive and consistent
        std::string pluginName = CpuDevicePlugin::getPluginName();
        EXPECT_FALSE(pluginName.empty());
        EXPECT_EQ(pluginName, "CPU");

        // Availability should be deterministic
        EXPECT_TRUE(CpuDevicePlugin::isAvailable());
        EXPECT_TRUE(CpuDevicePlugin::isAvailable()); // Should be consistent

        // Device count should be stable
        EXPECT_EQ(CpuDevicePlugin::getDeviceCount(), 1);
        EXPECT_EQ(CpuDevicePlugin::getDeviceCount(), 1); // Should be consistent
    }

    TEST_F(CpuDevicePluginTests, PluginNameImmutable) {
        // Plugin name should never change during runtime
        std::string name1 = CpuDevicePlugin::getPluginName();
        std::string name2 = CpuDevicePlugin::getPluginName();
        std::string name3 = CpuDevicePlugin::getPluginName();

        EXPECT_EQ(name1, name2);
        EXPECT_EQ(name2, name3);
        EXPECT_EQ(name1, "CPU");
    }

    // ============================================================================
    // CPU-Specific Behavior Tests
    // ============================================================================

    TEST_F(CpuDevicePluginTests, CpuAlwaysAvailable) {
        // CPU should be available regardless of system state
        // This is the fundamental guarantee of CPU as fallback device
        EXPECT_TRUE(CpuDevicePlugin::isAvailable());

        // Multiple checks should consistently return true
        for (int i = 0; i < 50; ++i) {
            EXPECT_TRUE(CpuDevicePlugin::isAvailable());
        }
    }

    TEST_F(CpuDevicePluginTests, SingleLogicalDevice) {
        // CPU plugin should always report exactly one logical device
        // regardless of actual core count
        EXPECT_EQ(CpuDevicePlugin::getDeviceCount(), 1);

        // This should be true even if system has many cores
        unsigned int coreCount = CpuDevicePlugin::getLogicalCoreCount();
        EXPECT_GE(coreCount, 1u); // May be > 1
        EXPECT_EQ(CpuDevicePlugin::getDeviceCount(), 1); // Always 1 logical device
    }

    TEST_F(CpuDevicePluginTests, DeviceCreationConsistency) {
        auto& registry = DeviceRegistry::instance();
        CpuDevicePlugin::registerDevices();

        // Create multiple CPU devices - should have identical properties
        std::vector<std::shared_ptr<ComputeDevice>> devices;
        for (int i = 0; i < 10; ++i) {
            auto device = registry.createDevice("CPU");
            EXPECT_NE(device, nullptr);
            devices.push_back(device);
        }

        // All devices should have identical properties
        for (const auto& device : devices) {
            EXPECT_EQ(device->getDeviceType(), DeviceType::Cpu);
            EXPECT_EQ(device->getName(), "CPU");
        }

        // But should be different instances
        for (size_t i = 0; i < devices.size(); ++i) {
            for (size_t j = i + 1; j < devices.size(); ++j) {
                EXPECT_NE(devices[i].get(), devices[j].get());
            }
        }
    }

    // ============================================================================
    // Integration Tests
    // ============================================================================

    TEST_F(CpuDevicePluginTests, RegistryIntegration) {
        auto& registry = DeviceRegistry::instance();

        // Register CPU devices
        CpuDevicePlugin::registerDevices();

        // Verify registry can list CPU device
        auto availableDevices = registry.listDevices();
        EXPECT_TRUE(std::find(availableDevices.begin(), availableDevices.end(), "CPU") != availableDevices.end());

        // Verify device creation through registry
        EXPECT_TRUE(registry.hasDevice("CPU"));
        auto device = registry.createDevice("CPU");
        EXPECT_NE(device, nullptr);

        // Verify device type matches plugin
        auto cpuDevice = std::dynamic_pointer_cast<CpuDevice>(device);
        EXPECT_NE(cpuDevice, nullptr);
    }

    TEST_F(CpuDevicePluginTests, PluginStaticInterface) {
        // Verify all required static methods exist and work
        EXPECT_NO_THROW(CpuDevicePlugin::registerDevices());
        EXPECT_NO_THROW(CpuDevicePlugin::isAvailable());
        EXPECT_NO_THROW(CpuDevicePlugin::getDeviceCount());
        EXPECT_NO_THROW(CpuDevicePlugin::getPluginName());
        EXPECT_NO_THROW(CpuDevicePlugin::getLogicalCoreCount());
        EXPECT_NO_THROW(CpuDevicePlugin::getAvailableMemory());

        // Verify return types and basic constraints
        EXPECT_TRUE(CpuDevicePlugin::isAvailable());
        EXPECT_EQ(CpuDevicePlugin::getDeviceCount(), 1);
        EXPECT_FALSE(CpuDevicePlugin::getPluginName().empty());
        EXPECT_GE(CpuDevicePlugin::getLogicalCoreCount(), 1u);
    }
}