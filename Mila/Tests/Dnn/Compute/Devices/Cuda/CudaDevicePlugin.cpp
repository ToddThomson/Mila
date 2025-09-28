#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <chrono>

import Mila;
import Compute.DeviceRegistry;
import Compute.CudaDevicePlugin;
import Compute.CudaDevice;

namespace Dnn::Compute::Devices::Tests
{
    using namespace Mila::Dnn::Compute;

    class CudaDevicePluginTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Check if CUDA is available for conditional testing
            int deviceCount;
            cudaError_t error = cudaGetDeviceCount(&deviceCount);
            has_cuda_ = (error == cudaSuccess && deviceCount > 0);
            cuda_device_count_ = has_cuda_ ? deviceCount : 0;
        }

        void TearDown() override {
            // Cleanup CUDA state if needed
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

    TEST_F(CudaDevicePluginTests, StaticInterfaceAvailability) {
        bool isAvailable = CudaDevicePlugin::isAvailable();

        if (has_cuda_) {
            EXPECT_TRUE(isAvailable);
        }
        else {
            EXPECT_FALSE(isAvailable);
        }

        // Should be consistent across multiple calls
        EXPECT_EQ(isAvailable, CudaDevicePlugin::isAvailable());
        EXPECT_EQ(isAvailable, CudaDevicePlugin::isAvailable());
    }

    TEST_F(CudaDevicePluginTests, StaticInterfaceDeviceCount) {
        int deviceCount = CudaDevicePlugin::getDeviceCount();

        if (has_cuda_) {
            EXPECT_EQ(deviceCount, cuda_device_count_);
            EXPECT_GE(deviceCount, 1);
        }
        else {
            EXPECT_EQ(deviceCount, 0);
        }

        // Should be consistent across multiple calls
        EXPECT_EQ(deviceCount, CudaDevicePlugin::getDeviceCount());
        EXPECT_EQ(deviceCount, CudaDevicePlugin::getDeviceCount());
    }

    TEST_F(CudaDevicePluginTests, StaticInterfacePluginName) {
        // Plugin name should be consistent regardless of CUDA availability
        EXPECT_EQ(CudaDevicePlugin::getPluginName(), "CUDA");
    }

    TEST_F(CudaDevicePluginTests, LegacyMethodCompatibility) {
        // Test legacy isCudaAvailable() method matches isAvailable()
        EXPECT_EQ(CudaDevicePlugin::isCudaAvailable(), CudaDevicePlugin::isAvailable());
    }

    // ============================================================================
    // Device Registration Tests
    // ============================================================================

    TEST_F(CudaDevicePluginTests, DeviceRegistration) {
        auto& registry = DeviceRegistry::instance();

        // Register CUDA devices through plugin
        EXPECT_NO_THROW(CudaDevicePlugin::registerDevices());

        if (has_cuda_) {
            // Verify CUDA devices were registered
            for (int i = 0; i < cuda_device_count_; ++i) {
                std::string deviceName = "CUDA:" + std::to_string(i);
                EXPECT_TRUE(registry.hasDevice(deviceName));

                // Verify device can be created through registry
                auto device = registry.createDevice(deviceName);
                EXPECT_NE(device, nullptr);
                EXPECT_EQ(device->getDeviceType(), DeviceType::Cuda);
                EXPECT_EQ(device->getName(), deviceName);
            }
        }
        else {
            // No CUDA devices should be registered
            EXPECT_FALSE(registry.hasDevice("CUDA:0"));
        }
    }

    TEST_F(CudaDevicePluginTests, DeviceRegistrationIdempotent) {
        auto& registry = DeviceRegistry::instance();

        // Register devices multiple times - should be safe
        EXPECT_NO_THROW(CudaDevicePlugin::registerDevices());
        EXPECT_NO_THROW(CudaDevicePlugin::registerDevices());
        EXPECT_NO_THROW(CudaDevicePlugin::registerDevices());

        if (has_cuda_) {
            // Should still have correct number of CUDA devices
            for (int i = 0; i < cuda_device_count_; ++i) {
                std::string deviceName = "CUDA:" + std::to_string(i);
                EXPECT_TRUE(registry.hasDevice(deviceName));

                // Device creation should still work
                auto device = registry.createDevice(deviceName);
                EXPECT_NE(device, nullptr);
                EXPECT_EQ(device->getDeviceType(), DeviceType::Cuda);
            }
        }
    }

    TEST_F(CudaDevicePluginTests, DeviceFactoryFunction) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA not available for device factory testing";
        }

        auto& registry = DeviceRegistry::instance();
        CudaDevicePlugin::registerDevices();

        std::string deviceName = "CUDA:0";

        // Create multiple device instances - should be independent
        auto device1 = registry.createDevice(deviceName);
        auto device2 = registry.createDevice(deviceName);

        EXPECT_NE(device1, nullptr);
        EXPECT_NE(device2, nullptr);
        EXPECT_NE(device1.get(), device2.get()); // Different instances

        // Both should have same properties
        EXPECT_EQ(device1->getDeviceType(), device2->getDeviceType());
        EXPECT_EQ(device1->getName(), device2->getName());

        // Cast to CUDA device and verify device ID
        auto cudaDevice1 = std::dynamic_pointer_cast<CudaDevice>(device1);
        auto cudaDevice2 = std::dynamic_pointer_cast<CudaDevice>(device2);

        EXPECT_NE(cudaDevice1, nullptr);
        EXPECT_NE(cudaDevice2, nullptr);
        EXPECT_EQ(cudaDevice1->getDeviceId(), 0);
        EXPECT_EQ(cudaDevice2->getDeviceId(), 0);
    }

    // ============================================================================
    // CUDA-Specific Tests
    // ============================================================================

    TEST_F(CudaDevicePluginTests, CudaRuntimeIntegration) {
        bool pluginAvailable = CudaDevicePlugin::isAvailable();
        int pluginDeviceCount = CudaDevicePlugin::getDeviceCount();

        // Plugin results should match direct CUDA runtime queries
        int runtimeDeviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&runtimeDeviceCount);
        bool runtimeAvailable = (error == cudaSuccess && runtimeDeviceCount > 0);

        EXPECT_EQ(pluginAvailable, runtimeAvailable);
        if (runtimeAvailable) {
            EXPECT_EQ(pluginDeviceCount, runtimeDeviceCount);
        }
        else {
            EXPECT_EQ(pluginDeviceCount, 0);
        }
    }

    TEST_F(CudaDevicePluginTests, DeviceCapabilityFiltering) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA not available for capability testing";
        }

        auto& registry = DeviceRegistry::instance();
        CudaDevicePlugin::registerDevices();

        // All registered devices should be usable (pass capability checks)
        for (int i = 0; i < cuda_device_count_; ++i) {
            std::string deviceName = "CUDA:" + std::to_string(i);

            if (registry.hasDevice(deviceName)) {
                auto device = registry.createDevice(deviceName);
                EXPECT_NE(device, nullptr);

                auto cudaDevice = std::dynamic_pointer_cast<CudaDevice>(device);
                EXPECT_NE(cudaDevice, nullptr);

                // Verify device properties meet minimum requirements
                const auto& props = cudaDevice->getProperties();
                
                // FIXME:
                //EXPECT_GE(props.major, 3); // Minimum compute capability 3.0
                //EXPECT_GE(props.totalGlobalMem, 1ULL << 30); // At least 1GB memory
            }
        }
    }

    TEST_F(CudaDevicePluginTests, MultipleDeviceSupport) {
        if (cuda_device_count_ < 2) {
            SUCCEED() << "Multiple CUDA devices not available for testing";
        }

        auto& registry = DeviceRegistry::instance();
        CudaDevicePlugin::registerDevices();

        // Verify each device has unique ID and properties
        std::vector<std::shared_ptr<CudaDevice>> devices;
        for (int i = 0; i < cuda_device_count_; ++i) {
            std::string deviceName = "CUDA:" + std::to_string(i);
            auto device = registry.createDevice(deviceName);
            auto cudaDevice = std::dynamic_pointer_cast<CudaDevice>(device);

            EXPECT_NE(cudaDevice, nullptr);
            EXPECT_EQ(cudaDevice->getDeviceId(), i);
            devices.push_back(cudaDevice);
        }

        // Verify all devices have unique device IDs
        for (size_t i = 0; i < devices.size(); ++i) {
            for (size_t j = i + 1; j < devices.size(); ++j) {
                EXPECT_NE(devices[i]->getDeviceId(), devices[j]->getDeviceId());
            }
        }
    }

    // ============================================================================
    // Exception Safety Tests
    // ============================================================================

    TEST_F(CudaDevicePluginTests, ExceptionSafetyRegistration) {
        // Device registration should never throw exceptions
        EXPECT_NO_THROW({
            for (int i = 0; i < 50; ++i) {
                CudaDevicePlugin::registerDevices();
            }
            });
    }

    TEST_F(CudaDevicePluginTests, ExceptionSafetyAvailabilityCheck) {
        // Availability checks should never throw
        EXPECT_NO_THROW({
            for (int i = 0; i < 100; ++i) {
                bool available = CudaDevicePlugin::isAvailable();
                bool legacyAvailable = CudaDevicePlugin::isCudaAvailable();
                EXPECT_EQ(available, legacyAvailable);
            }
            });
    }

    TEST_F(CudaDevicePluginTests, ExceptionSafetySystemQueries) {
        // System information queries should never throw
        EXPECT_NO_THROW({
            for (int i = 0; i < 10; ++i) {
                auto available = CudaDevicePlugin::isAvailable();
                auto deviceCount = CudaDevicePlugin::getDeviceCount();
                auto name = CudaDevicePlugin::getPluginName();

                // Validate consistency across calls
                EXPECT_EQ(name, "CUDA");
                EXPECT_GE(deviceCount, 0);
                EXPECT_EQ(available, (deviceCount > 0));
            }
            });
    }

    // ============================================================================
    // Edge Cases and Error Conditions
    // ============================================================================

    TEST_F(CudaDevicePluginTests, CudaUnavailableGracefulDegradation) {
        // Test behavior when CUDA is not available
        // This test validates graceful degradation without exceptions

        EXPECT_NO_THROW({
            CudaDevicePlugin::registerDevices();
            bool available = CudaDevicePlugin::isAvailable();
            int deviceCount = CudaDevicePlugin::getDeviceCount();
            std::string name = CudaDevicePlugin::getPluginName();

            // Should handle unavailability gracefully
            if (!available) {
                EXPECT_EQ(deviceCount, 0);
            }
            EXPECT_EQ(name, "CUDA");
            });
    }

    TEST_F(CudaDevicePluginTests, DeviceNamingConvention) {
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA not available for naming convention testing";
        }

        auto& registry = DeviceRegistry::instance();
        CudaDevicePlugin::registerDevices();

        // Verify device naming follows "CUDA:N" convention
        for (int i = 0; i < cuda_device_count_; ++i) {
            std::string expectedName = "CUDA:" + std::to_string(i);
            EXPECT_TRUE(registry.hasDevice(expectedName));

            auto device = registry.createDevice(expectedName);
            EXPECT_NE(device, nullptr);
            EXPECT_EQ(device->getName(), expectedName);
        }
    }

    // ============================================================================
    // Plugin Interface Consistency Tests
    // ============================================================================

    TEST_F(CudaDevicePluginTests, InterfaceConsistency) {
        // Verify plugin follows expected interface contract

        // Plugin name should be descriptive and consistent
        std::string pluginName = CudaDevicePlugin::getPluginName();
        EXPECT_FALSE(pluginName.empty());
        EXPECT_EQ(pluginName, "CUDA");

        // Availability should be deterministic
        bool available1 = CudaDevicePlugin::isAvailable();
        bool available2 = CudaDevicePlugin::isAvailable();
        EXPECT_EQ(available1, available2);

        // Device count should be stable
        int count1 = CudaDevicePlugin::getDeviceCount();
        int count2 = CudaDevicePlugin::getDeviceCount();
        EXPECT_EQ(count1, count2);

        // Availability and device count should be consistent
        EXPECT_EQ(available1, (count1 > 0));
    }

    TEST_F(CudaDevicePluginTests, PluginNameImmutable) {
        // Plugin name should never change during runtime
        std::string name1 = CudaDevicePlugin::getPluginName();
        std::string name2 = CudaDevicePlugin::getPluginName();
        std::string name3 = CudaDevicePlugin::getPluginName();

        EXPECT_EQ(name1, name2);
        EXPECT_EQ(name2, name3);
        EXPECT_EQ(name1, "CUDA");
    }

    // ============================================================================
    // Integration Tests
    // ============================================================================

    TEST_F(CudaDevicePluginTests, RegistryIntegration) {
        auto& registry = DeviceRegistry::instance();
        CudaDevicePlugin::registerDevices();

        if (has_cuda_) {
            // Verify registry can list CUDA devices
            auto availableDevices = registry.listDevices();
            for (int i = 0; i < cuda_device_count_; ++i) {
                std::string deviceName = "CUDA:" + std::to_string(i);
                EXPECT_TRUE(std::find(availableDevices.begin(), availableDevices.end(), deviceName) != availableDevices.end());
            }

            // Verify device creation through registry
            std::string firstDevice = "CUDA:0";
            EXPECT_TRUE(registry.hasDevice(firstDevice));
            auto device = registry.createDevice(firstDevice);
            EXPECT_NE(device, nullptr);

            // Verify device type matches plugin
            auto cudaDevice = std::dynamic_pointer_cast<CudaDevice>(device);
            EXPECT_NE(cudaDevice, nullptr);
            EXPECT_EQ(cudaDevice->getDeviceId(), 0);
        }
    }

    TEST_F(CudaDevicePluginTests, PluginStaticInterface) {
        // Verify all required static methods exist and work
        EXPECT_NO_THROW(CudaDevicePlugin::registerDevices());
        EXPECT_NO_THROW(CudaDevicePlugin::isAvailable());
        EXPECT_NO_THROW(CudaDevicePlugin::getDeviceCount());
        EXPECT_NO_THROW(CudaDevicePlugin::getPluginName());
        EXPECT_NO_THROW(CudaDevicePlugin::isCudaAvailable());

        // Verify return types and basic constraints
        EXPECT_FALSE(CudaDevicePlugin::getPluginName().empty());
        EXPECT_GE(CudaDevicePlugin::getDeviceCount(), 0);
        EXPECT_EQ(CudaDevicePlugin::isAvailable(), CudaDevicePlugin::isCudaAvailable());
    }

    // ============================================================================
    // Performance and Stress Tests
    // ============================================================================

    TEST_F(CudaDevicePluginTests, PerformanceConsistency) {
        // Verify plugin operations are fast and consistent
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < 1000; ++i) {
            CudaDevicePlugin::isAvailable();
            CudaDevicePlugin::getDeviceCount();
            CudaDevicePlugin::getPluginName();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Should complete quickly (less than 1 second for 1000 operations)
        EXPECT_LT(duration.count(), 1000);
    }
}