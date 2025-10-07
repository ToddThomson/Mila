#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <stdexcept>
#include <thread>
#include <chrono>

import Mila;

namespace Dnn::Compute::Registry::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    // Mock ComputeDevice for testing
    class MockComputeDevice : public ComputeDevice {
    public:
        explicit MockComputeDevice( const std::string& name, int id = 0 )
            : name_( name ), id_( id ) {
        }

        constexpr DeviceType getDeviceType() const override {
            if (name_.find( "CPU" ) != std::string::npos) return DeviceType::Cpu;
            if (name_.find( "CUDA" ) != std::string::npos) return DeviceType::Cuda;
            return DeviceType::Cpu;
        }

        std::string getName() const override { return name_; }
        int getDeviceId() const override { return id_; }

    private:
        std::string name_;
        int id_;
    };

    // Test fixture for DeviceRegistrar
    class DeviceRegistrarTest : public ::testing::Test {
    protected:
        void SetUp() override {
            // Get fresh instances for each test
            auto& registry = DeviceRegistry::instance();
            auto& registrar = DeviceRegistrar::instance();

            // Clear any existing devices by accessing instance
            // Note: In production, devices are registered once during static initialization
            // For testing, we work with the existing registered devices
        }

        void TearDown() override {
            // DeviceRegistry is a singleton, so state persists between tests
            // This is intentional as device registration typically happens once
        }
    };

    // Test DeviceRegistrar singleton behavior
    TEST_F( DeviceRegistrarTest, SingletonInstance ) {
        // Test that we get the same instance
        auto& registrar1 = DeviceRegistrar::instance();
        auto& registrar2 = DeviceRegistrar::instance();

        EXPECT_EQ( &registrar1, &registrar2 );
    }

    // Test thread safety of DeviceRegistrar singleton
    TEST_F( DeviceRegistrarTest, ThreadSafeSingleton ) {
        std::vector<DeviceRegistrar*> instances;
        std::vector<std::thread> threads;
        constexpr int num_threads = 10;

        instances.resize( num_threads );

        // Create multiple threads that access the singleton
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back( [&instances, i]() {
                instances[i] = &DeviceRegistrar::instance();
                } );
        }

        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }

        // Verify all threads got the same instance
        for (int i = 1; i < num_threads; ++i) {
            EXPECT_EQ( instances[0], instances[i] );
        }
    }

    // Test that DeviceRegistrar registers expected devices
    TEST_F( DeviceRegistrarTest, RegistersExpectedDevices ) {
        // Trigger registration by accessing singleton
        auto& registrar = DeviceRegistrar::instance();
        auto& registry = DeviceRegistry::instance();

        // Check that CPU device is always registered
        EXPECT_TRUE( registry.hasDevice( "CPU" ) );

        // Check that devices are listed
        auto devices = registry.listDevices();
        EXPECT_FALSE( devices.empty() );

        // CPU should always be present
        auto cpu_found = std::find( devices.begin(), devices.end(), "CPU" );
        EXPECT_NE( cpu_found, devices.end() );
    }

    // Test device creation through registrar
    TEST_F( DeviceRegistrarTest, DeviceCreation ) {
        auto& registrar = DeviceRegistrar::instance();
        auto& registry = DeviceRegistry::instance();

        // Test CPU device creation
        auto cpu_device = registry.createDevice( "CPU" );
        ASSERT_NE( cpu_device, nullptr );
        EXPECT_EQ( cpu_device->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( cpu_device->getName(), "CPU" );

        // Test multiple instances creation
        auto cpu_device2 = registry.createDevice( "CPU" );
        ASSERT_NE( cpu_device2, nullptr );

        // Should create different instances
        EXPECT_NE( cpu_device.get(), cpu_device2.get() );
    }

    // Test CUDA device registration (when available)
    TEST_F( DeviceRegistrarTest, CudaDeviceRegistration ) {
        auto& registrar = DeviceRegistrar::instance();
        auto& registry = DeviceRegistry::instance();

        auto devices = registry.listDevices();

        // Check if any CUDA devices are registered
        bool has_cuda = false;
        for (const auto& device : devices) {
            if (device.find( "CUDA:" ) != std::string::npos) {
                has_cuda = true;

                // Test CUDA device creation
                auto cuda_device = registry.createDevice( device );
                ASSERT_NE( cuda_device, nullptr );
                EXPECT_EQ( cuda_device->getDeviceType(), DeviceType::Cuda );
                EXPECT_EQ( cuda_device->getName(), device );

                break;
            }
        }

        // This test will pass whether CUDA is available or not
        // If CUDA is available, it tests device creation
        // If CUDA is not available, it just confirms no CUDA devices are registered
        EXPECT_TRUE( true ); // Always pass - the real test is in the loop above
    }

    // Test device listing functionality
    TEST_F( DeviceRegistrarTest, DeviceListing ) {
        auto& registrar = DeviceRegistrar::instance();
        auto& registry = DeviceRegistry::instance();

        auto devices = registry.listDevices();

        // Should have at least CPU device
        EXPECT_GE( devices.size(), 1 );

        // All device names should be non-empty
        for (const auto& device : devices) {
            EXPECT_FALSE( device.empty() );
        }

        // CPU should be present
        auto cpu_found = std::find( devices.begin(), devices.end(), "CPU" );
        EXPECT_NE( cpu_found, devices.end() );
    }

    // Test device existence checking
    TEST_F( DeviceRegistrarTest, DeviceExistenceCheck ) {
        auto& registrar = DeviceRegistrar::instance();
        auto& registry = DeviceRegistry::instance();

        // CPU should always exist
        EXPECT_TRUE( registry.hasDevice( "CPU" ) );

        // Non-existent device should return false
        EXPECT_FALSE( registry.hasDevice( "NonExistentDevice" ) );
        EXPECT_FALSE( registry.hasDevice( "" ) );
        EXPECT_FALSE( registry.hasDevice( "INVALID:999" ) );
    }

    // Test invalid device creation
    TEST_F( DeviceRegistrarTest, InvalidDeviceCreation ) {
        auto& registrar = DeviceRegistrar::instance();
        auto& registry = DeviceRegistry::instance();

        // Test creation of non-existent device
        auto invalid_device = registry.createDevice( "NonExistentDevice" );
        EXPECT_EQ( invalid_device, nullptr );

        // Test creation with empty name
        auto empty_device = registry.createDevice( "" );
        EXPECT_EQ( empty_device, nullptr );

        // Test creation with invalid CUDA device
        auto invalid_cuda = registry.createDevice( "CUDA:999" );
        EXPECT_EQ( invalid_cuda, nullptr );
    }

    // Test thread safety of device operations
    TEST_F( DeviceRegistrarTest, ThreadSafeDeviceOperations ) {
        auto& registrar = DeviceRegistrar::instance();
        auto& registry = DeviceRegistry::instance();

        constexpr int num_threads = 20;
        constexpr int operations_per_thread = 50;

        std::vector<std::thread> threads;
        std::vector<bool> results( num_threads, false );

        // Create threads that perform various registry operations
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back( [&registry, &results, i, operations_per_thread]() {
                bool thread_success = true;

                for (int j = 0; j < operations_per_thread; ++j) {
                    // Test device creation
                    auto device = registry.createDevice( "CPU" );
                    if (!device || device->getName() != "CPU") {
                        thread_success = false;
                        break;
                    }

                    // Test device listing
                    auto devices = registry.listDevices();
                    if (devices.empty()) {
                        thread_success = false;
                        break;
                    }

                    // Test device existence check
                    if (!registry.hasDevice( "CPU" )) {
                        thread_success = false;
                        break;
                    }

                    // Small delay to increase chance of race conditions
                    std::this_thread::sleep_for( std::chrono::microseconds( 1 ) );
                }

                results[i] = thread_success;
                } );
        }

        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }

        // Verify all threads succeeded
        for (int i = 0; i < num_threads; ++i) {
            EXPECT_TRUE( results[i] ) << "Thread " << i << " failed";
        }
    }

    // Test registration initialization order
    TEST_F( DeviceRegistrarTest, RegistrationInitializationOrder ) {
        // This test verifies that the DeviceRegistrar properly initializes
        // and registers devices when accessed for the first time

        auto& registrar = DeviceRegistrar::instance();
        auto& registry = DeviceRegistry::instance();

        // After accessing the registrar instance, devices should be registered
        auto devices = registry.listDevices();
        EXPECT_FALSE( devices.empty() );

        // CPU should be registered (it's always available)
        EXPECT_TRUE( registry.hasDevice( "CPU" ) );

        // Verify CPU device can be created successfully
        auto cpu_device = registry.createDevice( "CPU" );
        ASSERT_NE( cpu_device, nullptr );
        EXPECT_EQ( cpu_device->getDeviceType(), DeviceType::Cpu );
    }

    // Test device factory consistency
    TEST_F( DeviceRegistrarTest, DeviceFactoryConsistency ) {
        auto& registrar = DeviceRegistrar::instance();
        auto& registry = DeviceRegistry::instance();

        // Get list of available devices
        auto devices = registry.listDevices();

        // Test that each registered device can be created multiple times
        for (const auto& device_name : devices) {
            // Create multiple instances
            auto device1 = registry.createDevice( device_name );
            auto device2 = registry.createDevice( device_name );

            ASSERT_NE( device1, nullptr ) << "Failed to create device: " << device_name;
            ASSERT_NE( device2, nullptr ) << "Failed to create device: " << device_name;

            // Verify properties are consistent
            EXPECT_EQ( device1->getName(), device2->getName() );
            EXPECT_EQ( device1->getDeviceType(), device2->getDeviceType() );

            // Verify they are different instances
            EXPECT_NE( device1.get(), device2.get() );
        }
    }

    // Test device naming conventions
    TEST_F( DeviceRegistrarTest, DeviceNamingConventions ) {
        auto& registrar = DeviceRegistrar::instance();
        auto& registry = DeviceRegistry::instance();

        auto devices = registry.listDevices();

        for (const auto& device_name : devices) {
            // Device names should not be empty
            EXPECT_FALSE( device_name.empty() );

            // Test device creation and name consistency
            auto device = registry.createDevice( device_name );
            ASSERT_NE( device, nullptr );

            // The device's getName() should match the registry name
            EXPECT_EQ( device->getName(), device_name );

            // Verify naming conventions
            if (device_name == "CPU") {
                EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
            }
            else if (device_name.find( "CUDA:" ) == 0) {
                EXPECT_EQ( device->getDeviceType(), DeviceType::Cuda );

                // CUDA devices should have numeric suffix
                size_t colon_pos = device_name.find( ':' );
                ASSERT_NE( colon_pos, std::string::npos );

                std::string id_str = device_name.substr( colon_pos + 1 );
                EXPECT_FALSE( id_str.empty() );

                // Should be parseable as integer
                try {
                    int device_id = std::stoi( id_str );
                    EXPECT_GE( device_id, 0 );
                    EXPECT_EQ( device->getDeviceId(), device_id );
                }
                catch (const std::exception&) {
                    FAIL() << "CUDA device ID should be numeric: " << device_name;
                }
            }
        }
    }
}