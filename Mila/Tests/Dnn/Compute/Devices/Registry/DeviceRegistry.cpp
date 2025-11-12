#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <algorithm>

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

        std::string getDeviceName() const override { return name_; }
        int getDeviceId() const override { return id_; }

    private:
        std::string name_;
        int id_;
    };

    // Test fixture for DeviceRegistrar / DeviceRegistry
    class DeviceRegistrarTest : public ::testing::Test {
    protected:
        void SetUp() override {
            // Ensure registrar/registry are initialized (registration happens in registrar ctor)
            auto& registry = DeviceRegistry::instance();
            auto& registrar = DeviceRegistrar::instance();

            (void)registry;
            (void)registrar;
        }

        void TearDown() override {
            // DeviceRegistry is a singleton; state persists between tests by design
        }
    };

    // Test DeviceRegistrar singleton behavior
    TEST_F( DeviceRegistrarTest, SingletonInstance ) {
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

        for (int i = 0; i < num_threads; ++i)
        {
            threads.emplace_back( [&instances, i]() {
                instances[i] = &DeviceRegistrar::instance();
                } );
        }

        for (auto& thread : threads)
        {
            thread.join();
        }

        for (int i = 1; i < num_threads; ++i)
        {
            EXPECT_EQ( instances[0], instances[i] );
        }
    }

    // Test that DeviceRegistrar registers expected device types (API-aligned)
    TEST_F( DeviceRegistrarTest, RegistersExpectedDeviceTypes ) {
        auto& registry = DeviceRegistry::instance();

        // CPU factory should always be registered
        EXPECT_TRUE( registry.hasDeviceType( "CPU" ) );

        // Device types list should contain at least CPU
        auto types = registry.listDeviceTypes();
        auto cpu_it = std::find( types.begin(), types.end(), "CPU" );
        EXPECT_NE( cpu_it, types.end() );
    }

    // Test device creation through registry using getDevice (API changed)
    TEST_F( DeviceRegistrarTest, DeviceCreationAndCaching ) {
        auto& registry = DeviceRegistry::instance();

        // Create CPU device via getDevice
        auto cpu_device = registry.getDevice( "CPU" );
        ASSERT_NE( cpu_device, nullptr );
        EXPECT_EQ( cpu_device->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( cpu_device->getDeviceName(), "CPU" );

        // Repeated calls with same name should return the same cached instance
        auto cpu_device2 = registry.getDevice( "CPU" );
        ASSERT_NE( cpu_device2, nullptr );
        EXPECT_EQ( cpu_device.get(), cpu_device2.get() );
    }

    // Test CUDA device registration only via registry API (if available)
    TEST_F( DeviceRegistrarTest, CudaDeviceRegistrationIfAvailable ) {
        auto& registry = DeviceRegistry::instance();

        if (registry.hasDeviceType( "CUDA" ))
        {
            // If CUDA factory is registered, creating "CUDA:0" should not throw (but may if device invalid)
            EXPECT_NO_THROW( {
                auto cuda_device = registry.getDevice( "CUDA:0" );
                ASSERT_NE( cuda_device, nullptr );
                EXPECT_EQ( cuda_device->getDeviceType(), DeviceType::Cuda );
                } );
        }
        else
        {
            // If CUDA factory not registered, registry should not claim CUDA device type
            EXPECT_FALSE( registry.hasDeviceType( "CUDA" ) );
        }
    }

    // Test device listing after instantiation
    TEST_F( DeviceRegistrarTest, DeviceListingAfterInstantiation ) {
        auto& registry = DeviceRegistry::instance();

        // Ensure at least CPU exists by instantiating it
        auto cpu_device = registry.getDevice( "CPU" );
        ASSERT_NE( cpu_device, nullptr );

        auto devices = registry.listDevices();
        EXPECT_GE( devices.size(), 1 );

        auto cpu_it = std::find( devices.begin(), devices.end(), "CPU" );
        EXPECT_NE( cpu_it, devices.end() );

        for (const auto& dev : devices)
        {
            EXPECT_FALSE( dev.empty() );
        }
    }

    // Test getDevice throws for unknown device types / invalid names
    TEST_F( DeviceRegistrarTest, DeviceExistenceAndInvalidInputs ) {
        auto& registry = DeviceRegistry::instance();

        // Existing device should be retrievable
        EXPECT_NO_THROW( {
            auto cpu = registry.getDevice( "CPU" );
            ASSERT_NE( cpu, nullptr );
            } );

        // Non-existent device type should throw std::runtime_error
        EXPECT_THROW( registry.getDevice( "NonExistentDevice" ), std::runtime_error );
        EXPECT_THROW( registry.getDevice( "" ), std::runtime_error );
        EXPECT_THROW( registry.getDevice( "INVALID:999" ), std::runtime_error );
    }

    // Test thread safety of device operations using getDevice
    TEST_F( DeviceRegistrarTest, ThreadSafeDeviceOperations ) {
        auto& registry = DeviceRegistry::instance();

        constexpr int num_threads = 20;
        constexpr int operations_per_thread = 50;

        std::vector<std::thread> threads;
        std::vector<bool> results( num_threads, false );

        for (int i = 0; i < num_threads; ++i)
        {
            threads.emplace_back( [&registry, &results, i]() {
                bool thread_success = true;

                for (int j = 0; j < operations_per_thread; ++j)
                {
                    // Use getDevice (cached) to exercise registry concurrency
                    try
                    {
                        auto device = registry.getDevice( "CPU" );
                        if (!device || device->getDeviceName() != "CPU")
                        {
                            thread_success = false;
                            break;
                        }

                        auto devices = registry.listDevices();
                        if (devices.empty())
                        {
                            thread_success = false;
                            break;
                        }

                        if (!registry.hasDeviceType( "CPU" ))
                        {
                            thread_success = false;
                            break;
                        }
                    }
                    catch (...)
                    {
                        thread_success = false;
                        break;
                    }

                    std::this_thread::sleep_for( std::chrono::microseconds( 1 ) );
                }

                results[i] = thread_success;
                } );
        }

        for (auto& thread : threads)
        {
            thread.join();
        }

        for (int i = 0; i < num_threads; ++i)
        {
            EXPECT_TRUE( results[i] ) << "Thread " << i << " failed";
        }
    }

    // Test registration initialization order (registry/registrar cooperate)
    TEST_F( DeviceRegistrarTest, RegistrationInitializationOrder ) {
        auto& registrar = DeviceRegistrar::instance();
        auto& registry = DeviceRegistry::instance();

        auto types = registry.listDeviceTypes();
        EXPECT_FALSE( types.empty() );

        EXPECT_TRUE( registry.hasDeviceType( "CPU" ) );

        auto cpu_device = registry.getDevice( "CPU" );
        ASSERT_NE( cpu_device, nullptr );
        EXPECT_EQ( cpu_device->getDeviceType(), DeviceType::Cpu );

        (void)registrar;
    }

    // Test device factory consistency: getDevice returns same cached instance
    TEST_F( DeviceRegistrarTest, DeviceFactoryConsistency ) {
        auto& registry = DeviceRegistry::instance();

        // Instantiate devices list
        auto names = registry.listDeviceTypes();
        // Ensure CPU is present for testing
        if (std::find( names.begin(), names.end(), "CPU" ) == names.end())
        {
            names.push_back( "CPU" );
        }

        for (const auto& device_name : names)
        {
            // Create instances via getDevice
            std::shared_ptr<ComputeDevice> device1;
            std::shared_ptr<ComputeDevice> device2;

            // Some device types may throw on creation (e.g. CUDA when no runtime) - guard that
            try
            {
                device1 = registry.getDevice( device_name == "CPU" ? "CPU" : device_name + ":0" );
                device2 = registry.getDevice( device_name == "CPU" ? "CPU" : device_name + ":0" );
            }
            catch (...)
            {
                // Skip devices that cannot be instantiated in this environment
                continue;
            }

            ASSERT_NE( device1, nullptr ) << "Failed to create device: " << device_name;
            ASSERT_NE( device2, nullptr ) << "Failed to create device: " << device_name;

            EXPECT_EQ( device1->getDeviceName(), device2->getDeviceName() );
            EXPECT_EQ( device1->getDeviceType(), device2->getDeviceType() );

            // Registry caches instances: expect same pointer
            EXPECT_EQ( device1.get(), device2.get() );
        }
    }

    // Test device naming conventions using getDevice where possible
    TEST_F( DeviceRegistrarTest, DeviceNamingConventions ) {
        auto& registry = DeviceRegistry::instance();

        auto types = registry.listDeviceTypes();

        for (const auto& device_type : types)
        {
            // instantiate a representative device (index 0) where applicable
            std::string name = device_type == "CPU" ? "CPU" : device_type + ":0";

            try
            {
                auto device = registry.getDevice( name );
                ASSERT_NE( device, nullptr );

                EXPECT_EQ( device->getDeviceName(), name );

                if (device_type == "CPU")
                {
                    EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
                }
                else if (device_type == "CUDA")
                {
                    EXPECT_EQ( device->getDeviceType(), DeviceType::Cuda );
                    size_t colon_pos = name.find( ':' );
                    ASSERT_NE( colon_pos, std::string::npos );

                    std::string id_str = name.substr( colon_pos + 1 );
                    EXPECT_FALSE( id_str.empty() );

                    try
                    {
                        int device_id = std::stoi( id_str );
                        EXPECT_GE( device_id, 0 );
                        EXPECT_EQ( device->getDeviceId(), device_id );
                    }
                    catch (const std::exception&)
                    {
                        FAIL() << "Device ID should be numeric: " << name;
                    }
                }
            }
            catch (...)
            {
                // Skip devices that cannot be instantiated in this environment
                continue;
            }
        }
    }
}