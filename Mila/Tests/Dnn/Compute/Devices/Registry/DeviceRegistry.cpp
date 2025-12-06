#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <algorithm>
#include <utility>

import Mila;

// Private Module Imports for testing
//import Compute.DeviceRegistrar;

namespace Dnn::Compute::Registry::Tests
{
	using namespace Mila::Dnn;
	using namespace Mila::Dnn::Compute;

	class DeviceRegistryTest : public ::testing::Test
	{
	protected:
		void SetUp() override
		{
			// Ensure registrar/registry are initialized (registrar may register factories)
			//(void)DeviceRegistrar::instance();
			(void)DeviceRegistry::instance();
		}

		void TearDown() override
		{
			// DeviceRegistry is a singleton; state persists between tests by design
		}
	};

	TEST_F( DeviceRegistryTest, DeviceCreationAndCaching )
	{
		auto& registry = DeviceRegistry::instance();

		auto cpu_device = registry.getDevice( Device::Cpu() );

		ASSERT_NE( cpu_device, nullptr );
		EXPECT_EQ( cpu_device->getDeviceType(), DeviceType::Cpu );
		EXPECT_EQ( cpu_device->getDeviceId().type, DeviceType::Cpu );

		// Repeated calls with same id should return the same cached instance
		auto cpu_device2 = registry.getDevice( Device::Cpu() );
		
		ASSERT_NE( cpu_device2, nullptr );
		EXPECT_EQ( cpu_device.get(), cpu_device2.get() );
	}

	TEST_F( DeviceRegistryTest, CudaDeviceRegistrationIfAvailable )
	{
		auto& registry = DeviceRegistry::instance();

		if ( registry.hasDeviceType( DeviceType::Cuda ) )
		{
			auto cuda_device = registry.getDevice( Device::Cuda( 0 ) );

			ASSERT_NE( cuda_device, nullptr );
			EXPECT_EQ( cuda_device->getDeviceType(), DeviceType::Cuda );
			EXPECT_EQ( cuda_device->getDeviceId().type, DeviceType::Cuda );
		}
		else
		{
			// When CUDA not registered, creating a CUDA device should throw
			EXPECT_THROW( registry.getDevice( Device::Cuda( 0 ) ), std::runtime_error );
		}
	}

	TEST_F( DeviceRegistryTest, DeviceListingAfterInstantiation )
	{
		auto& registry = DeviceRegistry::instance();

		// Ensure at least CPU exists by instantiating it
		auto cpu_device = registry.getDevice( Device::Cpu() );
		ASSERT_NE( cpu_device, nullptr );

		auto devices = registry.listDeviceIds();
		EXPECT_GE( devices.size(), 1u );

		EXPECT_NE( std::find( devices.begin(), devices.end(), Device::Cpu() ), devices.end() );

		for ( const auto& dev : devices )
		{
			// device type must be a valid enum value (basic sanity)
			EXPECT_TRUE(
				dev.type == DeviceType::Cpu ||
				dev.type == DeviceType::Cuda ||
				dev.type == DeviceType::Metal ||
				dev.type == DeviceType::Rocm
			);
		}
	}

	TEST_F( DeviceRegistryTest, DeviceExistenceAndInvalidInputs )
	{
		auto& registry = DeviceRegistry::instance();

		EXPECT_NO_THROW( {
			auto cpu = registry.getDevice( Device::Cpu() );
			ASSERT_NE( cpu, nullptr );
			} );

		// Parsing invalid textual device identifiers should throw
		EXPECT_THROW( DeviceId::parse( "INVALID:999" ), std::invalid_argument );
		EXPECT_THROW( DeviceId::parse( "" ), std::invalid_argument );

		// Requesting an unregistered device type should throw (choose CUDA when not registered)
		if ( !registry.hasDeviceType( DeviceType::Cuda ) )
		{
			EXPECT_THROW( registry.getDevice( Device::Cuda( 0 ) ), std::runtime_error );
		}
	}

	TEST_F( DeviceRegistryTest, ThreadSafeDeviceOperations )
	{
        // TJT: This test fails intermittently. Determine root cause.
		auto& registry = DeviceRegistry::instance();

		constexpr int num_threads = 20;
		constexpr int operations_per_thread = 50;

		std::vector<std::thread> threads;
		std::vector<bool> results( num_threads, false );

		for ( int i = 0; i < num_threads; ++i )
		{
			threads.emplace_back( [&registry, &results, i]() {
				bool thread_success = true;

				for ( int j = 0; j < operations_per_thread; ++j )
				{
					try
					{
						auto device = registry.getDevice( Device::Cpu() );
						if ( !device || device->getDeviceId().type != DeviceType::Cpu )
						{
							thread_success = false;
							break;
						}

						auto devices = registry.listDeviceIds();
						if ( devices.empty() )
						{
							thread_success = false; break;
						}

						if ( !registry.hasDeviceType( DeviceType::Cpu ) )
						{
							thread_success = false; break;
						}
					}
					catch ( ... )
					{
						thread_success = false;
						break;
					}

					std::this_thread::sleep_for( std::chrono::microseconds( 1 ) );
				}

				results[i] = thread_success;
				} );
		}

		for ( auto& thread : threads ) thread.join();

		for ( int i = 0; i < num_threads; ++i )
		{
			EXPECT_TRUE( results[i] ) << "Thread " << i << " failed";
		}
	}

	TEST_F( DeviceRegistryTest, DeviceFactoryConsistency )
	{
		auto& registry = DeviceRegistry::instance();

		auto types = registry.listDeviceTypes();

		// Ensure CPU present for testing
		if ( std::find( types.begin(), types.end(), DeviceType::Cpu ) == types.end() )
		{
			types.push_back( DeviceType::Cpu );
		}

		for ( const auto& dt : types )
		{
			DeviceId id{ dt, 0 };

			// For CPU prefer explicit Cpu() helper
			if ( dt == DeviceType::Cpu ) id = Device::Cpu();

			std::shared_ptr<Device> device1;
			std::shared_ptr<Device> device2;

			try
			{
				device1 = registry.getDevice( id );
				device2 = registry.getDevice( id );
			}
			catch ( ... )
			{
				// Skip devices that cannot be instantiated in this environment
				continue;
			}

			ASSERT_NE( device1, nullptr );
			ASSERT_NE( device2, nullptr );

			EXPECT_EQ( device1->getDeviceType(), device2->getDeviceType() );
			EXPECT_EQ( device1->getDeviceId().type, device2->getDeviceId().type );

			// Registry caches instances: expect same pointer
			EXPECT_EQ( device1.get(), device2.get() );
		}
	}

	TEST_F( DeviceRegistryTest, DeviceNamingConventions )
	{
		auto& registry = DeviceRegistry::instance();

		auto types = registry.listDeviceTypes();

		for ( const auto& dt : types )
		{
			DeviceId id{ dt, 0 };
			if ( dt == DeviceType::Cpu ) id = Device::Cpu();

			try
			{
				auto device = registry.getDevice( id );
				ASSERT_NE( device, nullptr );

				// Basic invariants: device reports matching type and a valid id
				EXPECT_EQ( device->getDeviceType(), dt );
				EXPECT_EQ( device->getDeviceId().type, dt );

				if ( dt == DeviceType::Cuda )
				{
					EXPECT_GE( device->getDeviceId().index, 0 );
				}
			}
			catch ( ... )
			{
				// Skip devices that cannot be instantiated in this environment
				continue;
			}
		}
	}
}