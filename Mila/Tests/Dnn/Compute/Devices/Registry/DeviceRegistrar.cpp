#include <gtest/gtest.h>
#include <vector>
#include <thread>
#include <algorithm>
#include <utility>
#include <string>

import Mila;

// Private Module Imports
import Compute.Device;
import Compute.DeviceRegistrar;

namespace Dnn::Compute::Registry::Tests
{
	using namespace Mila::Dnn;
	using namespace Mila::Dnn::Compute;

	class MockComputeDevice : public Device
	{
	public:
		explicit MockComputeDevice( DeviceId id, std::string name = {} )
			: id_( id ), name_( std::move( name ) )
		{
		}

		constexpr DeviceType getDeviceType() const override
		{
			return id_.type;
		}

		std::string getDeviceName() const override
		{
			return name_;
		}
		DeviceId getDeviceId() const override
		{
			return id_;
		}

	private:
		DeviceId id_;
		std::string name_;
	};

	// Test fixture for DeviceRegistrar-focused tests
	class DeviceRegistrarTest : public ::testing::Test
	{
	protected:
		void SetUp() override
		{
			// Ensure registrar/registry are initialized (registration happens in registrar ctor)
			(void)DeviceRegistrar::instance();
			(void)DeviceRegistry::instance();
		}

		void TearDown() override
		{
		}
	};

	TEST_F( DeviceRegistrarTest, SingletonInstance )
	{
		auto& registrar1 = DeviceRegistrar::instance();
		auto& registrar2 = DeviceRegistrar::instance();

		EXPECT_EQ( &registrar1, &registrar2 );
	}

	TEST_F( DeviceRegistrarTest, ThreadSafeSingleton )
	{
		std::vector<DeviceRegistrar*> instances;
		std::vector<std::thread> threads;
		constexpr int num_threads = 10;

		instances.resize( num_threads );

		for ( int i = 0; i < num_threads; ++i )
		{
			threads.emplace_back( [&instances, i]() {
				instances[i] = &DeviceRegistrar::instance();
				} );
		}

		for ( auto& thread : threads ) thread.join();

		for ( int i = 1; i < num_threads; ++i )
		{
			EXPECT_EQ( instances[0], instances[i] );
		}
	}

	TEST_F( DeviceRegistrarTest, RegistersExpectedDeviceTypes )
	{
		auto& registry = DeviceRegistry::instance();

		// CPU factory should always be registered
		EXPECT_TRUE( registry.hasDeviceType( DeviceType::Cpu ) );

		auto types = registry.listDeviceTypes();
		
		EXPECT_NE( std::find( types.begin(), types.end(), DeviceType::Cpu ), types.end() );
	}

	TEST_F( DeviceRegistrarTest, RegistrationInitializationOrder )
	{
		auto& registrar = DeviceRegistrar::instance();
		auto& registry = DeviceRegistry::instance();

		auto types = registry.listDeviceTypes();
		EXPECT_FALSE( types.empty() );

		EXPECT_TRUE( registry.hasDeviceType( DeviceType::Cpu ) );

		auto cpu_device = registry.getDevice( Device::Cpu() );
		
		EXPECT_EQ( cpu_device->getDeviceId().type, DeviceType::Cpu);

		(void)registrar;
	}
}