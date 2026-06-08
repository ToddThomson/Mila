#include <gtest/gtest.h>
#include <string>

import Mila;

namespace Dnn::Compute::Tests
{
	using namespace Mila::Dnn::Compute;

	// Verify compile-time factory properties (Device::Cpu is constexpr)
	static_assert( Device::Cpu().type == DeviceType::Cpu );
	static_assert( Device::Cpu().index == 0 );

	class CpuDeviceTest : public ::testing::Test
	{
	protected:
		void SetUp() override
		{
			// Register CPU device factory only if not already registered.
			// DeviceRegistry::registerDevice throws if the id is already present.
			if ( !DeviceRegistry::instance().hasDevice( Device::Cpu() ) )
			{
				CpuDeviceRegistrar::registerDevices();
			}
		}

		void TearDown() override
		{
		}
	};

	TEST_F( CpuDeviceTest, GetDeviceType_Name_And_Id )
	{
		auto device = DeviceRegistry::instance().getDevice( Device::Cpu() );

		ASSERT_NE( device, nullptr );

		// Type
		EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );

		// Name - matches expected literal and DeviceId::toString() for Device::Cpu()
		EXPECT_EQ( device->getDeviceName(), std::string( "CPU:0" ) );
		EXPECT_EQ( device->getDeviceName(), Device::Cpu().toString() );

		// DeviceId
		DeviceId id = device->getDeviceId();
		EXPECT_EQ( id.type, DeviceType::Cpu );
		EXPECT_EQ( id.index, 0 );

		// Device::Cpu() factory equivalence
		EXPECT_EQ( id, Device::Cpu() );
	}

	TEST_F( CpuDeviceTest, PolymorphicBehavior )
	{
		auto device = DeviceRegistry::instance().getDevice( Device::Cpu() );

		ASSERT_NE( device, nullptr );

		Device* base = device.get();

		// Ensure virtual dispatch returns same id/name through base pointer
		EXPECT_EQ( base->getDeviceId(), device->getDeviceId() );
		EXPECT_EQ( base->getDeviceName(), device->getDeviceName() );

		// Also ensure base-reported type is Cpu
		EXPECT_EQ( base->getDeviceType(), DeviceType::Cpu );
	}
}