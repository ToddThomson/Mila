// Description: Unit tests for the device classes.
#include <gtest/gtest.h>
#include <memory>

import Mila;

namespace Dnn::Compute::Tests
{
    using namespace Mila::Dnn::Compute;

    class DeviceTest : public testing::Test {

    protected:
        DeviceTest() {}
    };
   
    TEST( DeviceRegistryTests, ListDevices ) {
        auto device_count = GetDeviceCount();
		int expected_device_count = 1 + device_count;
        auto devices = list_devices();
        EXPECT_EQ( devices.size(), expected_device_count );
        EXPECT_NE( std::find( devices.begin(), devices.end(), "CPU" ), devices.end() );
		for ( int i = 0; i < device_count; i++ ) {
			std::string device_name = "CUDA:" + std::to_string( i );
			EXPECT_NE( std::find( devices.begin(), devices.end(), device_name ), devices.end() );
		}
    }

}
