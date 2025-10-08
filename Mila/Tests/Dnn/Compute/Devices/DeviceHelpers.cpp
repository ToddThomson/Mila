#include <gtest/gtest.h>
#include <memory>

import Mila;

namespace Dnn::Compute::Devices::Tests
{
    using namespace Mila::Dnn::Compute;

    TEST( DeviceHelpersTests, ListDevices ) {
        auto gpu_device_count = Cuda::getDeviceCount();
		auto cpu_device_count = 1;

        int expected_device_count = cpu_device_count + gpu_device_count;

        std::vector<std::string> devices = listDevices();
       
        EXPECT_EQ( devices.size(), expected_device_count );
        EXPECT_NE( std::find( devices.begin(), devices.end(), "CPU" ), devices.end() );
        
        for ( int i = 0; i < gpu_device_count; i++ ) {
            std::string device_name = "CUDA:" + std::to_string( i );
            EXPECT_NE( std::find( devices.begin(), devices.end(), device_name ), devices.end() );
        }
    }
}