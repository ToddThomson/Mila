#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

import Mila;

// Private Module Imports
import Compute.DeviceRegistry;
import Compute.DeviceId;


namespace Dnn::Compute::Registry::Tests
{
    using namespace Mila::Dnn::Compute;

    TEST( DeviceRegistryHelpersTests, ListDevicesByName ) {
        auto& registry = DeviceRegistry::instance();

        int gpu_device_count = static_cast<int>( registry.getDeviceCount( DeviceType::Cuda ) );
        int cpu_device_count = static_cast<int>( registry.getDeviceCount( DeviceType::Cpu ) );

        int expected_device_count = cpu_device_count + gpu_device_count;

        std::vector<std::string> devices = listDevicesByName();

        EXPECT_EQ( static_cast<int>( devices.size() ), expected_device_count );

        // Verify CPU is present by comparing against canonical DeviceId representation
        std::string cpu_name = Device::Cpu().toString();
        EXPECT_NE( std::find( devices.begin(), devices.end(), cpu_name ), devices.end() );

        // Verify each CUDA device is present
        for ( int i = 0; i < gpu_device_count; ++i ) {
            std::string device_name = Device::Cuda( i ).toString();
            EXPECT_NE( std::find( devices.begin(), devices.end(), device_name ), devices.end() );
        }
    }

    TEST( DeviceRegistryHelpersTests, ListDevicesByTypeAndCount ) {
        auto& registry = DeviceRegistry::instance();

        int gpu_device_count = static_cast<int>( registry.getDeviceCount( DeviceType::Cuda ) );
        int cpu_device_count = static_cast<int>( registry.getDeviceCount( DeviceType::Cpu ) );

        std::vector<std::string> cuda_devices = listDevicesByType( DeviceType::Cuda );

        EXPECT_EQ( static_cast<int>( cuda_devices.size() ), gpu_device_count );

        for ( const auto& name : cuda_devices ) {
            EXPECT_NE( name.find( "CUDA" ), std::string::npos );
        }

        std::vector<std::string> cpu_devices = listDevicesByType( DeviceType::Cpu );

        EXPECT_EQ( static_cast<int>( cpu_devices.size() ), cpu_device_count );

        // Validate CPU canonical name is present in cpu_devices
        std::string cpu_name = Device::Cpu().toString();
        EXPECT_NE( std::find( cpu_devices.begin(), cpu_devices.end(), cpu_name ), cpu_devices.end() );
    }

    TEST( DeviceRegistryHelpersTests, GetBestDeviceAndDeviceCount ) {
        auto& registry = DeviceRegistry::instance();

        EXPECT_EQ( getDeviceCount( DeviceType::Cuda ), registry.getDeviceCount( DeviceType::Cuda ) );
        EXPECT_EQ( getDeviceCount( DeviceType::Cpu ), registry.getDeviceCount( DeviceType::Cpu ) );

        // getBestDevice for CPU should return a DeviceId with type == Cpu
        DeviceId best_cpu = getBestDevice( DeviceType::Cpu );
        EXPECT_EQ( best_cpu.type, DeviceType::Cpu );

        // getBestDevice for CUDA: if GPUs exist, ensure returned DeviceId is one of the CUDA devices;
        // otherwise index == -1 indicates none.
        int gpu_device_count = static_cast<int>( registry.getDeviceCount( DeviceType::Cuda ) );

        DeviceId best_cuda = getBestDevice( DeviceType::Cuda );

        if ( gpu_device_count > 0 ) {
            EXPECT_EQ( best_cuda.type, DeviceType::Cuda );

            std::vector<std::string> cuda_devices = listDevicesByType( DeviceType::Cuda );
            EXPECT_NE( std::find( cuda_devices.begin(), cuda_devices.end(), best_cuda.toString() ), cuda_devices.end() );
        }
        else {
            EXPECT_EQ( best_cuda.index, -1 );
        }
    }
}