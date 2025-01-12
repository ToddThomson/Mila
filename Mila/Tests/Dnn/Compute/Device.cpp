// Description: Unit tests for the device classes.
#include <gtest/gtest.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <memory>
#ifdef USE_OMP
#include <omp.h>
#endif

import Mila;

namespace Dnn::Compute::Tests
{
    using namespace Mila::Dnn::Compute;

    class DeviceTest : public testing::Test {

    protected:
        DeviceTest() {}
    };
   
    TEST( DeviceRegistryTests, CpuRegisterAndCreateDevice ) {
        auto& registry = DeviceRegistry::instance();
        registry.registerDevice( "MockDevice", []() { return std::make_unique<Cpu::CpuDevice>(); } );

        auto device = registry.createDevice( "MockDevice" );
        EXPECT_NE( device, nullptr );
    }

    TEST( DeviceRegistryTests, CreateNonExistentDevice ) {
        auto& registry = DeviceRegistry::instance();
        auto device = registry.createDevice( "NonExistentDevice" );
        EXPECT_EQ( device, nullptr );
    }

    TEST( DeviceRegistryTests, ListDevices ) {
        auto& registry = DeviceRegistry::instance();
        registry.registerDevice( "MockDevice1", []() { return std::make_unique<Cpu::CpuDevice>(); } );
        registry.registerDevice( "MockDevice2", []() { return std::make_unique<Cuda::CudaDevice>(); } );

        auto devices = registry.list_devices();
        EXPECT_EQ( devices.size(), 2 );
        EXPECT_NE( std::find( devices.begin(), devices.end(), "MockDevice1" ), devices.end() );
        EXPECT_NE( std::find( devices.begin(), devices.end(), "MockDevice2" ), devices.end() );
    }

}
