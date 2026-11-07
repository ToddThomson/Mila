/**
 * @file CudaDeviceFacts.Cuda.cpp
 * @brief What each CUDA device says about itself: its name, its stable identity, its speed facts.
 */

#include <gtest/gtest.h>

import Mila;
import std;

namespace Dnn::Compute::Devices::Tests
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;

    namespace
    {
        std::vector<DeviceId> cudaDevices()
        {
            std::vector<DeviceId> ids;

            for ( const DeviceId& id : DeviceRegistry::instance().listDeviceIds() )
            {
                if ( id.type == DeviceType::Cuda )
                {
                    ids.push_back( id );
                }
            }

            return ids;
        }
    }

    TEST( CudaDeviceFacts, EachDeviceIsNamedByItsUuidAsNvidiaSmiWritesIt )
    {
        const std::vector<DeviceId> ids = cudaDevices();

        if ( ids.empty() )
        {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::regex uuid( "GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}" );

        for ( const DeviceId& id : ids )
        {
            const std::shared_ptr<Device> device = DeviceRegistry::instance().getDevice( id );
            const std::string identity = device->getStableIdentity();

            // Printed so a run can be read against nvidia-smi -L, which orders the cards its own way.
            std::cout << std::format( "[identity] {} {} {}\n", id.toString(), device->getHardwareName(), identity );

            EXPECT_TRUE( std::regex_match( identity, uuid ) ) << id.toString() << ": " << identity;
        }
    }

    TEST( CudaDeviceFacts, NoTwoDevicesShareAnIdentity )
    {
        const std::vector<DeviceId> ids = cudaDevices();

        if ( ids.size() < 2 )
        {
            GTEST_SKIP() << "needs two CUDA devices";
        }

        std::set<std::string> seen;

        for ( const DeviceId& id : ids )
        {
            EXPECT_TRUE( seen.insert( DeviceRegistry::instance().getDevice( id )->getStableIdentity() ).second )
                << id.toString();
        }
    }

    TEST( CudaDeviceFacts, EachDeviceReportsItsNameAndSpeed )
    {
        const std::vector<DeviceId> ids = cudaDevices();

        if ( ids.empty() )
        {
            GTEST_SKIP() << "no CUDA device";
        }

        for ( const DeviceId& id : ids )
        {
            const std::shared_ptr<Device> device = DeviceRegistry::instance().getDevice( id );
            const DevicePerformance performance = device->getPerformance();

            EXPECT_FALSE( device->getHardwareName().empty() ) << id.toString();
            EXPECT_NE( device->getHardwareName(), device->getDeviceName() ) << id.toString();
            EXPECT_GT( performance.compute_unit_count, 0 ) << id.toString();
            EXPECT_GT( performance.clock_khz, 0 ) << id.toString();
            EXPECT_GT( performance.memory_clock_khz, 0 ) << id.toString();
            EXPECT_GT( performance.memory_bus_width_bits, 0 ) << id.toString();
        }
    }

    TEST( CudaDeviceFacts, AReadingCarriesWhatTheDeviceSays )
    {
        const std::vector<DeviceId> ids = cudaDevices();

        if ( ids.empty() )
        {
            GTEST_SKIP() << "no CUDA device";
        }

        for ( const DeviceId& id : ids )
        {
            const std::shared_ptr<Device> device = DeviceRegistry::instance().getDevice( id );
            const DeviceReading reading = DeviceReading::take( id );

            EXPECT_EQ( reading.device, id );
            EXPECT_EQ( reading.hardware_name, device->getHardwareName() );
            EXPECT_EQ( reading.stable_identity, device->getStableIdentity() );
            EXPECT_EQ( reading.performance.compute_unit_count, device->getPerformance().compute_unit_count );
            EXPECT_EQ( reading.performance.memory_bus_width_bits, device->getPerformance().memory_bus_width_bits );
        }
    }
}
