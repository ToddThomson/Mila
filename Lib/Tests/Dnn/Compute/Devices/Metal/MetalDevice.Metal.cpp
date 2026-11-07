/**
 * @file MetalDevice.Metal.cpp
 * @brief The Mac's Metal device: what it reports, what it allocates and what it refuses.
 */

#include <gtest/gtest.h>

import Mila;
import std;
import Compute.MetalBadAlloc;
import Compute.MetalBufferLocation;
import Compute.MetalCapabilities;
import Compute.MetalKernelLibrary;
import Compute.MetalKernelLimits;

namespace Mila::Tests::Dnn::Compute
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    TEST( MetalDevice, IsRegisteredAndPassesItsOwnChecks )
    {
        ASSERT_TRUE( DeviceRegistry::instance().hasDeviceType( DeviceType::Metal ) );

        const std::shared_ptr<MetalDevice> device = MetalDevice::at( 0 );

        EXPECT_EQ( device->getDeviceId(), Device::Metal( 0 ) );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Metal );
        EXPECT_FALSE( device->getDeviceName().empty() );
        EXPECT_EQ( metalRefusalReason( device->capabilities() ), std::nullopt );
    }

    TEST( MetalDevice, IsNamedForPeopleByItsChipAndForSettingsByItsType )
    {
        const std::shared_ptr<MetalDevice> device = MetalDevice::at( 0 );

        EXPECT_EQ( device->getHardwareName(), device->capabilities().device_name );
        EXPECT_FALSE( device->getHardwareName().empty() );
        EXPECT_EQ( device->getStableIdentity(), "metal" );
        EXPECT_EQ( findDeviceByStableIdentity( "metal" ), Device::Metal( 0 ) );
    }

    TEST( MetalDevice, ANumberNoMacHasIsNotADevice )
    {
        EXPECT_THROW( (void)MetalDevice::at( 1 ), std::runtime_error );
    }

    TEST( MetalDevice, ReadsItsNumbersFromTheDevice )
    {
        const MetalCapabilities& capabilities = MetalDevice::at( 0 )->capabilities();

        EXPECT_GE( capabilities.apple_gpu_family, kMinimumAppleGpuFamily );
        EXPECT_LE( capabilities.apple_gpu_family, kNewestKnownAppleGpuFamily );
        EXPECT_GT( capabilities.recommended_max_working_set_bytes, 0u );
        EXPECT_GT( capabilities.max_buffer_bytes, 0u );
        EXPECT_GT( capabilities.max_threadgroup_memory_bytes, 0u );
    }

    TEST( MetalDevice, MemoryInfoIsTheWorkingSetAndWhatIsLeftOfIt )
    {
        const std::shared_ptr<MetalDevice> device = MetalDevice::at( 0 );

        const DeviceMemoryInfo before = device->getMemoryInfo();
        EXPECT_EQ( before.total_bytes, device->capabilities().recommended_max_working_set_bytes );

        constexpr std::size_t bytes = std::size_t{ 64 } * 1024 * 1024;
        void* const contents = device->allocate( bytes );

        EXPECT_LE( device->getMemoryInfo().free_bytes + bytes, before.free_bytes );

        device->deallocate( contents );
    }

    TEST( MetalDevice, AnAllocationIsWritableMemoryThatLocatesToItsBuffer )
    {
        const std::shared_ptr<MetalDevice> device = MetalDevice::at( 0 );

        constexpr std::size_t bytes = 4096;
        auto* const contents = static_cast<std::byte*>( device->allocate( bytes ) );

        ASSERT_NE( contents, nullptr );
        contents[bytes - 1] = std::byte{ 7 };

        const MetalBufferLocation base = device->locate( contents );
        const MetalBufferLocation interior = device->locate( contents + 100 );

        EXPECT_NE( base.buffer, nullptr );
        EXPECT_EQ( base.offset, 0u );
        EXPECT_EQ( interior.buffer, base.buffer );
        EXPECT_EQ( interior.offset, 100u );

        device->deallocate( contents );

        EXPECT_THROW( (void)device->locate( contents ), std::invalid_argument );
    }

    TEST( MetalDevice, MoreThanTheLargestBufferIsRefusedWithBothNumbers )
    {
        const std::shared_ptr<MetalDevice> device = MetalDevice::at( 0 );
        const std::size_t limit = device->capabilities().max_buffer_bytes;

        try
        {
            (void)device->allocate( limit + 1 );

            FAIL() << "the allocation was not refused";
        }
        catch ( const MetalBadAlloc& error )
        {
            const std::string message = error.what();

            EXPECT_NE( message.find( std::to_string( limit ) ), std::string::npos ) << message;
            EXPECT_NE( message.find( std::to_string( limit + 1 ) ), std::string::npos ) << message;
        }
    }

    TEST( MetalKernelLibrary, EveryKernelCompilesAndHasAPipeline )
    {
        MetalKernelLibrary& kernels = MetalDevice::at( 0 )->kernels();
        const std::vector<std::string> names = kernels.kernelNames();

        EXPECT_FALSE( names.empty() );

        for ( const std::string& name : names )
        {
            EXPECT_NE( kernels.pipeline( name ), nullptr ) << name;
            EXPECT_GT( kernels.limits( name ).thread_execution_width, 0u ) << name;
            EXPECT_GT( kernels.limits( name ).max_total_threads_per_threadgroup, 0u ) << name;
        }
    }

    TEST( MetalKernelLibrary, AKernelIsPreparedOnce )
    {
        MetalKernelLibrary& kernels = MetalDevice::at( 0 )->kernels();

        EXPECT_EQ( kernels.pipeline( "fill_half" ), kernels.pipeline( "fill_half" ) );
    }

    TEST( MetalKernelLibrary, AnUnknownKernelIsRefusedByName )
    {
        try
        {
            (void)MetalDevice::at( 0 )->kernels().pipeline( "no_such_kernel" );

            FAIL() << "the kernel was not refused";
        }
        catch ( const std::runtime_error& error )
        {
            EXPECT_NE( std::string( error.what() ).find( "no_such_kernel" ), std::string::npos ) << error.what();
        }
    }
}
