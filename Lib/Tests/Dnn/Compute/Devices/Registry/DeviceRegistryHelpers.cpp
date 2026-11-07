#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <optional>
#include <stdexcept>

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
        int metal_device_count = static_cast<int>( registry.getDeviceCount( DeviceType::Metal ) );

        int expected_device_count = cpu_device_count + gpu_device_count + metal_device_count;

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

    TEST( DeviceRegistryHelpersTests, DeviceCountMatchesTheRegistry ) {
        auto& registry = DeviceRegistry::instance();

        EXPECT_EQ( getDeviceCount( DeviceType::Cuda ), registry.getDeviceCount( DeviceType::Cuda ) );
        EXPECT_EQ( getDeviceCount( DeviceType::Cpu ), registry.getDeviceCount( DeviceType::Cpu ) );
        EXPECT_EQ( getDeviceCount( DeviceType::Metal ), registry.getDeviceCount( DeviceType::Metal ) );
    }

    TEST( DeviceRegistryHelpersTests, EveryRegisteredDeviceIsFoundByItsOwnIdentity ) {
        auto& registry = DeviceRegistry::instance();

        for ( const DeviceId& id : registry.listDeviceIds() )
        {
            const std::string identity = registry.getDevice( id )->getStableIdentity();

            ASSERT_FALSE( identity.empty() ) << id.toString();
            EXPECT_EQ( findDeviceByStableIdentity( identity ), id ) << identity;
        }
    }

    TEST( DeviceRegistryHelpersTests, AnIdentityIsFoundWhateverItsCase ) {
        auto& registry = DeviceRegistry::instance();

        for ( const DeviceId& id : registry.listDeviceIds() )
        {
            std::string shouted = registry.getDevice( id )->getStableIdentity();
            std::ranges::transform( shouted, shouted.begin(),
                []( unsigned char c ) { return static_cast<char>( std::toupper( c ) ); } );

            EXPECT_EQ( findDeviceByStableIdentity( shouted ), id ) << shouted;
        }
    }

    TEST( DeviceRegistryHelpersTests, AnIdentityNoDeviceHasFindsNothing ) {
        EXPECT_EQ( findDeviceByStableIdentity( "GPU-00000000-0000-0000-0000-000000000000" ), std::nullopt );
        EXPECT_EQ( findDeviceByStableIdentity( "no-such-device" ), std::nullopt );
    }

    TEST( DeviceRegistryHelpersTests, AnEmptyIdentityIsRefused ) {
        EXPECT_THROW( (void)findDeviceByStableIdentity( "" ), std::invalid_argument );
    }

    TEST( DeviceRegistryHelpersTests, APrefixOneCudaDeviceHasFindsIt ) {
        auto& registry = DeviceRegistry::instance();

        if ( registry.getDeviceCount( DeviceType::Cuda ) == 0 )
        {
            GTEST_SKIP() << "no CUDA device";
        }

        const std::string identity = registry.getDevice( Device::Cuda( 0 ) )->getStableIdentity();

        // As CUDA_VISIBLE_DEVICES takes them: "GPU-" and the first group, unless another card shares it.
        const std::string prefix = identity.substr( 0, 12 );
        int sharing = 0;

        for ( const DeviceId& id : registry.listDeviceIds() )
        {
            sharing += registry.getDevice( id )->getStableIdentity().starts_with( prefix ) ? 1 : 0;
        }

        if ( sharing == 1 )
        {
            EXPECT_EQ( findDeviceByStableIdentity( prefix ), Device::Cuda( 0 ) );
        } else {
            EXPECT_THROW( (void)findDeviceByStableIdentity( prefix ), std::invalid_argument );
        }
    }

    TEST( DeviceRegistryHelpersTests, APrefixEveryCudaDeviceHasIsRefusedNamingThem ) {
        auto& registry = DeviceRegistry::instance();

        if ( registry.getDeviceCount( DeviceType::Cuda ) < 2 )
        {
            GTEST_SKIP() << "needs two CUDA devices";
        }

        try
        {
            (void)findDeviceByStableIdentity( "GPU-" );

            FAIL() << "a prefix two devices share was not refused";
        }
        catch ( const std::invalid_argument& error )
        {
            const std::string message = error.what();

            EXPECT_NE( message.find( Device::Cuda( 0 ).toString() ), std::string::npos ) << message;
            EXPECT_NE( message.find( Device::Cuda( 1 ).toString() ), std::string::npos ) << message;
        }
    }
}