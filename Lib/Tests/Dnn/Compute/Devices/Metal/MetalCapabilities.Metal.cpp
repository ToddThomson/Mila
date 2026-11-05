/**
 * @file MetalCapabilities.Metal.cpp
 * @brief Every refusal of a Mac Mila cannot run on, forced on one that passes (MetalBackend.md section 2.1).
 */

#include <gtest/gtest.h>

import std;
import Compute.MetalCapabilities;

namespace Mila::Tests::Dnn::Compute
{
    using namespace Mila::Dnn::Compute;

    namespace
    {
        MetalCapabilities passing()
        {
            MetalCapabilities capabilities;
            capabilities.device_name = "Test GPU";
            capabilities.macos_major_version = kMinimumMacOsMajorVersion;
            capabilities.apple_gpu_family = kMinimumAppleGpuFamily;
            capabilities.has_unified_memory = true;

            return capabilities;
        }
    }

    TEST( MetalCapabilities, AMacAtTheFloorIsNotRefused )
    {
        EXPECT_EQ( metalRefusalReason( passing() ), std::nullopt );
    }

    TEST( MetalCapabilities, AnOlderMacOsIsRefusedNamingBothVersions )
    {
        MetalCapabilities capabilities = passing();
        capabilities.macos_major_version = kMinimumMacOsMajorVersion - 1;
        capabilities.macos_minor_version = 4;

        const std::optional<std::string> reason = metalRefusalReason( capabilities );

        ASSERT_TRUE( reason.has_value() );
        EXPECT_NE( reason->find( std::format( "macOS {}", kMinimumMacOsMajorVersion ) ), std::string::npos ) << *reason;
        EXPECT_NE( reason->find( std::format( "macOS {}.4", kMinimumMacOsMajorVersion - 1 ) ), std::string::npos ) << *reason;
    }

    TEST( MetalCapabilities, AGpuBelowTheFloorIsRefusedNamingIt )
    {
        MetalCapabilities capabilities = passing();
        capabilities.apple_gpu_family = 0;

        const std::optional<std::string> reason = metalRefusalReason( capabilities );

        ASSERT_TRUE( reason.has_value() );
        EXPECT_NE( reason->find( "Test GPU" ), std::string::npos ) << *reason;
        EXPECT_NE( reason->find( "M-series" ), std::string::npos ) << *reason;
    }

    TEST( MetalCapabilities, MemoryNotSharedWithTheCpuIsRefusedNamingTheGpu )
    {
        MetalCapabilities capabilities = passing();
        capabilities.has_unified_memory = false;

        const std::optional<std::string> reason = metalRefusalReason( capabilities );

        ASSERT_TRUE( reason.has_value() );
        EXPECT_NE( reason->find( "Test GPU" ), std::string::npos ) << *reason;
        EXPECT_NE( reason->find( "shares its memory" ), std::string::npos ) << *reason;
    }
}
