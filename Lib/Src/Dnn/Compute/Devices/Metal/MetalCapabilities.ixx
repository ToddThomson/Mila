/**
 * @file MetalCapabilities.ixx
 * @brief What a Mac reports about its GPU, and whether Mila can run on it.
 */

export module Compute.MetalCapabilities;

import std;

namespace Mila::Dnn::Compute
{
    /// The oldest macOS Mila runs on (MetalBackend.md section 10).
    export constexpr int kMinimumMacOsMajorVersion = 27;

    /// The oldest GPU family Mila runs on: `MTLGPUFamilyApple7`, the M1.
    export constexpr int kMinimumAppleGpuFamily = 7;

    /// The newest GPU family Mila knows the name of. A later chip reports this family too, so it is read as this one.
    export constexpr int kNewestKnownAppleGpuFamily = 11;

    /**
     * @brief The three checks of MetalBackend.md section 2.1, as the device answered them.
     *
     * Read once from the device and never tabulated by chip. No field says whether BF16 is native or which matrix path
     * is faster: no API reports either (section 2.2).
     */
    export struct MetalCapabilities
    {
        std::string device_name;

        /// Logged only: Apple does not document its values.
        std::string architecture_name;

        int macos_major_version{ 0 };
        int macos_minor_version{ 0 };
        int macos_patch_version{ 0 };

        /// The highest `MTLGPUFamilyAppleN` the device supports up to the newest Mila knows, or 0 below the minimum.
        int apple_gpu_family{ 0 };

        bool has_unified_memory{ false };

        std::size_t recommended_max_working_set_bytes{ 0 };
        std::size_t max_buffer_bytes{ 0 };
        std::size_t max_threadgroup_memory_bytes{ 0 };
    };

    /**
     * @brief Why Mila cannot run on this GPU, in words for the person running it, or nothing when it can.
     *
     * A plain function of the answers, so that every refusal can be tested on a Mac that passes them all.
     */
    export [[nodiscard]] std::optional<std::string> metalRefusalReason( const MetalCapabilities& capabilities )
    {
        if ( capabilities.macos_major_version < kMinimumMacOsMajorVersion )
        {
            return std::format( "Mila needs macOS {} or later to use this Mac's GPU, and this Mac runs macOS {}.{}.",
                kMinimumMacOsMajorVersion, capabilities.macos_major_version, capabilities.macos_minor_version );
        }

        if ( capabilities.apple_gpu_family < kMinimumAppleGpuFamily )
        {
            return std::format( "Mila runs on the GPU of an Apple M-series chip, and {} is not one.", capabilities.device_name );
        }

        if ( !capabilities.has_unified_memory )
        {
            return std::format( "Mila needs a GPU that shares its memory with the CPU, and {} does not.", capabilities.device_name );
        }

        return std::nullopt;
    }
}
