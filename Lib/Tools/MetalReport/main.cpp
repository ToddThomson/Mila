/**
 * @file main.cpp
 * @brief What this Mac's GPU reports, and how long Mila's kernels take to load and prepare on it.
 *
 * MetalBackend.md sections 2.1 and 2.2. Run it on any chip Mila has not seen, for section 3.3's table and Phase 7.
 */

import Mila;
import std;
import Compute.MetalCapabilities;
import Compute.MetalKernelLibrary;
import Compute.MetalKernelLimits;

// The core count is in no Metal API, and in IORegistry here. Log only (MetalBackend.md section 2.2).
int gpuCoreCount();

namespace
{
    using Clock = std::chrono::steady_clock;

    double millisecondsSince( Clock::time_point start )
    {
        return std::chrono::duration<double, std::milli>( Clock::now() - start ).count();
    }

    double gibibytes( std::size_t bytes )
    {
        return static_cast<double>( bytes ) / ( 1024.0 * 1024.0 * 1024.0 );
    }
}

int main()
{
    using namespace Mila::Dnn::Compute;

    Mila::initialize();

    if ( !DeviceRegistry::instance().hasDeviceType( DeviceType::Metal ) )
    {
        std::println( "This Mac has no Metal device." );

        return 1;
    }

    std::shared_ptr<MetalDevice> device;

    try
    {
        device = MetalDevice::at( 0 );
    }
    catch ( const std::runtime_error& refusal )
    {
        std::println( "Refused: {}", refusal.what() );

        return 1;
    }

    const MetalCapabilities& capabilities = device->capabilities();

    std::println( "Device                 {}", capabilities.device_name );
    std::println( "Architecture           {} (log only: its values are undocumented)", capabilities.architecture_name );
    std::println( "GPU cores              {} (log only: from IORegistry)", gpuCoreCount() );
    std::println( "macOS                  {}.{}.{}", capabilities.macos_major_version, capabilities.macos_minor_version, capabilities.macos_patch_version );
    std::println( "GPU family             Apple{} (the newest Mila knows is Apple{})", capabilities.apple_gpu_family, kNewestKnownAppleGpuFamily );
    std::println( "Unified memory         {}", capabilities.has_unified_memory );
    std::println( "Working set            {:.2f} GiB", gibibytes( capabilities.recommended_max_working_set_bytes ) );
    std::println( "Largest buffer         {:.2f} GiB", gibibytes( capabilities.max_buffer_bytes ) );
    std::println( "Threadgroup memory     {} KiB", capabilities.max_threadgroup_memory_bytes / 1024 );

    MetalKernelLibrary& kernels = device->kernels();

    const Clock::time_point load_start = Clock::now();
    const std::vector<std::string> names = kernels.kernelNames();
    const double load_milliseconds = millisecondsSince( load_start );

    const Clock::time_point prepare_start = Clock::now();

    for ( const std::string& name : names )
    {
        (void)kernels.pipeline( name );
    }

    const double prepare_milliseconds = millisecondsSince( prepare_start );

    std::println( "" );
    std::println( "Kernels                {}", names.size() );
    std::println( "Loading them           {:.1f} ms", load_milliseconds );
    std::println( "Preparing them all     {:.1f} ms", prepare_milliseconds );
    std::println( "" );

    for ( const std::string& name : names )
    {
        const MetalKernelLimits limits = kernels.limits( name );

        std::println( "{:<24} width {:<4} largest threadgroup {}", name, limits.thread_execution_width, limits.max_total_threads_per_threadgroup );
    }

    return 0;
}
