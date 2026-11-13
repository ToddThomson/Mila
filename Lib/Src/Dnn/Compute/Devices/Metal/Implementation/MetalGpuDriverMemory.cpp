/**
 * @file MetalGpuDriverMemory.cpp
 * @brief What the GPU driver holds for every process, from IORegistry, the only place a Mac reports it.
 */

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

// "In use system memory" in the accelerator's PerformanceStatistics: the memory the GPU driver holds for GPU work, across
// processes, and not memory a process has locked (MetalBackend.md section 5, W3d). Apple does not document the key, so
// its absence is an answer: -1, and the caller reads nothing.
extern "C" long long milaMetalGpuDriverMemoryInUse()
{
    const io_service_t service = IOServiceGetMatchingService( kIOMainPortDefault, IOServiceMatching( "IOAccelerator" ) );

    if ( !service )
    {
        return -1;
    }

    long long in_use = -1;

    if ( const CFTypeRef statistics = IORegistryEntryCreateCFProperty( service, CFSTR( "PerformanceStatistics" ), kCFAllocatorDefault, 0 ) )
    {
        if ( CFGetTypeID( statistics ) == CFDictionaryGetTypeID() )
        {
            const auto value = CFDictionaryGetValue( static_cast<CFDictionaryRef>( statistics ), CFSTR( "In use system memory" ) );

            if ( value && CFGetTypeID( value ) == CFNumberGetTypeID() )
            {
                CFNumberGetValue( static_cast<CFNumberRef>( value ), kCFNumberLongLongType, &in_use );
            }
        }

        CFRelease( statistics );
    }

    IOObjectRelease( service );

    return in_use;
}
