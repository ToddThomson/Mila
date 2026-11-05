/**
 * @file GpuCoreCount.cpp
 * @brief The GPU's core count, from IORegistry: the only place a Mac reports it.
 */

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

int gpuCoreCount()
{
    const io_service_t service = IOServiceGetMatchingService( kIOMainPortDefault, IOServiceMatching( "AGXAccelerator" ) );

    if ( !service )
    {
        return 0;
    }

    int cores = 0;

    if ( const CFTypeRef property = IORegistryEntryCreateCFProperty( service, CFSTR( "gpu-core-count" ), kCFAllocatorDefault, 0 ) )
    {
        if ( CFGetTypeID( property ) == CFNumberGetTypeID() )
        {
            CFNumberGetValue( static_cast<CFNumberRef>( property ), kCFNumberIntType, &cores );
        }

        CFRelease( property );
    }

    IOObjectRelease( service );

    return cores;
}
