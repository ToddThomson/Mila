/*
 * The mock's runtime (section 3.2): a version from Version.txt, and one CUDA device once initialized.
 */

#include "Mock.Internal.h"

#include <stdbool.h>
#include <string.h>

#ifndef MILA_MOCK_VERSION
#error "MILA_MOCK_VERSION is defined by the mock's CMakeLists.txt, from Version.txt"
#endif

static bool initialized = false;

MILA_API int32_t mila_abi_version( void )
{
    return MILA_ABI_VERSION;
}

MILA_API mila_status mila_initialize( const char* log_level )
{
    mock_clear_error();

    if ( log_level == NULL )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "log_level must not be null" );
    }

    static const char* const levels[] = { "trace", "info", "warning", "error" };
    bool known = false;

    for ( size_t index = 0; index < sizeof( levels ) / sizeof( levels[0] ); ++index )
    {
        if ( strcmp( log_level, levels[index] ) == 0 )
        {
            known = true;
        }
    }

    if ( !known )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT,
            "log_level must be trace, info, warning or error, not '%s'", log_level );
    }

    initialized = true;

    return MILA_OK;
}

MILA_API mila_status mila_version( char** out_version )
{
    mock_clear_error();

    if ( out_version == NULL )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "out_version must not be null" );
    }

    return mock_copy_out( MILA_MOCK_VERSION, out_version );
}

MILA_API mila_status mila_cuda_devices_json( char** out_json )
{
    mock_clear_error();

    if ( out_json == NULL )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "out_json must not be null" );
    }

    if ( !initialized )
    {
        return mock_copy_out( "[]", out_json );
    }

    return mock_copy_out(
        "[{\"index\":0,\"name\":\"Mila mock device\",\"compute_capability_major\":12,"
        "\"compute_capability_minor\":0,\"total_memory_bytes\":17179869184,"
        "\"pci_domain\":0,\"pci_bus\":1,\"pci_device\":0}]",
        out_json );
}
