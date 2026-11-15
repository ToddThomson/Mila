/**
 * @file NetworkConnection.c
 * @brief One connection to one address through Network.framework, confirmed direct before a byte is sent.
 *
 * C, since Network.framework is a C API, and so no C++ crosses into it (HttpClient.md section 9). Each call waits on
 * the connection's own serial queue, where every completion runs; a completion owns nothing of its caller's, so one
 * that arrives after its caller stopped waiting writes only into the connection, which outlives it.
 */

#include <CoreFoundation/CoreFoundation.h>
#include <Network/Network.h>
#include <Security/Security.h>
#include <dispatch/dispatch.h>
#include <netinet/in.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "NetworkConnection.h"

enum
{
    kReceiveCapacity = 64 * 1024,

    /* What a connection waits to be made when the caller sets no limit: curl's own default. */
    kDefaultConnectSeconds = 300
};

struct MilaNetworkConnection
{
    nw_connection_t connection;
    dispatch_queue_t queue;
    char* server_name;

    dispatch_semaphore_t settled;
    dispatch_semaphore_t cancelled;
    dispatch_semaphore_t sent;
    dispatch_semaphore_t received;

    /* Written on the queue; read there, or after the semaphore that follows the write. */
    bool has_settled;
    nw_connection_state_t state;
    nw_error_t state_error;
    nw_error_t send_error;
    nw_error_t receive_error;
    char* receive_buffer;
    size_t received_length;
    bool peer_closed;

    /* Sends and receives whose completion has not run, so close frees nothing a completion still writes. */
    atomic_int outstanding;

    /* A wait ran out: a late completion may still signal, so nothing more is issued. */
    bool broken;
};

static dispatch_time_t deadline( long seconds )
{
    return seconds > 0 ? dispatch_time( DISPATCH_TIME_NOW, (int64_t)seconds * (int64_t)NSEC_PER_SEC ) : DISPATCH_TIME_FOREVER;
}

static void keepError( nw_error_t* slot, nw_error_t error )
{
    if ( *slot != NULL )
    {
        nw_release( *slot );
    }

    *slot = error;

    if ( error != NULL )
    {
        nw_retain( error );
    }
}

static void describe( nw_error_t error, const char* what, MilaNetworkOutcome* outcome )
{
    if ( error == NULL )
    {
        snprintf( outcome->message, sizeof( outcome->message ), "%s", what );

        return;
    }

    const nw_error_domain_t domain = nw_error_get_error_domain( error );
    const int code = nw_error_get_error_code( error );

    if ( domain == nw_error_domain_posix )
    {
        snprintf( outcome->message, sizeof( outcome->message ), "%s: %s", what, strerror( code ) );
    }
    else if ( domain == nw_error_domain_tls )
    {
        char text[ 256 ] = "";
        const CFStringRef described = SecCopyErrorMessageString( (OSStatus)code, NULL );

        if ( described != NULL )
        {
            CFStringGetCString( described, text, sizeof( text ), kCFStringEncodingUTF8 );
            CFRelease( described );
        }

        snprintf( outcome->message, sizeof( outcome->message ), "%s: %s (TLS %d)", what,
            text[ 0 ] != '\0' ? text : "the certificate or the handshake was refused", code );
    }
    else if ( domain == nw_error_domain_dns )
    {
        snprintf( outcome->message, sizeof( outcome->message ), "%s: the name did not resolve (DNS %d)", what, code );
    }
    else
    {
        snprintf( outcome->message, sizeof( outcome->message ), "%s (error %d)", what, code );
    }
}

static bool sameAddress( const struct sockaddr* left, const struct sockaddr* right )
{
    if ( left == NULL || right == NULL || left->sa_family != right->sa_family )
    {
        return false;
    }

    if ( left->sa_family == AF_INET )
    {
        const struct sockaddr_in* a = (const struct sockaddr_in*)left;
        const struct sockaddr_in* b = (const struct sockaddr_in*)right;

        return a->sin_port == b->sin_port && a->sin_addr.s_addr == b->sin_addr.s_addr;
    }

    if ( left->sa_family == AF_INET6 )
    {
        const struct sockaddr_in6* a = (const struct sockaddr_in6*)left;
        const struct sockaddr_in6* b = (const struct sockaddr_in6*)right;

        return a->sin6_port == b->sin6_port && memcmp( &a->sin6_addr, &b->sin6_addr, sizeof( a->sin6_addr ) ) == 0;
    }

    return false;
}

/* Whether the connection's path sends data to this address, the one the caller's check allowed. */
static bool reachesOnly( MilaNetworkConnection* state, const struct sockaddr* address )
{
    __block bool same = false;

    dispatch_sync( state->queue, ^{
        const nw_path_t path = nw_connection_copy_current_path( state->connection );

        if ( path == NULL )
        {
            return;
        }

        const nw_endpoint_t remote = nw_path_copy_effective_remote_endpoint( path );

        if ( remote != NULL )
        {
            same = nw_endpoint_get_type( remote ) == nw_endpoint_type_address
                && sameAddress( nw_endpoint_get_address( remote ), address );
            nw_release( remote );
        }

        nw_release( path );
    } );

    return same;
}

/* Whether the connection was made without a proxy; a report that cannot be read is not taken as no. */
static bool madeWithoutProxy( MilaNetworkConnection* state )
{
    __block bool direct = false;
    const dispatch_semaphore_t reported = dispatch_semaphore_create( 0 );

    nw_connection_access_establishment_report( state->connection, state->queue, ^( nw_establishment_report_t report ) {
        direct = report != NULL && !nw_establishment_report_get_used_proxy( report );
        dispatch_semaphore_signal( reported );
    } );

    // Its block runs exactly once (connection_report.h), so this wait ends.
    dispatch_semaphore_wait( reported, DISPATCH_TIME_FOREVER );
    dispatch_release( reported );

    return direct;
}

MilaNetworkConnection* milaNetworkConnect( const struct sockaddr* address, const char* tls_server_name,
    long timeout_seconds, MilaNetworkOutcome* outcome )
{
    memset( outcome, 0, sizeof( *outcome ) );

    MilaNetworkConnection* state = calloc( 1, sizeof( MilaNetworkConnection ) );
    char* buffer = malloc( kReceiveCapacity );

    if ( state == NULL || buffer == NULL )
    {
        free( state );
        free( buffer );
        outcome->result = MilaNetworkFailed;
        snprintf( outcome->message, sizeof( outcome->message ), "no memory for a connection" );

        return NULL;
    }

    state->receive_buffer = buffer;
    state->queue = dispatch_queue_create( "Mila.NetworkConnection", DISPATCH_QUEUE_SERIAL );
    state->settled = dispatch_semaphore_create( 0 );
    state->cancelled = dispatch_semaphore_create( 0 );
    state->sent = dispatch_semaphore_create( 0 );
    state->received = dispatch_semaphore_create( 0 );
    state->server_name = tls_server_name != NULL ? strdup( tls_server_name ) : NULL;
    atomic_init( &state->outstanding, 0 );

    nw_parameters_t parameters;

    if ( state->server_name != NULL )
    {
        const char* name = state->server_name;

        // The name the certificate must carry, since the endpoint is an address and names no host.
        parameters = nw_parameters_create_secure_tcp( ^( nw_protocol_options_t options ) {
            const sec_protocol_options_t security = nw_tls_copy_sec_protocol_options( options );
            sec_protocol_options_set_tls_server_name( security, name );
            sec_protocol_options_add_tls_application_protocol( security, "http/1.1" );
            sec_release( security );
        }, NW_PARAMETERS_DEFAULT_CONFIGURATION );
    }
    else
    {
        parameters = nw_parameters_create_secure_tcp( NW_PARAMETERS_DISABLE_PROTOCOL, NW_PARAMETERS_DEFAULT_CONFIGURATION );
    }

    // A preference only: a proxy may still be taken, which madeWithoutProxy finds.
    nw_parameters_set_prefer_no_proxy( parameters, true );

    const nw_endpoint_t endpoint = nw_endpoint_create_address( address );
    state->connection = nw_connection_create( endpoint, parameters );
    nw_release( endpoint );
    nw_release( parameters );

    nw_connection_set_queue( state->connection, state->queue );
    nw_connection_set_state_changed_handler( state->connection, ^( nw_connection_state_t now, nw_error_t error ) {
        state->state = now;

        if ( error != NULL )
        {
            keepError( &state->state_error, error );
        }

        if ( !state->has_settled
            && ( now == nw_connection_state_ready || now == nw_connection_state_failed || now == nw_connection_state_waiting ) )
        {
            state->has_settled = true;
            dispatch_semaphore_signal( state->settled );
        }

        if ( now == nw_connection_state_cancelled )
        {
            dispatch_semaphore_signal( state->cancelled );
        }
    } );

    nw_connection_start( state->connection );

    const long connect_seconds = timeout_seconds > 0 ? timeout_seconds : kDefaultConnectSeconds;

    if ( dispatch_semaphore_wait( state->settled, deadline( connect_seconds ) ) != 0 )
    {
        outcome->result = MilaNetworkTimedOut;
        snprintf( outcome->message, sizeof( outcome->message ), "no connection was made within %ld seconds", connect_seconds );
        milaNetworkClose( state );

        return NULL;
    }

    __block nw_connection_state_t settled = nw_connection_state_invalid;
    __block nw_error_t error = NULL;

    dispatch_sync( state->queue, ^{
        settled = state->state;
        error = state->state_error;

        if ( error != NULL )
        {
            nw_retain( error );
        }
    } );

    if ( settled != nw_connection_state_ready )
    {
        // Waiting means the system found no way to make it now and would try again later; to a caller that is a
        // connection not made.
        outcome->result = MilaNetworkFailed;
        describe( error, "the connection was not made", outcome );

        if ( error != NULL )
        {
            nw_release( error );
        }

        milaNetworkClose( state );

        return NULL;
    }

    if ( error != NULL )
    {
        nw_release( error );
    }

    if ( !reachesOnly( state, address ) || !madeWithoutProxy( state ) )
    {
        outcome->result = MilaNetworkNotDirect;
        snprintf( outcome->message, sizeof( outcome->message ),
            "the connection did not go straight to the address allowed, and a tool's network uses no proxy" );
        milaNetworkClose( state );

        return NULL;
    }

    outcome->result = MilaNetworkOk;

    return state;
}

MilaNetworkResult milaNetworkSend( MilaNetworkConnection* state, const char* data, size_t length,
    long timeout_seconds, MilaNetworkOutcome* outcome )
{
    memset( outcome, 0, sizeof( *outcome ) );

    if ( state->broken )
    {
        outcome->result = MilaNetworkFailed;
        snprintf( outcome->message, sizeof( outcome->message ), "the connection already timed out" );

        return outcome->result;
    }

    const dispatch_data_t content = dispatch_data_create( data, length, state->queue, DISPATCH_DATA_DESTRUCTOR_DEFAULT );
    atomic_fetch_add( &state->outstanding, 1 );

    nw_connection_send( state->connection, content, NW_CONNECTION_DEFAULT_MESSAGE_CONTEXT, true, ^( nw_error_t error ) {
        keepError( &state->send_error, error );
        atomic_fetch_sub( &state->outstanding, 1 );
        dispatch_semaphore_signal( state->sent );
    } );

    dispatch_release( content );

    if ( dispatch_semaphore_wait( state->sent, deadline( timeout_seconds ) ) != 0 )
    {
        state->broken = true;
        outcome->result = MilaNetworkTimedOut;
        snprintf( outcome->message, sizeof( outcome->message ), "the request was not taken within %ld seconds", timeout_seconds );

        return outcome->result;
    }

    __block nw_error_t error = NULL;

    dispatch_sync( state->queue, ^{
        error = state->send_error;

        if ( error != NULL )
        {
            nw_retain( error );
        }
    } );

    if ( error != NULL )
    {
        outcome->result = MilaNetworkFailed;
        describe( error, "the request was not sent", outcome );
        nw_release( error );

        return outcome->result;
    }

    outcome->result = MilaNetworkOk;

    return outcome->result;
}

MilaNetworkResult milaNetworkReceive( MilaNetworkConnection* state, char* buffer, size_t capacity,
    size_t* received, long timeout_seconds, MilaNetworkOutcome* outcome )
{
    memset( outcome, 0, sizeof( *outcome ) );
    *received = 0;

    if ( state->broken )
    {
        outcome->result = MilaNetworkFailed;
        snprintf( outcome->message, sizeof( outcome->message ), "the connection already timed out" );

        return outcome->result;
    }

    if ( state->peer_closed )
    {
        outcome->result = MilaNetworkClosed;

        return outcome->result;
    }

    const uint32_t most = (uint32_t)( capacity < (size_t)kReceiveCapacity ? capacity : (size_t)kReceiveCapacity );
    atomic_fetch_add( &state->outstanding, 1 );

    nw_connection_receive( state->connection, 1, most, ^( dispatch_data_t content, nw_content_context_t context, bool is_complete, nw_error_t error ) {
        state->received_length = 0;

        if ( content != NULL )
        {
            dispatch_data_apply( content, ^bool( dispatch_data_t region, size_t offset, const void* bytes, size_t size ) {
                memcpy( state->receive_buffer + offset, bytes, size );

                return true;
            } );

            state->received_length = dispatch_data_get_size( content );
        }

        // On a stream the reply ends with the peer's close, which arrives as a complete final context.
        if ( is_complete && ( context == NULL || nw_content_context_get_is_final( context ) ) )
        {
            state->peer_closed = true;
        }

        keepError( &state->receive_error, error );
        atomic_fetch_sub( &state->outstanding, 1 );
        dispatch_semaphore_signal( state->received );
    } );

    if ( dispatch_semaphore_wait( state->received, deadline( timeout_seconds ) ) != 0 )
    {
        state->broken = true;
        outcome->result = MilaNetworkTimedOut;
        snprintf( outcome->message, sizeof( outcome->message ), "no data arrived for %ld seconds", timeout_seconds );

        return outcome->result;
    }

    __block size_t length = 0;
    __block bool closed = false;
    __block nw_error_t error = NULL;

    dispatch_sync( state->queue, ^{
        length = state->received_length;
        closed = state->peer_closed;
        error = state->receive_error;

        if ( error != NULL )
        {
            nw_retain( error );
        }

        if ( length > 0 )
        {
            memcpy( buffer, state->receive_buffer, length );
        }
    } );

    *received = length;

    if ( length > 0 )
    {
        outcome->result = MilaNetworkOk;
    }
    else if ( closed )
    {
        outcome->result = MilaNetworkClosed;
    }
    else
    {
        outcome->result = MilaNetworkFailed;
        describe( error, "the reply was not received", outcome );
    }

    if ( error != NULL )
    {
        nw_release( error );
    }

    return outcome->result;
}

void milaNetworkClose( MilaNetworkConnection* state )
{
    if ( state == NULL )
    {
        return;
    }

    nw_connection_cancel( state->connection );

    // A connection that does not end, or a completion that does not run, keeps what it writes: it is left, not freed.
    if ( dispatch_semaphore_wait( state->cancelled, deadline( 10 ) ) != 0 )
    {
        return;
    }

    for ( int attempt = 0; attempt < 5000 && atomic_load( &state->outstanding ) > 0; ++attempt )
    {
        usleep( 1000 );
    }

    if ( atomic_load( &state->outstanding ) > 0 )
    {
        return;
    }

    dispatch_sync( state->queue, ^{} );

    nw_release( state->connection );
    keepError( &state->state_error, NULL );
    keepError( &state->send_error, NULL );
    keepError( &state->receive_error, NULL );
    dispatch_release( state->settled );
    dispatch_release( state->cancelled );
    dispatch_release( state->sent );
    dispatch_release( state->received );
    dispatch_release( state->queue );
    free( state->server_name );
    free( state->receive_buffer );
    free( state );
}
