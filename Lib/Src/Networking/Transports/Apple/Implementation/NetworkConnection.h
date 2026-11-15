/**
 * @file NetworkConnection.h
 * @brief One connection to one address through Network.framework, as C: what NetworkHttpTransport calls
 *        (HttpClient.md section 9).
 */

#pragma once

#include <stddef.h>
#include <sys/socket.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MilaNetworkConnection MilaNetworkConnection;

typedef enum MilaNetworkResult
{
    MilaNetworkOk = 0,
    MilaNetworkFailed = 1,
    MilaNetworkTimedOut = 2,

    /* The peer closed its side: no more bytes will arrive. */
    MilaNetworkClosed = 3,

    /* The connection reached somewhere other than the address given, or went through a proxy. */
    MilaNetworkNotDirect = 4
} MilaNetworkResult;

typedef struct MilaNetworkOutcome
{
    MilaNetworkResult result;
    char message[ 512 ];
} MilaNetworkOutcome;

/*
 * Connects to exactly this address, with TLS verified against tls_server_name when it is not NULL, and returns only
 * once the connection's own path says data goes to that address and its establishment report says no proxy was
 * used. NULL, with the outcome saying why, when it does not. Zero seconds waits as long as the system does.
 */
MilaNetworkConnection* milaNetworkConnect( const struct sockaddr* address, const char* tls_server_name,
    long timeout_seconds, MilaNetworkOutcome* outcome );

/* Sends all of these bytes. */
MilaNetworkResult milaNetworkSend( MilaNetworkConnection* connection, const char* data, size_t length,
    long timeout_seconds, MilaNetworkOutcome* outcome );

/* Receives up to capacity bytes into buffer; MilaNetworkClosed, with *received zero, once the peer has closed. */
MilaNetworkResult milaNetworkReceive( MilaNetworkConnection* connection, char* buffer, size_t capacity,
    size_t* received, long timeout_seconds, MilaNetworkOutcome* outcome );

/* Cancels the connection and waits for it to end. NULL is passed over. */
void milaNetworkClose( MilaNetworkConnection* connection );

#ifdef __cplusplus
}
#endif
