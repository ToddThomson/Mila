/**
 * @file UrlSessionFetch.h
 * @brief One request through Apple's URL loading, as C: what UrlSessionHttpTransport calls (HttpClient.md section 9).
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MilaUrlSessionHeader
{
    const char* name;
    const char* value;
} MilaUrlSessionHeader;

typedef struct MilaUrlSessionRequest
{
    const char* url;
    const MilaUrlSessionHeader* headers;
    size_t header_count;

    /* A POST of these bytes when has_body is set, a GET when not. */
    const char* body;
    size_t body_length;
    int has_body;

    /* Seconds with no data before the transfer is abandoned; zero disables. */
    long idle_timeout_seconds;
} MilaUrlSessionRequest;

/* Called once with the reply's status and headers, before any of its body. */
typedef void ( *MilaUrlSessionHeadCallback )( void* context, long status, const char* location, uint64_t content_length );

/* Called with each run of the body; returns zero to stop the transfer. */
typedef int ( *MilaUrlSessionDataCallback )( void* context, const char* data, size_t length );

typedef struct MilaUrlSessionOutcome
{
    /* Connection, TLS or I/O failure, distinct from any HTTP status. */
    int transport_failed;

    /* The data callback stopped the transfer. */
    int stopped;

    char message[ 1024 ];
} MilaUrlSessionOutcome;

/* Performs the request and returns when it has ended; the callbacks run on another thread meanwhile, one at a time. */
void milaUrlSessionFetch( const MilaUrlSessionRequest* request, void* context,
    MilaUrlSessionHeadCallback on_head, MilaUrlSessionDataCallback on_data, MilaUrlSessionOutcome* outcome );

#ifdef __cplusplus
}
#endif
