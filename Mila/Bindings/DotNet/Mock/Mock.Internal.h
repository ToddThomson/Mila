/*
 * What the mock's translation units share: the thread's error message, a growable text buffer, a
 * minimal JSON reader, and the fixed store.
 */

#ifndef MILA_MOCK_INTERNAL_H
#define MILA_MOCK_INTERNAL_H

#include "mila.h"

#include <stdbool.h>
#include <stddef.h>

/* ---- Errors -------------------------------------------------------------------------------------- */

/* Every exported body that returns a status calls this first, so a success leaves the message "". */
void mock_clear_error( void );

/* Records the message for the calling thread and returns status. */
mila_status mock_fail( mila_status status, const char* format, ... );

/* ---- Text ---------------------------------------------------------------------------------------- */

/*
 * A growable, null-terminated buffer. An allocation failure is remembered rather than reported at each
 * append, so a writer checks once, at mock_buffer_release.
 */
typedef struct mock_buffer
{
    char* data;
    size_t length;
    size_t capacity;
    bool failed;
} mock_buffer;

void mock_buffer_append( mock_buffer* buffer, const char* text, size_t length );
void mock_buffer_append_text( mock_buffer* buffer, const char* text );
void mock_buffer_append_char( mock_buffer* buffer, char character );

/* Appends text as a quoted JSON string. */
void mock_buffer_append_json_string( mock_buffer* buffer, const char* text );

/* The text, which the caller frees with free(), or null after an allocation failure. Empties the buffer. */
char* mock_buffer_release( mock_buffer* buffer );

void mock_buffer_free( mock_buffer* buffer );

/* A malloc'd copy for an out-parameter, or MILA_ERROR_RUNTIME when out of memory. */
mila_status mock_copy_out( const char* text, char** out_text );

/* Releases buffer into out_text, or MILA_ERROR_RUNTIME when an append ran out of memory. */
mila_status mock_release_out( mock_buffer* buffer, char** out_text );

/* ---- JSON reader --------------------------------------------------------------------------------- */

typedef struct mock_json_reader
{
    const char* at;
} mock_json_reader;

void mock_json_skip_whitespace( mock_json_reader* reader );

/* Skips whitespace, then consumes expected if it is next. */
bool mock_json_consume( mock_json_reader* reader, char expected );

/* Skips whitespace, then consumes the literal null if it is next. */
bool mock_json_consume_null( mock_json_reader* reader );

/* Reads a quoted string, unescaped to UTF-8, appending it to out. */
bool mock_json_read_string( mock_json_reader* reader, mock_buffer* out );

/* Skips one value of any kind. */
bool mock_json_skip_value( mock_json_reader* reader );

/* ---- The fixed store ----------------------------------------------------------------------------- */

typedef struct mock_model_record
{
    const char* name;
    const char* architecture;
    const char* family;
    const char* variant;
    const char* modalities_json;
    bool complete;
    bool begins_in_reasoning;
    unsigned long long bytes_on_disk;
} mock_model_record;

size_t mock_store_count( void );
const mock_model_record* mock_store_at( size_t index );

/* The record of that name, or null. */
const mock_model_record* mock_store_find( const char* name );

#endif
