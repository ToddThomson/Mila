/*
 * The mock's error message, text buffer and the ABI's two free functions.
 */

#include "Mock.Internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined( _MSC_VER )
#define MOCK_THREAD_LOCAL __declspec( thread )
#else
#define MOCK_THREAD_LOCAL _Thread_local
#endif

/* A fixed buffer keeps the message valid until the thread's next call without an allocation to leak at
 * thread exit; a longer message is truncated. */
static MOCK_THREAD_LOCAL char last_error_message[1024];

void mock_clear_error( void )
{
    last_error_message[0] = '\0';
}

mila_status mock_fail( mila_status status, const char* format, ... )
{
    va_list arguments;
    va_start( arguments, format );
    vsnprintf( last_error_message, sizeof( last_error_message ), format, arguments );
    va_end( arguments );

    return status;
}

MILA_API const char* mila_last_error_message( void )
{
    return last_error_message;
}

MILA_API void mila_string_free( char* text )
{
    free( text );
}

MILA_API void mila_tokens_free( int32_t* tokens )
{
    free( tokens );
}

static bool mock_buffer_reserve( mock_buffer* buffer, size_t additional )
{
    if ( buffer->failed )
    {
        return false;
    }

    size_t required = buffer->length + additional + 1;

    if ( required <= buffer->capacity )
    {
        return true;
    }

    size_t capacity = buffer->capacity == 0 ? 64 : buffer->capacity;

    while ( capacity < required )
    {
        capacity *= 2;
    }

    char* data = realloc( buffer->data, capacity );

    if ( data == NULL )
    {
        buffer->failed = true;

        return false;
    }

    buffer->data = data;
    buffer->capacity = capacity;

    return true;
}

void mock_buffer_append( mock_buffer* buffer, const char* text, size_t length )
{
    if ( !mock_buffer_reserve( buffer, length ) )
    {
        return;
    }

    memcpy( buffer->data + buffer->length, text, length );
    buffer->length += length;
    buffer->data[buffer->length] = '\0';
}

void mock_buffer_append_text( mock_buffer* buffer, const char* text )
{
    mock_buffer_append( buffer, text, strlen( text ) );
}

void mock_buffer_append_char( mock_buffer* buffer, char character )
{
    mock_buffer_append( buffer, &character, 1 );
}

void mock_buffer_append_json_string( mock_buffer* buffer, const char* text )
{
    mock_buffer_append_char( buffer, '"' );

    for ( const unsigned char* at = (const unsigned char*)text; *at != '\0'; ++at )
    {
        unsigned char character = *at;

        if ( character == '"' || character == '\\' )
        {
            mock_buffer_append_char( buffer, '\\' );
            mock_buffer_append_char( buffer, (char)character );
        }
        else if ( character < 0x20 )
        {
            char escaped[8];
            snprintf( escaped, sizeof( escaped ), "\\u%04x", (unsigned int)character );
            mock_buffer_append_text( buffer, escaped );
        }
        else
        {
            mock_buffer_append_char( buffer, (char)character );
        }
    }

    mock_buffer_append_char( buffer, '"' );
}

char* mock_buffer_release( mock_buffer* buffer )
{
    if ( buffer->failed )
    {
        mock_buffer_free( buffer );

        return NULL;
    }

    if ( !mock_buffer_reserve( buffer, 0 ) )
    {
        mock_buffer_free( buffer );

        return NULL;
    }

    char* data = buffer->data;
    data[buffer->length] = '\0';
    buffer->data = NULL;
    buffer->length = 0;
    buffer->capacity = 0;

    return data;
}

void mock_buffer_free( mock_buffer* buffer )
{
    free( buffer->data );
    buffer->data = NULL;
    buffer->length = 0;
    buffer->capacity = 0;
    buffer->failed = false;
}

mila_status mock_copy_out( const char* text, char** out_text )
{
    size_t length = strlen( text );
    char* copy = malloc( length + 1 );

    if ( copy == NULL )
    {
        return mock_fail( MILA_ERROR_RUNTIME, "out of memory" );
    }

    memcpy( copy, text, length + 1 );
    *out_text = copy;

    return MILA_OK;
}

mila_status mock_release_out( mock_buffer* buffer, char** out_text )
{
    char* text = mock_buffer_release( buffer );

    if ( text == NULL )
    {
        return mock_fail( MILA_ERROR_RUNTIME, "out of memory" );
    }

    *out_text = text;

    return MILA_OK;
}
