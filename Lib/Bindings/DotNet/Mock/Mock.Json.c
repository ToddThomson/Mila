/*
 * The least JSON reading the mock's chat protocol needs: strings with every escape, and skipping any
 * other value.
 */

#include "Mock.Internal.h"

#include <string.h>

void mock_json_skip_whitespace( mock_json_reader* reader )
{
    while ( *reader->at == ' ' || *reader->at == '\t' || *reader->at == '\n' || *reader->at == '\r' )
    {
        ++reader->at;
    }
}

bool mock_json_consume( mock_json_reader* reader, char expected )
{
    mock_json_skip_whitespace( reader );

    if ( *reader->at != expected )
    {
        return false;
    }

    ++reader->at;

    return true;
}

static bool mock_json_consume_literal( mock_json_reader* reader, const char* literal )
{
    mock_json_skip_whitespace( reader );
    size_t length = strlen( literal );

    if ( strncmp( reader->at, literal, length ) != 0 )
    {
        return false;
    }

    reader->at += length;

    return true;
}

bool mock_json_consume_null( mock_json_reader* reader )
{
    return mock_json_consume_literal( reader, "null" );
}

static int mock_json_hex_digit( char character )
{
    if ( character >= '0' && character <= '9' )
    {
        return character - '0';
    }

    if ( character >= 'a' && character <= 'f' )
    {
        return character - 'a' + 10;
    }

    if ( character >= 'A' && character <= 'F' )
    {
        return character - 'A' + 10;
    }

    return -1;
}

static bool mock_json_read_hex4( mock_json_reader* reader, unsigned long* out_value )
{
    unsigned long value = 0;

    for ( int index = 0; index < 4; ++index )
    {
        int digit = mock_json_hex_digit( reader->at[index] );

        if ( digit < 0 )
        {
            return false;
        }

        value = value * 16 + (unsigned long)digit;
    }

    reader->at += 4;
    *out_value = value;

    return true;
}

static void mock_json_append_code_point( mock_buffer* out, unsigned long code_point )
{
    if ( code_point < 0x80 )
    {
        mock_buffer_append_char( out, (char)code_point );
    }
    else if ( code_point < 0x800 )
    {
        mock_buffer_append_char( out, (char)( 0xC0 | ( code_point >> 6 ) ) );
        mock_buffer_append_char( out, (char)( 0x80 | ( code_point & 0x3F ) ) );
    }
    else if ( code_point < 0x10000 )
    {
        mock_buffer_append_char( out, (char)( 0xE0 | ( code_point >> 12 ) ) );
        mock_buffer_append_char( out, (char)( 0x80 | ( ( code_point >> 6 ) & 0x3F ) ) );
        mock_buffer_append_char( out, (char)( 0x80 | ( code_point & 0x3F ) ) );
    }
    else
    {
        mock_buffer_append_char( out, (char)( 0xF0 | ( code_point >> 18 ) ) );
        mock_buffer_append_char( out, (char)( 0x80 | ( ( code_point >> 12 ) & 0x3F ) ) );
        mock_buffer_append_char( out, (char)( 0x80 | ( ( code_point >> 6 ) & 0x3F ) ) );
        mock_buffer_append_char( out, (char)( 0x80 | ( code_point & 0x3F ) ) );
    }
}

static bool mock_json_read_unicode_escape( mock_json_reader* reader, mock_buffer* out )
{
    unsigned long code_point = 0;

    if ( !mock_json_read_hex4( reader, &code_point ) )
    {
        return false;
    }

    if ( code_point >= 0xDC00 && code_point <= 0xDFFF )
    {
        return false;
    }

    if ( code_point >= 0xD800 && code_point <= 0xDBFF )
    {
        if ( reader->at[0] != '\\' || reader->at[1] != 'u' )
        {
            return false;
        }

        reader->at += 2;
        unsigned long low = 0;

        if ( !mock_json_read_hex4( reader, &low ) || low < 0xDC00 || low > 0xDFFF )
        {
            return false;
        }

        code_point = 0x10000 + ( ( code_point - 0xD800 ) << 10 ) + ( low - 0xDC00 );
    }

    mock_json_append_code_point( out, code_point );

    return true;
}

bool mock_json_read_string( mock_json_reader* reader, mock_buffer* out )
{
    if ( !mock_json_consume( reader, '"' ) )
    {
        return false;
    }

    while ( *reader->at != '"' )
    {
        char character = *reader->at;

        if ( character == '\0' || (unsigned char)character < 0x20 )
        {
            return false;
        }

        ++reader->at;

        if ( character != '\\' )
        {
            mock_buffer_append_char( out, character );
            continue;
        }

        char escape = *reader->at;
        ++reader->at;

        switch ( escape )
        {
        case '"':
        case '\\':
        case '/':
            mock_buffer_append_char( out, escape );
            break;
        case 'b':
            mock_buffer_append_char( out, '\b' );
            break;
        case 'f':
            mock_buffer_append_char( out, '\f' );
            break;
        case 'n':
            mock_buffer_append_char( out, '\n' );
            break;
        case 'r':
            mock_buffer_append_char( out, '\r' );
            break;
        case 't':
            mock_buffer_append_char( out, '\t' );
            break;
        case 'u':
            if ( !mock_json_read_unicode_escape( reader, out ) )
            {
                return false;
            }

            break;
        default:
            return false;
        }
    }

    ++reader->at;

    return true;
}

static bool mock_json_skip_number( mock_json_reader* reader )
{
    const char* start = reader->at;

    while ( strchr( "+-0123456789.eE", *reader->at ) != NULL && *reader->at != '\0' )
    {
        ++reader->at;
    }

    return reader->at != start;
}

static bool mock_json_skip_container( mock_json_reader* reader, char close, bool keyed )
{
    if ( mock_json_consume( reader, close ) )
    {
        return true;
    }

    for ( ;; )
    {
        if ( keyed )
        {
            mock_buffer key = { 0 };
            bool read = mock_json_read_string( reader, &key );
            mock_buffer_free( &key );

            if ( !read || !mock_json_consume( reader, ':' ) )
            {
                return false;
            }
        }

        if ( !mock_json_skip_value( reader ) )
        {
            return false;
        }

        if ( mock_json_consume( reader, close ) )
        {
            return true;
        }

        if ( !mock_json_consume( reader, ',' ) )
        {
            return false;
        }
    }
}

bool mock_json_skip_value( mock_json_reader* reader )
{
    mock_json_skip_whitespace( reader );
    char next = *reader->at;

    if ( next == '"' )
    {
        mock_buffer ignored = { 0 };
        bool read = mock_json_read_string( reader, &ignored );
        mock_buffer_free( &ignored );

        return read;
    }

    if ( next == '[' )
    {
        ++reader->at;

        return mock_json_skip_container( reader, ']', false );
    }

    if ( next == '{' )
    {
        ++reader->at;

        return mock_json_skip_container( reader, '}', true );
    }

    if ( mock_json_consume_literal( reader, "true" ) || mock_json_consume_literal( reader, "false" ) )
    {
        return true;
    }

    if ( mock_json_consume_null( reader ) )
    {
        return true;
    }

    return mock_json_skip_number( reader );
}
