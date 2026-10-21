/*
 * The mock's fixed store (section 5): two loadable instruct models and one installed but incomplete.
 */

#include "Mock.Internal.h"

#include <stdio.h>
#include <string.h>

static const mock_model_record records[] = {
    { "mock-gemma-it", "gemma", "gemma", "it", "[\"text\",\"image\"]", true, true, 7340032000ull },
    { "mock-llama-it", "llama", "llama", "it", "[\"text\"]", true, false, 4613734400ull },
    { "mock-incomplete", "qwen", "qwen", "it", "[\"text\"]", false, false, 0ull },
};

size_t mock_store_count( void )
{
    return sizeof( records ) / sizeof( records[0] );
}

const mock_model_record* mock_store_at( size_t index )
{
    return &records[index];
}

const mock_model_record* mock_store_find( const char* name )
{
    for ( size_t index = 0; index < mock_store_count(); ++index )
    {
        if ( strcmp( records[index].name, name ) == 0 )
        {
            return &records[index];
        }
    }

    return NULL;
}

static void append_field( mock_buffer* json, const char* key, const char* value )
{
    mock_buffer_append_json_string( json, key );
    mock_buffer_append_char( json, ':' );
    mock_buffer_append_json_string( json, value );
    mock_buffer_append_char( json, ',' );
}

static void append_flag( mock_buffer* json, const char* key, bool value )
{
    mock_buffer_append_json_string( json, key );
    mock_buffer_append_char( json, ':' );
    mock_buffer_append_text( json, value ? "true" : "false" );
    mock_buffer_append_char( json, ',' );
}

static void append_record( mock_buffer* json, const mock_model_record* record )
{
    char weights_path[256];
    char tokenizer_path[256];
    char bytes_on_disk[32];
    snprintf( weights_path, sizeof( weights_path ), "/mock/store/%s/weights.safetensors", record->name );
    snprintf( tokenizer_path, sizeof( tokenizer_path ), "/mock/store/%s/tokenizer.json", record->name );
    snprintf( bytes_on_disk, sizeof( bytes_on_disk ), "%llu", record->bytes_on_disk );

    mock_buffer_append_char( json, '{' );
    append_field( json, "name", record->name );
    append_field( json, "architecture", record->architecture );
    append_field( json, "family", record->family );
    append_field( json, "variant", record->variant );
    append_field( json, "weight_quantization", "fp4" );
    append_flag( json, "instruct", true );
    append_flag( json, "thinking", record->begins_in_reasoning );
    append_flag( json, "tools", true );
    append_flag( json, "agentic", false );
    append_flag( json, "mtp", false );
    mock_buffer_append_text( json, "\"modalities\":" );
    mock_buffer_append_text( json, record->modalities_json );
    mock_buffer_append_char( json, ',' );
    append_field( json, "base_model", "" );
    append_field( json, "license", "apache-2.0" );
    append_field( json, "hub", "" );
    append_field( json, "owner", "" );
    append_field( json, "repository", "" );
    append_field( json, "revision", "" );
    append_field( json, "installed_at", "2026-10-21T00:00:00Z" );
    append_field( json, "weights_path", weights_path );
    append_field( json, "tokenizer_path", tokenizer_path );
    append_flag( json, "complete", record->complete );
    mock_buffer_append_text( json, "\"bytes_on_disk\":" );
    mock_buffer_append_text( json, bytes_on_disk );
    mock_buffer_append_char( json, '}' );
}

MILA_API mila_status mila_store_list_json( char** out_json )
{
    mock_clear_error();

    if ( out_json == NULL )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "out_json must not be null" );
    }

    mock_buffer json = { 0 };
    mock_buffer_append_char( &json, '[' );

    for ( size_t index = 0; index < mock_store_count(); ++index )
    {
        if ( index > 0 )
        {
            mock_buffer_append_char( &json, ',' );
        }

        append_record( &json, mock_store_at( index ) );
    }

    mock_buffer_append_char( &json, ']' );

    return mock_release_out( &json, out_json );
}
