/*
 * The mock's chat protocol (section 5): turns render as <role>content</role> lines and a reply's
 * <think>...</think> prefix parses as reasoning. A gemma-family model's reply begins in reasoning.
 */

#include "Mock.Internal.h"

#include <stdlib.h>
#include <string.h>

struct mila_chat_protocol
{
    const mock_model_record* record;
};

MILA_API mila_status mila_chat_protocol_for_installed_model( const char* name, mila_chat_protocol** out_protocol )
{
    mock_clear_error();

    if ( name == NULL || out_protocol == NULL )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "name and out_protocol must not be null" );
    }

    const mock_model_record* record = mock_store_find( name );

    if ( record == NULL )
    {
        return mock_fail( MILA_ERROR_NOT_FOUND, "no installed model is named '%s'", name );
    }

    mila_chat_protocol* protocol = malloc( sizeof( mila_chat_protocol ) );

    if ( protocol == NULL )
    {
        return mock_fail( MILA_ERROR_RUNTIME, "out of memory" );
    }

    protocol->record = record;
    *out_protocol = protocol;

    return MILA_OK;
}

/* A turn's text field: a string, or null or absent for empty, as the surface reads it. */
static mila_status read_text_field( mock_json_reader* reader, const char* key, mock_buffer* out )
{
    if ( mock_json_consume_null( reader ) )
    {
        return MILA_OK;
    }

    if ( !mock_json_read_string( reader, out ) )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "a turn's '%s' must be text", key );
    }

    return MILA_OK;
}

/* Renders each call as <tool_call name="...">arguments</tool_call>. */
static mila_status read_tool_call( mock_json_reader* reader, mock_buffer* rendered )
{
    if ( !mock_json_consume( reader, '{' ) )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "each tool call must be a mapping" );
    }

    mock_buffer name = { 0 };
    mock_buffer arguments = { 0 };
    mila_status status = MILA_OK;

    if ( !mock_json_consume( reader, '}' ) )
    {
        for ( ;; )
        {
            mock_buffer key = { 0 };

            if ( !mock_json_read_string( reader, &key ) || !mock_json_consume( reader, ':' ) )
            {
                mock_buffer_free( &key );
                status = mock_fail( MILA_ERROR_INVALID_ARGUMENT, "a tool call is not valid JSON" );
                break;
            }

            if ( key.data != NULL && strcmp( key.data, "name" ) == 0 )
            {
                status = read_text_field( reader, "name", &name );
            }
            else if ( key.data != NULL && strcmp( key.data, "arguments" ) == 0 )
            {
                status = read_text_field( reader, "arguments", &arguments );
            }
            else if ( !mock_json_skip_value( reader ) )
            {
                status = mock_fail( MILA_ERROR_INVALID_ARGUMENT, "a tool call is not valid JSON" );
            }

            mock_buffer_free( &key );

            if ( status != MILA_OK || mock_json_consume( reader, '}' ) )
            {
                break;
            }

            if ( !mock_json_consume( reader, ',' ) )
            {
                status = mock_fail( MILA_ERROR_INVALID_ARGUMENT, "a tool call is not valid JSON" );
                break;
            }
        }
    }

    if ( status == MILA_OK )
    {
        mock_buffer_append_text( rendered, "<tool_call name=\"" );
        mock_buffer_append_text( rendered, name.data != NULL ? name.data : "" );
        mock_buffer_append_text( rendered, "\">" );
        mock_buffer_append_text( rendered, arguments.data != NULL ? arguments.data : "" );
        mock_buffer_append_text( rendered, "</tool_call>" );
    }

    mock_buffer_free( &name );
    mock_buffer_free( &arguments );

    return status;
}

static mila_status read_tool_calls( mock_json_reader* reader, mock_buffer* rendered )
{
    if ( mock_json_consume_null( reader ) )
    {
        return MILA_OK;
    }

    if ( !mock_json_consume( reader, '[' ) )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "a turn's tool_calls must be a list" );
    }

    if ( mock_json_consume( reader, ']' ) )
    {
        return MILA_OK;
    }

    for ( ;; )
    {
        mila_status status = read_tool_call( reader, rendered );

        if ( status != MILA_OK )
        {
            return status;
        }

        if ( mock_json_consume( reader, ']' ) )
        {
            return MILA_OK;
        }

        if ( !mock_json_consume( reader, ',' ) )
        {
            return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "a turn's tool_calls is not valid JSON" );
        }
    }
}

static bool is_known_role( const char* role )
{
    return strcmp( role, "system" ) == 0 || strcmp( role, "user" ) == 0 || strcmp( role, "assistant" ) == 0 ||
        strcmp( role, "tool" ) == 0;
}

static mila_status read_turn_fields( mock_json_reader* reader, mock_buffer* role, mock_buffer* content, mock_buffer* calls )
{
    if ( mock_json_consume( reader, '}' ) )
    {
        return MILA_OK;
    }

    for ( ;; )
    {
        mock_buffer key = { 0 };

        if ( !mock_json_read_string( reader, &key ) || !mock_json_consume( reader, ':' ) )
        {
            mock_buffer_free( &key );

            return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "history is not valid JSON" );
        }

        const char* name = key.data != NULL ? key.data : "";
        mila_status status = MILA_OK;

        if ( strcmp( name, "role" ) == 0 )
        {
            status = read_text_field( reader, "role", role );
        }
        else if ( strcmp( name, "content" ) == 0 )
        {
            status = read_text_field( reader, "content", content );
        }
        else if ( strcmp( name, "tool_calls" ) == 0 )
        {
            status = read_tool_calls( reader, calls );
        }
        else if ( !mock_json_skip_value( reader ) )
        {
            status = mock_fail( MILA_ERROR_INVALID_ARGUMENT, "history is not valid JSON" );
        }

        mock_buffer_free( &key );

        if ( status != MILA_OK )
        {
            return status;
        }

        if ( mock_json_consume( reader, '}' ) )
        {
            return MILA_OK;
        }

        if ( !mock_json_consume( reader, ',' ) )
        {
            return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "history is not valid JSON" );
        }
    }
}

static mila_status render_turn( mock_json_reader* reader, mock_buffer* prompt )
{
    if ( !mock_json_consume( reader, '{' ) )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "each turn must be a mapping" );
    }

    mock_buffer role = { 0 };
    mock_buffer content = { 0 };
    mock_buffer calls = { 0 };
    mila_status status = read_turn_fields( reader, &role, &content, &calls );
    const char* role_name = role.data != NULL ? role.data : "";

    if ( status == MILA_OK && !is_known_role( role_name ) )
    {
        status = mock_fail( MILA_ERROR_INVALID_ARGUMENT,
            "a turn's role must be system, user, assistant or tool, not '%s'", role_name );
    }

    if ( status == MILA_OK )
    {
        mock_buffer_append_char( prompt, '<' );
        mock_buffer_append_text( prompt, role_name );
        mock_buffer_append_char( prompt, '>' );
        mock_buffer_append_text( prompt, content.data != NULL ? content.data : "" );
        mock_buffer_append_text( prompt, calls.data != NULL ? calls.data : "" );
        mock_buffer_append_text( prompt, "</" );
        mock_buffer_append_text( prompt, role_name );
        mock_buffer_append_text( prompt, ">\n" );
    }

    mock_buffer_free( &role );
    mock_buffer_free( &content );
    mock_buffer_free( &calls );

    return status;
}

static mila_status render_history( const char* history_json, mock_buffer* prompt )
{
    mock_json_reader reader = { history_json };

    if ( !mock_json_consume( &reader, '[' ) )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "history must be a list of turns" );
    }

    if ( mock_json_consume( &reader, ']' ) )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "history must hold at least one turn" );
    }

    for ( ;; )
    {
        mila_status status = render_turn( &reader, prompt );

        if ( status != MILA_OK )
        {
            return status;
        }

        if ( mock_json_consume( &reader, ']' ) )
        {
            break;
        }

        if ( !mock_json_consume( &reader, ',' ) )
        {
            return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "history is not valid JSON" );
        }
    }

    mock_json_skip_whitespace( &reader );

    if ( *reader.at != '\0' )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "history is not valid JSON" );
    }

    return MILA_OK;
}

MILA_API mila_status mila_chat_protocol_format_prompt(
    mila_chat_protocol* protocol,
    const char* history_json,
    const char* tools_json,
    char** out_prompt,
    int32_t* out_begins_in_reasoning )
{
    mock_clear_error();

    if ( protocol == NULL || history_json == NULL || out_prompt == NULL || out_begins_in_reasoning == NULL )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT,
            "protocol, history_json, out_prompt and out_begins_in_reasoning must not be null" );
    }

    mock_buffer prompt = { 0 };

    if ( tools_json != NULL )
    {
        mock_json_reader tools = { tools_json };

        if ( !mock_json_consume( &tools, '[' ) )
        {
            return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "tools must be a list of tool schemas" );
        }

        mock_buffer_append_text( &prompt, "<tools>" );
        mock_buffer_append_text( &prompt, tools_json );
        mock_buffer_append_text( &prompt, "</tools>\n" );
    }

    mila_status status = render_history( history_json, &prompt );

    if ( status != MILA_OK )
    {
        mock_buffer_free( &prompt );

        return status;
    }

    bool begins_in_reasoning = protocol->record->begins_in_reasoning;
    mock_buffer_append_text( &prompt, begins_in_reasoning ? "<assistant><think>" : "<assistant>" );
    status = mock_release_out( &prompt, out_prompt );

    if ( status == MILA_OK )
    {
        *out_begins_in_reasoning = begins_in_reasoning ? 1 : 0;
    }

    return status;
}

MILA_API mila_status mila_chat_protocol_parse_reply(
    mila_chat_protocol* protocol, const char* text, int32_t begins_in_reasoning, char** out_reply_json )
{
    mock_clear_error();

    if ( protocol == NULL || text == NULL || out_reply_json == NULL )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "protocol, text and out_reply_json must not be null" );
    }

    static const char open[] = "<think>";
    static const char close[] = "</think>";
    const char* reasoning_start = NULL;

    if ( begins_in_reasoning != 0 )
    {
        reasoning_start = text;
    }
    else if ( strncmp( text, open, sizeof( open ) - 1 ) == 0 )
    {
        reasoning_start = text + sizeof( open ) - 1;
    }

    mock_buffer reasoning = { 0 };
    const char* content = text;

    if ( reasoning_start != NULL )
    {
        const char* reasoning_end = strstr( reasoning_start, close );

        if ( reasoning_end == NULL )
        {
            mock_buffer_append_text( &reasoning, reasoning_start );
            content = reasoning_start + strlen( reasoning_start );
        }
        else
        {
            mock_buffer_append( &reasoning, reasoning_start, (size_t)( reasoning_end - reasoning_start ) );
            content = reasoning_end + sizeof( close ) - 1;
        }
    }

    mock_buffer json = { 0 };
    mock_buffer_append_text( &json, "{\"reasoning\":" );
    mock_buffer_append_json_string( &json, reasoning.data != NULL ? reasoning.data : "" );
    mock_buffer_append_text( &json, ",\"content\":" );
    mock_buffer_append_json_string( &json, content );
    mock_buffer_append_text( &json, ",\"tool_calls\":[],\"malformed_tool_calls\":[]}" );
    mock_buffer_free( &reasoning );

    return mock_release_out( &json, out_reply_json );
}

MILA_API void mila_chat_protocol_free( mila_chat_protocol* protocol )
{
    free( protocol );
}
