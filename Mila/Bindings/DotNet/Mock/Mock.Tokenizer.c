/*
 * The mock's byte tokenizer (section 5): one token per UTF-8 byte, id = byte + 3; ids 0 to 2 are bos, eos
 * and pad, and decode to nothing.
 */

#include "Mock.Internal.h"

#include <stdlib.h>
#include <string.h>

enum
{
    MOCK_TOKEN_EOS = 1,
    MOCK_FIRST_BYTE_TOKEN = 3,
    MOCK_VOCABULARY_SIZE = 256 + MOCK_FIRST_BYTE_TOKEN,
};

struct mila_tokenizer
{
    const mock_model_record* record;
};

MILA_API mila_status mila_tokenizer_from_store( const char* name, mila_tokenizer** out_tokenizer )
{
    mock_clear_error();

    if ( name == NULL || out_tokenizer == NULL )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "name and out_tokenizer must not be null" );
    }

    const mock_model_record* record = mock_store_find( name );

    if ( record == NULL )
    {
        return mock_fail( MILA_ERROR_NOT_FOUND, "no installed model is named '%s'", name );
    }

    mila_tokenizer* tokenizer = malloc( sizeof( mila_tokenizer ) );

    if ( tokenizer == NULL )
    {
        return mock_fail( MILA_ERROR_RUNTIME, "out of memory" );
    }

    tokenizer->record = record;
    *out_tokenizer = tokenizer;

    return MILA_OK;
}

MILA_API mila_status mila_tokenizer_encode(
    mila_tokenizer* tokenizer, const char* text, int32_t** out_tokens, int64_t* out_count )
{
    mock_clear_error();

    if ( tokenizer == NULL || text == NULL || out_tokens == NULL || out_count == NULL )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "tokenizer, text, out_tokens and out_count must not be null" );
    }

    size_t length = strlen( text );

    /* At least one element, so an empty text still gives a pointer the caller frees. */
    int32_t* tokens = malloc( ( length == 0 ? 1 : length ) * sizeof( int32_t ) );

    if ( tokens == NULL )
    {
        return mock_fail( MILA_ERROR_RUNTIME, "out of memory" );
    }

    for ( size_t index = 0; index < length; ++index )
    {
        tokens[index] = (int32_t)(unsigned char)text[index] + MOCK_FIRST_BYTE_TOKEN;
    }

    *out_tokens = tokens;
    *out_count = (int64_t)length;

    return MILA_OK;
}

MILA_API mila_status mila_tokenizer_decode(
    mila_tokenizer* tokenizer, const int32_t* tokens, int64_t count, char** out_text )
{
    mock_clear_error();

    if ( tokenizer == NULL || out_text == NULL || count < 0 || ( tokens == NULL && count > 0 ) )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT,
            "tokenizer and out_text must not be null, and tokens must hold count ids" );
    }

    mock_buffer text = { 0 };

    for ( int64_t index = 0; index < count; ++index )
    {
        int32_t token = tokens[index];

        if ( token < 0 || token >= MOCK_VOCABULARY_SIZE )
        {
            mock_buffer_free( &text );

            return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "token id %d is outside the vocabulary", (int)token );
        }

        if ( token >= MOCK_FIRST_BYTE_TOKEN )
        {
            mock_buffer_append_char( &text, (char)( token - MOCK_FIRST_BYTE_TOKEN ) );
        }
    }

    return mock_release_out( &text, out_text );
}

MILA_API mila_status mila_tokenizer_eos_token_id( mila_tokenizer* tokenizer, int32_t* out_id, int32_t* out_present )
{
    mock_clear_error();

    if ( tokenizer == NULL || out_id == NULL || out_present == NULL )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "tokenizer, out_id and out_present must not be null" );
    }

    *out_id = MOCK_TOKEN_EOS;
    *out_present = 1;

    return MILA_OK;
}

MILA_API void mila_tokenizer_free( mila_tokenizer* tokenizer )
{
    free( tokenizer );
}
