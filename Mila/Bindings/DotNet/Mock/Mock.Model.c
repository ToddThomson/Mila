/*
 * The mock's model (section 5): generate streams the bytes of a fixed reply, checks the stop source
 * between tokens, and fails on a prompt that begins with '!'.
 */

#include "Mock.Internal.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

enum
{
    MOCK_FIRST_BYTE_TOKEN = 3,
    MOCK_DEFAULT_CONTEXT_LENGTH = 4096,
};

static const char mock_reply[] = "Hello from the Mila mock.";

static const uint32_t known_set_bits = MILA_SET_TEMPERATURE | MILA_SET_TOP_K | MILA_SET_TOP_P |
    MILA_SET_PRESENCE_PENALTY | MILA_SET_THINKING | MILA_SET_SEED;

struct mila_model
{
    const mock_model_record* record;
    int64_t context_length;
};

struct mila_stop_source
{
    atomic_int requested;
};

static mila_status validate_load_options( const mila_load_options* options )
{
    if ( options->struct_size != sizeof( mila_load_options ) )
    {
        return mock_fail( MILA_ERROR_ABI, "mila_load_options.struct_size is %u; this library knows %u",
            (unsigned int)options->struct_size, (unsigned int)sizeof( mila_load_options ) );
    }

    if ( options->context_length < 0 )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "context_length must be 0 for automatic, or greater than zero" );
    }

    if ( options->device_index != 0 )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "no CUDA device has index %d", (int)options->device_index );
    }

    const char* kv_cache = options->kv_cache;

    if ( kv_cache != NULL && strcmp( kv_cache, "bf16" ) != 0 && strcmp( kv_cache, "fp8" ) != 0 )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "kv_cache must be 'bf16' or 'fp8', not '%s'", kv_cache );
    }

    return MILA_OK;
}

MILA_API mila_status mila_model_from_store( const char* name, const mila_load_options* options, mila_model** out_model )
{
    mock_clear_error();

    if ( name == NULL || out_model == NULL )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "name and out_model must not be null" );
    }

    if ( options != NULL )
    {
        mila_status status = validate_load_options( options );

        if ( status != MILA_OK )
        {
            return status;
        }
    }

    const mock_model_record* record = mock_store_find( name );

    if ( record == NULL )
    {
        return mock_fail( MILA_ERROR_NOT_FOUND, "no installed model is named '%s'", name );
    }

    if ( !record->complete )
    {
        return mock_fail( MILA_ERROR_REFUSED, "'%s' is missing a declared blob, so it cannot be loaded", name );
    }

    mila_model* model = malloc( sizeof( mila_model ) );

    if ( model == NULL )
    {
        return mock_fail( MILA_ERROR_RUNTIME, "out of memory" );
    }

    model->record = record;
    model->context_length = MOCK_DEFAULT_CONTEXT_LENGTH;

    if ( options != NULL && options->context_length > 0 )
    {
        model->context_length = options->context_length;
    }

    *out_model = model;

    return MILA_OK;
}

MILA_API mila_status mila_model_context_length( mila_model* model, int64_t* out_context_length )
{
    mock_clear_error();

    if ( model == NULL || out_context_length == NULL )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "model and out_context_length must not be null" );
    }

    *out_context_length = model->context_length;

    return MILA_OK;
}

MILA_API mila_status mila_model_family( mila_model* model, char** out_family )
{
    mock_clear_error();

    if ( model == NULL || out_family == NULL )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "model and out_family must not be null" );
    }

    return mock_copy_out( model->record->family, out_family );
}

static mila_status validate_generate_options( const mila_generate_options* options )
{
    if ( options->struct_size != sizeof( mila_generate_options ) )
    {
        return mock_fail( MILA_ERROR_ABI, "mila_generate_options.struct_size is %u; this library knows %u",
            (unsigned int)options->struct_size, (unsigned int)sizeof( mila_generate_options ) );
    }

    if ( ( options->set & ~known_set_bits ) != 0 )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "set names a knob this library does not know: 0x%x",
            (unsigned int)options->set );
    }

    if ( options->max_new_tokens <= 0 )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "max_new_tokens must be greater than zero" );
    }

    if ( ( options->set & MILA_SET_TEMPERATURE ) != 0 && !( options->temperature >= 0.0f ) )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "temperature must not be negative" );
    }

    if ( ( options->set & MILA_SET_TOP_K ) != 0 && options->top_k < 0 )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "top_k must not be negative" );
    }

    if ( ( options->set & MILA_SET_TOP_P ) != 0 && !( options->top_p > 0.0f && options->top_p <= 1.0f ) )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "top_p must be greater than 0 and at most 1" );
    }

    if ( ( options->set & MILA_SET_THINKING ) != 0 && options->thinking != 0 && options->thinking != 1 )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "thinking must be 0 or 1" );
    }

    return MILA_OK;
}

MILA_API mila_status mila_model_generate(
    mila_model* model,
    const int32_t* prompt,
    int64_t prompt_count,
    const mila_generate_options* options,
    mila_token_callback on_token,
    void* user_data,
    mila_stop_source* stop,
    mila_finish_reason* out_reason )
{
    mock_clear_error();

    if ( model == NULL || options == NULL || on_token == NULL || out_reason == NULL )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "model, options, on_token and out_reason must not be null" );
    }

    mila_status status = validate_generate_options( options );

    if ( status != MILA_OK )
    {
        return status;
    }

    if ( prompt == NULL || prompt_count <= 0 )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "the prompt must hold at least one token" );
    }

    if ( prompt[0] == '!' + MOCK_FIRST_BYTE_TOKEN )
    {
        return mock_fail( MILA_ERROR_RUNTIME, "the mock fails every prompt that begins with '!'" );
    }

    int64_t reply_length = (int64_t)( sizeof( mock_reply ) - 1 );
    mila_finish_reason reason = MILA_FINISH_STOP;

    for ( int64_t emitted = 0; emitted < reply_length; ++emitted )
    {
        if ( stop != NULL && atomic_load( &stop->requested ) != 0 )
        {
            reason = MILA_FINISH_CANCELLED;
            break;
        }

        if ( emitted == options->max_new_tokens )
        {
            reason = MILA_FINISH_LENGTH;
            break;
        }

        if ( prompt_count + emitted >= model->context_length )
        {
            reason = MILA_FINISH_CONTEXT_LIMIT;
            break;
        }

        int32_t token = (int32_t)(unsigned char)mock_reply[emitted] + MOCK_FIRST_BYTE_TOKEN;
        on_token( token, user_data );
    }

    *out_reason = reason;

    return MILA_OK;
}

MILA_API void mila_model_free( mila_model* model )
{
    free( model );
}

MILA_API mila_status mila_stop_source_create( mila_stop_source** out_stop )
{
    mock_clear_error();

    if ( out_stop == NULL )
    {
        return mock_fail( MILA_ERROR_INVALID_ARGUMENT, "out_stop must not be null" );
    }

    mila_stop_source* stop = malloc( sizeof( mila_stop_source ) );

    if ( stop == NULL )
    {
        return mock_fail( MILA_ERROR_RUNTIME, "out of memory" );
    }

    atomic_init( &stop->requested, 0 );
    *out_stop = stop;

    return MILA_OK;
}

MILA_API void mila_stop_source_request_stop( mila_stop_source* stop )
{
    if ( stop != NULL )
    {
        atomic_store( &stop->requested, 1 );
    }
}

MILA_API void mila_stop_source_free( mila_stop_source* stop )
{
    free( stop );
}
