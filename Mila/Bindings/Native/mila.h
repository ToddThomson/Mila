/*
 * mila.h -- the C ABI over Mila's binding surface (Specifications/DotNetBinding.md section 3).
 *
 * Experimental: any signature, struct or constant may change in any commit, and every change bumps
 * MILA_ABI_VERSION. C17; it compiles as C and as C++.
 */

#ifndef MILA_H
#define MILA_H

#include <stdint.h>

#if defined( _WIN32 )
#if defined( MILA_NATIVE_BUILDING )
#define MILA_API __declspec( dllexport )
#else
#define MILA_API __declspec( dllimport )
#endif
#else
#define MILA_API __attribute__( ( visibility( "default" ) ) )
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* 0 until the ABI is declared stable. A caller refuses a library whose mila_abi_version() differs. */
#define MILA_ABI_VERSION 0

/* ---- Status -------------------------------------------------------------------------------------- */

/* Every fallible call returns one; its out-parameters are written only on MILA_OK. */
typedef int32_t mila_status;

#define MILA_OK 0
#define MILA_ERROR_INVALID_ARGUMENT 1
#define MILA_ERROR_NOT_FOUND 2
#define MILA_ERROR_REFUSED 3
#define MILA_ERROR_UNSUPPORTED 4
#define MILA_ERROR_RUNTIME 5
#define MILA_ERROR_ABI 6

/* ---- Handles ------------------------------------------------------------------------------------- */

/*
 * Opaque. Each is released by its own _free, which accepts null. A handle is used by one thread at a
 * time; mila_stop_source_request_stop is the one call that may come from another thread mid-call.
 */
typedef struct mila_tokenizer mila_tokenizer;
typedef struct mila_chat_protocol mila_chat_protocol;
typedef struct mila_model mila_model;
typedef struct mila_stop_source mila_stop_source;

/* ---- Runtime (section 3.2) ----------------------------------------------------------------------- */

MILA_API int32_t mila_abi_version( void );

/* log_level is one of "trace", "info", "warning" and "error". */
MILA_API mila_status mila_initialize( const char* log_level );

/* The library's version as Version.txt writes it. Free with mila_string_free. */
MILA_API mila_status mila_version( char** out_version );

/*
 * A JSON array of {index, name, compute_capability_major, compute_capability_minor, total_memory_bytes,
 * pci_domain, pci_bus, pci_device}, in CUDA ordinal order; empty before mila_initialize.
 */
MILA_API mila_status mila_cuda_devices_json( char** out_json );

/*
 * The message of the last failed call on the calling thread, or "" after a success. Owned by the
 * library and valid until the next mila_ call on this thread: copy it at once.
 */
MILA_API const char* mila_last_error_message( void );

MILA_API void mila_string_free( char* text );
MILA_API void mila_tokens_free( int32_t* tokens );

/* ---- Store, tokenizer and chat protocol (section 3.3) -------------------------------------------- */

/*
 * The default store's installed models: a JSON array of {name, architecture, family, variant,
 * weight_quantization, instruct, thinking, tools, agentic, mtp, modalities, base_model, license, hub,
 * owner, repository, revision, installed_at, weights_path, tokenizer_path, complete, bytes_on_disk}.
 */
MILA_API mila_status mila_store_list_json( char** out_json );

MILA_API mila_status mila_tokenizer_from_store( const char* name, mila_tokenizer** out_tokenizer );

/* out_tokens is freed with mila_tokens_free. */
MILA_API mila_status mila_tokenizer_encode(
    mila_tokenizer* tokenizer, const char* text, int32_t** out_tokens, int64_t* out_count );

/* tokens may be null when count is 0. The text is the vocabulary's bytes, which need not end on a
 * character boundary when the ids do not. */
MILA_API mila_status mila_tokenizer_decode(
    mila_tokenizer* tokenizer, const int32_t* tokens, int64_t count, char** out_text );

/* out_present is 0, and out_id unspecified, when the vocabulary has no end-of-sequence token. */
MILA_API mila_status mila_tokenizer_eos_token_id( mila_tokenizer* tokenizer, int32_t* out_id, int32_t* out_present );

MILA_API void mila_tokenizer_free( mila_tokenizer* tokenizer );

MILA_API mila_status mila_chat_protocol_for_installed_model( const char* name, mila_chat_protocol** out_protocol );

/*
 * Render a conversation with the template's default variables. history_json is a JSON array of turns
 * {role, content, tool_calls: [{name, arguments}], tool_name, reasoning}; tools_json is a JSON array of
 * tool schemas, or null for none.
 */
MILA_API mila_status mila_chat_protocol_format_prompt(
    mila_chat_protocol* protocol,
    const char* history_json,
    const char* tools_json,
    char** out_prompt,
    int32_t* out_begins_in_reasoning );

/* The reply as JSON: {reasoning, content, tool_calls: [{name, arguments}], malformed_tool_calls}. */
MILA_API mila_status mila_chat_protocol_parse_reply(
    mila_chat_protocol* protocol, const char* text, int32_t begins_in_reasoning, char** out_reply_json );

MILA_API void mila_chat_protocol_free( mila_chat_protocol* protocol );

/* ---- The model (section 3.4) --------------------------------------------------------------------- */

typedef int32_t mila_finish_reason;

#define MILA_FINISH_STOP 0
#define MILA_FINISH_LENGTH 1
#define MILA_FINISH_CONTEXT_LIMIT 2
#define MILA_FINISH_CANCELLED 3

/* struct_size is sizeof( mila_load_options ); a size the library does not know is MILA_ERROR_ABI. */
typedef struct mila_load_options
{
    uint32_t struct_size;
    int64_t context_length; /* 0 = automatic: the planner's longest that fits */
    int32_t device_index;
    const char* kv_cache; /* "bf16" or "fp8"; null = "bf16" */
} mila_load_options;

/* Which of mila_generate_options' knobs are given; a knob not given is the publisher's setting. */
#define MILA_SET_TEMPERATURE 1u
#define MILA_SET_TOP_K 2u
#define MILA_SET_TOP_P 4u
#define MILA_SET_PRESENCE_PENALTY 8u
#define MILA_SET_THINKING 16u
#define MILA_SET_SEED 32u

typedef struct mila_generate_options
{
    uint32_t struct_size; /* sizeof( mila_generate_options ) */
    uint32_t set; /* MILA_SET_ bits */
    int64_t max_new_tokens; /* always given, greater than zero */
    float temperature;
    int32_t top_k;
    float top_p;
    float presence_penalty;
    int32_t thinking;
    uint64_t seed;
} mila_generate_options;

/*
 * Called on the generating thread, once per token, before mila_model_generate returns. It may call
 * mila_stop_source_request_stop and functions on other handles, never on the generating model.
 */
typedef void ( *mila_token_callback )( int32_t token, void* user_data );

/* Opens the session the record's family names. options may be null for every default. */
MILA_API mila_status mila_model_from_store( const char* name, const mila_load_options* options, mila_model** out_model );

MILA_API mila_status mila_model_context_length( mila_model* model, int64_t* out_context_length );

/* "gemma", "llama" or "qwen". Free with mila_string_free. */
MILA_API mila_status mila_model_family( mila_model* model, char** out_family );

/* options must not be null; stop may be. */
MILA_API mila_status mila_model_generate(
    mila_model* model,
    const int32_t* prompt,
    int64_t prompt_count,
    const mila_generate_options* options,
    mila_token_callback on_token,
    void* user_data,
    mila_stop_source* stop,
    mila_finish_reason* out_reason );

MILA_API void mila_model_free( mila_model* model );

MILA_API mila_status mila_stop_source_create( mila_stop_source** out_stop );

/* Safe from any thread, and from inside a token callback. */
MILA_API void mila_stop_source_request_stop( mila_stop_source* stop );

MILA_API void mila_stop_source_free( mila_stop_source* stop );

#ifdef __cplusplus
}
#endif

#endif
