using System.Runtime.InteropServices;

namespace Mila;

/// <summary>
/// Every entry point of mila.h, under its C name. Strings in are UTF-8; strings and token arrays out are
/// pointers the caller hands back to mila_string_free and mila_tokens_free (NativeResult does).
/// </summary>
internal static unsafe partial class NativeMethods
{
    internal const string Library = "MilaNative";

    // Runtime (section 3.2)

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int mila_abi_version();

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int mila_initialize(string logLevel);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int mila_version(out nint version);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int mila_cuda_devices_json(out nint json);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint mila_last_error_message();

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial void mila_string_free(nint text);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial void mila_tokens_free(nint tokens);

    // Store, tokenizer and chat protocol (section 3.3)

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int mila_store_list_json(out nint json);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int mila_tokenizer_from_store(string name, out TokenizerHandle tokenizer);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int mila_tokenizer_encode(TokenizerHandle tokenizer, string text, out nint tokens, out long count);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int mila_tokenizer_decode(TokenizerHandle tokenizer, int* tokens, long count, out nint text);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int mila_tokenizer_eos_token_id(TokenizerHandle tokenizer, out int id, out int present);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial void mila_tokenizer_free(nint tokenizer);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int mila_chat_protocol_for_installed_model(string name, out ChatProtocolHandle protocol);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int mila_chat_protocol_format_prompt(
        ChatProtocolHandle protocol,
        string historyJson,
        string? toolsJson,
        out nint prompt,
        out int beginsInReasoning);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int mila_chat_protocol_parse_reply(
        ChatProtocolHandle protocol,
        string text,
        int beginsInReasoning,
        out nint replyJson);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial void mila_chat_protocol_free(nint protocol);

    // The model (section 3.4)

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int mila_model_from_store(string name, in NativeLoadOptions options, out ModelHandle model);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int mila_model_context_length(ModelHandle model, out long contextLength);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int mila_model_family(ModelHandle model, out nint family);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int mila_model_generate(
        ModelHandle model,
        int* prompt,
        long promptCount,
        in NativeGenerateOptions options,
        delegate* unmanaged<int, nint, void> onToken,
        nint userData,
        StopSourceHandle stop,
        out int reason);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial void mila_model_free(nint model);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int mila_stop_source_create(out StopSourceHandle stop);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial void mila_stop_source_request_stop(StopSourceHandle stop);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial void mila_stop_source_free(nint stop);
}
