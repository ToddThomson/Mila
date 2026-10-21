# The C ABI and the .NET Binding

A C ABI over Mila's binding surface, and a C# library that projects it through P/Invoke: the second
language projection after Python, and the first consumer of the C ABI that `Direction.md` 8 decision 3
expects at the `AI` boundary.

**Status:** draft, 2026-10-21. **Experimental until v0.22.0 or later** (Todd, 2026-10-21): the ABI and the
C# API may change in any commit, and both say so to their callers (sections 3.1 and 4.1). Phase 1a is
written to be built by a cloud session from this document alone. Decisions and open questions are in
section 8.

---

## 1. What this is

The Python binding reaches Mila through pybind11, which binds C++ directly. .NET cannot call C++ classes;
it calls C functions. So the .NET binding is two pieces:

- **`MilaNative`**, a shared library exporting a C ABI (`mila.h`) over the binding surface that the Python
  extension already uses.
- **`Mila`**, a .NET assembly that calls `MilaNative` through source-generated P/Invoke and presents it
  in .NET's idiom: `IDisposable` handles, exceptions, `CancellationToken`.

It follows the binding rules the Python projection follows (`Mila/Bindings/README.md`,
`PythonBinding.md`):

- **Consumer-blind, session depth.** Nothing in either piece knows about Chat, MIS, agents or a wire
  protocol. It projects models, tokenizers, the store and the chat protocol, never components.
- **It projects; it defines nothing** (`Direction.md` 4, rule 5). A capability the C# library has is one the
  binding surface has. The one place this document bends the rule is named where it happens (section
  3.4, decided in section 8).
- **An app adds no capability to the library** (CLAUDE.md), and a binding is not an app either: a gap
  between Python and .NET is a defect in one of them.

What it is for: a C# developer streams a reply from an installed model in about thirty lines, with the
loop visible, as `Samples/QuickStart/Python/chat.py` does for Python.

## 2. Architecture

```
  +--------------------------+      +-----------------------------+
  |  Python: mila (pybind11) |      |  .NET: Mila.dll (C#)         |
  |  Mila_py.cpp             |      |  LibraryImport, SafeHandle   |
  +------------+-------------+      +--------------+--------------+
               |                                   |  P/Invoke
               |                    +--------------+--------------+
               |                    |  MilaNative: mila.h, C ABI  |
               |                    |  (section 3)                |
               |                    +--------------+--------------+
               |  C++, std-only                    |  C++, std-only
  +------------+-----------------------------------+--------------+
  |  Binding surface: Mila.Bindings, Mila.Bindings.Runtime,        |
  |  Mila.Bindings.ChatProtocol (today in the MilaPy target)       |
  +------------------------------+---------------------------------+
                                 |
                               Mila
```

- **Both languages stand on one surface.** The std-only modules in `Mila/Bindings/` (`Mila_py.Wrappers.ixx`
  and its partitions, `Mila_py.Runtime.ixx`, `Mila_py.ChatProtocol.ixx`) already hold every decision a
  projection needs: the deployment request a load sends, the refusal's words (`describeRefusal`), the
  sampling a knob not given falls back to. `MilaNative` calls that surface and adds only the C crossing.
  So Python and .NET cannot drift apart on behaviour, only on names.
- **The surface becomes its own target** (phase 1b). It is compiled into `MilaPy` today. It moves to a
  static library, `MilaBindingSurface`, that `MilaPy` and `MilaNative` both link. Its files keep their
  `Mila_py.` names until a rename commit of their own (`Specifications/README.md` rule 4).
- **At v0.22.0 the surface is re-pointed at `Mila::AI`** (phase 2). The ABI's model handle is already the
  shape `Mila::AI`'s is (section 3.4), so the C# library does not change when it is.

## 3. The C ABI (`Mila/Bindings/Native/mila.h`)

### 3.1 Conventions

- **C17, no C++ in the header.** It compiles as C. The mock (section 5) is written in C, which proves it.
- **Exported functions are `mila_` snake_case**, with the export macro `MILA_API` (`__declspec(dllexport)`
  building the library on Windows, `__declspec(dllimport)` using it, `visibility("default")` elsewhere) and
  the platform's default calling convention. No function is variadic.
- **Experimental and versioned.** `int32_t mila_abi_version(void)` returns `MILA_ABI_VERSION`, `0` until
  the ABI is declared stable. The C# library checks it on first use and refuses a mismatch with a message
  naming both numbers. Any change to a signature, struct or constant bumps it.
- **Every fallible call returns `mila_status`** (`typedef int32_t mila_status;`) and gives its results
  through out-parameters, which are written only on `MILA_OK`.

  | Constant | Value | Raised for |
  |---|---|---|
  | `MILA_OK` | 0 | |
  | `MILA_ERROR_INVALID_ARGUMENT` | 1 | a null handle or pointer, a bad value (`std::invalid_argument`) |
  | `MILA_ERROR_NOT_FOUND` | 2 | no installed model of that name |
  | `MILA_ERROR_REFUSED` | 3 | a deployment that does not fit the device, or a model this build cannot load |
  | `MILA_ERROR_UNSUPPORTED` | 4 | an entry point this build of `MilaNative` was compiled without |
  | `MILA_ERROR_RUNTIME` | 5 | anything else the library threw |
  | `MILA_ERROR_ABI` | 6 | an options struct whose `struct_size` the library does not know |

- **No exception crosses the ABI.** Every exported body is one `try` whose `catch` blocks map to a status
  and store the exception's `what()`. The map is by exception type, and today the surface throws
  `std::runtime_error` for a missing model and a refused deployment alike (`Mila_py.Wrappers.cpp`), so
  phase 1b gives those two their own types (section 6). Until then both are `MILA_ERROR_RUNTIME`; the
  message carries the library's words either way.
- **The error message is per thread.** `const char* mila_last_error_message(void)` returns the message of
  the last failed call on the calling thread, or `""` after a success. It is owned by the library and valid
  until the next `mila_` call on that thread, so a caller copies it at once.
- **Strings in are UTF-8, null-terminated, borrowed** for the duration of the call. **Strings out** are
  `char*` the library allocates, freed by `mila_string_free`. **Token arrays out** are `int32_t*` freed by
  `mila_tokens_free`. A caller never frees library memory any other way.
- **Counts are `int64_t`.** Token ids are `int32_t`: values, not extents (CLAUDE.md, `dim_t`).
- **Handles are opaque** (`typedef struct mila_model mila_model;`), created by a `_from_` or `_open`
  function, released by the matching `_free`, which accepts null. A handle is used by one thread at a time;
  `mila_stop_source_request_stop` is the one call that may come from another thread while a call is running.
- **Records cross as JSON text** where the surface already speaks JSON (the chat protocol) or a record has
  many fields (a stored model). Field names are the surface's own (`StoredModelInfo`), snake_case. Scalars
  and token streams never go through JSON.

### 3.2 Runtime

| Function | Projects |
|---|---|
| `int32_t mila_abi_version(void)` | |
| `mila_status mila_initialize(const char* log_level)` | `initialize`: `trace`, `info`, `warning`, `error` |
| `mila_status mila_version(char** out_version)` | `version()`, as `Version.txt` writes it |
| `mila_status mila_cuda_devices_json(char** out_json)` | `cudaDevices()`: a JSON array of `CudaDeviceInfo` |
| `const char* mila_last_error_message(void)` | |
| `void mila_string_free(char*)`, `void mila_tokens_free(int32_t*)` | |

### 3.3 Store, tokenizer and chat protocol

| Function | Projects |
|---|---|
| `mila_status mila_store_list_json(char** out_json)` | `ModelStoreHandle::list()` on the default store: a JSON array of `StoredModelInfo` |
| `mila_status mila_tokenizer_from_store(const char* name, mila_tokenizer** out)` | `Tokenizer::fromStore` |
| `mila_status mila_tokenizer_encode(mila_tokenizer*, const char* text, int32_t** out_tokens, int64_t* out_count)` | `encode` |
| `mila_status mila_tokenizer_decode(mila_tokenizer*, const int32_t* tokens, int64_t count, char** out_text)` | `decode` |
| `mila_status mila_tokenizer_eos_token_id(mila_tokenizer*, int32_t* out_id, int32_t* out_present)` | `eosTokenId`; `out_present` is 0 when the vocabulary has none |
| `void mila_tokenizer_free(mila_tokenizer*)` | |
| `mila_status mila_chat_protocol_for_installed_model(const char* name, mila_chat_protocol** out)` | `ChatProtocolHandle::forInstalledModel` |
| `mila_status mila_chat_protocol_format_prompt(mila_chat_protocol*, const char* history_json, const char* tools_json, char** out_prompt, int32_t* out_begins_in_reasoning)` | `formatPrompt` with the template's default variables; `tools_json` may be null |
| `mila_status mila_chat_protocol_parse_reply(mila_chat_protocol*, const char* text, int32_t begins_in_reasoning, char** out_reply_json)` | `parseReply` |
| `void mila_chat_protocol_free(mila_chat_protocol*)` | |

`history_json` and the reply JSON are the shapes `Mila_py.ChatProtocol.ixx` documents, unchanged.

### 3.4 The model

```c
typedef struct mila_model mila_model;
typedef struct mila_stop_source mila_stop_source;

typedef int32_t mila_finish_reason;   /* MILA_FINISH_STOP 0, _LENGTH 1, _CONTEXT_LIMIT 2, _CANCELLED 3 */

typedef struct mila_load_options {
    uint32_t struct_size;             /* sizeof( mila_load_options ) */
    int64_t context_length;           /* 0 = automatic: the planner's longest that fits */
    int32_t device_index;
    const char* kv_cache;             /* "bf16" or "fp8"; null = "bf16" */
} mila_load_options;

typedef struct mila_generate_options {
    uint32_t struct_size;             /* sizeof( mila_generate_options ) */
    uint32_t set;                     /* MILA_SET_TEMPERATURE 1, _TOP_K 2, _TOP_P 4, _PRESENCE_PENALTY 8,
                                         _THINKING 16, _SEED 32: which fields below are given */
    int64_t max_new_tokens;
    float temperature;
    int32_t top_k;
    float top_p;
    float presence_penalty;
    int32_t thinking;
    uint64_t seed;
} mila_generate_options;

typedef void ( *mila_token_callback )( int32_t token, void* user_data );
```

| Function | Projects |
|---|---|
| `mila_status mila_model_from_store(const char* name, const mila_load_options* options, mila_model** out)` | `GemmaSession`, `LlamaSession` or `QwenSession` `::fromStore`, by the record's `family`; `options` may be null |
| `mila_status mila_model_context_length(mila_model*, int64_t* out)` | `contextLength()` |
| `mila_status mila_model_family(mila_model*, char** out_family)` | the record's `family` |
| `mila_status mila_model_generate(mila_model*, const int32_t* prompt, int64_t prompt_count, const mila_generate_options* options, mila_token_callback on_token, void* user_data, mila_stop_source* stop, mila_finish_reason* out_reason)` | `generate` |
| `void mila_model_free(mila_model*)` | |
| `mila_status mila_stop_source_create(mila_stop_source** out)`, `void mila_stop_source_request_stop(mila_stop_source*)`, `void mila_stop_source_free(mila_stop_source*)` | `StopController`, over `std::stop_source` |

- **A field the `set` mask does not name is the model's publisher's setting** for the thinking the prompt
  was rendered with, as the surface already decides (`ChatConfiguration.md` 5). The mask is what makes
  "not given" crossable without `std::optional`.
- **`generate` is synchronous.** It calls `on_token` on the calling thread, once per token, before it
  returns. The callback may call `mila_stop_source_request_stop` and any function on another handle --
  decoding the token with a tokenizer is what a streaming caller does there, as Python's `on_token` does --
  but nothing on the model that is generating. A failure inside the callback leaves `generate`'s own
  status and message unaffected, since the callback's calls finish before `generate` reports.
- **`max_new_tokens` is always given and greater than zero**; it is a bound, not a sampling knob, so the
  mask does not cover it. A null `options` is `MILA_ERROR_INVALID_ARGUMENT`.
- **One model handle, not three -- the bend of rule 5.** The surface has a session class per family and
  Python projects each. The ABI opens one `mila_model` by store name and picks the session from the record's
  `family`, which is the selection `ChatProtocolHandle::forInstalledModel` already makes for the template.
  It adds no capability: the three sessions have one `generate` signature. It is the shape `Mila::AI`
  takes at v0.22.0 (`Direction.md` 4, rules 1 and 7), so the C# library is written once. Decided in
  section 8, item 1.

Not in phase 1: `load(path, ...)` from loose weights, `getConfig`, deployment queries, store install, pull
and remove, Gemma's and Qwen's family-specific protocol functions. Each is a later row in these tables,
written the same way.

## 4. The .NET library (`Mila/Bindings/DotNet/`)

### 4.1 Shape

- **`net10.0`**, the LTS release, supported to November 2028. Assembly and namespace `Mila`.
  `AllowUnsafeBlocks` and `Nullable` on; warnings are errors.
- **Marked experimental:** `[assembly: Experimental("MILA0001")]`, so a consumer's build reports the
  diagnostic until they suppress it -- the experimental status said once, by the compiler, rather than in
  prose on every type.
- **P/Invoke through `LibraryImport`** (source-generated, .NET 7 and later) with
  `StringMarshalling.Utf8`, all of it in one `internal static partial class NativeMethods`. No
  `DllImport`, no runtime IL stubs. Library name `"MilaNative"`: the runtime adds `.dll`, `.so` or
  `.dylib`, and `lib` on Unix.
- **Each handle is a `SafeHandle` subclass** whose `ReleaseHandle` calls the matching `_free`, and each
  public type that owns one is `IDisposable`.
- **A failed status becomes an exception** read with `mila_last_error_message` before any other call:
  `MILA_ERROR_INVALID_ARGUMENT` is `ArgumentException`; every other status is `MilaException`, whose
  `Status` property is the `MilaStatus` enum.
- **Records are deserialized with `System.Text.Json`** through a source-generated `JsonSerializerContext`,
  so the library is trim-safe and has no reflection-based serialization. No third-party package ships.
- **The token callback is an `[UnmanagedCallersOnly]` static method**, passed as `delegate* unmanaged`,
  with the managed `Action<int>` in a `GCHandle` as `user_data`. An exception from the caller's callback
  is caught there, never unwound through native frames: the method requests stop, stores the exception,
  and `Generate` rethrows it after the native call returns.
- **Cancellation is a `CancellationToken`**, registered to `mila_stop_source_request_stop` for the length
  of the call. A cancelled reply returns `FinishReason.Cancelled`; it does not throw.

### 4.2 API

| Type | Members |
|---|---|
| `MilaRuntime` (static) | `Initialize(LogLevel)`, `Version`, `CudaDevices` -> `IReadOnlyList<CudaDevice>` |
| `ModelStore` (static) | `List()` -> `IReadOnlyList<StoredModel>` |
| `Tokenizer` | `FromStore(name)`, `Encode(string)` -> `int[]`, `Decode(ReadOnlySpan<int>)`, `EosTokenId` -> `int?` |
| `ChatProtocol` | `ForInstalledModel(name)`, `FormatPrompt(IReadOnlyList<ChatTurn>, string? toolsJson)` -> `FormattedPrompt(Prompt, BeginsInReasoning)`, `ParseReply(text, beginsInReasoning)` -> `ParsedReply` |
| `Model` | `FromStore(name, LoadOptions?)`, `ContextLength`, `Family`, `Generate(ReadOnlySpan<int> prompt, Action<int> onToken, GenerateOptions?, CancellationToken)` -> `FinishReason` |
| Records | `CudaDevice`, `StoredModel`, `ChatTurn`, `ToolCall`, `FormattedPrompt`, `ParsedReply`, `LoadOptions`, `GenerateOptions` (each knob `T?`), enums `LogLevel`, `FinishReason`, `MilaStatus` |

`Model.Generate` throws `InvalidOperationException` when called while another `Generate` on the same model
is running, including from its own `onToken`. `GenerateOptions.MaxNewTokens` is a `long` defaulting to 64,
Python's `max_new_tokens` default; the sampling knobs are each `T?`. Every string out of the ABI is decoded
as strict UTF-8, so `Tokenizer.Decode` of ids that end part-way through a character throws
`DecoderFallbackException`, as Python's `decode` raises `UnicodeDecodeError`: a streaming caller holds the
ids and retries. Names are the Python projection's in .NET casing, without abbreviation (CLAUDE.md): a reader
who knows one knows the other.

### 4.3 Layout

```
Mila/Bindings/DotNet/
  Mila.slnx                       solution: the library, its tests, the sample
  Directory.Build.props           net10.0, Nullable, TreatWarningsAsErrors, the version from Version.txt
  Directory.Packages.props        central package management: every test package pinned exactly
  global.json                     the SDK band, and `dotnet test` in Microsoft Testing Platform mode,
                                  which xUnit v3 4.x requires on the .NET 10 SDK
  */packages.lock.json            every transitive package pinned; CI restores in locked mode
  Mila/                           the library (Mila.csproj)
  Mila.Tests/                     xUnit, over the mock (section 5)
  Samples/Chat/                   a console chat: store name in, streamed reply out
  Mock/                           the mock native library, C, with its own CMakeLists.txt
  README.md
```

## 5. The mock native library (`Mila/Bindings/DotNet/Mock/`)

A C17 implementation of all of `mila.h` with no Mila behind it, built by its own small `CMakeLists.txt`
into `MilaNative` -- the same name, so the C# library loads it unchanged. It is what phase 1a is built and
tested against, as Chat's desktop window was built against its mock session (`ChatDesktop.md` 5).

- **A fixed store** of two models, `mock-gemma-it` (family `gemma`) and `mock-llama-it` (family `llama`),
  and one name that is installed but incomplete, `mock-incomplete`, which refuses to load.
- **A byte tokenizer:** a token per UTF-8 byte, id = byte + 3; ids 0 to 2 are bos, eos and pad. Decoding
  inverts it, so an encode-decode round trip is exact.
- **A chat protocol** that renders turns as `<role>content</role>` lines and parses a reply's
  `<think>...</think>` prefix as reasoning, enough to test both JSON crossings.
- **`generate`** emits the bytes of a fixed reply, one callback per token, then reports `STOP`; it reports
  `LENGTH` when `max_new_tokens` is reached first, and `CANCELLED` when the stop source is triggered,
  checked between tokens. A prompt whose first token is the byte `!` fails with `MILA_ERROR_RUNTIME` and
  a message, so the error path is reachable.
- **Unknown names** are `MILA_ERROR_NOT_FOUND`, a null handle is `MILA_ERROR_INVALID_ARGUMENT`, a
  `struct_size` it does not know is `MILA_ERROR_ABI`; each sets the thread's message.

The tests point the library at the mock with `NativeLibrary.SetDllImportResolver` on the `Mila` assembly,
from the test project, which is given the mock's build directory at build time. The shipped library has
no resolver and reads no environment variable; the sample finds the mock as it would find `MilaNative`,
through the platform's library search (`LD_LIBRARY_PATH` on Linux, `PATH` on Windows).

## 6. `MilaNative` (`Mila/Bindings/Native/`)

- **`mila.h`**, the contract, and the C++ implementation of it over the binding surface, one `.cpp` per
  table in section 3 (`Mila.Native.Runtime.cpp`, `.Store.cpp`, `.Model.cpp` and so on).
- **A CMake target `MilaNative`**, a shared library, behind `MILA_ENABLE_DOTNET_BINDINGS`, default OFF
  while the binding is experimental, and requiring CUDA as `MilaPy` does.
- **Two exception types in the surface,** `ModelNotInstalledError` and `DeploymentRefusedBindingError`, each
  derived from `std::runtime_error` and each in its own module file, thrown where `Mila_py.Wrappers.cpp`
  throws `std::runtime_error` for those two cases today. pybind11 translates a `std::runtime_error`
  subclass as `RuntimeError`, so Python callers see no change; `MilaNative` maps them to
  `MILA_ERROR_NOT_FOUND` and `MILA_ERROR_REFUSED`.
- **Its own CUDA libraries.** `mila/__init__.py` loads the pinned CUDA libraries before the Python
  extension because Windows does not search `PATH` for an extension's dependencies (`PythonBinding.md`,
  Tier 1). Whether `MilaNative`'s dependencies resolve when .NET loads it is measured in phase 1b, not
  assumed; the C# library's answer is written into this section then.

## 7. Phases

1. **Structure, with the minimal API of section 3** (Todd, 2026-10-21), in two steps.
   - **1a, the contract and the C# library against the mock** (cloud session). `mila.h`, the mock, the C#
     library, its tests, the sample, the README and `.github/workflows/dotnet.yml` (on changes under
     `Mila/Bindings/DotNet/`, `Mila/Bindings/Native/mila.h` and itself: build the mock on Linux, then
     `dotnet test`). It touches only those paths and, for a contract change, this document, changed
     first. Done when the mock builds as C17 with warnings as errors under GCC or Clang; `dotnet build`
     is warning-free; `dotnet test` passes, covering every function in section 3 on its success path and
     every status in 3.1 that the mock can raise; a callback that throws ends `Generate` with that
     exception and no crash; cancelling from another thread returns `Cancelled`; and the sample runs
     against the mock.
   - **1b, `MilaNative` over the real library** (local). The binding surface split into
     `MilaBindingSurface` with `MilaPy` unchanged in behaviour (its Python tests pass), `MilaNative`
     built in `x64-profile`, and section 6's dependency question answered. Done when the C# sample
     streams a reply from an installed Gemma 4 12B on the GPU, unchanged from 1a, and from one Llama and
     one Qwen model.
2. **Re-pointed at `Mila::AI`** (v0.22.0). The surface's sessions give way to the handle; the ABI keeps
   `mila_model`, and the C# library changes only where the handle gains what the sessions lacked.
3. **A NuGet package** (open decision 3).

## 8. Open decisions

1. **One model handle in the ABI -- decided 2026-10-21 (Todd), for now.** One `mila_model`, for the
   reason section 3.4 gives. Considered: one per family, as Python has, which keeps rule 5 without
   exception and is retired with the sessions at v0.22.0, along with every C# caller written against it.
2. **Test framework -- decided 2026-10-21 (Todd), for now.** xUnit v3, pinned exactly in
   `Directory.Packages.props`. It is a test dependency only, never shipped, so it adds no `NOTICE.md` row.
3. **Distribution.** A NuGet package carrying `MilaNative` under `runtimes/win-x64/native` and
   `runtimes/linux-x64/native`; whether the CUDA libraries go in it, beside it, or are the user's install,
   which is the question the Python wheel answered with NVIDIA's own wheels and which NuGet has no
   equivalent for; the package name.
4. **Where the C# sample lives once the binding is stable:** `Samples/QuickStart/CSharp`, beside Python's,
   is a published surface the website links, so it waits for the experimental status to end.
