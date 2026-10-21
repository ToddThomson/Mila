# Mila for .NET

A C# library for running Mila's models from .NET: list the installed models, tokenize, render a
conversation in a model's chat template, and stream a reply token by token.

**Experimental.** The API may change in any commit until v0.22.0 or later. Using it reports the compiler
diagnostic `MILA0001`; suppress it in your project to acknowledge that:

```xml
<NoWarn>$(NoWarn);MILA0001</NoWarn>
```

The library calls `MilaNative`, a native library with a C interface (`Mila/Bindings/Native/mila.h`).
`MilaNative` over the real Mila library is not built yet; until it is, everything here runs against a
mock of it that answers with a fixed reply. The design is `Mila/Specifications/DotNetBinding.md`.

## Requirements

- The .NET 10 SDK.
- To build the mock: CMake 3.20 or later and GCC or Clang.

## Build and test

From this directory:

```sh
cmake -S Mock -B Mock/build
cmake --build Mock/build
dotnet test
```

The tests load the mock from `Mock/build`. To use a mock built elsewhere, pass
`-p:MilaNativeMockDirectory=<directory>` to `dotnet test`.

## Run the sample

`Samples/Chat` is a console chat: give it an installed model's name and talk to it. Ctrl-C stops a reply,
and an empty line quits. The sample finds `MilaNative` the way any .NET program does, so point the
system's library search at the mock first:

```sh
LD_LIBRARY_PATH=$PWD/Mock/build dotnet run --project Samples/Chat -- mock-gemma-it
```

Run it with no name to list the models you can choose.

## Using it

```csharp
using System.Text;
using Mila;

MilaRuntime.Initialize(LogLevel.Warning);

using Tokenizer tokenizer = Tokenizer.FromStore("gemma-4-12b-it-fp4");
using ChatProtocol protocol = ChatProtocol.ForInstalledModel("gemma-4-12b-it-fp4");
using Model model = Model.FromStore("gemma-4-12b-it-fp4");

ChatTurn[] history = [new ChatTurn("user", "Name three rivers.")];
FormattedPrompt prompt = protocol.FormatPrompt(history);
int[] tokens = tokenizer.Encode(prompt.Prompt);
List<int> held = [];

FinishReason reason = model.Generate(tokens, Show);

void Show(int token)
{
    held.Add(token);

    try
    {
        int[] ids = held.ToArray();
        string text = tokenizer.Decode(ids);
        Console.Write(text);
        held.Clear();
    }
    catch (DecoderFallbackException)
    {
        // Part of a character: wait for the token that completes it.
    }
}
```

- `Generate` calls your callback on the calling thread, once per token, and returns why the reply ended.
  An exception thrown from the callback stops the reply and is rethrown from `Generate`.
- A `CancellationToken` stops a reply from any thread; `Generate` then returns `FinishReason.Cancelled`.
- `GenerateOptions` sets the length and sampling; a setting left null is the model publisher's.
- `Tokenizer.Decode` throws `DecoderFallbackException` when the ids end part-way through a character,
  so the callback above holds ids until they decode.
- A failure in the library is a `MilaException` whose `Status` says why, such as `NotFound` for a model
  that is not installed or `Refused` for one that does not fit the device; an invalid argument is an
  `ArgumentException`.

## Layout

| Path | What it is |
|---|---|
| `Mila/` | The library |
| `Mila.Tests/` | Its tests, run against the mock |
| `Samples/Chat/` | The console chat |
| `Mock/` | The mock `MilaNative`, in C |
| `Directory.Packages.props`, `*/packages.lock.json` | Every package, pinned |
| `global.json` | The SDK, and `dotnet test` on Microsoft Testing Platform |
