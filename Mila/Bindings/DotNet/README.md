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
- CMake and a C compiler, to build the mock: CMake 3.20 or later with GCC or Clang on Linux; on Windows,
  Visual Studio's C++ tools and a CMake that knows that Visual Studio (4.2 or later for Visual Studio 2026).

## Build and test

From this directory:

```sh
dotnet test
```

The first build compiles the mock into `Mock/build` with CMake, and every build copies it next to the
programs that use the library. To use a `MilaNative` built elsewhere, pass
`-p:MilaNativeDirectory=<directory>`.

## Run the sample

`Samples/Chat` is a console chat: give it an installed model's name and talk to it. Ctrl-C stops a reply,
and an empty line quits.

```sh
dotnet run --project Samples/Chat -- mock-gemma-it
```

With no name it lists the models you can choose; `dotnet run` would otherwise take `mock-gemma-it` from
the project's launch profile, so pass `--no-launch-profile` for that.

## Visual Studio

Open `Mila.slnx` in Visual Studio 2026. It holds the library, the tests and the sample; Test Explorer runs
the tests, and starting the Chat project runs the sample on `mock-gemma-it`.

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
| `Mila.slnx` | The Visual Studio solution |
| `Directory.Packages.props`, `*/packages.lock.json` | The tests' and the sample's packages, pinned; the library has none |
| `global.json` | The SDK, and `dotnet test` on Microsoft Testing Platform |
