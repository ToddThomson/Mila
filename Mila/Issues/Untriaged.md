# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined after ninety days**, and anything user-reported is a pointer to its
GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

## Two of the agent's answers to a stop or a cut have no test that fails without them

`Mila/Src/AI/Agent.ixx` (`respond`, `dropIncompleteLastCharacter`) @ 6f077474

Found 2026-10-25 writing the limit tests: a round with several calls, stopped at its first, gives every call after
it a stopped result -- the tests' model makes one call per round, so nothing reaches them. And a result cut between
two bytes of one character drops the character's leading bytes; the test's result is ASCII, so no cut lands inside
a character.

## GPT-2 cannot be opened by name, because it has no deployment planner

`Mila/Src/Dnn/Models/Gpt/GptModel.ixx` (`load`); an absence -- no `planDeployment`, no `load( path, request )` @ 1f310d50

Found 2026-10-25 building the architecture list (`ModelHandle.md` 3.3): GPT-2 has a model and a tokenizer, but
its one `load` takes a config and no deployment request, so it cannot be an entry and `gpt2-small` cannot be
opened through the model handle. The spec's proof that adding an architecture is one list entry was GPT-2 (H5).

## Llama 3.1 8B FP4 answers as though a tool's result were made up

`Mila/Tests/AI/Agent.Cuda.cpp` (`askForTheWeather`); scratch-store package `Llama-3.1-8B-Instruct-fp4` @ efa24591

Measured 2026-10-25 through `Mila::AI::Agent`, the publisher's sampling, seeds 1 to 6, one question about the weather
in Paris and a tool returning "Sunny, 17 degrees Celsius." The rendered prompt is Meta's template: the result is an
`ipython` turn, quoted as the template quotes a string. With the tools in the first user message, the template's
default, it called once every time and then used the result once in six: "This is just an example response and may
not reflect the actual weather", "The function `get_weather` is not defined in the given code snippet". With them in
the system message, two of six; with that and Chat's when-to-call system prompt, one of six, three runs making no
call. Its manifest says `agentic: false`, so the agent's gate does not ask it to. 3.2 3B in the same runs: zero of
twenty-four. Data `D:\Claude\agent\llama_sweep.txt`, `llama_sweep2.txt`.

## Llama 3.2 3B offered a tool in the system message says it will call it and does not

`Mila/Apps/Chat/Prompts/tools-weather.json`; scratch-store package `Llama-3.2-3B-Instruct-fp4` @ e8df2aef

Measured 2026-10-25 at temperature 0, ten questions that need no tool and six about the weather now. With the
tools in the system message and the system prompt saying when to call, as Chat now renders them, 3B FP4 and BF16
alike answered all ten directly and called `get_weather` for none of the six: "I'll check the current weather
for Berlin." and stop, or a sentence and then the call, which the Llama grammar reads as text. With the tools in
the first user message, the template's default, it called for all sixteen. Meta's documented 3.2 prompt, with
pythonic calls (llama-models `text_prompt_format.md`), plus the same sentence: ten of ten, three of six. Data
`D:\Claude\llama-tools\` (`mila-3b-*`, `meta-3b-fp4.jsonl`, `chat-live-Llama-3.2-3B-Instruct-fp4.jsonl`).

## Chat runs a tool call whose arguments lack a field the tool requires

`Mila/Apps/Chat/Src/Chat.Session.ixx` (`dispatchTool`) @ e8df2aef

2026-10-25, Llama 3.1 8B FP4: the model called `get_weather` with `{}`, its schema requiring `location`; Chat
asked "Getting weather for {location}" with the placeholder unfilled and ran it. `ToolCalling.md` 3.2: the
arguments are checked against the schema before anything runs. Since Chat moved onto the agent, the call runs in
`Mila/Src/AI/Agent.ixx` (`answerCall`), which checks no arguments either.

## Llama 3.1 8B FP4 writes some tool calls in a shape its template does not define

`Mila/Src/Dnn/Models/LlaMa/Llama.ChatProtocol.ixx` (`templateReplyGrammar`); scratch-store package
`Llama-3.1-8B-Instruct-fp4` @ e8df2aef

Measured 2026-10-25 through the server at temperature 0, tools in the system message
(`chat_template_kwargs.tools_in_user_message: false`) and a system prompt saying when to call: of six weather
questions, three became calls, two were written `{"type": "function", "function": "get_weather", "parameters":
{...}}`, and one was a sentence and then the call. The server answered those three as text, as it does any call
that does not parse. The same questions on Q4_0 weights, Mila's and llama.cpp's alike, gave six well-formed calls,
and on FP8 weights quantized on load six as well. In Chat, which now renders that way, two of six (the answers
used the result) and two answered empty. Data `D:\Claude\llama-tools\mis-8b-fp4-guidance.jsonl`,
`mila-8b-fp8-guidance.jsonl`, `chat-live-Llama-3.1-8B-Instruct-fp4.jsonl`.

## Chat's tool definitions are documented as written into the system prompt by Chat

`Mila/Apps/Chat/Src/Chat.SystemPrompt.ixx` (`ToolDefinition`'s comment) @ 5e7de360

Since `+94` every family's template places the tools itself, and Chat writes none into the system turn; the
comment still says Chat serializes them, with Llama's pythonic wording. Fixing it brings the module under the
one-type rule, and it holds five types.

## .NET bindings for Mila

an absence -- no file in the repository or in `.internal/` names them @ a8f5360a

Todd, 2026-10-20, while choosing the toolkit for Chat's desktop window (`Mila/Specifications/ChatDesktop.md`
section 9): "We do have a future for .NET bindings." Like the Python binding: a projection of the C++ API
(`Mila/Bindings/`).

Specified 2026-10-21 in `Mila/Specifications/DotNetBinding.md`: P/Invoke over a C ABI, experimental until
v0.22.0 or later (Todd).

## An end user cannot install the Mila desktop app without a developer's toolkit

`Mila/Specifications/ChatDesktop.md` section 9.4; an absence -- the installer carries no CUDA runtime @ c8f71f60

Todd, 2026-10-24: the desktop app is the end-user showcase. A person who installs it has no CUDA Toolkit,
so the app ships the CUDA libraries its session loads (cuBLAS, cuBLASLt and the rest), which needs
NVIDIA's CUDA EULA redistribution terms read first. The .NET binding's option A ("the user installs CUDA
13") does not fit an end user. Also needed: the NVIDIA GPU requirement stated before the download, code
signing, an installer and updates.

## The installed Qwen 3.8 27B FP4 package does not say it can think or call tools

`%LOCALAPPDATA%\Mila\models\qwen3.8-27b-fp4.json` capabilities; an absence -- no `thinking`, no `tools` @ c8f71f60

Found 2026-10-24 driving the desktop session process: Todd's installed record declares `instruct` and `agentic`
only, so Chat and the desktop app run Qwen 3.8 27B FP4 with no reasoning switch and no effort levels, and the
console's status line has no Thinking clause (the phase 2 baseline shows the same). The fp4-cb3 package built
with `--thinking --tools` declares both. Not checked: whether the published package's `mila.json` or only this
install lacks them.

Also Gemma 4 12B FP4 (2026-10-24, the desktop window): Todd's `gemma-4-12b-it-fp4.json`, installed 2026-10-03,
has no `capabilities` block at all, so the session reports `thinking_capable: false` for a model that thinks.

## A tool call ran in the desktop window without the person being asked, once

`Mila/Apps/Desktop/Session/Src/Desktop.SessionProcess.ixx` `approveToolCall` @ a9087567

2026-10-24, the first scripted window run over the real session (Llama 3.2 3B FP4, tools-weather, scratch
store): the first turn's `get_weather` call ran and its result was stored, while the script clicked nothing; a
later call in a new conversation did ask. Not reproduced in five later runs (two window runs of the same
sequence with every page message recorded, two single-turn window runs, one over stdio): each asked. That run's
files were overwritten by the next, so what the page received is not known.

## A client that sends on `turn.end` is refused as busy

`Mila/Apps/Desktop/Session/Src/Desktop.SessionProcess.ixx` `respond` @ a9087567

`turn.end` is sent from inside the turn and `busy` clears after it, so a `turn.send` written on reading
`turn.end` gets "A reply is still being written" (seen 2026-10-24 from a stdio script). The page waits for
`session.state`, so it does not hit this; `ChatDesktop.md` 4.1 does not say which message a client waits for.

## The session sends a thought break when the model is not thinking

`Mila/Apps/Desktop/Session/Src/Desktop.SessionProcess.ixx` `thoughtBreak` @ a9087567

Gemma 4 12B Q4_0 with thinking off, 2026-10-24: `turn.thought_break` after the tool result, with no thought
text before or after it. `ChatDesktop.md` 4.1 says it means the reasoning resumed. The page shows nothing for it.

## The desktop Models view stays open after a load, and offers Load while one runs

`Mila/Apps/Desktop/Ui/views/models.js` @ a9087567

2026-10-24, the window over the real session: choosing Load leaves the person on the Models view once the model
is loaded, and the row's Load button stays during the load (progress shows above the list).

Also, installing `gpt2-small` from it (2026-10-24): after the install the published list is gone again ("Show
available models"); the list offers `gpt2-small`, a base model the app refuses to chat with, and labels its
quantization "none"; and the installed base model keeps a Load button that can only be refused.
