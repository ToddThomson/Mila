# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined after ninety days**, and anything user-reported is a pointer to its
GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

## Off Windows a tool's grant holds no folder, so no file tool can run there

`Mila/Src/AI/GrantFileSystem.Posix.ixx`; an absence -- no POSIX file checks behind the grant @ f42c0d95

Todd, 2026-10-27: the grant is Windows-only for now (`ToolCalling.md` 3.4 *Where*). Every POSIX call refuses, so a
grant of folders throws and a tool reaches none. The likely route is Linux's `openat2` with `RESOLVE_BENEATH` from
a held folder descriptor (Linux 5.6, man7.org openat2(2)), which decides in the kernel rather than by a final path,
plus a link-count check for hard links, which the man page does not mention. Unmeasured; macOS not looked at.

## Neither Chat nor the desktop app shows the person what a tool call would reach

`Mila/Apps/Chat/Src/Chat.Session.ixx` `askAboutCall`; an absence -- the question's needs and grant are not shown @ f42c0d95

`ToolCalling.md` 3.3 says the person is shown the tool's name and summary, every argument, and what it will touch.
`ToolQuestion` carries all of it, the grant included since `+103`, but `askAboutCall` passes on only the name and
the arguments, so the console line and the desktop app's approval card show neither the summary, the needs nor the
folders and network the call would be given. The desktop's card is `MilaApp.md` 4.5's schema, so showing them is a
protocol change too.

## Nobody has read whether Tavily's terms let the Mila app keep search results in a saved conversation

`Mila/Apps/Desktop/Session/Src/Desktop.WebSearch.ixx`; an absence -- Tavily's terms of service not read @ da647c29

Found 2026-10-26 adding Tavily web search: a saved conversation keeps every tool result, search results included, and
Tavily's credits page says nothing on keeping or showing results. Brave grants that only on some plans.

## Mila's HTTP transport names itself Mila 0.20 to every server

`Mila/Src/Distribution/CurlHttpTransport.ixx` (`CURLOPT_USERAGENT`, "Mila/0.20") @ da647c29

Found 2026-10-26 writing the desktop app's page reader, which fetches through it: the user agent is a written-in
string from the 0.20 cycle, not `Version.txt`'s, so model installs and web pages alike are requested as Mila 0.20.

## A program names a compute precision to open a model whose precision its package already decides

`Mila/Src/AI/Agent.ixx` (`Agent`, `BasicAgent`) and `Mila/Src/AI/ModelHandle.ixx` (`ModelHandle`) @ 793bddb5

Todd, 2026-10-26: `BasicAgent<Cuda, BF16>` reads as a BF16 model, but the same agent runs the FP4, FP8 and codebook
packages -- BF16 is the compute precision, and the weight format comes from the package named. Every installed
model's format runs at BF16 compute only (Qwen refuses another, `QwenModel.ixx` near 567); FP32 is used below
`Mila::AI`, by the HuggingFace reference tests and GPT-2's training. Outlined to Todd: `BasicAgent<TDeviceType>`
and `ModelHandle<TDeviceType>`, BF16 compute fixed inside; `Direction.md` 4.1 and `ModelHandle.md` 3 and 10.1 say
the pair. A format that changes the compute itself (Vnext's NVFP4 W4A4) is the package's choice too.

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

Todd, 2026-10-20, while choosing the toolkit for Chat's desktop window (`Mila/Specifications/MilaApp.md`
section 9): "We do have a future for .NET bindings." Like the Python binding: a projection of the C++ API
(`Mila/Bindings/`).

Specified 2026-10-21 in `Mila/Specifications/DotNetBinding.md`: P/Invoke over a C ABI, experimental until
v0.22.0 or later (Todd).

## The Mila desktop app's download is unsigned, has no updates and does not say it needs an NVIDIA GPU

`Mila/Specifications/MilaApp.md` section 9.4; an absence -- no signing, no update channel, no stated
requirement @ c8f71f60

Todd, 2026-10-24: the desktop app is the end-user showcase. Since +101 its installer carries the session, the
CUDA libraries it loads and the Visual C++ runtime, so no toolkit is needed. Still missing: the NVIDIA GPU
requirement stated before the download, code signing (SmartScreen warns on an unsigned one), and updates.

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
`session.state`, so it does not hit this; `MilaApp.md` 4.1 does not say which message a client waits for.

## The session sends a thought break when the model is not thinking

`Mila/Apps/Desktop/Session/Src/Desktop.SessionProcess.ixx` `thoughtBreak` @ a9087567

Gemma 4 12B Q4_0 with thinking off, 2026-10-24: `turn.thought_break` after the tool result, with no thought
text before or after it. `MilaApp.md` 4.1 says it means the reasoning resumed. The page shows nothing for it.

## The desktop Models view stays open after a load, and offers Load while one runs

`Mila/Apps/Desktop/Ui/views/models.js` @ a9087567

2026-10-24, the window over the real session: choosing Load leaves the person on the Models view once the model
is loaded, and the row's Load button stays during the load (progress shows above the list).

Also, installing `gpt2-small` from it (2026-10-24): after the install the published list is gone again ("Show
available models"); the list offers `gpt2-small`, a base model the app refuses to chat with, and labels its
quantization "none"; and the installed base model keeps a Load button that can only be refused.

## A tool that needs a shell has nothing on Windows to hold it to its grant

`Mila/Specifications/ToolCalling.md` sections 3.4 and 6; an absence -- no sandbox under the containment interface @ 571b5063

Todd, 2026-10-26: interested in Microsoft's execution containers (MXC) and NVIDIA OpenShell for the agent. OpenShell
sandboxes a whole agent; its Windows design (NVIDIA/OpenShell#2050, RFC 0013, tracked by #4450) runs MXC through
`wxc-exec.exe` -- AppContainer and a low-integrity token, unelevated, Windows 11 build 26100 or later. Per that design:
read-write and read-only paths are set at creation, denied paths "limited / unsupported"; network filtering by host, IP
or CIDR only, `allowedHosts`/`blockedHosts` "not yet enforced on Windows", so OpenShell's host proxy is the real network
enforcer; mapping five policies onto MXC gave 64 access-broadening errors. #2050 closed stale, "not planned",
2026-10-13; #4450 showed 6 of 9 sub-issues done. Fits Mila per tool, not per agent: the loop runs in the model's
process (section 1). Section 6 records MXC kept out for now (Todd, 2026-10-25). Measured 2026-10-26, a standalone
C++23 program calling `mxc_ffi.dll` v1.0.1 (outside the repository): a process container launches in 40-70 ms; reads
and writes outside the one read-write folder and all network were refused, beside the same accesses inside it
allowed; the process runs as the person's own account; with the UI disabled `whoami.exe` fails to start
(0xC0000142), allowed it runs. Unmeasured: whether it holds a grant as completely as section 3.4's file and network
checks describe.

## No CI job builds the desktop app's installer

`.github/workflows/desktop.yml`; an absence -- the `windows-installer` job removed @ 793bddb5

Removed 2026-10-26 when the installer began carrying the session (`MilaApp.md` 9.4): `npm run build:windows`
refuses to run without `Window/session/` staged from a C++ build, and the desktop workflow has no C++ build. Before
that the job's upload was an installer that ran only the mock. Nothing builds the installer now but a person
with a Release build directory (`Mila/Apps/Desktop/README.md`, *Staging it for the window*).
