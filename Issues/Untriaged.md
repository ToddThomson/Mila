# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined after ninety days**, and anything user-reported is a pointer to its
GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

## Chat unloads its model before refusing a context longer than the model was trained for

`Apps/Chat/Src/Chat.Session.ixx` `setContextLength`, `switchModel`; `Lib/Src/Agents/Agent.ixx` `check` @ 0.21.0-dev+108

Workaround chosen by Todd, 2026-10-31: kept for one change, to be fixed in Mila::Agents after the move. Since Chat's family
ceilings were deleted, `/context 300000` on Llama 3.2 3B releases the model, then the planner refuses ("planDeployment:
context length 300000 exceeds the trained maximum 131072"), leaving nothing loaded; before, Chat refused first and kept
the model. The same holds for a fixed context carried across a model switch. Proposed fix: `Agent::check` refuses a
fixed context past the trained maximum, before a program releases anything, in the library's words.

## A pnpm project may not build under Mila Desktop's commands, which tell Node to keep links as named

`Apps/Desktop/Session/Src/Desktop.ProgramSetup.ixx` (`NODE_OPTIONS`) @ 7cf45bdd

Measured 2026-10-31 (`Notebooks/MxcConformance.md` 7.1): in MXC's container Node's resolving of a script's real
path stats the drive's root, which is refused, so every `node` and `npm` run failed; `--preserve-symlinks
--preserve-symlinks-main` lets them run. pnpm lays `node_modules` out as links into its store and relies on real
paths; a pnpm project under those options is not measured. `dotnet build`'s first run in the container also reports
installing a development certificate; where it lands is not measured.

## A command run through the grant reaches every public host or none, never only the hosts a project needs

`Specifications/ContainedPrograms.md` decision 11; an absence -- no allow list by host name @ 7cf45bdd

Todd, 2026-10-31: after phase 3 (network per call), a list of hosts allowed per project in Mila Desktop, and a
question for each new host. MXC's egress rules match address, protocol and port only; names are out of its scope
(`docs/backends/process-container/networking.md`). Its proxy mode (`runtimeConfig.networkProxy`, "Model 2") sends
WinHTTP traffic and `HTTP_PROXY`/`HTTPS_PROXY` to a loopback proxy the host runs, and asks for `egress.default: deny`
with `ingress.default: allow`. Unmeasured: what that ingress opens, and which tools (npm, git, pip, curl, dotnet)
go through the proxy inside the container. The proxy would decide the grant's network, so it is `Lib/Src`.

## Microsoft has not been told of three accesses MXC's policy accepts that no grant gives

`D:\Claude\mxc-cpp` (`policy-conformance-2.ps1`, `-3.ps1`); an absence -- no report made to Microsoft @ 7b5248a2

Measured 2026-10-27 on MXC v1.0.1, Windows 11 build 26300.9711: with the UI on, a contained process reads the
person's screen in every `processContainer.ui.isolation` mode, and no policy field refuses it; and a file with one
name inside a read-write folder and another outside it, or in a denied folder, is accepted for read and write through
its inside name (bubblewrap on Linux the same). MXC's `SECURITY.md` sends security reports to MSRC, never to public
GitHub issues, and Codex ships with the UI on (`codex-rs/mxc-sandbox/src/policy.rs`), so a public issue would tell
everyone first. Todd, 2026-10-27: report after the design spec; Todd files, under his account.

Also measured 2026-10-27 (`ContainedPrograms.md` phase 1): with no folder granted, a contained process reads much of
`HKCU` -- `Environment`, `Software`, `Software\Classes` -- and the policy has no registry field. Todd, 2026-10-31
(`ContainedPrograms.md` decision 8): a known gap for Mila Desktop, and told to Microsoft with the other two.

## On NTFS a contained run walks its folders whole before every run

`Lib/Src/Agents/SecondNameWatch.ixx` (`watcher_reports_every_name_`) @ 7b5248a2

Todd, 2026-10-31 (`ContainedPrograms.md` decision 7, narrowed): `ReadDirectoryChangesW` on NTFS reports no name made
outside for a file inside, so off ReFS every run walks the granted folders whole -- about 115 ms for 7,500 files,
5.5 s for 358,000 (`Notebooks/MxcConformance.md` section 7). NTFS's change journal records a link added
(`USN_REASON_HARD_LINK_CHANGE`); whether it can be read without an administrator is unmeasured.

## The grant refuses a file with a second name even when every name lies inside the folder

`Lib/Src/Agents/ToolGrant.ixx` (`requireOneName`) @ 7b5248a2

Todd, 2026-10-27 (`ContainedPrograms.md` decision 5): kept for v0.21.0. `FindFirstFileNameW` and
`FindNextFileNameW` list a file's names, so the grant could accept a file whose names all lie within granted
folders giving at least the access asked, as the container's walk does (`ContainedPrograms.md` 4.3). A project
whose files are linked between two of its own folders is refused today.

## Two installed models that think are recorded as not thinking, so neither app offers their switch

`mila-store show gemma-4-12b-it-fp4` and `mila-store show Qwen3.8-27B-fp4` in Todd's store @ e9c9430f

Seen 2026-10-27: both list their capabilities as `instruct` alone, while the store's `gemma-4-26B-A4B-it-q4_0` and
the scratch store's `gemma-4-12b-it-q4_0` list `thinking` too. `ModelCapabilities::thinking` is read from the
record, so `Agent::thinks()` is false for both whatever the template's switch says, and Chat and Mila Desktop show
no thinking switch for them. Likely packages installed before the manifest carried thinking; not checked against
what the hub publishes now.

## In Get Things Done, Gemma 4 12B looks at the folder before it proposes a plan about half the time

`D:\Claude\filetools\drive_tasks.py` (the session drive's "the first call proposes a plan" check); `Apps/Desktop/Prompts/tasks.json` (`Apps/Chat/Prompts/tasks.json` until +105) @ cc07f776

Measured 2026-10-27 on Gemma 4 12B Q4_0 at the publisher's sampling: the first call was `list_folder` in 2 of 3 runs
on `+103`'s session and 3 of 3 on the session change after it, `propose_plan` otherwise; the plan, its steps and the
edit ran correctly every time. The same model chose both ways on one build, so the check measures the model's choice,
not the session. On `+105`'s session (Mila Desktop on `Mila::Agents` alone, same prompt and tool schemas byte for byte):
`list_folder` first in 4 of 6 runs, then a plan carried out correctly; `propose_plan` first in 1; and once no plan
at all -- three edits, then an answer claiming a README change it never made. Whether the prompt should allow a look first, or the check should, is not decided.

## Off Windows a tool's grant holds no folder, so no file tool can run there

`Lib/Src/Agents/GrantFileSystem.Posix.ixx`; an absence -- no POSIX file checks behind the grant @ f42c0d95

Todd, 2026-10-27: the grant is Windows-only for now (`ToolCalling.md` 3.4 *Where*). Every POSIX call refuses, so a
grant of folders throws and a tool reaches none. The likely route is Linux's `openat2` with `RESOLVE_BENEATH` from
a held folder descriptor (Linux 5.6, man7.org openat2(2)), which decides in the kernel rather than by a final path,
plus a link-count check for hard links, which the man page does not mention. Unmeasured; macOS not looked at.

## Nobody has read whether Tavily's terms let Mila Desktop keep search results in a saved conversation

`Apps/Desktop/Session/Src/Desktop.WebSearch.ixx`; an absence -- Tavily's terms of service not read @ da647c29

Found 2026-10-26 adding Tavily web search: a saved conversation keeps every tool result, search results included, and
Tavily's credits page says nothing on keeping or showing results. Brave grants that only on some plans.

## Mila's HTTP transport names itself Mila 0.20 to every server

`Lib/Src/Distribution/CurlHttpTransport.ixx` (`CURLOPT_USERAGENT`, "Mila/0.20") @ da647c29

Found 2026-10-26 writing Mila Desktop's page reader, which fetches through it: the user agent is a written-in
string from the 0.20 cycle, not `Version.txt`'s, so model installs and web pages alike are requested as Mila 0.20.

## A program names a compute precision to open a model whose precision its package already decides

`Lib/Src/Agents/Agent.ixx` (`Agent`, `BasicAgent`) and `Lib/Src/Deployment/ModelHandle.ixx` (`ModelHandle`) @ 793bddb5

Todd, 2026-10-26: `BasicAgent<Cuda, BF16>` reads as a BF16 model, but the same agent runs the FP4, FP8 and codebook
packages -- BF16 is the compute precision, and the weight format comes from the package named. Every installed
model's format runs at BF16 compute only (Qwen refuses another, `QwenModel.ixx` near 567); FP32 is used below
`Mila::Agents`, by the HuggingFace reference tests and GPT-2's training. Outlined to Todd: `BasicAgent<TDeviceType>`
and `ModelHandle<TDeviceType>`, BF16 compute fixed inside; `Direction.md` 4.1 and `ModelHandle.md` 3 and 10.1 say
the pair. A format that changes the compute itself (Vnext's NVFP4 W4A4) is the package's choice too.

## Two of the agent's answers to a stop or a cut have no test that fails without them

`Lib/Src/Agents/Agent.ixx` (`respond`, `dropIncompleteLastCharacter`) @ 6f077474

Found 2026-10-25 writing the limit tests: a round with several calls, stopped at its first, gives every call after
it a stopped result -- the tests' model makes one call per round, so nothing reaches them. And a result cut between
two bytes of one character drops the character's leading bytes; the test's result is ASCII, so no cut lands inside
a character.

## GPT-2 cannot be opened by name, because it has no deployment planner

`Lib/Src/Dnn/Models/Gpt/GptModel.ixx` (`load`); an absence -- no `planDeployment`, no `load( path, request )` @ 1f310d50

Found 2026-10-25 building the architecture list (`ModelHandle.md` 3.3): GPT-2 has a model and a tokenizer, but
its one `load` takes a config and no deployment request, so it cannot be an entry and `gpt2-small` cannot be
opened through the model handle. The spec's proof that adding an architecture is one list entry was GPT-2 (H5).

## Llama 3.1 8B FP4 answers as though a tool's result were made up

`Tests/Agents/Agent.Cuda.cpp` (`askForTheWeather`); scratch-store package `Llama-3.1-8B-Instruct-fp4` @ efa24591

Measured 2026-10-25 through `Mila::Agents::Agent`, the publisher's sampling, seeds 1 to 6, one question about the weather
in Paris and a tool returning "Sunny, 17 degrees Celsius." The rendered prompt is Meta's template: the result is an
`ipython` turn, quoted as the template quotes a string. With the tools in the first user message, the template's
default, it called once every time and then used the result once in six: "This is just an example response and may
not reflect the actual weather", "The function `get_weather` is not defined in the given code snippet". With them in
the system message, two of six; with that and Chat's when-to-call system prompt, one of six, three runs making no
call. Its manifest says `agentic: false`, so the agent's gate does not ask it to. 3.2 3B in the same runs: zero of
twenty-four. Data `D:\Claude\agent\llama_sweep.txt`, `llama_sweep2.txt`.

## Llama 3.2 3B offered a tool in the system message says it will call it and does not

`Apps/Chat/Prompts/tools-weather.json`; scratch-store package `Llama-3.2-3B-Instruct-fp4` @ e8df2aef

Measured 2026-10-25 at temperature 0, ten questions that need no tool and six about the weather now. With the
tools in the system message and the system prompt saying when to call, as Chat now renders them, 3B FP4 and BF16
alike answered all ten directly and called `get_weather` for none of the six: "I'll check the current weather
for Berlin." and stop, or a sentence and then the call, which the Llama grammar reads as text. With the tools in
the first user message, the template's default, it called for all sixteen. Meta's documented 3.2 prompt, with
pythonic calls (llama-models `text_prompt_format.md`), plus the same sentence: ten of ten, three of six. Data
`D:\Claude\llama-tools\` (`mila-3b-*`, `meta-3b-fp4.jsonl`, `chat-live-Llama-3.2-3B-Instruct-fp4.jsonl`).

## Llama 3.1 8B FP4 writes some tool calls in a shape its template does not define

`Lib/Src/Dnn/Models/LlaMa/Llama.ChatProtocol.ixx` (`templateReplyGrammar`); scratch-store package
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

`Apps/Chat/Src/Chat.SystemPrompt.ixx` (`ToolDefinition`'s comment) @ 5e7de360

Since `+94` every family's template places the tools itself, and Chat writes none into the system turn; the
comment still says Chat serializes them, with Llama's pythonic wording. Fixing it brings the module under the
one-type rule, and it holds five types.

## .NET bindings for Mila

an absence -- no file in the repository or in `.internal/` names them @ a8f5360a

Todd, 2026-10-20, while choosing the toolkit for Chat's desktop window (`Specifications/MilaDesktop.md`
section 9): "We do have a future for .NET bindings." Like the Python binding: a projection of the C++ API
(`Lib/Bindings/`).

Specified 2026-10-21 in `Specifications/DotNetBinding.md`: P/Invoke over a C ABI, experimental until
v0.22.0 or later (Todd).

## Mila Desktop's download is unsigned, has no updates and does not say it needs an NVIDIA GPU

`Specifications/MilaDesktop.md` section 9.4; an absence -- no signing, no update channel, no stated
requirement @ c8f71f60

Todd, 2026-10-24: Mila Desktop is the end-user showcase. Since +101 its installer carries the session, the
CUDA libraries it loads and the Visual C++ runtime, so no toolkit is needed. Still missing: the NVIDIA GPU
requirement stated before the download, code signing (SmartScreen warns on an unsigned one), and updates.

## The installed Qwen 3.8 27B FP4 package does not say it can think or call tools

`%LOCALAPPDATA%\Mila\models\qwen3.8-27b-fp4.json` capabilities; an absence -- no `thinking`, no `tools` @ c8f71f60

Found 2026-10-24 driving Mila Desktop's session process: Todd's installed record declares `instruct` and `agentic`
only, so Chat and Mila Desktop run Qwen 3.8 27B FP4 with no reasoning switch and no effort levels, and the
console's status line has no Thinking clause (the phase 2 baseline shows the same). The fp4-cb3 package built
with `--thinking --tools` declares both. Not checked: whether the published package's `mila.json` or only this
install lacks them.

Also Gemma 4 12B FP4 (2026-10-24, Mila Desktop's window): Todd's `gemma-4-12b-it-fp4.json`, installed 2026-10-03,
has no `capabilities` block at all, so the session reports `thinking_capable: false` for a model that thinks.

## A tool call ran in Mila Desktop's window without the person being asked, once

`Apps/Desktop/Session/Src/Desktop.SessionProcess.ixx` `approveToolCall` @ a9087567

2026-10-24, the first scripted window run over the real session (Llama 3.2 3B FP4, tools-weather, scratch
store): the first turn's `get_weather` call ran and its result was stored, while the script clicked nothing; a
later call in a new conversation did ask. Not reproduced in five later runs (two window runs of the same
sequence with every page message recorded, two single-turn window runs, one over stdio): each asked. That run's
files were overwritten by the next, so what the page received is not known.

## A client that sends on `turn.end` is refused as busy

`Apps/Desktop/Session/Src/Desktop.SessionProcess.ixx` `respond` @ a9087567

`turn.end` is sent from inside the turn and `busy` clears after it, so a `turn.send` written on reading
`turn.end` gets "A reply is still being written" (seen 2026-10-24 from a stdio script). The page waits for
`session.state`, so it does not hit this; `MilaDesktop.md` 4.1 does not say which message a client waits for.

## The session sends a thought break when the model is not thinking

`Apps/Desktop/Session/Src/Desktop.SessionProcess.ixx` `thoughtBreak` @ a9087567

Gemma 4 12B Q4_0 with thinking off, 2026-10-24: `turn.thought_break` after the tool result, with no thought
text before or after it. `MilaDesktop.md` 4.1 says it means the reasoning resumed. The page shows nothing for it.

## Mila Desktop's Models view stays open after a load, and offers Load while one runs

`Apps/Desktop/Ui/views/models.js` @ a9087567

2026-10-24, the window over the real session: choosing Load leaves the person on the Models view once the model
is loaded, and the row's Load button stays during the load (progress shows above the list).

Also, installing `gpt2-small` from it (2026-10-24): after the install the published list is gone again ("Show
available models"); the list offers `gpt2-small`, a base model the app refuses to chat with, and labels its
quantization "none"; and the installed base model keeps a Load button that can only be refused.

## A tool that needs a shell has nothing on Windows to hold it to its grant

`Specifications/ToolCalling.md` sections 3.4 and 6; an absence -- no sandbox under the containment interface @ 571b5063

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

Measured 2026-10-27 (`Specifications/Notebooks/MxcConformance.md`) and designed (`Specifications/ContainedPrograms.md`,
draft): Todd decided the same day that MXC goes into v0.21.0 as an experiment, off by default.

## No CI job builds Mila Desktop's installer

`.github/workflows/desktop.yml`; an absence -- the `windows-installer` job removed @ 793bddb5

Removed 2026-10-26 when the installer began carrying the session (`MilaDesktop.md` 9.4): `npm run build:windows`
refuses to run without `Window/session/` staged from a C++ build, and the app's workflow (`desktop.yml`) has no C++ build. Before
that the job's upload was an installer that ran only the mock. Nothing builds the installer now but a person
with a Release build directory (`Apps/Desktop/README.md`, *Staging it for the window*).
