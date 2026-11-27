# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined after ninety days**, and anything user-reported is a pointer to its
GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

## Mila asked in words whether to make a project, then offered it and went on before the answer, so no project was made

`Apps/Assistant/Prompts/general.json` (the `propose_project` passage) @ 0.21.0-dev+167

Found 2026-11-27 in Todd's sixth HelloWorld walk (`D:\Claude\helloworld\walk6`): to "Let's start with a simple HelloWorld C++23 project using CMake" Gemma 4 26B-A4B wrote "Would you like to start a project called "Hello World C++23"...?"; after "Yes" it called `propose_project`, whose result says the user answers in their next message, and in the same reply proposed the plan, which Todd approved. The offer was never answered, so the chat is in no project and its folder joined none. The same habit as asking for a folder in words (fixed that day for `request_folder`): the passage does not say never to ask in words first.

## A round rendered afresh after a reply with text and a call reads that text after the results, its turn closed and thinking opened outside it

`Lib/Src/Agents/Implementation/Agent.cpp` (`nextRound`, rendered whole when the framing changed) @ 0.21.0-dev+166

Found 2026-11-27 in Todd's fifth HelloWorld walk through the round record (`D:\Claude\helloworld` walk 5, turn 5 round 2): Gemma 4 26B-A4B wrote a sentence and then called `request_folder`; the folder changed the tools, so the round was rendered whole, and its prompt ended `...build the code.<turn|>\n<|channel>thought\n`. Gemma 4's own `chat_template.jinja` renders the same history the same way (checked with jinja2 the same day, `D:\Claude\gemma-render\render.py`): a message's content after its results, then the turn closed. So `Gemma.ChatProtocol` is faithful and the shape is the template's; what the model wrote (text, then the call, the turn open) is lost only when a response's rounds are rendered rather than continued. The model then closed every thought empty for the rest of the turn. Put to Todd: render the response's first prompt afresh with the new framing and continue from there with what the model generated, as the splice does.

## A chat cannot be shared with someone who does not have Mila

`Apps/Assistant/Ui` (the chat's menu) and `Apps/Assistant/Session` (the stored conversation) @ 0.21.0-dev+166

Todd, 2026-11-27, during his HelloWorld walks: the chat should have a share feature, so others can see a chat properly. Discussed the same day: a self-contained HTML file (the transcript as the window draws it, thinking and tool calls folded with `<details>`, no script, opens offline in any browser) as the main form; Markdown to paste into an issue or a message; PDF through Electron's own print to PDF. Made on this computer and sent by the person, never uploaded by Mila (AssistantInterface.md rule 7); what goes in (thinking, tool results, folder paths, file contents) shown and chosen before the file is made.

## A script the model writes and runs hides a failed step from the command's exit code

`Apps/Assistant/Session/Src/Tools/Assistant.CommandTool.ixx` (`script`) @ 0.21.0-dev+166

Found 2026-11-27 in Todd's second HelloWorld walk (Gemma 4 26B-A4B, `conversation-20261127171128-ecad1e`, turn 29): the model wrote `build.ps1` (cmake, cmake --build, `./build/Debug/HelloWorld.exe`) and ran `.\build.ps1`. The last line failed (NMake has no `Debug` folder) and the result read `Exit code: 0`, the error only under standard error: the wrapper runs the command's statements one at a time, and the script is one statement, whose cmdlet errors do not fail it. A program failing inside the script is not stopped at either. The prompt now asks for steps as commands of their own; the wrapper still lets a script through.

## A new piece of software was planned with CMake but no git, though the prompt asks for both

`Apps/Assistant/Prompts/general.json` (the `run_command` passage) @ 0.21.0-dev+166

Found 2026-11-27 replaying Todd's HelloWorld walk on Gemma 4 26B-A4B after the prompt gained "plan it with the build system and source control its developers normally use ... git for any project": the plan made `src`, `main.cpp` and `CMakeLists.txt`, configured and built into `build`, and never ran `git init` nor wrote a `.gitignore`. One run.

## Every CMake build in a command warns that `/Zi` is overridden by `/Z7`

`Apps/Assistant/Session/Src/Tools/Assistant.ProgramSetup.Windows.ixx` (`_CL_=/Z7`) @ 0.21.0-dev+166

Found 2026-11-27 in the same replay: `cmake --build build` on a default Debug project printed `cl : Command line warning D9025 : overriding '/Zi' with '/Z7'` on standard error. The 26B read past it; it is on every build, and a model told to fix warnings may chase one it cannot fix.

## The model wrote its reply twice around `note_understanding`

`Apps/Assistant/Prompts/general.json` (the `note_understanding` passage, "end the turn by calling") @ 0.21.0-dev+166

Found 2026-11-27 reading Todd's HelloWorld walk on Gemma 4 26B-A4B: turn 16 is the reply and the `note_understanding` call together, and turn 18, after the tool's result, is the same reply word for word. The page showed it once. A call always gets a round after it, so "end the turn by calling" cannot be done as written, and the result ("Noted for the project ...") does not say the reply was already given. Prompt and result changed the same day (call it before the reply; the result says to write the reply once); not yet seen working, since the 26B replay of the walk never called `note_understanding` at all.

## A command's output names a different folder from the one the person gave Mila

`Apps/Assistant/Session/Src/Implementation/SessionProcess.cpp` (the folder mapped into the container) @ 0.21.0-dev+166

Found 2026-11-27 reading Todd's HelloWorld walk on Gemma 4 26B-A4B. He gave `C:\Users\ToddT\Src\Mila\HelloWorld`, and `request_folder`'s result said so, but `New-Item`'s output inside the container read `Directory: D:\Mila\HelloWorld`. The model let it pass this time; a model that writes an absolute path, or a person reading the tool card, meets two names for one folder.

## A project made from a chat keeps none of that chat's folder, and its note is never brought up to date

`Data/TestUser/Mila/Assistant/Projects/project-20261127151632-dd1c55` (`project.json`, `understanding.json`) @ 0.21.0-dev+166

Found 2026-11-27 in the same walk: the folder was given after the chat moved into the HelloWorld project, yet the project's `folders` is empty, so a new chat in it starts with no folder. Its understanding note, written once when the project was made, still says the next step is "Ask Todd how he would like to begin", after the files were written, built and run; the model called `note_understanding` once and not after the plan or the build. In the replay the same day it was never called, so the project has no note.

## After suggesting a project, Gemma 4 26B wrote "thought" line after line as its reply, then calls and an answer it made up

`Data/TestUser/Mila/Assistant/Conversations/conversation-20261127163614-482543.json` (turn 12; the window over x64-claude-verify) @ 0.21.0-dev+166

Found 2026-11-27 at the start of Todd's clean walk, on the 5060 Ti. After `propose_project`'s result ("... in a sentence, say what you suggested"), the round thought twice (1,480 characters, shown as two "Thought for 350 tokens" blocks) about whether it should say anything before the person answered ("I shouldn't say anything more until they answer"), then wrote 18,058 characters of reply: "thought" on each line, ending in plain-text `propose_project(...)`, `offer_answered()` and "I've created the project HelloWorld for you", none of which happened. Whether the model wrote the word or a channel marker the reader hid (`<channel|>` is hidden, `<|channel>` opens reasoning) is not known: the tokens were not kept. The engine opens the thought channel itself after a tool result when thinking is on (`Gemma.ChatProtocol.ixx` `continueWithToolResults`). The same step ran clean in Todd's walk that morning and in the replay. Third unbounded round in two days (see the next entry); nothing ended it but the model.

## A round can think in a loop until the context is full, and the reply comes out empty

`D:\Claude\helloworld\g26.log` and `D:\Claudeound13ound-off.log` (the real session) @ 0.21.0-dev+166

Found 2026-11-26 testing commands for Todd: on Gemma 4 12B Q4_0, after `update_step` marked the last of five steps done at 44.8 s, the session sent nothing more until the script's 600 s limit stopped it. Not reproduced then. Reproduced 2026-11-27 on Gemma 4 26B-A4B (the 5060 Ti, Todd's HelloWorld walk replayed): the model had called `update_step` in_progress for each step beside its work, in the same round, and never done; after the last step's done, one round thought for 642 s, 185,930 characters ending in "Actually, I'll call `update_step` for all 4." over and over, and ended at the context (70,099 tokens) with no reply. Mila Assistant sets no `AgentLimits` (`SessionProcess.cpp`, `withLimits( AgentLimits{} )`), by ToolCalling.md 4's decision that the library sets none and an application sets its own, so only the context ends such a round. The second turn's plan had steps 1 and 2 left in_progress the same way, though both were done.

## After setup downloads a model, the first chat on it does not know it is that model, and asks whether the person means a machine learning model

`Apps/Assistant/Session/Src/Implementation/SessionProcess.cpp` (the system prompt); `Apps/Assistant/Prompts/interview.json` (`assess_needs`, "do not name any model") @ 0.21.0-dev+165

Found 2026-11-26 reading Todd's walk of the interview. His first chat on Gemma 4 26B-A4B after setup downloaded it opened "Are you ready with the new model?"; the model searched his chats for "new model", found nothing (the interview offered "a download" and never said "model"), and asked whether he meant a model for a project or a machine learning model. The system prompt says nothing of which model is running, that it was just downloaded, or what setup did.

## A change to Mila Assistant's prompts alone does not reach the build, so the session runs the old prompt

`Apps/Assistant/Session/CMakeLists.txt` (the `POST_BUILD` copy of `Prompts`) @ 0.21.0-dev+164

Found 2026-11-25 changing `Prompts/interview.json` during Todd's walk of the interview. The prompts are copied beside `mila-session.exe` by a `POST_BUILD` command, which runs only when the executable links; a build after editing a prompt alone links nothing, reports success, and the session goes on reading the old copy. Seen in `x64-claude-verify`; a VS build of `x64-profile` works the same way. Copied by hand with `cmake -E copy_directory` that day.

## Mila Assistant does not tell the model today's date or the time, so on Gemma and Qwen it does not know what day it is

`Apps/Assistant/Session/Src/Implementation/SessionProcess.cpp` (the system prompt, beside `person_name`); `Lib/Src/Dnn/Models/LlaMa/Llama.ChatProtocol.ixx` (`date_string`) @ 0.21.0-dev+164

Found 2026-11-25 scoping the interview's goals. The session's system prompt carries the person's name and nothing of when or where: no date, time or time zone. Llama's template fills `date_string` with today's date itself; Gemma's and Qwen's carry none, so the onboarding model answers "what's on today" or "next Tuesday" with no date at all. The computer knows all of it on Windows, macOS and Linux.

## Models a person installed with v0.20 are not found by 0.21, since the store moved and nothing moves it

`Lib/Src/Distribution/Store/ModelStore.ixx` (`resolveStoreRoot`); `Apps/Chat/Src/Settings/Chat.SettingsResolution.ixx` (`resolveConfigRoot`); `Specifications/ModelDistribution.md` @ 0.21.0-dev+163

Found 2026-11-21 moving each program into its own area of the Mila folder. The store moved from `%LOCALAPPDATA%\Mila` (and `~/.cache/mila`) to the folder's `Store`, and Chat's `chat.json` from the top of `%APPDATA%\Mila` (and `~/.config/mila`) to its `Chat`; a v0.20 install's models and settings stay where they were. Todd, 2026-11-21: "Move by hand for now" -- no code moves them yet. The proposal on the table was the library moving an old store into `Store` on first use, and Chat the same for `chat.json`.

## Gemma 4 12B FP4 often sends a file tool's path as `="."`, and some of its calls are not understood at all

`Lib/Src/Models/Gemma` (Gemma's chat protocol, its tool-call reading); `Apps/Assistant/Session` @ 0.21.0-dev+158

Found 2026-11-19 checking `AssistantInterface.md` phase 8 on the real session. Asked to read `notes.txt` and write
`summary.md` in a project's folder, the model's calls were stored with arguments `{"path":"=\".\""}` and `{"path":"=\"\""}`,
and two more were refused as `[tool call not understood: <|tool_call>call:list_folder{path:,"."}<tool_call|>]`; the
turn ended with no file written. The same prompt run again wrote the file with clean calls. Phase 4's run on the same
model stored `{"path":"=\".\""}` and `{"path":"=\"notes.txt\""}` before recovering. Not
seen on the Mac's Q4_0 in one run. Whether the model writes this or the reading takes it so is not known.

## Mila Assistant cannot install a gated model, since it no longer sends a token and has no step to ask for one

`Apps/Assistant/Session/Src/Assistant.Models.ixx` (`installModel`); `Lib/Src/Distribution/Hub/HuggingFaceHub.ixx` (`makeAnonymousModelHub`); `Specifications/Apps/AssistantOnboarding.md` 4.2 @ 0.21.0-dev+154

Found 2026-11-18 building `AssistantInterface.md` phase T1. Listing and installing now send no token. The design has the person told and asked before a token found on the computer is used for a model that needs one; that question is not built, so such a model's install fails with the publisher's refusal.

## A tool call that fails on what it met still marks the chat as having read from outside

`Lib/Src/Agents/Tools/ToolGate.ixx` (`answer`); `Specifications/ToolCalling.md` 3.3 *What a result may cause* @ 0.21.0-dev+160

Found 2026-11-19 at phase T2; narrowed 2026-11-20 at phase 5, where a refusal by the grant alone (`GrantRefusal`) no longer exposes. Every other failure still does, though many are the tool's own words with no outside text (*there is no file*, a path outside the folder). Telling those from an error that carries outside text, such as an HTTP error body, is undecided.

## A grant opens a device a link leads to before it refuses it, on every platform

`Lib/Src/Agents/Grants/GrantFileSystem.*.ixx` (`openExisting`, `openChild`); `Specifications/ToolCalling.md` 3.4 *Files on Linux*, *Not chosen* @ 0.21.0-dev+153

Found 2026-11-17 designing the Linux grant. A symbolic link in a granted folder to a device (`/dev/...` on macOS and
Linux) is followed and the device opened, and only then is it refused as not a file or folder; on Linux a device whose
open has an effect of its own is reached the same way. Windows' path form refuses device names, but a link to a device
was not looked at there. Linux could open with `O_PATH` and open again through `/proc` once checked; macOS and Windows
have no such call, so whether it is worth one platform's design differing is not decided.

## The terms a person accepts in Mila Assistant are Mila's MIT licence alone, while NVIDIA's and Microsoft's ask more

`Apps/Assistant/Session/Src/Settings/Assistant.MilaLicense.ixx` (what is accepted); `NOTICE.md` *What Mila Assistant carries*; `Specifications/AssistantOnboarding.md` 4.1 @ 0.21.0-dev+151

Found 2026-11-17 reading the terms of what the Windows installer carries. NVIDIA's CUDA EULA (v13.5, 1.1.2 item 5):
"The terms under which you distribute your application must be consistent with the terms of this Agreement",
NVIDIA's intellectual property included. Visual Studio Community's *Distributable Code* terms in 2013 and 2015 asked
that distributors and end users "agree to terms that protect the Distributable Code at least as much as this
agreement"; the 2026 text was not read (its page renders by script) and Microsoft's redistribution list says only
"subject to the License Terms for the software". The installer now carries NVIDIA's EULA in `Licences/`; the person
accepts Mila's `License.md`, which grants rights to Mila's own code and says nothing of the libraries beside it.
Not decided: whether the accepted terms name the third-party terms the app carries.

## With MXC built in, opening the container from an empty path throws instead of returning why there is none

`Lib/Src/Agents/Programs/MxcProgramContainer.Windows.ixx:221` (`std::filesystem::absolute( library )`) @ 0.21.0-dev+155

Found 2026-11-17: `MxcProgramContainer::open( {} )` throws `absolute: The filename, directory name, or volume label
syntax is incorrect`, where its contract and the absent twin's return the reason. It made the 14 `ContainedRunTest`
tests fail instead of skip until 0.21.0-dev+156, whose tests no longer open an empty path. No caller passes one today
(the session opens the file beside itself), so it is only the contract.

## Nothing at the operating system's level confines `mila-session`, so a defect in the grant reaches whatever the person can

`Apps/Assistant/Session/Src/main.cpp`; `Apps/Assistant/Window/src/main.js` (starts the session); `Specifications/Apps/MilaAssistant.md` section 9 item 4 (*Decided 2026-11-14*, App Sandbox) @ 0.21.0-dev+151

Found 2026-11-17 reviewing MXC v1.1.0 with Todd ("Mila has no other way to sandbox tool calls"). Every tool runs in
`mila-session` as the person, held only by the grant and the question before each call; MXC holds only the programs
`run_command` starts. A second layer would confine the session itself, on both platforms by one design (Todd's parity
rule, `CLAUDE.md`): an AppContainer on Windows; on macOS the App Sandbox, or a Seatbelt profile the window applies when it
starts the session. Todd decided 2026-11-14 that the App Sandbox belongs with a decision to sell through the Mac App
Store, since it moves the model store into the app's container and needs a security-scoped grant per folder, so that
road needs his word again. Not known: whether Metal and CUDA work inside either confinement, where the model store
and conversations would live, and how a folder the person chooses reaches a confined session on each platform.
Claude's call (Todd, 2026-11-17, to review later): captured here, not v0.21.0 work.

## A reply that falls into repeating itself runs for hours in Mila Assistant, with nothing but the person to stop it

`Apps/Assistant/Session/Src/Assistant.SessionProcess.ixx` (`useActivity`, `AgentLimits{}`); `Specifications/Apps/MilaAssistant.md` 3.6 and the *Activities* decision ("a reply runs to the context", Todd, 2026-11-13) @ 0.21.0-dev+149

Found 2026-11-17 measuring AssistantInterface.md 5.5 on the Mac (Gemma 4 12B Q4_0, thinking on, the publisher's
sampling, context 262,144 automatic): with Get Things Done's prompt and tools, "Make me a packing list for a weekend
hiking trip, then turn it into a checklist grouped by category", two `propose_plan` calls declined, then one round of
132,843 tokens (602,765 characters) of gear names repeated, 2 h 52 min, cut off by the measuring script's three-hour
limit with the context at 134,707. At that rate it would have run to the context in about five and a half hours
(*arithmetic*). The other 125 turns of the run ended on their own. Not known: how often, on which models, whether the
two declines lead to it, and whether llama.cpp's sampling of the same model loops on the same prompt. Data
`~/mila-work/phase3/full/tasks.jsonl` and `store-tasks` on the Mac.

Again the same day on the PC's RTX 4070 (the session's default card, unpinned), Gemma 4 12B **FP4** (not the format it
was trained in) at automatic context with the one general prompt of AssistantInterface.md 5.5: after `request_folder` was answered, the reply was "IsNullOrThought", one a line, until the
driving script's ten-minute limit; the conversation stored only the question. The same question on Q4_0, twice, was
answered correctly. Not known whether FP4 loops more often. Script `D:\Claude\phase3\realfolder\drive.mjs`.

A third time on 2026-11-21 checking AssistantInterface.md phase 6 on the PC (Gemma 4 12B FP4, `0.21.0-dev+160` plus
phase 6): the round after a `find_records` result ran over seven minutes at 100% GPU and was stopped by hand; the
script printed only answer text, so what the round wrote is not known, and the chat stored only the question. The same
turn run again straight after ended in four rounds. Script `D:\Claude\phase6\real.mjs` (now prints progress and stops a
round past 4,000 tokens).

## A model's licence on Mila Assistant's first-run card is a name, with no way to read its text

`Apps/Assistant/Ui/views/transcript.js` (`renderSuggestion`, *Licence:*), `Apps/Assistant/Session/Src/Assistant.Models.ixx` (`publishedModels`, `license`) @ 0.21.0-dev+146

Found 2026-11-15 building the first-run setup (AssistantOnboarding.md 4.1, question 16, which Todd took): the card's
licence name was to become a link that opens the text, but `models.list` carries the manifest's licence name only,
not where its text is published, so the page has nothing to open. Whether *Install* then says the person accepts it
is counsel's and Todd's.

## Mila Assistant's web search has no test, and nothing lets one give it a scripted server

`Apps/Assistant/Session/Src/Tools/Assistant.WebSearch.ixx` (`search`, `kEndpoint`), `Lib/Src/Agents/Grants/ToolGrant.ixx` (`transport()`); an absence -- no test names `WebSearch` @ 0.21.0-dev+145

Found 2026-11-15 moving web search onto `HttpClient` (Network.md phase 3): its request, its status wording and the
new 1 MB cut run untested. `search` reaches the network only through the grant's own checked transport, which
connects to public addresses alone, and the endpoint is a constant, so no test can answer it from loopback or a
scripted transport. The page reader is tested through `read( arguments, transport )`; web search has no such seam.

## A token loader test sets a setting the loader no longer reads

`Lib/Tests/Data/Loaders/TokenSequenceLoader.cpp` (`CustomConfig_VerboseLogging`, `config.max_queue_size = 5`), `Lib/Src/Data/Loaders/TokenSequenceLoader.Config.ixx` (`max_queue_size`) @ 0.21.0-dev+145

Found 2026-11-15 building on the Mac: Clang warns that `max_queue_size` is deprecated (the queue gave way to double
buffering), the only warning in an all-targets Mac build. MSVC says nothing. Either the field can go, or the test
is setting it for no effect. Not looked into further.

## A local hub's files are opened by a narrow path on Windows, and the warning that would say so is not silenced

`Lib/Src/Distribution/Implementation/LocalDirectoryHub.cpp` (`openForReading`) @ 0.21.0-dev+144

Found 2026-11-15 building the Networking move: MSVC 14.52 reports C4996 for the `std::fopen` there despite the
`#pragma warning( suppress : 4996 )` above it, the only warning in an all-targets build. The call opens
`path.string()`, which on Windows is the path in the ANSI code page, so a hub folder whose path holds a character
outside it would not open. Not tried.

## Mila Assistant's first run on the Mac installs from a local folder, not the publisher

`Lib/Src/Distribution/Hub/LocalDirectoryHub.ixx` (`MILA_HUB_DIRECTORY`) @ 0.21.0-dev+142

A workaround Todd chose, 2026-11-13: until evaluations exist and the published models are republished after 0.21 development (the two entries below), the first run is shown with `MILA_HUB_DIRECTORY` naming a folder that holds one locally packaged model, `gemma-4-12b-it-q4_0` with its new card (`Lib/Tools/ExportArtifact/ModelCards/gemma-4-12b-it-q4_0/`), so it is the only model the planner can place ("we do not need to load from HF for this pass"). The suggestion's ranking is unchanged; nothing is faked in it. Comes out when the publisher carries a model the Mac can run. Folder on the Mac: `~/mila-work/hub`. Widened 2026-11-24 (Todd: "A local 'test' store would be much better"): the onboarding of rounds 8 and 9 is developed on the PC against `Data/Hub` and a test person's Mila folder, `Data/TestUser`, both gitignored, its package carrying the `onboarding` mark; `gemma-4-12b-it-q4_0` is to be published before 0.21 (`ModelDistribution.md`, *A model replaced by another*).

## The published Gemma 4 12B fp4 cannot be loaded by this version of Mila

`Lib/Src/Dnn/Models/Gemma/GemmaModel.ixx` (`requireStoredTableMatches`) @ 0.21.0-dev+142

Found 2026-11-13 planning the published list on the Mac (`AssistantOnboarding.md`). `mila-llm/gemma-4-12b-it-fp4`'s weights header (digest d49c6c16...) stores `temb.wte` as F8_E4M3; since 0.21.0-dev+31 a quantized Gemma body is read only with the table as INT6 per 32, and the FP8 table is refused. The break was decided: `+31`'s commit is BREAKING for "quantized Gemma packages exported before 0.21.0-dev+31", with re-export or "the republished package" as the remedy, so the model was to be republished before 0.21.0 ships. `ModelDistribution.md` *Compatibility* (decided 2026-10-03, the same day) asks that a format a published minor wrote be read by the next minor; whether `+31` is the exception to it is not written down. Header kept in `~/mila-work/sweep/onboarding142/store/headers` on the Mac. Nothing is republished until 0.21 development is finished and the release backlog run down (Todd, 2026-11-13). Decided 2026-11-24 (Todd): the hub keeps one Gemma 4 12B; the FP4 is made private at 0.21's publish, and the Q4_0 replaces it (`ModelDistribution.md`, *A model replaced by another*). Still open there: whether 0.21 reads a 0.20 FP4 on a person's disk or refuses it with the Q4_0's install named.

## The published Llama manifests do not record which Llama they are, and this version of Mila refuses them

`Lib/Src/Deployment/Implementation/ModelHandle.cpp` (`check`, `ChatTemplateNotRecorded`) @ 0.21.0-dev+142

Found 2026-11-13, same run. `mila-llm/Llama-3.1-8B-Instruct-fp4` and `Llama-3.2-3B-Instruct-fp4` publish manifests with no `architecture_version`, which a Llama needs to choose its chat template; an install copies the empty value into the record, so the installed model is refused too, and the refusal's advice (remove it and install it again) cannot help. Same *Compatibility* rule as the entry above: the fix is republishing the manifests, or a reading of the 0.20 manifests that does not guess the version from the name. Nothing is republished until 0.21 development is finished and the release backlog run down (Todd, 2026-11-13).

## The turn after a long thinking reply waits while the reply's answer is read again

`Lib/Src/Dnn/Models/Gemma/GemmaModel.ixx` (prompt prefix reuse) @ 0.21.0-dev+138

Found 2026-11-13 in Mila Assistant on the Mac, Gemma 4 12B Q4_0, thinking on: the turn after a 2,048-token reply took 11.3 s to its first token, every other turn 1.17 to 1.36 s (the conversation's stored stats, timed from the round's start). Gemma 4's template leaves a reply's thinking out of the history, so the cached prompt stops matching where that reply began and its answer, about 1,400 tokens, is read again. Not checked: Qwen 3.8's template, which also drops past thinking, and what llama.cpp does with the same conversation. Logs `D:\Claude\mac\desktop138`.

## Three Gemma 4 26B tests fail on the PC because their BF16 weights file is 32 KB

`Data/models/gemma/gemma4_26b_a4b_it_qat_bf16.bin` (test data, gitignored) @ 0.21.0-dev+138

Found 2026-11-13 running the full suite in `x64-claude-verify`: `GemmaMixtureOfExpertsLoadCudaTests` (both loads) and `GemmaMixtureOfExpertsParityCudaTests.LayerStream_MatchesHuggingFaceReference` throw "Failed to read rank of 'tf_layer_24.ffn.mlp.fc_down.weight'". The file is 32,734 bytes, dated 2026-10-19; the worklog noted it absent at +76. Who wrote the stub is not known.

## Mila's prompt on a Mac is about 250 ms slower after 30 s of idle, its memory wired or not

`Lib/Src/Dnn/Compute/Devices/Metal/MetalExecutionContext.ixx` @ 0.21.0-dev+137

Found 2026-11-12 measuring residency (`Metal/MetalBackend.md` 5, W3a). Gemma 4 12B Q4_0 at 32K, Safari and Xcode open: a prompt at once after the first took 637 to 669 ms, and after 30 s of idle 885 to 1012 ms, with the weights and cache held wired by a renewed residency set or not. Not known: where the time goes (host encoding, GPU clocks, a lazily rebuilt object, the console path). LM Studio's time to first token stayed at 261 to 284 ms on the same schedule, but through its server with a different request, template and reply cap, against mila-chat driven through pipes: not a comparison. Like for like is MIS against LM Studio's server, the same request, one streaming client. Logs `D:\Claude\metal\sweep\switch` and `lmstudio`.

## On a 16 GB Mac, Gemma 4 12B's first prompt at automatic context waits 1 to 8 seconds for macOS to free memory

`Lib/Src/Dnn/Compute/Devices/Metal/MetalDevice.ixx` (`getMemoryInfo`) @ 0.21.0-dev+135

Found 2026-11-11 timing mila-chat's first reply. Automatic context takes 262,144, the model then holds 12.8 GB, and `getMemoryInfo` offers the planner the whole of `recommendedMaxWorkingSetSize` (12.71 GB) as though nothing else on the Mac needed memory. Metal stops holding idle buffers in place about 2 s after their last use, so between the load and the first prompt the weights become ordinary memory; the first prompt then pins 12.8 GB with almost nothing free, and macOS compresses 2 to 9 GB (part of it the weights) and swaps up to 0.9 GB before it can. First prefill of "Hello": 948 to 8423 ms at 262,144, 652 to 1154 ms at 8192. Since `0.21.0-dev+137` the load leaves no copy of the weights in the file cache (11 s to 1.7 s), and a trial of that read at +135 left this unchanged. Not measured: a prompt after the model has sat idle, and what context the Mac's real free memory would allow. Scripts and logs `D:\Claude\metal\first-prompt`. Measured since, 2026-11-12: idle, other applications, a renewed residency set and coming back after another app (`Metal/MetalBackend.md` 5, W3a and W3b; `D:\Claude\metal\sweep`). At `0.21.0-dev+138`, 2026-11-13, in the packaged Mila Assistant (its settings at `auto`): the plan came to 12,250,639,620 bytes, and the lock while Mila is in front held 11.4 to 11.5 GB of it before macOS refused the rest at the system's lock limit, leaving 0.06 GiB free. Each time Safari was in front macOS compressed 3 to 4 GiB of the let-go model, so each return re-read it. Logs `D:\Claude\mac\desktop138`.

## Nobody has measured what the first load of a model on a Mac pays to prepare its kernels

`Lib/Src/Dnn/Compute/Devices/Metal/Implementation/MetalKernelLibrary.cpp` (`pipeline`) @ 0.21.0-dev+134

Found 2026-11-11 moving the kernels' compile into the build (`MetalBackend.md` 4.3). The first time the Mac saw the new library, preparing all 217 pipelines took 12.4 s; every run after took 7 ms, and after one kernel changed, 72 ms. A model prepares only its own kernels, when its operations are constructed, and that first load was not timed: every kernel was already prepared by the time a model ran. Not known either: whether a Mac that compiled the same kernels from source at run time paid the same, since those readings may all have found pipelines already prepared.

## Sampling's Metal kernel draws two compiler warnings in every build

`Lib/Src/Dnn/Compute/Devices/Metal/Operations/Sampling/Kernels/Sampling.metal`:703 (`shared_base`, `shared_start`) @ 0.21.0-dev+134

Found 2026-11-11, once the build compiled the kernels and printed what the run-time compile had kept to itself. Both are threadgroup variables written by thread 0 and read by every thread after a barrier, reported as used uninitialized whenever the `if` is false (`-Wsometimes-uninitialized`). Not looked into: whether the read is always after the write, and how to say so to the compiler. With them gone the kernels could build with warnings as errors.

## Metal's kernel files share one scope, and all but one leave their macros defined

`Lib/Src/Dnn/Compute/Devices/Metal/Operations/Linear/Kernels/LinearQuantized.metal` (`MILA_QUANTIZED_LANES` and 12 more across the kernel files) @ 0.21.0-dev+134

Found 2026-11-11 reviewing the kernel library. The 23 kernel files compile as one source (`MetalBackend.md` 4.3), each repeating `#include <metal_stdlib>` and `using namespace metal;` as if it were alone. Every struct, constant and helper is in one global scope, kept apart only by an operation's prefix; no collision exists today. Only `Moe.metal` undefines its macros; the other 13 `MILA_` macros stay defined for every file after. A collision stops a Mac build.

## On a 16 GB Mac, Gemma 4 12B planned at 98.5% of the working set loads and then fails its first prompt for want of GPU memory

`Lib/Src/Deployment/Planning/PrefillScoreRowsRule.ixx` (`choosePrefillScoreRows`) @ 0.21.0-dev+133

Found 2026-11-10 checking the plan of the build with prefill score rows (`Specifications/Metal/MetalGlobalAttentionPrefill.md` fact 14). Q4_0, FP16, context 262,144: 11.408 GiB of an 11.840 GiB working set (96.4%) runs; with a 32-row score block, 11.658 GiB (98.5%), the load succeeds, device allocation equals the plan, and the first prefill of 512 fails with `kIOGPUCommandBufferCallbackErrorOutOfMemory`. Reproduced twice with the +131 binary run before, between and after. Not found: what the run needs beyond the plan, and whether a fused plan at 98.5% fails the same way (BACKLOG's Mac item says the planner takes a context at 99% of the working set and the model runs, measured on Llama). Worked around in the rule, which takes a block only where it fits twice over; that margin goes when the cost is priced.

## A small Gemma network's weights are priced 9,216 bytes under what it allocates on a Mac

`Lib/Tests/Dnn/Components/Transformers/Gemma/Gemma.PrefillScoreRows.Metal.cpp` @ 0.21.0-dev+133

Found 2026-11-10 writing the test of a build with a block of prefill score rows. The tiny Gemma network of `Lib/Tests/Common/TinyDecodeNetworks.h` (FP16, unquantized, context 256, chunk 64), priced by `getRequiredMemory` at the device's allocation granularity, reports 6,029,312 parameter bytes and allocates 6,038,528, with or without a block; its state and scratch match to the byte. Not looked into: which parameter, and whether a published model's plan carries the same gap. The test compares state and scratch only and says so.

## On a Mac, a sliding layer that prefilled unfused would be given score buffers narrower than the op strides them by

`Lib/Src/Dnn/Components/Transformers/Gemma/Gemma.ixx` (`prefillScoreWidth`) @ 0.21.0-dev+133

Found 2026-11-10 reading the unfused prefill for the block loop. `prefillScoreWidth` sizes the score buffers to the ring (window + chunk - 1) when only the sliding layers miss the fused kernel, which is what CUDA's op reads; `MetalGqaOp` strides its score rows by the context whatever the cache holds. No model reaches it: Metal fuses a head of 256 at every precision it runs. Since +133 the op refuses score buffers narrower than the context (`scoreBlockRows`) where it would have written past them, so the gap is a refusal and not a wrong answer.

## On a Mac under memory pressure, the first prompt timed after a held one reads up to a fifth slow, and the kernels' time does not

`Lib/Tools/MetalModelTiming/ModelTiming.cpp` (`depthRates`) @ 0.21.0-dev+131

Found 2026-11-10 measuring Gemma 4 12B Q4_0 beside llama.cpp (`Specifications/Notebooks/MetalRates.md` 7.1). 512 tokens after 4096 held read 133.0 tokens a second in eight readings of four loads with 0.5 GB of swap in use; with 1.1 GB in use, after a full-context planner load, the first timed reading of a load was 106.5 and 117.0 and the next three 133.0. The profiled GPU time was the same throughout, so the time is on the host, where is not known. The session of 2026-11-09 that gave 0.89 and 0.75 of llama.cpp read 140 and 106 to 110 in every reading, with 0.76 GB in use. A comparison takes several readings a cell and says how much swap was in use.

## On a Mac, an FP8 KV cache is widened to FP16 in a separate pass before prefill attention, where CUDA widens it inside the tile

`Lib/Src/Dnn/Compute/Devices/Metal/Operations/Attention/GQA/MetalGqaOp.ixx` (`gqa_expand_fp8_rows`) @ 0.21.0-dev+131

Found 2026-11-10 while checking which cache Gemma 4 12B ran with on the Mac (it was uncompressed FP16: nothing requests FP8, `Quantization.md` KV decision 5). `Quantization.md` KV decision 2 says the flash kernels read FP8 directly with no transient 16-bit buffers, and CUDA does; Metal's fused prefill expands each KV head's K and V into FP16 in the context's scratch first (`MetalBackend.md` 4.5, "twice what the unfused pipeline expanded"). Contradicts decision 2 and backend parity (`MetalBackend.md` 4.5 decision). Metal's FP8 prefill rate is unmeasured; CUDA's FP8 prefill penalty and its fix are in `Quantization.md` *The FP8 prefill penalty*.

## A TensorOp matrix product scoped to one SIMD-group gave wrong values past 128 channels on the M1 Pro, with no error

`Lib/Src/Dnn/Compute/Devices/Metal/Operations/Attention/GQA/Kernels/GroupedQueryAttention.Flash.metal` @ 0.21.0-dev+128

Found building the fused prefill, 2026-11-08 (`Specifications/Metal/MetalBackend.md` 4.5): `matmul2d` at `execution_simdgroup`, Q x K^T with K = 256 or 512 and P x V with N = 256 or 512 in the same kernel, FP16 and BF16, returned outputs 0.7 to 1.3 relative error from the reference, nothing reported; at 128 they matched. Summing over 128-channel slices fixed every geometry. Not isolated: which product, K or N, or whether wider scopes (`execution_simdgroups<N>`, which `gqa_value_tiles` uses at N = 16) have the same edge. Apple's MSL specification and `MPPTensorOpsMatMul2d.h` (macOS 27.2 SDK) state no such limit. Untested elsewhere than the M1 Pro.

## Metal's one-row FP4 and codebook products shift every code, and read their weights well under the M1 Pro's 200 GB/s

`Lib/Src/Dnn/Compute/Devices/Metal/Operations/Linear/Kernels/LinearQuantized.metal` (`linear_quantized_lanes`, `linear_codebook_lanes`) @ 0.21.0-dev+134

Q4_0 and six bits left the lanes for a kernel that masks its codes in place (`Specifications/Notebooks/MetalMatrixPath.md` 7.1, 7.2), and Q4_0 now runs within 2 to 7% of reading its bytes. FP4's one row still reads about 138 GB/s on Llama 3.1 8B (`MetalRates.md` 7), slower in the one-row shape than on its lanes with its codes decoded by arithmetic or by a table; the codebooks' lanes are unmeasured on a model.

## Gemma 4 12B's six-bit head reads at about 140 GB/s in a decode step, where the same kernel alone reads 164

`Lib/Src/Dnn/Compute/Devices/Metal/Operations/Linear/Kernels/LinearQuantized.metal` (`linear_quantized_row`, `Int6Row`) @ 0.21.0-dev+134

Found 2026-11-11 (`Specifications/Notebooks/MetalRates.md` 7.4): 5.39 ms a token for 755 MB in the model's profile; 1.245 ms for a quarter of the outputs alone, in a form that held the activations as vectors. Not separated: the form, the full head's size, or what runs before it. A wide RMSNorm launch likewise takes 7.3 us in the model against 4.1 alone.

## About 510 small kernel launches take 5.4 ms of Gemma 4 12B's 47 ms decode step on a Mac

`Lib/Src/Dnn/Compute/Devices/Metal/` (`split_columns`, `rope_decode`, `gqa_decode_merge`, `residual`, `geglu`, `scale`, `gqa_write_decode`) @ 0.21.0-dev+134

Found 2026-11-11 (`Specifications/Notebooks/MetalRates.md` 7.4): each already gives a thread one value, and a launch that does nothing costs 2.2 us, so only fewer launches would move them: CUDA's fusions, not yet on Metal. RoPE's decode takes 16 us a launch and the sliding layers' merge 20, neither looked at.

## Nobody has checked whether CUDA's one-row Q4_0 product is limited by arithmetic, as Metal's was

`Lib/Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/Int4/CudaInt4Gemm.cu`, `Kernels/DecodeRows/CudaDecodeRows.cu` @ 0.21.0-dev+127

Raised by Todd, 2026-11-07, after Metal's Q4_0 lanes were found limited by arithmetic. CUDA's path is a different design: it quantizes the activations to INT8 and multiplies on integer MMA. What an Nsight Compute profile of Gemma 4 12B's decode products on the 5060 Ti would say, memory- or compute-bound and at what share of the card's bandwidth, has not been looked at.

## A top-p test on 4099 FP16 logits disagreed with the reference on two draws, under two different Metal kernels

`Lib/Tests/Dnn/Samplers/Sampling.Metal.cpp`, logits `randomLogits( 4099, 90 )` with one at -30000, `top_p = 0.9` or `top_k = 40` @ 0.21.0-dev+127

Seen building the multi-threadgroup sampler, 2026-11-07: draws at uniforms 0.789 and 0.930 gave tokens 3330 and 3926 against the reference's 3328 and 3923, both with a histogram cut on float bin edges and with the radix cut on keys, so not the cut's arithmetic. Suspected, not shown: ties among 4099 FP16 logits at the nucleus edge, which the kernels keep whole and the test's reference splits by token order. Which of the two settings failed was not split. The test was replaced by ladders of distinct values.

## Mila's Metal rates on Llama 3.2 3B fell 5 to 8% between +122 and +127

`mila-metal-model-timing`, Llama 3.2 3B BF16 weights at FP16, M1 Pro @ 0.21.0-dev+127

2026-11-07, same tool and shapes as `Notebooks/MetalFirstModel.md` section 4: prefill 512 at 511 tokens/s (was 555), decode 19.1 (19.4), decode after 4096 17.5 (18.65 after the attention rebuild). llama.cpp b12138 on the same Mac the same day held or rose (627 / 25.3 / 23.3 against 649 / 24.3 / 22.4), so the Mac is not the cause. Spreads are within 3%. Not bisected. At `+130` (2026-11-09, `Notebooks/MetalRates.md` 7) decode is 20.88 and after 4096 18.88, above `+122`; prefill 512 is still 512.6.

## A session that fails as it starts hangs silently instead of saying why

`Apps/Assistant/Session/Src/Assistant.SessionProcess.ixx` (`run`, the `reader` jthread) @ 0.21.0-dev+126

Found running the packaged session on the Mac, 2026-11-07: it found its prompts against the working directory (fixed the same day in `main.cpp`'s `executable_directory`), so `useActivity( "ask" )` threw before the request loop. The exception left `run` while the reader thread was blocked reading standard input, and the `jthread`'s destructor joined it, so the process answered nothing and printed nothing until its input closed; `main`'s "Fatal error" is reached only then. A stack sample shows the main thread in `~jthread` -> `join` and the reader in `getline`. Not Mac-specific: any throw before the loop does this.

## An edit to a low-level module rebuilds about 530 test targets, which makes a forced failure there cost minutes

`Lib/Src/Dnn/Compute/Registry/DeviceRegistryHelpers.ixx`, `Devices/Device.ixx`; `Lib/Tests` (`import Mila;`) @ 0.21.0-dev+125

Found running change 1's forced failures (`Specifications/Notebooks/DeviceAssumptions.md`): each mutant of a device module rebuilt about 530 `MilaTests` targets, so eleven mutants were estimated at 40 to 50 minutes, and the run was cut to the two that mattered most. Two causes, not yet split by measurement: a module's bodies live in its interface unit, so a body edit changes what every importer sees; and most test files `import Mila;`, so every test depends on every module. `ninja -t query` on the build would show how much is each. Todd, 2026-11-07: noted, moved on. **Direction agreed the same day (Todd):** a module that is not a template, not on a per-token path, and has real bodies or includes a backend header gets an implementation unit in an `Implementation/` folder beside it; value types, accessor classes and templates stay in one `.ixx`; existing code moves when a change touches it substantially. First case: `Agent` and `ModelHandle` in the device-blind handle change, with one body edit's rebuild timed before and after; the convention goes into CLAUDE.md with that number, or is dropped if the saving is not there. **Measured at +127:** an edit to `Agent::baseModelRefusal` rebuilt 263 steps in 268 s in the interface and 19 in 20 s in `Implementation/Agent.cpp`; the convention is in CLAUDE.md. What remains here is the other cause. Separately, tests importing only what they use: white-box tests yes, contract tests and apps stay on `import Mila;`, to be measured after the file-name pass.

## `TPrecision` is named for no one thing, and documented six ways where it is documented at all

`Lib/Src` (about 109 files declare `TensorDataType TPrecision`, about 38 carry a `@tparam` for it); `Specifications/OperationDispatch.md` line 40 @ 0.21.0-dev+125

The decision is `OperationDispatch.md`'s: the activations' element type and the compute precision, the operation owning anything it widens internally, and through `TWeight`'s default the element type of unquantized weights. The `@tparam` lines say "compute and activation precision", "logits precision", "tensor data precision for this component", "abstract tensor precision", "tensor element type and cuBLASLt compute type" and, in `CudaMoeOp`, "activation and weight precision". `Linear` calls the same parameter `TComputePrecision`. No rule asks for `@tparam` lines; CLAUDE.md's Doxygen rule is the file comment's. Todd, 2026-11-07: document it to that meaning and rename it for its context, after the device decisions.

## The ROCm device folder is a Vulkan placeholder, and a ROCm build cannot compile

`Lib/Src/Dnn/Compute/Devices/Rocm/RocmDevice.ixx`, `Registry/DeviceRegistrar.ixx:70`, `Lib/CMakeLists.txt:1268` @ 0.21.0-dev+125

`RocmDevice.ixx` declares `module Compute.VulkanDevice` and a `VulkanDevice` returning `DeviceType::Vulkan`, which `DeviceType` does not have; it is not in the ROCm file set. `DeviceRegistrar` calls `RocmDeviceRegistrar::registerDevices()`, which no file defines. `DeviceType.ixx` still marks Metal and ROCm `FUTURE`.

## Metal's quantized model gate compares two engines' noise, so FP8 misses CUDA's bar by chance

`Lib/Tests/Dnn/Models/LlaMa/LlamaModel.HuggingFaceReference.Metal.cpp` (`Fp8`, `Fp8QuantizedAtLoad`, `Fp8Cache`, fail); `Specifications/Notebooks/MetalQuantizedWeights.md` 2.3 @ 0.21.0-dev+123

The gate holds Metal to "no further from HuggingFace FP32 than CUDA's run of the same weights". Both carry the format's same rounding, so it compares two draws of each engine's small difference: FP8 and the FP8 cache miss on one number each by 1% to 15%, with every argmax and greedy token HuggingFace's, and FP4 passes. Holding Metal to CUDA's run directly fails on FP4, where CUDA's prefill rounds activations to FP8 (`kUseFp8ActivationPrefill`, inferred cause of Metal scoring the sequence 0.012 better). Decided (Todd, 2026-11-06): HuggingFace FP32 on each format's weights decoded to full precision is the reference, with Phase 4's bars (`Metal/MetalBackend.md` section 10). Gemma 4 12B is gated that way first (`Notebooks/MetalGemma.md` 1.1); Llama's FP8 and FP4 tests are not yet moved to it, so they still fail as above. The 8B's FP32 run needs the layer-streamed driver Gemma's capture uses.

The `Fp8Cache` likelihood difference against the bar of 0.0009, Llama 3.2 3B on the M1 Pro: 0.0010 in FP16 at `+126`; at `+128` unfused 0.0019 FP16 and 0.0025 BF16 (with BF16's logits at 0.0558 against 0.0530); with the fused prefill as tuned and committed (2026-11-08, `Metal/MetalBackend.md` 4.5) 0.0028 FP16 and 0.0015 BF16, BF16's logits 0.0563. Every argmax unchanged. The FP8 weight tests moved under 1% with the fused prefill. The BF16 `SequenceLogLikelihoodAndRanges` miss of `+128` passes with it.

## CUDA's quantize at load writes codes for weights that are not finite, where Metal's refuses the tensor

`Lib/Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/Quantization/CudaFp8WeightQuantization.cu` (`quantize_fp8_per_channel_kernel`), and the FP4 kernel beside it @ 0.21.0-dev+122

Found writing Metal's quantize to CUDA's bytes, from reading, not run. `fmaxf` passes over a NaN, so a row holding one gets the scale of its other weights and the NaN becomes E4M3's not-a-number code; an infinity makes the scale infinite and every code of its row a NaN. Nothing is refused and nothing is logged, so a damaged checkpoint quantizes into a model that runs and is wrong. Metal counts such weights and refuses the tensor by name (`MetalLinearOp::quantize`, `Notebooks/MetalQuantizedWeights.md` 2.1), as its BF16-to-FP16 load already refuses a value FP16 cannot hold.

## Llama 3.2 3B at BF16 on Metal scores a sequence further from HuggingFace than either accepted BF16 run

`Lib/Tests/Dnn/Models/LlaMa/LlamaModel.HuggingFaceReference.Metal.cpp` (`Bf16.SequenceLogLikelihoodAndRanges`, fails); `Specifications/Notebooks/MetalFirstModel.md` 2 @ 0.21.0-dev+121

Mean negative log-likelihood over 1024 wikitext tokens: +1.25e-3 from HuggingFace FP32, against a bar of 8.97e-4, the further of HuggingFace's own BF16 (+9.0e-4) and Mila's CUDA BF16 (-5.5e-4). Its logits and greedy pass, at the same distance as the other BF16 runs, and FP16 on the same Mac is within 6.8e-5. The M1 selects FP16, so no model runs this path today. Not looked for: which operation loses it. A first step would compare per-layer outputs against CUDA BF16 on the same tokens.

## Every network chooses its load staging by `#ifdef MILA_HAS_CUDA` though the device traits name it

`Lib/Src/Dnn/Components/Transformers/LlaMa/Llama.ixx` (`loadParameters`); the same in `Gemma.ixx`, `Gemma.Drafter.ixx`, `Qwen.ixx`, `GptTransformer.ixx` @ 0.21.0-dev+120

Each streams its weights through `CudaPinnedMemoryResource` under `#ifdef MILA_HAS_CUDA` and `if constexpr ( Cuda )`, and through `CpuMemoryResource` otherwise. `DeviceTypeTraits<D>::host_staging_memory_resource` already names the staging resource per device (Cuda pinned, Cpu and Metal their own), and `TokenEmbedding` and `Component` use it. `streamTensorBlobs` takes a device index only for pinned memory, which is why the branch exists. Found running Llama on Metal, where the CPU branch serves and Metal's own staging resource goes unused.

## An agent cannot start a process and keep working while it runs

`Lib/Src/Agents/Programs/ProgramContainer.ixx`; `Lib/Src/Agents/Grants/ToolGrant.ixx` (`runProgram`); `Specifications/ContainedPrograms.md` 4 @ 0.21.0-dev+119

Todd, 2026-11-05: the case that matters is long-lived, an agent running a process across its tool calls. `ProgramContainer::run` starts a program and waits for it, and nothing it starts outlives the run, so a dev server cannot stay up while later tool calls test against it, except inside one command. The model sees output only when the run ends, and nothing can be typed into a running program. `run_command` caps a run at 600 seconds. Under the wait, `MxcProgramContainer` already uses `mxc_spawn_json`, reads each output stream on its own thread and polls `mxc_sandbox_try_wait` (`ContainedPrograms.md` 4.8), so MXC gives what a handle would stand on. Not designed: a handle that lasts across calls, how long it may live, when the person is asked, and how a model reads its output and stops it. Windows only, as all of the container is.

## Rope's angle loses accuracy far into a context, on both backends

`Lib/Src/Dnn/Compute/Devices/Cuda/Operations/Encodings/Rope/Kernels/Rope.Angle.cuh`; `Lib/Src/Dnn/Compute/Devices/Metal/Operations/Encodings/Rope/Kernels/Rope.metal` @ 0.21.0-dev+120

Both compute position times frequency in single precision before taking the cosine and sine. Arithmetic, not measured: at position 8192 and a frequency of 1, the angle's own rounding is about 5e-4 radians, and at 128000 about 8e-3. Found writing Metal's Rope test against a double-precision reference, which holds the bar only to position 2048. Reducing the angle before the multiplication would remove it.

## The token loader's test hung the full suite for two and a half hours

`Lib/Tests/Data/Loaders/TokenSequenceLoader.cpp` (`Reset_Functionality`); `Lib/Src/Data/Loaders/TokenSequenceLoader.ixx` (`~TokenSequenceLoader`, `producerThreadFunc`) @ 73e4c0ad

Seen 2026-11-04 in a full suite on the compaction build: the log stopped at this test at 15:56 and the process used no CPU when looked at, 2.5 hours later. It passed in 190 ms in the run before, and 123 repeats alone passed, cut off by the machine's reboot. Read, not proven: the destructor sets `stop_` without holding `mutex_`, so it can fall between the producer's check of `stop_` and its untimed `cv_producer_.wait`; the notify is lost, and `join` never returns. The test destroys the loader just as the producer reaches that wait. The likely fix is to set `stop_` under the lock.

## Compaction summarizes old tool results that could be cut without asking the model

`Lib/Src/Agents/Agent.ixx` `compactConversation` · `Specifications/Agents/Compaction.md` section 4 @ 0.21.0-dev+116

Tool results are most of the tokens in the sessions measured (`Notebooks/CompactionSummaries.md`), and every one but the latest
goes to the model to summarize. Hermes first replaces long old outputs with a stub, with no model call
(`agent/context_compressor.py`, read in part). A step here would sit after clearing reasoning and before the summary: if it
brings the prompt below the trigger, no summary is written, which on Qwen 3.8 could replace minutes of generation. Against it:
it changes the prompt from the first result cut, so what follows is read again, from zero on Qwen; and whether a model reads a
file again after losing it is unmeasured. Measured on the captured sessions first, as its own change (Todd, 2026-11-03: file it).
Beside it: Codex keeps recent user messages word for word along with its summary (`codex-rs/core/src/compact.rs`), where here
only the message being answered and the policy's `kept` list are sure to survive.

## An agent does not know how much of its context it has left

`Lib/Src/Agents/Agent.ixx` `addResult` · `Apps/Server/src/mila_llm_server/conversation.py`, `InferenceServerWire.md` 3.6 @ 0.21.0-dev+116

The model is told nothing of its budget, so it cannot plan around it: whether to read a large file, or finish now. Claude Code 2.1.300
sends `<total_tokens>N tokens left</total_tokens>` after every tool result. The agent could add one line to a result as it renders:
no tool, which would cost a round and be asked only when the model already suspects a problem; no new turn; and only the end of the
prompt changes, so the cached prefix holds. The number should be the reliable depth, not the context limit. It informs and never
enforces: the runtime keeps every limit. Unmeasured whether a model uses it or rushes because of it, every round or only at
thresholds; no evidence found beyond blogs. Measured on the captured sessions, with the pruning entry above (Todd, 2026-11-03: file it).
Master and worker streams would need this and more (`.internal/Ideas/AgentStreams.md`).

## The inference server's streamed chat reply arrives whole, after the model has finished

`Apps/Server/src/mila_llm_server/routes/factory.py` `_stream_chat` · `Specifications/InferenceServerWire.md` 3.5, Phase 4 @ 0.21.0-dev+116

Every chat API, asked to stream, sends an opening chunk and keepalives while the model works, then the reply in one piece
once the turn ends (found by reading it; Todd, 2026-11-03: "It should" stream). So a client shows nothing until the end,
however long the reply, and cannot time a prefill from its first token. Dated `+68`. The spec plans the fix as Phase 4 and
recommends it for v0.22.0, and nothing in `BACKLOG.md` or `Issues/` carried it. The reason it gives, that the parse runs
after the model finishes, no longer holds in the library: `Conversation::ReplyReader` has routed reasoning, answer and calls
as they arrive since Chat moved onto it. The binding does not expose the reader, and it knows no consumer, so that is
where the fix starts.

## Zeroing a view on CUDA writes past the view, and can run past its buffer

`Lib/Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/CudaTensorOps.Zero.ixx:69` @ 0.21.0-dev+117

`bytes` is `getStorageSize()`, the whole buffer's, and `dst` is `rawData()`, which starts at the view's offset. So a view at offset k zeroes k elements past the buffer's end. Found by reading, while the same line failed a test on Metal. Not run on CUDA, and whether any caller zeroes a view has not been looked at. `Tensor::getStorageSize` has other callers, none checked for views.

## Host accessors on a pinned or managed FP16 or BF16 tensor read elements as `float`

`Lib/Src/Dnn/Tensors/Tensor.ixx:351` `item`, `:386` `operator[]`, `:509` `data`; `TensorHostTypeMap.ixx:80` @ 0.21.0-dev+117

`TensorHostTypeMap` maps FP16 and BF16 to `float`, and the accessors cast the buffer to it, so element i reads 4 bytes at 4i of a 2-byte array. Found by reading while deciding that a Metal tensor has no host accessors (`MetalBackend.md` 5). Not run on CUDA, and which callers use these on a reduced-precision tensor has not been looked at. `Tensor::outputBuffer` uses the same map.

## Clang warns on two operations' `setParameters` and on `TensorOps.Random` reaching a TU-local entity

`Lib/Src/Dnn/Compute/Devices/Cuda/Operations/Convolutions/CudaCausalConv1dOp.ixx:58`, `.../DeltaNet/CudaGatedDeltaRuleOp.ixx:64`, `Lib/Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/CudaTensorOps.Random.ixx:266` @ 0.21.0-dev+114

Clang 22, a library build: `setParameters` overrides but is not marked `override` in both operations
(`-Winconsistent-missing-override`), and instantiating `fill_uniform` and `fill_normal` references the TU-local `cudaMalloc<float>`
and its `devPtr` and `size` (`-Wreference-tu-local-entity-in-other-tu`). 11 warnings, none from Agents. MSVC gives none of them.

## A pnpm project may not build under Mila Assistant's commands, which tell Node to keep links as named

`Apps/Assistant/Session/Src/Tools/Assistant.ProgramSetup.ixx` (`NODE_OPTIONS`) @ 7cf45bdd

Measured 2026-10-31 (`Notebooks/MxcConformance.md` 7.1): in MXC's container Node's resolving of a script's real
path stats the drive's root, which is refused, so every `node` and `npm` run failed; `--preserve-symlinks
--preserve-symlinks-main` lets them run. pnpm lays `node_modules` out as links into its store and relies on real
paths; a pnpm project under those options is not measured. `dotnet build`'s first run in the container also reports
installing a development certificate; where it lands is not measured.

## A command run through the grant reaches every public host or none, never only the hosts a project needs

`Specifications/ContainedPrograms.md` decision 11; an absence -- no allow list by host name @ 7cf45bdd

Todd, 2026-10-31: after phase 3 (network per call), a list of hosts allowed per project in Mila Assistant, and a
question for each new host. MXC's egress rules match address, protocol and port only; names are out of its scope
(`docs/backends/process-container/networking.md`). Its proxy mode (`runtimeConfig.networkProxy`, "Model 2") sends
WinHTTP traffic and `HTTP_PROXY`/`HTTPS_PROXY` to a loopback proxy the host runs, and asks for `egress.default: deny`
with `ingress.default: allow`. Unmeasured: what that ingress opens, and which tools (npm, git, pip, curl, dotnet)
go through the proxy inside the container. The proxy would decide the grant's network, so it is `Lib/Src`.

Seen again 2026-11-18 (`Specifications/Notebooks/MilaVisionDiscussion.md` 8, G6): the question for such a command shows the
command and cannot show what its program will send, so it cannot meet the vision's "approve that specific thing".

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
(`ContainedPrograms.md` decision 8): a known gap for Mila Assistant, and told to Microsoft with the other two.

## On NTFS a contained run walks its folders whole before every run

`Lib/Src/Agents/Programs/SecondNameWatch.ixx` (`watcher_reports_every_name_`) @ 7b5248a2

Todd, 2026-10-31 (`ContainedPrograms.md` decision 7, narrowed): `ReadDirectoryChangesW` on NTFS reports no name made
outside for a file inside, so off ReFS every run walks the granted folders whole -- about 115 ms for 7,500 files,
5.5 s for 358,000 (`Notebooks/MxcConformance.md` section 7). NTFS's change journal records a link added
(`USN_REASON_HARD_LINK_CHANGE`); whether it can be read without an administrator is unmeasured.

## The grant refuses a file with a second name even when every name lies inside the folder

`Lib/Src/Agents/Grants/ToolGrant.ixx` (`requireOneName`) @ 7b5248a2

Todd, 2026-10-27 (`ContainedPrograms.md` decision 5): kept for v0.21.0. `FindFirstFileNameW` and
`FindNextFileNameW` list a file's names, so the grant could accept a file whose names all lie within granted
folders giving at least the access asked, as the container's walk does (`ContainedPrograms.md` 4.3). A project
whose files are linked between two of its own folders is refused today.

## Two installed models that think are recorded as not thinking, so neither app offers their switch

`mila-store show gemma-4-12b-it-fp4` and `mila-store show Qwen3.8-27B-fp4` in Todd's store @ e9c9430f

Seen 2026-10-27: both list their capabilities as `instruct` alone, while the store's `gemma-4-26B-A4B-it-q4_0` and
the scratch store's `gemma-4-12b-it-q4_0` list `thinking` too. `ModelCapabilities::thinking` is read from the
record, so `Agent::thinks()` is false for both whatever the template's switch says, and Chat and Mila Assistant show
no thinking switch for them. Likely packages installed before the manifest carried thinking; not checked against
what the hub publishes now.

## In Get Things Done, Gemma 4 12B looks at the folder before it proposes a plan about half the time

`D:\Claude\filetools\drive_tasks.py` (the session drive's "the first call proposes a plan" check); `Apps/Assistant/Prompts/tasks.json` (`Apps/Chat/Prompts/tasks.json` until +105) @ cc07f776

Measured 2026-10-27 on Gemma 4 12B Q4_0 at the publisher's sampling: the first call was `list_folder` in 2 of 3 runs
on `+103`'s session and 3 of 3 on the session change after it, `propose_plan` otherwise; the plan, its steps and the
edit ran correctly every time. The same model chose both ways on one build, so the check measures the model's choice,
not the session. On `+105`'s session (Mila Assistant on `Mila::Agents` alone, same prompt and tool schemas byte for byte):
`list_folder` first in 4 of 6 runs, then a plan carried out correctly; `propose_plan` first in 1; and once no plan
at all -- three edits, then an answer claiming a README change it never made. Whether the prompt should allow a look first, or the check should, is not decided.

## Nobody has read whether Tavily's terms let Mila Assistant keep search results in a saved conversation

`Apps/Assistant/Session/Src/Tools/Assistant.WebSearch.ixx`; an absence -- Tavily's terms of service not read @ da647c29

Found 2026-10-26 adding Tavily web search: a saved conversation keeps every tool result, search results included, and
Tavily's credits page says nothing on keeping or showing results. Brave grants that only on some plans.

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

`Lib/Tests/Agents/Agent.Cuda.cpp` (`askForTheWeather`); scratch-store package `Llama-3.1-8B-Instruct-fp4` @ efa24591

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

Todd, 2026-10-20, while choosing the toolkit for Chat's desktop window (`Specifications/Apps/MilaAssistant.md`
section 9): "We do have a future for .NET bindings." Like the Python binding: a projection of the C++ API
(`Lib/Bindings/`).

Specified 2026-10-21 in `Specifications/DotNetBinding.md`: P/Invoke over a C ABI, experimental until
v0.22.0 or later (Todd).

## Mila Assistant's download is unsigned, has no updates and does not say it needs an NVIDIA GPU

`Specifications/Apps/MilaAssistant.md` section 9.4; an absence -- no signing, no update channel, no stated
requirement @ c8f71f60

Todd, 2026-10-24: Mila Assistant is the end-user showcase. Since +101 its installer carries the session, the
CUDA libraries it loads and the Visual C++ runtime, so no toolkit is needed. Still missing: the NVIDIA GPU
requirement stated before the download, code signing (SmartScreen warns on an unsigned one), and updates.

## The installed Qwen 3.8 27B FP4 package does not say it can think or call tools

`%LOCALAPPDATA%\Mila\models\qwen3.8-27b-fp4.json` capabilities; an absence -- no `thinking`, no `tools` @ c8f71f60

Found 2026-10-24 driving Mila Assistant's session process: Todd's installed record declares `instruct` and `agentic`
only, so Chat and Mila Assistant run Qwen 3.8 27B FP4 with no reasoning switch and no effort levels, and the
console's status line has no Thinking clause (the phase 2 baseline shows the same). The fp4-cb3 package built
with `--thinking --tools` declares both. Not checked: whether the published package's `mila.json` or only this
install lacks them.

Also Gemma 4 12B FP4 (2026-10-24, Mila Assistant's window): Todd's `gemma-4-12b-it-fp4.json`, installed 2026-10-03,
has no `capabilities` block at all, so the session reports `thinking_capable: false` for a model that thinks.

## A tool call ran in Mila Assistant's window without the person being asked, once

`Apps/Assistant/Session/Src/Assistant.SessionProcess.ixx` `approveToolCall` @ a9087567

2026-10-24, the first scripted window run over the real session (Llama 3.2 3B FP4, tools-weather, scratch
store): the first turn's `get_weather` call ran and its result was stored, while the script clicked nothing; a
later call in a new conversation did ask. Not reproduced in five later runs (two window runs of the same
sequence with every page message recorded, two single-turn window runs, one over stdio): each asked. That run's
files were overwritten by the next, so what the page received is not known.

## A client that sends on `turn.end` is refused as busy

`Apps/Assistant/Session/Src/Assistant.SessionProcess.ixx` `respond` @ a9087567

`turn.end` is sent from inside the turn and `busy` clears after it, so a `turn.send` written on reading
`turn.end` gets "A reply is still being written" (seen 2026-10-24 from a stdio script). The page waits for
`session.state`, so it does not hit this; `MilaAssistant.md` 4.1 does not say which message a client waits for.

## The session sends a thought break when the model is not thinking

`Apps/Assistant/Session/Src/Assistant.SessionProcess.ixx` `thoughtBreak` @ a9087567

Gemma 4 12B Q4_0 with thinking off, 2026-10-24: `turn.thought_break` after the tool result, with no thought
text before or after it. `MilaAssistant.md` 4.1 says it means the reasoning resumed. The page shows nothing for it.

## Mila Assistant's Models view stays open after a load, and offers Load while one runs

`Apps/Assistant/Ui/views/models.js` @ a9087567

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

## No CI job builds Mila Assistant's installer

`.github/workflows/assistant.yml`; an absence -- the `windows-installer` job removed @ 793bddb5

Removed 2026-10-26 when the installer began carrying the session (`MilaAssistant.md` 9.4): `npm run build:windows`
refuses to run without `Window/session/` staged from a C++ build, and the app's workflow (`assistant.yml`) has no C++ build. Before
that the job's upload was an installer that ran only the mock. Nothing builds the installer now but a person
with a Release build directory (`Apps/Assistant/README.md`, *Staging it for the window*).

## A chat's preview of Mila's last reply can stop mid-word, with nothing to say it was cut

`Apps/Assistant/Session/Src/Implementation/ConversationSummary.cpp` (`flattened`, `kReplyStartBytes`) @ 0.21.0-dev+149

Found 2026-11-16 checking AssistantInterface.md phase 2 on the real session with Gemma 4 12B: `reply_start` is cut at
160 bytes, which kept a character whole but ended "... at every step. This a". A wide window shows the whole 160
bytes in *All chats*, the overview and the risen conversation's lists, so the cut reads as the reply's own words. The
page cannot tell a cut from a reply that short; the summary could end a cut one at a word, or mark it.
