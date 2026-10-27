# Inventory of Chat's Session Library

Every job `MilaChatSession` does, who uses it, and where it goes once an agent is a model and a session
(`Direction.md` 4.3; `BACKLOG.md` *The Desktop Agent*). Read 2026-10-27 at `0.21.0-dev+103`, from
`Mila/Apps/Chat/Src/` and the desktop session's imports of it. A notebook: the decisions it leads to are carried to
`Direction.md` 4.3, `MilaApp.md` and `ChatConfiguration.md`.

## 1. What the desktop app reaches

`mila-session` imports seven of Chat's modules. `Assistant.SessionProcess` uses `ChatSession` for nearly everything it
does with a model: load, install, remove, the turn, stop, history, the activity's prompt, limits, rules and grant,
thinking, effort, context, sampling, seed, detail and the round measurements. `main.cpp` resolves its settings with
Chat's `resolveChatConfig`, **from Chat's own files** ("Settings resolve as mila-chat's do, from the same files"), and
the session's prompts are built from `Mila/Apps/Chat/Prompts` (`Desktop/Session/CMakeLists.txt:31`). Six desktop
modules import `Chat.Json` only for `nlohmann::json`.

## 2. The inventory

Destination: **Library** is `Mila::AI` (a `Mila/Src` change, proposed before it is built); **Each app** is code each
application keeps for itself; **Delete** has a library answer already, or no use.

| Job | Where today | Used by | Destination |
|---|---|---|---|
| Hold the program's prompt, tools, rules, grant, asker, limits and conversation across a release (`tool_rules_`, `grant_`, `limits_`, `tool_handlers_`, `held_conversation_`) | `ChatSession` | both | **Library**: the session (4.3) |
| Release a model before planning the next; reload at a new context keeping the conversation; open with no model | `switchModel`, `reloadAtContext`, `setAutomaticContext`, `releaseModel` | both | **Library** for the order and for keeping the session; **each app** for its policy around a failure (Chat puts the old context back) |
| Refuse a name before anything is released: not installed, files missing, base model, template unknown, family not plannable, stored format unreadable, quantize-on-load not allowed | `resolveModel`, `requireChatTemplate` | both | **Library** for the check, returning a reason; **each app** for the words (`RefusalAudience` and its `mila-store` remedies are Chat's) |
| The deployment request: device, KV cache, a quantization asked for on load, the context fixed or automatic | `deploymentRequestFor`, `activeDeploymentRequest` | both | **Each app** sets its own; `DeploymentRequest` is already the library's |
| The automatic context's ceiling and a model's highest context | `FamilyTraits::max_context` | both | **Delete**: the planner caps at the model's trained maximum when the ceiling is 0 (`DeploymentRequest::withAutomaticContextLength`), and refuses a fixed context past it. See decision 1 |
| The context opened at when nothing names one | `FamilyTraits::default_context`, 4096 for every family | both | **Each app**, as one number: it no longer differs by family |
| Thinking on or off, and effort | `setThinking`, `setThinkingEffort`, `templateVariables`, `qwenEffortName`, `Qwen::reasoningEffortFromScale` | both | **Each app**, from the template's own variables and their listed values (`TemplateVariable::values`). The 1-5 scale mapped onto Qwen's enum is model-specific code and goes |
| Tools in the system message (`tools_in_user_message = false`) | `templateVariables` | both | **Each app**, by the variable's name, as now; the template's default stays the library's |
| Sampling: the user's knobs over the publisher's recommendation for the thinking in effect | `sessionSampling` | both | **Library**: `Agent::sampling()` already merges the publisher's; it takes only a whole `SamplingParams` override, so Chat repeats the merge. Proposed: an override per knob |
| A turn's stop source, made fresh each turn | `respond`, `stop` | both | **Each app**; the library takes a `std::stop_token` already |
| Agent events turned into the front end's events; the turn's display; the raw transcript for the console's stream check | `generateResponse`, `endRound`, `SessionEvents`, `TextChannel`, `ParsedResponse` | both | **Each app**: presentation. The desktop maps agent events to its protocol directly |
| Asking the person about a call | `askAboutCall`, `ToolDecision` | both | **Each app**, given the whole `ToolQuestion` (today the source, summary, needs and grant are dropped: `Untriaged.md`) |
| Round measurements: first token, decode rate, gaps, stalls, reasoning tokens | `RoundTimer`, `RoundStats` | both | **Each app**, by the standing rule that diagnostics live in the consumer (the comment on `RoundTimer` says so). See decision 4 |
| The context floor for `/context` | `contextFloor`, `ContextFloor`, `ThinkingEffort` budgets | both | **Each app**; `agent.prompt()` is the library's part already |
| Tools from a prompt file: `ToolDefinition`, `SystemPromptLoader`, `activeTools`, `agentTool`, `parametersJson` | `Chat.SystemPrompt`, `ChatSession` | both | **Each app** (decided: tools are the application's). Each builds `Tool` objects for the session; the store of handlers kept until a model loads goes, since the session holds tools before a model exists |
| Install, remove, list installed and published models, describe one, with progress and the fit column | `Chat.ModelCatalog` | both | **Each app** for the words and tables; the work is the library's already (`ModelResolver`, `ModelStore`, `InstalledModel`, `fittingWeightQuantizations`) |
| The model last chosen | `readLastChosenModel`, `writeLastChosenModel` (`chat-state.json`) | both, **one file** | **Each app**, in a file of its own |
| Layered settings: compiled defaults, files, flags, session overrides | `Chat.SettingsResolution`, `Chat.Settings`, `ChatConfig`, `ConfigError`, `ModelNotFound` | both, **Chat's files** | **Chat**. The desktop app gets settings of its own. See decision 2 |
| Seeding the sampler | `seed` | both | **Each app**, through `model()` as now |
| Detail level and log level | `setDetail`, `DetailLevel` | both | **Each app** |
| Device memory and the card's name for listings | `queryDeviceMemory`, `queryDeviceName` | Chat, desktop listing | **Each app** |
| Byte, size and time formatting; console colours | `formatBytes`, `formatDuration`, `Chat.Ansi` | both; colours Chat only | **Each app**; `Chat.Ansi` is console code inside the session library today |
| `nlohmann::json` under another name | `Json.ixx` (`Chat.Json`) | both | **Delete**: import `nlohmann.json` directly, as `Mila/Src/AI` does |
| `ModelType`, `QuantizationMode`, `KvCache` | aliases of `Dnn::ModelType`, `Dnn::WeightQuantization`, `Dnn::KvCacheCompression` | both | **Delete the aliases**, use the library's names; keyword parsing (`parseQuantization`, `parseKvCache`) stays in each app |
| `ModelSize` | `Chat.ModelSize` | nobody | **Delete**: re-exported by `Chat.Config` and named nowhere |
| A canned `get_weather` tool | desktop `main.cpp:221`, `tools.json`, `tools-weather.json` | desktop | **Delete** from the desktop app. See decision 5 |

## 3. What this corrects

- **Two of the "family-specific types" are not.** `Direction.md` 4.3 and the BACKLOG item first named `ModelType` and
  `QuantizationMode` among Chat's family-specific types (corrected the same day). Both are aliases of the library's own enums;
  `QuantizationMode` is a weight format and knows no family. The family-specific code is `FamilyTraits`, the Qwen
  effort mapping, and two `family == Gemma || Llama || Qwen` checks in `ModelCatalog` (`:446`, `:672`) that repeat
  what `requirePlannable` and the instruct capability already decide.
- **Round measurements stay in the applications.** On 2026-10-27 they were put to Todd as a likely `Mila::AI` gap;
  the standing rule, and the comment on `RoundTimer`, put them in the consumer.

## 4. Proposed for `Mila::AI`

Each a `Mila/Src` change, to be proposed in full before it is built.

1. **`Session` and the agent over it** (4.3): `Agent` created from a handle and a session; giving the session back
   releases the model; template variables and sampling the next model's template does not accept are dropped and
   reported.
2. **A check before a release.** Whether a name opens as an agent -- installed, complete, instruct, a template
   this build has, plannable, a stored format it reads, and a quantization asked for on load that the weights
   allow -- without loading anything, returning the reason as a value. Without it a program that releases first
   loses its working model to a typo. Built from what `InstalledModel`, `chatProtocolFor`, `requirePlannable` and
   `requireStoredQuantizationMatches` already do.
3. **Sampling overridden per knob**, each knob the program sets standing over the publisher's recommendation for
   the thinking in effect, so no application repeats that merge.

## 5. Decisions

All five agreed 2026-10-27 (Todd), as proposed.

1. **Gemma's automatic context ceiling.** `FamilyTraits` caps Gemma and Llama at 131072 and Qwen at 262144; with it
   deleted, the automatic context can reach the model's trained maximum where the card holds it. Gemma's quality
   has not been gated across the contexts the planner may choose (`BACKLOG.md` *Gemma 4 Complete*). The planner
   decides, as it does for every other program, and that item gates the quality.
2. **The desktop app's settings stop being Chat's.** The desktop session reads settings of its own, which its
   Settings view writes; Chat keeps `ChatConfiguration.md`'s layers.
3. **The desktop app's prompts move to `Mila/Apps/Assistant/`**, with a loader of its own, and Chat's `Prompts`
   keeps Chat's.
4. **Round measurements stay in each application** (section 3).
5. **The canned weather tool leaves the desktop app**, which then offers only its own tools; Chat keeps it as its
   probe.

## 6. Order

1. Propose 4.1 to 4.3; build them with library tests, Chat moved onto the session in the same change (it is the
   only consumer of the agent that changes model), gated by the console byte gate.
2. The desktop session onto `Mila::AI` alone: its own settings, prompts, store commands and event mapping. Gated by
   the session drives and the window drive, and `mila-session` linking no module of Chat's. Built in `+105`, with
   `Agent::sampling()` and `Agent::thinks()` made public so the app shows the sampling in effect without repeating
   the merge, and effort offered in the template's own values (`MilaApp.md` 3.7).
3. `MilaChatSession` dissolved into Chat; `FamilyTraits`, `ModelSize`, the aliases and the Qwen mapping deleted.
