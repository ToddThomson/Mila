# Model Handle

Specification and implementation plan for the one place a model's name becomes the object that runs it:
the concrete type that loads its weights, the tokenizer that reads its text, the grammar it was trained on,
and what it can do. The handle is the foundation `Mila::AI` is built on (`Direction.md` 3.2, 5.1); this
document stops at the handle, and `Mila::AI` and the agent core get their own.

Written 2026-09-25 at `0.21.0-dev+7`. The capabilities (Phase 1) and the protocol (Phase 2) are built, the
protocol finished at `+94`; the architecture list, the handle and the factory (Phase 3) at `+96`, all but
`adopt` (10.4, moved to `Vnext.md`). Where this document and the code differ, the code is what was decided since and the section says
so.

---

## 1. Problem Statement

A model's name becomes a running model in three places, in two languages, each its own way, and each also
re-derives what the model can do from which family it belongs to.

| Consumer | Erasure | Per-family sites today |
|---|---|---|
| Chat | `ModelVariant`, a `std::variant` of four `unique_ptr`s (`Chat.Session.ixx`), ten `std::visit` sites | 28 `ModelType::` references across five files |
| Python binding | one session class per family (`LlamaSession`, `GemmaSession`, `QwenSession`) | the classes themselves, `Tokenizer::fromStore`'s architecture switch, `requireArchitecture` |
| MIS | `ModelFamily` enum and `SESSION_FOR` (`model_worker.py:38`) | 26 `ModelFamily.<member>` references outside `config.py` |

What Chat's branches decide falls into seven kinds, and every one of them is a fact about the model rather
than about Chat:

| Decision | Chat site | What varies by family |
|---|---|---|
| Load the weights | `Chat.Session.ixx` | the concrete model type |
| Load the tokenizer | `Chat.Session.ixx` | `BpeTokenizer::loadGemma` / `loadLlama32` / `loadQwen` |
| Render the prompt | `Chat.Session.ixx` | three renderers, in three places (section 1.1) |
| Declare the tools | `Chat.Session.ixx` | prose plus JSON for Llama, `serializeToolDeclarations` for Gemma, inside `formatPrompt` for Qwen |
| Parse a tool call | `Chat.Session.ixx` | Gemma in-stream, `Qwen::parseToolCall`, Chat's own `ToolCallParser` |
| Stop on a tool call | `Chat.Session.ixx` | Gemma's `<tool_call|>` token id; MIS's `_stop_markers_for` has a second table |
| Route the reasoning channel | `Chat.Session.ixx` | `kGemmaChannel` or Qwen's `<think>` pair |

And `Chat.FamilyTraits.ixx` answers "does it have a reasoning channel" and "how much context can it
address" with a switch on the family, so two models of one family cannot differ.

### 1.1 The grammar is not where `Direction.md` 3.1 says it is

The model's native grammar is `Mila/Src`'s by rule (`Direction.md` 3.1). Today:

| Family | In `Mila/Src` | Elsewhere |
|---|---|---|
| Qwen | `Qwen.Protocol.ixx`, 523 lines -- Chat and MIS both call it | nothing |
| Gemma | `Gemma.Protocol.ixx`, 1296 lines -- MIS calls it | **Chat renders its own** (`formatGemmaPrompt`, `Chat.ixx:1359`), with a thinking-effort prose scale the library's renderer does not have |
| Llama | **nothing** | Chat's `MessageFormatter` (203 lines) and `ToolCallParser` (242), and MIS's `_build_llama_prompt` -- which **disagree**: Chat declares tools in the Llama 3.2 zero-shot tool format, MIS as a sentence and a JSON dump |

The Llama disagreement is the defect the Gemma fold fixed at `+34` (MIS advertised tools in a form the
model was never trained on), still present for the one family that was not folded.

### 1.2 Consequence for Phase 4 of `Deployment.md`

`FamilyTraits` caps Gemma at 131072. Gemma 4 12B's weights header says 262144 (`max_position_embeddings`,
`Gemma.md` section 2), and since `0.21.0-dev+7` the binding and MIS plan against the header. On a card
that holds more than 131072 tokens of Gemma, Chat and a Python program would now choose different
contexts for the same model. Not measured; it follows from the two ceilings. Section 3.2 settles which
source is the fact.

---

## 2. What The Handle Is

One object per opened model, holding:

- the **model**, erased to its `LanguageModel<TDeviceType, TPrecision>` base;
- its **tokenizer**;
- its **protocol**, the grammar it was trained on;
- its **capabilities**, each read from the one place that knows it;
- its **plan**, the `DeploymentPlan` it was loaded with;
- its **record**: name, architecture, variant, lineage.

The family is erased. Device and precision are not: they stay compile-time, as they are everywhere in Mila,
so the handle is `ModelHandle<TDeviceType, TPrecision>` (name open, 10.1).

---

## 3. Design

### 3.1 The fact the design rests on

**The erasure already exists.** `GemmaModel`, `LlamaModel`, `QwenModel` and `GptModel` all derive from
`LanguageModel<TDeviceType, TPrecision>` (`LanguageModel.ixx:54`), whose `generate` is one virtual call,
`onGenerating`, per call. For a fixed device and precision a `unique_ptr<LanguageModel<Cuda, BF16>>` holds
any of them. Chat needs a variant only because it also carries an FP32 Llama.

So the handle adds no per-token cost and no new virtual layer beneath `generate`; `Direction.md` 4.2 rule 4
("one virtual call per respond, none per layer or per token") is already true of the base. What the base
lacks is what the handle has to supply: the plan, the grammar, the tokenizer and the capabilities.

Two members every family has and the base lacks -- `getDeploymentPlan()` and `promptPrefixReuse()`
(`AnyPosition` for Gemma and Llama, `SavedPosition` for Qwen, which resumes only from the end of the previous
prompt, `PromptCaching.md`) -- **stay on the families** (decided 2026-10-25, Todd). This section first moved them
onto the base so that a caller holding the base could reach them; the caller holds the handle instead, and the
factory reads both at the one point where the concrete type is known (3.6). The base is unchanged, and GPT-2,
which has no plan, owes it nothing.

### 3.2 Capabilities: each fact from the one place that knows it

`Direction.md` 3.2 lists "reasoning channel, vision, prefix reuse" as read from the manifest. Reading the
code, they come from three different places, and putting each in the wrong one recreates the problem:

| Capability | The fact belongs to | Source | Absent |
|---|---|---|---|
| Trained maximum context | the weights | the weights header's `max_seq_length`, which the planner already reads as `trained_maximum` | n/a -- every weights file has it |
| Instruction-tuned | the weights | manifest `instruct` (exists) | false |
| Reasoning channel | the weights: one family can publish a checkpoint with or without one | manifest `reasoning` (new); false narrows the template's thinking variable to its off value | the architecture's default (3.3) |
| Input modalities | the weights carry them; Mila must also load them | manifest `modalities` (new), intersected with what the family's loader reads | `["text"]` |
| Draft model | the package carries one, or does not | manifest (new, 2026-10-04), naming the draft's weights in the package | none |
| Prompt-prefix reuse | Mila's implementation of the architecture | the model type (3.1) | n/a |
| Template variables (thinking, effort, kept reasoning) | the checkpoint's template | the protocol (3.4), by the template's own names, values and defaults: Qwen 3.8 `enable_thinking` (true), `reasoning_effort` (`xhigh`, `medium`, `low`) and `preserve_thinking` (true); Gemma 4 `enable_thinking` (false) and `preserve_thinking` (false); Llama 3 none | n/a |
| Streaming by channel | the application's display | stays in the application (`ROADMAP.md`, Model Handle) | n/a |
| Default context | the application's configuration | stays in the application (Chat layer 1) | n/a |

Three consequences worth stating:

- **No manifest field for the context ceiling.** The weights header already states it and the planner
  already reads it, so a `maximum_context_length` in the manifest (`ChatConfiguration.md` section 5
  proposed one) would be a second source that can disagree -- which is exactly what `FamilyTraits`' 131072
  is today (1.2). A checkpoint whose header overstates what it was trained to is fixed in its header, at
  conversion. Open, 10.3.
- **Prefix reuse is not a manifest field.** Llama reports false because Mila has not implemented it, not
  because the weights forbid it; publishing a manifest could never change that, and a Llama that gains it
  gains it in code.
- **A manifest that omits a field loads.** The ROADMAP criterion. Every new field has a stated default,
  and for `reasoning` the default is the architecture's, so every model published before this spec keeps
  the behaviour it has now.

### 3.3 Identity: one list of architectures, in `Mila/Src`

The one place an architecture name becomes a type is a compile-time list:

```cpp
// Dnn/Models/Architectures.ixx
using Architectures = ArchitectureList<GemmaArchitecture, LlamaArchitecture, QwenArchitecture>;
```

Each family contributes one descriptor, in its own directory beside its model and protocol
(`Gemma/Gemma.Architecture.ixx`), as built at `+96`:

```cpp
struct GemmaArchitecture
{
    static constexpr std::string_view name = "gemma";                        // what a manifest's architecture says
    static constexpr std::array<std::string_view, 1> template_versions{ "4" }; // architecture_version values
    static constexpr bool drafts = true;                                     // this build decodes with its MTP drafter

    template<DeviceType D, TensorDataType P>
    using Model = GemmaModel<D, P>;

    static std::shared_ptr<BpeTokenizer> loadTokenizer( const std::filesystem::path& path );
    static std::unique_ptr<Conversation::Protocol> chatProtocol( std::string_view version );  // null: none here
};
```

`ArchitectureList` (`Dnn/Models/ArchitectureList.ixx`) finds a descriptor by name -- `visit( name, visitor )`
calls the visitor with `std::type_identity` of the one named, and says whether one was -- and gives `contains`,
`forEach` and `names()` for refusals. There is no registrar and no string-keyed runtime table (`CLAUDE.md`: the
`OperationRegistry` pattern is retired); a descriptor missing a member a caller uses is a compile error. Names
compare exactly, as manifests write them.

Every place in `Mila/Src` that turned an architecture into a type goes through it: `chatProtocolFor`,
`InstalledModel::loadTokenizer` and `drafter`, `requirePlannable` and `planDeployment` in
`Deployment.InstalledModelPlanner`, and the handle's factory. **Each architecture decides what its versions
mean**: Gemma and Qwen take an empty `architecture_version` as their one template; Llama, with two, refuses
it.

**Adding an architecture** is the family directory, which is written anyway, plus one entry in the list.
That is the ROADMAP's "added in one place" read honestly: one place outside the family's own code. One
shared file still names every family: `Dnn.ModelType`'s enum, which `LanguageModelConfig` and the
quantization dispatch key on and which a new family's config extends regardless.

What this section first proposed and the build no longer needs: `applyVariant`, because a load now reads
the weight format from the weights header and a request names one only to quantize on load (`Deployment.md`
12.6), so the four hand-kept mappings it would have absorbed are gone; and `reasoning_by_default`, because
whether a model thinks is the manifest's `thinking` capability and the template's own variables (3.2, 3.4).

`Direction.md` 3.2 places the identity "beside the manifest reader". It cannot sit literally there:
`Distribution/` is independent of `Dnn` in both directions, and the list names `Dnn` types. It sits beside
the families it names, and reads the architecture string the manifest reader produces.

### 3.4 The protocol: the grammar behind one interface, in `Mila/Src`

A virtual interface in `Dnn.Models.Conversation`'s namespace, implemented once per **template** -- the chat
template ships with the weights, and the tool grammar is part of it, so it is not stable within a family
(Gemma 3 had no tool tokens, Qwen 2.5 wrote JSON inside `<tool_call>`, Llama 3.1 and 3.2 differ). Today each
family has one, and the architecture picks it (decided 2026-10-14, Todd); a package naming its own handler
waits for a family that ships a second template.

```cpp
class Protocol
{
public:
    virtual Prompt formatPrompt( std::span<const Turn> history, const PromptOptions& options ) const = 0;
    virtual const ReplyGrammar& replyGrammar() const noexcept = 0;
    ReplyReader readReply( const Prompt& prompt ) const;                 // a reply as it streams
    Reply parseReply( std::string_view text, const Prompt& prompt ) const;   // the same reader, fed whole
    virtual std::string formatToolResult( std::string_view name, std::string_view result ) const = 0;
    virtual const ProtocolMarkers& markers() const noexcept = 0;
    virtual std::span<const TemplateVariable> templateVariables() const noexcept = 0;   // empty when none
};

struct Prompt
{
    std::string text;
    bool reply_begins_in_reasoning{ false };           // Qwen's primer leaves <think> open; Gemma's, on a thinking resume
};

struct TemplateVariable
{
    std::string name;                                  // as the template spells it: "enable_thinking"
    std::vector<std::string> values;                   // every value the template accepts, as text
    std::string default_value;                         // what the template uses when the variable is not set
};

struct PromptOptions
{
    std::map<std::string, std::string> template_variables;   // unset: the template's default; else a listed value
    std::string tools_json;                            // the family serializes it its own way
};

struct Reply
{
    std::string reasoning;                             // the reasoning channel's text, empty when none
    std::string content;                               // what the user is shown
    std::vector<ToolCall> tool_calls;                  // every call, in the order the model made them
    std::vector<std::string> malformed_tool_calls;     // each call span the grammar could not read, as written
};

struct ProtocolMarkers
{
    std::string reasoning_open, reasoning_close;
    std::string tool_call_open, tool_call_close;
    std::string tool_response_open;
    std::vector<std::string> stop_after;               // a closed call, and the open of a result (the engine's to write)
};
```

Built at `0.21.0-dev+68` as `Dnn.Models.Conversation.*`, one module per type, with `QwenChatProtocol` and
`GemmaChatProtocol` beside each family's free functions. **The reply is read against its prompt**: whether it
begins inside the reasoning channel is a fact of the rendering -- Qwen's primer leaves `<think>` open while
thinking is on, Gemma's leaves its channel open only on a resume after a result with thinking on -- so
`formatPrompt` returns it and `parseReply` takes it. **The stop is the open of a result, never a call's
close** (2026-10-20): stopping at a call's close kept only the first of several, and both families write every
call a turn needs before handing over -- Gemma 4 with `<|tool_response>`, its own stop, Qwen 3.8 by ending the
turn (`InferenceServerWire.md` 3.4 and W2).

Each member answers one of section 1's seven decisions: `formatPrompt` renders and declares tools,
`parseReply` parses, `markers()` gives the stop and the channel routing, `formatToolResult` renders a
tool's answer as the text that continues the turn -- what the agent core's splice appends to the live cache,
though how it is used is that spec's decision. The existing free functions (`Gemma::formatPrompt`,
`Qwen::parseToolCall`, and the rest) stay, and the implementations call them; the binding's
`qwen_format_prompt` and its siblings keep working until the binding moves onto `Mila::AI`.

**A reply is read as it streams, by one reader** (`0.21.0-dev+94`). Each template declares a `ReplyGrammar` --
its reasoning, call and result markers, the label a reasoning span opens with, a call written with no marker
(Llama's bare `{`), text that ends the reply, control text that is never answer -- and a function that reads one
call span's arguments. `ReplyReader` reads a reply fed in pieces by that grammar into `ReplyEvent`s (reasoning or
answer text, a span opening or closing, a call opening or closing) and a `Reply`; `parseReply` is the same reader
fed the whole text, and is no longer virtual. So what streamed is what was parsed: the events' text is the
`Reply`'s, and how the reply was cut into pieces changes neither (`Conversation.ReplyReader.cpp`). Text that may
still become a marker, and an incomplete UTF-8 character, are held until the next piece settles them. The reader
works on text rather than token ids, so every family streams, and a call is read as it closes rather than after
the turn. Two readings changed at the move, on output no measured model writes: Qwen's reasoning is a
`<think>` span wherever it opens, where before it was everything before the first `</think>`, and a Qwen result
the model writes itself is dropped, as Gemma's always was.

**The whole reply, not the first call** (2026-10-13, `InferenceServerWire.md` 3.2). `parseReply` replaced a
`parseToolCall` that returned one `optional<ToolCall>`: a turn can hold several calls, and OpenAI's `tool_calls`
and Anthropic's content blocks are lists, so every call after the first was lost on every server endpoint. It
also takes over the answer extraction each consumer did for itself (`Gemma::extractAnswer`, the server's
`answer_text`).

**A malformed call is never silent** (decided 2026-10-14, Todd). Today Qwen throws and Gemma returns nothing
and leaves the text, so the same failure is an exception from one family and invisible from the other.
`parseReply` does neither: it does not throw, and each call span the grammar cannot read -- an unclosed one
included -- is returned in `malformed_tool_calls` as the model wrote it, beside whatever calls did parse, and
kept out of `content`. What a consumer does with it --
Chat shows it, the server logs it and answers in prose -- is the consumer's; that it learns of it is not.

**A result names its tool; the loop resolves which call it answers** (decided 2026-10-14). The template
renders a history; it does not own the tool loop. Which call a result answers is known to whoever ran the
call -- Chat, or the server's client, which sends an id -- so it is resolved there, and a `Tool` turn carries
the name of the tool that produced it (`Conversation::Turn::tool_name`). Gemma's grammar renders the name
(`response:get_weather{...}`); Qwen's and Llama's render content only and ignore it. Ids stay out of the
library: they are how one wire format finds the name. Turns render in the order they are given, never
reordered, and pairing by position is rejected -- it has the protocol infer a fact about the flow that only
the loop's owner holds, and it misnames a result returned out of order or alone. A Gemma call and its
results share one model turn, so consecutive `Tool` turns after an `Assistant` turn with calls fold into it,
and a history ending on a `Tool` turn resumes that turn open; both follow from the conversation, so
`continue_open` is not a member of the interface.

**Why virtual rather than a concept.** The handle holds one protocol chosen by name at run time, so a concept
would need a variant again. The call is once per turn, never per token.

**Why in `Mila/Src` rather than `Mila/AI/`.** `Direction.md` 3.1: the model's native grammar is the
library's and consumer-blind. The interface holds no conversation and dispatches no tool; those are the agent
core's.

**Reasoning is the checkpoint's, exactly** (decided 2026-10-14, Todd). Thinking, reasoning and effort are
model-specific, and what a template offers is not a list of modes but its own **variables** -- what HF's
`apply_chat_template(**kwargs)` takes. Qwen 3.8's has `enable_thinking` (default true), `reasoning_effort`
(`xhigh` by default, `medium`, `low`; read only while thinking is on, and anything else raises) and
`preserve_thinking` (default true, which renders past turns' reasoning back into history). Gemma 4's has
`enable_thinking` and `preserve_thinking`, both false by default. Llama 3's has none. So there is no
family-neutral effort type and no invented name -- not even `off` or `on` -- and no layer, the library, an
application or an end user, invents a level or its text. `templateVariables()` lists each variable by the
template's name, with every value it accepts and its default; a prompt sets some of them, leaves the rest at
the template's default, and an unknown name or value is refused with what is accepted, never approximated.
Where the template ignores a variable (effort with thinking off), so does the renderer. An application may map
its own control onto a variable; the mapping is its own and lands only on an accepted value. Qwen's
`ReasoningEffort` stays an implementation detail behind its free functions. `Turn` gains `reasoning`, the
text of the turn's reasoning channel, so that `preserve_thinking` renders what the template renders.

**Llama is folded into the library**, as Gemma was at `+34`: `LlamaChatProtocol` (`+68`) renders Meta's own
templates, with parity against them rather than against Chat's `MessageFormatter` and `ToolCallParser`, which
Chat stopped calling and deleted at `+94`; a Llama whose record names no `architecture_version` is refused by
Chat's catalogue. MIS's `_build_llama_prompt` is deleted when MIS moves. **Llama ships two
templates**: 3.1's and 3.2's differ in `date_string`'s default (a fixed 26 Jul 2024, and today) and in 3.1's
built-in tools, which the protocol does not take -- so `llama31()` and `llama32()` are two handlers, and the
architecture alone cannot pick between them. The package names it: the manifest's `architecture_version`
(`+94`), which each descriptor reads (3.3).

**Gemma is made one renderer.** The Gemma protocol renders every variable of Google's template, `enable_thinking`
and its `<|think|>` trigger included, so Chat's `formatGemmaPrompt` -- whose remaining difference is the effort
prose 10.6 deleted -- is replaced rather than matched.

**The oracle is the checkpoint's own template** (2026-10-14). Parity is measured against `chat_template.jinja`
as the checkpoint ships it, rendered by `transformers`' `apply_chat_template`, not against today's Chat or server
output: those are earlier readings of the same template, and a reading can be wrong. Qwen 3.8's shows it -- its
default effort is `xhigh` where Mila assumed `medium`, it groups consecutive tool results into one user turn,
and it writes tool signatures with `tojson`'s separators and key order. Where an existing renderer differs from
the template, the template wins and the difference is recorded as a finding.

### 3.5 The handle, in the Mila library

```cpp
// Src/AI/ModelHandle.ixx, module AI.ModelHandle, namespace Mila::AI
template<DeviceType TDeviceType, TensorDataType TPrecision>
class ModelHandle
{
public:
    static ModelHandle open( std::string_view name, const DeploymentRequest& request,
                             const ModelStore& store = ModelStore{} );                       // section 3.6
    static std::expected<DeploymentPlans, DeploymentRefusal> planFor( name, request, store );

    LanguageModel<TDeviceType, TPrecision>& model() noexcept;         // descent: generate, observe, save
    template<typename TModel> TModel* as() noexcept;                  // the typed model, or null

    BpeTokenizer& tokenizer() const noexcept;
    const Conversation::Protocol* protocol() const noexcept;          // null for a base model
    const ModelCapabilities& capabilities() const noexcept;           // the package's, modalities this build runs
    PromptPrefixReuse promptPrefixReuse() const noexcept;             // the implementation's, read at load (3.1)
    const DeploymentPlan& plan() const noexcept;
    const InstalledModel& installed() const noexcept;                 // the store's record: lineage, licence, files
    const std::string& name() const noexcept;
};
```

- **Descent is one call** (`Direction.md` 4.2 rule 3): `model()` reaches everything `LanguageModel` exposes,
  including observation, and `as<GemmaModel<Cuda, BF16>>()` reaches the typed model and its components. A
  wrong type is null, never undefined behaviour.
- **A base model has no protocol.** `instruct` false means there is no template to render, so `protocol()` is
  null and `Mila::AI` refuses to converse with it -- by reading the handle, not by a family check.
- **The handle is small and holds no conversation.** It is what a model *is*; `Mila::AI` is what a program
  does with one.

**`Mila::AI` is part of the Mila library, not a library of its own** (`Direction.md` decision 2, revised
2026-10-25): `Src/AI/` is built into the Mila target, above the model level, which never imports it, so the
wheel and a `FetchContent` consumer receive it with everything else. The `MilaAI` target in `Mila/AI/` holds
the broker client built to the design `ToolCalling.md` replaced, and goes when that is refit.

### 3.6 The factory

`open( name, request )`, as built:

1. Finds `name` in the store, case-folded (`InstalledModel::find`). It never installs (`ModelDistribution.md`:
   install and load are separate verbs); a name not there is refused.
2. `requirePlannable`: the files are present, and the architecture is in the list (3.3) -- unknown is a
   refusal naming the list.
3. For an instruct model, the protocol from the descriptor's `chatProtocol( architecture_version )`; a
   version this build has no template for is refused here, before any file is read.
4. The tokenizer, through the descriptor.
5. `Architecture::Model<D, P>::load( weights, request )`: plan, check, execute (`Deployment.md` section 7). A
   refusal propagates as `DeploymentRefusedError`; the application words it (`Deployment.md` section 6). The
   typed model's plan and prefix reuse are read here, the one point the type is known (3.1).

The factory takes a `DeploymentRequest` rather than a device from the first commit (`ROADMAP.md`), so the
planner is the handle's entry point and not a signature change later. `planFor( name, request )` is steps 1
and 2 and the family's `planDeployment`, allocating nothing.

### 3.7 A composed network meets the same handle

`Direction.md` decision 7: a C++ concept, checked at compile time, no registration. Because the base is
already the contract (3.1), the concept is deliberately thin:

```cpp
template<typename TModel, DeviceType D, TensorDataType P>
concept HandleModel = std::derived_from<TModel, LanguageModel<D, P>>;

template<HandleModel<D, P> TModel>
static ModelHandle adopt( std::unique_ptr<TModel> model,
                          std::shared_ptr<BpeTokenizer> tokenizer,
                          std::unique_ptr<Conversation::Protocol> protocol,   // null for a base model
                          Capabilities capabilities );
```

A developer's model has no manifest, so what the factory reads from the record the developer passes. Not built:
what `plan()` returns for a model the planner did not load is open (10.4), and `adopt`'s shape follows it.

### 3.8 A model deployed for a use case

*Direction, decided 2026-10-04 (Todd); not built.* "`Mila::AI` can deploy a model for a specific use case; we
should have full control of the model features loaded." A handle is a model **and the features its use case
selected**, with nothing else resident. The capabilities in 3.2 are what a package offers; a deployment request
selects from them (`Deployment.md` 2.1) -- modalities, the draft model, context and cache format -- and the
factory builds exactly the selection the plan priced. A feature not selected is not built.

Each feature is a component that is built or not, never a template axis of the core blocks: an image path,
an audio path, a vision tower and a draft model attach to the network beside its blocks, so a selection is
not a new instantiation and the build cost does not multiply with combinations. The draft model is the one to
watch: it reads the network's embedding table, its final-normed hidden state and the caches of its last
sliding and last global layer (`Gemma4Mtp.md`).

`Mila::AI` names use cases over this -- a coding agent, a vision assistant, a long-document reader -- each a
selection with measured defaults (a context from the model's profile, a draft model where its speedup is
measured), overridable before creation, and never a second path: a preset produces an ordinary request.

---

## 4. Alternatives Rejected

**A variant of every family, as Chat has.** Every new family edits every `std::visit`, and the variant grows
by one per family and precision. The base class already erases the family.

**Erase device and precision too.** A handle over `LanguageModel<Cuda, BF16>` and `<Cuda, FP32>` behind
one interface would make precision a run-time value, against Mila's first design rule. The handle stays a
template; `Mila::AI` fixes the pair a program uses.

**Capabilities in the manifest, all of them.** What `Direction.md` 3.2 first said. Prefix reuse and the
context ceiling are not facts a publisher can state (3.2); a manifest that claimed prefix reuse for a Llama
would be a lie the library then believed.

**A registrar the families call at start-up.** String-keyed and run-time, the pattern `OperationRegistry`
is being retired from; a missing family is then a run-time miss rather than a compile error.

**The protocol in `Mila/AI/`.** It would make the grammar the agent layer's, and the binding's
`qwen_format_prompt` would then depend on `Mila.AI` for a template. The grammar is the model's.

**A concept for the protocol.** Needs a variant to hold one chosen at run time; see 3.4.

---

## 5. Traps

- **Parity before deletion.** Every renderer that moves -- Llama's into the library, Gemma's two into one --
  is proven byte-identical against the checkpoint's own template (P1), on recorded fixtures, before any
  caller switches; where the renderer it replaces disagrees, the caller changes behaviour when it switches,
  and the release notes say so. The Gemma fold's parity step (`+34`) is the recipe: declarations, calls,
  responses and answers identical; parsed arguments equal after decoding.
- **Gemma's `call_id` stays a digest** of name and arguments, not a counter, because MIS's Responses path
  re-parses the transcript on every request. Any protocol member that mints an id must keep that.
- **C1128.** The factory instantiates every family's `load` in one translation unit, which is the shape that
  has crossed 65535 sections before (`ExportArtifact` at 61393). Measure the factory's TU under
  RelWithDebInfo before it grows; split per architecture if it nears the limit.
- **`import Mila.AI` in an ordinary `.cpp`** will meet the MSVC defect that `import Mila;` does (C2079 with std
  headers). The binding's wrapper pattern -- a module implementation unit behind a std-only interface --
  already works around it.
- **The manifest default for `reasoning` must be the architecture's**, not false. False would silently turn
  thinking off for every Gemma and Qwen published before this spec.

---

## 6. Validation

Bounds written before any run; each gate forced to fail once. All in the maintainer's environment
(`x64-profile`, both GPUs).

- **H1. The handle runs what the direct type runs.** For every installed architecture, greedy generation of a
  fixed prompt through `ModelHandle::open` is token-identical to the typed `load`, on both cards.
- **H2. No per-token cost.** Decode throughput through the handle within the run-to-run spread of the direct
  typed model, three runs each, Gemma 4 12B FP4 and Llama 3.1 8B FP4 (`ROADMAP.md`'s criterion).
- **H3. A silent manifest loads.** A manifest with none of the new fields opens, and reports the
  architecture's defaults.
- **H4. The manifest wins over the family.** A Qwen manifest declaring `reasoning: false` opens with no
  reasoning capability, and its protocol renders the thinking-off primer.
- **H5. Adding an architecture is one list entry.** GPT-2 was to be the proof, and cannot be: it has no
  planner and no `load( path, request )`, so it cannot be an entry (`Mila/Issues/Untriaged.md`, 2026-10-25).
  As built, H5 is the structure plus N1: every `Mila/Src` site that named a family goes through the list
  (3.3), and a name the list lacks is refused rather than routed.
- **H6. A composed network meets the contract.** A test builds a small `LanguageModel` subclass from
  components, adopts it with no registration, and generates through the handle.
- **P1. Protocol parity.** Each family's `Protocol` produces prompts byte-identical to the checkpoint's own
  `chat_template.jinja` under `apply_chat_template`, on fixtures recorded from it: every template variable at
  each of its values, tools, one and several calls, results in and out of call order, multi-turn, and the
  resume after a result (3.4).
- **Negatives.** N1: an architecture removed from the list -- opening its model is a refusal naming the list,
  not a misroute into another family (the "else means llama" defect MIS had). N2: `as<>` with the wrong type
  returns null. N3: the manifest default for `reasoning` set to false -- H3 fails for Gemma and Qwen.

---

## 7. Phasing

**Phase 1 -- capabilities.** The manifest fields and their defaults. *Exit:* H3, H4, N3. Independent of
everything else. Built: `ModelCapabilities` in the manifest, read through `InstalledModel` (`+61`); the two
members stay on the families (3.1).

**Phase 2 -- the protocol.** The interface, Gemma and Qwen implementations over their free functions,
Llama's fold into the library, Gemma's renderers made one. *Exit:* P1 for all three families. No application
switches yet. Built through `+94`.

**Phase 3 -- identity, handle and factory.** The architecture list, the descriptors, `ModelHandle` in the Mila
library (3.5), `open`, `planFor`, `adopt`. *Exit:* H1, H2, H5, H6, N1, N2. Built at `+96` but `adopt` and H6,
which wait on 10.4 and are `Vnext.md`'s. `Tests/AI/ModelHandle.Cuda.cpp` holds H1, N1 and N2; H1 compares on every CUDA device that
holds the model, and on 2026-10-25 did so for Gemma 4 12B FP4 and Llama 3.2 3B FP4 on both cards and Qwen 3.8
27B FP4 on the 16 GB one. Each gate was made to fail once: `as<>` without its type check, a request's context
ignored, a list that matches any name, the tokenizer read before the template's refusal, Gemma not drafting.
H2, measured once on 2026-10-25 and not kept as a test (the handle's `model()` is the object the typed `load`
returns, so the call is the same one): 256 greedy tokens after a 64-token prompt, context 4096, RTX 5060 Ti,
three runs each, Gemma 4 12B FP4 54.60-54.65 tok/s through the handle and 54.52-54.55 typed, Llama 3.1 8B FP4
81.23-81.25 and 81.22.

The applications move in the Applications theme, onto `Mila::AI` rather than onto the handle, so each moves
once (`ROADMAP.md`, "one release rather than three"). The criterion "no dispatch site outside the handle
carries a per-family branch" is therefore met when they do, and `ModelVariant`, the per-family session
classes, `ModelFamily`, `FamilyTraits`' two weight facts, and Chat's and MIS's grammar copies are deleted
there.

---

## 8. Non-Goals

- Choosing a package for a base-model name, or naming an intent (`Direction.md` 6.5).
- Opening a loose weights file with no store record. `Direction.md` 4.2 rule 7: in v0.22.0 a model is named.
- Erasing device or precision (4).
- Conversation state, tool dispatch, the splice, and the autonomy policy: `Mila::AI` and the agent core.

---

## 9. Relationship To Other Specs

- `Direction.md` -- 3.2 and 5.1 are this spec's charter; 3.2's capability list is corrected by 3.2 here.
- `Deployment.md` -- the factory is `load( path, request )` behind a name; the plan is the handle's.
- `ModelDistribution.md` -- the manifest and the store the factory reads; gains two fields.
- `ChatConfiguration.md` -- section 5's `maximum_context_length` is replaced by the weights header (10.3).
- `PromptCaching.md` -- what `promptPrefixReuse` reports.
- `PythonBinding.md` -- the per-family sessions this replaces, via `mila.AI`.
- `ModelFamilyParity.md` -- what each family can do; its grammar rows are this spec's Phase 2.

---

## 10. Open Decisions

1. **The class name.** *Decided 2026-10-25 (Todd): `ModelHandle`.* `Mila::AI::Model` reads better in a program
   and collides in reading, not in code, with `Mila::Dnn::Model`; revisit when `Mila::AI`'s own object shows how
   often a program types it.
2. **Chat's FP32 Llama.** *Settled by events: retired.* Chat's variant holds three BF16 types and no FP32 one.
3. **The context ceiling's one source.** Recommend the weights header (3.2), with `ChatConfiguration.md`'s
   `maximum_context_length` withdrawn and `FamilyTraits`' 131072 for Gemma deleted rather than moved. Before
   deciding, measure Gemma 4 12B above 131072 -- the 262144 in its header is HF's `max_position_embeddings`,
   and whether quality holds out there is a question the header cannot answer. If it does not, the header is
   what changes, at conversion.
4. **`plan()` for an adopted model.** *Reopened 2026-10-25.* The planner did not load it. The second option
   below was recommended and agreed, and does not hold as written: `planOnDevice` is generic over the network but
   also takes the package's `WeightsMetadata` and trained maximum, which a composed network has none of, and a
   network handed to `adopt` is already built, so a plan made then is priced against a device that already holds
   it. Options: the handle's plan is optional, null for an adopted model; `adopt` takes the network unbuilt, with a
   request and the facts the planner reads, and builds what it plans; or `adopt` takes the plan the developer
   made. `adopt` and H6 wait on this, and both moved to `Vnext.md` (Todd, 2026-10-25): v0.21.0 finishes the
   Mila app, which opens only installed models.
5. **The Python projection of descent.** `mila.AI` projects the contract; how far `model()` reaches from
   Python -- the `LanguageModel` surface, or also observation -- is `PythonBinding.md`'s to decide with
   `Mila::AI`'s spec.
6. **Gemma's effort scale.** *Decided 2026-10-14 (Todd): deleted.* "Mila chat should not impose a made up
   setting", and the same for the inference server. Chat's five-step scale and its Gemma sentences go, and
   its reasoning setting becomes the loaded model's own template variables (3.4). What replaces the server's fixed
   thinking-off is `InferenceServerWire.md` 3.6's to settle.
