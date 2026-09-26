# Model Handle

Specification and implementation plan for the one place a model's name becomes the object that runs it:
the concrete type that loads its weights, the tokenizer that reads its text, the grammar it was trained on,
and what it can do. The handle is the foundation `Mila::AI` is built on (`Direction.md` 3.2, 5.1); this
document stops at the handle, and `Mila::AI` and the agent core get their own.

Written 2026-09-25 at `0.21.0-dev+7`. **Draft.** It proposes changes to `Mila/Src` -- two members on the
`LanguageModel` base, a protocol interface, Llama's grammar moved into the library, and the architecture
list -- and none of them is agreed until the decisions in section 10 are.

---

## 1. Problem Statement

A model's name becomes a running model in three places, in two languages, each its own way, and each also
re-derives what the model can do from which family it belongs to.

| Consumer | Erasure | Per-family sites today |
|---|---|---|
| Chat | `ModelVariant`, a `std::variant` of four `unique_ptr`s (`Chat.ixx:73`), ten `std::visit` sites | 28 `ModelType::` references across five files |
| Python binding | one session class per family (`LlamaSession`, `GemmaSession`, `QwenSession`) | the classes themselves, `Tokenizer::fromStore`'s architecture switch, `requireArchitecture` |
| MIS | `ModelFamily` enum and `SESSION_FOR` (`model_worker.py:38`) | 26 `ModelFamily.<member>` references outside `config.py` |

What Chat's branches decide falls into seven kinds, and every one of them is a fact about the model rather
than about Chat:

| Decision | Chat site | What varies by family |
|---|---|---|
| Load the weights | `Chat.ixx:2621` | the concrete model type |
| Load the tokenizer | `Chat.ixx:2451` | `BpeTokenizer::loadGemma` / `loadLlama32` / `loadQwen` |
| Render the prompt | `Chat.ixx:1290` | three renderers, in three places (section 1.1) |
| Declare the tools | `Chat.ixx:2863` | prose plus JSON for Llama, `serializeToolDeclarations` for Gemma, inside `formatPrompt` for Qwen |
| Parse a tool call | `Chat.ixx:593`, `:607` | Gemma in-stream, `Qwen::parseToolCall`, Chat's own `ToolCallParser` |
| Stop on a tool call | `Chat.ixx:1078` | Gemma's `<tool_call|>` token id; MIS's `_stop_markers_for` has a second table |
| Route the reasoning channel | `Chat.ixx:831` | `kGemmaChannel` or Qwen's `<think>` pair |

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

Two members move onto the base, because every family already has them and a caller holding the base cannot
reach them:

- `getDeploymentPlan()` -- each family keeps an identical `plan_` today.
- `supportsPromptPrefixReuse()` -- on `QwenModel` only (`QwenModel.ixx:338`). On the base it is virtual and
  false; Gemma, whose generate loop reuses a cached prefix (`PromptCaching.md`), returns true.

### 3.2 Capabilities: each fact from the one place that knows it

`Direction.md` 3.2 lists "reasoning channel, vision, prefix reuse" as read from the manifest. Reading the
code, they come from three different places, and putting each in the wrong one recreates the problem:

| Capability | The fact belongs to | Source | Absent |
|---|---|---|---|
| Trained maximum context | the weights | the weights header's `max_seq_length`, which the planner already reads as `trained_maximum` | n/a -- every weights file has it |
| Instruction-tuned | the weights | manifest `instruct` (exists) | false |
| Reasoning channel | the weights: one family can publish a checkpoint with or without one | manifest `reasoning` (new) | the architecture's default (3.3) |
| Input modalities | the weights carry them; Mila must also load them | manifest `modalities` (new), intersected with what the family's loader reads | `["text"]` |
| Prompt-prefix reuse | Mila's implementation of the architecture | the model type (3.1) | n/a |
| Trained reasoning-effort levels | the grammar | the protocol (3.4): Qwen has three, Gemma none | n/a |
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
(`Gemma/GemmaArchitecture.ixx`):

```cpp
struct GemmaArchitecture
{
    static constexpr std::string_view name = "gemma";          // what a manifest's architecture says

    template<DeviceType D, TensorDataType P>
    using Model = GemmaModel<D, P>;

    using Protocol = GemmaProtocol;                              // section 3.4

    static std::shared_ptr<BpeTokenizer> loadTokenizer( const std::filesystem::path& path );

    static void applyVariant( DeploymentRequest& request, std::string_view variant );

    static constexpr bool reasoning_by_default = true;          // section 3.2, when the manifest is silent
};
```

The factory folds over the list comparing names; an unknown name is a refusal listing the names the list
holds. There is no registrar and no string-keyed runtime table (`CLAUDE.md`: the `OperationRegistry`
pattern is being retired); a descriptor missing a member is a compile error.

**Adding an architecture** is the family directory, which is written anyway, plus one entry in the list.
That is the ROADMAP's "added in one place" read honestly: one place outside the family's own code.

`applyVariant` absorbs a mapping that exists four times today and must be kept in step by hand: the
binding's `applyQuantizationVariant` and `applyQwenQuantizationVariant`, and Chat's `deploymentRequestFor`
and `applyQwenQuantization`. Qwen's differs from the others (no FP8 KV type, and the codebook variant), which
is why it is per architecture.

`Direction.md` 3.2 places the identity "beside the manifest reader". It cannot sit literally there:
`Distribution/` is independent of `Dnn` in both directions, and the list names `Dnn` types. It sits beside
the families it names, and reads the architecture string the manifest reader produces.

### 3.4 The protocol: the grammar behind one interface, in `Mila/Src`

A virtual interface in `Dnn.Models.Conversation`'s namespace, implemented once per family:

```cpp
class Protocol
{
public:
    virtual std::string formatPrompt( std::span<const Turn> history, const PromptOptions& options ) const = 0;
    virtual std::optional<ToolCall> parseToolCall( std::string_view text ) const = 0;
    virtual std::string formatToolResult( std::string_view name, std::string_view result_json ) const = 0;
    virtual const ProtocolMarkers& markers() const noexcept = 0;
    virtual std::span<const ReasoningEffort> reasoningEfforts() const noexcept = 0;
};

struct PromptOptions
{
    bool enable_reasoning{ false };
    std::optional<ReasoningEffort> reasoning_effort;   // only a level reasoningEfforts() lists
    std::string tools_json;                            // the family serializes it its own way
};

struct ProtocolMarkers
{
    std::string reasoning_open, reasoning_close;
    bool primer_opens_reasoning{ false };              // Qwen leaves <think> open when reasoning is on
    std::string tool_call_open, tool_call_close;
    std::string tool_response_open;
    std::vector<std::string> stop_after;               // text that ends a turn early: a closed tool call
};
```

Each member answers one of section 1's seven decisions: `formatPrompt` renders and declares tools,
`parseToolCall` parses, `markers()` gives the stop and the channel routing, `formatToolResult` renders a
tool's answer as the text that continues the turn -- what the agent core's splice appends to the live cache,
though how it is used is that spec's decision. The existing free functions (`Gemma::formatPrompt`,
`Qwen::parseToolCall`, and the rest) stay, and the implementations call them; the binding's
`qwen_format_prompt` and its siblings keep working until the binding moves onto `Mila::AI`.

**Why virtual rather than a concept.** The handle holds one protocol chosen by name at run time, so a concept
would need a variant again. The call is once per turn, never per token.

**Why in `Mila/Src` rather than `Mila/AI/`.** `Direction.md` 3.1: the model's native grammar is the
library's and consumer-blind. The interface holds no conversation and dispatches no tool; those are the agent
core's.

**What stays out.** Only what the checkpoint was trained on is protocol. Chat's five-step thinking-effort
scale for Gemma is a sentence Chat writes into the system turn -- Gemma's `<|think|>` is a boolean with no
trained budget -- so it stays Chat's text (open, 10.6). Qwen's three levels are trained, with exact wording,
so they are protocol.

**Llama is folded into the library**, as Gemma was at `+34`: Chat's `MessageFormatter` and
`ToolCallParser` become `Llama.Protocol.ixx`, with parity demonstrated against Chat's output before either
application stops calling its own. MIS's `_build_llama_prompt` is then the third copy, and the one that
disagrees; it is deleted when MIS moves, and the zero-shot tool format is the one kept, because it is what
Llama 3.x was fine-tuned on.

**Gemma is made one renderer.** `Gemma::formatPrompt` gains the `<|think|>` trigger and the empty
thought-channel primer that only Chat's copy has, so Chat's `formatGemmaPrompt` has nothing left but the
effort prose.

### 3.5 The handle, in `Mila/AI/`

```cpp
template<DeviceType TDeviceType, TensorDataType TPrecision>
class ModelHandle
{
public:
    static ModelHandle open( std::string_view name, const DeploymentRequest& request );   // section 3.6

    LanguageModel<TDeviceType, TPrecision>& model() noexcept;         // descent: generate, observe, save
    template<typename TModel> TModel* as() noexcept;                  // the typed model, or null

    const BpeTokenizer& tokenizer() const noexcept;
    const Conversation::Protocol* protocol() const noexcept;          // null for a base model
    const Capabilities& capabilities() const noexcept;
    const DeploymentPlan& plan() const noexcept;
    const ModelRecord& record() const noexcept;
};
```

- **Descent is one call** (`Direction.md` 4.2 rule 3): `model()` reaches everything `LanguageModel` exposes,
  including observation, and `as<GemmaModel<Cuda, BF16>>()` reaches the typed model and its components. A
  wrong type is null, never undefined behaviour.
- **A base model has no protocol.** `instruct` false means there is no template to render, so `protocol()` is
  null and `Mila::AI` refuses to converse with it -- by reading the handle, not by a family check.
- **The handle is small and holds no conversation.** It is what a model *is*; `Mila::AI` is what a program
  does with one.

`Mila/AI/` is a new library target, `Mila::AI`, in module `Mila.AI`, depending on `Mila::Mila` and never
imported by it (`Direction.md` 8.2). The wheel and a `FetchContent` consumer both receive it.

### 3.6 The factory

`open( name, request )`:

1. Locates `name` in the store. It never pulls (`ModelDistribution.md`: pull and load are separate verbs).
2. Resolves the record's architecture through the list (3.3); unknown is a refusal naming the list.
3. `Architecture::applyVariant( request, record.variant )`: the caller's request with the weights' format.
4. `Architecture::Model<D, P>::load( weights, request )`: plan, check, execute (`Deployment.md` section 7).
   A refusal propagates as `DeploymentRefusedError`; the application words it (`Deployment.md` section 6).
5. Loads the tokenizer, constructs the protocol when `instruct` is true, and reads the capabilities.

The factory takes a `DeploymentRequest` rather than a device from the first commit (`ROADMAP.md`), so the
planner is the handle's entry point and not a signature change later. A preview without loading --
`ModelHandle::planFor( name, request )`, returning the plans or the refusal -- is the same first three steps
and the family's `planDeployment`.

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

A developer's model has no manifest, so what the factory reads from the record the developer passes. What
`plan()` returns for a model the planner did not load is open (10.4).

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
  is proven byte-identical against the renderer it replaces, on recorded fixtures, before any caller
  switches. The Gemma fold's parity step (`+34`) is the recipe: declarations, calls, responses and answers
  identical; parsed arguments equal after decoding.
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
- **H5. Adding an architecture is one list entry.** GPT-2 is the proof: it has a model and a tokenizer and no
  protocol. Adding `Gpt2Architecture` touches its own directory and the list, nothing else; `gpt2-small`
  opens, generates, and reports no protocol.
- **H6. A composed network meets the contract.** A test builds a small `LanguageModel` subclass from
  components, adopts it with no registration, and generates through the handle.
- **P1. Protocol parity.** Each family's `Protocol` produces byte-identical prompts, declarations and parses
  to the renderer it replaces, on fixtures recorded from today's Chat and MIS.
- **Negatives.** N1: an architecture removed from the list -- opening its model is a refusal naming the list,
  not a misroute into another family (the "else means llama" defect MIS had). N2: `as<>` with the wrong type
  returns null. N3: the manifest default for `reasoning` set to false -- H3 fails for Gemma and Qwen.

---

## 7. Phasing

**Phase 1 -- capabilities.** The manifest fields and their defaults; `getDeploymentPlan` and
`supportsPromptPrefixReuse` onto the base. *Exit:* H3, H4, N3. Independent of everything else.

**Phase 2 -- the protocol.** The interface, Gemma and Qwen implementations over their free functions,
Llama's fold into the library, Gemma's renderers made one. *Exit:* P1 for all three families. No application
switches yet.

**Phase 3 -- identity, handle and factory.** The architecture list, the descriptors, `Mila/AI/` as a target
in the wheel and a `FetchContent` consumer, `ModelHandle`, `open`, `planFor`, `adopt`. *Exit:* H1, H2, H5,
H6, N1, N2.

The applications move in the Applications theme, onto `Mila::AI` rather than onto the handle, so each moves
once (`ROADMAP.md`, "one release rather than three"). The criterion "no dispatch site outside the handle
carries a per-family branch" is therefore met when they do, and `ModelVariant`, the per-family session
classes, `ModelFamily`, `FamilyTraits`' two weight facts, and Chat's and MIS's grammar copies are deleted
there.

---

## 8. Non-Goals

- Choosing a package for a base-model name, or naming an intent (`Direction.md` 6.5).
- Opening a loose weights file with no store record. `Direction.md` 4.2 rule 7: in v0.21.0 a model is named.
- Erasing device or precision (4).
- Conversation state, tool dispatch, the splice, and the autonomy policy: `Mila::AI` and the agent core.

---

## 9. Relationship To Other Specs

- `Direction.md` -- 3.2 and 5.1 are this spec's charter; 3.2's capability list is corrected by 3.2 here.
- `Deployment.md` -- the factory is `load( path, request )` behind a name; the plan is the handle's.
- `ModelDistribution.md` -- the manifest and the store the factory reads; gains two fields.
- `ChatConfiguration.md` -- section 5's `maximum_context_length` is replaced by the weights header (10.3).
- `PromptCaching.md` -- what `supportsPromptPrefixReuse` reports.
- `PythonBinding.md` -- the per-family sessions this replaces, via `mila.AI`.
- `ModelFamilyParity.md` -- what each family can do; its grammar rows are this spec's Phase 2.

---

## 10. Open Decisions

1. **The class name.** `ModelHandle` is the working name. `Mila::AI::Model` reads better in a program and
   collides in reading, not in code, with `Mila::Dnn::Model`. Recommend `ModelHandle` until `Mila::AI`'s spec
   shows how often a program types it.
2. **Chat's FP32 Llama.** The only reason Chat's variant has a fourth member. Keep it through a second handle
   instantiation, or retire FP32 inference from Chat. Recommend retiring it: every published model is
   BF16-compute.
3. **The context ceiling's one source.** Recommend the weights header (3.2), with `ChatConfiguration.md`'s
   `maximum_context_length` withdrawn and `FamilyTraits`' 131072 for Gemma deleted rather than moved. Before
   deciding, measure Gemma 4 12B above 131072 -- the 262144 in its header is HF's `max_position_embeddings`,
   and whether quality holds out there is a question the header cannot answer. If it does not, the header is
   what changes, at conversion.
4. **`plan()` for an adopted model.** The planner did not load it. Options: the handle's plan is optional; or
   `adopt` takes a request and plans the network it is given, since `planOnDevice` is already generic over the
   network. Recommend the second: it keeps rule 2 of `Direction.md` 4.2 true for every `AI`.
5. **The Python projection of descent.** `mila.AI` projects the contract; how far `model()` reaches from
   Python -- the `LanguageModel` surface, or also observation -- is `PythonBinding.md`'s to decide with
   `Mila::AI`'s spec.
6. **Gemma's effort scale.** Recommend it stays Chat's system-turn text (3.4). The alternative is a library
   "effort" option for Gemma that is a sentence, not a trained parameter, which is a prompt the library would
   be writing on the model's behalf.
