# Mila Direction: v0.21.0 and After

The positioning, layering and release plan for the releases after v0.20. v0.20 is the base this
builds on; it is not revised by this document.

Written 2026-09-19; adopted 2026-09-23, when section 5 became v0.21.0 and its open decisions were
settled (section 8). Supersedes `MilaProductFamily.md` for everything after the v0.20 tag. That spec
remains the definition v0.20 ships under. Releases step by one minor at a time; section 6 is the
releases after v0.21.0, numbered as each is scheduled.

---

## 1. The Trait

**Mila is for developers to harness intelligence.**

*Harness* means taking what a model can do and putting it to work inside your own program, under your
control. Every release from v0.21.0 on is judged by one question: does it let a developer do that more
directly, with less in the way and nothing sealed off?

Mila's version of harnessing has three properties that together are not available elsewhere. Each is
already true at the model level in v0.20; the releases below carry them up to the application:

| Property | What it means | Grounded in |
|---|---|---|
| **In process** | The model, the conversation and the tool loop live in your program. A tool result is spliced into the live KV cache and generation continues; nothing is re-sent or re-read. | No execution engine; token-level splice (`MilaProductFamily.md` Decided 1) |
| **Planned** | What a model will cost on your hardware is decided before anything is allocated, and you can read the decision. | `Deployment.md` |
| **Reachable** | From the application object, one call reaches the model, and from there every component, operation and kernel. | Explicit composition; Observability |

The principle that joins them: **hide complexity, never capability.**

### 1.1 The building blocks are never lost

Mila is built from explicit components, and developers build with them: `Linear`, attention,
normalization and the rest compose into a network in ordinary C++, which can then be trained, run and
inspected. That is what Mila is made of, and it stays a first-class way to use Mila in every release.

`Mila::AI` adds an entry at the top. It does not replace the one at the bottom. A developer can harness
a published model through `Mila::AI`, or build a network from components, and **the two paths meet**:
a model composed from components can be put to work through `Mila::AI` on the same terms as a published
one. Components are a public building surface, documented, tested and taught; they are never
demoted to an implementation detail of the models Mila ships.

---

## 2. What Changes From v0.20

v0.20's public message is *reference implementation*: an LLM stack written to be read. That message
was right for v0.20 and stays on v0.20's surfaces. It is too small an identity to build on, because it
describes the reader and not the builder.

| | v0.20 | v0.21 | After v0.21 |
|---|---|---|---|
| The developer | reads Mila | builds on Mila | delegates tasks to Mila |
| Enters at | a model: `GemmaModel<...>::load`, or components | `Mila::AI`, or components | `Mila::AI` with tools and a policy, or components |
| Headline claim | a 27B model on a 16 GB card, and you can read it (the 2.82-bit build that would make it 12 GB is held for the next release — `ROADMAP.md`) | a local model at work in your program in ten lines, and one call from there to the kernel | an agent finishes a multi-step task in your process without re-reading its history, and every step is auditable |

**What does not change.** Validation stays token-for-token against the reference implementation.
Readability becomes the guarantee under the trait: the reason a developer can trust what they harness
is that they can open it. Speed is still evidence, never the lead. Mila still runs on the hardware the
user already has, and it is still a library.

**The public message claims only what the tagged release ships.** v0.20's truthful-today rule carries
forward unchanged: the website and README move to the trait in the release that makes it true,
v0.21.0, and not before.

---

## 3. Layering

"Adaptor" is retired as a concept. It named Chat and MIS by what they were not (the library) and
sorted them by who closes the generation loop. From v0.21.0 the loop closes inside `Mila::AI`, and
Chat and MIS are applications built on it, in the same position as any developer's program.

```
Applications      Chat      MIS (wire)      your program      Python / other languages
                    \           |                |                /
                     +----------+----------------+---------------+
                                         |
Mila.AI            Mila::AI  -- conversation, tools, agent core, autonomy policy
                   Model handle + factory (architecture -> concrete type, from the manifest)
                                         |
Mila (Src)         planDeployment -> DeploymentPlan -> load
                   Models, Components, Operations, Tensors, Compute
                                         |
                   CUDA / CPU
```

### 3.1 Mila (`Mila/Src`)

Unchanged in role. It owns everything model-intrinsic and consumer-blind: the model as composable C++,
generation, sampling, the KV cache, the model's native grammar including the token-level splice, and
deployment planning. It knows nothing about conversations, tools, agents or applications. Its
components remain a public surface a developer builds with directly (1.1), not only the material the
shipped models are made from.

### 3.2 Mila.AI

A new library, a peer beside `Mila/Src` that depends on it and is never imported by it. This is the
"native agent core" `MilaProductFamily.md` left as an open decision, grown to own the application
contract. It holds:

- **The model handle and factory.** The one place an architecture name becomes a concrete type, reading
  capabilities (reasoning channel, vision, prefix reuse) from the model's manifest, never from its
  family (5.1). It replaces the three erasures that exist today: Chat's `ModelVariant`, the binding's
  per-family session classes and MIS's family enum.
- **`Mila::AI`.** The application-facing object.
- **The agent core.** Parse a tool call, dispatch it, splice the result into the open turn, continue.
- **The autonomy policy** (after v0.21.0).

`MilaProductFamily.md`'s standing rule, that there is no "runtime services" layer, still holds for
`Mila/Src`. Mila.AI is that layer, built on purpose as its own library, so that `Mila/Src` never
acquires it one convenience class at a time.

### 3.3 Applications

Chat, MIS, a developer's program and a Python script are the same kind of thing: a consumer of
`Mila::AI`. What makes each distinct is its own concern and nothing more. For Chat that is terminal
rendering and the human approval gate; for MIS it is the HTTP wire shapes and per-request
statelessness.

The v0.20 rule carries over with a new subject: **a gap between what `Mila::AI` can do and what an
application reaches is a defect in the application.** Nothing an application needs may be private to
it. If Chat does something a developer's program cannot, it belongs in Mila.AI.

MIS keeps its second job as the conformance oracle, and it becomes the control arm of the autonomy
measurement (section 6).

---

## 4. `Mila::AI`

### 4.1 Shape

```cpp
auto ai = Mila::AI::create( "gemma-4-12b-it-fp4" );

auto response = ai.respond( "Explain transformer attention." );
std::cout << response.text();
```

With tools, streaming, and a look underneath:

```cpp
auto ai = Mila::AI::create( "Qwen3.8-27B-cb2-3" )
    .withTool( readFile )
    .withTool( runTests );

ai.respond( "Why does the parser test fail?", onToken );

const DeploymentPlan& plan = ai.plan();   // what was decided, and what bound it
auto& model = ai.model();                 // the handle; from here, the typed model and its components
```

(Method names illustrate the shape; the object's name is decided in section 8.)

### 4.2 Rules

1. **Small.** `Mila::AI` is a contract, not a catalogue: create, respond (whole or streamed), tools,
   the conversation it holds, `plan()`, `model()`. A capability one model has and another lacks is
   discovered from the handle, never added as a method that throws on the others.
2. **It never decides a deployment.** Creation calls the planner and keeps the plan. There is one
   decision site, `Deployment.md`'s, and `Mila::AI` exposes its result rather than deriving one.
3. **Descent is one call.** `plan()` and `model()` are part of the contract, not a debugging aid. From
   the handle a developer reaches the typed model, and from the typed model everything v0.20 already
   exposes, including Observability's named activations.
4. **The forward pass stays explicit.** The handle erases one type at the moment a session opens: one
   virtual call per `respond`, none per layer or per token. Everything beneath it is compile-time
   dispatched exactly as today.
5. **Other languages project it; they define nothing.** Python gains `mila.AI` with the same contract
   and the same plan. The per-family session classes are retired. A .NET projection, if it comes,
   follows the same rule.
6. **A composed model is a first-class model.** The handle's contract is one a network built from
   components can meet, so `Mila::AI` can be created over a developer's own model as well as over a
   store name. The factory is a convenience for published models, never the only way in.
7. **A model is named.** In v0.21.0 creation takes a store name. Naming a base model and letting Mila
   pick the package, and naming an intent, are later questions (section 6.5).

---

## 5. v0.21.0 — Intelligence Your Program Owns

**The release makes one claim, and it is a pair.** A local model is at work inside your program in ten
lines of C++, *and* one call takes you from there to the kernel. The first half without the second is
a wrapper; the second without the first is v0.20.

It adds no new family. Every workstream proves the contract on the four families v0.20 already
validates, and the next architecture then arrives in one place (section 6.2). The same release
finishes Qwen 3.8 and Gemma 4 — including Qwen's dense members, which reuse Llama's blocks, and
Gemma's image input, which enters through an embedder rather than a tower — so the contract is proven
on families with nothing left over. The ROADMAP carries that half and its criteria.

### 5.1 One model handle

The foundation of Mila.AI. It starts with the manifest: capabilities (reasoning channel, context
limits, modality) are declared in the record, which is additive because the manifest tolerates unknown
fields and `instruct` already proves the pattern. Today `Chat.FamilyTraits.ixx` derives them from the
family. The factory then reads the record rather than switching on family. Design of record:
[`ModelHandle.md`](ModelHandle.md), which finds that not every capability belongs in the manifest.

**Streaming is not a manifest capability.** Whether a display can route a model's output token by
token depends on whether the application has written that path for the model's markers, not on the
weights; `Chat.FamilyTraits.ixx` records it as a fact about the display, and it stays with the
application.

*Success:* a new architecture is added in one place; MIS serves every architecture Chat does; no
dispatch site carries a per-family branch.

### 5.2 Deployment planning on one device

`Deployment.md` Phases 1 to 4: pricing without a bound context, the build executing the chunk it is
given, `planDeployment` on one device, then the binding and MIS. Phase 5 (devices) comes after
v0.21.0 (6.3).

*Success:* `Deployment.md` gates G1 to G4 and negatives N1 to N4; Chat's `"auto"` choices are
reproduced by the planner on the measured models before Chat's own code is deleted.

### 5.3 The agent core, with token-level splice

The in-process loop, extracted from Chat into Mila.AI, with the delivered splice that
`MilaProductFamily.md` decided on and deferred: tool-result tokens appended to the live cache, no
re-render and no re-tokenize. Qwen's refusal of prefix reuse is a model property the core reads from
the handle.

*Success:* across a multi-turn tool session, prefill tokens per turn equal the tokens the turn added,
measured, for every family that permits prefix reuse; for Qwen the refusal is reported, not
discovered.

### 5.4 `Mila::AI`

The object in section 4, on the handle, the plan and the core.

*Success:* one ten-line program runs Gemma, Llama and Qwen by changing only the name, streaming and
calling a tool; `plan()` and `model()` reach the objects that actually ran; and a sample creates an
`AI` over a network it composed from components itself, with no factory registration.

### 5.5 Chat and MIS rebuilt as applications

Both consume `Mila::AI` and nothing beneath it except through `model()`. Their directories leave
`Mila/Adaptors/` for `Mila/Applications/Chat` and `Mila/Applications/Server`.

*Success:* neither contains model-specific code; MIS's tool flows (the v0.20 foreign-harness criteria)
still pass unchanged against Codex and Claude Code.

### 5.6 A developer can start

The application developer is a new reader, and the barrier section 7 names is theirs first. A
QuickStart that builds a program against Mila from source (CMake `FetchContent`), creates an `AI`,
calls a tool and prints the plan; the Python QuickStart moves to `mila.AI`.

*Success:* both QuickStarts run from a clean machine with only the documented prerequisites.

### 5.7 The public message moves

README, website and `MilaProductFamily.md`'s public descendants are rewritten to the trait, in the
v0.21.0 release and not before.

*Success:* no public surface describes Mila as a reference implementation first, or uses "adaptor".

---

## 6. After v0.21.0 — Intelligence That Acts

**The claim is a pair.** An agent finishes a multi-step task on your machine, in your process, without
re-reading its own history, *and* every step it took can be audited. Autonomy without the audit is a
runaway; the audit without autonomy is v0.21.0.

These are the releases after v0.21.0, one minor at a time. Which subsections land in which release is
decided when each is scheduled, in `ROADMAP.md`.

### 6.1 Autonomy

The agent is `Mila::AI` with tools and an autonomy policy. It is not a separate product. The design is
already written in `MilaProductFamily.md` (Agentic) and carries over as specified: the policy is a
runtime object, not a template parameter; loop detection by repeated call identity and by no
observable delta; the shell as one built-in tool behind one guarded door; escalate-to-human as a tool;
interrupts on turn boundaries only; a persisted, versioned trace. Chat becomes `Mila::AI` under the
policy "escalate before every tool", so Chat and an agent differ by one value.

*Success:* the two-loop measurement from `MilaProductFamily.md` Decided 1 — the same model and tools
driven in process and through MIS by a foreign harness, on the same task set — reporting pass rate,
turns, **prefill tokens per turn**, wall-clock, and termination split into "stopped when done" and
"stopped when stuck".

### 6.2 The next agentic model

The selection rule is unchanged: the leading open models that suit an agentic workflow on hardware the
user has. The named candidates are Muse Glimmer 30B and a dense Qwen member; which ships does not
change the plan. Both break two v0.20 assumptions: a second family with a reasoning channel, and (for
Muse Glimmer) a vision tower feeding the text model. v0.21.0's handle is what lets each arrive in one
place.

*Success:* greedy decode token-for-token against the reference; image-conditioned generation validated
against the same oracle; the model passes the 6.1 measurement as an agent.

### 6.3 More than one device

`LayerSplit.md`, landing as `Deployment.md` Phase 5: the device list, automatic placement, and ranked
alternatives across device sets. Required by 6.2 wherever one card cannot hold the model.

*Success:* `LayerSplit.md` section 10 and `Deployment.md` G5.

### 6.4 Memory

An `AI` saves its session, the conversation and the KV cache (and Qwen's recurrent state), and
restores it later without paying the prefill again. Memory in Mila means the state the model already
has, kept; retrieval stores are an application's concern.

*Success:* a restored session produces the same next tokens as one that was never interrupted, with
zero prefill on restore.

### 6.5 Composition and selection

- **Two `AI`s in one process.** Cooperating agents on one machine is first a memory question, and the
  planner is the one thing that can answer it. This reverses `Deployment.md`'s non-goal "two models
  sharing devices in one process", on purpose and in that spec.
- **Naming a base model.** `Deployment.md` 12.1's recorded direction: the caller names "Gemma 4 12B" and
  the planner chooses among that model's installed packages. It needs a stated rule for ranking a
  quality difference against memory.
- **Naming an intent** (`"coding"`, `"assistant"`) ships only if each mapping is backed by an evaluation
  Mila publishes. Without one, an intent name is a promise to keep choosing the best model for a task
  indefinitely, which is the treadmill `Positioning.md` §3 warns against.

*Success:* two `AI`s planned jointly on one device set, each passing G1 against the joint plan; a base
model name resolves to the same package the objective ranks first.

---

## 7. Honest Risks

- **The barrier grows with the audience.** A reader will fight a toolchain to read something
  remarkable; an application developer will not. v0.20's toolchain (VS 2026 18.6.2+, CUDA 13, C++23
  modules, an NVIDIA card) is unchanged. C++23 module interfaces are not portable between compilers,
  so a prebuilt C++ distribution is not available the way a C library's is. Source consumption through
  CMake is the v0.21.0 answer; whether a C ABI at the `Mila::AI` boundary is needed is decision 3.
- **The field above the model is crowded.** "Create an assistant, give it tools" exists elsewhere. Mila
  is distinct only through section 1's three properties together. A v0.21.0 that shipped `Mila::AI`
  without the splice, the plan or the descent would be one more wrapper, and every workstream's success
  bar is written to prevent that.
- **The top of the stack takes the attention.** A new entry point draws the documentation, the samples
  and the public message toward itself, and the component path can erode by neglect without anyone
  deciding it should. 1.1 is the guard: each release's QuickStarts and samples keep a component-built
  path, and the tests for components are not traded for tests of `Mila::AI`.
- **Autonomy amplifies model weakness.** Carried from `MilaProductFamily.md` unchanged: guardrails and
  loop detection are what make an edge-sized model survivable unsupervised, not polish.
- **One developer.** Each release is sized to one maintainer. v0.21.0 is the largest yet — section 5
  and the completion of two families — and it is bounded by a fixed date whose scope drains rather
  than by its list (`ROADMAP.md`). The releases after it take section 6 a subsection at a time. An
  item that cannot name its success bar in this document does not enter any of them.

---

## 8. Decisions

Settled 2026-09-23, each on the leaning this section recorded, when section 5 became v0.21.0.

1. **The name of the object: `Mila::AI`,** in module `Mila.AI` — the working name, and the one the
   trait suggests.
2. **Where Mila.AI lives: `Mila/AI/`,** a peer of `Mila/Src` and `Mila/Bindings`, built as its own
   library target that the wheel and a `FetchContent` consumer both get.
3. **A C ABI at the `AI` boundary: decided during v0.21.0** against the QuickStart experience, shipped
   no earlier than the release after it. It would make .NET and other compilers' C++ straightforward,
   and `Mila::AI`'s rule 1 keeps it small enough to be cheap.
4. **Where Chat and MIS live: `Mila/Applications/Chat` and `Mila/Applications/Server`.** Samples stay
   teaching code; these stay maintained products.
5. **Sequencing against the ROADMAP: section 5 is v0.21.0,** together with the completion of Qwen 3.8
   and Gemma 4, before any new chassis — which agrees with the rule that the handle is a precondition
   for every new model. Muse Glimmer follows in ROADMAP's Future.
6. **Tool registration: a compiled-in registry** — a callable plus a schema, registered on the `AI`.
   Declarative subprocess tools alongside it remain open for section 6, carried from
   `MilaProductFamily.md` Open Decision 5.
7. **How a composed model meets the handle: a C++ concept** the model satisfies — no registration,
   checked at compile time — so a developer's network and a published one pass through the same check.

---

## 9. Non-Goals Through Section 6

- A remote or hosted model behind `Mila::AI`. Every `AI` is backed by Mila running locally.
- Serving many users: batching, scheduling, multi-tenancy.
- A model zoo. Selection stays the agentic-workflow-on-your-hardware rule.
- A throughput race. Speed remains evidence.
- Training through `Mila::AI`. Training stays at the component and model level.

---

## 10. Vocabulary

- **Library**, never runtime, engine, framework or substrate.
- **AI** is the application object. **Model**, **weights** and **package** keep their meanings from
  `CLAUDE.md`.
- **Application** replaces adaptor.
- **Harness** is a verb in the trait. As a noun it keeps its industry meaning (Codex, Claude Code), so
  Chat stops being "the chat harness" and becomes Chat, the application.

---

## 11. Relationship To Other Specs

- `MilaProductFamily.md` — v0.20's definition. Its Agentic design, measurement and open decisions carry
  into section 6.1 by reference; its adaptor layering is replaced by section 3.
- `Deployment.md` — Phases 1 to 4 are v0.21.0 (5.2), Phase 5 comes after it (6.3); its non-goal on
  shared devices is reversed by 6.5.
- `ModelHandle.md` — the handle and factory of 3.2 and 5.1.
- `ModelFamilyParity.md` — the definition of a finished family, which 5's "families that are finished"
  is held to, and the 16 GB reference card.
- `LayerSplit.md` — after v0.21.0 (6.3).
- `ModelDistribution.md` — the store `Mila::AI::create` names models from.
- `Observability.md` — part of what "reachable" means.
- `.internal/Marketing/Positioning.md` — governs v0.20's public surfaces; section 5.7 replaces it at
  v0.21.0.
