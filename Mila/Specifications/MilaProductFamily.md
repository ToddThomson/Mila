# Mila Product Family: Runtime Library + Adaptors

**Date:** 2026-07-06 (locked 2026-07-07)
**Status:** Locked positioning for the v0.20 first production release — stable in shape,
refinable in detail. The Definition below is the product identity v0.20 ships under.
**Component:** whole-project positioning (`Mila/Src`, `Mila/Adaptors/Inference`, `Mila/Adaptors/Chat`, future agent)

## Definition

**Mila is an inference runtime library for single-user edge devices, plus a small family of
adaptors distinguished by who closes the generation loop.**

The runtime owns everything model-intrinsic and consumer-blind: the model as ordinary composable
C++ (no execution engine), generation, sampling, KV cache, and the model's native token grammar —
including token-level splice into the live cache. A native agent core above it owns the in-process
loop (parse -> dispatch -> splice -> continue) with a session-warm cache. Three adaptors consume
it: **MIS** exports the loop over the wire to foreign harnesses — interop and a ruthless
validation oracle, at O(context) per turn, an accepted cost, not a flaw; **Chat** closes the loop
in-process with a human in the gate; **Agentic** closes the loop on itself under an explicit
autonomy policy.

Mila is a **library, not a framework**: the application owns `main()`, the loop, and the tools;
Mila makes the model a first-class object inside them. (The tempting analogy — "a game engine for
inference" — points the wrong way: engines own the loop and call your code. Mila is raylib, not
Unity.) The corollary is a standing boundary rule: the runtime accepts only what is model-intrinsic
and consumer-blind. There is no "runtime services" middle layer for chat sessions, conversation
history, or prompt builders — that is how a hidden execution engine accretes one convenience class
at a time. Consumer state lives in the native agent core; policy, UI, and tools live in the
adaptors.

Mila is judged by craft and by the structural advantages of erasing boundaries — not by throughput
leaderboards, model breadth, or ecosystem compatibility.

## Release Boundary (v0.20)

The first production release ships **the runtime plus its two proven adaptors** — Chat and MIS —
with the Definition's claims demonstrable: clone the repo, build it, load Gemma 4 12B FP4 on a
12 GB card, chat through the harness, drive it from a foreign harness through MIS, and read the
entire path from prompt to kernel with no hidden engine. Comprehensibility is part of the
deliverable, not an aside: the release includes a guided reading path that lets a strong C++
developer trace one token's journey (embed -> attend -> sample -> decode) through the actual
source unaided.

**Explicitly post-release:** the Agentic adaptor, the extraction of the native agent core, and the
*delivered* token-level splice. Token-level splice is decided as the direction (see Decided); it
does not have to be built for v0.20. Agentic is the most tempting creep vector in this document —
it has no bounded checklist and real engineering risk (see Honest Risk) — and admitting it into
the release scope is how the release never ships. The grammar-in-runtime consolidation is the one
item from this spec that belongs in v0.20, because it is a correctness fix for drift already
shipped in two adaptors; its depth within v0.20 (canonical C++ consumed by Chat first; pybind for
MIS if bounded) is scoped at execution time.

## Purpose

This document gives shape to what Mila could become as a *product family*, without yet
committing to a milestone. It captures an architectural thesis reached in discussion: Mila is
not "an inference server." It is an **inference runtime library** with a small number of
**adaptors** layered on top, distinguished by who consumes the generated tokens. Writing it down
so the direction is refinable rather than re-derived each time.

It is a positioning + layering spec. It intentionally does not prescribe APIs; it fixes the
boundaries those APIs must respect.

## Thesis

The frontier stack splits the model behind an HTTP wire protocol because it *has to*:
multi-tenant GPUs behind a scheduler, a harness written in another language on another machine,
weights sold as access rather than shipped, decoupling for hot-swap. Every one of those reasons
evaporates on a single-user edge device. There, the wire protocol is pure impedance matching.

What makes Mila unusual is not that it runs on the edge — `libllama` and MLC-LLM do too. It is
that Mila has **no execution engine**: the model is ordinary C++ you compose in a translation
unit, not a runtime graph fed a config. So the agent loop and the model are the *same kind of
object*, composable in one program. The boundary that can be deleted is not just the HTTP hop but
the engine/runtime indirection beneath it. That combination is the "there is nothing like Mila"
claim, and it only becomes true (rather than aesthetic) when an adaptor closes the loop natively,
at the metal.

## The Consumer Axis

The runtime generates tokens. It does not know or care who reads them. An **adaptor** bridges
those tokens to a specific consumer. The consumer is the distinguishing axis:

- **MIS** -> a **machine**. Tokens <-> JSON blocks over HTTP/SSE. Consumer is a foreign harness
  (Codex, Claude Code). The loop is *delegated out* of the process; MIS is stateless per request.
- **Chat** -> a **human**. Tokens <-> rendered terminal text. Consumer is a person at a prompt.
  The loop is closed *in-process*, with a human in the gate.
- **Agentic** -> **no external consumer**. The loop closes on itself. Tool calls dispatch, results
  splice back, the model decides the next move. The tokens *are* the control flow. Unsupervised,
  in-process, token-level, on-device.

## Layering

The three adaptors are not flat peers. Sort them by *who closes the loop*:

```
Runtime Library  (model, generation, sampling, KV cache, native token grammar)
   |
   +-- MIS ................ loop delegated OUT to a foreign harness (wire, stateless)
   |
   +-- Native Agent Core .. loop closed IN-process
   |        (parse tool call -> dispatch -> splice result -> continue; cache-warm)
   |
   |     +-- Chat ......... + human gate + TUI      -> supervised
   |     +-- Agentic ...... + autonomy policy       -> unsupervised, "at the metal"
```

MIS sits on its own branch: it *exports* the loop to whoever holds the wire. Chat and Agentic
share a **native agent core** and differ only by the gate — Chat has a human in it; Agentic
replaces that human with a policy. Once escalate-to-human is a tool (see Agentic), the gate is not
even a structural difference: Chat is the core under a policy that escalates before every tool. The
shared core (parse -> dispatch -> splice -> continue) is the code that must not be written twice.
Today it exists once, informally, inside the Chat sample.

## Layer Responsibilities

### Runtime Library (`Mila/Src`)

Owns everything model-intrinsic and consumer-blind:

- Model, generation loop primitive, sampling, KV cache (incl. sliding-window / bounded ring).
- **The native token grammar.** How Gemma emits and consumes `<|turn>` / `<|channel>` /
  `<|tool_call>` / `<|tool_response>` / `<|"|>` is a property of *the model*, not of HTTP and not
  of the terminal. Both consumers need to parse and format it identically.

> **Key correction this spec makes:** the tool-call grammar currently lives in *both adaptors*,
> reimplemented — C++ `GemmaToolCallParser` in Chat, Python `gemma_protocol.py` in MIS — and they
> have already drifted (MIS renders the `<|"|>` string delimiter; Chat still renders plain
> quotes). That divergence is not a bug to patch; it is the layering telling us the code is in the
> wrong place. The grammar belongs **down** in the runtime (canonical C++, exposed via pybind so
> MIS consumes the same source), after which the adaptors *cannot* diverge. This supersedes the
> earlier "fold gemma_protocol toward the chat harness or a shared spec" thread: fold it toward
> neither adaptor — fold it into the runtime. See `GemmaChatProtocol.md`, `ToolCalling.md`.

### Python binding surface (`Mila/Bindings`)

A runtime-adjacent projection of the runtime into Python (`mila.pyd`, module `Mila.Bindings`):
`Tokenizer` / `LlamaSession` / `GemmaSession` — load, generate, stream, config. It is
**consumer-blind** (no HTTP, no chat, no protocol) and is therefore a peer of the runtime, not
an adaptor — the same consumer-blind test that keeps the grammar in `Src`. It has two consumers,
which is why it is not owned by either: the MIS server imports it, and the HuggingFace parity /
converter tooling imports it directly (it is `GemmaSession`'s primary consumer). Decided
2026-07-07 to promote it out of `Adaptors/Inference` to a first-class `Mila/Bindings`.

### MIS — wire adaptor (`Mila/Adaptors/Inference`)

Owns only what is genuinely wire-specific: HTTP/SSE transport, OpenAI/Anthropic block shapes
(Responses items, Anthropic `tool_use`/`tool_result`), per-request statelessness, protocol
translation, bind/auth. Value: drive Mila with a best-in-class harness you did not write, and use
those harnesses as a ruthless validation oracle. Cost: the client resends full history each turn,
so MIS re-renders and re-prefills the whole context every turn — structurally O(context) per turn,
and it fights prompt caching. See `MilaISCodexAgent.md`, `PromptCaching.md`.

### Native Agent Core (shared by Chat + Agentic; to be extracted)

The in-process loop: stop generation at `<tool_call|>`, parse the native call, dispatch, splice
the result back into an OPEN model turn, resume. Mila already does the intra-turn splice via the
`continue_open` trick. The core holds the model and its KV cache **warm across the whole session**
and prefills only the delta — O(new tokens) per turn, not O(context). This is the concrete,
quantifiable advantage the wire cannot match, and it is structurally unavailable to a stateless
protocol.

### Chat — human adaptor (`Mila/Adaptors/Chat`)

Native agent core + human gate (tool approval) + TUI (`ConsoleRenderer`, channel-aware streaming).
Supervised: a person judges done-ness and catches thrash.

> **Status resolved (2026-07-07):** Chat is a first-class adaptor, a peer of MIS under
> `Mila/Adaptors/` — no longer a `Samples/` demo. It is a maintained surface: it gains rigor and
> tests over time and its "freely editable scratch" boundary is retired. (This does not conflict
> with extracting the shared native agent core post-release; the core lands *beneath* Chat, which
> is already first-class in the release.) See Decided item 2.

### Agentic — autonomous adaptor (future)

Native agent core + **autonomy policy**, no human gate, no presentation layer. The product vision:
a single native artifact — model, loop, and tools fused, no engine, no wire — running a task
autonomously on-device, offline, private, with microsecond tool dispatch. This is the adaptor that
makes the thesis shippable.

What it must own that neither MIS nor Chat fully does:

- **Autonomy policy.** Termination conditions (done? stuck? budget blown?), step/token budgets,
  and *semantic* loop-detection ("I have re-read this file five times") — distinct from the
  existing token-repeat degeneration backstop.
- **Guardrails at the tool boundary.** With no human approving tools, sandboxing, allow-lists,
  filesystem jails, and resource caps must be structural. "At the metal" cuts both ways: a native
  binary with direct exec/syscall-level tool access is exactly as dangerous as it is fast.
- **Observability without a watcher.** A structured trace of (thought -> call -> result ->
  decision) for post-hoc audit and debugging — the honest version of what MIS's SSE stream was
  faking for a human.
- **Result extraction.** The agent must produce an artifact (a patch, an answer, a side effect)
  and *know* it is finished.

**What a tool is.** Three shapes are available, and the choice decides whether "microsecond tool
dispatch" survives contact with a real task. Compiled-in C++ functions — a callable plus a schema,
registered at core construction — deliver the claim, at the cost that adding a tool means a rebuild.
Subprocess tools declared in a manifest are what every other agent does; they forfeit the claim.
Dynamically loaded modules are ruled out: ABI surface against a C++23 module library, for a middle
ground neither end wants. *Leaning:* the compiled-in registry, with **the shell as one built-in
tool** rather than as the tool mechanism. Everything that matters for latency (file read, grep,
patch apply) stays a native call; open-ended capability arrives through a single door. The dividend
is the guardrail model: with one door to the host, the jail, the allow-list, and the resource caps
have exactly one place to live instead of being cross-cutting.

**The autonomy policy is a runtime object, not a template parameter.** Mila's idiom points the other
way — `TWeightQuantization`, `TKvPolicy` and `TRopePolicy` train the hand to reach for
`Agent<TAutonomyPolicy>`. Compile-time policies exist here because they change types and select
kernels. An autonomy policy does neither, and its content — step budget, token budget, allow-list —
is per-task data that differs between two runs of the same binary. It is an object with a decision
point (`shouldContinue` over the step trace), and templating it would buy nothing while making the
budget a compile-time constant.

**Semantic loop detection, concretely.** The phrase reads as an open research problem; most of it is
mechanical. Three signals, cheapest first: **repeated call identity** — hash `(tool, normalized
args)` and treat a repeat with no intervening state change as stuck; **no observable delta** — a run
of N steps in which no tool changed anything (no file written, no new bytes read, no exit code
moved) is a stuck signal regardless of what the model believes about its progress; and **thought
similarity**, which needs embeddings and a forward pass per step and is therefore deferred. The
first two catch the observed perseveration failure without any semantic machinery.

**Unsupervised is not uninterruptible.** Removing the human gate does not remove the human. Three
requirements follow. An interrupt must land on a **turn boundary, never mid-splice** — otherwise the
KV cache is left describing a turn that did not happen. The trace must be **persisted, not
in-memory**: for an unsupervised run it is not a debugging aid but the only artifact a human ever
sees, which makes its format a versioned deliverable rather than a logging concern. And
**escalate-to-human is itself a tool**, available under any policy. That last one collapses a
distinction: Chat is then the native agent core under the policy "escalate before every tool", so
the gate is one policy value rather than a structural difference between two adaptors.

## Forcing Functions (why the order matters)

1. **Token-level splice becomes mandatory.** For Chat, splicing tool-result tokens straight into
   the live KV cache (no re-render, no re-tokenize) is an optimization you could defer. For an
   autonomous loop that may run hundreds of turns with no human to end it, re-prefilling each turn
   is fatal. Building Agentic *is* committing to token-level splice in the runtime.
2. **Grammar-in-runtime pays maximum dividends at the agentic layer.** The agentic adaptor is
   almost nothing but grammar + dispatch + cache. If the grammar is in the right layer, this
   adaptor is thin; if it is not, the adaptor re-implements it a third time.
3. **Host-out-of-the-loop, extended one level up.** The D1 decode-ahead work got the host out of
   the *per-token* path. The agentic loop wants the same discipline on the *per-turn* path: tool
   dispatch becomes the one deliberate place the loop touches the host, and even that is a native
   call, not a process or network hop.

## Honest Risk

**Autonomy amplifies model weakness.** A 12B FP4 model driving itself with no human gate is a
different reliability regime than Claude driving it through MIS — one bad `<|tool_call>` cascades
with nobody to interrupt. Gemma has been observed perseverating on a single label for ~1000
tokens; in supervised Chat that is an annoyance, in an unsupervised loop it is a runaway. So the
guardrails and semantic loop-detection above are not polish — they are what make an edge-sized
model *survivable* in autonomy. This is the central engineering risk of the agentic adaptor and it
should be entered eyes-open.

**The model coupling is tighter than "advances in parallel"** (revised 2026-08-12). Muse Glimmer 30B
is the post-v0.20 target, and it was picked because it is tuned for tool use, long tasks and failure
recovery — the model roadmap's next step was chosen to serve *this* adaptor. Its DFlash drafter head
also makes speculative decoding worth more here than in Chat: a hundred-turn unsupervised run has
nobody waiting on the first token. The blocker is neither grammar nor loop but hardware — ~16 GB at
FP4 before any KV cache, against a 12,282 MiB card — which puts the compute sponsorship ask on this
adaptor's critical path rather than beside it.

## Positioning

This aligns with Mila's stated purpose — a mastery project for C++ metal-LLM contributors, judged
by craft, not a vLLM/llama.cpp throughput competitor. "No hidden execution engine, every forward
pass explicit" is today a claim about the *model*. The agentic adaptor extends that same
philosophy *up through the loop*: no hidden engine, no hidden wire, the tool-call -> dispatch ->
cache-splice cycle all explicit and in one native artifact. That is where "there is nothing like
Mila" stops being aspirational.

The adaptor axis and the model-capability axis are separable in code and converged in purpose: the
work advances in parallel, but the next model target was chosen for this adaptor and the hardware
that would validate it is shared. See Honest Risk.

## Decided

1. **Grammar depth: token-level splice** (decided 2026-07-07). The runtime exposes the grammar not
   merely as parse/format helpers over strings but a level deeper — tool-result tokens appended
   straight into the live KV cache. String helpers alone would leave Mila as "llama.cpp with nicer
   C++": a craft difference, not a structural one. The structural claim — session-warm cache,
   O(new tokens) per turn against the wire's O(context) — is measurable, demonstrable, and
   unavailable to any architecture with a serialization boundary. Everything else in this spec
   (Agentic's viability, Chat's thinness, the "at the metal" positioning) already assumed this
   answer; it is now explicit. String-level parse/format helpers still exist as the first
   deliverable and as the surface MIS consumes via pybind — the splice is the depth they are built
   on, not an alternative to them.

   **How the structural claim gets measured** (added 2026-08-12): same model, same tools, two loops
   — Agentic in-process against that identical model driven through MIS by a foreign harness on the
   same task set. Same weights, so the experiment isolates the loop, which is the only thing the
   adaptor is. The reported numbers are task pass rate, turns to completion, **prefill tokens per
   turn** (where O(new tokens) against O(context) either shows up or does not), wall-clock, and
   termination correctness split two ways — stopped when done, stopped when stuck. MIS is already
   the correctness oracle; this makes it the control for the loop as well.

2. **Chat is a first-class adaptor** (decided 2026-07-07). Chat and MIS are peer adaptors under
   `Mila/Adaptors/`; Chat is no longer a `Samples/` demo. It becomes a maintained surface (gains
   tests and rigor; the "freely editable scratch" boundary is retired) and builds under its own
   `MILA_ENABLE_ADAPTORS` gate rather than `MILA_ENABLE_SAMPLES`. This resolves Open Decision 1.
   It is orthogonal to the native-agent-core extraction, which remains post-release and lands
   *beneath* Chat — Chat being first-class now is what the v0.20 release ships.

## Open Decisions (to refine on return)

1. **Chat's status.** RESOLVED 2026-07-07 — Chat is a first-class adaptor under `Mila/Adaptors/`,
   a maintained surface, no longer a throwaway sample. See Decided item 2. (The native agent core
   extraction, formerly bundled into this decision, is a separate post-release item; it lands
   beneath the now-first-class Chat rather than replacing it.)
2. **Where the native agent core physically lives.** Runtime-adjacent library, or a peer component
   the two native adaptors depend on? *Leaning:* runtime-adjacent — a peer library depending on
   the runtime, never inside it. Same consumer-blind test as everywhere else: the core knows about
   sessions; the runtime must not.

   **It has a second tenant, and a hard sequencing constraint (agreed 2026-08-12).** The same layer
   should own **one typed model handle and one factory** mapping an architecture to its concrete
   instantiation. That erasure exists three times today, in two languages — Chat's `ModelVariant`
   (`Chat.ixx`), the binding's `LlamaSession`/`GemmaSession`, and MIS's `ModelFamily` enum — and the
   duplication is not theoretical: GPT-2 is absent from MIS *because the second erasure was never
   written for it* (`Server/model_worker.py:40`, "gpt2 has a record shape and no session"), a gap
   that reads as a policy.

   The cost is not the type list, which grows linearly under the crest-not-zoo selection rule. It is
   the **six `std::visit` sites, today carrying zero `if constexpr`** — uniform only while every
   model does the same things. The first chassis with a capability the others lack (a vision tower,
   MTP) makes every one of them conditional, in each of the three places.

   **Sequencing: after the v0.20 tag, and BEFORE the next chassis expansion.** Before, because the
   chassis is what turns three erasures into four and six clean visits into six conditional ones.
   After the tag, because it is structural rather than hardening. And after the manifest carries
   model capabilities (a v0.20 item), so the factory reads them from the record instead of baking a
   fourth family switch into the thing built to remove family switches.

   This does not touch the thesis: "every forward pass explicit" is a claim about the forward pass,
   and a handle erases one type at the session boundary — one virtual call per `generate()`, none
   per layer or per token, everything inside still compile-time dispatched. Nor is it the
   string-keyed `OperationRegistry` being phased out; that is per-operation dispatch, a different
   layer, and `familyFromArchitecture` is already this pattern at model level.
3. **Autonomy policy surface.** What is the minimal viable policy (termination + budget + semantic
   loop guard) that makes an edge-sized model survivable unsupervised? *Narrowed 2026-08-12:* the
   policy is a runtime object rather than a template parameter, and the loop guard's first two
   signals (repeated call identity, no observable delta) are settled as mechanical. Open: the
   budget defaults, and whether the policy owns result extraction or merely tests for it.
4. **Guardrail model.** Capability allow-list, filesystem jail, dry-run, resource caps — what is
   the mandatory floor for a native binary with direct tool access? *Narrowed 2026-08-12:* if the
   shell is one tool rather than the tool mechanism, the floor is that tool's argument validator
   plus a jail rooted at a declared working set. Open: whether write access is dry-run by default.
5. **Tool registration shape.** Compiled-in registry (leaning), subprocess manifest, or loadable
   modules — see Agentic. The leaning makes adding a tool a rebuild; open whether that is
   acceptable for the artifact Agentic is meant to be, or whether a declarative subprocess tool
   is needed alongside the native ones.

## Related Specifications

- `GemmaChatProtocol.md` — the native token grammar (candidate home for the runtime-level spec).
- `ToolCalling.md` — planned tool-calling design.
- `MilaISCodexAgent.md` — the proven MIS wire-adaptor agentic loop (Codex).
- `PromptCaching.md` — the caching the wire adaptor fights and the native core exploits.
- `SpeculativeDecoding.md`, `TokenSampling.md`, `SlidingWindowKvCache.md` — runtime primitives the
  native core rides on.
