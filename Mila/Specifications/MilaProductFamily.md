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
replaces that human with a policy. The shared core (parse -> dispatch -> splice -> continue) is
the code that must not be written twice. Today it exists once, informally, inside the Chat sample.

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
should be entered eyes-open. It also couples the adaptor's viability to model quality (MTP / MoE
roadmap) without being blocked on it.

## Positioning

This aligns with Mila's stated purpose — a mastery project for C++ metal-LLM contributors, judged
by craft, not a vLLM/llama.cpp throughput competitor. "No hidden execution engine, every forward
pass explicit" is today a claim about the *model*. The agentic adaptor extends that same
philosophy *up through the loop*: no hidden engine, no hidden wire, the tool-call -> dispatch ->
cache-splice cycle all explicit and in one native artifact. That is where "there is nothing like
Mila" stops being aspirational.

This is a distinct track from the model-capability roadmap (MTP -> MoE). It is a product/adaptor
axis, and it can advance in parallel.

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
3. **Autonomy policy surface.** What is the minimal viable policy (termination + budget + semantic
   loop guard) that makes an edge-sized model survivable unsupervised?
4. **Guardrail model.** Capability allow-list, filesystem jail, dry-run, resource caps — what is
   the mandatory floor for a native binary with direct tool access?

## Related Specifications

- `GemmaChatProtocol.md` — the native token grammar (candidate home for the runtime-level spec).
- `ToolCalling.md` — planned tool-calling design.
- `MilaISCodexAgent.md` — the proven MIS wire-adaptor agentic loop (Codex).
- `PromptCaching.md` — the caching the wire adaptor fights and the native core exploits.
- `SpeculativeDecoding.md`, `TokenSampling.md`, `SlidingWindowKvCache.md` — runtime primitives the
  native core rides on.
