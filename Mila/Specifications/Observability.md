# Observability

**Status: design, opened 2026-08-24. Carved into v0.20.**

This answers the positioning question `TransformerApiReadiness.md` item 6 parked the same
day: *"needs a positioning decision on whether observability is a Mila feature at all."*
It is one, and it is close to the centre rather than at the edge.

The scope line in that review reads *"Out of scope — new design. Anything that makes the
surface bigger is a feature and waits, however legitimate the need. Observability is the
live example."* That line still governs the rest of the review. This document is the
declared exception to it, and Section 10 states the exception's boundary so it stays one.

---

## 1. Why this is not a diagnostic

Mila's positioning is that it is the stack you *read* — every forward pass explicit, no
hidden execution engine, device and precision visible as types. A reference implementation
that cannot be looked into is a contradiction in terms, and today Mila cannot be looked
into by anyone holding a model.

The tax is documented, not felt. Four independent investigations each paid it:

- **The Qwen Phase 4 rotary defect** was localized by comparing intermediates against a
  reference, and every piece of that machinery was built outside `Mila/Src`. A consumer
  debugging their own model cannot reproduce any of it. (`BACKLOG.md:26`)
- **The Qwen attention intermediates** were reachable only because the test harness owns
  `QwenAttentionBlockWorkspace`. DeltaNet blocks self-allocate, so an entire block kind —
  48 of 64 layers on the target model — is opaque to every caller.
- **The memory diagnostics** build a transformer *directly*, with no weight load, because a
  loaded model cannot be walked.
- **Gemma's NaN fingerprint** reaches its probe through `if constexpr ( requires { ... } )`,
  so a signature change silently stops the probing rather than failing to compile.

Each was solved by scaffolding. None of the scaffolding is available to a user, and none of
it survived into the library.

## 2. The benchmark

The bar is not "Mila has a hook." The bar is what a PyTorch user does without thinking
about it.

| PyTorch | Mila today |
|---|---|
| `named_modules()` | `getComponents()`, `findComponent("gpt2.lenc.wte")` — public, `CompositeComponent.ixx:284`, `:383` |
| `named_parameters()` | `getParameters() -> std::vector<ITensor*>`, `getGradients()`, `getParameterNames()` — public, `Component.ixx:705`, `:718`, `:664` |
| `tensor.shape` / `.dtype` / `.device` | `ITensor` — entirely public, including `rawData()`; `ITensor.ixx:39` |
| `print(model)` | `toString()` per component; `CompositeComponent` renders one level |
| `register_forward_hook(fn)` | nothing at component level |
| `output_hidden_states=True` | nothing |

## 3. The audit: most of the spine is already built, and already public

This is the finding that sizes the work. The hard part of exposing a compile-time-typed
tensor library — type erasure — is **already solved**. `ITensor` is a fully public interface
carrying shape, element count, dtype and dtype name, device id and device type, storage
bytes, a stable identifier, an optional name, the owning memory resource, and a raw pointer
(`ITensor.ixx:39-162`). Nothing new is needed to hand a tensor of unknown precision to a
consumer.

Traversal is likewise done. `CompositeComponent` publicly exposes dotted-path resolution
with absolute and relative forms (`findComponent`, `:284`), existence (`hasComponent`,
`:373`), and ordered children (`getComponents`, `:383`). `Component` publicly exposes name,
type, device, parameter and gradient tensors, parameter names, parameter count, measured
memory statistics and predicted memory (`Component.ixx:517-718`, all above the `protected:`
at `:720`).

**So the "locked door" is one accessor.** `LanguageModel::getNetwork()` is protected
(`LanguageModel.ixx:257`), and the model's entire public surface is `generate`,
`savePretrained` and `seedSampler`. A consumer holding the object the library asks them to
hold cannot reach a tree that is otherwise fully public and fully navigable. That is why
every diagnostic in this repository constructs a transformer directly instead of loading a
model — not because introspection is missing, but because the handle users actually have
cannot reach it.

## 4. What is genuinely missing

1. **Reachability.** The door above.
2. **Activations.** `getParameters()` exposes what a component *holds*. Nothing exposes
   what *flows through* it. This is the only new mechanism in this document.
3. **A composed structural view.** The per-component pieces exist; nothing renders the tree
   in one call. See Section 6 for what Mila's version can carry that PyTorch's cannot.
4. **Extent** — not an observability problem. Section 9.

## 5. The constraint that shapes the design

**`Component` declares no virtual `forward`.** Neither does `CompositeComponent` or
`Network`; the review confirmed the only virtual `forward`/`backward` pair in the hierarchy
is the token-typed one on `LanguageModelNetwork` (`TransformerApiReadiness.md` item 8). Each
component kind has its own signature over its own types.

The consequence is decisive: **activation observation cannot be implemented by intercepting
a base-class call.** There is nothing to wrap. Hooks must be *emitted* by the component
that owns the value, not *installed over* it.

This is the right answer on its own merits, and it is why the current probe is shaped the
way it is. When a component publishes a stage, the stage becomes a declared property of that
component — named deliberately, documented on the symbol, testable — rather than an accident
of where an interceptor happened to sit. It also fixes by construction the review's specific
complaint that the existing stage vocabulary (`"embedding"`, `"layer_N"`) is undocumented and
defined at the call site.

The cost is that instrumentation lands component by component. The tree already has a
precedent for exactly this: `getRequiredMemory` is non-pure "only so the contract can land
family by family" (`Component.ixx:599-608`). Observability follows that pattern.

## 6. Design

### 6.1 The inspection door is its own door

**Decided 2026-08-24: `getNetwork()` stays protected for v0.20.** `LanguageModel` instead
gains an inspection surface of its own — describe the composition, resolve a path, attach an
observer. Recorded at length because the opposite choice is one line of code and will be
proposed again by anyone who has not seen the reasoning.

Behind the accessor sit `forward`, `backward`, `prefill`, `prefillFrom`, `decode`,
`setStageProbe` and `rewindKvCache`, plus everything inherited from `Network`,
`CompositeComponent` and `Component`.

**The case for making it public is not weak, and it does not lose on principle.** Mila's
pitch is explicit composition and no hidden execution engine, so protecting the network reads
as hiding the engine; every other member of the hierarchy is public, and this is the one
special case users actually hold. There is also a real need the inspection door does not
serve — a user writing their own generation loop, sampler, or speculative-decoding experiment
needs `prefill` and `decode`, and "use our loop or nothing" is not a reference-implementation
posture. The usual argument for a narrow surface, that it cannot be changed later, is weak
here because backward compatibility is not a goal.

It loses on timing, and on a fact found while making the case.

**There is no read-only middle.** The obvious compromise — a `const` accessor giving full
read access and no mutation — does not work. `ComponentPtr` is
`std::shared_ptr<Component<...>>` (`CompositeComponent.ixx:68`), non-const, and
`getParameters()` returns `std::vector<ITensor*>`, also mutable. Constness leaks at the first
traversal step. So the compromise is not one line — it requires const-ifying the traversal
API first, a larger change than the thing it enables — and, more importantly, **exposing the
network exposes mutable access to every component and every parameter tensor**, not merely to
prefill and decode. A caller can `loadParameter` into a live model or write through
`rawData()`. The surface is materially wider than its member list suggests.

**Timing is decisive.** Four of those seven members are under active revision in this same
release: `forward`/`backward` (`TransformerApiReadiness.md` item 8, delete from the base),
`prefill`/`prefillFrom` (item 7, collapse to one), `setStageProbe` (item 6), and the missing
`resetKvCache` (item 2). Publishing a surface while subtracting from it in four places is
wrong under either answer to the principle — and the review exists precisely because what
ships in v0.20 becomes what people read, quote and build against.

**The invariant argument is real but secondary.** The network is mid-flight state, not a
passive object: `prefillFrom` requires positions `[0, start_offset)` already resident and does
not verify it, `decode` trusts the caller's position against actual cache state, and
`logits_ptr_` is borrowed until the next call. Misuse is silent wrong output rather than an
exception. That argues for designing the entry point deliberately, not for hiding it forever.

**The root cause is the absence of an inspection surface, not the access modifier.** Making
the network public would fix the symptom by exposing everything, including a mutable parameter
surface nobody designed. The inspection door is the designed fix for the complaint actually
raised, which was read-only from the start.

**Cost of this choice, stated rather than discovered.** If the network later goes public there
are two paths to the tree, and a synonym is a defect. The mitigation is that the inspection
door is deliberately not a second traversal API: it resolves and describes, and hands back the
same `Component` and `ITensor` types everything else already uses.

**Revisit in v0.21**, once items 2, 7 and 8 have landed. Publishing is purely additive at that
point, and the user-written generation loop can shape the entry point instead of being served
by accident.

### 6.2 View — the structural half

A recursive description of the composition: path, component type, children, parameter
tensors with shape and dtype, and both memory numbers — `getMemoryStats()` for what is
allocated and `getRequiredMemory()` for what was predicted.

Two properties Mila's view can carry that PyTorch's has no place for, and they are the ones
that make it a reference implementation rather than a model printer:

- **The per-role quantization policy**, which on Qwen differs by role within a single block.
- **The `OperationTraits` specialization actually selected** — the dispatch decision itself,
  resolved at compile time and invisible in any dynamic framework.

Rendering is a consumer's business; the library returns the description.

**Stated limitation: the view cannot report a component's signature.** `getParameters()` says
what a component *holds*; nothing on `Component` says what *passes through* it — no input
arity, no input or output shapes, no dtype flow. So `describe()` can report parameters, memory
and policy, and cannot answer "what does this take and return", which is among the first
things a reader wants. This is not a parity gap, since PyTorch cannot answer it either; it is
a place Mila could exceed the benchmark and currently does not. The shapes are knowable rather
than absent — `BuildContext` carries them at build time. Whether the signature becomes part of
the component contract is a larger question than the view, and is **deferred to v0.21** (11.1).
Note the output half of that question is already answered here: `getOutputs()` returns real
tensors, so what the view cannot report is inputs specifically.

### 6.3 Observe — the activation half

A component publishes named stages. An observer attaches at the model by path or pattern and
receives `(path, pass, stage, const ITensor&)`.

`ITensor` is the payload precisely because it is already public and already erased — an
observer written against it works across every precision and device without templates, which
is the property that makes this usable at all.

**Every compute path publishes, and the record says which one did.** Not publishing on every
path is the existing probe's defect exactly — *"it fires on prefill and not decode with nothing
saying that is intentional"* — so a value that first goes bad during generation is invisible.
Gelu has only `forward`, but leaf components elsewhere have more: `rope_->prefill`,
`conv_qk_->prefill`, `delta_rule_->prefill` and `attn_->prefill` all appear in
`Qwen.AttentionBlock.ixx:836`, each with a decode counterpart. A component therefore publishes
`"output"` from two or three different passes, and `(path, stage)` alone cannot tell them
apart.

Inferring the pass from the shape is not an answer — `[B, 1, D]` is decode *or* a one-token
prefill. Four consumers need the distinction outright: a prefill-versus-decode parity harness,
which exists precisely because those paths differ numerically (Section 9); the perplexity
consumer, which wants prefill only; a NaN hunt during generation, which wants decode only; and
any cross-pass equivalence test.

```
enum class ComputePass { Forward, Prefill, Decode, Backward };
```

**Declare the enum complete now and implement inference only.** `Backward` stays out of v0.20
per Section 10, but naming it here makes adding it additive rather than a signature change —
and *"a signature change silently stops the probing"* is the exact failure mode
`TransformerApiReadiness.md` item 6 records against the current mechanism.

The pass is an enum rather than a prefix on the stage name. `"prefill.output"` would be
stringly typed, would force every component to concatenate, and would turn filtering into a
substring match.

**The transport is `IExecutionContext`.** An observer attached at the model has to reach a
leaf, and the two obvious routes are both bad: threading a member through every component, or
walking the tree to install one per node. The execution context is already shared parent to
child across the whole tree, so attachment propagates with no new plumbing. Three further
properties fall out rather than being designed: its lifetime already spans a forward pass and
is owned above the components, which is exactly an observation session's scope; it already
carries `synchronize()` (`IExecutionContext.ixx:44`), which 6.5 needs; and
`getDeviceScratchBuffer()` established it as the ambient-services object for operations, so
ambient observation is the same category rather than an intrusion.

One cost to weigh rather than assume: `IExecutionContext` is deliberately minimal — three
members — so widening it from compute infrastructure to compute plus diagnostics is a real
change of role even with the scratch buffer having softened that line.

**Checked 2026-08-25: two models cannot share a context, so the transport is scoped correctly.**
Four independent facts, each sufficient alone. `createExecutionContext` returns
`std::make_unique` on every call (`ExecutionContextFactory.ixx:23`) — no cache, no pool, no
per-device instance. All four transformers take a `DeviceId` and mint their own in the
constructor (`Qwen.ixx:184`, `Llama.ixx:154`, `GptTransformer.ixx:100`, `Gemma.ixx:168`), with
no overload accepting an existing one. `Component::setExecutionContext` is protected
(`Component.ixx:765`, inside the protected section at `:720`), so no external caller can inject
one. And it throws if the context is already set (`Component.ixx:779`). One model tree owns
exactly one context and propagates it only to its own children.

**This is currently an accident, and adopting the transport makes it load-bearing.** Nothing in
the type system forbids a future constructor overload that accepts a context, and that is a
reasonable thing to want — a shared CUDA stream across two models is a legitimate optimization.
The moment it exists, it is also a cross-model observation leak. So **one context, one model
tree** must be stated as a contract on `setExecutionContext` rather than left as a property of
today's construction path. Standalone-mode components each create their own and are therefore
their own observation scope, which is the right answer for a component under test.

**Stated on the symbol 2026-08-29** (`Component.ixx`, `setExecutionContext`), naming the
observation-leak consequence so a future overload is rejected on its merits rather than
tripping over an unexplained throw.

**Filter at attach, not at publish.** Putting the observer on a shared context means the naive
emission has to pattern-match a path on every call — a string comparison per component per
token, worse than the null check Section 7 is measuring. Instead `observe( pattern, passes,
sink )` walks the tree once, resolves the matches and marks them, so publication stays a cheap
test and a path is formatted only for a component actually selected.

Attachment therefore carries a **pass filter** alongside the path pattern, resolved at the same
walk. The per-component flag becomes a small mask rather than a bool, so publication is an AND
and a branch instead of a load and a branch — still nothing against a kernel launch — and an
observer wanting decode only stops paying for prefill altogether.

### 6.4 Capability is answered, never implied

The current default "accepts the probe and never fires it, so an empty result means 'not
instrumented' and not 'nothing to report'" (`LanguageModelNetwork.ixx:138-141`) — a false
negative in a NaN detector, and the review's sharpest objection to shipping the mechanism as
it stands.

**A component must be able to state which stages it publishes, and on which passes**, so a
caller can tell an uninstrumented component from a clean one. Attaching to a path that
publishes nothing is an error the caller can see, not silence.

The pass qualifier is not decoration: **the stage set genuinely differs by pass.** An attention
block's prefill produces score intermediates its decode never materializes, and a
flash-attention prefill publishes different stages again from the cuBLASLt one. So the
capability answer is a set of (stage, passes) pairs rather than a list of names, and a caller
asking "can I watch this value during generation" gets a real answer instead of an empty
callback. 6.3 makes every compute path publish; this is how a caller learns what that means
for a particular component before running anything.

**The rule has a boundary, and it is not "never return empty" (decided 2026-08-25).** The
objection is to an absence that reads as a *verdict*. A probe that fires nothing looks like
"clean", which is why the current default is a false negative. `getOutputs()` returning `{}`
looks like nothing at all — a view renders no row for that node and no reader draws a
conclusion from it — so the plain empty vector is honest, and `std::optional` would buy
ceremony rather than safety. The test to apply when adding any accessor here: **can a caller
mistake the empty answer for a finding?** Where the answer is yes, say so explicitly; where it
is no, return empty.

### 6.5 Lifetime is stated before the first emission site

Activations are pooled and overwritten by design — the Qwen prefill loop overwrites the
block output every chunk. An observer therefore receives a **borrowed view, valid for the
duration of the callback only**, with an explicit copy for anything that must outlive it.

Reading a workspace slot after `prefill()` returns is safe today only because exactly one
block is live at a time. That is an accident of the current implementation, and BACKLOG
already records the lesson: *a probe must not change what it observes, and a general hook has
to state its own lifetime.* The contract is written into this document before the first hook
site exists, not discovered per call site.

Attachment is scoped, so detachment is not something a caller can forget.

**Publication never synchronizes**, because synchronizing is the clearest way for a probe to
change what it observes. So the borrowed view is doubly qualified: valid for the callback
only, and **ordered on the component's stream rather than valid on the host**. An observer
that wants host-readable values reaches `IExecutionContext::synchronize()` — deliberately, at
its own cost, on the transport 6.3 already hands it — or issues its own copy on that stream.

This has to be stated on the symbol rather than in this document alone. The failure mode of
getting it wrong is reading a buffer mid-kernel and receiving plausible garbage, which is the
one class of defect an observability feature must not manufacture.

### 6.6 The convenience case

Possible is not the same as easy, and easy is the actual complaint. A named capture of every
block output — the equivalent of `output_hidden_states=True` — is what turns the mechanism
into a feature. It is a thin consumer of 6.3 and belongs in the library, not in each caller.

### 6.7 Instrumentation depth: every component (decided 2026-08-25)

**All of them.** 27 classes derive from `Component` or `CompositeComponent`; excluding the two
base classes and the two training-only ones leaves **23 in scope — 16 leaves and 7 composites**.
That is an afternoon of mechanical work, not a programme, and the cost measurement in Section 7
removed the only argument for holding back.

Partial coverage is worse than it sounds, and not merely less useful. It makes the 6.4
capability query load-bearing for **ordinary** use rather than for edge cases: every question a
user has would begin with "is that component instrumented?" The complaint this document exists
to answer is *view into any layer*, and "some layers" is the state of the tree today with a
larger number attached.

**The tiers are of effort, not of coverage.**

- **Mechanical.** Roughly ten are a single line — `Gelu`, `Swiglu`, `Softmax`, `RmsNorm`,
  `LayerNorm`, `Residual`, `Activation`, `Linear`, `TokenEmbedding`, `Lpe`. Appendix A is the
  template for all of them.
- **Judgement, still in v0.20.** `Rope` publishes q and k as separate roles;
  `GroupedQueryAttention` and `MultiHeadAttention` expose their intermediates; `GatedDeltaRule`
  carries recurrent state and `CausalConv1d` a ring. These six or seven are precisely the
  components `BACKLOG.md:26` was written about — the attention intermediates that required
  harness ownership, and the DeltaNet block that is opaque to every caller today.

**Stage names are derived, not invented.** The real risk in instrumenting the whole tree at once
was never effort, it was naming under time pressure, because 6.4 makes stage names queryable and
therefore public the moment they ship. That risk mostly dissolves: **the vocabulary is already
written down in the tree, as tensor names.** `Gelu` names its buffer
`this->getName() + ".output"` (`Gelu.ixx:515`); the Qwen attention workspace names its slots
`"q_perm"`, `"preatt"`, `"att"`, `"v_out"` (`Qwen.AttentionBlock.ixx:291-297`).

So the rule is: **the stage name is the tensor's own name with the component prefix removed.**
Naming becomes a lookup rather than an invention, 23 components cannot drift into 23
conventions, and item 6's complaint — that the stage vocabulary is undocumented and defined at
the call site — is answered by construction.

**Consequence to state rather than discover: the same buffer can be published twice.** With
pooled and installed outputs a child's output tensor can literally be its parent block's output
buffer, so a composite publishing its block output and its last child publishing its own emit
the same memory under two names. That is two names for one value at one point, not a defect,
and **composites publish anyway** — a block's output is semantically the layer output, which is
what 6.6's convenience capture means by `output_hidden_states` and what any per-layer consumer
is asking for.

## 7. Cost — MEASURED 2026-08-25: no measurable cost, the runtime design stands

**Result first.** An unattached publication check is not measurable on `Gelu::forward`. The
runtime design in Section 6 stands and the `constexpr` gate described below is **not needed**
— do not build it.

Method: `Gelu<Cuda, FP32>` with a private flag, a public setter (so the branch cannot be folded
away), and `if ( is_observed_ ) { ... }` after `operation_->forward`. 2000 warm-up calls, then
five repeats of 50000 calls, no synchronization inside the loop — the branch is host-side, so
what is under test is its share of *enqueue* cost. Median of five, `x64-claude-verify`
Release, `Device::Cuda(0)`. Baseline measured, stub applied and measured, stub reverted and
baseline measured again, so the two identical builds bracket the instrumented one.

| Arm | decode `[1,1,3072]` | prefill `[1,512,3072]` |
|---|---|---|
| Baseline A | 7506 ns | 12356 ns |
| Publication stub | 7673 ns | 12134 ns |
| Baseline B (after revert) | 8102 ns | 12107 ns |

**The bracket is the finding.** On decode the two identical builds differ from each other by
7.9%, which is larger than the gap between either of them and the instrumented build — and the
stub is *faster* than baseline B. On prefill the stub lands between the two baselines. The
effect moves in both directions and is smaller than build-to-build drift, which is the
signature of no effect rather than a small one.

Taken alone, baseline A against the stub reads as "+2.2% on decode". That number is an
artifact, and it is exactly what the bracket exists to catch.

**Why the result is unsurprising, stated as context rather than as the evidence.** A forward at
decode shape costs ~7.5 microseconds of host time, and total time tracks enqueue time almost
exactly, so the call is host-bound on launch overhead (WDDM is expensive here) rather than on
kernel work. A predicted branch against that is four orders of magnitude down.

**What this does not measure.** One component in isolation, not a whole model, and not the
attached case. The harness is `GeluCudaTests.DISABLED_PublishCost`, kept in the tree.

### 7.1 Whole-model — MEASURED 2026-08-29: the section 10 criterion is met

**Result first.** Publication is not measurable on decode tok/s with no observer attached, on
two chassis, and the criterion is met. The arms differ by one line — a temporary
`if constexpr` gate on `publish()`, applied and reverted, which is the fallback design this
section declines to ship. Run order baseline / instrumented / baseline; seven repeats of the
`decodeSecondsPerToken` subtraction per arm; `x64-claude-verify` Release, RTX 5060 Ti pinned
by UUID, both models DRAM-resident with GiB to spare. Harness:
`ObservationCostCudaTests` (`Mila/Tests/Dnn/Observation/ObservationCost.Cuda.cpp`).

| Arm | Llama 3.2 3B FP4 | Gemma 4 12B FP4 |
|---|---|---|
| Baseline A | 10.2346 ms/token | 23.1355 ms/token |
| Instrumented | 10.3609 ms/token | 23.1575 ms/token |
| Baseline B | 10.3573 ms/token | 23.1642 ms/token |

**The bracket is the finding, again.** On Llama the instrumented build sits 0.03% from
baseline B while the two *identical* baselines differ from each other by 1.20% — forty times
the gap being tested for. On Gemma the instrumented arm lands *between* the two baselines.
Within-arm spread over seven repeats was 0.22-0.62%, so the measurement resolves well below
the drift it is competing with.

Taken alone, baseline A against the instrumented build reads "+1.23% on Llama". That is the
same artifact section 7 produced at component scale, at a scale where it looks far more
convincing, and it is what the bracket exists to catch. A single pair would have reported a
regression that does not exist.

**Why these models rather than the 27B.** The criterion was originally written as "the 27B at
16K", which is the configuration least able to answer it. The overhead is *fixed* per token,
so its share is largest where a decode step is cheapest: Llama 3B at 10 ms/token is roughly
three times more sensitive than the 27B at ~33 ms. The 27B is worse still on the 12 GiB card,
where the 16K build predicts 10.27 GiB against ~10.85 free and the number would carry the
WDDM pager rather than the branch. Publication is a `Component` facility — the cost is
(publish calls per token) x (cost per call), with nothing family-specific in it — so the model
is a *sample*, and the right sample is the one where an effect would be largest. Finding
nothing there settles the 27B a fortiori. **Section 10's criterion is restated accordingly**:
no measurable movement in decode tok/s on a DRAM-resident model at the point of maximum
sensitivity, on at least two chassis.

---

The original gating argument, retained because it explains why the measurement came first:

If a per-component check is free when unattached, the runtime design in Section 6 stands. If
it is not — 64 layers of components, fired per token, is not self-evidently noise — then the
answer is a compile-time gate, and that is a different API shape rather than a tuning pass.
Sequencing this after the interface is written is how the interface gets written twice.

Stating the direction first: the change must not be measurable on decode tok/s with no
observer attached, on the 27B at 16K.

**The prior says it will pass.** The virtual boundary at the network is already justified on
exactly this reasoning — *"one dispatch per layer per token step is negligible against the
per-layer GEMMs"* (`LanguageModelNetwork.ixx:37`). Per-component publication is roughly an
order of magnitude finer than per-layer dispatch and still a predicted branch measured against
a GEMM. That is a prior, not a result, and it is why this is a measurement rather than an
argument.

**If it had failed, the answer would have been a whole-feature gate, not a policy** — one
module-level `constexpr` constant consumed through `if constexpr`, removing publication from
builds that decline it. One axis, two values, no propagation into any type, unlike the template
policy Section 11.2 rejects. Recorded for the record only: the measurement passed, so the gate
is not being built.

## 8. What this subsumes

The design is gated on three real consumers, not on generality:

1. **Item 9, corpus perplexity** (`Qwen3.8.md:398`) — the last unmet Phase 5 exit criterion.
   Teacher-forced log-likelihood is an observer on the head's output, summing over positions.
   The metric stays outside `Mila/Src`; the library owes logits, the consumer owes the number.
2. **The Qwen parity harness** — replaces `QwenAttentionBlockWorkspace` reach-in, and makes
   the DeltaNet block visible for the first time.
3. **`setStageProbe` and Gemma's `fingerprintPrefill`** — **DONE 2026-08-29, and both are
   gone rather than rewired.** The typedef, the silent base default and both family
   overrides are deleted; so are `fingerprintPrefill`, `summarizeActivation` and
   `ActivationSummary` on `GemmaModel`. The fingerprint is now `Tools/ExportArtifact`'s
   `fingerprintModel`, attaching `observe( "*", inference() )` against `LanguageModel` and
   reading the head's first publication — so `Mila/Src` carries no diagnostic of its own and
   every chassis gains one. `TransformerApiReadiness.md` item 6 is closed.

Three consumers exist before the API does. That is the guard against speculative generality.

## 9. What this does not solve

**Observation shows what flows; it cannot show what was never computed.**

`final_rmsnorm` and `lm_head` are built one row wide (`Qwen.ixx:574`), and prefill slices the
final position before the head (`Qwen.ixx:293`). No view, hook or probe produces a logit at
position 5, because that tensor does not exist.

So item 9 additionally needs **head width as an explicit build parameter, carried into
`getRequiredMemory`**. That is build geometry, small, independent of every design choice
here, and it survives all of them unchanged. Qwen has already paid twice for memory that a
build consumed and a prediction did not name; an unpredicted head buffer would be the third.

The number the width budget is sized against is **0.474 MiB per row**: `LmHeadLinearType` is
`Linear<TDeviceType, TPrecision, ...>` (`Qwen.ixx:174`) and `Linear::TensorType` is
`Tensor<TComputePrecision, MR>`, so the row is BF16 — 248320 x 2. Nothing downstream upcasts
it; the sampler reduces the BF16 row on device. An earlier comment at `Qwen.ixx:290` priced
the same row at FP32 and has been corrected.

## 10. Boundary

The exception to the review's scope line is only an exception if it has edges.

**In:** the inspection door on `LanguageModel`; the structural view; activation publication on
**every inference pass** — forward, prefill and decode — from **every one of the 23 in-scope
components** (6.7); the capability answer; the lifetime contract; the convenience capture;
tests pinning all of it.

**Out:** `ComputePass::Backward` and gradient observation — the enum names the pass so adding
it later is additive, but inference is what v0.20 ships; the compare-against-a-reference-file
tool, which is a `Tools/` consumer rather than library surface; and **any Python binding
exposure** — binding scope is settled at session depth, never components, and "as easy as
PyTorch" must not be read as reopening that.

**Also out: lifecycle.** `build`, `loadParameter` and serialization are not observation
targets. Watching those is progress reporting — a different feature, a different consumer, and
the container work already has a logging sink for it. Holding observation to *values flowing
through compute* is what stops this becoming a general event bus.

## 11. Open decisions and rejected alternatives

- ~~**Naming.**~~ **Settled by what shipped.** `+25` landed `publish` and
  `getObservableStages`; the walk is `observe`, which is what this document already called it.
  `describe` was never built and is not needed by any consumer.
- ~~**Path matching.**~~ **Decided: glob with `*`** (2026-08-26). Resolved once by
  `CompositeComponent::observe`, so publication still matches nothing at run time and 6.3's
  filter-at-attach rule holds. Deliberately not regular expressions: `*` covers every pattern
  the three named consumers need -- one component (`"*.lm_head"`), a family of layers
  (`"*.blk_*"`), the whole tree (`"*"`) -- and a richer syntax would be a vocabulary to learn
  for no consumer that exists. `findComponent`'s exact resolution is untouched and still
  serves parameter loading.

### 11.2 The walk, and why it landed late (2026-08-26)

`observe( pattern, passes, sink )` on `CompositeComponent`, forwarded by `LanguageModel`
alongside `stopObserving()` and `componentPaths()`. It returns the **match count**, and that
return is load-bearing: a pattern matching nothing is, downstream, indistinguishable from a
run with nothing to report -- the same false negative this document criticizes `setStageProbe`
for (6.x, 10).

**What its absence cost, recorded because the failure is instructive.** Publication shipped in
`+25` without this walk. The first real consumer -- the Qwen 3.8 item 9 gate, which wanted
last-position logits -- could not reach `lm_head` from outside the model, because the network
sits behind a protected accessor. It bolted a `LanguageModel::lastPositionLogits` onto the
model instead: a second door beside the missing one, narrower, answering exactly one question
where the walk answers all of them. **The accessor has been deleted.** The gate now reads its
logits by observing `"*.lm_head"` and taking the first publication of a `generate()` call,
which is the prefill's, and it reproduces every number the accessor produced to four decimal
places.

The lesson is about sequencing, not design: a mechanism without its door does not get used,
it gets routed around. Ship the attach path with the publication next time.

**Reading VALUES from a published tensor needs the concrete type back.** The sink receives
`const ITensor&`, whose `rawData()` is type-erased, so a consumer that wants numbers rather
than shapes does a `dynamic_cast` to `Tensor<TPrecision, MR>` and its own `toHost`. That works
and is what both consumers do, but it means every value-reading sink has to name the model's
compute precision. Whether observation should offer a typed convenience is a v0.21 question;
the capability is present either way.
- ~~**Does a component's signature become part of its contract?**~~ **Decided: deferred to
  v0.21** — see 11.1, which also corrects the failure history originally cited for it.
- ~~**`ROADMAP.md`.**~~ **Done 2026-08-26.** v0.20 read as a hardening release; it now states the
  pair — a 27B model at 2.82 bits on a 12 GB card *that you can open and read* — and observability
  is a theme of its own with its own success criteria, joined to a matching BACKLOG bucket. The
  rewrite also corrected the Future tail, which still described Qwen as an unbuilt third
  architecture family.

### 11.1 The component signature — DEFERRED to v0.21 (decided 2026-08-25)

**The question:** should `Component` describe what passes *through* it, alongside what it
holds? Signature here means arity, the role of each input, shapes, and dtypes.

**Decision: not in v0.20.** Three things settled it, and the first two emerged after this
section was first written.

**`getOutputs()` absorbed the output half.** When this section was drafted, "signature" meant
inputs *and* outputs. 6.3 then settled on `getOutputs()` returning real `const ITensor*`, and a
tensor already carries shape, dtype, device and name. So "what does this return" is answered,
the memory cross-check is available without any declaration, and the Qwen gate-half example
below is visible directly in `q_proj`'s output shape. What remains unanswered is only the
**input** half — arity and roles — plus the ability to answer before build.

**The failure history cited below belongs to a different check.** This section claimed the
cross-check was the one item here with measured defects behind it, citing both Qwen memory
failures. That attribution is wrong. `Qwen3.8.md:1866` records the larger one: `getRequiredMemory`
passed every block `.withInstalledOutput( isInferenceMode() )` while `onBuilding` never performed
the installation, so DeltaNet components self-allocated six `[512 x 5120]` buffers each, about
6.5 GiB across 48 layers. **Every one of those shapes was correct** — what was wrong was who
allocated them. The same holds for the ~92 MB per layer nothing predicted: right shapes, wrong
pooling. Both are *ownership* defects, which a declared shape sails past and which Gemma's Gate B
equality — `getRequiredMemory` against `getMemoryStats()` — catches without any part of this
section.

**It has no consumer, by this document's own test.** Section 8 gates the design on three real
consumers precisely to guard against speculative generality. Item 9 needs logits (`getOutputs()`),
the Qwen parity harness needs named intermediates (publication), and Gemma's fingerprint needs a
NaN hunt (publication). None needs an input declaration. Checked against the rest of the Qwen 3.8
plan as well — item 9, the chunked UT-transform prefill kernel, and the state-plus-conv-ring
snapshot roundtrip — and none of those wants it either.

**Sequencing, which is the positive reason rather than the absence of one.** Input roles are
better designed *after* instrumenting all 23 components than before. That pass is where one
learns which components have roles worth naming and whether the prefill/decode duality bites in
practice. Designing it now is designing in the abstract; designing it then is designing from 23
worked examples.

**What deferring costs, plainly.** The structural view cannot say "this takes five inputs named
q, k, v, a, b", so a reader of a multi-input component still goes to the code; and nothing can be
answered before build. Neither blocks anything, and neither is the complaint this document exists
to answer — *view into any layer* is delivered in full by `getOutputs()` plus publication from
every component on every pass. The input declaration is an additional capability, not the missing
half of this one.

The argument below is retained unchanged as the v0.21 starting point.

**This recovers information rather than inventing it.** PyTorch cannot answer the question
because `forward(*args)` genuinely does not know until it is called. Mila's situation is the
inverse: `Linear::forward( const TensorType& ) -> TensorType&` is checked by the compiler, so
the signature is not merely known but proven — and then discarded before anything can query
it. What is proposed is making a compile-time fact legible at runtime.

**Why it cannot be one uniform accessor.** The shape of the answer differs across the tree.
`RmsNorm` is one in, one out, same dtype. `Linear` changes the trailing extent.
`TokenEmbedding` takes `[B, T]` INT32 and returns `[B, T, D]` BF16, so the **dtype changes**.
`GroupedQueryAttention` takes three inputs, `DeltaRule` takes five, and `CausalConv1d` carries
state between calls that appears in neither its inputs nor its outputs.

**Three tiers of knowability, and one complication.** Declared extents are config facts known
at construction; concrete shapes resolve at build, where `BuildContext` supplies the batch and
prefill width — the same values `getMemoryStats()` already derives its bytes from. The
complication is that **a component does not have one signature, it has two**: prefill sees
`[B, chunk, D]`, decode sees `[B, 1, D]`, and the final prefill chunk is narrower than the
rest. Any description must say which pass it describes, or describe both. This is what makes
the item larger than "report the shapes".

**Roles must be declared, and that is the same act as declaring a stage.** In
`DeltaRule( q, k, v, a, b )`, the fact that `a` is the gate exists only as a C++ parameter
name and cannot be recovered at runtime. So roles are stated, not discovered — which is
precisely what Section 5 already requires a component to do for its published stages. The two
share a mechanism, and the second is nearly free once the first exists.

**What it buys, strongest first.**

- **A correctness gate, not a documentation feature.** A component that declares its output
  shape *and* predicts its bytes can be held to both agreeing, and a test can enforce it. The
  tree has no way to write that test today. Both Qwen memory defects were the same species — a
  prediction that did not match what the build performed — so this is the one item here with a
  measured failure history behind it.
- **It makes a load-bearing boundary visible.** Qwen's q projection is double width,
  `[query|gate]`, and the config carries two accessors differing by exactly the gate half. The
  comment on them reads *"confusing them is how the gate goes missing"*
  (`Qwen.Config.ixx:356`) — and that confusion produced the Section 2 parameter undercount. A
  declared signature is where such a fact stops being tribal knowledge.
- **It answers a reader's first question.** Knowing what shape reaches `q_norm` currently
  means reading the code or observing a run.

**What it costs.** A declaration on every component — a change to the base contract, not to
reporting, which is why this is a decision rather than a view feature. The real risk is
**drift**: a declared shape the code does not produce is worse than no declaration. The
mitigation is specific and cheap — assert the declaration against the actual tensor at the
emission sites Section 6.3 already adds, so drift cannot survive a debug run.

**Non-goal.** A symbolic shape algebra that infers and checks shapes across the graph as a
type system. The scope is declared roles plus extents in the config's own terms, resolved
concretely after build.

**Relation to Section 5.** That section argues `Component` is a noun rather than a verb. A
signature is a statement *about* compute without being compute, so it does not contradict the
position — but it is the first thing proposed in this document that tests its boundary, and
the boundary should be restated deliberately if this is adopted.

### 11.2 Rejected: a policy-based compile-time observer

**Considered and rejected 2026-08-25. Recorded because it is the idiomatic instinct for this
codebase and will be proposed again.**

The proposal: make the observer a compile-time policy — `Component<TDeviceType, TPrecision,
TObserver>` — so a null observer compiles away and the feature carries provably zero runtime
cost. It sits naturally beside `TWeightQuantization` and the rest of Mila's compile-time axes.

**It fails on the same structural point `TransformerApiReadiness.md` item 8 already settled**
for a trainability parameter, and the argument transfers without modification: `LanguageModel`
holds `unique_ptr<LanguageModelNetwork<TDeviceType, TPrecision>>`, so a third parameter either
propagates into `LanguageModel` — destroying the type erasure that base exists to provide — or
pins a model to one observer forever. CRTP and mixin variants fail identically.

The collision is sharper here than it was there. `LanguageModelNetwork` is deliberately where
the policy parameters stop, because *"GemmaModel should not be a different type because its
weights are FP4."* An observer policy would have to stop at that same line, yet observation's
entire purpose is reaching **below** it, into intra-block stages. The user attaches at the
model, and the model is erased over precisely the parameter the design would require.

Two independent reasons, either sufficient on its own:

- **Observation is inherently a runtime choice.** Compile-time selection means deciding what to
  look at when the library is built, so a consumer debugging their own model would rebuild Mila
  to inspect a different layer. That is the exact inverse of the goal in Section 2.
- **Instantiation cost.** A further axis multiplies every component across the existing device,
  precision and weight-policy matrix, in a tree where module compilation is already the slow
  part of the build.

**What survives from the idea.** The zero-overhead goal is legitimate and is not abandoned —
Section 7 keeps it as a measured requirement, and names the whole-feature `constexpr` gate as
the remedy if the measurement fails. That gate is a policy in spirit without being one in the
type system, which is the distinction that matters.

## Appendix A — Gelu, worked

`Gelu` is the floor case: a stateless leaf with one compute path and one output. It is here to
show what the design costs a component that gains nothing from it, because that is the number
that decides whether instrumenting the tree is affordable.

### What the base provides

```cpp
// Component
public:
    /// Tensors this component owns as outputs. Empty when it does not describe itself;
    /// a view renders no row rather than drawing a conclusion (6.4).
    virtual std::vector<const ITensor*> getOutputs() const
    {
        return {};
    }

    /// Stages this component publishes, and the passes each is published on.
    virtual std::vector<ObservableStage> getObservableStages() const
    {
        return {};
    }

protected:
    // Resolved once by the attach walk, so publication never matches a path or a pass
    // at call time.
    ComputePassMask observed_passes_{};

    void publish( ComputePass pass, std::string_view stage, const ITensor& value ) const
    {
        if ( observed_passes_.contains( pass ) )
        {
            this->getExecutionContext()->notifyObservation(
                this->getName(), pass, stage, value );
        }
    }
```

### What Gelu adds

```cpp
        std::vector<const ITensor*> getOutputs() const override
        {
            if ( output_ == nullptr )
            {
                return {};
            }

            return { output_.get() };
        }

        std::vector<ObservableStage> getObservableStages() const override
        {
            return { { "output", ComputePass::Forward } };
        }
```

and one line inside `forward`, after the operation and before the return:

```cpp
            operation_->forward( input, *output_view_ );

            // The live narrowed view, never the built ceiling in output_ -- a wider
            // earlier call leaves stale values in its tail.
            this->publish( ComputePass::Forward, "output", *output_view_ );

            return *output_view_;
```

No new members, and no new imports: `Dnn.ITensor` is already imported at `Gelu.ixx:27`.

### What this makes concrete

**`getOutputs()` and `publish` deliberately name different tensors.** `output_` is the built
ceiling, which is what a memory cross-check compares against `getMemoryStats()`.
`*output_view_` is the live value narrowed to this call (`Gelu.ixx:186`), which is what an
observer must receive. One component, two correct answers chosen by which question is being
asked — the clearest demonstration in the tree that the structural and dynamic halves of this
document are not substitutes for each other.

**Outputs only; inputs are not published.** A component's input is its producer's output, which
that component already published. Publishing both would duplicate every value in the graph
except at entry points, where the caller already holds it. This halves the emission count and
removes the question of what to name an input role.

**`backward` is untouched.** Gelu allocates `input_grad_` in training mode, and
`ComputePass::Backward` is out of scope per Section 10. The pass is named in the enum so adding
it later changes no signature.

**Gelu carries one half of the cross-check and not the other.** `getOutputs()[0]->shape()` and
`getMemoryStats().device_state_bytes` can be held to each other in a single assertion. The
*prediction* half cannot: Gelu has no `getRequiredMemory` override and falls through to the
base's throwing default, so the two contracts land independently and Gelu sits on one side.

### Why the Section 7 measurement belongs here

Gelu is nearly the cheapest component in the tree, so publication is the largest fraction of its
work anywhere. If the branch is noise here it is noise everywhere, and the stub needed to prove
it is the three additions above. The ratio under test is a host-side branch against a kernel
launch; the launch is expected to dominate by orders of magnitude, which is the prior Section 7
records rather than a result.

## Cross-references

`TransformerApiReadiness.md` item 6 (the parked positioning question) and item 8 (no virtual
`forward` in the hierarchy); `BACKLOG.md:26` for the requirements the Qwen rotary hunt paid
for; `Qwen3.8.md:398` for item 9 and `:574` for the head width; `MemoryFootprint.md` for the
prediction contract the structural view reports against.
