# Transformer API Readiness (v0.20)

**Status: review in progress, findings only.** Started 2026-08-24 as a read-through of
`GemmaTransformer` and its `LanguageModelNetwork` base, asking what must change before the
first production release. Nothing here is classified or scheduled yet -- the keep /
subtract / defer assessment is deliberately deferred until the read is complete.

## Why now

Breaking changes are still free. Nothing outside the repo depends on
`LanguageModelNetwork`'s shape, and backward compatibility is not a goal. The moment v0.20
ships, this surface becomes what people read, quote and build against. Every item below
costs a rename or a deletion today and costs an argument afterwards.

## Scope line

Holding this line is what keeps the backtrack bounded:

- **In scope -- hardening.** Removing members that were never designed, closing
  asymmetries, making silent no-ops say what they are. All of it makes the public
  surface *smaller*.
- **Out of scope -- new design.** Anything that makes the surface bigger is a feature
  and waits, however legitimate the need. Observability is the live example.

---

## Findings

### 1. `getMemoryStats()` hand-enumerates 25 pointers, and Qwen already fixed it

`Gemma.ixx:355-366` sums a braced list of 25 raw pointers -- seven loose `gqa_*_`
members (`:825-831`) plus all 18 `block_workspace_` slots via `.get()`. Add a slot to
`GemmaBlockWorkspace` and forget this list and the result **under**counts silently,
which is the one failure direction `MemoryFootprint.md` exists to prevent.

Qwen solved this: `QwenGqaWorkspace` owns the seven as a unit with `deviceStorageBytes()`,
its header stating the reason -- owned together so a caller cannot allocate half of it --
and `Qwen.ixx:356-358` is three lines. The improvement was never back-ported.

### 2. `resetKvCache` is not on the base while `rewindKvCache` is

`LanguageModelNetwork.ixx:189` declares `rewindKvCache` virtual with a default. There is no
`resetKvCache` on the base at all; `Gemma.ixx:313` and `Qwen.ixx:326` each define one
non-virtually. Half the KV lifecycle is polymorphic and half is not, so a caller holding
`LanguageModelNetwork*` can rewind a cache it cannot reset. The block interface has both
(`ITransformerBlock`), so the gap is only at network level.

### 3. `getExecutionContext()` re-declared in all four families

`Network.ixx:166` already declares it public with the identical signature, and forwards
to `CompositeBase`. Each family declares it again and forwards to `Network`:
`Gemma.ixx:582`, `GptTransformer.ixx:535`, `Llama.ixx:503`, `Qwen.ixx:511`. Non-virtual,
so dispatch is unaffected. A three-level forwarding chain in which two levels add nothing.

### 4. `getModelType()` is four constants whose only callers are tests

`Gemma.ixx:343`, `GptTransformer.ixx:464`, `Llama.ixx:350`, `Qwen.ixx:489` -- four
independent non-virtual methods each returning a constant. Every call site is an
`EXPECT_EQ` in a test. Because it is not virtual, a caller holding `LanguageModelNetwork*` --
the only caller who genuinely cannot tell the family -- cannot reach it.

### 5. `onTrainingModeChanging` is a pure forwarding override in all four families

`Gemma.ixx:778`, `Llama.ixx:661`, `GptTransformer.ixx:688`, `Qwen.ixx:647`. Each body is
a single `NetworkBase::onTrainingModeChanging( training_mode )` call.

### 6. `setStageProbe` -- an undesigned diagnostic in the public surface

`Mila.ixx:159` exports `Dnn.LanguageModelNetwork`, so `setStageProbe` and the `StageProbe`
typedef are public C++ API. Its sole consumer is an internal tool:
`ExportArtifact.ixx:954` reaches `GemmaModel::fingerprintPrefill`, which installs a probe
to find the first prefill stage producing a NaN.

**The need is legitimate.** Two loads of one model can hold byte-identical parameters and
compute differently, and only activations show where they diverge; the probe is on the
real prefill path rather than a parallel one because a second implementation is free not
to reproduce the bug. That reasoning is sound and is recorded on the base.

**The shape is not designed.** Every defect is an absence rather than an error: the stage
vocabulary (`"embedding"`, `"layer_N"`) is undocumented and defined at the call site; it
fires on prefill and not decode with nothing saying that is intentional; the base default
(`LanguageModelNetwork.ixx:140`) silently accepts and never fires, so on Llama and Gpt
"unsupported" and "clean" are indistinguishable -- a false negative in a NaN detector; no
test pins the contract; and the one consumer detects it with
`if constexpr ( requires { ... } )`, so a signature change silently stops the probing.

Two coherent ends, and the current state is neither: take it off the public base and let
the tool reach the mechanism deliberately (hardening, in scope), or design observability
as a feature -- named stages, both compute paths, an honest capability answer, a
documented contract, a test (new design, out of scope, needs a positioning decision on
whether observability is a Mila feature at all).

### 7. `prefill` / `prefillFrom` -- the pure/default assignment is inverted

`prefillFrom( input, start_offset )` is the primitive: the caller passes the full prompt
and says positions `[0, start_offset)` are already resident in the KV caches. `prefill`
is that with offset 0, and both implementers say so literally --
`return prefillFrom( input, 0 )` at `Gemma.ixx:204` and `Qwen.ixx:220`.

The base has it the other way round: `prefill` is **pure** (`LanguageModelNetwork.ixx:113`)
while `prefillFrom` carries a default that throws (`:169`). The derived form is
compulsory and the general one optional.

`ITransformerBlock.ixx:53` already models this correctly one level down -- a single
`prefill( input, position_offset )`, no second name for offset zero.

The cost lands on the caller, which branches on whether a number is zero
(`GemmaModel.ixx:509-511`):

```
auto& logits = reused
    ? this->getNetwork().prefillFrom( prefill_input, reuse )
    : this->getNetwork().prefill( prefill_input );
```

Collapsing to `prefill( input, start_offset = 0 )` is a subtraction: two one-line
delegations deleted, two base declarations become one, four call sites in `Models/`.

It also surfaces something the split currently hides. Prefix reuse exists in Gemma and
Qwen and not in Llama or Gpt, which today reads as `prefillFrom` throwing. One method
makes that a stated capability gap instead of an incidental one.

### 8. `forward()` / `backward()` are pure on the base and throw in two families

`LanguageModelNetwork.ixx:84` and `:100` are pure virtual; `Gemma.ixx:192` and `:198` throw
"inference-only", as does Qwen. The base's own header names LlamaTransformer and
GptTransformer as its subjects, so the training half of the interface is there for the
training families and the inference-only families satisfy it with a refusal.

This was initially flagged as the item that could swallow the release, on the assumption
that fixing it meant splitting the base along the inference/training line. **Investigation
2026-08-24 says it does not.** Three facts, each checked:

- No call site reaches `forward` or `backward` through a `LanguageModelNetwork&` or `*`. The
  only training caller is `BardTrainer.ixx:379`, which holds a concrete
  `make_unique<GptTransformer<...>>`, so `:435` and `:456` are concrete calls.
- The base's only polymorphic client is `LanguageModel` (`LanguageModel.ixx:182`, `:257`),
  and everything it does through that reference is prefill / decode / rewind.
- `Component`, `CompositeComponent` and `Network` declare no virtual `forward` or
  `backward` at all, so the token-typed pair here is the sole declaration in the hierarchy
  and removing it leaves nothing dangling.

**Recommendation, not yet decided:** delete both from `LanguageModelNetwork`. `GptTransformer`
and `LlamaTransformer` keep their implementations as their own concrete API, unchanged;
Gemma and Qwen delete two throwing overrides each. `LanguageModelNetwork` then becomes what it
is already used as -- the inference contract.

#### The split is not a deficiency in the type system

An earlier draft of this item said the inference/training split "is not expressed in the type
system", which reads as something to remedy. It is not. **The split is only a problem while
the base claims training it does not have**; delete the claim and there is nothing left to
express, because every member then describes something all four families do.

Three ways to model it were considered and all are worse than subtracting:

- **A capability template parameter** (`LanguageModelNetwork<TDeviceType, TPrecision,
  TCapability>`) is structurally impossible here. `LanguageModel` holds
  `unique_ptr<LanguageModelNetwork<TDeviceType, TPrecision>>`, so a third parameter either
  propagates into `LanguageModel` -- destroying the type erasure this base exists to provide --
  or pins a model to one capability forever. CRTP and mixin variants fail on the same point:
  anything that makes the base's type depend on trainability pushes that dependency up into
  the model layer.

- **A second interface**, `TrainableLanguageModelNetwork : LanguageModelNetwork` carrying the
  pair, is the textbook answer and would have zero users: `LanguageModel` would still hold the
  inference base and `BardTrainer.ixx:379` would still hold a concrete `GptTransformer`. A type
  that exists so a distinction has somewhere to live is ceremony. It stays the right move *if*
  polymorphic training ever earns it -- the change is purely additive at that point, and shaped
  by a real caller rather than a guess.

- **Demoting to a throwing default** is one line and leaves the training methods on every
  language model's public surface -- `setStageProbe`'s silent default with a louder failure.

The deciding argument is that "inference-only" is a **status, not a property**. Gemma has no
`backward` because none was written, not because the architecture forbids one; training
revival is live in this project. Encode the current status as a permanent type property and it
has to be unwound the day Gemma grows a backward. That is the same error rejected earlier in
this review for `ITransformerInferenceBlock`: naming a restriction rather than a property.

Separate and not a reason to change any of the above: `LlamaTransformer::backward` exists
and compiles, but Llama runs GQA and GQA backward has never been validated. "Llama is
training-capable" is a claim about shape, not correctness.

### 8a. `backward` returns a type that cannot hold what it documents

Found 2026-08-24 while correcting the Doxygen, and **not fixed by item 8** -- the signature
travels with the method to `GptTransformer` and `LlamaTransformer` under any outcome there.

The declared return is `TokenIndexType&` -- the INT32 token-index tensor -- while the
documentation said "Gradient w.r.t. the input embeddings". An INT32 tensor cannot carry a
float gradient. Both implementations return whatever the token embedding's backward hands
back (`Llama.ixx:320-323`, `GptTransformer.ixx:308-311`), and the embedding's input is
discrete, so there is no input gradient for it to produce.

Either the return is vestigial and `backward` should return `void`, or it was meant to be
the embedding-output gradient and the type is wrong. The comment has been corrected to
describe the type honestly; the signature has not been touched. Deciding this needs
someone who knows what the training loop wanted from the return value -- `BardTrainer`
discards it (`:456`).

### 9. Already-documented items confirmed live

- The tie-source split is visible within one file: `getMemoryStats():374` reads the
  member `tie_word_embeddings_` (set by `loadParameters():589` from checkpoint metadata)
  while `getRequiredMemory():460` reads `config_.getTieWordEmbeddings()`. Documented at
  `MemoryFootprint.md` s6.3; this read adds nothing new.
- `withInstalledOutput` appears at `Gemma.ixx:405` in `getRequiredMemory` and is absent
  from the matching `onBuilding` context at `:663-665`, which is the asymmetry
  `MemoryFootprint.md` s4.5 proposes to remove.

### 10. Internal, not API: the empty block-kind branch

`Gemma.ixx:686-737` -- 52 lines whose two arms are character-identical apart from
`GlobalBlockType` versus `LocalBlockType`. Same workspace install, same build, same flash
settings, same `push_back`; roughly half the lines are comment, and both arms restate the
flash rationale that belongs beside `useFlashPrefillForContext()` where the decision is
made. The same two-arm shape repeats at `:430/436` and `:868/873`.

Qwen's equivalent branch is load-bearing -- different workspaces per kind, no flash flags
on DeltaNet -- so the structure is not wrong in general. It is empty specifically in Gemma.

---

## Cross-references

`Workspaces.md` for the shared-buffer survey and the `setState` question;
`MemoryFootprint.md` s4.5 for `withInstalledOutput` and s6.3 for the tie source;
`BACKLOG.md` for the Gemma block-level Gate A case and `makeGemmaBlockWorkspace`.
