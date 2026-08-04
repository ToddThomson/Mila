# Memory Footprint

Specification and implementation plan for answering, without allocating anything:
**will this model run on this hardware at this context length?**

The answer must be available before a multi-gigabyte download, and for hardware the
user does not own. That rules out any mechanism that requires loading weights or
committing device memory.

---

## 1. Problem Statement

There is currently no way to know a model's VRAM cost except to load it and watch.
That is a 7 GB download and a 40-second load to discover a refusal, and it cannot
answer "would a 16 GB card help?" at all.

The cost of not knowing is on record. Llama 3.1 8B at FP4 measures ~10.7 GB, about
the same as Gemma 4 12B at FP4, because the Llama chassis never received Gemma's
memory gates (BACKLOG, Models). Nothing in the config predicts that. It took a
measurement to find, and a footprint report would have shown it without one.

---

## 2. What Decides The Answer

| Term                      | Where it comes from                                  | Status                      |
|---------------------------|------------------------------------------------------|-----------------------------|
| Weights                   | geometry x quantization policy                        | exact, closed form          |
| KV cache                  | geometry x context x KV policy                        | exact, closed form          |
| Prefill activation        | `resolvePrefillChunkSize`, `computeChunkRowCostBytes` | exact, already pure functions |
| cuBLASLt workspace        | `kCublasLtWorkspaceSize`                              | fixed 4 MB constant         |
| Free VRAM                 | `cudaMemGetInfo`                                      | read live, never modeled    |

The last row is a deliberate choice. The ~1176 MiB baseline measured on the 4070 is
not a CUDA context -- it is the desktop compositor and whatever else is resident.
Modeling it means predicting the user's browser. Reading free bytes accounts for all
of it exactly, at the cost of one call.

Worked example, Llama 3.1 8B, from geometry alone:

```
KV @ 8192, batch 1   2 x 32 layers x 8192 x 8 kv_heads x 128 head_dim x 2 B  = 1.074 GB
FP4 bytes/param      0.5 packed + 4/128 scale                                = 0.53125
Per layer            218.1M params x 0.53125                                 = 115.9 MB
Body                 x 32 layers                                             = 3.71 GB
Tables               2 x (128256 x 4096 x 2 B), untied, unquantized           = 2.10 GB
```

The KV figure reproduces the independently recorded ~1.07 GB. The table figure is
the defect named in section 8.2, visible here as arithmetic rather than as a
surprise on a VRAM meter.

---

## 3. Design

### 3.1 The fact the design rests on

**Construction allocates nothing.** `Component.ixx:65-68` states it as the lifecycle
contract, and `Linear::onBuilding()` confirms it: `initializeParameters( context )`
is called at `Linear.ixx:748`, from `onBuilding`, not from the constructor. Weights,
output buffers, KV cache and operation buffers are all `build()`.

Therefore `make_unique<ConcreteTransformerType>( ... )` at `GemmaModel.ixx:694`
produces the entire graph, correctly shaped and configured, for zero device bytes.
The expensive line is the next one.

### 3.2 The mechanism

Construct the graph. Do not build it. Ask it what building would cost.

The estimate is produced by the real components, holding their real configs, so
there is no second derivation of anything. Gemma's `head_dim` is decoupled from
`embedding_dim / num_heads`; under this design that is simply whatever the
constructed object holds, and cannot diverge.

### 3.3 Alternatives rejected

**A closed-form estimator module outside the components.** Re-derives every shape
from metadata. Agrees with Llama and diverges quietly on Gemma, which is the worst
failure ordering -- correct on the family tested first. Rejected: the divergence is
structural, not a matter of care.

**A `simulate` argument on `build()`.** `build()` is `final` and delegates to
`onBuilding`, so the flag threads into every override and needs a guard at every
allocation site. A missed guard makes the non-allocating probe allocate: a mystery
OOM inside the tool built to prevent OOM, failing silently and in the worst
direction. It also makes `built_` lie, since `forward()` only checks that build
completed. Rejected.

**A dry build that allocates and measures `cudaMemGetInfo`.** Exact, including
allocator rounding. But it answers only for a model already downloaded, on the
machine in front of you -- neither of the two cases that motivate the feature.
Retained as a *test oracle* (section 7), not as the product.

---

## 4. The Contract

### 4.1 Component

```cpp
virtual MemoryStats getRequiredMemory( const BuildContext& context ) const;
```

A peer of `onBuilding( const BuildContext& )`, taking the same argument type. That
is load-bearing: it receives exactly what `build()` receives, so it cannot be
answering a subtly different question. It pairs with `getMemoryStats()` as *what
this would need* against *what this has*.

Both return `MemoryStats`, so a comparison is category-by-category -- a disagreement
names parameters, state or gradients rather than handing back a number that is off
by 400 MB.

### 4.2 Operation

State does not live in the component. `GroupedQueryAttention::getMemoryStats()`
reads `operation_->getStateMemorySize()`, and the KV cache is allocated inside
`CudaGqaOp`. The contract therefore extends to operations:

```cpp
virtual std::size_t getRequiredStateMemorySize( const BuildContext& context ) const;
```

Peer of the existing `getStateMemorySize()`, and matching it in both name and return
type: operations own only state -- parameters and gradients are component-owned -- so
a `MemoryStats` return would have two categories permanently zero.

It defaults to 0 rather than throwing, unlike the component contract. That is not an
inconsistency: `getStateMemorySize()` already defaults to 0 on the premise that an
operation allocates no state unless it says so, and only `CudaGqaOp` and `CudaRopeOp`
override it. The two therefore agree for every stateless operation without either
being written.

The KV capacity rule -- full context, or `min( T, window + prefill_chunk - 1 )` for
the bounded ring -- belongs here, next to `getCacheCapacity()` which already states it.

Operations are constructed with their component, before `build()`, so virtual
dispatch is available. No template dispatch is duplicated: the component already
names its operation type through `OperationTraits`.

### 4.3 Composite

`getMemoryStats()` is pure virtual on `Component` and each composite implements its
own; there is no base-class recursion to inherit. `getRequiredMemory` mirrors
whatever its `getMemoryStats()` twin does, which for `GemmaTransformer`
(`Gemma.ixx:339`) is three parts:

1. sum over `getComponents()`
2. plus the transformer's own pooled tensors -- `block_workspace_`, `gqa_preatt_`,
   `gqa_att_`, and the decode variants
3. minus the tied `lm_head` correction

Part 2 is the prefill activation term. It is the largest term the closed-form
approach could not attribute, and it is contained in one function in one file.

### 4.4 Children do not all receive the parent's context

Discovered while implementing Phase 2, and it limits the "virtual recursion is
automatic" property claimed above.

`GemmaBlock::onBuilding` does not cascade one `BuildContext` to its children. It
derives **seven** of them -- `stream_shape`, `qproj_shape`, `qknorm_shape`,
`kknorm_shape`, `gate_up_shape`, `hidden_shape`, `qkv_ctx_shape` -- because the
QK-norms see per-head rows, the gate/up projection is double width, and the GQA layer
is built at full context length while everything else is built at the prefill chunk.

So `for ( child : getComponents() ) stats += child->getRequiredMemory( context )` is
**wrong** for this composite: it would size every child against the parent's shape.
`getRequiredMemory` has to name each child and hand it the same context `onBuilding`
would, which is the hand-written composition the static design was rejected for.

The mitigation is the one that has worked at every other level: **extract the context
derivation** so `onBuilding` and `getRequiredMemory` share it, rather than deriving
seven shapes twice. What remains hand-written is only *which child gets which
context*, and that mismatch is what Gate A's composite comparison detects.

### 4.5 Installation is an intent, and must be declared

The recurring hazard of this design is a value that `onBuilding` assigns and a
pre-build `getRequiredMemory` reads as garbage. Six instances so far --
`RmsNorm::outer_shape_`, `Rope::q_shape_`, `GemmaBlock`'s child pointers,
`prefill_chunk_size_` via `prefillScoreWidth`, `tie_word_embeddings_`, and the
`output_installed_` / `workspace_installed_` flags.

The last is the one that cannot be fixed by extracting a helper, because the value is
not derived -- it is *decided by the parent*, which installs a shared slot into each
child between constructing it and building it. Pre-build the flag is false everywhere,
so every pooled activation is counted twice: once by the child that would otherwise
self-allocate it, and once by the pooling parent that actually owns it.

`BuildContext::withInstalledOutput()` carries that intent down. A component predicts
against `output_installed_ || context.hasInstalledOutput()`, while `onBuilding` keeps
using the member, which by then is accurate. Gate A caught this as an overcount that
scaled exactly per layer -- 2x on two layers, 2.9x on four -- which is what a
per-child double count looks like from the outside.

Note this makes the composite comparison in section 7 load-bearing rather than
belt-and-braces. A block whose child list drifts from its context list produces a
plausible number, not an obviously wrong one.

---

## 5. Entry Point

A static sibling of `fromPretrainedImpl` that shares its prologue verbatim and stops
one line early:

```cpp
static MemoryStats requiredMemoryImpl(
    const std::filesystem::path& path,
    const GemmaModelConfig& model_config,
    DeviceId device_id )
{
    // identical: reader, artifact quantization check, configFromMetadata,
    //            context_length validation
    auto network = std::make_unique<ConcreteTransformerType>(
        metadata.model_name, network_config, device_id );

    BuildContext build_context(
        shape_t{ 1, context_length }, RuntimeMode::Inference, false );

    return network->getRequiredMemory( build_context );
    // no build(), no loadParameters(), no model
}
```

The public entry reuses `fromPretrained`'s existing runtime-to-compile-time
quantization dispatch, so the probe and the load resolve to the same template
combination by construction.

Only the artifact header is read. The safetensors `__metadata__` block carries the
full `PretrainedMetadata` geometry and sits at the front of the file, so the
pre-download case is a range request, not a schema change to `mila.json`.

The library returns measurements. Whether a given headroom counts as "too tight" is
adaptor policy -- see section 6.5 -- and stays out of the runtime.

---

## 6. Traps

### 6.1 Prefill chunk resolution

`resolvePrefillChunkSize` runs inside Gemma's `onBuilding` and threads its result to
every block via `block_context` (`Gemma.ixx:473-483`). It is a pure function of
(B, T_ctx). `getRequiredMemory` must run it before recursing, or every block's
attention scratch is sized against a default.

### 6.2 Sharing makes a naive child-sum overcount

Gemma installs shared outputs and shared weights, and `Linear::onBuilding` skips
self-allocation when a slot is installed (`Linear.ixx:764`). Pooling, output sharing
and weight tying all break `sum(children)`. The corrections already exist in
`getMemoryStats()` and `parameterCount()`; they must be mirrored, not re-derived.

### 6.3 The tied load peak -- RESOLVED, and a residual inconsistency

**No longer applies to Gemma.** `Gemma.ixx:571` installs the shared table *before*
`lm_head_->build()`, driven by `config_.getTieWordEmbeddings()` rather than by load
metadata, so the head never allocates its own `[vocab, model_dim]` weight. The ~1 GB
load-time transient this section was written against is gone, and the probe's settled
figure is the real high-water.

What remains is a **source inconsistency**, filed in BACKLOG: `onBuilding` ties from
config, while `getMemoryStats` subtracts the double-count from the
`tie_word_embeddings_` member, which is assigned from checkpoint metadata at load.
Between `build()` and `loadParameters()` the two disagree and `getMemoryStats`
double-counts ~2.0 GB on Gemma 4 12B. `getRequiredMemory` uses the config source,
because that is the one available when the decision is actually made.

### 6.4 The scratch buffer is not visible at build time

`getDeviceScratchBuffer` grows on demand during `forward()`, freeing and
re-allocating on each grow (`CudaExecutionContext.ixx:220`). No build-shaped
contract sees it. This is the one term that stays outside the model; section 7
quantifies it rather than predicting it.

### 6.5 On Windows, "fits" is not binary

WDDM oversubscribes into shared host memory rather than failing. Contexts of 65536
and above measured 12282/0 MiB and kept running, pathologically slowly. The number
is exact; the verdict needs a margin, and that margin exists to catch spill, not to
cover measurement error. Refuse only when weights alone exceed free VRAM; otherwise
warn and proceed. A false refusal is worse than the problem being solved.

---

## 7. Test Strategy

Two comparisons, catching different failure modes.

**Gate A -- composition.** `getRequiredMemory( context )` on an unbuilt graph
against `getMemoryStats()` after a real `build( context )`, per component and per
model. Catches a wrong formula in a leaf and a missing child in a composite. Needs a
device but no weights and no checkpoint.

**Gate B -- reality.** The same figure against the `cudaMemGetInfo` delta across a
real build. Catches what `MemoryStats` cannot see: allocator rounding, and the
section 6.4 scratch high-water once a forward pass has run. Gate B is not expected
to be exact; its job is to *quantify and bound* the residual, which is currently
unattributed.

Coverage must be over real components. A mock's `getRequiredMemory` agrees with its
own `getMemoryStats` and proves nothing -- this repeats a defect already paid for
once, where a base-class contract verified against a mock stayed green while five
real composites bypassed it.

Expect several build rounds before Gate A is green. The first disagreement is the
useful output: its size identifies the tensor class that was missed.

---

## 8. Related Defects

### 8.1 The lifecycle documentation stated the opposite of the truth

Fixed 2026-08-04. Two places in `Src` documented *"After construction -- parameters
only"*: `Component.MemoryStats.ixx` and `Component.ixx:567`. Parameters are allocated
in `onBuilding`. This is the load-bearing fact for the whole design -- taken at face
value it forecloses the approach entirely. CLAUDE.md does not carry the claim.

`Component.ixx:570` also referred to a `setEvaluation( false )` that does not exist;
the API is `setTrainingMode()`. Corrected in the same pass.

### 8.2 The Llama chassis has no memory gates

`TokenEmbedding` and `lm_head` carry no quantization policy and there is no weight
tying (BACKLOG, Models). The probe will faithfully report ~2.1 GB of BF16 tables on
an FP4 8B. That is a true measurement of a defect, not an estimator bug, and the two
should not be conflated when the numbers are reviewed.

Note the correction to the BACKLOG arithmetic: Gemma's `TableQuantizationPolicy` is
`PerChannelFp8<>` even when the body is FP4, so a quantized table costs ~0.53 GB
against 1.05 GB BF16 -- about 0.52 GB saved per table, not the 0.79 GB recorded.

---

## 9. Phasing

Each phase is separately verifiable. No phase depends on a later one being right.

**Phase 1 -- contract and leaves.** `getRequiredMemory` on `Component` and
`Operation`; implementations for `Linear`, `TokenEmbedding`, `RmsNorm`, and their
operations. Gate A per component. Fix 8.1 in the same commit.
*Verifiable:* a single `Linear` predicts its own build exactly, at every policy.

**Green 2026-08-04**, first build round: full ctest suite passing, Chat coherent. Two
decisions taken while implementing:

- The component default **throws**; the operation default **returns 0**. Not an
  inconsistency -- see section 4.2. A component that allocates and reports nothing is
  a silent underestimate, which is the one failure direction this exists to prevent.
- An **installed** weight is *reported*, not skipped, even though
  `initializeParameters()` will not allocate it. This matches `getMemoryStats()` and
  leaves the tying composite to subtract it exactly once (section 6.2). Skipping it
  here would have made a tied table disappear from the model total, and Gate A caught
  the divergence before it was built.

**Phase 2 -- Gemma composite.** `GemmaBlock` and `GemmaTransformer`, including chunk
resolution (6.1) and the sharing corrections (6.2). Gate A at model level.
*Verifiable:* predicted total matches `getMemoryStats()` for 12B FP4 across a
context sweep.

**Green 2026-08-04**, five Gate A cases: all-local, heterogeneous, four-layer (the
case that pins RoPE deduplication -- two layers cannot distinguish a correct dedup
from an off-by-one), tied embeddings, and construct-allocates-nothing. Two defects
found by the gate rather than by inspection: the tying source disagreement (6.3) and
the pooled-output double count (4.5).

**Phase 3 -- entry point and residual.** `requiredMemoryImpl` plus the public static.
Gate B against `cudaMemGetInfo`, attributing the gap between predicted and measured.
*Verifiable:* the residual is a named, bounded number instead of ~3.8 GB of
unexplained difference.

**Phase 4 -- Llama.** Same contract on the Llama chassis. Expected to expose 8.2 as
a reported figure.

**Phase 5 -- adaptor.** Chat `/model` pre-flight, warn-and-proceed per 6.5, plus a
context sweep -- the probe is cheap enough to search for the largest context that
fits.

---

## 10. Non-Goals

- **Predicting throughput.** This answers whether it fits, not how fast it runs.
- **Modeling the user's other processes.** Free VRAM is read, never predicted.
- **Training footprints.** Gradient buffers are allocated on first
  `setEvaluation( false )`, outside `build()`. The contract has a category for them
  and Phase 1-5 leave it zero.
- **A `mila.json` schema change.** The geometry is already in the artifact header.
