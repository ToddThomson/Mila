# Workspaces

**Status: survey, not a design of record.** This document maps what exists as of
2026-08-23 so the shared-buffer mechanisms can be reviewed against each other rather
than one at a time. Nothing here is settled policy. Where a decision is owed it is
named as such and points at the document that owns it.

---

## 1. Why this document exists

Five distinct mechanisms in Mila hand a device buffer to code that does not own it.
All five are called some variation of "workspace" or "scratch". They differ in owner,
in lifetime, in form, in how they are installed, and -- this is the part that matters --
in whether the memory-footprint mechanism can see them at all.

That last difference was never decided. It falls out of whether a given buffer happens
to be a `Tensor` or a raw `cudaMalloc`, which in turn falls out of when each mechanism
was written. `MemoryFootprint.md` records the cost of this twice already: a per-layer
double count (s4.5), a ~6.5 GiB understatement (s4.5), and a ~230 MiB residual that
turned out to be scratch the prediction path cannot observe (s6.4).

---

## 2. The five mechanisms

| # | Mechanism | Owner | Form | Reached by | Visible to `getRequiredMemory` |
|---|---|---|---|---|---|
| 1 | Block activation pooling | Transformer | `struct` of `shared_ptr<Tensor>` | `installSharedWorkspace` then `installSharedOutput` | Yes |
| 2 | Attention transient sharing | Transformer | `struct` of `unique_ptr<Tensor>` | `allocateAndWireGqaWorkspace` | Yes |
| 3 | Grow-on-demand device scratch | ExecutionContext | raw `void*` | `getDeviceScratchBuffer( bytes )` | **No** |
| 4 | cuBLASLt workspace | ExecutionContext | raw `void*` | `getCublasLtWorkspace()` | **No** |
| 5 | Operation-private scratch | Operation | `mutable Tensor` members | direct member access | Yes |

### 2.1 Block activation pooling

`GemmaBlockWorkspace` (`Gemma.Block.ixx:102`), `QwenAttentionBlockWorkspace`
(`Qwen.AttentionBlock.ixx:117`), `QwenDeltaNetBlockWorkspace` (`Qwen.DeltaNetBlock.ixx:122`).

One named slot per block-graph position, sized `[B, chunk, width]`, owned by the
transformer and installed into every layer of the matching kind. The argument is that
inference is strictly sequential, so exactly one block is live at a time and 47/48 of
per-layer retained activations are never read again. Components view prefixes of
max-geometry slots. Aliasing is argued per slot in each struct's header comment.

This is the mechanism that carries the accounting rules in `MemoryFootprint.md` s4.5,
including the `BuildContext::withInstalledOutput` intent channel and the open proposal
to remove it.

### 2.2 Attention transient sharing

`QwenGqaWorkspace` (`Qwen.AttentionBlock.ixx:226`).

Structurally the same idea as mechanism 1 -- a transformer-owned struct of tensors,
one set shared across a sequentially-executed stack -- but it holds attention
internals rather than component outputs, and it is reached by a different path. It
exposes `state()`, which flattens to the `GqaState` struct of raw pointers the kernels
take, and `deviceStorageBytes()` for the transformer's own accounting. Its factory
takes `score_width` explicitly so a caller cannot allocate for the flash path and then
run the cuBLASLt one.

It is declared in `Dnn.Components.QwenAttentionBlock` despite belonging to neither that
block nor any other -- the transformer owns it and shares it across the attention layers.
The 2026-08-23 rename from `Dnn.Components.QwenBlock` sharpened this rather than fixing
it: the module now names one block class, and the GQA workspace is not part of it.

### 2.3 Grow-on-demand device scratch

`CudaExecutionContext::getDeviceScratchBuffer` (`CudaExecutionContext.ixx:234`).

Raw `cudaMalloc`, `mutable`, grown on demand and never shrunk. On a grow it frees the
old pointer and allocates a new one, so **the returned pointer is invalidated by any
later call with a larger request**. Callers must fetch at `forward()` time and must not
cache across calls. Reuse across sequential operations is safe because the context has
one stream.

`getScratchHighWaterBytes()` (`:214`) exists specifically so the footprint tooling can
attribute this buffer after the fact. That is attribution, not prediction: the value is
only meaningful once something has run.

Known consumers include the FP8 two-phase dequantization staging buffer and the GQA
decode split path (`CudaGqaOp.ixx:834`).

### 2.4 cuBLASLt workspace

`CudaExecutionContext::getCublasLtWorkspace` (`:190`). Fixed 4 MiB
(`kCublasLtWorkspaceSize`, `:337`), allocated once at context creation, freed in
`releaseResources()`. Handed to cuBLASLt, whose contract owns what happens inside it.

This one is a third-party requirement rather than a Mila design choice. It is listed
because it is invisible to the footprint for the same reason mechanism 3 is, and a
review of "what does the prediction path not see" has to account for it.

### 2.5 Operation-private scratch

`CudaSamplingOp::prob_scratch_`, `reduction_scratch_`, `index_scratch_`
(`CudaSamplingOp.ixx:70-72, 278-280`).

`mutable Tensor` members, sized at construction from config, never shared with anything.
Owned, typed, tracked by the memory resource tracker, and visible to the footprint
because they are tensors.

**This mechanism is not a problem.** It reads as part of the family only because of the
word "scratch". A review should confirm it and move on.

---

## 3. What the survey shows

### 3.1 Footprint visibility is a consequence of form, not a decision

Whether a buffer participates in "does this model fit" depends on whether it is a
`Tensor` or a raw `cudaMalloc`. The memory resource tracker sees tensors; it cannot see
mechanisms 3 and 4. Nobody chose to exclude them.

The consequence is already a named number: `MemoryFootprint.md` s6.4 measures the
scratch residual at ~230 MiB, essentially model-independent, and it stayed an
unexplained margin until it was chased.

This is the finding the other four follow from, and the one worth deciding first.

### 3.2 Four verbs mean the same thing

`installSharedWorkspace`, `allocateAndWireGqaWorkspace`, `installSharedOutput`, and
`ITransformerBlock::setState` all mean "someone else owns this buffer".
`installSharedWeight` (`Linear.ixx:705`, `:736`) is a fifth instance of the same shape
for parameters rather than activations.

Mechanisms 1 and 2 are structurally identical and are reached by different paths for no
reason beyond the order in which they were written.

`setState` is the odd one and is treated separately in s3.7: it is mechanism 2's
per-block install step, but it is spelled as a generic state setter on a polymorphic
interface, and its argument names a specific attention kind.

### 3.3 One concept, two construction idioms -- and one of them is unreachable from a test

Qwen exports free factories: `makeQwenAttentionBlockWorkspace` (`Qwen.AttentionBlock.ixx:174`),
`makeQwenDeltaNetBlockWorkspace` (`Qwen.DeltaNetBlock.ixx:203`),
`makeQwenGqaWorkspace` (`Qwen.AttentionBlock.ixx:278`).

Gemma builds its workspace inside a private transformer member,
`GemmaTransformer::allocateBlockWorkspace` (`Gemma.ixx:1110`), with an inline `slot()`
lambda. There is no exported factory.

That is not only a style split. It decides testability: `Qwen.DeltaNetBlock.Cuda.cpp`
constructs a workspace directly and asserts predicted-equals-built for both the pooled
and self-allocated arms. Nothing can do that for Gemma, and `Gemma.Block.Cuda.cpp`
calls `getRequiredMemory` nowhere.

`MemoryFootprint.md` s4.5 states the rule that **a Gate A case is owed per block kind**.
Qwen satisfies it -- attention and DeltaNet both have block-level cases. Gemma does not:
its local and global kinds share one max-geometry workspace and neither has a
block-level predict-versus-build test. That is the exact shape of the defect that cost
~6.5 GiB on Qwen.

### 3.4 Ownership has no stated rule

Transformer (1, 2), ExecutionContext (3, 4), Operation (5). Nothing says which a new
shared buffer should pick, so the next one is picked by proximity to whatever file the
author had open.

### 3.5 The grow-on-demand hazard is enforced by a comment

Mechanism 3 can move its pointer under a caller that held it. The rule -- fetch at
`forward()`, never cache -- lives in `CLAUDE.md` and in the accessor's Doxygen. Nothing
in the type system carries it. Grow-only scratch has already turned an out-of-bounds
read intermittent once.

### 3.6 A private type is duplicated per family

`WorkspaceWidths` is defined identically and privately in `Gemma.ixx:902` and
`Qwen.ixx:782`, each with its own `computeWorkspaceWidths`. Low stakes -- both are
implementation detail of one class -- but it is the same concept twice.

### 3.7 Mechanism 2's install path runs through an interface that cannot express it

`ITransformerBlock::setState( const GqaState& )` is how the shared GQA workspace reaches an
individual block. Three things are wrong with that, in increasing order of cost.

**The name states nothing.** It means "wire the shared GQA transient workspace", which
is what its own Doxygen has to say underneath it. Both real implementers forward
straight to `attn_->setState( state )` (`Gemma.Block.ixx:357`, `Qwen.AttentionBlock.ixx:419`), so
the interface method exists only to reach an inner component of one block composition.

**The signature names an attention kind.** `GqaState` appears in the signature of an
interface whose purpose is to abstract over block kinds.

**The mismatch is absorbed silently.** `QwenDeltaNetBlock::setState` is an empty body
(`Qwen.DeltaNetBlock.ixx:356`), and `Qwen.ixx:940` pushes the state into every layer
unguarded:

```
for ( auto* layer : layers_ )
    layer->setState( gqa_state );
```

The transformer branched on `isFullAttentionLayer( i )` at build time and then discards
that knowledge, relying on an empty override to absorb the difference. This is the same
shape as the `withInstalledOutput` defect in `MemoryFootprint.md` s4.5: information the
caller held, thrown away, with a silent no-op standing in for a type-level guarantee.

`supportsKvCache()` shows the same seam from the other side -- `QwenDeltaNetBlock`
returns `false` (`:366`) with a comment explaining that the question does not apply to a
recurrent state. Two of the interface's six methods are meaningless for one of its three
implementers, which is what it looks like when an interface models one block kind and a
second kind is fitted into it afterwards.

---

## 4. The organizing question

Full unification is not the goal: mechanism 4 is a third-party contract and mechanism 5
is correct as written. The question the review can actually settle is narrower:

> **What makes a device buffer visible to `getRequiredMemory`, and is that a property
> we chose or one we inherited?**

Every workspace defect to date is an answer to that question going wrong. Settle it and
the rest follow: mechanisms 1 and 2 converge on one install path, mechanism 3 either
becomes a `Tensor` or becomes a declared term in the prediction, and the
`withInstalledOutput` question resolves as a consequence rather than as its own debate.

---

## 5. Decisions owed

- **Remove `BuildContext::withInstalledOutput` by splitting binding from allocation.**
  Proposed and not decided. Owned by `MemoryFootprint.md` s4.5, which carries the full
  argument. If adopted, mechanism 1's factories return described-but-unallocated slots
  and `installSharedWorkspace` moves ahead of prediction.

- **Whether mechanisms 3 and 4 become predictable terms or stay attributed after the
  fact.** Owned by `MemoryFootprint.md` s6.4.

- **Module structure.** Each block module exports its workspace, its factory, and its
  block class from one file (`Qwen.AttentionBlock.ixx` exports five entities across three
  concepts in 947 lines). Partitions would give one concept per file; a partition is not
  independently importable, so it buys file organization and not dependency decoupling.
  `QwenGqaWorkspace` is the case where the owning module is wrong rather than merely
  crowded.

- **A block-level Gate A case for the Gemma block kinds**, per the rule in s4.5. Needs
  an exported `makeGemmaBlockWorkspace` first, which the Qwen families already have.

- **Take `setState` off `ITransformerBlock`** (s3.7). The transformer holds the concrete
  block type at the point where it knows the kind -- it branches on
  `isFullAttentionLayer( i )` in the build loop -- so the GQA workspace can be wired
  there instead of through a polymorphic call that one implementer has to no-op. If it
  stays on the interface it needs a name that says what it does and an argument that
  does not name one attention kind.
