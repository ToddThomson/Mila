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

| Term | Where it comes from | Status |
|---|---|---|
| Weights | geometry x quantization policy | exact, closed form |
| KV cache | geometry x context x KV policy | exact, closed form |
| Prefill activation | geometry x the chunk the rule in section 11 picks | exact at that chunk |
| Scratch | the largest single operation request (11.4) | exact, closed form (draft) |
| cuBLASLt workspace | `kCublasLtWorkspaceSize` | fixed 4 MB constant |
| Driver allocation rounding | each allocation over 1 MiB rounded up to the device's allocation granularity (11.8) | exact on Windows, measured |
| Available device memory | read from the device by Mila when the chunk is resolved (11.3) | read live, never modeled |

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

**The flag has a second failure mode, and it is the expensive one.** `withInstalledOutput`
is a *promise by the parent*, and nothing in the type system holds the parent to it.
`QwenTransformer` passed it to every block while installing a workspace into only one of
the two block kinds; the DeltaNet blocks predicted their twenty component outputs at zero
and then allocated every one of them -- 138.2 MiB per layer, ~6.5 GiB across the 27B's 48
DeltaNet layers. Unlike the double count, which overstates and merely refuses
configurations that would fit, this *understates*, so the model loads and then
oversubscribes the card: it capped the 27B at 512 context and made WDDM page the weights
for a 5x decode penalty. It survived because Gate A only ever ran on an all-attention
configuration, where one block kind and one workspace made the promise true by accident.

The rule that follows: **a Gate A case is owed per BLOCK KIND, not per model.** A model
whose stack is heterogeneous has as many pooling contracts as it has kinds, and a passing
gate on one of them says nothing about the others. That rule survives whatever replaces
the flag -- it is about heterogeneous stacks, not about how installation is signalled.

**PROPOSED, NOT DECIDED -- remove the flag by splitting binding from allocation.**

The account above stops one level short of the cause. The parent is not computing anything
unknowable when it calls `withInstalledOutput`: the pooling predicate is `isInferenceMode()`,
the same expression that guards the install twenty lines later. `QwenTransformer` writes it
three times -- onto the block context (`Qwen.ixx:382`) and once per block kind at the install
(`:602`, `:626`) -- and the ~6.5 GiB understatement above was the first site existing while
the third did not. One concept, three authorings, nothing tying them together.

Why the value has to travel at all is the deeper fact. `makeQwenDeltaNetBlockWorkspace()` does
two things in one call: it **describes** the slot set, and it **allocates** it. The description
is pure -- `(config, device, B, chunk, name_prefix)` in, a fixed set of named shapes out. The
allocation is what makes the call impossible before the fit is known. From that fusion
everything else follows: prediction must precede allocation, allocation *is* installation,
so prediction precedes installation, so `output_installed_` is false at prediction time, so a
parallel intent channel must exist.

The proposal is to split the two:

- the workspace factory returns slots **described and unallocated** -- shapes and names, no
  device memory, consistent with the rule that construction allocates nothing;
- `installSharedWorkspace` moves ahead of prediction and binds slot to child, setting
  `output_installed_` as a **fact**, per slot;
- `build()` materializes. `onBuilding` already validates that an installed slot covers the
  build shape (`Linear.ixx:978`); it would allocate there rather than find memory waiting.

What that deletes: `withInstalledOutput` / `hasInstalledOutput` and the `installed_output_`
field (`Component.BuildContext.ixx:208`, `:219`); the `pooled` disjunction and re-stamping
lambda in all three block predictors (`Gemma.Block.ixx:476`, `Qwen.AttentionBlock.ixx:530`,
`Qwen.DeltaNetBlock.ixx:463`); and the duplicated predicate, leaving the install calls as the
sole statement of the pooling decision. Every leaf's
`!output_installed_ && !context.hasInstalledOutput()` collapses to `!output_installed_` --
the same expression `getMemoryStats` already uses, so prediction and measurement read one
source instead of two.

It also removes a constraint the single bool cannot express. The flag is one value over a
block's entire child set: the predictors stamp `pooled` onto every child unconditionally,
which is sound only while a workspace covers every slot the block would otherwise
self-allocate. A partial workspace understates, silently, in the expensive direction --
the Qwen failure mode again, reachable without anyone breaking a promise. Binding per slot
makes a partial workspace expressible and correct.

The cost is a contract change to `installSharedWorkspace` and `onBuilding` across every
pooled component, plus a split of each workspace factory. The argument for paying it is that
the alternatives -- collapsing the predicate to one site, or asserting post-build that a
declared installation happened -- both leave the promise in place and rely on convention to
keep it true.

### 4.6 Sparse layers: resident is not active

A mixture-of-experts layer holds every expert on the device and reads only `top_k` of them
per token, so one parameter figure answers two different questions: whether the model
loads, and what a token costs. `MemoryStats::device_inactive_parameter_bytes` is the part of
`device_parameter_bytes` a single token does not read, and `activeDeviceParameterBytes()`
is the rest.

It is a **subset**, not a fourth category: nothing is allocated for it, so
`totalDeviceBytes()` and every fit verdict are unchanged by it. `operator+=` sums it, which
carries it through the composite aggregation in section 4.3 with no composite written for
it. `MixtureOfExperts` is the only component that sets it, identically in `getMemoryStats`
and `getRequiredMemory`; every dense component leaves it zero, so a dense report cannot
change. A future residency design that streams experts would add its own term beside this
one rather than reinterpret it. See `Gemma4MoE.md` Phase 4.

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

`resolvePrefillChunkSize` runs inside `onBuilding` and threads its result to every block
via `block_context` (`Gemma.ixx:618-625`). `getRequiredMemory` must resolve the chunk
before recursing, or every block's activation buffers are sized against a default.

Because it resolves it anyway, the chunk is a value the prediction already holds.
`getDeploymentFootprint` returns it beside the memory answer; `getRequiredMemory` forwards
to that and keeps its own signature. Every family exposes the resolution as
`prefillChunking(B, T_ctx)`, where its rung table lives once. A caller choosing a context
length needs it: see ChatConfiguration.md section 6.

**Draft, section 11:** the resolution stops being a function of (B, T_ctx) alone. It also
takes the available device memory, and it walks the rungs by asking this same prediction
for the whole footprint at each one, so the chunk it resolves and the footprint it reports
cannot disagree.

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

### 6.4 The scratch buffer, and what the residual is

`getDeviceScratchBuffer` (`CudaExecutionContext.ixx`) serves one buffer to every operation on a
context. Before Phase 6 step 2 it grew on demand during forward passes, stayed out of
`MemoryStats`, and was visible only through a `getScratchHighWaterBytes()` accessor. Since step 2
every network reserves it at `build()` to the largest request its operations make, reports it as
`MemoryStats::device_scratch_bytes`, and a larger request throws; the accessor is deleted. A
permanent allocation is accounted for, not inspected.

**Draft, section 11.4:** scratch is predicted as the largest single request, because
that is exactly what sizes the buffer.

**Attributed 2026-09-13.** Temporary counters in the CUDA allocator split the residual on
both cards, after a load and a full-chunk prefill (11.5):

| Model, chunk, RTX 5060 Ti | Driver rounding | Scratch | Everything else, including the 4 MiB cuBLASLt workspace |
|---|---|---|---|
| Llama 3.1 8B FP4, 256 | 0.08 GiB | 0.11 GiB | 0.04 GiB |
| Gemma 4 12B FP4, 1024 | 0.27 GiB | 0.11 GiB | 0.05 GiB |
| Qwen 3.8 27B FP4, 256 | 0.57 GiB | 0.09 GiB | 0.04 GiB |
| Qwen 3.8 27B cb2-3, 256 | 0.73 GiB | 0.07 GiB | 0.05 GiB |
| Gemma 4 26B-A4B FP4, 64 | 0.29 GiB | 0.25 GiB | 0.07 GiB |

Rounding is the largest term, and it follows the number and sizes of allocations, as the
2026-08-04 note below suspected. The same runs on the RTX 4070 add about 0.31 GiB to the last
column once more than about 8 GiB is written: the Windows budget cut in 11.5. The 2026-08-04
figures were taken on the RTX 4070, with loads past 8 GiB, so they include that cut.

**Measured 2026-08-04, and it falsified the standing hypothesis at the time.** Scratch was
expected to be most of the residual, and to be larger for Gemma because Gemma
quantizes its tied table to FP8 during load while Llama does not:

```
              residual    scratch          unattributed
Llama 8B      0.667 GiB   0.219 (32.8%)    0.449 GiB
Gemma 12B     1.265 GiB   0.250 (19.8%)    1.015 GiB
```

Scratch is ~230 MiB on both and essentially model-independent. It is a real term
worth reporting, but it is not what makes Gemma's residual twice Llama's.

**Leading explanation for the remainder: per-allocation rounding.** `cudaMalloc`
rounds every request up, so the overhead scales with allocation *count*, not bytes.
Gemma has roughly twice Llama's tensor count -- 48 layers with five norms and a
five-way Linear set against 32 layers with two norms and four Linears -- and its
unattributed remainder is roughly 2.3x Llama's. That correlation is suggestive rather
than established; the cheap test is to read `MemoryAllocationStats::allocationCount`
and divide. *Established 2026-09-13 by allocator counters instead: see the table above.*

Note also the measurement noise floor: `consumed` moved 50-70 MiB between runs of the
same configuration, so nothing below ~0.1 GiB should be read as signal.

### 6.5 On Windows, "fits" is not binary -- measured

**Confirmed 2026-08-04 with a mechanism, not just an outcome.** Llama 3.1 8B unquantized
at context 4096, predicted 20.81 GiB, loaded on a 12 GB 4070:

```
Task Manager    11.7 GiB dedicated + ~9 GiB shared  = ~20.7 GiB
predicted                                             20.81 GiB
generation      3.1 tok/s, token gaps 318.2 ms median / 318.8 p99 / 318.8 max
```

The footprint agrees with an independent, out-of-process measurement to within half a
percent -- on the unquantized path, which Gate B does not yet cover.

**The spill is to system RAM over PCIe, not to disk.** 9 GiB in 318 ms is ~28 GB/s,
which is PCIe 4.0 x16 at practical efficiency. A 7 GB/s NVMe would need 1.4 s per
token, four times what was measured. The variance settles it further: 0.6 ms spread
between median and maximum across a whole turn is a fixed-size DMA every token, where
filesystem paging would be jittery.

Shared memory disappears from Task Manager when the session is idle and returns on the
next prompt, which is why an idle snapshot can read 11.6 dedicated / 0.1 shared and look
like the model fits.

That is a residency effect, not a lifetime one. The spilled bytes are persistent
component tensors, allocated for the model's lifetime; they are not released between
prompts, and generation would not stay correct if they were. What cycles is the GPU
aperture *mapping*: "Shared GPU memory" counts system memory currently mapped
GPU-accessible, so when the device idles those mappings are torn down and the bytes
sit in ordinary paged system memory, counted against the process rather than a GPU
figure. The next submission re-maps them.

Idle demotion appears not to reach the pagefile. A cold prompt after the shared figure
drops costs 2103 ms of prefill against 1145 ms warm -- a 958 ms penalty, which is
*less* than reading 9 GiB back from a 7 GB/s NVMe would take on its own. The penalty
being under the disk floor is the argument; re-establishing residency for roughly 2.4M
pages accounts for it without any storage traffic, and the GPU moves that same volume
across PCIe in 318 ms during generation. Not conclusive against a PCIe 5.0 drive; the
clean discriminator is disk read throughput during the cold prompt.

The user-visible consequence is worth stating in the warning: on an oversubscribed
model the first prompt after a pause costs about a second more than the rest.

The practical consequence for a footprint report: **a model that does not fit still runs,
at roughly a thirteenth of its speed.** Refusing would block it; saying nothing would
leave the user to discover it by waiting. Warn.

WDDM oversubscribes into shared host memory rather than failing. Contexts of 65536
and above measured 12282/0 MiB and kept running, pathologically slowly. The number
is exact; the verdict needs a margin, and that margin exists to catch spill, not to
cover measurement error. *Superseded by 11.6:* Chat holds no margin; the prediction includes rounding (11.8)
and is compared with free memory as read. Refuse only when weights alone exceed free VRAM; otherwise
warn and proceed. A false refusal is worse than the problem being solved.

**Two measured additions, 2026-09-13.** On a card that drives a display, Windows also lowers
the process's video memory budget partway through a load (11.5), so a model that fitted the
free memory read at the start can still spill. On Linux none of this section applies: there is
no shared-memory fallback, and an allocation past the device's memory fails the load (11.10).

**Measured 2026-09-14: the spill happens only in a process that sees one GPU.** A standalone probe
allocating and writing 1 GiB blocks: the RTX 4070 alone held 20 GiB, as did the RTX 5060 Ti alone;
with both GPUs visible, the RTX 4070 failed after 11 GiB and the RTX 5060 Ti after 15, and listing
both UUIDs in `CUDA_VISIBLE_DEVICES` failed the same way. So on a machine with two GPUs, a Windows
process behaves like Linux here: an allocation past the device fails the load. Everything above about
spilling was measured with one GPU visible.

---

## 7. Test Strategy

Two comparisons, catching different failure modes.

**Gate A -- composition.** `getRequiredMemory( context )` on an unbuilt graph
against `getMemoryStats()` after a real `build( context )`, per component and per
model. Catches a wrong formula in a leaf and a missing child in a composite. Needs a
device but no weights and no checkpoint.

**Gate B -- reality.** The same figure against the `cudaMemGetInfo` delta across a
real build. Catches what `MemoryStats` cannot see: allocator rounding, and, until Phase 6
step 2 reserves it at build, the section 6.4 scratch buffer once a forward pass has run. Gate B is not expected
to be exact; its job is to *quantify and bound* the residual, attributed 2026-09-13
in section 6.4.

Coverage must be over real components. A mock's `getRequiredMemory` agrees with its
own `getMemoryStats` and proves nothing -- this repeats a defect already paid for
once, where a base-class contract verified against a mock stayed green while five
real composites bypassed it.

Expect several build rounds before Gate A is green. The first disagreement is the
useful output: its size identifies the tensor class that was missed.

**Draft, section 11.** The chunk rule adds two comparisons and narrows one:

- **Scratch joins Gate A.** Reserved at `build()` to its prediction and reported (11.4), so
  predicted equals reported covers it, for every family, with pre-quantized weights and with
  weights quantized during the load. A prediction that is too small throws at the first forward
  that needs more, after a load, a prefill of at least one full chunk, and decode.
- **The rule itself.** The same inputs give the same chunk. The chunk never grows when the
  available memory shrinks. The chosen rung's predicted footprint plus scratch is at most the
  available memory, and the next rung up exceeds it.
- **Gate B** keeps bounding the residual, which no longer includes scratch once 11.4 lands.
  Once rounding is predicted (11.8) the residual is the small-allocation overhead and the fixed remainder,
  under 30 MiB on the two models measured.

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
*Verifiable:* the residual is a named, bounded number instead of an unexplained
difference.

**Green 2026-08-04. Measured, Gemma 4 12B FP4 on the 4070:**

```
context 8192    predicted 8.649 GiB   reported 8.649 GiB
                consumed  9.972 GiB   residual 1.323 GiB (13.3%)

context 4096    weights 6.33 GiB   state 1.97 GiB   total 8.31 GiB
context 32768   weights 6.33 GiB   state 2.49 GiB   total 8.83 GiB
```

Three things this establishes:

- **Predicted equals reported to the byte on a real model**, not just on the synthetic
  configs Gate A uses. The two accountings agree at 12B scale.
- **The context curve is sub-linear, as the bounded ring intends.** Eight times the
  context costs +0.52 GiB of state, because only the global layers' KV grows; the
  sliding layers hold a window-bounded ring. A model that scaled linearly here would
  have indicated the ring was not being used.
- **The residual is 1.323 GiB and is not yet attributed.** It is everything a
  build-time contract cannot see: the grow-on-demand execution-context scratch
  (section 6.4), allocator rounding, and any load-path staging that outlives the load.
  On a 12 GB card that is the margin between fits and does not, so it is reported as a
  measured bound rather than folded into the estimate.

**Phase 4 -- Llama.** Same contract on the Llama chassis. Expected to expose 8.2 as
a reported figure.

**Green 2026-08-04.** Llama 3.1 8B FP4 at context 8192: predicted 9.732 GiB ==
reported 9.732 GiB, consumed 10.331 GiB, residual 0.598 GiB (5.8%).

It did expose 8.2, and larger than recorded:

```
context   Llama 3.1 8B FP4    Gemma 4 12B FP4
8192      9.73 GiB            8.65 GiB
32768     12.08 GiB           8.83 GiB
```

An 8B costs more than a 12B and the gap widens with context. Two causes, separable
because the report splits weights from state: 1.438 GiB of unquantized untied tables,
and an attention scratch that grows 3.12 GiB across an 8x context increase against
Gemma's 0.52 GiB. The second is the larger effect at long context and is a different
defect from the one 8.2 describes.

Note also the residual is 5.8% here against Gemma's 13.3%. Gemma quantizes its tied
table to FP8 during load and Llama does not, which makes load-path staging the leading
suspect for the unattributed term rather than anything in the prefill path.

**Phase 5 -- adaptor.** Chat `/model` pre-flight, warn-and-proceed per 6.5, plus a
context sweep -- the probe is cheap enough to search for the largest context that
fits.

**Green 2026-08-04, and it produced the empirical confirmation of 6.5.** Loading
Llama 3.1 8B **unquantized** at context 4096 predicted weights 14.96 GiB against
10.82 GiB free. It loaded and generated -- at **3.1 tok/s**, against roughly 40 for a
model that fits. Nothing failed; the driver spilled.

Two corrections that fell out of seeing it run:

- The message said *"This will not run."* It ran. A refusal on this path would have
  blocked a working, if slow, configuration -- which is the whole reason 6.5 says warn
  rather than refuse. The wording now states the consequence, not a prohibition.
- **Context is not the lever when the weights alone overflow**, because they do not
  shrink with it. The sweep is offered only on the softer path, where the weights fit
  and trimming context can bring working memory under the line. On the harder path the
  advice is quantization, named as the concrete command.

Free VRAM reaches the adaptor through `Device::getMemoryInfo()` rather than
`cudaMemGetInfo` in Chat: `mila-chat uses no CUDA APIs directly` is a stated property of
that target, and the same accessor is what MIS and the `mila` CLI will need.

**Phase 6 -- prefill chunk rule. DRAFT, section 11.** Each step is separately verifiable.

1. **Load-owned staging.** The quantize-on-load staging buffer is a buffer of its own on the
   execution context, grown during the load and freed by `releaseLoadStaging()` when the load
   returns, and the FP8 per-channel site stages in row blocks like FP4 (11.4). *Verifiable:* weights and scales bit-identical to the current
   path; after a load from BF16 weights, device memory consumed equals that of exported weights
   within noise; load time measured at more than one staging size before one is chosen.
2. **Scratch accounted.** `getRequiredScratchBytes` on the operations that request forward
   scratch (11.4), combined by maximum. The network reserves the execution context's scratch to
   that maximum during `build()` and reports it once in `getMemoryStats()` and
   `getRequiredMemory()`, so Gate A covers it like every other allocation, and a request above
   the reserved size throws instead of growing. `getScratchHighWaterBytes()` is deleted from
   `IExecutionContext`, `CudaExecutionContext` and `Model`, with its four test uses: a permanent
   allocation is accounted for, not inspected. The load staging peak is predicted beside the
   footprint, never reported by an accessor.
   *Verifiable:* Gate A exact with scratch included, on Gemma, Qwen and Llama at every rung, with
   exported weights and with weights quantized during the load; a full-chunk prefill and decode
   on each run without the reserved size being exceeded; an operation predicting one byte less
   than it requests makes the first forward that needs it throw (negative, reverted).
3. **The rule.** Free device memory read by Mila (11.3), the rung walk over the whole footprint
   with rounding included (11.8) in all three transformers, the budget constants and row-cost
   models deleted, and `PrefillChunking`'s members renamed `fits_available_memory` and
   `isMemoryConstrained()`. No public input is added. *Verifiable:* the rule tests in section 7,
   and Gate A exact at every rung.
4. **Callers.** Chat's automatic-context scan runs after a switch releases the outgoing model, so
   each prediction reads the free memory the load will see; MIS and the binding change nothing.
   Chat's fixed allowances are removed: the 10% of the device total that
   `resolveAutomaticContext` holds back (at least 512 MiB) and the 12.5% of the prediction that
   `practicalDeviceBytes` (`Chat.Footprint.ixx:102`) adds. They stack, and they guess. On the
   headless RTX 5060 Ti they grade Gemma 4 26B-A4B FP4 at context 8192 and chunk 64 as 15.83 GiB
   against a 14.34 GiB budget, so the scan finds no context, while section 11.9's measurements put
   the load at chunk 128 about 0.17 GiB under the free memory (derived, with exported weights).
   Grading the model catalogue against device capacity rather than free memory
   (`Chat.Footprint.ixx:145`) is a separate question and is not changed here. *Verifiable:* the
   chunk a scan reports equals the chunk the load builds when free memory has not changed between
   them; no fixed percentage
   remains in Chat's fit path; the 26B on the RTX 5060 Ti grades as fitting at the context and
   chunk the rule picks, and does not spill there; a configuration that measurably spills still
   grades as not fitting.
5. **Documents.** ChatConfiguration.md section 6, `Gemma4InferenceReview.md` 6.4, the budget
   comments in `Gemma.ixx`, `Qwen.ixx` and `Llama.ixx`, and the `PrefillChunking` Doxygen describe
   the rule instead of the budgets.
6. **Driver rounding (11.8)** lands with step 3, and step 3's gate covers it. The Linux measurement in 11.10
   is still owed and does not block it.

The gates for each step are recorded before any run, below.

**Phase 6 step 1 gate -- load-owned staging. Written 2026-09-14, before any run.**

*What it assumes, to confirm before code:*

- The staging size is not a public input. The load owns one buffer of a fixed size, 256 MiB today,
  the ceiling the three capped sites already use. For the sweep below, a temporary change lets the
  test set that size; it is removed after the run, and the size the sweep chooses stays internal to
  the load.
- No accessor is added to `Mila/Src` for any of this. What a criterion needs to observe beyond
  exported bytes, tokens and free device memory is read through temporary counters, removed after
  the run.

*Where:* the RTX 5060 Ti pinned by UUID for every model-level criterion. The component criteria run
in the full suite on both cards. The reference for bit identity is an export from a build of the
committed tree before the change, taken at a short path without touching the working tree.

*Criteria:*

1. **Nothing fitted changes.** `ExportArtifact` output is byte-identical before and after the
   change at the default staging size, for:
   - Llama 3.2 3B from BF16 at FP4 and at FP8 (the FP8 per-channel site)
   - Gemma 4 12B from BF16 at FP4 (the FP4 `Linear` site and the FP8 embedding table)
   - Qwen 3.8 27B `plan` from the fitted source (the 2.54 GiB head through the FP4 site)

   With the temporary sweep change set to 16 MiB, Llama 3.2 3B at FP4 and at FP8 is also
   byte-identical, which forces row blocks through every site that was previously staged whole.
2. **The expert bank is unchanged.** The FP4 bank gates in `MixtureOfExperts.Cuda.cpp` pass
   unchanged at the default size and at 1 MiB.
3. **A fitted load settles where its export does.** For each BF16 and exported pair (Llama 3.2 3B,
   Llama 3.1 8B and Gemma 4 12B at FP4, context 8192), loaded in turn in one process:
   - device memory consumed after the load differs by less than half the staging size
   - after the same full-chunk prefill and eight greedy decode steps, consumed memory still differs
     by less than half the staging size, and the generated tokens are identical

   The noise floor is 50-70 MiB (6.4), so half of 256 MiB separates staging returned from
   staging kept.
4. **Staging is bounded by its size.** Temporary counters on the staging allocation, the method
   used to attribute the residual on 2026-09-13, show the buffer never exceeds the staging size,
   at 256 MiB and, through the temporary change, at 16 MiB, on every load in criteria 1 and 3, and
   on the 26B from BF16 at 256 MiB.
5. **The gate can fail.** Each of these edits must fail the criterion named, and is reverted:
   - the staging buffer not freed at the end of the load: criterion 3
   - the FP8 per-channel site left staging the whole tensor: criterion 4 at 16 MiB on Llama 3.2
     3B FP8
   - a row block starting one row late: criterion 1 at 16 MiB
6. **Nothing else moves.** The full suite passes on both cards, Gemma 4 12B token parity and every
   Gemma, Qwen and Llama footprint literal included, with the temporary sweep change and counters
   removed.

*Measured and recorded, not asserted:* load time of Llama 3.1 8B and Gemma 4 12B from BF16 at FP4,
at staging sizes of 256, 64 and 16 MiB set through the temporary change. Timed after one discarded
load, with the source file resident in host memory, and the file's residency stated beside each
number. The size the load keeps is chosen from these results, as a separate decision after the run.

**Phase 6 step 1 result (2026-09-14, RTX 5060 Ti pinned by UUID).** The reference exports came from a build
of the working tree before the change, whose `Mila/Src` was identical to the committed tree, so no
repository operation was needed.

| Criterion | Result |
|---|---|
| 1. Nothing fitted changes | **byte-identical, 6 of 6**: Llama 3.2 3B FP4 and FP8, Gemma 4 12B FP4, Qwen 3.8 27B `plan` at the default size; Llama 3.2 3B FP4 and FP8 at 16 MiB |
| 2. Expert bank unchanged | the FP4 bank, FP8 table and quantized `Linear` tests, 21 of 21, at the default size and at 1 MiB |
| 3. A fitted load settles where its export does | consumed-memory difference after load and after generation: Llama 3.2 3B 14 MiB, Llama 3.1 8B 0 MiB, Gemma 4 12B 0 MiB; tokens identical for all three |
| 4. Staging bounded by its size | high-water at the default size: Llama 3.2 3B 96 MiB, Llama 3.1 8B 224 MiB, Gemma 4 12B, Qwen `plan` and the 26B 256 MiB; at 16 MiB, 16 MiB |

The 26B FP4 test still fails only its fit criterion, 15,982,200,832 predicted bytes against 15,908,995,072
free, as it did before the change; its weights, expert-bank bytes and eight greedy tokens are unchanged.

**The negatives fail the criteria named.**

- Staging not freed: criterion 3 fails on Llama 3.1 8B (224 MiB kept) and Gemma 4 12B (256 MiB kept).
  Llama 3.2 3B keeps 82 MiB, under the 128 MiB bound, so this criterion cannot see a model whose largest
  fitted tensor is that small.
- The FP8 per-channel site staging the whole tensor: criterion 4 fails, 96 MiB against a 16 MiB limit on
  Llama 3.2 3B FP8. Its export stays byte-identical, as it should: staging more changes no fitted byte.
- Every row block after the first starting one row early, in the FP8 per-channel kernel: criterion 1 fails,
  the Llama 3.2 3B FP8 export at 16 MiB no longer matches the reference. One row early rather than late,
  because late reads past the end of the source on the last block.

**Load time does not depend on the staging size** on these models. Warm, after one discarded load, source
files held in the host file cache:

| Model, from BF16 at FP4 | 256 MiB | 64 MiB | 16 MiB |
|---|---|---|---|
| Llama 3.1 8B | 4.19, 4.18 s | 4.21, 4.17 s | 4.26, 4.29 s |
| Gemma 4 12B | 12.34, 12.34 s | 12.31, 12.33 s | 12.30, 12.57 s |

The spread is under 2% at every size, so the staging size is free to be chosen on memory alone.

**Criterion 6, with every temporary change and negative removed.** All targets build clean. The full suite
runs 1962 tests on each card, pinned by UUID: 1960 pass, 1 skipped (the long-standing Swiglu BF16 backward)
and 1 fails, the 26B FP4 fit, on its two fit assertions only -- 83.8 MiB over free memory on the RTX 5060 Ti,
and unreachable on the 12 GiB RTX 4070. Gemma 4 12B token parity, the Qwen and 26B layer-streamed harnesses
and every footprint literal pass on both. The new `QuantizeOnLoad.Footprint.Cuda.cpp` passes on both.

**Phase 6 step 2 gate -- scratch accounted. Written 2026-09-14, before any run.**

*What the code does today, which the gate is written against.* Three operations request forward
scratch, and every size is fixed once the operation is built:

| Request | Where | Bytes | When |
|---|---|---|---|
| Staged prefill GEMM | `CudaLinearOp.ixx:1295` | strip rows x input width x 2 | `kUsesStagedPrefill`: FP8 per-channel without the W8A16 GEMM, codebook, FP4 without the FP8-activation or fused path |
| FP8-activation prefill | `CudaLinearOp.ixx:791` | strip rows x input width, plus bucket x input width, each 16-byte aligned, plus bucket x 4 | `kUseFp8ActivationPrefillPath`; the bucket is the plan bucket of the prefill chunk |
| Fused decode attention | `CudaGqaOp.ixx:834` | B x heads x `kMaxDecodeSplits` x (head width + 2) x 4 | BF16, `use_flash_decode_`, and a supported head geometry |

The strip width comes from the operation's own `max_staging_bytes_`, a constructor argument tests
override, so a prediction reads the instance's value, never `kMaxStagingBytes`. Load staging is not
forward scratch (step 1).

*What it assumes, to confirm before code:*

- `MemoryStats` gains `device_scratch_bytes`, counted in `totalDeviceBytes()`. Only a network sets it;
  no component or operation does, so summing across the tree cannot count it twice. `Operation` gains
  `getRequiredScratchBytes( const BuildContext& )`, defaulting to 0, and composites combine it by maximum.
- `IExecutionContext` gains `reserveScratch( std::size_t bytes )`, a no-op on CPU. A context that has
  reserved -- zero bytes included -- throws `std::logic_error` naming the requested and reserved bytes on
  any larger request; a context that never reserved keeps growing on demand, which is what a component
  built directly in a test gets. The reservation is made in `onBuilding` of every transformer: Gemma,
  Llama, Qwen and GPT-2, which reserves its prediction even when that is zero, so no network is left
  growing unaccounted.
- `getScratchHighWaterBytes()` is deleted from `IExecutionContext`, `CudaExecutionContext` and `Model`,
  and its four test uses with it. No accessor replaces it: what a criterion needs beyond `MemoryStats`
  is read through temporary counters, removed after the run.

*Where:* the RTX 5060 Ti pinned by UUID for the model criteria; Qwen 3.8 27B cb2-3 on the RTX 4070, the
card that model is measured on. The component criteria run in the full suite on both cards.

*Criteria:*

1. **Gate A holds with scratch in it.** Predicted equals reported, every category including
   `device_scratch_bytes`, in every existing component and transformer footprint test and at model level
   for Gemma 4 12B FP4, Llama 3.1 8B FP4 and Qwen 3.8 27B FP4 from exported weights, Gemma 4 12B FP4 from
   BF16 weights, and Qwen 3.8 27B cb2-3. Gemma 4 12B is also taken at context 512, where the prefill
   chunk and so the FP8-activation bucket differ from context 8192. *Amended during the run: the gate
   first named context 1024, but Gemma takes a 1024-row chunk at context 8192 as well, so that case
   repeated the same bucket.*
2. **The reservation is exactly what forward uses.** Temporary counters record the largest scratch request
   each context receives. After a prompt that fills one prefill chunk and eight decode steps, the largest
   request equals `device_scratch_bytes`, exactly, on every model load in criterion 1. `Linear.Cuda.cpp`'s
   footprint tests do the same for one operation at the default strip width and at a `max_staging_bytes`
   that forces strips.
3. **No request exceeds the reservation.** Every generation in the full suite on both cards runs under the
   throw, GPT-2 included.
4. **Scratch moves from generation to build.** Against step 1's recorded consumption on the RTX 5060 Ti, the
   growth from after the load to after generation in `QuantizeOnLoad.Footprint.Cuda.cpp` -- Llama 3.2 3B
   +50 MiB, Llama 3.1 8B +116 MiB, Gemma 4 12B +120 MiB -- falls by `device_scratch_bytes` rounded up to
   2 MiB, within 16 MiB, for both weight sources.
5. **The accessor is gone.** A search of the tree finds no `getScratchHighWaterBytes`, and the Gate B tests
   report their residual against a prediction that now includes scratch.
6. **The gate can fail.** Each of these edits must fail the criterion named, and is reverted:
   - `CudaLinearOp::getRequiredScratchBytes` one byte short of the FP8-activation request: criterion 3,
     the first full-chunk prefill on Gemma 4 12B FP4 throws
   - Gemma's transformer not reserving: criterion 4, scratch is allocated during generation again, which
     `ScratchReservation.Cuda.cpp` asserts as device memory growth under 16 MiB. *Amended during the run:
     the gate first named criterion 1, but reported scratch comes from the operations rather than from the
     reservation, so criterion 1 cannot see a missing reservation and criterion 3 cannot either.*
   - a composite summing its children's scratch instead of taking the maximum: criterion 2, the
     reservation exceeds the largest request
7. **Nothing else moves.** All targets build; the full suite passes on both cards with temporary counters
   and negatives removed, Gemma 4 12B token parity, the Qwen and 26B layer-streamed harnesses and
   `QuantizeOnLoad.Footprint.Cuda.cpp` included. Footprint literals that change by exactly the scratch
   term are updated to the new totals and listed in the result, each with its scratch bytes.

**Phase 6 step 2 result (2026-09-14).** The design landed with one change from the assumptions above:
`device_scratch_bytes` is combined by maximum inside `MemoryStats::operator+=` and set by the two leaf
components whose operations request scratch, `Linear` and `GroupedQueryAttention`, instead of by a
network-only traversal. Every composite already aggregates its children through `operator+=`, so the field
reaches the network with no composite changed. Fused decode is declared on the build context with
`BuildContext::withFusedDecode`, which Gemma and Qwen set on their block contexts in `onBuilding` and in
`getRequiredMemory` alike; the calls to `setUseFlashDecode( true )` after build are gone.

| Criterion | Result |
|---|---|
| 1. Gate A with scratch | predicted equals reported: Gemma 4 12B FP4 at context 8192 and 512, Llama 3.1 8B FP4, Qwen 3.8 27B FP4 (RTX 5060 Ti), Qwen 3.8 27B cb2-3 at context 4096 (RTX 4070), and the loads from BF16 weights of Llama 3.2 3B, Llama 3.1 8B and Gemma 4 12B; `Linear.Cuda.cpp` asserts it per category |
| 2. Reservation equals the largest request | exactly, on every load in criterion 1: Gemma 4 12B 121,901,056 bytes at context 8192 and 119,932,928 at 512, Llama 3.1 8B 119,539,712, Qwen FP4 90,243,328, cb2-3 74,712,064, Llama 3.2 3B 51,906,560 |
| 3. No request exceeds the reservation | every generation above, and the full suite on the RTX 5060 Ti, runs under the throw |
| 4. Scratch moves from generation to build | growth from load to generation, against step 1: Llama 3.2 3B +50 to +0 MiB, Llama 3.1 8B +116 to +0 MiB, Gemma 4 12B +120 to +2 MiB; each fall is the scratch rounded up to 2 MiB, within 2 MiB |
| 5. The accessor is gone | no `getScratchHighWaterBytes` remains in the tree's code |

No existing footprint literal changed: the literal tests compare parameter and state bytes, which scratch
does not enter. The 26B FP4 prediction rose by exactly its scratch, 27,396,096 bytes, to 16,009,596,928.

The per-operation counter check in `Linear.Cuda.cpp` at a forced strip width was not added: every quantized
`Linear` shape of the five models is held to the same equality through the model runs.

**Negatives.**

- `CudaLinearOp`'s FP8-activation scratch one byte short: criterion 3 fails. Gemma 4 12B reserves
  121,901,055 bytes and the first full-chunk prefill throws, naming the 121,901,056-byte request.
- Gemma's transformer not reserving: criterion 4 fails, 120 MiB of growth during generation against the
  16 MiB bound. Predicted still equals reported, as the amended criterion says it must.
- Scratch summed across the tree instead of taking the maximum: criterion 2 fails, and only criterion 2.
  Gemma 4 12B reserves 12,386,304,000 bytes against a largest request of 121,901,056, and Llama 3.1 8B
  7,415,791,616 against 119,539,712. Both reservations succeed, predicted still equals reported, nothing
  throws and memory does not grow, so both reservation tests pass. **No permanent test catches an
  over-reservation**: criterion 2 is observable only through the temporary counters, which are removed.

**Criterion 7, with every temporary change and negative removed.** All targets build clean. Every GPU run above,
and the first two full suites, **saw one GPU at a time** through `CUDA_VISIBLE_DEVICES`: 1967 tests on each card,
1965 pass, 0 fail, 2 skipped. That is not how the suite normally runs, and it hid two aborts. A process that sees
one GPU places an allocation past the device's memory in host memory; a process that sees both fails it (a
standalone probe: the RTX 4070 alone held 20 GiB, the RTX 4070 with both visible failed after 11 GiB, and so did
listing both UUIDs). With both visible, CUDA's device 0 is the 12 GiB RTX 4070.

The run that counts is Todd's: a clean `x64-profile` build, both GPUs visible. Its first run aborted in the 26B FP4
test and, with that disabled, in `ScratchReservationCudaTests.Qwen38_27B_Fp4_Context8192` -- both load about 15
GiB, the allocation fails, and `CudaDeviceMemoryResource::do_deallocate` rethrows from a destructor. With both
**disabled**, 1965 tests run: 1962 pass, 0 fail, 3 skipped (Swiglu BF16 backward, and the growth check on the Gemma
4 12B context 8192 and cb2-3 cases), 24 disabled. Which cases the growth check skips, and whether
`QuantizeOnLoad.Footprint.Cuda.cpp`'s Gemma 4 12B comparison fails (323 MiB in one run, a pass in the next),
depends on when the Windows budget cut lands on that display card. All of it is tracked in
`Mila/Issues/Untriaged.md` for rc.1. The cb2-3 tests, including the new context 4096 case, pass on the RTX 4070
with both GPUs visible. Chat: `ChatRichTextTests` 33 of 33, and piped sessions with Gemma 4 12B FP4 and Llama 3.2
3B FP4 answer a factual question and write a correct function, both with one GPU visible and with both.

**Phase 6 step 3 gate -- the rule. Written 2026-09-15, before any code or run.**

*What the code does today, which the gate is written against.* Each transformer resolves its chunk in
`prefillChunking( B, T_ctx )` from a row-cost model and a fixed budget: `Gemma.ixx:464` (budget `:119`, row cost
`:952`, global KV `:1007`), `Qwen.ixx:525` (budget `:134`, row cost `:938`, KV `:967`) and Llama's free function
`computePrefillChunking` (`Llama.ixx:85`, cap `:67`). `getRequiredMemory` resolves the chunk first and then sizes
everything at it (`Gemma.ixx:352`, `Qwen.ixx:447`, `Llama.ixx:383`); `onBuilding` resolves it the same way; the
model entry points return `prefillChunking( 1, context_length )` beside the memory (`GemmaModel.ixx:610`,
`QwenModel.ixx:627`, `LlamaModel.ixx:506`). Predicted bytes come from `storageBytes` (68 sites in `Mila/Src`),
reported bytes from `getStorageSize()` (76 sites in `Mila/Src/Dnn/Components`), and neither rounds. Nothing in
`Mila/Src` queries the allocation granularity. Chat reads `isBudgetConstrained()` (`Chat.Footprint.ixx:527`), and
two footprint tests assert that a long context is budget-constrained (`GemmaModel.Footprint.Cuda.cpp:184`,
`LlamaModel.Footprint.Cuda.cpp:161`).

*What it assumes, to confirm before code:*

- **Free memory reaches the walk inside the transformer.** Chunk resolution reads
  `DeviceRegistry::instance().getDevice( device )->getMemoryInfo()` for the transformer's own device, at the moment
  it resolves -- in `getRequiredMemory` and at the start of `onBuilding`, before anything is allocated. The rung walk
  is a private function that takes the bytes as an argument. `LanguageModelConfig`, `BuildContext`, the entry
  points and `prefillChunking( B, T_ctx )`'s signature do not change. A reading with `total_bytes == 0` (the CPU
  device, or a failed query) takes the largest rung the context permits and reports that it fits. A transformer a
  unit test builds directly reads free memory like any other; its footprint is small, so it takes the largest rung.
- **The walk asks the prediction.** The body of each `getRequiredMemory` becomes a footprint at a given chunk; the
  walk calls it from the largest rung down and stops at the first whose `totalDeviceBytes()` is at most the free
  memory. `getRequiredMemory`, `onBuilding` and `prefillChunking` all go through the walk, so the chunk reported,
  the chunk built and the footprint reported cannot disagree. Deleted: the three budgets, both `computeChunkRowCostBytes`,
  `prefillGlobalKvBytes`, `prefillKvBytes`, `computePrefillChunking`, the measured table above `kQwenPrefillActivationBudgetBytes`,
  and the budget comments. Kept: the rung tables, the floors, `kGemmaPrefillChunkOverride`.
- **Rounding is applied on both sides of Gate A by one rule.** A helper beside `storageBytes` rounds one
  allocation's bytes up to a multiple of the granularity when it exceeds 1 MiB, and every prediction and report of a
  device allocation goes through it, one allocation at a time -- the 68 and 76 sites, the workspaces'
  `deviceStorageBytes()`, `ropeCacheBytes`, and the scratch reservation. Predicted equals reported therefore still
  holds exactly, and a site that sums two allocations before rounding breaks it. The CUDA device reads its
  granularity once with `cuMemGetAllocationGranularity`; how it reaches the helper inside `Mila/Src` is proposed at
  code time, not assumed here. A device with no granularity (CPU) rounds nothing.
- **Mila links the CUDA driver API, decided with Todd 2026-09-15.** `cuMemGetAllocationGranularity` is declared in
  `cuda.h` (CUDA 13.3 `include/cuda.h:13544`), not in the runtime API Mila uses, so `CUDA::cuda_driver` joins the
  `PUBLIC` CUDA libraries (`Mila/CMakeLists.txt:811`). It adds no compiled unit: `cuda.h` is included by
  `CudaDevice.ixx` alone, beside a runtime header of the same size. The driver library comes with the NVIDIA
  driver, so no artifact ships it: the Linux wheel's `auditwheel repair` gains `--exclude libcuda.so.1`
  (`Docker/build-wheel.sh:110`), and the Windows wheel, which has no repair step, needs nothing.
  Rejected: the documented 2 MiB written as a constant (a number from the development machine), and measuring
  one allocation against free memory at run time (anything else allocating in between corrupts it).
- **Load staging stays outside the rule.** The rule compares the footprint after the load. A load that quantizes
  its weights adds up to the staging size on top while it runs (11.4); where the driver does not spill, a pick
  within that of the free memory fails the load. That is left open with the staging size, as 11.4 already says.
- **Steps 3 and 4 land in one commit.** From step 3 on, Chat's scan reads free memory through the prediction, and
  today it scans before releasing the outgoing model (`Chat.ixx:1490`), so a switch would scan against the old
  model's memory until step 4 moves it.
- **Renames:** `fits_activation_budget` to `fits_available_memory`, `isBudgetConstrained()` to
  `isMemoryConstrained()`, at their definitions and their three uses.

*Where:* the RTX 5060 Ti pinned by UUID for criteria 1-5, which drives no display, so the Windows budget cut does
not enter; Qwen 3.8 27B cb2-3 also on the RTX 4070 for criterion 1, the card it is measured on. Criterion 7 in Todd's
environment: a clean `x64-profile` build with both GPUs visible.

*Criteria:*

1. **Gate A holds with rounding, at every rung.** Predicted equals reported, every category, in every existing
   component and transformer footprint test, and at model level at each rung the context permits -- forced through
   a temporary change to the free memory the walk is given, removed after the run -- for Gemma 4 12B FP4 (context
   8192, rungs 1024 to 64), Qwen 3.8 27B FP4 (context 8192) and cb2-3 (context 4096), and Llama 3.1 8B FP4
   (context 8192, rungs 512 to 128), from exported weights, and Gemma 4 12B from BF16 weights at its natural pick.
   The predicted totals at adjacent rungs differ, so equal totals identify the rung built.
2. **The prediction meets the driver.** After a load, a prefill of one full chunk and eight decode steps,
   consumed device memory less the predicted total is at least 0 and under 64 MiB, for Gemma 4 12B FP4, Qwen 3.8
   27B FP4 and cb2-3, and Llama 3.1 8B FP4 at their natural pick. Measured 2026-09-15, the same quantity is 300
   MiB and 780 MiB without rounding (11.8), so the bound separates predicted rounding from absent rounding.
   `GetRequiredMemory_BoundsActualConsumption` in both footprint tests keeps its `predicted <= consumed` and
   tightens its residual bound from a quarter of consumption to 64 MiB.
3. **The rule picks what 11.2 says.** For each model in criterion 2 at contexts 4096 and 8192, and Gemma 4 12B and
   Llama 3.1 8B at 16384: the chosen rung's predicted total is at most the free memory read, and either the next
   rung up exceeds it or the chunk is the largest rung the context permits. Through the temporary change, with the
   free memory stepped down in 64 MiB steps from the device's total to the weights alone, the chunk never grows as
   memory shrinks, and below the floor's total the floor is returned with `fits_available_memory` false.
4. **Prediction and load agree, and only the load warns.** `getDeploymentFootprint` then `fromPretrained`, with
   nothing allocated between, give equal totals (criterion 1) on every load in criterion 2. A prediction whose floor
   does not fit logs nothing; the build of the same deployment logs exactly one warning.
5. **Section 11.9 comes true.** With one GPU visible, Qwen 3.8 27B FP4 at context 8192 on the RTX 5060 Ti takes a
   rung above 64 and generates; Gemma 4 26B-A4B FP4 at context 8192 takes a rung whose predicted total fits its free
   memory and its load consumes no more than the free memory it read. Each pick is recorded against 11.9's.
6. **The gate can fail.** Each of these edits must fail the criterion named, and is reverted:
   - rounding removed from the helper, on both sides at once: criterion 2 (Gate A still passes, which is why
     criterion 2 exists)
   - one site rounding the sum of two allocations instead of each: criterion 1
   - the walk returning the rung above the first that fits: criterion 3
   - `onBuilding` resolving against a fixed free-memory figure instead of the reading: criterion 4
7. **Nothing else moves.** All targets build. The full suite passes in Todd's environment with zero failures and
   nothing newly disabled or skipped, with the temporary change and negatives removed; Gemma 4 12B token parity,
   the Qwen and 26B layer-streamed harnesses, `QuantizeOnLoad.Footprint.Cuda.cpp` and `ScratchReservation.Cuda.cpp`
   included. The two tests that assert a budget-constrained long context are rewritten to criterion 3's properties
   against the free memory they read. Footprint literals that change by exactly their rounding are updated and
   listed with the rounding bytes. The new link reaches every consumer: the FetchContent and CPM packaging
   gates configure, build and link; the Linux CUDA build links in WSL with CI's configure; a Linux wheel built by
   `Docker/build-wheel.sh` does not contain `libcuda`; the Windows wheel imports in a clean environment. The
   peak memory of a clean `x64-profile` build is recorded before and after the change. Chat: `ChatRichTextTests`, and a piped session with Gemma 4 12B FP4 and Llama 3.2
   3B FP4 under automatic context, including a switch between them.

*Measured and recorded, not asserted:* the time one chunk resolution takes on Gemma 4 12B, the 26B and Qwen 3.8 27B,
warm; and the wall time of Chat's automatic-context scan for Gemma 4 12B FP4 before and after, since 11.2 says a walk
multiplies the cost of each prediction and that cost inside the scan was never measured.

**Phase 6 step 3 result, criteria 1-6 (2026-09-15, RTX 5060 Ti with one GPU visible unless named).** Criterion 7,
the measured costs and step 4 are not done; this is not a green result.

| Criterion | Result |
|---|---|
| 1. Gate A with rounding, every rung | predicted equals reported, every category, to the byte: Gemma 4 12B FP4 at context 8192, rungs 64-1024; Llama 3.1 8B FP4 at 8192, rungs 128-512; Qwen 3.8 27B FP4 at 8192, rungs 64-512; cb2-3 at 4096, rungs 64-1024 on the RTX 5060 Ti and 64-512 on the RTX 4070. Not loaded because the rung exceeds the card's free memory: Qwen FP4 rung 1024 (15.397 GiB against 14.816 free) and cb2-3 rung 1024 on the RTX 4070 (by 5 MiB). The full suite, every existing footprint test included, passed before the codebook fix below: 1968 run, 1967 pass, 1 skipped (Swiglu BF16 backward); it is rerun under criterion 7. |
| 2. The prediction meets the driver | consumed less predicted, after a full-chunk prefill and eight decode steps: Gemma 4 12B 22.3 MiB, Llama 3.1 8B 22.0, cb2-3 32.5, Qwen FP4 42.6, all at context 8192. After the load alone, in the two Gate B tests: Gemma 4 12B 6 MiB, Llama 3.1 8B 8 MiB, from 298 and 33 MiB before rounding. |
| 3. The rule picks what 11.2 says | zero violations over free memory stepped down in 64 MiB from the device total to the weights: Gemma 4 12B and Llama 3.1 8B at 4096, 8192 and 16384; Qwen FP4 and cb2-3 at 4096 and 8192. Every drop in chunk happened where the larger rung's total first exceeded the free memory. |
| 4. Prediction and load agree, only the load warns | equal totals on every load in criterion 1. With free memory below Gemma 4 12B's floor: the prediction logged no warning, the build of the same deployment exactly one. |
| 5. Section 11.9 comes true | Qwen 3.8 27B FP4 at 8192 takes 512 rows (64 before; 11.9 predicted 512) and generates. Gemma 4 26B-A4B FP4 at 8192 takes 256 rows (11.9 predicted 256): 14.777 GiB predicted against 14.803 free, 14.791 consumed, 12 MiB left, greedy tokens unchanged. |

**Negatives, criterion 6.** Each failed the criterion named, and was reverted:

- rounding removed from the helper on both sides: criterion 2, residual 306.8 MiB on Gemma 4 12B, while Gate A still
  held at all five rungs -- which is why criterion 2 exists
- Linear's weight and scales rounded as one allocation: criterion 1, parameter bytes 6,830,943,744 predicted against
  7,090,695,680 reported on Gemma 4 12B, at every rung
- Qwen's walk returning the rung above the first that fits: criterion 3, 20 picks whose total exceeded the free memory
- Llama's build resolving with no free-memory reading: criterion 4 (read through Gate A at forced rungs), state
  2,004,363,776 predicted against 4,577,569,280 built at rung 128, the build having taken 512

**A defect criterion 1 found, fixed.** `Linear::getMemoryStats` never reported a codebook format's table and high-bit
plane, which `getRequiredMemory` has always counted, so a loaded cb2-3 reported 960 MiB fewer parameter bytes than it
holds, on both cards. No earlier test compared cb2-3's parameter bytes. Both are now reported.

**Changes the result forced.** The Gemma 4 26B-A4B FP4 test's three layout literals changed by exactly their rounding
(one layer's bank 428,212,224 to 432,013,312 bytes), and its skip on not fitting became the failure its comment
promised once this rule landed. The test stays disabled: in a process that sees both GPUs it targets the 12 GiB card.

**Found by the run, decided 2026-09-16.** On the RTX 4070, which drives a display, the two Gate B tests measure 224 and 225 MiB
beyond the prediction, the Windows budget cut in 11.5, and fail the 64 MiB bound criterion 2 set for them. On
2026-09-16 the same tests read 720-1180 MiB there (11.5), with nothing in Mila changed. Decided: both tests run on the
lowest CUDA device that drives no display (`Mila/Tests/Common/DeviceWithoutDisplay.h`, NVML display active, matched
by PCI address), hold the 64 MiB bound there, and skip the bound, printing the residual, where every visible device
drives one. The exact agreements with `getMemoryStats` are asserted on any device. Measured after the change, both
GPUs visible: Gemma 4 12B 6 MiB and Llama 3.1 8B 21 MiB on CUDA 1, the RTX 5060 Ti. The step 1 criterion 3 test
(`QuantizeOnLoad.Footprint.Cuda.cpp`) follows the same rule: on the RTX 4070 its Llama 3.1 8B exported arm read
10389.6 MiB after generation in one run and 10679.0 MiB in the next and failed the 128 MiB comparison; on the RTX
5060 Ti the two arms differ by 0.0-14.0 MiB for all three models, and token agreement is asserted on any device.

**Criterion 7, in progress.** Every Windows target builds, and the full suite passes on the RTX 5060 Ti pinned by
UUID: 1968 run, 1967 pass, 1 skipped. The FetchContent consumer configures, builds and links, with the driver
library on its link line. In WSL with CI's configure, the CUDA build passes in 1520 seconds and the CPU-only build
in 469; clang caught one portability defect this change introduced, a missing `import Dnn.RuntimeMode;` in the three
transformers, since fixed. Two deviations from CI, both recorded rather than worked around: the local CUDA configure
turns the Python bindings off, because this machine has no Python development headers, and the local CPU suite run
under `ctest -j 4` fails 15 tests that pass on Windows from the same tree and pass when rerun alone, which is test
parallelism rather than this change. A full rebuild with the change peaks at 1,410 MiB in its largest compiler process
and 12,108 MiB across every compiler process at once, at ninja's default parallelism on ten cores.

**Not run, and why.** Three of criterion 7's checks were declined rather than forced, Todd 2026-09-15:

- **The before-and-after build memory comparison.** The "before" half needs the committed tree built separately, and
  the figure above is the "after" half alone. What the link adds is one library on the link line and no new translation
  unit, and the largest single compiler process sets a build's memory floor.
- **The Linux wheel.** `Docker/build-wheel.sh` clears `out/wheel/mila_llm-*linux*.whl` before building, and that
  directory holds the released beta.3 wheels. The `--exclude libcuda.so.1` line is therefore reviewed, not exercised;
  the next release build exercises it.
- **The Windows wheel import.** Its script builds into the `x64-wheel` preset directory, which is not this session's
  to overwrite.

**The two measured costs are not measured.** The gate asks for the time of one chunk resolution and for Chat's
automatic-context scan before and after. The "before" scan cannot be timed without the committed tree, for the same
reason as the build comparison, and neither figure gates anything: the walk asks the same prediction at up to five
rungs, and the scan's own probes are what dominate it.

Still owed: the run in Todd's environment.

**Phase 6 step 4 result (2026-09-15).** Chat's two fixed allowances are gone: `resolveAutomaticContext` budgets against
the device's free memory with nothing held back, and `practicalDeviceBytes` -- the 12.5% it added to every prediction --
is deleted, so `gradeFootprint` grades the prediction itself. Because every prediction now picks its chunk against free
memory, a scan taken while a model is resident measures a card that still holds it, so Chat releases first: a model
switch releases before the scan it already released before the load, and `/context auto` releases, scans, then loads.
`/context` and the "context N would fit" suggestion read the scan the last load ran, carried in one session member,
rather than taking a new one; before the first such scan, `/context` reports the number startup resolved. MIS, the
Python binding and the catalogue's grading against device capacity are unchanged.

Verified in piped sessions on the RTX 5060 Ti: Llama 3.2 3B FP4 opens at 62464 under auto and `/context` names it as
measured at startup; `/model load gemma-4-12b-it-fp4` switches, resolves 131072 for Gemma after the release, and
`/context` then reports 131072 from that scan; the switched-in model answers correctly. `/context auto` on Llama
releases, scans, reloads, and `/context` afterwards reports that scan, held back to keep a full prefill chunk.
`ChatRichTextTests` passes 33 of 33, and the full suite passes 1967 with none failing.

---

## 10. Non-Goals

- **Predicting throughput.** This answers whether it fits, not how fast it runs.
- **Modeling the user's other processes.** Free VRAM is read, never predicted.
- **Training footprints.** Gradient buffers are allocated on first
  `setEvaluation( false )`, outside `build()`. The contract has a category for them
  and Phase 1-5 leave it zero.
- **A `mila.json` schema change.** The geometry is already in the artifact header.
- **Operating system memory policy.** Windows budget changes are measured and recorded (11.5), not
  predicted, and Mila holds no memory back for them (11.3).

---

## 11. Prefill Chunk Rule

**DRAFT 2026-09-13, for review.** Decided with Todd on 2026-09-13: the rule in 11.2; that scratch is
predicted rather than allowed for; that Mila predicts only its own allocations; and that no number measured on
the development machine enters `Mila/Src`, because Mila is a static library inside the user's `main()`.
Zero-filling memory to make the driver commit it was dropped earlier as redundant and slow; nothing here
depends on it. Decided on 2026-09-15: available memory is read by Mila with no public input (11.3), and the
driver's rounding of Mila's allocations is predicted (11.8).

**Post-v0.20 direction (2026-09-16): `Deployment.md`.** The rule below stays the rule, but where it runs
moves. Deployment planning takes one free-memory reading per device and resolves the chunk before
construction, the build executes the chunk it is given, and pricing stops reading the bound context
(`Deployment.md` sections 4 and 7, Phases 1-2). Until then this section describes the tree as it is.

### 11.1 What is wrong today

Each family chooses its prefill chunk against a fixed activation budget: 1536 MiB for Gemma
(`Gemma.ixx:119`) and Llama (`kPrefillScratchByteCap`, `Llama.ixx:67`), and 512 MiB for Qwen
(`Qwen.ixx:134`), measured on the 12 GiB card. The budget caps the chunk-scaled buffers and knows nothing
about the weights or the card, so a constant that fits one model on one card is wrong for the next. With
the scratch measured in 11.4 added:

- Gemma 4 26B-A4B FP4 at context 8192: the budget admits 512 rows, 15.14 GiB, against 14.82 GiB free on
  the RTX 5060 Ti. It does not fit.
- Llama 3.1 8B FP4 at context 16384: the cap admits 512 rows, 11.35 GiB, against 10.85 GiB free on the
  RTX 4070. It does not fit.
- Qwen 3.8 27B FP4 at context 8192 on the RTX 5060 Ti: the budget forces the 64-row floor and its
  "cannot prefill efficiently" warning, 13.61 GiB, where 512 rows is 14.20 GiB and fits.

`Gemma.ixx:116` and `Gemma4InferenceReview.md` 6.4 call a live-memory budget a BACKLOG follow-up. There is
no such BACKLOG item.

### 11.2 The rule

**The chunk is the largest rung whose whole predicted device footprint fits the available device memory.**

- **Rungs** stay per family: 1024, 512, 256, 128, 64 for Gemma and Qwen; 512, 256, 128 for Llama. A rung
  longer than the context is skipped, as today.
- **The whole predicted device footprint** at a rung is parameters, plus state -- KV caches, pooled
  workspaces, the GQA transient and every chunk-scaled buffer -- plus predicted scratch (11.4). It comes
  from the same `getRequiredMemory` arithmetic that reports the footprint, so the number the rule compares
  is the number the footprint reports at that rung. There is no separate row-cost model to drift from it.
- **Available device memory** is the device's free memory, read by Mila (11.3).

The rungs are walked from the largest down, and the first that fits is the chunk. One rule serves Gemma,
Qwen and Llama. Routed layers need nothing of their own: their per-layer buffers are already in the
footprint.

This deletes `kGemmaPrefillActivationBudgetBytes`, `kQwenPrefillActivationBudgetBytes` and the measured
table above it, `kPrefillScratchByteCap`, the row-cost models (`computeChunkRowCostBytes` on Gemma and
Qwen, the scratch arithmetic in Llama's `computePrefillChunking`), and the KV terms subtracted from the
budgets (`prefillGlobalKvBytes`, `prefillKvBytes`). `kGemmaPrefillChunkOverride` stays as the debug
override.

A prediction now walks up to five rungs instead of one. Measured 2026-08-17, one Gemma 48-layer
prediction costs 1-2 ms warm, so a full walk stays near 10 ms, and most walks stop at the first or second
rung. What that does to Chat's automatic-context scan, which predicts at every candidate context, is not
yet measured.

### 11.3 Available device memory is read by Mila

**Decided with Todd, 2026-09-15: there is no public input.** Nothing on `LanguageModelConfig`, the model
entry points or the Python binding changes. An input was drafted here so a caller could hold memory back for
what the prediction leaves out; no end user was found who would set it, and the largest term it would have
covered, driver rounding, is predicted instead (11.8).

- **What is read:** the device's free memory, through `Device::getMemoryInfo()`, when the chunk is resolved --
  in `fromPretrained` and in `getDeploymentFootprint`. A prediction describes the load that would happen at
  that moment, so two predictions made at different moments can disagree.
- **What is left out:** the fixed remainder (11.5) and the small-allocation overhead (11.8), under 30 MiB
  together on the models measured, and on a card that drives a display the Windows budget cut (11.5). Mila
  holds nothing back for any of them. A pick that leaves less than the cut free on a display card spills once
  Windows lowers the budget; a pick that leaves less than the shortfall free fails where the driver does not
  spill (11.10).
- **Only device bytes count.** Host-resident allocations, such as Qwen's embedding table, do not.

**Open for the Phase 6 step 3 gate:** how the free memory reaches each transformer's rung walk without a
public addition, and what a transformer built directly, as unit tests do, uses. To confirm there: the
transformer reads it from its own device where it resolves the chunk, and the rung walk takes it as an
argument so the rule tests in section 7 can give it a fixed value.

### 11.4 Scratch is predicted

The execution context's scratch buffer (`CudaExecutionContext::getDeviceScratchBuffer`) is a single
allocation that grows to the largest request any operation makes and never shrinks. Its cost is the
largest request, not a sum, and every request is sized from shapes fixed at build time:

| Request | Where | Bytes |
|---|---|---|
| Quantize-on-load staging, FP4 `Linear` | `CudaLinearOp.ixx:490` | min(BF16 source bytes, 256 MiB) |
| Quantize-on-load staging, FP8 per-channel `Linear` | `CudaLinearOp.ixx:475` | BF16 source bytes plus 4, no ceiling |
| Quantize-on-load staging, FP8 embedding table | `CudaTokenEmbeddingOp.ixx:184` | min(BF16 source bytes, 256 MiB) |
| Quantize-on-load staging, FP4 expert bank | `CudaMoeOp.ixx:167` | min(BF16 source bytes, 256 MiB) |
| Staged BF16 prefill GEMM | `CudaLinearOp.ixx:1310` | strip rows x input width x 2; one strip is the whole matrix unless that exceeds 256 MiB (32 MiB for codebook weights), then 16-row strips |
| FP8-activation prefill, FP4 weights | `CudaLinearOp.ixx:795` | strip rows x input width, plus bucket(rows) x input width, each 16-byte aligned, plus bucket(rows) x 4 |
| Fused decode attention | `CudaGqaOp.ixx:834` | batch x heads x 128 x (head width + 2) x 4 |

The quantize-on-load rows apply only when the stored weights are unquantized and the policy quantizes; the
entry point knows which from the weights header. The FP8-activation row is the one that changes with the
chunk.

**Load staging is not forward scratch, decided 2026-09-14.** The four quantize-on-load rows share the forward
buffer today, so their staging stays allocated after the load. `Quantization.md` *Load Pipeline* moves it
to a buffer the load owns and frees. After that change the rule's total counts forward scratch only -- the
last three rows -- and a load from full-precision weights settles at the same footprint as its exported
form. The staging buffer becomes a load peak on top of the built model, because `build()` allocates before
`loadParameters()`. On Windows the overshoot spills and returns; on Linux it decides whether the load
succeeds (11.10). The prediction reports that peak beside the steady footprint. Whether the rule also
checks it is decided together with the staging size.

`Operation` gains `getRequiredScratchBytes( const BuildContext& )`, defaulting to 0, the peer of
`getRequiredStateMemorySize`. Components and composites combine their children by maximum, not sum, so
no child reports it in `MemoryStats`, which is summed across the tree. The network reserves the execution
context's scratch to that maximum during `build()` and reports it once, at its own level, in
`getMemoryStats()` and `getRequiredMemory()`; the rule's total is then `memory.totalDeviceBytes()` with
nothing added. A request above the reserved size throws, which is how an under-prediction shows. A
component built directly, with no network to reserve for it, keeps today's grow-on-demand buffer.
`getScratchHighWaterBytes()` is deleted (Phase 6 step 2).

Measured 2026-09-13 after a load and a full-chunk prefill: 0.07-0.11 GiB with pre-quantized weights
(Qwen 3.8 cb2-3 and FP4, Gemma 4 12B, Llama 3.1 8B), and 0.25 GiB when the weights are quantized during
the load, which is the 256 MiB staging cap -- the load staging the paragraph above moves out of the steady
footprint.

### 11.5 What the prediction does not count

Measured 2026-09-13 with temporary counters in the CUDA allocator, on both cards, after a load and a
full-chunk prefill. The prediction equals what Mila requests from its allocator to within 23 MiB; the
difference is buffers that bypass the allocator, such as the RoPE tables, which the prediction does count.
What remains between the prediction and the memory in use:

| Term | Measured | Owner |
|---|---|---|
| Driver rounding of each allocation | 0.02-0.73 GiB across the models tested | predicted, 11.8 |
| Scratch | 0.07-0.25 GiB | predicted, 11.4 |
| Fixed remainder, including the 4 MiB cuBLASLt workspace | 0.02-0.07 GiB | not predicted (11.3) |
| Windows budget cut on a card that drives a display | 313-319 MiB on the RTX 4070; none on the headless RTX 5060 Ti. 2026-09-16, Gate B after a load on the RTX 4070: Gemma 4 12B 354-1180 MiB over ten runs, Llama 3.1 8B 321 and 369 MiB; on the RTX 5060 Ti, 6-20 and 21 MiB | not predicted (11.3) |

**The fixed remainder** is the same after the load as after the prefill in every run; it does not grow
with use.

**The Windows budget cut is not memory Mila uses.** Reproduced without Mila on the RTX 4070 that drives the
display: once the process has written about 8.1 GiB, Windows lowers the process's video memory budget by
326 MiB, then returns 12 MiB. CUDA's free memory equals that budget less the process's usage at every
reading, so it falls by the same amount. Memory that is allocated but never written does not trigger it.
Reserving memory up front (`SetVideoMemoryReservation`) does not prevent it, and neither the trigger nor the
size can be read before it happens. It is Windows policy on that machine, so `Mila/Src` neither models it nor
holds memory back for it (11.3). A caller that wants to know whether a card drives a display can
ask NVML, which reported the RTX 4070 as driving one and the RTX 5060 Ti as not. Display mode, asked
first, is deprecated on driver 610.88; display active answers.

**The cut moves with the desktop.** On 2026-09-16 the residual on the RTX 4070 read 720 MiB twice with
about 1450 MiB of the card in use before the run, and 1025 MiB with 1805 MiB in use, the card then
holding Visual Studio, the Claude app, three Edge WebView2 hosts and Docker Desktop. During the 1025 MiB
run NVML showed the process physically take 8914 MiB, CUDA context included, while CUDA's free memory fell
by 9563 MiB: the difference is budget, not allocation. Usage alone does not predict the size, since 224
MiB was read at a similar figure the day before. A measurement of consumption on a display card is a
measurement of that desktop.

None of the measured sizes in this section is a constant in `Mila/Src`. They record the size of what the
prediction leaves out.

### 11.6 Callers

- **Chat** predicts against the free memory the load will see. Under automatic context a model switch
  releases the outgoing model before the scan; today the scan runs first, against device capacity
  (`Chat.ixx:1490`). The chunk a scan reports is then the chunk the load builds, as long as free memory does
  not change in between. The fixed allowances Chat uses today -- 10% of the device total in
  `resolveAutomaticContext` and 12.5% of the prediction in `practicalDeviceBytes` -- are removed, not
  re-tuned (Phase 6 step 4). They stood in for scratch, rounding and the display cut without knowing which
  applied, so a headless card paid for all three. What Chat tells the user on a card that drives a display
  is decided at the step 4 gate. "Held to a full prefill chunk" keeps its meaning: the available memory
  forced a rung below the largest the context permits.
- **MIS, the Python binding and a user's own `main()`** change nothing: the entry points they already call
  read the free memory.

### 11.7 When nothing fits

If the floor rung does not fit the available memory, the transformer builds at the floor, `PrefillChunking`
reports that it does not fit, and the load logs one warning. What happens next depends on the platform: on
Windows, in a process that sees one GPU, the load spills and runs slowly (6.5); on Linux, and on Windows in a
process that sees more than one GPU, the build throws `CudaBadAlloc` if the device really lacks the memory (6.5,
11.10). The library does not refuse ahead of that, and whether to warn, proceed or refuse
is the caller's decision. A prediction never warns.

`PrefillChunking::fits_activation_budget` and `isBudgetConstrained()` are renamed `fits_available_memory`
and `isMemoryConstrained()` (confirmed 2026-09-15), since no activation budget remains.

### 11.8 Driver allocation rounding

**Decided with Todd, 2026-09-15: the footprint predicts rounding.** A device allocation larger than 1 MiB
counts as its size rounded up to a multiple of the device's allocation granularity. An allocation of 1 MiB or
less counts as its size.

- **Documented:** CUDA packs requests of 1 MiB or less into shared blocks and rounds larger ones up to the
  allocation granularity. `cuMemGetAllocationGranularity` returned 2 MiB on both cards.
- **Size:** across the models tested the rounding is 0.02 GiB (Llama 3.1 8B at 512 rows) to 0.73 GiB
  (Qwen 3.8 cb2-3), and it moves with the chunk: Gemma 4 12B is 0.27 GiB at 1024 rows and 0.40 GiB at 64.
- **Not measured:** Linux, and other driver versions.

**Measured 2026-09-15** on the RTX 5060 Ti, which drives no display. Every device allocation site was logged
by temporary code, since removed. The allocations live after a load and a full-chunk prefill were replayed
in their original order in a process with nothing else on the device; each replay ran twice with identical
results.

| Model, chunk | Allocations over 1 MiB | Their rounding, measured | Rounded up to 2 MiB, predicted | Allocations of 1 MiB or less | Their packing overhead | Model process beyond the replay |
|---|---|---|---|---|---|---|
| Qwen 3.8 27B cb2-3, 256 | 876 | 765.6 MiB | 765.6 MiB | 1284 | 12.7 MiB | about 14 MiB |
| Gemma 4 12B FP4, 1024 | 414 | 284.4 MiB | 284.4 MiB | 1212 | 2.1 MiB | about 14 MiB |

The rule for large allocations is exact. The prediction therefore falls short of what a load consumes by the
small-allocation overhead and the fixed remainder (11.5), under 30 MiB on both models and below the 50-70 MiB
run-to-run noise in free memory (6.4). The shortfall is always in the same direction; nothing models it.

**How accurate the rule needs to be.** The prediction only chooses a rung, so it needs to be accurate enough to
choose the right one. Prefill time for a 4096-token prompt at context 8192, measured the same day on the same
card, each rung against the one above it, three runs each with a spread under 0.3%:

| Step | Gemma 4 12B FP4 | Qwen 3.8 27B cb2-3 |
|---|---|---|
| 1024 to 512 | +3% | +8% |
| 512 to 256 | +29% | +15% |
| 256 to 128 | +63% | +37% |
| 128 to 64 | +84% | +55% |

Qwen 3.8 cb2-3 on the RTX 4070 at context 4096 followed the same shape (+9%, +24%, +38%, +63%), but its
1024-row load left no free memory, so its first step may include spill. Unpredicted, rounding exceeds a rung's
memory step on Qwen, so leaving it to the caller either costs rungs worth 15% or more or makes the load spill.
Predicted, what remains is too small to change a rung.

Carried to the Phase 6 step 3 gate, to confirm there: rounding is applied per allocation by one helper beside
`storageBytes`, including the allocations that bypass the allocator (scratch, RoPE tables, the cuBLASLt
workspace); the granularity is read from the device, not written as a constant; the rounded bytes are reported
in the existing `MemoryStats` fields rather than a new one; and Linux takes the same rule (11.10).

### 11.9 What the rule picks on the measured models

Predicted footprint plus the scratch measured for each model, identical on both cards. Free memory was
measured inside the test process: 14.82 GiB on the RTX 5060 Ti, 10.85 GiB on the RTX 4070. "Rule" is the
rung this section picks with that free memory as the available memory.

| Model, context | Card | Today | Rule |
|---|---|---|---|
| Gemma 4 26B-A4B FP4, 8192 | RTX 5060 Ti | 512 rows, 15.14 GiB, does not fit | 256 rows, 14.67 GiB |
| Gemma 4 12B FP4, 131072 | RTX 4070 | 64 rows, 10.12 GiB | 1024 rows, 10.78 GiB |
| Qwen 3.8 27B FP4, 8192 | RTX 5060 Ti | 64 rows, 13.61 GiB | 512 rows, 14.20 GiB |
| Qwen 3.8 27B cb2-3, 8192 | RTX 4070 | 64 rows, 9.57 GiB | 1024 rows, 10.81 GiB |
| Llama 3.1 8B FP4, 16384 | RTX 4070 | 512 rows, 11.35 GiB, does not fit | 256 rows, 9.50 GiB |

Three of the five picks -- the 26B, the 12B and cb2-3 -- leave less than 0.15 GiB of the free memory
unused, and 11.5 measured more than that beyond prediction plus scratch for each: 0.36 GiB for the 26B on
the RTX 5060 Ti, and on the RTX 4070 the rounding and fixed remainder plus about 0.31 GiB of budget cut. With
nothing held back, those three loads would spill. Rounding, the largest of those terms, is predicted (11.8);
the picks in this table predate that decision and do not include it.

The 26B row was measured quantizing on load, so its scratch is 0.25 GiB of load staging. Once staging
belongs to the load (11.4) it is expected to fall to the 0.07-0.11 GiB the exported models measured. That
has not been measured on the 26B, and no exported 26B FP4 weights exist yet.

### 11.10 Linux

Mila runs on Linux as well, and every measurement in this section was taken on Windows. What is known to
differ, and what is not:

- **No budget cut.** The cut in 11.5 is Windows policy for a card that drives a display. The Linux driver
  has no per-process video memory budget.
- **Running out fails instead of spilling.** An allocation past the device's memory throws `CudaBadAlloc`
  during the build; there is no shared-memory fallback. Chat already words its warning for this
  (`kDriverOversubscribesToHostMemory`, `Chat.Footprint.ixx`). On Linux, and on Windows in a process that
  sees more than one GPU, a prediction that falls short of consumption is the difference between a load that
  runs and one that fails, not between fast and slow. With rounding predicted (11.8) the shortfall measured
  on Windows is under 30 MiB, and nothing holds it back (11.3).
- **Free memory is the whole device's.** `cudaMemGetInfo` reports what is free on the device across all
  processes; there is no per-process budget to read.
- **Driver rounding is unmeasured.** Linux takes the rule in 11.8. A driver that rounds finer than the
  granularity it reports makes the prediction high, which costs at most a rung; one that rounds coarser makes
  it low. Only a measurement on the Linux driver tells which.
- **WSL2 does not stand in for it.** Mila's Linux build runs GPU work under WSL2 on the development
  machine, but WSL2 reaches the GPU through the Windows driver, so it would measure the Windows driver
  again. The Linux numbers need a native Linux machine with an NVIDIA GPU, and none is available as of
  2026-09-13.
