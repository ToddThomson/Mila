# Layer Split Across Devices

Specification and implementation plan for running one model's transformer blocks across more than
one CUDA device, so that a model too large for any single card in the machine runs on the cards
together.

Draft, 2026-09-16. Post-v0.20; nothing here is committed to a release.

**This spec is the mechanism; `Deployment.md` is the decision.** Whether a model is split, across which
devices, and where the cut falls are chosen by the deployment planner, as one dimension of the plan
alongside context length and prefill chunk. Nothing here chooses a device on its own.

---

## 1. Problem Statement

Every model Mila runs has to fit on one card. Today the machine holds two separate pools, and a
model that fits neither is refused even when the two together would hold it with room to spare.

That limit decides which models Mila can target at all, so it matters most for the models Mila
wants next:

| Model | Weights at the build Mila would ship | One 16 GiB card | Both cards (~26 GiB) |
|---|---|---|---|
| Gemma 4 26B-A4B, experts FP4, rest BF16 | 15.76 GiB (`MixtureOfExperts.md` §8) | does not fit | fits, with context |
| Gemma 4 26B-A4B, `PerGroupFp4<64>` throughout | 13.54 GiB | fits at context 8192 only at chunk 256, 12 MiB spare | fits, at long context |
| Muse Glimmer 30B (dense, text and vision), FP4 | ~16 GiB before KV | does not fit | fits |
| Qwen 3.8 27B, `PerGroupFp4` | 12.71 GiB resident | fits at context 8192 | fits, at long context |

The last row is the proof vehicle, not the reason (section 10).

**What it buys, derived from the footprint tables and not measured:** about 26 GiB usable across
the dev machine's two cards. At the 26B's FP4 rate (13.54 GiB / 25.23B = 0.537 GiB per billion
parameters) that holds a model of about **45B total parameters** with room left for KV cache; at
the 2.82-bit codebook rate (about 0.41 GiB per billion), about **55-60B**. A split does not reach
the 100B+ MoE class, which needs expert weights in host memory; 32 GB of host RAM rules that out on
this machine.

**What it does not buy is decode speed.** At batch 1 exactly one stage computes a decode step at a
time, so decode time is the sum of each block's time on the card it runs on, plus the crossings.
Prefill could overlap the stages; whether that pays is open (3.6).

---

## 2. What Decides The Design

### 2.1 The machine

Measured 2026-08-24 and 2026-09-12 (`nvidia-smi`, and `MixtureOfExperts.md` §8.1 for the links):

| | RTX 5060 Ti | RTX 4070 |
|---|---|---|
| Architecture | Blackwell, sm_120 | Ada, sm_89 |
| Memory | 16,311 MiB | 12,282 MiB, drives the display |
| Link | CPU-attached, PCIe Gen5 x8; measured ~28 GB/s to host, ~24 GB/s from host | chipset-attached, Gen4 x4, shares DMI with D:; measured ~6.5 GB/s both ways |
| Memory bandwidth | ~448 GB/s | ~504 GB/s |

There is no NVLink, and the driver grants no peer access between the two cards
(`cudaDeviceCanAccessPeer` is 0 both ways, Windows and WSL, section 11 Phase 0). **Every byte that
moves between the cards passes through host memory and the 4070's x4 link.**

With two GPUs visible, `cudaMalloc` fails at a card's physical capacity instead of spilling to host
memory (measured 2026-09-14, `MemoryFootprint.md` 6.5). A split therefore sees hard limits on both
cards, which is what makes a placement prediction checkable.

### 2.2 Where to cut

What matters over this link is how often data crosses, not how much. Per decode token, for a
model with `L` layers and width `d`:

| Cut | Crossings per decode token | Size of each |
|---|---|---|
| **Between blocks** (this spec) | 1 per boundary | `d` x 2 B at BF16: 5.5 KiB at 2816, 7.5 KiB at 3840, 13 KiB at 6656 |
| Attention on one card, experts on the other (expert parallelism) | 2 per MoE layer: 60 on the 26B | the routed tokens' hidden states |
| Within each weight (tensor parallelism) | 2 all-reduces per layer | activation-sized |

Measured (Phase 0): a decode crossing at width 3840 costs 35-72 µs and a 1024-row prefill chunk
crosses in about 1.3 ms, both including the destination's synchronize. Only the first cut keeps the number of crossings independent of the layer count, and at batch 1
nothing is gained by the others: expert parallelism pays off only with batches large enough to keep
every expert busy.

### 2.3 The fact the design rests on

**Between blocks, a transformer passes exactly one tensor: the hidden state.** Every family's loop is
embedding, blocks, final norm, head:

- Gemma: `Gemma.ixx:241` (prefill), `:270` (decode)
- Qwen: `Qwen.ixx:245`, `:326` (`scoreTokens`), `:370`
- Llama: `Llama.ixx:175`, `:210`

Everything else a block needs lives inside it and moves with it: its KV cache and sliding-window
ring, the DeltaNet recurrent state and convolution window (`Qwen.DeltaNetBlock.ixx`), Gemma's
`layer_scalar` (a host float, `Gemma.Block.ixx:918`), and the MoE expert bank. The block interface
already states the contract: `ITransformerBlock::prefill( input, position_offset )` and
`decode( input, position )` return a block-owned `[B, T, model_dim]` tensor
(`ITransformerBlock.ixx:53-59`).

---

## 3. Design

### 3.1 Terms

One concept, one name, everywhere below and in the code:

- **Stage** — a contiguous run of blocks bound to one execution context on one device.
- **Boundary** — the copy of the hidden state from the last block of one stage to the first of the next.
- **Placement** — the mapping of blocks to stages, and of the embedding and head to the first and last
  stage. Carried by `DevicePlacement` (section 6).
- **`StagedTransformerBlock`** — the one object that walks the stages. It implements
  `ITransformerBlock`, so a transformer holds one and calls it as it calls any block (section 7,
  Phase 1).

### 3.2 The mechanism

The stages are an ordered list inside the transformer's `StagedTransformerBlock`, which walks blocks
only. The embedding, final norm and head stay in the family's own code, bound to stage 0's context and
the last stage's context respectively (section 4). Prefill and decode run:

```
hidden = embedding( tokens )                      // family code, stage 0's context
hidden = staged_block.prefill( hidden, offset )   // StagedTransformerBlock:
    for each stage s:
        bind s's device                           //   section 3.4
        if s > 0: hidden = boundary( hidden, s )  //   section 3.3
        for each block in s: hidden = block( hidden, offset )
logits = head( norm( hidden ) )                   // family code, last stage's context
```

With one stage this is the loop that exists today, with no boundary and no bind beyond what already
happens. The split stays explicit, visible and in one place: there is no scheduler and no hidden
execution engine.

A stage's blocks, workspaces and scratch reservation all belong to that stage's context:

- **Shared workspaces are per stage.** Each transformer owns workspaces every block writes into
  (GQA: `Gemma.ixx:789`, `Qwen.ixx:772`, `Llama.ixx:724`; block: `Gemma.ixx:784`, `Qwen.ixx:764`;
  DeltaNet: `Qwen.ixx:768`). A stage allocates its own set, sized from its own blocks.
- **Scratch reservation is per context.** `MemoryStats` combines scratch by maximum
  (`MemoryFootprint.md` Phase 6 step 2), which is correct within a stage and wrong across two, so
  the maximum is taken per stage.
- **Load staging is per context.** `releaseLoadStaging()` is called on every stage's context after
  the load.

### 3.3 The boundary

A boundary copies `[B, T, model_dim]` from one stage's buffer into a destination buffer the next
stage owns, allocated once at build for the prefill chunk and sliced for decode.

Two candidates were measured in Phase 0:

1. **Direct device-to-device `cudaMemcpy`**, which is what `copy()` (`TensorOps.Transfer.ixx:87`)
   issues for two device tensors. Without peer access the CUDA runtime stages it through host
   memory itself.
2. **Host-staged through a pinned buffer**: device-to-host on the source stage's stream, a sync of
   that stream, then host-to-device on the destination stage's stream.

The destination stage must not start before the copy has landed. Phase 0 showed candidate 1 gives
that: a read queued on the destination's own stream immediately after the call, with no synchronize
between, saw the new bytes every time.

**Recommended: candidate 1**, for a sequential stage walk and for concurrent prefill alike. It works
on Windows and WSL, costs the same as candidate 2 at decode and 15-20% less at prefill (1.3 against
1.5-1.6 ms per 1024-row chunk), and needs no pinned buffer, so the boundary owns only its
destination device buffer.

Phase 0 established its ordering on both sides. It waits for work still queued on the source stage's
stream, so it reads a chunk's finished output. It also delivers that output even when the next
chunk's work is queued on the source stream immediately after the call returns, including on
Windows, where the call returns before the copy lands. That second property is what concurrent prefill
(3.6) needs: stage 0 can be given chunk `k+1` straight after chunk `k`'s crossing is issued.

### 3.4 Device binding

**Kernel launches do not select a device.** They use their context's stream and rely on the calling
thread's current device. Today one context per process makes that true by construction:
`CudaExecutionContext::initializeResources` binds once (`CudaExecutionContext.ixx:474`).

With two contexts on one thread, the current device is whatever last changed it, and several sites
change it without restoring:

- `CudaDeviceMemoryResource::do_allocate` binds its device and leaves it bound
  (`CudaDeviceMemoryResource.ixx:88`), so building stage 1 leaves stage 1's device current.
- `RopeCacheRegistry::acquire` allocates with `cudaMalloc` and no bind (`CudaRopeOp.Cache.ixx:115`),
  so a table keyed to one device can land on another. The key already carries `device_id`
  (`:43`); the allocation does not use it.
- `CudaExecutionContext`'s scratch, staging and reservation buffers allocate without binding
  (`Mila/Issues/Untriaged.md`, "CudaExecutionContext allocates its buffers without selecting its
  device").

**The rule: a stage binds its device on entry, in prefill, decode, scoring and build alike, and every
allocation site that does not own its device binds explicitly.** The first half makes every launch
inside a stage correct regardless of what ran before it. The second closes the allocation sites
above, which run during build and load, outside any stage walk.

This is the defect class behind both multi-GPU failures on record: a stream created on the wrong
device (fixed 2026-08-25) and the buffers above. Both were invisible with one GPU visible. The gates
in section 10 therefore run with both GPUs visible, and one negative removes the bind.

### 3.5 Tied tables

Gemma (all sizes), Llama 3.2 3B and GPT-2 tie the embedding table to the head
(`Gemma.ixx:591-606`). With a split the embedding runs on stage 0 and the head on the last stage, so
one table is needed in two places. Two options:

1. **A copy on each end.** Costs one table on stage 0: ~0.69 GiB of FP8 on the 26B build
   (`MixtureOfExperts.md` §8). No new code path.
2. **A host-resident gather for the embedding**, as Qwen does for its untied table
   (`EmbeddingTableResidency::Host`, `Qwen.ixx:150`), keeping the device copy with the head. Costs a
   host copy of the table and a ~10 KiB per-token gather over the first stage's link.

Undecided (section 12). Qwen has untied tables and needs neither, which is one reason it goes first.

### 3.6 Alternatives rejected

**Expert parallelism** (the expert bank on one card, everything else on the other). Sixty host-staged
round trips per decode token on the 26B, each with a sync, against one for a block split. Pays off only
with large batches. `FfnAndMoE.md` §13 keeps it out of scope and this spec does not change that.

**Tensor parallelism.** Two host-staged all-reduces per layer over a chipset x4 link. Not viable
without peer access.

**Concurrent decode.** At batch 1 there is one token in flight, and the next cannot start until it is
sampled, so there is nothing for a second stage to overlap. It becomes a question only if MIS ever
serves concurrent requests.

**Concurrent prefill is not rejected; it is open (section 12).** Chunked prefill is a sequence of
micro-batches: stage 0 can start chunk `k+1` while stage 1 processes chunk `k`, because stage 0's
work on `k+1` depends only on stage 0's own KV and recurrent state, which already hold chunks up to
`k`. What it would gain is unmeasured. It depends on whether one host thread can issue both stages'
launches faster than the two cards consume them, and launch overhead is a large share of short-prompt
prefill today. A search of the Gemma and Qwen block paths and their operations found no host
synchronize or device-to-host copy during prefill, so no stage waits on the host partway through a
chunk.

**A split below the transformer, inside `Linear` or the operations.** Moves device placement into
components that are single-device by contract, and multiplies crossings by the number of
operations. Rejected for the same reason expert parallelism is.

---

## 4. Binding A Second Context

A transformer today owns one execution context, created from the `DeviceId` its constructor takes
(`Gemma.ixx:164`), and binds it after building its graph (`:178`). Two rules in `CompositeComponent`
decide how a context reaches the blocks:

- `addComponent` refuses a child that **already** has a context (`CompositeComponent.ixx:145-153`).
- `onExecutionContextSet` gives the parent's context only to children **without** one (`:826-834`).

So the constructor binds the later stages between building the graph and binding its own context:

```cpp
explicit GemmaTransformer( const std::string& name, const GemmaConfig& config, DevicePlacement placement )

createGraph();                                          // addComponent: no child has a context yet

for ( each block in stage k > 0 )                       // and the final norm and head, on the last stage
    block->setExecutionContext( contexts_[ k ].get() );

this->setExecutionContext( contexts_[ 0 ].get() );      // reaches only the children still unbound
```

- **The blocks stay direct children of the transformer.** No stage level enters the component tree, so
  the save and load traversals, which push a scope per child (`Gemma.Block.ixx:535-545`), see the same
  paths, and no published tensor is renamed. V2 checks the vocabulary regardless.
- **`CompositeComponent` gains no mechanism.** Both rules already exist.

**The contract has to change with it.** The comment at `addComponent` says children "must share the
parent's ExecutionContext"; the skip in `onExecutionContextSet` is what permits otherwise, and nothing
designed it for this. Phase 2 rewrites the contract — a child bound before its parent keeps its own
context, and it must have the parent's device type, which `setExecutionContext` already checks
(`Component.ixx:921`) — and adds a test that pins it, so changing the skip fails a test instead of a
split.

**Rejected shapes.** A `Stage` composite owning its context adds a tree level and, through the per-child
scope, very likely renames every block tensor. Holding stages outside the tree takes the blocks out of
what save, load, footprint and observation walk. Relaxing `addComponent` removes the check for every
composite to allow it for one.

**What the single context still does.** Each transformer uses it in 10-13 places (Gemma 10, Qwen 13,
Llama 13). Per stage in Phase 2: workspace allocation (`Gemma.ixx:968`, `:982`), `reserveScratch`
(`:739`), `releaseLoadStaging` (`:589`) and the synchronizes during load (`:567`, `:584`). Stage 0 only:
construction and the network's `getDeviceId()`. Free-memory reads leave the transformer altogether
(`Deployment.md` section 4).

---

## 5. Placement

Placement is the planner's device dimension (`Deployment.md`, Phase 5). It is decided before
construction, from prices that read no bound context and from one free-memory reading per device
(`Deployment.md` sections 4 and 3.2). What this spec fixes is what the planner must respect.

**Devices.** The request names an ordered device list or `auto`. The planner's objective adds devices
only when no single device holds an acceptable deployment (`Deployment.md` section 5, rule 2). With an
explicit list, the list's order is the layer order: the first device takes block 0.

**The prefill chunk is shared.** A chunk's rows flow through every stage, so one chunk size applies to
all of them. Among devices, chunk rungs and split points the planner searches:

```
for chunk in rungs, largest first:
    for each split point assignment:
        every stage: requiredMemory( stage, chunk ).totalDeviceBytes() <= reading( stage.device ).budget
    if any assignment fits: pick the one with the largest minimum headroom fraction; stop
```

Both loops are small (at most eight rungs; for two devices, at most one split point per layer), and
every evaluation is closed-form. Each stage is priced against its own device's reading, including the
display card's share.

**Per-stage terms.** Stage 0 carries the embedding (device-side unless host-resident); the last stage
carries the final norm, the head and, if tied, its table copy (3.5); every stage carries its blocks,
its workspaces, its scratch and its boundary buffer. A single-device placement is the one-stage case of
the same computation and must reproduce today's numbers exactly.

**`MemoryStats` becomes per device.** A network with more than one stage reports one `MemoryStats`
per device, and the total. The plan carries the same structure (`Deployment.md` 3.3), so an adaptor
can say which card is short.

---

## 6. Model Layer

The model layer (`GemmaModel`, `QwenModel`, `LlamaModel` over `LanguageModel`) assumes one device in
three places a split breaks, and several it does not.

### 6.1 Two types

- **Public: a device list, or `auto`,** in the `DeploymentRequest` (`Deployment.md` 3.3). A user names
  cards and never counts blocks. A single `DeviceId` remains the one-card case.
- **Internal: `DevicePlacement`,** in `Detail` — the ordered stages, each a `DeviceId` and a block
  count. The plan holds it; the transformer's constructor takes it (section 4); a test constructs one
  directly to force a split point. A `DeviceId` converts implicitly to one stage holding every block.

### 6.2 What a split breaks

1. **Decode-ahead assumes one device.** The generation loop (`GemmaModel.ixx:394-468`) samples on the
   network's context (`LanguageModel.ixx:396`), writes the token into `decode_token_device_`, which is
   allocated on `getDeviceId()` (`GemmaModel.ixx:503`), and queues the next forward before the host has
   read the token back. With a split the logits are on the output device and the next forward starts
   on the input device. The sampler binds the **output** stage's context, and the sampled token (4 B)
   crosses back to the input device before the next decode. Phase 0 shows the crossing is ordered; what
   it costs decode-ahead is unmeasured (Phase 3 exit).
2. **`synchronize()` drains one stage.** `Network::synchronize()` syncs one context, and the model calls
   it to drain in-flight work on cancellation and on a stop token (`GemmaModel.ixx:419`, `:448`). With a
   split it must drain every stage, or the output device may still be computing when `generate` returns.
3. **The footprint is one number.** `getDeploymentFootprint` returns one `MemoryStats`. The plan's per-device
   footprints replace it for any caller that decides (`Deployment.md` 3.3).

### 6.3 What it does not break

- **Prompt upload** already targets the input device (`GemmaModel.ixx:673`).
- **`observe()`** hands each tensor ordered on its publishing component's stream, which is per stage by
  construction.
- **Loading and saving** go component by component, and each component writes through its own context.
- **The weight format** selects the transformer type before construction (`QuantizationDispatch.ixx`)
  and does not depend on placement.
- **The build** must carry both architectures, since both cards run one compiled library. The default
  list does (`MILA_LIBRARY_CUDA_ARCHITECTURES`, `Mila/CMakeLists.txt`); the single-architecture presets
  `x64-release-ada` and `x64-release-blackwell` cannot run a split on this machine.

### 6.4 Input and output device

A model has an **input device** (stage 0: token ids, embedding) and an **output device** (the last stage:
logits, sampling). `getDeviceId()` keeps meaning the input device. Every current caller of it in
`Mila/Src/Dnn/Models` and `Core/LanguageModel.ixx` is audited for which of the two it means.

---

## 7. Families

| | Blocks | Tables | Notes |
|---|---|---|---|
| **Qwen** | `ITransformerBlock`, heterogeneous (attention, DeltaNet) | untied; embedding host-resident on CUDA | Has `scoreTokens`, the only on-device teacher-forced scoring path. First. |
| **Gemma** | `ITransformerBlock`, heterogeneous (sliding, global; dense or MoE) | tied (3.5) | Needs `scoreTokens` ported before its gate can run. |
| **Llama** | concrete `std::vector<std::shared_ptr<Block>>` | 3.2 3B tied, 3.1 8B untied | Also has a training `backward` (`Llama.ixx:227`), which the split does not cover. |
| **GPT-2** | concrete vector | tied | **Out of scope.** Small enough that a split has no use; it is the training reference. |

**The block walk exists in seven places across three families, plus six more block loops.** Prefill,
decode and scoring walk the blocks (`Gemma.ixx:241`, `:270`; `Qwen.ixx:245`, `:326`, `:370`;
`Llama.ixx:175`, `:210`), and every family also loops its blocks for `setState` (`Gemma.ixx:988`,
`Qwen.ixx:1057`, `Llama.ixx:653`), `resetKvCache` and `rewindKvCache`. Each becomes per stage. Phase 1
therefore puts all of them behind one `StagedTransformerBlock` before any stage exists.

---

## 8. User Surface

Every surface sends the same `DeploymentRequest`; none decides anything itself (`Deployment.md`
section 8). Adaptor work is outside the feature freeze by definition.

- **C++.** `planDeployment( path, request )` with `devices` set to a list or `auto`, then
  `fromPretrained( path, plan )`. A single `DeviceId` still works. A request that cannot be met anywhere
  yields a plan that says so, with the per-device shortfall.
- **Chat.** `--device 1,0` or `"device": [1, 0]`, and `"device": "auto"`. A single number still means one
  card, as today. `ChatConfig::device_index` (`Chat.Config.ixx:217`) becomes the request's device field,
  and it already reaches the load, the pre-flight and the fit column, which is where the plan now comes
  from. `context_length: "auto"` is resolved in the same plan. `/models` renders the plan's binding
  constraint, naming the short card; the stats line names every card in use.
- **Python binding.** `from_store( name, context_length="auto", devices="auto" )`, accepting an int or a
  list. `devices` replaces `device_index` (`Mila_py.cpp:382` and siblings): one concept, one name. The
  QuickStart Python samples and `getting-started.md` change in the same commit.
- **MIS.** `MILA_DEVICES` (a list or `auto`) replaces `MILA_DEVICE_INDEX` (`config.py:43`,
  `model_worker.py:163`, the README table).
- **Ordinals.** Lists hold CUDA ordinals, which differ from nvidia-smi's order on a mixed-generation
  machine, this one included (`Chat.Config.ixx:213`). User-facing text says which is meant.

---

## 9. Traps

- **Cross-architecture results never agree token-for-token.** Greedy generation forks between Ada and
  Blackwell at BF16, FP8 and FP4 alike, at a token index set by the prompt (measured 2026-08-24,
  `Qwen3.8.md` §8). Any split on this machine puts blocks on both architectures, so no gate may
  compare a cross-card split token-for-token against a single card. Section 10 uses teacher-forced
  scoring instead.
- **A single-GPU run proves nothing about binding.** Both multi-GPU defects on record passed pinned to
  one card. Every stage gate runs with both GPUs visible, and each result states which GPUs were
  visible.
- **The display card's free memory moves.** `readFreeDeviceBytes` on the 4070 changes between the
  prediction and the build (Gate B residual 224 MiB on the 4070 against 6 MiB on the 5060 Ti,
  2026-09-15). A placement that is exact on the headless card may miss on the display card. The plan
  records one reading per device and the load builds from the plan (`Deployment.md` 3.2), so the
  prediction and the build cannot read different values.
- **Chipset contention during load.** The 4070's link shares DMI with D:, where the weights live, so
  loading stage 1 competes with reading the file. A load-time cost only; record it, do not optimise it.
- **The boundary buffer is a new allocation.** It is counted in the stage's footprint, or Gate A
  misses by exactly its size.
- **On Windows the copy between cards returns before it finishes.** A 7.5 MiB direct copy returns in
  a median 112-118 µs, and every so often a call blocks for 15-16 ms while the runtime catches up;
  under WSL the same call blocks for the whole 1.3 ms. Ordering is unaffected, but any host-side
  timing of a single stage on Windows charges crossing time to whichever later call happened to
  block. A per-stage timing synchronizes the destination first.

---

## 10. Validation

All gates run in the maintainer's environment: `x64-profile`, both GPUs visible, CUDA device 0 = the
RTX 4070. Each gate's bound is written into this spec before its
first run, and each is forced to fail once.

**V1. Same-device split is bit-identical.** Two stages with two contexts **on one card**, split at
block `k`, against the unsplit model: prefill logits, a run of decode logits and `scoreTokens` are
bit-identical, for `k` in {1, N/2, N-1}. This isolates the stage machinery from the architecture,
because nothing crosses a card. It is the primary correctness gate.

**V2. The vocabulary and the footprint are unchanged.** Same configuration as V1: the flat parameter
vocabulary is byte-identical to the unsplit one (section 4), and the sum of the per-stage predictions
equals the unsplit prediction plus exactly the boundary buffer.

**V3. Architecture baseline, measured first.** A model that fits both cards alone: Qwen 3.8 27B
cb2-3 (11.05 GiB). Mean per-token negative log-likelihood from `scoreTokens` over the scoring corpus,
4070 alone against 5060 Ti alone. Their difference, delta, is the size of the architecture effect.
Recorded here before V4 is written.

*Measured 2026-09-16* with `QwenPackedArtifactTests.DISABLED_CorpusPerplexity` (x64-claude-verify,
Release, built from `c6f53c7c`), each run pinned by UUID. Protocol: wikitext-2 test, non-overlapping
1024-token segments, teacher-forced, head width 64, 16,240 scored positions.

| Card | Mean NLL, nats/token | Perplexity |
|---|---|---|
| RTX 4070 | 2.0169 | 7.515 |
| RTX 5060 Ti | 2.0167 | 7.514 |
| RTX 5060 Ti, repeated | 2.0167 | 7.514 |

**delta = 0.0002 nats/token**, at the limit of the four decimals the test prints. Two limits on using it
for V4, both to settle before V4's bound is written:

- **Resolution.** The harness Phase 3 builds for V4 prints the summed log-probability at full
  precision and re-measures delta the same way, so the bound is not set from a rounded number.
- **Different kernels.** cb2-3 runs codebook GEMV; the FP4 model V4 splits runs FP4 kernels, whose
  cross-architecture difference this does not measure. No FP4 Qwen fits the 4070, and no other family
  has `scoreTokens` yet.

**V4. Cross-card split is within the baseline.** Qwen 3.8 27B FP4 split across both cards, against
the same model on the 5060 Ti alone: the difference in mean NLL is no larger than delta. A split
puts only part of the model on Ada, so it should land inside the all-Ada difference. The bound is
fixed from V3's number before the run.

**V5. Placement is exact per device.** `Deployment.md` G1 with a split: the plan's footprint equals
the built network's for every stage on its own device, at every chunk rung and split point the planner
can pick. Gate B's residual holds per device, with the display card's bound stated separately.

**V6. Decode-ahead survives the split.** Decode rate for Qwen 3.8 27B FP4 split across both cards,
decode-ahead on against decode-ahead forced off, recorded with the single-card rate on the 5060 Ti. A
measurement, not a pass/fail: it states what the token's crossing back to the input device costs
(section 6.2).

**Negatives, each of which must fail its gate:**

- N1. Remove the bind at stage entry: fails with both GPUs visible (illegal access or V4), passes pinned
  to one card. That asymmetry is the evidence the gate sees the defect.
- N2. Boundary copy one row off: V1 fails.
- N3. Stage 1's workspace sized from stage 0's blocks: V5 fails.
- N4. The planner prices every stage against stage 0's reading: V5 fails on the 4070.
- N5. `synchronize()` drains stage 0 only: a stop token on the output device leaves work in flight
  after `generate` returns, detected by a query of the output stage's stream at return.

---

## 11. Phasing

Each phase is independently buildable and leaves the single-device path unchanged.

**Phase 0 — measure the crossing.** Scratchpad probe only, no `Mila/Src` change. On this machine,
both GPUs visible, Windows and WSL: whether `cudaDeviceCanAccessPeer` reports access (expected: no);
whether a direct device-to-device `cudaMemcpy` between the two cards succeeds and what it costs;
the cost of the pinned host-staged copy (3.3, candidate 2); each at a decode crossing (7.5 KiB) and a
prefill crossing (7.5 MiB), a few thousand repetitions, median and 99th percentile. *Exit:* the numbers
recorded here and the boundary mechanism chosen. **Measured 2026-09-16** — see the Phase 0 result
below; candidate 1 recommended.

**Phase 1 — one staged block, one stage.** No behaviour change.

- **`StagedTransformerBlock`**, a new module `Dnn.Components.StagedTransformerBlock` beside
  `ITransformerBlock.ixx`, in namespace `Detail`. It implements `ITransformerBlock` — `prefill`,
  `decode`, `setState`, `supportsKvCache`, `resetKvCache`, `rewindKvCache` — over an ordered list of
  non-owning `ITransformerBlock*`. It is not a `Component` and never joins the component tree, so no
  tensor name and no footprint can change.
- **Gemma and Qwen** replace `std::vector<TransformerBlockType*> blocks_` (`Gemma.ixx:771`,
  `Qwen.ixx:760`) with one staged block, and route prefill, decode, scoring, `setState`, reset and
  rewind through it.
- **Llama**: `LlamaBlock` also derives from `ITransformerBlock`, as `GemmaBlock` does. Its `prefill`,
  `decode`, `setState`, `supportsKvCache` and `resetKvCache` already match and gain `override`; it gains a
  `rewindKvCache` delegating to its attention. `LlamaTransformer` does not override `rewindKvCache`, so
  prefix reuse stays off for Llama. The transformer keeps its `shared_ptr` vector for `forward`,
  `backward` and `zeroGradients` and uses the staged block for inference. Llama's `decode` stops
  overwriting `block_input_ptrs_` and `block_output_ptrs_` (`Llama.ixx:208-217`), which `backward` reads;
  that changes behaviour only for a `backward` after a `decode`, which is already wrong.
- **Export.** Whether the module must be re-exported from `Mila.ixx` is decided by building, not argued:
  first build every consumer (`mila-chat`, the QuickStart C++ sample, the Python binding) with it
  imported only by the families. MSVC requires a type used as a member of a public template to be visible
  in the consumer (`Mila.ixx` header), so re-export is expected; `Detail` marks it as not API either way.

*Gate, written before any code:* a temporary test hashes prefill logits and 16 decode steps for Gemma 4
12B FP4, Llama 3.2 3B and Llama 3.1 8B, and the summed log-probability of Qwen cb2-3's `scoreTokens`. It
is built and run on the unchanged tree, then after the change; every hash is identical, and the test is
deleted. Negative: a staged block that skips its last block changes every hash. Then the full suite in
`x64-profile` with both GPUs visible, `ChatRichTextTests`, piped Chat sessions for the three families,
and the WSL CPU-only build.

**Phase 2 — stages on one device.** `DevicePlacement` (section 6.1) on each transformer's constructor;
the binding order and the rewritten `CompositeComponent` contract with its test (section 4); per-stage
workspaces, scratch and load staging; the bind at stage entry. *Exit:* V1 and V2 on Qwen 3.8 27B FP4 on
the 5060 Ti.

**Phase 3 — the boundary, the binding sites and the model layer.** Phase 0's mechanism; explicit binds at
the three allocation sites in 3.4; the sampler on the output stage, the token's crossing back, and
`synchronize()` over every stage (section 6.2). *Exit:* V3, then V4, V6 and N1, N2 and N5, both GPUs
visible, with the split point forced by a test-constructed `DevicePlacement`.

**Phase 4 — placement.** Waits for `Deployment.md` Phases 1-3: pricing without a bound context, the build
executing the plan's chunk, and `planDeployment` on one device. Placement then lands as the planner's
device dimension (`Deployment.md` Phase 5): per-device footprints in the plan, and the joint search over
devices, chunk rungs and split points. *Exit:* V5, N3 and N4 on Qwen 3.8 27B FP4 across both cards, at a
context length the 5060 Ti alone cannot hold.

**Phase 5 — user surface.** Section 8 through Chat, the Python binding and MIS, including `"auto"` devices.
*Exit:* a Chat session on the split Qwen answers coherently with `"device": "auto"`; `/models` names the
short card for a model that fits neither card alone nor both; a QuickStart Python session and an MIS
request each run split.

**Phase 6 — Gemma.** `scoreTokens` ported; tied tables decided and built (3.5). *Exit:* V1, V2, V4
and V5 on Gemma 4 12B; then Gemma 4 26B-A4B with BF16 non-expert weights across both cards at a
context the single-card build cannot reach, with V4 against the FP4 build on the 5060 Ti alone.

**Phase 7 — Llama.** *Exit:* V1, V2 and V5 on Llama 3.1 8B. V4 waits for a Llama `scoreTokens`.

### Phase 0 result (2026-09-16)

**Environment.** Both GPUs visible (`CUDA_VISIBLE_DEVICES` unset): CUDA 0 = RTX 4070, driving the
display; CUDA 1 = RTX 5060 Ti, headless. Driver 610.88. Windows: MSVC, CUDA 13.4. WSL (Ubuntu-Dev):
g++, CUDA 13.3. The probe uses the CUDA runtime only, with no Mila code, and was run once per platform.
Both cards otherwise idle.

**Peer access:** `cudaDeviceCanAccessPeer` = 0 in both directions, on both platforms.

**Every mechanism works without peer access**, and every copy verified byte-for-byte: direct
`cudaMemcpy` with `DeviceToDevice` or `Default`, `cudaMemcpyPeer`, staged through pinned memory
(synchronous, or on streams), and staged through pageable memory.

**Ordering.** Each iteration writes new bytes to the source, copies, then immediately queues a read
of the destination on the destination's own stream, with no synchronize between. Stale reads per
direction, identical on both platforms:

| Mechanism | 7.5 KiB (x2000) | 7.5 MiB (x300) | 64 MiB (x30) |
|---|---|---|---|
| direct `cudaMemcpy` DeviceToDevice | 0 | 0 | 0 |
| `cudaMemcpyPeer` | 0 | 0 | 0 |
| staged pinned, streams | 0 | 0 | 0 |
| Negative: copy skipped on odd iterations | 1000 | 150 | 15 |

The negative landing at exactly half shows the check detects a stale read.

**Cost of one crossing, including the destination's synchronize** (median; the range covers both
directions):

| | Windows, direct | Windows, staged pinned | WSL, direct | WSL, staged pinned |
|---|---|---|---|---|
| Decode, 7,680 B | 35-48 µs | 29-54 µs | 53-72 µs | 87-99 µs |
| Prefill chunk, 7,864,320 B | 1.29-1.32 ms | 1.51-1.57 ms | 1.31-1.32 ms | 1.54-1.61 ms |
| 64 MiB | 10.6-10.7 ms (6.3 GB/s) | 12.5-13.0 ms (5.2-5.4 GB/s) | 10.6-10.7 ms | 12.6-13.1 ms |

The staged path costs the sum of its two halves, measured separately below (for example
1204 + 334 µs against 1570 µs measured). Direct costs about the slower half alone, which suggests the
runtime overlaps the two halves; that is inferred from the numbers, not observed.

**The link, one direction at a time** (pinned, 64 MiB, both platforms within 3%): 4070 to host
6.55 GB/s, host to 4070 6.61 GB/s; 5060 Ti to host 27.9-28.5 GB/s, host to 5060 Ti 23.6-24.0 GB/s.
Every crossing is bounded by the 4070's link. A synchronize with nothing queued, and a
`cudaSetDevice` pair, each cost 0.2 µs.

**Against the work around it (derived, not measured together):** at ~40 tokens/s a token takes
~25 ms, so a 35-72 µs crossing is 0.1-0.3% of it. A 1024-row prefill chunk that takes several hundred
milliseconds pays 1.3 ms. The crossing does not decide whether a split is worth it.

**Windows returns early** (section 9): the direct copy's median without the synchronize is 112-118 µs
at 7.5 MiB with a 99th percentile of 15-16 ms; under WSL it is 1.28 ms, blocking.

**Source-side ordering.** Each iteration puts the previous bytes on both cards, queues a write of new
bytes to the source on the source's own stream (standing in for a stage's last kernel), and starts
the crossing at once. `cudaStreamQuery` just before the crossing confirms the write had not finished.
Identical on Windows and WSL, both directions:

| Mechanism | Size | Write pending at start | Crossings that read the old bytes |
|---|---|---|---|
| direct `cudaMemcpy` DeviceToDevice | 7.5 KiB / 7.5 MiB / 64 MiB | 1997-2000 of 2000 / 300 of 300 / 30 of 30 | 0 / 0 / 0 |
| `cudaMemcpyPeer` | same | same | 0 / 0 / 0 |
| staged pinned, D2H on source stream | same | same | 0 / 0 / 0 |
| Negative: source write skipped on odd iterations | same | half | 1000 / 150 / 15 |

**The next chunk queued straight after the crossing.** As above, but a second write (standing in for
chunk `k+1`) is queued on the source stream immediately after the crossing call returns. Which bytes
reach the destination, identical on Windows and WSL, both directions:

| Mechanism | 7.5 KiB (x2000) | 7.5 MiB (x300) | 64 MiB (x30) |
|---|---|---|---|
| direct `cudaMemcpy` DeviceToDevice | chunk `k` every time | chunk `k` every time | chunk `k` every time |
| `cudaMemcpyPeer` | chunk `k` every time | chunk `k` every time | chunk `k` every time |
| staged pinned, D2H on source stream | chunk `k` every time | chunk `k` every time | chunk `k` every time |
| Negative: chunk `k+1` queued before the crossing | chunk `k+1` every time | chunk `k+1` every time | chunk `k+1` every time |

No crossing delivered a mix of the two. The stand-in for a kernel is a host-to-device copy queued on the
same stream; a kernel launch is queued the same way, but the probe launches no kernel.

---

## 12. Open Decisions

**Decided 2026-09-16:** a split has an `auto` (section 8), chosen by the deployment planner
(`Deployment.md`); a named single device never splits. The stages bind by construction order, not
through a `Stage` composite (section 4).

1. **Tied tables** (3.5): a copy on each end, or a host-resident embedding gather.
2. **Device order under `auto`.** With an explicit list the list's order is the layer order. With `auto`
   the planner orders the devices, and the rule is open: by free memory, by CUDA ordinal, or by which
   order leaves the largest minimum headroom.
3. **Chat's default device.** Stays `0`, as today, or becomes `"auto"`. Recommended: `0` until `auto` has
   run through Phase 5's exit, then decide.
4. **Concurrent prefill** (3.6): whether stages overlap during chunked prefill. Decided by
   measurement after Phase 3, against the sequential split on the same model. It must be
   bit-identical to the sequential split, since each card runs the same kernels in the same order.
   The boundary mechanism does not constrain it (3.3).
5. **`/device` in Chat.** Whether a session can move cards without restarting (`Mila/Issues/Future.md`,
   "A session cannot move cards without restarting").

---

## 13. Non-Goals

- Expert parallelism and tensor parallelism (3.6).
- Concurrent decode across stages. Concurrent prefill is an open decision, not a non-goal (12.4).
- Splitting training: `backward`, gradients and optimizer state across devices.
- GPT-2.
- A CPU stage (host offload of blocks). A stage is `TDeviceType`-homogeneous with its transformer.
- Peer-to-peer and NVLink-specific paths. Nothing on the target hardware has them; the boundary
  mechanism may use them later without changing the stage contract.

---

## 14. Relationship To Other Specs

- `Deployment.md` — the planner that decides devices, placement, context and chunk. This spec supplies
  its device dimension.
- `MemoryFootprint.md` — the footprint contract that prices each stage.
- `MixtureOfExperts.md` §8 — the 26B residency table and the expert-streaming axis (§8.1). Streaming is
  a decode-only way to run a model that does not fit; a split also covers prefill.
- `Gemma4MoE.md` — the 26B implementation record; Phase 6 here follows its Phase 8.
- `Qwen3.8.md` §8 — the cross-architecture finding and the scoring corpus.
- `WeightTying.md` — the tied table (3.5).
- `Workspaces.md` — the shared workspaces that become per stage.
- `SpeculativeDecoding.md` — the other use of two cards: a drafter on one, the target on the other.
  Independent of this spec.
- `ChatConfiguration.md` — the device list in Chat (section 8).
