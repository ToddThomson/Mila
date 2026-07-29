# Model Serialization

Design notes and build plan for saving and restoring a Mila model: the `ModelArchive` checkpoint
path, its relationship to the flat `.bin` pretrained path, and the sequenced work to make either one
trustworthy.

Scoped 2026-07-29 from a read of the shipped code. Defects found during that read are tracked in
[BACKLOG.md](../../BACKLOG.md); this document carries the design and the order of work.

---

## Why

Two unrelated needs keep arriving at the same code:

**Training cannot resume.** MNIST and Bard train to convergence and then have nowhere to put the
result. The BF16 train-from-scratch path was repaired in `e585be9d`, which makes a long run worth
starting and therefore worth checkpointing.

**Distribution wants a smaller artifact.** The converter always writes BF16 because quantization is a
load-time policy, so the only artifact Mila can currently produce for Gemma 4 12B is 22.2 GB. That
number is what ruled out hosting in [PythonBinding.md](PythonBinding.md). A pre-quantized artifact
would be roughly a third of it.

These want different formats. Conflating them is the main risk this document exists to prevent.

---

## Two artifacts, not two modes

`SerializationMode` (`Serialization/SerializationMode.ixx`) offers `Checkpoint`, `WeightsOnly` and
`Architecture`, which reads as three flags over one format. It is not.

| | Training checkpoint | Distribution artifact |
|---|---|---|
| Contents | parameters + optimizer moments + step/epoch + RNG state | parameters only |
| Lifetime | written repeatedly, read by the same build | written once, read by strangers |
| Compatibility | may break between versions | must not |
| Access pattern | read once, sequentially | memory-mapped, random access by tensor name |
| Size pressure | none — it is a local file | decisive — it is a download |
| Fit | `ModelArchive` (zip, per-component blobs) | flat `.bin` + `PretrainedModelReader` |

The flat path already exists and is good: a `MILA` magic header, a tensor index, mmap plus a pinned
double-buffered staging thread (`PretrainedReader.ixx:148`, `:392`). Routing distribution through
`ModelArchive` would replace a memory-mapped read with zip decompression into a heap buffer, and
produce a 22 GB zip file. **`ModelArchive` is the checkpoint format. The flat `.bin` is the
distribution format. `WeightsOnly` on the archive is a debugging convenience, not the delivery
vehicle.**

The consequence for packaging: a small hostable artifact needs **quantized-tensor serialization in
the flat format**, not a mode flag on the archive. That is a genuine feature and belongs after
v0.20.

---

## Current surface

More exists than a first look suggests. The pieces below are shipped and working.

| Piece | Where | State |
|---|---|---|
| `ModelArchive` | `Serialization/ModelArchive.ixx` | zip-backed; scoped paths, `ScopedScope`, JSON metadata, blob read/write/read-into |
| `ZipSerializer` / `ArchiveSerializer` | `Serialization/` | miniz backing; `[[nodiscard]] bool` on every fallible call |
| `writeTensorBlob` | `Tensors/Tensor.Serialization.ixx:142` | writes `<prefix>/meta.json` + `<prefix>/data.bin` |
| `readTensorBlob<MR>` | `Tensors/Tensor.Serialization.ixx:172` | reads both back into a `TensorBlob<MR>`; pinned-MR aware |
| `ITensorBlob` / `TensorBlobView` | `Tensors/Tensor.Serialization.ixx:53`, `:110` | the type-erased boundary between a byte source and a component |
| `Component::loadParameter( name, ITensorBlob& )` | `Core/Component.ixx:509` | implemented by all five parameter-owning components |
| `Network::save()` | `Core/Network.ixx:273` | writes `network/meta.json`, `architecture.json`, then recurses |
| `ComponentFactory::readComponentMeta` | `Core/ComponentFactory.ixx:68` | reads a component's `meta.json` under its scope |
| `PretrainedModelReader` | `Serialization/PretrainedReader.ixx` | the flat `.bin` path; mmap + pinned double-buffer |

**The load side is further along than "absent".** `ITensorBlob` is already the virtual boundary
between a byte source and a component, `readTensorBlob` already produces one from an archive, and
`loadParameter` is already implemented by `Linear`, `RmsNorm`, `LayerNorm`, `TokenEmbedding` and
`Lpe` — exactly the five components that own parameters. What is missing is not a load mechanism but
the **traversal** that walks an archive and drives those five implementations. The flat path has such
a traversal (`Gemma.ixx:383`, `Llama.ixx:377`, `GptTransformer.ixx:532`, each hardcoded to
`PretrainedModelReader&`); the archive path has none.

---

## What is missing

- **No `Component::load_`.** `save_` is pure virtual with 25 overrides; there is no counterpart, so
  nothing mirrors the save traversal.
- **No `Network::load()`.** `Network.ixx:83` documents a "concrete class MUST provide a static
  `Load()`" convention in a comment block and provides no support for it. `GptModel.ixx:164` declares
  `fromCheckpoint()` and throws, naming what is absent (`GptConfig::fromArchive`, `GptTransformer`
  save/load).
- **`getParameterNames()` has zero overrides.** `Core/Component.ixx:518` documents it as "the
  canonical parameter name list in the same stable order used by `save_()` and `loadParameter()`".
  No component implements it, so that stable order does not exist and save and load have no agreed
  vocabulary.
- **No optimizer state anywhere.** `SerializationMode::Checkpoint` is documented as "architecture +
  weights + optimizer state". `AdamWConfig` serializes hyperparameters; the moments, step count and
  master parameters have no representation. `Checkpoint` cannot currently mean what it says.
- **The transformer load traversals are welded to the flat reader.** `loadParameters(
  PretrainedModelReader& )` takes a concrete type, not a blob source, so nothing can feed it from an
  archive.
- **No quantized tensor representation** in either format. A `PerGroupFp4<128>` weight is packed
  nibbles plus per-group scales; a `PerChannelFp8<>` weight is FP8 plus FP32 scales. Neither
  `TensorMetadata` nor the flat tensor index can express the scale companion.

---

## Defects in the shipped path

Three, all found 2026-07-29. The first two are tracked under Production Hardening in
[BACKLOG.md](../../BACKLOG.md); the third is the layout defect and is filed alongside them.

**1. Four parameter-owning components have an empty `save_`.** Of the ten empty overrides, six are
correct — `Softmax.ixx:213`, `SoftmaxCrossEntropy.ixx:168`, `Rope.ixx:197`, `Residual.ixx:224`,
`MultiHeadAttention.ixx:248` and `GroupedQueryAttention.ixx:406` all report `parameterCount() == 0`.
Four are not: `RmsNorm.ixx:169`, `LayerNorm.ixx:210`, `TokenEmbedding.ixx:250` and `Lpe.ixx:274`.
`Linear` is the only component that writes tensor blobs, so a saved transformer holds the projection
weights and nothing else, and reports success.

**2. `Linear::save_` truncates and mislabels any non-FP32 weight.** `Linear.ixx:306` and `:308` take
dtype and byte count from the device tensor; `:317` stages through a host
`Tensor<dtype_t::FP32, CpuMemoryResource>`; `:323` writes that FP32 buffer using the BF16 byte count.
Half the buffer, labelled with the wrong dtype. The bias block repeats it at `:330`, `:332`, `:341`,
`:347`.

**3. `CompositeComponent::save_` collides every descendant into one scope.** `Network::save()` pushes
`components/<name>` per top-level child (`Network.ixx:510`), but the composite's own recursion at
`CompositeComponent.ixx:783` calls `component->save_( archive, mode )` with **no `ScopedScope`**. So
every descendant of a composite writes to its parent's scope: in a 48-block transformer, every
`Linear` in every block writes `tensors/weight/data.bin` at the same path, each overwriting the last.
Compounding it, the same function records `type`, `version`, `child_count` and `child_names` through
`archive.addMetadata()` (`:764`–`:782`), which is the **archive-global** store — `ZipSerializer.ixx:426`
writes to the unscoped path `metadata/<key>`, bypassing `scopedPath()` entirely, so every composite in
the model overwrites the same four keys. This defect makes the other two moot: fixing the four empty
`save_` bodies without fixing the layout produces a larger archive that is still wrong.

---

## Design decisions

**`ITensorBlob` is the load contract, and it does not change.** Every load path — flat file, archive,
and any future source — produces `ITensorBlob`s and calls `loadParameter`. The five existing
implementations handle shape validation, precision conversion and device upload, and they stay
untouched. This is what makes the archive load path small rather than a second implementation of
weight loading.

**`getParameterNames()` is the join between save and load.** Once each parameter-owning component
returns its canonical names, `save_` iterates that list instead of hand-rolling per-tensor blocks, and
the default `load_` iterates the same list to read them back. The two sides cannot drift because they
read the same vector. Implementing it is the precondition for both, not a tidy-up afterwards.

**Do not extend `NetworkFactory` for models.** `Core/NetworkFactory.ixx` is a string-keyed runtime
registry mapping `"network_type"` to a factory lambda — the same pattern
[CLAUDE.md](../../CLAUDE.md) records as being phased out with `OperationRegistry`. It is also
unnecessary here: `GemmaModel::fromCheckpoint` knows its own type at compile time, exactly as
`fromPretrained` does. Leave the factory in place for the generic `Network` case that MNIST uses;
route concrete models through their own static factory.

**Saving stays a member, loading stays a static factory.** `saveCheckpoint( path )` on the model;
`fromCheckpoint( path, ... )` as a static, matching `fromPretrained`. The asymmetry is correct —
loading constructs an object, saving does not. Both are thin wrappers over
`save( ModelArchive&, SerializationMode )`. Do not name the general API after
`SerializationMode::Checkpoint`, which is one mode of three. `GptModel.ixx:164` already declares
`fromCheckpoint`, so the vocabulary is chosen; do not invent more.

**One staging implementation, not five.** The device-to-host branch in `Linear::save_` will be needed
by all five parameter-owning components. It belongs in `Tensor.Serialization.ixx` next to
`writeTensorBlob`, taking an `ITensor*` and mirroring the device dtype rather than widening to FP32.
Copying the current branch four more times would replicate defect 2 four more times.

**A component that cannot serialize its parameters must throw.** Silence is what produced defect 1.
`GptModel::fromCheckpoint` is the model: refuse clearly rather than succeed emptily.

---

## Build plan

Phases 0–5 are the training-checkpoint spine and are sequenced so each one is independently
verifiable. Phase 6 is what `Checkpoint` mode needs to be honest. Phase 7 is the distribution
artifact and is vNext.

> **Status 2026-07-29 — Phases 0 and 1 done and verified; Phases 2 and 3 implemented, their machinery
> verified, their per-component criteria not yet met.** Build, full suite, and the new serialization
> coverage all green. Two design decisions were added during implementation: `Linear` and
> `TokenEmbedding` refuse outright on the quantized path rather than write a blob nothing can read
> back (Phase 3), and the device staging buffer is **pinned** rather than `CpuMemoryResource` —
> every reduced precision Mila uses is `is_device_only`, so `Tensor<BF16, CpuMemoryResource>` is not a
> valid template-id and same-dtype staging is impossible on host memory. That constraint, not
> carelessness, is why the original code widened to FP32.
>
> **Phases 4 and 5 done and verified 2026-07-29** — build and full ctest green, including a real
> two-layer GPT round trip. Phase 4's `Component::load_` carries a working default rather than being
> pure: it walks the same `getParameterNames()` vector `save_` walked, so the two directions cannot
> drift. Phase 5 turned out cheaper than scoped because `GptConfig` already had
> `toMetadata()`/`fromMetadata()`, which also exposed that the hand-rolled metadata block it replaced
> wrote `mlp_hidden_dim` where `fromMetadata` reads `hidden_dim` and omitted `use_bias` entirely.
>
> **The finding that mattered most came from a compile error, not a test.** Adding a virtual `load_`
> forced instantiation of dormant hand-rolled `load_` methods in the transformer blocks, which
> revealed that `GptBlock`, `Llama.Block`, `Gemma.Block`, `MLP` and `GatedMLP` all overrode `save_`
> with an **unscoped** walk — so the Phase 1 scoping repair never ran on any real model, and the
> Phase 1 test could not see it because its mock composite does not override `save_`. Lesson worth
> carrying: when a fix lands in a virtual hook, a mock proves the hook works, not that anything calls
> it. Four of the five now inherit the base traversal; `Gemma.Block` calls the base and then writes
> its own `layer_scalar`, which the hand-rolled version never wrote at all.
>
> **Coverage is GPT-only.** `Llama.Block`, `Gemma.Block` (including the net-new `layer_scalar` path),
> `RmsNorm`/`TokenEmbedding` `getParameterNames()`, and `GptModel::saveCheckpoint`/`fromCheckpoint`
> themselves are all still unexercised. **Phase 6 is not started.**

### Phase 0 — make the current path honest

Freeze-compatible; adds no capability. In `Component::save_`'s contract, a component reporting
`parameterCount() > 0` with no implementation throws. The four offenders start throwing, the six
parameterless no-ops are annotated as deliberate, and `Network::save()` on a transformer fails loudly
instead of writing a misleading archive.

*Done when:* saving a Gemma or Llama network raises, naming the first component that cannot
serialize.

### Phase 1 — archive layout

Fix defect 3. `CompositeComponent::save_` pushes a `ScopedScope` per child; composite metadata moves
from `addMetadata()` to `writeMetadata( "meta.json", ... )` so it lands under the composite's scope.
Write the resulting path grammar into this document as the format definition.

*Done when:* a two-level composite round-trips to distinct, predictable paths, pinned by a test that
lists the archive and asserts the full path set.

### Phase 2 — the parameter-name contract

Implement `getParameterNames()` on `Linear`, `RmsNorm`, `LayerNorm`, `TokenEmbedding` and `Lpe`,
returning the names `loadParameter` already accepts. Rewrite `Linear::save_` to iterate it.

*Done when:* for each of the five, `getParameterNames()` matches the set `loadParameter` accepts —
asserted directly, since a mismatch here silently breaks the round trip later.

### Phase 3 — complete the save side

Add the shared staging helper described above; fix defect 2 by mirroring the device dtype; implement
`save_` for the four components via the helper and `getParameterNames()`.

*Done when:* a saved Gemma archive contains one blob per parameter of every component, with byte
counts matching `parameterCount() * elementSize()`.

### Phase 4 — the archive load traversal

Add `Component::load_( ModelArchive&, SerializationMode )` mirroring `save_`, with a **default
implementation on `Component`**: for each name in `getParameterNames()`, `readTensorBlob` under the
component's scope and hand it to `loadParameter`. That default covers all five leaves with no
per-component code. `CompositeComponent::load_` recurses under the same scopes Phase 1 established;
`Network::load()` reads `architecture.json` and drives it.

*Done when:* save then load on a small built network reproduces every parameter bit-for-bit, on both
CPU and CUDA.

### Phase 5 — the model-level API

`saveCheckpoint( path )` and `fromCheckpoint( path, ... )`, on `GptModel` first — its stub already
exists and GPT-2 is the smallest real model. Then MNIST.

*Done when:* Bard trains N steps, checkpoints, restarts from the checkpoint, and continues with the
same loss trajectory. That is the acceptance test for the whole spine; nothing short of it proves the
round trip.

### Phase 6 — optimizer state

`AdamW` moments, step count, and the FP32 master parameters. Until this lands, `Checkpoint` and
`WeightsOnly` produce the same archive and resuming a run restarts the optimizer, which for Adam
means a visible loss bump at the resume point.

*Done when:* the Phase 5 resume test shows no discontinuity at the seam.

### Phase 7 — the distribution artifact (vNext)

Quantized-tensor serialization in the **flat** format: an FP4 tensor is packed nibbles plus a
per-group scale tensor, an FP8 tensor is FP8 plus per-channel FP32 scales, and the tensor index needs
to express that pairing. This is what turns a 22.2 GB Gemma BF16 artifact into roughly 7 GB and makes
hosting viable. Depends on nothing in Phases 0–6 except the vocabulary.

---

## Freeze boundary (v0.20)

In bounds — defect repair in shipped code:

- Phase 0 (throw instead of no-op)
- Phase 1 (the scoping and metadata-namespace fix)
- Phase 2 and 3 are arguable: they repair `save_` implementations that exist and are wrong, but the
  four empty ones are closer to unwritten than broken. Treat as a scope call.

Out of bounds — feature additions, vNext:

- Phase 4 (the load traversal) and Phase 5 (the model API)
- Phase 6 (optimizer state)
- Phase 7 (quantized distribution artifact)

Phase 0 alone is worth doing under the freeze regardless of where the rest lands: it converts a
silent-corruption path into an honest refusal, which is what hardening toward a production release
means.

---

## Open decisions

1. **Whether Phases 2–3 ride the freeze.** They are the difference between "save refuses" and "save
   works", and they are two of the cheaper phases. Doing 0–3 and stopping leaves a save path that is
   correct and a load path that does not exist, which is a defensible place to pause.
2. **Whether `Architecture` mode survives.** Nothing consumes it, and reconstructing a network from
   serialized topology is what `NetworkFactory`'s string registry exists for — the pattern being
   phased out. If concrete models reconstruct themselves through `fromCheckpoint`, `Architecture`
   mode has no caller.
3. **Checkpoint format versioning.** The flat format has `MAGIC` + `VERSION` and rejects mismatches
   (`PretrainedReader.ixx:392`, `:805`). The archive has `network/meta.json` with a version field that
   nothing validates. Decide the compatibility promise before the first checkpoint is written by a
   user, not after.
