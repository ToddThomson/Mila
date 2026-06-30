# Weight Tying

Specification and implementation plan for sharing the token embedding table and
language-model head projection. The delivered target is **Gemma 4 12B**; the
same architecture-agnostic plumbing pre-paves a deferred follow-up for Llama 3.2
(see §6).

---

## 1. Problem Statement

Mila loads the token embedding table (`temb.wte`) and the language-model
projection head (`lm_head.weight`) as two independent device allocations even
when the source model ties them (`tie_word_embeddings = true`). The converters
currently copy the embedding tensor into a second blob so the loader "needs no
tying logic" (`Llama/convert_weights.py`, lm_head block). This doubles the
VRAM cost of the single largest tensor in these models.

---

## 2. VRAM Impact

| Model         | Shape (vocab x dim) | BF16 size | Tied in HF? | Saving   |
|---------------|---------------------|-----------|-------------|----------|
| Llama 3.2 1B  | 128256 x 2048       | ~524 MB   | yes         | ~524 MB  |
| Llama 3.2 3B  | 128256 x 3072       | ~789 MB   | yes         | ~789 MB  |
| Llama 3.1 8B  | 128256 x 4096       | ~1050 MB  | no          | 0        |
| Gemma 4 12B   | 262144 x 3840       | ~2013 MB  | yes         | ~2.0 GB  |

The Gemma 4 12B figure is the priority and the reason this work exists: the
12 GB dev card is already memory-constrained (BACKLOG, Gemma Step 5 footprint
analysis), and reclaiming ~2 GB is the single largest available lever that
requires no kernel changes. Llama 3.2 3B/1B (~789 MB / ~524 MB) are on models
that already fit comfortably, so they are deferred (§6); Llama 3.1 8B is untied
in HF and gains nothing (D6).

---

## 3. Design Decisions

**D1 — Shared device allocation, not blob reuse.**
Feeding the embedding blob data into `lm_head` at load time (two device copies)
saves nothing in VRAM. The saving requires both components to hold a pointer
into the same CUDA allocation.

**D2 — Share at the post-load step, not at build time.**
`TokenEmbedding::wte_` is allocated inside `onBuilding`, before weights are
read from disk. The correct aliasing point is after `loadParameters` streams
all blobs: check the `tie_word_embeddings` flag, then install the shared pointer
into `lm_head`.

**D3 — `shared_ptr` upgrade for `wte_` only; `Linear::weight_` is already `shared_ptr`.**
`Linear::weight_` is **already** `std::shared_ptr<WeightTensorType>` (allocated
with `make_shared`) — the Linear side needs no ownership change, only the
additive `installSharedWeight()`. Only `TokenEmbedding::wte_` is still
`unique_ptr` and must become `shared_ptr` within the same type so the
transformer can share ownership after load. All existing call sites use `.get()`
and are unaffected.

The mechanism that makes this safe is one level below the component: `Tensor`
already holds its storage as `std::shared_ptr<TensorBuffer>`. The shared
`TokenEmbedding::wte_` and the `lm_head` weight therefore keep the single
underlying device buffer alive as long as *either* component refers to it —
teardown order is irrelevant, and no `Tensor::view()` (sub-slice) machinery is
needed because the alias is a full-shape share, not a slice.

For an unquantized `lm_head` (the only case that exists — see D4),
`WeightTensorType` is exactly `Tensor<TPrecision, MR>`, identical to
`TokenEmbedding`'s `EmbeddingTensorType`, so `getWeightTensorShared()` assigns
into `installSharedWeight` with no conversion.

**D4 — `lm_head` is never quantized, so tying is always safe.**
Both transformers alias `using LmHeadLinearType = Linear<TDeviceType, TPrecision>`
(`Gemma.ixx`, `Llama.ixx`) — the model's `TWeightQuantization` policy is **not**
forwarded to `lm_head`. Even for Gemma 4 12B FP4 the `lm_head` is BF16. So
`kIsQuantized` on the `lm_head` is always `false` and the shared BF16 embedding
always feeds a BF16 matmul — there is no live "tied + quantized" path to guard.

The `if constexpr (kIsQuantized) throw` in `installSharedWeight` (§5.4) is kept
as defensive insurance, but it is unreachable by construction: if `lm_head` were
ever quantized, `WeightTensorType` would become e.g. `Tensor<FP4, MR>` and
`installSharedWeight(shared_ptr<Tensor<FP4, MR>>)` would fail to compile against
the BF16 embedding — a build error, not a runtime throw. The guard documents the
invariant; it does not protect a reachable code path.

**D5 — Gemma sqrt(hidden_size) scale moves from converter to runtime.**
The Gemma converter currently folds `sqrt(hidden_size)` into the stored
embedding blob so the transformer forward path stays structurally identical to
Llama (`Gemma.ixx` file header, Step 5d decision). If the embedding and
`lm_head` share storage, `lm_head` would receive a sqrt(d)-scaled weight and
produce wrong logits. The fix is to store the embedding raw (unscaled) and
apply the scale in `TokenEmbedding::forward` via a new `embedding_scale` field
on `TokenEmbeddingConfig`. This is one elementwise scalar multiply on the
embedding output — negligible against the surrounding GEMMs. Requires a
checkpoint re-convert for Gemma. `TensorOps::scale` already exists (landed
2026-06-22 for the Gemma `layer_scalar` work).

**D6 — Llama 3.1 8B (untied) requires no change.**
The `tie_word_embeddings: false` metadata flag leaves the load path unchanged.
The existing two-blob path is correct for 8B.

**D7 — Double-counting correction in `MemoryStats`.**
After tying, both `TokenEmbedding` and `Linear` report the shared allocation as
their own parameter bytes. The owning transformer overrides `getMemoryStats()`
to subtract the `lm_head` weight contribution once when tied.

**D8 — Tying reduces steady-state VRAM, not load-time peak (accepted for v1).**
Aliasing happens *after* streaming, by which point `lm_head` has already
allocated its own ~2 GB weight in `onBuilding`; that allocation is freed only
when `installSharedWeight` reassigns the `shared_ptr`. So the load-time peak is
unchanged (both allocations coexist during streaming) while steady-state drops
by one `vocab x dim` tensor. This is the correct tradeoff: the constraint being
relieved is the post-load budget that feeds State / KV-cache (the larger context
window), not the load transient. Avoiding the transient would require the tie
flag at *build* time so `lm_head` skips its allocation, but `build()` currently
runs before `loadParameters` reads metadata; reordering that is out of scope.

---

## 4. Affected Files

### Target — Gemma 4 12B (requires checkpoint re-convert)

The component plumbing (metadata flag, `wte_` ownership, `installSharedWeight`,
`embedding_scale`) is architecture-agnostic and lands here. Gemma additionally
requires the sqrt(d) scale-move (D5), which mandates the re-convert.

| File | Change |
|------|--------|
| `Mila/Src/Dnn/Serialization/PretrainedReader.ixx` | Add `bool tie_word_embeddings` to `PretrainedMetadata`; parse from JSON |
| `Mila/Src/Dnn/Components/Embeddings/TokenEmbedding.Config.ixx` | Add `float embedding_scale_` (default 1.0) with fluent setter and getter |
| `Mila/Src/Dnn/Components/Embeddings/TokenEmbedding.ixx` | `wte_` from `unique_ptr` to `shared_ptr` (leave `wte_grad_`); add `getWeightTensorShared()`; apply `embedding_scale` in `forward` when scale != 1.0 |
| `Mila/Src/Dnn/Components/Linear/Linear.ixx` | Add `installSharedWeight()` (`weight_` is already `shared_ptr`) |
| `Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.ixx` | Set `embedding_scale = sqrt(embedding_dim)` on config before build; `tie_word_embeddings_` member; post-load aliasing; `getMemoryStats()` correction; update file-header comment |
| `Mila/Tools/Converters/Gemma/convert_weights.py` | Write raw (unscaled) embedding; write `tie_word_embeddings: true`; skip `lm_head.weight` blob |

### Deferred — Llama 3.2 1B/3B (Good First Issue, §6)

Once the target lands, the embedding-scale field defaults to identity (1.0) for
Llama and the plumbing is already shipped and unit-tested. The remaining surface
is small and architecture-local.

| File | Change |
|------|--------|
| `Mila/Src/Dnn/Components/Transformers/LlaMa/Llama.ixx` | `tie_word_embeddings_` member; post-load aliasing; `getMemoryStats()` correction (identical to Gemma) |
| `Mila/Tools/Converters/Llama/convert_weights.py` | Write `tie_word_embeddings` to metadata; skip `lm_head.weight` blob when tied |

Llama 3.1 8B is untied in HF — no change (D6).

---

## 5. Component-Level Change Detail (Gemma target)

### 5.1 `PretrainedMetadata` — `PretrainedReader.ixx`

Add one field after `use_bias`:

```cpp
bool tie_word_embeddings = false;
```

In `parseMetadataJSON`, alongside the existing field reads, using the same
`extract_bool` helper the other flags use:

```cpp
metadata_.tie_word_embeddings = extract_bool( "tie_word_embeddings" );
```

Caveat (pre-existing, not introduced here): `extract_bool` is positional — it
matches the first `true`/`false` token after the key name and can bleed into the
next field's value. It is correct for `tie_word_embeddings` as long as the
converter writes a literal `true`/`false` immediately after the key, which it
does. Hardening the parser is tracked separately.

### 5.2 `TokenEmbeddingConfig` — `TokenEmbedding.Config.ixx`

Add a scale field (default 1.0 = identity, preserves all existing behavior and
makes the field a no-op for Llama):

```cpp
template <typename Self>
decltype(auto) withEmbeddingScale( this Self&& self, float scale )
{
	self.embedding_scale_ = scale;
	return std::forward<Self>( self );
}

float getEmbeddingScale() const noexcept { return embedding_scale_; }
```

Include in `validate` (scale must be finite and > 0), `toMetadata`,
`fromMetadata`, and `toString`.

Private member:

```cpp
float embedding_scale_{ 1.0f };
```

### 5.3 `TokenEmbedding` — `TokenEmbedding.ixx`

Change `wte_` from `unique_ptr` to `shared_ptr`. **`wte_grad_` stays
`unique_ptr`** — tying is inference-only (§8), the gradient is never shared, so
upgrading it would be noise:

```cpp
// Before:
std::unique_ptr<EmbeddingTensorType> wte_{ nullptr };

// After:
std::shared_ptr<EmbeddingTensorType> wte_{ nullptr };
```

Update `initializeParameters` to use `make_shared`:

```cpp
wte_ = std::make_shared<EmbeddingTensorType>( device_id, wte_shape, this->getName() + ".wte" );
```

`initializeParameterGradients` is unchanged (`wte_grad_` remains `unique_ptr`).

Add a shared-ownership accessor:

```cpp
std::shared_ptr<EmbeddingTensorType> getWeightTensorShared() const noexcept
{
	return wte_;
}
```

Apply the scale in `forward`. After the op writes into `output_`, multiply by
`embedding_scale` when it is not 1.0. `TensorOps::scale(in, scalar, out, ctx)`
already exists, and the in-place form (`in` and `out` are the same tensor) is
valid for the scalar-multiply kernel:

```cpp
if ( config_.getEmbeddingScale() != 1.0f )
	scale( *output_, config_.getEmbeddingScale(), *output_,
		   this->getExecutionContext() );
```

All other members (`loadParameter`, `getParameters`, `parameterCount`,
`getMemoryStats`, `onBuilding`) dereference `wte_.get()` and are unchanged.

### 5.4 `Linear` — `Linear.ixx`

`weight_` is **already** `std::shared_ptr<WeightTensorType>` and already
allocated with `make_shared` — no ownership change required. The only change is
the additive `installSharedWeight`. It must be called only after `onBuilding` has
run (so `operation_` is live and `setParameters` is safe to call again):

```cpp
// Replace the owned weight allocation with a shared tensor from the token
// embedding table. Called by the owning transformer when tie_word_embeddings
// is set. Precondition: the component is built; !kIsQuantized (D4).
void installSharedWeight( std::shared_ptr<WeightTensorType> shared_weight )
{
	if constexpr ( kIsQuantized )
	{
		throw std::logic_error( std::format(
			"Linear '{}': installSharedWeight requires an unquantized lm_head; "
			"tied weights and per-tensor quantization are mutually exclusive",
			this->getName() ) );
	}

	weight_ = std::move( shared_weight );
	operation_->setParameters( weight_.get(), bias_.get() );
}
```

`loadParameter`, `getParameters`, `parameterCount`, and `getMemoryStats` all
access `weight_.get()` and require no changes.

### 5.5 `GemmaTransformer` — `Gemma.ixx`

In `createGraph`, set the embedding scale on the `TokenEmbeddingConfig` before
building (replaces the converter-side sqrt(d) fold, D5):

```cpp
TokenEmbeddingConfig embedding_config;
embedding_config
	.withVocabSize( static_cast<size_t>( config_.getVocabSize() ) )
	.withEmbeddingDim( static_cast<size_t>( config_.getEmbeddingDim() ) )
	.withEmbeddingScale( static_cast<float>(
		std::sqrt( static_cast<double>( config_.getEmbeddingDim() ) ) ) );
```

Add a member flag populated at load time:

```cpp
bool tie_word_embeddings_{ false };
```

Modify `loadParameters` to read the flag and perform the post-stream aliasing.
The body below is the generic transformer load pattern (Llama uses the identical
one — §6.1); only the surrounding class differs:

```cpp
void loadParameters( PretrainedModelReader& reader )
{
	const auto& metadata = reader.getPretrainedMetadata();
	tie_word_embeddings_ = metadata.tie_word_embeddings;

	const int device_index = this->getExecutionContext()->getDeviceId().index;

	auto consume = [&]( const std::string& full_name, const Serialization::ITensorBlob& blob )
	{
		auto [component_path, param_name] = parseParameterPath( full_name );
		ComponentPtr target = this->findComponent( component_path );
		target->loadParameter( param_name, blob );

		if constexpr ( TDeviceType == DeviceType::Cuda )
			this->getExecutionContext()->synchronize();
	};

#ifdef MILA_HAS_CUDA
	if constexpr ( TDeviceType == DeviceType::Cuda )
		reader.streamTensorBlobs<CudaPinnedMemoryResource>( consume, device_index );
	else
#endif
		reader.streamTensorBlobs<CpuMemoryResource>( consume );

	if constexpr ( TDeviceType == DeviceType::Cuda )
		this->getExecutionContext()->synchronize();

	if ( tie_word_embeddings_ )
		lm_head_->installSharedWeight( token_embedding_->getWeightTensorShared() );
}
```

When `tie_word_embeddings` is true, `lm_head.weight` is absent from the file.
`streamTensorBlobs` delivers only the blobs present in the index, so no load
failure occurs for the missing blob.

`getMemoryStats` is **already overridden** in `GemmaTransformer` — it sums
`child->getMemoryStats()` over `getComponents()` and adds attention state. There
is no `NetworkBase::getMemoryStats()` to delegate to. Amend the existing method:
after the child-sum loop (and before `return stats`), subtract the shared
`lm_head` weight contribution once when tied, so it is not counted as both
embedding and `lm_head` parameter bytes (D7):

```cpp
if ( tie_word_embeddings_ && lm_head_ )
	stats.device_parameter_bytes -= lm_head_->getMemoryStats().device_parameter_bytes;
```

Remove the file-header comment that says the scale is folded into the converter;
update it to say the scale is applied at runtime via `TokenEmbeddingConfig`.

### 5.6 `Gemma/convert_weights.py`

Remove the `* normalizer` from the embedding write:

```python
# Raw embedding (no sqrt(hidden_size) fold -- scale is applied at runtime via
# TokenEmbeddingConfig::embedding_scale; lm_head shares this unscaled storage).
writer.add_tensor( 'temb.wte', _tensor_to_numpy( embed, dtype ) )
```

Add `tie_word_embeddings: True` to the metadata dict and skip the `lm_head.weight` blob:

```python
# tie_word_embeddings is always true for Gemma 4: lm_head shares the embedding.
# 'tie_word_embeddings': True  (in the metadata dict above)
print( "  lm_head tied to embed_tokens -- skipping second blob "
	   "(GemmaTransformer aliases at load time)" )
```

The `lm_head` block at the end of `convert_gemma` is removed entirely.

---

## 6. Deferred — Llama 3.2 1B/3B (Good First Issue)

This follow-up is intentionally split out. After the Gemma target lands, the
shared plumbing (§5.1–5.4) is in place and `embedding_scale` defaults to identity
for Llama, so the Llama work is code-local and small. It is a good candidate for
an external contributor.

**Validation caveat:** the code change is small, but the acceptance test (§7.5)
needs the Llama 3.2 checkpoint, a HuggingFace reference, and the greedy-parity
oracle harness. That rig is not trivial to stand up externally. Decide before
labeling the GitHub issue whether the contributor owns parity too, or only the
code while the maintainer runs parity.

### 6.1 `LlamaTransformer` — `Llama.ixx`

Add the `tie_word_embeddings_` member, the post-load aliasing, and the
`getMemoryStats` correction — **identical** to the Gemma versions in §5.5
(`loadParameters` body and `getMemoryStats` amendment are the same pattern).
Llama needs no `embedding_scale` (it stays 1.0), so there is no `createGraph`
scale change.

### 6.2 `Llama/convert_weights.py`

Replace the current `lm_head` block with a flag-aware version. The metadata
dict written earlier gains `tie_word_embeddings`:

```python
tie = bool( getattr( config, 'tie_word_embeddings', False ) )

# ... (in the metadata dict already written above) ...
#   'tie_word_embeddings': tie,

if not tie:
	lm_head_key = 'lm_head.weight'
	lm_head_tensor = (
		state_dict[lm_head_key]
		if lm_head_key in state_dict
		else state_dict['model.embed_tokens.weight']
	)
	writer.add_tensor( 'lm_head.weight', _tensor_to_numpy( lm_head_tensor, dtype ) )
else:
	print( "  lm_head tied to embed_tokens -- skipping second blob "
		   "(LlamaTransformer aliases at load time)" )
```

Note: existing Llama checkpoints lack the flag, load as untied, and behave
exactly as today (zero saving). The re-convert is what realizes the saving — the
stored embedding *format* is already raw, unlike Gemma where the stored *data*
changes.

---

## 7. Test Plan

### 7.1 `TokenEmbedding.Cuda.cpp` — new case

`Forward_WithEmbeddingScale_ScalesOutput`: build a `TokenEmbedding<Cuda, BF16>`
with `embedding_scale = 2.0f`. Assert every output element equals exactly
`2 * reference` (FP32 exact for scale = 2). Assert that `embedding_scale = 1.0`
(default) produces identical output to the unmodified path.

### 7.2 `Linear.Cuda.cpp` — new cases

`InstallSharedWeight_SetsParameterAndMatchesDirectLoad`: allocate a BF16
tensor of the correct shape, call `installSharedWeight`, verify
`getParameters()[0]` returns the same raw pointer, and that `forward` output
matches a reference run where the blob was loaded directly.

`InstallSharedWeight_QuantizedPath_Throws`: instantiate
`Linear<Cuda, BF16, PerGroupFp4<128>>` and assert `installSharedWeight`
throws `std::logic_error`. This exercises the guard in isolation; per D4 the
guard is unreachable in the real model set (no quantized `lm_head` exists), but
the unit test pins the documented invariant.

### 7.3 `GemmaTransformer.Cuda.cpp` — full load-tie round-trip (DEFERRED)

The intended test synthesizes a small pretrained artifact (two layers, small
vocab and dim) with `tie_word_embeddings: true` and `lm_head.weight` absent, then
after `loadParameters` asserts the shared pointer identity and the no-double-count
`getMemoryStats`. **Deferred** (2026-07-01): `PretrainedModelReader` is mmap/file-
only with no C++ writer (the checkpoint writer is Python-only, `Tools/.../common.py`),
and `GemmaTransformer::token_embedding_` / `lm_head_` are private. A byte-exact
test-local writer would be brittle scaffolding against an undocumented format.
Tracked in BACKLOG as a reusable C++ test-checkpoint writer (also unblocks the
deferred Llama tying test, §6).

Coverage in the meantime:
- The aliasing primitive is unit-tested at the component level by §7.2
  (`installSharedWeight` pointer identity + shared-byte reporting + forward) and
  by `getWeightTensorShared` returning the embedding's shared storage.
- The post-load wiring and the D7 `getMemoryStats` / `parameterCount` corrections
  are exercised end-to-end by the validated Gemma 4 12B chat run (§7.4): coherent
  output requires the tie to have installed the shared raw table correctly.

### 7.4 Parity — Gemma 4 12B (acceptance gate)

Re-convert Gemma 4 12B with the sqrt(d)-free converter. Run
`GemmaModel.Parity.Cuda.cpp`. Expected token ids are unchanged (the scale is
still applied; only the application point moved). Also verify that the chat
sample produces coherent text with the re-converted checkpoint, and confirm the
~2 GB steady-state reduction via `getMemoryStats`.

### 7.5 Parity — Llama 3.2 3B (deferred, §6)

Re-convert Llama 3.2 3B with the updated converter (tied blob skipped). Run the
greedy-parity oracle. Token-for-token match confirms aliasing does not alter
inference numerics. Gated with the deferred Llama follow-up.

---

## 8. Implementation Sequence

Gemma target:

1. Add `tie_word_embeddings` to `PretrainedMetadata` and parse it — zero
   behavioral change until the flag is `true`.
2. Add `embedding_scale` to `TokenEmbeddingConfig`; apply in
   `TokenEmbedding::forward`. Unit-test the scale path (§7.1).
3. Change `TokenEmbedding::wte_` to `shared_ptr` (leave `wte_grad_` as
   `unique_ptr`); add `getWeightTensorShared()`.
4. Add `Linear::installSharedWeight()` (`weight_` is already `shared_ptr`).
   Unit-test install + quantized-throw (§7.2).
5. Update `Gemma/convert_weights.py` — raw embedding, write flag, skip lm_head
   blob. Validate the converter produces a parseable checkpoint.
6. Wire `GemmaTransformer`: set `embedding_scale = sqrt(dim)`, add
   `tie_word_embeddings_`, post-load aliasing, `getMemoryStats` correction,
   header comment. Run the aliasing isolation gate (§7.3), then re-convert and
   run Gemma parity (§7.4, acceptance gate).

Deferred (good first issue, §6): add the `tie_word_embeddings_` member +
post-load aliasing + `getMemoryStats` correction to `LlamaTransformer`; update
`Llama/convert_weights.py`; re-convert Llama 3.2 3B and run parity (§7.5).

---

## 9. Non-Goals

- KV-cache weight sharing (orthogonal concern).
- Training with tied weights — gradient aliasing requires a single accumulated
  `wte_grad`; out of scope as Llama and Gemma are inference-only models in Mila.
- Llama 3.1 8B — untied in HF (`tie_word_embeddings: false`); no change.
- Any change to quantization policies, `OperationTraits`, or CUDA kernels.
