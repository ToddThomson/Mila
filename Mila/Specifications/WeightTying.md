# Weight Tying

Specification and implementation plan for sharing the token embedding table
and language-model head projection across Llama 3.2 and Gemma 4.

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

The Gemma 4 12B figure is the priority: the 12 GB dev card is already
memory-constrained (BACKLOG, Gemma Step 5 footprint analysis), and reclaiming
~2 GB is the single largest available lever that requires no kernel changes.

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

**D3 — `shared_ptr` upgrade for `wte_` and `lm_head` weight tensor.**
Both `TokenEmbedding::wte_` and `Linear::weight_` are currently `unique_ptr`.
Changing them to `shared_ptr` within the same type allows the transformer to
share ownership after load without redesigning either component. All existing
call sites use `.get()` and are unaffected.

**D4 — Quantized `lm_head` and tying are mutually exclusive.**
When `TWeightQuant::kIsQuantized`, `Linear::loadParameter` calls
`operation_->quantize(blob, *weight_, *weight_scales_, ...)`. A tied `lm_head`
must bypass quantization entirely and operate on the BF16 embedding tensor
directly. This is correct: `lm_head` is a single large matmul over the
vocabulary; quantizing it separately is only meaningful when it carries
independent weights. Enforced at load time: if `tie_word_embeddings &&
kIsQuantized`, `installSharedWeight` throws `std::logic_error`.

**D5 — Gemma sqrt(hidden_size) scale moves from converter to runtime (Phase 2).**
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

---

## 4. Affected Files

### Phase 1 — Llama (no checkpoint re-convert required)

| File | Change |
|------|--------|
| `Mila/Src/Dnn/Serialization/PretrainedReader.ixx` | Add `bool tie_word_embeddings` to `PretrainedMetadata`; parse from JSON |
| `Mila/Src/Dnn/Components/Embeddings/TokenEmbedding.ixx` | `wte_` from `unique_ptr` to `shared_ptr`; add `getWeightTensorShared()` |
| `Mila/Src/Dnn/Components/Linear/Linear.ixx` | `weight_` from `unique_ptr` to `shared_ptr`; add `installSharedWeight()` |
| `Mila/Src/Dnn/Components/Transformers/LlaMa/Llama.ixx` | `tie_word_embeddings_` member; post-load tying step; `getMemoryStats()` correction |
| `Mila/Tools/Converters/Llama/convert_weights.py` | Write `tie_word_embeddings` to metadata; skip `lm_head.weight` blob when tied |

### Phase 2 — Gemma (requires checkpoint re-convert)

| File | Change |
|------|--------|
| `Mila/Src/Dnn/Components/Embeddings/TokenEmbedding.Config.ixx` | Add `float embedding_scale_` (default 1.0) with fluent setter and getter |
| `Mila/Src/Dnn/Components/Embeddings/TokenEmbedding.ixx` | Apply `embedding_scale` scalar multiply to forward output when scale != 1.0 |
| `Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.ixx` | Set `embedding_scale = sqrt(embedding_dim)` on config before build; post-load tying step; `getMemoryStats()` correction |
| `Mila/Tools/Converters/Gemma/convert_weights.py` | Write raw (unscaled) embedding; write `tie_word_embeddings: true`; skip `lm_head.weight` blob |

---

## 5. Component-Level Change Detail

### 5.1 `PretrainedMetadata` — `PretrainedReader.ixx`

Add one field after `use_bias`:

```cpp
bool tie_word_embeddings = false;
```

In `parseMetadataJSON`, alongside the existing field reads:

```cpp
if ( auto it = j.find( "tie_word_embeddings" ); it != j.end() )
	meta.tie_word_embeddings = it->get<bool>();
```

### 5.2 `TokenEmbedding` — `TokenEmbedding.ixx`

Change `wte_` and its gradient from `unique_ptr` to `shared_ptr`:

```cpp
// Before:
std::unique_ptr<EmbeddingTensorType> wte_{ nullptr };
std::unique_ptr<EmbeddingTensorType> wte_grad_{ nullptr };

// After:
std::shared_ptr<EmbeddingTensorType> wte_{ nullptr };
std::shared_ptr<EmbeddingTensorType> wte_grad_{ nullptr };
```

Update `initializeParameters` to use `make_shared`:

```cpp
wte_ = std::make_shared<EmbeddingTensorType>( device_id, wte_shape, this->getName() + ".wte" );
```

Update `initializeParameterGradients` similarly:

```cpp
wte_grad_ = std::make_shared<EmbeddingTensorType>( device_id, wte_->shape(), this->getName() + ".wte.grad" );
```

Add a shared-ownership accessor:

```cpp
std::shared_ptr<EmbeddingTensorType> getWeightTensorShared() const noexcept
{
	return wte_;
}
```

All other members (`loadParameter`, `getParameters`, `parameterCount`,
`getMemoryStats`, `onBuilding`) dereference `wte_.get()` and are unchanged.

### 5.3 `Linear` — `Linear.ixx`

Change `weight_` from `unique_ptr` to `shared_ptr`:

```cpp
// Before:
std::unique_ptr<WeightTensorType> weight_{ nullptr };

// After:
std::shared_ptr<WeightTensorType> weight_{ nullptr };
```

Update `initializeParameters` to use `make_shared`:

```cpp
weight_ = std::make_shared<WeightTensorType>( device_id, weight_shape, this->getName() + ".weight" );
```

Add `installSharedWeight`. This must be called only after `onBuilding` has
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

### 5.4 `LlamaTransformer` — `Llama.ixx`

Add a member flag populated at load time:

```cpp
bool tie_word_embeddings_{ false };
```

Modify `loadParameters` to read the flag and perform the post-stream aliasing:

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

Override `getMemoryStats` to avoid counting the shared allocation twice (D7):

```cpp
MemoryStats getMemoryStats() const override
{
	auto stats = NetworkBase::getMemoryStats();

	if ( tie_word_embeddings_ && lm_head_ )
		stats.device_parameter_bytes -= lm_head_->getMemoryStats().device_parameter_bytes;

	return stats;
}
```

### 5.5 `Llama/convert_weights.py`

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

---

## 6. Phase 2 Detail

### 6.1 `TokenEmbeddingConfig` — `TokenEmbedding.Config.ixx`

Add a scale field (default 1.0 = identity, preserves all existing behavior):

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

### 6.2 `TokenEmbedding::forward` — `TokenEmbedding.ixx`

After the op writes into `output_`, apply the scale when it is not 1.0.
`TensorOps::scale(in, scalar, out, ctx)` already exists:

```cpp
if ( config_.getEmbeddingScale() != 1.0f )
	scale( *output_, config_.getEmbeddingScale(), *output_,
		   this->getExecutionContext() );
```

The in-place form (`in` and `out` are the same tensor) is valid for the scalar
multiply kernel. No new kernel is introduced.

### 6.3 `GemmaTransformer` — `Gemma.ixx`

In `createGraph`, set the embedding scale before building:

```cpp
TokenEmbeddingConfig embedding_config;
embedding_config
	.withVocabSize( static_cast<size_t>( config_.getVocabSize() ) )
	.withEmbeddingDim( static_cast<size_t>( config_.getEmbeddingDim() ) )
	.withEmbeddingScale( static_cast<float>(
		std::sqrt( static_cast<double>( config_.getEmbeddingDim() ) ) ) );
```

Add `tie_word_embeddings_` member and the same post-load aliasing and
`getMemoryStats` correction as `LlamaTransformer` (§5.4).

Remove the file-header comment that says the scale is folded into the converter;
update it to say the scale is applied at runtime via `TokenEmbeddingConfig`.

### 6.4 `Gemma/convert_weights.py`

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
throws `std::logic_error`.

### 7.3 `LlamaTransformer.Cuda.cpp` — new case

`LoadParameters_TiedEmbedding_SharedAllocation`: synthesize a small pretrained
artifact (two layers, vocab 256, dim 64) with `tie_word_embeddings: true` and
`lm_head.weight` absent from the blob index. After loading, assert:

- `lm_head_->getParameters()[0]` raw pointer equals
  `token_embedding_->getWeightTensorShared().get()->rawData()`.
- `getMemoryStats().device_parameter_bytes` does not double-count the
  embedding allocation.

### 7.4 Parity — Llama 3.2 3B

Re-convert Llama 3.2 3B with the updated converter (tied blob skipped). Run the
greedy-parity oracle. Token-for-token match confirms aliasing does not alter
inference numerics.

### 7.5 Parity — Gemma 4 12B (Phase 2)

Re-convert Gemma 4 12B with the sqrt(d)-free converter. Run
`GemmaModel.Parity.Cuda.cpp`. Expected token ids are unchanged (scale is still
applied; only the application point moved). Also verify that the chat sample
produces coherent text with the re-converted checkpoint.

---

## 8. Implementation Sequence

1. Add `tie_word_embeddings` to `PretrainedMetadata` and parse it — zero
   behavioral change until the flag is `true`.
2. Change `TokenEmbedding::wte_` and `wte_grad_` to `shared_ptr`; add
   `getWeightTensorShared()`.
3. Change `Linear::weight_` to `shared_ptr`; add `installSharedWeight()`.
4. Update `Llama/convert_weights.py` — write flag; skip blob when tied.
   Validate converter produces a parseable checkpoint.
5. Add `tie_word_embeddings_` member to `LlamaTransformer`; wire post-load
   aliasing and `getMemoryStats` correction. Run Llama 3.2 3B parity test.
6. (Phase 2) Add `embedding_scale` to `TokenEmbeddingConfig` and apply in
   `TokenEmbedding::forward`. Unit-test the scale path.
7. (Phase 2) Update `Gemma/convert_weights.py` — raw embedding, skip lm_head
   blob. Wire `GemmaTransformer` tying and scale. Re-convert and run Gemma
   parity test.

---

## 9. Non-Goals

- KV-cache weight sharing (orthogonal concern).
- Training with tied weights — gradient aliasing requires a single accumulated
  `wte_grad`; out of scope as Llama and Gemma are inference-only models in Mila.
- Llama 3.1 8B — untied in HF (`tie_word_embeddings: false`); no change.
- Any change to quantization policies, `OperationTraits`, or CUDA kernels.
