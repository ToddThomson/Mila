# Gemma 4 Modality

Image and audio input for Gemma 4: the 12B's image and audio paths, the 26B-A4B's vision tower, and how
an image or a clip reaches the model from Chat.

*Status: design, 2026-10-04; the mask (5.4) is built, the rest is not. Section 1 answers the two questions that
sized Gemma's image path, which closed their BACKLOG entry; the work is `BACKLOG.md`'s "Gemma 4 12B is multimodal
and Mila drops its image and audio weights" and "The 26B-A4B's vision tower is dropped when it loads". Chat accepts
images and audio (Todd, 2026-10-04).*

---

## 1. The Answers

Read 2026-10-04 from transformers 5.12.1 (`models/gemma4_unified/modular_gemma4_unified.py`,
`models/gemma4/modeling_gemma4.py`, `masking_utils.py`), the checkpoints' `config.json` and
`processor_config.json`, and the 12B's safetensors header.

**Positions: one per token, as for text.** An image's soft tokens take ordinary consecutive positions in
the sequence, and RoPE treats them as it treats text. The image's own two-dimensional layout is carried
inside the embedder, by a learned position table added before the projection into the decoder. There is
no multimodal RoPE.

**Attention: bidirectional within an image.** Both models set `use_bidirectional_attention: "vision"`.
Each contiguous run of image soft tokens attends to every token of that run, earlier or later; every
other token, the image's delimiters included, stays causal. The rule is an OR over the existing mask
(`blockwise_overlay`): a sliding layer keeps its window's lower bound and gains the span's upper bound.
Audio soft tokens are causal.

So the second question is no work at all, and the first is a mask change in every prefill attention path.
That change, not the embedders, is the size of the 12B's image path.

## 2. The 12B's Image Path

`Gemma4UnifiedForConditionalGeneration`: encoder-free. No vision tower; pixels are projected into the
decoder.

### 2.1 Preprocessing

From the processor (`Gemma4UnifiedImageProcessor`), defaults from `processor_config.json`:

1. **Resize**, keeping the aspect ratio, to the largest size whose sides are multiples of 48 and whose
   16-pixel patch count is at most `max_soft_tokens x 9` (`get_aspect_ratio_preserving_size`), bicubic
   with antialiasing (torchvision `resize`, `resample` 3). `max_soft_tokens` is 280 by default; the model
   accepts 70, 140, 280, 560 or 1120.
2. **Rescale** to [0, 1] (x 1/255). No mean or standard deviation: `do_normalize` is false.
3. **Patchify** into 16 x 16 patches with (x, y) grid positions, then **merge** each 3 x 3 group into one
   48 x 48 model patch of 6912 values, its position the group's grid position divided by 3
   (`patches_merge`).

An image becomes `n = (H / 48) x (W / 48)` soft tokens, **at most 280 and usually fewer**: the count
follows the resized shape.

### 2.2 The embedder

`model.vision_embedder.*` and `model.embed_vision.*`, BF16:

```
patch [6912] -> LayerNorm (patch_ln1) -> Linear 6912->3840 + bias (patch_dense) -> LayerNorm (patch_ln2)
  + pos_embedding[x, 0] + pos_embedding[y, 1]           pos_embedding [1120, 2, 3840]
  -> LayerNorm (pos_norm) -> RMSNorm, no scale -> Linear 3840->3840 (embed_vision.embedding_projection)
```

About 50M parameters, 100 MB at BF16. The output replaces the token embedding at each soft-token
position **after** the embedding's sqrt(3840) scale: soft tokens are not scaled.

### 2.3 In the prompt

The chat template emits one `<|image|>` per image; the processor expands it to
`<|image>` + n x `<|image|>` + `<image|>` (ids 255999, 258880, 258882), n known only after the resize.
The soft tokens are the `<|image|>` run; the delimiters are text.

## 3. The 12B's Audio Path

Not an encoder. `model.embed_audio.embedding_projection` [3840, 640] is the only audio tensor:

```
16 kHz mono samples, zero-padded to a multiple of 640, cut into 640-sample frames (40 ms)
  -> RMSNorm per frame, no scale -> Linear 640->3840
```

One soft token per frame, no downsampling, 750 frames (30 s) per clip
(`Gemma4UnifiedAudioFeatureExtractor`, `processor_config.json`). The prompt form is `<|audio>` +
n x `<|audio|>` + `<audio|>` (ids 256000, 258881, 258883). About 2.5M parameters.

## 4. The 26B-A4B's Vision Tower

`Gemma4ForConditionalGeneration`; no audio. Read at outline depth on 2026-10-04 (`Gemma4VisionModel`); the
tower's layers are read in full when it is built.

- The same preprocessing, without the 3 x 3 merge: 16-pixel patches, `2 x (p - 0.5)` on the pixels.
- `patch_embedder`: Linear 768->1152 plus a learned (x, y) table, [2, 10240, 1152].
- 27 encoder layers, width 1152, 16 heads of 72, GeGLU 4304, two-dimensional RoPE (theta 100),
  attending bidirectionally over one image's patches.
- Pooler: 3 x 3 average by position to n soft tokens (280 by default), x sqrt(1152), then standardize
  (`std_bias`, `std_scale`).
- `embed_vision`: RMSNorm without scale, Linear 1152->2816.

About 411 million parameters (`Gemma4MoE.md`). Placement, prompt and mask are the 12B's; the tower is the
new piece, and it runs at prefill only.

## 5. Design

### 5.1 Where each step lives

- **The application decodes files.** A PNG or JPEG becomes RGB pixels, a WAV becomes 16 kHz mono samples,
  in the application: Chat in C++, the inference server in Python. A vendored decoder is a `NOTICE.md`
  entry.
- **The library does everything the model defines.** Resize, patchify, merge, framing, the embedder, the
  placement and the mask are the model's, so they live in `Mila/Src` beside it, and every application
  gets the same result from the same pixels. The boundary type is a decoded image (width, height, RGB8)
  or a clip (16 kHz float samples).

### 5.2 The prompt

Generation takes the prompt's tokens and the media it refers to: an ordered list of images and clips,
one per placeholder. The family's protocol (`Gemma.Protocol.ixx`) expands each placeholder to its
delimited run once the item's n is known. A prompt with no media is today's path, byte for byte.

**Prefix reuse keys on content, not token ids.** Every image's run is the same `<|image|>` id repeated, so
`GemmaModel`'s token-equality test (`kv_token_history_`) would call two different images equal. Reuse
compares each media item's identity (a hash of its decoded content) as well as the tokens, and a rewind
that lands inside a run is refused: the positions before it attended to the positions after it.

### 5.3 Components

Built or not, per deployment (`Deployment.md` 2.1, `ModelHandle.md` 3.8); a modality not selected
allocates nothing, and its weights stay in the file.

- `GemmaImageEmbedder` (2.2): `LayerNorm` exists; the factorized position lookup and the patch preparation
  are new.
- `MultimodalEmbedder`: RMSNorm without scale, then a `Linear`. The 12B's image and audio paths and the
  26B's tower all end in it.
- Soft-token placement: overwrite rows of the token embedding's output with the embedder's, before
  layer 0.
- `GemmaVisionTower` (section 4), for the 26B.

### 5.4 The mask

Every prefill attention path takes, per query row, a key upper bound: the row itself for text, the end
of its run for an image soft token. The bound is non-decreasing across a chunk, so the flash kernels'
tile skipping generalizes rather than breaking. Gemma's prefill reaches two kernels, each with a BF16 and
an FP8 cache: `Gqa.Flash.Packed` (sliding layers, head 256, the ring) and `Gqa.Flash.WideHead` (global
layers, head 512). Those carry the bound; every other prefill path -- the FA-2 ring kernel, the cuBLASLt
softmax path, FP32 -- refuses a prompt with an image run rather than attending it causally. GQA has no CPU
path. Decode is untouched: images arrive only in prefill.

**A run never crosses a prefill chunk.** A row can attend a later row only if that row's keys exist when
it runs. The chunker ends a chunk before a run that would not fit and starts the next one at it; a run
longer than the chunk is refused with the chunk it needs. At 280 tokens against chunks of hundreds or
more, this changes where a chunk ends, not how many there are. The sliding ring holds
`window + chunk - 1` rows, so a run inside one chunk is always resident for its own rows.

**A run is no longer than the sliding window.** HuggingFace's mask is the sliding mask OR the run: a run
longer than the window would let its last rows see run tokens below their window. The kernels keep each
row's window lower bound, which equals that OR exactly when the run fits the window (1024 on Gemma 4,
against 280 tokens), so a longer run is refused rather than masked differently.

**The bounds travel on the execution context**, as the decode position does (Todd, 2026-10-04): the
network sets each chunk's per-row key bounds with `setPrefillKeyBounds( position_offset, bounds )`, which
checks them -- non-decreasing, each at least its own position, none past the chunk -- and an attention op
asks for the bounds of its own chunk, getting none when none are set and a refusal when they were set for
another chunk.

**Built 2026-10-04** (`0.21.0-dev+35`): the bounds on the context, both kernels, and the op's refusals, gated by
`CudaGqaOp.KeyBounds.Cuda.cpp` against exact attention. A prompt without an image prefills as before -- measured on
the 12B Q4_0 through ProfileModel against a build of `+34`, alternating two rounds of five runs: 8K +0.45%, 32K
-0.27%, inside the 1.5% the baseline itself moved between rounds (RTX 5060 Ti).

### 5.5 Weights

The converter keeps `model.vision_embedder.`, `model.embed_vision.` and `model.embed_audio.`
(`convert_weights.py` skips them today), at BF16. The manifest declares the modalities the package
carries, and the handle reads them from there rather than from the family's name (`BACKLOG.md`, Model
Handle). Adding tensors to a package leaves it readable by the previous minor (`ModelDistribution.md`
*Compatibility*). The 12B's one republish carries them (`ModelFamilyParity.md` G6).

### 5.6 Footprint

The planner prices a selected modality like any feature: its weights, the patch staging (one image is
`n x 6912` values), and the embedder's scratch at its largest image. `PlanEqualsBuild` holds a selection.
The 26B's tower also prices its encoder's activations over 2,520 patches.

### 5.7 Applications

- **Chat** accepts an image or an audio clip and attaches it to the next message: `/image <path>` and
  `/audio <path>` (PNG and JPEG; WAV at any rate, mixed to mono and resampled to 16 kHz in Chat).
  For a model whose manifest declares no such modality, Chat refuses the attachment by name. Chat renders
  Gemma's prompt through the library's template first (`BACKLOG.md`, "Chat renders Gemma's prompt itself"),
  since that template is what expands the runs.
- **The inference server** accepts image and audio content blocks in both wire protocols.
- **The Python binding** passes media with the prompt.
- All three reach the model through `Mila::AI`, not around it.

## 6. Gates

- **Embedder parity:** the image and audio embedders' outputs against HuggingFace's on the same
  preprocessed tensors (`pixel_values`, `image_position_ids`; the framed samples).
- **The mask:** a run's first row changes when its last row's input changes, a text row before the run
  does not, and a run placed across a natural chunk boundary gives the output it gives inside one chunk.
- **The model:** token for token against HuggingFace on an image prompt and on an audio prompt, fed
  HuggingFace's preprocessed tensors, by the oracle the text path uses.
- **Preprocessing:** Mila's resize against torchvision's on the same images, its difference measured and
  recorded. A bicubic resize is not bit-reproducible across libraries, so it is gated by its effect: the
  answer to an image prompt with Mila's pixels against the answer with HuggingFace's.
- **Prefix reuse:** two prompts that differ only in their image never reuse each other's cache.
- **The applications:** Chat answers about an attached image and an attached clip; the inference server
  and the binding do the same.
- **The 26B's tower:** tower parity against HuggingFace, then an image prompt token for token, at a
  context the plan states.

## 7. Phasing

1. **The mask**, across every prefill path, on synthetic runs; no weights involved.
2. **The 12B's image path:** converter, embedder, placement, the prompt with media, prefix reuse by
   content. Parity, then token for token.
3. **Audio:** a framing and one `MultimodalEmbedder`.
4. **The applications:** Chat's `/image` and `/audio`, the inference server, the binding.
5. **The 26B's tower**, first to drain if the date presses (`ROADMAP.md`).

## 8. Decisions

Decided 2026-10-04 (Todd, as recommended):

1. **The soft-token budget** is the processor's 280 to start. A per-image choice from 70 to 1120 is
   added only when a use case needs it; more tokens for detail cost context, and a run above the chunk
   is refused (5.4).
2. **The resize** reproduces torchvision's antialiased bicubic, the reference's own; the preprocessing gate
   (section 6) measures what remains.
3. **Chat's image decoder** is a vendored single-file decoder (`stb_image`), its licence read at the source
   and recorded in `NOTICE.md`, rather than a system library a user would have to install.

## 9. Non-Goals

- Video. The 12B accepts it (frames at 70 soft tokens, with timestamps); not in this release.
- Audio on the 26B-A4B, which has none.
- The E2B and E4B models' audio and vision encoders.
- Generating images or audio.
