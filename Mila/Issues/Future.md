# Future

Next-cycle work: real, and for a later release. Flat and coarse **by design** — detailed tasking
happens only when an item promotes into a release, and elaborating it here is work spent on a
plan that will be rewritten before it is used.

Moved out of `BACKLOG.md` so that file means exactly one thing: work committed to the release in
flight. Triage flow and categories are in [README.md](README.md).

---

- **[gate] One typed model handle + factory, before ANY next chassis** — the architecture-to-concrete
  erasure exists three times in two languages (Chat's `ModelVariant`, the binding's `*Session`
  classes, MIS's `ModelFamily`), which is why GPT-2 is missing from MIS. Lands in the runtime-adjacent
  native agent core; sequencing in `MilaProductFamily.md` Open Decision 2. **After the v0.20 tag,
  before the chassis expansion below.**
- **The library should own architectural identity** — the set of architectures is the set of model
  classes `Mila/Src` implements, held today as a compile-time type and an unvalidated manifest string
  with nothing connecting them, so each consumer writes its own bridge (`familyFromArchitecture` in
  `Chat.ModelCatalog.ixx:159`, `architecture == "gemma"` at `Mila_py.Wrappers.cpp:413`). Home is
  `Distribution`, beside the manifest reader, not `Dnn`. The library owns the identity only; traits
  merely keyed on it stay with the consumer they describe.
  [[project_architecture_identity_ownership]]
- **Qwen 3** — the dense decoder, thinking-mode suppression, model-agnostic tool calling, and FP8 KV
  cache; the `OperationTraits<GqaOp, Cuda, BF16, PerChannelKvFp8<>>` specialization lands here.
- **Qwen 3.8 FP4 — FP8 the embedding table so the model is wholly device-resident.**
  `QwenOraclePrecisionPlan::EmbeddingTable` is `NoWeightQuant` and host-resident, so every decode step
  gathers a row over PCIe; at `PerChannelFp8<>` the 1.271 B table halves to ~1.19 GiB and sits beside
  the 12.31 GiB FP4 body inside a 15.93 GiB card. Gemma 4's D4 Design B is the precedent and
  `TokenEmbedding` already accepts a per-channel table policy. `Qwen.PrecisionPlan.ixx:153`
- **Architecture / MoE** — the presumptive post-v0.20 tentpole; one router chassis unlocks Gemma
  26B-A4B, Qwen3-30B-A3B and gpt-oss-20b. [[project_moe_tentpole_direction]]
- **Gemma 4 MTP** — the self-speculative drafter, sequenced ahead of MoE.
- **Ministral** — SWA transformer; reuses the Llama foundation, Qwen 3 tool calling, and the Gemma 4
  SWA mask + bounded-KV ring.
- **v0.20 library-frozen tails** — the Generation API surface tail (`SamplerConfig` rename, Llama/Gpt
  seedable sampling, eager sampler, config-accessor propagation, `contextLength()` hoist), the
  Sample-API device-sampler migration for Llama/Gpt, and the Optimizer-dispatch migration onto
  `OperationTraits`. All `Mila/Src`, which is why they wait. Adaptor work does not.
- **Model serialization** — the remaining checkpoint round-trip and distribution phases. Design and
  phase plan in `Specifications/ModelSerialization.md`.
- **Retire quantize-on-load — one load shape for every policy.** `Linear::loadParameter` refuses a
  compute-precision blob, uploads packed bytes, binds, derives; the dtype sniff at `Linear.ixx:601`
  and `CudaLinearOp::quantize()` go, and FP8/FP4 fitting joins the sub-4-bit fitter in
  `Tools/Quantization` — one producer for every format, and model production stops needing a GPU. The
  codebook path is already this shape (`:574`). Depends on the FP4/FP8 codecs, and takes Chat's
  load-time quantization keyword with it. An API change to `Mila/Src`, which is why it waits.
  [[project_quantization_offline]]
- **Python binding — numeric access, not component access.** Add a session-level `forward()` returning
  logits, plus final hidden states, to `LlamaSession`/`GemmaSession`; from Python a parity run can
  compare token ids and nothing else today. Component, tensor and training bindings are ruled out:
  `TDeviceType x TPrecision x TWeightQuantization` is erased only at the session PIMPL.
  `Mila_py.Wrappers.ixx:362`
- **API Coherence** — the pre-1.0 consistency pass, and the precursor to any API-stability promise.
  First named item: **`loadModel`/`saveModel` and `loadCheckpoint`/`saveCheckpoint`** — verb plus what
  you get, both directions. "Pretrained" is relative to a fine-tuning stage Mila does not have and is
  doubly wrong on the write side; "artifact" is build vocabulary for a file that is simply a model;
  `from` names the *source* form, so `fromCheckpoint` earns it and `fromModel` cannot. Document the
  distinction: a checkpoint carries epoch and loss as one of a series, a model is terminal. One
  wrinkle: `Network::load( archive, mode )` restores into an existing graph. **The methods are the
  small half** — `kArtifactMinimumMilaVersion`, `ModelDistribution.md`, both model cards,
  `from_pretrained`, MIS and the samples all speak the old vocabulary. Sequence with the
  `ExportArtifact` rename and the binding's `quantize_fp8` fix.
- **Warnings-as-errors ratchet.** Requires the `/external:W0` isolation first; enforce in **CI only**,
  never locally; ratchet on the count *not increasing* before demanding zero; **MSVC first**, since
  `/WX` across three compilers means the union of three opinions must be zero; and dormant-but-retained
  code warns by nature — suppress per-file in CMake pointing at the owning task, never with
  `#pragma warning` in module code. Land it **after** v0.20 ships.
- **Training (advanced)** — Llama fine-tuning, loss-function GPU migration, gradient checkpointing,
  and BF16/GQA training.
- **Performance** — Gemma 4's levers (the fused W4A16 prefill GEMM, flash-attention on the global
  layers) and the codebook path's own, each a measured gap with its numbers in `Qwen3.8.md` §8: the
  decode GEMVs' bandwidth shortfall against FP4 (amortize the unpack across output rows, or bucket
  activations by code); staging the sub-4-bit prefill to FP8 so it reaches the sm89 tensor GEMM
  instead of a BF16 one, gated on e4m3's 3 mantissa bits over 2.82-bit codes; the per-chunk staging
  dequantize, unmeasured across the rung ladder (`Qwen.ixx:110`); tensor cores for the DeltaNet
  chunked kernel, worth ~13% of prefill; and whether Gemma's ring softmax is reachable from Qwen's 16
  full-attention layers. [[project_w4a16_prefill_gemm]]
- **Whole-model prefix caching for Qwen** — the 48 DeltaNet layers need the snapshot/restore copy and
  the 16 attention layers the positional rewind, and nothing combines them. Deferred as a policy
  question: how many prefixes to hold in host RAM, and eviction. `Qwen3.8.md` §8, `PromptCaching.md`
- **Native low-precision compute (Blackwell+)** — the microscaling data path and finer per-arch gating.
- **Compute backends beyond CUDA** — ROCm and Metal; `DeviceType::Rocm` / `::Metal` are reserved and
  unimplemented.
- **Platform portability — aarch64 + coherent memory.** Mila has never been built on ARM.
- **Model loading** — a load-time FP4 sidecar cache, and concurrent/async read I/O for real queue depth.
- **Ungated GPT-2 zero-auth quick-start** — a first-run HTTPS weights fetch.
- **`ComponentType` vitality** — does `getType()` earn its keep, or does the unused converter surface
  retire?
- **Discoverability** (internal, not a README theme) — the site is live at `mila.toddt.me`.
- **A value-reading observation sink has to name the model's compute precision.** The sink gets
  `const ITensor&`, whose `rawData()` is type-erased, so anything wanting numbers does a
  `dynamic_cast` to `Tensor<TPrecision, MR>` then its own `toHost`; all three consumers do this.
  Whether observation should offer a typed convenience is decided-deferred to v0.21 —
  `Observability.md` §11.2.
- **Remove FP16 (superseded by BF16) — measure first.** Woven through live code; trace
  live-vs-dead first, and 8 `REVIEW:` markers already scope it. Note the odd row it collides with:
  CUDA `LayerNormOp` is registered at FP32 and FP16 and *not* BF16, so deleting the FP16 row leaves
  CUDA LayerNorm FP32-only. Pinned by a `static_assert`, so this work must confront it.
