# Mila — Roadmap

Where Mila is going — the durable narrative of each release and what it means.

- **Open tasks** -> [BACKLOG.md](BACKLOG.md) · **Completed work** -> the
  [release notes](https://github.com/ToddThomson/Mila/releases)
- **How versions, branches, and releases work** -> [RELEASING.md](RELEASING.md)
- **Design rationale** -> `Mila/Specifications/`

The roadmap shows the **release in flight**, plus a **Future** tail. A release is reached through the
**themed workstreams** below; their tasks live in [BACKLOG.md](BACKLOG.md).

---

## v0.21.0 — The Families, Complete

**Release Date:** _Target — 2026-10-31_

**The release makes one claim, and it is a pair.** The model families Mila ships are *finished* —
nothing about Qwen 3.8 or Gemma 4 is half-reached — **and** adding to them is an edit in one place.
Neither half stands alone. Completion without the handle is four bridges to maintain instead of one,
and every future family pays that tax again. The handle without completion is an abstraction nobody
has stressed, which is the kind that holds until the first real load.

Gemma 4 gaining images is what proves both at once. It is a new modality arriving in a family that
already exists — exactly the case the handle claims to make cheap, and exactly the case that is
ruinous without it.

**The date is fixed and the scope drains.** This is an ambitious arc for the time, deliberately so.
What has not landed by 2026-10-31 returns to [`Vnext.md`](Mila/Issues/Vnext.md) and the release ships
with what it has; the date does not move to accommodate the backlog. A theme that loses items still
ships its narrowed claim, stated as narrowed — the same rule that made v0.20 withdraw its 2.82-bit
headline rather than soften it.

**The arc runs in three stages, in dependency order.**

1. **The handle**, because everything after it is cheaper on the other side. A modality declared in a
   manifest is one edit once the factory reads the record, and three edits while `ModelVariant`, the
   binding's session classes and the inference server's family enum each infer behaviour from a
   family name.
2. **Qwen 3.8**, the smaller of the two families and the one already closest to done. It also
   settles the handle on a family with no new surface, before a second modality arrives to test it.
3. **Gemma 4**, ending in modality — the largest single item in the release and the one whose size
   is not yet known, which is why the design questions in its own theme are answered first rather
   than discovered during implementation.

Pre-1.0 still holds: breaking changes are acceptable, and this release carries several.

### Model Handle

*One place that turns a model's name into the type that loads it, reading what the model declares
rather than inferring it from the family.*

It starts at the manifest. A model's capabilities — reasoning channel, streaming, context limits,
and now modality — are declared in its own record, which is additive rather than breaking because
the manifest already tolerates unknown fields and `instruct` proves the pattern. Today
`Chat.FamilyTraits.ixx` derives those same facts from `family == Gemma`, which is correct only while
the set of families is the set it was written against.

The factory then reads the record. A family enum remains, because a model still resolves to a
concrete compile-time type and that is the point of Mila; what goes is every *second* place that
re-derives a model's behaviour from which family it belongs to.

**The thesis is not what is being traded away.** The erasure is one call when a session opens, not
one per layer or per token. Everything inside the forward pass stays exactly as explicit as it is
now — a handle that hid the forward pass would be a wrapper, and the previous release spent itself
proving Mila is not one.

**Success criteria:** a new architecture is added in one place; no dispatch site outside the handle
carries a per-family branch; a model's declared capabilities come from its manifest, and a model
whose manifest omits a field loads rather than failing; and the erasure costs no measurable decode
throughput, since it happens once per session and not once per token.

### Adaptor Convergence

*Chat, the inference server and the Python binding consume the handle instead of each maintaining a
copy of it.*

The handle is only demonstrated when the three bridges it replaces are actually gone. This is the
half that makes the claim checkable from outside: the server refusing a model the chat harness runs
is the visible symptom, and it disappears or the handle did not land.

Both adaptors stay open to change throughout, on the standing rule that an adaptor adds no capability
to the library — where one cannot reach something Mila already does, that gap is the work rather than
a feature request.

**Success criteria:** the inference server serves every architecture the chat harness does, GPT-2
included, and its family enum is gone; the binding's per-family session classes are replaced by the
handle at session depth, with no component-level surface added; and the foreign-harness tool flows
established in v0.20 — Codex CLI and Claude Code CLI over the OpenAI and Anthropic wire shapes —
still pass unchanged.

### Qwen 3.8 Complete

*Everything the family was specced to be, including the parts v0.20 shipped around.*

v0.20 promoted Qwen 3.8 out of a research track and into the release, and the promotion was earned —
but it shipped the model rather than the family. Three things were left: the 2.82-bit build is
finished, validated and **unpublished**, so the claim the release was written around had to be
narrowed to the FP4 build at 16 GB; the KV cache is uncompressed on a card the weights already fill,
which is what bounds the context the model is actually sold at; and the dense members were never
built, although they reuse Llama blocks rather than the DeltaNet chassis and are cheap.

The user-visible one is smaller and worse: a 27B answers in a single block after a long silence,
because the per-token router was written against Gemma's four control tokens and Qwen's
`<think>`/`</think>` pair was never wired to it. It is the longest wait in Mila with nothing on
screen.

**Publishing the 2.82-bit build is not release work** — weights publish on their own schedule and a
release never re-publishes them — but the claim below depends on it, which is why it is named here.

**Success criteria:** a 27B model runs on a 12 GB card from a package a stranger can fetch;
FP8 KV cache compression measured against BF16 at the context the model is claimed for, by the same
protocol the weight allocation used, and the freed margin spent deliberately rather than absorbed;
the quality gate re-run at whatever context the release ends up advertising rather than stopping at
16K; a dense member decodes token-for-token against HuggingFace at BF16 and FP8; and Qwen streams
its reasoning and its answer as separate channels.

### Gemma 4 Complete

*The family Mila validates most closely, finished — including the half of the checkpoint the loader
currently discards.*

Gemma 4 is the family with the most Mila behind it and the most left on the table. The 26B-A4B
mixture of experts **already works** — router, expert bank, FP4 expert bank and streaming converter,
gated against HuggingFace at BF16 and FP4 — and appears in no capability row because no published
package uses it. Two protocol defects survive into multi-step tool use: a model working through a
sequence loses the reasoning that led to each step, and a malformed call parses as a call with no
arguments rather than as prose, which a client that executes `{}` would act on.

**The headline is modality, and it is the largest single item in the release.** The 12B is
encoder-free — no vision tower — so images enter through `vision_embedder` and an embedding
projection directly into the decoder at 280 soft tokens, and audio through `embed_audio`. The
converter skips both today. That makes this materially smaller than the vision-language work in
**Future** below, and it is also the thing that de-risks it: patch embedding, soft-token placement,
image templates, the adaptor surface on both wire protocols, manifest modality and footprint
accounting all carry over, leaving a tower as the only genuinely new piece. It also runs on a 12 GB
card, so unlike that work it is not gated on hardware.

**Its size is not yet known, and that is stated rather than estimated.** Two questions decide it:
whether image soft tokens must attend bidirectionally within a prefill, when every attention path in
Mila is causal, and the position scheme for image spans. Both are answered before anything is built.

Two further items are gated on a measurement rather than committed on intent, and each carries its
own stop condition: Google's quantization-aware 4-bit build is only worth loading if it beats Mila's
FP4, and its published drafter is only worth a speculative loop if a K-token verify costs
meaningfully less than K decodes on a bandwidth-bound FP4 path.

**Success criteria:** Gemma 4 12B accepts an image and answers about it, with the embedder gated
against HuggingFace and then token-for-token on a mixed prompt; modality is declared in the manifest
and read by the handle rather than by a family test; the 26B-A4B is fetchable and named in a
capability row; reasoning survives across tool calls within a turn and a malformed call is refused
rather than executed as empty; and each measurement-gated item has a recorded result, including the
result "not worth doing".
---

## Future

Uncommitted work — no release, no date. An item **promotes** into the Current release, acquiring its
own version, date, and tag, when it is scheduled.

- **Muse Glimmer 30B — the named model target, and not the next release.** Meta's Apache 2.0,
  ungated 30B, chosen for *why it exists* rather than for what it resembles: it is tuned for tool
  use, long tasks, and failure recovery, which is the model an on-device agentic loop actually needs.
  It arrives after the `Direction.md` section 5 slices, not among them, because those add no chassis
  and this is a second architecture. The
  [product definition](Mila/Specifications/MilaProductFamily.md) already reserves the **Agentic
  adaptor** — the loop closing on itself, on-device — as the post-release member of the family, and
  this is the model that makes it real rather than aspirational.
  Its text tower is close enough to Gemma 4 that the chassis carries over: a repeating
  local/local/local/global attention pattern, final logit softcapping, GQA, RMSNorm with a post-norm,
  a SiLU-gated FFN, a bounded sliding-window ring, and per-layer RoPE. It is dense, so MoE is not a
  prerequisite. Three details are new and each is silent if assumed away — a non-standard QK scale, an
  output multiplier, and RoPE disabled on the global layers rather than merely retuned.
  **The real work is that it is a vision-language model.** A fifty-layer ViT with window attention, 2D
  position embeddings and its own RoPE feeds a projector into the text model. That used to mean a
  second architecture from nothing, which is what made this a tentpole rather than a chassis
  extension. **v0.21.0's Gemma 4 modality work changes the sum**: Gemma 4 12B is encoder-free, so
  building its image path delivers patch embedding, soft-token placement, image templates, the
  adaptor surface on both wire protocols, manifest modality and footprint accounting — leaving the
  tower itself as the only genuinely new piece here.
  **The binding constraint is hardware, not code.** Around 31B parameters is roughly 16 GB at FP4
  before any KV cache, against 12 GB on the card every current Mila claim is validated on. Mila's bar
  is token-for-token agreement with the HuggingFace reference, and that cannot be established on
  hardware which cannot load the model — so this target and the compute ask in
  [SPONSORING.md](SPONSORING.md) are one decision, not two.
  Success bar: greedy text decode matches the reference token-for-token; image-conditioned generation
  validated against the same oracle; tool calling driven end-to-end through MIS; and the Agentic
  adaptor closing a multi-step task on-device.
- **v0.20 library-frozen tails** — the Generation API surface tail (SamplerConfig rename, Llama/Gpt
  seedable sampling, eager sampler, accessor propagation), the Sample-API device-sampler migration for
  Llama/Gpt, a second module-compiler oracle (GCC 16) with a broadened Linux compiler matrix, and the
  ungated GPT-2 zero-auth quick-start (a first-run HTTPS weights fetch). These are library-side, which
  is why they wait; adaptor work does not.
- **Ministral** — Ministral transformer with Sliding Window Attention; 3B Instruct (BF16) and 8B
  Instruct (FP8). Builds on the Llama foundation and the Qwen 3 tool-calling pipeline, reusing the SWA
  mask + bounded-KV ring cache from Gemma 4.
- **Training (advanced)** — the second training release, and large enough to be one: **BF16
  training** and **GQA training**, plus a full LLaMA fine-tuning pipeline, loss-function GPU
  migration, gradient checkpointing, and checkpoint save/restore. Sized honestly, because v0.20's
  training scope was narrowed on the evidence that this half was further out than it looked: GQA
  backward does not exist (`CudaGqaOp::backward` throws, which is what the one deliberate compiler
  warning reports), the loss path is still host-side in both samples, and BF16 needs gradient checks
  at its own tolerance. The BF16 optimizer machinery is in the tree and guarded by
  `AdamW.MixedPrecision.Cuda.cpp` — dormant and tested, in the same spirit as the GQA
  expanded-layout substrate — so this release starts from working parts rather than from repair.
  **Sequencing:** the releases after v0.20 work through `Direction.md` section 5, of which v0.21.0
  is the first slice; Muse Glimmer follows those. Training (advanced), Qwen 3 and MoE all come after
  it rather than compete with it — MoE in particular is no longer a prerequisite for anything on the
  critical path, since the Muse Glimmer decoder is dense.
- **Architecture** — additional attention variants. The Mixture-of-Experts components this entry used
  to describe — the `GatedMLP` reusable gated FFN, the grouped `MoeOp`, `Router` and
  `MixtureOfExperts`, specified in `Specifications/FfnAndMoE.md` — **landed during v0.20's `rc.1`**
  and are validated against HuggingFace on the Gemma 4 26B-A4B; what remained was a published package,
  which v0.21.0 takes. Speculative decoding moved with it, since Google's published drafter is what
  makes it concrete rather than general.
- **Performance** — Gemma 4 prefill/decode competitiveness levers (the fused W4A16 prefill GEMM, the
  flash-attention global prefill kernel, the FP4 decode-matvec bandwidth campaign), the codebook path's
  own two (staging to FP8 so the sub-4-bit projections reach the same tensor-core GEMM the FP4 path
  already uses, and closing the codebook GEMV's bandwidth gap), tensor parallelism, and deterministic
  gradient accumulation. Each of these is a measured gap rather than a suspicion; the numbers behind
  them live in `Specifications/Qwen3.8.md` and BACKLOG.
- **Native low-precision compute (Blackwell+)** — microscaling data-path support, finer per-arch gating
  (sm_120, CUTLASS 4.x), and the "compute precision as a first-class axis" design question.
- **Compute backends beyond CUDA** — ROCm (AMD) and Metal (Apple silicon) device backends. Both are
  reserved in `DeviceType` and neither is implemented; Mila is CUDA and CPU today. Because the device
  type is a compile-time template parameter and dispatch resolves through an explicit `OperationTraits`
  table, a backend should be a new partition of specializations rather than conditional compilation
  threaded through the components — that is the design claim, and a port is the first honest test of
  whether it holds across a second GPU vendor. Gated on hardware access (see SPONSORING.md).
  Success bar: an existing validated model path reproduces its token-for-token reference result on the
  new backend, with the component sources unchanged.
- **Platform portability** — `aarch64` as a build and correctness target (Mila is x86-64 Windows and
  Linux today), broadening the Linux compiler matrix, and a third compute-capability gate alongside
  sm_89 and sm_120. Grace-Blackwell-class hardware also puts a coherent unified-memory model in front
  of memory resources and the weight loader, both of which assume discrete device VRAM with explicit
  host-to-device staging — a design question, not only a port.
- **Model loading** — the load-time FP4 sidecar cache and concurrent read I/O.
