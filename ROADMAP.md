# Mila — Roadmap

Milestone vision and success criteria. This file is the durable narrative of where Mila is
going and what each release means — not a task tracker.

- **Open tasks** live in [BACKLOG.md](BACKLOG.md).
- **Completed, validated work** lives in [CHANGELOG.md](CHANGELOG.md).
- **Design rationale** lives under `Mila/Specifications/`.

---

## Versioning

Mila uses a repeating release-cycle model. Field conventions: **minor** = feature-set era ·
**patch** = running build counter · **pre-release tag** = maturity stage
(`alpha.N` -> `beta.N` -> `rc.N` -> unsuffixed stable). Each feature set opens a new minor and
runs its own ladder; features are never added inside a hardening ladder — a stabilizing release
only takes patch-level fixes. Mila is pre-1.0, so any release may carry breaking changes:
`0.20.0` "production" means validated and polished, **not** API-frozen. An API-stability promise
is a separate, deliberate `1.0.0` decision, intentionally deferred (for a mastery-first project,
possibly far off by design).

The minor jumps `0.13` -> `0.20` to mark the production tier; this is forward in semver
(`0.20.0 > 0.13.x`, minor compared before patch), avoiding the version *decrease* that `0.2.x`
would have caused.

| Cycle | Version | Title |
|---|---|---|
| Current | `0.13.x-alpha.6` | Feature freeze, FIXME/TODO burndown, debug-strip — earn the right to call it beta |
| First production | `0.20.x-beta.1` -> `-rc.1` -> `0.20.0` | Public release — packaging, docs, contributor onboarding |
| Feature cycle | `0.21.0` (own alpha->beta ladder) | Qwen 3 architecture + thinking mode — Qwen 3 8B Instruct |
| Feature cycle | `0.22.0` (own alpha->beta ladder) | Ministral architecture + SWA — Ministral 3B and 8B Instruct |

---

## Alpha.6 — Consolidation (Feature Freeze) — Current

**The last alpha milestone: deliver the final feature, then stop — burn down the debt that
would otherwise make the public release embarrassing to open to contributors.**

Alpha.6 is the bridge between "the features work" and "the tree is honest enough to call it
beta." It is not a feature milestone in the additive sense; its one feature delivery is token
sampling (the last open capability — greedy-only is too thin a story to ship publicly), after
which the feature set for the first production release is frozen. The rest is consolidation:

- **Token sampling** — the final feature (temperature / top-k / top-p); see [BACKLOG.md](BACKLOG.md).
- **FIXME/TODO debt burndown** — triage the source markers: *disguised features* (commented-out
  core paths) are functional gaps that must be finished, because you cannot honestly feature-freeze
  around commented-out code; *known-limitation* markers are fixed or demoted to tracked BACKLOG
  tasks (never shipped as a literal `FIXME` in public source); `REVIEW` design notes are
  low-urgency, addressed opportunistically. Live triage status and counts are in [BACKLOG.md](BACKLOG.md).
- **Debug-instrumentation strip** — the `std::cout` / `std::cerr` / `printf` usage; same trust concern.

**Why a distinct phase.** "Beta" is a public trust signal. Applying the beta label to a tree
full of "this is broken" markers and commented-out core paths overpromises. Alpha.6 is the work
that *earns the right* to the beta tag, so the `0.20.x-beta.1` that follows is honest.

**Exit:** feature set frozen, no literal `FIXME` in public source, debug instrumentation gone.
The version then jumps from the `0.13.x` alpha line to the `0.20.x` first-production line.

---

## 0.20 — First Production Release

**The first public, contributor-ready release. Hardening ladder: `0.20.x-beta.1` -> (optional)
`-rc.1` -> `0.20.0` (production-tagged).**

This is the trust-establishing milestone — the first release Mila would stake its name on. It
adds no new model features beyond the Alpha.6-frozen set; it is reached when GPT-2 and Llama
inference are validated across FP32, BF16, FP8, and FP4, tool calling is validated on Llama
3.2 3B and 3.1 8B Instruct, and the library is stable, documented, and packaged well enough for
external contributors to work with confidently. "Production" means validated and polished, not
API-frozen — Mila stays pre-1.0 and breaking changes remain allowed (see Versioning). The
requirement table is the gate list to reach `0.20.0`; the `-beta` / `-rc` stages are the
hardening iterations along the way, each a tagged release.

| Item | Required |
|---|---|
| Llama 3.2 1B FP32 validated against HuggingFace | Yes |
| Llama 3.2 3B BF16 validated against HuggingFace | Yes |
| Llama 3.1 8B Instruct FP8 validated against BF16 baseline | Yes |
| Tool calling validated on Llama 3.2 3B and 3.1 8B Instruct | Yes |
| API documentation complete and published | Yes |
| CPU reference ops — GPT-2 (Alpha.1) lineage only; Llama/Qwen/Ministral CPU ops are contributor-driven, not a gate | Scoped |
| Debug instrumentation fully gated or removed | Yes |
| Test coverage of core components | Yes |
| CONTRIBUTING.md with coding standards | Yes |
| getting-started.md onboarding guide (user-first, contributor superset) | Yes |
| `find_package(Mila)` packaging validated by an external consumer build | Yes |
| Published Docker runtime image (slim multi-stage GPU runtime, release-tagged) | Yes |
| Ungated GPT-2 quick-start path for zero-auth first run | Yes |
| good-first-issue labels on GitHub | Yes |

The engineering work behind these requirements — packaging restructuring, module-hygiene
sweeps, public-API narrowing, release/CI, and project-hygiene tasks — is tracked in
[BACKLOG.md](BACKLOG.md). The themes that define *what this release means* follow below.

**Compute backend scope.** Mila is GPU-first. The CUDA backend is the validated, supported
inference path; correctness is established token-for-token against HuggingFace, which serves as
the reference oracle (the original llm.c-derived CPU path is no longer needed for that role). The
CPU backend is retained as the always-available baseline and contributor on-ramp, but full CPU
op parity across architectures is explicitly **not** a beta gate. GPT-2 (the Alpha.1 / llm.c
lineage) keeps its CPU reference; Llama, Qwen, and Ministral are CUDA-first, and their CPU ops are
filled in by contributors as demand arises. This is safe by construction: compile-time
`OperationTraits` dispatch means a missing CPU op costs nothing on the GPU path and produces a
localized compile error — never a silent wrong answer — if a CPU model is instantiated without it.
So the gaps are intentional good-first-issue work rather than hidden incompleteness, and the
user-facing model paths stay GPU so an end user never hits an unimplemented CPU op.

CPU op coverage by component (the CUDA backend implements every row). Legend: ✅ wired via
`OperationTraits`; ◐ implemented but still on the legacy registry/typemap dispatch; — not
implemented (contributor opportunity).

| Component | Lineage | CPU | Notes |
|---|---|---|---|
| GELU | GPT-2 / shared | ✅ | |
| Residual | GPT-2 / shared | ✅ | |
| Softmax | GPT-2 / shared | ✅ | |
| LayerNorm | GPT-2 | ✅ | migrated to `OperationTraits` (Alpha.5) |
| Linear | GPT-2 / shared | ◐ | `CpuLinearOp` exists; traits specialization is the last typemap holdout (BACKLOG) |
| MultiHeadAttention | GPT-2 | ✅ | |
| LPE (learned positional) | GPT-2 | ✅ | `CpuEncoderOp` |
| AdamW (optimizer) | shared | ✅ | `CpuAdamWOptimizer` |
| SoftmaxCrossEntropy (loss) | training | — | `CpuSoftmaxCrossEntropyOp` exists but is not built/wired |
| TokenEmbedding | Llama | — | GPT-2 embedding is folded into `CpuEncoderOp` |
| RoPE | Llama | — | good-first-issue |
| RmsNorm | Llama | — | good-first-issue |
| SwiGLU (SiLU) | Llama | — | good-first-issue |
| GroupedQueryAttention | Llama | — | good-first-issue |

Net: the GPT-2 lineage runs end-to-end on CPU; the Llama lineage is CUDA-only until a contributor
adds the five Llama CPU ops above.

**Distribution.** Beta introduces a published Docker image so users can run Mila without
standing up the bleeding-edge build toolchain. Two images, two roles: a slim multi-stage
**runtime** image (built in the CUDA `-devel` base, artifacts copied into a `-runtime`
base) for users, and the existing `-devel` **dev container** for contributors. Images are
published on release tags (not a rolling `:latest`) to keep maintenance bounded for a solo
maintainer. Gated model weights are never baked into the image — they remain a user-supplied,
offline conversion step mounted in at run time; the ungated GPT-2 path provides an
out-of-the-box first run with no HuggingFace auth.

Pre-converted Mila-format weights for permissively-licensed models (GPT-2 first; Qwen and
Mistral as they land) are hosted in a public Hugging Face repository and fetched on first
run via direct `resolve/` URLs over HTTPS — no Python, venv, or HuggingFace auth at runtime.
The weight blob carries a format magic and `version`, so hosted artifacts are versioned
against the Mila format and an incompatible build fails loudly rather than mis-loading; the
writer (`common.py`) and reader `VERSION` constants must be kept in sync and a re-publish is
required on any format bump. Llama and other gated weights stay a user-supplied offline
conversion step — redistributing them would transfer Meta's license obligations onto the
project.

---

## 0.21 — Qwen 3 — Planned (next feature cycle)

**Qwen 3 transformer architecture with thinking mode and model-agnostic tool calling,
validated on Qwen 3 8B Instruct at BF16 and FP8. FP8 KV cache compression introduced
and validated on Qwen 3 8B.**

A fresh cycle on top of the `0.20.0` production line, running its own `alpha -> beta` ladder
(rc optional). The `0.21` cycle adds Qwen 3 as Mila's second supported architecture family. The Qwen 3 dense
decoder shares Mila's existing building blocks — RMSNorm, SwiGLU, GQA, RoPE — so the
model component is a thin addition on the established Llama foundation. The primary new
work is in the Chat layer: the ChatML prompt template, model-agnostic `ToolCallParser`,
and thinking mode token suppression.

The FP8 quantization infrastructure delivered in Alpha.5 is exercised on Qwen 3 8B,
providing a second independent architecture validation at a scale where VRAM constraints
are meaningful (Qwen 3 8B at FP8 targets ~9–10 GB, within the RTX 4070 12 GB budget). FP8
KV cache compression is introduced as a symmetric K/V policy (`PerChannelKvFp8<>`); the
`KvCachePolicy` extension point and policy struct are already in place from Alpha.5. Qwen 3
8B is the appropriate validation target — at this scale and context length the KV cache is
large enough for compression to be practically meaningful.

**Success criterion:** Greedy decode of Qwen 3 8B Instruct at BF16 and FP8 each match
HuggingFace token-for-token on identical prompts. Tool calling validated end-to-end
using the model-agnostic pipeline. Thinking mode token suppression confirmed in the
Chat CLI. FP8 KV cache compression produces acceptable output quality degradation
relative to the BF16 baseline on Qwen 3 8B.

Phases: (1) Qwen 3 transformer component — `Qwen3Config`/`Qwen3Transformer`/`Qwen3Model`/
`Qwen3.Presets.ixx`, reusing `LlamaBlock` unchanged. (2) Qwen 3 tokenizer
(`BpeTokenizer::loadQwen3`, vocab 151936, ChatML + thinking + tool-call special tokens) and
`convert_qwen3_weights.py`. (3) Model-agnostic tool calling — `ToolCallParser` pluggable
boundary strategies (`Llama32Strategy`/`Qwen3Strategy`), `Qwen3ChatTemplate`, stop on
`<|im_end|>`, `ThinkingFilter` for `<think>`…`</think>` suppression. (4) FP8 KV cache
compression in `CudaGqaOp` (per-head per-token symmetric K/V; `if constexpr (TKvPolicy::
kIsActive)`; scale tensors in `GroupedQueryAttention::build()`; prefill + decode write
kernels and read dequant). Per-phase tasks live in [BACKLOG.md](BACKLOG.md) once this
milestone is active.

---

## 0.22 — Ministral — Planned (feature cycle)

**Ministral transformer architecture with Sliding Window Attention, validated on Ministral
3B Instruct at BF16 and Ministral 8B Instruct at FP8.**

The `0.22` cycle, again its own `alpha -> beta` ladder on top of `0.21`. It introduces the
Ministral transformer as a new first-class component built on the Llama 3.2 foundation. The primary architectural addition is Sliding Window Attention (SWA),
used on interleaved layers in the Ministral 8B model. The FP8 quantization infrastructure
delivered in Alpha.5 is applied directly to Ministral 8B, bringing it within the 12 GB
VRAM budget of consumer Ada Lovelace GPUs (validated at context_length 2048: ~10.2 GB
total on an RTX 4070).

Ministral 3B has no SWA and uses standard global GQA, making it a clean BF16 validation
gate before the combined SWA + FP8 work is exercised on the 8B model. The model-agnostic
tool calling pipeline delivered in the `0.21` Qwen 3 cycle applies directly here via a `MistralStrategy`.

**Success criterion:** Greedy decode of Ministral 3B Instruct at BF16 and Ministral 8B
Instruct at FP8 each match HuggingFace token-for-token on identical prompts. Tool calling
validated end-to-end on both models using the model-agnostic pipeline from the `0.21` cycle.

Phases: (1) Ministral transformer — `MinistralConfig` (`withSlidingWindowSize()`, `0`
disables SWA), `MinistralBlock` (even layers global GQA, odd layers SWA), SWA masking in
`GroupedQueryAttention` (CPU + CUDA parity), `MinistralTransformer`, presets for 3B (no SWA)
and 8B (`sliding_window=1024`). (2) Mistral v3 tokenizer (vocab 32768) +
`convert_ministral_weights.py`; `ModelType::Ministral` and `MistralStrategy`. (3) Ministral
3B BF16 validation. (4) Ministral 8B FP8 validation. Per-phase tasks live in
[BACKLOG.md](BACKLOG.md) once this milestone is active.

---

## Later — Unscheduled

Future cycles beyond Ministral, deferred until the library has a stable contributor base.
Each becomes its own minor cycle (`0.23`+) with its own ladder when picked up.

**Training** — Full LLaMA fine-tuning pipeline. Loss function GPU migration.
Gradient checkpointing. Checkpoint save and restore.

**Architecture** — Mixture of Experts components. Speculative decoding.
Additional attention variants.

**Performance** — Flash Attention integration. Tensor parallelism.
Deterministic gradient accumulation for training reproducibility.
