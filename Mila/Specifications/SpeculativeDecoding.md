# Speculative Decoding

Design Contract for Draft-Verify Accelerated Decoding in Mila's Generation Loop

Status: **DRAFT / proposed milestone** (no code yet). Sibling of the planned-feature specs
(`TokenSampling.md`, `PromptCaching.md`, `ToolCalling.md`).

---

## 1. Overview

Autoregressive decode emits one token per target forward pass, and on consumer GPUs that pass is
memory-bandwidth-bound: each step re-streams the full weight set to produce a single token
(see the decode-perf notes — 8B FP4 sits at ~57 tok/s, bandwidth-bound). Speculative decoding
breaks the one-token-per-pass floor: a cheap **drafter** proposes `K` candidate tokens, the target
model **verifies all `K` in a single forward pass**, and every token whose drafted distribution the
target agrees with is accepted. The target still emits at least one correct token per step, so the
output is **lossless** — identical in distribution to plain autoregressive generation — while the
amortized cost per accepted token drops.

Gemma 4 ships its own MTP variant (a 4-layer drafter that shares the embedding table and builds on
the target's last-layer activations). Rather than hard-wire that one bespoke drafter, this spec makes
**the drafter a compile-time policy axis** — the same orthogonal-axes pattern the rest of Mila is
built on (`TWeightQuantization`, `OperationTraits`, `TKvPolicy`). The reusable, high-value
engineering is not any single drafter; it is the **draft -> batched-verify -> accept/reject ->
KV-rollback scaffold**, which is identical across every drafter family and lands directly on top of
the two-phase prefill/decode loop, the bounded sliding-window KV cache, and the lean `generate()`
primitive already in place.

The first shippable drafter carries **no weights and no training** (Prompt Lookup), which respects
the inference-only reality of the Gemma/GQA path. The performance drafter (EAGLE-1/2) slots into the
same scaffold as a second policy.

---

## 2. Key Insight — the scaffold is drafter-agnostic

Every speculative-decoding method decomposes into four steps, only the first of which differs
between methods:

```
1. DRAFT    drafter proposes K tokens  d[1..K]           <- the ONLY method-specific step
2. VERIFY   one target forward over the K drafted tokens -> per-position target distribution p[0..K]
3. ACCEPT   walk positions; accept longest prefix the target agrees with (rule per Section 5)
4. COMMIT   emit accepted tokens + one target "bonus" token; roll the KV cache back to accepted length
```

Steps 2-4 are shared code. They are also exactly the operations Mila already performs elsewhere:
step 2 is a `K`-wide **prefill-shaped** forward (the model already has a prefill path that processes
many tokens at once); step 4's "roll the KV cache back" is the one genuinely new primitive, and it
is where the interaction with the bounded sliding-window ring must be designed carefully
(Section 6).

So the milestone is: **build steps 2-4 once as a `SpeculativeDecoder` orchestrator tool, and express
the drafter (step 1) as a swappable compile-time policy.** Gemma's own 4-layer drafter, EAGLE,
Medusa, and Prompt Lookup then all become drafter policies behind one scaffold.

---

## 3. Architecture — Dispatch and Ownership

### 3.1 An orchestrator tool, not a graph component

`SpeculativeDecoder` is an orchestration tool owned by the `LanguageModel` base, **not** a
`Component` — structurally the sibling of `TokenSampler` (see `TokenSampling.md` s3.1) and
`Optimizer`:

- it owns no graph parameters and has no position in the module/operation graph;
- it drives the target `forward()` and the drafter; it is invoked *by* `generate()`, it does not
  live inside `forward()`;
- it takes the model's `IExecutionContext*` (shared, never owned) so the draft, the verify forward,
  the sampler, and the KV-cache writes all run on the same stream — a drafter that owned its own
  context/stream would force a per-step cross-stream sync and defeat the point.

The `LanguageModel` base owns the single `SpeculativeDecoder`, and `generate()` selects between the
plain decode loop and the speculative loop.

### 3.2 The drafter as a compile-time policy axis

The drafter is resolved the same way weight quantization and operations are — a compile-time policy,
not a runtime string:

```cpp
template <DeviceType TDeviceType, TensorDataType TPrecision, typename TDrafter = NoSpeculation>
class SpeculativeDecoder;
```

Drafter policies (each a small type exposing the same `propose(...)` contract, Section 4):

| Policy | Weights | Training | Hidden states used | Lossless | Fit |
|---|---|---|---|---|---|
| `PromptLookupDrafter<NGram>` | none | none | none | yes | ship first — pure algorithm |
| `EagleDrafter<...>` | 1 decoder layer | external | **last only** | yes | performance play; reuses decoder-layer component + tied `lm_head` |
| `GemmaMtpDrafter<...>` | 4 layers | ships pretrained | last-layer activations | yes | Google's own; heaviest |
| `MedusaDrafter<Heads>` | MLP heads | external | last only | *not* under non-greedy | de-prioritized (not strictly lossless) |

`NoSpeculation` is the default — `generate()` falls straight through to the existing decode loop,
zero overhead, so nothing regresses for models without a drafter.

**Deliberately out of scope:** EAGLE-3 (low/mid/high multi-layer feature fusion) and DeepSeek-style
per-depth MTP modules. Both require taps on *intermediate* decoder-layer hidden states, which the
chassis does not expose and which would be real surgery through the stack for a marginal acceptance
gain. EAGLE-1/2's **last-hidden-only** contract is the sweet spot and matches what the model already
produces.

### 3.3 What the drafter policies reuse (why EAGLE is nearly a drop-in)

The chassis already provides every ingredient EAGLE-1/2 needs:

- a **decoder-layer component** (the Gemma/Llama block) — EAGLE's drafter *is* one such layer;
- the **tied `lm_head`** (weight-tying shipped, see `WeightTying.md`) — the drafter reuses it to turn
  its hidden state into a token distribution instead of carrying its own output projection;
- the **token embedding table** — shared, not duplicated;
- the target's **last hidden state**, already produced each step.

The only thing EAGLE adds that the repo does not have is an externally-trained draft-layer weight set
for the specific target model. That is the single external dependency, and it is why Prompt Lookup
(which needs none) leads the phase plan.

### 3.4 Module layout (proposed)

| Role | Type | Module | Location |
|---|---|---|---|
| Orchestrator (owned by model) | `SpeculativeDecoder<Device,Precision,Drafter>` | `Dnn.Speculation.SpeculativeDecoder` | `Speculation/SpeculativeDecoder.ixx` |
| Drafter contract | `DrafterConcept` | `Dnn.Speculation.Drafter` | `Speculation/Drafter.ixx` |
| Zero-weight drafter | `PromptLookupDrafter<NGram>` | `Dnn.Speculation.PromptLookupDrafter` | `Speculation/PromptLookupDrafter.ixx` |
| Feature drafter | `EagleDrafter<Device,Precision>` | `Dnn.Speculation.EagleDrafter` | `Speculation/EagleDrafter.ixx` |
| Construction config | `SpeculationConfig` | `Dnn.Speculation.SpeculationConfig` | `Speculation/SpeculationConfig.ixx` |

Any device kernels a drafter needs (e.g. EAGLE's draft-layer forward, a tree-attention mask) are
resolved through `OperationTraits` exactly like graph ops — a missing specialization is a hard
compile error, not a runtime miss.

---

## 4. Interface Contract

### 4.1 Drafter contract

```cpp
// Propose up to max_draft candidate tokens continuing `context`.
// Returns the count actually proposed (may be < max_draft, e.g. Prompt Lookup found no match).
// `last_hidden` is the target's last-layer hidden for the most recent accepted token
// (ignored by PromptLookupDrafter; consumed by EagleDrafter).
std::size_t propose( std::span<const int32_t> context,
                     const TensorView& last_hidden,
                     std::span<int32_t> draft_out,      // length max_draft
                     std::span<float>   draft_prob_out, // q(d_i) under the drafter, for the accept rule
                     const SpeculationParams& params );
```

- `draft_prob_out` carries the drafter's probability `q` for each proposed token. It is required by
  the **lossless stochastic** acceptance rule (Section 5); greedy ignores it. Prompt Lookup has no
  meaningful `q` and therefore is only exactly-lossless in the greedy/argmax regime (Section 5.3).

### 4.2 SpeculationConfig (construction-time) vs SpeculationParams (per-call)

Mirrors the `SamplingConfig` / `SamplingParams` split in `TokenSampling.md`:

- **`SpeculationConfig`** (construction-time): `max_draft_length` (K, e.g. 4), drafter-specific fixed
  fields (EAGLE draft-layer dims, Prompt Lookup n-gram order).
- **`SpeculationParams`** (per-call, part of / adjacent to `GenerateParams`):
  `num_assistant_tokens` (dynamic K per Gemma's docs), `assistant_tokens_schedule`
  (`constant | heuristic`).

### 4.3 Verify forward contract

Verification is a single target forward over the `K` drafted tokens plus the anchor token,
processed as a prefill-shaped batch (`outer_size == K` on the decode-with-multiple-tokens path).
It must return the target distribution `p` at **every** drafted position, not just the last —
acceptance needs `p[i]` for each `i`. This is the one place the decode path must yield per-position
logits rather than only the final position.

---

## 5. Acceptance Rule (the lossless guarantee)

### 5.1 Greedy

Accept `d[i]` iff `argmax(p[i]) == d[i]`, stopping at the first mismatch. At the mismatch (or after
the last accepted token) emit `argmax(p[j])` as the bonus token. Trivially lossless: the emitted
sequence is exactly what greedy autoregression would have produced.

### 5.2 Stochastic (speculative sampling, Leviathan/Chen)

For each drafted token in order, draw `r ~ U[0,1)` and accept if `r < min(1, p[i](d_i) / q[i](d_i))`.
On the first rejection, sample the replacement token from the normalized residual
`normalize(max(0, p[i] - q[i]))` and stop. If all `K` are accepted, sample one bonus token from
`p[K]`. This provably preserves the target distribution `p` — it is lossless by construction, not
approximately.

### 5.3 Dependency on TokenSampler

The rule needs the target's **probabilities** `p[i](·)`, and for the stochastic path the residual
draw is itself a sample from a distribution. This couples speculative decoding to `TokenSampling.md`:
the device sampler must expose a "given logits and a `q`, produce accept-decision + residual-sampled
token" primitive, reusing the same softcap -> temperature -> top-k/top-p pipeline and the same
host-drawn `r` (so parity oracles stay deterministic). Practically: **land device sampling first
(that milestone), then build stochastic speculative decoding on it.** Greedy speculative decoding
(5.1) has no such dependency and can be validated against the existing greedy path immediately.

---

## 6. KV-Cache Rollback — the one genuinely new primitive

The verify forward writes `K` speculative entries into the KV cache. When only `m < K` are accepted,
the cache must be rewound to length `accepted`, because the rejected entries were computed from a
continuation the target did not take.

- **Contiguous KV cache:** rollback is a write-pointer decrement — cheap.
- **Bounded sliding-window ring (`SlidingWindowKvCache.md`):** the ring write index advanced by `K`
  and may have **wrapped**, overwriting slots that held still-live earlier keys. Rewinding the index
  is not enough if a wrap clobbered a slot that a later query still needs. The design must guarantee
  rewindability, e.g. by requiring `K <= (window - live_length)` headroom before drafting on a
  local (bounded) layer, or by staging speculative KV writes in a scratch region and committing only
  the accepted prefix into the ring. **This is the primary correctness risk of the whole feature and
  must be settled before the EAGLE phase** — Gemma interleaves bounded local layers with full global
  layers, so both cache kinds are live simultaneously and roll back together.

The rollback primitive is added to the KV-cache interface and implemented per policy
(`NoKvCompression`, bounded ring, and any FP8 KV variant), validated in isolation before wiring into
the loop.

---

## 7. Validation

- **Losslessness is the headline invariant.** For any drafter, greedy speculative decoding must
  produce **token-for-token identical** output to the plain greedy path on the same prompt/seed. This
  is a build-vs-build comparison in the Chat harness (the project's edit-a-constant-then-rebuild
  workflow), and it protects the existing Gemma parity.
- **Stochastic parity:** with a fixed seed and injected `r` stream, the speculative path must select
  the same tokens as a reference non-speculative stochastic path consuming the same `r` — the
  acceptance rule is deterministic given `(p, q, r)`.
- **KV-rollback unit oracle:** after a forced partial-accept, the cache contents and subsequent
  logits must equal those of a run that never drafted past the accepted length. Exercise the wrap
  case on a bounded layer explicitly.
- **Acceptance-length metric:** report mean accepted tokens per step and realized tok/s speedup;
  these are performance signals, never correctness gates (a drafter that accepts nothing is *slow*,
  not *wrong*).

---

## 8. Implementation Plan

Each phase advances only when its parity gate is clean.

- **Phase A — scaffold + Prompt Lookup (no weights, no training).** `SpeculativeDecoder` orchestrator,
  `DrafterConcept`, `PromptLookupDrafter<NGram>`, `SpeculationConfig`/`Params`; the `K`-wide verify
  forward yielding per-position logits; **greedy** acceptance (5.1); the KV-rollback primitive
  (Section 6) for the contiguous/global cache. Wired into `generate()` behind `NoSpeculation`-default
  policy selection. **Gate:** greedy output token-for-token identical to the plain path; measurable
  speedup on repetitive/grounded text (code, RAG, tool-call echo — the MIS path).
- **Phase B — bounded-ring rollback.** Extend the rollback primitive to the sliding-window ring with
  the wrap-safety guarantee (Section 6); validate on Gemma's interleaved local/global layers.
  **Gate:** rollback unit oracle green including the wrap case; Gemma greedy speculative output
  identical to plain.
- **Phase C — stochastic acceptance.** Add the speculative-sampling rule (5.2) on top of the device
  `TokenSampler` primitive (5.3). **Gate:** injected-`r` parity vs the non-speculative stochastic
  path.
- **Phase D — EAGLE drafter.** `EagleDrafter` as a second policy: one decoder-layer forward over
  `(last_hidden, shifted-token-embedding)`, reusing the tied `lm_head` and embedding table; loader
  for an external EAGLE head; start with a **linear** draft chain (defer tree drafts). **Gate:**
  lossless parity retained; acceptance length and tok/s reported vs Prompt Lookup.
- **Phase E (optional) — Gemma MTP drafter / tree drafts.** Wrap Google's 4-layer drafter as a policy
  and/or add dynamic tree drafting (EAGLE-2 style). Only if profiling shows the linear-chain ceiling
  is the bottleneck.

### 8.1 Locked / proposed decisions

- **Drafter is a compile-time policy** (`TDrafter`), default `NoSpeculation` (zero-overhead
  fall-through). Not a runtime registry.
- **Prompt Lookup leads** — it needs no weights and no training, so Phase A ships without a checkpoint
  hunt and proves the entire scaffold; EAGLE's external-weight dependency is isolated to Phase D.
- **Last-hidden-only drafter contract** — excludes EAGLE-3 / per-depth MTP by design (no
  intermediate-layer taps).
- **Lossless is non-negotiable** — greedy and stochastic paths both preserve the target distribution
  exactly; Medusa's non-lossless non-greedy mode is why it is de-prioritized.
- **Depends on device sampling** for the stochastic path (Phase C after `TokenSampling.md`); greedy
  (Phases A/B) does not.

### 8.2 Boundary note

This is core `Mila/Src/` work (new `Speculation/` subtree, KV-cache interface change, a `generate()`
branch, per-position logits from the decode path) and therefore requires explicit agreement before
implementation — this document is the proposal, not a green light. The Chat harness
(`Samples/Chat/Src/`) is the validation driver and is freely editable.

---

## 9. Relationship to other specs

- `TokenSampling.md` — the device sampler the stochastic acceptance rule builds on (hard dependency
  for Phase C).
- `SlidingWindowKvCache.md` — the bounded ring whose rollback semantics are the main correctness risk
  (Phase B).
- `WeightTying.md` — the tied `lm_head` the EAGLE drafter reuses.
- `Gemma.md` — the target chassis (interleaved local/global layers) speculative decoding must respect.
- Gemma 4 MTP (`ai.google.dev/gemma/docs/mtp`) — the reference for one drafter policy
  (`GemmaMtpDrafter`), not the whole design.
