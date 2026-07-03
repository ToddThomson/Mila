# Gemma 4 Inference Pipeline Review

Findings from a full-path review (2026-07-02) of the Gemma 4 12B FP4 generation
pipeline: `LanguageModel::generate()` -> `GemmaModel::onGenerating` ->
`GemmaTransformer::prefill/decode` -> `GemmaBlock` -> `CudaGqaOp` /
`CudaLinearOp` decode matvec -> `TokenSampler` -> `stochastic_kernel`. The
pipeline is correct and HF-validated token-for-token; this review is about
performance headroom and a small number of latent defects.

Three defects were recorded in `BACKLOG.md` (Gemma 4 section) during the review.
The performance items below are ranked recommendations, not scheduled work,
until they are adopted into BACKLOG.

---

## 1. Scope and Method

Traced by reading, not profiling (profiling is Recommendation 0):

| Layer | Files |
|-------|-------|
| Generation loop | `LanguageModel.ixx`, `GemmaModel.ixx` (`onGenerating`) |
| Network | `Gemma.ixx` (`prefill`/`decode`, chunk heuristic, GQA workspace) |
| Block | `Gemma.Block.ixx` (per-layer op sequence, both paths) |
| Attention op | `CudaGqaOp.ixx`, `Gqa.Prefill.Bf16.cu`, `Gqa.Cache.Bf16.cu`, `CudaAttention.Softmax.Bf16.cu` |
| Linear decode | `CudaLinearOp.ixx` (`outer_size == 1` matvec paths), `CudaMatVecBias.Bf16.cu` |
| Sampler | `TokenSampler.ixx`, `Sampling.cu` (`argmax_kernel`, `stochastic_kernel`) |
| Harness | `Chat.ixx` (`generateResponse`, `buildInputTokens`, tool-round loop) |

All byte/launch counts below are derived from the code and the 12B geometry;
they are estimates to be calibrated by the in-tree `ProfileModel` + Nsight
before any kernel work is committed.

---

## 2. Platform and Cost Model

### 2.1 Resident footprint (corrected 2026-07-02)

The checkpoint is BF16; Linear weights are quantized to FP4 per-group at
`loadParameter()` time (quantize-on-load). Post the two v0.20 memory gates
(weight tying, bounded KV ring) and with `kGemmaPrefillChunkOverride = 32`,
Gemma 4 12B FP4 **fits the 12 GB RTX 4070** and runs VRAM-resident:

| Component | Size | Notes |
|-----------|------|-------|
| FP4 linear weights + scales | ~5.8 GB | ~10.9B linear params packed 2/byte + FP32 group-128 scales (~0.34 GB) |
| Tied BF16 embedding / lm_head table | ~2.01 GB | 262144 x 3840 x 2B, shared (WeightTying.md) |
| KV caches | ~0.48 GB | 40 local rings (1055 rows) + 8 global full-context (8K build) |
| Per-layer activations + GQA workspace | ~0.4-0.7 GB | sized at chunk 32 |
| **Total** | **~8.7-9 GB** | fits 12 GB; decode is weight-bandwidth-bound, not paging-bound |

### 2.2 Decode bandwidth budget (per token, 8K-context build)

| Read per token | Bytes | Share |
|----------------|-------|-------|
| FP4 linears + scales (all 48 layers) | ~5.8 GB | ~70% |
| lm_head matvec (BF16 tied table) | ~2.01 GB | ~24% |
| KV cache (local rings + global full reads, see 4.3) | ~0.48 GB | ~6% |
| Activations / attention scratch | noise | |

Total ~8.3 GB/token. At ~430 GB/s effective on the 4070 that is a ~19 ms/token
(**~52 tok/s**) bandwidth ceiling. Anything measured meaningfully below the
ceiling is launch/sync/sampler overhead — which is what most of the decode
recommendations attack.

**Measured 2026-07-03 (section 10.1):** 37.7 tok/s greedy = 26.5 ms/token, of
which ~24.9 ms is GPU-busy — the FP4 matvec sustains 379 GB/s (not 430), so the
current-kernel ceiling is ~46 tok/s and the host-side gap is only ~1.5-2 ms.

### 2.3 Decode launch count

Per layer per token: input_norm, qkv matvec, split, q_norm, k_norm, rope,
v_norm, GQA (write_kv, permute_q, QK GEMM, softmax, AV GEMM, unpermute),
o_proj, post_attn_norm, res_1, pre_ffn_norm, fc_gate_up, geglu, fc_down,
post_ffn_norm, res_2, layer_scalar scale = **~23 launches**. Times 48 layers
plus embedding/final-norm/lm_head/sampler: **~1,100 launches per token**, plus
two host round-trips (stream sync + synchronous D2H token copy). Gemma is
roughly 2x Llama's per-layer launch count (sandwich norms + QK/V norms + split
+ scale), so the earlier Llama conclusion "bandwidth-bound, Graphs ruled out"
does not automatically transfer.

### 2.4 Prefill cost model at chunk 32

Chunked prefill re-reads **all** linear weights every chunk (~5.8 GB per 32
prompt tokens). lm_head runs once per prefill (last position only).

| Prompt length | Chunks | Weight traffic | Floor at ~430 GB/s |
|---------------|--------|----------------|---------------------|
| 512   | 16  | ~93 GB   | ~0.22 s |
| 2048  | 64  | ~371 GB  | ~0.86 s |
| 8192  | 256 | ~1.48 TB | ~3.4 s  |

Weight re-read dominates prefill at chunk 32 (attention scratch + KV traffic
adds ~1 GB/chunk, ~20%). Prefill weight traffic scales as `1/chunk` — the chunk
size is the primary TTFT lever (see 5.1).

**Measured 2026-07-03 (section 10.2): this floor model is superseded.** A
2048-token prefill takes 20.8 s, 24x the table above, because the W4A16 prefill
GEMM is compute-bound (~2.5 TFLOPS, ~40 us per chunk-row), not weight-read-bound;
the chunk lever only pays after that kernel is fixed.

---

## 3. Defects Found (recorded in BACKLOG)

1. **[latent, B>1] Chunked-prefill last-position view ignores the batch
   stride.** `GemmaTransformer::prefill` (`Gemma.ixx`) and
   `LlamaTransformer::prefill` (`Llama.ixx`, the origin) extract the final
   position as `view({B,1,model_dim}, (T_last-1)*model_dim)` — correct only for
   batch row 0. Latent (inference builds are always B = 1) but the code carries
   B through every shape as if batched.
2. **[minor, API edge] `max_new_tokens = 0` still emits one token.**
   `GemmaModel::onGenerating` emits the prefill-sampled token before the
   `max_new` bound is consulted (decode loop starts at step 1). Same structure
   in Llama/Gpt.
3. **Prefill per-layer `res0` copy is suspected redundant.**
   `GemmaBlock::prefill` copies its input into `res0_`, but `decode()` feeds
   the same input reference through the identical Residual structure with no
   copy, and no component in the block writes the previous block's output
   buffer. Verify against the HF-greedy parity test, then delete (one launch +
   one full stream read/write per layer per chunk) — or document the real
   aliasing hazard.

Hygiene (not BACKLOG-tracked): `GemmaModel::makeTokenTensor` stages the prompt
through a pageable CPU tensor and allocates a fresh device tensor per
`generate()` call; the pinned `StagingMR` alias defined at the top of
`GemmaModel.ixx` is unused (an existing REVIEW comment already flags the
double copy). Prefill-only, once per call — minor.

---

## 4. Decode Findings

### 4.1 Per-token sync structure (D1)

Each decode token costs two host round-trips:

1. `getLanguageNetwork().synchronize()` — `cudaStreamSynchronize` on the
   network stream (`GemmaModel::onGenerating`), required only because
2. the sampling kernel runs on the **default** stream (`TokenSampler::sample`,
   the Phase A MSVC-reachability workaround), followed by a synchronous D2H
   copy of the 4-byte token.

Fix (= the already-planned TokenSampling.md Phase D): run the sampler on the
decode stream, drop the explicit per-token `synchronize()`, keep one sync point
(the D2H copy). Follow-on once streams are unified: **decode-ahead** — the
sampler already writes the next token into `decode_token_device_` on-device, so
decode N+1 can be enqueued *before* the host reads back token N for the
stop-check/`on_token` callback. Hides the entire host round-trip + callback
latency; costs one wasted decode step at the stop token.

### 4.2 Launch-count reduction (D2)

In cheap-to-expensive order:

- **Decode `split` is unnecessary.** At T = 1 the fused QKV output's Q/K/V
  sections are contiguous slices; views replace a kernel plus a full read/write
  of the QKV buffer per layer. (Prefill genuinely needs the split — rows
  interleave.)
- **Fold `layer_scalar` into the `res_2` residual add** (alpha on the epilogue)
  instead of a separate full-stream `scale()` per layer (both paths).
- **Fuse QK-norm + RoPE (+ v_norm + KV-cache write)** — four to five
  tiny-tensor launches per layer that are one kernel's worth of work.

### 4.3 Fused decode attention (D3)

`decode_optimized` is 6 launches per layer, and its two cuBLASLt GEMMs are
M = 1 matvec-shaped calls computed over the **full `cache_capacity_`** columns
every token: a global layer in an 8K build reads its entire ~16.8 MB K/V cache
from token 1 onward (~134 MB/token constant across the 8 global layers,
regardless of actual length), and the softmax masks the garbage. A single
fused flash-decode-style kernel (QK dot + online softmax + AV accumulate over
only the live rows):

- drops 5 launches/layer (~240/token),
- eliminates the preatt/att global-memory round-trips,
- reads only `actual_len` (global) / window (local) cache rows.

This is the flagship kernel-craft piece and feeds directly into the Blackwell
native-FP4/attention work. The existing full-cache path stays as the oracle,
mirroring the bounded-ring methodology.

### 4.4 lm_head decode read (D4)

The tied BF16 table costs ~2.01 GB of the ~8.3 GB per-token read (~24%). An
FP8 lm_head — either a separate quantized copy (+1 GB VRAM, keeps BF16
embeddings) or quantizing the shared table (needs an embedding-quality eval;
Gemma scales embeddings by sqrt(d)) — is worth roughly +12-15% decode. The
+1 GB variant is a close call against the 4070's ~2.5-3 GB headroom; measure
free VRAM at the chunk-32 operating point before choosing. Note the softcap is
monotonic so greedy decode is insensitive to small logit error; sampled decode
needs the eval.

### 4.5 Sampler kernel (D5)

`stochastic_kernel` is deliberately correctness-first: a **single block** (one
SM), a 40-iteration top-k binary search that re-reads the full 262k-element
scratch each iteration (~42 MB read by one SM), an optional 40-iteration top-p
search, and a **thread-0 serial** inverse-CDF walk over up to 262k elements.
`argmax_kernel` (greedy) is also single-block. Estimated 0.5-1.5 ms/token —
measure first. Fix shape: multi-block two-stage reductions + a parallel scan
for the CDF. The injected-r oracle (TokenSampling.md) makes this refactor
safely testable.

**Measured 2026-07-03 (section 10.3): 11.05 ms/token** — 7-20x the estimate;
29.9% of sampled-decode wall (the chat default path). Promoted to the top
decode item in the re-ranked table (10.4). `argmax_kernel` measured 26 us —
the greedy path is a non-issue.

### 4.6 CUDA Graphs (deliberately deferred)

Re-measure only after D1/D2/D3 have removed the launches they were going to
hide; the Llama measurement showed Graphs do not pay once bandwidth-bound, and
the fusion work attacks the same overhead while also reducing traffic.

---

## 5. Prefill Findings

### 5.1 Chunk 32 weight re-read is the dominant TTFT term (P1)

See 2.4. `kGemmaPrefillChunkOverride = 32` (`Gemma.ixx`) is the operating point
that keeps the 48 per-layer activation buffers small enough to fit the 4070 —
the heuristic (`computeGemmaPrefillChunkSize`) is attention-scratch-sized and
activation-blind, so it cannot be re-enabled as-is. The two tracked BACKLOG
items are the fix, in this order of value:

- **Pool the 48 per-layer activation buffers** (only one layer is live at a
  time in the sequential forward). Pooling frees enough State on the 4070 to
  run chunk 512 -> **16x less prefill weight traffic** — the single largest
  measurable win in this review. Elevated to inference-path defect with a full
  fix design in **section 7**.
- **Activation-aware chunk budget** so the heuristic picks the largest chunk
  that fits a real VRAM budget (and the constexpr override reverts to 0).

Full design for the replacement heuristic — alternatives considered, cost
model, budget rule, drift guard — in **section 6**.

### 5.2 Prefill softmax is one thread per query row (P2)

`prefill_softmax_bf16_kernel` (`Gqa.Prefill.Bf16.cu`) assigns one *thread* per
query row: three serial passes over up to `cache_capacity_` columns (8192 on
global layers), uncoalesced (adjacent threads are a full row apart). At chunk
32 the whole launch is `B x NH x chunk = 512` threads = **2 blocks on the
entire GPU**. The decode softmax already got the warp-per-row rewrite
(`softmax_decode_forward_bf16_kernel`, the +20% Llama decode win); port that
pattern to the prefill and prefill-ring variants. Small, isolated, oracle
already exists (bounded-vs-full parity tests).

### 5.3 Prefill GEMMs span full cache capacity from chunk 0 (P3)

The QK/AV prefill plans are built at `N = cache_capacity_`, so every chunk
computes scores against all capacity columns even when only
`position_offset + chunk_len` are live — ~2x FLOP/traffic waste integrated over
a full-context prompt, far worse for short prompts in a big-context build (a
1K prompt in an 8K build computes ~8x the needed columns). The partial-plan
cache already keys by `chunk_len`; extend the key to a live-length bucket
(round `position_offset + chunk_len` up to 512/1024) and pick the plan per
chunk. Global layers only (local layers are already capacity-bounded by the
ring).

### 5.4 Incremental prefill / prefix reuse (P4)

`Chat::generateResponse` re-prefills the **entire** conversation every turn and
every tool round (`kMaxToolRounds = 4` full re-prefills per tool-using turn),
yet decode already leaves the assistant's response tokens in the KV cache — the
true delta is only the new user message or spliced tool response. A
`prefillFrom(position)` compute primitive (already anticipated by the
Generation API design and PromptCaching.md) plus prefix matching in the harness
turns per-turn latency from linear-in-history into linear-in-delta. Composes
cleanly with the bounded ring: the ring holds exactly the window a continued
decode needs. The biggest user-visible chat win.

---

## 6. Prefill Chunk: Alternatives Considered and Heuristic v2

Design follow-up to 5.1 (discussed 2026-07-02): is chunked prefill the right
mechanism at all, and if so, what should set the chunk for Gemma 4?

### 6.1 Chunking is the right mechanism

The alternatives all reduce back to it:

- **Full-sequence prefill** is chunk = T. At Gemma's width it is hopeless: the
  fc_gate_up buffer alone is 30720 wide, so full-context activation buffers
  across 48 layers cost tens of GB.
- **Flash-style fused prefill attention** eliminates only the
  `[B, NH, chunk, T]` preatt/att scratch — the term the v1 heuristic budgets —
  but not the FFN activation buffers, which are the dominant term (6.3). Even
  with a flash prefill you still chunk; it just removes one term from the cost
  model. Worth doing eventually alongside the fused decode attention (D3), not
  as the chunk fix.
- **Runtime-adaptive chunking** buys nothing: buffers are allocated once at
  build time, so the footprint is set by the max chunk regardless of what runs.
  (Partial chunks already work — the plan cache handles them.)
- Industry practice is the same mechanism: vLLM's chunked-prefill token budget
  and llama.cpp's `n_ubatch` are both a fixed chunk against a memory budget.
  The relevant difference is that ggml sizes **one shared compute buffer** for
  the worst-case graph and reuses it across layers — which is exactly the
  activation-pooling BACKLOG item (option D). Mila's per-layer-owned buffers
  are the anomaly, not the chunk.

Conclusion: improve the heuristic; pooling later changes its constant (48
private buffer sets -> 2-3 shared slabs), not the mechanism.

### 6.2 Why the v1 heuristic failed on Gemma

`computeGemmaPrefillChunkSize` is a verbatim copy of Llama's
`computePrefillChunkSize` (`Llama.ixx`): a fixed 1536 MB cap, a {512..16}
ladder, and a cost model containing **only the shared GQA attention scratch**.
It was never wrong for Llama — just unexercised: 3B/8B FP4 leave enough
headroom on 12 GB that the blind spot never bit. Gemma 12B is the first model
wide (GeGLU 30720) and deep (48 layers) enough to expose that the dominant
chunk-scaled term is the per-layer activation buffers, which the v1 model
cannot see. Hence the manual `kGemmaPrefillChunkOverride = 32` operating point.

### 6.3 The correct per-chunk-row cost model (Gemma 12B, 8K build)

All terms computable from `GemmaConfig` at `onBuilding` time:

| Term | Per chunk-row | Notes |
|------|---------------|-------|
| Per-layer activation buffers x 48 | ~10.4 MB | sum of component output widths ~= 105K elements x 2B ~= 217 KB/layer; fc_gate_up (30720) dominant |
| Shared GQA scratch (preatt/att/q_perm/v_out) | ~0.56 MB | `NH x (2 x T_ctx + 2 x HS_max) x 2B` — the ONLY term v1 sees |
| Bounded-ring KV growth (`capacity = window + chunk - 1`) | ~0.33 MB | 8 KB/token/layer x 40 local layers |
| **Total** | **~11.3 MB/row** | x 512 ~= 5.8 GB — matches the measured State floor that forced the override |

### 6.4 Heuristic v2

```
budget = free VRAM (cudaMemGetInfo, after params + KV caches are allocated)
         - safety margin (~1 GB: cuBLASLt workspace, FP8 dequant staging,
           WDDM/driver slack)
chunk  = largest c in {512, 256, 128, 64} with row_cost x c <= budget
```

- **Ladder floor is 64, not 16.** The chunk is the M dimension of every prefill
  GEMM; M = 32 is tensor-core-hostile on top of the 16x weight re-read. If 64
  does not fit, log a clear "this card cannot prefill this model efficiently"
  warning rather than silently limping at 16.
- **No gain past ~512:** weight amortization is done and the O(T^2) attention
  term does not care; 512 stays the ceiling.
- On the 4070 today (~2-2.5 GB budget / 11.3 MB per row ~= 195): picks
  **128** — a 4x prefill-traffic improvement over the forced 32 from the
  heuristic fix alone, no kernel work.
- **After pooling** (section 7) the row cost collapses to ~1 MB, the same
  heuristic lands at 512 on the 4070, and it remains as the guard that keeps a
  future 26B or 128K-context build honest.
- `kGemmaPrefillChunkOverride` stays as the debug/sweep escape hatch, default
  back to **0** once the budget is honest.

### 6.5 Drift guard

The analytic row-cost formula will rot as the block graph changes — that is
the v1 story. Pin it with a test instead of maintenance: build one `GemmaBlock`
at two chunk sizes and assert the formula's slope matches the measured
`getMemoryStats().device_state_bytes` delta. The formula is then allowed to be
a simple closed form because the test catches divergence. (The self-maintaining
alternative — probe-build at a small chunk, measure the slope, rebuild at the
chosen chunk — was considered and set aside: it adds a rebuild pass for no
accuracy the pinning test does not already provide.)

---

## 7. Activation Buffer Sharing (Pooling) — Benefit and Fix Design

Elevated 2026-07-02 from "VRAM lever D, deferred" to **inference-path defect**:
it is the single reason the chunk-32 operating point exists, and therefore the
root cause of the 16x prefill weight re-read in 2.4.

### 7.1 The problem: component-owned outputs are a training-first design

Every component allocates its own output tensor at build shape and retains it
for its lifetime — `output_ = std::make_unique<TensorType>(...)` uniformly
across `RmsNorm.ixx`, `Linear.ixx`, `Residual.ixx`, `Swiglu.ixx`,
`GroupedQueryAttention.ixx` (plus the block-owned `res0_/q_/k_/v_` scratch in
`Gemma.Block.ixx`). That ownership model is **correct for training**: backward
needs each layer's retained activations. On the inference-only Gemma path it is
pure waste: the forward is strictly sequential, exactly **one layer is live at
a time**, and 47/48 of the retained bytes can never be read again. The design
is not wrong in general — it is wrong *for this path*, and Gemma 12B (wide
GeGLU x 48 layers) is the first model where the waste is the binding
constraint.

### 7.2 Benefit

| Quantity | Today (per-layer ownership) | Pooled (one shared slot set) |
|----------|-----------------------------|-------------------------------|
| Activation bytes per chunk-row | ~10.4 MB (48 x ~217 KB) | ~0.23 MB (~45x less) |
| Activations at chunk 32 | ~333 MB | ~7 MB |
| Activations at chunk 512 | ~5.3 GB (why the override exists) | ~118 MB |
| Total State at chunk 512 (8K build) | ~5.9 GB | ~0.6 GB (incl. GQA scratch + ring growth) |

Downstream consequences on the 4070:

- **Chunk 512 fits** -> 16x less prefill weight traffic than today's forced 32:
  2048-token prompt floor ~0.86 s -> ~54 ms; 8K prompt ~3.4 s -> ~0.22 s.
  Heuristic v2 (6.4) then picks 512 automatically. *(Traffic floor only —
  measured correction in 10.2: the prefill GEMM is compute-bound, so the
  wall-clock end-state is ~1-2 s for 2048 tokens and requires the P0 GEMM fix
  to land alongside pooling.)*
- ~0.33 GB freed immediately even at chunk 32.
- The FP8 lm_head option (4.4, +1 GB) becomes comfortable instead of tight.
- State stops being the fit constraint for the future 26B / long-context builds.

Decode speed is unchanged (decode uses T = 1 views of the same buffers); the
benefit there is freed VRAM only.

### 7.3 Fix design

Three in-tree precedents make this a pattern-extension, not new architecture:
the shared GQA transient workspace (`GqaState` + `setState`, wired by the
transformer to all 48 layers), `Linear::installSharedWeight` (weight tying —
post-build replacement of a component's own allocation with shared storage),
and the `ExecutionContext` grow-on-demand scratch.

- **Transformer-owned `BlockActivationWorkspace`**: one tensor per block graph
  position (~17 slots: normed, qkv, q/k/v splits, q/k/v-norm outs, attn out,
  o out, o_normed, res1, ffn_in, gate_up, geglu out, down out, ffn_normed,
  stream/res2), each sized `[B, chunk, max(local, global) width]` — the same
  max-geometry convention the GQA workspace already uses (HS_max prefix).
  Total ~230 KB per chunk-row.
- **`installSharedOutput(std::shared_ptr<TensorType>)`** on the five component
  types, mirroring `installSharedWeight`; `output_` becomes `shared_ptr`. The
  existing `output_view_` reshape logic is untouched — it views a prefix of the
  (possibly wider) shared slot, exactly as GQA views `q_permute` today.
- **Defer-allocation flag on `BuildContext`** so components skip self-allocation
  when an install is coming — avoids a transient 48-layer allocation peak
  during build.
- **Aliasing analysis** — slot-per-graph-position keeps intra-block liveness
  identical to today (a block behaves exactly like the single-block case). The
  only new cross-block fact is the residual stream: block i+1's *input* is the
  shared stream slot, which is last **read** at `res_1` (mid-block) and only
  **written** by block i+1's own `res_2` (end of block). A single stream slot
  is therefore safe — no ping-pong needed. The prefill `res0` copy (defect 3,
  section 3) is definitively removable in this world.
- **`getMemoryStats` correctness**: count the workspace once at the
  transformer; installed outputs must not be double-counted per component —
  same shape as the tied-weight D7 correction.
- **Scope guard**: Gemma wiring only. Self-allocation stays the component
  default, so training models and Llama are untouched (Llama can adopt later
  for a smaller win).

### 7.4 Phasing and validation

- **Phase 0** — settle defect 3 (`res0` copy) against the HF-greedy parity
  test; pin the parity baseline.
- **Phase 1** — workspace struct + wiring plumbing; pool only the block-owned
  `res0_/q_/k_/v_` scratch (no component API change). ~15% of the per-layer
  bytes; proves the wiring end to end.
- **Phase 2** — `installSharedOutput` + defer flag on RmsNorm / Linear /
  Swiglu / Residual / GroupedQueryAttention; `GemmaBlock` installs slots into
  its children at build. Validation: HF-greedy token-for-token parity +
  a closed-form State assertion (mirror of
  `StateMemory_MatchesClosedFormAndShrinks` from the bounded-ring work).
- **Phase 3** — heuristic v2 (6.4) + revert `kGemmaPrefillChunkOverride` to 0;
  chunk lands at 512 on the 4070.

---

## 8. Ranked Recommendations

**Superseded 2026-07-03 by the measured re-ranking in section 10.4** (Rec 0 is
done); kept for the original estimates the measurements are compared against.

| # | Item | Effort | Expected gain (est.) | Risk |
|---|------|--------|----------------------|------|
| 0 | Profile decode + long prefill (ProfileModel + Nsight) | hours | calibrates everything below | none |
| 1 | D1 + D2 cheap batch (stream unify, drop per-token sync, split->views, scale fold) | days | +5-15% decode | low |
| 2 | P1 buffer pooling (section 7) + heuristic v2 (section 6) -> chunk 512 on the 4070 | medium | 16x less prefill weight traffic (heuristic alone: 4x) | low (parity oracle + closed-form State test) |
| 3 | P4 incremental prefill (`prefillFrom` + Chat prefix match) | medium | per-turn TTFT ~O(delta) instead of O(history) | medium (API + ring interaction tests) |
| 4 | P2 warp-per-row prefill softmax port | small | prefill attention kernel time; GPU no longer 2-blocks-busy | low (parity oracle exists) |
| 5 | D3 fused decode attention | large | ~240 launches/token + live-length reads; flagship kernel work | medium (full-cache oracle stays) |
| 6 | D4 lm_head FP8 | medium | +12-15% decode | eval needed; +1 GB variant vs 4070 headroom |
| 7 | D5 sampler scale-up | small-medium | ~0.5-1.5 ms/token if profile confirms | low (injected-r oracle) |
| 8 | P3 live-length GEMM buckets | small | up to ~2x prefill attention FLOPs (global layers) | low |

If only three: **1 -> 2 -> 3** — the overhead batch pays immediately, pooling +
chunk is the biggest raw number, incremental prefill is what users feel most in
chat.

---

## 9. Cross-References

- `BACKLOG.md` — Gemma 4 section: the three defects (recorded 2026-07-02), the
  VRAM-footprint item (pooling = option D, activation-aware chunk = option A),
  FP8 KV on global layers.
- `TokenSampling.md` — Phase D (stream unification, sampler hoist), injected-r
  oracle. This review adds the concrete per-token sync tally (4.1) and the
  262k-vocab kernel-shape numbers (4.5).
- `PromptCaching.md` + BACKLOG "Generation API" preamble — `prefillFrom` /
  `rewindKvCache` primitives (5.4).
- `SlidingWindowKvCache.md` — bounded ring; supplies the oracle methodology
  reused by D3/P2, and the layer geometry used in the cost model.
- `WeightTying.md` — why lm_head is BF16 and shared (4.4 trades against it).

---

## 10. Measured Calibration (2026-07-03 — Recommendation 0 done)

Three Nsight Systems captures (nsys 2026.2.1, x64-profile build, RTX 4070,
8K-context build, `kGemmaPrefillChunkOverride = 32`, greedy unless noted;
capture region = the measured run only, via cudaProfilerApi + NVTX).
`ProfileModel` gained `--model gemma` and `--temperature` for this pass.

| Capture | Shape | Wall (NVTX) |
|---------|-------|-------------|
| Decode, greedy | 256 tokens, 16-token prompt | 6.96 s (255 decode steps) |
| Prefill | `--seq-len 2048` (64 chunks of 32) | 20.77 s |
| Decode, sampled | 128 tokens, temperature 0.7 | 4.95 s (127 decode steps) |

### 10.1 Decode (greedy): 37.7 tok/s — bandwidth story confirmed, two surprises

26.5 ms/token wall, ~24.9 ms GPU-busy. Launch count ~1,060/token and ~2.4
syncs/token as modeled (2.3, 4.1), but the host-side gap is only ~1.5-2
ms/token — mostly hidden behind GPU work.

| Kernel group | ms/token | Share | Note |
|---|---|---|---|
| FP4 matvec (192/token) | 15.3 | 60% | ~379 GB/s effective (75% of 504 peak) |
| lm_head BF16 matvec | 4.15 | 16% | ~484 GB/s (96% of peak) — already optimal |
| RmsNorm (337 launches/token) | 2.63 | 10% | single-block launches; ~11.6 us wide / ~2 us narrow |
| Attention (GEMMs + softmax + permutes + KV write) | ~2.4 | 9% | ring QK ~94 GB/s, M = 1 over full capacity |
| RoPE / residual / GeGLU / split / scale | ~0.45 | 2% | |
| argmax sampler | 0.03 | ~0% | greedy sampling is a non-issue |

Corrections to the 2.2 model:

- The FP4 matvec sustains 379 GB/s, not ~430; the realistic current-kernel
  ceiling is ~46 tok/s and the measured 37.7 is ~82% of it.
- **NEW headroom (D6):** the lm_head matvec proves ~484 GB/s on the same access
  pattern; closing half the FP4 matvec gap is worth ~1.7-3 ms/token — as much
  as the entire D1+D2 batch.
- **RmsNorm is the D2 sleeper:** the sandwich norms cost more than
  split+scale+rope combined. Norm fusion (or at least a multi-block norm) moves
  to the front of D2.
- D1 shrinks to ~6-8%; D3 (~2.4 ms addressable) and D4 (~2 ms, FP8 only)
  confirmed as modeled.

### 10.2 Prefill: 20.8 s for 2048 tokens — compute-bound, not traffic-bound

The 2.4 floor (~0.86 s) is off 24x, and the miss invalidates the traffic-centric
model: 87.9% of the wall is `fp4a16_wmma_gemm_kernel` (12,288 launches, 17.97 s,
`CudaW4A16Gemm.Wmma.cu`), and its per-launch time scales with chunk rows.
Fitting the M = 16 (825 us, decode capture) and M = 32 (1,462 us) points:

    t_launch(M) ~= 188 us fixed (weight read, ~160 GB/s) + ~40 us per chunk-row

i.e. ~2.5 effective TFLOPS, while the cuBLASLt BF16 attention GEMMs in the same
capture run ~26 TFLOPS. One full weight pass costs 281 ms (~21 GB/s effective).

- **~15.6 s of the 20.8 s is chunk-size-independent** per-row compute. Pooling
  -> chunk 512 with today's kernel buys ~12%, not 16x — the 16x was traffic,
  and traffic is not the binding term.
- **NEW item (P0): fix the W4A16 prefill GEMM first.** Candidate shape: FP4 ->
  BF16 dequant-staging + cuBLASLt (the FP8 path's existing 2-phase approach),
  bandwidth-bound at ~2.8x today's kernel even at chunk 32 — and it restores
  the chunk lever so pooling pays out.

  **SHIPPED + re-measured 2026-07-03 (same day):** 2-phase dequant-staging is
  the new default FP4 batch path behind a `kUseFusedFp4Gemm` A/B toggle
  mirroring the FP8 baseline. 2048-token prefill: **20.77 s -> 10.21 s
  (2.03x)**; the linear term dropped 17.97 s -> 7.39 s (2.4x): dequant 3.98 s
  (40% of wall, 62 ms/weight-pass ~= 275 GB/s — vectorization headroom to
  ~400 GB/s is worth ~1.2 s) + cuBLASLt GEMMs 3.41 s (~13 TFLOPS aggregate at
  M = 32; fc_gate_up hits ~26, the small shapes drag). The chunk lever is now
  live: dequant traffic scales 1/chunk and GEMM efficiency rises with M, so
  pooling -> chunk 512 projects ~10.2 s -> ~3-3.5 s, and P2 (the softmaxes,
  now 2.01 s = 20% of wall) -> ~1.5 s. Chat coherent; HF-greedy FP4 parity
  test not yet re-run (opt-in).
- The honest end-state is **FLOP-bound**: a 2048-token prefill carries ~44.7
  TFLOP of linear GEMM work; at 26-50 TFLOPS (measured M = 32 cuBLASLt ->
  larger-M efficiency) that is **~1-2 s** (8K prompt ~3.5-7 s) — 10-20x better
  than today, but not the 54 ms the traffic-only model in 7.2 suggested.
- **P2 confirmed and quantified:** the two one-thread-per-row prefill softmaxes
  cost 2.0 s = 10% of prefill (global variant 1.8 ms/launch, 2 blocks on the
  whole GPU).
- **P3 demoted:** the full-capacity global attention GEMMs cost only ~82 ms.
- **P4 elevated:** at ~99 tok/s prefill throughput, an 8K-history re-prefill is
  ~80 s of per-turn chat latency.

### 10.3 Sampler: stochastic_kernel is 11 ms/token — 7-20x the 4.5 estimate

At temperature 0.7 the single-block `stochastic_kernel` averages 11.05 ms/token
(128 instances, 29.9% of the capture, stable 10.9-11.6 ms). Sampled decode —
the chat default — runs at 39.0 ms/token (**25.6 tok/s vs 37.7 greedy**): the
sampler alone costs ~30% of every sampled token. D5 is promoted from "measure
first" to the top decode item: small, isolated, injected-r oracle already
specified, +~40% on the path users actually run.

**FIXED + re-measured 2026-07-03 (same day):** the multi-block pipeline
(TokenSampling.md section 5) measures **55.5 us/token** — a 200x kernel
reduction — and sampled decode landed at **35.7 tok/s** (28.0 ms/token, the
predicted number). The residual ~1.4 ms/token gap to the greedy capture is
run-to-run core-clock variance on the tiny latency-bound kernels (their medians
moved capture-to-capture while the bandwidth-bound matvec medians did not) —
the same kernels D2's norm fusion attacks.

### 10.4 Re-ranked recommendations (measured)

| # | Item | Measured cost | Expected gain | Effort |
|---|------|---------------|---------------|--------|
| 1 | D5 sampler scale-up (multi-block + parallel scan) | 11.05 ms/token sampled | sampled decode 39 -> ~28 ms/token (+~40%) | small-medium |
| 2 | **P0** W4A16 prefill GEMM (NEW: dequant-staging or WMMA rework) | 87.9% of prefill at ~2.5 TFLOPS | ~3x prefill now; unlocks the chunk lever | medium |
| 3 | P1 pooling + heuristic v2 (chunk 512) | State floor confirmed | with P0: 2048-prefill ~1-2 s; VRAM freed regardless | medium |
| 4 | **D6** FP4 decode matvec bandwidth (NEW: 379 -> 450+ GB/s) | 15.3 ms/token at 75% peak | ~2-3 ms/token decode | medium |
| 5 | D2 + norm fusion (RmsNorm first, then split/scale/rope) | ~3.2 ms/token | ~2-3 ms/token decode | days |
| 6 | P2 warp-per-row prefill softmax port | 2.0 s per 2048 prefill | nearly all of it | small |
| 7 | P4 incremental prefill (`prefillFrom` + prefix match) | ~80 s per 8K re-prefill | per-turn O(delta) | medium |
| 8 | D3 fused decode attention | ~2.4 ms/token | most of it | large |
| 9 | D4 lm_head FP8 | 4.15 ms/token (already at peak BW) | ~2 ms/token | medium + eval |
| 10 | D1 stream unify + decode-ahead | 1.5-2 ms/token | ~6-8% decode | days |

P3 drops off the ranked list; CUDA Graphs stay deferred (4.6). If only three:
**1 -> 2 -> 3** — the sampler pays immediately on the chat path, and the
prefill GEMM + pooling pair is the TTFT story. Full decode stack
(1+4+5+8+9+10): sampled decode 39 -> ~17-19 ms/token (~55 tok/s).
