# RoPE In Attention

**Status:** Draft, 2026-10-01. No code. Both facts that decide it (section 5) were measured the same day and both
pass; Todd, the same day, takes it forward for what it does to latency: the stored-once cache buys the 26B-A4B its
largest prefill chunk at 64K on a 16 GB card (section 3). What remains before it is a design is section 4.3's quality
gate and the kernels' shape. Admitted to v0.21 on 2026-10-04 (Todd), after the 26B-A4B's context profile measured it
reliable to 96K on recall and losing the middle of a conversation at 128K (`ContextProfile.md` section 8): the stored-once
cache is for reaching 96K, or a vision tower at a useful context, not 128K -- and for the 12B's global layers too.

**Area:** where rotary position embedding is applied, and what the KV cache stores as a result. RoPE as a
separate operation is `CudaRopeOp`; its angles are calculated, not stored (`MemoryFootprint.md` 8.4).

---

## 1. Where RoPE runs today

`CudaRopeOp` is its own pass between the QKV projection and attention. Every rotation calculates each cos and sin
once per (token, pair) through one device function (`Rope.Angle.cuh`) and rotates every Q and K head of the token
with it, in one launch (`Rope.Rotation.cuh`). The op holds no state. The KV cache stores keys already rotated.

The proposal, Todd's: RoPE as a **compile-time policy** -- table (retired 2026-10-01), calculated (today), fused into
attention -- so the separate op stays the readable reference and a fused form is a gated, performant alternative
chosen by type, as `TKvCachePolicy` and `TWeightQuantization` already are.

## 2. Two different fusions

"Fused RoPE" names two things that buy different things and cost different things.

| | Fused at the cache write | Fused at the attention read |
|---|---|---|
| Where keys rotate | while written to the cache | while attention reads the cache |
| What the cache holds | rotated keys, as today | unrotated keys |
| What it buys | one pass over Q and K and one launch less | unrotated storage (section 3) |
| Cost | none measurable | trigonometry per cached key per decode step (section 4.2) |
| Numerics | can stay bit-identical | changes the FP8 cache's rounding order (section 4.3) |
| Needs a policy axis | no | yes |

**Write-time fusion** belongs with the "split + RoPE + cache write" fusion already listed among the launches left
to remove (`DecodeGraph.md`, `ModelFamilyParity.md` L5). It is a kernel fusion, not a design decision, and needs
nothing from this document.

The rest of this document is about **read-time fusion**.

## 3. What read-time fusion would buy

- **Gemma's global keys stored once.** On Gemma 4's global layers the checkpoint has one projection for keys and
  values (K = V), but the cache holds two tensors: K = RoPE(k_norm(x)) and V = v_norm(x). If keys are rotated
  inside attention, the cache need not hold rotated keys, and if k_norm(x) and v_norm(x) can be recovered from one
  stored tensor, it holds one tensor where it holds two. Re-derived 2026-10-01 from the footprint the planner prices
  (`GemmaLogLikelihoodCudaTests.DISABLED_KvCache_26B_Q4_0_Fp8Global_ChunkPricing`, Q4_0, FP8 global cache, context
  65536, 15,172 MiB free on the RTX 5060 Ti): the global cache's two tensors are about 650 MiB, so storing once frees
  about 325 MiB, and the planner's largest chunk, 1024 rows, goes from 216 MiB over to about 109 MiB under -- where
  today it takes 128. At 32K, the same cache moving the chunk from 256 to 1024 made the whole prefill 1.41x faster
  (`Quantization.md`, decode at 32K). The earlier "about 96K" estimate is withdrawn; capacity at the largest chunk is
  the figure that matters to a user waiting on a prompt.

  **Capacity, from the planner, 2026-10-03** (`ProfileModel --model gemma --quantization q4_0 --kv-cache fp8`, the
  26B-A4B quantized on load from Google's QAT checkpoint, `0.21.0-dev+32`, 15,904,800,768 bytes free on the RTX 5060
  Ti). Today 81920 fits at a 256-row chunk with 22 MiB to spare; 98304 plans 15,992,329,216 bytes, 87 MB over; 131072
  plans 16,330,494,976, 426 MB over. Between the two refusals the deployment grows 10,320 bytes a token: five global
  layers, two KV heads of 512 at one byte, keys and values, plus their scales. Storing keys once removes 5,160 of
  those, so 131072 would plan about 15,654 MB, **about 251 MB under** -- the 26B-A4B at 128K on a 16 GB card, at
  whatever chunk that margin buys.
- **Nothing else that Mila can use.** Unrotated keys would let a cached prefix be reused at a shifted position, but
  that changes outputs, and `PromptCaching.md`'s invariant is that reuse never does.

## 4. The case against

### 4.1 The premise is unverified

Storing once requires k_norm(x) and v_norm(x) to come from one stored tensor -- for instance both RMSNorm of the same
x differing only by their weights, or v_norm weightless, so the cache stores the normalized x and attention applies
k_norm's weight and the rotation on read. Not checked against the HuggingFace Gemma 4 code. If they differ in any
other way, section 3's first item, and with it the case for read-time fusion, does not exist.

### 4.2 It moves the cost to where it grows

Rotating at read time re-rotates every cached key in every global layer on every decode step: trigonometry
proportional to the context, the per-key repetition the shared-angle kernel was built to remove (`MemoryFootprint.md`
8.4), now multiplied by the context length. Past position 105615 the accurate `cosf` takes its slow argument
reduction. Gemma's global layers rotate 64 of 256 pairs (partial rotary 0.25), which helps by four.

Whether the work hides behind the cache read is unmeasured, and there may be no slack: the HS-512 global decode
kernel already reads about 7% below bandwidth (`Quantization.md`, "Where compression buys speed"). The ways out
each give something up:

- **Angle recurrence across a tile** (rotate by a fixed step from a base angle): not bit-identical to the
  reference, and its error accumulates along the tile.
- **A key-side table** of the rotated pairs only: 64 pairs x 2 x 4 bytes = 512 bytes a token, shared across layers.
  Possibly still a net win against the stored-once saving; it is also the table just retired.

### 4.3 It changes the FP8 cache's numerics

Today keys are rotated, then quantized to FP8. Read-time rotation quantizes unrotated keys and rotates after
dequantizing. Not bit-identical to the reference, so each family that takes it needs the long-context quality gate
again (`Quantization.md` KV decision 6): hours of book scoring per arm, and decision 6's own open arms are still open.

### 4.4 It benefits one layer type of one family

Only checkpoints with K = V gain memory: Gemma's global layers. Llama and Qwen project K and V separately. The policy
would be carried by kernels every family shares, and the GQA kernel set is already large -- BF16 and FP8 caches,
bounded and unbounded, several head sizes, prefill and decode. A fused variant doubles it. The separate op stays the
readable reference; the performant kernels get harder to read, and the instantiation count rises again (Chat's
largest object is about 80K sections, built with `/bigobj`).

### 4.5 Speed alone does not justify it

Decode replay (`DecodeGraph.md`) removed most launch overhead; the separate RoPE pass costs about 1.4 to 2 us a
layer at decode and is now cheaper than the table was in prefill. Read-time fusion removes that pass and adds
section 4.2's work.

Measured in whole models, 2026-10-01 (nsys, kernel time, Q4_0, RTX 5060 Ti, a 32,512-token prompt then 256 tokens at
that depth; the 26B-A4B with the FP8 global cache, the 12B with BF16 caches): RoPE is 0.3% of either model's prefill
(25.7 ms of 8.3 s, 39.6 ms of 14.1 s) and 0.7% and 0.4% of its decode (63 and 79 us a token). Removing the pass
outright is the most any form of fusion could save, so speed cannot carry this proposal; capacity has to.

### 4.6 It is past the target

The 26B-A4B fits 65536 on 16 GB with the FP8 global cache, which was the target. 96K is beyond it, and v0.21 is dated
2026-11-20.

## 5. What decides it

Two facts, each cheap, in this order. Either failing ends the proposal; it stays in `Mila/Issues/` and triage
sends it to `Future.md`.

1. **Whether one stored tensor serves both k_norm and v_norm** on Gemma 4's global layers: read the HuggingFace
   Gemma 4 attention code and the checkpoint's norm weights. Minutes. Gate: K and V must be expressible as
   per-channel scalings (or identity) of one normalized tensor, with the rotation applied to K only.
2. **What read-time key rotation costs** inside the global decode kernel: a microbenchmark at 32K and 64K against
   today's kernel, the method `MemoryFootprint.md` 8.4 used for the RoPE kernel (nsys, kernel time per call,
   median of 200, RTX 5060 Ti). Gate written before the run: the fused decode attention within 5% of today's at
   64K, with the accurate angle function. A key-side table of rotated pairs measured as the alternative if not.

If both pass, read-time fusion is a candidate for the release after v0.21: a compile-time policy on Gemma's global
attention only, gated per section 4.3, with section 3's figure re-derived from the footprint first.

**Both measured, 2026-10-01; both pass.**

1. **One stored tensor serves both.** HuggingFace's Gemma 4 attention (transformers 5.12.1, `modeling_gemma4.py`) on a
   K = V layer sets `value_states = key_states`, the raw projection; V = `v_norm( raw )` with `with_scale=False`, and
   K = RoPE( `k_norm( raw )` ), the same RMSNorm with the same epsilon times its weight. So K = RoPE( V * w_k ),
   rotation on the first 128 of 512 dims only. The 384 unrotated dims of K are V times w_k, and w_k folds into the
   query once per step: the attention kernel reads them from V's own tile, and builds only 64 rotated pairs per key.
2. **Building them costs less than the bytes saved.** `Profiling/Microbenchmarks/RotateOnRead.cu`, decode's tile loop
   over one global layer of the 26B-A4B (2 KV heads, FP8), RTX 5060 Ti, median of 200: at 65536 positions, today's two
   tensors 342.5 us; V alone with the 64 pairs rotated by the accurate `sincosf` 235.0 us, **0.69 of today**; an angle
   recurrence 0.67; a table of angles 0.91, its reads costing more than the trigonometry. At 32768, 0.72. The gate was
   parity within 5%. No MMA competes for issue in the microbenchmark, so the real kernel's margin is smaller; the
   accurate angle function is kept, since the recurrence buys 2%.

Section 4.2's cost is therefore not the obstacle it was feared to be, and section 4.5 changes: read-time fusion is now
a speed gain at depth as well as a capacity one. Section 4.3 -- the FP8 cache quantizing unrotated values, so the
long-context quality gate again -- stands, as does 4.4's cost in kernels. One more change rides with it: the cache's V
could be stored FP16 rather than BF16 -- the same bytes, three more significand bits, bounded by the weightless
`v_norm` -- which removes the per-tile narrowing the FP16 PV path pays over a BF16 cache (`GqaFlashAttention.md` 5.8).
