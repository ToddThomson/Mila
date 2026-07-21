# Sliding-Window KV Cache (Bounded Ring)

Specification and implementation plan for bounding the KV-cache allocation of
Gemma 4's local (sliding) attention layers to a fixed ring buffer instead of the
full context length. This is **gate 2 of 2** of the v0.20 Gemma memory-management
work (gate 1 = Weight Tying, `WeightTying.md`, shipped 0.20.0-alpha.6+77). The
delivered target is **Gemma 4 12B**; the bounded ring is reused later by the
Ministral sliding-window Future Direction.

---

## 1. Problem Statement

Gemma 4 interleaves 40 **sliding** (local, window = 1024) attention layers with 8
**global** (full-attention) layers over 48 layers (`sliding_window_pattern = 6`,
every 6th layer global). A sliding layer mathematically attends only to the most
recent `window` keys — and Mila already enforces this correctly as a **mask** in
the softmax (`prefill_softmax_bf16_kernel`, `window_start = max(0, abs_t - window + 1)`,
`Gqa.Prefill.Bf16.cu`).

But the cache is still allocated at the full context length. `CudaGqaOp`
unconditionally allocates `k_tensor_` / `v_tensor_` as `[B, NKV, T, HS]`
(`initializeState_optimized`, `CudaGqaOp.ixx`), and the QK / attention-value
cuBLASLt plans use `T` as the inner GEMM dimension. So a sliding layer pays
**full-T memory and full-T compute** to then mask away everything older than
`window`. At long context this is the dominant persistent allocation: it grows
linearly with context even though the layer can never read past the window.

`window` is already a runtime config field (`GqaConfig::getWindow()`,
default 0 = global/unbounded). What is missing is making the **allocation** and
the **GEMM/softmax extents** follow the window rather than the context length.

---

## 2. Memory Impact

KV-cache element cost is `2 (K+V) x NKV x HS x bytes` per token per layer
(BF16, `NoKvCompression` => cache dtype = compute precision = BF16, 2 bytes).

| Layer kind | count | NKV | HS  | per-token KV  |
|------------|-------|-----|-----|---------------|
| sliding    | 40    | 8   | 256 | 8 KB/token/layer  |
| global     | 8     | 1   | 512 | 2 KB/token/layer  |

So per token: **sliding = 320 KB/token**, **global = 16 KB/token** (total
336 KB/token, matching the ~0.35 MB/token measured floor).

After this change the **sliding** contribution becomes a fixed ring of
`capacity = window + prefill_chunk - 1` rows (see D2), independent of context:

| Context T | Sliding KV (full) | Sliding KV (bounded) | Global KV (unchanged) |
|-----------|-------------------|----------------------|-----------------------|
| 4 K       | 1.25 GB           | ~0.34 GB             | 64 MB                 |
| 8 K       | 2.5 GB            | ~0.34 GB             | 128 MB                |
| 32 K      | 10 GB             | ~0.34 GB             | 512 MB                |
| 128 K     | 40 GB             | ~0.34 GB             | 2 GB                  |
| 256 K     | 80 GB             | ~0.34 GB             | 4 GB                  |

(`~0.34 GB` = 40 layers x 1087 slots x 8 KB at `window = 1024`, `chunk = 64`.)

The headline: Gemma's persistent-KV growth slope drops from **336 KB/token to
16 KB/token** (the 8 global MQA layers only) — a 21x flatter curve. On the 12 GB
dev card this moves Gemma 4 12B FP4 from an effective ~512–4096 context (WDDM
thrash) to a safe ~32K, with headroom toward ~64K. Every additional GB of VRAM
then buys ~64K tokens of context (16 KB/token).

**What this does NOT shrink** (see §5): the shared per-chunk attention scratch
and the GeGLU FFN activation floor — both are chunk-bound transients sized by the
*global* geometry, not the persistent cache. Those are the province of chunked
prefill, not of this gate.

---

## 3. Approach Decision — Bounded Ring (Option A), not Flash Rewrite (Option B)

The hard constraint: a prefill **chunk** at positions `[off, off + chunk)` needs
the **union** of every query's window resident at once — up to `window + chunk - 1`
distinct keys — so a ring of exactly `window` cannot hold a chunk's working set
during its GEMM. Two ways to resolve it:

**Option A (chosen) — ring capacity = `window + prefill_chunk - 1`, reuse the
existing cuBLASLt plan structure.** Substitute `capacity` for `T` as the K/V
cache row count and the GEMM inner dimension, write with ring wrap
(`pos % capacity`), and teach the softmax to reconstruct each ring slot's
absolute position for masking. Capacity is ~1087 vs the ideal 1024 — a 6%
overshoot that is negligible against the 80 GB -> 0.34 GB win.

**Option B (rejected, out of scope) — ring capacity = exactly `window`, full
flash / block-sparse prefill.** Saves ~63 rows. Costs a streaming
flash-attention kernel (online softmax, block iteration over the K/V tiles).
Disproportionate to the marginal memory it reclaims; revisit only if a model
with `window >> context` and a hard scratch ceiling demands it.

Option A confines all new logic to: the cache allocation size, the write
indexing (wrap), the plan inner-dimension (`T -> capacity`), and **one** softmax
masking change (slot -> absolute-position). Softmax and the attention-value
product are set operations over keys, so ring (rotated) slot order is irrelevant
— no sort, no reordering. This keeps the change minimal and reversible: with the
identity policy, `capacity == T` and the path is byte-identical to today.

---

## 4. Design Decisions

**D1 — `SlidingWindowKvCache` is a `TKvPolicy` sibling, not a new axis.**
It joins `NoKvCompression` / `PerChannelKvFp8<>` on the existing KV-cache policy
axis (`Quantization/KvCache/Policy.ixx`). It satisfies the `KvCachePolicy`
concept with `kIsActive = true` and carries **no** `kStorageDtype` (it is
uncompressed — orthogonal to quantization; a `bounded + FP8` policy is a later
composition). It is **not** the window number: the window stays a runtime
`GqaConfig` field. The policy only signals "bound the cache to the window."

```cpp
export struct SlidingWindowKvCache
{
    static constexpr bool kIsActive = true;
};
static_assert(KvCachePolicy<SlidingWindowKvCache>);
```

**D2 — Ring capacity = `min(T, window + prefill_chunk - 1)`, requires `window > 0`.**
The bounded policy is only valid on a layer with a positive window; global layers
(`window == 0`) must keep `NoKvCompression`. The op computes
`cache_capacity_ = min(T, window + prefill_chunk_size - 1)` in `build()` and
asserts `window > 0` under the bounded policy. When `T <= window + chunk - 1` (short
context) the cache is already full-size and the ring degenerates to the linear
cache — correct and harmless.

**D3 — `CudaGqaOp` gains a compile-time `bool kBounded` axis; policy selects it
via `OperationTraits`.** `CudaGqaOp<TPrecision, bool kBounded = false>`.
`NoKvCompression -> kBounded = false` (capacity = T, identical to today);
`SlidingWindowKvCache -> kBounded = true`. Two new traits rows
(`OperationTraits.Cuda.ixx`). All bounded behavior is `if constexpr (kBounded)`
guarded — no runtime branch on the hot path, and the unbounded path compiles to
exactly the current code.

**D4 — Per-block-kind policy selection in Gemma (the heterogeneous-layer axis
already exists).** Gemma's global and sliding blocks are *already* separate
compile-time instantiations selected by the `kGlobal` flag (`Gemma.ixx`
`GlobalBlockType` / `LocalBlockType`). The sliding block carries
`TKvPolicy = SlidingWindowKvCache`; the global block keeps `NoKvCompression`.
This rides the existing per-layer-geometry mechanism — no new dispatch, no
`std::variant`, no runtime layer-kind check. (Plumbing prerequisite: confirm
`GemmaBlock` threads its `TKvPolicy` down into its `GroupedQueryAttention`
member — §6.0.)

**D5 — The persistent KV cache shrinks; the shared scratch does not.**
The QK plan output `preatt` is written into the **shared**
`gqa_preatt_ [B, NH, chunk, T_ctx]` workspace (`Gemma.ixx`
`allocateAndWireGqaWorkspace`), which is sized by the global layers and reused by
all blocks. A bounded sliding layer writes only `capacity` columns per row but
keeps the buffer pitch `T_ctx` (cuBLASLt `ldc = T_ctx`, `N = capacity` — a
column-prefix of each row, exactly the prefix pattern already used for
`q_permute` at `HS_max`). This means:
- **No new scratch allocation** — bounded layers reuse the shared buffer.
- The shared scratch stays `[B, NH, chunk, T_ctx]` (global-bound) — gate 2 does
  not and cannot shrink it (only flash on the global layers, or a smaller
  context, would).
- Bonus: the sliding QK/AV GEMMs run at `N = capacity` instead of `N = T`, so
  bounded sliding layers also prefill/decode **faster** — a welcome side effect
  while the 12 GB card is WDDM-bound.

**D6 — Slot -> absolute-position masking is the only new kernel math.** With a
ring, `preatt` column `j` is ring slot `j`, holding the key whose absolute
position `p_j` is the unique value in `[end - capacity + 1, end]` with
`p_j % capacity == j` (`end = cached_seq_len - 1`). The softmax keeps slot `j`
for query at `abs_t` iff `max(0, abs_t - window + 1) <= p_j <= abs_t` (window +
causal), else zeros it. A slot whose `p_j` falls outside the resident range
`[max(0, end - capacity + 1), end]` (cold/garbage slot before the ring fills) is
also masked. The exp / sum / normalize then runs over the kept slots in whatever
(rotated) order they sit — order-independent, so correctness does not depend on
sorting the ring.

**D7 — Decode is the easy half; prefill is the hard half; they share one
buffer.** Decode (`chunk = 1`) needs only `window` resident keys and softmax is
trivially order-independent — the ring write + slot mask is small. Prefill must
hold `window + chunk - 1` and apply causal masking *within* the chunk (a slot may
hold a same-chunk key at position `> abs_t`). Because prefill and decode share
the one ring buffer, capacity is sized for prefill (`window + chunk - 1`) and
decode tolerates the few extra slots (masked). The change is therefore
all-or-nothing per layer, but can be **developed and validated** decode-first,
then prefill, against the full-cache oracle (§7).

**D8 — Validate against the full-cache path as the oracle.** The
`NoKvCompression` path is numerically validated (Gemma 4 12B parity vs HF). Every
bounded result is checked against it: for `seq <= window` the bounded and full
outputs must be **bit-equivalent** (no eviction, identical masked set); for
`seq > window` the bounded output must match the full-cache path that already
applies the same window mask. No new reference is needed — the oracle is in-tree.

---

## 5. Why Chunked Prefill Stays (Interaction Analysis)

This gate does **not** make chunked prefill redundant, and chunked prefill does
**not** subsume this gate. They attack orthogonal terms:

- **Chunked prefill bounds the per-chunk *transients*** — the shared attention
  scratch `gqa_preatt_/gqa_att_ [B, NH, chunk, T_ctx]` and, dominant, the GeGLU
  FFN activations (`gate_up = 30720`, `hidden = 15360`, 48 layers), both sized at
  the prefill chunk. The FFN floor is **independent of attention entirely** and
  is the largest State term on the 12 GB card. Only chunking controls it; flash
  attention would not touch it.
- **This gate bounds the persistent *KV cache*** — the 40 sliding layers'
  `[B, NKV, T, HS]` allocations, the only term that grows with context.

So the State budget after both is: resident params (tied) + shared attention
scratch (chunk x T, global-bound, capped) + GeGLU FFN floor (chunk-bound) +
**bounded** sliding KV + global KV (16 KB/token). Chunked prefill owns the first
group; this gate owns the last two.

**Is chunked prefill still the best choice short of flash attention? Yes.** A
full flash rewrite (Option B applied to *all* layers) would additionally fuse the
global-layer score matrix so `gqa_preatt_/att_` no longer materializes
`[B, NH, chunk, T]` — but that is **one** shared buffer already capped at ~1.5 GB
by `computeGemmaPrefillChunkSize`, well below the FFN floor it cannot help. The
cost/benefit does not justify it for Gemma on this hardware. Chunked prefill +
the bounded ring is the right pairing.

One mild new coupling to note: `capacity = window + chunk - 1`, so a larger
prefill chunk slightly inflates the sliding ring (chunk 64 -> +63 rows on 1024).
Negligible, and the chunk-size heuristic (sized on the global geometry) is
unchanged and still valid. Making that heuristic activation-aware (so it accounts
for the FFN floor) remains separate BACKLOG work and is out of scope here.

---

## 6. Affected Files

| File | Change |
|------|--------|
| `Mila/Src/Dnn/Quantization/KvCache/Policy.ixx` | Add `SlidingWindowKvCache` struct (D1) + `static_assert` |
| `Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqaOp.ixx` | Add `bool kBounded` template arg; compute `cache_capacity_` in `build()`; size cache `[B,NKV,capacity,HS]`; ring-wrap writes; pass `capacity` as plan inner dim and `cached_seq_len_` to softmax; `if constexpr (kBounded)` guards throughout |
| `Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/CudaGqa.Plans.ixx` | QK / AV plan builders: separate the K/V cache row count (`capacity`) from the `preatt` output leading dim (`T_ctx` pitch, `ldc`); `N = capacity` |
| `Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/Kernels/Gqa.Prefill.Bf16.cu` / `.Fp32.cu` | Prefill softmax: slot -> absolute-position masking (D6); new param `cached_seq_len` (= `end + 1`) and `capacity` |
| `Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/Common/Kernels/CudaAttention.Softmax.*.cu` | Decode softmax: same slot -> absolute-position masking (D6) |
| `Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Attention/GQA/Kernels/Gqa.Cache.*.cu` | `kvcache_write_kv`: ring-wrap destination index `(start_pos + t) % capacity` under bounded mode |
| `Mila/Src/Dnn/Compute/Devices/Cuda/Operations/OperationTraits.Cuda.ixx` | Two rows: `(GroupedQueryAttentionOp, Cuda, {FP32,BF16}, SlidingWindowKvCache) -> CudaGqaOp<…, true>` |
| `Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.Block.ixx` | Thread `TKvPolicy` into the `GroupedQueryAttention` member (D4 plumbing prereq §6.0) |
| `Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.ixx` | Select `SlidingWindowKvCache` for `LocalBlockType`, `NoKvCompression` for `GlobalBlockType` |

### 6.0 Plumbing prerequisite

`GroupedQueryAttention<Device, Precision, TKvPolicy>` already resolves its op via
`OperationTraits<…, TKvPolicy>` and exposes `kKvCompressed` / `kCacheDtype`
(`GroupedQueryAttention.ixx`). Confirm `GemmaBlock` forwards its own `TKvPolicy`
into the GQA member's third template argument (today it likely defaults to
`NoKvCompression`). This is the one wiring gap to close before D4 takes effect.

---

## 7. Test Plan

All cases compare `SlidingWindowKvCache` against the validated `NoKvCompression`
oracle on the same `GqaConfig` (D8). Extend the existing GQA op test file in
place (inventory first — do not create a parallel suite).

### 7.1 Policy + plumbing (Phase 0)
- `SlidingWindowKvCache_SatisfiesConcept` — compile-time `static_assert`.
- `BoundedOp_CapacityEqualsContext_WhenWindowGEContext` — with `window >= T`,
  `cache_capacity_ == T`; bounded output is **bit-identical** to the full path
  (degenerate ring, no eviction).
- `BoundedOp_RequiresPositiveWindow_Throws` — bounded op with `window == 0`
  throws in `build()` (D2).

### 7.2 Bounded decode (Phase 1)
- `Decode_NoEviction_MatchesFullCache` — `seq <= window`: token-for-token
  identical to the full-cache decode.
- `Decode_PastWindow_MatchesMaskedFullCache` — `seq > window`: matches the
  full-cache path (which already window-masks). Exercises ring wrap + slot mask.

### 7.3 Bounded prefill (Phase 2)
- `Prefill_SingleChunk_MatchesFullCache` — one chunk, `chunk < window`.
- `Prefill_MultiChunk_AcrossWindow_MatchesFullCache` — several chunks with
  `off + chunk` crossing the window boundary; checks intra-chunk causal masking
  on rotated ring slots and cross-chunk eviction.
- `Prefill_PartialFinalChunk_MatchesFullCache` — `chunk_len < prefill_chunk`
  (the partial-chunk `row_offset` path).
- `PrefillThenDecode_MatchesFullCache` — full session: chunked prefill followed
  by decode steps over the same ring buffer.

### 7.4 Gemma acceptance (Phase 3)
- Re-run `GemmaModel.Parity.Cuda` — token ids unchanged vs HF (window math is
  identical; only the storage extent moved).
- `getMemoryStats` at context 4K and 32K — confirm the sliding KV floor is
  ~0.34 GB (flat) rather than growing with context; confirm chat coherence.

---

## 8. Implementation Sequence

0. **Policy + plumbing** — add `SlidingWindowKvCache`; add `kBounded` to
   `CudaGqaOp` with capacity computation; two `OperationTraits` rows; close the
   `GemmaBlock` `TKvPolicy` forwarding gap. With `kBounded = false` nothing
   changes — reversible checkpoint. Land §7.1.
1. **Bounded decode** — ring-write single token; decode plan at `N = capacity`;
   decode softmax slot -> abs (D6). Land §7.2.
2. **Bounded prefill** — ring-write chunk; prefill plan at `N = capacity`,
   `ldc = T_ctx`; prefill softmax slot -> abs with intra-chunk causal. Land §7.3.
3. **Wire Gemma + measure** — `LocalBlockType` -> `SlidingWindowKvCache`,
   `GlobalBlockType` -> `NoKvCompression`; re-run parity; confirm the flat
   sliding KV floor and chat coherence (§7.4).

---

## 9. Non-Goals / Future Directions

- **Bounded + FP8 KV** (`SlidingWindowKvCache` composed with FP8 storage) — the
  global layers become the new context wall (16 KB/token, 4 GB at 256K); FP8 KV
  on the *global* layers is the next lever. Separate policy composition.
- **Flash / block-sparse prefill** (Option B) — would additionally bound the
  shared global-layer attention scratch; below the FFN floor in payoff (§5).
- **Activation-aware chunk heuristic** — `computeGemmaPrefillChunkSize` is
  attention-scratch-sized and blind to the GeGLU FFN floor; orthogonal BACKLOG
  item.
- **GQA training / backward** — Llama and Gemma are inference-only; the bounded
  ring is an inference-path concern.
- **Llama sliding layers** — Llama 3.x is global-only; no sliding layer to bound.
