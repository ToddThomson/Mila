# FP8-Activation Prefill GEMM (W4A8-FP8)

Status: **Implemented + shipped ON** (spec 2026-07-12; landed + validated + profiled same day). The path is
live behind `kUseFp8ActivationPrefill = true` (default on): `Forward_MatchesReference` 5e-2 + Gemma token
parity green (per-tensor weight scale sufficed — no per-channel escalation needed), and it profiled 1.24x
faster prefill @48K on the RTX 4070 (1056 -> 1307 tok/s, flash on in both). Owner surface:
`Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Linear/CudaLinearOp.ixx` and its FP4 dequant kernels. This is
an **internal op optimization** — no component API change.

Measured caveat (nsys @48K): the linear GEMMs are only ~24% of Gemma prefill, so the ~2x GEMM speedup yields
1.24x end-to-end; attention (~62%: global flash + local sliding-window) is the dominant remaining cost — the
next levers are the flash kernel ladder and flashing the local layers, not more matmul work.

---

## 1. Goal

Close the one remaining prefill gap to llama.cpp on Gemma 4 12B (and the Llama targets): run the batched
prefill linear GEMMs on **FP8 tensor cores (~2x BF16)** instead of the current BF16 path, without changing the
`Linear` component's BF16-in / BF16-out contract.

### Measured motivation (RTX 4070, 2026-07-12)

- Gemma 4 12B FP4 prefill, 22496 tokens @ 48K: **Mila 1056 tok/s vs llama.cpp (Q4_K_M/int8) 2063 tok/s = 1.95x
  behind.** The gap is *entirely* the activation-precision / tensor-core-rate factor (chunk sizing is already
  correct; see [[project_w4a16_prefill_gemm]]).
- Standalone cuBLASLt microbench (`scratchpad/fp8_gemm_bench.cu`): **FP8xFP8 = ~2.0x BF16xBF16 on the 4070**
  (1.90-2.11x across prefill shapes), and **regular FP32 accumulate already hits the 2x** (fast-accum not
  needed -> we keep accumulate precision). Consumer-Ada FP8 is NOT gated.
- Expected outcome: prefill from ~1.95x-behind to **~1.1-1.3x** (competitive). End-to-end is < 2x because
  attention stays BF16 and there is an activation-quantize pass; the linear GEMMs dominate prefill, so most of
  the gap closes.

### Why FP8, not int8 (llama.cpp's MMQ)

Same 2x on Ada, but FP8 E4M3 is floating-point (tolerates activation outliers int8 clips -> more likely to
pass the existing oracle), and cuBLASLt has a **native** FP8 GEMM — we reuse the tuned library instead of
hand-rolling an int8 MMQ kernel. (The Stage-1/2 fused W4A16 experiments showed a hand kernel does not casually
beat cuBLASLt; see [[project_w4a16_prefill_gemm]].)

---

## 2. Hard constraint (the pivot)

**Gemma 4 12B linear weights MUST remain FP4 in VRAM** (packed nibbles + per-group FP32 scales, quantized at
`loadParameter()` time — unchanged). FP8 *storage* would be ~12 GB of weights and will not fit the 12 GB 4070.

Therefore FP8 is applied **only transiently for the prefill GEMM**:

- Weights: FP4 storage -> **transient FP4->FP8 E4M3 upcast** into a scratch buffer (this replaces the current
  FP4->BF16 staging; it is **half the bytes** — 1 B/elem vs 2 B/elem — so staging traffic drops).
- FP4->FP8 is numerically ~lossless: FP4 E2M1 has 8 magnitude levels; after applying the per-group scale these
  land cleanly on FP8 E4M3's finer grid. The FP8 weight scale is computed **once at load** (static, like the
  FP4 group scales) — see Section 5.

Decode (`outer_size == 1`, the `cuda_matvec_decode_bf16_qfp4` matvec) is **untouched**: it stays FP4, is
bandwidth-bound, and gains nothing from FP8 tensor cores.

---

## 3. Data path (prefill / batched, `outer_size > 1`)

Contract preserved: BF16 activation in, BF16 activation out. Everything below is internal to `CudaLinearOp`.

```
BF16 activation A[M,K]  --quantize-->  FP8 A8[M,K] (E4M3) + activation scale sA   (DYNAMIC, per-forward)
FP4 weight W (stored)   --upcast-->    FP8 W8[N,K] (E4M3) + weight scale sB        (STATIC, from load)
cuBLASLt FP8xFP8 GEMM (TN, FP32 accumulate, A_scale=sA, B_scale=sB) -> D[M,N] BF16
+ bias (BF16) -> output
```

- **Accumulate:** FP32 (regular, NOT fast-accum — the microbench shows regular already gives 2x, so keep the
  precision).
- **Output:** BF16 (`CUDA_R_16BF`), preserving the op boundary.
- **Layout:** cuBLASLt FP8 requires the **TN** form (op(A)=T, op(B)=N), column-major, 16-byte aligned. The
  microbench (`fp8_gemm_bench.cu`) is the working reference for the exact descriptor/scale/layout setup.

This mirrors the existing 2-phase FP4 path structure exactly (dequant kernel -> scratch -> cuBLASLt plan),
so it slots into the same branch of `CudaLinearOp::forward`.

---

## 4. Scope / non-goals

In scope:
- Prefill/batched FP8xFP8 path in `CudaLinearOp` for `PerGroupFp4` weights on CUDA/BF16.
- A new FP4->FP8 weight upcast kernel (sibling of `cuda_fp4_dequantize_to_bf16`).
- A BF16->FP8 activation quantize kernel + dynamic scale.
- cuBLASLt FP8 plan (new plan variant / entry in the linear plan cache).
- A compile-time (initially) toggle to A/B against the BF16 path, like `kUseFusedFp4Gemm`.

Out of scope (explicitly):
- Decode path (stays FP4 matvec).
- FP8 *weight storage* (rejected — VRAM).
- int8 / MMQ (rejected — FP8 is better here).
- The hand-rolled fused W4A16 kernel (Stage 2, parked; `kUseFusedFp4Gemm=false`).
- Attention (stays BF16; a separate lever).
- FP8 for the Llama/Gpt path beyond what falls out for free — validate on Gemma first.

---

## 5. Design decisions (the real choices for the implementation session)

1. **Weight FP8 scale granularity** — *the key numerics risk.*
   - Simplest: **per-tensor** FP8 weight scale (single FP32 `sB`), computed once at load as
     `absmax(dequant_fp4_weights) / 448` (E4M3 max). The per-group FP4 scales are absorbed into the FP8 values.
   - Risk: FP4 uses per-group scales *because* weights have per-group dynamic-range variation; a single
     per-tensor FP8 scale may lose too much and fail Gemma token parity. E4M3's 4-bit exponent gives wide
     range, so per-tensor may suffice — but if parity fails, escalate to **per-channel (per-N-row)** scaling
     via cuBLASLt's vector/outer scale (or block scaling on CUDA 12.4+). Decide by measurement, not a priori.
   - Compute the FP8 weight scale(s) at load, store alongside the FP4 weight (small).

2. **Activation FP8 scale granularity** — dynamic (activations change per forward).
   - Default: **per-tensor absmax** (one `sA` per forward) — simplest, single cuBLASLt A_scale.
   - Fallback if parity/accuracy needs it: **per-token (per-row) absmax** (cuBLASLt vector scaling). Per-token
     is the standard robust choice (llama.cpp's Q8_1 is per-32-block); start per-tensor, escalate if needed.

3. **Scratch buffers.** Reuse `ExecutionContext::getDeviceScratchBuffer` (grow-on-demand) for the FP8 weight
   staging (half the current BF16 staging) and the FP8 activation buffer. **Fetch at forward() time, never
   cache the pointer** (may be reallocated on grow — same rule as the FP4/FP8 staging today).

4. **Toggle.** Start with a compile-time `kUseFp8ActivationPrefill` (mirrors `kUseFusedFp4Gemm`) for clean A/B.
   Consider a runtime toggle later for one-build A/B (as noted for the fused kernel).

5. **cuBLASLt plan.** Add an FP8 plan variant to the linear plan cache keyed on `outer_size` (like the BF16
   plan). The FP8 descriptor needs: `CUBLAS_COMPUTE_32F`, `CUDA_R_32F` scale type, TRANSA=T/TRANSB=N,
   A_SCALE_POINTER=sA, B_SCALE_POINTER=sB, E4M3 A/B layouts, BF16 D. No fast-accum.

---

## 6. Validation plan (the gate)

Same oracle that already blesses FP4 weights — this change lives under it:

1. `Linear.Cuda` `Forward_MatchesReference` for `Linear<Cuda, BF16, PerGroupFp4<128>>`, `forward_atol 5e-2`,
   with `kUseFp8ActivationPrefill=true`. If per-tensor scales are too coarse, this is where it shows; escalate
   scale granularity (Section 5.1/5.2) until green — do NOT loosen the tolerance without cause.
2. **Gemma 4 12B token-for-token parity** vs the BF16 prefill path (the decisive end-to-end gate — activation
   FP8 is the one lossy step).
3. Chat coherence smoke (the real harness prompt).

If the oracle cannot be met at any practical scale granularity, that is the finding: FP8 *activations* are too
coarse for these models, and the lever is int8-with-per-block-scale or nothing. (Not expected — FP8 E4M3 is
the industry-standard inference activation format — but it is the honest failure branch.)

---

## 7. Performance expectation & how to confirm

- GEMM: ~2x (measured 1.90-2.11x). Staging traffic: ~half (FP8 vs BF16 weight staging).
- End-to-end prefill: **~1.95x-behind -> ~1.1-1.3x** vs llama.cpp @48K (attention + activation-quant don't
  halve). Confirm with `ProfileModel --model gemma --phase prefill --seq-len 22496 --context-length 49152
  --quantization fp4` with the toggle on vs off, against the 1056 tok/s baseline.
- **Re-measure the CURRENT binary** each time (a stale ProfileModel drove a whole session of wrong analysis —
  see [[feedback_save_per_result_not_per_session]]).

---

## 8. Implementation checklist (for the new session)

1. `cuda_fp4_dequantize_to_fp8` kernel (+ per-tensor/per-channel FP8 weight scale, computed at load).
2. `cuda_quantize_bf16_to_fp8` activation kernel + dynamic scale (per-tensor first).
3. cuBLASLt FP8 plan builder + plan-cache entry (TN, scales, E4M3->BF16, FP32 accum). Reference:
   `scratchpad/fp8_gemm_bench.cu`.
4. Wire the FP8 branch in `CudaLinearOp::forward` behind `kUseFp8ActivationPrefill` (prefill/batched only;
   decode matvec untouched).
5. Load-time: compute + store the FP8 weight scale(s) alongside the existing FP4 quantization.
6. Validate (Section 6). Escalate scale granularity only if the oracle demands it.
7. Profile (Section 7); keep the toggle OFF in the shipped default until it beats the BF16 path AND passes
   parity (same discipline as `kUseFusedFp4Gemm`).

---

## References

- Measured baselines, the stale-binary correction, and the FP8 microbench: [[project_w4a16_prefill_gemm]],
  worklog `WORKLOG.md`.
- Existing FP4/FP8 quant pipeline: `Quantization.V2.md`, `CudaLinearOp.ixx`.
- Reusable FP8 GEMM microbenchmark: `scratchpad/fp8_gemm_bench.cu` (nvcc -arch=sm_89, cuBLASLt).
