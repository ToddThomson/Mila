# Mixture of Experts

## Status

Design of record for Mila's mixture-of-experts path: the router, the stacked
expert bank, how they execute, how the bank is quantized, and what a model
whose bank does not fit could do about it. It extends `FfnAndMoE.md` §8/§9
(decision B), and it **corrects three assumptions** in that spec that reading a
real checkpoint falsified (Section 3).

**One model runs on it: Gemma 4 26B-A4B.** That model's own facts —
configuration, weight inventory, block topology, format, memory, chassis deltas
and the bar it is held to — are `Gemma.md` §10, the Gemma family's design. How
it was built and gated is the notebook `Notebooks/Gemma4MoE.md`. The next
families the path is meant for are Qwen3-30B-A3B and gpt-oss-20b
(`Mila/Issues/Future.md`, "Architecture / MoE").

**Split 2026-09-29.** Until then this document was both the MoE design and the
26B-A4B's. Sections 3, 5, 6 and 7.1-7.5 kept their numbers, which other
documents and code cite; the 26B-A4B's sections moved to `Gemma.md` §10, and the
notebook carries the map from the old numbers.

---

## 1. Scope

An MoE layer replaces one dense feed-forward network with a bank of `E`
equal-width experts, of which a router selects `top_k` per token, plus, in some
families, always-on shared experts. Everything in this document is
family-neutral; where a family decides something differently — how its router
normalizes, what its experts read, how its branches join — the family's design
owns it, as `Gemma.md` §10 does for Gemma.

---

## 2. Weight Layout

- **The bank is stacked.** Each projection is one tensor with a leading expert
  dimension, in PyTorch's `[experts, out, in]` order: `gate_up_proj`
  `[E, 2I, H]` and `down_proj` `[E, H, I]`, for model width `H` and expert width
  `I`.
- **`gate_up_proj` chunks gate first, then up**, and the activation applies to
  the gate half — the `[gate | up]` order `Linear`'s fused `fc_gate_up` already
  uses. An expert row is laid out exactly like a dense gated FFN's fused
  projection row, which is what lets the bank reuse `Linear`'s quantizers
  (Section 7).
- **Stacking is a converter capability, not a load-time step** (Section 3(a)).
  Gemma 4 ships its experts pre-stacked and loads them directly; a family that
  ships per-expert tensors is stacked by its converter.
- **Shared experts are not in the bank** (Section 3(c)).

---

## 3. Corrections to `FfnAndMoE.md`

Three §8/§13 statements did not survive contact with the Gemma 4 26B-A4B
checkpoint. They were recorded here rather than silently diverged from, and
`FfnAndMoE.md` §8 now points at this document.

**(a) "The converter / `WeightsReader` fuses E experts into the stacked
tensors at load time."** Not on Gemma 4's path — upstream already ships
stacked, so the load is a direct upload. The fusion path is still needed, but
for models that ship per-expert (Qwen3-30B-A3B, gpt-oss-20b), not for Gemma 4.
Stacking is therefore a **converter capability, not a load-time step**, and the
Gemma 4 path must not be built through a fuse-then-upload detour it does not
need.

**(b) "This maps directly onto the vendored CUTLASS grouped-GEMM kernels."**
**Correct as written, including for block-scaled FP4 on SM120.** An earlier
revision of this document claimed otherwise; that claim was the error, not §8.
CUTLASS is no longer vendored (Section 7.3); the kernels named here are
upstream's.

Verified in the tree Mila then pinned:
`include/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp` (`_array_`
is CUTLASS's name for the Ptr-Array/grouped variant), plus
`builders/sm120_blockscaled_mma_builder.inl`. The official example is
`examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu`
with `ArchTag = cutlass::arch::Sm120`, present since CUTLASS 3.9.0 and in every
version Mila has pinned. The field reports of garbage output on SM120 trace to
the **arch flag** (`compute_120a` versus `compute_120f`), not to a missing or
defective kernel — Section 7.2(b).

**CUTLASS remains the only library route for a grouped block-scaled FP4 GEMM.**
cuBLASLt has no grouped/varying-shape matmul in CUDA 13.3 or 13.4
(`cublasLtGemmGroupedBatchedEx` appears in documentation ahead of both headers),
so unlike the dense path in Section 7.1 there is no library alternative here.
**Gemma 4 26B-A4B does not take that route:** its experts are Q4_0
(`Gemma.md` §10.4), which no CUTLASS block-scaled collective serves, so its
grouped prefill is Mila's own INT8 kernel (Section 7.7).

**(c) §13 open decision, "whether shared experts are `GatedMLP` instances
composed beside the router or a distinct always-on path."** **Decided:** the
always-on branch is an ordinary `GatedMLP` composed beside the router, with its
own pre- and post-norms. It is not an entry in the expert bank, it is not
routed, and it must not be folded into the grouped path — its width (2112 in
Gemma 4 26B-A4B) differs from an expert's (704), so folding it would break the
uniform expert extent the grouped kernel depends on.

---

## 4. Routing

**Routing is a first-class operation** (`RouterOp`), not a subroutine buried in
a family's kernel. Its output — expert indices (INT32) and combine weights — is
the interface `MoeOp` consumes, and it is the seam a second MoE family plugs
into without touching the kernel.

- **`RouterOp` is selection only:** from `[N, E]` router logits to `top_k`
  indices and weights. What precedes it — a norm, a scale, the projection — is
  composed in the `Router` component from existing components, so a family's
  routing chain is wiring, not a new kernel.
- **The order of softmax, selection and renormalization is the family's.**
  Gemma takes the softmax over all `E` before selecting and renormalizes after
  (`Gemma.md` §10.3); a family that softmaxes over the selected experts only is
  a different `RouterOp` behaviour, and must be matched to its reference, not
  assumed.
- **Ties select the lower expert index**, and the gate compares the selected
  *set*, never its order: the combine is a sum over the selected experts, so any
  order is correct (`Gemma4MoE.md` Phase 5).
- **The router is a precision holdout**: its projection is never quantized
  (Section 7.5).

---

## 5. Component Decomposition

The chassis does not change. A family's block keeps its attention half and its
heterogeneous-layer machinery (`ITransformerBlock`); the MoE layer replaces the
feed-forward slot. Gemma's wiring, where every layer is an MoE layer with a
dense branch beside it, is `Gemma.md` §10.3.

New components:

| Concept | Component | Config | Operation |
|---|---|---|---|
| Routing | `Router` | `RouterConfig` | `RouterOp` |
| Expert bank + combine | `MixtureOfExperts` | `MixtureOfExpertsConfig` | `MoeOp` |

**`Router` is a sibling of `MixtureOfExperts`, not its child** (decided in Phase 6,
`Gemma4MoE.md`). Gemma routes on the raw residual while its experts read
`pre_feedforward_layernorm_2` of it, so a bank that owned its router would need a
family-specific two-input signature. `MixtureOfExperts` takes the router's weights and
indices as input, and the block wires the two.

`MixtureOfExperts` orchestrates; `MoeOp` holds the kernel. This is the division
`Linear` established and `OperationDispatch.md` requires. There is **no**
`Expert` component and **no** `ExpertBank` — an expert is a row of a stacked
tensor, never an object. `GatedMLP` remains the single-expert semantics and
the correctness oracle the grouped path is validated against, exactly as
`FfnAndMoE.md` §8 specifies.

---

## 6. Execution Model

Two paths, split on `M`, mirroring the prefill-GEMM / decode-matvec division
`Linear` already carries.

**What runs today:** a Q4_0 bank runs both paths below -- the gather-matvec for
one token (`MoeGather.cu`), the grouped GEMM for more (`MoeGrouped.cu`). An FP4
or unquantized bank runs the Phase 6 two-pass kernel (`Moe.cu`) for every
forward, one thread per output value, reading each token's selected expert rows
in place: the correctness baseline, gated against HuggingFace, and not a GEMM.

**Prefill (`M > 1`) — grouped GEMM.** Tokens are permuted into expert-major
order, one segment per expert, and a grouped GEMM walks the segments against
`experts.gate_up_proj`. Segment extents are data-dependent, so the launch is
sized from a device-side histogram of the routing result. Which kernel serves it
follows the bank's format (Section 7): at Q4_0 it is the INT8 prefill `Linear`
already runs (`Quantization.md`, Q4_0 decision 4) made grouped (Section 7.7).

Built for Q4_0, in five steps on the stream, none reading back to the host:

1. **Routing**, one block: rows counted per expert, then each placed at its
   expert's offset plus the count of earlier rows routed there, so a segment
   lists its rows in ascending (token, slot) order and the permutation does not
   depend on thread timing.
2. **Tile table** for a token range: each expert's rows in that range, cut into
   M tiles of 64 -- half `Linear`'s, since a 1024-row chunk averages 64 rows an
   expert on the 26B-A4B. The launch is sized to the bound
   `rows / 64 + experts`; blocks past the table's count exit.
3. **Gated GEMM**: the input quantized to INT8 per 32-element block, as
   `Linear`'s, then each tile's gathered rows against its expert's gate and up
   rows **interleaved** in the B tile, so a thread's two adjacent columns are
   one unit's gate and up and the activation runs on registers. Out: FP32 gated
   values.
4. **Gated values to INT8**, the same quantizer, into the scratch the input's
   codes held.
5. **Combine GEMM and reduction.** Each (token, slot)'s down projection in FP32,
   then each token's rows summed in slot order, weighted, to BF16. The rows are
   written over the gated buffer, dead once quantized, **as many tokens at a
   time as it holds** -- a quarter of a chunk on the 26B-A4B, whose hidden width
   is four times the expert width -- so the down weights are read once per
   pass. Materializing every row at once would take 92 MB at 1024 rows, more
   than the about 60 MiB the plan has spare at 8192 (G5), so the planner would
   narrow the chunk.

Every output depends on its own row alone, so the result is bit-identical
however tokens are batched. Scratch beyond the pooled slots: the routing, the
tile table and the INT8 codes, 6.5 MB at 1024 rows on the 26B-A4B, under the
10 MiB the network already reserves.

**Decode (`M == 1`, `outer_size == 1`) — gather-matvec.** A single token visits
`top_k` of `E` experts. Permutation, segmentation and grouped launch all cost
more than the arithmetic they organize. The decode path indexes the selected
expert rows directly out of the `[E, ...]` tensor and runs a fused
gather-matvec, never materializing gathered weights. It is memory-bound on the
selected rows, so its floor is the active expert bytes over the card's
bandwidth — about 1.8 ms per token for Gemma 4 26B-A4B on the RTX 5060 Ti
(Section 8). Built for Q4_0: the gated pass is one warp per (slot, unit), the
combine one warp per output column over every slot, each lane a whole 32-code
group at a time; 2.27 ms a token on the 26B-A4B (`Gemma4MoE.md`, "G5b decode
result").

This is the decode case Mila actually ships. The prefill path is the one that
touches enough weight mass for a GEMM to matter.

**A decode step is recorded once and replayed** (`DecodeGraph.md`), so the router
and the expert bank obey its contract: no per-token value as a launch argument,
no host state a later call reads, no allocation or device-to-host read after the
first call. `DecodeGraph.md` §3 traced the routed decode path and found no
per-token dependency, and `GemmaModel` turns replay on for every load, the 26B
included. **No test replays a routed network** — `DecodeReplay.Cuda.cpp` covers
Llama, a dense Gemma and Qwen — so the contract is traced on this path, not
gated. Any kernel that replaces the Phase 6 one inherits the contract.

---

## 7. Quantization

The expert bank is most of an MoE model's parameters — 91% of Gemma 4
26B-A4B's — so it is where quantization matters. **The bank carries `Linear`'s
weight policies with `Linear`'s row layout** (Section 2), so a stacked tensor is
`E x rows` output channels of the existing quantizer, and quantize-on-load and
`ExportArtifact` share each policy's one rounding rule.

**The format is the model's, not the bank's.** A producer's
quantization-aware weights run in the format they were trained for
(`ModelFamilyParity.md` §9 item 14); for Gemma 4 26B-A4B that is Q4_0
(`Gemma.md` §10.4). The grouped kernel follows the format:

| Format | Grouped prefill | Where |
|---|---|---|
| Q4_0 (`PerGroupInt4<32>`) | Mila's own INT8 grouped GEMM | Sections 6, 7.7 |
| `PerGroupFp4<g>` | the Phase 6 two-pass kernel today | Section 7.6 |
| NVFP4 | CUTLASS's SM120 block-scaled grouped GEMM | Sections 7.1-7.5; not a policy yet |

Sections 7.1 to 7.5 were written when the 26B-A4B was to land on Mila's FP4 and
move to NVFP4 next. Their measurements stand and other specifications cite them,
so they are kept under their numbers, but **none of it is on the 26B-A4B's
path**: NVFP4 needs FP4 activations, which that model does not run, and
CUTLASS's SM120 grouped GEMM is block-scaled FP4, which Q4_0 is not. They are the
record for a future NVFP4 policy on any model.

### 7.1 NVFP4 as a sibling policy

NVFP4 is **not** `PerGroupFp4<128>` with different constants. It is block-16
E2M1 with an **FP8 E4M3 scale per block** plus a per-tensor FP32 global scale —
a different storage format, so it is a **new `WeightQuantPolicy`**, not a
retuning of the existing one. `PerGroupFp4<128>` (FP32 per-group scales) stays
exactly as it is.

The seam already exists. `Policies.ixx` grew **companion-tensor traits** so
`PerGroupCodebook3` could carry its third-bit plane, with structural detection
and `if constexpr` guards at every consumer. NVFP4's FP8 block-scale plane is
the same shape of problem and rides the same mechanism. This is a real fit, not
an analogy.

Per `Quantization.md`, quantization is **offline**: `Tools/ExportArtifact` is
the only writer, weights declare their policy in
`__metadata__["mila_quantization"]`, and a load refuses a policy that is not
the compiled one. NVFP4 adds an export path and traits rows; it does not add a
runtime mode.

### 7.1a The DENSE path needs no kernel work at all — measured

**cuBLASLt already performs native NVFP4 GEMM on SM120**, and the numbers are
not marginal. `Mila/Profiling/Microbenchmarks/CublasLtScaleModes.cu`, both
cards, CUDA 13.3, Gemma 4 12B prefill shapes at M=1024:

| arm | 4070 (SM 8.9) | 5060 Ti (SM 12.0) |
|---|---|---|
| BF16 reference | 52.5-60.3 | 50.8-51.3 |
| FP8 + `SCALAR_32F` (ships today) | 103.9-119.9 | 183.0-197.2 |
| FP8 + `OUTER_VEC_32F` | no algorithm | no algorithm |
| FP4 + `VEC16_UE4M3` | no algorithm | **329.3-361.3 TFLOP/s** |

361 against the `mxf4nvf4` instruction ceiling of 415.6 is **88% of peak**, and
**1.8x the FP8 path Mila ships**. The mechanism is one descriptor attribute —
`CUBLASLT_MATMUL_DESC_A_SCALE_MODE = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`
with A and B as `CUDA_R_4F_E2M1` — not a kernel project. Every mode that matters
is in **CUDA 13.3**; 13.4 adds only packed MX-style layouts Mila does not use.

Two consequences that reshape the rest of this section:

- The dense expert GEMM, the shared-expert GEMM and every `Linear` in the model
  reach NVFP4 through the library. **CUTLASS is needed only for the grouped
  path** (Section 3(b)), where cuBLASLt has no equivalent.
- The remaining risk is **numerics, not throughput**: this is W4A4, and
  per-tensor FP8 activations already produced incoherent Gemma once. Block-16
  scaling is a far finer instrument than what failed, so it is open rather than
  lost — but it gates on token parity and coherent generation, never on the
  per-layer tolerance alone.

Also measured: `OUTER_VEC_32F` returns no algorithm on **either** card, so the
per-token scale epilogue cannot be folded into the GEMM that way.
`Fp8ActivationPrefill.md` attributes that limitation to Ada; it is not an Ada
limitation.

### 7.2 The SM120 question

The hardware is not in doubt, and Section 7.1a settles the dense path. What
remains in doubt is the **toolchain** for the grouped path, where three separate
issues are routinely conflated into one. They are not the same problem and they
do not have the same answer.

**(a) `tcgen05` is absent from SM120 hardware — not a toolchain restriction.**
The CuTeDSL `tcgen05` MMA ops (`python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py`)
admit only datacenter Blackwell: `BlockScaledMmaOp` lists `sm_100a`, `sm_100f`,
`sm_103a`, `sm_107a` and `sm_110a`. That list reflects the silicon — SM120 and
SM121 have no TMEM / UMMA, so no API, Python or C++, can emit `tcgen05.mma` for
them, and the DSL's own error redirects to the warp-level ops
(`cute/nvgpu/warp/mma.py`, admitting only `sm_120a/f` and `sm_121a/f`). Block-scaled
FP4 on SM120 is reachable, but through `mma.sync` (Section 7.3), and that is what
the C++ SM120 paths emit. An earlier revision of this spec read the arch list as
a Python-DSL-only restriction the C++ API escapes; it is not.

**(b) The build flag is a live trap, and Mila is currently in it.** Plain
`sm_120` does **not** carry the block-scaled MMA capability; it needs the `a`
(arch-conditional) or `f` (family-conditional) suffix. The failure mode is the
dangerous one — not a compile error but a silent omission, with the assertion
surfacing far downstream. PyTorch shipped exactly this bug by stripping the
suffix during arch auto-detection.

Mila sets `120` — no suffix — in `x64-release-blackwell`, in the library's own
`MILA_LIBRARY_CUDA_ARCHITECTURES` default, and in all four published-artifact
lists. The library default is the one most easily missed. Nothing is
broken today (the preset description correctly notes nothing in Mila emits
sm_120 instructions yet), but this is **step 0** of any NVFP4 work, not a
detail to discover later.

Which suffix: dense block-scaled MMA is **family**-specific and available via
`120f` from PTX ISA 8.8; only **sparse** `mma.sp` with `.kind::mxf4nvf4`
requires arch-specific `120a`. Mila needs dense, so `120f` suffices and is
forward-compatible across the whole 12.x family — the better choice for the
published-artifact lists, where `120a` would pin to one architecture. This rig
has **CUDA 13.4** and **CMake 4.0.1**, both of which support the `f` suffix, so
this is expressible today with no toolchain upgrade.

**(c) CUTLASS grouped block-scaled GEMM on SM120 — present and supported; the
field failures are (b) in disguise.** The collective ships in every CUTLASS
version Mila has pinned, and `79d_blackwell_geforce_nvfp4_grouped_gemm.cu` is
an official SM120 example (Section 3(b) lists the verified paths). The reports
of garbage output are builds using `compute_120a`, where the TMA
warp-specialized tactics fail to initialize; `compute_120f` restores them,
measured **14.6 -> 39.0 tok/s** on an NVFP4 MoE model.

An earlier revision of this document called this a broken template with a
non-upstream fix. That was wrong: it is the **same arch-flag trap as (b)**,
which is the third place in this section where that flag turned out to be the
actual story. Read (b) as the root cause and this as a symptom.

Note that those numbers are from **RTX PRO 6000** (188 SMs, 96 GB). The 5060 Ti
has 36 SMs. Do not transfer the throughput.

CUTLASS's documented SM120 constraints do apply and shape any grouped kernel:
**TN layout only** (A row-major, B column-major), **cluster fixed at 1x1x1**
(no multicast), NVFP4 tile shapes limited to `128x128x128`, `256x128x128` and
`128x128x256`, and `EpilogueScheduleAuto` mandatory. The 128-row minimum tile
is an independent reason the decode path cannot use the grouped kernel: at
`M == 1` it would waste 127 of 128 rows, which is Section 6's gather-matvec
split arrived at from the kernel side.

### 7.3 What Mila writes, and what it does not

CUTLASS is not in the build. It was fetched from `e75938f3` to `rc.1+23` with
nothing including it, and removed at `rc.1+24`. The absence is not a design
position: CUTLASS **returns in the commit that adds the first kernel built on
it**, behind an off-by-default `MILA_ENABLE_CUTLASS` option, pinned to the
CUTLASS release current at that time. That is no longer the 26B-A4B's grouped
prefill, which is Mila's own INT8 kernel (Section 7.7). Checked 2026-09-29:
CUTLASS 4.8.0 is current, and PR #3082 (`is_family_of()` for the SM12x guard
in `MmaSM120BlockScaledOp`) was closed unmerged. Whether Section 7.2(c)'s `120f`
route builds and runs on 4.8.0 without it has not been tried in Mila; it is the
first thing an NVFP4 grouped kernel must establish. For an NVFP4 world the split
after Section 7.1a is clean:

- **Dense GEMM — the library.** cuBLASLt reaches 88% of the instruction ceiling
  on NVFP4 and 88-96% on FP8. Mila's own history says what happens to a hand
  kernel here: the WMMA FP4 GEMM measured 4x slower at Stage 1 and 1.1-2x
  slower at Stage 2, both against cuBLASLt. Do not re-litigate this.
- **Grouped GEMM — CUTLASS.** No library alternative exists (Section 3(b)).
- **Hand-written — only where neither serves.** The decode gather-matvec, whose
  shape no GEMM library expresses and which the 128-row minimum tile rules out
  of the grouped kernel anyway.

Useful context if a hand-written kernel is ever reached for: SM120 uses
warp-level `mma.sync`, **not** `tcgen05`/TMEM, so every SM100-derived kernel
(DeepGEMM, CUTLASS SM100 collectives, WGMMA flash attention) fails to compile
or crashes — while the SM8x programming model Mila's kernels already use ports
directly. SM120 also has **99 KB shared memory per SM** against SM100's 228 KB,
so inherited SM100 tile designs do not fit. The instruction is fixed-shape:
`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`,
quad-based scale distribution, published SM12x references at ~60% of peak —
which is the number to weigh against cuBLASLt's 88% before writing anything.

**The decision NVFP4 now gates on is numerics, not throughput.** Throughput is
measured and favourable. Validate W4A4 against Gemma token parity and coherent
generation before adopting it; if FP4 activations cannot hold, the dense path
stays on `PerGroupFp4<128>`, and the MoE path is unaffected.

### 7.4 Decoupling

MoE and NVFP4 are **two milestones, not one**. An MoE model lands in the format
its producer trained — Q4_0 for Gemma 4 26B-A4B (`Gemma.md` §10.4) — and is
complete and shippable there. NVFP4 is a policy addition that any `Linear` and
the `MoeOp` can adopt afterwards, on a model whose producer ships no
quantization-aware format, or as Mila's own fallback. Whether FP4
**activations** hold their quality is a per-model numerics investigation with
its own oracle runs, and nothing about the router, the expert bank or the
grouped dispatch depends on its outcome.

---

### 7.5 The NVFP4 pipeline — the activation half is what remains

Mila is most of the way to an end-to-end NVFP4 pipeline already. Stating what
ships is the point of this section, because the remaining work is far narrower
than "add NVFP4" suggests and the architecture it plugs into is the one Qwen 3.8
established.

**Shipped:**

- **Quantized weights, several formats.** `PerGroupFp4<128>` (E2M1 with an FP32
  per-group scale), `PerChannelFp8<>`, `PerGroupCodebook2/3`. Offline through
  `Tools/ExportArtifact`, declared in `__metadata__["mila_quantization"]`, and a
  load refuses a policy that is not the compiled one.
- **Companion-tensor traits.** `Policies.ixx` already serves a format whose
  storage spills into a second plane, structurally detected, `if constexpr` at
  every consumer. NVFP4's block-scale plane needs no new mechanism.
- **Per-role bit allocation.** `WeightQuantization::Plan` and
  `Qwen.PrecisionPlan.ixx` — a family's designed allocation read from a
  pre-quantized artifact, with no quantize-on-load path because the codes are
  fitted offline.
- **Runtime activation quantization.** `cuda_quantize_bf16_to_fp8_per_token`
  produces FP8 E4M3 with per-token absmax scales, shipped on and gated green.
- **The GEMM.** Measured at 329.3-361.3 TFLOP/s for FP4 x FP4 with
  `VEC16_UE4M3` (Section 7.1a). No kernel to write.

**What remains is one kernel and one vocabulary extension.**

**(a) An activation quantizer at block-16 granularity.** The existing quantizer
emits one E4M3 scale per token; NVFP4 wants `e2m1` values with a `ue4m3` scale
per 16 elements along K. Same shape of kernel — one pass over the activations,
absmax then quantize — at a different granularity, writing the scale plane
cuBLASLt expects. The layout is a contract, not a choice: the scale tensor must
match what `VEC16_UE4M3` reads, and getting it wrong produces wrong numbers
rather than an error, so it needs a CPU reference codec the way
`CodebookPacking.ixx` has one.

The cost trade is favourable and worth recording, because it inverts the usual
objection to activation quantization. The pass this **deletes**
(`dequantize_fp4_to_fp8`) is `O(weights)`; the pass it **adds** is
`O(tokens x hidden)`. At a 512-token prompt that is the difference between
51.6% of prefill and something near noise — the weight dequant is expensive
precisely because it is paid per forward regardless of how few tokens are in
flight.

**(b) Plans must allocate activation precision, not only weight bits.**
`LanguageModelConfig` is explicit that "a plan allocates WEIGHT bits", and that
is the line NVFP4 crosses. A uniform W4A4 model is the aggressive variant; the
useful one holds the layers that carry accuracy at higher precision and runs the
rest at W4A4. That is the same idea a precision plan already encodes, extended
one axis. The expert bank is 91% of the 26B-A4B's parameters and the obvious
W4A4 candidate; the router, the norms and the embedding are the obvious
holdouts.

**What has to be validated, and why the FP8 result does not predict it.**
Per-tensor FP8 activations failed here once: one outlier token set the tensor
scale, crushed every other token's resolution, and the error compounded over 48
layers. That is an argument about **granularity**, and NVFP4 moves granularity
by three orders of magnitude — one scale per 16 elements against one per tensor.
It is a reason to expect a different outcome, not the same one. The gate is
unchanged and is not the per-layer tolerance: **token parity against the
reference, and coherent generation**, because 30 layers of top-8 routing
compound and a 5e-2 per-layer pass has already proven it can hide that.

Sequencing note: (a) is independently useful. An NVFP4 activation quantizer plus
the `VEC16_UE4M3` GEMM is a **dense-model** win on every `Linear` Mila has, on
any Blackwell card, with no MoE machinery involved. It does not need an MoE
model to land.

### 7.6 The bank's formats today

`MixtureOfExperts` and `MoeOp` take `TWeightQuantization`, and the bank runs
unquantized, at `PerGroupFp4<g>` (`Gemma4MoE.md` Phase 8) or at
`PerGroupInt4<32>`, Q4_0 (Phase 9). Under FP4 each pass of the Phase 6 kernel
decodes the selected expert's codes against their group scales inline; nothing
is dequantized ahead of the matvec. Q4_0 has its own kernels (Section 7.7).
Which format a policy is comes from two concepts beside the
policies, `HasFp4E2M1Codes` and `HasInt4Codes`, which the bank, `CudaMoeOp` and
`CudaLinearOp` all ask. `CudaMoeOp` refuses every other policy at construction,
FP8 included, and an expert width that is not a multiple of the group at build.

### 7.7 Q4_0 on the bank

The bank is built (`Gemma4MoE.md` Phase 9, `0.21.0-dev+24`), and G5b gives it
kernels at speed (`Gemma4MoE.md`, "G5b decode result" and "G5b prefill
result"). Every kernel in it is Mila's own:

- **The bank has `PerGroupInt4<32>`**, laid out and rounded as `Linear`'s Q4_0
  (Section 7's lead, `Quantization.md`, Q4_0), quantized on load by `Linear`'s
  INT4 quantizer over the stack's `E x rows` output channels.
- **Prefill** is `Linear`'s Q4_0 INT8 path — activations to INT8 per 32-element
  block, a k32 MMA against the packed codes — made grouped over the expert
  segments of Section 6; the two share their tile copies, fragment unpack and
  MMA through `Int4Int8Mma.cuh`.
- **Decode** is the Q4_0 matvec (code times BF16 activation, summed in FP32 per
  block, times the scale) in gather form over the selected rows, measured
  against the floor of Section 8: 2.27 ms a token against 1.8.
- **The router** is one warp per row: top-k by k rounds of a warp argmax, and
  only the selected experts' exponentials, since the softmax denominator cancels
  in the renormalization. 7 us a layer at decode, where Phase 5's one thread per
  row took 53.

---

## 8. Expert Residency — an Open Design Axis

Every fit analysis so far assumes the whole expert bank is resident. That
assumption is worth challenging rather than inheriting, because the **working
set is far smaller than the resident set**: in Gemma 4 26B-A4B, top-8 of 128
means a token touches 6.25% of each layer's experts.

The arithmetic bounds it before any design work. Per decode token, 8 experts x
5,947,392 parameters x 30 layers = **1.43B parameters**, or **~0.80 GB at 4.5
bits** -- four bits of code and half a bit of scale per weight, which both
`PerGroupFp4<64>` and Q4_0 carry; the 26.8 MB a layer's eight experts read was
measured (`Gemma4MoE.md`, Rates Baseline). The 5060 Ti negotiates **PCIe Gen5 x8
(~31.5 GB/s)** — measured, and better than the 4070's Gen4 x4 (~7.9 GB/s), which
corrects a note that had the slots the other way round. That puts a hard ceiling
of ~25 ms per token, so **~39 tokens/s if every active expert is streamed every
token** — against a resident bank's floor of about 1.8 ms for the same 0.80 GB at
the card's 448 GB/s. Streaming everything is therefore not a free lunch; it is
about fourteen times the resident floor, before any compute. (Until 2026-09-29
this paragraph counted the codes alone, 0.71 GB and 1.6 ms.)

Three things decide whether a partial-residency design beats that ceiling, and
none is answered here:

1. **Routing locality.** A resident cache of hot experts only pays if the hit
   rate is high. Nothing in this document knows Gemma 4's routing entropy, and
   it is measurable offline from the router weights plus a corpus — cheap, and
   it gates the whole idea.
2. **The prefetch window is closed by a serial dependency.** Layer N+1's router
   consumes layer N's output, so the next layer's expert set is not knowable
   while the current layer computes. There is no cross-layer prefetch without
   speculating on the routing, which is a different and larger design.
3. **Prefill does not benefit.** A 1024-token chunk with top-8-of-128 touches
   essentially every expert in every layer, so the full bank must be resident
   during prefill regardless. Residency streaming is a **decode-only**
   technique, and it therefore cannot solve a fit problem — only the
   steady-state one.

The honest framing: this is an axis for running a model that **otherwise would
not load at all**, not an optimization for one that fits. Decide it after a
model's fit analysis (`Gemma.md` §10.5 for the 26B-A4B), not instead of it.

`MemoryStats` carries the **sparse-layer term** — resident-but-inactive
parameter bytes (`MemoryFootprint.md` §4.6, `Gemma4MoE.md` Phase 4) — and
`GemmaModel`'s footprint reports it for the routed model (Phase 8's entry-point
gate). **Nothing a user sees reports it yet**: neither the deployment plan nor
Chat's footprint display carries an active figure. It is a prerequisite of
publishing an MoE model, not a follow-up: a footprint report that counts an MoE
layer as dense is wrong by a factor of 14 on the only number a user checks
before loading. A residency design would need a second term on top, separating
resident from active from streamed.

---

## 9. What a Second Family Reuses

Of what Gemma 4 26B-A4B added (`Gemma.md` §10.6), these are family-neutral and
serve Qwen3-30B-A3B and gpt-oss-20b unchanged: `Router` and `RouterOp`, with a
family's own routing chain composed around the selection (Section 4);
`MixtureOfExperts` and `MoeOp`, both paths of Section 6; the sparse-layer
footprint term (Section 8); and the bank's quantization policies (Section 7).
What each new family brings is its routing order, its block wiring, and — for a
family that ships per-expert tensors — a converter that stacks them.

---

## 10. Validation

The oracle discipline is unchanged: token-for-token against HuggingFace on the
same weights, per `Testing.md`.

- `MoeOp` is validated against a loop of `GatedMLP` over the same stacked
  weights — the `FfnAndMoE.md` §8 oracle — or, on CPU, where `GatedMLP<Cpu>`
  does not compile, against a definition in the test and HuggingFace's eager
  experts (`Gemma4MoE.md` Phase 6).
- `Router` is validated against the HF routing chain on fixed hidden states;
  index agreement is exact, weight agreement is `atol`-bounded. **A
  must-differ threshold here is absolute, never a multiple of `atol`.**
- Routing is discrete, so a near-tie that flips one expert selection changes
  the output materially while the norms stay small. Generation must be
  validated, not just the oracle — layers of top-k routing compound.
- The decode gather-matvec is validated against the prefill path at `M == 1`,
  which is the only comparison that isolates it. Today both are the one two-pass
  kernel, and the gate is bit-identity (`Gemma4MoE.md` Phase 7); a grouped GEMM
  prefill is gated against that kernel when it lands.
- **Short-prompt greedy tokens need not see a routing error.** On the 26B-A4B a
  router fed the wrong input still generated HuggingFace's eight tokens at FP4;
  only the layer-streamed BF16 hidden-state gate failed (`Gemma4MoE.md` Phase 8).
  A publish gate for an MoE model must be shown to fail on a routing edit.

What a published MoE model must additionally hold is its family's bar — for the
26B-A4B, `Gemma.md` §10.7.

---

## 11. Open Decisions

- **Whether the grouped prefill reads its segment extents back to the host.** A
  launch sized on the host costs a synchronization per layer; one scheduled on
  the device does not. Decided with the kernel.
- Load-balancing auxiliary loss is **out of scope** — Mila does not train an MoE
  model, and routing weights ship fitted.
- Expert parallelism across devices remains out of scope per
  `FfnAndMoE.md` §13; the stacked layout must not preclude it and need not
  enable it.
