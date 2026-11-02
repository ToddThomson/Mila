# Microbenchmarks

Standalone measurement tools. **Not part of the CMake build** — each is one translation
unit compiled by hand with `nvcc`, because they target specific architecture flags that no
preset carries and they are run deliberately, not on every build. Nothing here ships.

They exist to answer questions *before* committing to a kernel, which is the expensive
mistake: `ProfileModel` tells you where a real model spends its time, and these tell you
what the hardware and the vendor libraries are capable of in the first place.

---

## The build-flag trap, before anything else

Blackwell's block-scaled MMA instructions need the **family-conditional** target. Get this
wrong and there is no warning:

```
nvcc -arch=sm_120f ...      # WRONG -- silently emits .target sm_120
```

That drops the `f`, and ptxas then rejects every block-scaled instruction with
`Feature '.block_scale' not supported on .target 'sm_120'`. The form that works:

```
nvcc -gencode=arch=compute_120f,code=sm_120f ...
```

Dense block-scaled MMA is family-specific, so `120f` suffices and is forward-compatible
across all of 12.x; only sparse `mma.sp` needs the arch-specific `120a`. Mila's own presets
carry bare `"120"`, which is fine only while nothing in the tree emits these instructions.

**Pin the card by UUID**, never by index — the CUDA ordinal does not match `nvidia-smi`:

```
CUDA_VISIBLE_DEVICES=GPU-<uuid> ./MmaInstructionPeak.exe
```

---

## `MmaInstructionPeak.cu`

Back-to-back `mma.sync` from registers — the **instruction ceiling**, not a GEMM. An upper
bound no tiled kernel can exceed.

```
nvcc -gencode=arch=compute_120f,code=sm_120f -O3 MmaInstructionPeak.cu -o MmaInstructionPeak.exe
```

Carries a deliberate control arm: `mxf8f6f4` is measured with `e4m3 x e4m3` **and**
`e2m1 x e4m3`. If those differ, operand width affects throughput; if they match, the gain is
the datapath. Keep the control when adding arms — it is what stops a datapath win being
misattributed to a format.

`kChains` / `kIters` are the knobs. Raising chains and lowering iterations distinguishes an
issue-limited result from an ILP-limited one.

## `CublasLtGemmThroughput.cu`

What cuBLASLt **actually banks** at real prefill shapes, DRAM-resident, BF16 vs FP8.

```
nvcc -gencode=arch=compute_120f,code=sm_120f -O3 CublasLtGemmThroughput.cu -o CublasLtGemmThroughput.exe -lcublasLt
```

This is the arm that matters for any "should we hand-write a GEMM?" question. Pair it with
`MmaInstructionPeak`: the gap between the two is the only headroom a hand-written kernel can
win, and the vendor library is usually closer to the ceiling than expected.

## `CublasLtScaleModes.cu`

Which scale modes the card actually accepts, and what they cost.

```
nvcc -gencode=arch=compute_120f,code=sm_120f -O3 CublasLtScaleModes.cu -o CublasLtScaleModes.exe -lcublasLt
```

Mila's FP8 prefill runs `dequantize_fp4_to_fp8` → GEMM → `apply_per_token_scales`, and both
flanking passes exist only because the GEMM could not be told about the scales. This asks the
heuristic, per card, whether that is still true. Measured 2026-09-12:

| arm | 4070 (SM 8.9) | 5060 Ti (SM 12.0) |
|---|---|---|
| BF16 | 52.5–60.3 | 50.8–51.3 |
| FP8 + `SCALAR_32F` (ships) | 103.9–119.9 | 183.0–197.2 |
| FP8 + `OUTER_VEC_32F` | no algorithm | no algorithm |
| FP4 + `VEC16_UE4M3` | no algorithm | **329.3–361.3** |

**cuBLASLt does native NVFP4 on SM120 at 88% of the instruction ceiling, in CUDA 13.3.** No
CUTLASS, no hand-written kernel. `OUTER_VEC_32F` is available on neither card.

Run it on **both** cards — the Ada/Blackwell split is the whole point, and a Blackwell-only
result is a per-architecture path rather than a replacement.

*Harness note:* the BF16 arm must not have a scale mode set, or cuBLASLt reports NO ALGORITHM
and the table claims the card cannot do BF16. That guard is in the code; do not remove it.

## `CudaContextFloor.cu`

Splits "what the CUDA driver reserves" from "what Mila adds", stage by stage.

```
nvcc -gencode=arch=compute_120f,code=sm_120f CudaContextFloor.cu -o CudaContextFloor1Arch.exe -lcublasLt
```

Build it a second time with a multi-arch gencode list to price the fatbin. Use it before
concluding that a model "does not fit" — VRAM arithmetic is easy to get wrong from the
outside, and a model's file size is **not** its resident size.

## `RotateOnRead.cu`

Whether a decode step can build Gemma's global keys from a cache that holds only V (`RopeInAttention.md` section 5,
fact 2): decode's tile loop over one global layer, today's two tensors against V alone with 64 pairs rotated per key --
by `sincosf`, by an angle recurrence, or from a table.

```
nvcc -gencode=arch=compute_120,code=sm_120 -O3 RotateOnRead.cu -o RotateOnRead.exe
```

No MMA competes for issue in it, so its K = V margin is an upper bound on the real kernel's. Measured 2026-10-01 on
the RTX 5060 Ti: at 65536 positions `sincosf` runs 0.69 of today's time, the recurrence 0.67, the table 0.91.

## `AllocationLayout.cu`

Whether packing tensors into fewer device allocations costs anything: the same tensors as one allocation each, packed
into slabs, as one block, and in one reserved address range mapped a chunk at a time. Each arm is allocated, read with
no pressure, read while a second process holds memory the card does not have spare, and read again after it frees it.

```
nvcc -gencode=arch=compute_120,code=sm_120 -O3 AllocationLayout.cu -o AllocationLayout.exe -lcuda
AllocationLayout.exe <tensor list> 3072 20 <arm>
```

The tensor list is `layer bytes` per line, written from a weights file. **Run one arm per process:** an arm run after
another under pressure inherits whatever the driver moved to system memory, which once read as a layout effect.

Measured 2026-10-03, RTX 5060 Ti, Windows (WDDM), Qwen 3.8 27B FP4's 1,203 device tensors (13,019 MiB), both
orders: every arm reads at 381-384 GB/s quiet, under pressure and after. One allocation per tensor consumes 13,540
MiB; one block 13,044; a mapped range 13,044 at 2 MiB chunks (twice the placement time) and 13,054 at 64; slabs of
256 MiB 13,150. The pressure process's 3 GiB never displaced the resident arm: free memory did not fall, and its
allocation went to system memory instead. On this driver the allocation made when the card is full is the one that
spills, whatever the layout.

## `VerifyRows.cu`

What a speculative verify of R rows costs against one decode, on the Gemma 4 12B's Q4_0 Linear shapes
(`Gemma4Mtp.md` 4.7): today's decode matvec copied verbatim, the same matvec with R accumulators (bit-identical per
row), and a tensor-core product with the rows as the `mma` n = 8 operand, in six tile/warp configurations.

```
nvcc -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_89,code=sm_89 -O3 VerifyRows.cu -o VerifyRows.exe
```

Every arm is checked against an FP64 reference and against the one-row decode before it is timed. Measured
2026-10-05: per token, R = 5 rows cost 1.91 decodes as a matvec and 1.01 as the product on the RTX 5060 Ti (1.78
and 1.06 on the 4070); the product's R = 1 is 1.01x decode. Table in `Gemma4Mtp.md` 4.7.

A `cute` arm compiles when CUTLASS's include directory is on the path (`-std=c++17 --expt-relaxed-constexpr
-Xcompiler=/Zc:__cplusplus,/Zc:preprocessor -I<cutlass>/include`; measured with v4.8.0, kept outside the tree):
CuTe's TiledMMA in the canonical sm80 structure -- warps tile M, a cp.async pipeline walks K through shared memory,
the codes are widened into a swizzled BF16 tile and read by ldmatrix. Same accuracy as the product; 1.32x decode at
every R on the 5060 Ti (272 to 331 GB/s), best at 4 warps x 3 stages. Shared memory holds it to 2 to 4 warps per SM
against the product's 48, and each stage serializes a widening pass between two barriers -- the structure, at eight
rows, not CuTe as notation.

A `cute-split` arm then gives CuTe the product's structure: warps splitting K, register prefetch, and the product's k
permutation passed as the TiledMMA's `PermutationMNK`, so `partition_A`/`partition_B` and `cute::copy` deliver the
operands and `cute::gemm` multiplies. Its outputs match the product's bit for bit, which shows the permutation is
expressed exactly; it runs at 1.41x decode. The SASS shows why: per 64 columns it issues 32 single-byte weight loads
and 48 byte permutes where the product issues one 32-bit load per row and block -- the SM80 atom's register order
interleaves rows g and g + 8, so a copy of 4-bit elements cannot vectorize across them. CUTLASS's remedy is an offline
reorder of the stored codes (`examples/55_hopper_mixed_dtype_gemm`), which for Mila would be a new Q4_0 storage
format.

*Harness note:* clear an output on the benchmark's own stream. A non-blocking stream does not order against a
legacy-stream `cudaMemset`, and a clear that landed after the kernel once read as wrong results on one card only.

## `ExpertUnion.cu`

What the Gemma 4 26B-A4B's Q4_0 expert bank costs a verify of R rows against one decode (`Gemma4Mtp.md` 4.7, step
5), on real routing: each reply row's experts as `Tools/Drafting routing --target 26b --output <dir>` writes them,
over random weights of the 26B's shapes, three 428 MB banks in rotation so every call is DRAM-resident. Four arms,
each checked bit for bit against R one-row decodes before it is timed: today's gather kernels once per row, the same
kernels at R tokens in today's block order, the same kernel bodies with the R tokens' blocks for the same rows run
together (token-first), and a gate-and-up pass grouped by distinct expert.

```
nvcc -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_89,code=sm_89 -O3 ExpertUnion.cu -o ExpertUnion.exe
ExpertUnion.exe chat.routing code.routing prose.routing
```

Measured 2026-10-06: token-first runs within 3 to 8% of the union's own bytes at one row's rate on both cards -- L2
turns the repeated reads of a shared expert into hits once the rows that chose it run together -- where today's order
is 15 to 30% above it. The grouped arm is slower than both past R = 2. Table in `Gemma4Mtp.md` 4.7.

## `kernel_shares.py`

Groups an nsys kernel summary into attention / GEMM / plumbing / other, so a profile answers
*where does the time go* rather than listing 200 kernels.

```
nsys profile --trace=cuda --sample=none -o run ./YourProgram
nsys stats --report cuda_gpu_kern_sum --format csv --output run run.nsys-rep
python kernel_shares.py <directory holding the *_cuda_gpu_kern_sum.csv files>
```

**It prints every uncategorised kernel above 1%, and that output is the point.** The
classifier has twice been wrong in a way that silently moved a large share into `other` — a
namespace that did not match the pattern, and a weight-unpack kernel under a different name
in a different quantization format. Read the residual every time; a share table with a large
unexplained `other` is not a result.

---

## Reading a number from any of these

- **State the card.** The two development GPUs differ in architecture, SM count and
  bandwidth, and a number without its card is not a measurement.
- **State residency.** These are written to be DRAM-resident; an L2-resident result ranks
  candidates differently.
- **State the direction the number must move before running it.** Deciding afterwards is how
  a flat result becomes a success story.
- **Ratios survive; absolutes travel badly.** Throughput scales with SM count, so a figure
  from a 188-SM part says little about a 36-SM one.
