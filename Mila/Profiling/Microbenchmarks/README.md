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
