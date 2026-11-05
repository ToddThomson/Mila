# Metal precision timing on the M1 Pro

Evidence for `Metal/MetalBackend.md` sections 2 and 3, 2026-11-04. It answers one question Todd put: is FP32 math on
the M1 about 3 times slower than FP16? It also records what the same run showed about BF16 and TensorOps.

True as of its date, on one chip. Not a model run: uniform random values in [-1, 1], so no subnormals, and no FP4 or
FP8 weights.

**Machine:** MacBook Pro 2021, Apple M1 Pro, 14-core GPU, 16 GB, macOS 27.2, on AC, Metal Shading Language 4.1, default
math mode.

The program, the gate and the raw log are outside the tree, in `D:\Claude\metal\precision-speed\` (`precision.m`,
`gate.md`, `analyze.py`, `all.log`). `MetalBackend.md` Phase 2 replaced it in the tree with `Lib/Tools/MetalMatrixTiming`, on Linear's shapes.

---

## 1. The gate, written before the run

**Claim:** FP32 math is too slow, likely 3 times slower than FP16.

**Verdict, per kernel and shape, on FP32 math time over FP16 math time:** 2.5 or more confirms it there; 1.5 to 2.5 is
slower by less than claimed; under 1.5 does not confirm it there. The matrix multiply shapes decide the overall
answer. If they disagree, each is reported and no single number is made from them.

**Validity:**

- An arm's timing counts only if its output matches a double-precision CPU reference built from that arm's own rounded
  inputs, at shapes S and D: relative L2 error under 0.05 for 16-bit math, under 0.01 for FP32 math with 16-bit output,
  under 1e-5 for FP32 throughout. Shape L is too large for the reference; it runs the same kernels.
- A deliberately misaligned output must fail the check.
- 3 warm-up dispatches per arm, then 15 rounds with every arm once per round, order rotated each round; three separate
  process runs. A result differing between runs by more than 15% is reported as unstable.
- GPU timestamps, one command buffer per dispatch, waited on.

## 2. The arms

One program, with the same input values rounded by the GPU into each format. Storage/math: FP16/FP16, BF16/BF16,
FP32/FP32, FP16/FP32, BF16/FP32.

| Kernel | What it does | Tells us |
|---|---|---|
| `register-matrix`, `register-scalar` | Loops in registers, no memory traffic after the first load | Arithmetic alone |
| `gemm-direct` | SIMD-group matrices loaded straight from device memory, 8 x 8 tiles; storage and math the same type only | Mila's own kernel in its simplest efficient form. Not tuned |
| `gemm-copy` | Tiles copied through threadgroup memory, 64 elements of K at a time | The only own-kernel form that can store 16 bits and compute in 32 |
| `matmul2d` | Metal's TensorOps, tiled 64 or 8 by 32, four SIMD-groups; in the mixed arms the output is FP32 | Apple's kernel |
| `elementwise` | `c = a * b + b` over 64M elements | Bandwidth, no matrix unit |

**Shapes, M x K x N:** S = 512 x 1024 x 1024, near cache. L = 2048 x 4096 x 4096, prefill-like. D = 8 x 8192 x 8192, one
pass over large weights, decode-like.

## 3. Results

Median milliseconds, the middle of three runs' medians.

### FP16, BF16, FP32 with storage and math the same

| Kernel, shape | FP16 | BF16 | FP32 | BF16 / FP16 | FP32 / FP16 | Gate on FP32 |
|---|---|---|---|---|---|---|
| `register-matrix` | 0.654 | 0.747 | 0.748 | 1.14 | 1.14 | not confirmed |
| `register-scalar` | 2.072 | 5.012 | 2.978 | 2.42 | 1.44 | not confirmed |
| `gemm-direct` S | 1.570 | 1.632 | 1.643 | 1.04 | 1.05 | not confirmed |
| `gemm-direct` L | 100.694 | 105.722 | 303.233 | 1.05 | **3.01** | **confirmed** |
| `gemm-direct` D | 2.576 | 2.764 | 4.383 | 1.07 | 1.70 | slower by less |
| `gemm-copy` S | 5.974 | 5.023 | 9.176 | 0.84 | 1.54 | slower by less |
| `gemm-copy` L | 475.768 | 425.632 | 680.990 | 0.89 | 1.43 | not confirmed |
| `gemm-copy` D | 6.854 | 6.810 | 10.948 | 0.99 | 1.60 | slower by less |
| `matmul2d` S | 0.388 | 0.745 | 0.685 | 1.92 | 1.77 | slower by less |
| `matmul2d` L | 25.095 | 48.335 | 84.859 | 1.93 | **3.38** | **confirmed** |
| `matmul2d` D | 2.791 | 1.337 | 6.670 | 0.48 | 2.39 | slower by less |
| `elementwise` | 2.360 | 2.358 | 3.243 | 1.00 | 1.37 | not confirmed |

`gemm-copy` is dominated by its copies, which is why its ratios sit lower; it is not the form anyone would choose where
`gemm-direct` is available.

### 16-bit storage, FP32 math

| Kernel, shape | FP16/FP32 | BF16/FP32 | FP16/FP32 over the fastest FP16/FP16 |
|---|---|---|---|
| `gemm-copy` S | 7.691 | 7.242 | 19.8 x `matmul2d`, 4.90 x `gemm-direct` |
| `gemm-copy` L | 559.018 | 522.241 | 22.3 x `matmul2d`, 5.55 x `gemm-direct` |
| `gemm-copy` D | 8.786 | 8.145 | 3.41 x `gemm-direct` |
| `matmul2d` S | 0.390 | 0.762 | 1.01 |
| `matmul2d` L | 25.123 | 43.674 | 1.00 |
| `matmul2d` D | 2.806 | 1.343 | 1.09 x `gemm-direct`, 1.01 x FP16 `matmul2d` |
| `elementwise` | 2.032 | 2.031 | 0.86 |

### Relative L2 error against the double-precision reference

| Kernel, shape | FP16/FP16 | BF16/BF16 | FP32/FP32 | FP16/FP32 | BF16/FP32 |
|---|---|---|---|---|---|
| `gemm-direct` S | 4.68e-3 | 3.30e-3 | 5.73e-7 | | |
| `gemm-direct` D | 1.32e-2 | 3.31e-3 | 1.62e-6 | | |
| `gemm-copy` S | 4.68e-3 | 3.30e-3 | 5.73e-7 | 2.07e-4 | 1.66e-3 |
| `gemm-copy` D | 1.32e-2 | 3.31e-3 | 1.62e-6 | 2.08e-4 | 1.66e-3 |
| `matmul2d` S | 2.07e-4 | 1.66e-3 | 5.73e-7 | 5.65e-7 | 2.11e-7 |
| `matmul2d` D | 2.08e-4 | 1.66e-3 | 1.62e-6 | 1.60e-6 | 8.24e-7 |

All pass the gate. The misaligned output scored 1.415 and the true one 0.00468, in every run.

**Added beyond the gate, for the kernels it left unchecked:** `elementwise` passes against its inputs, and
`register-scalar` against its fixed point of 0.5. `register-matrix` was compared with the FP32 run: BF16 matches it
exactly, and FP16 is 8.45% off. The FP16 figure is rounding in 512 accumulations reaching several hundred, not a kernel
fault, and the timing stands.

### Stability

The worst difference between runs is 12.5% (`matmul2d` S, BF16/FP32), inside the 15% limit. Within a run, kernels under
about 10 ms drift by up to 57% between the first and last three rounds, from a few slow rounds, while their medians
hold between runs. No shape-L arm drifts by more than 4.1%. Thermal state reported no warning before or after.

## 4. Verdict

**Confirmed at prefill size:** FP32 with FP32 storage is 3.01 and 3.38 times FP16 at shape L. At D it is 1.70 and 2.39,
and at S 1.05 and 1.77, so the size of the penalty follows the shape.

**The claim as a cause:** arithmetic alone costs 1.14 to 1.44. Most of the penalty at L therefore appears to be data
size and memory, not the math. This is inferred from the shape dependence; no run isolated it.

**16-bit storage with FP32 math** is out for own kernels through threadgroup memory, at 5.55 times own FP16 at L. It survives in TensorOps,
where it costs nothing, and at decode in an own kernel Phase 2 found (`MetalMatrixPath.md`), widening in registers.

## 5. What else it showed

1. **TensorOps FP16 inputs with FP32 output cost 1.00 to 1.01 times FP16 output**, with error 5.65e-7 against 2.07e-4.
2. **TensorOps FP16 to FP16 shows the error of a wide accumulator rounded on output:** 2.07e-4 is the error of
   `gemm-copy` FP16/FP32, whose accumulation is FP32. Inferred; Apple does not document it.
3. **BF16 in own kernels accumulates with less error than FP16, despite 3 fewer significand bits** (D: 3.31e-3 against
   1.32e-2; `register-matrix` exact against 8.45%). So its SIMD-group matrix does not appear to round to BF16 at every
   step. That supports Todd's premise that BF16 is emulated on the M1, and does not prove it.
4. **FP16 by hand loses accuracy with K:** 1.32e-2 at K = 8192, about 60 times TensorOps at the same shape.
5. **Scalar BF16 costs 2.42 times FP16**, and 1.68 times FP32. Elementwise BF16 costs no more than FP16.
6. **Explained since, in `MetalMatrixPath.md` section 4:** TensorOps BF16 at D takes 0.48 of FP16, though it takes 1.9 times
   at S and L. The tile at D makes FP16 slow, on this program's weight layout, so FP16's `matmul2d` figure at D is not a fair one.
7. **TensorOps takes a quarter of `gemm-direct` at S and L** and about the same at D. `gemm-direct` is untuned, so this
   is not a verdict on either. llama.cpp keeps TensorOps off before M5 on its own measurements (its source comment:
   M2 Ultra about 5% slower, M4 no significant difference), which is the reason to tune before concluding.

## 6. What went wrong on the way

- **TensorOps wrote nothing, and looked 10 to 190 times faster.** Binding each tensor with `setBytes` at its own index
  gave all-zero output and no error. The reference check caught it (error exactly 1). Passing the resource
  ids in one argument struct fixed it, and that is the form used here.
- **The first full attempt is discarded.** Todd accidentally pulled the MagSafe cable at about 17:31, the Mac went to
  battery, where its idle sleep is 1 minute, and it slept mid-run. The clean runs went from 17:41 to 17:44 under
  `caffeinate -dimsu`, on AC at the start and end of each, with no sleep event in the window.
