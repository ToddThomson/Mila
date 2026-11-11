# Gemma 4 12B's global-layer prefill attention on Metal (head 512)

**Role: Plan.** The open problem in Mila's Metal prefill and what is known about it, written 2026-11-10 so that the next
piece of work starts from the evidence. Everything below is measured unless marked *arithmetic* or *inference*. The
evidence itself is in `Notebooks/MetalRates.md` section 7.3; read it in full first. A design this plan arrives at is
carried to `MetalBackend.md` 4.5, which owns the fused prefill.

**Status, 2026-11-10: built at `0.21.0-dev+133`, as sections 7 and 8 say, with the three differences section 9
records.** The decision is carried to `MetalBackend.md` 4.5 and `Deployment.md` 5. Sections 1 to 6 are the road here;
facts 8 to 14 and sections 7 to 9 are where it stands. Not run: the rates beside llama.cpp with the built change (the
arm's are in fact 13's session and section 2).

## 1. The task, and only this task

**Scope: Gemma 4 12B only. The global (full-attention) layers only. Prefill at depth.** Not Llama, not the sliding layers,
not decode, not the Q4_0 expansion pass. Todd has corrected drift away from this scope several times; stay inside it.

**Required output before any code: one design, argued from the facts in section 3, with its arithmetic and the risks that
could sink it** (Todd, 2026-11-10, after a session of arms tried one after another without a design: "throwing money at
non-returning attempts"). No build is proposed until the design has survived its own premises. Any `Lib/Src` change needs Todd's agreement first
(CLAUDE.md). Anything over about 10 minutes of machine time: state the estimate and ask.

## 2. The problem

Mila's Gemma 4 12B Q4_0 prefill (512 tokens added, FP16, M1 Pro 16 GB) against llama.cpp b12138 on the same Mac:
0.99 of llama.cpp from empty, 0.91 after 4096 held and 0.88 after 16,384, measured 2026-11-10 (MetalRates 7.1). The
0.89 and 0.75 this plan first carried were one session of the night before whose Mila rates do not repeat; llama.cpp's
do. With the global layers unfused, a whole prompt a block, the same session gave 0.93 after 4096 and 0.97 after
16,384. Kernel-by-kernel profile (MetalRates 7.3), GPU ms a prompt of 512:

| Kernel | Held 0 | 4096 | 16384 |
|---|---|---|---|
| Matrix product (flat, ALU-bound) | 2929 | 2923 | 2950 |
| Attention, sliding layers (40 layers, head 256; flat once the 1024 window fills) | 93 | 396 | 402 |
| **Attention, global layers (8 layers, head 512)** | **23** | **318** | **1553** |

Everything except the global layers is flat with depth: **the depth fall is the global layers' attention.** Rates differ
between sessions by up to ~20%; the same kernel measured 1,187 to 1,553 ms after 16384 across today's sessions. Compare
only arms alternated in one session.

Gemma's global layer: 16 query heads share 1 KV head; head 512; K = V in the checkpoint (two cache tensors today);
partial rotary. Arithmetic: after 16384 held, the global layers' Q x K^T plus P x V for a 512 prompt is about 2.2 TFLOP;
the matrix product kernel achieves roughly 3.7 TFLOPS on this chip, so the floor is very roughly 0.6 s against today's
1.2 to 1.5 s.

## 3. Facts established today

The kernel: `Lib/Src/Dnn/Compute/Devices/Metal/Operations/Attention/GQA/Kernels/GroupedQueryAttention.Flash.metal`
(`gqa_flash_prefill`, at HEAD), host side `MetalGqaOp.ixx` (`flashTileFor`, `flashPrefill`). Design record
`MetalBackend.md` 4.5; its tuning history `MetalRates.md` 6.1 (64 keys a tile beat 32 at head 512:
383 to 308 ms; staging K/V once per threadgroup lost at head 256 then).

Today's head-512 kernel: threadgroup = 8 SIMD-groups = 8 of the 16 query heads, 8 query rows, 64-key tiles; each
SIMD-group holds four 8 x 128 FP32 output slices; Q x K^T summed over four 128-channel slices; K and V read from device
memory by each SIMD-group. Grid 128 threadgroups a layer for a 512 prompt (64 row blocks x 2 head halves); each streams
the whole cache up to its rows, so each key row is read 128 times a layer (*arithmetic*: about 4.3 GB of cache reads a
global layer after 16K, for a 33.5 MB cache).

1. **Registers (Xcode 27.2, GPU capture replayed with profiling, Performance > Shaders):** today's head-512 kernel 254
   allocated registers, high register 254, **1.14 KiB spilled**. Decode partial 104, merge 80, writes 42, no spills.
2. **Thread limit = register proxy:** 254 registers gives `maxTotalThreadsPerThreadgroup` 384. Every fused prefill
   kernel (heads 128, 256, 512) is at 384 today.
3. **GPU counters (one dispatch, after 4096):** occupancy 11%, ALU limiter 44, buffer read limiter 57, last-level cache
   51; no limiter saturated. Same occupancy from empty.
4. **What sets the 384 (compile-only probe, parts removed):** two independent causes, each enough alone:
   (a) the four FP32 output slices a SIMD-group holds (four slices and no products: still 384; one slice and no
   products: 704); (b) a TensorOp product whose right operand is in **device** memory holds the operand in registers.
   One lone TensorOp: P x V 8x128x64 from device 448 threads, from threadgroup 1024; Q x K^T 8x64x128 from device 832,
   from threadgroup 1024; at 16 keys both 1024.
5. **TensorOps constraints on the M1 Pro:** `matmul2d` M must be a multiple of 8 (4-row kernel refused at compile);
   one TensorOp wider than 128 channels at `execution_simdgroup` returned wrong values silently (MetalBackend 4.5);
   threadgroup memory max 32 KB (a 32-key tile of all 512 channels plus partials was refused at 39 KB).
6. **Arms that lost (all pass the 128 `GqaMetal*` tests; GPU ms after 16384 against today's in the same session):**
   - Channel split, 4 SIMD-groups a head, device operands, 64 keys: 1,704 vs 1,273 (+34%); still 254 registers, spill
     256 B; occupancy 8%.
   - Split + K/V staged in threadgroup memory, 16 keys, 576 threads: 2,028 vs 1,463 (+39%).
   - Unsplit + staged slice by slice (16 barriers a tile), 384 threads: 64 keys 1,849 vs 1,192 (+55%); 32 keys 3,311.
   - Head 128 staged unsplit at 8 rows reached 576 threads and was 13-21% faster (Llama; out of scope, not kept) -- the
     only staging win, and the only one that lifted the threads without an exchange.
   Reading: more threads alone is not the measure; every route off the cap tried so far paid more in barriers or
   exchange than it gained. Metal has no asynchronous copy, so staging cannot overlap loads with products as CUDA does.
7. **CUDA's answer at head 512** (`GqaFlashAttention.md` 5.8, "head size 512 goes key-major"): keys on the
   MMA's M dimension (S^T = K Q^T), each warp owns 8 rows across the whole head with no split-K exchange, O^T = V^T P^T
   in 128 FP32 registers a warp, 64 rows a block sharing each K/V tile, 32-key tiles with K and V double-staged and
   overlapped, Q half in registers half in shared memory; FP32 accumulation kept, FP16's 2x rate declined as a
   precision trade. 1.38-1.60x over the packed kernel. Backend parity is a standing rule (MetalBackend 4.5 decision):
   CUDA's design is the reference, translated to what Metal can do.

8. **Occupancy follows the thread limit** (2026-11-10, the compile-only probe of section 5): the matrix product
   `linear_tiles_half_128_16_4` is given 832 threads a threadgroup, the fused prefill kernels 384, the unfused softmax
   1024. 832 / 384 is 2.17, and the counters' occupancy is 24% against 11%, 2.18. The product holds nothing between
   tiles: operands and destination are device tensors. Fact 4 put one output slice held with no product at 704, so no
   kernel that carries a head-512 output in registers reaches the product's limit.
9. **MLX v0.32.4 never runs a fused kernel at head 512 on a chip without neural accelerators** (read 2026-11-10,
   `mlx/backend/metal/scaled_dot_product_attention.cpp`): `has_fused_kernel` refuses 512 there, so a prompt under 1024
   rows takes the plain matmul, softmax, matmul; from 1024 rows, causal, FP16 or BF16, `use_blocked_d512` selects
   `sdpa_full_self_attention_blocked`: blocks of 256 query rows, one score buffer of a block by every key, `steel_matmul`
   for Q x K^T over the query-head group (M = group x rows), a causal mask, the softmax in place, `steel_matmul` for
   scores x V. No merge between blocks: each row's softmax is whole. Its fused 512 kernel is for chips with the
   accelerators, from 1024 query blocks up.

10. **Unfused, the global layers are faster at depth** (2026-11-10, one session, the sliding layers fused in both builds;
    MetalRates 7.3). GPU ms a prompt of 512, fused then unfused: 23.1 and 32.4 from empty, 306.5 and 213.7 after 4096,
    1189.7 and 765.3 after 16,384. The whole prompt: 3285 and 3297, 3872 and 3790, 4762 and 4346, which is 8.7% faster
    after 16,384, 2.1% after 4096 and 0.3% slower from empty. After 16,384 the unfused time is attention x V 381,
    Q x K^T 276, the softmax 100 and the permutes 8. The plan at a 20,480 context grew from 7.706 to 8.956 GiB: two
    score buffers of a 1024-row chunk by 16 heads by the context.
11. **The unprofiled prompt is the kernels' sum** (same session): 3.27 s from empty and 3.85 s after 4096 unprofiled,
    3.29 to 3.31 s and 3.87 to 3.88 s of profiled GPU time. Nothing outside the kernels is waiting to be found.
12. **Gemma's plan at its full context has 0.43 GiB to spare** (2026-11-10, `--planner-only`): context 262,144, limited by
    the trained maximum, 11.408 GiB of an 11.840 GiB working set. *Arithmetic:* one score row of the 16 heads across
    262,144 keys is 8.4 MB in FP16, so a block of 8 query rows is 67 MB a score buffer.

13. **The gain depends on the block** (2026-11-10, the block loop as an arm, one process; MetalRates 7.3). The global
    layers' GPU ms a prompt after 16,384, against the fused kernel's 1190: 756 at 512 positions a block, 948 at 128,
    1008 at 32, 1239 at 8. After 4096, against 307: 219, 257, 270, 342. So 32 positions is still 15% and 12% ahead of
    the fused kernel and 8 is behind it. The loss at small blocks is in attention x V and the softmax, whose dispatches
    are then a few dozen threadgroups each. The attention tests pass with the unfused reference in blocks of 8 and 24.

14. **A block that takes the last of the memory does not run** (2026-11-10, the built change, `--planner-only`, two
    runs with the +131 binary before, between and after). At a context of 262,144 the plan without a block is 11.408 GiB,
    96.4% of the 11.840 GiB working set, and runs. Given a block of 32 rows it is 11.658 GiB, 98.5%, loads, and its first
    prefill of 512 fails: `kIOGPUCommandBufferCallbackErrorOutOfMemory`. What the run needs beyond the plan is not found
    (`Issues/Untriaged.md`).

## 4. Options not yet tried (inference; none designed)

- **Global layers on the matrix product** (Q x K^T and P x V through the 89%-ALU tiled product, a softmax pass between).
  Measured unblocked, fact 10: 765 ms against 1190 after 16,384. Section 7 proposes it in blocks of query rows.
- **More query rows per threadgroup reading each tile** (blocks), to cut the 128x re-streaming, without raising
  per-SIMD-group registers.
- **Spill reduction** via `[[max_total_threads_per_threadgroup(256)]]` on today's kernel (verify what the attribute does
  in Apple's documentation first).
- **What MLX does at head 512** for prefill: fact 9. It blocks by query rows, not by keys, so it needs no online
  merge and its memory is bounded by the rows a block takes. MLX v0.32.4 source is at `D:\Claude\mlx-src` (MIT; may be
  read and cited). llama.cpp's source must NOT be read (CLAUDE.md *Sources*).

## 5. Tools and recipes (all work)

- **Mac:** `ssh -i D:\Claude\mac\mila_mac -o UserKnownHostsFile=D:\Claude\mac\known_hosts toddt@mac-dev.local`; tree
  `~/mila-work/Mila`, build `out/build/macos-clang-release`. Sync PC to Mac: `python D:\Claude\mac\sync_all.py`.
  Full Xcode 27.2 is installed; Developer Mode is on.
- **Profile by kernel:** `mila-metal-model-timing --family gemma --weights Data/Models/Gemma/gemma4_12b_it_qat_q4_0.safetensors
  --precision fp16 --no-planner --context 20480 --prefill-depths 0,4096,16384 --rounds 1 --runs 1 --profile-dispatches`
  (~7.5 min a run). Template with alternation: `D:\Claude\metal\prefill2\unsplit.sh`. Baseline binary of HEAD:
  `~/mila-work/rows4/TA`.
- **Thread-limit probe (headless, ~20 s, no build):** `D:\Claude\metal\prefill2\limits.mm`, built at
  `~/mila-work/capture/limits <tree> <kernel>...`; compiles Mila's kernel source at run time, as `MetalKernelLibrary` did until `+134`. Scratch
  tree `~/mila-work/probe-tree`; probe kernels `probe_kernels.metal`, extra instantiations `inst*.metal`.
- **Registers and spills:** capture with `D:\Claude\metal\prefill2\capture.mm` + `capture.sh <name>` (dylib injected into
  `MilaTests --gtest_filter=GqaMetalFused/Fp16.GemmasGlobalLayers`), then Todd opens the `.gputrace` in Xcode (Replay,
  "Profile after replay", Performance > Shaders) and screenshots with Cmd+Shift+3; fetch the newest `~/Desktop/*.png`
  with `cat` over SSH. `gpudebug` works headless but cannot profile on M1 ("requires M3 / A17 Pro or later").
- **GPU counters:** `xctrace record --template "Metal System Trace" --instrument "Metal GPU Counters"` over the WHOLE run
  (a `--window` keeps only ~0.25 s of counters), export tables, join with `D:\Claude\metal\attention2\join2.py`.
  Exporting a 4096-depth run took ~2 h; prefer depth 0 or a short run.
- **Tests on the Mac:** `MilaTests --gtest_filter='GqaMetal*'` (128 tests, ~1 min).
- Today's arm code (staged kernels, split kernel, shared softmax helper) is saved at
  `D:\Claude\metal\prefill2\arm_staged\` for reference; the tree is back at HEAD for both files.

## 6. Starting point

The kernel files are as committed at `0.21.0-dev+131`; none of the arms above is in the tree. A related parity gap is
filed in `Issues/Untriaged.md`: Metal widens the FP8 cache in a separate pass where CUDA widens it inside the tile
(`Quantization.md` KV decision 2).

**The Mac's copy is checked by hash before a build, not assumed.** On 2026-11-10 the PC's tree was at HEAD and the Mac's
`MetalGqaOp.ixx` and `GroupedQueryAttention.Flash.metal` still held the last arm. Both were copied from the PC, all 674
files of `Lib/Src` and the timing tool then matched, and the Mac's timing binary was rebuilt from them.

## 7. The design (agreed, Todd, 2026-11-10; nothing built)

**The global layers' prefill at a head of 512 runs the unfused pipeline, a block of query rows at a time.** The pipeline
is Phase 3's, which the fused kernel's tests already use as their reference, and the shape is MLX's on chips without
neural accelerators (fact 9).

- **A block** is R query positions and the group's 16 heads: those rows of Q permuted, Q x K^T through the matrix product
  (M = 16 R), the causal softmax, attention x V, the rows of the output permuted back. Each row's softmax is whole, so
  blocks are not merged and nothing is carried between them. As the arm built it, a block is the operation's own unfused
  prefill called on R rows at their offset: no kernel changes.
- **R comes from the memory the plan has left**, not from the chunk: the context is planned first, as today, and the
  score buffers then take what a block of up to a whole chunk needs, one row of the group across the context being
  `16 x context x 2` bytes in FP16. *Arithmetic:* a block of 128 positions is 84 MB a buffer at a 20,480 context, 268 MB
  at 65,536 and 1.07 GB at 262,144.
- **Under 32 positions a block the fused kernel runs instead** (fact 13: 8 positions is behind it). So the fused 512
  kernel stays built and tested, and is what a context too long for a 32-position block gets; on a 16 GB Mac that is
  only near Gemma's trained maximum (fact 12).
- **The planner prices the score block** where it priced no score buffer, and the operation is told the block it was
  given, as it is told the chunk today.
- Heads 128 and 256 stay fused.

**What it bends.** `MetalBackend.md` 4.5, agreed on backend parity, says one fused kernel a layer at heads 128, 256 and
512. The same section says that where the fused kernel is slower than Phase 3's pipeline at a head size on the M1, the
numbers go to Todd before anything else is decided. Fact 10 is those numbers.

**Agreed (Todd, 2026-11-10):** the direction and the block-size arm first, then, with fact 13 in hand, the three points
that follow from it: the block from the memory the plan has left, the fused kernel under 32 positions a block, and the
full context on a 16 GB Mac staying fused for now. What `Lib/Src` changes to carry it is section 8, agreed the same day.

**Risks:**

1. **Small blocks: measured, fact 13.** The gain falls from 36% of the kernel at a whole chunk to 15% at 32 positions
   and is gone at 8. Nothing deeper than 16,384 held has been timed at any block size.
2. **The full-context plan has 0.43 GiB of room** (fact 12). A 32-position block there is 268 MB a score buffer, so two
   do not fit and one would leave the plan at about 98.5% of the working set (arithmetic). The softmax writes a second
   buffer today; whether it can run in place is not checked. Until it is, the full context on a 16 GB Mac stays fused.
3. From empty the unfused path is 9 ms slower, 0.3% of a prompt.
4. The scores are rounded to FP16 before the softmax, where the fused kernel keeps them in FP32. The HuggingFace gate
   on Gemma 4 12B Q4_0 decides.
5. Chips with neural accelerators are unmeasured; MLX fuses 512 there.
6. **Two paths at one head size**, chosen by the plan: the gate has to cover both, at a block of a whole chunk and at
   the floor.

**Not in it:** the sliding layers, whose attention grows 0.30 s by 4096 held and is flat after, and the value product's
tile shape.

## 8. What `Lib/Src` changes (agreed, Todd, 2026-11-10; nothing built)

The planner already chooses one thing against memory after the context: the prefill chunk, from a family's rungs, each
rung priced by the network's own `getRequiredMemory` (`Deployment/Planning/PrefillChunkRule.ixx`). The score block is
chosen the same way, after it.

1. **The build context carries the block.** `BuildContext` gains the prefill score rows beside the prefill size: zero,
   its default, means no score buffer, which is today's fused build. A network reads it where it reads the chunk.
2. **A rule chooses it** (`Deployment/Planning`, one function in its own module as the chunk's is): the largest rung, no
   larger than the chunk, whose whole footprint fits the budget; zero when none does, or when no layer of the network
   wants one. Rungs are the chunk's halved down to the floor: 1024, 512, 256, 128, 64, 32. It runs after the context and
   the chunk are chosen and can only add memory a plan had spare, so **no plan's context or chunk changes**. That is the
   property its tests assert.
3. **The plan records it**, beside the chunk, so `load(path, plan)` builds what was priced. `load(path, config, device)`
   names no block and stays fused.
4. **The operation says what it prefers.** `supportsFlashPrefill` stays as it is: the fused kernel exists at 512. A
   second answer, per head size, says the unfused pipeline is the faster one where it can have a block: true for 512 on
   Metal, by fact 13, false everywhere else and on CUDA. The transformer runs a layer unfused when the operation
   prefers it and the context carries a block; otherwise fused, as today.
5. **The workspace's score buffers take the block's rows** in place of the chunk's: `makeGqaWorkspace` and
   `gqaWorkspaceDeviceBytes` gain the rows as an argument, and CUDA's callers pass the chunk, so nothing of CUDA's
   moves.
6. **`MetalGqaOp::prefill` runs its unfused pipeline a block at a time**, as the arm did, for any batch count, and no
   longer refuses a chunk longer than its score rows.
7. **The softmax in place, as a second step of the same change.** `gqa_prefill_softmax` reads and writes each score
   from the one thread that owns its column (read of the kernel, 2026-11-10), so one buffer can serve as both. It
   halves the score memory, which doubles the block a plan can afford, and is what lets a 16 GB Mac's full context
   have a 32-position block at all. Its own test: the same values, bit for bit, as with two buffers.

**Gemma only in effect.** The rule is not written for a family: it asks the network. Only Gemma has a head of 512;
Llama and Qwen 3.8 answer that no layer wants a block, and their plans are untouched.

**What a user sees.** Gemma 4 12B on a Mac takes more memory than it does today wherever there is memory to spare, up
to 1.25 GiB at a 20,480 context with a 1024 chunk (7.71 to 8.96 GiB, measured), and prefills a long conversation
faster. Chat's footprint and `/context` report the plan's number, so they show it.

**Gates.** The attention tests with the unfused path in blocks at the floor and at a whole chunk, more than one batch
among them, each failing when the block loop is broken on purpose; the rule's tests, model-free; the HuggingFace gate
on Metal for Gemma 4 12B Q4_0 with the global layers unfused; rates at 0, 4096 and 16,384 held beside llama.cpp, one
session, before and after; the context a 16 GB Mac plans, before and after, which must stay 262,144.

**Specs it edits when built:** `MetalBackend.md` 4.5 (the decision and what replaced it at 512), `Deployment.md` (the
rule and the plan's new value), and this plan's status.

## 9. As built (`0.21.0-dev+133`, 2026-11-10)

Section 8's seven steps, in its order. Where the build differs from what section 8 says:

1. **A request that fixes its context is given a block too.** Section 8 item 3 said `load(path, config, device)` names
   no block and stays fused. That load is a planned request like any other, given its chunk by the planner, and it is
   given its block the same way. Section 8's own figure for what a user sees is such a load.
2. **A block is taken only where it fits twice over** (`choosePrefillScoreRows`). Fact 14: the block that fitted once at
   the full context did not run. So the full context on a 16 GB Mac stays fused, as section 7 agreed it would for now,
   and the softmax in place halves what a block costs without yet buying that plan a block. The margin stands in for a
   cost nobody has found; when it is found and priced, the rule goes back to fitting once.
3. **An FP8 cache stays fused.** `prefersUnfusedPrefill` is true for a head of 512 over an uncompressed cache only: no
   FP8 cache was measured unfused, and its rows would be widened again for every block.

**The block is the score buffers' rows.** `MetalGqaOp` takes no block size: it runs as many query positions at a time as
the buffers it was wired hold, one batch at a time, so the transformer's sizing and the operation's loop cannot disagree.
It refuses buffers narrower than the context it strides them by.

**Measured, M1 Pro 16 GB, Gemma 4 12B Q4_0, FP16, a fixed context of 20,480, the +131 binary then this one in one
session, tokens a second for 512 added:** 156.6 and 156.4 from empty, 132.9 and 136.2 after 4096 held, 107.7 and 118.7
after 16,384 held. The plan grows from 7.706 to 8.331 GiB: one score buffer of a 1024-row chunk, where the arm's two were
1.25 GiB. The plan at the full context is unchanged: 262,144, 11.408 GiB, fused.

**Gates run:** the attention tests in blocks at the floor, at a whole chunk, off a divisor, on the ring and with more
than one batch; seven ways of breaking the block loop, each caught; the softmax in place against two buffers, equal to
the bit; a small Gemma network built with a block against the same weights fused, 2.3e-3 of the logits' RMS apart at
every block size; the rule's tests, model-free; Gemma 4 12B's HuggingFace gate on Metal, FP16 and BF16, 12 of 12. Its
BF16 window-edge check reads 2.36e-2 against a bar of 2.54e-2, where the fused build read 2.14e-2
(`BACKLOG.md`, the window-edge item).
