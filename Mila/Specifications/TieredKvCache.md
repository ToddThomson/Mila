# Tiered KV Cache

**Status:** Preliminary, 2026-10-18. No code, nothing measured for it yet. Todd's proposal: a KV cache that is one
pool over two surfaces -- device memory first, pinned host memory for what the device cannot hold -- so a model
whose weights fill a 16 GB card still reaches the 64K minimum agentic context. Never the whole cache in host memory.
Section 7 lists what must be measured before this becomes a design.

**Area:** where a KV cache's positions live, and what the planner prices for each surface. What the cache stores
(BF16 or FP8) is `Quantization.md`'s; the sliding-window ring is `SlidingWindowKvCache.md`'s and stays on the device.

---

## 1. The gap

The minimum context for agentic work is 64K (Todd, 2026-10-15; `Qwen3.8.md`, *Agentic deployment on the 16 GB
card*). On the RTX 5060 Ti 16 GB no model of 26B parameters or more at FP4 or better reaches it today.

Qwen 3.8 27B FP4, from the measurements of 2026-10-15:

| | MiB |
|---|---|
| Budget the process can use | 15,168 |
| Everything but the cache, at a 512-row prefill chunk (13,811 + 0.61 x chunk) | ~14,123 |
| Left for the cache | ~1,045 |
| Cache at 64K, FP8 (16 full-attention layers x 4 KV heads x 256 x 2 x 1 byte = 32 KiB a token) | 2,048 |
| **Short** | **~1,000** |

32,768 at a 512-row chunk with an FP8 cache fits with 9 MiB to spare, measured, which is the same arithmetic at half
the context. Gemma 4 26B-A4B Q4_0 is not priced here; its global layers' cache is the one to price, and storing their
keys once (`RopeInAttention.md`) halves it.

## 2. Device levers come first

What frees device memory costs nothing at decode, so the host tier carries only what they leave:

| Lever | Frees on Qwen FP4 | Where it is tracked |
|---|---|---|
| A weight arena: one allocation per model or layer, not one per tensor rounded to 2 MiB | 497 MiB, measured | BACKLOG, "Each weight tensor over 1 MiB is its own allocation" |
| FP8 group scales in place of BF16 | ~0.37 GiB, estimated | a weight-format change: re-export and measure |
| A 256-row prefill chunk in place of 512 | ~156 MiB, from the footprint model | prefill 8.1 s to 10.3 s on an ~8K prompt, measured |

The first two together would leave about 140 MiB, some 4,500 tokens, for the host tier at 64K -- arithmetic on the
table above, not a measurement.

## 3. The proposal

One logical cache of N positions per full-attention layer, split at H:

- **Positions [0, H) in pinned host memory**, the oldest. They are written once, when prefill or a spill fills them,
  and read every step.
- **Positions [H, N) on the device**, the newest, where every decode step writes its token.

H is the plan's: zero when the device holds the context, and otherwise the overflow (section 5). Nothing is lost or
approximated; the values are the ones the cache would hold on the device, in the format the cache already uses.

**The host tier is prefetched, not read in place.** Attention reads every cached position every step, so the bytes
crossing the bus per token are H x (bytes a token) whatever the schedule. What the schedule decides is whether they
overlap compute. Qwen runs three DeltaNet layers between attention layers. While they run, the next attention layer's
host slice is copied into a device staging buffer, double-buffered. A decode step then costs about
max(compute, transfer) rather than their sum. A kernel reading mapped host memory directly cannot overlap that way:
the read stalls the attention that issued it.

**Cost model, to be measured (section 7).** The 5060 Ti's link reads as PCIe 5.0 x8 (`nvidia-smi`, 2026-10-18);
about 25 GB/s is assumed. At H x 32 KiB:

| Host tier | Bytes a step | Transfer | Decode today ~30 ms a step |
|---|---|---|---|
| 140 MiB (after section 2) | 140 MiB | ~6 ms | hidden under compute |
| 1 GiB (no device levers) | 1 GiB | ~43 ms | bounded by transfer, ~23 tokens a second against ~33 |

Prefill reads the host tier once per chunk, as far as it is filled. A 64K prompt at 512-row chunks with a 1 GiB host
tier moves about 96 GiB over the bus, about 4 seconds at the assumed rate; with 140 MiB, under one.

## 4. Constraints it must keep

- **Decode replay** (`DecodeGraph.md`). A replayed step takes no per-token launch value. The host tier is a fixed
  region once full, so every prefetch copy has a fixed size and fixed pointers; the attended length is read from the
  decode position on the device, as today.
- **The plan prices both surfaces exactly** (`MemoryFootprint.md`, `Deployment.md`). The staging buffers are device
  memory and are priced with the rest; the host tier is priced against what the operating system lets the process
  lock, which WDDM limits, and a plan that cannot lock it is a refusal, not a smaller tier.
- **A caller's setting is a constraint the plan honours.** A fixed context is split around; the device tier and
  chunk follow from it. The same principle as the prefill-chunk entry in `Mila/Issues/Untriaged.md`.
- **Prefix reuse.** The host tier is the oldest positions, which is the prompt's prefix; a saved position
  (`PromptCaching.md`, Qwen's `savePosition`) should land inside the tiers without a copy between them.
- **The format a model trained for is untouched.** This changes where values live, not what they are.

## 5. The planner

The plan gains the split: device positions, host positions, the staging buffers, and the decode cost it expects,
stated so a caller can refuse it. Automatic context fills the device tier first and opens the host tier only past
it; a fixed context opens it exactly as far as needed. Whether the planner may choose a host tier unasked, or only
when the caller allows one, is open (section 7).

## 6. Not this

- **The whole cache in host memory.** Never proposed: every step would move the full cache, 2 GiB at 64K FP8.
- **Offloading weights instead.** The same bytes cross the bus every step, for no gain, and the cache is what grows.
- **Evicting or compressing old positions** (attention-sink and keep-the-important-tokens schemes). It changes what
  the model sees, against a model running as it was trained.

## 6a. Prior work, searched 2026-10-18

Most of it targets data-centre hardware or batches of requests; none found measures exact attention over a
host-resident cache on one consumer card over PCIe.

| Work | What it does | Bearing here |
|---|---|---|
| FlexGen (Sheng et al., arXiv 2303.06865, 2023) | places weights and cache across GPU, CPU and disk by a linear program, both compressed to 4 bits | a throughput system for batches; its 4-bit cache is a format change this design does not make |
| HGCA (Deng et al., arXiv 2507.03153, 2025) | attends on the CPU to *selected* host-resident entries, merged with the GPU part by log-sum-exp | sparse, so approximate: out, as eviction is |
| Fluxion (Yao et al., arXiv 2605.07719, May 2026) | block-sparse CPU-GPU hybrid; finds both GPU-only and hybrid designs leave the GPU idle during decode | sparse, so out; the idle-GPU finding is the risk section 3's prefetch has to answer |
| InfiniGen (via the survey arXiv 2603.20397) | predicts which entries the next layer needs and prefetches only those | approximate selection: out |
| KVPR (Jiang et al., arXiv 2411.17089, Findings of ACL 2025) | sends a layer's input activations for part of the cache and recomputes those keys and values on the GPU while the rest transfers | exact. Pays only where a position's activations are smaller than its cache entries; for Qwen 3.8's grouped attention they are larger (5,120 values against 2,048), so it moves more bytes, not fewer |
| DirectKV (OSDI '26) | GPU kernels read a CPU-resident cache in place | measured on NVLink-C2C (GH200), not PCIe; over PCIe a read in place cannot overlap, as section 3 says |

Read from abstracts and a survey, not the full papers; their hardware and numbers are the authors'.

## 7. Before this is a design

1. **The bus, measured.** Pinned host to device copy rate on the 5060 Ti, at the slice sizes above, overlapped with
   a decode step's kernels.
2. **The device levers, measured on both models.** The arena on Qwen FP4 and Gemma 26B-A4B, then the 64K plan priced
   with it, so the host tier's size is a measurement and not section 2's arithmetic.
3. **Gemma 4 26B-A4B priced** at 64K: global layers only, with and without keys stored once.
4. **The decode target** (Todd): the lowest rate at 64K on the 16 GB card that is acceptable, which bounds H.
5. **Which families.** Qwen and Gemma cache only some layers; Llama caches every layer, so its host tier costs the
   most per position.
6. **Whether a host tier is opt-in** for a caller, or the planner's to choose.
