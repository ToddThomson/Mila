# A KV cache that grows with the conversation

**Role: Design. Status: parked, 2026-11-12 (Todd); nothing built.** Growth commits a plan's memory later; it does not make
the plan fit the machine, and on a Mac that is what makes the first prompt wait: the planner budgets from a limit, not
from what is free, takes the most context that limit holds, and a load holds the weights twice (`Metal/MetalBackend.md`
5). The load is fixed first, then the planner's budget where memory is shared (`Deployment.md` 12.10), with probes. This
design comes back only if, after both, a measurement shows a prompt still waiting for cache memory. Section 4.2, no read
of an unwritten row, stands on its own and is not parked. Draft of 2026-11-11 follows, corrected where a fact was wrong.

How a layer's KV cache holds memory on any compute backend:
its address range reserved at the plan's full context, memory committed as the conversation fills it, and no kernel
reading a row nobody wrote. The contract and the device's part are backend-neutral (sections 2 and 3); each backend
provides them its own way (sections 4 to 6). **Metal is built first** (Todd, 2026-11-11), because on a Mac the memory a
cache reserves is taken from everything else on the machine. CUDA builds it when a need of its own asks for memory that
grows, such as card memory caching MoE layers (section 5). Decisions are asked in section 12. Everything below is
measured unless marked *arithmetic* or *inference*.

## 1. The problem

Today every backend allocates a layer's cache whole, for the plan's full context, when the network is built
(`CudaGqaOp.ixx` and `MetalGqaOp.ixx`, `build`). On a card that costs nothing else. On a Mac it is the reason the first
prompt is slow.

Mila Chat on the M1 Pro (16 GB), Gemma 4 12B Q4_0, FP16, one-shot "Hello", a fresh process each time, `0.21.0-dev+135`:

| Context | First prefill, ms | Memory pinned by the GPU after it |
|---|---|---|
| automatic (262,144, the trained maximum) | 1598 to 8423, fifteen readings | 12.8 GB |
| 8192 | 771 to 1154, six readings | 8.8 GB |

The difference between the rows is the full-attention layers' caches (*arithmetic*: 8 global layers, one KV head of 512,
K and V in FP16, is 16 KiB a position, 4 GiB at 262,144). A "Hello" uses a few dozen rows of them. The load fills the
16 GB; Metal stops holding idle buffers in place about 2 s after their last use; the first prompt then asks for all
12.8 GB at once with almost nothing free, and macOS compresses and swaps several gigabytes before the GPU can have it
(section 4.1, facts M1 to M4).

**The aim:** a conversation commits the cache memory its rows need, a little ahead of them, and nothing more; the
plan's context stays the ceiling it is today.

## 2. The contract, on every backend

### 2.1 No kernel reads a row nobody wrote

A cache row is read only once written. Rows past the held length are never loaded, whatever the tile shape; a kernel
that works in whole tiles bounds the last one by the rows held.

- **CUDA keeps this today.** Its flash prefill stages a key tile of `min(kFlashTileKeys, block_max_key - tile_start + 1)`
  rows into shared memory and reads no other (`Gqa.Flash.Bf16.cu`, lines 152 to 161), its decode zeroes the ragged tail
  as it stages it (`Gqa.Decode.Bf16.cu`), and it clears no cache.
- **Metal does not, in one case** (fact M5): its fused prefill reads a whole tile past the rows held when fewer than a
  tile's rows are held, and clears the whole cache at build so that those rows are zero. Section 4.2 is the fix.
- It is what makes uncommitted memory safe to leave uncommitted: nothing reads it.

### 2.2 Reserve the ceiling, commit as it fills

- **A cache's address range is reserved at its full capacity** when the network is built, and never moves: the layout,
  `[batch, KV heads, rows, head size]`, and every kernel stay as they are, and so do `KvCacheView` (the drafter reads
  another layer's cache by address) and a recorded decode step (`DecodeGraph.md`).
- **Memory is committed per KV head**, since each head owns a fixed range of rows: to hold rows `[0, n)`, each head's
  range `[head * capacity, head * capacity + n)` is committed, rounded up to the device's granularity. The capacity is
  rounded up so that each head's rows start on that granularity; it is only addresses, so the rounding costs nothing.
- **The sliding-window rings** (`SlidingWindowKvCache.md`) are bounded by the window and a chunk and written in their
  first prompt; whether they grow too is asked in section 12.

### 2.3 The network asks for rows before the work

- **`IKvCacheLifecycle` gains one call, "hold rows up to n"**, beside `initializeKvCache`, `resetKvCache` and
  `rewindKvCache`. The network calls it before work that writes past what it last asked for: a prefill writing up to
  `offset + T`, a decode writing position `p`. It asks in steps of **256 rows** (mlx-lm's, fact M9).
- **Its default commits nothing and returns**, for an operation whose cache is allocated whole, so every backend's
  network makes the same call and no CUDA operation changes until CUDA grows.
- **It is host work done before the work is encoded, never inside a decode step's launches.** A decode step stays a pure
  function of device memory (CLAUDE.md, `DecodeGraph.md`): under replay the network asks between replayed steps, once
  every 256 tokens, and the recording is untouched because no address changed. The rows committed are host state the
  operation keeps, written only by this call, which no decode step makes; `IKvCacheLifecycle`'s rule that the operation
  keeps no fill count is untouched, since the fill count stays the network's.
- **An operation refuses to write past the rows committed**: it checks on the host before encoding and throws a logic
  error naming the layer. On Metal a write to unmapped memory is lost with no error (fact M8), so this check is the only
  guard; its test breaks the commit on purpose and expects the throw.

### 2.4 Reset keeps; release gives back (agreed, Todd, 2026-11-11)

**Release is dropped (Todd, 2026-11-12).** The planner makes deployment decisions, and with a context sized to the
machine there is nothing worth giving back when a conversation ends. The section stays as the record of what was weighed;
that `resetKvCache` keeps its memory still holds.

- **`resetKvCache` keeps what is committed**, as `IKvCacheLifecycle.ixx` says today ("preserving the allocation"). It
  runs at the start of every full prefill that follows a decode (`GroupedQueryAttention.ixx`, `forward`), so a new
  prompt that cannot reuse the held prefix resets and then commits its rows straight away. Giving memory back there would
  only hand it to the system and take it back.
- **Nothing resets when a conversation ends** (no caller in `Mila::Agents`, Chat or the server, 2026-11-11), so the rows
  a long conversation committed would stay committed until the next prompt or the model's unload.
- **So a separate call gives memory back, "release down to n rows"**: everything committed above `n` rows, rounded up
  to a step, is returned; the reservation and its addresses stay, so `KvCacheView` and a recorded decode step stay valid.
  `n` is never below one step, so the next short prompt maps nothing.
- **The agent's session calls it**, since only the session knows a conversation has ended: through the model handle
  (`ModelHandle.md`) to the network, which calls it on every layer. The path is `Mila::Agents` and `Mila::Deployment`
  source, so it needs Todd's direction when built (CLAUDE.md, *API Boundary*).
- **When it is called:** when a conversation is cleared or its session ends, down to one step; and, proposed and not yet
  agreed, after compaction (`Agents/Compaction.md`), down to the rows the conversation holds once compacted (section 12).
- **A server's policy between requests** (keep memory warm, or give it back after a quiet spell) is a separate decision,
  not this design's.

### 2.5 What the plan means

- **The plan's context is the ceiling, priced and refused exactly as today.** `planDeployment` still prices every cache
  at its full capacity, so a model that cannot reach its context on a device is refused at load, as now.
- **What a load allocates is less than the plan** where the backend grows: the caches start at one step. *Arithmetic:*
  Gemma 4 12B's global caches at the first prompt hold 256 rows, 4 MiB, where today they hold 4 GiB.
- **A long conversation commits as it goes.** Whether a commit can fail where the plan said the context fits is an
  unknown on each backend (sections 4.4, 5.3).

## 3. The device's part

A device that can grow memory provides a growable allocation beside its ordinary one. Three operations, and nothing a
backend-neutral caller has to know about pages or heaps:

- **Reserve** a range of a given size, returning its address. Nothing is committed.
- **Commit** a list of byte ranges inside it. Already committed parts are left alone, so commit is idempotent; each range
  is rounded out to the device's granularity, which the device reports.
- **Release** a reservation: everything committed in it given back, and the range freed.

It is a capability, as `IKvCacheLifecycle` is: a device that does not provide it is not asked, and its operations
allocate whole, as today. The cache is its first user. The same three operations are what an expert cache needs on a
card (section 5), so they are written for any user, not for KV rows.

## 4. Metal (built first)

### 4.1 Facts

"Pinned" in this document is what `vm_stat` counts as wired; `Metal/MetalBackend.md` 5 (W1 to W5) holds what wiring
means on a Mac and what releases it.

M1. **First prompt, by context** (2026-11-11, `vm_stat` at 5 Hz beside each run): at 262,144 the GPU's pinned memory goes
    from 1.2 to 12.8 GB in the first prompt. With the unchanged binary, 7.4 and 8.5 GB were decompressed in the first
    prompt's 2 to 3 s, in the two runs read this way. With the weights read past the file cache (section 7), 2.3 GB went
    through the compressor in the fastest reading (948 ms), and about 10 GB were decompressed and 0.95 GB swapped in in
    the slowest (4105 ms). At 8192 the pinned memory goes to 8.8 GB, with 2.7 GB decompressed (unchanged binary) and
    0.4 GB (the load arm).
M2. **macOS unpins idle GPU memory** (standalone probe, `D:\Claude\metal\first-prompt\loadprobe.mm`): 7.6 GB of buffers
    pinned by a GPU read were unpinned between 1 and 2 s of GPU idle, with or without a residency set holding a standing
    residency request. *Corrected 2026-11-12:* it is macOS's wired collector, not Metal, and MLX's maintainers call it a
    macOS bug (`Metal/MetalBackend.md` 5, W3). Pinning them again took 0.11 s for 7.6 GB and 0.24 s for 12.6 GB when nothing had been compressed.
    MLX documents the same ("macOS can drop residency after GPU idle despite a standing request",
    `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS`, ml-explore/mlx `docs/src/usage/environment_variables.rst`).
M3. **An untouched buffer costs nothing until the GPU first uses it, then all of it** (same probe): a 5 GB shared buffer
    moved free memory by nothing when made, and the GPU's first read of 16 KB of it pinned all 5 GB, in 0.18 to 0.41 s.
    So a cache that is to grow cannot be one ordinary buffer.
M4. **A residency set pins at load and does not keep it** (same probe): every buffer and a 5 GB cache in one set with a
    standing request pinned 12.8 GB at once, 4.46 s with the memory full; 2 s idle later they were unpinned anyway (M2).
    *Corrected 2026-11-12:* the probe made one request and never renewed it. Renewing it on an interval is MLX's interim
    fix and has not been tried here (W3).
M5. **Metal's fused prefill reads unwritten rows in one case, and the cache is cleared for it.** The kernel
    (`GroupedQueryAttention.Flash.metal`) reads whole tiles of `TKeys` rows. When the last tile of a range would pass the
    rows held, it starts earlier so that it ends on the last held row and masks the overlap (line 196). Only when fewer
    than `TKeys` rows are held does the tile start at row 0 and run past them; the key and value tensors are declared
    `TKeys` rows whatever is held (line 208). Masked rows weigh zero, and zero times a value that is not finite is not
    finite, so `MetalGqaOp::build` clears every row of K and V and an FP8 cache's scales (since `+129`, `bc4a7438`), and
    the cache is given at least one tile of rows (`cacheRowsFor`). An FP8 cache widens `max(columns, tile.keys)` rows
    (`flashPrefill`) for the same case. *Arithmetic:* the clear writes 4 GiB at 262,144.
M6. **Every other Metal reader is bounded by what is held.** Decode reads positions up to the decode position
    (`gqa_decode_partial`, `p <= position`); the drafter's shared-cache attention uses the same decode kernels; the
    unfused prefill reads the rows attended (`block_columns`).
M7. **The M1 Pro has placement sparse buffers** (2026-11-11, `sparse.mm` on the Mac, macOS 27.2): `supportsPlacementSparse`
    is true, a 16 KiB sparse page is one 16 KiB tile, and a Metal 4 command queue can be made.
M8. **What the SDK says of them** (Xcode 27.2 headers): a placement sparse buffer is made at its full length with
    `newBufferWithLength:options:placementSparsePageSize:`, the options choosing its storage mode; memory comes from heaps
    of type `MTLHeapTypePlacement`, mapped a range of tiles at a time by `MTL4CommandQueue
    updateBufferMappings:heap:operations:count:` and unmapped the same way. At sparse tier 1, a read of an unmapped range
    returns zero and a write to one does nothing (`MTLResource.h`, `MTLBufferSparseTier1`).
M9. **MLX's policy** (mlx-lm 0.32.2, `mlx_lm/models/cache.py`, MIT): a cache starts empty and grows 256 rows at a time;
    the sliding layers' cache grows the same way up to the window, then rotates in place. Its mechanism is a new, larger
    array and a copy of the old one at every growth, so addresses move and the cache briefly exists twice; Mila takes the
    policy and not the mechanism. MLX's allocator puts every buffer in a residency set (`mlx/backend/metal/resident.cpp`),
    and mlx-lm raises the limit to the recommended working set while it generates.

### 4.2 No read of an unwritten row

- **The short tile is declared with the rows held.** In the one case of M5 the key and value tensors of the tile take the
  held row count as their extent, not `TKeys`. Apple's TensorOps header says a product edge-checks against its tensors'
  extents (`MPPTensorOpsMatMul2d.h`, around line 142); what it reads past them is not stated, which the gate settles.
  Every other tile keeps its fixed extent, as the header recommends for tiles wholly inside.
- **An FP8 cache widens only the rows held.**
- **The clear goes**, and so does the one-tile minimum in `cacheRowsFor`.
- **Gate: a cache full of NaN.** Before a prefill shorter than one tile, and before one that ends partway through a tile,
  every row of the cache is set to NaN; the output must match the reference. That tests the decision, that no read
  depends on an unwritten row, and not that zeros happen to be there. It fails when the bounded extent is reverted.

It lands on its own, before the growth. *Inference:* the load also stops writing 4 GiB (M5); not measured.

### 4.3 The growable allocation on Metal

- **Reserve** is a placement sparse buffer of the full length, entered in `MetalDevice`'s allocation map like any buffer,
  so binding by address (`MetalBackend.md` 5) is unchanged, if such a buffer has a host address (unknown 3).
- **Commit** maps the pages not yet mapped from the device's placement heaps, adding a heap when the last is full; the
  granularity is the 16 KiB page (*arithmetic*: 16 rows at head 512 in FP16, 32 at 256, 64 at 128, twice that for FP8
  codes, 4096 rows of FP32 scales; 256 rows is whole pages at every one).
- **Release** unmaps the pages and returns heaps with nothing mapped.
- **The heaps are kept resident through one residency set on the queue**, MLX's pattern (M9). M2 says that does not keep
  memory pinned through idle, so nothing here depends on it.

### 4.4 Unknowns, settled before any code

Each a probe outside the tree, like M7's; together about 15 minutes of machine time (estimate).

1. **The M1 Pro's sparse tier**: `sparseBufferTier` on a placement sparse buffer it makes.
2. **Shared storage**: whether such a buffer can be `MTLStorageModeShared`, as every Mila buffer is, or must be private.
   The cache is written and read only by kernels, so private may serve, if the allocation map can hold it.
3. **A host address**: whether it has `contents`. If not, the map keys it by `gpuAddress`.
4. **Two queues**: mappings are made on a Metal 4 command queue and Mila encodes on a classic `MTLCommandQueue`. Whether
   the classic queue's work sees a mapping once a shared event orders them, and what a commit of 16 ranges costs
   (*arithmetic*: Gemma 4 12B's 8 global layers, K and V, one head each).
5. **Whether a placement heap's memory is committed when the heap is made**, so that a heap's size is a real granularity.
6. **What a commit does when memory is short**: fails, and how, or succeeds and leaves macOS to compress.
7. **Rates**: whether a placement sparse buffer reads slower than an ordinary one, at prefill after 0, 4096 and 16,384
   held, and decode, one session.

## 5. CUDA (when a need asks for it)

### 5.1 The need

Nothing on CUDA needs it today: a card's memory is the model's. It would be built for a need of CUDA's own. The one in
view: card memory used as a cache of MoE expert weights, filled and emptied as experts are used (`MixtureOfExperts.md`,
expert residency), which needs the same reserve, commit and release as section 3. The KV cache would then grow on CUDA
too, through the same contract, freeing the memory a short conversation does not use for that cache.

### 5.2 The facts it would build on

- **CUDA's driver has the operations** (NVIDIA CUDA Driver API, *Virtual Memory Management*, read 2026-11-11):
  `cuMemAddressReserve` reserves a virtual address range; `cuMemCreate` makes physical memory, in multiples of the
  granularity `cuMemGetAllocationGranularity` reports; `cuMemMap` maps it into a reserved range, and `cuMemSetAccess`
  grants the device access; `cuMemUnmap`, `cuMemRelease` and `cuMemAddressFree` undo them. That reserving commits no
  memory is the page's description rather than a stated guarantee.
- **`cuMemUnmap` takes a whole mapped range, never part of one.** So a commit is mapped as its own piece, and release
  unmaps the pieces it mapped.
- **CUDA's kernels already keep section 2.1** (section 2.1's first bullet), so CUDA needs no kernel change to grow.
- **A captured CUDA graph records addresses**; a reserved range keeps them, so replay is untouched (`DecodeGraph.md`).

### 5.3 Unknowns, when it is built

The granularity on Mila's cards and the step it implies (*inference*: coarser than Metal's 16 KiB page, so the step may
be set by it); the cost of a map and its access grant; whether a commit can fail where the plan said the context fits;
rates of attention over a mapped range against an ordinary allocation.

## 6. Other backends

A backend provides section 3 or does not. One that does not allocates whole and keeps section 2.1, which every backend
keeps regardless. The host device has no compute operations (`CpuDeviceTypeTraits.ixx`) and no cache to grow.

## 7. Not in this design

- **What the planner offers as Metal's free memory.** `getMemoryInfo` offers the whole recommended working set, as though
  nothing else on the Mac needed memory (`Issues/Untriaged.md`). *2026-11-12:* this is the cause, not a separate change;
  it comes before any growth (`Deployment.md` 12.10).
- **The weights' load on Metal.** Reading each tensor from the file into its buffer without the file cache took Chat's
  load from about 10 s to about 2 s (2026-11-11, an arm on the Mac's copy). Its own change, MLX's way of loading
  (`mlx/io/load.cpp`, `ParallelFileReader`). *2026-11-12:* the load holds the weights twice, and fixing it is the next
  change (`Metal/MetalBackend.md` 5, *Weights*).
- **A prompt after the model has sat idle.** M2 suggests it pays the pinning again; not measured.
- **Shrinking as a conversation goes.** Rows are given back only by release (section 2.4), never by reset or rewind.

## 8. Phases

1. **Metal: the probe of section 4.4.** Its answers go into section 4.1 before anything is built; an answer that breaks
   the design comes back to Todd.
2. **Metal: no read of an unwritten row** (section 4.2). Lands on its own.
3. **The contract and Metal's growth**: `IKvCacheLifecycle`'s new call with its default, the networks' calls before
   prefill and decode, the device capability of section 3, Metal's implementation of it (4.3), `MetalGqaOp`'s rows and
   its host check. No CUDA source changes beyond the interface's default.
4. **Measured in Chat on the M1 Pro**, section 9's numbers before and after.
5. **CUDA**, when section 5.1's need is taken up: its own spec section, its probe of section 5.3, then the build.

## 9. Gates

- Phase 2: the NaN gate; every `GqaMetal*` test; each failing when the change is reverted on purpose.
- Phase 3: attention tests over a cache that grows across a step boundary in prefill and in decode, at every head size
  and both cache types; a forced missing commit caught by the host check; the drafter's tests over a grown cache;
  Gemma 4 12B's and Llama 3.2 3B's HuggingFace gates on Metal; the full suite on CUDA, which must not move.
- Phase 4, Chat on the M1 Pro, Gemma 4 12B Q4_0, alternated with the binary before, one session: first prefill at
  automatic context and at 8192, with the pinned memory and the compressor's traffic; the load's time; prefill at 0,
  4096 and 16,384 held, and decode; the context the Mac plans, which must stay 262,144.

## 10. What it edits when built

- `IKvCacheLifecycle.ixx`: the two new calls, "hold rows up to n" and "release down to n rows"; `resetKvCache`'s
  documentation says it keeps what is committed.
- `ModelHandle.md` and `Direction.md` 4.3: the session's release when a conversation ends.
- `DecodeGraph.md`: that the network asks for rows between replayed steps, never inside one.
- `MetalBackend.md` 4.5 (the clear and the one-tile minimum, replaced by section 4.2) and 5 (the growable allocation).
- `Deployment.md`: one line, that where a backend grows, a plan's caches are reserved and committed as they fill.
- This document's status.

## 11. Sources

- mlx-lm 0.32.2, `mlx_lm/models/cache.py` (`KVCache`, `RotatingKVCache`), MIT; ml-explore/mlx at `36969c0`,
  `mlx/backend/metal/resident.cpp`, `mlx/backend/metal/allocator.cpp`, `mlx/io/load.cpp`, MIT.
- Apple, Metal framework headers in the macOS SDK of Xcode 27.2: `MTLDevice.h`, `MTLHeap.h`, `MTLResource.h`,
  `MTLBuffer.h`, `MTL4CommandQueue.h`; Metal Performance Primitives, `MPPTensorOpsMatMul2d.h`.
- NVIDIA, CUDA Driver API, *Virtual Memory Management* (`group__CUDA__VA`).

## 12. Open decisions

1. **The step**: 256 rows, mlx-lm's, or another figure after phase 4's rates.
2. **A heap's size on Metal**: one figure for every cache, or sized to the model, after unknown 5.
3. **The sliding rings**: left as they are, or grown up to the window, as mlx-lm's are.
4. **Release after compaction**: whether compaction calls release down to the rows the compacted conversation holds
   (section 2.4). Decided: reset keeps, release gives back when a conversation ends (Todd, 2026-11-11). *Moot:* release is
   dropped (Todd, 2026-11-12).
5. **What Chat shows**: the plan's footprint, as now, or the plan's ceiling and what is committed.
6. **Where the device capability lives**: an interface of its own in `Compute`, beside `IKvCacheLifecycle`, or members of
   the device classes that provide it.
