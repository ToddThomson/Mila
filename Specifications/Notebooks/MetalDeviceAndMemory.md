# Metal Phase 1: device, memory, one kernel

Phase 1 of `Specifications/Metal/MetalBackend.md`, on the M1 Pro. The gate is written before the run.

## 1. The gate

**Question:** can Mila allocate, dispatch and read back through its own device and memory types?

| # | Check | Passes when | Forced failure that shows it can fail |
|---|---|---|---|
| 1 | The tree builds | All targets on `macos-clang-release` with Metal on, no warnings from Mila's Metal files | |
| 2 | Refusals | Each of the three checks refuses alone, in plain words, naming the value found | Remove one check: its test fails |
| 3 | Allocation map | Base and interior addresses resolve with the right offset; one past the end, a neighbour's gap and a removed allocation refuse | Off-by-one in the end test: its test fails |
| 4 | Scalar fill, a real kernel | Every element of FP32, FP16, BF16 and all integer types reads back as the expected value | A kernel that writes nothing: its test fails |
| 5 | The format is the one claimed | A value FP16 can hold and BF16 cannot reads back differently in each, each as its format rounds it | Bind `fill_half` to BF16: its test fails |
| 6 | Host values, copy, zero | Values read back, in the same type and across types | |
| 7 | Order | Work encoded on one context runs in order, across compute and blit, and is visible after `synchronize()` | Blit before compute: its test fails |
| 8 | A kernel error surfaces | An unknown kernel name refuses, and names it | |
| 9 | Report | Runs, and compiles and creates every kernel | |
| 10 | Windows | All targets and the suite unaffected | |

Reference values come from the test itself, in double precision or from the format's definition. A test that only checks
completion is not a test here (`MetalBackend.md` 4.2).

## 2. What is measured

All on AC, under `caffeinate -dimsu`, power recorded at both ends, the sleep log checked.

- **First-load compile cost:** compiling Mila's kernel source, and creating every pipeline.
- **Commit cost per dispatch:** what it costs to commit and wait, against encoding many dispatches into one command buffer.
- **Binding:** per buffer on the encoder and through the allocation map, against one residency set with GPU addresses.

## 3. Gate results

On `macos-clang-release`, macOS 27.2, at `0.21.0-dev+117` plus this change.

| # | Check | Result |
|---|---|---|
| 1 | The tree builds | All targets with Metal on, from a clean directory; no warnings from the Metal files |
| 2 | Refusals | Pass. Removing each of the three checks fails its test |
| 3 | Allocation map | Pass. The end off by one, and a dropped offset, each fail their test |
| 4 | Scalar fill | Pass, on FP32, FP16, BF16 and six integer types. A kernel that writes nothing fails |
| 5 | The format is the one claimed | Pass. 1 + 2^-10 reads back whole in FP16 and as 1 in BF16; 70000 is infinity in FP16 and 70144 in BF16. BF16 through `fill_half` fails |
| 6 | Host values, copy, zero | Pass, in one type and across types, and on views |
| 7 | Order | Pass. A blit run ahead of the open work fails. So do a read back and a host fill that do not wait |
| 8 | A kernel error surfaces | Pass, by name |
| 9 | Report | Runs: Apple M1 Pro, `applegpu_g13s`, 14 cores, Apple7, 11.84 GiB working set, 8.88 GiB largest buffer, 9 kernels |
| 10 | Windows | `x64-claude-verify`, all targets, 0 warnings. The suite at 2475 pass, 12 skip and the same 4 filed failures as before this change; the separate test binaries pass. Todd's `x64-profile` was not run |

The Mac's suite: `MilaTests` 1291 pass and 1 skip, of which 54 are Metal's; Agents 10, Chat 33, Desktop 14. 12 of 12 forced
failures were caught (`D:\Claude\metal\phase1\mutate.py`, run on the Mac's copy, restored after each).

**What the tests found:** the first zero of a view wrote its whole buffer's size from the view's offset, because
`Tensor::getStorageSize` is the buffer's. Metal counts a tensor's own elements. CUDA's zero has the same line and is filed in
`Issues/Untriaged.md`, unrun there.

## 4. Measurements

From `D:\Claude\metal\phase1\measure.cpp`, raw metal-cpp so that each arm is the Metal calls alone, except the map arm, which
uses `MetalAllocationMap`'s lookup. 300 allocations, each dispatch binding its own three buffers, 1024 FP16 elements each,
values checked after every arm. Three process runs of 3 rotations of 400 steps, on AC under `caffeinate -dimsu`, no sleep in
the window. Medians, which agree between runs within 1%.

**Compile.**

| | Time |
|---|---|
| The first source this process compiles, never seen before | 33 ms (five fresh processes, 33 to 38 ms) |
| A later source, never seen before | 3.7 ms for 4 kernels |
| A source the cache already has | 27 us. Through Mila, in a fresh process: 0.6 to 0.9 ms for all 9 |
| Creating one pipeline | 66 us new, 31 us cached. Through Mila, all 9: 0.8 ms |

So a first load costs about 35 ms today, on 9 small kernels. **No `metallib` or pipeline cache is earned.** Phase 3 adds real
kernels, so this is asked again there.

**Commit.**

| | Time |
|---|---|
| One dispatch, committed and waited for | 222 us |
| 12 dispatches, a commit for each | 2724 us |
| 12 dispatches, one commit | 270 us |
| 200 dispatches, one commit, bound through the map | 798 us |

Arithmetic, on these numbers: a commit and wait is about 220 us and each further dispatch about 3 us. **A decode step commits
once.** This is what the context already does, since an operation encodes and only `synchronize()` commits. A dispatch here
does almost no work, so these are the overhead and not a kernel's time.

**Binding.**

| | Encoding one dispatch | 200 dispatches, one commit |
|---|---|---|
| Buffers already known | 0.30 us | |
| Through the map | 0.27 us | 798 us |
| Through the map, with an autorelease pool per dispatch, as `dispatch` does | 0.32 us | |
| One residency set and GPU addresses | 0.23 us | 768 us |

The map costs nothing measurable at 300 allocations: the known-buffers arm is slower than it, within the noise of this
timing. A residency set saves about 30 us in 200 dispatches, 4%, on kernels that do almost nothing. **Binding per buffer through the map
stays.** Besides the 4% being small, Metal cannot see which buffers a kernel reaches through an address it was handed, so it
could not order the dispatches that depend on each other, and Mila would have to place barriers itself. That is inferred
from what the binding gives Metal, and was not tested: the arm's dispatches here are independent. Phase 4 can reopen this
with a real decode step.

### metal-cpp in three module interfaces, against none

Phase 1 kept metal-cpp out of every interface, with three `.ixx` and `.cpp` pairs, for fear of Clang's source-location
ceiling, and without measuring it. This compares that with the same code folded mechanically into the three `.ixx` files:
headers in the global module fragment, bodies after the class. `MetalCppImplementation.cpp` stays in both. Same Mac, on AC,
one job at a time, Clang 23.1.5, in separate copies of the tree.

| | Split | Folded |
|---|---|---|
| Clean build, `ninja` only, two runs | 73.3, 73.2 s | 76.1, 76.1 s |
| Compile one file that does `import Mila;` (`Metal.TensorOps.Metal.cpp`), median of 5 | 2120 ms | 2175 ms |
| Source locations that file uses, of 2048 MB | 63.8 MB, 3.1% | 105.3 MB, 5.1% |
| Source location entries in its modules | 990,855 | 1,564,708 |
| Binary module files in `Mila.dir`, Metal's | 3.3 MB | 41.7 MB |
| Binary module files in `Mila.dir`, all | 173.4 MB | 211.5 MB |
| `libMila.a` | 12.7 MB | 12.7 MB |
| Rebuild after appending a comment to the execution context's source | 2.6 s | 31.4 s |

Source locations are arithmetic from the lowest module base `-print-stats` reports, 2^31 less that base, so local locations
are not counted. `-Rsloc-usage` printed nothing we could read. The file is a test with three further imports, not Chat, whose
command the script did not find. In the folded copy all targets built and the 54 Metal tests passed; the rest of the
suite was not run there.

The ceiling is not close in either arm. Each interface that carries metal-cpp adds about 14 MB of source locations
(arithmetic: 41.5 MB over three), so the count of such modules is what would matter: 68, as CUDA has for its runtime
header, would be about 950 MB. The operations need none, since they go through `MetalExecutionContext::dispatch`.

**Folded** (Todd, 2026-11-04), with the bodies in the class, as the library writes them, which is not quite the form measured
above. The split's one gain was the rebuild after editing these three files. The Phase 1 results in section 3 are from
before the fold, and were run again after it.

## 5. What went wrong on the way

- **The first timings were discarded.** The encode arm rebound the same three buffers, which may be cheaper than a model's
  binds, so each dispatch was given its own. It made no difference.
- **A run's numbers came from the previous binary** when a compile failed and the script carried on. They were thrown away,
  and the script now removes the binary first.
- **The first 200-dispatch arm reported wrong values:** the probe's rotation wrote over a buffer the check reads. The
  regions were separated, and the timings it gave were not used.
