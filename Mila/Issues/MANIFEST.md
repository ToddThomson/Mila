# Triage manifest

Every `BACKLOG.md` item, its origin line in the file at `586e327a`, and its destination. This is
the recovery index: if a move turns out wrong, this says what moved and from where.

**No item is deleted in this pass, and no item is folded into a spec.** Those are the only lossy or
expensive operations, and both need evidence gathered per item. Candidates carry a flag instead:

- `dup?` — the record looks to exist elsewhere already; verify, then delete in a later pass.
- `spec?` — belongs in its owning spec; verify the spec text, then fold and delete in a later pass.

Destinations: **keep** (same bucket) · **→bucket** (different BACKLOG bucket) · **Future** ·
**Contributor** · **Declined** · **merge** (folded into another entry, named).

---

## Models — 39 items

| Line | Item | Destination | Tags |
|---|---|---|---|
| 36 | `ExecutionContextFactory` `#ifdef` in module purview | merge → *`#ifdef` in module purviews* | `architecture` `build` `mila-src` |
| 39 | `Component` documents a contract it does not declare | →API Documentation | — |
| 43 | `Component` carries training's bookkeeping without training's act | merge → *API Coherence* | `api` `mila-src` |
| 47 | `GemmaConfig::getRotaryDimForLayer()` is dead code | Contributor | `models` `mila-src` |
| 51 | FP32 materializing softmax kernels store and reload | merge → *Materializing softmax kernels* | `perf` `mila-src` |
| 55 | Nothing stops a new BF16 softmax kernel reacquiring it | merge → *Materializing softmax kernels* | `perf` `mila-src` |
| 59 | Parity script cites `kGemmaDumpActivations`, which is gone | Contributor | `docs` `observability` |
| 64 | Consumer cannot instantiate a CUDA component without a non-public import | →Production Hardening | — |
| 69 | No build or CI step runs `compute-sanitizer` | Future | `ci` `mila-src` |
| 74 | Llama chassis never received Gemma's memory gates | Future | `models` `quantization` `mila-src` |
| 79 | Gate B has no unquantized case | **keep** | — |
| 82 | Attribute the Gate B residual `[~]` | **keep** (in progress) | — |
| 87 | `cudaMemGetInfo` cannot see WDDM's shared allocation | **keep** + `spec?` → `MemoryFootprint.md` | — |
| 92 | DECISION OWED — `withInstalledOutput` is an unenforced promise | **keep** | — |
| 97 | Gemma owes a block-level Gate A case | **keep** | — |
| 102 | Leaf-level Gate A for `Rope` is unwritten | **keep** | — |
| 105 | GPT-2 has no `getRequiredMemory` | **keep** (absorbs 175) | — |
| 108 | A pre-flight that cannot answer says nothing at all | Contributor | `adaptors` |
| 111 | `FamilyTraits::default_context` is a compiled-in guess | Future | `adaptors` |
| 115 | The published model cards still say `/install` | →Model Distribution | — |
| 118 | `/context`, `/set`, `/thinking`, `/model` have no tests | Future | `adaptors` |
| 122 | An `unknown` GPU FIT verdict prints no reason | Contributor | `adaptors` |
| 126 | `/models` measures a per-model context and throws it away | Future | `adaptors` |
| 131 | `temperature`/`top_k`/`top_p` have no command-line flags | Contributor | `adaptors` |
| 135 | A per-row disk figure should be reclaimable bytes | Future | `distribution` |
| 138 | No Llama parity test exists `[~]` | **keep** (in progress) | — |
| 143 | RoPE scaling is disabled on the Llama load path | **keep** | — |
| 146 | Triage `Llama.Block.ixx:132` view-aliasing | **keep** | — |
| 148 | Tool calling validated on Llama 3.2 3B and 3.1 8B | **keep** | — |
| 149 | GQA standalone-`forward()` stub | merge → *GQA forward paths* (Future) | `mila-src` |
| 153 | GQA `forward()` fallback is stale | merge → *GQA forward paths* (Future) | `mila-src` |
| 155 | `CudaMhaOp.ixx:433` `active_max_seq_len_` reason unrecorded | Future | `mila-src` |
| 157 | `GptModel.ixx:386` hardcodes `eos_token_ = 50256` | Contributor | `models` `mila-src` |
| 158 | `LlamaModel`'s context-overflow guard has no test | →Test Suite Revival | — |
| 162 | `cuda_fp4a16_gemm` falls through to a silent no-op | Future | `quantization` `mila-src` |
| 167 | `AttentionOutputGate` has two callers, one not attention | merge → *API Coherence* | `api` `mila-src` |
| 171 | The MTP head cannot be gated against HuggingFace | merge → *Gemma 4 MTP* | `models` |
| 175 | `getRequiredMemory` unimplemented on nine components | merge → line 105 (**keep**) | — |
| 179 | Llama converter writes `norm_eps`; reader parses `norm_epsilon` | Contributor | `models` `docs` |

**Models totals:** 11 keep · 4 to another bucket · 7 Contributor · 16 to Future (in 12 entries,
4 merged into existing) · 1 merged into a keeper · 1 `spec?` flag · 0 deleted.

The bucket goes from 39 items to 11 — and the 11 that remain are the ones that name a Models
success criterion directly: the footprint gates (79, 92, 97, 102, 105), the Llama parity chain
(138, 143, 146), tool calling (148), and the WDDM measurement note (87).

---

## Observability — 1 item

| Line | Item | Destination | Tags |
|---|---|---|---|
| 186 | `matchesPath`'s `*` crosses the path separator; its doc says otherwise | **keep** | — |

Stays. `*` selected 816 components on a 48-layer model, so "attaches by pattern" is met only in a
degraded sense, and the contradiction sits in public Doxygen. Todd's call which side to fix — the
glob or the doc — but the doc needs an edit either way, because `:402` ("any run of characters")
and `:405` ("but not their children") already contradict each other before you reach the code.

---

## Test Suite Revival — 8 items

| Line | Item | Destination | Tags |
|---|---|---|---|
| 195 | Re-green the authored component / tensor / tokenizer suites `[~]` | **keep** (in progress) | — |
| 198 | Core `Tensor.ixx` coverage to the value-type archetype `[~]` | **keep** (in progress) | — |
| 202 | Backfill inference-drought coverage `[~]` | **keep** (in progress) | — |
| 205 | Test coverage behind the samples `[~]` | **keep** (in progress) | — |
| 208 | A test discards a `[[nodiscard]] GenerateStatus` | Contributor | `models` `ci` |
| 212 | Backward-path kernels disabled or unverified | **keep** | — |
| 217 | `[crash]` GPT-2 CPU path to current standards | Future — **contested, see below** | `mila-src` |
| 222 | `ResidualConfig` advertises an unimplemented scaling factor | →Training Revival | — |

**217 is the one judgement call I am least sure of.** Production Hardening's criteria say outright
"GPU-first: the CUDA backend is the validated inference path; full CPU op parity is not a gate",
which sends it to Future. Against that: it is tagged `[crash]`, CPU inference has never run a
prompt shorter than its built context, and there is a CPU-only build the samples are meant to run
in. If the QuickStart samples touch that path on a CPU-only machine it is an onboarding failure and
belongs back in the release. Flagged rather than settled.

## Training Revival — 8 items

| Line | Item | Destination | Tags |
|---|---|---|---|
| 230 | Data-loader contract tests `[~]` | **keep** (in progress) | — |
| 233 | Re-enable the AdamW path `[~]` | **keep** (in progress) | — |
| 236 | Training-loop integration test `[~]` | **keep** (in progress) | — |
| 238 | Optimizer step-convergence test | **keep** | — |
| 240 | TrainingMode / RuntimeMode coverage | **keep** | — |
| 243 | `[decoupled]` Revive the loss + backward path | Future (absorbs PH:287) | `training` `mila-src` |
| 245 | `[net-new, training-only]` Revive the `Dropout` component | Future | `training` `mila-src` |
| 246 | Validation — FP32 training path, CI-gated | **keep** | — |

**243 revises what I proposed for the first fifteen.** I had PH:287 (`SoftmaxCrossEntropy` never
compiled) merging *into* this item. Seeing the bucket whole, this item does not itself earn
admission — it is marked `[decoupled]`, off the critical path, and the Training criterion is
explicit that both samples compute loss host-side. So the two travel together to Future instead.
The one piece that stays in the release is the separate, freeze-compatible truthfulness fix:
deleting the `<CrossEntropyOp, Cuda, BF16>` row that resolves to a silent stub.

## API Documentation — 2 items

| Line | Item | Destination | Tags |
|---|---|---|---|
| 252 | Tier 3 — semantic staleness (retired-world prose) | **keep** | — |
| 254 | Nothing checks Doxygen when doc drift is introduced | **keep** | — |

Both name their criterion directly. This bucket also *receives* three items: Models:39, and
PH:311 (`TPrecision`/`TWeightQuant`) from the first fifteen.

---

## Running totals after five buckets — 58 items

| | Models | Obs | Test | Train | API Doc | Total |
|---|---|---|---|---|---|---|
| keep | 11 | 1 | 6 | 6 | 2 | **26** |
| → another bucket | 4 | 0 | 1 | 0 | 0 | **5** |
| Contributor | 7 | 0 | 1 | 0 | 0 | **8** |
| Future | 16 | 0 | 1 | 2 | 0 | **19** |
| merged into a keeper | 1 | 0 | 0 | 0 | 0 | **1** |
| deleted | 0 | 0 | 0 | 0 | 0 | **0** |

`BACKLOG.md` after this half of the pass: **190 → 162 items, 820 → 720 lines.** The 28 that left
reconcile exactly — 8 to `Contributor.md`, 19 to `Future.md`, 1 folded into the GPT-2
`getRequiredMemory` keeper.

Bucket counts, before → after: Models 39 → 11 · Test Suite Revival 8 → 6 · Training Revival 8 → 7 ·
API Documentation 2 → 3 · Observability 1 → 1. The three that grew did so by receiving relocations:
Production Hardening 30 → 31, Model Distribution 30 → 31.

## Product Family — Adaptor Validation — 29 items

Line numbers are from `BACKLOG.md` at `586e327a`, before any of this pass was applied.

| Line | Item | Destination | Tags |
|---|---|---|---|
| 601 | `gemma_protocol.py` retired in place — delete it when ready | **keep** | — |
| 604 | In-turn thoughts dropped between tool calls | **keep** | — |
| 606 | Buffer Gemma Anthropic streaming only when tools are present | Future | `adaptors` |
| 607 | MIS Gemma 4 tool-calling validated end-to-end `[~]` | **keep** (in progress) | — |
| 611 | Qwen streams nothing — the harness routes by Gemma's control ids | **keep** | — |
| 615 | Qwen tool results are not merged into one user turn | Future | `adaptors` `mila-src` |
| 619 | Prompt-prefix reuse unavailable on DeltaNet, and silent | **keep** | — |
| 624 | Model capabilities belong in the manifest, not a family switch | Future | `adaptors` `distribution` |
| 630 | `ToolCallParser::parse` routes any `[` into the tool-call path | Contributor | `adaptors` |
| 635 | `ModelSize` is dead | Contributor | `adaptors` |
| 638 | A session cannot move cards without restarting | Future | `adaptors` |
| 642 | Library log output collides with Chat's spinner | merge → *Chat's console output* | `adaptors` |
| 647 | `printThinking` takes the plain-text path | merge → *Chat's console output* | `adaptors` |
| 651 | Wrapped list items do not hang-indent | Contributor | `adaptors` |
| 654 | `Chat.StreamingDisplay` has no tests | merge → *Chat and the quick starts have no test path* | `adaptors` `ci` |
| 658 | Chat's `context_length` needs an `auto` | merge → *Where Chat's configuration lives* | `adaptors` |
| 663 | Decide where a user's Chat config lives | merge → *Where Chat's configuration lives* | `adaptors` |
| 668 | A non-interactive `chat` must name its model | Future | `adaptors` `docs` |
| 673 | The download bar restarts per file and never says which | Future | `distribution` `mila-src` `breaking` |
| 678 | `mila serve <args>` is broken on Windows | **keep** | — |
| 683 | `Chat.Json` duplicates the `nlohmann.json` module | Contributor | `adaptors` `build` |
| 687 | `main.cpp` re-checks what the store already guarantees | Future | `adaptors` |
| 690 | Rework Chat configuration to layered resolution `[~]` | **keep** (in progress) | — |
| 693 | `import Mila;` breaks the standard library in a consumer TU | →Production Hardening | — |
| 698 | Make `packaging_fetchcontent_consumer` instantiate a model | →Production Hardening | — |
| 702 | The Python binding discards `GenerateStatus` | →Production Hardening | — |
| 707 | Neither Chat nor the quick starts have a test model | merge → *Chat and the quick starts have no test path* | `adaptors` `ci` |
| 713 | A full `mila-chat` QA pass is owed | merge → *Chat's model-and-session commands have no tests* | `adaptors` |
| 717 | Does a Python completion sample need a `GptSession`? | Future | `binding` |

**Product Family totals:** 7 keep · 3 to Production Hardening · 4 Contributor · 15 to Future (in 11
entries, 1 merged into an existing one) · 0 deleted.

The three that moved to Production Hardening are the C++ and Python consumer story — they name that
bucket's "an external consumer can build against Mila via FetchContent" criterion, not this one's.
The seven that stay are the criterion itself: the grammar-written-twice cleanup (601), the foreign
harness round-trips (604, 607), Qwen on the same terms as any other model (611), and its
prefix-reuse refusal surfacing as a model property (619).

## Production Hardening — 65 items

Line numbers are from `BACKLOG.md` at `586e327a`. The first fifteen were dispositioned before the
full pass was agreed; two of those calls changed on review and are marked.

### Top level — 30

| Line | Item | Destination | Tags |
|---|---|---|---|
| 261 | The Phase 5 prompt set is six prompts | merge → *Widen the quality harness* | `models` `quantization` `mila-src` |
| 266 | The 16K perplexity gate needs a re-run before a 32K claim | **keep** + `dup?` → `Qwen3.8.md:435-439` | — |
| 270 | CUDA device 0 is not `nvidia-smi`'s | Contributor | `docs` `api` |
| 275 | The head's two paths do not agree to the last digit | **keep** + `dup?` → `Qwen3.8.md:509`, `:546` | — |
| 279 | Do not build the device-side scoring reduction | Declined | `perf` `quantization` `measured` |
| 283 | Only Qwen can be scored | merge → *Widen the quality harness* | `models` `quantization` `mila-src` |
| 287 | `SoftmaxCrossEntropy` has never been compiled | merge → *Revive the loss + backward path* — **revised** | `training` `mila-src` |
| 293 | §8 gates the oracle on a test that cannot pass | **keep** + `spec?` → `Qwen3.8.md:693-700` | — |
| 297 | Every FP4/FP8 number is Ada-at-x16 | Future | `perf` `docs` |
| 300 | `PerGroupFp4` FP32 scales vs FP16 simulation | Future + `spec?` → `Qwen3.8.md` §4/§5 | `quantization` `mila-src` `breaking` |
| 305 | The FP4/FP8 wire formats are defined only by a kernel | Future + `spec?` → `Quantization.md` Part II | `quantization` `ci` `gate` |
| 311 | `TPrecision` / `TComputePrecision` | →API Documentation | — |
| 317 | `getStorageSize` is implemented three times | merge → *API Coherence* | `api` `mila-src` |
| 322 | Isolate third-party warnings structurally | merge → *Warnings-as-errors ratchet* — **already ruled there** | `build` `ci` `blocked` |
| 326 | `save_` public on `Component`, protected on `CompositeComponent` | merge → *API Coherence* | `api` `mila-src` |
| 331 | BPE pre-tokenization runs the ASCII fallback everywhere `[~]` | **keep** (in progress) | — |
| 337 | The BPE fallback warning fires for every session | Future | `tokenizer` `adaptors` `mila-src` |
| 342 | Two CUDA memory resources throw with no message | Contributor | `mila-src` |
| 346 | `GroupedQueryAttention.ixx:216` C4702 left deliberately | merge → *Warnings-as-errors ratchet* | `build` `ci` |
| 350 | `Chat.Footprint.ixx` defines a variable in the GMF | merge → *`#ifdef` inside module purviews* | `architecture` `build` `mila-src` |
| 354 | `IExecutionContext` is exported but unreachable | **keep** | — |
| 358 | If C1128 recurs, move `/bigobj` project-wide | Future | `build` |
| 361 | The README's six CI badges are decorative fiction | **keep** | — |
| 365 | Three different GCC floors, only one measured | **keep** | — |
| 368 | Both onboarding docs state build-option defaults backwards | **keep** | — |
| 373 | A FetchContent consumer inherits Mila's `docs` target | **keep** | — |
| 378 | The preset list names four presets that do not exist | **keep** | — |
| 383 | `getting-started.md:229` pins the dev container a release behind | **keep** | — |
| 386 | `CLAUDE.md` documents the retired Chat alias set | Future | `docs` |
| 390 | `CMakeLists.txt:266` pins curl at 8.11.1 | **keep** | — |

**Two revisions to the earlier proposal.** 322 was already ruled in `Future.md`'s warnings-ratchet
entry ("Land it after v0.20 ships"), so it was never Todd's call to make — it merges rather than
being asked about. And 287 was a *third* copy: `Test Suite Revival:195` parks
`SoftmaxCrossEntropy` and `Training Revival:243` carries it as `[decoupled]`.

### Release mechanics — 10

| Line | Item | Destination | Tags |
|---|---|---|---|
| 396 | `[gate]` The wheel matrix has never had a clean-room run on Windows | **keep** | — |
| 401 | The published wheel still stops before Blackwell | **keep** | — |
| 406 | Add Python 3.14 once 3.12 is proven | Future | `ci` `build` |
| 410 | Publish `mila-llm-server` to PyPI | **keep** | — |
| 414 | CI jobs have no `timeout-minutes` | merge → *CI has three cost and reliability gaps* | `ci` |
| 418 | Split the packaging gate into its own job | merge → *CI has three cost and reliability gaps* | `ci` |
| 423 | A `dev` push and an open PR run the pipeline twice | merge → *CI has three cost and reliability gaps* | `ci` |
| 428 | `actions/setup-python@v5` declares deprecated Node 20 | Contributor | `ci` |
| 431 | Add the Samples build to CI | **keep** | — |
| 432 | Broaden CI compiler coverage | Future | `ci` `build` |

### Container — 17

| Line | Item | Destination | Tags |
|---|---|---|---|
| 436 | Publish the Docker runtime image `[~]` | **keep** (in progress) | — |
| 443 | Every container build path defaults to an unusable arch | **keep** | — |
| 448 | Decide the container tag scheme | **keep** | — |
| 453 | ONE image holding all of Mila, with two entry points | **keep** + `dup?` — reads as a shipped decision | — |
| 457 | The real split is devel vs runtime | **keep** + `dup?` — reads as a shipped decision | — |
| 462 | Docker Hub Overview page needs a source in the repo | **keep** | — |
| 465 | Nothing cites `scripts/dockerhub/` | Future | `ci` `docs` |
| 469 | The runtime image ships a binding that cannot import `[~]` | **keep** (in progress) | — |
| 474 | The `ldd` gate passes when the file it checks is absent | **keep** | — |
| 478 | The binding's staged extensions accumulate in the source tree | Future | `build` `binding` |
| 483 | The wheel VERSION file is written into the source tree | **keep** | — |
| 487 | `Docker/build-mis.sh:76` looks broken on the current image | **keep** | — |
| 492 | The devel image's `mila-chat` wrapper shares its name | Contributor | `build` `docs` |
| 497 | `Docker/README.md:69` credits ChatApp with `MODELS_DIR` | Contributor | `docs` |
| 501 | Stage model weights off the Windows bind mount | Future | `perf` `build` |
| 502 | Reproducible container build `[~]` | **keep** (in progress) | — |
| 504 | Linux/clang as a first-class platform `[~]` | **keep** (in progress) | — |

**453 and 457 are flagged rather than kept confidently.** Both read as design decisions for work the
`[~]` runtime-image item says has already shipped — one image with three verified verbs, and a
`Dockerfile.runtime` that exists. Verify, then delete.

### Library hygiene — 8

| Line | Item | Destination | Tags |
|---|---|---|---|
| 509 | Dispatch error UX `[~]` | **keep** (in progress) | — |
| 511 | Five files hand-roll the staging memory resource | merge → *`#ifdef` inside module purviews* | `architecture` `build` `mila-src` |
| 516 | Module import hygiene | Future | `build` `mila-src` |
| 519 | Delete the 16 `REVIEW:` markers already dispositioned | Contributor | `mila-src` |
| 522 | The `fopen` → `<fstream>` conversion in three modules | Future | `build` `mila-src` |
| 527 | ProgressReporter | Future | `api` `mila-src` |
| 530 | `Version::getMajor()`/`getMinor()`/`getPatch()` are non-const | Contributor | `api` `mila-src` |
| 532 | Guided reading path | **keep** | — |

**Production Hardening totals:** 33 keep · 1 to another bucket · 7 Contributor · 1 Declined ·
23 to Future · 0 deleted. Four `dup?` and three `spec?` flags raised.

---

## Final totals — all 190 items

| | Count |
|---|---|
| Stayed in `BACKLOG.md` | **96** |
| → `Future.md` | 74 entries, absorbing more items through merges |
| → `Contributor.md` | 24 entries |
| → `Declined.md` | 1 entry |
| **Deleted** | **0** |

`BACKLOG.md`: **190 → 96 items, 820 → 451 lines.** Lines per item is 4.7, against CLAUDE.md's
threshold of 4 — close, and the remaining excess is in the Container and Model Distribution buckets
where several items carry five lines.

**Flags to resolve in a later pass, with evidence:**

- **`dup?` (4)** — PH:266, PH:275, Container:453, Container:457. Each looks like a record that
  already exists elsewhere; verify the other copy before deleting.
- **`spec?` (4)** — PH:293 (`Qwen3.8.md` §8's unmeetable oracle gate), PH:300 (§4/§5 state 4.125
  bits where the runtime ships 4.25), PH:305 (`Quantization.md` needs a normative FP4/FP8 layout),
  Models:87 (`MemoryFootprint.md` needs the WDDM measurement note). PH:293 is the substantive one.
- **Contested (1)** — Test Suite Revival:217, the `[crash]` GPT-2 CPU path. Sent to Future by the
  "full CPU op parity is not a gate" criterion, against the fact that it is a crash on a path the
  QuickStart samples may reach.
- **No ROADMAP theme — resolved.** The `#### Website` sub-bucket had no theme to name, so nothing in
  it could be admitted. The website is a separate project with its own cadence and publish, so it
  now has its own funnel at [`Web/Issues/`](../../Web/Issues/README.md): the six items moved to
  `Web/Issues/Backlog.md`, joined by two website items that had gone to `Contributor.md` (the blog
  `discussion:` line and the orphaned Achilles assets). Same entry format, its own four area tags —
  `content`, `layout`, `brand`, `publish` — because nearly everything on a website is `docs`, which
  tags nothing.

---

## Website — 8 items moved to `Web/Issues/`

| Origin | Item | Destination |
|---|---|---|
| Website:581 | Reconcile `start.md` with the Get Started band `[~]` | `Web/Issues/Backlog.md` |
| Website:586 | The home page hardcodes `0.20.0-beta.3` in three places | `Web/Issues/Backlog.md` |
| Website:591 | The Evaluating band leaves a stopped container behind | `Web/Issues/Backlog.md` |
| Website:596 | `docs.md:28` states "quantization has no checkpoint format" | `Web/Issues/Backlog.md` |
| Website:599 | The site links GitHub and nothing else | `Web/Issues/Backlog.md` |
| Website:601 | Mila is a library, never a "runtime" `[~]` | `Web/Issues/Backlog.md` |
| Website:606 | A blog post ships with no `discussion:` line | `Web/Issues/Backlog.md` (via `Contributor.md`) |
| Website:608 | Two orphaned brand assets carry the old Achilles mark | `Web/Issues/Backlog.md` (via `Contributor.md`) |

The release coupling survives the split and is worth restating: the home page hardcodes the Mila
version in three places, so a release tag and a site publish still have to name the same version.
Separate project means separate cadence, not separate from the release.

Production Hardening 65 (31 + 10 Release mechanics + 17 Container + 8 Library hygiene) ·
Model Distribution 39 (31 + 8 Website) · Product Family — Adaptor Validation 29.

Fifteen of Production Hardening's have dispositions proposed before the full pass was agreed; those
carry forward with two corrections — its warnings-isolation item was already ruled in `Future.md`,
and its `SoftmaxCrossEntropy` item was a third copy.
