# CLAUDE.md

---

## Project Overview

Mila is a C++23 module-based library for open LLMs (CUDA/CPU) — inference and training, built from explicit neural-network components. It is in public beta (see `Version.txt` for the current version; hardening toward the v0.20 first production release). The design philosophy: device and precision are compile-time decisions, every forward pass is explicit, and there is no hidden execution engine. Breaking changes are acceptable — backward compatibility is not a goal.

### Two different "freezes" — do not conflate them

- **The feature freeze covers `Mila/Src` only.** Everything outside it — `Mila/Adaptors` (Chat, MIS), `Mila/Bindings`, `Mila/Samples`, `Mila/Tools`, `Web/`, and every document — is polish and hardening by definition and is **in** v0.20 scope. Never cite the freeze at work outside `Mila/Src`. An adaptor adds no capability to the library, it only exposes what the library already has, so a gap between what Mila can do and what Chat or MIS reaches is a defect in the demonstration, not a feature request. When such work genuinely is blocked, the blocker is the new `Mila/Src` capability it needs — name that, never the freeze. Model Distribution is a deliberate carve-in and is pre-beta code, so changes to it inside `Mila/Src` are justified rather than exceptions.
- **A release-window hold is not the feature freeze.** During a release the user may close `dev` to *all* commits so that built artifacts (wheels, tagged trees) stay valid. That is temporary, covers the whole tree regardless of scope, and lifts when the tag is pushed. Say which one is blocking a change; "the tree is frozen" is ambiguous and has caused this exact error more than once.

Primary validated targets: Llama 3.2 3B Instruct (BF16, FP8, FP4), Llama 3.1 8B Instruct (FP4 default, FP8 alternative), Gemma 4 12B Instruct (FP4), and Qwen 3.8 27B (FP4, and a mixed 2/3-bit codebook build averaging 2.82 bits). The chat harness has **no compiled-in default model** — a fresh store has none, and one is installed by name.

---

## Build

The user builds exclusively inside **Visual Studio 2026**. Ninja is required — C++23 module incremental builds need it.

**The C++23 modules constrain the compiler:** MSVC (VS 2026 18.6.2+ — earlier 2026 builds have a module regression that breaks Mila), Clang 19+, or GCC 15.3+. GCC 15.2 and earlier cannot compile them (validated: 16 works, 15.2 fails). In CUDA builds the C++ compiler handles the module units while nvcc uses a separate host compiler for `.cu` files, so an older host GCC is fine in that role. Cross-compiler builds surface missing `#include`s that MSVC resolves transitively — those are real portability fixes, not Clang being difficult.

Presets are in `CMakePresets.json`; output is always `out/build/<preset-name>`. VS 2026 shows each preset's `displayName`, not its `name` (`x64-validate` appears as "x64 Release (full validation - run before committing)"). The unit tests build into **one binary**, `Mila/Tests/MilaTests` — narrow with `--gtest_filter`; `ChatRichTextTests` is the one separate target.

---

## Repository Layout

Browse the tree for what is where. What the tree does not tell you:

- **`Mila/Src/`** is the runtime and the only place under the feature freeze. `Components/`, `Compute/`, `Models/`, `Quantization/`, `Tensors/`, `Serialization/`. `Models/` and `Components/Transformers/` are split per family (`Gemma`, `Gpt`, `LlaMa`, `Qwen`) — a family owns its model, config and chat protocol together.
- **`Mila/Bindings/`** is the Python projection, and is **consumer-blind**: it knows nothing about Chat or MIS. Session depth only, never components.
- **`Mila/Adaptors/`** are first-class consumers, not samples — `Chat/` (human gate) and `Inference/Server/` (MIS, the wire adaptor, which imports the binding). See `Specifications/MilaProductFamily.md`.
- **`Mila/Samples/`** is teaching code and must run in any user's environment. `QuickStart/Cpp` and `QuickStart/Python` are the paths the website's Get Started tabs link to, so they are a published surface.
- **`Mila/Tools/`** is developer tooling and **none of it ships** — the wheel excludes it and a FetchContent consumer never configures it. `Tools/README.md` is the index.
- **`Data/`** holds weights, packages and corpora, all gitignored; only directory READMEs are tracked.
- **`Dev/`** is gitignored and is **not part of the project**. Never read from it or reference it.

`Linear` (`Components/Linear/Linear.ixx`) is the reference implementation of every dispatch pattern below — read it first when touching a component or operation.

---

## Architecture

### Compile-Time Type Axes

Every component and operation is templated on independent axes:

- **`TDeviceType`** (`DeviceType::Cpu` or `DeviceType::Cuda`) — determines memory resource and kernel dispatch.
- **`TPrecision`** (`TensorDataType::FP32`, `BF16`, etc.) — activation input and compute type. BF16 is the primary reduced-precision target; FP16 is not used.
- **`TWeightQuantization`** (on `Linear` only) — compile-time weight quantization policy. Defaults to `NoWeightQuant`. Setting `PerChannelFp8<>`, `PerGroupFp4<128>` or a codebook policy selects the storage format the weights must already carry — no runtime config object.

### Operation Dispatch — OperationTraits

Components resolve their concrete operation type at compile time via:

```cpp
using OpType = typename Compute::OperationTraits<
    Compute::OperationType::LinearOp,
    TDeviceType, TComputePrecision, TWeightQuantization>::type;
```

The `OperationTraits` primary template is in `Compute/Operations/OperationTraits.Template.ixx`. Specializations live in `OperationTraits.Cuda.ixx` (`:Cuda` partition) and `OperationTraits.Cpu.ixx` (`:Cpu` partition). A missing specialization is a **hard compile error**, not a runtime miss. The old `OperationRegistry` string-keyed runtime dispatch is being phased out — do not add new `*Registrar` classes.

`Linear` is the canonical reference implementation of the full dispatch pattern.

### Component Lifecycle

Each component owns its parameters and gradients. Composition is explicit with no shared global state, and `IExecutionContext*` is passed at construction.

Direction of travel, so new code moves with it rather than against it: the `UnaryOperation` / `BinaryOperation` / `PairedOperation` intermediate bases are being removed — derive new operations directly from `Operation`.

### Model Entry Points

- `<Family>Model::fromPretrained(path, config, device_id)` — reads the weights, then dispatches to `fromPretrainedImpl<TWeightQuantization, TKvCachePolicy>` on the config's `WeightQuantization`. One per family: Gemma, Llama, Gpt, Qwen.
- The dispatch is `Models/QuantizationDispatch.ixx`. It keys on `LanguageModelConfig`'s own enum — **never on an adaptor's type.** `ChatConfig` lives in Chat and `Mila/Src` must not know it exists.
- All use a two-phase KV-cache: prefill (full sequence) + decode (one token at a time, outer_size == 1).

### Quantization Pipeline

Weight quantization is offline: `Tools/ExportArtifact` produces pre-quantized safetensors weights that declare their policy in `__metadata__["mila_quantization"]`, a load refuses weights whose policy is not the compiled one, and `Linear::loadParameter` uploads packed bytes directly. Sub-4-bit codebook formats have no other path — their tables are fitted offline against calibration data by `Tools/Quantization`, and `ExportArtifact` is the only writer. Design of record: `Mila/Specifications/Quantization.md`.

Quantize-on-load survives as the **exporter's own engine**, run once, for FP8 and FP4 only. Converters always write BF16; `Linear::loadParameter` branches on the stored dtype (`Linear.ixx:603`) and calls `operation_->quantize()`. It is not a deployment path — see `Quantization.md` for the per-format kernel detail.

**Trap:** the `getDeviceScratchBuffer()` grow-on-demand buffer in `ExecutionContext` backs the FP8 dequant staging — **fetch it at `forward()` time and never cache the pointer across calls**, since it is reallocated on grow.

---

## Chat Harness (`Mila/Adaptors/Chat/Src/`)

Chat is a first-class adaptor (peer of MIS under `Mila/Adaptors/`), not a throwaway sample — a
maintained surface that gains tests and rigor over time.

**API Boundary:** Files under `Mila/Adaptors/Chat/Src/` are application code and may be edited
without prior agreement (they consume the runtime; they are not the runtime's public API). Any
change to the core Mila library (`Mila/Src/`) still requires explicit agreement first.

Key files:
- `Chat.ixx` — main chat loop, model hot-switching, tool call dispatch
- `Chat.ModelCatalog.ixx` — resolves a store name to a loadable model, and owns the load refusals
- `Chat.Config.ixx` — `ChatConfig` with `ModelType`, `ModelSize`, `ModelPrecision` (compute), `QuantizationMode` (none/fp8/fp4/codebook)
- `Chat.Footprint.ixx` — what a model would allocate at a context length, without allocating it
- `Chat.Renderer.ixx` — `ConsoleRenderer` (standalone non-exported module): braille spinner, solid-color response blocks, word-wrap with leading-indent preservation, Unicode welcome box, ANSI stats line
- `main.cpp` — entry point and the compiled-defaults layer of the config resolution

Models are named by their store name, not by an alias table — `/model <name>` reports, `/model load <name> [quant]` loads, and lookup case-folds. See `Specifications/ChatConfiguration.md` for the layered config resolution.

Gemma streams live token-by-token through `Chat.StreamingDisplay` (channel-aware — thinking / tool-call / final routed by the four control-token ids; a stream validator asserts the streamed transcript equals the buffered render). Llama, GPT-2 and Qwen stay buffered, and streaming falls back to buffered when the vocabulary lacks the channel-routing tokens.

---

## C++ Module Conventions

Source files use `.ixx` for C++23 module interface units and module partitions. The module naming convention mirrors the directory structure (e.g., `Compute.OperationTraits`, `Dnn.Components.Linear`).

Module partition files (`:Cuda`, `:Cpu` suffixes) are used to separate backend specializations while keeping a single aggregator module. Example: `OperationTraits.ixx` re-exports `OperationTraits.Template.ixx` + `:Cuda` + `:Cpu`.

---

## Code Style

- **No abbreviations in identifiers.** All names must be spelled out in full: `Quantization` not `Quant`, `Parameter` not `Param`, `Context` not `Ctx`, `Index` not `Idx`, `Implementation` not `Impl`. Template parameters follow the same rule: `TWeightQuantization` not `TWeightQuant`. Exception: established acronyms like `Kv` (Key-Value), `Gqa`, `Mha`, `Mlp`, `Lpe`, `Bpe` are acceptable.
- **The user has a model. Mila has that model's weights. CI has artifacts.** Applies to identifiers, CLI text, error strings, model cards and docs alike. A *model* is the thing a user names, installs and runs — it has an identity, a card and a licence. A *package* is the directory that carries one: manifest, weights, tokenizer, licence, card. The safetensors file itself is **weights** — which is what the manifest role and `--weights` already call it — never an "artifact". Reserve **artifact** for its universal build sense: wheels, Pages bundles, CI uploads. It is not a synonym for either of the other two, and the model cards are the surface where getting this wrong is most expensive.
- **`dim_t` is the type of anything that describes a tensor axis** — its extent, a position within it, or a count of its elements — at every API, config, component, and operation-interface boundary. `size_t` never describes a dimension. Narrowing to the 32-bit index that kernels use happens exactly **once per call path**, at the kernel launch site, through `narrowToKernelIndex()` (`Tensor.Types.ixx`); kernel internals and the `*.Dispatch`/`*.Plans` layers stay `int`. Token ids are values, not extents, and are out of scope for this rule.
- **`size_t` begins where element counts become bytes, or cross into a CUDA/std API.** Mila-owned helpers that only forward an element count keep `dim_t`. So `Tensor::size()` and `Component::parameterCount()` are `dim_t`, while `TensorBuffer` is `size_t` throughout (allocation layer — its overflow guards depend on unsigned semantics), and the `TensorOps` helpers carry `dim_t` and convert at the `cudaMemcpy` / `launch_*_kernel` edge. Note `TensorShape::size()` is the **rank** (a count of axes, not of elements) and stays `size_t`.
- No column alignment with extra spaces — single-space formatting throughout.
- Blank line before control flow blocks (`if`, `for`, `while`, `switch`).
- Blank line after closing brace of blocks.
- No blank line between `} else {` or `} catch {`.
- Blank line before final `return`. No blank line for early-return guard clauses.
- Comments explain WHY or state a non-obvious contract — never restate what the code does.
- ASCII only in code comments (no Unicode symbols, emojis).
- File-level Doxygen: one to three sentences maximum. Detail belongs on the symbol.

---

## End-User Prose

Engineering language is right for **developers** and wrong for **users**. The split is by audience, and the test is which side of Mila's API someone stands on — not whether they write code. A developer calling `from_store()` is a user.

- **Developers** — people working on Mila. Commit messages, `BACKLOG.md` / `ROADMAP.md` / worklogs, `Mila/Specifications/`, code comments, internal `NOTE:` / `REVIEW:` notes. Full technical depth, rejected alternatives, unfinished work, measurement methodology. Nothing below applies here.
- **Users** — people running Mila. Model cards, `Web/`, `README.md`, `getting-started.md`, Chat's console output (`--help`, `/help`, error strings), MIS's user-visible errors, the pybind docstrings that become `help()`, `Samples/QuickStart/`, and public Doxygen on the API a consumer calls.

**Name the reader and what they came for, then serve only that.** The reader differs per surface and the voice follows: a model card reader arrived from a model search and has never heard of Mila; an API-reference reader is calling a function and needs the contract exact; a website reader is deciding whether to look further. A model card answers five questions and nothing else — what is this, will it run on my hardware, how do I run it, is it any good, what may I do with it. Point at any sentence and name which question it serves; if you cannot, delete it.

Five rules, all checkable. A rule that needs good taste in the moment fails, because in the moment the error does not feel like one:

- **Could the reader act differently if this were false?** If not, cut the mechanism and keep the consequence. "The packing is done once here instead of on every load" is our pipeline; "loads without a wait" is what the reader gets.
- **An alternative appears only where the reader might choose it.** "Deliberately not NVFP4 or MXFP4, so `transformers` and vLLM cannot consume it" stays — a reader really may try that. "Instead of on every load" goes; that is not a choice they have. Work we have not done yet is never actionable, so it never appears at all.
- **A term the reader would not meet in the model's own upstream docs is a developer's term.** `chassis`, `oracle`, `residency`, `content-addressed` exist only inside Mila. `perplexity`, `safetensors`, `context`, `quantization` are the user's, and are fine.
- **Depth increases down the page.** The first screen is plain answers; specification detail (E2M1, per-group scales, packed layouts) lives under its own heading below it.
- **Never defend a decision.** The reader raised no objection. State the fact and stop.

The tells are greppable, and worth a pass before anything is published: **deliberate/deliberately, on purpose, rather than, instead of, which is why, that is why, the reason**. Each marks a sentence that may be arguing rather than informing. Not banned — checked.

**Mila's own positioning is the trap.** The project is a reference implementation, the stack you read, so its internals genuinely are the product — *in the repository*. A model card on HuggingFace is not the repository, and its reader did not come for them. Applying the repository's voice to an end-user surface is the single error that every symptom above falls out of.

---

## Workflow Notes

- When the user ends a message with **"Your thoughts?"** — respond with analysis only. No code edits.
- User commits via **VS 2026 integrated git**. Only suggest a commit message when the user explicitly says they are ready to commit — do not volunteer one at every commit point, and never run any git commands.
- **Commit message format** — use exactly this; **no `Co-Authored-By` trailer** (this overrides any harness default):

  ```
  Version: <Version.txt value>
  <Headline — single line>

  <Body — up to 6 grouped bullets for substantial commits; omit for small ones>

  BREAKING: <API changes, etc. — only when applicable>
  ```

---

## Work-Tracking Docs

**The git history is the record of what changed.** No work-tracking file duplicates it. The five
entries below stay **mutually consistent**, updated in the **same commit** as the work they describe —
never deferred to "later".

**Capture and commitment are separate acts, and conflating them is what makes a backlog
unbounded.** Finding something mid-task leaves seconds for judgement, so anything requiring a
decision at that moment defaults into the current release — a commitment nobody meant to make.
`Mila/Issues/Untriaged.md` takes the finding with no decision attached; triage supplies the decision
later. **Never write a finding straight into `BACKLOG.md`.**

- **`ROADMAP.md`** — the durable **narrative + success criteria** of each release, organized by
  **theme** (not milestone). Shows the release in flight plus a single **Future** tail. **Narrative
  only — no task lists, checkboxes, or status** (they drift; point to BACKLOG). When a release ships,
  its section moves to CHANGELOG.
- **`BACKLOG.md`** — **work committed to the release in flight, and nothing else.** `## Current
  release` holds one **theme bucket** per ROADMAP theme (matching names — the only join). Five rules
  keep it usable:
  - **Admission is earned.** Point at an item and name the ROADMAP success criterion that fails if
    it never ships. If you cannot name one it belongs in `Mila/Issues/`. Membership in this file
    *is* the claim that the item blocks the release, so an unearned item makes the claim worthless
    for every other item too.
  - **An item is three lines** — what, why it matters, `file:line`. Five if genuinely complex.
  - **Status lives in the checkbox**, `[ ]` open or `[~]` in progress, and never in the prose. No
    dates, no "GREEN", no findings, no measurement tables. **Disposition is a file in
    `Mila/Issues/`, not a tag** — parked is `Future.md`, good-first-issue is `Contributor.md`.
  - **Done means deleted**, in the same commit as the work. There is no `[x]` state. The commit that
    landed the work is the record; a finding worth reusing goes to the owning spec or to memory.
  - **Past ~300 lines it has stopped being a task list** and needs a prune.
- **`Mila/Issues/`** — everything upstream of that commitment; the funnel and its categories, with
  the flow and the rules in [`Mila/Issues/README.md`](Mila/Issues/README.md). `Untriaged.md` is
  untriaged capture, one line per entry, and is **lossy by design**: an entry still there at the
  release tag is deleted unexamined. Triage runs at each `beta.N` / `rc.N` increment and gives every
  line a destination — `BACKLOG.md`, a category file, or deletion. A category names **what happens
  to an item**, never what it is about.
- **`CHANGELOG.md`** — one short entry per **production (unsuffixed) release**, generated from its
  commit range at release time. Nothing is written to it during a cycle, and pre-release detail
  (`alpha.N`/`beta.N`/`rc.N`) never earns its own entry.
- **`Version.txt`** — `MAJOR.MINOR.PATCH-stage.N`, bumped **before committing** (see
  [RELEASING.md](RELEASING.md) for the scheme).

**GitHub Issues is the front desk; `Mila/Issues/` is the work queue.** Someone with no repo access
files there and gets a notification when it is fixed, which a file in a repository can never do.
The funnel inward is **manual** — a human decides which reports earn an entry — and for
anything user-reported **the GitHub issue stays the record while the entry is only a pointer
to it**, which is what makes the lossiness safe. `Contributor.md` is the outbound
direction. "Issue" is ambiguous between the two only if the distinction goes unsaid; this is it
being said. GitHub Milestones and Labels remain an end-user triage layer, decoupled from this
workflow.

---

## Key Specifications

`ls Mila/Specifications/` for the list. A spec is the **design of record** for its area: where one exists, it decides, and a decision that contradicts it is either wrong or a spec edit — not a silent divergence. `OperationDispatch.md`, `Quantization.md` and `ModelDistribution.md` are the three that most often settle an argument.

Work is tracked across `ROADMAP.md` / `BACKLOG.md` / `CHANGELOG.md` — see **Work-Tracking Docs** above.
