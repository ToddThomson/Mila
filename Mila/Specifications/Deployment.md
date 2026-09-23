# Deployment Planning

Specification and implementation plan for deciding, once and in the library, how a model will run
on the hardware in front of it: which devices, how its blocks are placed across them, what context
length, and what prefill chunk. The decision is a value, the **plan**, that a load then executes
without deciding anything again.

Written 2026-09-16. Phases 1 to 4 are committed to v0.21.0 (2026-09-23, `Direction.md` 5.2); Phase 5
comes after it. Open decisions 1 and 3 were settled on their recommendations the same day.

---

## 1. Problem Statement

Every "auto" in Mila today is decided somewhere different, and one of them is decided twice.

| Decision | Made in | How |
|---|---|---|
| Prefill chunk | `Mila/Src`, inside each transformer's `onBuilding` (`Gemma.ixx:623`, `Qwen.ixx:645`, `Llama.ixx:577`) | reads free device memory at build |
| The same prefill chunk, predicted | `Mila/Src`, in `getRequiredMemory` (`Gemma.ixx:347`) | reads free device memory again |
| Context length `"auto"` | Chat, `resolveAutomaticContext` (`Chat.Footprint.ixx:448`) | scans candidate lengths through `getDeploymentFootprint` |
| Whether a model fits | Chat, `Chat.ModelCatalog.ixx` | grades against one device's free memory |
| All of the above | MIS and the Python binding | nothing: the caller passes explicit values |

Three consequences:

- **The prediction and the build can disagree.** Both read live free memory, at different moments.
  On a card that drives a display the free memory moves between the two readings, so the build can
  take a different chunk rung than the prediction named (`MemoryFootprint.md` 11.3). It is one of
  two candidate causes of the open `GemmaFootprintCudaTests.GetRequiredMemory_BoundsActualConsumption`
  failure in the maintainer's environment, not yet diagnosed.
- **Users of different surfaces get different products.** A Chat user gets an automatic context
  length; a Python or MIS user must know the number.
- **Every new knob would repeat the pattern.** Splitting a model across devices (`LayerSplit.md`)
  needs a device list and a placement chosen against the footprint. Built the way context length was,
  it would be a fourth decision site, in one adaptor.

---

## 2. What A Deployment Is

The values that decide how a model runs, and which of them a caller can choose today:

| Knob | Selectable at run time | Today |
|---|---|---|
| Devices | one CUDA ordinal | caller; `LayerSplit.md` makes it a list |
| Placement of blocks across devices | no | new, `LayerSplit.md` |
| Context length | yes, `withContextLength` | caller, or Chat's `"auto"` |
| Prefill chunk | no | the transformer, at build |
| Weight format | only for a BF16 package, by quantize-on-load to FP8 or FP4; a pre-quantized package fixes it (`requireStoredQuantizationMatches`) | caller |
| KV-cache compression | **no**: FP8 is refused on the unquantized path and ignored on the quantized ones (`QuantizationDispatch.ixx:103`) | not a knob |
| Head width for scoring | yes, `withLanguageModelHeadPositions` | caller, and it is a purpose, not a resource trade |
| Batch | fixed at 1 | not a knob |

The planner decides the first four. Weight format stays the caller's (section 12). KV-cache
compression becomes a planner input only when it is a real choice. Head width and batch are part of
the request, never decided.

---

## 3. Design

### 3.1 The fact the design rests on

**Pricing a deployment allocates nothing.** Construction allocates nothing (`MemoryFootprint.md`
3.1), and `getRequiredMemory( BuildContext )` is exact against the build (Gate A, `MemoryFootprint.md`
Phase 6 step 3). So every candidate deployment can be priced, many times, before anything is
committed to a device.

### 3.2 Plan, then execute

```cpp
DeploymentPlan plan = GemmaModel<Cuda, BF16>::planDeployment( path, request );   // decides; allocates nothing
auto model = GemmaModel<Cuda, BF16>::load( path, plan );               // executes exactly this plan
```

The plan is decided once, from one reading of each device's free memory, and the load executes it.
**The load never re-derives a value the plan holds.** The structural problem in section 1
disappears; it is not handled more carefully.

### 3.3 The types

- **`DeploymentRequest`** — what the caller asks for. Each plannable knob is either a value or `auto`:
  devices (a list, or `auto`), context length (a number, or `auto` between a floor and a ceiling, as
  Chat's `resolveAutomaticContext` already takes them, `Chat.Footprint.ixx:448`). Also
  the fixed inputs: weight format, head width, and **headroom** — bytes to leave free on each device.
  Headroom is a caller input: how much another process or a display needs is a fact about the user's
  machine, not a Mila constant.
- **`DeviceReading`** — one device's identity, free bytes and allocation granularity, taken once by the
  planner.
- **`DeploymentPlan`** — the resolved values (devices, `DevicePlacement`, context length, prefill chunk),
  the readings they were decided against, the per-device and total `MemoryStats`, and for each value
  the **constraint that bound it** (section 6). Also:
  - **What it was priced for.** The package facts pricing read: architecture, geometry and stored
    weight format. Not a content hash — two packages with the same geometry and format price
    identically, so a plan is valid for either, and invalid for anything else (section 7).
  - **Its ranked alternatives.** The objective's best plan for each device set it considered — each card
    alone, and the cards together — ordered by the objective, each a complete plan with its own
    footprints and binding constraints. The first choice is the plan itself. A caller who prefers "both
    cards at 65536" to "one card at 8192" takes that alternative instead, without planning again.
  - **Whether it is feasible.** A request nothing can meet still yields a plan, marked infeasible, naming
    the constraint that failed. There is no separate error path for "does not fit".

`DeploymentFootprint` and `PrefillChunking` (`Component.MemoryStats.ixx:236`, `:263`) are the pricing of
one candidate and stay as they are. A plan is a decision among candidates. **"Footprint" names a price;
"plan" names a decision.** Neither is used for the other.

### 3.4 The plan across the lifecycle

A plan's parts are consumed at different stages, and each stage receives only its own part:

| Part | Consumed at | Carried by |
|---|---|---|
| Weight format | dispatch, before construction: it selects the transformer type (`QuantizationDispatch.ixx`) | `WeightQuantization` (existing) |
| Devices and block placement | construction: execution contexts bind once, here | `DevicePlacement` (`LayerSplit.md` section 4) |
| Context length, batch, prefill chunk | build | `BuildContext`, which gains the resolved chunk |
| Readings, footprints, binding constraints | nothing in the graph | `DeploymentPlan`, kept by the model |

A transformer never sees a `DeploymentPlan`. `DeviceId` keeps meaning one device.

### 3.5 Alternatives rejected

**Keep "auto" in the adaptors.** What exists today. Every adaptor re-implements it or goes without,
and a Mila capability (splitting a model) would reach users only through whichever adaptor implemented
it. Rejected.

**Decide inside the build, more carefully.** Takes one free-memory reading during `build()` and
threads it to the prediction. It still makes the decision after construction has bound devices, so it
cannot choose devices or placement. Rejected.

**A single `fit()` call that loads.** Hides the decision inside the load, so a caller cannot see or
report what was chosen, or price a deployment without committing to it. `fit` also means *train* in
most ML libraries, and Mila trains. Rejected in favour of plan, then execute.

**Prior art.** llama.cpp decides the same question inside its load: `--fit` (on by default) adjusts
"unset arguments to fit in device memory", with a per-device margin `--fit-target` (default 1024) and a
minimum context `--fit-ctx` (default 4096); `--split-mode layer`, the default, splits layers and KV
across GPUs pipelined (llama.cpp `tools/server/README.md`, read 2026-09-16). Two of this design's
choices match it: only unset values are decided (section 5, rule 1), and the margin is per device
(`DeploymentRequest` headroom). It differs in returning the decision as a value before anything loads,
and in never falling back to host memory for layers (section 13).

cuBLASLt's matmul API is prior art for the shape rather than the problem, and this design takes three of
its ideas, not its interface: a query that returns **ranked candidates, each with its own resource cost**
(its heuristic's per-algorithm workspace size; here, alternatives with per-device footprints); a **check**
that puts a caller's own choice through the same validation as a chosen one (here, a fully fixed request,
section 7); and **a decision valid only for the inputs it was made for** (an algorithm for its descriptors;
here, a plan for the package facts it priced). It does not take cuBLASLt's opaque ranking, which is what
the binding constraint replaces, nor its attribute-setter style, which Mila's builders already cover.

### 3.6 One path, three depths

Simple to use, powerful when needed, and consistent throughout. Each depth is the same mechanism with more
of it visible:

```cpp
// 1. Just run it: every plannable knob auto.
auto model = GemmaModel<Cuda, BF16>::load( path, DeploymentRequest{} );

// 2. Look first: plan, inspect, then run exactly that.
DeploymentPlan plan = GemmaModel<Cuda, BF16>::planDeployment( path,
    DeploymentRequest{}.withContextLength( 32768 ).withHeadroom( 512 * MiB ) );

if ( !plan.feasible() )
    report( plan.bindingConstraint() );

auto model = GemmaModel<Cuda, BF16>::load( path, plan );

// 3. Choose among trades: take an alternative the objective ranked second.
auto model = GemmaModel<Cuda, BF16>::load( path, plan.alternatives()[ 0 ] );
```

(Names illustrate the shape; section 12 leaves them open.)

Three rules keep it consistent:

- **One request type for every surface.** C++, Chat, the Python binding and MIS all send a
  `DeploymentRequest`; no surface has a knob the others cannot express.
- **Every load is plan, check, execute.** A request with every knob fixed is planned and checked like any
  other, so an explicit deployment and an automatic one cannot diverge (section 7).
- **A model reports the plan it runs.** A loaded model exposes its `DeploymentPlan`, so what an adaptor
  shows is the object that ran, not a second derivation of it.

Per family the surface is two entry points, `planDeployment` and `load`, and three types.
Nothing else is added.

---

## 4. Pricing Without A Bound Context

The planner must price a graph for devices it has not bound to, and for more than one candidate device.
Today pricing reads the bound context in two ways:

- **Allocation granularity**: `allocationGranularity( this->getDeviceId() )` at every report and
  prediction site added by `MemoryFootprint.md` Phase 6 step 3 (about 30 files; for example
  `Rope.ixx:284`, `MixtureOfExperts.ixx:302`).
- **Free memory**: `readFreeDeviceBytes( this->getDeviceId() )` in each transformer's
  `getRequiredMemory`, `prefillChunkingFor` and `resolvePrefillChunkSize`.

**The rule: `getRequiredMemory` reads nothing from the bound context.** Granularity arrives in the
`BuildContext` it is given, and free memory is never read below the planner. That makes pricing a pure
function of configuration and `BuildContext`, which is what `MemoryFootprint.md` 4.1 already claims
("it receives exactly what `build()` receives").

Two consequences worth having:

- **One graph prices every candidate device.** The planner constructs a graph once and evaluates many
  `BuildContext`s against it.
- **A deployment can be priced for hardware that is not present**, which is `MemoryFootprint.md`'s
  original goal and is not true today, since a price needs a context on a real device.

The chunk rungs (`kGemmaPrefillChunkRungs` and siblings) become a family trait the planner reads;
`prefillChunkingFor` and `resolvePrefillChunkSize` leave the transformers.

---

## 5. The Objective

"Auto" hides decisions unless the order in which it trades one knob against another is stated. The
planner applies this order, strictly:

1. **Honour every value the caller fixed.** If they cannot all be met, the plan says so and names the
   binding constraint (section 12 decides whether the load then refuses).
2. **Fewest devices.** A model on one card beats the same model split: no crossings, and no second card
   (possibly the display) carrying part of it. Devices are only added when no single named device can
   hold an acceptable deployment.
3. **A full prefill chunk over a longer context.** Among fitting context lengths, the largest one whose
   prefill chunk is not reduced by memory.
4. **Otherwise, the largest fitting context.** When every fitting context prefills below a full chunk.
5. **Otherwise, no plan**, with the reason.

Rules 3 and 4 are the order Chat's `"auto"` applies today (`Chat.Footprint.ixx:509-532`: a
constrained chunk costs prefill throughput, and giving up context to avoid it is worth doing only while
there is context to give up). Phase 3 moves that rule, it does not change it; candidate context lengths
keep Chat's step of 1024 (`Chat.Footprint.ixx:368`) and its early stop once the weights alone exceed the
budget.

The budget on each device is its free memory at the reading, less the requested headroom. A caller who
wants a different trade fixes that knob.

---

## 6. The Plan Record

For each resolved value the plan carries a **binding constraint** as an enumerated reason, not text:
for example `ContextBoundByMemory`, `ContextBoundByFullPrefillChunk`, `ContextBoundByTrainedMaximum`,
`SplitBecauseNoSingleDeviceFits`, `NoDeploymentWeightsExceedAllDevices`. With the readings and the
per-device footprints, that is enough for any adaptor to explain the choice in its own voice. The
library never phrases it.

This is not a diagnostic addition to `Mila/Src`. It is the answer to the question every adaptor already
asks and currently answers for itself.

---

## 7. The Load Executes The Plan

- **Check.** `load( path, plan )` first compares the package's facts with the ones the plan was
  priced for, and refuses a mismatch naming both. It refuses an infeasible plan, naming its binding
  constraint. It never substitutes a value of its own: no fallback chunk, no default device.
- **Execute.** It dispatches on the plan's weight format, constructs with the plan's `DevicePlacement`,
  builds with the plan's `BuildContext` (context length and resolved chunk), loads, and keeps the plan.
- **`load( path, request )`** is `planDeployment` followed by the same check and execute.
- **A fully fixed request is not a second path.** Every knob set explicitly, including a single device, is
  still planned and checked; the planner's job reduces to pricing it. `load( path, config,
  device )` becomes exactly that request.
- **The build uses the chunk it is given.** `resolvePrefillChunkSize` is deleted from `onBuilding`; a
  `BuildContext` without a resolved chunk is a programming error, not an invitation to decide.
- If free memory fell between planning and loading, the load fails cleanly on allocation
  (`0.20.0-rc.1+11`). It does not re-plan.

The anti-pattern this section rules out is already in the tree at a smaller scale: when cuBLASLt's
heuristic returns no algorithm, `CublasLtPlan.ixx:333-337` logs "will use default at execution" and
continues, deciding again at execution. Recorded in `Mila/Issues/Vnext.md`.

---

## 8. Entry Points And Adaptors

- **Library.** `planDeployment( path, request )` per family, beside `load` and
  `getDeploymentFootprint`. The family comes from the package; a family-neutral entry point that reads
  it from the metadata is an open question (section 12).
- **Chat.** `"auto"` for `context_length` and for `device` both become fields of one request. An explicit
  value that does not fit is refused with the plan's binding constraint, no longer attempted with a
  warning (section 12.2); `--help` and `ChatConfiguration.md` §6 change with Phase 3.
  `resolveAutomaticContext` and the fit grading in `Chat.ModelCatalog.ixx` are replaced by reading the
  plan. `/models` renders the plan's binding constraint.
- **Python binding.** `from_store( name, context_length="auto", devices="auto" )`: the same request.
  The QuickStart samples and `getting-started.md` change with it.
- **MIS.** Its settings (`config.py`) take `auto` for context and devices and pass them through the
  binding. MIS gains the automatic context it has never had.

---

## 9. Traps

- **One reading per device, per plan.** A second reading anywhere below the planner reintroduces the
  disagreement this design exists to remove. Negative N1 (section 10) guards it.
- **Read after the graph exists.** Constructing a graph creates its execution context, which holds
  device memory, and the build reads free memory with that context alive. A reading taken before
  construction sees more than any build will find. Measured in Phase 1: moving Gemma's reading ahead
  of construction moved Chat's auto context on the RTX 4070 from 121856 to 122880, deterministically,
  with Gate A and G3 both still passing -- only G2 saw it. The planner reads after it constructs.
- **The display card.** Its free memory moves on its own. The plan is exact against its reading; the
  load can still fail if the desktop takes memory in between, and fails cleanly.
- **Search cost.** Contexts x chunk rungs x device sets x split points. Construction allocates nothing
  but creates execution contexts; the planner constructs one graph per candidate weight format and
  varies only `BuildContext` and placement.
- **Behaviour Chat users rely on.** Chat's `"auto"` is a published surface. Phase 3 must reproduce its
  choices exactly on the measured models before deleting it (G2).

---

## 10. Validation

All gates run in the maintainer's environment (`x64-profile`, both GPUs visible), and each is forced to
fail once. Bounds are written here before any run.

- **G1. Plan equals build, per device.** For every model and card in the matrix below, and every chunk
  rung the planner can pick: the plan's per-device `MemoryStats` equal what the built network reports,
  category by category (Gate A, from one reading). Gate B's residual bound per device is carried over
  from `MemoryFootprint.md` unchanged.
- **G2. Chat's choices are preserved.** Before Phase 3, record what Chat's `"auto"` picks (context
  length, chunk, `bounded_by_prefill`) for Gemma 4 12B FP4, Llama 3.1 8B FP4, Qwen 3.8 27B FP4 and cb2-3,
  on each card that holds them, pinned by UUID, with the free memory each run saw. After Phase 3, the
  planner given those readings picks the same values.
- **G3. Pricing does not read the bound context.** The same graph priced with two different granularities
  in `BuildContext` gives two different, individually correct predictions.
- **G4. A fixed request agrees with the planner.** For each plan G1 builds, a request fixing every knob to
  that plan's values yields an identical plan: same values, footprints and feasibility.
- **G5. Alternatives are real plans.** For a model that fits one card, each alternative loads and passes
  G1 on its own device set.
  **Recorded 2026-09-23 at `0.21.0-dev+3`** (`x64-claude-verify`, Release, both GPUs visible, card chosen
  with `--device`, `mila-chat -p --output-format json`, whose `context_measurement` carries these fields).
  Free memory is the scan's own reading; the planner is given it after Phase 3 and must pick the same row.

  | Model | Card (UUID) | Free bytes | Context | Chunk | Unconstrained chunk | Held back for a full chunk |
  |---|---|---|---|---|---|---|
  | gemma-4-12b-it-fp4 | 4070 `GPU-11770557` | 11645485056 | 121856 | 1024 | 1024 | yes |
  | llama-3.1-8b-instruct-fp4 | 4070 `GPU-11770557` | 11645485056 | 13312 | 512 | 512 | yes |
  | qwen3.8-27b-fp4 | 4070 `GPU-11770557` | -- | -- | -- | -- | -- |
  | qwen3.8-27b-cb2-3 | 4070 `GPU-11770557` | 11645485056 | 3072 | 1024 | 1024 | yes |
  | gemma-4-12b-it-fp4 | 5060 Ti `GPU-9a81c7d1` | 15908995072 | 131072 | 1024 | 1024 | no |
  | llama-3.1-8b-instruct-fp4 | 5060 Ti `GPU-9a81c7d1` | 15908995072 | 33792 | 512 | 512 | yes |
  | qwen3.8-27b-fp4 | 5060 Ti `GPU-9a81c7d1` | 15908995072 | 4096 | 1024 | 1024 | yes |
  | qwen3.8-27b-cb2-3 | 5060 Ti `GPU-9a81c7d1` | 15908995072 | 64512 | 1024 | 1024 | yes |

  Qwen FP4 on the 4070 has no plan: the scan found no fitting context, fell back to 4096, and Chat
  attempted the load anyway and failed on allocation (exit 5, weights 13.20 GB against 10.85 GB). After
  Phase 3 that row is a refusal naming `NoDeploymentWeightsExceedAllDevices`, not a failed load.

- **Negatives.** N1: the transformer re-reads free memory during build — G1 fails on the display card
  when free memory is changed between plan and build. N2: rules 3 and 4 swapped — G2 fails. N3:
  granularity taken from the bound context — G3 fails. N4: a plan priced for Gemma 4 12B passed with
  Llama 3.1 8B's package — the load refuses before constructing anything. N5: an infeasible plan passed
  to `load` — refused, and no device memory is allocated.

---

## 11. Phasing

**Phase 1 — pricing without a bound context.** Granularity through `BuildContext`; free-memory reads leave
`getRequiredMemory`. No behaviour change. *Exit:* Gate A exact on every existing footprint test; G3.
*Met at `0.21.0-dev+4`:* `BuildContext` carries granularity and free memory and a prediction without them
throws; each model entry point takes one reading after construction. G3 is
`GemmaRequiredMemoryCudaTests.PricesAtTheGranularityTheContextCarries`, forced to fail by N3 (one component
reading its own device) while every Gate A case still passed. G2 re-run: all eight rows identical.

**Phase 2 — the build executes a chunk it is given.** `BuildContext` carries the resolved chunk;
`resolvePrefillChunkSize` leaves `onBuilding`; the rung walk moves to one planner-owned function reading one
reading. *Exit:* G1 on the single-device matrix; N1. Re-run the open display-card footprint failure and
record whether it clears.

**Phase 3 — `planDeployment` on one device.** The request, plan and binding constraints; Chat's `"auto"`
context and fit grading move into it and are deleted from Chat. *Exit:* G2 and N2; Chat piped sessions for
every family.

**Phase 4 — the binding and MIS.** `context_length="auto"` for Python and MIS; samples and documents
updated. *Exit:* a QuickStart Python session and an MIS request each run with `auto`.

**Phase 5 — devices.** The device list, `"auto"` devices and placement: `LayerSplit.md` Phases 4 and 5 land
here as a new dimension of the planner, not beside it.

---

## 12. Open Decisions

1. **Does the planner ever choose the weight format? DECIDED 2026-09-23: no, for this spec's
   phases** — the caller fixes it; the plan reports whether it fits. For a BF16 package, FP8 against
   FP4 is a quality trade, not only a memory one.

   **Direction beyond these phases (Todd, 2026-09-16):** a caller names only the base model — "Gemma 4
   12B" — and Mila chooses among that model's published pre-quantized packages in the store, then plans the
   deployment on the user's hardware. That makes the package itself a plannable knob, chosen by the same
   objective with the variants' footprints as candidates, and it needs a statement of how a quality
   difference between variants is ranked against memory. Not designed here; it builds on Phase 3 and on
   `ModelDistribution.md`'s store.
2. **A fixed value that does not fit. DECIDED 2026-09-16: refused.** The plan is infeasible and a load of it
   is refused, whichever depth the caller used (section 7). This changes a published Chat behaviour: an
   explicit `context_length` is today "honoured as written, with a warning if it will not fit" (Chat's
   `--help`, `ChatConfiguration.md` §6); from Phase 3 it is refused, naming the binding constraint.
6. **The library's default request. DECIDED 2026-09-16:** every plannable knob `auto`, weight format the
   package's own, headroom zero. Adaptors may pin any of it — Chat's device default is `LayerSplit.md` 12.3.
7. **Names.** `DeploymentRequest`, `DeploymentPlan`, `planDeployment`, `alternatives`, `feasible` and
   `bindingConstraint` are working names.
3. **Headroom defaults. DECIDED 2026-09-23: zero in the library; the adaptors choose.** The
   alternative was Chat, MIS and the binding sharing one non-zero default. llama.cpp defaults its per-device margin to
   1024; Chat removed its own stacked margins in `MemoryFootprint.md` Phase 6 step 4 because the
   prediction already counts what Mila allocates.
4. **A family-neutral `planDeployment`.** One entry point reading the family from the package, or one per
   family. Chat's catalog already dispatches on family; the binding does too.
5. **Planning for hardware that is not present.** Section 4 makes it possible. Whether the request can
   carry described devices instead of readings is a later decision.

---

## 13. Non-Goals

- Re-planning during generation.
- Kernel or throughput autotuning. The plan decides memory-bound values; it does not benchmark.
- Training deployments.
- Two models sharing devices in one process. Chat releases a model before loading the next.

---

## 14. Relationship To Other Specs

- `MemoryFootprint.md` — the pricing contract. Section 11's chunk rule moves here in Phase 2; section 4.1's
  claim that pricing receives exactly what the build receives becomes true in Phase 1.
- `ChatConfiguration.md` §6 — Chat's `"auto"` context, which moves here in Phase 3.
- `LayerSplit.md` — devices and placement, the planner's device dimension (Phase 5).
- `Quantization.md` — quantize-on-load, which makes weight format selectable for BF16 packages.
