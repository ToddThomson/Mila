# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined at the production release tag**, and anything user-reported is a
pointer to its GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

## A fix to the published site sits in the repo until someone dispatches the workflow

`.github/workflows/publish-site.yml` (`on: workflow_dispatch:`) @ `c60c100a`

Found reading Google Search Console for `mila.toddt.me`: 141 pages reported **"Excluded by 'noindex'
tag"**, against 1 indexed page and 14 others crawled-or-discovered-but-not-indexed.

The `noindex` post-processing step was removed from the workflow at `0589673f` on 2026-08-13. The
workflow only runs on dispatch, and the next dispatch was 2026-09-11 — so the live site served
`noindex` across the whole `/api/` tree for 29 days after the repository had stopped saying it.
Verified on the live site 2026-09-22: no `robots` meta and no `X-Robots-Tag` on `/api/index.html`,
`/api/annotated.html`, `/api/files.html` or `/api/classes.html`. GSC validation is running.

Nothing reconciles the site source in the repository against what is actually deployed, and the
workflow's own verification step reads one file (`build/site/api/index.html`) out of the tree it
assembles.

## The scratch reservation test measures growth on the card that drives the display

`Mila/Tests/Dnn/Models/ScratchReservation.Cuda.cpp:150` @ `1f823df4`

`ScratchReservationCudaTests.Gemma4_12B_Fp4_Context8192` bounds device memory growth during
generation at 64 MiB, measured on the current CUDA device -- ordinal 0, the RTX 4070 that drives
the display, shared with 18 desktop processes. Same binary, three runs on 2026-09-23: 2.0, 1.9 and
116.6 MiB; a full-suite run earlier the same day read 375.3 MiB. Scratch predicted and reported were
identical in every run. The file's own note (`:40`) records noise reaching 47.6 MiB.

## A skipped test sends the reader to a backlog entry that no longer exists

`Mila/Tests/Dnn/Components/FFN/Swiglu/Swiglu.Cuda.cpp:358` @ `1f823df4`

`SwigluCudaTests/Bf16.Backward_MatchesReferenceGradients` skips with "BF16 SwiGLU backward
grad-dtype mismatch (FP32 kernel grads vs BF16 tensors) -- see BACKLOG". `BACKLOG.md` at `+3` has
no SwiGLU entry, so the skip's only record of why it exists points at nothing.

## The deployment types repeat their namespace in their names

`Mila/Src/Deployment/` @ `0.21.0-dev+6`

Moved to the top level in `Mila::Deployment` at `+6` (Todd, 2026-09-24: `Dnn/` held it inconsistently with its
siblings, which each build or run a network), with the type names `Deployment.md` 12.7 recorded left as they
were: `Mila::Deployment::DeploymentPlan`, `DeploymentPlans`, `DeploymentRequest`, `DeploymentRefusal`,
`DeploymentRefusedError`. The namespace would let them drop the prefix, as `Mila::Dnn::Conversation` did for
`Turn` and `ToolCall`. One layering fact the move made visible: unlike `Distribution/`, which is independent
of `Dnn` both ways, deployment imports `Dnn.Component`, `Serialization.WeightsReader` and `Compute.DeviceId`,
and `Dnn/Models`' three families import it. See the `DeviceReading` entry below.

## `DeviceReading` is the wrong name for what a plan was decided against

`Mila/Src/Deployment/DeviceReading.ixx` @ `0.21.0-dev+6`

Todd, 2026-09-23: the name is poor; kept for Phase 3 and to be replaced. What the type carries: one
device's identity, its free and total memory, and its allocation granularity, taken once after the graph is
constructed (`DeviceReading::take`). Every plan and refusal holds one (`DeploymentPlan::reading()`,
`DeploymentRefusal::reading()`), Chat's `-p` JSON reports its fields, and `Deployment.md` 3.3 and 9 use the word.
Decision 12.5 (planning for a device that is described rather than read) bears on the name, since a
described device is not a reading of anything.

## The /model list column prices capacity, which the planner cannot yet be asked about

`Mila/Adaptors/Chat/Src/Chat.ModelCatalog.ixx` (`largestFittingContext`) @ `0.21.0-dev+6`

Deployment Phase 3 moved Chat's load and `/context` onto `planDeployment`, but the listing's GPU FIT column
still walks its own context ladder through `getDeploymentFootprint` and `gradeFootprint`. It answers "could
this card run it at all" against the card's TOTAL memory, deliberately, so a resident model is not charged
to every row. A plan is decided against a live reading of FREE memory, so moving the column onto plans needs
a described reading -- `Deployment.md` 12.5, undecided. `Deployment.md` section 8 says the column renders
plans; it does not yet.

## A find_package consumer may be told C++20 where Mila's modules need C++23

`Mila/CMakeLists.txt:71-75` @ `a8d30790`

When CMake's feature table lacks `cxx_std_23` (the comment names Clang 21.x), the exported `Mila` target
advertises `cxx_std_20` to consumers, who compile Mila's installed module units under it. Those units
already use C++23 library facilities -- `std::ranges::fold_left` at `Mila/Src/Dnn/Tensors/Tensor.ixx:806`
-- which libc++ and libstdc++ provide only in C++23 mode. Not reproduced: whether CMake's synthetic target
then compiles at C++20, or the compiler's default wins, was not tested. Found while checking whether
`std::expected` (C++23) could appear in the public API for `Deployment.md` 3.3.

## A process's second automatic load chooses a shorter context than its first

`Mila/Src/Deployment/DeviceReading.ixx` (`take`) @ `0.21.0-dev+7`

Found verifying Deployment Phase 4 through the Python binding on the RTX 4070: four `GemmaModel.from_store(
"gemma-4-12b-it-fp4")` loads in one process, each deleted before the next, chose 121856, 120832, 120832, 120832.
Stable after the first, so not a leak -- memory the first load leaves held in the process (lazily loaded kernel
modules or a library workspace are candidates; not measured). A fresh process always chooses 121856. It reaches
any caller that loads twice in one process: Chat's `/model` switching and a Python program. Not measured how many
bytes, or which.

