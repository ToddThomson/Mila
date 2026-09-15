# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined at the production release tag**, and anything user-reported is a
pointer to its GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

## The clang portability build has no parallelism cap and dies at the memory frontier

`.github/workflows/build-pipeline.yml`, the `Build` step @ `2ff42c28`

`cmake --build build -- -k 0 -j $(nproc)` is the only build path in the tree with no memory cap;
`Docker/build-chat.sh:35` caps at 4 and explains why ("build parallelism is a MEMORY limit, not a
core-count one"), and `Dockerfile.runtime` sets `CMAKE_BUILD_PARALLEL_LEVEL=4` after the uncapped
build wedged Docker Desktop twice. The `ubuntu-24.04` runner is 4 vCPU / 16 GB and the tail
translation units — the Qwen dispatch tree, `Mila_py.Wrappers.cpp` — cost several GB each.

Found on the beta.3 release PR: run 34617000055 died `exit 137` at target `[1087/1246]`, with the
last compiler output at 15:54:37 and the kill at 16:45:35 — fifty-one minutes of silence, which is
thrash rather than a crash. The identical job had passed in 1h8m13s an hour earlier on the same
content, so it is flaky at the frontier and this release is what pushed it there. Re-running cleared
it; Todd chose that over changing the workflow mid-window.

## Two dockerhub scripts cannot execute under WSL, because checkout gives them CRLF

`scripts/dockerhub/verify-image.sh` @ `2ff42c28`

`git ls-files --eol scripts/dockerhub/` reports `i/lf w/crlf` for `verify-image.sh` and
`build-runtime-image.sh`, and `i/lf w/lf` for `publish-image.sh`. Running the first under WSL fails
instantly with `env: $'bash\r': No such file or directory`. The stored form is LF, so a Linux clone
is fine and no published artifact is affected — `.gitattributes` carries only `* text=auto`, so a
Windows checkout writes CRLF. `publish-image.sh` escaped because a tool last wrote it with LF and
the tag checkout rewrote nothing.

Found running RELEASING step 9 on a Windows box. Worked around by extracting the stored blob
(`git show <tag>:<path>`), which the release window required anyway since the tree had to stay clean
for the push gate. `*.sh text eol=lf` in `.gitattributes` would close it.

## A seven-level relative include leaves a consumer about thirty characters of path budget

`Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Activations/Elementwise/Kernels/ElementwiseActivation.cu:21` @ `2ff42c28`

The include is `"../../../../../../../Components/Activations/Activation/Kernels/ElementwiseActivation.h"`,
86 characters. MSVC opens the unresolved form rather than collapsing `../` first, so the limit
applies to the whole string: in the CPM gate's cache the path reached 264 against MAX_PATH's 260 and
threw C1083 on a header that exists at a 176-character resolved path. `LongPathsEnabled=1` does not
help, because the compiler does not opt in through its manifest.

The budget for a consumer's Mila source root is `260 - 86 - 76 = ~97` characters; a typical
`…\myapp\out\build\x64-debug\_deps\mila-src` is about 66, so it fits with roughly thirty to spare and
a deeper project path does not. Surfaced because `173b17c3` made the gate's CPM cache tag-keyed,
adding exactly 15 characters — correctly, to stop a warm cache validating the previous release — and
that tipped it over; at beta.2 the same path measured 249. Two other five-plus-level relative
includes exist in `Src`.

## Publishing the container images compiles the tree twice, about sixty-six minutes

`scripts/dockerhub/publish-image.sh` @ `2ff42c28`

`MILA_CLEAN_BUILD=1` is forced on every invocation to clear BuildKit's cache mounts, which
`--no-cache` alone leaves intact — the mechanism that stopped two wrong images shipping from another
tree's objects. RELEASING's flow is build-to-verify at step 9.3 then build-to-push at step 9.6, so
each release pays that compile twice: measured 33m17s and then 38m45s, of which only five or six
minutes was upload.

A push-only mode would halve it, at the cost of weakening the guarantee that what ships is what was
gated — which is the whole reason the forced rebuild exists, so this is a trade to think about
rather than an obvious fix.

## The packaging fixtures are an unread detector for source-tree pollution

`Mila/Tests/Packaging/{fetchcontent,cpm}_consumer/Mila/Adaptors/Inference/Server/mila.cp313-win_amd64.pyd`
@ `a87e8315`

Two 16 MB fossils dated 2026-07-21, left by the third binding destination that
`Mila/Bindings/CMakeLists.txt:112-118` records as removed. They landed there because the copy was
source-relative and a subproject build makes the consumer's root `CMAKE_SOURCE_DIR` — the same
defect class as the `tokenize` and wheel-VERSION items, both closed at `+40`. Nothing looks in
these directories, so the evidence sat for six weeks. Found widening the FetchContent gate. The
open question is whether the gate should assert the fixture directories are clean afterwards,
which is what would have caught this in a day rather than six weeks.

## `RelWithDebInfo` is the only build type that reports C4702, and nothing watches it

`Mila/Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/CudaTensorOps.Transfer.ixx:296` @ `a5650805`

Eleven `unreachable code` warnings appear under `RelWithDebInfo` and under no other build type:
Release and Debug both report zero on the identical tree, measured across five configurations.
They also need the extras — with `MILA_ENABLE_TESTING/SAMPLES/ADAPTORS/TOOLS/PROFILING` off the
library alone reports none, so the instantiations that trigger them come from outside `Mila/Src`.
Ten of the eleven are the `copyFromBlob` fall-through fixed at `+42`; the eleventh was
`GroupedQueryAttention::backward`, whose tail became unreachable once the operation it dispatched
into always threw, and which `+43` resolved by declaring the refusal at the component boundary
instead. **Unverified there: whether moving the throw up a level moves the warning up with it.**
MSVC deduced never-returns through the operation call, so it may deduce the same through
`attn_->backward` at `Llama.Block.ixx:373` and report the Llama backward chain as unreachable —
which would be true, and is the reason to look rather than to assume. No preset the project
watches is `RelWithDebInfo`, so nothing will surface it on its own.
Bisected to at least `+38` by building `git archive` exports at a short path
(`MAX_PATH` defeats a scratchpad build), so they are older than the day they were first noticed —
first noticed only because a clean full `x64-profile` build is rarer than an incremental one. The
open question is whether any preset the project actually watches should be `RelWithDebInfo`, since
`x64-validate` is the pre-commit gate and is Release, and therefore blind to this whole class.

## MIS reports every response as `finish_reason: "stop"`, including truncated ones

`Mila/Adaptors/Inference/Server/src/mila_llm_server/routes/completions.py:49` @ `9c431945`

Five sites hardcode it -- `chat.py:66`, `completions.py:49`, `factory.py:137`, `:155`, `:200` --
so an OpenAI or Anthropic client is told a reply ended naturally when it was cut off by
`max_tokens` or by context exhaustion. The live Anthropic path returns `stop_reason: "end_turn"`
on every response for the same reason. Until now this was not fixable: the binding discarded
`GenerateStatus`, so MIS had nothing truthful to report and a constant was the only option. The
binding's `generate` now returns the reason, and `ModelWorker.generate` /
`ModelWorker.generate_streaming` are the two places it would be threaded through -- neither
currently propagates it to the routes. Mapping is not one-to-one: OpenAI spells the cap `length`
and Anthropic spells it `max_tokens`, and neither protocol has a spelling for `context_limit`, so
the decision owed is what each protocol reports for a context overflow.

## A failed `--model` names a remedy that only exists inside the session it refused to open

`Mila/Adaptors/Chat/Src/Chat.ModelCatalog.ixx:485` @ `840568de`

`resolveStoredName`'s two refusals advise `/model install <name>` (:478) and `/model list --online`
(:486). Both are REPL commands, and both are correct on the path where the session opens anyway --
which `main.cpp:1057` and `Chat.Config.ixx:201` describe as the deliberate design. But the same
exception is thrown on the command-line path, where `main.cpp:823` deliberately exits instead:
"opening a session that ignored the one instruction it was given is worse than refusing". That exit
is right; the message travelling with it is not, because the user is back at their shell and cannot
type what it suggests. In the published container it is wrong twice over -- the reachable remedy
there is the image's own `install` verb, which is step 1 of the website's Evaluating band, and the
message never mentions it. Found by running the site's step 2 against a store where step 1 had not
run. The decision owed is whether the refusal text varies by path, or whether one wording can serve
both.

## A public component method takes a type the umbrella does not export

`Mila/Src/Dnn/Components/Transformers/Qwen/Qwen.DeltaNetBlock.ixx:363` @ `a395fe76`

`void setState( const GqaState& ) override` is public on a public component, but `Mila.ixx` never
exports `Compute.GqaState`, so a consumer with `import Mila;` cannot name the argument and cannot
call the method. Found because clang rejects what MSVC accepts: the name is reachable through the
component modules, and `Qwen.DeltaNetBlock.Cuda.cpp` compiled on MSVC while failing on clang with
`use of undeclared identifier 'GqaState'`. Worked around at `+42` with a direct
`import Compute.GqaState;` in the test, matching what `CudaGqaOp.Cuda.cpp` already does -- the
umbrella was left alone because widening it is a public-API decision. Same class as the notes
already in `Mila.ixx` for `Serialization.Tensor` and the weight-quantization policies: a type in a
public interface that the umbrella does not re-export, which fails asymmetrically and so goes
unnoticed. The decision owed is whether `GqaState` joins the export list, or `setState` stops being
part of the public component surface. Worth asking the same question of every other type named in a
public component signature, since nothing checks this.

## Nothing in the repository compiles the C++ quick start the website links to

`Mila/Samples/QuickStart/Cpp/main.cpp` @ `00978057`

It is a standalone FetchContent project, so the main tree never adds it -- `x64-validate`
does not build it and neither does CI. `packaging_fetchcontent_consumer` has its own
`main.cpp` rather than this one. The only build that touches it is
`Docker/Dockerfile.runtime:347`, which copies it into `/root/myapp` for the devel image, so a
break reaches a published surface and is caught by a container build or by a reader. Found
editing its not-installed message and having nowhere to compile-check the edit.
Configuring it standalone with `-DFETCHCONTENT_SOURCE_DIR_MILA=<tree>` works and needs no
network, which is what a gate would do; the file's own comment already names that override.
Same shape as the Doxygen entry in BACKLOG: a published artefact whose only checker is the
publish itself.

## A mid-download transport failure tells the reader nothing about what to do next

`Mila/Src/Distribution/ModelStore.ixx:457` @ `00978057`

Walking the website's Evaluating band, `install` died at 35% of a 2.86 GiB transfer with
`ModelStore: fetch of llama32_3b_instruct_fp4.safetensors (mila-llm/Llama-3.2-3B-Instruct-fp4)
failed: Transferred a partial file (TransportError)`. The design handles this well —
`ensureBlob` names the partial after the digest precisely so the next invocation resumes
(`:342`) — but the message says none of that, so a first-time evaluator on the highest-stakes
path reads a raw transport error as "this is broken" rather than "run it again, it continues
from 1 GiB". The remedy exists and the text does not name it. Whether resume actually engages
on the container's named-volume path is untested: `verify-image.sh` uses a throwaway volume by
design, so it always restarts from zero and cannot demonstrate it.

## An FP4 model on Turing has a fallback path that may be unreachable dead code

`Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Linear/CudaLinearOp.ixx:882` @ `d4c61b15`

When `use_wmma_fp4_gemm_` is false (SM < 8.0), the FP4 Linear dispatches to a non-WMMA
`cuda_fp4a16_gemm` rather than refusing -- so the Linear layer is written to serve Turing. But both
GQA flash prefill entry points throw outright on `sm_major < 8` (`Gqa.Flash.Fa2.cu:513`,
`Gqa.Flash.Wmma.cu:632`), and every bound model uses GQA, so no prefill can reach that GEMM on such
a card. Either the scalar path is dead code behind a refusal, or there is a non-flash attention
route that makes it live and nothing says which. Found deciding the published architecture lists,
which now start at 80 and so compile it for nobody. Worth resolving before someone maintains a
kernel that cannot execute.

## Published binaries may not carry the third-party notices they are required to

`NOTICE.md:39` @ `41c89f23`

NOTICE.md closed with "Mila is distributed as source... **A binary distribution that
linking them would need to carry their notices** -- that decision is open", pointing at a
BACKLOG section the hard prune has since deleted. The premise expired when the project began
publishing: the `mila-llm` wheels carry nlohmann/json, miniz, CUTLASS and pybind11, and the
container images additionally carry curl. The paragraph is rewritten and the table is now
mechanically gated, but **whether each published artifact actually ships the notices is
untested and unknown** -- that is the part this entry is for. Triage should decide whether the
wheel and image builds embed NOTICE.md, and whether anything already published needs a
follow-up. Licence texts to be read at source, not from this file.

## CUTLASS is five releases behind, and CI structurally cannot gate the bump

`CMakeLists.txt:320` @ `41c89f23`

The pin is v4.5.1; latest stable is v4.7.1. This is the one stale pin left after miniz and curl
moved, and it is deferred rather than forgotten: bumping CUTLASS recompiles every CUDA kernel,
and `build-pipeline.yml` says in its own header that GPU correctness tests are not run because
hosted runners have no GPU. So no amount of green CI would mean anything here -- it needs the
local suite on real hardware, which makes it a scheduled piece of work rather than a pin bump.
Worth doing sooner than that implies: PR #3082 (`is_family_of()` for the SM12x block-scaled arch
guard) sits somewhere in that range and bears directly on whether NVFP4 is reachable on the 5060
Ti. googletest v1.17.0 -> v1.18.0 is also outstanding and is trivial by comparison; it can ride
along or wait. The weekly `Dependency pins` workflow now reports both.

## Llama prefill has no flash path and is 3.8x slower than a larger Gemma

`Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Gqa/` @ `41c89f23`

Measured on the 5060 Ti, one build, both models FP4, 22496 tokens at 49152 context: Gemma 4
12B reaches ~1461 tok/s and dispatches `gqa_flash_prefill_mma_bf16_kernel`, while Llama 3.1 8B
reaches ~382 tok/s and falls to `Gqa::prefill_softmax_bf16_kernel`. A smaller model is 3.8x
slower because flash prefill is wired on Gemma's blocks and not Llama's. Per-kernel shares at
that length: Llama is 75.3% attention against Gemma's 59.5%. Found while sweeping GEMM share
across length and family, not while looking at Llama -- so nothing here says whether the
absence is a deliberate scoping decision or an oversight, which is what triage needs to settle.

## User-facing surfaces call Gemma 4 12B "the flagship", ranking a model on the user's behalf

`README.md:107` @ `5063ee29`

Four end-user sites carry it: the `README.md:107` heading "Gemma 4 12B — the flagship",
`Mila/Samples/QuickStart/Python/quickstart.py:20` and `Mila/Samples/QuickStart/Cpp/main.cpp:40`
("The published flagship"), and the `Mila/Bindings/Mila_py.cpp:951` docstring "(the flagship)",
which prints in Python's `help()`. Which model suits a reader is theirs to decide against their card;
the label ranks one for them, and it dates from when Gemma was Chat's compiled-in default, which no
longer exists. The samples name Gemma because an example needs a model, which is the fact to state.
`main.cpp:40` also still gives `/install`, a verb that no longer exists. The website's copy of the
same label is widened into `Web/Issues/Backlog.md`'s chat-default entry; the developer-doc uses were
fixed in the same change that captured this.

## CUDA RMSNorm backward ignores the unit offset its forward applies

`Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Normalizations/RmsNorm/RmsNormOp.ixx:334` @ `5063ee29`

`forward` passes `config_.getUnitOffset()` to the kernel (`RmsNormOp.Dispatch.ixx:40` declares
`weight_offset`), so the forward computes `x * rstd * (weight + offset)`. `backward` passes weight,
rstd and geometry but no offset, so any norm configured with `withUnitOffset( 1.0f )` — every Qwen
norm (`Qwen.ixx:807`, `Qwen.DeltaNetBlock.ixx:855`, `Qwen.AttentionBlock.ixx:889`) — differentiates
as if the offset were zero. Nothing ships wrong today because Qwen is inference-only and no test
trains through an offset norm; it is latent until one does. Found writing `CpuRmsNormOp`, whose
backward applies the offset, so a CPU/CUDA gradient comparison on an offset norm would disagree.

## Any MSVC translation unit that loads a model is near the object section limit

`Mila/Profiling/ProfileModel/CMakeLists.txt:32` @ `36d87f7a`

`x64-profile` failed with C1128 on `ProfileModel.ixx` once the Gemma 4 MoE work gave `GemmaModel` a
dense and a routed dispatch for every quantization. It now carries `/bigobj`, as `mila-chat` already
did (`Mila/Adaptors/Chat/CMakeLists.txt:41`). Section counts, Release `x64-claude-verify` against
RelWithDebInfo `x64-profile`, which runs about 1.28x higher: `ProfileModel.ixx` 51609 / ~66000,
`ExportArtifact.ixx` 46926 / 59830, `Gemma.MixtureOfExperts.Cuda.cpp` 44694 / 56375,
`Chat.ModelCatalog.ixx` 41647 / 52666 -- the limit is 65535 and Debug was not measured. So the
next family or quantization added breaks targets one at a time, and a consumer's own app calling
`fromPretrained` -- the C++ quick start included -- inherits the same exposure with no flag. The
decision owed is whether `Mila` exports `/bigobj` as a PUBLIC MSVC compile option, which changes
the flags every consumer compiles with, or targets keep adding it as they cross.

## A composite cannot keep a derived child out of its flat save without re-implementing the walk

`Mila/Src/Dnn/Components/MixtureOfExperts/Router.ixx:185` @ `66d831b6`

`Router` holds an `RmsNorm` child whose weight is derived from `scale` and must never be written, so
it overrides `saveFlatTensors` and re-spells its children by hand (`owned_prefix + "proj"`) rather
than calling the base walk. It has to, because `CompositeComponent::childFlatPrefix`
(`CompositeComponent.ixx:1070`) is private and the base recursion (`:749`) has no way to exclude a
child. The consequence is a second copy of the flat-naming rule that will not follow a change to the
first. Applied silently during the router work and only now flagged. Candidate changes: a
per-child "derived, not serialized" marker the base walk honours, or `childFlatPrefix` made
protected so an override at least shares the naming rule.

## The MoE expert bank reaches into Linear's kernel header to quantize

`Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Moe/CudaMoeOp.ixx:169` @ `66d831b6`

`CudaMoeOp::quantize` includes `../Linear/Kernels/Quantization/CudaFp4WeightQuantization.cuh` by
relative path and calls `Linear::cuda_quantize_fp4_per_group` directly, because the shared
per-group FP4 quantizer only exists as a private detail of `CudaLinearOp:Quantize`, which is not
exported. It works -- the quantizer is row-generic -- but one operation now depends on another
operation's kernel namespace, and a change made for Linear lands silently in the expert bank. The
E2M1 *decode* was lifted to `Helpers/Fp4E2M1.h` in the same work; the *encode* was not.
`CudaTokenEmbeddingOp:Quantize` has its own copy of the same boundary pattern. Candidate change: move
the per-group FP4 quantizer into a shared weight-quantization kernel location both operations import.

## Llama still holds the GQA transient as seven loose members

`Mila/Src/Dnn/Components/Transformers/LlaMa/Llama.ixx:710` @ `66d831b6`

`GqaWorkspace` (`Compute.GqaWorkspace`) now owns the seven GQA scratch tensors as one unit for Qwen and
Gemma, with `state()` and `deviceStorageBytes()`. Llama still declares them as separate `unique_ptr`
members, builds the `GqaState` by hand (`:633-639`) and sums them in a hand-written list in
`getMemoryStats` (`:361-363`) -- the list that under-counts silently when a tensor is added. Left alone
when the workspace moved because the agreed scope was Qwen and Gemma, and Llama has no memory-footprint
gates to catch a mistake. The change is mechanical: one member, one factory call, two accounting lines.

## Every family chooses its prefill chunk against a fixed activation budget blind to weights and device

`Mila/Src/Dnn/Components/Transformers/Gemma/Gemma.ixx:119` @ `bbdce9b6`

Gemma and Llama cap chunk-scaled activations at 1536 MiB (`Gemma.ixx:119`, `Llama.ixx:67`), Qwen at a
measured 512 MiB (`Qwen.ixx:134`). None knows how much of the device the weights already hold, so a
constant that fits one model on one card is wrong for the next. On the RTX 5060 Ti the 26B-A4B FP4
weights take 13.54 of 14.80 GiB free, the budget still admits 1,376 MiB of activations at context 8192,
and the load predicts 83.8 MiB over; Qwen needed its own smaller constant for the same reason on the
12 GB card. `Gemma.ixx:116` and `Gemma4InferenceReview.md:531` both call the live-memory version a
BACKLOG follow-up, and `BACKLOG.md` has no such item. Direction agreed 2026-09-13, taken up in its own
session: the largest rung whose whole predicted footprint fits the available memory. Specified as a draft in
`Mila/Specifications/MemoryFootprint.md` section 11; driver allocation rounding (11.8) is still open.

Its visible symptom, `GemmaModel.MixtureOfExperts.Fp4.Cuda.cpp`, is now **disabled**
(`DISABLED_Fp4Load_FitsSection8AndMatchesHuggingFaceGreedy`, 2026-09-14, Todd: fix during rc.1). On the RTX 5060 Ti
alone it fails only its fit: 16,009,596,928 bytes predicted against 15,894,315,008 free after Phase 6 steps 1 and
2. In a normal run both GPUs are visible and CUDA's device 0 is the 12 GiB RTX 4070, where the load cannot be
placed at all. Until 2026-09-15 that aborted the whole test run; it now fails the test with the allocation error,
and it fails the same way with the RTX 5060 Ti as device 0, where it does not fit without spilling either.
Its fit assertions also skip rather than fail since the same day. Both must be undone when the rule lands.

## The scratch reservation test's growth check fails on a display card and cannot see a saturated one

`Mila/Tests/Dnn/Models/ScratchReservation.Cuda.cpp` @ `c1e439e2`

The check that device memory does not grow during generation -- the only permanent detector of a network that
fails to reserve its scratch -- read 29.1 MiB for Qwen 3.8 27B cb2-3 on the RTX 4070 at the end of a full
suite run, and 0.0 MiB in three runs of that test alone and in one run straight after Qwen FP4. The 4070 drives
the display, and MemoryFootprint.md 11.5 records Windows lowering that process's video memory budget once it
has written about 8 GiB, which moves CUDA's free-memory reading by the same amount; the suite writes far more
than that first. The opposite blind spot is a card the load saturates: Qwen FP4 on the 4070 read 0.0 MiB
because a full card reports no change. The check **skips instead of failing** (2026-09-14, Todd: fix during
rc.1), which also lets a genuinely missing reservation read as skipped. The proposed guard is to measure only on
a card that is neither saturated nor driving a display, the second read from NVML's display mode, and to restore
the failure.

Two full runs of `x64-profile` with both GPUs visible (device 0 the RTX 4070) show which test it lands on is a
matter of timing: the first skipped the Llama 3.1 8B case (47.6 MiB), the second skipped the Gemma 4 12B context
8192 case and the cb2-3 case instead, while the Llama case passed. No test fails on it, but on that card it
cannot catch a missing reservation either.

## No permanent test catches a scratch reservation larger than any request

`Mila/Tests/Dnn/Models/ScratchReservation.Cuda.cpp` @ `c1e439e2`

Phase 6 step 2's third negative -- scratch summed across the tree instead of taking the maximum -- reserved
12,386,304,000 bytes for Gemma 4 12B against a largest request of 121,901,056, and every permanent test passed:
predicted still equals reported, nothing throws, memory does not grow. Only the temporary counters used during
that run saw it. A per-model literal of the reserved bytes in the reservation test (121,901,056 for Gemma 4 12B
FP4 at context 8192) would fail on it, as the other footprint literals do.

## The quantize-on-load footprint test can fail on a card that drives a display

`Mila/Tests/Dnn/Models/QuantizeOnLoad.Footprint.Cuda.cpp:132` @ `c1e439e2`

`Gemma4_12B_FittedLoadSettlesAtExport` compares device memory consumed by a BF16 load against its exported form
within 128 MiB. In a full `x64-profile` run with both GPUs visible, on the RTX 4070, it failed at 323 MiB after the
load and 295 MiB after generation; in the next identical run it passed. 323 MiB is the Windows video memory budget
cut MemoryFootprint.md 11.5 measured on that card (313-326 MiB), which lands wherever the process has written about
8 GiB, so whichever comparison straddles it fails. On the headless RTX 5060 Ti the same test reads 0 MiB. Unlike
the growth check above, this one still fails rather than skips, so it can turn a suite red by timing alone.

## The Qwen 3.8 27B FP4 scratch reservation case is disabled because the model does not fit a 12 GiB card

`Mila/Tests/Dnn/Models/ScratchReservation.Cuda.cpp:155` @ `c1e439e2`

`DISABLED_Qwen38_27B_Fp4_Context8192` (2026-09-14) loads about 15 GiB. With both GPUs visible it runs on the RTX
4070 and the load's allocation fails -- until 2026-09-15 by aborting the whole run, now as a test failure; Todd's run hit
this with the same two exceptions. It passed on the RTX 5060 Ti alone, and on the RTX 4070 alone only because a
process that sees one GPU spills the overflow to host memory. The case needs a way to run where the model fits,
or a skip when the device is too small, before it is re-enabled.

## CudaExecutionContext allocates its buffers without selecting its device

`Mila/Src/Dnn/Compute/Devices/Cuda/CudaExecutionContext.ixx:274` @ `c1e439e2`

The forward scratch, load staging buffer and scratch reservation call `cudaMalloc` without selecting the
context's device, where `CudaDeviceMemoryResource::do_allocate` calls `Cuda::setCurrentDevice` first. The
constructor selects the device once (`initializeResources`), so in a process that sees two GPUs these buffers
land on the wrong device only if something else changes the current device between construction and the
allocation. Found 2026-09-14 chasing the allocation-failure abort, which it played no part in.

## A test comment still says Qwen 3.8 27B cb2-3 cannot load at context 2048 on the RTX 4070

`Mila/Tests/Dnn/Models/Qwen/QwenModel.Load.Cuda.cpp:1468` @ `c1e439e2`

The comment above `Generation_RunsAndStaysInsideTheVocabulary` says the load "dies" at context 2048, which is why
that test uses 512. On 2026-09-14 `ScratchReservationCudaTests.Qwen38_27B_Codebook_Context4096` loaded and generated
at context 4096 on the RTX 4070 with both GPUs visible. The comment predates the footprint reductions since
(RoPE tables sized to the context among them) and misled a prediction about which tests would fail.

## The FP8 per-tensor weight quantizer has no caller and still stages the whole tensor

`Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Linear/CudaLinearOp.Quantize.ixx:106` @ `c1e439e2`

`Detail::quantize_fp8_per_tensor` and `cuda_quantize_fp8_per_tensor` (`CudaFp8WeightQuantization.cu:251`)
are exported, documented as the Ada cuBLASLt path, and called from nowhere: `CudaLinearOp::quantize` reaches
only the per-channel bridge. The per-channel call site allocated four extra staging bytes "so the same
scratch buffer covers both variants", which is how the dead path kept a live cost. Left untouched when the
per-channel path moved to row blocks through the load-owned staging buffer, so it is now the one quantizer
that still requires the whole BF16 tensor on the device. Found implementing Phase 6 step 1. The decision
owed is deletion, or row blocks with a running maximum if a per-tensor path is wanted.

## Two exception classes report the same CUDA runtime error

`Mila/Src/Dnn/Compute/Devices/Cuda/Helpers/CudaUtils.h:15` @ `d531a026`

`cudaCheck` (`CudaUtils.h:31`) throws `CudaException`, and it is what the 157 kernel launch sites call as
`cudaCheck( cudaGetLastError() )`. `cudaCheckStatus` and `cudaCheckLastError` (`CudaError.ixx:133`, `:145`)
throw `CudaError`, used from the module code. Both derive from `std::runtime_error`, both carry the error code
and source location, and they format different messages ("[CUDA ERROR] at file" against "CUDA runtime API
error"). `CudaError::getError()` is commented out, so only `CudaException` exposes the code. A caller that
wants to catch a CUDA failure has to name both or fall back to `std::runtime_error`. Found 2026-09-15 surveying
CUDA error handling after the allocation-failure abort.

## Running out of CUDA memory arrives as four different exception types

`Mila/Src/Dnn/Compute/Devices/Cuda/CudaPinnedMemoryResource.ixx:101` @ `d531a026`

`CudaDeviceMemoryResource` throws `CudaBadAlloc`, a `std::bad_alloc` carrying the size and device. The pinned
(`:101`) and managed (`CudaManagedMemoryResource.ixx:89`) resources throw a bare `std::bad_alloc`; the managed
one builds the same message and discards it. `CudaExecutionContext`'s scratch, load staging, pinned staging and
cuBLASLt workspace throw `std::runtime_error`, as do the scratch buffers in `CudaTensorOps.Random.ixx`, and
`RopeCacheRegistry::acquire` throws `CudaError`. So a caller cannot catch "the model does not fit" as one type,
and `import Mila;` does not export `CudaBadAlloc`, so a consumer cannot name even the device one. Found
2026-09-15, same survey.

## Five device allocations ignore cudaMalloc's result

`Mila/Src/Dnn/Compute/Devices/Cuda/Operations/Linear/CudaLinearOp.ixx:1509` @ `d531a026`

`CudaLinearOp.ixx:1509` and `:1510` (the FP8 scale scalars), `TensorOps.Fill.cu:112` and `:133` (fill staging)
and `CudaLinearGelu.cu:62` (a cuBLASLt workspace) call `cudaMalloc` as a statement. A failure goes unnoticed,
the pointer stays null and the next kernel or copy uses it. Found 2026-09-15 enumerating allocation sites for
the allocation-failure fix, which clears the error only where a failure is detected.

## A 4 GB single-tensor limit marked DEBUG ships in TensorBuffer

`Mila/Src/Dnn/Tensors/TensorBuffer.ixx:224` @ `d531a026`

Any tensor whose storage is 4 GiB or more throws `std::length_error` from the constructor. The check sits
between `// DEBUG:` and `// END DEBUG:` comments with no condition around it, so every build carries it.
Found 2026-09-15 writing the allocation-failure test, which had to call the memory resource directly because
a tensor large enough to fail on the device never reaches `cudaMalloc`.
