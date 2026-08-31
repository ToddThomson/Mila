# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined at the release tag**, and anything user-reported is a pointer to
its GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

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
