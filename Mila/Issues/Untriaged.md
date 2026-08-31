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
Ten of the eleven are the `copyFromBlob` fall-through fixed at `+42`; the eleventh is the GQA
backward entry below. Bisected to at least `+38` by building `git archive` exports at a short path
(`MAX_PATH` defeats a scratchpad build), so they are older than the day they were first noticed —
first noticed only because a clean full `x64-profile` build is rarer than an incremental one. The
open question is whether any preset the project actually watches should be `RelWithDebInfo`, since
`x64-validate` is the pre-commit gate and is Release, and therefore blind to this whole class.

## CUDA `GroupedQueryAttention::backward` is dead code and the compiler says so

`Mila/Src/Dnn/Components/Attention/GQA/GroupedQueryAttention.ixx:219` @ `a5650805`

`return *input_grad_;` draws C4702 because line 217's `operation_->backward(...)` provably never
returns — the CUDA GQA backward throws unconditionally, so the whole tail is unreachable. This is
not a code-shape problem and was deliberately left reporting the truth when the ten sibling
warnings were fixed at `+42`: silencing it would hide an unimplemented path. Matches the standing
note that GQA backward has never been validated. The decision owed is whether backward is
implemented, or declared unsupported at the component's own boundary so the throw is the
documented contract rather than an accident of the operation layer.
