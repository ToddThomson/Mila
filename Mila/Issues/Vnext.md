# Vnext

**The seed corpus for the next release's backlog.** When the release in flight goes to production,
this is what `BACKLOG.md` is rewritten from, so an item here carries a real intention to do it in
that cycle. That is the difference from [`Future.md`](Future.md), which carries zero commitment:
"we mean to do this next" belongs here, "someday, if the hardware or the reason arrives" belongs
there.

A shortlist, not a plan — tasking happens on promotion, when the next release has a ROADMAP section
and its items can face the real admission test. Triage flow and categories are in
[README.md](README.md); the tag set is [Tags.md](Tags.md).

---

## One typed model handle + factory

`architecture` · `mila-src` · `gate`

The architecture-to-concrete erasure exists three times in two languages — Chat's `ModelVariant`,
the binding's `*Session` classes, MIS's `ModelFamily` — which is why GPT-2 is missing from MIS.

Lands in the runtime-adjacent native agent core; sequencing in `MilaProductFamily.md` Open
Decision 2. ROADMAP already calls it the first work after the v0.20 tag and a precondition for
every model entry below it.

## Warnings-as-errors ratchet

`build` · `ci` · `blocked`

Enforce in **CI only**, never locally; ratchet on the count *not increasing* before demanding zero;
**MSVC first**, since `/WX` across three compilers means the union of three opinions must be zero.
Dormant-but-retained code warns by nature — suppress per-file in CMake pointing at the owning task,
never with `#pragma warning` in module code.

Blocked on isolating third-party warnings first: `/external:I` + `/external:W0` (`-isystem` for
Clang/GCC), targeting third-party header text pulled into Mila's own TUs rather than their sources
— `/W4` at `Mila/CMakeLists.txt:87` is `PRIVATE` and never reached them. Two frictions: those
headers enter through module global module fragments, and `/external:` does nothing for nvcc
diagnostics.

`GroupedQueryAttention.ixx:216`'s C4702 is the case that decides the shape. It is left deliberately —
it self-clears when the GQA training path is built, where a suppression would have to be remembered.
A blanket `/WX` forces it silent; escalating only the defect-class codes leaves it visible.

## v0.20 library-frozen tails

`api` · `mila-src`

The Generation API surface tail (`SamplerConfig` rename, Llama/Gpt seedable sampling, eager
sampler, config-accessor propagation, `contextLength()` hoist), the Sample-API device-sampler
migration for Llama/Gpt, and the Optimizer-dispatch migration onto `OperationTraits`.

All `Mila/Src` capability deferred out of v0.20 by the freeze rather than declined, so the release
that lifts the freeze is where they land.
