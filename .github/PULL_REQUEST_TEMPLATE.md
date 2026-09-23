<!-- Thank you for contributing to Mila. Please fill out the sections below. -->

## Summary

<!-- What does this PR change, and why? One or two sentences. -->

## Related issues

<!-- e.g. Closes #123. Remove if none. -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Refactor / internal cleanup
- [ ] Documentation
- [ ] Build / CI / tooling

## How you built and tested it

- Compiler and preset:
- Tests run, and the result:

<!--
If your change touches CUDA, fill this in too. CI compiles the CUDA tree but executes
nothing on a GPU, so what you write here is the only record of what ran on one.
-->

- GPU and compute capability: <!-- e.g. RTX 4090, sm_89 -->
- Driver and CUDA Toolkit version:
- Architectures the build targeted: <!-- CMAKE_CUDA_ARCHITECTURES / MILA_LIBRARY_CUDA_ARCHITECTURES -->

A build from the supported architecture list still runs on a newer card, by JIT from older
PTX, and at the prompt that is indistinguishable from a native build. So "it ran on my 5090"
and "it was compiled for sm_120" are separate claims — pin the architecture if you mean the
second.

## If you ticked "Performance improvement"

Three things make a number checkable by someone who is not you:

- [ ] **Baseline and change built from the same tree**, one after the other.
- [ ] **Where the weights were** — resident in GPU memory, or streaming from disk. The ranking
      between candidates changes between those two.
- [ ] **More than one run.** A single measurement is a sample, not a property.

<!-- Numbers, and the card they came from: -->

## Checklist

- [ ] This PR targets the `dev` branch
- [ ] Code follows the standards in [CONTRIBUTING.md](CONTRIBUTING.md) / [CLAUDE.md](CLAUDE.md)
- [ ] Builds clean (note your compiler: MSVC 2026 / Clang 21)
- [ ] Tests added or updated for the change, and `ctest` passes — on `x64-validate`, or on
      `x64-debug-cpu-only` / `linux-clang-cpu-release` if you have no NVIDIA GPU
- [ ] A new kernel or operation has a `*_MatchesReference` test against the CPU path
- [ ] Documentation / Doxygen updated where the public API changed
- [ ] Relevant design doc under `Mila/Specifications/` updated for significant changes

## Notes for reviewers

<!-- Anything reviewers should focus on, known limitations, or follow-up work. -->

---

**What CI can and cannot tell you.** It compiles the whole tree under clang on Linux with CUDA
enabled, and runs the CPU test suite. You need neither a GPU nor a CUDA Toolkit for that to
pass — it is there to check what your machine may not be able to build. It runs no kernel and
proves nothing about numerical correctness on a device, so a green tick is not a validated GPU
change; a maintainer re-runs those on real cards, which is the slow part of review.

If CI goes red for something that looks like Mila rather than your change, say so in a comment.
That is a bug on our side.
