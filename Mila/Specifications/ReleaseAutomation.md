# Release Automation

Specification for where each of Mila's checks runs, and why. The change it proposes is to move
continuous integration off GitHub-hosted runners and onto the maintainer's own machines, keeping on
GitHub only the three things that genuinely need to be there: the release build's public evidence,
the Pages deploy, and a weekly pin report.

Draft, 2026-09-19. Post-v0.20; nothing here is committed to a release, and v0.20.0 ships on the
procedure in [RELEASING.md](../../RELEASING.md) unchanged.

---

## 1. Problem Statement

Mila has one maintainer, and release management is the part of the work he least wants to spend time
on. A tagged release costs a day or more. The largest single line in that day is waiting: the
`dev -> master` pull request at `RELEASING.md` step 3 runs the full pipeline and takes about seventy
minutes, during which nothing else can proceed.

The five workflows are not equally responsible:

| Workflow | Trigger | Blocks the maintainer |
|---|---|---|
| `build-pipeline.yml` | `push: master`, `pull_request: dev` | No — the release merge no longer waits on it |
| `gates.yml` | `push`, `pull_request` | No — three offline jobs, about a minute |
| `wheel-cleanroom.yml` | `workflow_dispatch` | Yes, at release step 1 |
| `web.yml` | dispatch | Build-without-deploy check |
| `dependency-pins.yml` | `cron: 0 7 * * 1` | No — reports weekly, while he sleeps |
| `publish-site.yml` | `workflow_dispatch` | No — manual, once per release |

So "eliminate CI" is too broad a statement of the goal. Two of the five never cost him anything, and
one of those is load-bearing for a reason recorded below. The target is `build-pipeline.yml` and
`wheel-cleanroom.yml`, which together are effectively all of the waiting.

**Acted on, 2026-09-20, and the conclusion was stronger than this section's.** `build-pipeline.yml`
now triggers only on `push: master` and `pull_request: dev`; the three offline jobs moved to their own
`gates.yml`.

The release PR was removed from its triggers because `RELEASING.md`'s cross-platform policy *already*
requires a passing WSL build before `dev -> master`, and by its own table that build does strictly
more — it compiles under clang **and** runs the suite on real hardware, where a hosted runner executes
nothing. So the release never depended on this workflow for an answer, only for a badge. §2's reasoning
reached that conclusion without anyone drawing it.

**What survives is not about releases at all.** `pull_request: dev` covers the one case no maintainer
machine can: a contributor's fork. CONTRIBUTING.md directs PRs at `dev`, and the work the project
actually offers is CPU-operation work — issue #22 is the standing example — so the likely author has
no CUDA Toolkit and cannot discover that they broke the CUDA tree. This run is the only thing that
can tell them. That reframes the job's reliability from a
convenience into a first impression, which is why the NVML load-time dependency and the missing
`timeout-minutes` were fixed in the same change.

## 2. What decides where a check runs

**A check belongs on the machine that can answer its question, and nowhere further away.** Applied
honestly this moves almost everything local, because of a fact that is easy to miss: for Mila's
purposes the maintainer's machines are *better* instruments than GitHub's runners, not worse.

- A hosted runner has **no GPU and no model weights** (`RELEASING.md:462`, `:518`). Every test that
  touches a device or a checkpoint is skipped there.
- The WSL2 distribution and the dev container both see both cards, and the container reaches the
  weights through a bind mount.

So the Linux/clang portability build — the one thing CI is genuinely for, since MSVC resolves
includes transitively that clang does not — runs *more* thoroughly on the maintainer's machine than
on the runner it currently runs on. The procedure already concedes this: `RELEASING.md:160` says
that when dev CI goes red on a compile error, the way to reach green is the local WSL loop.

There is exactly one question a local machine cannot answer, in §4.

## 3. The design — three rings and three residents

### Ring 1 — every commit, seconds, offline

`notice-gate`, `version-sites-gate`, `doxygen-gate`. No toolchain, no network, no GPU. These are the
checks whose current placement is hardest to defend: a stale version string is reported today after a
ten-minute queue, and it is a sub-second check.

They are now `gates.yml`, separated from the pipeline on 2026-09-20 so the pipeline's triggers could
narrow without taking them down. That is a staging post, not the ring: the ring is the pre-commit hook,
which reports before the commit rather than after the push.

Delivered as a git `pre-commit` hook, which Visual Studio's commit button honours, plus a named entry
point for running them by hand.

### Ring 2 — before pushing, or on demand, ~20 minutes

The Linux/clang build and the full suite, plus the CPU-only configuration. This is `compile-and-gate`
and `cpu-only-tests` moved to WSL2 or the dev container.

**Built from a `git archive` export, not from the working tree.** This is what buys back the one
property local CI otherwise loses — that the build is of a clean checkout rather than of whatever is
on disk. The idiom is already established for baseline builds: export, never stash.

### Ring 3 — release only

Wheel validation (§4), both container images, `verify-image.sh`, and the Hugo build that `web.yml`
does today.

### Resident on GitHub, deliberately

- **The release build and its badge.** `build-pipeline.yml` retriggered on `push: branches: [master]`.
  `master` moves exactly once per release, at the step 4 merge, so this fires once per release and
  the badge URL's `?branch=master` filter resolves correctly. It is public evidence that the last
  tagged release builds clean — a more useful claim than a badge describing the last `dev` commit,
  which is a tree nobody consumes.

  **Two constraints, or it silently becomes a blocker again.** It must never be a required status
  check, because then GitHub can refuse the merge; and the maintainer must not watch it before
  tagging, because then it is a gate with extra steps. Its value is that it arrives *while* he tags,
  in the gap between the step 4 merge and the step 6 tag — so a red result still precedes the
  immutable artifact.

- **The contributor check.** `build-pipeline.yml` on `pull_request: dev`. Added to this list on
  2026-09-20, and the only resident that is not about releases. A fork's branch has never been built
  on a machine the maintainer controls, so the "local instruments are better" argument in §2 simply
  does not reach it — "local" means *his* machine. It costs nothing while there are no contributors,
  because with no PRs it never fires.

  **It answers a compile question and no correctness question**, which matters most for the
  contributions Mila most wants. A kernel, a quantization format, an attention variant — none of it
  can be validated by a runner with no GPU, so every such PR ends on the maintainer's cards. The
  lever on that cost is not CI: it is the `*_MatchesReference` equivalence test the PR template asks
  for. With one, validation is a build and a `ctest`; without one, it is reading a kernel and
  inventing the test yourself. That is the difference between a contribution that helps and one that
  is a second job.

- **`publish-site.yml`.** A deploy, not a check. Manual, once per release, off the critical path.
- **`dependency-pins.yml`.** A weekly notification. A local machine cannot reliably do "weekly while
  nobody is looking", and this blocks nothing.

## 4. The wheel clean room, and the one question a local machine cannot answer

`wheel-cleanroom.yml` runs a four-leg matrix — `windows-latest` and `ubuntu-latest` against Python
3.12 and 3.13 — and `verify_wheel_cleanroom.py` asserts the **absence of a CUDA Toolkit** before it
asserts anything else. That absence is the whole point: a wheel that quietly leans on a host Toolkit
passes on a developer machine exactly the way a correct one does.

An absence cannot be proven on a machine that has one, and `mila/__init__.py` makes this worse on
purpose — `_register_cuda_libraries` keeps an installed Toolkit as a Windows backstop, so a wheel
with a missing dependency degrades into working rather than failing.

**The failure mode is narrower than the check, and the narrower thing is locally decidable.** What
the clean room actually guards against is a wheel binding to host libraries instead of its own. That
is answerable by enumerating the process's loaded modules after `import mila` and asserting every
CUDA library resolved from `site-packages` — which works *with* a Toolkit installed, because it
inspects provenance rather than availability.

The technique was demonstrated on 2026-09-19 while verifying the `>=13.0` dependency floor: after
`import mila`, `psapi.EnumProcessModules` reported `cublas64_13.dll`, `cublasLt64_13.dll`,
`curand64_10.dll` and both nvrtc libraries all resolving from the venv, with CUDA v13.4 registered
as a fallback directory and unused. (Set `argtypes` on the psapi calls — without them ctypes
truncates the 64-bit module handles and the enumeration silently returns nothing.)

Proposed replacement:

| Leg | Local substitute | Strength |
|---|---|---|
| Linux 3.12 / 3.13 | `python:3.13-slim` container — genuinely carries no Toolkit | **Equal** to today |
| Windows 3.12 / 3.13 | Module-provenance assertion on the maintainer's machine | **Weaker**: proves the wheel does not *use* the host Toolkit, not that it survives without one |

A truly clean Windows leg would need a virtual machine. Windows Sandbox and Hyper-V are unavailable
on Windows 11 Home, so this would mean a third-party hypervisor maintained for one check per
release. Not judged worth it; the gap is recorded here instead.

## 5. What is lost

Stated plainly, so the trade is made once rather than rediscovered:

- **The Windows absence proof**, downgraded to a provenance proof (§4).
- **Continuous evidence for a reader.** Between releases the badge describes the last release rather
  than the current `dev` head. This is accurate, not misleading, but it is less than a project of
  Mila's positioning might be expected to show.

  Worth less than it sounds, on the evidence. Over the five days to 2026-09-20 the `dev` badge was red
  continuously and **not once for a portability regression** — first exit 137 on a runner shutdown, then
  a test binary that could not start without a driver. Thirty runs held five successes. A signal that is
  red regardless of the code is not evidence, and the badge was showing it on the README.
- **The property that CI is indifferent to the developer's machine.** Ring 2's `git archive` export
  recovers the clean-checkout half of this. It does not recover the clean-*machine* half: a
  dependency installed only on the maintainer's box would go unnoticed. The FetchContent and CPM
  consumer gates are the existing defence and stay.

## 6. Rejected alternatives

- **Publishing the site from a `gh-pages` branch push.** `publish-site.yml:26-31` records that
  first-party Pages actions were chosen over exactly this, and that the branch-push approach is what
  froze the published docs. Re-creating a diagnosed failure to remove a workflow that costs nothing
  is a bad trade.
- **Making the master-push run a required status check.** Restores the block this specification
  exists to remove.
- **Triggering the badge run on the tag rather than on the master push.** Gets the run but not a
  clean badge — tag-triggered runs do not filter into the `?branch=` badge URL — and it lands after
  the immutable artifact rather than before it.
- **Deleting `dependency-pins.yml` for symmetry.** It blocks nothing and does something a local
  machine does badly.

- **A self-hosted runner on the maintainer's cards, to give contributor PRs real GPU coverage.**
  This is the first idea anyone has after reading §3's note that CI cannot validate a kernel, and it
  is the one thing that must not be built. A `pull_request` from a fork carries arbitrary code; a
  self-hosted runner on a public repository executes it on the maintainer's desktop, which holds the
  signing path for every wheel and image Mila publishes. GitHub's own guidance says not to. GPU
  validation stays a human step, and the equivalence test the PR asks for is what keeps it cheap.

## 7. Phasing

1. **Ring 1.** Additive and safe at any time, including before a release.
2. **Ring 2.** Wrap what is already run by hand; add the `git archive` export.
3. ~~**Retrigger `build-pipeline.yml` on `push: branches: [master]`**, and remove the `push`/
   `pull_request` triggers.~~ **Done, 2026-09-20**, and it was not the point of no return this step
   assumed. That framing took the release PR's run for a gate; `RELEASING.md` had already made the WSL
   build the gate, so nothing was being relied upon. `pull_request: dev` stays and is not release
   machinery — it is the contributor path (§1).
4. **Ring 3**, including the clean-room substitute.
5. **Fold `web.yml`** into ring 3 and retire it.

## 8. Open decisions

- Whether `RELEASING.md` steps 3 and 4 collapse. The `dev -> master` pull request exists only to run
  CI; with CI local, a merge performed in Visual Studio would do the same work. The PR also produces
  a reviewable diff of a release, which has some value even with no second reviewer.
- Whether ring 2 runs in WSL2 or the dev container by default. The container is the more faithful
  environment and reaches the weights; WSL2 is faster to start.
- Whether the badge should claim anything about the suite, or only about the build.
