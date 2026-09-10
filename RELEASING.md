# Releasing

How Mila is versioned, branched, validated, and tagged into a consumable release. Planning and
progress live in [ROADMAP.md](ROADMAP.md) / [BACKLOG.md](BACKLOG.md) / [CHANGELOG.md](CHANGELOG.md);
this document is only the release mechanics.

One thing to internalize up front: the version scheme carries a **stage** (the codebase's maturity,
not a task or phase label) and a ticking **build** counter held in semver build metadata, detailed in
the next section.

---

## Versioning

Mila uses a repeating **release-cycle** model: `MAJOR.MINOR.PATCH-stage.X+build` (e.g.
`0.20.0-alpha.6+56`).

- **minor** — feature-set era.
- **patch** — part of the target release (usually `.0`).
- **stage** — the codebase's maturity: `alpha.X -> beta.X -> rc.X ->` unsuffixed stable. `X` is the
  stage **checkpoint ordinal**. It is a pure release-provenance count: it does not name, count, or
  correspond to any unit of planned work.
- **build** — a per-commit counter carried as **semver build metadata** (after `+`). It counts the
  `dev` commits accumulated toward the checkpoint named in the stage field, and **resets to `+1` when
  that checkpoint is tagged**. It is **ignored for version precedence** by the spec.

**The whole string points forward.** `Version.txt` names *what is being built*, never what was last
built — the git tag is the record of what shipped. So on `dev`, `0.20.0-beta.2+7` reads "the 0.20.0
release, seven commits into the work toward the beta.2 checkpoint". The stage ordinal is bumped **at
the moment a checkpoint is tagged**, not before the next one is cut (see step 12 of *Cutting a
release*), so the working tree never reports a version that has already shipped.

The next checkpoint's name is a **placeholder, not a commitment** — a tree that says `beta.2` may
well be tagged `rc.1` instead. Nothing downstream reads the interim value; if the call changes, edit
`Version.txt` in the same commit that prepares the tag.

Each feature set opens a new minor and runs its own ladder; features never land inside a hardening
ladder — a stabilizing release takes only patch-level fixes. Mila is pre-1.0, so any release may
carry breaking changes: `0.20.0` "production" means validated and polished, **not** API-frozen. An
API-stability promise is a separate, deliberate `1.0.0` decision, intentionally deferred. (How
releases land on `master` and ramp through the stage ladder is the **Branching** section below.)

**Why the build counter sits in build metadata.** Everything before the dash is the *target
release*, which must not move every commit, so a free-running counter cannot live in the patch slot.
Putting it after `+` makes it **build metadata**, which semver compares as equal regardless of value
(`alpha.6+56` and `alpha.6+57` have the same precedence). That is safe here for three reasons: tag
resolution is by **exact tag string** (the CPM gate pins an explicit `GIT_TAG`, never a semver
range), so precedence is never used to pick a build; every tagged checkpoint **ticks `stage.X`**,
so no two checkpoints ever differ by build metadata alone; and because the stage points at the *next*
checkpoint, the version a dev tree compares equal to is one that has not been released — so the
equality can never be mistaken for an already-published tag. The build counter is therefore pure
provenance — it distinguishes dev commits between checkpoints, never two releases. (Caveat: OCI/Docker
image tags forbid `+`, so the optional runtime-image tag must sanitize it — drop the metadata or map
`+` to `-`.)

**The `0.13 -> 0.20` jump.** The minor was jumped from `0.13` to `0.20` to mark the production tier,
and the pre-release ladder rebased onto the `0.20.0` target. This stays forward in semver
(`0.20.0-… > 0.13.46-…`, minor compared first), keeping the timeline monotonic past already-published
tags like `v0.13.46-alpha.5` — which is why the target is `0.20.0` and not `0.13.0` (the latter would
sort *below* what is already released).

| Stage | Meaning | Example |
|---|---|---|
| `alpha.X` | features still landing; unstable | `0.20.0-alpha.6+119` |
| `beta.X` | feature-frozen; hardening only | `0.20.0-beta.N+M` (the stage `dev` is in now) |
| `rc.X` | release candidate | `0.20.0-rc.1+N` |
| _(none)_ | production-tagged | `0.20.0` |

Last checkpoint tagged: **`v0.20.0-beta.3`** (observability, and the container images published).

**`Version.txt`** at the repo root is the single source of truth. It feeds `project(VERSION ...)`
(the numeric triple) and the prerelease label separately; see `cmake/MilaVersion.cmake` — which
parses the numeric triple and carries the `-stage.X+build` tail as the prerelease label. (Today it
carries `+build` verbatim inside that label, which still reports correctly; a future one-line regex
tweak can split the `+build` metadata into its own field.) `Version.txt` is bumped **before
committing** — every commit carries the version it introduces — so the tag `vX.Y.Z-stage.X+build`
always points at a tree whose `Version.txt` matches it, and a consumer fetching that tag gets a Mila
that reports that exact version.

---

## Branching

Mila uses two long-lived branches with a strict one-way flow between them:

| Branch | Role | Audience | Contents |
|---|---|---|---|
| **`dev`** | working trunk / workbench | maintainers, contributors | every commit; CI-gated; `+build` ticks here |
| **`master`** | release front door (GitHub **default branch**) | end users | **only tagged releases** — nothing else |

**Why `master` is the default branch — the audience split made structural.** End users consume Mila
by pinning a **tagged release** via CMake FetchContent, so the branch a repo visitor lands on should
be the released artifact, not in-flight work. `master` (releases only) is that front door; `dev` is
the workbench. Contributors branch from and target `dev` (see
[getting-started](getting-started.md)).

**Work flows one way** — from the dev machine, out to `dev`, then a chosen checkpoint is promoted to
`master`:

```
dev machine  ->  dev (GitHub)  ->  master (GitHub)
  git push       maintainer         release PR of a chosen dev checkpoint,
                 commits            tagged vX.Y.Z[-stage.X], --prerelease for any pre-release tag
```

**`master` invariants:**

- Every commit on `master` **is** a tagged release — a `dev` checkpoint promoted through a single
  `dev -> master` PR and tagged `vX.Y.Z-stage.X` (pre-release) or `vX.Y.Z` (production).
- `master` **never** receives a direct commit and **never** carries an untagged one; it changes
  *only* via a release PR. (A stray direct edit to `master` is what diverged it from `dev` and caused
  the README merge conflict in the first release — treat `master` as release-only.)

**A cycle is a ramp of releases on `master`.** One feature-set era plays out as a sequence of
`dev -> master` release PRs climbing the stage ladder, each arrow its own tagged checkpoint:

```
alpha.1 -> alpha.2 -> ... -> beta.1 -> ... -> rc.1 -> ... -> X.Y.Z (production)
```

A pre-release ladder always opens on a **new** target, never on a version already shipped — once
`0.20.0` is tagged, the next cycle opens `0.21.0-alpha.1`. This stays monotonic
(`0.20.0 < 0.21.0-alpha.1 < … < 0.21.0`) because semver compares the numeric triple before the
pre-release tag, so a pre-release always sorts above every earlier stable and below its own final.
The one hazard: **never open a ladder on a shipped version** — `0.20.0-alpha.1` sorts *below*
`0.20.0` and rewinds the timeline; a hardening ramp opens on the next patch instead
(`0.20.1-alpha.1`).

---

## Cross-platform build policy

MSVC / VS 2026 is the primary development toolchain; the Linux/clang path is where portability
regressions hide (MSVC silently resolves includes and conformance that clang rejects). Two of the
three Linux surfaces are **maintainer gates** (do we ship a tree that compiles everywhere and passes
on hardware); the third is an **end-user feature** we verify but do not gate with:

| Surface | Role | What it does | GPU? |
|---|---|---|---|
| **GitHub CI** (`build-pipeline.yml`) | maintainer gate | *compiles* the full tree under clang-21 + packaging gates | no (hosted, no GPU) |
| **WSL** (`linux-clang-debug` preset) | maintainer gate | compiles the full tree under clang **and runs the CUDA test suite on real hardware** | yes |
| **Devcontainer** (`Docker/build-chat.sh`) | end-user convenience | a completely known build environment — clone, one step, a running Mila | yes |

The division that matters:

- **CI compiles but never executes on a GPU** ("device tests would be theater" on hosted runners);
  **WSL** adds the real-hardware test execution CI structurally cannot do. Together they are the
  portability + correctness gate.
- The **devcontainer is not a portability gate** — it is the *end-user's* known-good build
  environment, a convenience/onboarding feature we **ship** (the goal is "using Mila is a single easy
  step"). It is scoped to the Chat runtime path today, so it is deliberately blind to the test/binding
  tree; do **not** mistake a green devcontainer for portability coverage (that misread is how the
  `stop_token` and `std::min` clang breaks reached CI ungated — CI/WSL catch those, the container
  never will).

**The rules:**

1. **On `dev`, CI is the portability tripwire.** Do not run the Linux builds on every commit — the
   VS 2026 inner loop stays fast. Let CI catch the breaks.
2. **When dev CI goes red on a compile error, switch to the local WSL loop to reach green — do not
   debug portability through CI.** Each CI round trip is ~25 minutes; WSL gives the full error list
   in minutes. (CI runs `ninja -k 0`, so one red run now reports *every* error, not just the first —
   fix the batch locally, then push once.) CI should *confirm* green, not be the tool you iterate in.
3. **Before merging `dev -> master`:** the **WSL build** must pass (the portability + test gate), and
   the **devcontainer build** must still succeed — the latter because a broken end-user onboarding
   path is a shipped-product defect, not because it is a portability oracle.

---

## Everyday commit (on `dev`)

1. Make your change.
2. Bump `Version.txt` to the version this commit introduces.
3. In VS 2026, select the preset shown as **"x64 Release (full validation - run before committing)"**
   (its CMake `name` is `x64-validate` — the folder under `out/build/`), build, then `ctest` it:
   ```
   ctest --test-dir out/build/x64-validate --output-on-failure
   ```
   Expect: unit tests + `packaging_fetchcontent_consumer` green. FetchContent is the only
   supported consumption path. `packaging_cpm_consumer` is a separate preset, at step 7.
4. Commit and push to `dev`.

`dev` is the CI-gated trunk; releases reach `master` only through a `dev -> master` PR (see
**Branching**).

---

## The release surface

The website is the index of every way into Mila: four Get Started tabs (`#p-cpp`, `#p-python`,
`#p-docker`, `#p-clone`) and the `#evaluate` band. They end at the same place — a model answering on
the reader's own GPU — and differ only in what they consume to get there. **A release is finished
when every one of those five names the tag being released and still works**, which is more than
tagging: three of the five reach a registry Git cannot reach.

| Onboarding path | Site anchor | What the reader consumes | Published by | Step |
|---|---|---|---|---|
| C++ | `#p-cpp` | the git tag, via FetchContent | pushing the tag | 6, gated at 7 |
| Python | `#p-python` | `mila-llm` wheels on PyPI | `scripts/pypi/…` + hand upload | 1 validates, 8 publishes |
| Docker — environment | `#p-docker` | `toddthomson/mila-llm:<version>-devel` | `scripts/dockerhub/publish-image.sh` | 9 |
| Clone | `#p-clone` | the tag, built from source | pushing the tag | 6 |
| Try it | `#evaluate` | `toddthomson/mila-llm:<version>-runtime` | the same script, the same invocation | 9 |
| _the index itself_ | mila.toddt.me | the Pages deployment | `publish-site.yml`, dispatched | 10 |

**The site publishes last, and that order is load-bearing.** Its copy names two image tags and a
FetchContent pin, so dispatching it before those exist advertises commands that fail — the coupling
runs the other way from every other step, which take their input from the tag. It deploys from
**`dev`**, not from the tag, so the strings it names are the ones the step-2 prep commit wrote and
which are still on `dev` when the dispatch runs.

Two things belong to the surface but do **not** move with a release, so they are preconditions
rather than steps:

- **The published models.** All five paths need one in the store — four say how, and the C++ tab
  locates one it assumes is already there — but weights are published on their own schedule by
  `Mila/Tools/Publishing/publish_model.py`, and a card declares `minimum_mila_version` (`0.20.0`
  today), never a checkpoint. **A release never re-publishes weights.** What a release can break is
  the *name* in the copy, which is why the model names in `Web/layouts/index.html` are read back at
  step 10.
- **The dev container** (`Docker/docker-compose.yml`, `Docker/build-chat.sh`) — built by the reader
  from a clone and pushed nowhere. It is gated before the merge by the cross-platform build policy
  above. It is not the same image as `-devel`, which is built from `Docker/Dockerfile.runtime` and
  *is* published; do not let the shared word "container" collapse them.

---

## Cutting a release

Releases are **manual** — there is no release workflow. The GitHub Release object is
human-facing only (a curated changelog and download link); consumers resolve by git **tag**,
not by the Release. See the note below.

**Wheel validation comes first, and that is a correction.** It used to sit after the tag, on the
reasoning that the wheel version needs `Version.txt` stripped of `+build` — but that happens in the
prep commit on `dev`, and the tag contributes nothing to a wheel: the merge commit's tree is
byte-identical to `dev`'s head (verified at `beta.2`). Only the CPM gate is genuinely post-tag,
because it clones from GitHub at the tag. At `beta.2` we tagged, published a Release and a
Discussion, and only then discovered the wheels could not be validated at all. Under this order that
lands while nothing is immutable yet.

1. **Validate the wheels from a `dev` snapshot, before anything is permanent.** Build all four from
   the current `dev` head — still carrying `+build`, so they version as `0.20.0b3.devN` — and take
   them through [Publishing the wheels](#publishing-the-wheels) steps 1, 2, 3 and 4. **TestPyPI takes
   only `.devN` snapshots; the plain release version goes only to PyPI.** A filename is burned
   permanently on first upload, so validating at `0.20.0b2` leaves no second attempt if a fix is
   needed — and a stray `0.20.0b2` upload once poisoned that release's `Requires-Python` at `>=3.13`
   (PyPI fixes it at the release level from the first file and never updates it), which broke both
   3.12 legs of the clean room three weeks later and could not be repaired: delete does not free
   filenames, and yank changes neither metadata nor an exact pin. The `.devN` snapshot is the same
   binaries under a disposable version, so it costs one extra wheel build and nothing else.
   **First time only:** the clean-room workflow is `workflow_dispatch` and so is dispatchable only
   once `wheel-cleanroom.yml` is on `master`. Until the merge that puts it there, this step has to
   follow the merge instead — once, and never again.
2. **Release-prep commit on `dev`** — set `Version.txt` to the checkpoint string with the `+build`
   metadata **dropped**: `0.20.0-beta.2+7` becomes `0.20.0-beta.2`. A tag never carries build
   metadata, so this is what lets step 5's drift check pass. If the checkpoint is being renamed from
   its working placeholder (`beta.2` -> `rc.1`), this is the commit that does it. Reconcile
   BACKLOG / ROADMAP in the same commit, and **bump every stage string written in prose**: the
   `README.md` status callout and its *Current Status* heading, and **this document's own
   "Last checkpoint tagged" line** above. `master` is the branch a visitor lands on, so a missed bump
   leaves the front page advertising the previous checkpoint for the whole next cycle — and a
   procedure that misreports the last release is worse than one that says nothing.
   **Bump the QuickStart pin to the tag being released** — the `GIT_TAG` in
   `Mila/Samples/QuickStart/Cpp/CMakeLists.txt`, and in `Mila/Samples/QuickStart/Cpp/README.md` and
   `getting-started.md` §7, where it appears in copy-paste blocks (the README also carries it in a
   `URL` archive line). Nothing checks these strings: `packaging_fetchcontent_consumer` overrides
   the ref with `SOURCE_DIR`, and `packaging_cpm_consumer` reads its tag from `Version.txt`, not from
   the sample. They went stale once already, pointing at an unreleased `v0.20.0` — a downstream
   consumer copying the sample got a checkout failure on their first build.
   **Bump the website's copy in the same commit** — `Web/layouts/index.html` hardcodes the version
   at five sites, and nothing derives them: the C++ tab's `GIT_TAG` (`#p-cpp` step 1) and its sample
   output (`#p-cpp` step 3), the `-devel` image tag (`#p-docker` step 1), and the `-runtime` image
   tag in **both** `#evaluate` commands. They have already gone out of step with each other once —
   the tab pinned `beta.2` while the output beside it read `beta.3`. Bump
   `scripts/dockerhub/verify-image.sh`'s `MILA_IMAGE` default with them, since it exists to run the
   `#evaluate` commands and a stale default verifies the previous release. This is a `dev` commit
   and publishes nothing; the site goes live at step 10, after the images it names.
   **Clear any "not published yet" copy the release makes false** — today the `#p-docker` panel
   carries a flag saying both tags are local, and `getting-started.md` and `README.md` each carry a
   note calling the slim runtime image "planned". If step 9 then fails, nothing false has reached a
   reader: the site is not dispatched until step 10.
   **CHANGELOG only at a production (unsuffixed) release** — generate one short entry from the
   commit range since the previous production tag, and collapse that line's `alpha.N`/`beta.N`/`rc.N`
   sections into it. A pre-release flip writes nothing to CHANGELOG.
3. Open a `dev -> master` pull request. CI validates on the PR.
4. Merge to `master`.
5. **Drift check (by eye — this used to be an automated gate):** the tag you are about to
   create must be exactly `v` + the contents of `Version.txt`, e.g. a `Version.txt` of
   `0.13.46-alpha.5` -> tag `v0.13.46-alpha.5`. A tag that disagrees with `Version.txt` makes a
   semver consumer fetch a tree that reports a different version.
6. Tag `master` and push the tag. **Tagging `master` is the release** — CPM/FetchContent fetch
   this git tag directly, and GitHub auto-generates the source archives at it. Nothing else is
   required for the library to be consumable downstream.
7. **Post-tag smoke test — the one step that genuinely needs the tag,** because it clones from
   GitHub at it. Select the preset shown as **"x64 Release (CPM release-access gate)"**
   (CMake `name` `x64-release-cpm-gate`) and run:
   ```
   ctest --test-dir out/build/x64-release-cpm-gate -R packaging_cpm_consumer --output-on-failure
   ```
   **Pass the tag explicitly — `-DMILA_CPM_GIT_TAG=v0.20.0-beta.2` — rather than relying on the
   default.** The gate derives it from `Version.txt` when unset, which lines up here, but a build
   directory reused across releases once kept a tag it was configured with months earlier, tested the
   *previous* release, and passed off a warm cache in 134 seconds. The configure now prints
   `CPM release-access gate: testing <repo>@<tag>` and the run announces the same before it fetches:
   **read that line.** A pass that does not name the tag you just pushed is not a pass.
   Still run it **before step 12**, which moves `Version.txt` off the tag.
8. **Upload the wheels to PyPI** — [Publishing the wheels](#publishing-the-wheels) step 5. Build them
   from the tagged tree first (steps 1 and 2 there), since these carry the release version rather than
   the `.devN` snapshot validated in step 1. Like step 7 this must happen **before step 12**: the
   wheel version derives from `Version.txt`, so a wheel built after the next checkpoint opens carries
   the wrong version entirely.
9. **Publish the container images** — [Publishing the container images](#publishing-the-container-images).
   Two tags, `-runtime` and `-devel`, from one script invocation. Post-tag by construction: the
   script refuses to build unless the tag exists on `origin`, `HEAD` is that tag, and the tree is
   clean. **Its own sitting** — the build is long and runs in WSL2, so do not start it alongside a
   native build competing for the same machine.
10. **Publish the website** — [Publishing the website](#publishing-the-website). Dispatch
   `Publish Mila Site`. **Last, and only once steps 6, 8 and 9 are all green**, because the copy
   published here names the tag, the wheel and both image tags; a reader who lands on a command
   naming something that does not exist has no way to tell a typo from a release in progress. The
   deploy replaces the live site wholesale and has no staging, so read the assembled build first.
11. **(Optional, human-facing) Publish a GitHub Release** for a curated changelog:
   ```
   gh release create v0.13.46-alpha.5 --notes-file release-notes.md --prerelease
   ```
   **Not `--generate-notes`.** GitHub builds those notes from the pull requests merged between two
   tags; Mila lands all work as direct commits on `dev` and opens exactly one PR per release, so
   `beta.2` would have produced a one-line release for 48 commits of work. The substance lives only in
   the commit messages, so the body is **authored from the commit range** — which is what `beta.1`
   actually did, hand-written with only its trailing `Full Changelog` footer generated. One summary
   serves both destinations: it becomes the Release body at any tag, and distils into the CHANGELOG
   entry at a production release.
   Apply `--prerelease` to **every** `dev -> master` pre-release flip — `alpha.N`, `beta.N`, and
   `rc.N` alike — and drop it **only** for the final production tag. GitHub never awards the "Latest
   release" badge to a prerelease, so this is what keeps the last production release badged as Latest
   throughout the next cycle's pre-release ramp. Or draft it in the **Releases** web UI for full
   hand-curation. Nothing downstream depends on this, so do it on your own schedule.
12. **Open the next checkpoint on `dev`** — bump `Version.txt` to the *next* stage ordinal with the
   counter reset, e.g. having just tagged `v0.20.0-beta.2`, `dev` becomes `0.20.0-beta.3+1` (or
   `0.20.0-rc.1+1`, if that is the call). Its own `dev` commit, same sitting as the tag. Skipping it
   leaves the working tree reporting an already-shipped version — the failure mode this scheme exists
   to prevent. After a **production** tag, this is where the next cycle opens instead
   (`0.21.0-alpha.1+1`); never reopen a ladder on a shipped version.

**Steps 2 to 12 are one window, and `dev` stays closed to unrelated commits across it.** Four
steps build from `dev` or from the tag it produced — the wheels, both images, and the site — and
each one that lands after an unrelated commit is built from a tree the tag does not describe. This
is a **release-window hold**, temporary and tree-wide, and it lifts when step 12 commits. It is not
the feature freeze, which is permanent for the cycle and scoped to `Mila/Src`; say which one is
refusing a change.

---

## Publishing the wheels

`mila-llm` on PyPI is a published release artifact, built by two scripts and uploaded by hand. The
wheel version comes from `Version.txt`: a working `+38` tree produces the `0.20.0b3.dev38` snapshot,
and the prep commit's stripped `0.20.0-beta.2` produces the release `0.20.0b2`.

**This section is run twice, and the two runs differ only in which version they carry.** Steps 1-4
run at **release step 1**, on a `dev` snapshot, and validate. Steps 1, 2 and 5 run again at
**release step 8**, from the tagged tree, and publish. The binaries are the same; validating a
throwaway version is what keeps the release filename unburned.

**The CUDA toolkit and the architecture list are pinned in the tree, not taken from the machine.**
Both wheels and both images carry `80;86;89;90;120` — the published-artifact list, one entry
narrower than the library's portable default, because SM 8.0 is the floor Mila's own kernels draw
(`CudaLinearOp.ixx:661`, and both GQA flash prefill paths throw below it). The four sites are the
two wheel presets, `ARCHITECTURES` in `publish-image.sh`, and `MILA_IMAGE_CUDA_ARCHITECTURES` in
`Dockerfile.runtime`; they are one list and move together. The toolkit is declared the same way:
`$cudaVersion` in `scripts/pypi/build-wheel-windows.ps1` for Windows, the base image in
`Docker/Dockerfile.wheel` and `Docker/Dockerfile.runtime` for Linux — currently **13.3** across all
four. Before that the Windows wheel took whatever `CUDA_PATH` resolved to, so installing a toolkit
on the dev box silently changed a published wheel; 13.4 landed mid-cycle and would have split one
release across two toolchains. **Moving to a new toolkit is an edit to all of those together, and
never inside a release window** — `MILA_WHEEL_CUDA_VERSION` overrides the Windows one for a
maintainer on a different box, and whichever is used gets printed at the top of the run.

**What the declared toolkit is not.** It constrains the six published binaries and nothing else: a
FetchContent or clone consumer builds with their own CUDA, and `getting-started.md` states the
user-facing floor (13.0 or newer) independently of it. A wheel user needs **no toolkit at all** —
`pyproject.toml`'s `nvidia-*` dependencies supply the runtime, so what they need is a driver. The
minor version is therefore invisible downstream (CUDA minor version compatibility holds within 13),
and only the **major** is load-bearing, since those dependencies pin `>=13.0,<14.0`; the wheel
script asserts that and leaves the minor as a declaration. Do not reach for this pin to answer a
question about what hardware or software a user must have — that answer lives in
`getting-started.md` and in the architecture list, not here.

**A PyPI upload cannot be undone.** Release metadata is immutable and a filename can never be reused,
so a wheel published before it was verified stays wrong until the *next* release — which is exactly
how the live page came to advertise Linux while shipping only `win_amd64`. That is what the TestPyPI
step below exists to prevent, and why it is not optional.

1. **Build all four wheels** from the tagged tree — one per interpreter (3.12, 3.13) per platform.
   Each script clears only its own platform's wheels from `out/wheel`, because all four land there
   and all four are published from one glob.
   - Windows: `scripts/pypi/build-wheel-windows.ps1` — enters the VS developer shell, then configures the
     `x64-wheel` preset once per interpreter and packages from a copy of the package tree. The
     interpreter list is `$interpreters` at the top of the script and must match `requires-python`
     in `pyproject.toml`.
   - Linux: from `Docker/`, build and run the wheel container —

     ```bash
     docker compose -f docker-compose.wheel.yml build
     docker compose -f docker-compose.wheel.yml run --rm mila-wheel mila-build-wheel
     ```

     It must be the **wheel** container (Ubuntu 24.04), not the dev container: `auditwheel` derives
     the manylinux tag from the build distro, so 26.04 would produce a wheel that locks out the
     current LTS.
2. **Check what is actually in `out/wheel`** — exactly four files, all carrying the release version
   and nothing else. A leftover wheel from an earlier build is published alongside the intended one by
   the same glob, and that cannot be withdrawn. Expect the directory to hold the previous
   checkpoint's `.devN` wheels when you arrive here: each script clears only its own platform's, so a
   stale wheel survives whenever the interpreter list or the platform set has changed.
3. **Upload to TestPyPI** with `twine upload --repository testpypi out/wheel/*.whl` — never to PyPI
   first, and **only ever a `.devN` snapshot.** TestPyPI
   burns a filename permanently on first upload exactly as PyPI does, so uploading the plain release
   version there leaves no second attempt if a fix is needed. Worse, a stray `0.20.0b2` upload once
   pinned that release's `Requires-Python` at `>=3.13` for good — PyPI fixes it at the release level
   from the first file it sees and never revisits it — which broke both 3.12 legs of the clean room
   three weeks later, unrepairably: delete does not free filenames, and yank changes neither metadata
   nor an exact pin.
4. **Dispatch the `Wheel clean room` workflow** (Actions -> Wheel clean room -> Run workflow) with the
   exact version (`0.20.0b3.dev38`) and index `testpypi`. It runs a four-leg matrix — `windows-latest` and
   `ubuntu-latest` x Python 3.12 and 3.13, one leg per published wheel, none of them carrying a CUDA
   Toolkit — and runs `scripts/pypi/verify_wheel_cleanroom.py`, which asserts that absence *before* it
   asserts anything else. A developer machine cannot answer this question, because a wheel quietly
   leaning on a host Toolkit passes there exactly the way a correct one does.
   All four legs must be green. The version is pinned exactly because PyPI carries an older
   `mila-llm` that can outrank a TestPyPI build; the script re-asserts the version it actually got.
5. **Upload to PyPI** with `twine upload out/wheel/*.whl`. Only now, at release step 8, and only if
   step 4 was green on all four legs. This is the release version, and it is the only place it is
   ever uploaded.

**The upload client is `twine`, and its credentials are not in the repository.** `~/.pypirc` names
the two indexes and the username `__token__`; the API tokens themselves live in the OS keyring
(Windows Credential Manager), one per index, so an upload runs without a prompt on a configured box
and not at all on any other. A maintainer setting up a new machine installs `twine`, writes that
`.pypirc`, and lets the first upload store the tokens.

The workflow is `workflow_dispatch`, so it is dispatchable only once `wheel-cleanroom.yml` is on the
default branch (`master`). Until the merge that first puts it there, the validation run has to
follow the merge rather than precede it — a one-off, called out at release step 1.

---

## Publishing the container images

`toddthomson/mila-llm` on Docker Hub carries two tags per release, and they are two of the five
onboarding paths: **`-runtime`** backs the `#evaluate` band (run a model, nothing else) and
**`-devel`** backs the `#p-docker` tab (a built tree the reader edits). Both come from
`Docker/Dockerfile.runtime` as separate targets sharing one builder stage, so one script builds
both and the compile happens once. `scripts/dockerhub/publish-image.sh` is the only writer.

This runs at **release step 9**, and it is post-tag by construction rather than by convention: the
script's gates refuse to build unless the tag resolves locally, exists on `origin`, equals `HEAD`,
and the tree is clean. A `-devel` image ships `/src`, so a reader reads and edits that source — if
it came from anywhere but a public tag there is nothing for them to reproduce against.

1. **Log in first.** `docker login`. The script never handles credentials; it only checks that you
   did, and only when `--push` is passed.
2. **Check out the tag**, so `HEAD` is it. Run from WSL2 — this is a Linux image build, and it is
   long. Do not start it while a native build is running; the two contend for the same machine.
3. **Build both images, without pushing:**
   ```bash
   scripts/dockerhub/publish-image.sh v0.20.0-beta.3
   ```
   Build-only is the default and `--push` is a deliberate second decision, the same shape as the
   site workflow. `MILA_CLEAN_BUILD` is forced rather than inherited: `--no-cache` invalidates
   layers but leaves BuildKit cache mounts intact, and two wrong images have already been built
   from another tree's objects that way.
   The architecture list is `80;86;89;90;120`, fixed in the script and **not** the library's
   portable list — it drops Turing, which both GQA flash prefill paths refuse outright. `native` —
   the local scripts' default — is wrong twice here, since it does not resolve on a GPU-less
   builder and the image is pulled by hardware the builder never saw.
4. **Run the `#evaluate` sequence against the local `-runtime` image, on a GPU host:**
   ```bash
   MILA_IMAGE=toddthomson/mila-llm:0.20.0-beta.3-runtime scripts/dockerhub/verify-image.sh
   ```
   It reproduces the website's two commands with exactly two substitutions — the image reference
   and a throwaway store volume, fresh per run so `install` cannot report success by finding the
   model already there. **When the `#evaluate` copy changes, this script changes with it; that
   coupling is the point.** CI cannot stand in for this: a hosted runner has no GPU, so nothing in
   the image has been executed until you run this.
5. **Walk the `-devel` tab by hand.** No script covers it, and its three steps are a different
   claim from `-runtime`'s: land in a shell in a built tree, `mila install` a model, then build and
   run `~/myapp` against the compiled library. The tab's whole promise is that the first build takes
   seconds, so an image that builds but carries a cold or missing build tree passes steps 3 and 4 —
   neither of which touches `-devel` — and fails the reader.
6. **Push**, once 4 and 5 are green:
   ```bash
   scripts/dockerhub/publish-image.sh v0.20.0-beta.3 --push
   ```
   It asks for the version string rather than a `y/n`, because a reflexive "y" is not a decision.
   **A pushed tag cannot be withdrawn, only superseded** — which is the one thing here that is
   gentler than PyPI, and the reason a re-publish over a bad image is a real remedy.

**No `latest`.** A bare `docker run toddthomson/mila-llm` resolves to it, so pointing it at a
pre-release makes the beta the default for everyone who does not read the tag list. It starts
existing at the first unsuffixed release and tracks the newest one's `-runtime`. The rule lives in
`TARGETS`/the header comment of `publish-image.sh`; this is not an open decision.

> **One-off for `v0.20.0-beta.3`:** `toddthomson/mila-llm:0.20.0-beta.3-runtime` is **already on
> Docker Hub**, pushed 2026-08-31 from a dirty tree with the gates bypassed knowingly, to prove the
> site's `#evaluate` band from the registry. The gated publish above **must overwrite it** — the
> failure mode is seeing the tag already present and skipping step 9. `-devel` at that version has
> never been published.

---

## Publishing the website

mila.toddt.me is the index of every onboarding path, so it is the last thing published and the only
one whose copy is invalidated by the others rather than the reverse. It is a Hugo site plus the
Doxygen API reference, deployed to GitHub Pages by `.github/workflows/publish-site.yml`.

**It publishes from `dev`, not from the tag**, and it is `workflow_dispatch:` only — running the
workflow *is* the decision to publish, and the deploy replaces the live site wholesale with no
staging environment behind it. An earlier version auto-published on any push touching `Web/**`,
which shipped unready work in one direction and stayed silent for two weeks in the other.

1. **Read back the strings step 2 wrote.** Five in `Web/layouts/index.html` (the C++ tab's `GIT_TAG`
   and sample output, the `-devel` tag, and both `-runtime` commands), plus any "not published yet"
   flag the release has now falsified. Every model name in the copy must be one the store actually
   serves — `gemma-4-12b-it-fp4` and `Llama-3.2-3B-Instruct-fp4` today.
2. **Build it without deploying** — dispatch **`Mila Web`** (`web.yml`), which runs the same Hugo
   build and the same JSON-LD validation and never touches Pages. Inspect its artifact. This is the
   staging step the publish workflow structurally lacks.
3. **Dispatch `Publish Mila Site`.** The build job gates the deploy job: a broken Hugo build, a
   missing assembled file, or malformed structured data fails before anything goes live. Doxygen
   runs under `WARN_AS_ERROR`, so **doc-comment drift in `Mila/Src` blocks the publish** — it is a
   source problem, and the fix belongs in the source, not in a workflow flag.
4. **Open the published page and run one command out of each panel.** The C++ tab's `GIT_TAG`
   against the tag you pushed, `pip install mila-llm` resolving to the release version, and both
   `#evaluate` commands from the registry with every local copy deleted first — a local image makes
   a `docker run` succeed regardless of what was pushed.

The API reference deploys in the **same** Pages artifact and therefore tracks `dev`, not the tag: a
Pages deployment replaces the whole site, so publishing the authored site alone would delete `/api`
from it. That asymmetry is deliberate — reference docs generated from source should follow source.

---

## Notes

- **Releases are created by hand (no release workflow).** The GitHub Release is human-facing
  only — CPM and FetchContent resolve by git **tag**, and GitHub serves source archives from the
  tag regardless — so the Release object is curated manually (`gh release create` or the web UI)
  rather than auto-cut on tag push by a third-party action. This keeps release timing and content
  under explicit control, and the drift check (tag == `Version.txt`) moves to step 5 above. A
  tag-triggered workflow (`release.yml`, `softprops/action-gh-release`) previously did this; it
  was removed deliberately.
- **Tag format:** `vX.Y.Z` or `vX.Y.Z-PRERELEASE`. The CPM gate uses an explicit `GIT_TAG`
  (not CPM's `@version` shorthand, which mishandles the `-alpha.N` pre-release suffix).
- **Testing an older tag mid-development:** the CPM gate derives its tag from `Version.txt` when
  `MILA_CPM_GIT_TAG` is empty, but you can point it at any already-pushed tag:
  `-DMILA_CPM_GIT_TAG=v0.13.45-alpha.5` (also `-DMILA_CPM_GITHUB_REPOSITORY=<owner/repo>`). Note
  that an explicit value **persists in the cache**, so clear it (`-DMILA_CPM_GIT_TAG=`) to go back
  to the derived one.
- **Stale CPM cache:** the gate keeps a source cache across runs for speed, now under
  `.../Mila/Tests/Packaging/cpm-cache/<tag>`, with its build work directory keyed on the tag as
  well. Both are per-tag deliberately: sharing them is half of how a gate once validated the
  previous release and reported success. If a re-run misbehaves after a failed attempt, delete that
  tag's folder to force a clean fetch. **A passing run is only evidence for the tag it names** —
  the configure line and the run's first message both print it.
