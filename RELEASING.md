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
  stage **checkpoint ordinal** — it ticks each time a checkpoint is tagged within a stage. It is a
  pure release-provenance count: it does not name, count, or correspond to any unit of planned work.
- **build** — a per-commit counter carried as **semver build metadata** (after `+`). It counts the
  `dev` commits accumulated toward the next checkpoint and **resets at every checkpoint tag**, so it
  reads as "commits since the last release" — not a repo-lifetime counter. It is **ignored for
  version precedence** by the spec.

Each feature set opens a new minor and runs its own ladder; features never land inside a hardening
ladder — a stabilizing release takes only patch-level fixes. Mila is pre-1.0, so any release may
carry breaking changes: `0.20.0` "production" means validated and polished, **not** API-frozen. An
API-stability promise is a separate, deliberate `1.0.0` decision, intentionally deferred. (How
releases land on `master` and ramp through the stage ladder is the **Branching** section below.)

**Why the build counter sits in build metadata.** Everything before the dash is the *target
release*, which must not move every commit, so a free-running counter cannot live in the patch slot.
Putting it after `+` makes it **build metadata**, which semver compares as equal regardless of value
(`alpha.6+56` and `alpha.6+57` have the same precedence). That is safe here for two reasons: tag
resolution is by **exact tag string** (the CPM gate pins an explicit `GIT_TAG`, never a semver
range), so precedence is never used to pick a build; and every tagged checkpoint **ticks `stage.X`**,
so no two checkpoints ever differ by build metadata alone. The build counter is therefore pure
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
| `beta.X` | feature-frozen; hardening only | `0.20.0-beta.1` (now) |
| `rc.X` | release candidate | `0.20.0-rc.1+N` |
| _(none)_ | production-tagged | `0.20.0` |

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
   Expect: unit tests + `packaging_find_package_consumer` + `packaging_fetchcontent_consumer` green.
4. Commit and push to `dev`.

`dev` is the CI-gated trunk; releases reach `master` only through a `dev -> master` PR (see
**Branching**).

---

## Cutting a release

Releases are **manual** — there is no release workflow. The GitHub Release object is
human-facing only (a curated changelog and download link); consumers resolve by git **tag**,
not by the Release. See the note below.

1. Open a `dev -> master` pull request. CI validates on the PR.
2. Merge to `master`.
3. **Drift check (by eye — this used to be an automated gate):** the tag you are about to
   create must be exactly `v` + the contents of `Version.txt`, e.g. a `Version.txt` of
   `0.13.46-alpha.5` -> tag `v0.13.46-alpha.5`. A tag that disagrees with `Version.txt` makes a
   semver consumer fetch a tree that reports a different version.
4. Tag `master` and push the tag. **Tagging `master` is the release** — CPM/FetchContent fetch
   this git tag directly, and GitHub auto-generates the source archives at it. Nothing else is
   required for the library to be consumable downstream.
5. **(Optional, human-facing) Publish a GitHub Release** for a curated changelog:
   ```
   gh release create v0.13.46-alpha.5 --generate-notes --prerelease
   ```
   Apply `--prerelease` to **every** `dev -> master` pre-release flip — `alpha.N`, `beta.N`, and
   `rc.N` alike — and drop it **only** for the final production tag. GitHub never awards the "Latest
   release" badge to a prerelease, so this is what keeps the last production release badged as Latest
   throughout the next cycle's pre-release ramp. Or draft it in the **Releases** web UI for full
   hand-curation. Nothing downstream depends on this, so do it on your own schedule.
6. **Post-tag smoke test:** select the preset shown as **"x64 Release (CPM release-access gate)"**
   (CMake `name` `x64-release-cpm-gate`) and run:
   ```
   ctest --test-dir out/build/x64-release-cpm-gate -R packaging_cpm_consumer --output-on-failure
   ```
   This git-clones Mila from GitHub at the tag and builds a consumer against it via CPM,
   proving the release is actually consumable downstream. The gate's tag defaults to the
   current `Version.txt`, so at this moment it lines up with the tag you just pushed.

---

## Notes

- **Releases are created by hand (no release workflow).** The GitHub Release is human-facing
  only — CPM and FetchContent resolve by git **tag**, and GitHub serves source archives from the
  tag regardless — so the Release object is curated manually (`gh release create` or the web UI)
  rather than auto-cut on tag push by a third-party action. This keeps release timing and content
  under explicit control, and the drift check (tag == `Version.txt`) moves to step 3 above. A
  tag-triggered workflow (`release.yml`, `softprops/action-gh-release`) previously did this; it
  was removed deliberately.
- **Tag format:** `vX.Y.Z` or `vX.Y.Z-PRERELEASE`. The CPM gate uses an explicit `GIT_TAG`
  (not CPM's `@version` shorthand, which mishandles the `-alpha.N` pre-release suffix).
- **Testing an older tag mid-development:** the CPM gate defaults its tag to `Version.txt`,
  but you can point it at any already-pushed tag:
  `-DMILA_CPM_GIT_TAG=v0.13.45-alpha.5` (also `-DMILA_CPM_GITHUB_REPOSITORY=<owner/repo>`).
- **Stale CPM cache:** the CPM gate keeps a source cache across runs for speed
  (`.../Mila/Tests/Packaging/cpm-cache`). If a re-run misbehaves after a failed attempt,
  delete that folder to force a clean fetch.
