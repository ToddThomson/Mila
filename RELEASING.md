# Releasing

How Mila is versioned, branched, validated, and tagged into a consumable release. Planning and
progress live in [ROADMAP.md](ROADMAP.md) / [BACKLOG.md](BACKLOG.md); this document is only the
release mechanics.

Two things to internalize up front. **Every version produces exactly one tag, and a dev build is
never tagged** — `Version.txt` carries a ticking **build** counter in semver build metadata while a
version is being built, and drops the whole pre-release tail at the tag. And **a tag is not a
publish**: a minor publishes to all five channels, a patch is a git tag for source consumers and
costs almost nothing. Both are detailed in the next section.

---

## Versioning

Mila uses a **one-tag cycle**: `MAJOR.MINOR.PATCH-dev+N` while the release is being built, and
`MAJOR.MINOR.PATCH` at the tag — `0.21.0-dev+7` becomes `v0.21.0`.

- **minor** — the release. It steps by **one** per cycle.
- **patch** — a fix release on an already-tagged minor; usually `.0`.
- **dev** — the only stage there is, and it carries no ordinal, because there is no checkpoint to
  number.
- **build** — a per-commit counter carried as **semver build metadata** (after `+`). It counts the
  `dev` commits accumulated toward the release, and **resets to `+1` when that release is tagged**.
  It is **ignored for version precedence** by the spec.

**The whole string points forward.** `Version.txt` names *what is being built*, never what was last
built — the git tag is the record of what shipped. So on `dev`, `0.21.0-dev+7` reads "the 0.21.0
release, seven commits in". The tail is dropped and the counter reset **at the moment the release is
tagged** (see step 12 of *Cutting a release*), so the working tree never reports a version that has
already shipped.

**`dev` carries the next patch by default.** Having tagged `v0.22.0`, `dev` becomes `0.22.1-dev+1`,
because the cheaper of the two acts below is the one that should need no decision to reach.
Promoting the cycle to a minor is a deliberate rename in the release-prep commit, made when the goal
in [ROADMAP.md](ROADMAP.md) has actually landed. The working version is therefore a placeholder in
exactly one bit, and step 2 is where that bit is resolved.

### A tag and a publish are different acts, and only one is expensive

Tagging `master` **is** the release for a source consumer: CPM and FetchContent fetch the git tag
directly and GitHub generates the source archives at it, so the whole cost is the gate that proves
the tree. Publishing is the separate act of carrying a *built* artifact to somebody — wheels to
PyPI, images to Docker Hub, and the site and Release body that name them — and every part of it is
immutable once uploaded. Fusing the two makes the cheap act inherit the expensive one's price, which
is the only reason a source release ever looked costly.

They are separated by the slot, so the version number says which one a reader is holding:

| | What it is | Reaches | Cost |
|---|---|---|---|
| **minor** — `0.22.0` | a **publish** | source, PyPI, Docker Hub, the site, a Release body | the full procedure below |
| **patch** — `0.22.1` | a **tag** | source consumers only | the gate, a merge, the tag, the CPM gate |

A patch tag builds no wheel, builds no image, dispatches no site, and **gets no GitHub Release
object** — the Release marks a publish, which is what keeps GitHub's "Latest release" badge pointing
at the version that actually exists on PyPI and Docker Hub. It gets no `ROADMAP.md` section and no
`BACKLOG.md` bucket either: a fix is not a goal, and its commit is its own record.

**The sixteen version sites do not move on a patch.** They name the last *published* release, which
is the wording they have always carried — so a reader copying a `GIT_TAG` out of
`getting-started.md` gets the combination that was built, clean-roomed and documented rather than the
newest source tag, and a consumer who wants a particular fix bumps the pin themselves. That is what
lets a patch tag touch exactly one tracked file.

**What the source consumer is not getting, stated plainly:** no wheel or image was built from a patch
tag, so no clean room has ever seen that tree. It costs them nothing, because they compile it
themselves and the CPM gate proves they can. It does mean the binary channels sit visibly behind the
tag list between minors.

**Each minor carries one or two goals**, which is what makes one publish per cycle sufficient: a
release small enough to state in a sentence converges without a hardening ramp, and providing that
ramp is what the four-rung ladder existed for. The patches between minors are where the tree stays
reachable while that goal is being built. Mila is pre-1.0, so any release may carry breaking
changes: "production" means validated and polished, **not** API-frozen. An API-stability promise is
a separate, deliberate `1.0.0` decision, intentionally deferred. (How either kind of tag lands on
`master` is the **Branching** section below.)

**Why the build counter sits in build metadata.** Everything before the dash is the *target
release*, which must not move every commit, so a free-running counter cannot live in the patch slot.
Putting it after `+` makes it **build metadata**, which semver compares as equal regardless of value
(`dev+56` and `dev+57` have the same precedence). Under the ladder that took a three-part argument to
justify. It now takes one sentence: **a dev build is never tagged**, so no two releases can differ by
build metadata alone, and precedence is never asked to tell two snapshots apart. The counter is pure
provenance — it distinguishes dev commits within a cycle, never two releases. (Caveat: OCI/Docker
image tags forbid `+`. `publish-image.sh` refuses such a tag outright rather than sanitizing one,
because it publishes only from a release tag, which never carries the metadata.)

**The retired ladder, and the `0.13 -> 0.20` jump.** Through `0.20.0` a cycle ramped
`alpha.X -> beta.X -> rc.X ->` unsuffixed stable, each rung its own tag, and the minor was jumped
from `0.13` to `0.20` to mark the production tier. Both are history as of the `0.21.0` cycle.

The ladder went because four tagged rungs is four releases' worth of mechanics for one release's
worth of work. The jump went because tenths are a finite ordered resource with `1.0` at the end of
them: spent at one per release they run out in a few cycles, `1.0` arrives as arithmetic rather than
as the deliberate API-freeze decision it is meant to be, and the release numbers march straight
through the era names `Specifications/Direction.md` uses for work that is years out. **The minor now
steps by one.**

Old tags stay valid and keep sorting correctly, because semver compares the numeric triple before the
pre-release tail: `0.13.46-alpha.5 < 0.20.0-beta.3 < 0.20.0 < 0.21.0-dev+1 < 0.21.0`.

| Stage | Meaning | Example |
|---|---|---|
| `dev` | the release being built; `+N` ticks once per commit | `0.21.0-dev+7` |
| _(none)_ | tagged release | `0.21.0` |

Last checkpoint tagged: **`v0.20.0`** (observability, and the container images published).

**`Version.txt`** at the repo root is the single source of truth. It feeds `project(VERSION ...)`
(the numeric triple) and the prerelease label separately; see `cmake/MilaVersion.cmake` — which
parses the numeric triple and carries the `-dev+N` tail as the prerelease label, then translates the
pair into the PEP 440 spelling the wheel needs (`0.21.0-dev+7` -> `0.21.0.dev7`). That translation
**refuses a tail it does not recognise** rather than guessing, because the version it produces goes
on a published artifact and a published version can never be reused. `Version.txt` is bumped
**before committing** — every commit carries the version it introduces — so the tag `vX.Y.Z` always
points at a tree whose `Version.txt` matches it, and a consumer fetching that tag gets a Mila that
reports that exact version.

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

**Work flows one way** — from the dev machine, out to `dev`, then one chosen commit is promoted to
`master`:

```
dev machine  ->  dev (GitHub)  ->  master (GitHub)
  git push       maintainer         release PR of a chosen dev commit,
                 commits            tagged vX.Y.Z
```

**`master` invariants:**

- Every commit on `master` **is** a tagged release — a `dev` commit promoted through a single
  `dev -> master` PR and tagged `vX.Y.Z`.
- `master` **never** receives a direct commit and **never** carries an untagged one; it changes
  *only* via a release PR. (A stray direct edit to `master` is what diverged it from `dev` and caused
  the README merge conflict in the first release — treat `master` as release-only.)

**Every arrow is one tag.** Work lands on `dev` as ordinary commits, the counter ticks, and a single
`dev -> master` release PR promotes the commit that gets tagged. Most arrows are patches, which is
the point of the split above — they are cheap, so they are frequent:

```
0.21.0-dev+1 .. +N   ->  v0.21.0    a publish: source, PyPI, Docker Hub, site, Release body
0.21.1-dev+1 .. +N   ->  v0.21.1    a tag: source consumers, same day it is green
0.21.2-dev+1 .. +N   ->  v0.21.2
0.22.0-dev+1 .. +N   ->  v0.22.0    the goal landed, so the cycle was promoted to a minor
```

Each opens on a **new** version, never on one already shipped, and stays monotonic because semver
compares the numeric triple before the pre-release tail — so a `-dev` snapshot sorts above every
earlier release and below its own final. The one hazard: **never reopen on a shipped version**, since
`0.21.0-dev+1` after `0.21.0` has shipped sorts *below* it and rewinds the timeline.

The promotion to a minor happens in the release-prep commit, not here: `dev` runs on patch numbering
until the goal in [ROADMAP.md](ROADMAP.md) has actually landed, and renaming `0.21.3-dev+9` to
`0.22.0` at step 2 is the act that says it has.

---

## Cross-platform build policy

MSVC / VS 2026 is the primary development toolchain; the Linux/clang path is where portability
regressions hide (MSVC silently resolves includes and conformance that clang rejects). Two of the
three Linux surfaces are **maintainer gates** (do we ship a tree that compiles everywhere and passes
on hardware); the third is an **end-user feature** we verify but do not gate with:

| Surface | Role | What it does | GPU? |
|---|---|---|---|
| **GitHub CI** (`build-pipeline.yml`) | maintainer gate | *compiles* the full tree under clang-21 + packaging gates | no (hosted, no GPU) |
| **WSL** (`linux-clang-debug` preset) | maintainer gate | compiles the full tree under clang **and runs the CUDA test suite on real hardware** — minus anything needing weights, see below | yes |
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

**Running the WSL gate.** Take the tree across with `git archive`, not rsync:

```bash
git archive --format=tar HEAD | wsl -d <distro> -- bash -c 'rm -rf ~/mila-<version> \
  && mkdir -p ~/mila-<version> && tar -x -C ~/mila-<version>'
```

That carries exactly the tracked tree — 54 MB, no `Data/` (342 GB), no venvs, no `Dev/` — and it
tests what a consumer actually clones. Then `cmake --preset linux-clang-debug`; the preset already
pins clang, `/usr/local/cuda`, `gcc-15` as the nvcc host and a job pool of 4.

**Run the test binaries directly. Do not use `ctest` here.** Measured at `0.20.0`: `MilaTests` plus
`ChatRichTextTests` run in **30 seconds**, against roughly **30 minutes** under `ctest`, which
starts one process per case. The devcontainer docs were moved off `ctest` for the same reason and
this leg never was.

```bash
cd out/build/linux-clang-debug/Mila/Tests && ./MilaTests && ./ChatRichTextTests
```

Running both is what makes that equivalent to `ctest`: `ctest` reports a higher number only because
it also runs the separate `ChatRichTextTests` target. On Windows, where the gap is 149 s against
596 s, the everyday `ctest` invocation above stays as written.

**What this gate cannot tell you.** A `git archive` tree has no weights and no tokenizers, so every
model-parity, tokenizer and footprint test self-skips — 75 of them at `0.20.0`, leaving 1900 + 33
passing. So a green WSL run proves **portability and component correctness on Linux, not token
parity and not footprint prediction**. Both of those are covered only by the Windows `x64-validate`
run, which is worth knowing because the footprint tests are what validate
`kCudaAllocationGranularityBytes` — the constant is therefore checked on one platform only.
Adding a Python binding to this leg needs `python3.14-dev` in the distro; without it pybind11 fails
configure, and `-DMILA_ENABLE_PYTHON_BINDINGS=OFF` is the way past, since the wheel container
already compiles the binding under Linux at step 1.

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

**Step 3 is the gate.** GitHub CI does not run on a `dev` push, and by design: it compiles on a
machine with no GPU, so it can tell you less than the build you just ran. Releases reach `master`
through a `dev -> master` PR (see **Branching**), and that PR's gate is the WSL build, not CI.

---

## The release surface

The website is the index of every way into Mila: four Get Started tabs (`#p-cpp`, `#p-python`,
`#p-docker`, `#p-clone`) and the `#evaluate` band. They end at the same place — a model answering on
the reader's own GPU — and differ only in what they consume to get there. **A publish is finished
when every one of those five names the version being released and still works**, which is more than
tagging: three of the five reach a registry Git cannot reach.

**This whole section is about a publish.** Two of the five paths — `#p-cpp` and `#p-clone` — consume
nothing but the git tag, which is why a patch tag reaches them the moment it is pushed and costs
nothing beyond the gate. The other three are the expense, and they are why a minor is a different
kind of act. The surface as a whole still names the last *publish*, so a patch tag changes nothing
on this page.

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

**Before step 1, review the dependency pins.** Run
`python scripts/dependencies/check_pins.py --check-upstream` and decide, per pin, whether it moves
this cycle. This comes first for the same reason wheel validation does: a bump changes what the
wheels and images contain, so taking one after validating them throws that validation away. The
weekly `Dependency pins` workflow reports the same thing between releases, but a production tag is
where the decision is actually owed — a pin left alone is a choice, not an oversight, once it has
been looked at. A bump that touches CUDA needs the **local** GPU suite; CI has no GPU and cannot
gate it. Whatever moves, move its `NOTICE.md` row in the same commit or the `notice-gate` CI job
fails.

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
2. **Release-prep commit on `dev`** — set `Version.txt` to the release version with the whole
   pre-release tail **dropped**: `0.21.0-dev+7` becomes `0.21.0`. A tag never carries a tail, so this
   is what lets step 5's drift check pass. Reconcile BACKLOG / ROADMAP in the same commit. `master`
   is the branch a visitor lands on, so a missed bump leaves the front page advertising the previous
   release for the whole next cycle — and a procedure that misreports the last release is worse than
   one that says nothing. Two commands and an audit cover it, below.
   **Bump the sixteen version sites with one command:**
   ```
   python scripts/release/version_sites.py --set 0.21.0
   python scripts/release/version_sites.py --check --expect
   ```
   Run `--expect` *after* `Version.txt` is set. Before that it reports a pass with "nothing to
   compare", because the sites correctly name the last release while `Version.txt` still names a dev
   snapshot — which is the normal state on every other day of the cycle.
   Those sixteen are the QuickStart `GIT_TAG`s a reader copies
   (`Mila/Samples/QuickStart/Cpp/CMakeLists.txt`, that sample's README including its `URL` archive
   line, `getting-started.md` §7), the website's five (`Web/layouts/index.html`: the C++ tab's
   `GIT_TAG` and its sample output, the `-devel` tag, and the `-runtime` tag in **both** `#evaluate`
   commands), `scripts/dockerhub/overview.md`'s three, `verify-image.sh`'s `MILA_IMAGE` default, the
   `README.md` status callout, and this document's "Last checkpoint tagged" line. Nothing derived
   them and nothing checked them, which cost a downstream consumer a checkout failure against an
   unreleased tag and once left the C++ tab pinned a release behind the output beside it. The
   `version-sites-gate` CI job now asserts they agree with each other on every commit; `--expect`
   is the stronger release-time assertion that they name *this* release. A site that moved makes the
   script abort rather than skip — fix the pattern, never the file.
   **The prose is separate, and is not bumped — it is rewritten.**
   `python scripts/release/version_sites.py --audit-prose` lists the sentences a release can
   falsify — the *Current Status* section, the README status callout, and the `pre-1.0` claims that
   go at 1.0. No pattern can decide what those should say instead. **The listed set is not the whole
   set:** at `0.20.0` the beta wording also sat in `SECURITY.md` and the feature-request template,
   which nothing had listed, so grep the tree for the phrasing being retired as well. This is a
   `dev` commit and publishes nothing; the site goes live at step 10, after the images it names.
   **Clear any "not published yet" copy the release makes false** — today the `#p-docker` panel
   carries a flag saying both tags are local, and `getting-started.md` and `README.md` each carry a
   note calling the slim runtime image "planned". If step 9 then fails, nothing false has reached a
   reader: the site is not dispatched until step 10.
   **There is no CHANGELOG to update.** The release's prose record is the GitHub Release body at
   step 11, written once from the commit range. The file was deleted at `0.20.0` — it duplicated
   that body, written by the same hand from the same commits at the same moment.
3. Open a `dev -> master` pull request. CI validates on the PR.
4. Merge to `master`.
5. **Drift check (by eye — this used to be an automated gate):** the tag you are about to
   create must be exactly `v` + the contents of `Version.txt`, e.g. a `Version.txt` of `0.21.0` ->
   tag `v0.21.0`. A tag that disagrees with `Version.txt` makes a semver consumer fetch a tree that
   reports a different version.
6. Tag `master` and push the tag. **Tagging `master` is the release** — CPM/FetchContent fetch
   this git tag directly, and GitHub auto-generates the source archives at it. Nothing else is
   required for the library to be consumable downstream.
7. **Post-tag smoke test — the one step that genuinely needs the tag,** because it clones from
   GitHub at it. Select the preset shown as **"x64 Release (CPM release-access gate)"**
   (CMake `name` `x64-release-cpm-gate`) and run:
   ```
   ctest --test-dir out/build/x64-release-cpm-gate -R packaging_cpm_consumer --output-on-failure
   ```
   **Pass the tag explicitly — `-DMILA_CPM_GIT_TAG=v0.21.0` — rather than relying on the
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
11. **Publish a GitHub Release.** This is the release's only prose record — since the CHANGELOG was
   deleted at `0.20.0`, nothing else says what shipped, so it is a required step rather than the
   optional flourish it used to be.
   ```
   gh release create v0.21.0 --notes-file release-notes.md
   ```
   **Not `--generate-notes`.** GitHub builds those notes from the pull requests merged between two
   tags; Mila lands all work as direct commits on `dev` and opens exactly one PR per release, so
   `beta.2` would have produced a one-line release for 48 commits of work. The substance lives only in
   the commit messages, so the body is **authored from the commit range** — which is what `beta.1`
   actually did, hand-written with only its trailing `Full Changelog` footer generated.
   **No `--prerelease`.** There are no pre-release tags to apply it to since the ladder was retired,
   so every release takes GitHub's "Latest release" badge, which is what you want. The flag survives
   only as something to *not* copy off an older release's command line. Or draft it in the
   **Releases** web UI for full hand-curation. No consumer resolves through it, but it is the only
   place the release is described, so it lands in the same sitting as the tag.
12. **Open the next version on `dev`** — bump `Version.txt` with the counter reset. Having tagged
   `v0.21.0`, `dev` becomes **`0.21.1-dev+1`**: the default is the next patch, because a tag needs
   no decision and a publish does. Its own `dev` commit, same sitting as the tag. Skipping it leaves
   the working tree reporting an already-shipped version — the failure mode this scheme exists to
   prevent — and it is what lifts the release-window hold below, so a deferred step 12 keeps `dev`
   closed. Never reopen on a shipped version.
   **Open on the patch even when the next minor's themes are already picked**, which after a publish
   they will be. `ROADMAP.md` names the goal; `Version.txt` names the next *tag*, and the next tag
   is whatever gets cut first — very likely a patch. Opening on the minor instead forces the version
   to move backwards the first time a fix needs to reach source consumers: `0.22.0-dev+7` would have
   to become `0.21.1` for the prep commit and then climb back. The only case that legitimately opens
   straight onto a minor is one where there is nothing yet to patch, which is true exactly once —
   at the release that opens a new numbering era.
   **After a publish, this is also where the next release's narrative opens.** Delete the shipped
   section from `ROADMAP.md` — the GitHub Release body is now the record of what it contained — then
   pick the next release's **themes out of [`Mila/Issues/Vnext.md`](Mila/Issues/Vnext.md)** and write
   the narrative around them. That order is deliberate: the themes come from what the work has
   actually accumulated into, not from what would be satisfying to announce. The rules for the pick,
   and the trap in it, are in [`Mila/Issues/README.md`](Mila/Issues/README.md).

**Steps 2 to 12 are one window, and `dev` stays closed to unrelated commits across it.** Four
steps build from `dev` or from the tag it produced — the wheels, both images, and the site — and
each one that lands after an unrelated commit is built from a tree the tag does not describe. This
is a **release-window hold**, temporary and tree-wide, and it lifts when step 12 commits.

### A patch tag runs steps 2 to 7 and 12, and nothing else

Set `Version.txt` to the patch version, open and merge the PR, check the drift by eye, tag, run the
CPM gate, then open the next patch on `dev`. Steps 1, 8, 9, 10 and 11 are the *publish*, and a patch
is not one.

Step 2 shrinks with them: no `--set`, no prose audit, no BACKLOG or ROADMAP reconcile.
`Version.txt` is the only tracked file a patch tag touches.

**No release-window hold applies.** The hold exists because four steps build artifacts from the
tree; a patch builds none, so there is nothing a later commit could invalidate. `dev` stays open
throughout.

The cross-platform gate still applies in full — **the WSL build is what a patch tag is actually
asserting**, since a source consumer compiles this tree and a portability break is precisely their
failure mode. The devcontainer check is a publish concern and is only owed when the patch touches
`Docker/` or the build system.

---

## Publishing the wheels

`mila-llm` on PyPI is a published release artifact, built by two scripts and uploaded by hand. The
wheel version comes from `Version.txt`: a working `0.21.0-dev+38` tree produces the `0.21.0.dev38`
snapshot, and the prep commit's stripped `0.21.0` produces the release `0.21.0`.

**This section is run twice, and the two runs differ only in which version they carry.** Steps 1-4
run at **release step 1**, on a `dev` snapshot, and validate. Steps 1, 2 and 5 run again at
**release step 8**, from the tagged tree, and publish. The binaries are the same; validating a
throwaway version is what keeps the release filename unburned.

**The CUDA toolkit and the architecture list are pinned in the tree, not taken from the machine.**
Both wheels and both images carry `80;86;89;90;120` — the supported set, because SM 8.0 is the
floor Mila's own kernels draw (`CudaLinearOp.ixx:661`, and both GQA flash prefill paths throw
below it). The five sites are the two wheel presets, `MILA_LIBRARY_CUDA_ARCHITECTURES` in
`Mila/CMakeLists.txt`, `ARCHITECTURES` in `publish-image.sh`, and `MILA_IMAGE_CUDA_ARCHITECTURES`
in `Dockerfile.runtime`; they are one list and move together. The library default was the sixth
entry out of step until `rc.1+27`, when Turing was dropped from it — a compile pass for hardware
the runtime refuses. The toolkit is declared the same way:
`$cudaVersion` in `scripts/pypi/build-wheel-windows.ps1` for Windows, the base image in
`Docker/Dockerfile.wheel` and `Docker/Dockerfile.runtime` for Linux — currently **13.3** across all
four. Before that the Windows wheel took whatever `CUDA_PATH` resolved to, so installing a toolkit
on the dev box silently changed a published wheel; 13.4 landed mid-cycle and would have split one
release across two toolchains. **Moving to a new toolkit is an edit to all of those together, and
never inside a release window** — `MILA_WHEEL_CUDA_VERSION` overrides the Windows one for a
maintainer on a different box, and whichever is used gets printed at the top of the run.

**What the declared toolkit is not.** It constrains the six published binaries and nothing else: a
FetchContent or clone consumer builds with their own CUDA. The user-facing docs
(`getting-started.md`, `README.md`, `CONTRIBUTING.md`, `Web/content/start.md`) name the same
version CI builds with, so a toolkit move edits them too. A wheel user needs **no toolkit at all** —
`pyproject.toml`'s `nvidia-*` dependencies supply the runtime, so what they need is a driver. The
minor version is therefore invisible downstream (CUDA minor version compatibility holds within 13),
and only the **major** is load-bearing, since those dependencies pin `>=13.0,<14.0`; the wheel
script asserts that and leaves the minor as a declaration. Do not reach for this pin to answer a
question about what hardware or software a user must have — that answer lives in
`getting-started.md` and in the architecture list, not here.

**The `>=13.0` floor is verified, not assumed, and the clean room cannot verify it.**
`verify_wheel_cleanroom.py` installs the wheel and lets pip resolve the newest satisfying
`nvidia-*` — 13.8.0.4 at the time of writing — so the floor is never the version under test. It was
checked directly against the `0.20.0b3` wheels, built on CUDA 13.3 and declaring the same
dependency floors as `pyproject.toml` does today; a rebuild that changes the CUDA symbol surface
invalidates the check rather than the conclusion:

- **Windows**, empirically. A clean 3.13 venv with `nvidia-cublas==13.0.0.19` and
  `nvidia-curand==10.4.0.35` imports, and `Llama-3.2-3B-Instruct-fp4` generates greedily
  token-for-token identically to the same venv at 13.8.0.4. The loaded DLLs were confirmed to be the
  site-packages copies, not the machine's toolkit — which is what `_register_cuda_libraries`
  preloading the wheel's copies buys.
- **Linux**, by symbol. All 49 CUDA symbols the extension leaves undefined are exported by the 13.0
  libraries, and all four version nodes it requires (`libcudart.so.13`, `libcublas.so.13`,
  `libcublasLt.so.13`, `libcurand.so.10` — NVIDIA versions by SONAME, stable across 13.x) are
  defined there. Not run end to end: the manylinux wheel is cp312/cp313 and the WSL distribution
  carries only 3.14.

The risk this retires is a wheel that is immutable once uploaded declaring a floor nothing had
loaded. Re-check it when the cuBLASLt surface grows — the thirteen `cublasLt*` entry points are the
part most likely to acquire a newer one. `nvidia-cuda-nvrtc` arrives transitively as
`nvidia-cublas`'s own dependency and needs no declaration here.

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
   the same glob, and that cannot be withdrawn. Expect the directory to hold the previous release's
   `.devN` wheels when you arrive here: each script clears only its own platform's, so a stale wheel
   survives whenever the interpreter list or the platform set has changed.
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
   scripts/dockerhub/publish-image.sh v0.21.0
   ```
   Build-only is the default and `--push` is a deliberate second decision, the same shape as the
   site workflow. `MILA_CLEAN_BUILD` is forced rather than inherited: `--no-cache` invalidates
   layers but leaves BuildKit cache mounts intact, and two wrong images have already been built
   from another tree's objects that way.
   The architecture list is `80;86;89;90;120`, fixed in the script — the supported set, which
   excludes Turing because both GQA flash prefill paths refuse it outright. `native` — the local
   scripts' default — is wrong twice here, since it does not resolve on a GPU-less builder and the
   image is pulled by hardware the builder never saw.
4. **Run the `#evaluate` sequence against the local `-runtime` image, on a GPU host:**
   ```bash
   MILA_IMAGE=toddthomson/mila-llm:0.21.0-runtime scripts/dockerhub/verify-image.sh
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
   scripts/dockerhub/publish-image.sh v0.21.0 --push
   ```
   It asks for the version string rather than a `y/n`, because a reflexive "y" is not a decision.
   **A pushed tag cannot be withdrawn, only superseded** — which is the one thing here that is
   gentler than PyPI, and the reason a re-publish over a bad image is a real remedy.
7. **Update the Overview page.** Its source is `scripts/dockerhub/overview.md`, and the short
   description is `scripts/dockerhub/description.txt`. Move the image tags in `overview.md` to the new
   version in the release commit, then paste both into the repository's settings on Docker Hub. Never
   edit the page in the browser without the file: the file is the record.

**No `latest` on a pre-release.** A bare `docker run toddthomson/mila-llm` resolves to it, so
pointing it at a pre-release makes a snapshot the default for everyone who does not read the tag
list. It tracks the newest release's `-runtime`. This is not an open decision, and
`publish-image.sh` **enforces** it rather than stating it: the tag is added — as an alias of
`<version>-runtime`, not a third build — only when the version carries no `-dev`, `-alpha.`,
`-beta.` or `-rc.` suffix, and a pre-release run prints that it is leaving `latest` alone.
**`v0.20.0` is the release where `latest` first existed**, so it is now always in the push list.

The retired ladder's suffixes stay in that list because tags carrying them exist and are fetchable.
`-dev` joined it for a case that should never arise: a cycle never tags a dev build, and the script
already refuses a tag carrying `+build`, but a hand-typed `v0.21.0-dev` would clear every other gate
and publish a snapshot as the default image — which a push cannot take back.

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
   flag the release has now falsified. Every model named in the copy must be **installable by a
   stranger** — public on the Hub, not merely present in your own store. Today that is
   `gemma-4-12b-it-fp4`, `Llama-3.2-3B-Instruct-fp4` and `Qwen3.8-27B-fp4`. Check the last one
   anonymously: `curl -s -o /dev/null -w '%{http_code}' https://huggingface.co/api/models/mila-llm/<name>`
   returns 200 when it is public and 401 when it is private *or absent* — the Hub hides the
   difference, so a repo you can see while logged in tells you nothing. The site names Qwen in four
   places (hero lede, SEO description, models band, `/docs/`), and it was published separately from
   any release, so this is the check that stops the site advertising weights nobody can fetch.
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
- **Tag format:** `vX.Y.Z`. The CPM gate uses an explicit `GIT_TAG` (not CPM's `@version`
  shorthand, which mishandles a pre-release suffix — which the retired ladder's tags carry).
- **Testing an older tag mid-development:** the CPM gate derives its tag from `Version.txt` when
  `MILA_CPM_GIT_TAG` is empty, but you can point it at any already-pushed tag:
  `-DMILA_CPM_GIT_TAG=v0.20.0-beta.3` (also `-DMILA_CPM_GITHUB_REPOSITORY=<owner/repo>`). Note
  that an explicit value **persists in the cache**, so clear it (`-DMILA_CPM_GIT_TAG=`) to go back
  to the derived one.
- **Stale CPM cache:** the gate keeps a source cache across runs for speed, now under
  `.../Mila/Tests/Packaging/cpm-cache/<tag>`, with its build work directory keyed on the tag as
  well. Both are per-tag deliberately: sharing them is half of how a gate once validated the
  previous release and reported success. If a re-run misbehaves after a failed attempt, delete that
  tag's folder to force a clean fetch. **A passing run is only evidence for the tag it names** —
  the configure line and the run's first message both print it.
