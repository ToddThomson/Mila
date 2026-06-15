# Releasing & Work-Tracking

How Mila is versioned, planned, validated, and tagged. The [ROADMAP](ROADMAP.md) shows *where* Mila
is going; this document is the home for the *mechanics* — the version scheme, how the roadmap and its
milestones map to GitHub, label conventions, and the release procedure — kept out of the ROADMAP so
that stays a clean, public-facing narrative.

Two things to internalize up front: the version scheme carries a **stage** (the codebase's maturity,
not a milestone label) and a ticking **build** counter held in semver build metadata (next section),
and there are **two distinct validation moments**, only one of which involves a tag (further down).

---

## Versioning

Mila uses a repeating **release-cycle** model: `MAJOR.MINOR.PATCH-stage.X+build` (e.g.
`0.20.0-alpha.6+56`).

- **minor** — feature-set era.
- **patch** — part of the target release (usually `.0`).
- **stage** — the codebase's maturity: `alpha.X -> beta.X -> rc.X ->` unsuffixed stable. `X` is the
  stage **checkpoint ordinal** — it ticks each time a checkpoint is tagged within a stage, and is
  **decoupled from milestones** (it does not name or count them). Milestones are named by theme and
  namespaced by their planned release (see below), so a stage number never appears in a milestone
  title.
- **build** — a running per-commit counter carried as **semver build metadata** (after `+`). It ticks
  every commit as provenance and is **ignored for version precedence** by the spec.

Each feature set opens a new minor and runs its own ladder; features never land inside a hardening
ladder — a stabilizing release takes only patch-level fixes. Mila is pre-1.0, so any release may
carry breaking changes: `0.20.0` "production" means validated and polished, **not** API-frozen. An
API-stability promise is a separate, deliberate `1.0.0` decision, intentionally deferred.

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

**The `0.13 -> 0.20` jump.** The minor jumps to mark the production tier, and the jump lands now:
the current pre-releases rebase onto the `0.20.0` target. This stays forward in semver
(`0.20.0-… > 0.13.46-…`, minor compared first), keeping the timeline monotonic past already-published
tags like `v0.13.46-alpha.5` — which is why the target is `0.20.0` and not `0.13.0` (the latter would
sort *below* what is already released).

| Stage | Meaning | Example |
|---|---|---|
| `alpha.X` | features still landing; unstable | `0.20.0-alpha.6+56` (now) |
| `beta.X` | feature-frozen; hardening only | `0.20.0-beta.1+N` |
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

## The roadmap: releases, milestones, Future Directions

The ROADMAP shows **two production releases at a time** — the one in flight (top) and the one after
(**vNext**) — plus a non-tracked **Future Directions** tail.

- **Current release** — a pinned version, a *Committed* Release Date, and an eventual git tag.
  Reached through one or more **milestones**.
- **vNext** — identified by **theme, not version**; a target *range*, no tag. Its version, date, and
  tag crystallize when it promotes to Current.
- **Future Directions** — uncommitted vision; no milestone, no date.

A **milestone** is a step inside a release box. The milestones of a box share its version and are
distinguished by **theme** (a short descriptive name), **not** by stage or number. Stage is a
property of the codebase's maturity, not a milestone label: several milestones can land within the
same stage, and the `alpha -> beta` transition is a maturity judgment over the whole box (it opens
with Production Hardening), not the closing of any one milestone. Milestones are therefore namespaced
by their planned **release**, never by stage — which is also how a recurring theme (e.g. API
Documentation, Production Hardening) stays unique across releases.

---

## Milestones <-> GitHub Milestones

Every milestone maps **1:1 to a GitHub Milestone**. The GitHub Milestone title uses the form
**`Release v<version> - <theme>`** — release version (the namespace) plus a short theme, **no
stage** — so it is readable, self-locating in GitHub's flat namespace, and a recurring theme stays
unique by release:

- `Release v0.20.0 - Consolidation`
- `Release v0.20.0 - API Documentation` (a recurring theme; next cycle is `Release v0.21.0 - API
  Documentation` — the **release** disambiguates, never the stage)
- `Release v0.20.0 - Production Hardening` — the **single** hardening milestone; `beta.X` and `rc.X`
  are *tags* cut from it as it converges, **not** separate milestones (only the checkpoint ordinal
  and the tag iteration move; the milestone does not split)
- `vNext - Qwen 3` — the exception: no version yet, so no `Release v…` prefix; it is renamed to the
  full form when it promotes to Current

User issues (bugs, feature requests) are filed against the milestone they belong to.

---

## Dates and progress

**Milestones are dateless.** With no date, **task completion is the progress metric**: each
milestone's `- [ ]` checklist in the ROADMAP is its GitHub Issue set (one box ~ one Issue), so
GitHub's milestone progress bar (closed / total) shows how far along it is. The milestone is done
when the boxes are checked.

**Releases carry a Release Date**, following a 3-tier precision rule (GitHub's date field drives
"overdue" styling, so only populate it when you will be held to it):

| Tier | Release Date | GitHub date fields | Applies to |
|---|---|---|---|
| **Committed** | exact date | milestone fields stay empty; the date is the release/tag target | the Current release |
| **Target** | a range in prose ("H2 2026") | empty | vNext |
| **Unscheduled** | "Unscheduled" | none | Future Directions |

(If a GitHub-visible date is ever wanted, mirror the Committed Release Date onto the *terminal*
milestone's due date, since its completion is the release. Default is to leave milestone date fields
empty and let the progress bar carry the signal.)

---

## Roadmap lifecycle and promotion

Status is encoded by **position** (Current is at the top of the ROADMAP) and **GitHub open/closed**
— there is no "Current/Planned" label. *Open* means on-the-board-and-unshipped; future milestones
are created **open** up front so issues can be filed against them before work starts.

When the Current release ships, four things happen together (one event, kept mutually consistent in
the same commit — the same rule that binds ROADMAP / BACKLOG / CHANGELOG):

1. its prose moves to [CHANGELOG.md](CHANGELOG.md);
2. its box is **deleted** from the ROADMAP;
3. its GitHub Milestones **close**;
4. **vNext promotes to Current** — acquiring a version number, a Committed Release Date, and a tag
   target — and a new vNext is drawn from Future Directions.

The ROADMAP therefore always shows exactly two release boxes plus Future Directions; it never
sprawls.

---

## Labels

Labels carry what milestones cannot:

- `release:<version>` — *(optional)* unions the milestones of a multi-milestone release box (e.g.
  `release:0.20` over its alpha + beta milestones) into one filter. Only earns its keep when a box
  spans more than one milestone; vNext (single milestone, no version) gets none until it grows one.
- `future` — the GitHub home for Future-Directions issues that have no milestone.
- `bug` / `feature` / `good first issue` and `area:*` — orthogonal type/area classification.

**Discipline:** never add a label that merely restates an issue's Milestone field — that is the
double-bookkeeping the work-tracking model exists to avoid.

---

## The two validation moments

| | Pre-commit | Post-tag (release smoke test) |
|---|---|---|
| Preset | `x64-validate` | `x64-release-cpm-gate` |
| What it checks | unit tests + find_package + FetchContent gates | a **pushed tag** is fetchable + buildable via CPM |
| Consumes | the **local working tree** | the **GitHub remote** at the tag |
| Tag / `Version.txt` dependency | **none** | needs the matching tag already pushed |
| Network | no | yes (git clone of the repo) |

**Pre-commit has no coupling to `Version.txt` or any tag** — bump the version before or
after running `x64-validate`, it makes no difference. There is nothing to sequence.

**The CPM gate is inherently post-push.** The tag for the current version does not exist
until you push it, so this gate can only be run *after* a release is tagged. It is a
smoke test, not a gate you clear before committing.

---

## Everyday commit (on `dev`)

1. Make your change.
2. Bump `Version.txt` to the version this commit introduces.
3. Configure + build the **`x64-validate`** preset, then `ctest` it:
   ```
   ctest --test-dir out/build/x64-validate --output-on-failure
   ```
   Expect: unit tests + `packaging_find_package_consumer` + `packaging_fetchcontent_consumer` green.
4. Commit and push to `dev`.

`dev` is the CI-gated trunk during alpha (it is the GitHub default branch); `master` tracks
tagged releases.

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
   Use `--prerelease` for any `-alpha`/`-beta` tag; drop it for a final release. Or draft it in
   the **Releases** web UI for full hand-curation. Nothing downstream depends on this, so do it
   on your own schedule.
6. **Post-tag smoke test:** configure the **`x64-release-cpm-gate`** preset and run:
   ```
   ctest --test-dir out/build/x64-release-cpm-gate -R packaging_cpm_consumer --output-on-failure
   ```
   This git-clones Mila from GitHub at the tag and builds a consumer against it via CPM,
   proving the release is actually consumable downstream. The gate's tag defaults to the
   current `Version.txt`, so at this moment it lines up with the tag you just pushed.

---

## Notes

- **Never commit directly to `master`.** It changes *only* via a `dev -> master` release PR.
  A stray direct edit to `master` is what caused the README merge conflict in the first release
  — `master` and `dev` diverged and had to be hand-reconciled. Treat `master` as release-only.
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
- **At beta:** the GitHub default branch switches from `dev` to `master` so the front door
  is the released artifact. Until then, releases are infrequent and the audience targets `dev`.
