# Releasing Mila

How Mila is versioned, validated, and tagged. The key thing to internalize: there are
**two distinct validation moments**, and only one of them involves a tag.

---

## Versioning

`Version.txt` at the repo root is the single source of truth (SemVer: `X.Y.Z` or
`X.Y.Z-PRERELEASE`, e.g. `0.13.46-alpha.5`). It feeds `project(VERSION ...)` (the numeric
triple) and the prerelease label separately; see `cmake/MilaVersion.cmake`.

`Version.txt` is bumped **before committing** — every commit carries the version it
introduces. Because the version bump is part of the same commit you later tag, the tag
`vX.Y.Z-PRERELEASE` always points at a tree whose `Version.txt` matches it, and a consumer
fetching that tag gets a Mila that reports that exact version.

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
