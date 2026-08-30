# Web — Backlog

Open website work. Status lives in the checkbox on the heading — `[ ]` open, `[~]` in progress — and
never in the prose. **Done means deleted**, in the same commit as the work; the commit is the record.

Format, tags and the split from `Untriaged.md` are in [README.md](README.md).

---

## `[~]` Reconcile `Web/content/start.md` with the Get Started band

`content` · `layout`

The four tabs landed as `#qs` in `Web/layouts/index.html`, but the nav and the home-page "Get
started" box still point at `/start/` with older clone-and-build content, so the site has two
getting-started surfaces.

That page's §3 is retired in every sentence — conversion as the path, "no separate quantized
checkpoint to manage", and "Llama and Gemma are gated", now backwards. This reconcile owns §3.

## `[ ]` The home page hardcodes `0.20.0-beta.3` in three places

`layout` · `publish`

Two image tags — the Docker panel and the Evaluating band — plus the FetchContent pin. Every later
release breaks those commands until the copy is updated with it, so the site and the release tag
must ship naming the same version.

The C++ tab compounds it by pinning `v0.20.0-beta.2` while its sample output reads `0.20.0-beta.3`.
`Web/layouts/index.html` — `#p-docker`, `#evaluate`, `#p-cpp` steps 1 and 3.

## `[ ]` The Evaluating band's commands leave a stopped container behind on every run

`content`

No `--rm`, so a QA afternoon accumulated four and `docker image rm` then failed with a conflict the
user has no context for. Nothing is lost — the model lives in the named volume.

**The devel tab must NOT get `--rm`**: that image is a configured environment where the reader edits
`~/myapp`, and its gap is the opposite — nothing says how to re-enter it. `Web/layouts/index.html`,
`#evaluate`

## `[ ]` `Web/content/docs.md:28` states "quantization has no checkpoint format"

`content`

True when written, false now — every published model is a quantized checkpoint. The surrounding
point, that the type chooses the reduced-precision path, still stands and should survive the
correction.

## `[ ]` The site links GitHub and nothing else

`layout` · `publish`

No HuggingFace, no PyPI, so the primary marketing site does not point at the model store or the
package. [[project_four_channel_roles]]

## `[~]` Mila is a library, never a "runtime"

`content`

The noun names an engine you hand a model to, so it argues with "no hidden execution engine" in the
same breath. Three user-facing sites remain: `Web/content/docs.md:38`,
`blog/implementing-gemma-4.md:4`, `blog/gemma-4-docker-openai-api.md:4`.

Not a sweep — "at runtime", "runtime dispatch" and the two places naming what Mila is *not* are
correct. Whether `Mila/Src`'s own design name changes is a separate open call, and it belongs to the
library rather than here.

## `[ ]` A blog post ships with no `discussion:` line

`content`

`Web/content/blog/longer-context-fixed-the-crash.md`.

## `[ ]` Two orphaned brand assets still carry the old Achilles mark

`brand`

`icon.png` at the repo root and `Web/static/achilles.png`, neither referenced by any page, template,
README or the Doxyfile — which sets no `PROJECT_LOGO` at all.

Delete rather than replace: `Brand/generate.py` emits the current mark into `Web/static/` only, so a
root copy would be a second source to drift.
