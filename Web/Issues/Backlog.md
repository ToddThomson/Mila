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

## `[ ]` The home page hardcodes the version at five sites, and nothing derives them

`layout` · `publish`

The C++ tab's `GIT_TAG` and its sample output, the `-devel` tag, and the `-runtime` tag in both
Evaluating commands. Every later release breaks those commands until the copy is bumped with it, so
the site and the release tag must ship naming the same version. They have already drifted from each
other once — the tab pinned `beta.2` against `beta.3` output for a whole cycle.

Held today by procedure rather than by derivation: `RELEASING.md` step 2 lists all five, plus
`scripts/dockerhub/verify-image.sh`'s `MILA_IMAGE` default, as one edit.
`Web/layouts/index.html` — `#p-cpp` steps 1 and 3, `#p-docker`, `#evaluate`.

## `[ ]` The Docker tab never says how to get back into the container

`content`

That image is a configured environment where the reader edits `~/myapp`, so leaving it is not the
end of the session — but the tab stops at `docker run` and nothing names `docker start -ai`. The
reader's second visit begins with a container they cannot find.

**It must not gain `--rm`**, which is what makes this the opposite gap from the Evaluating band's.
`Web/layouts/index.html`, `#p-docker`

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

## `[ ]` The models band calls Gemma 4 the chat default, and there is no default

`layout`

Nothing in the chat harness carries a compiled-in default model: a search of
`Mila/Adaptors/Chat/Src/` for `default_model`, `defaultModel` and `DEFAULT_MODEL` finds nothing, and
`CLAUDE.md:14` says so outright. Confirmed against the published runtime image — bare `chat` refuses
and names the flag that fixes it. So a reader who installs Gemma, types `chat`, and expects it to be
picked up gets a refusal instead.

Deleting the two words is the whole fix; the rest of the entry already carries the weight.
`Web/layouts/index.html`, the models band.

## `[ ]` The precision column offers deployments a reader cannot reach

`layout`

`FP4 / FP8` on Llama 3.1 8B and `BF16 / FP8 / FP4` on Llama 3.2 3B sit under a heading about models
you can run, so they read as precisions to choose between. `applyRequestedQuantization` refuses to
reload pre-quantized weights as anything else, so every published model is FP4 at runtime and the
others are converter capabilities.

Mila's `BACKLOG.md` records this against `README.md:163,165` and does not know the website repeats
it — widen that entry, or close both together. `Web/layouts/index.html`, the models band.

## `[ ]` Two orphaned brand assets still carry the old Achilles mark

`brand`

`icon.png` at the repo root and `Web/static/achilles.png`, neither referenced by any page, template,
README or the Doxyfile — which sets no `PROJECT_LOGO` at all.

Delete rather than replace: `Brand/generate.py` emits the current mark into `Web/static/` only, so a
root copy would be a second source to drift.
