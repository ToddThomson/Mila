# Web — Issues

The website is a separate project from the library, with its own cadence and its own publish. It
does not appear in Mila's `ROADMAP.md`, so its work cannot be admitted to `BACKLOG.md` — an item
there has to name a release success criterion, and the site has none to name. That is why this
directory exists rather than a Website bucket one level up.

`Backlog.md` holds open work; `Untriaged.md` is capture. One of Mila's category files is ruled out
here rather than merely absent, so nobody creates it on the "a file appears when it has its first
entry" rule:

- **No `Future.md`.** The site has no release boundary to be on the far side of, so work is either
  open or done.

There is no contributor file here either, but that is no longer a local rule: Mila deleted its own,
and the reasoning this page reached first — a website item that happens to be small is just a small
item in `Backlog.md` — is now the general one in
[`Mila/Issues/README.md`](../../Mila/Issues/README.md). Nobody outside this repository works on the
website, so the outbound direction is empty here in any case.

`Declined.md` is still available if a site idea is worth recording as considered-and-rejected.

## Format

Entries use the same three-part shape as [`Mila/Issues/`](../../Mila/Issues/README.md) — heading,
metadata line, body — so moving an item between the two costs nothing. The metadata line is tags in
`Backlog.md`, and `<anchor> @ <sha>` in `Untriaged.md`.

## Tags

Mila's [`Tags.md`](../../Mila/Issues/Tags.md) set is wrong here — nearly everything on a website is
`docs`, which tags nothing. These four follow the tree instead:

| Tag | Means |
|---|---|
| `content` | Prose under `Web/content/` — pages and blog posts. |
| `layout` | `Web/layouts/`: the home page bands, templates, nav. |
| `brand` | Marks, icons and other assets under `Web/static/`. |
| `publish` | The Pages workflow, and anything coupling the site to a release tag. |

Mila's **constraint** and **sequencing** tags carry over unchanged where they apply — `blocked`,
`gate`, `next`.

## The release coupling

One thing here is not independent: the home page hardcodes the Mila version in three places, so a
release tag and a site publish have to name the same version. That item is why "separate project"
means separate cadence, not separate from the release.
