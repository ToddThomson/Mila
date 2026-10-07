# Web — Untriaged

Captured, not yet judged. Writing here needs no judgement — finding something mid-task leaves
seconds, and a format that asks for more is a format that goes unused. Facts may grow; judgement
may not.

Entries are **deleted unexamined** at each site publish. Format, tags and the split from
`Backlog.md` are in [README.md](README.md); entries here carry an anchor where `Backlog.md` carries
tags.

---

## The C++ tab loads a model from a store it never says how to fill

`Web/layouts/index.html:107` — `#p-cpp` step 2

`ModelStore{}.locate( "gemma-4-12b-it-fp4" )`, with the comment "the store is the only source --
loading never downloads" beside it, and no step anywhere in the panel that installs one. Every other
path carries an explicit install step of its own: Python `pull`s in step 2, both Docker panels and
Clone each run an install verb. So the tab whose reader has the longest build ahead of them is the
one that reaches a runtime refusal, and the refusal it reaches names REPL commands
(`Mila/Adaptors/Chat/Src/Chat.ModelCatalog.ixx:485`) that a C++ consumer has no session to type
into. Found mapping the five onboarding paths against RELEASING.md.

## The site's crawl signals lag the site

`Web/hugo.toml:13-17` @ `0.21.0-dev+39`

Moved from `Mila/Issues/Untriaged.md` at its 2026-10-07 triage. Verified: the `hugo.toml` comment still says the
`/api/` tree is marked noindex by a workflow step that was removed at `0589673f`, and `enableRobotsTXT = false` with
no layout supplying a sitemap line. From a cloud review, unverified: the home page's sitemap `lastmod` reads
2026-08-15 because git info is off and the workflows check out shallow; two blog posts link duplicate GitHub
Discussions (`flash-decoding-mqa-cuda.md:8`, `sponsor.md:23`); old `toddthomson.github.io/Mila/` Doxygen URLs have no
redirect.
