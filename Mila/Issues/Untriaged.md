# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined at the production release tag**, and anything user-reported is a
pointer to its GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

## A fix to the published site sits in the repo until someone dispatches the workflow

`.github/workflows/publish-site.yml` (`on: workflow_dispatch:`) @ `c60c100a`

Found reading Google Search Console for `mila.toddt.me`: 141 pages reported **"Excluded by 'noindex'
tag"**, against 1 indexed page and 14 others crawled-or-discovered-but-not-indexed.

The `noindex` post-processing step was removed from the workflow at `0589673f` on 2026-08-13. The
workflow only runs on dispatch, and the next dispatch was 2026-09-11 — so the live site served
`noindex` across the whole `/api/` tree for 29 days after the repository had stopped saying it.
Verified on the live site 2026-09-22: no `robots` meta and no `X-Robots-Tag` on `/api/index.html`,
`/api/annotated.html`, `/api/files.html` or `/api/classes.html`. GSC validation is running.

Nothing reconciles the site source in the repository against what is actually deployed, and the
workflow's own verification step reads one file (`build/site/api/index.html`) out of the tree it
assembles.

