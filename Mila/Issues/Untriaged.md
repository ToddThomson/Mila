# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined at the release tag**, and anything user-reported is a pointer to
its GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

## The packaging fixtures are an unread detector for source-tree pollution

`Mila/Tests/Packaging/{fetchcontent,cpm}_consumer/Mila/Adaptors/Inference/Server/mila.cp313-win_amd64.pyd`
@ `a87e8315`

Two 16 MB fossils dated 2026-07-21, left by the third binding destination that
`Mila/Bindings/CMakeLists.txt:112-118` records as removed. They landed there because the copy was
source-relative and a subproject build makes the consumer's root `CMAKE_SOURCE_DIR` — the same
defect as the open `tokenize` and wheel-VERSION items. Nothing looks in these directories, so the
evidence sat for six weeks. Found widening the FetchContent gate.
