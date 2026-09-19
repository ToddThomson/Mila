# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined at the production release tag**, and anything user-reported is a
pointer to its GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

## The wheel's CUDA runtime dependencies admit 13.0 for a binary built against 13.3

`Mila/Bindings/Package/pyproject.toml:47` / `:55` (`nvidia-cublas`, `nvidia-cuda-runtime`
`>=13.0,<14.0`) @ b5593e32

Noticed while moving the docs to state one CUDA version, the one CI builds with.
`RELEASING.md` (*What the declared toolkit is not*) treats the minor as invisible downstream on
driver minor-version compatibility; that covers the driver, not whether a library built against
13.3 headers loads against the 13.0 runtime libraries pip may resolve. Unverified either way.
