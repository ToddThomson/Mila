# Contributor

Good-first-issue shaped: bounded, self-contained, and reachable by a strong developer without a
tour of the codebase. This is the **outbound** queue — these are what get mirrored *to* GitHub
Issues with a label when someone asks how to help.

An item here is not a commitment. Nothing in this file blocks a release; if it did, it would be in
[`BACKLOG.md`](../../BACKLOG.md) instead. Triage flow and categories are in [README.md](README.md).

---

- **Llama 3.2 1B/3B weight tying** — the aliasing plumbing shipped; add `tie_word_embeddings_` +
  post-load aliasing + `getMemoryStats` correction to `LlamaTransformer`.
  See `Specifications/WeightTying.md` §6.
- **Llama-lineage CPU ops** (`RmsNormOp`, `SwigluOp`, `RopeOp`, `TokenEmbeddingOp`,
  `CrossEntropyOp`) in `OperationTraits.Cpu.ixx` — demand-driven; absence is zero-cost on the GPU
  path.
