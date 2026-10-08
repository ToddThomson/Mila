# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined after ninety days**, and anything user-reported is a pointer to its
GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

## MIS sends Llama a different prompt from Meta's chat template

`Mila/Adaptors/Inference/Server/src/mila_llm_server/prompt.py` `_build_llama_prompt`, `protocols/openai/chat.py`
`parse_chat_request` @ `539396e`

Found 2026-10-07 building `Tools/Evaluation`. A Chat Completions request with no system message gets
"You are a helpful assistant." as its system turn; Meta's template writes a "Cutting Knowledge Date / Today Date"
header there instead. Llama's template is still written out in MIS, where Gemma's and Qwen's render through the
runtime. The evaluation tool sends token ids to `/v1/completions` and so does not see it; a score taken through
`/v1/chat/completions` would. 2026-10-08: `run_inspect.py` takes Hub benchmark scores through that route, and
inspect-ai sends one user message with no system message, so every such score carries MIS's added system turn.
The route also defaults `temperature` to 0.6 when a request omits it (inspect-ai sends 0).

## MIS reports every buffered reply as `finish_reason: "stop"`

`Mila/Adaptors/Inference/Server/src/mila_llm_server/routes/factory.py` `_dispatch`, `model_worker.py`
`ModelWorker.generate` @ `539396e`

Found 2026-10-07 building `Tools/Evaluation`. The binding's `generate` returns why it stopped (`stop`,
`length`, `context_limit`, `cancelled`); the worker discards it and the route hard-codes `"stop"`, so a reply cut
at `max_tokens` is indistinguishable from a finished one.

## MIS streamed replies ignore `stop`

`Mila/Adaptors/Inference/Server/src/mila_llm_server/routes/factory.py` `_stream` @ `539396e`

Found 2026-10-07 adding `stop` to the buffered Completions and Chat Completions paths for `Tools/Evaluation`.
The streaming path still emits past a stop sequence; the MIS README says so.

## `routes/chat.py` and `routes/completions.py` in MIS are never registered

`Mila/Adaptors/Inference/Server/src/mila_llm_server/routes/chat.py`, `routes/completions.py` @ `539396e`

Found 2026-10-07 reading the Chat Completions path. `app.py` registers only `routes.factory` and `routes.health`;
nothing imports these two, and `chat.py` carries its own request schema and prompt assembly.

## The Qwen 3.8 2.82-bit model card says "residency"

`Mila/Tools/ExportArtifact/ModelCards/Qwen3.8-27B-cb2-3/README.md:53` @ `539396e`

Found 2026-10-07 reading the cards' quality sections for the evaluation discussion. "The cost of the smaller
residency" -- a term `CLAUDE.md` lists as one only Mila uses. The section's numbers are also against Mila's own FP4
build, not the upstream model.

## Qwen 4 27B is expected around November 2026 and Mila has no Qwen 4 chassis

`Mila/Specifications/Qwen4.md` @ 539396e

Compared Qwen3.8-Flash-Next, the Qwen 4 architecture preview, against the Qwen 3.8 chassis and wrote the
spec. Its section 9 phases 0 to 2 (the tiny reference, the converter skeleton, grouped RmsNorm, dilated
CausalConv1d, the DeltaNet gate activation, and the gated residual, n-gram, PLE and QSA indexer components)
need no 27B checkpoint. 2026-10-08: Phases 0 and 1 are built and their CPU gates passed; Phase 2 has
`GatedResidual` and `NgramEmbedding` on CPU, gated against the tiny reference.
