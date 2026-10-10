# Specifications

The design documents behind Mila, and the notebooks that record how their decisions were reached. This
index is the list: every file here has one row, added in the commit that adds the file.

---

## Roles

Every document plays one of four roles. The role decides how it is edited and whether it can be cited as
deciding anything.

| Role | What it is | How it changes |
|---|---|---|
| **Design** | The design of record for one area. Where one exists it decides, and a decision that contradicts it is either wrong or an edit to it (CLAUDE.md, *Key Specifications*). | Edited in place; always current. |
| **Notebook** | Dated evidence: each phase's gate written before its run, the result, the forced failure, what was corrected. True as of its dates, never current. | Append-only. |
| **Plan** | Work with gates and open decisions, spanning several areas. | Drains as the work lands. |
| **Direction** | Positioning and the release arc. | Rarely, by discussion. |

## Rules

1. **A decision made in a notebook or a plan is carried to the design that owns the area, in the same
   change.** The notebook keeps the evidence and the plan keeps a pointer. A design that learns of a decision
   only second-hand goes stale: `MixtureOfExperts.md` still planned FP4 and CUTLASS for the 26B-A4B after
   `ModelFamilyParity.md` §9 item 14 had decided that a model runs in the format its producer trained.
2. **Notebooks live in `Notebooks/`**, named for what they record. A notebook is cited for evidence, never
   as the authority on a design.
3. **A document that no longer decides anything is archived in this index, not moved.** Its row leaves its
   area's table for the *Archive* heading at the end, naming what superseded it, and the file stays where
   it is, so every citation of it still resolves. A document is archived after a reading shows its area is
   gone or superseded, never on its date alone. A shipped design is not finished while live code still
   follows it: it stays the design of record for that code.
4. **A move is its own commit**, with every reference updated and a search showing no old name left.

Files move into `Notebooks/` one at a time, each after a reading; until a file moves, its row says where it
belongs.

---

## Index

The area tables list the documents that still decide or record something live; the *Archive* heading at
the end lists those that no longer do. *Notes* records only what has been checked. "Read before relying"
means a status line or target is out of date and the rest has not been re-read.

### Direction

| Document | Role | Decides or records | Notes |
|---|---|---|---|
| [Direction.md](Direction.md) | Direction | Mila's trait, layering and release plan from v0.21.0 | Supersedes `MilaProductFamily.md` after the v0.20 tag |
| [MilaProductFamily.md](MilaProductFamily.md) | Direction | The product definition v0.20 shipped under | Superseded from v0.21.0; `Direction.md` carries its Agentic design by reference, so it is archived only once nothing cites it as live |

### Library architecture

| Document | Role | Decides or records | Notes |
|---|---|---|---|
| [OperationDispatch.md](OperationDispatch.md) | Design | Compile-time operation dispatch through `OperationTraits` | |
| [Compute.md](Compute.md) | Design | Mixed-precision CUDA compute | Read before relying. Read 2026-09-29: its registrar contract (§4) and checklist (§12) are superseded by `OperationDispatch.md`, which removed the registrars, and it names SwiGLU as the reference op where CLAUDE.md names `Linear`. The rest still describes patterns in the tree — per-op `*.Dispatch.ixx` partitions, `fp32_compute_type` — but was not checked against the kernels, so it is not archived. Cited by no file |
| [FfnAndMoE.md](FfnAndMoE.md) | Design | The feed-forward component family, and the MoE decision (grouped GEMM over stacked experts) it constrains | The MoE design itself is `MixtureOfExperts.md` |
| [MixtureOfExperts.md](MixtureOfExperts.md) | Design | The MoE path: router, stacked expert bank, execution, kernels per weight format, expert residency | Family-neutral; its one model is the Gemma 4 26B-A4B (`Gemma.md` §10) |
| [Observability.md](Observability.md) | Design | Looking into a running network | |
| [Notebooks/Workspaces.md](Notebooks/Workspaces.md) | Notebook | Survey of the five shared-buffer mechanisms, 2026-08-23 | Its own status: "not a design of record"; decisions owned by `MemoryFootprint.md` |
| [Notebooks/TransformerApiReadiness.md](Notebooks/TransformerApiReadiness.md) | Notebook | Pre-v0.20 review of `GemmaTransformer`'s public API | Findings only; item 6 answered by `Observability.md`; `LanguageModelNetwork.ixx` cites items 2, 7 and 8 for its shape |
| [Testing.md](Testing.md) | Design | Component test methodology | |
| [Testing.Tensors.md](Testing.Tensors.md) | Plan | Coverage worklist for the Tensor suite | Core `Tensor.ixx` done; the wider `Tensors/` tree open. Ten test files cite it for the value-type archetype. Drains when every row is covered |

### Models

| Document | Role | Decides or records | Notes |
|---|---|---|---|
| [Gemma.md](Gemma.md) | Design | The Gemma 4 chassis; §10 the Gemma 4 26B-A4B — configuration, topology, Q4_0 format, memory, the bar | |
| [GemmaChatProtocol.md](GemmaChatProtocol.md) | Design | Gemma 4's turn and tool protocol, and Chat's implementation of it | |
| [Gemma4Modality.md](Gemma4Modality.md) | Design | Image and audio input for Gemma 4: the 12B's embedders, the 26B-A4B's vision tower, bidirectional image spans, and the applications' attach path | Draft; the mask is built, the rest is not |
| [Gemma4Mtp.md](Gemma4Mtp.md) | Design | Speculative decoding of Gemma 4 12B and 26B-A4B with Google's draft models | Draft; stage 1 built and measured, the loop not built. Supersedes `SpeculativeDecoding.md` for Gemma |
| [Notebooks/Gemma4MoE.md](Notebooks/Gemma4MoE.md) | Notebook | Gemma 4 26B-A4B, phases 1-9 | Decisions owned by `Gemma.md` §10 and `MixtureOfExperts.md`; carries the map from the section numbers it cites to where they moved. Keeps its name because code cites it by name |
| [Notebooks/Gemma4InferenceReview.md](Notebooks/Gemma4InferenceReview.md) | Notebook | Review of the Gemma 4 12B generation path, 2026-07-02 | Its recommendations went to `BACKLOG.md`; five specs and a test cite its sections as evidence |
| [Qwen3.8.md](Qwen3.8.md) | Design and notebook | The Qwen 3.8 27B chassis, and its measurements | To split. Read before relying: its opening calls it a research track outside v0.20, and it shipped in v0.20 |
| [Qwen4.md](Qwen4.md) | Design | The Qwen 4 architecture (`qwen4_exp`): gated residual, sparse attention, n-gram embedding, and what a Qwen 4 27B port builds | Draft; no code. Written from the Qwen3.8-Flash-Next preview before any 27B checkpoint exists. Section 9 is its phased plan, phases 0-2 buildable now; section 10 is closed by the 27B's own config |
| [ModelFamilyParity.md](ModelFamilyParity.md) | Plan | The parity matrix, each family's stages, and their open decisions | Decisions carry to each area's design (rule 1). Candidate to split: the matrix is standing, the plan drains |
| [ModelHandle.md](ModelHandle.md) | Design | How a model's name becomes the object that runs it | Draft; decisions in its section 10 |

### Attention and the KV cache

| Document | Role | Decides or records | Notes |
|---|---|---|---|
| [GqaFlashAttention.md](GqaFlashAttention.md) | Design | Fused prefill attention | Read before relying: its status says flash is off, and since `0.21.0-dev+13` it is every family's prefill path |
| [GqaDecodeAttention.md](GqaDecodeAttention.md) | Design | Decode attention on the tensor cores | |
| [GqaAttentionExtent.md](GqaAttentionExtent.md) | Design | Attended length against physical stride in unbounded prefill | Interim by its own status, to be subsumed by flash prefill, which now runs everywhere. Archive candidate, after a reading |
| [GqaMemory.md](GqaMemory.md) | Design | Reducing GQA's buffers (Llama 3.2 3B era) | Read before relying: Phase 1 recorded complete, Phases 2-3 not recorded either way |
| [SlidingWindowKvCache.md](SlidingWindowKvCache.md) | Design | The bounded KV ring for sliding-window layers | |
| [PromptCaching.md](PromptCaching.md) | Design | KV prefix reuse across calls | |
| [RopeInAttention.md](RopeInAttention.md) | Design | RoPE fused into attention, and what the cache would store | Draft; no code. Two measurements (its section 5) decide whether it becomes a design |

### Memory, deployment and devices

| Document | Role | Decides or records | Notes |
|---|---|---|---|
| [MemoryFootprint.md](MemoryFootprint.md) | Design | Predicting what a load allocates, without allocating it | |
| [Deployment.md](Deployment.md) | Design | The deployment planner: context length, prefill chunk, devices | |
| [LayerSplit.md](LayerSplit.md) | Design | One model's blocks across several devices | Draft; `Deployment.md` Phase 5, after v0.21.0 |
| [WeightTying.md](WeightTying.md) | Design | One table for a tied embedding and output head | |

### Quantization

| Document | Role | Decides or records | Notes |
|---|---|---|---|
| [Quantization.md](Quantization.md) | Design | Weight policies (FP8, FP4, codebook, Q4_0) and KV-cache compression | |
| [Fp8ActivationPrefill.md](Fp8ActivationPrefill.md) | Design | The FP8-activation prefill GEMM for FP4 weights | |

### Generation

| Document | Role | Decides or records | Notes |
|---|---|---|---|
| [TokenSampling.md](TokenSampling.md) | Design | Sampling on the device | |
| [DecodeGraph.md](DecodeGraph.md) | Design | Replaying a decode step as one recorded graph, and the contract every decode op keeps | |
| [SpeculativeDecoding.md](SpeculativeDecoding.md) | Design | Draft-verify decoding | Draft; no code. For Gemma superseded by `Gemma4Mtp.md` (2026-10-04): Google's drafter has no cache of its own and is a feature built or not, not a `TDrafter` axis, so its §3.2, §3.3 and §6 do not describe it. Archive candidate once nothing else plans on it |

### Measurement

| Document | Role | Decides or records | Notes |
|---|---|---|---|
| [ContextProfile.md](ContextProfile.md) | Design | Measuring a model configuration at each context length it holds: fit, loss by band, recall, instruction retention, tool calls, turn cost | Draft; no code |
| [ModelEval.md](ModelEval.md) | Design | Standard task benchmarks run paired against a reference engine: arms, controls, statistics, the noise floor, and the model card's quality claim | Draft; the tools for Phases 1 to 4 built (`Tools/Evaluation`: IFEval, GSM8K, RULER, BFCL; transformers, MIS and llama.cpp arms), none yet run on real weights |

### Distribution and serialization

| Document | Role | Decides or records | Notes |
|---|---|---|---|
| [ModelDistribution.md](ModelDistribution.md) | Design | Naming, packaging, the manifest and the local store | |
| [ModelSerialization.md](ModelSerialization.md) | Design | Checkpoints and the flat weights path | |
| [HttpClient.md](HttpClient.md) | Design | The HTTP transport behind fetching | Built: `Distribution/HttpClient.ixx`, `HttpTransport.ixx` and `CurlHttpTransport.ixx` implement it and cite it. Its status line still says "proposed" |

### Applications

| Document | Role | Decides or records | Notes |
|---|---|---|---|
| [ChatConfiguration.md](ChatConfiguration.md) | Design | Chat's layered configuration | |
| [PythonBinding.md](PythonBinding.md) | Design | The Python binding as a product, and its samples | |
| [Notebooks/MilaCommandLineReview.md](Notebooks/MilaCommandLineReview.md) | Notebook | Review of the `mila` command line utility, 2026-10-10 | Findings, and one decision: a published utility is for the end user only. What else it encompasses is unresolved, and no finding is admitted until it is |
| [MilaISCodexAgent.md](MilaISCodexAgent.md) | Design | The inference server's bridge from OpenAI Responses tool schemas to Llama's tool format | Describes code that ships: `protocols/openai/tool_bridge.py`, imported by `responses.py`. Written 2026-05-15 against Llama 3.2 3B; not re-read against today's bridge |

### Release

| Document | Role | Decides or records | Notes |
|---|---|---|---|
| [ReleaseAutomation.md](ReleaseAutomation.md) | Design | Where each check runs | Draft |

---

## Archive

Documents that no longer decide anything. Each stays where it is, so its citations resolve; read it as
history, and read what superseded it for how things work now.

| Document | Was | Superseded by | Archived |
|---|---|---|---|
| [ToolCalling.md](ToolCalling.md) | Version 0.1 plan for tool calling in the inference server, against Llama 3.1 8B FP8 at alpha.4 | What shipped instead of it. Its central decision — parse Llama's calls in C++ through a pybind11 `parse_tool_call`, gated by `MILA_TOOL_CALLING_ENABLED` — was not built: the server parses Llama's calls in `protocols/openai/tool_bridge.py` (`MilaISCodexAgent.md`), and Gemma's and Qwen's through the library's grammar bindings, `gemma_parse_tool_call` and `qwen_parse_tool_call` (`GemmaChatProtocol.md`; `ModelFamilyParity.md` 3.3). Neither the binding nor the flag exists | 2026-09-29, after a reading |
