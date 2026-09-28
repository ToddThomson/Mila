# Model Family Parity

What each supported model family can do in Mila, measured against the others and against what the base
model can do upstream. The matrix is the definition of a finished family: every gap is closed, or recorded
as not applicable with the architectural reason.

Written 2026-09-25 at `0.21.0-dev+7`, from a survey of the code at that commit. The rules in section 1
are Todd's (2026-09-25). **This is a pass over the existing families, not a ROADMAP theme** (Todd,
2026-09-25). It runs family by family, Gemma 4 first. Starting a family turns its gaps into
`BACKLOG.md` entries under that family's existing theme -- after checking `Vnext.md`, `Future.md` and
`Untriaged.md` for entries that already cover them -- and the family is finished when those entries are
gone. A cell here changes in the commit that closes it.

Section 8, the implementation plan, was added 2026-09-26 at `0.21.0-dev+8`, from a review of the Gemma chassis
and its mixture-of-experts path; its code anchors were checked at that commit.

---

## 1. The Rules

**Families share capabilities.** The families were built one after another -- Llama 3.2 and 3.1 first,
then Gemma 4 12B, then Qwen 3.8 27B with two quantization policies, then the Gemma 4 mixture of experts --
and each stage added capabilities, and in places removed them, without carrying them back to the families
already built. The order a family was built in is never a reason for it to lack a capability. A difference
is legitimate only when the architecture rules it out, and then the reason is written in section 5.

**The reference card is 16 GB.** Development until 2026-08 ran on a 12 GB RTX 4070, and some upstream
capabilities were left out or deferred to fit it. The 16 GB RTX 5060 Ti is now the card Mila is designed
for: **a capability the base model has upstream is in scope unless it cannot fit 16 GB at the published
precision**, and "it did not fit 12 GB" no longer justifies leaving it out. 12 GB remains a supported card
for the models that fit it -- it is a tier, not the ceiling.

**Every automatic value is a validated value.** Since `Deployment.md` Phase 3 the planner may choose any
context length up to a model's trained maximum. A context the planner can choose is a context whose quality
has been measured; where it has not, the family is not finished.

---

## 2. The Models In Scope

| Model | Family | Published packages | Card it needs |
|---|---|---|---|
| Llama 3.2 1B Instruct | Llama | none -- converts and runs, unpublished (`Future.md`, "Publish `Llama-3.2-1B-Instruct-fp4` as the evaluation model") | 8 GB (about 1 GB of FP4 weights) |
| Llama 3.2 3B Instruct | Llama | `Llama-3.2-3B-Instruct-fp4` | 12 GB |
| Llama 3.1 8B Instruct | Llama | `Llama-3.1-8B-Instruct-fp4` | 12 GB |
| Gemma 4 12B | Gemma | `gemma-4-12b-it-fp4` | 12 GB |
| Gemma 4 26B-A4B | Gemma (mixture of experts) | none -- in `Mila/Src`, gated, unpublished | 16 GB, and not yet at context 8192 (3.6) |
| Qwen 3.8 27B | Qwen | `Qwen3.8-27B-fp4`; `Qwen3.8-27B-cb2-3` fitted, unpublished | FP4: 16 GB. cb2-3: 12 GB |

---

## 3. Parity Inside Mila

Survey of `0.21.0-dev+7`. **Y** has it, **--** missing, **n/a** ruled out by the architecture (section 5).

### 3.1 Generation

| Capability | Llama | Gemma | Qwen | Anchor |
|---|---|---|---|---|
| Sampling on the device: top-k, top-p, seedable | -- | Y | Y | Llama samples on the host, applies temperature and top-k only (so `top_p` is ignored) and seeds from the clock, so `seedSampler` has no effect (`LlamaModel.ixx:391`) |
| Sampling overlapped with the next forward | -- | Y | Y | Llama synchronizes after every prefill and decode |
| Prompt-prefix reuse | -- | Y | n/a | Gemma `GemmaModel.ixx:412`; Qwen `supportsPromptPrefixReuse`, `QwenModel.ixx:338` |
| Correct at the trained context length | partial | Y | Y | Llama's frequency scaling applied since 8.4 L2, and matched to HuggingFace past 8192 on the 3.2 1B; quality across the planner's range is L3 |
| Flash-attention prefill | Y | Y | Y | every BF16 build of every family, with no context threshold (8.4, L4) |

### 3.2 Weights and memory

| Capability | Llama | Gemma | Qwen | Anchor |
|---|---|---|---|---|
| FP4, FP8 and BF16 weights; quantize-on-load | Y | Y | Y | `QuantizationDispatch.ixx` |
| Embedding table and output head quantized with the body | -- | Y | Y | Llama's are always BF16 (`Llama.ixx:86`, `:88`): about 1.4 GiB on 3.1 8B |
| A tied checkpoint's embedding and head held once | -- | Y | n/a | Llama 3.2 1B and 3B tie them upstream (3.1 8B does not); the converter copies the table into `lm_head.weight` (`convert_weights.py:226`) and the transformer loads both, a second BF16 copy of 0.73 GiB on the 3B and 0.49 GiB on the 1B, in the package and on the device. Gemma shares one table. Qwen 3.8 27B does not tie |
| Sub-4-bit codebook weights | -- | -- | Y | the fitting and packing tools are Qwen's (`Tools/Quantization/pack_qwen.py`, `qwen_plan.py`); the dispatch is Qwen's own (`dispatchQwenWeightPlan`) |
| KV-cache compression | -- | -- | -- | `PerChannelKvFp8<>` specified, not built (`QuantizationDispatch.ixx:103`) |
| Sequence log-likelihood at the network layer, for the quality harness | Y | Y | Y | `sequenceLogLikelihood` on `GemmaTransformer` and `QwenTransformer`, reached through `Tests/Common/LogLikelihoodHarness.h`; what an in-library perplexity gate measures. Until `0.21.0-dev+9` it was Qwen's `scoreTokens`, public on `QwenModel` with a head width on every family's deployment request; both left the public surface (8.2, G1) |
| Deployment planning, exact footprint | Y | Y | Y | `Deployment.md` Phases 1 to 4 |
| Per-block activations pooled across layers | Y | Y | Y | one block workspace installed on every block: Gemma's and Qwen's `allocateBlockWorkspace`, Llama's `makeLlamaBlockWorkspace` since 8.4 L4, where the 32 per-block sets had cost 2,762 MiB from chunk 512 to 1024 on the 3.1 8B |

### 3.3 Grammar

| Capability | Llama | Gemma | Qwen | Anchor |
|---|---|---|---|---|
| Chat template and tool grammar in `Mila/Src` | -- | Y | Y | Llama's is in Chat (`Chat.MessageFormatter.ixx`, `Chat.ToolCallParser.ixx`) and MIS (`prompt.py`), and the two declare tools differently (`ModelHandle.md` 1.1) |
| One renderer per family | -- | -- | Y | Gemma: Chat renders its own (`Chat.ixx:1359`) beside `Gemma::formatPrompt` |
| Grammar projected to Python | -- | Y | Y | `gemma_*` and `qwen_*` in `Mila_py.cpp`; no `llama_*` |
| Reasoning channel | n/a | Y | Y | |
| Tool calling in Chat and MIS | Y | Y | Y | |

### 3.4 Validation

| Capability | Llama | Gemma | Qwen | Anchor |
|---|---|---|---|---|
| Token-for-token agreement with HuggingFace, in the suite | Y | Y | Y | `GemmaModel.Parity.Cuda.cpp`, `QwenModel.Parity.Cuda.cpp`; Llama's `Llama.HuggingFaceReference.Cuda.cpp` (tiny model, in the suite) and `Llama.LogLikelihood.Cuda.cpp` (the real 3.2 1B, opt-in) |
| Throughput harness | -- | Y | Y | `GemmaModel.Rates.Cuda.cpp`; Qwen's `DISABLED_PrefillRate` / `DISABLED_DecodeRate*` |
| Quality measured across the planner's range | -- | -- | partial | Qwen to 16K (`Qwen3.8.md` §8) while the planner chooses up to 64512 for cb2-3 on 16 GB; Gemma unmeasured above 131072 against a header of 262144; Llama as the RoPE row above |

### 3.5 Applications

Not a model property -- streaming is the display's (`ModelHandle.md` 3.2) -- but held to the same rule.

| Capability | Llama | Gemma | Qwen |
|---|---|---|---|
| Chat streams the reply as it is generated | -- | Y | -- |

### 3.6 Gemma 4 26B-A4B Against The 12B

The two share `GemmaTransformer` and `GemmaModel`, so every row above that the transformer decides holds for
both, and scoring (3.2) lands in both at once. The cells where the 26B-A4B differs:

| Capability | 12B | 26B-A4B | Anchor |
|---|---|---|---|
| Fits the 16 GB card at context 8192 | Y | -- | predicted 14.88 GiB against 14.80 GiB free, 83.8 MiB over (`Gemma4MoE.md` Phase 8). The remaining cause is the routed buffers, allocated per layer rather than pooled: each layer's `MixtureOfExperts` output (`MixtureOfExperts.ixx:388`) and FP32 gated scratch (`CudaMoeOp.ixx:126`), about 0.49 GiB at chunk 512 |
| Throughput measured | Y | -- | no rate for the 26B is recorded anywhere; its CUDA kernels are the Phase 6 correctness baseline and were never tuned |
| Token-for-token agreement with HuggingFace, in the suite | Y | partial | eight greedy tokens at FP4 (`GemmaModel.MixtureOfExperts.Fp4.Cuda.cpp`) and a layer-streamed BF16 hidden-state gate; the BF16 model fits neither card, so no whole-model BF16 run exists |

---

## 4. Upstream Capabilities Mila Does Not Yet Have

What the base model can do in its reference implementation, against Mila. Under section 1's hardware rule
each row is in scope unless it cannot fit 16 GB; the last column says whether that has been priced.

| Model | Upstream capability | In Mila | Fits 16 GB | Tracked |
|---|---|---|---|---|
| Gemma 4 12B | Image and audio input, encoder-free: patches and waveforms projected into the decoder, no vision tower | -- the converter skips `embed_vision` and `embed_audio` | not priced; the embedders are small beside the decoder | BACKLOG, Gemma 4 Complete (two entries) |
| Gemma 4 12B | Multi-token prediction: a dedicated draft model for speculative decoding | -- | not priced | BACKLOG (measurement only); `SpeculativeDecoding.md` draft; loop in `Vnext.md` |
| Gemma 4 12B | 262144-token context | planner allows it; Chat caps 131072; quality unmeasured above 131072 | yes, at FP4 | `ModelHandle.md` 10.3 |
| Gemma 4 12B | Quantization-aware 4-bit checkpoint (int4, group 32) | -- | yes | BACKLOG (measurement only) |
| Gemma 4 26B-A4B | Mixture of experts | Y, in `Mila/Src`, unpublished | at FP4, once the routed buffers are pooled (3.6) | BACKLOG, "Announce the Gemma 4 26B-A4B mixture of experts" |
| Qwen 3.8 27B | Vision tower (27 layers, width 1152) and multimodal positions (mrope) | -- out of scope for the first chassis (`Qwen3.8.md` §1) | not priced; tight beside the FP4 build (13.2 GB of weights on device) | nowhere |
| Qwen 3.8 27B | Multi-token prediction head, one layer (~0.45 B) | -- both converters skip `mtp.*` | not priced | nowhere |
| Qwen 3.8 27B | 262144-token context, 1M by YaRN | the planner allows 262144; quality measured to 16K | only with KV-cache compression, and not at FP4 | BACKLOG, "Qwen's perplexity gate has only been run to 16K"; KV compression entry |
| Qwen 3.8 | Smaller dense members | -- | yes | BACKLOG, "The smaller dense Qwen members were never built" |
| Llama 3.1 / 3.2 | 128K context by Llama 3 RoPE scaling | Y since 8.4 L2; quality past 10240 unmeasured (L3) | 8B: only with KV compression | 8.4 |
| Llama 3.1 | Built-in tools (the `ipython` role, `<|python_tag|>` ... `<|eom_id|>`) | custom JSON tools only (`Chat.MessageFormatter.ixx:143`) | yes | nowhere |

"Nowhere" is a finding: a capability the base model has that no Mila record names.

---

## 5. Differences The Architecture Decides

The only legitimate entries of **n/a** above, each with its reason:

- **Qwen: no prompt-prefix reuse.** 48 of its 64 layers carry a recurrent state, a lossy summary of every
  position, which cannot be rolled back to a prefix (`QwenModel.ixx:9`).
- **Llama: no reasoning channel.** Llama 3.x was not trained with one.
- **Gemma: sliding-window attention and its bounded KV ring.** A property of the architecture; the other
  families' attention is full.
- **Gemma 4 26B-A4B: a mixture of experts.** A model, not a capability the others lack.

A proposed n/a is argued here before the matrix may say it.

---

## 6. What The Survey Shows

- **Llama is furthest behind**, because it was first: device sampling, prefix reuse, flash prefill, table
  quantization, the grammar and a parity test were all built after it and not carried back. Most of its
  gaps are the other families' finished work applied to a third family, not new design.
- **Some settings are accepted and silently ignored.** The scoring head width is honoured by one family in
  three -- and on inspection it is a measurement knob that should never have been on the public request --
  and Llama ignores `top_p`. That is the failure `QuantizationDispatch.ixx` exists to prevent -- a
  request that reads as honoured and is not -- and it is worse than a refusal.
- **Quality validation stops short of what the planner now chooses** for all three families (3.4). This is
  the gap `Deployment.md` Phase 3 opened, and it is the one a user meets first.
- **Four upstream capabilities are recorded nowhere** (section 4): Qwen's vision and MTP head, Llama's
  built-in tools, and until this document the 16 GB rule itself.
- **The 26B-A4B is recorded as needing only a package, and it needs more** (3.6). Its BACKLOG entry says
  "a published package and a capability row, not more implementation", while the model does not fit the
  reference card at context 8192 and its speed has never been measured.

---

## 7. Keeping It True

- The matrix changes in the same commit as the work that changes a cell.
- **A new capability gets a row with every family's cell filled** -- Y, --, or n/a argued in section 5 --
  in the commit that adds it to the first family. That is the step whose absence produced this document.
- **A new family gets a column with every row filled** before it is announced.
- An upstream capability the base model gains, or one found missing, gets a section 4 row whether or not
  work is planned.

---

## 8. Implementation Plan

The order each family's pass runs in, what each stage depends on, and the gate that ends it. Status is not
kept here: it lives in `BACKLOG.md`, and a stage names the entries it closes by their headings. Section 9
holds the decisions a stage waits on.

### 8.1 Two Constraints Every Pass Obeys

**A published package moves once per family per release.** Several stages change what a package holds --
tensor names, the tables' format, weights the converter used to skip -- and each republish costs a re-export,
a card update and every user's re-fetch. A family's package-changing work is collected and republished
together, at the end of its pass. The rename has a cost in the meantime: once the tree renames a tensor, the
tree refuses the package that carries the old name, so between the rename and the republish a `dev` build
loads only a locally re-exported package (about two minutes with `ExportArtifact`).

**Shared work is written once, in the family that needs it first.** Gemma's log-likelihood is the second
implementation of the window loop and the first with a softcap, so it lifts the host reduction into a helper
the third family calls, rather than making a third copy.

**Every pass audits the family's public API for calls that exist only for internal use.** A test, a
diagnostic, a profiling run or a measurement is not a reason for a `Mila/Src` public member. For each public
member of the family's model, network and config types, and for each setting they add to the shared
deployment request, name the end user who calls it and what goes wrong for them without it. Where the only
callers are tests or tools, the member moves to the consumer: a harness in `Tests/` or `Tools/` that builds
the network itself. Or it is deleted, when a general mechanism such as `LanguageModel::observe` already
serves it. The same audit checks names: a name must say what it produces in the domain's own term, so that a
reader who knows the domain can tell what it returns without opening its documentation. What the audit finds
is removed in the pass, not recorded for later.

**Every module a stage touches exports one type** (CLAUDE.md, *C++ Module Conventions*). A type other
modules must name is exported from its own file; a type only its own file uses is not exported at all. A
touched file that predates the rule (`Vnext.md`, "Forty-six module files export more than one type") is
brought under it in the same stage, so the backlog of old files drains as the passes reach them. For each
extra type in a touched file, first ask whether other modules need to name it. A type that stops needing to
be exported is often the better fix than a new file.

### 8.2 Gemma 4

Seven stages. G1 to G3 need no architecture change and can start at once; G4 is the one structural change;
G5 and G6 finish the 26B-A4B; G7 is modality, sized by its own two open questions.

**G1 -- Sequence log-likelihood, in the quality harness.** *Closes:* "Gemma cannot score a text, so its
quality cannot be measured". With it go "The Gemma parity script compares two different precisions" and "A
parity script tells the reader to diff against a debug flag that no longer exists", which are the same
validation tooling.

The log-likelihood a model assigns to a given text is a measurement, and no user surface calls it. Adaptors,
Bindings, Samples and Tools hold no reference to either `scoreTokens` or `withLanguageModelHeadPositions`. So
Gemma gains it where the measurement lives, and Qwen's public copy is taken back in the same stage (8.1's
audit):

- **Off the public surface.** The head width leaves `LanguageModelConfig`, `DeploymentRequest` and
  `DeploymentPlan`, and `QwenModel::scoreTokens` is deleted. No model class gains a replacement.
- **At the network layer, renamed.** `LanguageModelNetwork::scoreTokens` becomes `sequenceLogLikelihood`,
  after the type it returns. The head width becomes `withLogLikelihoodWindow` on `QwenConfig` and
  `GemmaConfig`: the positions the output head evaluates per pass during a log-likelihood run. Generation
  never sets it, so at its default of 1 every existing graph and footprint is unchanged.
- **Gemma's implementation.** `GemmaTransformer::sequenceLogLikelihood` runs Qwen's window loop
  (`Qwen.ixx:280`), with the head and final norm built at the window in `onBuilding` and `getRequiredMemory`
  (`Gemma.ixx:552`, `:365`). The dense and routed models share it. The final-logit softcap, which Gemma
  otherwise applies at the sampler, is applied to each row before the log-softmax. The host reduction
  (`accumulateWindowLogProbability`, `Qwen.ixx:892`) moves out of `QwenTransformer` into a shared function
  taking the softcap, 0 meaning none.
- **The harness builds the network itself.** It reads the weights header, builds the network at the
  context under measurement and calls `loadParameters`, as the tiny-model tests already construct
  `GemmaTransformer` directly. It chooses its context explicitly and needs no planner. Because no
  `GemmaModel` is involved, the model's prompt-prefix history cannot go stale behind a log-likelihood run.
  That trap exists only if the method sits on the model.
- **Two touched modules export more than one type, and are split here.** `LanguageModelNetwork.ixx`
  exports `SequenceLogLikelihood` beside the network (`:37`, `:79`); the struct moves to its own
  `SequenceLogLikelihood.ixx`, which also takes the shared host reduction. `LanguageModelConfig.ixx`
  exports `WeightQuantization`, `KvCacheCompression` and `LanguageModelConfig` (`:91`, `:234`, `:251`).
  Both enums are used across the library, not only by that class, so they are two new files whichever way
  the open ruling in `Vnext.md` goes, each carrying its free functions.
- **Cost of the wider window, measured rather than assumed.** At a window above 1, the FP8 tied head leaves
  the decode matvec for the staged prefill path, whose BF16 staging is striped at 256 MiB against a
  1.875 GiB table. The harness's footprint has to show what that reserves.
- *Gate, written before the run:*
  - The tiny routed checkpoint's HuggingFace capture gains HuggingFace's per-position log-likelihood, and
    Mila's total at FP32 matches it within 1e-4. This is the only gate that sees a missing softcap: the
    softcap is monotonic, so argmax is blind to it, and two widths would agree on the same wrong number.
  - Window 1 and window 64 on the 12B at FP4 agree within a relative perplexity difference of 2e-3 (set at
    1e-3; see section 9, item 13).
  - Each position's argmax equals the token greedy generation chose, on the tiny FP32 model; on the 12B the
    agreement is reported, not asserted (section 9, item 13).
  - Footprint agreement stays exact at window 64, and every footprint is unchanged at window 1.
  - Qwen's `DISABLED_QualityGateAcrossContextLengths` reproduces its recorded numbers, bit for bit.
  - *The gate can fail:* with the softcap dropped, the HuggingFace gate must fail.
- *Result, 2026-09-26* (`x64-claude-verify` Release, RTX 5060 Ti pinned by UUID unless stated):
  - HuggingFace gate: Mila's total is within 3.1e-7 to 4.9e-7 of HuggingFace's -72.0917 at windows 1, 3 and 8,
    against 1e-4. Without the softcap it misses by 4.0e-3, so the gate can fail. Argmax against greedy, footprint
    at a widened window and the clamp all pass. The tiny capture's weights regenerated byte-identical.
  - Qwen: `DISABLED_CorpusPerplexity` through the harness reads 2.0167110553096492 nats per token, the baseline
    recorded before the change to all 17 digits, at the same chunk (1024) and positions (16,240).
  - Full suite unpinned, both cards visible: 2015 pass, 0 fail, 4 skipped (three need the 16 GB card and see the
    4070 as device 0; the long-standing Swiglu BF16 backward). ChatRichTextTests 33 pass; Chat answers the
    tools-weather prompt through a real Gemma tool call.
  - **12B, both gates fail as written.** Window 1 against 64 over 16,195 wikitext positions: 7.4960777 against
    7.4950178 nats, a relative perplexity difference of 1.06e-3 against the 1e-3 bound (Qwen measured 2.7e-4).
    Argmax equals the greedy token at 31 of 32 positions; the one miss is 0.75 logits apart. Both are decisions
    for section 9, item 13, and both were then diagnosed (`Gemma.LogLikelihood.Cuda.cpp`, DISABLED diagnostics
    A to D, and `Tools/Converters/Gemma/gemma_4_BF16/head_paths_reference.py`):
  - **The window gap is two roundings in the head, and one is avoidable.** Against logits computed exactly in
    FP64 from `temb.wte` over 64 positions, window 1 (the decode matvec) is identical to the exact logits rounded
    once to BF16 -- mean |error| 0.015564 both -- which is the best a head that writes BF16 logits can do.
    Window 64 (the staged GEMM) is 0.017216, and a model of its one extra step -- each `fp8 * scale` rounded
    to BF16 before the GEMM -- reproduces it to four digits (0.017215). Relative to exact perplexity, window
    1 is -6.4e-4 and window 64 is +4.9e-4: opposite signs, so they differ by ~1.1e-3, the corpus measurement.
    The BF16 logits alone cost ~6e-4, because Gemma's raw logits reach magnitudes where one BF16 step is 0.25
    to 0.5.
  - **The argmax miss is the model's sensitivity to last-bit differences, not a defect in the log-likelihood.**
    A prefill is deterministic -- the same tokens give bit-identical rows, whatever ran before -- but the row at
    a position depends on how many rows the prefill had, and decode is different arithmetic again. On FP4
    weights the first difference is layer 5's global attention (7e-4 relative) and grows to 7.1e-2 by the last
    layer; on FP8 weights, whose prefill keeps BF16 activations, it starts at layer 0's `o_proj` (1.5e-4, the
    BF16 GEMM's row-count-dependent algorithm). HuggingFace at BF16 shows the effect too: the row at position 8
    moves by a mean 0.13 to 0.15 logits between a 9- and a 29-token prefill (eager and sdpa).
  - **Averaged over 63 positions** (a 64-token prefix against the same positions inside 128, relative L2 at layer
    47; `hf_fp8_layer_comparison.py`, diagnostic E), with HuggingFace given the weights Mila's prefill multiplies
    by: Mila FP8 2.1e-2, HuggingFace on those FP8 weights 2.0e-2, on Mila's FP4 weights 2.6e-2, on BF16 2.0e-2 --
    and **Mila FP4 7.0e-2**. The single-position figures read earlier (6.6e-2 for FP8) were noise, and the
    conclusion drawn from them -- that FP8 activations do not amplify -- was wrong. The extra sensitivity is
    Mila's FP4 prefill arithmetic, the W4A8 path, which quantizes activations to FP8 per token and re-expresses
    the FP4 weights in FP8 under one scale per tensor; not the FP4 weights, and not the model.
  - **Parity on identical weights**, the same measurement against HuggingFace rather than against itself: Mila FP8
    is within 5.6e-3 (layer 0) to 2.4e-2 (layer 47) -- the size of HuggingFace's own row-count dependence, a third
    of what FP8 quantization does to HuggingFace. Mila FP4 is 3.5e-2 at layer 0 and 0.44 at layer 47, more than
    FP4 quantization itself moves HuggingFace (0.37).
  - **Mila implements W4A8 exactly; W4A8 is what differs.** Layer 0's four FP4 projections, recomputed in FP64
    from the package's own FP4 bytes and Mila's captured inputs: Mila's output is within 2.5e-5 to 1.1e-4 of
    exact W4A8 as `Fp8ActivationPrefill.md` specifies it, and exact W4A8 is 2.0e-2 to 3.6e-2 from exact W4A16,
    the arithmetic decode uses. Two percent per projection, four projections a layer, 48 layers, is the FP4
    build's sensitivity, its distance from HuggingFace, and most of its prefill-against-decode gap.
  - **Perplexity, the same four segments (4,092 positions), mean nats:** HuggingFace BF16 6.388; FP8 weights:
    HuggingFace 6.523, Mila 6.507. FP4 weights with BF16 activations (HuggingFace) 7.580; W4A8 emulated in
    HuggingFace 7.336; Mila FP4 7.253. The FP8 arms agree. W4A8 scores *better* than W4A16 on the same weights:
    on raw wikitext this instruction-tuned model is confidently wrong, so rounding that flattens its
    distributions raises the probability of the tokens that occur. Raw wikitext therefore cannot judge a
    precision change on this model -- a direct input to section 9, item 10.
  - Consequence for G2: a perplexity measures prefill arithmetic at a given segment length and prefill chunk,
    and the protocol fixes both, as it fixes the window.
  - **The 12B's perplexity on raw wikitext is about 1,800, and that is the model, not the measurement.**
    HuggingFace's own BF16 run on the same 256 token ids reads 2,182: Gemma 4 12B *it* is instruction-tuned, and
    raw wikitext is far from its training format. The tokenizers agree id for id. Every segment must start with
    `<bos>`, which the tokenizer does not add; without it the same run read 24,986. This bears on G2's corpus
    (section 9, item 10).

**G2 -- Quality across the planner's range.** *Closes:* "Gemma's quality above 131072 tokens has never been
measured". *Needs:* G1, and a long-document corpus whose licence is verified at its source (section 9).

- Loss by context length on the RTX 5060 Ti, 8K to 262144, first on the published FP4 package (the result
  below) and then, as the gate's run, on the Q4_0 package built from Google's QAT weights (section 9, item 14).
  The result decides `ModelHandle.md` 10.3 -- Chat's 131072 cap is deleted, or the ceiling comes down to the
  last band the gate passes. The card holds all of it: the planner chooses 262144 for the published FP4
  package on the RTX 5060 Ti (`GemmaModel.from_store`, `auto`, 2026-09-26).
- *Corpus* (section 9, item 10): the PG-19 test split, whose longest books fill one 262144-token window on
  their own. `Data/Datasets/PG19/README.md` records the source and how to fetch it.
- *The text goes inside a model turn.* Gemma 4 *it* predicts running text only in its own format. On the same
  32,000 characters of a PG-19 book (diagnostic H, `DISABLED_BookFormat`, RTX 4070): 9.26 nats per token as
  bare text after `<bos>`, 9.61 with Gutenberg's 70-column wraps joined into paragraphs, and **3.79** with the
  joined text as the model's reply to "Continue this book." (0.88 nats per character, against 2.23). A first
  G2 run on bare text read 9.2 to 10.7 nats per token and gained nothing from context through 131072, so it
  measured nothing and was stopped. This is also why raw wikitext reads about 1,800 (G1 result).
- *Protocol.* One network built at 262144, window 64, prefill chunk as the library's rule chooses it. Each
  book that fills the window -- at least two, taken in the test split's order -- is `<bos>`, a user turn "Continue this book.", the model
  primer with thinking off, then the book with its wraps joined -- 262144 tokens in all, scored at prefixes of
  8192, 16384, ... 262144. The prompt is scored alone and taken off, so every figure covers the book's tokens
  only. A book takes 45 minutes. Scoring is causal and every prefix is a whole number of prefill chunks, so the positions a shorter prefix scores come
  out the same in the longer one, and the prefix of 2L less the prefix of L is the book's log-likelihood over
  positions L to 2L given everything before them. The top band, 131072 to 262144, is the question itself.
  The cumulative perplexity at each prefix is `Qwen3.8.md` §8's measurement -- one segment of that length,
  its ratio against the shortest -- and is reported beside the bands. Window 64 takes the FP8 head's staged
  path, a bias of about 1e-3 in perplexity (G1), identical at every length, so it cancels in every ratio;
  window 1 would add about 20 minutes a book in the head alone.
- *Gate, written before the run (section 9, item 15).* The first gate -- the band 131072 to 262144 within 1.05
  times the band below -- failed on book 30312 (1.67) for a reason it was not written to catch: the FP4 loss
  rises from 16K, not past 131072. It is replaced by three tests, on at least two books, every band and every
  book reported; the ratio of each band to the one below stays in the report and decides nothing.
  1. *The whole book predicts no worse than the sliding window alone,* in every band from 8K to 262144. Each
     band is scored twice on the same target tokens: with the whole book before it, as the protocol does, and
     fresh -- `<bos>`, the same turn and primer, then only the 1024 tokens before the band (Gemma's sliding
     window) as unscored context. The first must not exceed the second. This asks whether the global layers
     use the long context or damage it, which is the published FP4 package's failure (diagnostic I: 4.1 to 4.8
     with the book before it, 3.7 to 3.8 fresh), and it needs no reference build at any length. It assumes
     BF16 itself gains from context past 32K, which is unmeasured; a band that fails for a reason other than
     quantization still marks where the model stops using its context, which is where the cap belongs.
  2. *Cost over BF16 at most 0.05 nats per token in every band to 32K,* against HuggingFace on the BF16
     checkpoint (`hf_long_context_loss.py`), the only lengths where BF16 runs on this machine. QAT at Q4_0
     measured +0.03 there in HuggingFace. The margin covers Mila's own offset against HuggingFace, 0.04 over
     the first 8K on FP4, where the prefill rounds activations to FP8; the Q4_0 prefill keeps them at BF16.
  3. *Mila agrees with llama.cpp on Google's GGUF at every band to 262144* (`llama_cpp_long_context_loss.py`):
     no band past 32K differs by more than the largest difference over 0 to 32K plus 0.02 nats per token. The
     Q4_0 weights are identical (8.2, G2 result), the tied table is not (FP8 against Q6_K), and llama.cpp runs
     8-bit activations, so the two need not agree exactly; a difference that grows with length is Mila's own
     long-context arithmetic, which nothing independent has checked past 32K.
  All three pass on every book to 262144 and Chat's cap is deleted; otherwise the ceiling is the last band
  where all three pass.
- The QAT measurement (now answered, and the build admitted: "Gemma 4 12B at 4 bits predicts worse the longer
  the context..." in `BACKLOG.md`) uses the same corpus. Its metric is not perplexity: G1 showed raw-text
  perplexity cannot rank precisions on this model, since extra rounding scored better. It was planned as the KL
  divergence of each 4-bit build's next-token distribution from BF16's; it was answered by the per-token loss
  over BF16 on the same tokens (+0.03 against +0.69), a gap no choice of metric reverses.
- *Result so far, 2026-09-26* (`x64-claude-verify` Release, published FP4 weights; G2 on the RTX 5060 Ti and the
  diagnostics on the RTX 4070, each pinned by UUID; `Gemma.LogLikelihood.Cuda.cpp`, diagnostics H to J):
  - **Loss rises with context, from about 16K, and keeps rising through 262144.** Mean nats per book token by band,
    window 64, chunk 1024:

    | Book | 0-8K | 8K-16K | 16K-32K | 32K-64K | 64K-128K | 128K-256K |
    |---|---|---|---|---|---|---|
    | 30312 | 3.787 | 3.816 | 4.111 | 4.448 | 4.790 | 5.306 |
    | 3608 | 3.794 | 3.670 | 3.650 | 3.834 | 4.178 | -- |

    Book 30312's top band is 1.67 times the perplexity of the band below it, against the 1.05 bound, but the rise is
    not a failure that starts past 131072: perplexity roughly doubles from 8K to 128K before the top band adds to
    it. Book 3608 is flat to 32K and then rises the same way. Book 30312 took 2,694 s. Book 3608's 262144 prefix ran
    out of device memory at 71 minutes -- the 5060 Ti then showed 15 GB held with only `devenv.exe` listed on it, so
    another process shared the card; 262144 fits with little room to spare. The third book did not run.
  - **The text does not get harder** (diagnostic I, `DISABLED_LossAlongTheBook`). Book 30312's 8K stretches starting
    at 0, 8K, 16K, 32K, 64K and 96K, each alone in a fresh turn, read 3.784, 3.678, 3.788, 3.738, 3.739 and 3.766.
    The same text read with the book before it reads 4.1 to 4.8. **The model predicts the book worse with 64K of it
    in context than with none**, which a model using its context correctly does not do.
  - **One long reply is not the cause** (diagnostic I). The book's first 64K as four 16K model turns, each after a
    user "Continue.", read 3.793, 4.032, 4.163 and 4.314 -- a little below the single turn's rise, and still rising.
  - **The prefill attention kernel is not the cause** (diagnostic J, `DISABLED_DecodeAgainstPrefillAlongTheBook`).
    The same 257 targets scored by decode, one token at a time, and by prefix differences, at window 1: after
    4096 tokens decode 3.883 and prefill 3.925; after 32768, 3.904 and 3.933. The two paths agree to the same small
    margin at both depths (the W4A8 prefill's known gap, G1), so whatever grows with length is something they share:
    the weights, the RoPE tables, the KV cache, or the model itself.
  - **HuggingFace rises the same way: the rise is not Mila's.** `hf_long_context_loss.py` (beside the G1 script)
    scores the first 32768 tokens of book 30312's turn in HuggingFace, on the FP4 weights rebuilt as Mila quantizes
    them, in 1024-token chunks through its KV cache; its prompt tokenizes id for id as Mila's. By 8K band:
    3.830, 3.812, 4.024, 4.173 -- 4.099 over 16K-32K, where Mila reads 4.111, and 3.812 against Mila's 3.816 over
    8K-16K. Mila's lower 3.787 over the first 8K is its W4A8 prefill against HuggingFace's W4A16, the direction G1
    measured. On the FP4 weights, Gemma 4 12B *it* predicts a book worse the more of it it has read, in both
    implementations.
  - **BF16 does not rise: the FP4 weights cost the long context.** The same HuggingFace run on the BF16 checkpoint,
    by 8K band: 3.641, 3.339, 3.433, 3.487. The model gains from context after the first 8K and stays below that
    level through 32K. FP4 minus BF16, band by band: +0.19, +0.47, +0.59, +0.69 nats per token -- the cost of the
    FP4 weights grows with context, to about twice BF16's perplexity over 24K-32K. One book, one 32K window. This
    is a finding about the published FP4 package, not about Chat's 131072 cap, and it bears on the QAT
    measurement: a quantization-aware checkpoint is exactly what might keep what FP4 loses here.
  - **The attention projections carry the loss.** The same run with FP4 on one sublayer and BF16 on the other,
    the tied table at FP8 in both, as cost over BF16 by band: attention alone +0.11, +0.33, +0.43, +0.47;
    feed-forward alone +0.08, +0.10, +0.11, +0.13. The feed-forward cost is the flat price of 4-bit weights; the
    attention cost grows with context, and the two add to within 0.1 of the whole FP4 build's. An FP4 build whose
    attention projections stay FP8 is the candidate this points at -- measured next, not assumed. The published
    sensitivity analysis of FP4 by component (Cim, Topcu and Kandemir, arXiv 2603.08747, on Qwen2.5) ranks the
    feed-forward projections most sensitive and attention "substantially less", with no context-length axis. Over
    the first 8K the two here are close (+0.08 and +0.11); the ranking inverts only with length. A sensitivity
    ranking measured at short context calls attention safe, which is wrong for the lengths agentic use runs at.
  - **Attention is more sensitive, not worse quantized.** Relative weight error by projection over every sixth
    layer, sliding and global: Mila's FP4 (128-element groups) 0.110 to 0.116 on q, k, v and o, and 0.110 to 0.112
    on gate, up and down; NVFP4 (16-element groups, E4M3 block scales) 0.094 to 0.095 everywhere; Q4_0 (32, the
    format Google's QAT targets) 0.088 to 0.096; FP8 per row 0.026. Gemma's weights carry no outlier structure that
    coarse groups trip over, so NVFP4 cuts the error by only about 15%. The same error costs attention more as the
    context grows. Whether NVFP4 moves the curve at all, and whether FP8 attention restores it, are both queued.
  - **Google's quantization-aware weights keep the long context at 4 bits.** `google/gemma-4-12B-it-qat-q4_0-unquantized`
    (Apache 2.0, BF16 weights trained for llama.cpp's Q4_0), the same book, tokens and harness. As stored: 3.649,
    3.342, 3.458, 3.532 -- within 0.05 of the original BF16 in every band. Rounded to Q4_0 (32-element groups,
    FP16 scale; tied table at FP8 as in the package): 3.653, 3.324, 3.463, 3.518 -- within 0.03 of BF16 through
    32K, where Mila's FP4 of the original weights is 0.69 behind. The loss is not inherent to 4-bit weights; it is
    what rounding costs weights that were not trained to be rounded. This is most of the QAT BACKLOG entry's
    question answered in advance of its KL measurement, and in QAT's favour. The QAT weights rounded to Mila's
    FP4 instead: 3.731, 3.484, 3.632, 3.729 -- +0.09 rising to +0.24 over BF16, a third of the original weights'
    +0.69 at 32K, and still rising with context. QAT fitted the weights to Q4_0's grid, evenly spaced integers in
    32-element groups; E2M1 levels in 128-element groups discard part of that fit.
  - **The QAT weights are not on llama.cpp's Q4_0 grid, and no E2M1 format can hold that grid.** Rounding the
    stored QAT weights to Q4_0 moves them by 0.051 relative (layers 0, 5, 23 and 47, every projection): the
    checkpoint holds the weights before rounding. Google's own GGUF is exactly that rounding -- see the next
    result. E2M1's magnitudes, {0, 0.5, 1, 1.5, 2, 3, 4, 6} times a scale, miss at least two of Q4_0's
    integer steps at any scale. What a converter can still do is choose each group's scale by squared error
    rather than by absmax: relative weight error 0.110 for Mila's FP4 today, 0.095 with best scales at 128, 0.079
    at 32, and 0.078 for NVFP4 with best E4M3 block scales (0.098 at absmax). Whether that moves the curve is
    queued: the QAT weights as NVFP4 with best scales.
  - **Google's Q4_0 GGUF is llama.cpp's reference rounding of Google's BF16 QAT checkpoint, bit for bit**
    (2026-09-27, header and values read from `gemma-4-12b-it-qat-q4_0.gguf` and
    `gemma-4-12B-it-qat-q4_0-unquantized`). Every code and every FP16 scale bit of all 328 Q4_0 tensors
    (10,899,947,520 codes, 340,623,360 scales) equals `quantize_row_q4_0_ref` applied to the BF16 weights, with
    no exception: d = the group's signed extreme over -8, code `min(15, trunc(fl(x * fl(1/d)) + 8.5))`, FP16
    scale. Dividing by d, or fusing the multiply-add, also reproduces every code on these weights -- they sit away
    from the rounding boundaries -- so the rule as written is kept (`Quantization.md`, Q4_0). The GGUF holds 328 Q4_0 tensors (every projection; no `attn_v` on the
    8 global layers, K=V), the tied embedding at **Q6_K** -- llama.cpp's own choice, not a trained format -- and
    338 FP32 tensors, whose norms and `layer_output_scale` (HF `layer_scalar`) equal the checkpoint's exactly, with
    no 1 added. No row is permuted: shapes are [out, in] with 32-element groups along the input, as in HF.
    `rope_freqs` is p-RoPE written out (64 of 256 pairs at 1, the rest 1e30). The GGUF carries no image or audio
    weights; the BF16 checkpoint carries both. So the BF16 checkpoint rounded by the reference rule gives the
    GGUF's weights exactly, with the embedding from BF16 rather than from Q6_K, and the GGUF is its bit-exact
    oracle.
  - **Mila's Q4_0 build equals the GGUF, bit for bit** (2026-09-27, `0.21.0-dev+11` tree plus the Q4_0 work). The
    BF16 QAT checkpoint through the existing converter (`gemma4_12b_it_qat_bf16.bin`, 578 tensors) and
    `ExportArtifact --quantization q4_0` (6.65 GiB: 5.08 GiB of codes, 0.63 GiB of FP16 scales, 0.94 GiB FP8
    embedding): all 328 Q4_0 tensors equal the GGUF in every code and every scale bit
    (`q4_0_package_gate.py`, 27 s). The gate was checked against a planted one-code and one-scale-bit change, which
    it reports by tensor and row range. The checkpoint names its vision stack `model.vision_embedder.`, which the
    original 12B checkpoint does not use.
  - **FP8 attention restores it.** The original weights with attention at FP8 per row and the feed-forward at FP4:
    3.683, 3.417, 3.513, 3.595 -- +0.04, +0.08, +0.08, +0.11 over BF16, against +0.19 rising to +0.69 for the
    all-FP4 build, and no worse than the feed-forward-alone arm. The cost is bytes: the attention projections are
    2.41 billion of the body's 10.90 billion weights, so the body grows from 5.39 GiB at 4.25 bits to 6.44 GiB,
    about 1.05 GiB more on the device.
  - **NVFP4 on the original weights barely moves the curve.** 3.898, 3.750, 3.965, 4.063 -- +0.26, +0.41, +0.53,
    +0.58 over BF16, against Mila's FP4 at +0.19, +0.47, +0.59, +0.69: worse over the first 8K, a little better
    beyond it. The emulation rounds each block's E4M3 scale to nearest, which on Gaussian test weights shrinks the
    largest element of 14% of blocks by more than 3%; a variant rounding the scale up, so that nothing clips, is
    queued to tell the format from the emulation.
  - **QAT weights as NVFP4 with best block scales: flatter, not better.** 3.811, 3.511, 3.612, 3.691 -- +0.17, +0.17,
    +0.18, +0.20 over BF16, against the same weights in Mila's FP4 at +0.09, +0.15, +0.20, +0.24. The cost barely
    grows with context, but it starts higher; both NVFP4 arms are worse than 128-element FP4 over the first 8K, which
    the weight error does not predict and is unexplained.
  - **The early-context penalty was the emulation's clipping; NVFP4 still does not fix the long context.** With each
    block scale rounded up to the next E4M3 value, so that no block's largest element exceeds 6 times its scale:
    3.787, 3.714, 3.977, 4.153 -- +0.15, +0.38, +0.54, +0.67 over BF16. Slightly better than Mila's FP4 early, the
    same by 32K. (The first attempt returned NaN: a scale a hair above 448 rounded up into E4M3's NaN code.)
  - **llama.cpp, as users run it.** `llama_cpp_long_context_loss.py` drives llama.cpp b11216's `llama.dll` through
    its C API: `llama-perplexity` cannot run this protocol (it tokenizes without parsing control tokens, scores only
    the second half of each window and overwrites each window's first token with `<bos>`). Same 17 prompt ids as
    Mila and HuggingFace; largest logit exactly 30, so the softcap is applied. LM Studio's
    `lmstudio-community/gemma-4-12B-it-Q4_K_M` (original weights, 6.87 GiB against Mila's 6.33 GiB FP4 package):
    3.789, 3.620, 3.729, 3.796 -- +0.15, +0.28, +0.30, +0.31 over HuggingFace's BF16. At about the same size, the
    format LM Studio users get by default keeps more than half of what Mila's FP4 loses at 32K. llama.cpp's own
    arithmetic (8-bit activations in its quantized GEMMs, FP16 KV cache) puts a small offset on any comparison
    across implementations.
  - **Google's QAT GGUF in llama.cpp matches BF16, and confirms the emulation.** `google/gemma-4-12B-it-qat-q4_0-gguf`:
    3.647, 3.321, 3.462, 3.511 -- +0.01, -0.02, +0.03, +0.02 over BF16, and within 0.007 of the HuggingFace
    emulation of QAT at Q4_0 in every band. Google's own file in llama.cpp's own kernels agrees with the
    emulation, so the emulated rows above stand, and the cross-implementation offset is negligible here. This is
    what a llama.cpp or LM Studio user of Gemma 4 QAT has today: BF16's long-context quality at 4.5 bits.
  - **The training wins, not the format.** The original weights rounded to Q4_0: 3.943, 3.798, 3.995, 4.134 --
    +0.30, +0.46, +0.56, +0.65 over BF16, the same rise as Mila's FP4 and worse over the first 8K. Q4_0 on the QAT
    weights costs +0.03. A Q4_0 policy in Mila pays only together with Google's QAT checkpoint; the checkpoint is
    what carries the long context.
  - **On QAT weights, FP8 attention adds nothing.** QAT weights with attention at FP8 and the feed-forward at Q4_0:
    3.641, 3.342, 3.462, 3.525 -- +0.00, +0.00, +0.03, +0.04 over BF16, the same as all-Q4_0 (+0.01, -0.02, +0.03,
    +0.03). The training already protects attention, so a QAT package is all Q4_0: one format, and none of the
    1.05 GiB that FP8 attention costs.
  - **Query and key carry most of the attention loss, not all of it.** The original weights at FP4 with q_proj and
    k_proj at FP8: 3.689, 3.502, 3.651, 3.751 -- +0.05, +0.16, +0.22, +0.26 over BF16, against +0.69 for all-FP4
    and +0.11 for all attention at FP8. Query and key (1.21 billion weights; +0.53 GiB at FP8) recover about 60% of
    the loss; value and output (1.20 billion; the other +0.52 GiB) hold most of the rest, and it still grows with
    context. On the global layers k_proj also makes V. arXiv 2607.08734 ("The Illusion of Equivalency") reports
    query and key as the most quantization-sensitive projections; this agrees, and adds that the effect scales
    with context.
  - **Query and key alone reproduce most of it.** FP4 on q_proj and k_proj only, everything else BF16 (tied table
    at FP8): 3.790, 3.613, 3.795, 3.864 -- +0.15, +0.27, +0.36, +0.38 over BF16, against +0.47 at 32K for FP4 on
    all of attention. Eleven percent of the body's weights produce about 80% of attention's long-context cost, and
    it grows with context the same way: rounding error in queries and keys perturbs every attention score, and the
    more keys there are, the more of them a perturbed score lets through.

**G3 -- One protocol.** *Closes:* "Chat renders Gemma's prompt itself", "Gemma loses its own reasoning between
tool calls in a turn", "A malformed Gemma tool call parses as a call with no arguments", and "`gemma_protocol.py`
is dead". Independent of every other stage.

- The template first: `Gemma::formatPrompt` gains thinking on, proven byte-identical to Chat's output on
  recorded fixtures before Chat calls it (the `+34` fold's recipe). The two behaviour fixes then land on the
  single template, each with a fixture that fails before it: `extractAnswer` strips prior turns' reasoning
  and keeps the current turn's; `parseArguments` refuses a partial parse and the call is returned as prose,
  as Qwen's bridge does.
- *Gate:* the Gemma protocol suite, MIS's suite, and the live Chat tool-call recipe
  (`--system-prompt tools-weather`).

**G4 -- The feed-forward sublayer becomes a type.** Not yet a BACKLOG entry (section 9). *Needs:* nothing;
it must land before G5 and before any Gemma package is published or republished, since it renames tensors.

Today `GemmaBlock` and `GemmaTransformer` carry two positional flags, `kDelegatedFeedForward` and
`kMixtureOfExperts` (`Gemma.Block.ixx:105`), where one without the other exists only to be rejected by a
`static_assert` (`:124`) and the model always sets both equal (`GemmaModel.ixx:605`). The routed sublayer is
six members null on a dense model and four `if constexpr` sites. The footprint path re-implements the
dense/routed dispatch (`GemmaModel.ixx:286`) and re-spells the transformer type (`:698`) instead of going
through `dispatchChassis`. A routed FP8 request instantiates a whole routed FP8 transformer
(`QuantizationDispatch.ixx:77`) only for `CudaMoeOp`'s constructor to refuse it at runtime.

- One `GemmaTransformer`, one `GemmaBlock`, one `GemmaModel`. The flags are replaced by an enum,
  `GemmaFeedForward { Dense, Routed }`, mapped through a traits struct to `GemmaDenseFeedForward` or
  `GemmaRoutedFeedForward`, each a `CompositeComponent` owning its norms, children, build, footprint and
  workspace slots. The routed sublayer is constrained to the policies the expert bank implements, so a
  routed FP8 build is refused in `dispatchChassis` with a named reason, before instantiation.
- `Gemma4MoE.md` Phase 2b (delete `kDelegatedFeedForward` and the inline FFN) is the first step of this
  stage, and the tensor rename it carries happens once, together with the sublayer's. `getDeploymentFootprint`
  routes through `dispatchChassis`; `GemmaBlock`'s `TWeightQuant` / `TKvPolicy` take their full names.
- The new enum, its traits and the two sublayer types are four files.
- `Gemma.Block.Workspace.ixx`, which G4 changes for the routed slots, exports two types
  (`GemmaBlockWorkspace` and `GemmaBlockWorkspaceWidths`, `:44`, `:99`). It also lists the slots twice:
  `GemmaBlockWorkspaceWidths::slotWidths()` holds the widths the footprint prices, in order, and
  `makeGemmaBlockWorkspace` allocates the slots member by member. A slot added to one list and not the
  other is caught only by the byte total in Gate A, and a slot whose width changes in one list alone can
  pass it. Both are derived from one table of slots, and the workspace answers its own footprint
  (`requiredBytes( config, B, chunk )`, name to settle), which the transformer calls at `Gemma.ixx:804`.
  The widths then have no caller outside the file and stop being exported. The routed slots, which the
  table carries today as `routed_stream_slots = 4`, move with the routed sublayer.
- Separate transformer types for the routed model were considered and rejected: attention and the whole
  transformer are identical between the two, so every fix would land twice. Qwen splits its block types
  because its two token mixers coexist in one model; Gemma's dense and routed layers share the mixer.
- *Gate:* 12B token parity (`GemmaModel.Parity.Cuda.cpp`); footprint drift gates unchanged; the tiny routed
  wiring gate (`Gemma.MixtureOfExperts.Cuda.cpp`) including its forced-failure negative, re-run against the
  new type; the 26B layer-streamed BF16 gate and FP4 greedy tokens `818 5279 529 7001 563 5213 50429 84750`;
  full suite. `Gemma4MoE.md` Phase 8 decision 4 and `Gemma.md` §7 are amended in the same change.

**G5 -- The 26B-A4B on the reference card.** *Closes:* the 3.6 fit cell. *Needs:* G4.

- Pool the routed buffers inside `GemmaRoutedFeedForward`: one gated scratch and pooled outputs across layers,
  as every other activation slot is pooled through the block workspace. The scratch stays FP32. The
  unpooled special case in `GemmaBlock::getRequiredMemory` (`Gemma.Block.ixx:495`) becomes local to the
  sublayer, then goes.
- Then measure: the 26B-A4B joins the rate harness (`GemmaModel.Rates.Cuda.cpp`), prefill and decode, on the
  5060 Ti. The expert floor is about 1.6 ms per token (0.71 GB of active expert bytes at 448 GB/s,
  `MixtureOfExperts.md` 8.1). The decode kernels are the untuned baseline -- one thread per output value,
  6 to 11 blocks at decode on a 36-SM card, byte-wide uncoalesced weight reads, a one-thread-per-row router --
  so a large gap to the floor is expected. The number, not the expectation, decides whether rewriting them is
  admitted (section 9).
- *Gate:* the FP4 fit test at context 8192 passes; Gate A exact; greedy tokens unchanged.

**G6 -- Publish.** *Closes:* "Announce the Gemma 4 26B-A4B mixture of experts". *Needs:* G4, G5, and G1's
harness for the card's quality line.

- The 26B-A4B is a first publish, so it is not held for the 12B. The 12B's republish -- G4's rename, plus the
  image and audio embedders if G7 stays in the release -- happens once, when both are known.

**G7 -- Image and audio input.** *Closes:* "Two questions decide the size of Gemma's image path" and "Gemma 4
12B is multimodal and Mila drops its image and audio weights".

- The two questions are reading, not building, and they decide what the 12B's one republish carries, so they
  are answered early -- before G4 lands. The build follows its BACKLOG entry and is the first item to drain
  if the answers make it large.

**Independent of the order:** the drafter measurement ("Nobody knows whether Google's drafter would make Gemma
4 12B decode faster") needs only today's prefill path and can run whenever a card is free.

**Recommended order:** G1, then the G7 questions, then G2 with the QAT measurement, G3, G4, G5, G6, and the G7
build last. G1 is already next; the G7 questions are cheap and decide the republish; G2 is correctness and
is only machine time once the corpus is chosen. G3 can move anywhere, since nothing depends on it.

**Deferred, and why.** The grouped W4A16 prefill for the expert bank, CUTLASS's SM120 grouped GEMM and the
`120f` architecture target are performance work with no parity cell. The CUTLASS path also needs NVFP4
activations, which `Gemma4MoE.md` keeps off this model. Codebook weights and KV-cache compression stay as
section 9 item 1 records.

### 8.3 Qwen 3.8

Not started. Most of Qwen's gaps already have entries under Qwen 3.8 Complete: KV-cache compression,
quality beyond 16K, streaming in Chat, the smaller dense members. So its pass is mostly a reconciliation.
The one new item is pricing the vision tower against 16 GB, where the FP4 build already holds 13.2 GB of
weights; the vision tower is currently recorded nowhere. G1's shared reduction is a refactor of Qwen's own
code, gated by Qwen's scores staying bit-identical.

### 8.4 Llama 3.x

Started 2026-09-27, and the largest. Section 9 item 3's order: the measurement tooling first, as G1 was for
Gemma, because every later gate reads it; then correctness, then parity inside Mila, then grammar. The 1B
shares the 3B's chassis, so every Llama cell covers it; what it lacks of its own is a published package, and
whether that publish joins the pass or stays in `Future.md` is decided at L7. The Llama half of
`Gemma4MoE.md` Phase 2b -- delegating its FFN, which renames its tensors -- moves here from Gemma's pass
(section 9), so that Llama republishes once, together with whatever L6 changes in its package.

**L1 -- Sequence log-likelihood and HuggingFace token parity, in the suite.** *Closes:* 3.2's log-likelihood
row and 3.4's first row. *Needs:* nothing. `LlamaTransformer` gains `sequenceLogLikelihood` over G1's shared
reduction (no softcap) and `withLogLikelihoodWindow` on its config, reached only through
`Tests/Common/LogLikelihoodHarness.h` (8.1's audit). The reference is a tiny random Llama, as Gemma's MoE
gates use, not the real 3.2 1B: every real Llama 3.x checkpoint carries RoPE scaling, so no real model can
match HuggingFace before L2, and the real 1B's token parity moves to L2's gate.
`Tools/Converters/Llama/hf_llama_tiny_reference.py` builds it (head dimension 128, a group of 2, so L4 reuses
it), converts it with the converter, and records logits and per-position log-likelihood;
`Llama.HuggingFaceReference.Cuda.cpp` reproduces both. *Gate:* logits within 1e-4 and the total within 1e-4
of HuggingFace's at FP32; each scored position's argmax is the greedy token; window 1 and window 64 on the
3.1 8B at FP4 within G1's 2e-3.

- *Result, 2026-09-27* (RTX 5060 Ti). FP32 logits within 7.4e-6 at the prefill and every decode step; the total
  within 4.8e-6 of HuggingFace's -191.2256 at windows 1, 3 and 8; argmax is the greedy token at all six
  positions. BF16 within 2.1e-2 relative L2, and one argmax differs: step 1, where HuggingFace's own two best
  logits are 0.0087 apart, under one BF16 step at that magnitude (0.0156), which the BF16 head cannot order.
  The BF16 gate accepts a disagreement only inside that step -- set after this run, and said so. The 3.1 8B
  at FP4, 15,479 wikitext positions at context 1024: perplexity 11.8945 at window 1 and 11.8941 at window 64,
  2.8e-5 apart (the head is BF16, so there is no staged FP8 path to widen it as Gemma's is), 60 s against 15 s.
- The pass's audit removed from `LlamaTransformer` what nothing reached: five build-shape members never
  assigned (so `toString` and the saved metadata reported zeros), an unused prefill buffer, and two helpers,
  one of them the `createConfigFromMetadata` whose commented-out RoPE line `Vnext.md` cites.

**L2 -- RoPE scaling.** *Closes:* 3.1's context row and 4's Llama 3 RoPE row (`Vnext.md`, "Llama's
long-context scaling factor is stored, printed, and never used"). *Needs:* L1. Llama 3's frequency scaling
in the Rope component and its CPU and CUDA operations, from the factor, the low- and high-frequency factors
and the original context the checkpoint's `rope_scaling` carries. Below the original 8192 the scaled and
unscaled tables differ only in the low-frequency bands, so the gate must read past it -- or shrink it: a tiny
model with an original context of 32 shows the scaling inside a 32-token capture, and at head dimension 128
puts frequencies in all three of the rule's bands. *Gate:* L1's tiny-model gates on a capture with Llama 3
scaling, and *the gate can fail*: with scaling off it must; then the real 3.2 1B against HuggingFace at FP32
over 10240 wikitext tokens, mean log-likelihood within 1e-3 nats per token below 8192 and past it, with
scaling off missing past it; and its greedy tokens.

- *Built.* `RopeFrequencyScaling` (its own module) on `RopeConfig`, applied where the cos/sin table is built and
  part of the table's cache key; `RotaryLayout` left `Rope.Config.ixx` and `WeightsMetadata` left
  `WeightsReader.ixx` under the one-type rule. `LlamaConfig::withRoPEFrequencyScaling` replaces the bare factor
  nothing read. The converter writes the rule and its four values (`rope_scaling`: `none` or `llama3`), and
  `ExportArtifact` carries them. **A Llama file that records neither is refused**: 3.1 and 3.2 are wrong at
  every position without it, so every Llama file converted before this change must be converted and exported
  again -- including the two published packages, which L7's republish replaces.
- *Result, 2026-09-27* (RTX 5060 Ti). Tiny scaled capture: FP32 logits within 7.0e-6, the total within 1.1e-5
  at windows 1, 3 and 8, BF16 argmax equal at every step; the same weights read without the scaling miss by
  0.25 nats. The real 3.2 1B, 10240 tokens: 2.274533 and 1.855534 nats per token below and past 8192, within
  3.3e-8 and 5.1e-7 of HuggingFace; without the scaling, 2.301501 and 1.894908 -- 0.027 and 0.039 worse. That
  is what every Llama Mila shipped has run at: not only past 8192, at every position. Greedy tokens equal
  HuggingFace's.

**L3 -- Quality across the planner's range.** *Closes:* 3.4's third row, and section 9 item 2. *Needs:* L1,
L2. G2's protocol and its three tests on the same PG-19 books, inside a Llama user turn, on the 3.1 8B at
FP4 to the largest context the planner chooses on the reference card after L4.

- *Protocol, decided 2026-09-28 (Todd).* One network built at 69632, the planner's choice for the 3.1 8B on
  the RTX 5060 Ti at chunk 1024 (L4), window 64. Bands 0-8K, 8K-16K, 16K-32K, 32K-64K and 64K-69632, the last
  the top of the range and the question. Five books: G2's two (30312 and 3608), so the families compare on the
  same text, then the next three in the test split's order that fill 69632 tokens. The turn is
  `<|begin_of_text|>`, a user turn "Continue this book.", the assistant header, then the book with its wraps
  joined -- no system block, as Chat renders it, although HuggingFace's 3.1 template adds a dated one by default;
  HuggingFace and llama.cpp are fed the same ids.
- *Test 1's short arm is 1024 tokens,* as Gemma's, in 1024-target blocks each after the turn and only the 1024
  book tokens before the block, so that the two families' tables compare. Llama has no sliding window, so the
  length is a choice, not the architecture's. An 8192 arm -- Llama's original context -- may be reported beside
  it and gates nothing.
- *Test 2's bound is written after a HuggingFace measurement and before Mila's run.* Meta publishes no
  quantization-aware 3.1 8B, so the package rounds weights not trained to be rounded, which is what cost Gemma
  +0.69 at 32K; section 9 item 14's fallback for that case is FP4 with all of attention at FP8, which Mila cannot
  build today. One book to 32K in HuggingFace -- BF16, FP8, FP4, FP4 with attention at FP8, and Q4_0, each as
  Mila rounds it -- decides the package's format and the bound.
- *Test 3 runs on a Q4_0 build,* since the FP4 package has no llama.cpp counterpart: the 8B rounded by the
  reference Q4_0 rule against a Q4_0 GGUF of the same checkpoint, first checked equal code for code (llama.cpp
  permutes the query and key rows, which the check undoes). It tests Mila's long-context arithmetic to 69632 on
  identical weights, which is what test 3 is for.
- *HuggingFace by weight format, 2026-09-28* (`Tools/Converters/Llama/hf_llama_long_context_loss.py`, books 30312
  and 3608 to 32K, the body rounded as Mila rounds it, embedding and head BF16; the text with CRLF joined, below).
  Cost over BF16, nats per token, by 8K band:

  | Weights | Book 30312 | Book 3608 | Mean |
  |---|---|---|---|
  | BF16 (loss) | 2.620, 2.562, 2.535, 2.631 | 2.180, 2.164, 2.091, 2.013 | |
  | FP8 | +0.003, +0.002, +0.003, +0.000 | -0.000, +0.000, +0.001, -0.001 | 0.001 |
  | FP4, attention at FP8 | +0.040, +0.020, +0.029, +0.027 | +0.033, +0.033, +0.033, +0.022 | 0.030 |
  | Q4_0 | +0.035, +0.032, +0.034, +0.036 | +0.037, +0.027, +0.040, +0.027 | 0.034 |
  | FP4 | +0.067, +0.041, +0.051, +0.048 | +0.056, +0.053, +0.059, +0.047 | 0.053 |

  **Llama does not lose long context to 4-bit weights.** Every format's cost is flat across the bands on both
  books, where Gemma's FP4 cost rose from +0.19 to +0.69 over the same range. Q4_0 on the original weights and FP4
  with attention at FP8 agree within 0.004 on average and both stay under 0.05 in every band; Q4_0's body is about
  3.7 GiB against about 4.0. On Gemma the two separate: Q4_0 on the original weights cost +0.30 to +0.65, FP8
  attention +0.04 to +0.11. The same sweep on the text before its line endings were fixed read Q4_0 ahead of FP8
  attention in every band -- an artifact of that text, and the first statement of this decision rested on it.
- *Decided 2026-09-28 (Todd), on that result.* Test 2's bound is 0.05 nats per token, Gemma's; today's FP4 package
  fails it (+0.067 over book 30312's first 8K), so **the Llama package leaves FP4** at L7's republish. Section 9
  item 14's fallback is corrected to depend on the family. **Q4_0 or FP4 with attention at FP8 is decided after
  the Q4_0 GEMM queued for Gemma lands**, on prefill rate measured beside this table: the two are the same quality
  within 0.004, Q4_0 is about 0.4 GiB smaller, one format across the two families, and a policy Llama already has,
  where the mixed build needs a new one; Q4_0's prefill stages its weights to BF16 until that GEMM exists. First
  proposed as Q4_0 outright, on the sweep over the damaged text. *How, decided 2026-09-28 (Todd):* when the GEMM
  lands, Q4_0's prefill is measured and each projection is timed in the existing FP8 and FP4 builds (nsys); the
  faster format per role bounds any mixed build from above, since a projection runs the same kernel at the same
  shape in either. If Q4_0 meets that bound, Q4_0 is the package and nothing is built. Otherwise the mixed build
  is built -- `LlamaBlock` and `LlamaTransformer` taking a precision plan as Qwen's blocks do, and
  `Fp8AttentionPrecisionPlan<TBody>` in a module of its own -- and decides by L3's protocol and the prefill
  harness in Mila's own arithmetic, with `PlanEqualsBuild` pricing its footprint: the estimate decides whether a
  build is worth making, never the package.
- *Two defects found by the first Mila run, 2026-09-28, both fixed before any result stands.* Mila's FP4 read
  3.085 and 2.459 nats per token over the first 8K of books 30312 and 3608, against 2.567 and 2.122 in
  HuggingFace on the same format -- far past any arithmetic offset -- because the two were not scoring the same
  tokens.
  1. **Mila's Llama 3 tokenizer did not produce HuggingFace's ids.** It encoded a pre-token by the longest
     vocabulary match at each position; tiktoken, and HuggingFace's `ignore_merges` tokenizer, emit a pre-token
     the vocabulary holds whole and otherwise merge adjacent units lowest rank first, a token's rank being its id.
     The two agree on most words and not on rarer ones (" seminary": 54675 88 against 5347 3367), so about one
     token in seventeen of English prose differed -- 2,913 of 45,870 on book 30312, 807 of 14,167 on wikitext.
     The premise that `ignore_merges` made Llama 3 "max-munch equivalent" was written into the tokenizer, the
     Qwen loader's comments, the Converters README, `Qwen3.8.md` and a test that asserted it. Separately, the ASCII
     fallback pattern -- the one MSVC runs for all text, since its `std::regex` has no `\p{L}` -- matched
     contractions case-sensitively where the pattern's `(?i:...)` does not, so a quote before a capital after a
     line break ("'Didn") split differently; Qwen's fallback had the same fault. Fixed in `BpeTokenizer.ixx`
     (rank path) and `BpePreTokenizationMode.ixx`, gated by `BpeTokenizerLlama32.Encode_MatchesHuggingFaceIds`
     and `Encode_QuoteBeforeCapital_MatchesHuggingFaceIds` (each checked failing on the old code). All five L3
     books now tokenize to HuggingFace's ids for all 69,618 book tokens; what still differs anywhere is non-ASCII
     text through the fallback (20 regions in 68,538 wikitext tokens). Every Llama and GPT-2 encode in Mila --
     Chat, the binding -- took the old path. L2 compared HuggingFace's own ids and is unaffected; L1's window
     gate and L4's flash A/B tokenized wikitext through Mila, and were re-run (RTX 4070, the 3.1 8B at FP4): at
     context 1024 perplexity is **8.303, where it read 11.89** -- the tokenizer alone cost the 8B 0.36 nats per
     token on wikitext -- with windows 1 and 64 1.6e-5 apart; at context 4096, 6.807 with flash against 6.803 with
     cuBLASLt, 6.3e-4 nats per token apart (from 9.47 and 8.3e-4).
  2. **PG-19's books are stored with CRLF line endings, and the wrap join never saw a bare newline.** All 100 test
     books; every line break, paragraph breaks included, became "\r " -- no paragraph survived. Mila's `joinWraps`
     and the Python harnesses did the same, so every Mila, HuggingFace and llama.cpp comparison in G2 and here
     read identical text, but every level was measured on it, diagnostic H's included. CRLF now becomes LF first
     in all three; G2's results before 2026-09-28 read the old text, and its next run reads the new.
- *Mila on the Q4_0 build, 2026-09-28* (`LlamaQualityCudaTests.DISABLED_AcrossContextLengths_Q4_0`, RTX 5060 Ti, the
  tokenizer and line-ending fixes in, chunk 128 -- the library's rule on the live reading, below the recorded
  readings' 1024 because Q4_0 stages its weights; 56 minutes for five books). Nats per book token, whole book
  against 1024 tokens of it alone:

  | Book | 0-8K | 8K-16K | 16K-32K | 32K-64K | 64K-69632 |
  |---|---|---|---|---|---|
  | 30312 | 2.6539 / 2.6713 | 2.5932 / 2.6381 | 2.6176 / 2.6476 | 2.6568 / 2.6847 | 2.6097 / 2.6813 |
  | 3608 | 2.2168 / 2.2385 | 2.1925 / 2.2320 | 2.0849 / 2.1020 | 2.1165 / 2.1444 | 2.1459 / 2.2279 |
  | 10321 | 3.2545 / 3.2666 | 3.2364 / 3.3035 | 3.2297 / 3.2997 | 3.0840 / 3.1503 | 3.0197 / 3.0697 |
  | 10356 | 2.6639 / 2.7458 | 2.4720 / 2.5650 | 2.5126 / 2.6311 | 2.6279 / 2.7188 | 2.7536 / 2.8509 |
  | 10762 | 3.1432 / 3.1470 | **3.1872 / 3.1709** | **3.1357 / 3.1233** | **3.1489 / 3.1335** | **3.1054 / 3.0687** |
  | Pooled | 2.7865 / 2.8138 | 2.7363 / 2.7819 | 2.7161 / 2.7607 | 2.7268 / 2.7663 | 2.7269 / 2.7797 |

  **Test 1 passes on four books and fails on the fifth.** Book 10762 -- George Eliot's *Impressions of Theophrastus
  Such*, eighteen separate character essays -- reads 0.013 to 0.037 nats worse with the whole book before each band
  than with 1024 tokens of it, in every band past 8K; the other four books gain 0.02 to 0.10 from the whole book,
  and so does the pool in every band. **The failure is the model's, not the quantization's.** Test 1 on book 10762 in
  HuggingFace to 32K (`hf_llama_long_context_loss.py --short-context 1024`), whole book less short context by 8K band:
  BF16 -0.0001, +0.025, +0.009, +0.022; Q4_0 -0.004, +0.016, +0.005, +0.019. BF16 fails it by more than Q4_0 does: on a
  collection of unrelated essays the 3.1 8B predicts slightly worse with the whole book before it, however it is
  quantized. HuggingFace's Q4_0 matches Mila's in both arms to 0.001 (8K-16K: 3.1873 / 3.1718 against 3.1872 /
  3.1709), which checks Mila's short-context arm as well. The gate as written counts these bands as failures and
  reads a failure that is the model's as marking where it stops using its context -- which, from one essay
  collection, would put the 8B's ceiling at 8192 while the other four books and the pool gain from context in every
  band to 69632. How test 1 treats a book the model itself does not use is open (section 9).
  **Test 2 passes on both books it has references for**: cost over HuggingFace BF16 +0.034, +0.031, +0.035 on 30312
  and +0.037, +0.028, +0.033 on 3608 over 0-8K, 8K-16K and 16K-32K, against 0.05. Mila agrees with HuggingFace's Q4_0
  emulation of the same weights to 0.001 in every one of those bands.
  **Test 3 passes on all five books.** llama.cpp b11216 on the reference GGUF (below; `llama_cpp_long_context_loss.py
  --family llama`, 1024-token chunks, RTX 5060 Ti), Mila less llama.cpp by band: 30312 -0.0005, +0.0006, +0.0007,
  +0.0023, +0.0023; 3608 +0.0010, +0.0025, +0.0021, +0.0034, +0.0051; 10321 0.0000, +0.0004, +0.0012, +0.0031,
  +0.0023; 10356 +0.0005, +0.0007, +0.0009, +0.0014, +0.0012; 10762 -0.0004, +0.0007, +0.0012, +0.0011, +0.0009. The
  largest difference anywhere is 0.005, against bounds of 0.021 to 0.023. llama.cpp reads book 10762 as Mila does,
  so its test 1 result is not Mila's arithmetic.
- *Test 3's weights, 2026-09-28.* Published Q4_0 GGUFs of the 8B cannot serve: unsloth's is fitted with an
  importance matrix and mixes in Q4_1 (four `ffn_down`) and a Q6_K head. `Tools/Converters/Llama/llama_q4_0_gguf.py`
  writes one from Meta's BF16 checkpoint -- projections by gguf-py's port of the reference Q4_0 rule, query and key
  rows permuted as llama.cpp's converter does, embedding and head BF16, metadata and the RoPE scaling table from a
  published GGUF. `ExportArtifact --quantization q4_0` on the converted checkpoint (5.61 GiB, 14 s) equals it in
  every code and every scale bit of all 224 Q4_0 tensors (`llama_q4_0_package_gate.py`); the same gate without
  undoing the query and key permutation reports 598 million codes differing, all in those two projections.

**L4 -- Flash-attention prefill, and the context threshold retired in every family.** *Closes:* 3.1's flash
row (`Vnext.md`, "Llama prefill has no flash path and is 3.8x slower than a larger Gemma") and the rest of
`Untriaged.md`'s long-context prefill entry ("Gemma and Qwen still switch flash off below a context
threshold"). *Needs:* L1 for its end-to-end gate; L2, or section 9 item 2's stopgap, before it reaches a
user, because removing the score buffers raises the context the planner chooses for Llama.

- *Measured, 2026-09-27* (`CudaGqaFlashPrefillRate.DISABLED_AcrossPromptLengths`, one binary, both paths
  on one op, chunk 512, attention alone). The packed kernel beats the cuBLASLt pipeline at every prompt
  length from 512 to 32768 and at every unbounded geometry, on both cards:

  | Geometry | RTX 5060 Ti | RTX 4070 |
  |---|---|---|
  | Llama 3.1 8B (32 / 8 / 128) | 6.5x at 512, 7.4x at 32K | 6.4x, 8.4x |
  | Llama 3.2 3B (24 / 8 / 128) | 7.4x, 8.9x | 5.9x, 8.2x |
  | Qwen 3.8 full attention (24 / 4 / 256) | 4.4x, 3.5x | 4.6x, 3.8x |
  | Gemma 4 global (16 / 1 / 512) | 2.4x, 1.45x | 3.7x, 1.9x |

  The op is built at the prompt length; a deployed op is built at the whole context, which can only add to
  the cuBLASLt side. The threshold existed because Stage 2d lost at short prompts (`GqaFlashAttention.md`
  5.7); the packed kernel does not, so no unbounded layer keeps it. Gemma's sliding layers run the bounded
  ring kernel, which this table does not cover, and are measured the same way before Gemma's threshold goes.
- *Llama.* Flash on for every BF16 build; FP32 keeps cuBLASLt. The seven loose workspace tensors become
  `GqaWorkspace` from `makeGqaWorkspace`, as Qwen's are, with the score width derived from the same
  decision, so the 3.1 8B at 131072 and chunk 512 stops reserving 4 GiB per score buffer. The footprint
  (`requiredMemoryAtChunk`) reads the same width. Fused decode is wired in the same change
  (`withFusedDecode`): its kernel already passes at Llama's geometry
  (`CudaGqaDecodeParity.FusedMatchesCublasLt_LlamaConfig`) and nothing turns it on; it ships only if the
  decode rate does not fall. The chunk rungs (512, 256, 128, against 1024 first in the other families) were
  set while the score buffers scaled with the chunk; they are re-measured, not assumed.
- *Llama result, 2026-09-27* (3.1 8B FP4, RTX 5060 Ti, one build; the cuBLASLt arm is the same network with a
  full-width score buffer installed and flash off, `Llama.LogLikelihood.Cuda.cpp`). Flash scores 15,479
  wikitext positions at context 4096 at 2.248195 nats per token against cuBLASLt's 2.249025, 8.3e-4 apart.
  Prefill: 4,193 tokens/s against 1,569 at an 8K prompt (2.7x), 2,383 against 509 at 32K (4.7x). Fused decode
  in a 32K deployment: 66.8 tokens/s against 38.3 after a 512-token prompt, 46.8 against 34.6 after 16K. The
  tiny model's BF16 gates hold with both on. RTX 4070: 9.6e-4 apart; prefill 3,698 against 1,486 at 8K and
  1,388 against 415 at 32K (where the 12 GB card plans a 128-row chunk); decode 78.0 against 40.1 and 57.1
  against 34.8. The planner's choice for the 8B on `DeploymentPlanner.G2.Cuda.cpp`'s recorded readings rises
  from 13312 to 19456 on the 4070 and from 33792 to 50176 on the 5060 Ti.
- *Chunk rungs and llama.cpp, 2026-09-27* (3.1 8B FP4, RTX 5060 Ti, one build,
  `LlamaFlashPrefillCudaTests.DISABLED_PrefillRateByChunk_Fp4`: each chunk built directly, the planner bypassed,
  best of three). A 1024 rung was 12.9% faster than 512 at an 8K prompt and 6.9% at 32K, but cost 2,762 MiB,
  about 20,800 tokens of context at 136 KiB a token: the 32 blocks each allocated their own activations at the
  chunk, about 5.4 MB a row, where Gemma's and Qwen's share one block workspace (3.2). The planner prefers the
  largest context at which the top rung fits, so the rung alone would have traded 20,800 tokens for that gain.
- *Pooled.* One `LlamaBlockWorkspace` (`Llama.Block.Workspace.ixx`, 13 slots, 164 KiB a row on the 8B) is
  installed on every block in inference, as Qwen's attention workspace is, and the per-block copy of the block's
  input left with it: `res_1` reads the input directly, which the stream slot's alias rule makes safe. Training
  still self-allocates. The rungs are now 1024, 512, 256 and 128. Footprint equals the build at every rung
  (`PlanEqualsBuild`), and Gate B at context 8192 predicts 6.793 GiB against 6.799 consumed. 1024 now costs 92 MiB
  over 512, about 700 tokens. The planner's choice on the recorded readings rises to 38912 on the 4070 and 69632
  on the 5060 Ti, both at chunk 1024 (from 19456 and 50176 at 512). Prefill, same test and card: 5,206 / 4,763 /
  4,238 / 3,319 / 2,210 tokens/s at an 8K prompt and chunks 2048 / 1024 / 512 / 256 / 128; 2,766 / 2,559 / 2,404
  / 2,073 / 1,578 at 32K. Every gate re-ran on the pooled build: the 3.2 1B against HuggingFace 1.7e-7 and 2.5e-7
  nats a token below and past 8192, greedy tokens equal; flash against cuBLASLt 8.3e-4; window 1 against 64 on the
  8B at 9.6e-6 (perplexity 11.8863, from 11.8945 at the old chunk of 512: 0.07% with the chunk, cause not
  measured); fused decode 66.5 and 46.4 tokens/s, unchanged. 2048, above every
  family's top rung, would add 9.3% at 8K and 8.1% at 32K for 184 MiB more, about 1,400 tokens.
- *llama.cpp.* b11216 on a Q4_0 GGUF of the same model (`unsloth/Llama-3.1-8B-Instruct-GGUF`,
  `llama-bench -ngl 99 -fa 1 -p 8192,32768 -n 0 -r 3`, its default 512-token micro-batch): 3,686 and 1,883
  tokens/s. Mila at chunk 1024 is 1.29x and 1.36x ahead.
- *Gemma and Qwen.* `kGemmaFlashPrefillMinContext` and `kFlashPrefillMinContext` are deleted and the flash
  decision is precision alone (`usesFlashPrefill()`), after the ring kernel measured 2.0x to 3.5x ahead at the
  sliding layers' geometry (`GqaFlashAttention.md` 5.7). Gemma keeps its score buffer at the ring capacity so a
  standalone op's cuBLASLt fallback stays valid; Qwen's is one row.
- *The threshold had been hiding a head-size limit.* The unbounded kernel serves head sizes 128, 256 and 512 and
  the ring kernel multiples of 16 to 256; below 16384 nothing reached them, so nothing asked. With the threshold
  gone the tiny Gemma models (head size 64) threw at prefill, and so would Llama 3.2 1B in BF16 (also 64). Each
  kernel file now owns a `...supported( head_size )` its launcher checks through, `CudaGqaOp::supportsFlashPrefill`
  and `GroupedQueryAttention::supportsFlashPrefill` carry it up, and every family's decision is BF16 *and*
  supported, per layer kind in Gemma, whose score width is the whole context whenever a layer that needs it is
  left on cuBLASLt. `Llama.Cuda.cpp`'s `Bf16_UnsupportedFlashHeadSize_PrefillsThroughCublasLt` holds it.
- *Gate, written before the run:*
  - `CudaGqaFlashPrefillParity` at the 3B and 8B geometries, on both cards. *Passed 2026-09-27.*
  - L1's HuggingFace token parity and log-likelihood hold with flash on.
  - The 3.1 8B at FP4 scores the same book prefix with flash and with cuBLASLt within 0.02 nats per token
    in every band that fits both (Gemma's A/B agreed within 0.016 bare and 0.003 in a turn).
  - Footprint agreement stays exact (`PlanEqualsBuild`), at every rung.
  - Prefill rate at 8K and 32K on the RTX 5060 Ti, against the cuBLASLt build and against llama.cpp on a
    Q4_0 GGUF of the same model, recorded. *Recorded 2026-09-27.*

**L5 -- Sampling on the device, overlapped with the next forward.** *Closes:* 3.1's first two rows. Applies
Gemma's and Qwen's sampler; `top_p` and `seedSampler` start working.

**L6 -- Embedding table and output head quantized with the body, and held once where the checkpoint ties
them.** *Closes:* 3.2's second and third rows. The converter writes a tied checkpoint's table once and records
the tie, and the transformer shares it between the embedding and the head, as Gemma's does. Changes the
package, so it is collected for L7.

**L7 -- One republish, grammar in `Mila/Src`, and the rest of the matrix.** Phase 2b's FFN delegation and
L6 ride one republish (8.1). The grammar is `ModelHandle.md` Phase 2's Llama half. Then a rate harness and
streaming in Chat.

---

## 9. Open Decisions

1. ~~**Where the work is committed.**~~ **Decided 2026-09-25 (Todd): not a ROADMAP theme.** A parity pass
   over the existing families, Gemma 4 first. Each family's gaps become `BACKLOG.md` entries under its
   existing theme, whose success criteria gain the parity clause that admits them.

   **Gemma, started 2026-09-25.** Three entries admitted: scoring and head width, quality across the
   planner's range, one template. Deferred rather than admitted, with the reason: codebook weights (the
   12B's FP4 build already fits 12 GB; the case for it is the 26B-A4B on a 12 GB card) and KV-cache
   compression (missing in all three families; Qwen's entry leads it). Its section 4 rows -- image and
   audio, the drafter, the QAT build, the 26B-A4B -- were already in `BACKLOG.md`; the drafter and QAT
   *implementations* sit in `Vnext.md` behind their measurements, which the ROADMAP text keeps out of this
   release (open, item 5).
2. ~~**Llama above the context it is accurate at, until RoPE scaling lands.**~~ **Resolved 2026-09-27: the
   scaling landed first** (8.4, L2), before flash raised the planner's choice for the 3.1 8B from 13312 to
   19456 on the 4070 and from 33792 to 50176 on the 5060 Ti. No stopgap was written.
3. **The order of the three kinds of work.** Recommend correctness first (RoPE scaling, then quality across
   the planner's range), then parity inside Mila (sampling, flash prefill, tables,
   `sequenceLogLikelihood`), then grammar -- which is also `ModelHandle.md` Phase 2, so it builds the handle's
   foundation at the same time.
4. **Llama 3.1's built-in tools.** In scope under section 1, but they are a tool-execution convention
   (`ipython`) as much as a grammar. Decide with the agent core's spec whether they are a Llama capability or
   a `Mila::AI` one.
5. **Gemma's drafter and QAT implementations under the 16 GB rule.** Section 1 puts both in scope, but
   `ROADMAP.md`'s Gemma 4 Complete text makes only their measurements this release. Recommend keeping it
   that way: each measurement carries a stop condition that can decline the work, so admitting the
   implementation first would commit to something the number may reject. **QAT resolved 2026-09-27:** its
   measurement answered in QAT's favour and the implementation is admitted (item 14). **Drafter decided
   2026-09-27 (Todd): measurement only in 0.21**, on the QAT drafter against the Q4_0 12B.
6. **Admit G4 and G5 (8.2).** Neither has an entry. G4 is tracked only as `Gemma4MoE.md`'s Phase 2b and Phase 8
   decision 4, which no work-tracking file names. G5 is what makes "the 26B-A4B is fetchable" true on the
   reference card. Both earn admission under Gemma 4 Complete's existing criterion, and the 26B entry's claim
   that it needs "not more implementation" is corrected in the same edit. Recommend admitting both, paired
   with no removal: this is scope the survey found, not scope invented.
7. **G4's four design choices.** (a) Enum and traits, recommended, against a template template parameter for
   the sublayer type. (b) The sublayer's child name, which becomes part of every FFN tensor name: `ffn`
   proposed. (c) Regenerate the tiny routed checkpoint and its HuggingFace capture under the new names,
   recommended, against a converter that accepts both. (d) Phase 2b and the sublayer types as one commit or
   two; both land before any Gemma publish either way.
8. **Llama's half of Phase 2b moves to Llama's pass** (8.4). `Gemma4MoE.md` pairs the two families so they
   republish once together. That pairing predates the one-republish-per-family rule (8.1), under which each
   family's rename rides its own pass.
9. **Whether the 26B's decode kernels are admitted** -- decided on G5's measured rate against the expert
   floor, the way the drafter and QAT questions are decided on theirs.
10. ~~**The long-document corpus for G2**: which one, and its licence verified at the source.~~ **Decided
    2026-09-26 (Todd): the PG-19 test split** (DeepMind, `github.com/google-deepmind/pg19`): 100 Project
    Gutenberg books published before 1919, the dataset Apache 2.0 and the texts public domain, read at the
    source 2026-09-26. Kept in `Data/Datasets/PG19/raw/`, gitignored, never redistributed. The test split's
    books run to 4.5 million characters, so single books fill a 262144 window. The books are almost certainly
    in Gemma's training data, which matters little for a measurement that compares one build with itself
    across lengths, and would matter for a comparison against published numbers. The same decision replaces
    wikitext perplexity with KL divergence from BF16 as the QAT measurement's metric (8.2, G2).
11. ~~**G1's names and tolerance.**~~ **Decided 2026-09-26 (Todd):** `sequenceLogLikelihood` and
    `withLogLikelihoodWindow`, and window 1 against window 64 within a relative perplexity difference of
    1e-3. `Qwen3.8.md` measured 7.513 against 7.515 and set no bound, so this is the first recorded one.
12. ~~**Does the impl-only exception let `Gemma.Block.Workspace.ixx` keep two types?**~~ **Decided
    2026-09-26 (Todd): no.** A type other modules must name gets its own file; a type only its own file
    uses is not exported (CLAUDE.md). Resolved in G4: the widths stop being exported once the workspace
    prices itself from the one table of slots it allocates from.
13. ~~**G1's two 12B gates, both failed as written, both now diagnosed (8.2, G1 result).**~~ **Decided 2026-09-26
    (Todd):** the window bound is 2e-3 for G1, with the cause named in the test; the FP8 staged path's scale moves
    after the dot product as its own change, gated at Linear level against exact FP64, and that change sets the
    bound back to 1e-3. The 12B argmax check reports its agreement count; the tiny FP32 model asserts it.
    The analysis as it stood:
    (a) *The window bound.* The 1.06e-3 is the FP8 head's staged path rounding each weight to BF16 before the
    GEMM, against a matvec that is already as exact as BF16 logits allow. Fixing the staged path -- scale after
    the dot product, as the matvec does -- would bring window 64 to window 1's accuracy, and the 1e-3 bound
    would then test what it was meant to. The same path serves every per-channel FP8 Linear above one row, so it
    is a `Mila/Src` kernel change with its own gate. Or the bound widens to 2e-3 with the mechanism recorded.
    (b) *The argmax gate.* Exact agreement between prefill and decode is not a property BF16 inference of this
    model has, in Mila or in HuggingFace; it holds on the tiny FP32 model, where it passes. The 12B check
    reports its agreement count instead of asserting it.
14. ~~**The Gemma 4 12B package** (8.2, G2 result).~~ **Decided 2026-09-27 (Todd):** where the producer
    publishes quantization-aware weights, Mila runs those weights in the format they were trained for; where it
    does not, FP4 with all of attention at FP8. For Gemma 4 12B that is Google's QAT weights at Q4_0, which needs
    a `PerGroupInt4<32>`-class policy (`Vnext.md`'s QAT entry, its blocking measurement answered). Measured on one
    book, cost over BF16 at 24K-32K: today's FP4 +0.69; FP4 with query and key at FP8 +0.26 (+0.53 GiB); FP4 with
    all attention at FP8 +0.11 (+1.05 GiB); QAT at Q4_0 +0.03 (about +0.3 GiB; confirmed in llama.cpp). The
    fallback is all of attention, not query and key alone. **The Q4_0 policy is admitted to 0.21 (Todd,
    2026-09-27)** -- deliberate scope growth against the 2026-12-15 date, since Gemma 4 is committed and must
    hold up at agentic context lengths; `BACKLOG.md` carries it. **Source and delivery decided 2026-09-27
    (Todd):** built from Google's BF16 QAT checkpoint by the reference Q4_0 rule (bit-identical to Google's GGUF,
    8.2 G2 result), and delivered as a Mila package. Google's only complete source is 22 GiB (the 26B-A4B's far
    more); its GGUF is small but has no image or audio weights and a Q6_K embedding. This refines item 16: a
    recipe where the producer's trained format is usable as shipped, a package where its only complete source
    is full precision. **The fallback corrected 2026-09-28 (Todd):** "FP4 with all of attention at FP8" was
    Gemma's measurement, not a rule. Llama 3.1 8B, with no quantization-aware checkpoint, loses no long context at
    4 bits, and Q4_0 matches FP8 attention there within 0.004 nats per token in fewer bytes (8.4, L3). Without quantization-aware
    weights, the format is chosen per family by the long-context measurement. The change rides G4's
    single republish, and before it: a second book, and G2 re-run in Mila on the Q4_0 build, with its prefill's
    activation precision gated by the curve.
15. ~~**G2's gate.**~~ **Decided 2026-09-27 (Todd):** three tests replace "the top band within 1.05 of the band
    below", which failed on book 30312 (1.67) because the loss rises from 16K, not past 131072 (8.2, G2). A
    gate against BF16 at every length, as first proposed here, cannot run: BF16 has no reference past 32K on
    this machine. So the length test is the model against itself -- the whole book predicts no worse than the
    1024-token sliding window alone -- with the cost over BF16 bounded at 0.05 nats per token to 32K, and Mila
    checked against llama.cpp on Google's GGUF to 262144. Chat's 131072 cap (`ModelHandle.md` 10.3) is decided
    on that run.
16. **Recipes rather than packages, and what Mila is at the edge.** Agreed in discussion 2026-09-27 (Todd):
    where a producer publishes its trained quantized format, Mila installs it rather than republishing weights;
    and Mila's place beside llama.cpp is a library you build with and measure through, not breadth. Where each is
    written (`ModelDistribution.md`, `Direction.md`) is open, with the format principle, in `Untriaged.md`.
17. **How test 1 treats a book the model does not use (8.4, L3).** Written for G2 and taken for L3: in every band of
    every book, the whole book must predict no worse than 1024 tokens of it alone; a failure that is the model's
    marks where it stops using its context. Book 10762, a collection of unrelated essays, fails it past 8K on the
    Llama Q4_0 build -- and on BF16, by more. Read as written, that one book caps the 8B at 8192 while four books and
    the pool gain in every band to 69632. (a) Keep it as written. (b) Gate test 1 on the pool of books, band by band,
    report every book, and trace any book that fails to BF16 where BF16 runs (to 32K): a failure BF16 shares is the
    model's and is recorded, not gated. (c) Per book, gate the build against BF16 -- the quantized build's whole-book
    gain no smaller than BF16's less a margin -- which reaches only 32K. Recommend (b), for G2 as well: it keeps the
    question test 1 was written to ask (does quantization damage long context) and keeps a model's own limit on one
    kind of text from being read as its limit on all of them.

---

## 10. Relationship To Other Specs

- `ModelHandle.md` -- its capability sources (3.2) and protocol (3.4) are this matrix's grammar and
  capability rows; its Phase 2 closes section 3.3.
- `Deployment.md` -- the planner whose range section 1's third rule and row 3.4 hold quality to.
- `Qwen3.8.md`, `Gemma.md`, `Gemma4MoE.md` -- each family's design of record; section 4 points into them.
- `MixtureOfExperts.md` -- the expert bank's design of record; G5's floor and the deferred grouped prefill.
- `SpeculativeDecoding.md` -- the draft design behind the MTP rows.
- `PromptCaching.md` -- the prefix-reuse row.
- `TokenSampling.md` -- the device sampler Llama does not use.
