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
| Llama 3.2 3B Instruct | Llama | `Llama-3.2-3B-Instruct-fp4` | 12 GB |
| Llama 3.1 8B Instruct | Llama | `Llama-3.1-8B-Instruct-fp4` | 12 GB |
| Gemma 4 12B | Gemma | `gemma-4-12b-it-fp4` | 12 GB |
| Gemma 4 26B-A4B | Gemma (mixture of experts) | none -- in `Mila/Src`, gated, unpublished | 16 GB (~13 GB at FP4) |
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
| Correct at the trained context length | -- | Y | Y | Llama's `rope_scaling` is stored and never applied; the Rope component has no scaling knob (`Vnext.md`, "Llama's long-context scaling factor is stored, printed, and never used") |
| Flash-attention prefill | -- | Y | Y | Llama's attention scratch spans the full context (`Llama.ixx:380`) |

### 3.2 Weights and memory

| Capability | Llama | Gemma | Qwen | Anchor |
|---|---|---|---|---|
| FP4, FP8 and BF16 weights; quantize-on-load | Y | Y | Y | `QuantizationDispatch.ixx` |
| Embedding table and output head quantized with the body | -- | Y | Y | Llama's are always BF16 (`Llama.ixx:86`, `:88`): about 1.4 GiB on 3.1 8B |
| Sub-4-bit codebook weights | -- | -- | Y | the fitting and packing tools are Qwen's (`Tools/Quantization/pack_qwen.py`, `qwen_plan.py`); the dispatch is Qwen's own (`dispatchQwenWeightPlan`) |
| KV-cache compression | -- | -- | -- | `PerChannelKvFp8<>` specified, not built (`QuantizationDispatch.ixx:103`) |
| Scoring head width honoured (`withLanguageModelHeadPositions`) | -- | -- | Y | accepted on every request, read only by Qwen -- a setting silently ignored |
| Sequence log-likelihood (`scoreTokens`) | -- | -- | Y | `QwenModel.ixx:317`; what an in-library perplexity gate measures |
| Deployment planning, exact footprint | Y | Y | Y | `Deployment.md` Phases 1 to 4 |

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
| Token-for-token agreement with HuggingFace, in the suite | -- | Y | Y | `GemmaModel.Parity.Cuda.cpp`, `QwenModel.Parity.Cuda.cpp`; Llama has only `LlamaModel.Footprint.Cuda.cpp`, and no parity script under `Tools/Converters/Llama` |
| Throughput harness | -- | Y | Y | `GemmaModel.Rates.Cuda.cpp`; Qwen's `DISABLED_PrefillRate` / `DISABLED_DecodeRate*` |
| Quality measured across the planner's range | -- | -- | partial | Qwen to 16K (`Qwen3.8.md` §8) while the planner chooses up to 64512 for cb2-3 on 16 GB; Gemma unmeasured above 131072 against a header of 262144; Llama as the RoPE row above |

### 3.5 Applications

Not a model property -- streaming is the display's (`ModelHandle.md` 3.2) -- but held to the same rule.

| Capability | Llama | Gemma | Qwen |
|---|---|---|---|
| Chat streams the reply as it is generated | -- | Y | -- |

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
| Gemma 4 26B-A4B | Mixture of experts | Y, in `Mila/Src`, unpublished | yes, ~13 GB at FP4 | BACKLOG, "Announce the Gemma 4 26B-A4B mixture of experts" |
| Qwen 3.8 27B | Vision tower (27 layers, width 1152) and multimodal positions (mrope) | -- out of scope for the first chassis (`Qwen3.8.md` §1) | not priced; tight beside the FP4 build (13.2 GB of weights on device) | nowhere |
| Qwen 3.8 27B | Multi-token prediction head, one layer (~0.45 B) | -- both converters skip `mtp.*` | not priced | nowhere |
| Qwen 3.8 27B | 262144-token context, 1M by YaRN | the planner allows 262144; quality measured to 16K | only with KV-cache compression, and not at FP4 | BACKLOG, "Qwen's perplexity gate has only been run to 16K"; KV compression entry |
| Qwen 3.8 | Smaller dense members | -- | yes | BACKLOG, "The smaller dense Qwen members were never built" |
| Llama 3.1 / 3.2 | 128K context by Llama 3 RoPE scaling | -- the factor is never applied, for either (32.0 on 3.2, 8.0 on 3.1) | 8B: only with KV compression | `Vnext.md` |
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
  three, and Llama ignores `top_p`. That is the failure `QuantizationDispatch.ixx` exists to prevent -- a
  request that reads as honoured and is not -- and it is worse than a refusal.
- **Quality validation stops short of what the planner now chooses** for all three families (3.4). This is
  the gap `Deployment.md` Phase 3 opened, and it is the one a user meets first.
- **Four upstream capabilities are recorded nowhere** (section 4): Qwen's vision and MTP head, Llama's
  built-in tools, and until this document the 16 GB rule itself.

---

## 7. Keeping It True

- The matrix changes in the same commit as the work that changes a cell.
- **A new capability gets a row with every family's cell filled** -- Y, --, or n/a argued in section 5 --
  in the commit that adds it to the first family. That is the step whose absence produced this document.
- **A new family gets a column with every row filled** before it is announced.
- An upstream capability the base model gains, or one found missing, gets a section 4 row whether or not
  work is planned.

---

## 8. Open Decisions

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
2. **Llama above the context it is accurate at, until RoPE scaling lands.** The planner chooses 13312 on
   the 4070 and 33792 on the 5060 Ti for Llama 3.1 8B. Either implement the scaling first, or lower the
   ceiling Llama reports until it lands -- which is a stopgap, and says so where it is written.
3. **The order of the three kinds of work.** Recommend correctness first (RoPE scaling, then quality across
   the planner's range), then parity inside Mila (sampling, flash prefill, tables, head width,
   `scoreTokens`), then grammar -- which is also `ModelHandle.md` Phase 2, so it builds the handle's
   foundation at the same time.
4. **Llama 3.1's built-in tools.** In scope under section 1, but they are a tool-execution convention
   (`ipython`) as much as a grammar. Decide with the agent core's spec whether they are a Llama capability or
   a `Mila::AI` one.
5. **Gemma's drafter and QAT implementations under the 16 GB rule.** Section 1 puts both in scope, but
   `ROADMAP.md`'s Gemma 4 Complete text makes only their measurements this release. Recommend keeping it
   that way: each measurement carries a stop condition that can decline the work, so admitting the
   implementation first would commit to something the number may reject.

---

## 9. Relationship To Other Specs

- `ModelHandle.md` -- its capability sources (3.2) and protocol (3.4) are this matrix's grammar and
  capability rows; its Phase 2 closes section 3.3.
- `Deployment.md` -- the planner whose range section 1's third rule and row 3.4 hold quality to.
- `Qwen3.8.md`, `Gemma.md`, `Gemma4MoE.md` -- each family's design of record; section 4 points into them.
- `SpeculativeDecoding.md` -- the draft design behind the MTP rows.
- `PromptCaching.md` -- the prefix-reuse row.
- `TokenSampling.md` -- the device sampler Llama does not use.
