# Mila — Backlog

**Work committed to the release in flight, and nothing else.** Narrative and success criteria are in
[ROADMAP.md](ROADMAP.md); everything upstream of the commitment is in
[`Mila/Issues/`](Mila/Issues/README.md). Completed work is in the git history.

**Admission:** name the ROADMAP success criterion that fails if this never ships. If you cannot,
it belongs in `Mila/Issues/`.

Each `###` bucket is a theme of the release in flight, its name matching the ROADMAP section — the
only join. **This file describes a publish (a minor); a patch tag has no backlog.**

**Entry shape**, one level deeper than the same shape in `Mila/Issues/`:

```markdown
#### Llama throws away its long-context scaling factor

`open` · `llama`

The load path reads `rope_scaling` from the model metadata and discards it — the
`.withRoPEScalingFactor()` call at `Llama.ixx:703` is commented out, for a reason recorded as
unclear. 3.1 8B cannot reach the context length it advertises.
```

The **heading states the problem** and has to read cold: no term that exists only inside Mila.
The **metadata line** carries status — `open`, `in progress`, `done` — then area tags from
[Tags.md](Mila/Issues/Tags.md); status never appears in the prose. The **body** carries whatever
detail the work needs and ends in an anchor, unless the finding is an absence, in which case say so
rather than inventing a location.

**The gate is the entry count, and it only goes down.** A release in flight burns down, so an
addition is paired with a removal, or it is a deliberate admission that scope grew. **What bounds
this file is the release it describes and the date that release carries, not a fixed number.**
v0.21.0 opens deliberately large against 2026-10-31: whatever has not landed by then goes back to
[`Vnext.md`](Mila/Issues/Vnext.md) and the release ships narrowed, rather than the date moving to
accommodate the list. A backlog that grows while the date holds is the signal to drain it early.

**Done means deleted**, in the same commit as the work — `done` is a working-tree marker and is
never committed.

---

## Current release (v0.21.0)

The whole arc, implemented in three stages in the order the buckets appear: the handle first because
everything after it is cheaper on the other side, then Qwen 3.8 as the smaller family, then Gemma 4
ending in modality. The stages are a sequence, not a partition — an item is workable whenever its
own blockers are gone.

### Model Handle

#### One typed model handle + factory

`open` · `architecture` · `mila-src`

The architecture-to-concrete erasure exists three times in two languages — Chat's `ModelVariant`,
the binding's `*Session` classes, MIS's `ModelFamily` — which is why GPT-2 is missing from MIS.

Lands in the runtime-adjacent native agent core; sequencing in `MilaProductFamily.md` Open
Decision 2. ROADMAP already calls it the first work after the v0.20 tag and a precondition for
every model entry below it.

### Adaptor Convergence

#### The inference server refuses a model the chat harness runs

`open` · `adaptors` · `binding` · `gpt`

GPT-2 loads in Chat and is rejected by MIS — not by decision, but because the server's own family
enum was never extended to it. `model_worker.py`'s session pick branches on family, and so does
`/v1/models`; three latent `else means llama` sites in the same shape were fixed at `rc.1+31`, each
correct only while there were exactly two families. Read from outside, the gap looks like a policy
about which models are servable.

This is the visible half of the handle landing: the enum goes, and the gate is that every
architecture Chat runs is servable.

#### The Python binding carries one session class per family

`open` · `binding` · `api`

The per-family session types under `Mila/Bindings/` are the second of the three bridges, and the one
a Python consumer actually meets. They must consume the handle at session depth without the binding
gaining a component-level surface — it is consumer-blind by design and stays that way.

The gate is that adding a family adds no binding type. The finding is a duplication rather than a
defect at one line, so it has no single anchor.

#### MIS tool calling beyond the three flows the release names

`open` · `gemma` · `adaptors`

N sequential distinct tool calls within one turn, and channel-content parser polish. Moved from the
v0.20 backlog at `rc.1+21`: the release criterion names plain-chat, single-tool and
tool-result-resume only.

### Qwen 3.8 Complete

#### Publish the 2.82-bit Qwen 3.8 27B so a 12 GB card can run a 27B model

`open` · `distribution` · `quantization` · `qwen`

The package exists and is validated; nobody outside can fetch it.
`https://huggingface.co/api/models/mila-llm/Qwen3.8-27B-cb2-3` answers 401 anonymously, where
every other published model answers 200. It is 11.1 GiB against 15.1 GiB for the FP4 build of the
same model, which is what would let a 12 GB card run a 27B model at all — the FP4 build needs 16 GB.

Held out of v0.20.0 deliberately (Todd, 2026-09-21): a capability with no published package is not
announced, the same rule that kept the Gemma 4 MoE work unannounced. The cost is that
**`ROADMAP.md`'s v0.20 headline claim was written around this model** and had to be narrowed to the
FP4 build at 16 GB. Publishing it is what makes the stronger claim sayable, so it wants to land
early in the cycle rather than at the end. The fitted source is gitignored and not reproducible
from the repo, so no reader can work around its absence.

#### Qwen's KV cache is uncompressed on a card its weights already fill

`open` · `qwen` · `quantization` · `mila-src`

`PerChannelKvFp8<>` is specified and not built; `QuantizationDispatch.ixx:103` records that KV-cache
compression is not a live knob. On a 27B at FP4 the weights take the card, so the KV cache is what
buys context back, and halving it is the difference between the context length that fits and the one
the model is sold at.

The freed margin is then a decision rather than a windfall — more context, or more bits where the
quality gate says they are worth most. v0.20 deliberately did not pre-empt that, so it is part of
this work rather than a consequence of it.

#### Qwen answers in one block after a long silence

`open` · `qwen` · `adaptors`

`FamilyTraits::streaming_capable` is false for Qwen (`Chat.FamilyTraits.ixx`), because the harness
routes tokens by Gemma's four control-token ids and nothing else has them. Qwen has one marker pair,
`<think>`/`</think>`, which is enough to separate reasoning from answer; the per-token router has
simply not been written for it.

Not a gap against any other model — Llama and GPT-2 are buffered too, and Gemma is the only family
that streams. It matters most on Qwen because a 27B is the longest wait to sit through with nothing
on screen.

#### The smaller dense Qwen members were never built

`open` · `qwen` · `mila-src`

The family shipped as the 3.8-27B hybrid alone, so "Qwen 3.8" names one model rather than a family.
The dense members reuse the Llama blocks rather than the DeltaNet chassis, which is what makes them
cheap next to everything already landed.

The finding is an absence — there is no partial implementation to point at. Gate: a dense member
decodes token-for-token against HuggingFace at BF16 and FP8.

#### Qwen's perplexity gate has only been run to 16K

`open` · `qwen` · `measured`

From 8K to 16K the FP4 oracle improves 7.2% while the 2.82-bit plan improves only 3.4%, so the
quantized arm captures about half the benefit of the extra context — the compounding signature the
recurrent layers make plausible. It was held out of v0.20 because nothing claimed a context above
16K: the model card stopped there and recorded the ratio as flat from 1K.

**The condition it was waiting on has fired.** KV-cache compression above buys context back, and a
release that advertises a longer one has to gate quality at that length rather than at the length
the old card stopped at. The table and caveats are in `Qwen3.8.md` §8 item 9;
`DISABLED_QualityGateAcrossContextLengths` is the harness.

#### The head's two paths disagree in the third decimal, so perplexity must fix the width

`open` · `qwen` · `measured`

Same weights, same corpus: width 1 (the decode matvec) and width 64 (the W4A8-FP8 GEMM) do not
produce identical numbers. Small, but head width is part of the measurement protocol rather than a
free performance knob, so both arms of a quantization comparison have to use the same one.

Probably already recorded at `Qwen3.8.md:509` and `:546` — verify, and if so this entry is a
duplicate and should be deleted rather than worked.

#### A Qwen load test discards a `[[nodiscard]]` status and warns on every build

`open` · `qwen` · `ci`

`QwenModel.Load.Cuda.cpp:205` calls `model->generate(...)` for its side effects inside a lambda,
producing C4834. The status is the only channel reporting why generation stopped, so a test that
ignores it cannot tell a completed run from an aborted one — and the other call sites in the same
file (`:168`, `:523`, `:813`) already bind it.

Assert it instead of casting it away. Also one entry on the warnings-as-errors ratchet's bill.

### Gemma 4 Complete

#### Two questions decide the size of Gemma's image path, and neither is answered

`open` · `gemma` · `mila-src`

Answer both before building, because one of them reaches every attention path in the library.

**Do image soft tokens attend bidirectionally within a prefill?** Every attention path in Mila is
causal. If a bidirectional span is required, that is a mask change across the attention machinery
rather than a component added beside it, and it resizes everything downstream of it.

**What is the position scheme for an image span?** 280 soft tokens arrive at `model_patch_size` 48
inside a sequence whose positions are otherwise one per token.

The finding is an absence: no code answers either question today, and the entry below cannot be
honestly sized until they do. Whatever the answers are, they are recorded in `Gemma4.md` rather than
in the implementation that assumes them.

#### Gemma 4 12B is multimodal and Mila drops its image and audio weights

`open` · `gemma` · `mila-src` · `adaptors`

The 12B is **encoder-free** (`Gemma4UnifiedForConditionalGeneration`): no vision tower. Images enter
through `vision_embedder` (`patch_dense`, `patch_ln1`, `patch_ln2`, `pos_norm`) and
`embed_vision.embedding_projection` into the decoder itself — `patch_size` 16, `model_patch_size` 48,
280 soft tokens, `mm_embed_dim` 3840; audio through `embed_audio` (`audio_embed_dim` 640).
`convert_weights.py:116` skips `model.embed_vision.` and `model.embed_audio.`. Open before sizing:
whether image soft tokens attend bidirectionally within a prefill (every Mila attention path is causal),
the position scheme for image spans, and the audio front end. Work: patch embedding component, soft-token
placement before layer 0, image decode/resize/normalize (a vendored decoder is a NOTICE entry; decode
belongs in adaptors), template image tokens, Chat attach, MIS image content blocks for both protocols,
converter keeps the embedders, manifest declares modality, footprint counts image prefill. Gate:
embedder parity against HuggingFace, then token-for-token on an image prompt.

#### Announce the Gemma 4 26B-A4B mixture of experts

`open` · `gemma` · `distribution`

Landed in `Mila/Src` during rc.1 — the router and expert bank on CPU and CUDA, wired into
`GemmaModel` with an FP4 expert bank and a streaming converter, gated against HuggingFace at BF16
and FP4. It appears in no README capability row, no CLAUDE.md target and no release note, because
no published package uses it, so a user cannot run it.

Held out of v0.20.0 deliberately (Todd, 2026-09-21) on the same rule as the entry above. What it
needs to become announceable is a published package and a capability row, not more implementation.
Expect this to be rediscovered by anyone grepping the tree for MoE and wondering why it is silent.

#### Gemma loses its own reasoning between tool calls in a turn

`open` · `gemma` · `adaptors`

Google's multi-turn rule is to strip thoughts from *prior* turns and keep the current turn's.
`extractAnswer` (`Gemma.Protocol.ixx:1288`) removes every channel span from a response rather than a
leading run, so a model working through a multi-step tool sequence starts each step without the
reasoning that led to it. Moved out of v0.20 at `rc.1+24` (Todd): a behaviour change inside Gemma's
protocol is too late in the cycle.

#### A malformed Gemma tool call parses as a call with no arguments instead of failing

`open` · `gemma` · `mila-src`

`parseArguments` (`Gemma.Protocol.ixx:480`) breaks out of its loop at the first key not followed by
`:` and returns what it has accumulated, so a partial parse is indistinguishable from a call that
genuinely took no arguments. Seen driving Codex through MIS: the model emitted
`call:exec_command{cmd="cat line_count.txt"}` — `=` and plain quotes, off the trained grammar — and
`gemma_parse_tool_call` returned `{'name': 'exec_command', 'arguments': '{}'}`. Codex rejected the
empty call and the model retried correctly, so that flow recovered; a client that executes `{}`
would not. Qwen's bridge treats a malformed call as prose, which is the behaviour to match.

Held for the same reason as the entry above (`rc.1+24`): a behaviour change inside Gemma's protocol
is too late in this cycle. `Chat.ToolCallParser.ixx`'s over-eager `[` test — "Any response
containing a bracket enters the tool-call parser" below — is the same failure shape in the adaptor
rather than the library.

#### The Gemma parity script compares two different precisions and calls it parity

`open` · `gemma`

`gemma_greedy_parity.py:70` loads Mila through the binding's FP4 default and diffs it against a
BF16 HuggingFace reference, so any divergence it reports mixes quantization error with a real
defect and a clean run proves less than it appears to. `GemmaModel.load` now takes `quantization=`,
so the honest comparison is one argument away — on a card that can hold a BF16 12B. Either way the
script should state which precision it ran.

Full path: `Mila/Tools/Converters/Gemma/gemma_4_BF16/gemma_greedy_parity.py`.

#### `gemma_protocol.py` is dead and can be deleted

`open` · `gemma` · `binding`

Its 856 lines are superseded by `Gemma.Protocol.ixx` plus `gemma_bridge.py`, nothing imports it, and
it carries a header saying so. Kept on disk under the retire-don't-delete rule, which is the correct
state for now; removing it is a one-file deletion whenever the reconciled grammar has been driven
long enough to be sure.

#### Gemma 4 12B decodes one token per forward pass, and Google ships a drafter for it

`open` · `gemma` · `perf` · `mila-src`

Every Gemma 4 size ships a dedicated draft model for speculative decoding (ai.google.dev/gemma/docs/core,
read 2026-09-17). `SpeculativeDecoding.md` is a DRAFT that places Google's drafter last (phase E) behind
prompt lookup and EAGLE; with a published drafter it moves forward. First step, before any code: measure
what a K-token verify forward costs against K decodes on the 5060 Ti with today's prefill path, since
FP4 decode is bandwidth-bound and the verify goes through prefill GEMM — if K=4 costs near 4 decodes
there is no win. Then pin the drafter checkpoint layout (tensor names, how it combines the target's last
hidden state). Work: draft/verify/accept/rewind loop in `generate()`, logits at every verify position,
wrap-safe rewind on the sliding ring (`rewindKvCache` exists; speculative wrap unverified), drafter KV
cache, the target's last hidden state exposed, converter/footprint/Chat stats. Gate: greedy output
token-for-token identical to plain decode.

#### Google's quantization-aware 4-bit Gemma 4 cannot be loaded without losing what QAT bought

`open` · `gemma` · `quantization` · `mila-src` · `distribution`

`google/gemma-4-12b-it-qat-w4a16-ct` (compressed-tensors, read 2026-09-17): `pack-quantized`, `int`,
`num_bits` 4, `symmetric`, `strategy` group, `group_size` 32, targets `Linear`; `lm_head` and the
image/audio embedders ignored. Mila's FP4 is E2M1 at group 128 — re-quantizing QAT weights onto that
grid discards the training that fitted them to the int4 grid. Measure first: wikitext perplexity of
the QAT checkpoint against Mila's published FP4 and BF16; if QAT does not beat FP4, stop. Work: a
`PerGroupInt4<32>` symmetric policy (OperationTraits rows, W4A16 GEMM with an int4 value table and
group-32 scales — the FP4 kernel's nibble lookup is the part that changes), ExportArtifact transcoding
int32 `pack-quantized` into Mila's nibble layout with `mila_quantization` metadata, footprint (4.5 bits
per weight with 16-bit scales against FP4's 4.25 — scale dtype unverified). The embedding stays Mila's
FP8 tied table, which the QAT build leaves unquantized. Publish as its own model. Depends on the
compressed-tensors import below.

#### Mila cannot import the format most quantized models on the Hub are published in

`open` · `quantization` · `distribution` · `mila-src`

compressed-tensors (the vLLM project's format) is safetensors plus a `quantization_config` in
`config.json`: a `format` (`pack-quantized`, `int-quantized`, `float-quantized`,
`nvfp4-pack-quantized`), per-group schemes (bits, `int`/`float`, symmetric, strategy tensor/channel/
group/block, `group_size`), and `targets`/`ignore`. A packed int4 Linear carries `weight_packed` (int32,
eight values each), `weight_scale` per group, `weight_shape`, and `weight_zero_point` only when
asymmetric. Packing order and scale dtype are from memory — settle them with a safetensors header read
before code.

Import it in `ExportArtifact` only, as a transcode into Mila's own safetensors with
`mila_quantization` metadata; the load contract, loaders, store and adaptors stay untouched, and a
layout mismatch surfaces at export rather than at load. Mapping, per format: `pack-quantized` int4
symmetric -> a new `PerGroupInt4<G>` (first consumer: the Gemma 4 QAT entry above);
`float-quantized` FP8 per-channel -> the existing `PerChannelFp8` (check scale shape and dtype agree);
`nvfp4-pack-quantized` -> the native NVFP4 direction on SM120 (`Fp8ActivationPrefill.md`). Refuse any scheme with no matching policy, naming the scheme.

#### A parity script tells the reader to diff against a debug flag that no longer exists

`open` · `gemma` · `docs` · `observability`

`kGemmaDumpActivations` is gone from `Mila/Src`, but
`Mila/Tools/Converters/Gemma/gemma_4_BF16/hf_gemma_activation_dump.py:4` still names it as the
thing to compare its output with. Anyone following the script for a parity investigation starts by
looking for a flag that is not there.

The replacement is `LanguageModel::observe` over `"*.tf_layer_*"`.
`GemmaModel::fingerprintPrefill` is **not** the substitute — it localizes a NaN rather than
comparing per-layer activations.
