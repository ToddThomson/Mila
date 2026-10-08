# Qwen 4

The Qwen 4 architecture (`qwen4_exp`): what it adds to the Qwen 3.8 hybrid, which of Mila's Qwen 3.8 pieces
carry over, and what a Qwen 4 27B port has to build.

*Status: draft, 2026-10-08; no code. Written ahead of the Qwen 4 27B release, expected around November 2026
(Todd, 2026-10-08). No Qwen 4 27B checkpoint exists yet. Everything here is read from the architecture's one
open-weights release, `Qwen/Qwen3.8-Flash-Next`, which its model card calls "this experimental preview of the
architecture that will underpin Qwen4", and from the reference implementation in `transformers` 5.16.0
(`models/qwen4_exp/modeling_qwen4_exp.py`). Section 9 is the implementation plan; its phases 0 to 2 need no
27B checkpoint. Section 10 lists what only the 27B's own `config.json` can settle; until it is read, every 27B
number here is a placeholder.*

---

## 1. Sources and Their Standing

| Source | What it settles | Standing |
|---|---|---|
| `Qwen/Qwen3.8-Flash-Next` `config.json` | The architecture's fields and one set of values | Read 2026-10-08 |
| `Qwen/Qwen3.8-Flash-Next` `model.safetensors.index.json` | Tensor names and which modules exist (1,658 tensors, 359,999,963,128 bytes in BF16) | Read 2026-10-08. Shapes were **not** read: the HF weights CDN is outside the cloud session's network policy. Every shape below is derived from the config and the reference code |
| `transformers` v5.16.0 `modeling_qwen4_exp.py`, `configuration_qwen4_exp.py` | The arithmetic, exactly | Read 2026-10-08. **This is a reference the Qwen 3.8 port did not have for its MTP head.** It ignores `mtp.*` here too |
| Model card | Parameter counts, layout, sampling, chat controls | Read 2026-10-08 |
| `chat_template.jinja` | Turn format, `reasoning_effort`, `preserve_thinking` | Read 2026-10-08 |

**Naming.** "Qwen3.8-Flash" is Qwen Cloud's API-only product built on Flash-Next. It has no open weights;
`Qwen/Qwen3.8-Flash` does not exist on the Hub. The architecture is `Qwen4ExpForConditionalGeneration`, with
text config `qwen4_exp_text`. This document calls the architecture **Qwen 4** and the preview **Flash-Next**.

---

## 2. The Architecture (Flash-Next values)

| Field | Flash-Next | Qwen 3.8 27B, for contrast | Note |
|---|---|---|---|
| `hidden_size` | 2560 | 5120 | |
| `num_hidden_layers` | 48 | 64 | |
| `layer_types` / `full_attention_interval` | 3 linear : 1 full, interval 4 | same | Full attention at layer indices 3, 7, ... 47 |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 24 / 2 / 256 | 24 / 4 / 256 | |
| Query projection | 2 x 24 x 256, per-head [query, gate] | same | The `view(..., head_dim * 2)` then `chunk` layout the 3.8 converter already de-interleaves |
| Attention output gate | sigmoid, hard-coded in the reference | sigmoid | |
| `partial_rotary_factor`, `rope_theta` | 0.25 (64 of 256), 1e7 | same | |
| `rope_parameters.mrope_section` | [11, 11, 10], interleaved | same | Text-only input reduces it to plain RoPE |
| Linear attention: key heads / value heads / head dim / conv | 16 / 48 / 128 / 4 | same | `mamba_ssm_dtype: float32` |
| **`output_gate_type`** | **`"sigmoid"`** | `"swish"`, never read | **Live in `qwen4_exp`**: it is the DeltaNet gated norm's activation (Section 3.2) |
| `hc_count` / `hc_lowrank` | 4 / 320 | none | Gated Residual (Section 3.1) |
| `indexer_n_heads` / `indexer_kv_heads` / `indexer_head_dim` | 4 / 1 / 128 | none | QSA (Section 3.3) |
| `indexer_budget` / `indexer_compress_ratio` | 2048 tokens / 4 | none | 512 blocks kept |
| `ngram_size` / `heads_per_ngram` / `ngram_vocab_size_base` | 3 / 8 / 20,000,000 | none | N-gram embedding (Section 3.4) |
| `ple_layer_ids` / `ple_embed_dim` / `ple_conv_kernel_size` | [2] / 2560 / 4 | none | **1-based**: the reference injects before layer *index* 1 |
| `split_ngram_parts` / `make_ngram_vocab_size_divisible_by` | 128 / 128 | none | The table ships as 128 shards |
| `num_experts` / `num_experts_per_tok` / `moe_intermediate_size` | 512 / 10 / 640 | dense FFN, 17408 | MoE on **every** layer (Section 3.5) |
| `shared_expert_intermediate_size` | 640 | none | One shared expert, with a scalar sigmoid gate |
| `norm_topk_prob` | `True` (default; absent from the file) | none | |
| `vocab_size` / `tie_word_embeddings` | 248,320 / false | same | |
| `max_position_embeddings` | 262,144 | 262,144 | Card: "extensible up to 1,000,000" |
| `mtp_num_hidden_layers` | 1, hybrid, full attention | 1 | Out of scope, as for 3.8 |
| Vision tower | 27 layers, width 1152, out 2560 | 27 layers, out 5120 | Out of scope, as for 3.8 |
| Final norm | **none**: the stream mixer feeds `lm_head` directly | `model.norm` | Section 3.6 |

The model card counts Flash-Next as 125B parameters (6B active), plus 51B of n-gram embedding and 4B of MTP.
The index's 360 GB of BF16 agrees: 180B parameters.

---

## 3. What Is New

Every formula is the reference's. `n = hc_count`, `H = hidden_size`. `RMSNorm` without a qualifier is the
family's unit-offset form, `x * rsqrt(mean(x^2) + eps) * (1 + w)` computed in FP32. "Grouped" means each
H-wide slice of an n*H vector is normalized on its own, with one n*H weight across all slices.

### 3.1 Gated Residual (hyper-connections)

The residual stream is **n*H wide**: 10,240 on Flash-Next. There is no `input_layernorm` and no
`post_attention_layernorm`; the index has neither, and the gated residual's norm takes their place.

```
S_0 = [e, e, ..., e]                                         # token embedding repeated n times

for each sublayer f in (mixer, moe) of each layer:
    N  = GroupedRMSNorm_hc(S)                                # weight hc_norm [n*H]
    r  = sigmoid( up( silu( down(N) / n ) ) )                # down [rank, n*H], up [n*H, rank]
    x  = mean_over_streams( r * N )                          # [H], the sublayer's input
    y  = f(x)                                                # [H]
    g  = 2 * sigmoid( inject(N) / n )                        # inject [n, n*H]; one scalar per stream
    S  = S + concat_i( g_i * y )                             # adds to S, not to N

x_final = hyper_connection_mixer(S)       # the same N, r and mean, with no inject
logits  = lm_head( x_final )
```

Consequences for Mila:

- **The block skeleton changes for every block kind.** `QwenAttentionBlock` and `QwenDeltaNetBlock` are
  pre-norm, single-stream blocks: `x <- x + f(norm(x))`. Under Qwen 4 a block reads and writes an n*H state,
  and its sublayer reads a reduced H-wide input. The mixers themselves (GQA with output gate, Gated DeltaNet)
  are unchanged.
- **One new component, `GatedResidual`**, with two modes: *read and inject*, used twice per layer, and
  *read only*, used once by the final mixer. Its weights are small: `down` + `up` + `inject` = 2 * 320 * 10240
  + 4 * 10240 = 6.6M per instance on Flash-Next, 97 instances including the mixer, 0.64B in all.
- **RmsNorm needs a grouped mode**, or a reshape that `RmsNorm`'s weight shape tolerates: normalize each
  `H` slice, scale by an `n*H` weight. The same grouped norm appears in PLE (Section 3.4).
- **Activations are n times wider** wherever the stream itself is held: the residual slots, the norm, the
  `down` input. A sublayer's own buffers stay H-wide.
- **Precision.** The `r` and `g` gates are sigmoids that compound across 96 sublayers, which is the argument
  `DeltaNetGating` makes for holding a and b out of quantization (`Qwen.DeltaNetBlock.ixx`, *PER-ROLE
  PRECISION*). Proposed: `down`, `up`, `inject` are a new plan role that defaults to BF16. They are 0.5% of
  Flash-Next's 125B core parameters.

### 3.2 DeltaNet Gated Norm Activation

`Qwen4ExpTextRMSNormGated(head_v_dim, activation = config.output_gate_type or config.hidden_act)`. On
Flash-Next that is **sigmoid**. `configuration_qwen4_exp.py` accepts only `sigmoid` and `silu`. Mila hard-codes
`AttentionOutputGate<..., ActivationType::Silu>` for this gate (`Qwen.DeltaNetBlock.ixx:287`) and describes
`output_gate_type` as dead. That is true for Qwen 3.8 and false here. The fix is a template parameter, since
the activation selects a type. The weight convention is unchanged: raw weight, not unit-offset.

The DeltaNet's tensors keep their 3.8 names (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`,
`conv1d`, `A_log`, `dt_bias`, `norm`, `out_proj`), so the 3.8 converter's `in_proj_qkv` and `conv1d` split
carries over unchanged.

### 3.3 Qwen Sparse Attention (QSA)

Every full-attention layer carries an **indexer** that restricts each query to a budget of key blocks:

```
qk       = index_qk_proj(x)                        # [ (4 + 1) * 128 ]
q_h      = RoPE( RMSNorm_q(qk[h]) ), h = 0..3      # q_layernorm [128]
k_raw[t] = qk[4]                                   # cached per token: no norm, no RoPE

for each query at position p:
    visible     = 0..p
    blocks      = floor( (p + 1) / 4 ) complete blocks of 4 tokens: block b = tokens 4b .. 4b+3
    kc[b]       = RoPE_at(4b)( RMSNorm_k( mean_fp32( k_raw[4b .. 4b+3] ) ) )   # k_layernorm [128]
    score[b]    = sum_h relu( q_h . kc[b] ) / sqrt(128)
    selected    = top_512(score) blocks  +  the tail, tokens 4 * blocks .. p
    attention over the selected tokens only (the ordinary GQA with output gate)
```

- **QSA equals dense causal attention exactly while a query sees at most 2051 tokens** (512 full blocks plus
  a 3-token tail): the top-512 then selects every block. This is the first gate's foundation. A chassis with
  Mila's existing dense GQA is *exactly* Qwen 4 for every position below 2051, so the parity of everything
  else can be settled before the indexer exists.
- **The compressed key `kc[b]` depends only on its block**, never on the query. The reference recomputes it
  for every query; Mila computes it once when a block completes and caches it. That makes the indexer's cache
  `[T/4, 128]` plus at most 3 raw keys, a quarter of the reference's `[T, 128]`.
- **RoPE on the indexer.** The reference passes the model's cos/sin, built for the 64 rotated dimensions,
  to `apply_rotary_pos_emb` on a 128-wide head. By its partial-rotary convention the first 64 of 128 rotate.
  Verify this against the tiny reference (Section 8) rather than reading it from the code.
- **Ties.** `torch.topk` does not specify which of two equal scores it keeps. The gate compares selected
  *sets* and accepts a difference only where the reference's scores tie, as `MixtureOfExperts.md` §4 does
  for the router.
- **What it buys.** At 262K context, a dense decode step reads 262,144 x 2 x 256 x 2 x 2 B = 512 MiB of KV
  per full-attention layer. QSA reads 2051 rows of it, plus 65,536 compressed keys (16 MiB). Decode attention
  becomes O(budget) plus a scan that is O(T/4) at 128 dimensions.
- **What Mila builds.** An indexer component (projection, two norms, RoPE, the compressed-key cache). A
  top-k over a block count read from the device. A sparse attention: either gather the selected KV rows into
  a dense 2051-row workspace and reuse `GqaDecodeAttention`, or a block-sparse flash kernel. Prefill needs a
  per-query selection, so it is the harder half. Decision 5 in Section 10.

### 3.4 N-gram Per-Layer Embedding (PLE)

Before the layer whose 1-based index is in `ple_layer_ids`, a hashed n-gram embedding is added to every
stream:

```
history  = previous 2 token ids (state; eos when absent) ++ this chunk's ids
shift_s  = the id s positions back, or eos if that crosses the last EOS before this token
             (an EOS token belongs to the segment it ends)
for order k in {2, 3}:
    mixed_k = (shift_0 * m_0) XOR (shift_1 * m_1) [XOR (shift_2 * m_2)]      # int64, wrapping
    for each of 8 heads j of order k:
        id = floor_mod(mixed_k, P_j) + offset_j                              # P_j: primes just above 20M
E  = concat over 16 heads of table[id]                                       # 16 x 160 = 2560 = ple_embed_dim
K  = GroupedRMSNorm( key_proj(E) )           [n*H]       # norm_key
V  = value_proj(E)                           [H]
Q  = GroupedRMSNorm( S )                     [n*H]       # norm_query, over the stream state
s_i = <K_i, Q_i> / sqrt(H);  s_i = sign(s_i) * sqrt( max(|s_i|, 1e-6) )
G_i = sigmoid(s_i) * V                                    # per stream
S  = S + G + silu( conv1d_dilated( GroupedRMSNorm(G) ) )  # norm_conv; depthwise, kernel 4, dilation 3
```

- **The table is the largest tensor in the model.** On Flash-Next: about 16 x 20M rows x 160 = 51.2B
  parameters, 102 GB in BF16. A token gathers 16 rows of 320 B, so it is a gather, never a multiply.
  Device residency is out of the question. Host residency at BF16 needs about 102 GB of RAM. The candidates
  are pinned host memory at FP8 (51 GB), or a memory-mapped file with the page cache as residency. Prefill
  gathers about 32K random rows per 2K-token chunk, which is where the memory-mapped option is at risk.
  The 27B's table size is unknown (Section 10, decision 2), and it decides this.
- **The hash must match bit for bit.** Multiply in `uint64_t`, which wraps exactly as torch's int64 does,
  then reinterpret as `int64_t` and **floor**-mod: C++ `%` on a negative operand returns a negative
  remainder where `torch.remainder` does not. An off-by-one-row error here produces no NaN and no crash,
  only a quietly worse model. The gate is exact integer equality on ids.
- **The constants ship in the checkpoint**: `layer_multipliers` [3], `ngram_heads_vocab_sizes` [16] and
  `ngram_heads_offsets` [16], all int64. Nothing needs to recompute primes. Proposed: the converter moves
  these 35 integers into the config metadata, so the C++ weights path never has to carry an int64 tensor.
- **The dilated conv is new.** `CausalConv1d` has no dilation. The PLE conv needs dilation 3, so its
  rolling window is (4 - 1) x 3 = 9 positions of n*H channels.
- **The two pieces of recurrent state** (the 2-token id history and the 9-position conv window) behave like
  the DeltaNet state. They cannot rewind; they are saved and restored with it (Section 4).

### 3.5 Mixture of Experts

Flash-Next puts an MoE on every layer. **Whether the 27B does is the first question its config answers.**
Qwen's dense members have so far been dense, and a 27B whose name is its total parameter count suggests
dense. If it is dense, this section does not apply and the 3.8 SwiGLU FFN carries over.

If it is MoE, the reference's chain is:

```
logits   = router(x)                                  # bare linear [E, H]: no norm, no scale
w, idx   = top_k( softmax_E(logits) ), renormalized   # norm_topk_prob
y        = sum_k w_k * expert_idx_k(x)  +  sigmoid( shared_gate(x) ) * shared_expert(x)
```

- `RouterOp`'s selection order is Gemma's (softmax over all, select, renormalize). The routing chain around
  it is bare wiring, with no per-expert scale (`MixtureOfExperts.md` §4).
- The stacked layout `gate_up_proj [E, 2I, H]`, `down_proj [E, H, I]` is `MixtureOfExperts.md` §2 exactly,
  so it uploads directly.
- **The shared expert has a scalar sigmoid gate** (`shared_expert_gate`, [1, H]). Gemma 4's always-on branch
  has none. It is a `GatedMLP` beside the router, per `MixtureOfExperts.md` §3(c). Its width equals an
  expert's on Flash-Next, but its gate differs from a routing weight, so it stays out of the bank.
- **Limits to check**: `kMaximumTopK = 16` (`Router.cuh:10`); the Q4_0 grouped prefill handles at most 256
  experts (`Moe.cuh:89`, refused at `CudaMoeOp.ixx:134`). Flash-Next's 512 needs that limit raised or FP4.
- The sparse-layer footprint term must reach the deployment plan and Chat before an MoE model publishes
  (`MixtureOfExperts.md` §8).

### 3.6 No Final Norm

There is no `model.norm`. `hyper_connection_mixer`, the read-only gated residual of Section 3.1, reduces the
n streams to H, and `lm_head` reads it. `QwenTransformer`'s `rmsn_final` becomes that mixer.

---

## 4. State, Rewind and Replay

| State | Grows with context | Rewinds | Held by |
|---|---|---|---|
| KV cache, full-attention layers | yes | positionally | as Qwen 3.8 |
| QSA compressed keys `[T/4, 128]` + raw keys of the open block | yes | positionally, to `floor(p/4)` blocks | new |
| DeltaNet recurrent state + conv window | no | no: save and restore | as Qwen 3.8 |
| PLE id history (2 ids) + dilated conv window (9 x n*H) | no | no: save and restore | new |
| Gated residual | none | n/a | stateless |

Under decode replay (`DecodeGraph.md`), three of the new pieces need attention:

- **The compressed key is appended only when a block completes**, which is every fourth token. Under replay
  that must be a device-side condition on the decode position, never a host branch: a host `if` is recorded
  once and replayed every token.
- **The top-k reads its block count from the execution context's decode position**, not from a launch
  argument.
- **The n-gram hash reads the sampled token from device memory** and updates the id history in the same
  kernel, as the embedding gather already does.

Prefix reuse keeps the 3.8 shape: `savePosition` snapshots every non-rewinding state, and the new rows of
the table above join that snapshot.

---

## 5. Memory

Flash-Next as the worked example; the 27B column is filled in when its config is read.

| Term | Formula | Flash-Next |
|---|---|---|
| KV per token | L_full x 2 x kv_heads x head_dim x 2 B | 12 x 2 x 2 x 256 x 2 = 24 KiB (Qwen 3.8 27B: 64 KiB) |
| QSA keys per token | L_full x 128 x 2 B / 4 | 768 B (3 KiB if raw keys are cached as the reference does) |
| DeltaNet state | L_lin x v_heads x 128 x 128 x 4 B | 36 x 48 x 64 KiB = 108 MiB |
| PLE state | 9 x n*H x 2 B + 2 ids | 180 KiB |
| Expert bank | L x E x 3 x H x I | 120.8B parameters; 68 GB at 4.5 bits per weight |
| Active experts per token | L x top_k x 3 x H x I | 2.36B; 1.33 GB at 4.5 bits per weight |
| N-gram table | heads x ~base x (ple_embed_dim / heads) | 51.2B; 102 GB BF16 |
| Gated residual weights | (2L + 1) x (2 x rank x n*H + n x n*H) | 0.64B |

At 262K context Flash-Next needs 6 GiB of KV against the 27B 3.8's 16 GiB. Flash-Next itself fits no
consumer card: streaming every active expert over PCIe Gen5 x8 caps decode near 24 tok/s
(`MixtureOfExperts.md` §8 arithmetic). That is a reason to port the 27B and not the preview.

---

## 6. Where It Lives

**Proposed: a `Qwen4` family beside `Qwen`**, in `Models/Qwen4/` and `Components/Transformers/Qwen4/`.
CLAUDE.md has a family own its model, config and protocol together, and the change in Section 3.1 reaches
every block's skeleton, so a flag on the 3.8 blocks would reach every line of them. What is shared is
already a component: `GatedDeltaRule`, `CausalConv1d`, `AttentionOutputGate`, `Gqa`, `Rope`, `Router`,
`MixtureOfExperts`. The chat protocol is shared, not duplicated (Section 7).

New components, each its own module per the one-type-per-file rule: `GatedResidual` (+ config),
`NgramEmbedding` (+ config), `PerLayerEmbedding` (+ config), `QsaIndexer` (+ config), and the sparse
attention operation with its `OperationTraits` specializations. Extended: `RmsNorm` (grouped mode),
`CausalConv1d` (dilation), the DeltaNet gate activation as a template parameter.

---

## 7. Converter and Protocol

**Converter** (`Tools/Converters/Qwen4/`, beside the 3.8 one, reusing its helpers):

- Prefix `model.language_model.`, as for 3.8. Skip `model.visual.` and `mtp.` (`SKIPPED_PREFIXES`,
  `Tools/Converters/Qwen/convert_weights.py:82`).
- `q_proj` per-head de-interleave, plus the `in_proj_qkv` and `conv1d` split: as for 3.8.
- Experts: already stacked; pass through.
- **N-gram shards concatenate in numeric order**, `shard_0 .. shard_127`. The index lists them lexically
  (`shard_1, shard_10, shard_100, ...`), and a lexical concatenation is a silent permutation of the table.
  Then pad to `make_ngram_vocab_size_divisible_by`.
- The int64 buffers move to config metadata (Section 3.4).

**Chat protocol.** The turn format, `<think>` and the XML `<tool_call><function=...>` form are 3.8's, which
`Qwen.Protocol.ixx` already speaks. Two controls are new:

- `reasoning_effort` (`xhigh` default, `medium`, `low`) adds a fixed sentence of reasoning instructions to the
  system turn when thinking is on.
- `preserve_thinking`, on by default, keeps earlier assistant turns' `<think>` blocks in the history. Without
  it, only turns after the last user query keep theirs. Check what Mila's protocol does with history thinking
  before relying on either default.

---

## 8. Gates

1. **A tiny reference, built now, before any checkpoint exists.** `Tools/Converters/Qwen4/hf_qwen4_tiny_reference.py`,
   in the pattern of `Tools/Converters/Llama/hf_llama_tiny_reference.py`: a random-init `Qwen4ExpForCausalLM`
   with every norm drawn away from 1, saved, converted, captured. The geometry must exercise what a real
   checkpoint at short context would hide:
   - `indexer_budget` 8 with `compress_ratio` 4, so QSA *selects* within a 32-token capture;
   - `ngram_vocab_size_base` near 1,000, so hash collisions and the floor-mod are exercised;
   - a PLE layer that is not layer 0, and EOS tokens inside the captured sequence;
   - `full_attention_interval` 4, so one layer of each kind appears;
   - MoE with 8 experts and top-2, in a variant that matches what the 27B turns out to be.
2. **Component gates against that capture**: n-gram ids (exact integer equality), PLE output, gated residual,
   indexer selected sets, sparse attention output, then full logits at prefill and at each decode step.
3. **Dense equivalence**: a Qwen 4 network with its indexer bypassed matches the reference exactly for
   contexts of at most 2051 tokens (Section 3.3).
4. **Real checkpoint**: layer-stream parity, as `Tools/Converters/Qwen/qwen38_BF16/hf_qwen_layer_stream.py`
   did for 3.8, then the 3.8 quality and long-context measurements.
5. **Replay self-check** passes on the third decode step with QSA and PLE in the graph.

---

## 9. Implementation Plan

Six phases. **Phases 0 to 2 need no 27B checkpoint**: they are tested against the tiny reference (8.1),
which `transformers` 5.16 builds today. Phases 3 to 6 start when the 27B's files appear. Every `Mila/Src`
change needs agreement before it starts (CLAUDE.md, *Chat Harness*, API boundary), so each phase below is a
proposal to agree, not a commitment. Whether any of it joins the release in flight is triage's decision; the
capture is in `Mila/Issues/Untriaged.md`.

**Ordering principle.** Cheap and certain before expensive and conditional. A piece the preview and every
plausible 27B share comes first. Within that, an extension of an existing component comes before a new
one, because the existing component's own tests guard it. Everything that depends on the 27B's numbers,
or on whether it is dense or MoE, waits for its config.

Each phase's gate is written here, before its run, as `Qwen3.8.md`'s phases were; a result that misses it
is recorded, not re-gated.

### Phase 0 -- Reference and tooling

No `Mila/Src` change.

1. **The tiny reference** (8.1): `Tools/Converters/Qwen4/hf_qwen4_tiny_reference.py`. A dense variant and
   an MoE variant, so either 27B is covered. It records, per component, the inputs and outputs the Phase 1
   and 2 gates compare against: n-gram ids, PLE output, each gated residual's `x` and updated `S`, the
   indexer's selected token sets per query, attention output, and logits at prefill and at each decode
   step. The capture includes EOS tokens inside the sequence and a prefill split across two chunks.
2. **The converter skeleton**: `Tools/Converters/Qwen4/convert_weights.py`, reusing the 3.8 converter's
   split and de-interleave helpers. It converts the tiny reference's checkpoint, and it is written against
   Flash-Next's tensor names until the 27B's exist. It concatenates the n-gram shards in numeric order and
   moves the 35 int64 constants into config metadata (Section 7).

**Gate.** The capture is deterministic: two runs with the same seed produce identical files. The converter
writes every tensor of the tiny checkpoint exactly once and skips nothing but `model.visual.` and `mtp.`. Fed
the shards in lexical order, it refuses.

### Phase 1 -- Extensions of existing components

Each is small, family-neutral and guarded by the tests that already pass.

1. **The DeltaNet gate activation as a template parameter** of `QwenDeltaNetBlock` (Section 3.2), defaulted
   to `Silu`.
2. **A grouped mode for `RmsNorm`**: a group size in `RmsNormConfig`, each group normalized on its own, one
   weight across the full width, unit offset as configured (Section 3.1). CPU and CUDA.
3. **Dilation in `CausalConv1d`**: a `withDilation` on `CausalConv1dConfig`; the kernel steps its look-back
   by the dilation, and the retained state is (kernel - 1) x dilation positions deep. CPU and CUDA.

**Gate.**
- Qwen 3.8's tests pass unchanged, and its parity test produces bit-identical logits: the defaults are
  today's behaviour.
- A grouped `RmsNorm` whose group is the full width is bit-identical to today's. With 4 groups it matches
  the reference's `Qwen4ExpTextRMSNorm(group_size = H)` within the family's existing norm tolerance.
- A dilation-1 `CausalConv1d` is bit-identical to today's. At dilation 3 it matches `torch.nn.Conv1d` with
  that dilation, and a prefill of T tokens followed by decode steps equals one prefill of the whole sequence.

### Phase 2 -- New components

One type per module file, each with its config in its own file, its operations under `OperationTraits`
with CPU and CUDA specializations; the CPU operation comes first and is the CUDA kernel's reference.

1. **`GatedResidual`**: both modes, *read and inject* and *read only* (Section 3.1).
2. **`NgramEmbedding`**: the hash as its own operation, the gather, and the 2-id history as device state
   that the hash kernel reads and updates (Section 3.4). The table is device-resident at test scale; where
   the real one lives is Phase 6.
3. **`PerLayerEmbedding`**: composes 2, Phase 1.2 and Phase 1.3 (Section 3.4).
4. **`QsaIndexer`**: the projection, its two norms and RoPE, the compressed-key cache with its open block,
   and the block top-k, which reads the block count from the decode position (Section 3.3).
5. **Sparse decode attention**: gathers the selected KV rows into a dense workspace of at most budget + 3
   rows and runs the existing decode attention over it. Decode only; sparse prefill is Phase 5.

**Gate.**
- `GatedResidual`: `x` and `S` match the capture in both modes.
- `NgramEmbedding`: ids **equal** the capture's, every one, across the EOS tokens and the chunk boundary;
  embeddings then match exactly, since they are a gather.
- `PerLayerEmbedding`: output matches the capture, including a decode step after a chunked prefill.
- `QsaIndexer`: the selected set of every query equals the capture's, except where the reference's scores
  tie (Section 3.3, *Ties*).
- Sparse decode attention: output matches dense attention restricted to the captured selection.
- Under replay, components 2, 4 and 5 each pass a third-decode-step self-check on their own, given the
  position by `context->setDecodePosition( p )` (CLAUDE.md, *Quantization Pipeline*).

### Phase 3 -- Release day

Starts when `Qwen/Qwen4-27B`, or whatever the 27B is named, appears on the Hub.

1. Read its `config.json`, weight index, chat template and model card. Diff its `modeling_qwen4_exp.py`
   against 5.16's: a changed formula is a changed component, and a Phase 2 gate catches it once the tiny
   reference is re-run.
2. Close Section 10's questions 1 to 4. Add the 27B's column to Section 2 and its numbers to Section 5.
3. Re-run the tiny reference against the release's `transformers`. Any Phase 1 or 2 gate that fails is the
   first work of Phase 4.
4. The fit analysis: the device budget per context length, and the n-gram table's size against host memory.
   This settles questions 2 and 7.

**Exit.** No question 1 to 4 is open, and the spec says what the 27B is.

### Phase 4 -- The chassis, dense attention

1. `Qwen4Config`, `Qwen4AttentionBlock` and `Qwen4DeltaNetBlock` over the n-stream state, with dense GQA
   in place of QSA. `Qwen4Transformer`: expands the embedding into the streams, runs the PLE layer and the
   blocks, collapses through the final mixer into `lm_head`. `Qwen4Model` with `planDeployment` and `load`
   per the other families.
2. The feed-forward: the 3.8 SwiGLU if the 27B is dense; if it is MoE, the bare router chain, the shared
   expert's sigmoid gate, and the bank's expert-count limit raised for the formats the plan uses (Section 3.5).
3. The converter finished against the 27B's real tensor names and shapes.
4. A BF16 load first, quantize-on-load where it fits, as the 3.8 bring-up did.

**Gate.**
- Tiny reference: full logits at prefill and at each decode step match the capture with the indexer
  bypassed, at a capture length that keeps every query under the budget (8.3).
- Real checkpoint: layer-stream parity (8.4) at contexts of at most 2051 tokens, where dense attention is
  exactly QSA.

### Phase 5 -- QSA in the network

1. The indexer in every full-attention layer; decode through Phase 2.5.
2. Sparse prefill. Decision 5 picks between a per-query gather and a block-sparse flash kernel; the gather
   is the fallback that is correct first.
3. Rewind and `savePosition` extended to the compressed-key cache and the PLE state (Section 4).

**Gate.**
- Tiny reference: logits match the capture with selection active, prefill and decode.
- Real checkpoint: logits beyond 2051 tokens match the reference within the 3.8 parity tolerance, and the
  selected sets of a sampled layer and position match it.
- The replay self-check passes on the full network (8.5).
- Decode at long context reads at most budget + 3 KV rows per full-attention layer, confirmed by a counter,
  not inferred from timing.

### Phase 6 -- The product

1. **The n-gram table's residency** as decided in Phase 3: pinned host memory, FP8 storage, or a
   memory-mapped file. Measure the prefill gather's cost at a 2K chunk.
2. **The precision plan and export**: the roles of Section 10 question 6, through `Tools/ExportArtifact`.
3. **Footprint and deployment**: the compressed-key cache, the PLE state and the n-gram table's host bytes
   in `MemoryStats`, plus the sparse-layer term if the 27B is MoE (`MixtureOfExperts.md` §8). Then
   `planDeployment`.
4. **Chat and MIS**: `reasoning_effort` and `preserve_thinking` in the protocol (Section 7).
5. **Quality and long-context measurements** as for 3.8. The model card, the store name, the package.

**Gate.** The 3.8 publishing bar: quality within the agreed band of the BF16 reference, a footprint report
that matches what the load allocates, and Chat and MIS running the model by its store name.

Out of scope throughout, as for 3.8: the vision tower and the MTP head.

---

## 10. Open Decisions

Questions 1 to 4 are answered by the 27B's own files on release day; the rest are Mila's.

1. **Is the 27B dense or MoE?** Decides whether Section 3.5 applies at all.
2. **How large is its n-gram table, and where does it live?** Pinned host memory at BF16, at FP8, or a
   memory-mapped file. If it scales like Flash-Next's it outweighs the rest of the model.
3. **`hc_count`, `hc_lowrank`, the QSA budget, `ple_layer_ids`**: the same values as Flash-Next or not.
4. **`output_gate_type`**: still sigmoid.
5. **QSA kernel**: gather into a dense workspace and reuse the decode attention, or a block-sparse flash
   kernel. Decode can start with the first; prefill probably needs the second.
6. **Precision roles**: the gated-residual weights, the indexer, the router and the n-gram table, each held
   out or quantized. Proposed defaults are in Sections 3.1 and 3.4.
7. **Target hardware**: the 12 GiB and 16 GiB cards Qwen 3.8 27B ships for, or larger.
8. **Family placement**: a new `Qwen4` family (Section 6) or extending `Qwen`.
