# Mila weight quantization

Fits, encodes and gates a sub-4-bit weight quantization scheme, offline. Nothing here is
specific to one model family: it was written for the Qwen3.8-27B chassis
(`Mila/Specifications/Qwen3.8.md`, section 8) and proven on a Llama 3.2 3B proxy, and the
scheme tables are the only part that knows which family it is looking at.

The gate is Phase 0 of that spec: fake-quantization experiments that must pass
IQ2_XXS-class generation quality before the sub-4-bit CUDA kernels are built. Results of
record live in the spec ("Phase 0 first results"); this file is how to reproduce them.

## Layout

| File | Holds |
|---|---|
| `formats.py` | grouping, the fake-quant level sets, k-means codebook fitting, the scheme tables |
| `fit.py` | activation calibration (`collect_importance`) and sequential GPTQ |
| `artifact.py` | Mila-named safetensors emission, shape checks, metadata |
| `evaluate.py` | perplexity, greedy generation and agreement, the synthetic self-test |
| `packing.py` | the packed-layout codec, and the C++ test fixture generator |
| `quality_gate.py` | the command line, and the orchestration that wires the above |

`packing.py` is held to `Src/Dnn/Quantization/Weight/CodebookPacking.ixx`, which is the
normative statement of the layout; regenerate the fixture after any layout change:

```
python packing.py --emit-fixture ../../Tests/Dnn/Quantization/CodebookOracle.Fixture.h
```

## Environment

Runs in the Converters venv (`Mila/Tools/Converters/.venv` — Python 3.13, torch + CUDA,
transformers, `gguf`, `accelerate`). The proxy model is `meta-llama/Llama-3.2-3B-Instruct`
(HF cache); the baseline is `unsloth/Llama-3.2-3B-Instruct-GGUF` UD-IQ2_XXS.

The corpus directory (`corpus/`, gitignored) holds wikitext-2-raw `wiki.test.raw` and
`wiki.train.raw`, from the `ggml-org/ci` mirror on Hugging Face.

## Modes

```
# Quantizer numerics on synthetic tensors, no model
python quality_gate.py --self-test

# A scheme vs the BF16 reference (same process, same probe)
python quality_gate.py --model meta-llama/Llama-3.2-3B-Instruct --scheme codebook \
    --calibrated --calib-text corpus/wiki.train.raw \
    --protect-first 2 --protect-last 2 \
    --eval-text corpus/wiki.test.raw --ppl-tokens 131072

# GPTQ error compensation over the codebook scheme
python quality_gate.py --model meta-llama/Llama-3.2-3B-Instruct --gptq \
    --calib-text corpus/wiki.train.raw --protect-first 2 --protect-last 2 \
    --eval-text corpus/wiki.test.raw --ppl-tokens 131072

# As above, additionally emitting the Mila artifact (safetensors; artifacts/ is
# gitignored). Every tensor is packed through packing.py and verified to
# dequantize bit-for-bit to the weights that entered the evaluated model.
python quality_gate.py --model meta-llama/Llama-3.2-3B-Instruct --gptq \
    --calib-text corpus/wiki.train.raw --protect-first 2 --protect-last 2 \
    --eval-text corpus/wiki.test.raw --ppl-tokens 131072 \
    --emit-artifact artifacts/llama32_3b_gptq_codebook.safetensors

# The pass line: a GGUF quantization through the identical probe
python quality_gate.py --model meta-llama/Llama-3.2-3B-Instruct \
    --gguf-file Llama-3.2-3B-Instruct-UD-IQ2_XXS.gguf \
    --eval-text corpus/wiki.test.raw --ppl-tokens 131072
```

```
# One codebook across the gate_proj/up_proj pair, as a Mila fc_gate_up artifact
# must carry. Costs 0.77%; see Qwen3.8.md "Mila's fused tensors". --emit-artifact
# turns this on by itself, because the fused tensor is otherwise inexpressible.
python quality_gate.py --model meta-llama/Llama-3.2-3B-Instruct --gptq \
    --fuse-gate-up-codebook \
    --calib-text corpus/wiki.train.raw --protect-first 2 --protect-last 2 \
    --eval-text corpus/wiki.test.raw --ppl-tokens 131072
```

## What the artifact contains

Mila-named tensors, in the safetensors container Mila reads. Per unprotected decoder
layer, seven tensors over two fused linears:

| Tensor | Policy | dtype and shape |
|---|---|---|
| `tf_layer_N.fc_gate_up.weight` | `PerGroupCodebook2<32>` | U8 `[2*inter, hidden/4]` |
| `tf_layer_N.fc_gate_up.weight_scale` | | F16 `[2*inter, hidden/32]` |
| `tf_layer_N.fc_gate_up.weight_codebook` | | F32 `[4]` |
| `tf_layer_N.fc_down.weight` | `PerGroupCodebook3<64>` | U8 `[hidden, inter/4]` |
| `tf_layer_N.fc_down.weight_scale` | | F16 `[hidden, inter/64]` |
| `tf_layer_N.fc_down.weight_codebook` | | F32 `[8]` |
| `tf_layer_N.fc_down.weight_high_plane` | | U8 `[hidden, inter/8]` |

Attention carries no codebook record. Mila fuses q/k/v into one `fc_qkv_proj` and the
Phase 0 research allocation puts q/k at cb8 and v at cb4 -- two formats in one tensor,
which one codebook cannot express. Section 8 step 5 settles it the other way: attention
and `lm_head` stay BF16 and quantize to FP4 at load. **So the evaluated model and the
artifact are not the same network** -- the perplexity a run reports covers the research
allocation, including quantized attention, while the artifact is the FFN subset.

Names come from the converter's map (`Tools/Converters/common.py`), so the packer and the
BF16 converter cannot disagree about what fuses. Shapes are checked against what
`Linear::initializeParameters` allocates before anything is written.

## Rules that keep the numbers honest

- Publish wikitext numbers only. The built-in short probe flatters (~2.45x where the
  corpus says ~2.99x).
- Always pass `--calib-text` with held-out text; calibrating on the eval text inflates
  results and the harness warns when it happens.
- The perplexity ratio (candidate / BF16 reference, same probe, same code path) is the
  comparison unit, never absolute perplexity.
- Compare arms built by the SAME harness. Changing the fit changes how many draws the
  RNG makes, so a number from an earlier build is not a baseline -- run both arms back to
  back at matched settings and read only the delta.
- Runs are deterministic and must stay that way. Before `enforce_determinism()` the same
  configuration returned ratios spanning 1.792 to 1.915 (sigma 2.7%), which is wider than
  most effects worth measuring. If a change makes the harness non-reproducible, that is a
  defect in the change, not a cost of doing business.
- Needs a 12 GiB card. Reserved memory climbs across the layer walk and spills into shared
  memory near the end without the per-layer `empty_cache()`; a spilled pass still completes
  but crawls.
