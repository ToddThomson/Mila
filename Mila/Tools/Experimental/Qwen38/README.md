# Qwen3.8 Phase 0 — quality gate harness

Phase 0 of `Mila/Specifications/Qwen3.8.md` (section 8): fake-quantization experiments that
must pass IQ2_XXS-class generation quality before the sub-4-bit CUDA kernels are built.
Results of record live in the spec ("Phase 0 first results"); this file is how to reproduce
them.

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

# As above, additionally emitting the packed artifact (npz; artifacts/ is
# gitignored). Every tensor is packed through packing.py and verified to
# dequantize bit-for-bit to the weights that entered the evaluated model.
python quality_gate.py --model meta-llama/Llama-3.2-3B-Instruct --gptq \
    --calib-text corpus/wiki.train.raw --protect-first 2 --protect-last 2 \
    --eval-text corpus/wiki.test.raw --ppl-tokens 131072 \
    --emit-artifact artifacts/llama32_3b_gptq_codebook.npz

# The pass line: a GGUF quantization through the identical probe
python quality_gate.py --model meta-llama/Llama-3.2-3B-Instruct \
    --gguf-file Llama-3.2-3B-Instruct-UD-IQ2_XXS.gguf \
    --eval-text corpus/wiki.test.raw --ppl-tokens 131072
```

## Rules that keep the numbers honest

- Publish wikitext numbers only. The built-in short probe flatters (~2.45x where the
  corpus says ~2.99x).
- Always pass `--calib-text` with held-out text; calibrating on the eval text inflates
  results and the harness warns when it happens.
- The perplexity ratio (candidate / BF16 reference, same probe, same code path) is the
  comparison unit, never absolute perplexity.
