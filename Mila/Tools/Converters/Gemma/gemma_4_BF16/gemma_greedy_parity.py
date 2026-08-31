# Gemma 4 token-for-token greedy parity: HuggingFace vs Mila (Step 5f).
#
# Builds on hf_gemma_greedy_validation.py (the HF reference) by also running Mila's
# GemmaModel (the kept `mila.GemmaModel` binding) on the SAME prompt ids and diffing
# the greedy output token-for-token. The Mila tokenizer is validated separately
# (BpeTokenizerGemma suite), so feeding identical ids here isolates MODEL parity:
# the embedding sqrt(d) scale, sandwich/QK norms, dual RoPE, the local AND global
# K=V / head_dim-512 attention, GeGLU, and the lm_head.
#
# Prereqs:
#   - Converted weights: Tools/Converters/Gemma/convert_weights.py -> <mila>.bin
#   - The `mila` pybind extension built (Mila/Bindings).
#   - pip install torch transformers  (the Converters venv already has these).
#
# Usage:
#   python gemma_greedy_parity.py --mila-bin <...>/gemma4_12b_it_bf16.bin \
#       --prompt "The capital of France is" --max-new-tokens 64
#
# Greedy parity note: bf16 reductions differ slightly between implementations, so
# some divergence depth is expected. The harness reports the matched-prefix length
# and the first divergence rather than asserting an exact full match.

import argparse
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser(description="Gemma 4 HF<->Mila greedy parity")
    ap.add_argument("--hf-model", default="google/gemma-4-12b-it")
    ap.add_argument("--mila-bin", required=True, help="Converted Mila weights .bin")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--context-length", type=int, default=4096)
    ap.add_argument("--device-index", type=int, default=0)
    args = ap.parse_args()

    try:
        import mila
    except ImportError:
        print("Error: the Mila pybind extension `mila` is not importable.\n"
              "Build + install it first (Mila/Bindings), then re-run.")
        sys.exit(1)

    dev = f"cuda:{args.device_index}"

    # ---- Shared prompt ids (HF tokenizer; Mila's is validated separately) ----
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    ids = tokenizer.encode(args.prompt, return_tensors="pt", add_special_tokens=True)
    prompt_ids = ids[0].tolist()
    print(f"prompt     : {args.prompt!r}")
    print(f"prompt ids : {prompt_ids}\n")

    # ---- HF reference (greedy, forced to max_new_tokens) ---------------------
    print(f"Loading HF {args.hf_model} (bf16, {dev})...")
    hf = AutoModelForCausalLM.from_pretrained(args.hf_model, dtype=torch.bfloat16).to(dev).eval()
    with torch.no_grad():
        hf_out = hf.generate(ids.to(dev), do_sample=False, num_beams=1,
                             min_new_tokens=args.max_new_tokens,
                             max_new_tokens=args.max_new_tokens, use_cache=True)
    hf_gen = hf_out[0].tolist()[len(prompt_ids):]
    print(f"HF   gen   : {hf_gen}")
    print(f"           : {tokenizer.decode(hf_gen)!r}\n")

    # ---- Mila ----------------------------------------------------------------
    print(f"Loading Mila {args.mila_bin}...")
    mila.initialize("warning")
    model = mila.GemmaModel.from_pretrained(
        args.mila_bin, context_length=args.context_length, device_index=args.device_index)
    print(f"  config: {model.get_config()}")

    mila_gen = []
    reason = model.generate(prompt_ids, mila_gen.append,
                            max_new_tokens=args.max_new_tokens,
                            temperature=0.0, top_k=1)   # greedy argmax
    print(f"Mila gen   : {mila_gen}  [{reason}]")
    print(f"           : {tokenizer.decode(mila_gen)!r}\n")

    # ---- Compare -------------------------------------------------------------
    matched = 0
    for a, b in zip(hf_gen, mila_gen):
        if a != b:
            break
        matched += 1

    n = min(len(hf_gen), len(mila_gen))
    print("=" * 60)
    print(f"Matched prefix: {matched}/{n} tokens")

    if matched == len(hf_gen) and len(mila_gen) >= len(hf_gen):
        print("PARITY: exact match over the full HF window.")
    elif matched == n:
        print(f"Matched all {n} comparable tokens"
              + (" (Mila stopped early at EOS)." if len(mila_gen) < len(hf_gen) else "."))
    else:
        hf_tok = hf_gen[matched] if matched < len(hf_gen) else None
        mila_tok = mila_gen[matched] if matched < len(mila_gen) else None
        print(f"DIVERGENCE at token {matched}:")
        print(f"  HF  : {hf_tok} {tokenizer.decode([hf_tok])!r}" if hf_tok is not None else "  HF  : <end>")
        print(f"  Mila: {mila_tok} {tokenizer.decode([mila_tok])!r}" if mila_tok is not None else "  Mila: <end>")


if __name__ == "__main__":
    main()
