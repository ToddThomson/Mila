"""Quality gate for a sub-4-bit weight quantization scheme.

Fake-quantizes a known-good model with an exact storage scheme -- group sizes, zero
points, FP16 scales, codebooks -- and measures generation quality against the BF16
reference on the same probe. Every format is simulated as quantize -> dequantize on
checkpoint weights; no CUDA kernel is involved. The gate: a scheme must reach
IQ2_XXS-class generation quality, and the result can reshape or kill a bit allocation
before any kernel exists.

Written for the Qwen3.8-27B chassis (Specifications/Qwen3.8.md, section 8) and proven
on a Llama 3.2 3B proxy; the machinery is not specific to either. Section 8 records the
results, and this file is how to reproduce them.

Modes:
  --self-test          quantizer numerics on synthetic heavy-tailed tensors (no model)
  --model PATH_OR_ID   apply a scheme to a transformers causal LM and evaluate against
                       the BF16 reference in the same process

Examples:
  python quality_gate.py --self-test
  python quality_gate.py --model meta-llama/Llama-3.2-3B-Instruct --scheme codebook
"""

import argparse
import sys

# Imported before torch reaches this module: formats sets CUBLAS_WORKSPACE_CONFIG, which
# must land before any CUDA context exists or determinism cannot be enabled.
from formats import GENERATOR_SEED, SCHEMES, enforce_determinism

import torch

from artifact import write_artifact
from evaluate import (DEFAULT_PROMPTS, FALLBACK_EVAL_TEXT, agreement_length, generate,
                      perplexity, self_test)
from fit import apply_scheme, collect_importance, gptq_apply


def read_text(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def evaluate_model(args, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model} (bfloat16) on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(device).eval()

    eval_text = read_text(args.eval_text) if args.eval_text else FALLBACK_EVAL_TEXT

    print("\n[reference: bf16]")
    reference_ppl = perplexity(model, tokenizer, eval_text, device,
                               max_tokens=args.ppl_tokens)
    print(f"  perplexity {reference_ppl:.3f}")
    reference_generations = []
    for prompt in DEFAULT_PROMPTS:
        reference_generations.append(
            generate(model, tokenizer, prompt, device, args.max_new_tokens))

    if args.gguf_file:
        # External baseline: same probe, same code path; the GGUF weights are
        # dequantized by gguf-py at load, so kernel differences are excluded and
        # the comparison isolates quantization damage.
        label = f"gguf: {args.gguf_file}"
        print(f"\n[candidate {label}]")
        del model
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            args.gguf_repo, gguf_file=args.gguf_file, dtype=torch.bfloat16)
        model.to(device).eval()
    elif args.gptq:
        calibration_text = eval_text
        if args.calib_text:
            calibration_text = read_text(args.calib_text)
        else:
            print("  WARNING: calibrating on the eval text; pass --calib-text")
        artifact = {} if args.emit_artifact else None
        artifact_policies = {} if args.emit_artifact else None

        if artifact is not None and not args.fuse_gate_up_codebook:
            # Mila's fc_gate_up is one tensor and carries one codebook, so the pair must
            # be fitted jointly for the artifact to be expressible at all. Forced rather
            # than refused, and stated because it moves the measured number: the joint
            # fit costs 0.77% (spec section 8).
            print("  --emit-artifact implies --fuse-gate-up-codebook: "
                  "fc_gate_up carries one table for both projections")
            args.fuse_gate_up_codebook = True

        label = (f"gptq codebook, {args.gptq_samples}x{args.gptq_seqlen} calibration"
                 + (f", protect {args.protect_first}+{args.protect_last}"
                    if args.protect_first or args.protect_last else ""))
        print(f"\n[candidate {label}]")
        torch.manual_seed(GENERATOR_SEED)
        gptq_apply(model, tokenizer, calibration_text, device, args, artifact=artifact,
                   artifact_policies=artifact_policies)

        if artifact is not None:
            write_artifact(args.emit_artifact, artifact, artifact_policies, model)
    else:
        importance = None
        if args.calibrated:
            calibration_text = eval_text
            if args.calib_text:
                calibration_text = read_text(args.calib_text)[:200_000]
            print("\n[calibration] collecting per-channel activation energy ...")
            importance = collect_importance(
                model, tokenizer, [calibration_text] + DEFAULT_PROMPTS, device)
            print(f"  {len(importance)} tensors calibrated"
                  + (" (held-out text)" if args.calib_text else " (WARNING: eval text)"))

        label = (f"scheme: {args.scheme}"
                 + (", calibrated" if args.calibrated else "")
                 + (f", protect {args.protect_first}+{args.protect_last}"
                    if args.protect_first or args.protect_last else ""))
        print(f"\n[candidate {label}]")
        torch.manual_seed(GENERATOR_SEED)
        apply_scheme(model, args.scheme, importance=importance,
                     protect_first=args.protect_first, protect_last=args.protect_last)

    scheme_ppl = perplexity(model, tokenizer, eval_text, device,
                            max_tokens=args.ppl_tokens)
    print(f"  perplexity {scheme_ppl:.3f}  "
          f"(reference {reference_ppl:.3f}, ratio {scheme_ppl / reference_ppl:.3f})")

    for prompt, reference_tokens in zip(DEFAULT_PROMPTS, reference_generations):
        candidate_tokens = generate(model, tokenizer, prompt, device, args.max_new_tokens)
        agree = agreement_length(reference_tokens, candidate_tokens)
        print(f"\n  prompt: {prompt}")
        print(f"  greedy agreement with bf16: {agree}/{len(reference_tokens)} tokens")
        print(f"  output: {tokenizer.decode(candidate_tokens, skip_special_tokens=True)!r}")


# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", help="transformers model id or local path")
    parser.add_argument("--scheme", choices=sorted(SCHEMES), default="codebook")
    parser.add_argument("--eval-text", help="text file for perplexity (built-in fallback)")
    parser.add_argument("--gguf-repo", default="unsloth/Llama-3.2-3B-Instruct-GGUF",
                        help="repo for --gguf-file")
    parser.add_argument("--gguf-file",
                        help="evaluate a GGUF quantization as the candidate instead of a scheme")
    parser.add_argument("--ppl-tokens", type=int, default=0,
                        help="cap perplexity evaluation at N tokens (0 = full text)")
    parser.add_argument("--calibrated", action="store_true",
                        help="weight the codebook fit by per-channel activation energy")
    parser.add_argument("--calib-text",
                        help="held-out text file for calibration (default: the eval text)")
    parser.add_argument("--gptq", action="store_true",
                        help="sequential GPTQ error compensation over the codebook scheme")
    parser.add_argument("--gptq-samples", type=int, default=16,
                        help="calibration samples for GPTQ Hessians")
    parser.add_argument("--gptq-seqlen", type=int, default=1024,
                        help="tokens per GPTQ calibration sample")
    parser.add_argument("--emit-artifact", metavar="PATH",
                        help="with --gptq: write the Mila-named FFN codebook tensors as "
                             "safetensors, verifying each against the model bit-for-bit")
    parser.add_argument("--fuse-gate-up-codebook", action="store_true",
                        help="fit ONE codebook across the concatenated gate_proj/up_proj "
                             "pair, as a Mila fc_gate_up artifact would have to carry")
    parser.add_argument("--protect-first", type=int, default=0,
                        help="hold the first N decoder layers at fp4")
    parser.add_argument("--protect-last", type=int, default=0,
                        help="hold the last N decoder layers at fp4")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    enforce_determinism()
    torch.manual_seed(GENERATOR_SEED)
    device = torch.device(args.device)

    if args.self_test:
        self_test(device)
    elif args.model:
        evaluate_model(args, device)
    else:
        print("Nothing to do: pass --self-test or --model. See --help.")
        sys.exit(2)


if __name__ == "__main__":
    main()
