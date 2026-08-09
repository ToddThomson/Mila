"""
Tokenizer round-trip and the sampling knobs, on one prompt.

Blocking generation rather than streaming (chat.py has the streaming loop), so
what is on show here is the small stuff you actually poke at: how text becomes
tokens and back, and what temperature / top_k / top_p do to the same prompt.

    python generate.py --sweep
    python generate.py --family llama --quantization fp8
    python generate.py --raw --prompt "The three laws of robotics are"

The prompt is wrapped in the model's instruct template. Pass --raw to skip that and
feed the text through untouched -- worth doing once, to see what an instruct-tuned
model does with an instruction it was not handed in the form it was trained on (it
degenerates into punctuation within a dozen tokens).
"""

import argparse
import time

import common

DEFAULT_PROMPT = "In three sentences, explain what a KV cache is and why it matters."

# One-turn instruct templates. chat.py explains the Gemma grammar and carries the
# multi-turn version; the single-turn form is repeated here so this file stands on
# its own. The markers are registered vocabulary tokens, so they are written as
# literal text and the tokenizer encodes each as one token.
TEMPLATES = {
    # The trailing empty <|channel>thought<channel|> primes thinking OFF -- without
    # it the 12B narrates a reasoning section at you.
    "gemma": "<bos><|turn>user\n{}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
    "llama": ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
              "<|start_header_id|>assistant<|end_header_id|>\n\n"),
}

# Registered tokens that must never reach the screen if one slips through.
CONTROL_TOKENS = (
    "<bos>", "<eos>", "<pad>", "<|turn>", "<turn|>", "<|channel>", "<channel|>",
    "<|think|>", "<|eot_id|>",
)


# Named sampling settings for --sweep, from fully deterministic to loose.
SWEEP = (
    ("greedy      ", dict(temperature=0.0, top_k=0, top_p=1.0)),
    ("balanced    ", dict(temperature=0.6, top_k=40, top_p=0.9)),
    ("creative    ", dict(temperature=1.0, top_k=100, top_p=0.95)),
)


def strip_control_tokens(text):
    for token in CONTROL_TOKENS:
        text = text.replace(token, "")

    return text


def show_round_trip(tokenizer, text):
    """
    Encode, inspect, decode. The round-trip is exact for text the vocabulary
    covers; where it is not, the difference is worth seeing rather than hiding.
    """
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print(f"vocab size        {tokenizer.vocab_size}")
    print(f"bos / eos / pad   {tokenizer.bos_token_id} / {tokenizer.eos_token_id} / {tokenizer.pad_token_id}")
    print(f"text              {text!r}")
    print(f"tokens ({len(ids)})       {ids}")

    pieces = [tokenizer.token_to_string(token_id) for token_id in ids]
    print(f"pieces            {pieces}")
    print(f"decoded           {decoded!r}")
    print(f"round-trip exact  {decoded == text}")
    print()


def run(model, tokenizer, prompt_tokens, max_new_tokens, temperature, top_k, top_p):
    """One blocking generation. Returns (text, generated token count, seconds)."""
    started = time.perf_counter()
    output = model.generate(prompt_tokens, max_new_tokens, temperature, top_k, top_p)
    elapsed = time.perf_counter() - started

    # generate() returns the prompt followed by the completion.
    generated = output[len(prompt_tokens):]

    return strip_control_tokens(tokenizer.decode(generated)), len(generated), elapsed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--family", default="gemma", choices=("gemma", "llama"))
    parser.add_argument("--quantization", choices=("bf16", "fp8", "fp4"),
                        help="Quantize Linear weights at load time. FP8 and FP4 require "
                             "SM >= 8.9. Default: fp4 for gemma, bf16 for llama.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--weights", help="Path to the Mila .bin weights.")
    parser.add_argument("--tokenizer", help="Path to the tokenizer .bin.")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--sweep", action="store_true",
                        help="Run the same prompt at three sampling settings instead of one.")
    parser.add_argument("--raw", action="store_true",
                        help="Feed the prompt as-is instead of wrapping it in the instruct template.")
    parser.add_argument("--log-level", default="warning", choices=("trace", "info", "warning", "error"))

    return parser.parse_args()


def main():
    args = parse_args()

    common.configure_console()

    mila = common.import_mila(args.log_level)
    weights, tokenizer_path = common.resolve_paths(args.family, args.weights, args.tokenizer)

    print(f"Loading {weights.name} (context {args.context_length}) ...", flush=True)
    load_started = time.perf_counter()
    tokenizer, model = common.load(
        mila, args.family, weights, tokenizer_path,
        args.context_length, args.device_index, args.quantization)
    print(f"Loaded in {time.perf_counter() - load_started:.1f}s\n")

    config = model.get_config()
    print(f"{args.family}: {config['num_layers']} layers, model dim {config['model_dim']}, "
          f"{config['num_heads']} heads / {config['num_kv_heads']} kv heads, "
          f"vocab {config['vocab_size']}\n")

    # The round-trip is shown on the bare prompt -- that is the text worth reading.
    # Generation runs on the templated form unless --raw.
    show_round_trip(tokenizer, args.prompt)

    prompt = args.prompt if args.raw else TEMPLATES[args.family].format(args.prompt)
    prompt_tokens = tokenizer.encode(prompt)

    settings = SWEEP if args.sweep else (
        ("as configured", dict(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)),
    )

    for label, knobs in settings:
        text, count, elapsed = run(
            model, tokenizer, prompt_tokens, args.max_new_tokens, **knobs)

        knob_summary = ", ".join(f"{name}={value}" for name, value in knobs.items())
        print(f"--- {label}  ({knob_summary})")
        print(text.strip())
        print(f"[{count} tokens in {elapsed:.2f}s, {count / max(elapsed, 1e-9):.1f} tok/s]\n")


if __name__ == "__main__":
    main()
