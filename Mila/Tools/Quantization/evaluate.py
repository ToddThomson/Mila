"""Measurement: what a scheme costs, against a reference on the same probe.

Perplexity plus short greedy generations. Nothing here quantizes anything -- it is the
half of the harness that has to stay honest about the other half, so it runs the
identical code path for every candidate and only the weights differ.
"""

import math

import torch

from formats import FORMATS, GENERATOR_SEED

FALLBACK_EVAL_TEXT = (
    "The design of a compiler begins with the shape of the language it must accept. "
    "A tokenizer walks the raw characters and produces a stream of terminals; a parser "
    "folds that stream into a tree whose structure mirrors the grammar. Every later "
    "stage - type checking, lowering, optimization, code generation - is a walk over "
    "that tree or over a graph derived from it. The discipline that keeps a compiler "
    "maintainable is the same one that keeps any large system maintainable: each stage "
    "states its invariants and refuses input that violates them. When an optimization "
    "pass assumes single assignment, the pass before it must establish single "
    "assignment, and the pass after it may destroy the property only by declaring so. "
    "Errors discovered late are expensive because the distance between the mistake and "
    "its symptom has grown; a good intermediate representation shortens that distance "
    "by making illegal states unrepresentable. None of this is specific to compilers. "
    "It is what engineering looks like when the cost of a wrong answer is high and the "
    "input space is adversarial."
)

DEFAULT_PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "List three differences between TCP and UDP.",
    "What is 17 * 23? Show the steps.",
]


# ---------------------------------------------------------------------------
# Self-test: quantizer numerics on synthetic heavy-tailed tensors
# ---------------------------------------------------------------------------

def synthetic_weights(device):
    torch.manual_seed(GENERATOR_SEED)
    gaussian = torch.randn(1024, 4096, device=device) * 0.02
    student_t = torch.distributions.StudentT(df=5.0).sample((1024, 4096)).to(device) * 0.02
    return {"gaussian": gaussian, "student_t_df5": student_t}


def self_test(device):
    print(f"Self-test on {device}: relative RMSE (lower is better)\n")
    header = f"{'distribution':<16}" + "".join(f"{name:>12}" for name in FORMATS)
    bits_row = f"{'bits/weight':<16}"
    print(header)

    for dist_name, weight in synthetic_weights(device).items():
        row = f"{dist_name:<16}"
        for format_name, quantize in FORMATS.items():
            dequantized, bits = quantize(weight)
            rel_rmse = ((dequantized - weight).norm() / weight.norm()).item()
            row += f"{rel_rmse:>11.4f} "
            if dist_name == "gaussian":
                bits_row += f"{bits:>11.4f} "
        print(row)

    print(bits_row)
    print("\nReading: cb4 (2.5 bits) vs int2 (2.5625 bits) is the codebook lever;")
    print("the student-t row is closer to real LLM weight tails than the gaussian row.")


# ---------------------------------------------------------------------------
# Model measurement
# ---------------------------------------------------------------------------

def perplexity(model, tokenizer, text, device, context=1024, stride=512, max_tokens=0):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    if max_tokens and input_ids.shape[1] > max_tokens:
        input_ids = input_ids[:, :max_tokens]
    log_likelihood = 0.0
    token_count = 0

    for begin in range(0, input_ids.shape[1] - 1, stride):
        window = input_ids[:, begin:begin + context]
        targets = window.clone()
        if begin > 0:
            targets[:, :context - stride] = -100
        loss = model(window, labels=targets).loss
        evaluated = (targets != -100).sum().item() - 1
        log_likelihood += loss.item() * evaluated
        token_count += evaluated
        if begin + context >= input_ids.shape[1]:
            break

    return math.exp(log_likelihood / token_count)


@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_new_tokens):
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    input_ids = encoded["input_ids"].to(device)
    output = model.generate(
        input_ids, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id)
    return output[0, input_ids.shape[1]:].tolist()


def agreement_length(reference_tokens, candidate_tokens):
    length = 0
    for a, b in zip(reference_tokens, candidate_tokens):
        if a != b:
            break
        length += 1
    return length
