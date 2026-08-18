"""Phase 0 quality gate for the Qwen3.8-27B chassis (Specifications/Qwen3.8.md, section 8).

Fake-quantizes a known-good model with the exact Section 5 storage scheme -- group sizes,
zero points, FP16 scales, codebooks -- and measures generation quality against the BF16
reference. Every format is simulated as quantize -> dequantize on checkpoint weights; no
CUDA kernel is involved. The gate: the planned scheme must reach IQ2_XXS-class generation
quality, and the result can reshape or kill the Section 5 bit allocation before any kernel
exists.

Modes:
  --self-test          quantizer numerics on synthetic heavy-tailed tensors (no model)
  --model PATH_OR_ID   apply a scheme to a transformers causal LM and evaluate against
                       the BF16 reference in the same process

Examples:
  python quality_gate.py --self-test
  python quality_gate.py --model meta-llama/Llama-3.2-3B-Instruct --scheme codebook
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Must be set before the CUDA context exists, hence before torch is imported. Without it
# torch.use_deterministic_algorithms() refuses to run cuBLAS GEMMs at all.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from safetensors.numpy import save_file

import packing

# The HF -> Mila tensor name map is the converter's, not this tool's: the packer and the
# BF16 converter must name the same tensor the same way or the artifact will not load.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Converters"))
from common import LLAMA_LAYER_TENSORS

GENERATOR_SEED = 20260816


def enforce_determinism():
    """Make a run reproducible, because this harness produces numbers of record.

    MEASURED 2026-08-17, without this: five runs of one identical configuration returned
    ratios 1.792, 1.807, 1.807, 1.847 and 1.915 -- sigma 2.7%, range 6.9%. The seed is set
    before the layer walk, so the random subsampling is already identical run to run; the
    spread comes from cuBLAS picking different algorithms depending on free workspace, and
    GPTQ compounds it because each layer quantizes against the previous layer's already
    quantized outputs. At that spread the gate cannot resolve anything below ~8%, which is
    most of what it exists to measure.

    Costs some throughput. A gate that is fast and irreproducible is worth less than one
    that is slow and repeatable.
    """
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def group_weight(weight, group_size):
    """[out, in] -> ([out, n_groups, group_size], original_in). Pads the tail group."""
    out_features, in_features = weight.shape
    pad = (-in_features) % group_size
    if pad:
        weight = torch.nn.functional.pad(weight, (0, pad))
    return weight.reshape(out_features, -1, group_size), in_features


def ungroup_weight(grouped, in_features):
    return grouped.reshape(grouped.shape[0], -1)[:, :in_features]


def as_fp16_storage(x):
    """Simulate FP16 storage of scales."""
    return x.half().float()


def guard_scale(scale):
    return torch.where(scale <= 0, torch.full_like(scale, 1e-8), scale)


# ---------------------------------------------------------------------------
# Fake-quantization formats. Each returns (dequantized_weight, bits_per_weight).
# Bits per weight include scale/zero-point metadata, matching Section 5.
# ---------------------------------------------------------------------------

def fake_int2_asymmetric(weight, group_size=32):
    """2-bit codes, FP16 scale, 2-bit zero point per group: 2 + 18/g bits."""
    grouped, in_features = group_weight(weight.float(), group_size)
    w_min = grouped.amin(-1, keepdim=True)
    w_max = grouped.amax(-1, keepdim=True)
    scale = guard_scale(as_fp16_storage((w_max - w_min) / 3.0))
    zero = (-w_min / scale).round().clamp(0, 3)
    codes = (grouped / scale + zero).round().clamp(0, 3)
    dequantized = (codes - zero) * scale
    return ungroup_weight(dequantized, in_features), 2.0 + 18.0 / group_size


def fake_int3_offset(weight, group_size=64):
    """3-bit offset-binary levels (+-0.5 .. +-3.5) * FP16 scale: 3 + 16/g bits."""
    grouped, in_features = group_weight(weight.float(), group_size)
    absmax = grouped.abs().amax(-1, keepdim=True)
    scale = guard_scale(as_fp16_storage(absmax / 3.5))
    codes = (grouped / scale - 0.5).round().clamp(-4, 3) + 0.5
    dequantized = codes * scale
    return ungroup_weight(dequantized, in_features), 3.0 + 16.0 / group_size


FP4_E2M1_LEVELS = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
)


def nearest_level_index(values, levels):
    """Index of the nearest entry of a sorted 1-D level tensor, per element."""
    midpoints = (levels[1:] + levels[:-1]) / 2.0
    return torch.bucketize(values, midpoints.to(values.device))


def nearest_level(values, levels):
    """Map each value to the nearest entry of a sorted 1-D level tensor."""
    return levels.to(values.device)[nearest_level_index(values, levels)]


def fake_fp4_e2m1(weight, group_size=128):
    """FP4 E2M1 absmax with FP16 scale: 4 + 16/g bits (today's PerGroupFp4 with FP16 scales)."""
    grouped, in_features = group_weight(weight.float(), group_size)
    absmax = grouped.abs().amax(-1, keepdim=True)
    scale = guard_scale(as_fp16_storage(absmax / 6.0))
    dequantized = nearest_level(grouped / scale, FP4_E2M1_LEVELS) * scale
    return ungroup_weight(dequantized, in_features), 4.0 + 16.0 / group_size


def kmeans_1d(values, k, iterations=30, weights=None):
    """1-D Lloyd k-means, quantile-initialized, optionally importance-weighted.

    Returns sorted centroids. With weights, minimizes sum(w * (v - c)^2) -- the
    imatrix-style objective: errors on high-activation channels cost more.
    """
    if weights is None:
        weights = torch.ones_like(values)
    quantiles = torch.linspace(0.5 / k, 1.0 - 0.5 / k, k, device=values.device)
    centroids = torch.quantile(values, quantiles)

    for _ in range(iterations):
        centroids, _ = centroids.sort()
        midpoints = (centroids[1:] + centroids[:-1]) / 2.0
        assignment = torch.bucketize(values, midpoints)
        sums = torch.zeros(k, device=values.device).scatter_add_(0, assignment, values * weights)
        counts = torch.zeros(k, device=values.device).scatter_add_(0, assignment, weights)
        empty = counts == 0
        if empty.any():
            refill = values[torch.randint(len(values), (int(empty.sum()),), device=values.device)]
            sums[empty] = refill
            counts[empty] = 1
        centroids = sums / counts

    centroids, _ = centroids.sort()
    return centroids


def codebook_samples(weight, group_size, importance=None, sample_limit=500_000):
    """Group-normalized values and their importance weights, subsampled.

    Split out so several tensors can be fitted jointly without materializing their
    concatenation: the normalization makes three float32 tensors the size of the
    input, so concatenating first doubles a peak that is already the largest
    transient in the pass.
    """
    grouped, in_features = group_weight(weight.float(), group_size)
    absmax = grouped.abs().amax(-1, keepdim=True)
    normalized = grouped / guard_scale(as_fp16_storage(absmax))

    flat = normalized.flatten()
    flat_weights = None
    if importance is not None:
        pad = grouped.shape[1] * grouped.shape[2] - in_features
        channel_weight = importance.float().to(flat.device).clamp_min(1e-12)
        if pad:
            channel_weight = torch.nn.functional.pad(channel_weight, (0, pad))
        flat_weights = channel_weight.expand(weight.shape[0], -1).flatten()

    if flat.numel() > sample_limit:
        keep = torch.randperm(flat.numel(), device=flat.device)[:sample_limit]
        return flat[keep], (flat_weights[keep] if flat_weights is not None else None)

    return flat, flat_weights


def fit_codebook_levels(weight, k, group_size, importance=None, sample_limit=500_000):
    """Fit a per-tensor k-entry codebook over group-normalized values.

    With importance (per-input-channel mean squared activation, [in]), the k-means
    fit is weighted so that channels the model actually drives hard are represented
    more accurately -- the data-dependent half of what IQ2-class formats do.
    """
    sample, sample_weights = codebook_samples(weight, group_size, importance, sample_limit)

    return kmeans_1d(sample, k, weights=sample_weights)


def fit_codebook_levels_joint(weights, k, group_size, importance=None,
                              sample_limit=500_000):
    """One codebook over several tensors that share an input axis.

    Each tensor is normalized and sampled on its own and only the samples are
    concatenated, so peak memory matches the single-tensor fit rather than the sum.
    The budget is split evenly, so the table represents both tensors equally
    regardless of their relative size.
    """
    per_tensor = max(1, sample_limit // len(weights))
    samples = []
    sample_weights = []

    for weight in weights:
        drawn, drawn_weights = codebook_samples(weight, group_size, importance, per_tensor)
        samples.append(drawn)
        sample_weights.append(drawn_weights)

    combined = torch.cat(samples)
    combined_weights = (torch.cat(sample_weights)
                        if sample_weights[0] is not None else None)

    return kmeans_1d(combined, k, weights=combined_weights)


def fake_codebook(weight, k, group_size, sample_limit=500_000, importance=None):
    """Per-tensor k-entry codebook over group-normalized values, FP16 absmax scale.

    log2(k) + 16/g bits per weight; the codebook itself amortizes to nothing.
    This is the LUT-dequantization variant Section 8 makes the default: on SM 8.9
    a non-uniform codebook prices identically to uniform INT2/INT3 steps.
    """
    grouped, in_features = group_weight(weight.float(), group_size)
    absmax = grouped.abs().amax(-1, keepdim=True)
    scale = guard_scale(as_fp16_storage(absmax))
    codebook = fit_codebook_levels(weight, k, group_size,
                                   importance=importance, sample_limit=sample_limit)
    dequantized = nearest_level(grouped / scale, codebook) * scale
    return ungroup_weight(dequantized, in_features), math.log2(k) + 16.0 / group_size


FORMATS = {
    "int2": lambda w: fake_int2_asymmetric(w, 32),
    "int3": lambda w: fake_int3_offset(w, 64),
    "cb4": lambda w: fake_codebook(w, 4, 32),
    "cb8": lambda w: fake_codebook(w, 8, 64),
    "fp4": lambda w: fake_fp4_e2m1(w, 128),
}

# The Section 5 plan mapped onto a Llama-shaped proxy. gate/up and v/o stand in for
# the FFN gate+up and DeltaNet v/gate/o rows (the 2.5-bit class); down and q/k stand
# in for the 3.25-bit class; lm_head for the FP4 row.
SCHEMES = {
    "bf16": {},
    "uniform": {
        "gate_proj": "int2", "up_proj": "int2", "v_proj": "int2", "o_proj": "int2",
        "down_proj": "int3", "q_proj": "int3", "k_proj": "int3",
        "lm_head": "fp4",
    },
    "codebook": {
        "gate_proj": "cb4", "up_proj": "cb4", "v_proj": "cb4", "o_proj": "cb4",
        "down_proj": "cb8", "q_proj": "cb8", "k_proj": "cb8",
        "lm_head": "fp4",
    },
    "fp4": {
        name: "fp4"
        for name in ("gate_proj", "up_proj", "down_proj",
                     "q_proj", "k_proj", "v_proj", "o_proj", "lm_head")
    },
}


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
# Model evaluation
# ---------------------------------------------------------------------------

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


CODEBOOK_PARAMETERS = {"cb4": (4, 32), "cb8": (8, 64)}


def collect_importance(model, tokenizer, texts, device):
    """Per-input-channel mean squared activation for every target linear, via hooks.

    This is the calibration pass -- the analog of llama.cpp's importance matrix.
    A handful of forward passes over ordinary text is enough to expose which
    channels the model drives hard.
    """
    sums = {}
    counts = {}
    hooks = []

    def make_hook(name):
        def hook(module, inputs):
            x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            sums[name] = sums.get(name, 0) + (x * x).sum(0)
            counts[name] = counts.get(name, 0) + x.shape[0]
        return hook

    target_suffixes = set(SCHEMES["codebook"])
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] in target_suffixes and hasattr(module, "weight"):
            hooks.append(module.register_forward_pre_hook(make_hook(name)))

    with torch.no_grad():
        for text in texts:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            model(input_ids[:, :1024])

    for hook in hooks:
        hook.remove()

    return {name: sums[name] / counts[name] for name in sums}


# ---------------------------------------------------------------------------
# GPTQ-style error compensation: quantize columns sequentially, folding each
# column's quantization error into the not-yet-quantized columns through the
# inverse Hessian of the layer's calibration activations. The remaining large
# lever after codebooks and calibration (spec section 8, Phase 0).
# ---------------------------------------------------------------------------

class _StopForward(Exception):
    pass


# (levels_or_None, group_size, absmax_divisor, bits_per_weight) per policy.
# levels None means fit a codebook per tensor. All three formats are
# "level set times per-group FP16 absmax scale", which is what lets one
# GPTQ inner loop serve them all.
GPTQ_FORMATS = {
    "cb4": (None, 32, 1.0, 2.5),
    "cb8": (None, 64, 1.0, 3.25),
    "fp4": (FP4_E2M1_LEVELS, 128, 6.0, 4.125),
}


def gptq_quantize_tensor(weight, hessian, levels, group_size, divisor,
                         block_size=128, damp_fraction=0.01):
    """Return (Q, codes, scale_bits): the compensated-quantized weight, its level
    indices [out, in] uint8, and its FP16 scale bits [out, in/group_size] uint16.
    Group scales are computed when the column walk reaches each group, from the
    residual-updated weights -- group sizes of 32/64/128 all divide block_size, so
    a group never straddles a block."""
    W = weight.float().clone()
    out_features, in_features = W.shape
    H = hessian.clone()

    diagonal = torch.arange(in_features, device=H.device)
    dead = torch.diag(H) == 0
    H[diagonal[dead], diagonal[dead]] = 1.0
    W[:, dead] = 0.0
    H[diagonal, diagonal] += damp_fraction * torch.diag(H).mean()

    Hinv = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Hinv)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros_like(W)
    codes = torch.zeros(out_features, in_features, dtype=torch.uint8, device=W.device)
    scale_bits = torch.zeros(out_features, in_features // group_size,
                             dtype=torch.uint16, device=W.device)
    levels = levels.to(W.device)
    scale = None

    for block_start in range(0, in_features, block_size):
        block_end = min(block_start + block_size, in_features)
        count = block_end - block_start
        W1 = W[:, block_start:block_end].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[block_start:block_end, block_start:block_end]

        for i in range(count):
            column = block_start + i
            if column % group_size == 0:
                group_end = min(i + group_size, count)
                absmax = W1[:, i:group_end].abs().amax(1)
                scale = guard_scale(as_fp16_storage(absmax / divisor))
                scale_bits[:, column // group_size] = scale.half().view(torch.uint16)
            w = W1[:, i]
            index = nearest_level_index(w / scale, levels)
            q = levels[index] * scale
            Q1[:, i] = q
            codes[:, column] = index.to(torch.uint8)
            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err

        Q[:, block_start:block_end] = Q1
        if block_end < in_features:
            W[:, block_end:] -= Err1 @ Hinv[block_start:block_end, block_end:]

    return Q, codes, scale_bits


# ---------------------------------------------------------------------------
# Mila artifact emission
#
# Only the FFN linears become codebook records. Mila fuses q/k/v into one
# fc_qkv_proj, and the Phase 0 research allocation puts q/k at cb8 and v at cb4 --
# two formats inside one tensor, which a single codebook cannot express. Section 8
# step 5 settles it the other way round: attention and lm_head stay BF16 in the
# artifact and quantize to FP4 at load, so the artifact carries codebook tensors only.
# ---------------------------------------------------------------------------

# Mila tensor stem -> the policy it is written with. The HF sources of each stem come
# from the converter's name map, so the two tools cannot disagree about what fuses.
ARTIFACT_POLICIES = {"fc_gate_up": "cb4", "fc_down": "cb8"}

# Policy spelling as the C++ side names it, for the artifact's metadata.
POLICY_TYPE_NAMES = {
    "cb4": "PerGroupCodebook2<32>",
    "cb8": "PerGroupCodebook3<64>",
}


def artifact_tensor_plan():
    """[(mila_stem, (hf_module_suffix, ...), policy)] for the codebook tensors.

    Derived from the converter's LLAMA_LAYER_TENSORS rather than restated, so the
    fusion this packs matches the fusion the BF16 converter writes. HF names are
    reduced to the module path a layer's named_modules() reports ('mlp.gate_proj'),
    which is how gptq_apply keys its targets.
    """
    plan = []

    for mapping in LLAMA_LAYER_TENSORS:
        stem = mapping.mila.removeprefix("tf_layer_{i}.").removesuffix(".weight")
        policy = ARTIFACT_POLICIES.get(stem)

        if policy is None:
            continue

        modules = tuple(
            source.removeprefix("model.layers.{i}.").removesuffix(".weight")
            for source in mapping.sources)
        plan.append((stem, modules, policy))

    return plan


def pack_codebook_tensor(name, policy, quantized, codes, scale_bits, levels, group_size):
    """Pack one GPTQ tensor through the normative layout and verify the packed form
    dequantizes bit-for-bit to the weights that entered the model. A failure here is
    a layout or codec bug, never a tolerance question.

    Returns (plane_two, plane_one_or_None, scale_bits) as numpy arrays.
    """
    codes_np = codes.cpu().numpy()
    scale_bits_np = scale_bits.cpu().numpy()
    codebook_np = levels.cpu().numpy().astype(np.float32)
    columns = codes_np.shape[1]

    if policy == "cb4":
        plane_two = packing.pack_two_bit_codes(codes_np)
        plane_one = None
        recovered = packing.unpack_two_bit_codes(plane_two, columns)
    else:
        plane_two, plane_one = packing.pack_three_bit_codes(codes_np)
        recovered = packing.unpack_three_bit_codes(plane_two, plane_one, columns)

    if not np.array_equal(recovered, codes_np):
        raise AssertionError(f"{name}: pack/unpack mismatch")

    stored_scale = scale_bits_np.view(np.float16).astype(np.float32)
    column_group = np.arange(columns) // group_size
    dequantized = codebook_np[codes_np] * stored_scale[:, column_group]

    if not np.array_equal(dequantized, quantized.cpu().numpy()):
        raise AssertionError(f"{name}: packed dequantization differs from model weights")

    return plane_two, plane_one, scale_bits_np


def record_mila_tensor(artifact, name, policy, pieces, codebook, group_size,
                       in_features):
    """Assemble one Mila tensor from its per-HF-linear packed pieces.

    Codes, scale bits and the high plane all concatenate along the output axis --
    packing is row-major and rows are independent, so packing each source and
    stacking is byte-identical to packing the fused matrix. The codebook cannot
    concatenate, which is why the sources were fitted against one shared table.

    The emitted shapes are checked against what Linear::initializeParameters
    allocates for the policy; a mismatch here fails the load with a shape error
    that says nothing about which side is wrong, so it is caught at the source.
    """
    planes_two = [piece[0] for piece in pieces]
    planes_one = [piece[1] for piece in pieces]
    scales = [piece[2] for piece in pieces]

    weight = np.concatenate(planes_two, axis=0)
    scale = np.concatenate(scales, axis=0).view(np.float16)
    out_features = weight.shape[0]

    expect_shape(name, "weight", weight,
                 (out_features, in_features * 2 // 8), np.uint8)
    expect_shape(name, "weight_scale", scale,
                 (out_features, in_features // group_size), np.float16)

    artifact[f"{name}.weight"] = weight
    artifact[f"{name}.weight_scale"] = scale
    artifact[f"{name}.weight_codebook"] = codebook.cpu().numpy().astype(np.float32)

    if planes_one[0] is not None:
        high_plane = np.concatenate(planes_one, axis=0)
        expect_shape(name, "weight_high_plane", high_plane,
                     (out_features, in_features // 8), np.uint8)
        artifact[f"{name}.weight_high_plane"] = high_plane

    entries = len(artifact[f"{name}.weight_codebook"])
    expect_shape(name, "weight_codebook", artifact[f"{name}.weight_codebook"],
                 (entries,), np.float32)


def emit_mila_layer(artifact, policies, layer_index, targets, packed_pieces,
                    fitted_levels):
    """Assemble one decoder layer's Mila codebook tensors from its packed HF pieces."""
    for stem, modules, policy in artifact_tensor_plan():
        missing = [module for module in modules if module not in packed_pieces]

        if missing:
            raise AssertionError(
                f"tf_layer_{layer_index}.{stem}: no packed record for {missing}")

        # A fused tensor carries exactly one codebook, so its sources must have been
        # fitted against one shared table. Compared by value rather than trusting the
        # flag: a per-linear fit would otherwise emit the first source's table for both
        # and produce an artifact that loads and decodes the second source wrongly.
        tables = [fitted_levels[module] for module in modules]

        for table in tables[1:]:
            if not torch.equal(table, tables[0]):
                raise AssertionError(
                    f"tf_layer_{layer_index}.{stem}: sources were fitted against "
                    f"different codebooks; a fused tensor can carry only one")

        group_size = CODEBOOK_PARAMETERS[policy][1]
        name = f"tf_layer_{layer_index}.{stem}"

        record_mila_tensor(artifact, name, policy,
                           [packed_pieces[module] for module in modules],
                           tables[0], group_size,
                           targets[modules[0]].weight.shape[1])

        policies[name] = POLICY_TYPE_NAMES[policy]


def expect_shape(name, suffix, array, shape, dtype):
    if array.shape != shape or array.dtype != dtype:
        raise AssertionError(
            f"{name}.{suffix}: emitted {array.dtype} {array.shape}, "
            f"Linear allocates {np.dtype(dtype)} {shape}")


def artifact_metadata(model, policies):
    """__metadata__ for the artifact: the architecture Mila's reader parses, plus the
    per-tensor policy map.

    kMilaQuantizationMetadataKey is a single string and this artifact carries two
    policies, so the string names the scheme and the map names what each tensor
    actually is. Which of the two a Mila loader should trust is still open -- the
    load path for a codebook model does not exist yet.
    """
    config = model.config
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads)

    mila_config = {
        "architecture": "llama",
        "model_name": config.name_or_path.rsplit("/", 1)[-1].replace(".", "_"),
        "vocab_size": config.vocab_size,
        "max_seq_length": config.max_position_embeddings,
        "embedding_dim": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "num_heads": config.num_attention_heads,
        "num_kv_heads": config.num_key_value_heads,
        "head_dim": head_dim,
        "hidden_dim": config.intermediate_size,
        "use_bias": False,
        "tie_word_embeddings": config.tie_word_embeddings,
        "activation": "silu",
        "norm_type": "rmsnorm",
        "attention_type": "gqa",
        "positional_encoding": "rope",
        "rope_theta": float(getattr(config, "rope_theta", 500000.0)),
        "norm_epsilon": float(config.rms_norm_eps),
    }

    return {
        "mila_config": json.dumps(mila_config),
        "mila_quantization": "codebook",
        "mila_codebook_policies": json.dumps(policies),
    }


def write_artifact(path, artifact, policies, model):
    """Write the packed records as safetensors, which is the container Mila reads.

    int64 has no Mila wire code and the writer orders the data region itself, so
    nothing here may carry an index tensor or depend on emission order.
    """
    save_file(artifact, path, metadata=artifact_metadata(model, policies))

    payload = sum(array.nbytes for array in artifact.values())
    print(f"  artifact: {len(artifact)} tensors over {len(policies)} Mila linears, "
          f"{payload / 1e9:.2f} GB packed payload, "
          f"every tensor verified bit-exact -> {path}")


def gptq_apply(model, tokenizer, calibration_text, device, args, artifact=None,
               artifact_policies=None):
    """Sequential GPTQ over the decoder stack.

    Per layer: capture the layer's calibration inputs, accumulate per-linear
    Hessians, quantize each target linear with compensation, then recompute the
    layer's outputs with the quantized weights so the next layer calibrates
    against what it will actually see at inference.

    artifact / artifact_policies are filled in place when an artifact is being
    written: the Mila-named packed tensors, and the policy each one carries.
    """
    policy_by_suffix = SCHEMES["codebook"]
    layers = model.model.layers
    num_layers = len(layers)

    token_ids = tokenizer(calibration_text, return_tensors="pt").input_ids[0]
    generator = torch.Generator().manual_seed(GENERATOR_SEED)
    starts = torch.randint(0, max(1, token_ids.numel() - args.gptq_seqlen),
                           (args.gptq_samples,), generator=generator)
    samples = [token_ids[s:s + args.gptq_seqlen].unsqueeze(0).to(device) for s in starts]

    captured = []
    original_forward = layers[0].forward

    def catcher(*call_args, **call_kwargs):
        hidden = call_args[0] if call_args else call_kwargs.pop("hidden_states")
        captured.append((hidden, call_kwargs))
        raise _StopForward

    layers[0].forward = catcher
    with torch.no_grad():
        for sample in samples:
            try:
                model(sample)
            except _StopForward:
                pass
    layers[0].forward = original_forward

    total_params = 0
    total_bits = 0.0
    skipped_layers = []

    for layer_index, layer in enumerate(layers):
        protected = (layer_index < args.protect_first
                     or layer_index >= num_layers - args.protect_last)

        targets = {}
        for name, module in layer.named_modules():
            suffix = name.rsplit(".", 1)[-1]
            if suffix in policy_by_suffix and hasattr(module, "weight"):
                targets[name] = module

        hessians = {}
        counts = {}
        hooks = []

        def make_hook(name):
            def hook(module, inputs):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        x.shape[1], x.shape[1], device=x.device)
                    counts[name] = 0
                hessians[name] += x.T @ x
                counts[name] += x.shape[0]
            return hook

        for name, module in targets.items():
            hooks.append(module.register_forward_pre_hook(make_hook(name)))
        with torch.no_grad():
            for hidden, kwargs in captured:
                layer(hidden, **kwargs)
        for hook in hooks:
            hook.remove()

        # Mila fuses gate_proj and up_proj into one tensor (tf_layer_N.fc_gate_up), and a
        # codebook is per tensor, so a fused artifact can carry only one table. Fitting the
        # pair jointly is the cheapest way to make the artifact expressible; this measures
        # what that costs. The two projections read the SAME input, so their Hessians and
        # hence the importance vector are identical -- only the fit is shared. Off by
        # default: the recorded Phase 0 numbers are per-HF-linear.
        shared_levels = {}
        if args.fuse_gate_up_codebook and not protected:
            gate_name = next((n for n in targets if n.endswith("gate_proj")), None)
            up_name = next((n for n in targets if n.endswith("up_proj")), None)
            gate_policy = policy_by_suffix.get("gate_proj")

            if gate_name and up_name and gate_policy in CODEBOOK_PARAMETERS:
                k, fit_group = CODEBOOK_PARAMETERS[gate_policy]
                importance = torch.diag(hessians[gate_name]) / counts[gate_name]
                shared = fit_codebook_levels_joint(
                    [targets[gate_name].weight.data, targets[up_name].weight.data],
                    k, fit_group, importance=importance)
                shared_levels[gate_name] = shared
                shared_levels[up_name] = shared

        packed_pieces = {}
        fitted_levels = {}

        for name, module in targets.items():
            policy = "fp4" if protected else policy_by_suffix[name.rsplit(".", 1)[-1]]
            levels, group_size, divisor, bits = GPTQ_FORMATS[policy]
            weight = module.weight.data
            if levels is None and name in shared_levels:
                levels = shared_levels[name]
            if levels is None:
                k = 4 if policy == "cb4" else 8
                importance = torch.diag(hessians[name]) / counts[name]
                levels = fit_codebook_levels(weight, k, group_size,
                                             importance=importance)
            quantized, codes, scale_bits = gptq_quantize_tensor(
                weight, hessians[name], levels, group_size, divisor)
            if artifact is not None and policy in CODEBOOK_PARAMETERS:
                packed_pieces[name] = pack_codebook_tensor(
                    f"model.layers.{layer_index}.{name}", policy, quantized, codes,
                    scale_bits, levels, group_size)
                fitted_levels[name] = levels
            module.weight.data = quantized.to(weight.dtype)
            total_params += weight.numel()
            total_bits += bits * weight.numel()

        if artifact is not None and not protected:
            emit_mila_layer(artifact, artifact_policies, layer_index, targets,
                            packed_pieces, fitted_levels)
        elif artifact is not None:
            skipped_layers.append(layer_index)

        hessians.clear()

        # clear() drops the references but not the reservation: torch's caching allocator
        # keeps freed blocks, so across 28 layers of differently-shaped Hessians and
        # float32 fit transients the reserved pool creeps monotonically -- measured 8.5 GB
        # climbing to 11.5 GB and spilling into shared memory on the last layers of a
        # 12 GiB card. Returning the blocks each layer bounds it, at the cost of some
        # reallocation, and is what lets this gate run on a smaller card at all.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            for i, (hidden, kwargs) in enumerate(captured):
                output = layer(hidden, **kwargs)
                captured[i] = (output[0] if isinstance(output, tuple) else output,
                               kwargs)

        print(f"  layer {layer_index + 1}/{num_layers} done"
              + (" (protected, fp4)" if protected else ""), flush=True)

    average = total_bits / total_params if total_params else 0.0
    print(f"  quantized {total_params / 1e9:.3f} B params, "
          f"{average:.3f} average bits/weight over quantized tensors")
    if model.config.tie_word_embeddings:
        print("  lm_head is tied to embed_tokens on this model: left in BF16")
    if skipped_layers:
        print(f"  artifact carries no codebook record for protected layers "
              f"{skipped_layers}: those stay BF16 and quantize to FP4 at load")
    return total_params, average


def layer_index_of(module_name):
    parts = module_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def apply_scheme(model, scheme_name, importance=None, protect_first=0, protect_last=0,
                 verbose=True):
    """Fake-quantize matching linear weights in place. Returns (params_quantized, avg_bits).

    importance: optional {module_name: per-channel mean squared activation} from
    collect_importance; feeds the weighted codebook fit.
    protect_first / protect_last: decoder layers held at fp4 instead of their
    sub-4-bit policy (the llama.cpp UD move -- edge layers are the sensitive ones).
    """
    policy_by_suffix = SCHEMES[scheme_name]
    total_params = 0
    total_bits = 0.0
    tied = model.config.tie_word_embeddings
    num_layers = model.config.num_hidden_layers

    for name, module in model.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        policy = policy_by_suffix.get(suffix)
        if policy is None or not hasattr(module, "weight"):
            continue
        if suffix == "lm_head" and tied:
            if verbose:
                print("  lm_head is tied to embed_tokens on this model: left in BF16")
            continue

        layer = layer_index_of(name)
        if layer is not None and (layer < protect_first or layer >= num_layers - protect_last):
            policy = "fp4"

        weight = module.weight.data
        if policy in CODEBOOK_PARAMETERS:
            k, group_size = CODEBOOK_PARAMETERS[policy]
            channel_importance = importance.get(name) if importance else None
            dequantized, bits = fake_codebook(weight, k, group_size,
                                              importance=channel_importance)
        else:
            dequantized, bits = FORMATS[policy](weight)
        module.weight.data = dequantized.to(weight.dtype)
        total_params += weight.numel()
        total_bits += bits * weight.numel()

    average = total_bits / total_params if total_params else 0.0
    if verbose:
        print(f"  quantized {total_params / 1e9:.3f} B params, "
              f"{average:.3f} average bits/weight over quantized tensors")
    return total_params, average


@torch.no_grad()
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


def evaluate_model(args, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model} (bfloat16) on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(device).eval()

    eval_text = FALLBACK_EVAL_TEXT
    if args.eval_text:
        with open(args.eval_text, encoding="utf-8") as handle:
            eval_text = handle.read()

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
            with open(args.calib_text, encoding="utf-8") as handle:
                calibration_text = handle.read()
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
                with open(args.calib_text, encoding="utf-8") as handle:
                    calibration_text = handle.read()[:200_000]
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
