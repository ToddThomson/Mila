"""Weight quantization formats: the level sets, the grouping, and the codebook fit.

Every format here is "a level set times a per-group FP16 absmax scale", which is what
lets one GPTQ inner loop serve them all (see fit.py). Each fake-quantization entry
returns (dequantized_weight, bits_per_weight), with bits including scale and zero-point
metadata so a scheme's average is directly comparable to a bit budget.

This module is the base of the import graph: fit, artifact and evaluate all read from it
and none of them read from each other's tables.
"""

import math
import os

# Must be set before the CUDA context exists, hence before torch is imported. Without it
# torch.use_deterministic_algorithms() refuses to run cuBLAS GEMMs at all. It lives here
# rather than in the entry point because this module is what every other one imports
# first, so the guard holds even for a caller that reaches past the CLI.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

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
#
# Keyed on the HuggingFace module suffix, which is what a decoder layer's
# named_modules() reports. A family whose projections are named differently -- Qwen3.8's
# DeltaNet layers, for instance -- needs its own table here, not an extension of this one.
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

CODEBOOK_PARAMETERS = {"cb4": (4, 32), "cb8": (8, 64)}

# (levels_or_None, group_size, absmax_divisor, bits_per_weight) per policy.
# levels None means fit a codebook per tensor. All three formats are
# "level set times per-group FP16 absmax scale", which is what lets one
# GPTQ inner loop serve them all.
GPTQ_FORMATS = {
    "cb4": (None, 32, 1.0, 2.5),
    "cb8": (None, 64, 1.0, 3.25),
    "fp4": (FP4_E2M1_LEVELS, 128, 6.0, 4.125),
}
