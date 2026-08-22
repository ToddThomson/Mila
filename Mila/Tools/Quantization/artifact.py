"""Emission of a Mila-named quantized artifact from packed codebook records.

Only the codebook tensors become artifact records. Mila fuses q/k/v into one
fc_qkv_proj, and the Phase 0 research allocation puts q/k at cb8 and v at cb4 --
two formats inside one tensor, which a single codebook cannot express. Qwen3.8.md
section 8 step 5 settles it the other way round: attention and lm_head stay BF16 in
the artifact and quantize to FP4 at load, so the artifact carries codebook tensors only.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

import packing
from formats import CODEBOOK_PARAMETERS

# The HF -> Mila tensor name map is the converter's, not this tool's: the packer and the
# BF16 converter must name the same tensor the same way or the artifact will not load.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Converters"))
from common import LLAMA_LAYER_TENSORS

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


def codebook_tensor_records(name, pieces, codebook, group_size, in_features):
    """{tensor_name: array} for one Mila codebook tensor, from its packed pieces.

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

    table = codebook.cpu().numpy().astype(np.float32)
    expect_shape(name, "weight_codebook", table, (len(table),), np.float32)

    records = {
        f"{name}.weight": weight,
        f"{name}.weight_scale": scale,
        f"{name}.weight_codebook": table,
    }

    if planes_one[0] is not None:
        high_plane = np.concatenate(planes_one, axis=0)
        expect_shape(name, "weight_high_plane", high_plane,
                     (out_features, in_features // 8), np.uint8)
        records[f"{name}.weight_high_plane"] = high_plane

    return records


def record_mila_tensor(artifact, name, policy, pieces, codebook, group_size,
                       in_features):
    """Assemble one Mila tensor into an in-memory artifact dict."""
    artifact.update(
        codebook_tensor_records(name, pieces, codebook, group_size, in_features))


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
