#!/usr/bin/env python3
# ============================================================================
# File: convert_llama32.py
# Convert Llama 3.2 weights to Mila format
# ============================================================================

"""
Convert Llama 3.2 weights from HuggingFace to Mila binary format.

Mila alpha.2 note:
    Only float32 is supported by Mila's TPrecision template instantiations in alpha.2.
    float16 / bfloat16 options are provided for forward compatibility but are NOT
    validated against Mila and will likely produce incorrect results at this stage.

Usage:
    python convert_llama32.py --model meta-llama/Llama-3.2-1B --output ../Weights/llama32/llama32_1b.bin
    python convert_llama32.py --model meta-llama/Llama-3.2-3B --output ../Weights/llama32/llama32_3b.bin
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from common import MilaWeightWriter, convert_dtype

# Supported Llama 3.2 text model variants
SUPPORTED_MODELS = [
    'meta-llama/Llama-3.2-1B',
    'meta-llama/Llama-3.2-3B',
    'meta-llama/Llama-3.2-1B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
]


def convert_llama32(model_name: str, output_path: str, dtype: str = 'bfloat16'):
    """
    Convert Llama 3.2 model to Mila component format.

    Llama 3.2 Architecture (Mila component mapping):
    - model.embed_tokens.weight          -> lenc.wte

    Per layer (i = 0..num_hidden_layers-1):
    - model.layers.{i}.input_layernorm.weight           -> tf_layer_{i}.ln_1.weight
    - model.layers.{i}.self_attn.q_proj.weight          -> tf_layer_{i}.fc_q_proj.weight
    - model.layers.{i}.self_attn.k_proj.weight          -> tf_layer_{i}.fc_k_proj.weight
    - model.layers.{i}.self_attn.v_proj.weight          -> tf_layer_{i}.fc_v_proj.weight
    - model.layers.{i}.self_attn.o_proj.weight          -> tf_layer_{i}.fc_out_proj.weight
    - model.layers.{i}.post_attention_layernorm.weight  -> tf_layer_{i}.ln_2.weight
    - model.layers.{i}.mlp.gate_proj.weight             -> tf_layer_{i}.mlp.fc_gate.weight
    - model.layers.{i}.mlp.up_proj.weight               -> tf_layer_{i}.mlp.fc_up.weight
    - model.layers.{i}.mlp.down_proj.weight             -> tf_layer_{i}.mlp.fc_down.weight

    - model.norm.weight                  -> ln_final.weight
    - lm_head.weight                     -> lm_head.weight  (tied with wte, written explicitly)

    Key differences vs GPT-2 conversion:
    - No weight transposition needed: Llama uses nn.Linear (weight shape is already
      [out_features, in_features]), vs GPT-2's Conv1D which needed .T
    - RMSNorm (no bias) instead of LayerNorm (weight + bias)
    - Separate Q/K/V projections instead of fused fc_qkv_proj (GQA: Q != K/V shapes)
    - SwiGLU FFN: three projections (gate, up, down) instead of two (fc_1, fc_2)
    - No positional embedding tensor (RoPE is computed, not learned)
    - No attention/MLP biases anywhere
    - Tied embeddings: lm_head.weight shares data with embed_tokens.weight
    """

    print(f"Loading {model_name} from HuggingFace...")
    # Load in float32 first for safe conversion; we'll cast per-tensor below
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    config = model.config

    print(f"Model config:")
    print(f"  vocab_size:          {config.vocab_size}")
    print(f"  hidden_size:         {config.hidden_size}")
    print(f"  num_hidden_layers:   {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print(f"  intermediate_size:   {config.intermediate_size}")
    print(f"  max_position_embeddings: {config.max_position_embeddings}")
    print(f"  rms_norm_eps:        {config.rms_norm_eps}")

    # rope_theta moved into rope_scaling/rope_parameters in newer transformers versions.
    # Handle all three locations defensively.
    rope_theta = 500000.0  # Llama 3.2 default fallback
    if hasattr(config, 'rope_theta'):
        rope_theta = config.rope_theta
    elif hasattr(config, 'rope_scaling') and isinstance(config.rope_scaling, dict):
        rope_theta = config.rope_scaling.get('rope_theta', rope_theta)
    elif hasattr(config, 'rope_parameters') and isinstance(config.rope_parameters, dict):
        rope_theta = config.rope_parameters.get('rope_theta', rope_theta)

    rope_scaling = getattr(config, 'rope_scaling', None)
    print(f"  rope_theta:          {rope_theta}")
    print(f"  rope_scaling:        {rope_scaling}")
    print(f"  tie_word_embeddings: {config.tie_word_embeddings}")

    # Derived GQA info
    head_dim = config.hidden_size // config.num_attention_heads
    gqa_ratio = config.num_attention_heads // config.num_key_value_heads
    print(f"  head_dim:            {head_dim}")
    print(f"  gqa_groups (Q/KV):   {gqa_ratio}:1")

    writer = MilaWeightWriter(output_path)

    # Set metadata
    writer.set_metadata({
        'architecture': 'llama',
        'model_name': model_name,
        'dtype': dtype,
        'vocab_size': config.vocab_size,
        'hidden_size': config.hidden_size,
        'num_layers': config.num_hidden_layers,
        'num_attention_heads': config.num_attention_heads,
        'num_key_value_heads': config.num_key_value_heads,
        'head_dim': head_dim,
        'intermediate_size': config.intermediate_size,
        'max_position_embeddings': config.max_position_embeddings,
        'rms_norm_eps': config.rms_norm_eps,
        'rope_theta': rope_theta,
        'use_bias': False,
        'activation': 'silu',
        'norm_type': 'rmsnorm',
        'attention_type': 'gqa',
        'positional_encoding': 'rope',
        'tie_word_embeddings': config.tie_word_embeddings,
    })

    state_dict = model.state_dict()

    # -------------------------------------------------------------------------
    # Token embeddings
    # -------------------------------------------------------------------------
    writer.add_tensor(
        'lenc.wte',
        convert_dtype(state_dict['model.embed_tokens.weight'].numpy(), dtype)
    )

    # -------------------------------------------------------------------------
    # Transformer layers
    # -------------------------------------------------------------------------
    for i in range(config.num_hidden_layers):
        prefix_hf   = f'model.layers.{i}'
        prefix_mila = f'tf_layer_{i}'

        print(f"  Converting layer {i}/{config.num_hidden_layers - 1}...")

        # --- Pre-attention RMSNorm (no bias) ---
        writer.add_tensor(
            f'{prefix_mila}.ln_1.weight',
            convert_dtype(state_dict[f'{prefix_hf}.input_layernorm.weight'].numpy(), dtype)
        )

        # --- Attention projections (separate Q / K / V due to GQA) ---
        # Q: shape [hidden_size, hidden_size]           e.g. [2048, 2048] for 1B
        # K: shape [kv_hidden, hidden_size]             e.g. [512,  2048] for 1B (8 heads * head_dim)
        # V: shape [kv_hidden, hidden_size]             e.g. [512,  2048] for 1B
        # No .T needed: nn.Linear stores as [out, in] which is already Mila's convention
        writer.add_tensor(
            f'{prefix_mila}.fc_q_proj.weight',
            convert_dtype(state_dict[f'{prefix_hf}.self_attn.q_proj.weight'].numpy(), dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.fc_k_proj.weight',
            convert_dtype(state_dict[f'{prefix_hf}.self_attn.k_proj.weight'].numpy(), dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.fc_v_proj.weight',
            convert_dtype(state_dict[f'{prefix_hf}.self_attn.v_proj.weight'].numpy(), dtype)
        )

        # --- Attention output projection ---
        # Shape: [hidden_size, hidden_size]  e.g. [2048, 2048]
        writer.add_tensor(
            f'{prefix_mila}.fc_out_proj.weight',
            convert_dtype(state_dict[f'{prefix_hf}.self_attn.o_proj.weight'].numpy(), dtype)
        )

        # --- Post-attention RMSNorm (no bias) ---
        writer.add_tensor(
            f'{prefix_mila}.ln_2.weight',
            convert_dtype(state_dict[f'{prefix_hf}.post_attention_layernorm.weight'].numpy(), dtype)
        )

        # --- SwiGLU FFN: gate, up, down projections ---
        # gate_proj: [intermediate_size, hidden_size]   e.g. [8192, 2048]
        # up_proj:   [intermediate_size, hidden_size]   e.g. [8192, 2048]
        # down_proj: [hidden_size, intermediate_size]   e.g. [2048, 8192]
        writer.add_tensor(
            f'{prefix_mila}.mlp.fc_gate.weight',
            convert_dtype(state_dict[f'{prefix_hf}.mlp.gate_proj.weight'].numpy(), dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.mlp.fc_up.weight',
            convert_dtype(state_dict[f'{prefix_hf}.mlp.up_proj.weight'].numpy(), dtype)
        )
        writer.add_tensor(
            f'{prefix_mila}.mlp.fc_down.weight',
            convert_dtype(state_dict[f'{prefix_hf}.mlp.down_proj.weight'].numpy(), dtype)
        )

    # -------------------------------------------------------------------------
    # Final RMSNorm (no bias)
    # -------------------------------------------------------------------------
    writer.add_tensor(
        'ln_final.weight',
        convert_dtype(state_dict['model.norm.weight'].numpy(), dtype)
    )

    # -------------------------------------------------------------------------
    # LM head (tied with wte in Llama 3.2, written explicitly for Mila)
    # Note: even though lm_head.weight == embed_tokens.weight in HF, we write
    # it as a separate tensor so Mila's weight loader doesn't need to know
    # about the tying. The metadata flag 'tie_word_embeddings' is set for
    # reference but Mila should use this tensor directly.
    # -------------------------------------------------------------------------
    lm_head_key = 'lm_head.weight'
    if lm_head_key in state_dict:
        lm_head_weight = state_dict[lm_head_key].numpy()
    else:
        # Tied: lm_head not stored separately in state_dict, use embed_tokens
        print("  Note: lm_head.weight not in state_dict (tied) - copying from embed_tokens")
        lm_head_weight = state_dict['model.embed_tokens.weight'].numpy()

    writer.add_tensor(
        'lm_head.weight',
        convert_dtype(lm_head_weight, dtype)
    )

    # -------------------------------------------------------------------------
    # Write to file
    # -------------------------------------------------------------------------
    writer.write()

    print(f"\nConversion complete!")
    print(f"  Output: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Llama 3.2 weights to Mila format')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=SUPPORTED_MODELS,
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for Mila weight file'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='float32',
        choices=['float32', 'float16', 'bfloat16'],
        help='Target dtype for weights (default: float32 — only float32 is supported in Mila alpha.2)'
    )

    args = parser.parse_args()

    if args.dtype != 'float32':
        print(f"WARNING: dtype '{args.dtype}' is not validated in Mila alpha.2.")
        print(f"         Only float32 is supported by Mila's TPrecision template instantiations.")
        print(f"         Proceeding, but results in Mila will likely be incorrect.\n")

    convert_llama32(args.model, args.output, args.dtype)