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
    python convert_llama32.py --model meta-llama/Llama-3.2-1B --output ../Weights/llama32/llama32_1b_fp32.bin
    python convert_llama32.py --model meta-llama/Llama-3.2-3B --output ../Weights/llama32/llama32_3b_fp32.bin

Mila component name mnemonics (2-4 chars):
    fc   = Linear (fully-connected)
    gelu = GELU activation
    sglu = SwiGLU activation
    ln   = LayerNorm
    rmsn = RMSNorm
    smax = Softmax
    mha  = MultiHeadAttention
    gqa  = GroupedQueryAttention
    res  = Residual
    mlp  = MLP
    tf   = Transformer
    temb = TokenEmbedding
    lpe  = LearnedPositionalEncoding
    rope = RoPE
    net  = Network

HuggingFace → Mila weight mapping:

    Token embedding:
        model.embed_tokens.weight                        -> temb.wte

    Per layer (i = 0..num_hidden_layers-1):
        model.layers.{i}.input_layernorm.weight          -> tf_layer_{i}.rmsn_1.weight
        model.layers.{i}.self_attn.q_proj.weight  \\
        model.layers.{i}.self_attn.k_proj.weight   |     -> tf_layer_{i}.fc_qkv_proj.weight
        model.layers.{i}.self_attn.v_proj.weight  /        (concatenated along dim 0)
        model.layers.{i}.self_attn.o_proj.weight         -> tf_layer_{i}.fc_out_proj.weight
        model.layers.{i}.post_attention_layernorm.weight -> tf_layer_{i}.rmsn_2.weight
        model.layers.{i}.mlp.gate_proj.weight \\
        model.layers.{i}.mlp.up_proj.weight    |          -> tf_layer_{i}.fc_gate_up.weight
                                               /             (concatenated along dim 0)
        model.layers.{i}.mlp.down_proj.weight            -> tf_layer_{i}.fc_down.weight

    Final RMSNorm:
        model.norm.weight                                -> rmsn_final.weight

    LM head:
        lm_head.weight                                   -> lm_head.weight
        (tied with embed_tokens in Llama 3.2 — written
         explicitly so Mila loader needs no tying logic)

    Notes:
        - HF uses 'layernorm' naming historically even though the actual
          implementation is LlamaRMSNorm — not LayerNorm.
        - No weight transposition needed: Llama uses nn.Linear [out, in]
          which is already Mila's convention.
        - LlamaBlock uses individual Linear components for FFN (no MLP
          composite) — fc_gate_up and fc_down have no mlp. prefix.
        - gate+up fused into single fc_gate_up tensor for Mila's SwiGLU
          which expects [gate | up] layout.
        - No positional embedding tensor — RoPE is computed, not learned.
        - No attention or MLP biases.
        - model_name sanitized: '.' replaced with '_' to avoid conflicts
          with Mila's component path separator.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM
from common import MilaWeightWriter, convert_dtype

# Supported Llama 3.2 text model variants
SUPPORTED_MODELS = [
    'meta-llama/Llama-3.2-1B',
    'meta-llama/Llama-3.2-3B',
    'meta-llama/Llama-3.2-1B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
]

def convert_llama32( model_name: str, output_path: str, dtype: str = 'float32' ):

    print( f"Loading {model_name} from HuggingFace..." )
    model = AutoModelForCausalLM.from_pretrained( model_name, torch_dtype=torch.float32 )
    config = model.config

    print( f"Model config:" )
    print( f"  vocab_size:              {config.vocab_size}" )
    print( f"  hidden_size:             {config.hidden_size}" )
    print( f"  num_hidden_layers:       {config.num_hidden_layers}" )
    print( f"  num_attention_heads:     {config.num_attention_heads}" )
    print( f"  num_key_value_heads:     {config.num_key_value_heads}" )
    print( f"  intermediate_size:       {config.intermediate_size}" )
    print( f"  max_position_embeddings: {config.max_position_embeddings}" )
    print( f"  rms_norm_eps:            {config.rms_norm_eps}" )

    # rope_theta moved into rope_scaling/rope_parameters in newer transformers versions.
    # Handle all three locations defensively.
    rope_theta = 500000.0  # Llama 3.2 default fallback
    if hasattr( config, 'rope_theta' ):
        rope_theta = config.rope_theta
    elif hasattr( config, 'rope_scaling' ) and isinstance( config.rope_scaling, dict ):
        rope_theta = config.rope_scaling.get( 'rope_theta', rope_theta )
    elif hasattr( config, 'rope_parameters' ) and isinstance( config.rope_parameters, dict ):
        rope_theta = config.rope_parameters.get( 'rope_theta', rope_theta )

    rope_scaling = getattr( config, 'rope_scaling', None )
    print( f"  rope_theta:              {rope_theta}" )
    print( f"  rope_scaling:            {rope_scaling}" )
    print( f"  tie_word_embeddings:     {config.tie_word_embeddings}" )

    head_dim  = config.hidden_size // config.num_attention_heads
    gqa_ratio = config.num_attention_heads // config.num_key_value_heads
    print( f"  head_dim:                {head_dim}" )
    print( f"  gqa_groups (Q/KV):       {gqa_ratio}:1" )

    # Sanitize model name — replace '.' with '_' to avoid conflicts
    # with Mila's component path separator '.'.
    raw_name   = model_name.rsplit( '/', 1 )[-1]
    model_id   = raw_name.replace( '.', '_' )
    print( f"  model_id (sanitized):    {model_id}" )

    writer = MilaWeightWriter( output_path )

    writer.set_metadata( {
        'architecture':        'llama',
        'model_name':          model_id,
        'dtype':               dtype,
        'vocab_size':          config.vocab_size,
        'hidden_size':         config.hidden_size,
        'embedding_dim':       config.hidden_size,
        'num_layers':          config.num_hidden_layers,
        'num_heads':           config.num_attention_heads,
        'num_kv_heads':        config.num_key_value_heads,
        'head_dim':            head_dim,
        'hidden_dim':          config.intermediate_size,
        'max_seq_length':      config.max_position_embeddings,
        'norm_eps':            config.rms_norm_eps,
        'rope_theta':          rope_theta,
        'use_bias':            False,
        'activation':          'silu',
        'norm_type':           'rmsnorm',
        'attention_type':      'gqa',
        'positional_encoding': 'rope',
        'tie_word_embeddings': config.tie_word_embeddings,
    } )

    state_dict = model.state_dict()

    # -------------------------------------------------------------------------
    # Token embedding
    # -------------------------------------------------------------------------
    writer.add_tensor(
        'temb.wte',
        convert_dtype( state_dict['model.embed_tokens.weight'].numpy(), dtype )
    )

    # -------------------------------------------------------------------------
    # Transformer layers
    # -------------------------------------------------------------------------
    for i in range( config.num_hidden_layers ):
        prefix_hf   = f'model.layers.{i}'
        prefix_mila = f'tf_layer_{i}'

        print( f"  Converting layer {i}/{config.num_hidden_layers - 1}..." )

        # Pre-attention RMSNorm (no bias).
        # Note: HF names this 'input_layernorm' historically even though
        # the actual implementation is LlamaRMSNorm — not LayerNorm.
        writer.add_tensor(
            f'{prefix_mila}.rmsn_1.weight',
            convert_dtype(
                state_dict[f'{prefix_hf}.input_layernorm.weight'].numpy(), dtype )
        )

        # Fused QKV projection — concatenate Q, K, V along dim 0.
        # Q: [num_heads * head_dim,    hidden_size]  e.g. [2048, 2048] for 1B
        # K: [num_kv_heads * head_dim, hidden_size]  e.g. [ 512, 2048] for 1B
        # V: [num_kv_heads * head_dim, hidden_size]  e.g. [ 512, 2048] for 1B
        # Fused: [(Q+K+V), hidden_size]              e.g. [3072, 2048] for 1B
        q_weight   = state_dict[f'{prefix_hf}.self_attn.q_proj.weight']
        k_weight   = state_dict[f'{prefix_hf}.self_attn.k_proj.weight']
        v_weight   = state_dict[f'{prefix_hf}.self_attn.v_proj.weight']
        qkv_weight = torch.cat( [q_weight, k_weight, v_weight], dim=0 )

        writer.add_tensor(
            f'{prefix_mila}.fc_qkv_proj.weight',
            convert_dtype( qkv_weight.numpy(), dtype )
        )

        # Attention output projection.
        # Shape: [hidden_size, hidden_size]  e.g. [2048, 2048]
        writer.add_tensor(
            f'{prefix_mila}.fc_out_proj.weight',
            convert_dtype(
                state_dict[f'{prefix_hf}.self_attn.o_proj.weight'].numpy(), dtype )
        )

        # Post-attention RMSNorm (no bias).
        # Note: HF names this 'post_attention_layernorm' historically —
        # same naming convention issue as input_layernorm above.
        writer.add_tensor(
            f'{prefix_mila}.rmsn_2.weight',
            convert_dtype(
                state_dict[f'{prefix_hf}.post_attention_layernorm.weight'].numpy(), dtype )
        )

        # Fused gate+up projection — concatenate gate and up along dim 0.
        # gate_proj: [intermediate_size, hidden_size]     e.g. [8192, 2048]
        # up_proj:   [intermediate_size, hidden_size]     e.g. [8192, 2048]
        # Fused:     [2*intermediate_size, hidden_size]   e.g. [16384, 2048]
        # SwiGLU splits the output into gate and up halves internally.
        # No mlp. prefix — LlamaBlock uses individual Linear components,
        # not an MLP composite.
        gate_weight    = state_dict[f'{prefix_hf}.mlp.gate_proj.weight']
        up_weight      = state_dict[f'{prefix_hf}.mlp.up_proj.weight']
        gate_up_weight = torch.cat( [gate_weight, up_weight], dim=0 )

        writer.add_tensor(
            f'{prefix_mila}.fc_gate_up.weight',
            convert_dtype( gate_up_weight.numpy(), dtype )
        )

        # Down projection.
        # Shape: [hidden_size, intermediate_size]  e.g. [2048, 8192]
        writer.add_tensor(
            f'{prefix_mila}.fc_down.weight',
            convert_dtype(
                state_dict[f'{prefix_hf}.mlp.down_proj.weight'].numpy(), dtype )
        )

    # -------------------------------------------------------------------------
    # Final RMSNorm (no bias)
    # -------------------------------------------------------------------------
    writer.add_tensor(
        'rmsn_final.weight',
        convert_dtype( state_dict['model.norm.weight'].numpy(), dtype )
    )

    # -------------------------------------------------------------------------
    # LM head
    # Tied with embed_tokens in Llama 3.2 — written explicitly so Mila's
    # weight loader needs no tying logic. The metadata flag tie_word_embeddings
    # is set for reference only.
    # -------------------------------------------------------------------------
    lm_head_key = 'lm_head.weight'
    if lm_head_key in state_dict:
        lm_head_weight = state_dict[lm_head_key].numpy()
    else:
        print( "  Note: lm_head.weight not in state_dict (tied) — copying from embed_tokens" )
        lm_head_weight = state_dict['model.embed_tokens.weight'].numpy()

    writer.add_tensor(
        'lm_head.weight',
        convert_dtype( lm_head_weight, dtype )
    )

    writer.write()

    print( f"\nConversion complete!" )
    print( f"  Output: {output_path}" )
    print( f"  Model:  {model_id}" )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert Llama 3.2 weights to Mila format' )
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
        help='Target dtype for weights (default: float32 — only float32 supported in Mila alpha.2)'
    )

    args = parser.parse_args()

    if args.dtype != 'float32':
        print( f"WARNING: dtype '{args.dtype}' is not validated in Mila alpha.2." )
        print( f"         Only float32 is supported by Mila's TPrecision template instantiations." )
        print( f"         Proceeding, but results in Mila will likely be incorrect.\n" )

    convert_llama32( args.model, args.output, args.dtype )