#!/usr/bin/env python3
# ============================================================================
# File: convert_weights.py
# Convert Llama weights to Mila format
# ============================================================================

"""
Convert Llama 3.x weights from HuggingFace to Mila binary format.

Supports Llama 3.2 (1B, 3B) and Llama 3.1 (8B) text model variants.
All share the same weight layout; dimensions are read from the model config
so no code changes are needed when adding new variants.

Models are loaded directly in the target dtype to minimise peak memory usage —
critical for the 8B variant under BF16 (~16 GB).

Usage:
    # Llama 3.2
    python convert_llama_weights.py --model meta-llama/Llama-3.2-1B --output ../Weights/llama/llama32_1b_bf16.bin
    python convert_llama_weights.py --model meta-llama/Llama-3.2-3B --output ../Weights/llama/llama32_3b_bf16.bin
    python convert_llama_weights.py --model meta-llama/Llama-3.2-3B-Instruct --output ../Weights/llama/llama32_3b_instruct_bf16.bin
    python convert_llama_weights.py --model meta-llama/Llama-3.2-1B --dtype float32 --output ../Weights/llama/llama32_1b_fp32.bin
    python convert_llama_weights.py --model meta-llama/Llama-3.2-3B --dtype float32 --output ../Weights/llama/llama32_3b_fp32.bin

    # Llama 3.1 8B
    python convert_llama_weights.py --model meta-llama/Llama-3.1-8B --output ../Weights/llama/llama31_8b_bf16.bin
    python convert_llama_weights.py --model meta-llama/Llama-3.1-8B-Instruct --output ../Weights/llama/llama31_8b_instruct_bf16.bin


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

HuggingFace -> Mila weight mapping:

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
        (tied with embed_tokens in Llama 3.2 -- written
         explicitly so Mila loader needs no tying logic)

    Notes:
        - HF uses 'layernorm' naming historically even though the actual
          implementation is LlamaRMSNorm -- not LayerNorm.
        - No weight transposition needed: Llama uses nn.Linear [out, in]
          which is already Mila's convention.
        - LlamaBlock uses individual Linear components for FFN (no MLP
          composite) -- fc_gate_up and fc_down have no mlp. prefix.
        - gate+up fused into single fc_gate_up tensor for Mila's SwiGLU
          which expects [gate | up] layout.
        - No positional embedding tensor -- RoPE is computed, not learned.
        - No attention or MLP biases.
        - model_name sanitized: '.' replaced with '_' to avoid conflicts
          with Mila's component path separator '.'.
        - bfloat16 tensors are stored as raw uint16 bytes (IEEE 754 BF16
          bit pattern) as expected by MilaWeightWriter and Mila's loader.
        - Llama 3.1 8B does not tie word embeddings (tie_word_embeddings=False);
          lm_head.weight is a separate tensor in the state_dict. The 3.2 1B/3B
          variants tie embeddings; the converter handles both cases.
        - Llama 3.1 8B uses rope_scaling with rope_type="llama3" for extended
          context (up to 131072 tokens). Mila uses standard RoPE with
          rope_theta=500000, which is accurate at context lengths <= 4096.
          The rope_scaling dict is printed for reference but not written to the
          Mila binary -- no Mila change required for Phase 5 validation.
"""

import sys
from pathlib import Path
sys.path.insert( 0, str( Path( __file__ ).parent.parent ) )

import argparse
import torch
from transformers import AutoModelForCausalLM
from common import MilaWeightWriter

def _check_hf_error( model_name: str, e: Exception ):
    name = type( e ).__name__
    msg  = str( e )
    if 'GatedRepo' in name or ('403' in msg and 'gated' in msg.lower()):
        print( f"\nError: '{model_name}' is a gated model." )
        print( f"  1. Accept Meta's license at https://huggingface.co/{model_name}" )
        print(  "  2. Authenticate: hf auth login" )
        sys.exit( 1 )
    if 'RepositoryNotFound' in name or '404' in msg:
        print( f"\nError: '{model_name}' not found on HuggingFace." )
        print(  "  Check the model name and your network connection." )
        sys.exit( 1 )
    raise e

# Supported Llama 3.x text model variants
SUPPORTED_MODELS = [
    'meta-llama/Llama-3.2-1B',
    'meta-llama/Llama-3.2-3B',
    'meta-llama/Llama-3.2-1B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
    'meta-llama/Llama-3.1-8B',
    'meta-llama/Llama-3.1-8B-Instruct',
]

TORCH_DTYPE_MAP = {
    'float32':  torch.float32,
    'bfloat16': torch.bfloat16,
}


def _tensor_to_numpy( tensor: torch.Tensor, dtype: str ):
    """
    Convert a torch tensor to a numpy array in the target Mila dtype.

    bfloat16 tensors cannot be represented natively in numpy, so they are
    returned as a uint16 view of the raw BF16 bit patterns. This matches
    the representation expected by MilaWeightWriter._get_dtype_code and
    Mila's binary weight loader.

    The input tensor may be in any dtype; conversion to the target dtype
    is performed before extraction.
    """
    if dtype == 'bfloat16':
        return tensor.to( torch.bfloat16 ).contiguous().view( torch.uint16 ).numpy()
    else:
        return tensor.to( torch.float32 ).numpy()


def convert_llama( model_name: str, output_path: str, dtype: str = 'float32' ):

    torch_dtype = TORCH_DTYPE_MAP[dtype]
    print( f"Loading {model_name} from HuggingFace (dtype={dtype})..." )

    try:
        model = AutoModelForCausalLM.from_pretrained( model_name, torch_dtype=torch_dtype )
    except Exception as e:
        _check_hf_error( model_name, e )

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
    rope_theta = 500000.0  # Llama 3.x default (all validated variants use 500000)
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

    # Sanitize model name -- replace '.' with '_' to avoid conflicts
    # with Mila's component path separator '.'.
    raw_name = model_name.rsplit( '/', 1 )[-1]
    model_id = raw_name.replace( '.', '_' )
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
        _tensor_to_numpy( state_dict['model.embed_tokens.weight'], dtype )
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
        # the actual implementation is LlamaRMSNorm -- not LayerNorm.
        writer.add_tensor(
            f'{prefix_mila}.rmsn_1.weight',
            _tensor_to_numpy( state_dict[f'{prefix_hf}.input_layernorm.weight'], dtype )
        )

        # Fused QKV projection -- concatenate Q, K, V along dim 0.
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
            _tensor_to_numpy( qkv_weight, dtype )
        )

        # Attention output projection.
        # Shape: [hidden_size, hidden_size]  e.g. [2048, 2048]
        writer.add_tensor(
            f'{prefix_mila}.fc_out_proj.weight',
            _tensor_to_numpy( state_dict[f'{prefix_hf}.self_attn.o_proj.weight'], dtype )
        )

        # Post-attention RMSNorm (no bias).
        # Note: HF names this 'post_attention_layernorm' historically --
        # same naming convention issue as input_layernorm above.
        writer.add_tensor(
            f'{prefix_mila}.rmsn_2.weight',
            _tensor_to_numpy( state_dict[f'{prefix_hf}.post_attention_layernorm.weight'], dtype )
        )

        # Fused gate+up projection -- concatenate gate and up along dim 0.
        # gate_proj: [intermediate_size, hidden_size]     e.g. [8192, 2048]
        # up_proj:   [intermediate_size, hidden_size]     e.g. [8192, 2048]
        # Fused:     [2*intermediate_size, hidden_size]   e.g. [16384, 2048]
        # SwiGLU splits the output into gate and up halves internally.
        # No mlp. prefix -- LlamaBlock uses individual Linear components,
        # not an MLP composite.
        gate_weight    = state_dict[f'{prefix_hf}.mlp.gate_proj.weight']
        up_weight      = state_dict[f'{prefix_hf}.mlp.up_proj.weight']
        gate_up_weight = torch.cat( [gate_weight, up_weight], dim=0 )

        writer.add_tensor(
            f'{prefix_mila}.fc_gate_up.weight',
            _tensor_to_numpy( gate_up_weight, dtype )
        )

        # Down projection.
        # Shape: [hidden_size, intermediate_size]  e.g. [2048, 8192]
        writer.add_tensor(
            f'{prefix_mila}.fc_down.weight',
            _tensor_to_numpy( state_dict[f'{prefix_hf}.mlp.down_proj.weight'], dtype )
        )

    # -------------------------------------------------------------------------
    # Final RMSNorm (no bias)
    # -------------------------------------------------------------------------
    writer.add_tensor(
        'rmsn_final.weight',
        _tensor_to_numpy( state_dict['model.norm.weight'], dtype )
    )

    # -------------------------------------------------------------------------
    # LM head
    # Llama 3.2 1B/3B: tied with embed_tokens (tie_word_embeddings=True);
    #   lm_head.weight may not be a separate key in the state_dict.
    # Llama 3.1 8B: separate tensor (tie_word_embeddings=False);
    #   lm_head.weight is always present.
    # Written explicitly in both cases so Mila's loader needs no tying logic.
    # -------------------------------------------------------------------------
    lm_head_key = 'lm_head.weight'
    if lm_head_key in state_dict:
        lm_head_tensor = state_dict[lm_head_key]
    else:
        print( "  Note: lm_head.weight not in state_dict (tied) -- copying from embed_tokens" )
        lm_head_tensor = state_dict['model.embed_tokens.weight']

    writer.add_tensor(
        'lm_head.weight',
        _tensor_to_numpy( lm_head_tensor, dtype )
    )

    writer.write()

    print( f"\nConversion complete!" )
    print( f"  Output: {output_path}" )
    print( f"  Model:  {model_id}" )
    print( f"  dtype:  {dtype}" )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert Llama 3.x weights to Mila format' )
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
        default='bfloat16',
        choices=['float32', 'bfloat16'],
        help='Target dtype for weights (default: bfloat16)'
    )

    args = parser.parse_args()

    convert_llama( args.model, args.output, args.dtype )