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

The HuggingFace -> Mila weight mapping lives in common.py (LLAMA_LAYER_TENSORS and
its neighbours), because the Qwen3.8 codebook packer needs the same map to name
what it emits. This file drives its emission from that map rather than restating it.

    Notes:
        - LlamaBlock uses individual Linear components for FFN (no MLP
          composite) -- fc_gate_up and fc_down have no mlp. prefix.
        - model_name sanitized: '.' replaced with '_' to avoid conflicts
          with Mila's component path separator '.'.
        - bfloat16 tensors are stored as raw uint16 bytes (IEEE 754 BF16
          bit pattern) as expected by MilaWeightWriter and Mila's loader.
        - Llama 3.1 8B does not tie word embeddings (tie_word_embeddings=False);
          lm_head.weight is a separate tensor in the state_dict. The 3.2 1B/3B
          variants tie embeddings; the converter handles both cases.
        - Llama 3.1 and 3.2 scale their rotary frequencies (rope_type "llama3"),
          which they read every position through. The rule and its four values
          are written as rope_scaling / rope_scaling_factor /
          rope_low_frequency_factor / rope_high_frequency_factor /
          rope_original_context_length; a checkpoint without scaling writes
          rope_scaling "none". Mila refuses a Llama file that says neither.
"""

import sys
from pathlib import Path
sys.path.insert( 0, str( Path( __file__ ).parent.parent ) )

import argparse
import torch
from transformers import AutoModelForCausalLM
from common import MilaWeightWriter, expand_llama_tensor_map

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


def _rope_scaling_metadata( config ) -> dict:
    """The rotary frequency scaling the checkpoint was trained with, as Mila's weights metadata names it."""
    # transformers 5 keeps it in rope_parameters (also exposed as rope_scaling); 4.x in rope_scaling, None when absent.
    parameters = getattr( config, 'rope_parameters', None ) or getattr( config, 'rope_scaling', None ) or {}
    rope_type = parameters.get( 'rope_type', parameters.get( 'type', 'default' ) )

    if rope_type == 'default':
        return { 'rope_scaling': 'none' }

    if rope_type == 'llama3':
        return {
            'rope_scaling':                 'llama3',
            'rope_scaling_factor':          float( parameters['factor'] ),
            'rope_low_frequency_factor':    float( parameters['low_freq_factor'] ),
            'rope_high_frequency_factor':   float( parameters['high_freq_factor'] ),
            'rope_original_context_length': int( parameters['original_max_position_embeddings'] ),
        }

    raise ValueError( f"rope_type '{rope_type}' is not one Mila implements" )


def convert_llama( model_name: str, output_path: str, dtype: str = 'float32' ):

    torch_dtype = TORCH_DTYPE_MAP[dtype]
    print( f"Loading {model_name} from HuggingFace (dtype={dtype})..." )

    try:
        model = AutoModelForCausalLM.from_pretrained( model_name, dtype=torch_dtype )
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
        **_rope_scaling_metadata( config ),
    } )

    state_dict = model.state_dict()

    # Llama 3.2 1B/3B tie lm_head to the embedding, so 'lm_head.weight' may not be
    # a separate key; Llama 3.1 8B does not tie and always has it. Written
    # explicitly in both cases so Mila's loader needs no tying logic.
    if 'lm_head.weight' not in state_dict:
        print( "  Note: lm_head.weight not in state_dict (tied) -- copying from embed_tokens" )
        state_dict['lm_head.weight'] = state_dict['model.embed_tokens.weight']

    reported_layer = -1

    for mapping in expand_llama_tensor_map( config.num_hidden_layers ):

        # One line per layer rather than per tensor; the map is flat, so the
        # layer index is recovered from the name it produced.
        if mapping.mila.startswith( 'tf_layer_' ):
            layer = int( mapping.mila.split( '.', 1 )[0].removeprefix( 'tf_layer_' ) )

            if layer != reported_layer:
                print( f"  Converting layer {layer}/{config.num_hidden_layers - 1}..." )
                reported_layer = layer

        sources = [state_dict[source] for source in mapping.sources]

        # A fused Mila tensor concatenates its sources along dim 0, in map order:
        # fc_qkv_proj as [Q | K | V] for GQA, fc_gate_up as [gate | up] for SwiGLU.
        tensor = sources[0] if len( sources ) == 1 else torch.cat( sources, dim=0 )

        writer.add_tensor( mapping.mila, _tensor_to_numpy( tensor, dtype ) )

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