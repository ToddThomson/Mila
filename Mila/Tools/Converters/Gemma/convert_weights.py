#!/usr/bin/env python3
# ============================================================================
# File: convert_weights.py
# Convert Gemma 4 weights to Mila format
# ============================================================================

"""
Convert Gemma 4 weights from HuggingFace to Mila binary format.

Target: Gemma 4 12B Unified (dense text chassis). The 5:1 sliding/global layer
interleave, decoupled head_dim, K=V global layers, GeGLU FFN, sandwich norm,
and QK-norm are all read from the model config, so the converter adapts to the
geometry rather than hardcoding it.

Three Gemma-specific transforms are folded in here (rather than at inference),
matching the Step 5d/5e design decisions:

  1. Embedding scale: HF multiplies the embedded hidden states by
     sqrt(hidden_size) at runtime. Mila keeps the token embedding and lm_head
     UNTIED, so we fold the scale into the written embedding table (temb.wte)
     and write lm_head its own UNSCALED copy from the same (HF-tied) tensor.
     The transformer forward then stays structurally identical to Llama.

  2. RAW RMSNorm weights: we write every norm weight AS-IS (all sandwich norms, both
     QK-norms, and the final norm). Gemma's RMSNorm is x_norm * (1 + weight) (HF
     Gemma3RMSNorm), but the +1 is NOT folded here -- Mila applies it at the kernel via
     RmsNormConfig::withUnitOffset(1.0) on every Gemma norm, so the stored weights stay
     identical to the HF checkpoint (zero-centered, directly comparable) and the shared
     RmsNorm kernel stays Llama-safe (offset 0 = raw). Do NOT add 1.0 here. See
     _rmsnorm_to_numpy. (Historical note: an earlier attempt folded +1 into the weights;
     the parity garbage that was blamed on it was actually elsewhere -- the (1+w) convention
     itself is correct, it just belongs at the kernel, not in the stored data.)

  3. K=V global layers: the 1-in-N global (full-attention) layers share K=V and
     have no v_proj. Their fused QKV blob is [Q | K] only; the sliding layers are
     the usual [Q | K | V]. Mila's GemmaBlock<kGlobal=true> aliases V to K.

CAUTION: the exact HF config attribute names and state_dict keys for Gemma 4
(Gemma4TextModel) are read defensively with the documented Gemma.md defaults as
fallbacks. Verify them against the installed `transformers` Gemma 4 implementation
on first run (the script prints every resolved value).

Mila tensor names (must match GemmaTransformer / GemmaBlock component paths):

    Token embedding:
        model.embed_tokens.weight * sqrt(hidden_size)    -> temb.wte

    Per layer (i = 0..num_hidden_layers-1):
        input_layernorm.weight            (+1)           -> tf_layer_{i}.input_norm.weight
        self_attn.q_proj | k_proj [| v_proj]             -> tf_layer_{i}.qkv_proj.weight
                                                              (V section dropped for K=V global layers)
        self_attn.q_norm.weight           (+1)           -> tf_layer_{i}.q_norm.weight
        self_attn.k_norm.weight           (+1)           -> tf_layer_{i}.k_norm.weight
        self_attn.o_proj.weight                          -> tf_layer_{i}.o_proj.weight
        post_attention_layernorm.weight   (+1)           -> tf_layer_{i}.post_attn_norm.weight
        pre_feedforward_layernorm.weight  (+1)           -> tf_layer_{i}.pre_ffn_norm.weight
        mlp.gate_proj | mlp.up_proj                      -> tf_layer_{i}.fc_gate_up.weight
        mlp.down_proj.weight                             -> tf_layer_{i}.fc_down.weight
        post_feedforward_layernorm.weight (+1)           -> tf_layer_{i}.post_ffn_norm.weight

    Final RMSNorm:
        model.norm.weight                 (+1)           -> rmsn_final.weight

    LM head (untied, unscaled):
        lm_head.weight (or embed_tokens.weight if tied)  -> lm_head.weight
"""

import sys
from pathlib import Path
sys.path.insert( 0, str( Path( __file__ ).parent.parent ) )

import argparse
import math
import torch
from transformers import AutoModelForCausalLM
from common import MilaWeightWriter


def _check_hf_error( model_name: str, e: Exception ):
    name = type( e ).__name__
    msg  = str( e )
    if 'GatedRepo' in name or ('403' in msg and 'gated' in msg.lower()):
        print( f"\nError: '{model_name}' is a gated model." )
        print( f"  1. Accept Google's license at https://huggingface.co/{model_name}" )
        print(  "  2. Authenticate: hf auth login" )
        sys.exit( 1 )
    if 'RepositoryNotFound' in name or '404' in msg:
        print( f"\nError: '{model_name}' not found on HuggingFace." )
        print(  "  Check the model name and your network connection." )
        sys.exit( 1 )
    raise e


SUPPORTED_MODELS = [
    'google/gemma-4-12b',
    'google/gemma-4-12b-it',
]

TORCH_DTYPE_MAP = {
    'float32':  torch.float32,
    'bfloat16': torch.bfloat16,
}


def _tensor_to_numpy( tensor: torch.Tensor, dtype: str ):
    """Convert a torch tensor to a numpy array in the target Mila dtype.

    bfloat16 is returned as a uint16 view of the raw BF16 bit pattern, matching
    MilaWeightWriter and Mila's loader.
    """
    if dtype == 'bfloat16':
        return tensor.to( torch.bfloat16 ).contiguous().view( torch.uint16 ).numpy()
    else:
        return tensor.to( torch.float32 ).numpy()


def _rmsnorm_to_numpy( weight: torch.Tensor, dtype: str ):
    """Convert an RMSNorm weight AS-IS (raw, zero-centered) -- do NOT add 1.0 here.

    Gemma's RMSNorm is x_norm * (1 + weight) (HF Gemma3RMSNorm). Mila applies the +1 at the
    kernel via RmsNormConfig::withUnitOffset(1.0) on every Gemma norm, so the stored weights
    must remain RAW (identical to the HF checkpoint, directly comparable, and the shared
    RmsNorm kernel stays Llama-safe with offset 0). All norms (sandwich + final + QK) write
    the raw weight. Confirmed via output_hidden_states + an fp32 oracle: HF residual is small
    (L0 ~= 88); raw-only gave ~1643 (18x), the missing +1 was the bug.
    """
    return _tensor_to_numpy( weight.to( torch.float32 ), dtype )


def _get( config, *names, default=None, required=False ):
    """Read the first present attribute from a HF config (defensive: Gemma 4
    field names are verified on first run)."""
    for n in names:
        if hasattr( config, n ) and getattr( config, n ) is not None:
            return getattr( config, n )
    if required and default is None:
        raise KeyError( f"none of {names} found on the model config" )
    return default


def convert_gemma( model_name: str, output_path: str, dtype: str = 'bfloat16' ):

    torch_dtype = TORCH_DTYPE_MAP[dtype]
    print( f"Loading {model_name} from HuggingFace (dtype={dtype})..." )

    try:
        model = AutoModelForCausalLM.from_pretrained( model_name, dtype=torch_dtype )
    except Exception as e:
        _check_hf_error( model_name, e )

    config = model.config
    # Gemma 4 may nest the text config under config.text_config for the unified
    # (multimodal) checkpoint; fall back to the top-level config for text-only.
    text_config = getattr( config, 'text_config', config )

    hidden_size   = _get( text_config, 'hidden_size', required=True )
    num_layers    = _get( text_config, 'num_hidden_layers', required=True )
    num_heads     = _get( text_config, 'num_attention_heads', required=True )
    num_kv_heads  = _get( text_config, 'num_key_value_heads', required=True )
    head_dim      = _get( text_config, 'head_dim', default=hidden_size // num_heads )
    intermediate  = _get( text_config, 'intermediate_size', required=True )
    vocab_size    = _get( text_config, 'vocab_size', required=True )
    max_seq_len   = _get( text_config, 'max_position_embeddings', required=True )
    rms_eps       = _get( text_config, 'rms_norm_eps', default=1e-6 )

    # Global-layer geometry (Gemma.md section 5 defaults).
    global_head_dim     = _get( text_config, 'global_head_dim', default=512 )
    num_global_kv_heads = _get( text_config, 'num_global_key_value_heads', default=1 )
    key_equals_value    = bool( _get( text_config, 'attention_k_eq_v', default=True ) )

    # Chassis: sliding/global interleave, dual RoPE, logit softcap.
    window                = _get( text_config, 'sliding_window', default=1024 )
    sliding_pattern       = _get( text_config, 'sliding_window_pattern', 'sliding_window_size', default=6 )
    rope_theta_local      = _get( text_config, 'rope_theta', 'rope_local_base_freq', default=10000.0 )
    rope_theta_global     = _get( text_config, 'rope_global_base_freq', 'global_rope_theta', default=1000000.0 )
    partial_rotary_factor = _get( text_config, 'partial_rotary_factor', default=0.25 )
    global_rotary_dim     = int( partial_rotary_factor * global_head_dim )
    final_softcap         = _get( text_config, 'final_logit_softcapping', default=30.0 )
    tie_embeddings        = bool( _get( text_config, 'tie_word_embeddings', default=True ) )

    def is_global_layer( i ): return ((i + 1) % sliding_pattern) == 0

    print( "Resolved Gemma config (verify against the installed transformers Gemma 4):" )
    for k, v in {
        'hidden_size': hidden_size, 'num_hidden_layers': num_layers,
        'num_attention_heads': num_heads, 'num_key_value_heads': num_kv_heads,
        'head_dim': head_dim, 'global_head_dim': global_head_dim,
        'num_global_key_value_heads': num_global_kv_heads, 'attention_k_eq_v': key_equals_value,
        'intermediate_size': intermediate, 'vocab_size': vocab_size,
        'max_position_embeddings': max_seq_len, 'rms_norm_eps': rms_eps,
        'sliding_window': window, 'sliding_window_pattern': sliding_pattern,
        'rope_theta_local': rope_theta_local, 'rope_theta_global': rope_theta_global,
        'global_rotary_dim': global_rotary_dim, 'final_logit_softcapping': final_softcap,
        'tie_word_embeddings': tie_embeddings,
    }.items():
        print( f"  {k:30s} {v}" )

    raw_name = model_name.rsplit( '/', 1 )[-1]
    model_id = raw_name.replace( '.', '_' ).replace( '-', '_' )

    writer = MilaWeightWriter( output_path )
    writer.set_metadata( {
        'architecture':            'gemma',
        'model_name':              model_id,
        'dtype':                   dtype,
        'vocab_size':              vocab_size,
        'hidden_size':             hidden_size,
        'embedding_dim':           hidden_size,
        'num_layers':              num_layers,
        'num_heads':               num_heads,
        'num_kv_heads':            num_kv_heads,
        'head_dim':                head_dim,
        'hidden_dim':              intermediate,
        'max_seq_length':          max_seq_len,
        'norm_epsilon':            rms_eps,
        'global_head_dim':         global_head_dim,
        'num_global_kv_heads':     num_global_kv_heads,
        'key_equals_value':        key_equals_value,
        'window':                  window,
        'sliding_window_pattern':  sliding_pattern,
        'global_rotary_dim':       global_rotary_dim,
        'rope_theta_local':        rope_theta_local,
        'rope_theta_global':       rope_theta_global,
        'final_logit_softcapping': final_softcap,
        'use_bias':                False,
        'activation':              'gelu_tanh',
        'norm_type':               'rmsnorm',
        'attention_type':          'gqa',
        'positional_encoding':     'rope',
        'tie_word_embeddings':     tie_embeddings,
    } )

    # For Gemma4ForConditionalGeneration (multimodal checkpoint), the text LM
    # is nested under .language_model; use it directly so state-dict key paths
    # are identical to the text-only (Gemma4ForCausalLM) variant.
    text_model = getattr( model, 'language_model', model )
    state_dict = text_model.state_dict()

    # Locate embed_tokens to determine the actual key prefix. Known layouts:
    #   text-only  (Gemma4ForCausalLM):              'model.'
    #   multimodal (Gemma4ForConditionalGeneration):  'model.language_model.'
    _EMBED_SUFFIX = 'embed_tokens.weight'
    _embed_key = next( (k for k in state_dict if k.endswith( _EMBED_SUFFIX )), None )
    if _embed_key is None:
        sample = '\n  '.join( list( state_dict.keys() )[:30] )
        raise KeyError( f"embed_tokens.weight not found in state_dict. First 30 keys:\n  {sample}" )
    _key_prefix = _embed_key[: -len( _EMBED_SUFFIX )]
    if _key_prefix != 'model.':
        print( f"  Note: multimodal checkpoint detected; key prefix is '{_key_prefix}'" )

    def sd( key ):
        # Converter keys follow the text-only convention: 'model.' prefix for
        # submodule tensors (embed, layers, norm). Replace it with the detected
        # prefix. Top-level keys (lm_head.weight) have no 'model.' prefix; try
        # bare first then fall back to the prefixed form.
        if key.startswith( 'model.' ):
            k = _key_prefix + key[len( 'model.' ):]
        else:
            k = key if key in state_dict else _key_prefix + key
        if k not in state_dict:
            raise KeyError( f"expected tensor '{key}' not in state_dict (tried '{k}')" )
        return state_dict[k]

    # ----- Token embedding: fold the sqrt(hidden_size) scale in -----
    normalizer = math.sqrt( hidden_size )
    embed = sd( 'model.embed_tokens.weight' )
    print( f"Embedding scale (normalizer = sqrt({hidden_size})) = {normalizer:.6f}" )
    writer.add_tensor( 'temb.wte', _tensor_to_numpy( embed.to( torch.float32 ) * normalizer, dtype ) )

    # ----- Transformer layers -----
    for i in range( num_layers ):
        hf   = f'model.layers.{i}'
        mila = f'tf_layer_{i}'
        kind = 'global' if is_global_layer( i ) else 'local'
        print( f"  Converting layer {i}/{num_layers - 1} ({kind})..." )

        writer.add_tensor( f'{mila}.input_norm.weight',
            _rmsnorm_to_numpy( sd( f'{hf}.input_layernorm.weight' ), dtype ) )

        # Fused QKV: [Q | K | V] for sliding, [Q | K] for K=V global layers.
        q = sd( f'{hf}.self_attn.q_proj.weight' )
        k = sd( f'{hf}.self_attn.k_proj.weight' )
        if is_global_layer( i ) and key_equals_value:
            qkv = torch.cat( [q, k], dim=0 )
        else:
            v = sd( f'{hf}.self_attn.v_proj.weight' )
            qkv = torch.cat( [q, k, v], dim=0 )
        writer.add_tensor( f'{mila}.qkv_proj.weight', _tensor_to_numpy( qkv, dtype ) )

        # QK-norm (per-head RMSNorm over head_dim) uses the RAW weight, like every other
        # Gemma 4 norm (see _rmsnorm_to_numpy). First place the raw-weight convention was
        # confirmed: q_norm/k_norm OUTPUT RMS == raw stored weight (q 1.02, k 0.12), not (1+w)
        # (2.03 / 1.12). Written raw via _rmsnorm_to_numpy below.
        writer.add_tensor( f'{mila}.q_norm.weight',
            _rmsnorm_to_numpy( sd( f'{hf}.self_attn.q_norm.weight' ), dtype ) )
        writer.add_tensor( f'{mila}.k_norm.weight',
            _rmsnorm_to_numpy( sd( f'{hf}.self_attn.k_norm.weight' ), dtype ) )

        # v_norm: Gemma 4 normalizes V per head too, but with_scale=False (NO learnable weight,
        # pure magnitude normalize V/RMS(V)). HF stores no v_norm.weight. Mila's RMSNorm always
        # applies a weight, so write a UNIT weight (normalize * 1 == normalize). head_dim is
        # global_head_dim (512) on the full-attention layers.
        _vnorm_hd = global_head_dim if is_global_layer( i ) else head_dim
        writer.add_tensor( f'{mila}.v_norm.weight',
            _tensor_to_numpy( torch.ones( _vnorm_hd ), dtype ) )

        writer.add_tensor( f'{mila}.o_proj.weight',
            _tensor_to_numpy( sd( f'{hf}.self_attn.o_proj.weight' ), dtype ) )

        writer.add_tensor( f'{mila}.post_attn_norm.weight',
            _rmsnorm_to_numpy( sd( f'{hf}.post_attention_layernorm.weight' ), dtype ) )

        writer.add_tensor( f'{mila}.pre_ffn_norm.weight',
            _rmsnorm_to_numpy( sd( f'{hf}.pre_feedforward_layernorm.weight' ), dtype ) )

        # GeGLU fused gate+up: [gate | up].
        gate = sd( f'{hf}.mlp.gate_proj.weight' )
        up   = sd( f'{hf}.mlp.up_proj.weight' )
        writer.add_tensor( f'{mila}.fc_gate_up.weight',
            _tensor_to_numpy( torch.cat( [gate, up], dim=0 ), dtype ) )

        writer.add_tensor( f'{mila}.fc_down.weight',
            _tensor_to_numpy( sd( f'{hf}.mlp.down_proj.weight' ), dtype ) )

        writer.add_tensor( f'{mila}.post_ffn_norm.weight',
            _rmsnorm_to_numpy( sd( f'{hf}.post_feedforward_layernorm.weight' ), dtype ) )

        # Gemma4Unified per-layer output scale: x_out = (x_mid + post_ffn(...)) * layer_scalar.
        # A learned [1] scalar (varies wildly per layer, e.g. L0 0.053). Written FP32 for
        # precision; GemmaBlock loads it and scales res2. WITHOUT this the residual stream
        # blows up ~18x (the 5f parity bug). NOTE: this is unique to Gemma 4 (not Gemma 3).
        writer.add_tensor( f'{mila}.layer_scalar',
            _tensor_to_numpy( sd( f'{hf}.layer_scalar' ).reshape( -1 ), 'float32' ) )

    # ----- Final RMSNorm -----
    writer.add_tensor( 'rmsn_final.weight',
        _rmsnorm_to_numpy( sd( 'model.norm.weight' ), dtype ) )

    # ----- LM head: untied, UNSCALED (own copy from the tied tensor) -----
    try:
        lm_head = sd( 'lm_head.weight' )
    except KeyError:
        print( "  Note: lm_head.weight tied -- writing an unscaled copy of embed_tokens" )
        lm_head = embed
    writer.add_tensor( 'lm_head.weight', _tensor_to_numpy( lm_head, dtype ) )

    writer.write()

    print( f"\nConversion complete!" )
    print( f"  Output: {output_path}" )
    print( f"  Model:  {model_id}  dtype: {dtype}" )


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description='Convert Gemma 4 weights to Mila format' )
    parser.add_argument( '--model', type=str, required=True, choices=SUPPORTED_MODELS,
        help='HuggingFace model name' )
    parser.add_argument( '--output', type=str, required=True,
        help='Output path for the Mila weight file' )
    parser.add_argument( '--dtype', type=str, default='bfloat16',
        choices=['float32', 'bfloat16'], help='Target dtype (default: bfloat16)' )

    args = parser.parse_args()
    convert_gemma( args.model, args.output, args.dtype )
