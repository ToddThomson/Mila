#!/usr/bin/env python3
# ============================================================================
# File: convert_weights.py
# Convert Gemma 4 weights to Mila format
# ============================================================================

"""
Convert Gemma 4 weights from HuggingFace to Mila binary format.

Targets: Gemma 4 12B (dense) and Gemma 4 26B-A4B (mixture of experts). The 5:1
sliding/global interleave, decoupled head_dim, K=V global layers, GeGLU FFN, sandwich
norm, QK-norm and the routed feed-forward are all read from config.json, so the
converter adapts to the geometry rather than hardcoding it.

THIS CONVERTER STREAMS. The 26B checkpoint is 48 GiB against 31.8 GB of host RAM, so
shards are read through safetensors one tensor at a time and written through
MilaStreamingWeightWriter, whose index is declared from the shard headers before any
data moves. On a dense model its output is byte-identical to the from_pretrained
converter it replaced (Specifications/Gemma4MoE.md Phase 8).

Gemma-specific transforms handled in this converter:

  1. Embedding scale + weight tying: HF multiplies the embedded hidden states by
     sqrt(hidden_size) at runtime. Mila ALSO applies this at runtime
     (TokenEmbedding::forward via TokenEmbeddingConfig::embedding_scale, set in
     GemmaTransformer::createGraph), so the embedding table is written RAW. With
     tie_word_embeddings (the Gemma 4 default) the lm_head shares this raw table:
     the lm_head.weight blob is omitted and GemmaTransformer aliases at load time
     (WeightTying.md).

  2. RAW RMSNorm weights: every norm weight is written AS-IS. Gemma 4's RMSNorm
     multiplies by the stored weight directly (HF Gemma4RMSNorm: x_norm * weight),
     unlike Gemma 3's x_norm * (1 + weight), and GemmaBlock runs every norm at
     RmsNormConfig unit offset 0. Do NOT add 1.0 here. See _rmsnorm_to_numpy.

  3. K=V global layers: the 1-in-N global (full-attention) layers share K=V and
     have no v_proj. Their fused QKV blob is [Q | K] only; the sliding layers are
     the usual [Q | K | V]. Mila's GemmaBlock<kGlobal=true> derives V from K.

  4. Mixture of experts (enable_moe_block): the always-on dense branch is written
     under `mlp`, where GemmaBlock<kMixtureOfExperts> delegates it. The expert bank
     ships stacked and is written as-is, gate first in gate_up_proj. The router's
     three tensors and the three extra norms are written raw, and per_expert_scale is
     NOT folded into down_proj.

Mila tensor names (must match GemmaTransformer / GemmaBlock component paths):

    Token embedding:
        embed_tokens.weight (raw; scale applied at runtime)  -> temb.wte

    Per layer (i = 0..num_hidden_layers-1):
        input_layernorm.weight                           -> tf_layer_{i}.input_norm.weight
        self_attn.q_proj | k_proj [| v_proj]             -> tf_layer_{i}.qkv_proj.weight
                                                              (V section dropped for K=V global layers)
        self_attn.q_norm.weight                          -> tf_layer_{i}.q_norm.weight
        self_attn.k_norm.weight                          -> tf_layer_{i}.k_norm.weight
        (none; unit weight written)                      -> tf_layer_{i}.v_norm.weight
        self_attn.o_proj.weight                          -> tf_layer_{i}.o_proj.weight
        post_attention_layernorm.weight                  -> tf_layer_{i}.post_attn_norm.weight
        pre_feedforward_layernorm.weight                 -> tf_layer_{i}.pre_ffn_norm.weight
        mlp.gate_proj | mlp.up_proj                      -> tf_layer_{i}[.mlp].fc_gate_up.weight
        mlp.down_proj.weight                             -> tf_layer_{i}[.mlp].fc_down.weight
      mixture of experts only:
        post_feedforward_layernorm_1.weight              -> tf_layer_{i}.post_ffn_norm_1.weight
        pre_feedforward_layernorm_2.weight               -> tf_layer_{i}.pre_ffn_norm_2.weight
        router.proj.weight                               -> tf_layer_{i}.router.proj.weight
        router.scale                                     -> tf_layer_{i}.router.scale
        router.per_expert_scale                          -> tf_layer_{i}.router.per_expert_scale
        experts.gate_up_proj                             -> tf_layer_{i}.experts.gate_up_proj
        experts.down_proj                                -> tf_layer_{i}.experts.down_proj
        post_feedforward_layernorm_2.weight              -> tf_layer_{i}.post_ffn_norm_2.weight
      every layer:
        post_feedforward_layernorm.weight                -> tf_layer_{i}.post_ffn_norm.weight
        layer_scalar                                     -> tf_layer_{i}.layer_scalar (FP32)

    Final RMSNorm:
        norm.weight                                      -> rmsn_final.weight

    LM head:
        (tied: omitted -- shares temb.wte at load time)
        lm_head.weight (untied case only)                -> lm_head.weight

Usage:
    python Gemma/convert_weights.py --model google/gemma-4-26B-A4B-it \
        --output <weights-dir>/gemma/gemma4_26b_a4b_it_bf16.bin

    # A local checkpoint directory converts with no hub access
    python Gemma/convert_weights.py --model <checkpoint-dir> --output <file>
"""

import sys
from pathlib import Path
sys.path.insert( 0, str( Path( __file__ ).resolve().parent.parent ) )

import argparse
import json
import re
from dataclasses import dataclass
from typing import Tuple

import torch

from common import MilaStreamingWeightWriter, ShardedCheckpoint


SUPPORTED_MODELS = [
    'google/gemma-4-12b',
    'google/gemma-4-12b-it',
    'google/gemma-4-26B-A4B',
    'google/gemma-4-26B-A4B-it',
]

# Checkpoint tensors the text chassis does not model. Named as prefixes rather than
# discovered, so a tensor family that appears in a future revision is reported as
# unconsumed instead of being silently dropped.
SKIPPED_PREFIXES = ( 'model.vision_tower.', 'model.embed_vision.', 'model.audio_tower.', 'model.embed_audio.' )


@dataclass( frozen=True )
class GemmaTensor:
    """One Mila tensor and how it is built.

    transform: 'concat' (sources along dim 0), 'norm' (a raw RMSNorm weight),
    'unit' (a weight of ones, `width` long, with no source), 'scalar' (flattened, FP32).
    """
    mila: str
    sources: Tuple[ str, ... ]
    transform: str = 'concat'
    width: int = 0


def _check_hf_error( model_name: str, e: Exception ):
    name = type( e ).__name__
    msg  = str( e )
    if 'RepositoryNotFound' in name or '404' in msg:
        print( f"\nError: '{model_name}' not found on HuggingFace." )
        print(  "  Check the model name and your network connection." )
        sys.exit( 1 )
    raise e


def _resolve_checkpoint( model_name: str ) -> Path:
    """Local directory holding the checkpoint, downloading it only if absent."""
    candidate = Path( model_name )

    if candidate.is_dir():
        return candidate

    if model_name not in SUPPORTED_MODELS:
        raise ValueError( f"'{model_name}' is neither a checkpoint directory nor one of {SUPPORTED_MODELS}" )

    from huggingface_hub import snapshot_download

    print( f"Resolving {model_name} (downloads only what is missing from the hub cache)..." )

    try:
        return Path( snapshot_download(
            model_name,
            allow_patterns=[ 'config.json', '*.safetensors', '*.safetensors.index.json' ] ) )
    except Exception as e:
        _check_hf_error( model_name, e )


def _value( config: dict, name: str, default ):
    """A config field, or the default when it is absent or null."""
    value = config.get( name )

    return default if value is None else value


def _tensor_to_numpy( tensor: torch.Tensor, dtype: str ):
    """Convert a torch tensor to a numpy array in the target Mila dtype.

    bfloat16 is returned as a uint16 view of the raw BF16 bit pattern, matching
    MilaStreamingWeightWriter and Mila's loader.
    """
    if dtype == 'bfloat16':
        return tensor.to( torch.bfloat16 ).contiguous().view( torch.uint16 ).numpy()
    else:
        return tensor.to( torch.float32 ).numpy()


def _rmsnorm_to_numpy( weight: torch.Tensor, dtype: str ):
    """Convert an RMSNorm weight AS-IS (raw) -- do NOT add 1.0 here.

    Gemma 4's RMSNorm is x_norm * weight (HF Gemma4RMSNorm), and GemmaBlock runs every Gemma
    norm at unit offset 0, so the stored weight is the one the kernel multiplies by. Confirmed
    at the QK norms: output RMS equals the raw weight (q 1.02, k 0.12), not 1 + weight
    (2.03 / 1.12). The ~18x residual blow-up once blamed on this was the missing layer_scalar.
    """
    return _tensor_to_numpy( weight.to( torch.float32 ), dtype )


def resolve_gemma_geometry( config: dict, max_layers: int = 0 ) -> dict:
    """Every geometry field Mila needs, read and validated from a Gemma 4 config.json."""
    # The published checkpoints are multimodal and nest the text geometry under text_config.
    text = config.get( 'text_config', config )

    hidden_size = text[ 'hidden_size' ]
    num_layers = text[ 'num_hidden_layers' ]
    num_heads = text[ 'num_attention_heads' ]
    global_head_dim = _value( text, 'global_head_dim', 512 )

    # Mila derives the interleave from a period, so a layer_types list that is not one is refused
    # rather than converted into the wrong chassis.
    layer_types = _value( text, 'layer_types', None )
    pattern = _value( text, 'sliding_window_pattern', 6 )

    if layer_types:
        if 'full_attention' not in layer_types:
            raise ValueError( 'layer_types has no full_attention layer' )

        pattern = layer_types.index( 'full_attention' ) + 1
        periodic = [ 'full_attention' if ((i + 1) % pattern) == 0 else 'sliding_attention'
                     for i in range( len( layer_types ) ) ]

        if layer_types != periodic:
            raise ValueError( f'layer_types is not a period of {pattern}; GemmaConfig carries a period' )

    # rope_theta and the partial rotary factor moved into per-layer-type rope_parameters in
    # transformers 5.x; the flat spellings are the fallback.
    rope = _value( text, 'rope_parameters', {} )
    sliding_rope = rope.get( 'sliding_attention', {} )
    full_rope = rope.get( 'full_attention', {} )

    rope_theta_local = sliding_rope.get( 'rope_theta',
        _value( text, 'rope_theta', _value( text, 'rope_local_base_freq', 10000.0 ) ) )
    rope_theta_global = full_rope.get( 'rope_theta',
        _value( text, 'rope_global_base_freq', _value( text, 'global_rope_theta', 1000000.0 ) ) )
    partial_rotary_factor = full_rope.get( 'partial_rotary_factor',
        _value( text, 'partial_rotary_factor', 0.25 ) )

    for field, refused in ( ( 'hidden_size_per_layer_input', 'per-layer inputs' ),
                            ( 'num_kv_shared_layers', 'shared KV layers' ),
                            ( 'attention_bias', 'attention biases' ),
                            ( 'use_double_wide_mlp', 'a double-wide MLP' ) ):
        if _value( text, field, 0 ):
            raise ValueError( f'{field} is set; GemmaBlock does not model {refused}' )

    if _value( text, 'use_bidirectional_attention', None ) == 'all':
        raise ValueError( 'use_bidirectional_attention is "all"; GemmaBlock is causal' )

    mixture_of_experts = bool( _value( text, 'enable_moe_block', False ) )

    if max_layers:
        num_layers = min( num_layers, max_layers )

    return {
        'hidden_size': hidden_size,
        'num_hidden_layers': num_layers,
        'num_attention_heads': num_heads,
        'num_key_value_heads': text[ 'num_key_value_heads' ],
        'head_dim': _value( text, 'head_dim', hidden_size // num_heads ),
        'global_head_dim': global_head_dim,
        'num_global_key_value_heads': _value( text, 'num_global_key_value_heads', 1 ),
        'attention_k_eq_v': bool( _value( text, 'attention_k_eq_v', True ) ),
        'intermediate_size': text[ 'intermediate_size' ],
        'vocab_size': text[ 'vocab_size' ],
        'max_position_embeddings': text[ 'max_position_embeddings' ],
        'rms_norm_eps': _value( text, 'rms_norm_eps', 1e-6 ),
        'sliding_window': _value( text, 'sliding_window', 1024 ),
        'sliding_window_pattern': pattern,
        'rope_theta_local': rope_theta_local,
        'rope_theta_global': rope_theta_global,
        'global_rotary_dim': int( partial_rotary_factor * global_head_dim ),
        'final_logit_softcapping': _value( text, 'final_logit_softcapping', 30.0 ),
        'tie_word_embeddings': bool( _value( text, 'tie_word_embeddings', True ) ),
        'enable_moe_block': mixture_of_experts,
        'num_experts': text[ 'num_experts' ] if mixture_of_experts else 0,
        'top_k_experts': text[ 'top_k_experts' ] if mixture_of_experts else 0,
        'moe_intermediate_size': text[ 'moe_intermediate_size' ] if mixture_of_experts else 0,
    }


def gemma_mila_metadata( geometry: dict, dtype: str, model_id: str ) -> dict:
    """The metadata block Mila's reader parses. Key order is the dense converter's, unchanged."""
    metadata = {
        'architecture':            'gemma',
        'model_name':              model_id,
        'dtype':                   dtype,
        'vocab_size':              geometry[ 'vocab_size' ],
        'hidden_size':             geometry[ 'hidden_size' ],
        'embedding_dim':           geometry[ 'hidden_size' ],
        'num_layers':              geometry[ 'num_hidden_layers' ],
        'num_heads':               geometry[ 'num_attention_heads' ],
        'num_kv_heads':            geometry[ 'num_key_value_heads' ],
        'head_dim':                geometry[ 'head_dim' ],
        'hidden_dim':              geometry[ 'intermediate_size' ],
        'max_seq_length':          geometry[ 'max_position_embeddings' ],
        'norm_epsilon':            geometry[ 'rms_norm_eps' ],
        'global_head_dim':         geometry[ 'global_head_dim' ],
        'num_global_kv_heads':     geometry[ 'num_global_key_value_heads' ],
        'key_equals_value':        geometry[ 'attention_k_eq_v' ],
        'window':                  geometry[ 'sliding_window' ],
        'sliding_window_pattern':  geometry[ 'sliding_window_pattern' ],
        'global_rotary_dim':       geometry[ 'global_rotary_dim' ],
        'rope_theta_local':        geometry[ 'rope_theta_local' ],
        'rope_theta_global':       geometry[ 'rope_theta_global' ],
        'final_logit_softcapping': geometry[ 'final_logit_softcapping' ],
        'use_bias':                False,
        'activation':              'gelu_tanh',
        'norm_type':               'rmsnorm',
        'attention_type':          'gqa',
        'positional_encoding':     'rope',
        'tie_word_embeddings':     geometry[ 'tie_word_embeddings' ],
    }

    if geometry[ 'enable_moe_block' ]:
        metadata[ 'num_experts' ] = geometry[ 'num_experts' ]
        metadata[ 'top_k_experts' ] = geometry[ 'top_k_experts' ]
        metadata[ 'expert_hidden_dim' ] = geometry[ 'moe_intermediate_size' ]

    return metadata


def expand_gemma_tensor_map( geometry: dict, prefix: str ):
    """The full HF -> Mila map, in the order the dense converter emitted."""
    pattern = geometry[ 'sliding_window_pattern' ]
    routed = geometry[ 'enable_moe_block' ]

    tensors = [ GemmaTensor( 'temb.wte', ( f'{prefix}embed_tokens.weight', ) ) ]

    for i in range( geometry[ 'num_hidden_layers' ] ):
        hf = f'{prefix}layers.{i}'
        mila = f'tf_layer_{i}'
        is_global = ((i + 1) % pattern) == 0
        key_equals_value = is_global and geometry[ 'attention_k_eq_v' ]

        qkv_sources = ( f'{hf}.self_attn.q_proj.weight', f'{hf}.self_attn.k_proj.weight' )

        if not key_equals_value:
            qkv_sources += ( f'{hf}.self_attn.v_proj.weight', )

        # Dense layers keep the inline FFN names; a routed block delegates its dense branch to `mlp`.
        feed_forward = f'{mila}.mlp' if routed else mila

        tensors += [
            GemmaTensor( f'{mila}.input_norm.weight', ( f'{hf}.input_layernorm.weight', ), 'norm' ),
            GemmaTensor( f'{mila}.qkv_proj.weight', qkv_sources ),
            GemmaTensor( f'{mila}.q_norm.weight', ( f'{hf}.self_attn.q_norm.weight', ), 'norm' ),
            GemmaTensor( f'{mila}.k_norm.weight', ( f'{hf}.self_attn.k_norm.weight', ), 'norm' ),
            # v_norm has no learnable scale (with_scale=False), so HF stores no weight; Mila's
            # RMSNorm always applies one, and a unit weight makes it the pure normalize.
            GemmaTensor( f'{mila}.v_norm.weight', (), 'unit',
                geometry[ 'global_head_dim' ] if is_global else geometry[ 'head_dim' ] ),
            GemmaTensor( f'{mila}.o_proj.weight', ( f'{hf}.self_attn.o_proj.weight', ) ),
            GemmaTensor( f'{mila}.post_attn_norm.weight', ( f'{hf}.post_attention_layernorm.weight', ), 'norm' ),
            GemmaTensor( f'{mila}.pre_ffn_norm.weight', ( f'{hf}.pre_feedforward_layernorm.weight', ), 'norm' ),
            GemmaTensor( f'{feed_forward}.fc_gate_up.weight',
                ( f'{hf}.mlp.gate_proj.weight', f'{hf}.mlp.up_proj.weight' ) ),
            GemmaTensor( f'{feed_forward}.fc_down.weight', ( f'{hf}.mlp.down_proj.weight', ) ),
        ]

        if routed:
            tensors += [
                GemmaTensor( f'{mila}.post_ffn_norm_1.weight', ( f'{hf}.post_feedforward_layernorm_1.weight', ), 'norm' ),
                GemmaTensor( f'{mila}.pre_ffn_norm_2.weight', ( f'{hf}.pre_feedforward_layernorm_2.weight', ), 'norm' ),
                GemmaTensor( f'{mila}.router.proj.weight', ( f'{hf}.router.proj.weight', ) ),
                GemmaTensor( f'{mila}.router.scale', ( f'{hf}.router.scale', ) ),
                GemmaTensor( f'{mila}.router.per_expert_scale', ( f'{hf}.router.per_expert_scale', ) ),
                GemmaTensor( f'{mila}.experts.gate_up_proj', ( f'{hf}.experts.gate_up_proj', ) ),
                GemmaTensor( f'{mila}.experts.down_proj', ( f'{hf}.experts.down_proj', ) ),
                GemmaTensor( f'{mila}.post_ffn_norm_2.weight', ( f'{hf}.post_feedforward_layernorm_2.weight', ), 'norm' ),
            ]

        tensors += [
            GemmaTensor( f'{mila}.post_ffn_norm.weight', ( f'{hf}.post_feedforward_layernorm.weight', ), 'norm' ),
            # A learned [1] per-layer output scale, written FP32 at every dtype.
            GemmaTensor( f'{mila}.layer_scalar', ( f'{hf}.layer_scalar', ), 'scalar' ),
        ]

    tensors.append( GemmaTensor( 'rmsn_final.weight', ( f'{prefix}norm.weight', ), 'norm' ) )

    if not geometry[ 'tie_word_embeddings' ]:
        tensors.append( GemmaTensor( 'lm_head.weight', ( 'lm_head.weight', ) ) )

    return tensors


def _output_shape( tensor: GemmaTensor, checkpoint: ShardedCheckpoint ):
    """The shape a mapping produces, from the source shapes alone."""
    if tensor.transform == 'unit':
        return ( tensor.width, )

    shapes = [ checkpoint.shape( source ) for source in tensor.sources ]

    if tensor.transform == 'scalar':
        count = 1

        for dim in shapes[ 0 ]:
            count *= dim

        return ( count, )

    if tensor.transform == 'concat':
        return ( sum( shape[ 0 ] for shape in shapes ), ) + tuple( shapes[ 0 ][ 1: ] )

    return tuple( shapes[ 0 ] )


def _materialize( tensor: GemmaTensor, checkpoint: ShardedCheckpoint, dtype: str ):
    if tensor.transform == 'unit':
        return _tensor_to_numpy( torch.ones( tensor.width ), dtype )

    sources = [ checkpoint.tensor( source ) for source in tensor.sources ]

    if tensor.transform == 'norm':
        return _rmsnorm_to_numpy( sources[ 0 ], dtype )

    if tensor.transform == 'scalar':
        return _tensor_to_numpy( sources[ 0 ].reshape( -1 ), 'float32' )

    joined = sources[ 0 ] if len( sources ) == 1 else torch.cat( sources, dim=0 )

    return _tensor_to_numpy( joined, dtype )


def _verify_geometry( entries, geometry: dict ):
    """Check every declared shape against what the Gemma components allocate.

    The declaration pass derives shapes from the checkpoint; this derives them from the
    config, independently -- worth an hour of conversion to find a disagreement first.
    """
    hidden = geometry[ 'hidden_size' ]
    heads = geometry[ 'num_attention_heads' ]
    pattern = geometry[ 'sliding_window_pattern' ]
    experts = geometry[ 'num_experts' ]
    expert_hidden = geometry[ 'moe_intermediate_size' ]

    for name, _, shape in entries:
        match = re.match( r'tf_layer_(\d+)\.(.+)$', name )

        if match is None:
            want = { 'temb.wte': ( geometry[ 'vocab_size' ], hidden ),
                     'rmsn_final.weight': ( hidden, ),
                     'lm_head.weight': ( geometry[ 'vocab_size' ], hidden ) }.get( name )
        else:
            layer = int( match.group( 1 ) )
            stem = match.group( 2 ).removeprefix( 'mlp.' )
            is_global = ((layer + 1) % pattern) == 0
            head_dim = geometry[ 'global_head_dim' ] if is_global else geometry[ 'head_dim' ]
            kv_heads = geometry[ 'num_global_key_value_heads' ] if is_global else geometry[ 'num_key_value_heads' ]
            kv_sections = 1 if is_global and geometry[ 'attention_k_eq_v' ] else 2

            want = {
                'qkv_proj.weight': ( (heads + kv_sections * kv_heads) * head_dim, hidden ),
                'q_norm.weight': ( head_dim, ),
                'k_norm.weight': ( head_dim, ),
                'v_norm.weight': ( head_dim, ),
                'o_proj.weight': ( hidden, heads * head_dim ),
                'fc_gate_up.weight': ( 2 * geometry[ 'intermediate_size' ], hidden ),
                'fc_down.weight': ( hidden, geometry[ 'intermediate_size' ] ),
                'router.proj.weight': ( experts, hidden ),
                'router.scale': ( hidden, ),
                'router.per_expert_scale': ( experts, ),
                'experts.gate_up_proj': ( experts, 2 * expert_hidden, hidden ),
                'experts.down_proj': ( experts, hidden, expert_hidden ),
                'layer_scalar': ( 1, ),
            }.get( stem, ( hidden, ) if stem.endswith( 'norm.weight' ) or 'norm_' in stem else None )

        if want is not None and shape != want:
            raise ValueError( f'{name}: checkpoint gives {shape}, the config implies {want}' )


def _report_unconsumed( checkpoint: ShardedCheckpoint, consumed, geometry: dict, max_layers: int ):
    """Account for every checkpoint tensor: consumed, deliberately skipped, or a gap."""
    skipped = { name for name in checkpoint.names() if name.startswith( SKIPPED_PREFIXES ) }

    if geometry[ 'tie_word_embeddings' ]:
        skipped |= { name for name in checkpoint.names() if name == 'lm_head.weight' }

    unconsumed = checkpoint.names() - consumed - skipped

    if max_layers:
        unconsumed = { name for name in unconsumed
                       if not _past_layer_cut( name, geometry[ 'num_hidden_layers' ] ) }

    print( f'\n  Checkpoint tensors: {len( consumed )} consumed, {len( skipped )} skipped' )

    if unconsumed:
        sample = '\n    '.join( sorted( unconsumed )[ :10 ] )
        raise ValueError(
            f'{len( unconsumed )} checkpoint tensors were neither consumed nor skipped:\n    {sample}' )


def _past_layer_cut( name: str, num_layers: int ) -> bool:
    match = re.search( r'layers\.(\d+)\.', name )

    return match is not None and int( match.group( 1 ) ) >= num_layers


def _text_prefix( checkpoint: ShardedCheckpoint ) -> str:
    """'model.' for a text-only checkpoint, 'model.language_model.' for the multimodal packaging."""
    suffix = 'embed_tokens.weight'
    key = next( (k for k in sorted( checkpoint.names() )
                 if k.endswith( suffix ) and not k.startswith( SKIPPED_PREFIXES )), None )

    if key is None:
        raise KeyError( 'embed_tokens.weight not found in the checkpoint' )

    return key[ : -len( suffix ) ]


def convert_gemma( model_name: str, output_path: str, dtype: str = 'bfloat16', max_layers: int = 0 ):

    root = _resolve_checkpoint( model_name )
    config = json.loads( (root / 'config.json').read_text( encoding='utf-8' ) )
    geometry = resolve_gemma_geometry( config, max_layers )

    print( 'Resolved Gemma config:' )
    for k, v in geometry.items():
        print( f'  {k:30s} {v}' )

    checkpoint = ShardedCheckpoint( root )
    prefix = _text_prefix( checkpoint )

    if prefix != 'model.':
        print( f"  Note: multimodal checkpoint detected; key prefix is '{prefix}'" )

    raw_name = model_name.rstrip( '/\\' ).replace( '\\', '/' ).rsplit( '/', 1 )[ -1 ]
    model_id = raw_name.replace( '.', '_' ).replace( '-', '_' )

    tensors = expand_gemma_tensor_map( geometry, prefix )

    writer = MilaStreamingWeightWriter( output_path )
    writer.set_metadata( gemma_mila_metadata( geometry, dtype, model_id ) )

    # ---- Declaration pass: shapes from the shard headers, no tensor data ----
    for tensor in tensors:
        writer.declare( tensor.mila, 'float32' if tensor.transform == 'scalar' else dtype,
            _output_shape( tensor, checkpoint ) )

    _verify_geometry( writer.entries, geometry )

    print( f'\n  {len( writer.entries )} Mila tensors, {writer.total_data_bytes() / 1024**3:.2f} GiB payload' )

    # ---- Data pass: one tensor at a time, source -> transform -> file ----
    consumed = set()
    reported_layer = -1

    with writer:
        for tensor in tensors:
            if tensor.mila.startswith( 'tf_layer_' ):
                layer = int( tensor.mila.split( '.', 1 )[ 0 ].removeprefix( 'tf_layer_' ) )

                if layer != reported_layer:
                    kind = 'global' if ((layer + 1) % geometry[ 'sliding_window_pattern' ]) == 0 else 'local'
                    print( f'  Converting layer {layer}/{geometry[ "num_hidden_layers" ] - 1} ({kind})...' )
                    reported_layer = layer

            consumed.update( tensor.sources )
            writer.write( tensor.mila, _materialize( tensor, checkpoint, dtype ) )

    _report_unconsumed( checkpoint, consumed, geometry, max_layers )

    print( '\nConversion complete!' )
    print( f'  Output: {output_path}' )
    print( f'  Model:  {model_id}  dtype: {dtype}' )


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description='Convert Gemma 4 weights to Mila format' )
    parser.add_argument( '--model', type=str, required=True,
        help=f'HuggingFace model name ({", ".join( SUPPORTED_MODELS )}) or a local checkpoint directory' )
    parser.add_argument( '--output', type=str, required=True,
        help='Output path for the Mila weight file' )
    parser.add_argument( '--dtype', type=str, default='bfloat16',
        choices=[ 'float32', 'bfloat16' ], help='Target dtype (default: bfloat16)' )
    parser.add_argument( '--max-layers', type=int, default=0,
        help='Convert only the first N layers -- a structural smoke test, not a model' )

    args = parser.parse_args()
    convert_gemma( args.model, args.output, args.dtype, args.max_layers )
