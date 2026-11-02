#!/usr/bin/env python3
# ============================================================================
# File: convert_weights.py
# Convert Qwen 4 (qwen4_exp) weights to Mila format
# ============================================================================

"""
Convert Qwen 4 weights from HuggingFace to Mila binary format.

A skeleton (Specifications/Qwen4.md section 9, Phase 0). No Qwen 4 27B checkpoint
exists yet, so the names are the architecture preview's, Qwen/Qwen3.8-Flash-Next,
which are also what transformers' qwen4_exp saves; it is exercised on the tiny
reference (hf_qwen4_tiny_reference.py). The Mila-side names are proposals: no Mila
component reads them until Phase 4.

Carried over from the Qwen 3.8 converter unchanged, through common.apply_transform:
the per-head de-interleave of the gated q_proj, the in_proj_qkv split into [q|k] and
[v], and the conv1d split on the same boundary. Norm weights are written raw, as for
3.8: the unit offset is applied at the kernel.

New for Qwen 4:

  1. The gated residual replaces input_layernorm and post_attention_layernorm. Each
     layer has two (attn_hyper_connection, mlp_hyper_connection), and the model has a
     read-only one in place of the final norm (hyper_connection_mixer).

  2. The n-gram table ships as split_ngram_parts shards per PLE layer. They are
     concatenated in NUMERIC order, shard_0 .. shard_{N-1}. A safetensors index lists
     them lexically (shard_1, shard_10, shard_100, ...), and a lexical concatenation
     is a silent permutation of the table: the model loads and is quietly worse. The
     concatenation refuses any other order.

  3. The n-gram hash constants (layer_multipliers, ngram_heads_vocab_sizes,
     ngram_heads_offsets) are int64 buffers in the checkpoint. They move into the
     metadata as decimal strings, so the weights path never carries an int64 tensor,
     and a multiplier above 2^53 survives a JSON reader that parses numbers as double.

  4. The expert bank is written stacked, [E, 2I, H] with gate first and [E, H, I]
     (MixtureOfExperts.md section 2). A checkpoint that ships it stacked passes
     through; one that ships per-expert tensors is stacked here.

  5. A dense feed-forward is accepted beside the MoE one, detected from the tensor
     names, since whether the 27B is dense is open (Qwen4.md section 10, question 1).

Usage:
    python Qwen4/convert_weights.py --model <checkpoint-dir> --output <weights-dir>/qwen4/model_bf16.bin
"""

import sys
from pathlib import Path
sys.path.insert( 0, str( Path( __file__ ).parent.parent ) )

import argparse
import importlib.util
import json
import math
import re

import torch

from common import MilaStreamingWeightWriter, ShardedCheckpoint, TensorMapping, apply_transform


def _load_qwen38_converter():
    """The Qwen 3.8 converter, by path: both directories name their module convert_weights."""
    path = Path( __file__ ).parent.parent / 'Qwen' / 'convert_weights.py'
    spec = importlib.util.spec_from_file_location( 'qwen38_convert_weights', path )
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )

    return module


_qwen38 = _load_qwen38_converter()

SKIPPED_PREFIXES = _qwen38.SKIPPED_PREFIXES

NGRAM_SHARD_PATTERN = re.compile( r'\.shard_(\d+)\.weight$' )

# Transforms this converter adds to common.apply_transform's.
LOCAL_TRANSFORMS = ( 'ngram_shards', 'stack_gate_up', 'stack', 'squeeze_conv' )


# ============================================================================
# Geometry
# ============================================================================

def resolve_qwen4_geometry( config: dict, checkpoint: ShardedCheckpoint, prefix: str,
                            max_layers: int = 0 ) -> dict:
    """Every geometry field Mila needs, read and validated from a qwen4_exp config.json."""
    text = config.get( 'text_config', config )

    num_layers = text[ 'num_hidden_layers' ]
    hidden_size = text[ 'hidden_size' ]
    num_heads = text[ 'num_attention_heads' ]
    head_dim = text.get( 'head_dim', hidden_size // num_heads )

    rope_parameters = text.get( 'rope_parameters', {} ) or {}
    rope_theta = float( text.get( 'rope_theta', rope_parameters.get( 'rope_theta', 1e7 ) ) )
    partial_rotary = float( text.get( 'partial_rotary_factor',
        rope_parameters.get( 'partial_rotary_factor', 0.25 ) ) )

    layer_types = text.get( 'layer_types' )

    if layer_types is None:
        interval = text.get( 'full_attention_interval', 4 )
        layer_types = [ 'linear_attention' if ( i + 1 ) % interval else 'full_attention'
                        for i in range( num_layers ) ]

    # The published config says full_attention where the reference builds qwen_sparse_attention.
    full_attention = [ kind != 'linear_attention' for kind in layer_types ]

    key_head_dim = text[ 'linear_key_head_dim' ]
    value_head_dim = text[ 'linear_value_head_dim' ]

    if key_head_dim != value_head_dim:
        raise ValueError(
            f'linear_key_head_dim ({key_head_dim}) != linear_value_head_dim ({value_head_dim}); '
            'the DeltaNet chassis carries a single linear_head_dim' )

    output_gate_type = text.get( 'output_gate_type' ) or text.get( 'hidden_act', 'silu' )

    if output_gate_type not in ( 'sigmoid', 'silu' ):
        raise ValueError( f"output_gate_type '{output_gate_type}' is neither sigmoid nor silu" )

    eos_token_id = text.get( 'eos_token_id', config.get( 'eos_token_id' ) )

    if isinstance( eos_token_id, list ):
        eos_token_id = eos_token_id[ 0 ]

    ple_layer_ids = sorted( set( text.get( 'ple_layer_ids' ) or [] ) )

    for layer_id in ple_layer_ids:
        if full_attention[ layer_id - 1 ]:
            raise ValueError( f'PLE on 1-based layer {layer_id}, a full-attention layer; the reference refuses it' )

    if ple_layer_ids and eos_token_id is None:
        raise ValueError( 'PLE needs eos_token_id: the n-gram history pads and resets with it' )

    indexer_fields = ( 'indexer_n_heads', 'indexer_kv_heads', 'indexer_head_dim',
                       'indexer_budget', 'indexer_compress_ratio' )
    indexer = { field: text.get( field ) for field in indexer_fields }

    if any( value is None for value in indexer.values() ):
        raise ValueError( f'QSA fields missing: {indexer}. A qwen4_exp model without an indexer is not this chassis' )

    if indexer[ 'indexer_kv_heads' ] != 1:
        raise ValueError( 'the QSA indexer has one key head in qwen4_exp' )

    ple_embed_dim = text.get( 'ple_embed_dim' ) or hidden_size
    ngram_heads = ( text.get( 'ngram_size', 3 ) - 1 ) * text.get( 'heads_per_ngram', 8 )

    if ple_embed_dim % ngram_heads != 0:
        raise ValueError( f'ple_embed_dim {ple_embed_dim} is not divisible by {ngram_heads} n-gram heads' )

    # The checkpoint, not the config, says which feed-forward a layer has: qwen4_exp's config always
    # carries num_experts, whether or not a dense variant uses it.
    names = checkpoint.names()
    mixture_of_experts = f'{prefix}layers.0.mlp.gate.weight' in names
    dense = f'{prefix}layers.0.mlp.gate_proj.weight' in names

    if mixture_of_experts == dense:
        raise ValueError( 'layer 0 has neither one MoE router nor one dense gate_proj; unknown feed-forward' )

    if max_layers:
        num_layers = min( num_layers, max_layers )
        full_attention = full_attention[ : num_layers ]
        ple_layer_ids = [ layer_id for layer_id in ple_layer_ids if layer_id <= num_layers ]

    geometry = {
        'hidden_size': hidden_size,
        'num_hidden_layers': num_layers,
        'num_attention_heads': num_heads,
        'num_key_value_heads': text[ 'num_key_value_heads' ],
        'head_dim': head_dim,
        'vocab_size': text[ 'vocab_size' ],
        'max_position_embeddings': text[ 'max_position_embeddings' ],
        'rms_norm_eps': text.get( 'rms_norm_eps', 1e-6 ),
        'rope_theta': rope_theta,
        'partial_rotary_factor': partial_rotary,
        'tie_word_embeddings': bool( text.get( 'tie_word_embeddings', False ) ),
        'full_attention': full_attention,
        'linear_num_key_heads': text[ 'linear_num_key_heads' ],
        'linear_num_value_heads': text[ 'linear_num_value_heads' ],
        'linear_head_dim': key_head_dim,
        'linear_conv_kernel_dim': text[ 'linear_conv_kernel_dim' ],
        'output_gate_type': output_gate_type,
        'hc_count': text[ 'hc_count' ],
        'hc_lowrank': text[ 'hc_lowrank' ],
        'ple_layer_ids': ple_layer_ids,
        'ple_embed_dim': ple_embed_dim,
        'ple_conv_kernel_size': text.get( 'ple_conv_kernel_size', 4 ),
        'ngram_size': text.get( 'ngram_size', 3 ),
        'heads_per_ngram': text.get( 'heads_per_ngram', 8 ),
        'ngram_heads': ngram_heads,
        'make_ngram_vocab_size_divisible_by': text.get( 'make_ngram_vocab_size_divisible_by', 128 ),
        'split_ngram_parts': text.get( 'split_ngram_parts', 512 ),
        'eos_token_id': eos_token_id,
        'mixture_of_experts': mixture_of_experts,
    }

    geometry.update( indexer )

    if mixture_of_experts:
        geometry.update( {
            'num_experts': text[ 'num_experts' ],
            'num_experts_per_tok': text[ 'num_experts_per_tok' ],
            'moe_intermediate_size': text[ 'moe_intermediate_size' ],
            'shared_expert_intermediate_size': text[ 'shared_expert_intermediate_size' ],
            'norm_topk_prob': bool( text.get( 'norm_topk_prob', True ) ),
        } )
    else:
        down = checkpoint.shape( f'{prefix}layers.0.mlp.down_proj.weight' )
        geometry[ 'intermediate_size' ] = down[ 1 ]

    return geometry


# ============================================================================
# HuggingFace -> Mila tensor map
# ============================================================================

def _gated_residual( mila: str, hf: str, read_only: bool = False ):
    mappings = [
        TensorMapping( f'{mila}.norm.weight', ( f'{hf}.hc_norm.weight', ) ),
        TensorMapping( f'{mila}.fc_down.weight', ( f'{hf}.input_mix_weight_down.weight', ), is_linear=True ),
        TensorMapping( f'{mila}.fc_up.weight', ( f'{hf}.input_mix_weight_up.weight', ), is_linear=True ),
    ]

    if not read_only:
        mappings.append(
            TensorMapping( f'{mila}.fc_inject.weight', ( f'{hf}.block_inject_weight.weight', ), is_linear=True ) )

    return mappings


def _attention( mila: str, hf: str ):
    return [
        TensorMapping( f'{mila}.fc_qkv_proj.weight',
            ( f'{hf}.self_attn.q_proj.weight', f'{hf}.self_attn.k_proj.weight', f'{hf}.self_attn.v_proj.weight' ),
            is_linear=True, transform='gated_qkv' ),
        TensorMapping( f'{mila}.q_norm.weight', ( f'{hf}.self_attn.q_norm.weight', ) ),
        TensorMapping( f'{mila}.k_norm.weight', ( f'{hf}.self_attn.k_norm.weight', ) ),
        TensorMapping( f'{mila}.fc_o_proj.weight', ( f'{hf}.self_attn.o_proj.weight', ), is_linear=True ),
        TensorMapping( f'{mila}.indexer.fc_qk_proj.weight', ( f'{hf}.self_attn.indexer.index_qk_proj.weight', ),
            is_linear=True ),
        TensorMapping( f'{mila}.indexer.q_norm.weight', ( f'{hf}.self_attn.indexer.q_layernorm.weight', ) ),
        TensorMapping( f'{mila}.indexer.k_norm.weight', ( f'{hf}.self_attn.indexer.k_layernorm.weight', ) ),
    ]


def _deltanet( mila: str, hf: str, key_dim: int, value_dim: int ):
    qk_rows = ( 0, 2 * key_dim )
    v_rows = ( 2 * key_dim, 2 * key_dim + value_dim )
    source = f'{hf}.linear_attn'

    return [
        TensorMapping( f'{mila}.fc_in_proj_qk.weight', ( f'{source}.in_proj_qkv.weight', ),
            is_linear=True, transform='rows', rows=qk_rows ),
        TensorMapping( f'{mila}.fc_in_proj_v.weight', ( f'{source}.in_proj_qkv.weight', ),
            is_linear=True, transform='rows', rows=v_rows ),
        TensorMapping( f'{mila}.fc_in_proj_z.weight', ( f'{source}.in_proj_z.weight', ), is_linear=True ),
        TensorMapping( f'{mila}.fc_in_proj_a.weight', ( f'{source}.in_proj_a.weight', ), is_linear=True ),
        TensorMapping( f'{mila}.fc_in_proj_b.weight', ( f'{source}.in_proj_b.weight', ), is_linear=True ),
        TensorMapping( f'{mila}.conv_qk.weight', ( f'{source}.conv1d.weight', ),
            transform='conv_rows', rows=qk_rows ),
        TensorMapping( f'{mila}.conv_v.weight', ( f'{source}.conv1d.weight', ),
            transform='conv_rows', rows=v_rows ),
        TensorMapping( f'{mila}.delta_rule.A_log', ( f'{source}.A_log', ) ),
        TensorMapping( f'{mila}.delta_rule.dt_bias', ( f'{source}.dt_bias', ) ),
        TensorMapping( f'{mila}.norm_gate.weight', ( f'{source}.norm.weight', ) ),
        TensorMapping( f'{mila}.fc_out_proj.weight', ( f'{source}.out_proj.weight', ), is_linear=True ),
    ]


def ngram_shard_names( names, table: str ):
    """The table's shards in numeric order, each index from 0 to N - 1 present exactly once."""
    shards = {}

    for name in names:
        if not name.startswith( f'{table}.shard_' ):
            continue

        match = NGRAM_SHARD_PATTERN.search( name )

        if match is None:
            raise ValueError( f'unrecognized n-gram shard name {name}' )

        shards[ int( match.group( 1 ) ) ] = name

    if sorted( shards ) != list( range( len( shards ) ) ):
        raise ValueError( f'{table}: shard indices are not 0 .. {len( shards ) - 1}: {sorted( shards )}' )

    return tuple( shards[ index ] for index in range( len( shards ) ) )


def _per_layer_embedding( mila: str, hf: str, names ):
    source = f'{hf}.ple'

    return [
        TensorMapping( f'{mila}.ngram_table',
            ngram_shard_names( names, f'{source}.ple_embedding.ngram_embedding' ), transform='ngram_shards' ),
        TensorMapping( f'{mila}.fc_key_proj.weight', ( f'{source}.key_proj.weight', ), is_linear=True ),
        TensorMapping( f'{mila}.fc_value_proj.weight', ( f'{source}.value_proj.weight', ), is_linear=True ),
        TensorMapping( f'{mila}.norm_key.weight', ( f'{source}.norm_key.weight', ) ),
        TensorMapping( f'{mila}.norm_query.weight', ( f'{source}.norm_query.weight', ) ),
        TensorMapping( f'{mila}.norm_conv.weight', ( f'{source}.norm_conv.weight', ) ),
        TensorMapping( f'{mila}.conv.weight', ( f'{source}.conv1d.weight', ), transform='squeeze_conv' ),
    ]


def _feed_forward( mila: str, hf: str, names, num_experts: int ):
    source = f'{hf}.mlp'

    if f'{source}.gate_proj.weight' in names:
        return [
            TensorMapping( f'{mila}.fc_gate_up.weight',
                ( f'{source}.gate_proj.weight', f'{source}.up_proj.weight' ), is_linear=True ),
            TensorMapping( f'{mila}.fc_down.weight', ( f'{source}.down_proj.weight', ), is_linear=True ),
        ]

    if f'{source}.experts.gate_up_proj' in names:
        gate_up = TensorMapping( f'{mila}.ffn.experts.gate_up_proj', ( f'{source}.experts.gate_up_proj', ) )
        down = TensorMapping( f'{mila}.ffn.experts.down_proj', ( f'{source}.experts.down_proj', ) )
    else:
        # Per-expert tensors, interleaved [gate_0, up_0, gate_1, up_1, ...] for the stacking transform.
        gate_up_sources = []

        for expert in range( num_experts ):
            gate_up_sources.append( f'{source}.experts.{expert}.gate_proj.weight' )
            gate_up_sources.append( f'{source}.experts.{expert}.up_proj.weight' )

        down_sources = tuple( f'{source}.experts.{expert}.down_proj.weight' for expert in range( num_experts ) )
        gate_up = TensorMapping( f'{mila}.ffn.experts.gate_up_proj', tuple( gate_up_sources ), transform='stack_gate_up' )
        down = TensorMapping( f'{mila}.ffn.experts.down_proj', down_sources, transform='stack' )

    return [
        TensorMapping( f'{mila}.ffn.router.proj.weight', ( f'{source}.gate.weight', ) ),
        gate_up,
        down,
        TensorMapping( f'{mila}.ffn.shared.fc_gate_up.weight',
            ( f'{source}.shared_expert.gate_proj.weight', f'{source}.shared_expert.up_proj.weight' ),
            is_linear=True ),
        TensorMapping( f'{mila}.ffn.shared.fc_down.weight', ( f'{source}.shared_expert.down_proj.weight', ),
            is_linear=True ),
        TensorMapping( f'{mila}.ffn.shared_gate.weight', ( f'{source}.shared_expert_gate.weight', ) ),
    ]


def expand_qwen4_tensor_map( geometry: dict, prefix: str, names ):
    """The full HF -> Mila map for a Qwen 4 model, every index and row split resolved."""
    key_dim = geometry[ 'linear_num_key_heads' ] * geometry[ 'linear_head_dim' ]
    value_dim = geometry[ 'linear_num_value_heads' ] * geometry[ 'linear_head_dim' ]
    num_experts = geometry.get( 'num_experts', 0 )

    mappings = [ TensorMapping( 'temb.wte', ( f'{prefix}embed_tokens.weight', ) ) ]

    for i in range( geometry[ 'num_hidden_layers' ] ):
        mila = f'tf_layer_{i}'
        hf = f'{prefix}layers.{i}'

        if ( i + 1 ) in geometry[ 'ple_layer_ids' ]:
            mappings.extend( _per_layer_embedding( f'{mila}.ple', hf, names ) )

        mappings.extend( _gated_residual( f'{mila}.attn_residual', f'{hf}.attn_hyper_connection' ) )

        if geometry[ 'full_attention' ][ i ]:
            mappings.extend( _attention( mila, hf ) )
        else:
            mappings.extend( _deltanet( mila, hf, key_dim, value_dim ) )

        mappings.extend( _gated_residual( f'{mila}.mlp_residual', f'{hf}.mlp_hyper_connection' ) )
        mappings.extend( _feed_forward( mila, hf, names, num_experts ) )

    mappings.extend( _gated_residual( 'final_mixer', f'{prefix}hyper_connection_mixer', read_only=True ) )
    mappings.append( TensorMapping( 'lm_head.weight', ( 'lm_head.weight', ), is_linear=True ) )

    return mappings


# The int64 n-gram buffers each PLE layer carries, moved into the metadata rather than written as tensors.
NGRAM_CONSTANTS = {
    'multipliers': 'layer_multipliers',
    'head_vocab_sizes': 'ngram_heads_vocab_sizes',
    'head_offsets': 'ngram_heads_offsets',
}


def _ngram_constants( checkpoint: ShardedCheckpoint, prefix: str, geometry: dict ):
    """Per PLE layer, the hash constants as int lists, checked against the table they index."""
    constants = {}

    for layer_id in geometry[ 'ple_layer_ids' ]:
        source = f'{prefix}layers.{layer_id - 1}.ple.ple_embedding'
        values = {}

        for key, buffer in NGRAM_CONSTANTS.items():
            tensor = checkpoint.tensor( f'{source}.{buffer}' )

            if tensor.dtype != torch.int64:
                raise ValueError( f'{source}.{buffer} is {tensor.dtype}; the hash needs int64' )

            values[ key ] = [ int( value ) for value in tensor.tolist() ]

        if len( values[ 'multipliers' ] ) != geometry[ 'ngram_size' ]:
            raise ValueError( f'{source}: {len( values[ "multipliers" ] )} multipliers for n-gram size '
                              f'{geometry[ "ngram_size" ]}' )

        if len( values[ 'head_vocab_sizes' ] ) != geometry[ 'ngram_heads' ] \
                or len( values[ 'head_offsets' ] ) != geometry[ 'ngram_heads' ]:
            raise ValueError( f'{source}: head constants do not have {geometry[ "ngram_heads" ]} entries' )

        running = 0

        for offset, size in zip( values[ 'head_offsets' ], values[ 'head_vocab_sizes' ] ):
            if offset != running:
                raise ValueError( f'{source}: head offsets are not the running sum of head vocabulary sizes' )

            running += size

        values[ 'total_rows' ] = running
        constants[ layer_id - 1 ] = values

    return constants


def qwen4_mila_metadata( geometry: dict, constants: dict, dtype: str, model_id: str ) -> dict:
    """The metadata block Mila's reader parses. No key may be a prefix of another up to its closing quote."""
    full_attention = geometry[ 'full_attention' ]

    metadata = {
        'architecture': 'qwen4',
        'model_name': model_id,
        'dtype': dtype,
        'vocab_size': geometry[ 'vocab_size' ],
        'max_seq_length': geometry[ 'max_position_embeddings' ],
        'embedding_dim': geometry[ 'hidden_size' ],
        'num_layers': geometry[ 'num_hidden_layers' ],
        'num_heads': geometry[ 'num_attention_heads' ],
        'num_kv_heads': geometry[ 'num_key_value_heads' ],
        'head_dim': geometry[ 'head_dim' ],
        'use_bias': False,
        'tie_word_embeddings': geometry[ 'tie_word_embeddings' ],
        'activation': 'silu',
        'norm_type': 'rmsnorm',
        'attention_type': 'gqa',
        'positional_encoding': 'rope',
        'rope_theta': geometry[ 'rope_theta' ],
        'norm_epsilon': geometry[ 'rms_norm_eps' ],
        'attention_output_gate': True,
        'partial_rotary_factor': geometry[ 'partial_rotary_factor' ],
        'full_attention_layers': ','.join( str( i ) for i, full in enumerate( full_attention ) if full ),
        'linear_num_key_heads': geometry[ 'linear_num_key_heads' ],
        'linear_num_value_heads': geometry[ 'linear_num_value_heads' ],
        'linear_head_dim': geometry[ 'linear_head_dim' ],
        'linear_conv_kernel_dim': geometry[ 'linear_conv_kernel_dim' ],
        'linear_output_gate': geometry[ 'output_gate_type' ],
        'hc_count': geometry[ 'hc_count' ],
        'hc_lowrank': geometry[ 'hc_lowrank' ],
        'indexer_num_heads': geometry[ 'indexer_n_heads' ],
        'indexer_head_dim': geometry[ 'indexer_head_dim' ],
        'indexer_budget': geometry[ 'indexer_budget' ],
        'indexer_compress_ratio': geometry[ 'indexer_compress_ratio' ],
        'eos_token_id': geometry[ 'eos_token_id' ],
        'ple_layers': ','.join( str( layer_id - 1 ) for layer_id in geometry[ 'ple_layer_ids' ] ),
        'ple_embed_dim': geometry[ 'ple_embed_dim' ],
        'ple_conv_kernel_size': geometry[ 'ple_conv_kernel_size' ],
        'ngram_size': geometry[ 'ngram_size' ],
        'ngram_heads_per_order': geometry[ 'heads_per_ngram' ],
        'mixture_of_experts': geometry[ 'mixture_of_experts' ],
    }

    if geometry[ 'mixture_of_experts' ]:
        metadata[ 'num_experts' ] = geometry[ 'num_experts' ]
        metadata[ 'top_k_experts' ] = geometry[ 'num_experts_per_tok' ]
        metadata[ 'expert_hidden_dim' ] = geometry[ 'moe_intermediate_size' ]
        metadata[ 'shared_expert_hidden_dim' ] = geometry[ 'shared_expert_intermediate_size' ]
        metadata[ 'normalize_top_k' ] = geometry[ 'norm_topk_prob' ]
    else:
        metadata[ 'hidden_dim' ] = geometry[ 'intermediate_size' ]

    # Decimal strings: a multiplier can exceed 2^53, which a reader parsing JSON numbers as double rounds.
    for layer, values in constants.items():
        for key in NGRAM_CONSTANTS:
            metadata[ f'ple_layer_{layer}_{key}' ] = ','.join( str( value ) for value in values[ key ] )

    return metadata


# ============================================================================
# Transforms
# ============================================================================

def transform( mapping: TensorMapping, sources, head_dim: int, divisor: int ):
    if mapping.transform not in LOCAL_TRANSFORMS:
        return apply_transform( mapping, sources, head_dim )

    if mapping.transform == 'ngram_shards':
        # Refused here, not only where the names are gathered, so no caller can concatenate out of order.
        if mapping.sources != ngram_shard_names( mapping.sources, mapping.sources[ 0 ].rsplit( '.shard_', 1 )[ 0 ] ):
            raise ValueError( f'{mapping.mila}: n-gram shards are not in numeric order' )

        table = torch.cat( sources, dim=0 )
        padded = math.ceil( table.shape[ 0 ] / divisor ) * divisor

        if padded != table.shape[ 0 ]:
            table = torch.nn.functional.pad( table, ( 0, 0, 0, padded - table.shape[ 0 ] ) )

        return table

    if mapping.transform == 'stack_gate_up':
        pairs = [ torch.cat( [ sources[ k ], sources[ k + 1 ] ], dim=0 ) for k in range( 0, len( sources ), 2 ) ]

        return torch.stack( pairs, dim=0 )

    if mapping.transform == 'stack':
        return torch.stack( list( sources ), dim=0 )

    # squeeze_conv: [channels, 1, kernel] -> [channels, kernel], torch's depthwise input-group axis dropped.
    weight = sources[ 0 ]

    return weight.reshape( weight.shape[ 0 ], weight.shape[ -1 ] )


def output_shape( mapping: TensorMapping, checkpoint: ShardedCheckpoint, divisor: int ):
    shapes = [ checkpoint.shape( source ) for source in mapping.sources ]

    if mapping.transform == 'ngram_shards':
        rows = sum( shape[ 0 ] for shape in shapes )

        return ( math.ceil( rows / divisor ) * divisor, shapes[ 0 ][ 1 ] )

    if mapping.transform == 'stack_gate_up':
        return ( len( shapes ) // 2, shapes[ 0 ][ 0 ] + shapes[ 1 ][ 0 ], shapes[ 0 ][ 1 ] )

    if mapping.transform == 'stack':
        return ( len( shapes ), ) + tuple( shapes[ 0 ] )

    if mapping.transform == 'squeeze_conv':
        return ( shapes[ 0 ][ 0 ], shapes[ 0 ][ -1 ] )

    return _qwen38._output_shape( mapping, checkpoint, 0 )


def _verify_geometry( entries, geometry: dict, constants: dict ):
    """Every declared shape against what the config implies, derived independently of the checkpoint."""
    H = geometry[ 'hidden_size' ]
    n = geometry[ 'hc_count' ]
    rank = geometry[ 'hc_lowrank' ]
    head_dim = geometry[ 'head_dim' ]
    q_width = geometry[ 'num_attention_heads' ] * head_dim
    kv_width = geometry[ 'num_key_value_heads' ] * head_dim
    linear_head_dim = geometry[ 'linear_head_dim' ]
    key_dim = geometry[ 'linear_num_key_heads' ] * linear_head_dim
    value_dim = geometry[ 'linear_num_value_heads' ] * linear_head_dim
    value_heads = geometry[ 'linear_num_value_heads' ]
    conv_kernel = geometry[ 'linear_conv_kernel_dim' ]
    indexer_heads = geometry[ 'indexer_n_heads' ] + geometry[ 'indexer_kv_heads' ]
    indexer_head_dim = geometry[ 'indexer_head_dim' ]
    ple_embed_dim = geometry[ 'ple_embed_dim' ]
    vocab_size = geometry[ 'vocab_size' ]

    expected = {
        'temb.wte': ( vocab_size, H ),
        'lm_head.weight': ( vocab_size, H ),
        'residual.norm.weight': ( n * H, ),
        'residual.fc_down.weight': ( rank, n * H ),
        'residual.fc_up.weight': ( n * H, rank ),
        'residual.fc_inject.weight': ( n, n * H ),
        'fc_qkv_proj.weight': ( 2 * q_width + 2 * kv_width, H ),
        'q_norm.weight': ( head_dim, ),
        'k_norm.weight': ( head_dim, ),
        'fc_o_proj.weight': ( H, q_width ),
        'indexer.fc_qk_proj.weight': ( indexer_heads * indexer_head_dim, H ),
        'indexer.q_norm.weight': ( indexer_head_dim, ),
        'indexer.k_norm.weight': ( indexer_head_dim, ),
        'fc_in_proj_qk.weight': ( 2 * key_dim, H ),
        'fc_in_proj_v.weight': ( value_dim, H ),
        'fc_in_proj_z.weight': ( value_dim, H ),
        'fc_in_proj_a.weight': ( value_heads, H ),
        'fc_in_proj_b.weight': ( value_heads, H ),
        'fc_out_proj.weight': ( H, value_dim ),
        'conv_qk.weight': ( 2 * key_dim, conv_kernel ),
        'conv_v.weight': ( value_dim, conv_kernel ),
        'norm_gate.weight': ( linear_head_dim, ),
        'delta_rule.A_log': ( value_heads, ),
        'delta_rule.dt_bias': ( value_heads, ),
        'ple.fc_key_proj.weight': ( n * H, ple_embed_dim ),
        'ple.fc_value_proj.weight': ( H, ple_embed_dim ),
        'ple.norm_key.weight': ( n * H, ),
        'ple.norm_query.weight': ( n * H, ),
        'ple.norm_conv.weight': ( n * H, ),
        'ple.conv.weight': ( n * H, geometry[ 'ple_conv_kernel_size' ] ),
    }

    if geometry[ 'mixture_of_experts' ]:
        experts = geometry[ 'num_experts' ]
        expert_width = geometry[ 'moe_intermediate_size' ]
        shared_width = geometry[ 'shared_expert_intermediate_size' ]
        expected.update( {
            'ffn.router.proj.weight': ( experts, H ),
            'ffn.experts.gate_up_proj': ( experts, 2 * expert_width, H ),
            'ffn.experts.down_proj': ( experts, H, expert_width ),
            'ffn.shared.fc_gate_up.weight': ( 2 * shared_width, H ),
            'ffn.shared.fc_down.weight': ( H, shared_width ),
            'ffn.shared_gate.weight': ( 1, H ),
        } )
    else:
        width = geometry[ 'intermediate_size' ]
        expected.update( {
            'fc_gate_up.weight': ( 2 * width, H ),
            'fc_down.weight': ( H, width ),
        } )

    divisor = geometry[ 'make_ngram_vocab_size_divisible_by' ]
    head_width = ple_embed_dim // geometry[ 'ngram_heads' ]
    unchecked = []

    for name, _, shape in entries:
        if name.endswith( '.ple.ngram_table' ):
            layer = int( name.split( '.', 1 )[ 0 ].removeprefix( 'tf_layer_' ) )
            want = ( math.ceil( constants[ layer ][ 'total_rows' ] / divisor ) * divisor, head_width )
        else:
            stem = name.split( '.', 1 )[ 1 ] if name.startswith( 'tf_layer_' ) else name

            for marker in ( 'final_mixer.', 'attn_residual.', 'mlp_residual.' ):
                if stem.startswith( marker ):
                    stem = 'residual.' + stem.removeprefix( marker )

            want = expected.get( stem )

        if want is None:
            unchecked.append( name )
            continue

        if shape != want:
            raise ValueError( f'{name}: checkpoint gives {shape}, the config implies {want}' )

    if unchecked:
        raise ValueError( f'no expected shape for {unchecked[ :10 ]}' )

    if len( { name for name, _, _ in entries } ) != len( entries ):
        raise ValueError( 'duplicate tensor names in the declared index' )


# ============================================================================
# Conversion
# ============================================================================

def convert_qwen4( model_name: str, output_path: str, dtype: str = 'bfloat16', max_layers: int = 0 ):
    root = _qwen38._resolve_checkpoint( model_name )
    config = json.loads( ( root / 'config.json' ).read_text( encoding='utf-8' ) )

    checkpoint = ShardedCheckpoint( root )
    prefix = _qwen38._text_prefix( checkpoint )
    names = checkpoint.names()

    geometry = resolve_qwen4_geometry( config, checkpoint, prefix, max_layers )
    constants = _ngram_constants( checkpoint, prefix, geometry )

    shards = geometry[ 'split_ngram_parts' ]

    for layer in constants:
        table = f'{prefix}layers.{layer}.ple.ple_embedding.ngram_embedding'
        found = len( ngram_shard_names( names, table ) )

        if found != shards:
            raise ValueError( f'{table}: {found} shards, split_ngram_parts is {shards}' )

    print( 'Resolved Qwen 4 config:' )

    for key, value in geometry.items():
        print( f'  {key:34s} {value}' )

    mappings = expand_qwen4_tensor_map( geometry, prefix, names )

    raw_name = model_name.rstrip( '/' ).rsplit( '/', 1 )[ -1 ]
    model_id = raw_name.replace( '.', '_' ).replace( '-', '_' )
    divisor = geometry[ 'make_ngram_vocab_size_divisible_by' ]
    head_dim = geometry[ 'head_dim' ]

    writer = MilaStreamingWeightWriter( output_path )
    writer.set_metadata( qwen4_mila_metadata( geometry, constants, dtype, model_id ) )

    for mapping in mappings:
        writer.declare( mapping.mila, dtype, output_shape( mapping, checkpoint, divisor ) )

    _verify_geometry( writer.entries, geometry, constants )

    print( f'\n  {len( writer.entries )} Mila tensors, {writer.total_data_bytes() / 1024**3:.3f} GiB payload' )

    consumed = set()

    with writer:
        for mapping in mappings:
            sources = [ checkpoint.tensor( source ) for source in mapping.sources ]
            consumed.update( mapping.sources )
            tensor = transform( mapping, sources, head_dim, divisor )
            writer.write( mapping.mila, _qwen38._tensor_to_numpy( tensor, dtype ) )

    for layer in constants:
        source = f'{prefix}layers.{layer}.ple.ple_embedding'
        consumed.update( f'{source}.{buffer}' for buffer in NGRAM_CONSTANTS.values() )

    _report_unconsumed( checkpoint, consumed, geometry[ 'num_hidden_layers' ], max_layers )

    print( '\nConversion complete!' )
    print( f'  Output: {output_path}' )
    print( f'  Model:  {model_id}  dtype: {dtype}' )


def _report_unconsumed( checkpoint: ShardedCheckpoint, consumed, num_layers: int, max_layers: int ):
    """Every checkpoint tensor consumed, moved to metadata, or under a skipped prefix; anything else refuses."""
    skipped = { name for name in checkpoint.names() if name.startswith( SKIPPED_PREFIXES ) }
    unconsumed = checkpoint.names() - consumed - skipped

    if max_layers:
        unconsumed = { name for name in unconsumed if not _qwen38._past_layer_cut( name, num_layers ) }

    print( f'\n  Checkpoint tensors: {len( consumed )} consumed, {len( skipped )} skipped (vision tower and MTP head)' )

    if unconsumed:
        sample = '\n    '.join( sorted( unconsumed )[ :10 ] )
        raise ValueError( f'{len( unconsumed )} checkpoint tensors were neither consumed nor skipped:\n    {sample}' )


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description='Convert Qwen 4 (qwen4_exp) weights to Mila format' )
    parser.add_argument( '--model', type=str, required=True,
        help='HuggingFace model name, or a local checkpoint directory' )
    parser.add_argument( '--output', type=str, required=True, help='Output path for the Mila weight file' )
    parser.add_argument( '--dtype', type=str, default='bfloat16', choices=[ 'float32', 'bfloat16' ],
        help='Target dtype (default: bfloat16)' )
    parser.add_argument( '--max-layers', type=int, default=0,
        help='Convert only the first N layers -- a structural smoke test, not a model' )

    args = parser.parse_args()
    convert_qwen4( args.model, args.output, args.dtype, args.max_layers )
