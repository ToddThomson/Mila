"""Convert Google's Gemma 4 draft model ("assistant") to Mila format.

The drafter is four decoder layers with query projections only -- they attend the target's own cache -- a
projection in from the target's width, one back out to it, and its own tied head (Specifications/Gemma4Mtp.md
section 2). Its checkpoint is separate from the target's, and so is its Mila file: a deployment that does not
select the drafter never reads it.

Mapping (HF -> Mila):

    pre_projection.weight                         -> drafter.pre_projection.weight      [hidden, 2 x target]
    model.layers.{i}.input_layernorm.weight       -> drafter.layer_{i}.input_norm.weight
    model.layers.{i}.self_attn.q_proj.weight      -> drafter.layer_{i}.q_proj.weight
    model.layers.{i}.self_attn.q_norm.weight      -> drafter.layer_{i}.q_norm.weight
    model.layers.{i}.self_attn.o_proj.weight      -> drafter.layer_{i}.o_proj.weight
    model.layers.{i}.post_attention_layernorm     -> drafter.layer_{i}.post_attn_norm.weight
    model.layers.{i}.pre_feedforward_layernorm    -> drafter.layer_{i}.ffn.pre_norm.weight
    model.layers.{i}.mlp.gate_proj | up_proj      -> drafter.layer_{i}.ffn.mlp.fc_gate_up.weight
    model.layers.{i}.mlp.down_proj.weight         -> drafter.layer_{i}.ffn.mlp.fc_down.weight
    model.layers.{i}.post_feedforward_layernorm   -> drafter.layer_{i}.ffn.post_norm.weight
    model.layers.{i}.layer_scalar                 -> drafter.layer_{i}.layer_scalar (FP32)
    model.norm.weight                             -> drafter.final_norm.weight
    model.embed_tokens.weight (tied head)         -> drafter.head.weight                [vocab, hidden]
    post_projection.weight                        -> drafter.post_projection.weight     [target, hidden]

Norms are written raw, as for the target (convert_weights.py).

Usage:
    python Gemma/convert_drafter.py --model google/gemma-4-12B-it-qat-q4_0-unquantized-assistant \
        --output <weights-dir>/Gemma/gemma4_12b_it_qat_drafter_bf16.bin
"""

import sys
from pathlib import Path
sys.path.insert( 0, str( Path( __file__ ).resolve().parent.parent ) )

import argparse
import json

from common import MilaStreamingWeightWriter, ShardedCheckpoint
from convert_weights import GemmaTensor, _materialize, _output_shape, _value

SUPPORTED_DRAFTERS = [
    'google/gemma-4-12B-it-qat-q4_0-unquantized-assistant',
    'google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant',
]


def _resolve_checkpoint( model_name: str ) -> Path:
    """Local directory holding the checkpoint, downloading it only if absent."""
    candidate = Path( model_name )

    if candidate.is_dir():
        return candidate

    if model_name not in SUPPORTED_DRAFTERS:
        raise ValueError( f"'{model_name}' is neither a checkpoint directory nor one of {SUPPORTED_DRAFTERS}" )

    from huggingface_hub import snapshot_download

    return Path( snapshot_download( model_name, allow_patterns=[ 'config.json', '*.safetensors' ] ) )


def resolve_drafter_geometry( config: dict ) -> dict:
    """The drafter's geometry, with every feature this converter does not model refused."""
    text = config[ 'text_config' ]
    num_layers = text[ 'num_hidden_layers' ]

    if _value( config, 'use_ordered_embeddings', False ):
        raise ValueError( 'use_ordered_embeddings is set; the centroid head is not modelled (Gemma4Mtp.md section 7)' )

    if _value( text, 'num_kv_shared_layers', 0 ) != num_layers:
        raise ValueError( 'a drafter layer projects its own keys and values; every layer must read the target cache' )

    if not _value( config, 'tie_word_embeddings', True ):
        raise ValueError( 'the drafter head is not tied to its embedding table' )

    layer_types = text[ 'layer_types' ]

    # GemmaConfig carries the interleave as a period, so the one period a drafter's layers can be is all sliding
    # layers then one full one.
    if layer_types != [ 'sliding_attention' ] * ( num_layers - 1 ) + [ 'full_attention' ]:
        raise ValueError( f'layer types {layer_types} are not sliding layers ending in one full-attention layer' )

    return {
        'hidden_size': text[ 'hidden_size' ],
        'target_hidden_size': config[ 'backbone_hidden_size' ],
        'num_hidden_layers': num_layers,
        'layer_types': layer_types,
        'num_attention_heads': text[ 'num_attention_heads' ],
        'head_dim': text[ 'head_dim' ],
        'global_head_dim': _value( text, 'global_head_dim', 512 ),
        'intermediate_size': text[ 'intermediate_size' ],
        'vocab_size': text[ 'vocab_size' ],
        'rms_norm_eps': _value( text, 'rms_norm_eps', 1e-6 ),
        'sliding_window': _value( text, 'sliding_window', 1024 ),
    }


def drafter_metadata( geometry: dict, model_id: str ) -> dict:
    """The drafter's own geometry under the keys Mila's reader parses; RoPE and the key/value heads are the target's."""
    return {
        'architecture':            'gemma_drafter',
        'model_name':              model_id,
        'dtype':                   'bfloat16',
        'vocab_size':              geometry[ 'vocab_size' ],
        'embedding_dim':           geometry[ 'hidden_size' ],
        'target_embedding_dim':    geometry[ 'target_hidden_size' ],
        'num_layers':              geometry[ 'num_hidden_layers' ],
        'sliding_window_pattern':  geometry[ 'num_hidden_layers' ],
        'num_heads':               geometry[ 'num_attention_heads' ],
        'head_dim':                geometry[ 'head_dim' ],
        'global_head_dim':         geometry[ 'global_head_dim' ],
        'hidden_dim':              geometry[ 'intermediate_size' ],
        'norm_epsilon':            geometry[ 'rms_norm_eps' ],
        'window':                  geometry[ 'sliding_window' ],
        'tie_word_embeddings':     True,
    }


def expand_drafter_tensor_map( geometry: dict ):
    tensors = [ GemmaTensor( 'drafter.pre_projection.weight', ( 'pre_projection.weight', ) ) ]

    for i in range( geometry[ 'num_hidden_layers' ] ):
        hf = f'model.layers.{i}'
        mila = f'drafter.layer_{i}'

        tensors += [
            GemmaTensor( f'{mila}.input_norm.weight', ( f'{hf}.input_layernorm.weight', ), 'norm' ),
            GemmaTensor( f'{mila}.q_proj.weight', ( f'{hf}.self_attn.q_proj.weight', ) ),
            GemmaTensor( f'{mila}.q_norm.weight', ( f'{hf}.self_attn.q_norm.weight', ), 'norm' ),
            GemmaTensor( f'{mila}.o_proj.weight', ( f'{hf}.self_attn.o_proj.weight', ) ),
            GemmaTensor( f'{mila}.post_attn_norm.weight', ( f'{hf}.post_attention_layernorm.weight', ), 'norm' ),
            GemmaTensor( f'{mila}.ffn.pre_norm.weight', ( f'{hf}.pre_feedforward_layernorm.weight', ), 'norm' ),
            GemmaTensor( f'{mila}.ffn.mlp.fc_gate_up.weight', ( f'{hf}.mlp.gate_proj.weight', f'{hf}.mlp.up_proj.weight' ) ),
            GemmaTensor( f'{mila}.ffn.mlp.fc_down.weight', ( f'{hf}.mlp.down_proj.weight', ) ),
            GemmaTensor( f'{mila}.ffn.post_norm.weight', ( f'{hf}.post_feedforward_layernorm.weight', ), 'norm' ),
            GemmaTensor( f'{mila}.layer_scalar', ( f'{hf}.layer_scalar', ), 'scalar' ),
        ]

    tensors += [
        GemmaTensor( 'drafter.final_norm.weight', ( 'model.norm.weight', ), 'norm' ),
        GemmaTensor( 'drafter.head.weight', ( 'model.embed_tokens.weight', ) ),
        GemmaTensor( 'drafter.post_projection.weight', ( 'post_projection.weight', ) ),
    ]

    return tensors


def _expected_shape( name: str, geometry: dict ):
    """Each tensor's shape from the config alone, independently of the checkpoint."""
    hidden = geometry[ 'hidden_size' ]
    heads = geometry[ 'num_attention_heads' ]

    if name == 'drafter.pre_projection.weight':
        return ( hidden, 2 * geometry[ 'target_hidden_size' ] )

    if name == 'drafter.post_projection.weight':
        return ( geometry[ 'target_hidden_size' ], hidden )

    if name == 'drafter.head.weight':
        return ( geometry[ 'vocab_size' ], hidden )

    if name == 'drafter.final_norm.weight':
        return ( hidden, )

    layer = int( name.split( '.' )[ 1 ].removeprefix( 'layer_' ) )
    stem = name.split( '.', 2 )[ 2 ]
    is_global = geometry[ 'layer_types' ][ layer ] == 'full_attention'
    head_dim = geometry[ 'global_head_dim' ] if is_global else geometry[ 'head_dim' ]

    return {
        'q_proj.weight': ( heads * head_dim, hidden ),
        'q_norm.weight': ( head_dim, ),
        'o_proj.weight': ( hidden, heads * head_dim ),
        'ffn.mlp.fc_gate_up.weight': ( 2 * geometry[ 'intermediate_size' ], hidden ),
        'ffn.mlp.fc_down.weight': ( hidden, geometry[ 'intermediate_size' ] ),
        'layer_scalar': ( 1, ),
    }.get( stem, ( hidden, ) )


def convert_drafter( model_name: str, output_path: str ):
    root = _resolve_checkpoint( model_name )
    config = json.loads( (root / 'config.json').read_text( encoding='utf-8' ) )
    geometry = resolve_drafter_geometry( config )

    print( 'Resolved drafter config:' )
    for k, v in geometry.items():
        print( f'  {k:20s} {v}' )

    checkpoint = ShardedCheckpoint( root )
    tensors = expand_drafter_tensor_map( geometry )

    raw_name = model_name.rstrip( '/\\' ).replace( '\\', '/' ).rsplit( '/', 1 )[ -1 ]
    model_id = raw_name.replace( '.', '_' ).replace( '-', '_' )

    writer = MilaStreamingWeightWriter( output_path )
    writer.set_metadata( drafter_metadata( geometry, model_id ) )

    for tensor in tensors:
        shape = _output_shape( tensor, checkpoint )
        want = _expected_shape( tensor.mila, geometry )

        if tuple( shape ) != tuple( want ):
            raise ValueError( f'{tensor.mila}: checkpoint gives {shape}, the config implies {want}' )

        writer.declare( tensor.mila, 'float32' if tensor.transform == 'scalar' else 'bfloat16', shape )

    consumed = set()

    with writer:
        for tensor in tensors:
            consumed.update( tensor.sources )
            writer.write( tensor.mila, _materialize( tensor, checkpoint, 'bfloat16' ) )

    unconsumed = checkpoint.names() - consumed

    if unconsumed:
        raise ValueError( f'{len( unconsumed )} checkpoint tensors were not consumed: {sorted( unconsumed )[ :10 ]}' )

    print( f'\n  {len( tensors )} Mila tensors, every checkpoint tensor consumed' )
    print( f'  Output: {output_path}' )


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description='Convert a Gemma 4 draft model to Mila format' )
    parser.add_argument( '--model', type=str, required=True,
        help=f'HuggingFace drafter name ({", ".join( SUPPORTED_DRAFTERS )}) or a local checkpoint directory' )
    parser.add_argument( '--output', type=str, required=True, help='Output path for the Mila weight file' )

    args = parser.parse_args()
    convert_drafter( args.model, args.output )
