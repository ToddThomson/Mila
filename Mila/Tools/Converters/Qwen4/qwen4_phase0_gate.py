# The Phase 0 gate of Specifications/Qwen4.md section 9, run end to end on the tiny reference.
#
#   1. The capture is deterministic: two runs with the same seed write identical captures and checkpoints --
#      every tensor bit-equal and the same metadata.
#   2. The converter writes every tensor of the tiny checkpoint exactly once and skips nothing but
#      model.visual. and mtp. -- convert_qwen4 refuses otherwise, so a conversion that completes passes. The
#      written index is then read back: names unique, and every tensor equal to the one the reference loads,
#      which is what catches a permuted n-gram table or a mis-stacked expert bank.
#   3. Fed the n-gram shards in lexical order, the converter refuses.
#
# Two checks the spec's gate does not name, because the tiny checkpoint alone would leave them open:
#   - transformers saves the expert bank one tensor per expert, while Flash-Next ships it stacked. The MoE
#     checkpoint is rewritten stacked and converted again; both conversions must be identical.
#   - --flash-next reads Qwen/Qwen3.8-Flash-Next's config.json and weight index (two small files, no
#     weights) and runs the converter's name map over them: every one of its tensors consumed, moved to
#     metadata or skipped, and every source the map names present.
#
# Both variants by default. No GPU; network only with --flash-next.
#
#   python qwen4_phase0_gate.py --work-dir ../../../../Data/Models/Qwen4/phase0_gate --flash-next

import argparse
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert( 0, str( Path( __file__ ).resolve().parent.parent ) )
sys.path.insert( 0, str( Path( __file__ ).resolve().parent ) )

import numpy as np
import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

from common import TensorMapping
from convert_weights import convert_qwen4, ngram_shard_names, transform


SCRIPT = Path( __file__ ).resolve().parent / 'hf_qwen4_tiny_reference.py'

DTYPE_CODES = { 0: np.float32, 2: np.uint16, 3: np.int32 }


def same_content( first: Path, second: Path ) -> bool:
    """Equal tensors and equal metadata. Not bytes: safetensors writes the metadata's keys in no fixed order."""
    from safetensors import safe_open

    with safe_open( str( first ), 'pt' ) as a, safe_open( str( second ), 'pt' ) as b:
        if a.metadata() != b.metadata() or set( a.keys() ) != set( b.keys() ):
            return False

        return all( torch.equal( a.get_tensor( key ), b.get_tensor( key ) ) for key in a.keys() )


def read_mila( path: Path ):
    """The MILA container's metadata and tensors, BF16 widened to FP32."""
    with open( path, 'rb' ) as f:
        magic, version, count = struct.unpack( 'III', f.read( 12 ) )

        if magic != 0x4D494C41:
            raise ValueError( f'{path}: not a MILA file' )

        metadata_size = struct.unpack( 'I', f.read( 4 ) )[ 0 ]
        metadata = json.loads( f.read( metadata_size ) )
        index = []

        for _ in range( count ):
            name_length = struct.unpack( 'I', f.read( 4 ) )[ 0 ]
            name = f.read( name_length ).decode( 'utf-8' )
            dtype, ndim = struct.unpack( 'II', f.read( 8 ) )
            shape = struct.unpack( 'I' * ndim, f.read( 4 * ndim ) )
            offset, nbytes = struct.unpack( 'QQ', f.read( 16 ) )
            index.append( ( name, dtype, shape, offset, nbytes ) )

        tensors = {}

        for name, dtype, shape, offset, nbytes in index:
            f.seek( offset )
            data = np.frombuffer( f.read( nbytes ), dtype=DTYPE_CODES[ dtype ] ).reshape( shape )

            if dtype == 2:
                data = ( data.astype( np.uint32 ) << 16 ).view( np.float32 )

            if name in tensors:
                raise ValueError( f'{path}: {name} written twice' )

            tensors[ name ] = torch.from_numpy( data.copy() )

    return metadata, tensors


def runtime_reference( model: Qwen4ExpForCausalLM, metadata: dict ):
    """What each Mila tensor must hold, built from the loaded model's runtime parameters -- not from the
    checkpoint files the converter read -- so the converter's own name map is not checking itself."""
    state = model.state_dict()
    expected = { 'temb.wte': state[ 'model.embed_tokens.weight' ], 'lm_head.weight': state[ 'lm_head.weight' ] }
    full_attention = { int( i ) for i in metadata[ 'full_attention_layers' ].split( ',' ) }
    ple_layers = { int( i ) for i in metadata[ 'ple_layers' ].split( ',' ) if i }
    heads = metadata[ 'num_heads' ]
    head_dim = metadata[ 'head_dim' ]

    def residual( mila, hf, read_only=False ):
        expected[ f'{mila}.norm.weight' ] = state[ f'{hf}.hc_norm.weight' ]
        expected[ f'{mila}.fc_down.weight' ] = state[ f'{hf}.input_mix_weight_down.weight' ]
        expected[ f'{mila}.fc_up.weight' ] = state[ f'{hf}.input_mix_weight_up.weight' ]

        if not read_only:
            expected[ f'{mila}.fc_inject.weight' ] = state[ f'{hf}.block_inject_weight.weight' ]

    for i in range( metadata[ 'num_layers' ] ):
        mila = f'tf_layer_{i}'
        hf = f'model.layers.{i}'
        residual( f'{mila}.attn_residual', f'{hf}.attn_hyper_connection' )
        residual( f'{mila}.mlp_residual', f'{hf}.mlp_hyper_connection' )

        if i in ple_layers:
            ple = f'{hf}.ple'
            expected[ f'{mila}.ple.ngram_table' ] = state[ f'{ple}.ple_embedding.ngram_embedding.weight' ]
            expected[ f'{mila}.ple.fc_key_proj.weight' ] = state[ f'{ple}.key_proj.weight' ]
            expected[ f'{mila}.ple.fc_value_proj.weight' ] = state[ f'{ple}.value_proj.weight' ]

            for norm in ( 'norm_key', 'norm_query', 'norm_conv' ):
                expected[ f'{mila}.ple.{norm}.weight' ] = state[ f'{ple}.{norm}.weight' ]

            conv = state[ f'{ple}.conv1d.weight' ]
            expected[ f'{mila}.ple.conv.weight' ] = conv.reshape( conv.shape[ 0 ], conv.shape[ -1 ] )

            for key, buffer in ( ( 'multipliers', 'layer_multipliers' ), ( 'head_vocab_sizes', 'ngram_heads_vocab_sizes' ),
                                 ( 'head_offsets', 'ngram_heads_offsets' ) ):
                written = [ int( value ) for value in metadata[ f'ple_layer_{i}_{key}' ].split( ',' ) ]

                if written != state[ f'{ple}.ple_embedding.{buffer}' ].tolist():
                    raise AssertionError( f'ple_layer_{i}_{key}: metadata differs from the reference buffer' )

        if i in full_attention:
            attention = f'{hf}.self_attn'
            query_gate = state[ f'{attention}.q_proj.weight' ].reshape( heads, 2, head_dim, -1 )
            expected[ f'{mila}.fc_qkv_proj.weight' ] = torch.cat( [
                query_gate[ :, 0 ].reshape( heads * head_dim, -1 ),
                query_gate[ :, 1 ].reshape( heads * head_dim, -1 ),
                state[ f'{attention}.k_proj.weight' ], state[ f'{attention}.v_proj.weight' ] ] )
            expected[ f'{mila}.q_norm.weight' ] = state[ f'{attention}.q_norm.weight' ]
            expected[ f'{mila}.k_norm.weight' ] = state[ f'{attention}.k_norm.weight' ]
            expected[ f'{mila}.fc_o_proj.weight' ] = state[ f'{attention}.o_proj.weight' ]
            expected[ f'{mila}.indexer.fc_qk_proj.weight' ] = state[ f'{attention}.indexer.index_qk_proj.weight' ]
            expected[ f'{mila}.indexer.q_norm.weight' ] = state[ f'{attention}.indexer.q_layernorm.weight' ]
            expected[ f'{mila}.indexer.k_norm.weight' ] = state[ f'{attention}.indexer.k_layernorm.weight' ]
        else:
            linear = f'{hf}.linear_attn'
            key_dim = metadata[ 'linear_num_key_heads' ] * metadata[ 'linear_head_dim' ]
            qkv = state[ f'{linear}.in_proj_qkv.weight' ]
            conv = state[ f'{linear}.conv1d.weight' ]
            conv = conv.reshape( conv.shape[ 0 ], conv.shape[ -1 ] )
            expected[ f'{mila}.fc_in_proj_qk.weight' ] = qkv[ : 2 * key_dim ]
            expected[ f'{mila}.fc_in_proj_v.weight' ] = qkv[ 2 * key_dim : ]
            expected[ f'{mila}.conv_qk.weight' ] = conv[ : 2 * key_dim ]
            expected[ f'{mila}.conv_v.weight' ] = conv[ 2 * key_dim : ]

            for projection in ( 'z', 'a', 'b' ):
                expected[ f'{mila}.fc_in_proj_{projection}.weight' ] = state[ f'{linear}.in_proj_{projection}.weight' ]

            expected[ f'{mila}.delta_rule.A_log' ] = state[ f'{linear}.A_log' ]
            expected[ f'{mila}.delta_rule.dt_bias' ] = state[ f'{linear}.dt_bias' ]
            expected[ f'{mila}.norm_gate.weight' ] = state[ f'{linear}.norm.weight' ]
            expected[ f'{mila}.fc_out_proj.weight' ] = state[ f'{linear}.out_proj.weight' ]

        mlp = f'{hf}.mlp'

        if metadata[ 'mixture_of_experts' ]:
            expected[ f'{mila}.ffn.router.proj.weight' ] = state[ f'{mlp}.gate.weight' ]
            expected[ f'{mila}.ffn.experts.gate_up_proj' ] = state[ f'{mlp}.experts.gate_up_proj' ]
            expected[ f'{mila}.ffn.experts.down_proj' ] = state[ f'{mlp}.experts.down_proj' ]
            shared = f'{mlp}.shared_expert'
            expected[ f'{mila}.ffn.shared.fc_gate_up.weight' ] = torch.cat(
                [ state[ f'{shared}.gate_proj.weight' ], state[ f'{shared}.up_proj.weight' ] ] )
            expected[ f'{mila}.ffn.shared.fc_down.weight' ] = state[ f'{shared}.down_proj.weight' ]
            expected[ f'{mila}.ffn.shared_gate.weight' ] = state[ f'{mlp}.shared_expert_gate.weight' ]
        else:
            expected[ f'{mila}.fc_gate_up.weight' ] = torch.cat(
                [ state[ f'{mlp}.gate_proj.weight' ], state[ f'{mlp}.up_proj.weight' ] ] )
            expected[ f'{mila}.fc_down.weight' ] = state[ f'{mlp}.down_proj.weight' ]

    residual( 'final_mixer', 'model.hyper_connection_mixer', read_only=True )

    return expected


def check_conversion( directory: Path, variant: str ):
    checkpoint = directory / 'hf_checkpoint'
    model = Qwen4ExpForCausalLM.from_pretrained( checkpoint, dtype=torch.float32 )

    if variant == 'dense':
        from hf_qwen4_tiny_reference import substitute_dense_feedforward
        from safetensors.torch import load_file

        # from_pretrained builds the reference's MoE layer; the dense variant's weights load after the swap.
        substitute_dense_feedforward( model, model.config )
        state = {}

        for shard in sorted( checkpoint.glob( '*.safetensors' ) ):
            state.update( load_file( str( shard ) ) )

        dense = { name: tensor for name, tensor in state.items() if '.mlp.' in name }
        missing, unexpected = model.load_state_dict( dense, strict=False )

        if unexpected:
            raise AssertionError( f'dense weights the substituted MLP does not have: {unexpected[ :5 ]}' )

    for dtype, tolerance in ( ( 'fp32', 0.0 ), ( 'bf16', None ) ):
        metadata, tensors = read_mila( directory / f'qwen4_tiny_{variant}_{dtype}.bin' )
        expected = runtime_reference( model, metadata )

        if set( tensors ) != set( expected ):
            raise AssertionError( f'{dtype}: written {sorted( set( tensors ) - set( expected ) )[ :5 ]}, '
                                  f'missing {sorted( set( expected ) - set( tensors ) )[ :5 ]}' )

        for name, want in expected.items():
            want = want.to( torch.float32 )

            if dtype == 'bf16':
                want = want.to( torch.bfloat16 ).to( torch.float32 )

            if tuple( tensors[ name ].shape ) != tuple( want.shape ):
                raise AssertionError( f'{dtype} {name}: shape {tuple( tensors[ name ].shape )}, want {tuple( want.shape )}' )

            if not torch.equal( tensors[ name ], want ):
                raise AssertionError( f'{dtype} {name}: values differ from the reference' )

        print( f'  {variant} {dtype}: {len( tensors )} tensors, each equal to the reference model\'s' )


def check_lexical_refusal( directory: Path ):
    checkpoint = directory / 'hf_checkpoint'
    index = json.loads( ( checkpoint / 'model.safetensors.index.json' ).read_text() ) \
        if ( checkpoint / 'model.safetensors.index.json' ).exists() else None

    from common import shard_header

    names = set( index[ 'weight_map' ] ) if index else set( shard_header( checkpoint / 'model.safetensors' ) )
    table = next( name.rsplit( '.shard_', 1 )[ 0 ] for name in names if '.shard_' in name )
    numeric = ngram_shard_names( names, table )
    lexical = tuple( sorted( numeric ) )

    if lexical == numeric:
        raise AssertionError( 'the tiny checkpoint has too few shards for lexical and numeric order to differ' )

    mapping = TensorMapping( 'tf_layer_1.ple.ngram_table', lexical, transform='ngram_shards' )
    sources = [ torch.zeros( 1, 1 ) for _ in lexical ]

    try:
        transform( mapping, sources, 0, 128 )
    except ValueError as error:
        print( f'  lexical shard order refused: {error}' )

        return

    raise AssertionError( 'the converter concatenated n-gram shards in lexical order' )


def check_stacked_passthrough( directory: Path ):
    """A stacked copy of the per-expert MoE checkpoint converts to the same file."""
    import shutil
    from safetensors.torch import load_file, save_file

    source = directory / 'hf_checkpoint'
    stacked = directory / 'hf_checkpoint_stacked'
    stacked.mkdir( exist_ok=True )
    shutil.copy( source / 'config.json', stacked / 'config.json' )

    state = load_file( str( source / 'model.safetensors' ) )
    per_expert = {}
    rewritten = {}

    for name, tensor in state.items():
        match = re.match( r'(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$', name )

        if match is None:
            rewritten[ name ] = tensor
            continue

        per_expert.setdefault( match.group( 1 ), {} ).setdefault( int( match.group( 2 ) ), {} )[ match.group( 3 ) ] = tensor

    if not per_expert:
        raise AssertionError( 'the MoE checkpoint has no per-expert tensors; the stacking path went untested' )

    for bank, experts in per_expert.items():
        order = range( len( experts ) )
        rewritten[ f'{bank}.gate_up_proj' ] = torch.stack(
            [ torch.cat( [ experts[ e ][ 'gate_proj' ], experts[ e ][ 'up_proj' ] ] ) for e in order ] )
        rewritten[ f'{bank}.down_proj' ] = torch.stack( [ experts[ e ][ 'down_proj' ] for e in order ] )

    save_file( rewritten, str( stacked / 'model.safetensors' ), metadata={ 'format': 'pt' } )

    output = directory / 'qwen4_tiny_moe_stacked_fp32.bin'
    convert_qwen4( stacked.as_posix(), str( output ), 'float32' )

    _, from_stacked = read_mila( output )
    _, from_per_expert = read_mila( directory / 'qwen4_tiny_moe_fp32.bin' )

    if set( from_stacked ) != set( from_per_expert ):
        raise AssertionError( 'stacked and per-expert conversions write different tensor sets' )

    for name, tensor in from_per_expert.items():
        if not torch.equal( tensor, from_stacked[ name ] ):
            raise AssertionError( f'{name}: stacked and per-expert conversions differ' )

    print( f'  stacked expert bank: converts to the same {len( from_stacked )} tensors as the per-expert one' )


class IndexOnlyCheckpoint:
    """The names of a checkpoint known only by its weight index."""

    def __init__( self, names ):
        self._names = set( names )

    def names( self ):
        return self._names


def check_flash_next_names():
    from huggingface_hub import hf_hub_download

    from convert_weights import NGRAM_CONSTANTS, SKIPPED_PREFIXES, expand_qwen4_tensor_map, resolve_qwen4_geometry

    repository = 'Qwen/Qwen3.8-Flash-Next'
    config = json.loads( Path( hf_hub_download( repository, 'config.json' ) ).read_text() )
    index = json.loads( Path( hf_hub_download( repository, 'model.safetensors.index.json' ) ).read_text() )
    names = set( index[ 'weight_map' ] )

    prefix = 'model.language_model.'
    checkpoint = IndexOnlyCheckpoint( names )
    geometry = resolve_qwen4_geometry( config, checkpoint, prefix )
    mappings = expand_qwen4_tensor_map( geometry, prefix, names )

    sources = [ source for mapping in mappings for source in mapping.sources ]
    missing = [ source for source in sources if source not in names ]

    if missing:
        raise AssertionError( f'{len( missing )} mapped sources absent from Flash-Next: {missing[ :5 ]}' )

    consumed = set( sources )

    for layer_id in geometry[ 'ple_layer_ids' ]:
        source = f'{prefix}layers.{layer_id - 1}.ple.ple_embedding'
        consumed.update( f'{source}.{buffer}' for buffer in NGRAM_CONSTANTS.values() )

    skipped = { name for name in names if name.startswith( SKIPPED_PREFIXES ) }
    unaccounted = names - consumed - skipped

    if unaccounted:
        raise AssertionError( f'{len( unaccounted )} Flash-Next tensors unaccounted for: {sorted( unaccounted )[ :5 ]}' )

    tables = sum( 1 for mapping in mappings if mapping.transform == 'ngram_shards' )
    print( f'  Flash-Next: {len( names )} tensors -- {len( consumed )} consumed or moved to metadata, '
           f'{len( skipped )} skipped -- into {len( mappings )} Mila tensors, {tables} n-gram table of '
           f'{geometry[ "split_ngram_parts" ]} shards' )


def main():
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( '--work-dir', type=Path, required=True )
    parser.add_argument( '--variants', nargs='+', choices=[ 'moe', 'dense' ], default=[ 'moe', 'dense' ] )
    parser.add_argument( '--flash-next', action='store_true',
        help="Also run the name map over Qwen/Qwen3.8-Flash-Next's config and weight index (network)" )
    arguments = parser.parse_args()

    work = arguments.work_dir.resolve()

    for variant in arguments.variants:
        first = work / f'{variant}_a'
        second = work / f'{variant}_b'

        for directory, extra in ( ( first, [] ), ( second, [ '--skip-conversion' ] ) ):
            command = [ sys.executable, str( SCRIPT ), '--variant', variant, '--output-dir', str( directory ) ] + extra
            subprocess.run( command, check=True )

        print( f'\nPhase 0 gate, {variant}:' )

        compared = [ f'qwen4_tiny_{variant}_reference.safetensors' ]
        compared += sorted( str( path.relative_to( first ) ) for path in ( first / 'hf_checkpoint' ).glob( '*.safetensors' ) )

        for relative in compared:
            if not same_content( first / relative, second / relative ):
                raise AssertionError( f'{relative} differs between two runs with the same seed' )

        print( f'  deterministic: {len( compared )} files identical in content across two runs' )

        check_conversion( first, variant )
        check_lexical_refusal( first )

        if variant == 'moe':
            check_stacked_passthrough( first )

    if arguments.flash_next:
        print( '\nPhase 0 gate, Flash-Next names:' )
        check_flash_next_names()

    print( '\nPhase 0 gate passed.' )


if __name__ == '__main__':
    main()
