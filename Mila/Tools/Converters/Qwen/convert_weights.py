#!/usr/bin/env python3
# ============================================================================
# File: convert_weights.py
# Convert Qwen 3.8 weights to Mila format
# ============================================================================

"""
Convert Qwen 3.8 weights from HuggingFace to Mila binary format.

Target: Qwen/Qwen3.8-27B, the 64-layer hybrid stack that interleaves three Gated
DeltaNet layers per full-attention layer (`full_attention_interval: 4`). Text only:
the vision tower and the MTP draft head are out of scope for this chassis
(Specifications/Qwen3.8.md section 1) and are skipped at the index.

THIS CONVERTER STREAMS, and that is not a refinement. The checkpoint is 51.8 GiB
against 31.8 GB of host RAM, so `from_pretrained` -- which every other converter here
uses -- cannot run at all. Shards are read through safetensors mmap one tensor at a
time, and the output is written through MilaStreamingWeightWriter, whose index is
declared from the checkpoint's own shard headers before a byte of data moves.

Four transforms have no counterpart in the Llama or Gemma converters. Each is a
contract stated in the consuming block's header, and all four live in the shared name
map (`common.expand_qwen_tensor_map`) so a quantizing packer rearranges identically:

  1. De-interleave `q_proj`. `attn_output_gate` makes the query projection double
     width, and the reference views it as [..., heads, 2 * head_dim] and chunks the
     last axis -- so the checkpoint stores [q_h0 | gate_h0 | q_h1 | gate_h1 | ...].
     QwenAttentionBlock's fused projection is [query | gate | key | value] with query
     and gate as CONTIGUOUS halves, so the query half is de-interleaved before the
     three sources concatenate.

  2. Split `in_proj_qkv` [10240, 5120] into [q|k] at 4096 and [v] at 6144. One tensor
     cannot carry two storage policies, and the precision plan puts DeltaNet q/k a
     half step above v (section 5's counts only add up this way).

  3. Split `conv1d.weight` [10240, 1, 4] on the same boundary, into [4096, 4] and
     [6144, 4]. Exact rather than approximate: the convolution is depthwise, so two
     convolutions over a partition of the channels compute what one over all of them
     computes. The singleton axis is torch's Conv1d input-group axis and is dropped.

  4. Write norm weights RAW. Qwen's stream norms scale by (1 + weight) with weights
     stored zero-centered, and Mila applies the +1 at the kernel
     (RmsNormConfig::withUnitOffset), so stored weights stay identical to the
     checkpoint. Do NOT pre-add 1. `linear_attn.norm` -- the DeltaNet mixer's gated
     norm -- is genuinely raw on both sides and needs no distinction here, but the two
     conventions do live in one model.

A_log and dt_bias load into the GatedDeltaRule component rather than into any
projection: the rule derives the decay and the forget gate from them
(g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b)) instead of taking g and
beta ready-made.

Usage:
    python Qwen/convert_weights.py --model Qwen/Qwen3.8-27B \
        --output <weights-dir>/qwen/qwen38_27b_bf16.bin

    # Structural smoke test -- the first 4 layers (3 DeltaNet + 1 full attention)
    python Qwen/convert_weights.py --model Qwen/Qwen3.8-27B --max-layers 4 \
        --output <weights-dir>/qwen/qwen38_27b_l4_bf16.bin
"""

import sys
from pathlib import Path
sys.path.insert( 0, str( Path( __file__ ).parent.parent ) )

import argparse
import json
import re
import struct

import torch
from safetensors import safe_open

from common import MilaStreamingWeightWriter, apply_transform, expand_qwen_tensor_map


SUPPORTED_MODELS = [
    'Qwen/Qwen3.8-27B',
]

# Checkpoint tensors this chassis does not model. Named as prefixes rather than
# discovered, so a tensor family that appears in a future revision is reported as
# unconsumed instead of being silently dropped.
SKIPPED_PREFIXES = ( 'model.visual.', 'mtp.' )

TORCH_DTYPE_MAP = {
    'float32':  torch.float32,
    'bfloat16': torch.bfloat16,
}


def _resolve_checkpoint( model_name: str ) -> Path:
    """Local directory holding the checkpoint, downloading it only if absent.

    A local path is taken as-is so a converted-from-disk checkpoint needs no hub
    access at all.
    """
    candidate = Path( model_name )

    if candidate.is_dir():
        return candidate

    from huggingface_hub import snapshot_download

    print( f"Resolving {model_name} (downloads only what is missing from the hub cache)..." )

    try:
        return Path( snapshot_download(
            model_name,
            allow_patterns=[ 'config.json', '*.safetensors', '*.safetensors.index.json' ] ) )
    except Exception as e:
        _check_hf_error( model_name, e )


def _check_hf_error( model_name: str, e: Exception ):
    name = type( e ).__name__
    msg  = str( e )
    if 'GatedRepo' in name or ('403' in msg and 'gated' in msg.lower()):
        print( f"\nError: '{model_name}' is a gated model." )
        print(  "  Accept the license on the model page, then authenticate: hf auth login" )
        sys.exit( 1 )
    if 'RepositoryNotFound' in name or '404' in msg:
        print( f"\nError: '{model_name}' not found on HuggingFace." )
        print(  "  Check the model name and your network connection." )
        sys.exit( 1 )
    raise e


class ShardedCheckpoint:
    """Read-only view of a sharded safetensors checkpoint.

    Shapes come from the shard headers, which cost one small read each -- so the whole
    output index can be declared before any tensor data is touched.
    """

    def __init__( self, root: Path ):
        self.root = root
        index_path = root / 'model.safetensors.index.json'

        if index_path.exists():
            weight_map = json.loads( index_path.read_text( encoding='utf-8' ) )[ 'weight_map' ]
        else:
            # Single-shard checkpoint: no index file is written for one file.
            weight_map = { name: 'model.safetensors'
                           for name in _shard_header( root / 'model.safetensors' ) }

        self.weight_map = weight_map
        self.shapes = {}
        self.dtypes = {}

        for shard in sorted( set( weight_map.values() ) ):
            for name, entry in _shard_header( root / shard ).items():
                self.shapes[ name ] = tuple( entry[ 'shape' ] )
                self.dtypes[ name ] = entry[ 'dtype' ]

        self._handles = {}

    def shape( self, name: str ):
        if name not in self.shapes:
            raise KeyError( f"expected tensor '{name}' not in the checkpoint" )

        return self.shapes[ name ]

    def tensor( self, name: str ):
        shard = self.weight_map[ name ]

        if shard not in self._handles:
            self._handles[ shard ] = safe_open( self.root / shard, framework='pt' )

        return self._handles[ shard ].get_tensor( name )

    def names( self ):
        return set( self.weight_map )


def _shard_header( path: Path ):
    """The safetensors header: an 8-byte length, then that many bytes of JSON."""
    with open( path, 'rb' ) as f:
        length = struct.unpack( '<Q', f.read( 8 ) )[ 0 ]
        header = json.loads( f.read( length ) )

    header.pop( '__metadata__', None )

    return header


def _tensor_to_numpy( tensor: torch.Tensor, dtype: str ):
    """Convert a torch tensor to a numpy array in the target Mila dtype.

    bfloat16 is returned as a uint16 view of the raw BF16 bit pattern, matching
    MilaStreamingWeightWriter and Mila's loader.
    """
    if dtype == 'bfloat16':
        return tensor.to( torch.bfloat16 ).contiguous().view( torch.uint16 ).numpy()
    else:
        return tensor.to( torch.float32 ).contiguous().numpy()


def _text_prefix( checkpoint: ShardedCheckpoint ) -> str:
    """The checkpoint's prefix for the text model's submodules.

    Two packagings exist and differ only here: 'model.' for a text-only
    Qwen3_5ForCausalLM, 'model.language_model.' for the published multimodal
    Qwen3_5ForConditionalGeneration. Detected from embed_tokens rather than assumed.
    """
    suffix = 'embed_tokens.weight'
    key = next( (k for k in checkpoint.names()
                 if k.endswith( suffix ) and not k.startswith( SKIPPED_PREFIXES )), None )

    if key is None:
        raise KeyError( 'embed_tokens.weight not found in the checkpoint' )

    return key[ : -len( suffix ) ]


def resolve_qwen_geometry( config: dict, max_layers: int = 0 ) -> dict:
    """Every geometry field Mila needs, read and validated from a Qwen config.json.

    Shared with the quantizing packer rather than restated there: these fields decide
    which block kind each layer is and how the DeltaNet projections split, so two
    tools disagreeing about them produces an artifact that loads into the wrong
    chassis. The two refusals below are the same for both callers.
    """
    # The published checkpoint nests the text geometry under text_config; a text-only
    # packaging would carry it at the top level.
    text_config = config.get( 'text_config', config )

    hidden_size    = text_config[ 'hidden_size' ]
    num_layers     = text_config[ 'num_hidden_layers' ]
    num_heads      = text_config[ 'num_attention_heads' ]
    num_kv_heads   = text_config[ 'num_key_value_heads' ]
    head_dim       = text_config.get( 'head_dim', hidden_size // num_heads )
    intermediate   = text_config[ 'intermediate_size' ]
    vocab_size     = text_config[ 'vocab_size' ]
    max_seq_len    = text_config[ 'max_position_embeddings' ]
    rms_eps        = text_config.get( 'rms_norm_eps', 1e-6 )
    interval       = text_config[ 'full_attention_interval' ]
    output_gate    = bool( text_config.get( 'attn_output_gate', False ) )
    tie_embeddings = bool( text_config.get( 'tie_word_embeddings', False ) )

    # rope_theta and the partial rotary factor moved into rope_parameters in
    # transformers 5.x; both spellings are read so either packaging converts.
    rope_parameters = text_config.get( 'rope_parameters', {} ) or {}
    rope_theta = float( text_config.get( 'rope_theta',
        rope_parameters.get( 'rope_theta', 1e7 ) ) )
    partial_rotary = float( text_config.get( 'partial_rotary_factor',
        rope_parameters.get( 'partial_rotary_factor', 0.25 ) ) )

    key_heads   = text_config[ 'linear_num_key_heads' ]
    value_heads = text_config[ 'linear_num_value_heads' ]
    key_head_dim   = text_config[ 'linear_key_head_dim' ]
    value_head_dim = text_config[ 'linear_value_head_dim' ]
    conv_kernel    = text_config[ 'linear_conv_kernel_dim' ]

    # Mila's DeltaNet geometry carries ONE head width. The two published fields are
    # equal at 128, and the delta rule's state is [key_head_dim, value_head_dim], so a
    # checkpoint that separated them would need a chassis change rather than a
    # converter change -- refused here rather than silently converted.
    if key_head_dim != value_head_dim:
        raise ValueError(
            f'linear_key_head_dim ({key_head_dim}) != linear_value_head_dim '
            f'({value_head_dim}); QwenConfig carries a single linear_head_dim' )

    if not output_gate:
        raise ValueError(
            'attn_output_gate is false; the ungated geometry is plain grouped-query '
            'attention and QwenAttentionBlock refuses it' )

    if max_layers:
        num_layers = min( num_layers, max_layers )

    return {
        'hidden_size': hidden_size, 'num_hidden_layers': num_layers,
        'num_attention_heads': num_heads, 'num_key_value_heads': num_kv_heads,
        'head_dim': head_dim, 'attn_output_gate': output_gate,
        'intermediate_size': intermediate, 'vocab_size': vocab_size,
        'max_position_embeddings': max_seq_len, 'rms_norm_eps': rms_eps,
        'rope_theta': rope_theta, 'partial_rotary_factor': partial_rotary,
        'full_attention_interval': interval, 'tie_word_embeddings': tie_embeddings,
        'linear_num_key_heads': key_heads, 'linear_num_value_heads': value_heads,
        'linear_head_dim': key_head_dim, 'linear_conv_kernel_dim': conv_kernel,
    }


def qwen_mila_metadata( geometry: dict, dtype: str, model_id: str ) -> dict:
    """The metadata block Mila's reader parses, from resolved geometry.

    Keys the reader's parser extracts, plus the Qwen-specific geometry. The parser
    searches for each quoted key and ignores what it does not know, so no key may be
    a prefix of another up to its closing quote. Shared with the packer: a quantized
    artifact and a BF16 one must declare the same architecture or they load into
    different models.
    """
    return {
        'architecture':            'qwen',
        'model_name':              model_id,
        'dtype':                   dtype,
        'vocab_size':              geometry[ 'vocab_size' ],
        'max_seq_length':          geometry[ 'max_position_embeddings' ],
        'embedding_dim':           geometry[ 'hidden_size' ],
        'num_layers':              geometry[ 'num_hidden_layers' ],
        'num_heads':               geometry[ 'num_attention_heads' ],
        'num_kv_heads':            geometry[ 'num_key_value_heads' ],
        'head_dim':                geometry[ 'head_dim' ],
        'hidden_dim':              geometry[ 'intermediate_size' ],
        'use_bias':                False,
        'tie_word_embeddings':     geometry[ 'tie_word_embeddings' ],
        'activation':              'silu',
        'norm_type':               'rmsnorm',
        'attention_type':          'gqa',
        'positional_encoding':     'rope',
        'rope_theta':              geometry[ 'rope_theta' ],
        'norm_epsilon':            geometry[ 'rms_norm_eps' ],
        'attention_output_gate':   geometry[ 'attn_output_gate' ],
        'full_attention_interval': geometry[ 'full_attention_interval' ],
        'partial_rotary_factor':   geometry[ 'partial_rotary_factor' ],
        'linear_num_key_heads':    geometry[ 'linear_num_key_heads' ],
        'linear_num_value_heads':  geometry[ 'linear_num_value_heads' ],
        'linear_head_dim':         geometry[ 'linear_head_dim' ],
        'linear_conv_kernel_dim':  geometry[ 'linear_conv_kernel_dim' ],
    }


def convert_qwen( model_name: str, output_path: str, dtype: str = 'bfloat16',
                  max_layers: int = 0 ):

    root = _resolve_checkpoint( model_name )
    config = json.loads( (root / 'config.json').read_text( encoding='utf-8' ) )
    geometry = resolve_qwen_geometry( config, max_layers )

    hidden_size  = geometry[ 'hidden_size' ]
    num_layers   = geometry[ 'num_hidden_layers' ]
    num_heads    = geometry[ 'num_attention_heads' ]
    num_kv_heads = geometry[ 'num_key_value_heads' ]
    head_dim     = geometry[ 'head_dim' ]
    intermediate = geometry[ 'intermediate_size' ]
    vocab_size   = geometry[ 'vocab_size' ]
    interval     = geometry[ 'full_attention_interval' ]

    key_heads      = geometry[ 'linear_num_key_heads' ]
    value_heads    = geometry[ 'linear_num_value_heads' ]
    key_head_dim   = geometry[ 'linear_head_dim' ]
    conv_kernel    = geometry[ 'linear_conv_kernel_dim' ]

    print( 'Resolved Qwen config:' )
    for k, v in geometry.items():
        print( f'  {k:28s} {v}' )

    checkpoint = ShardedCheckpoint( root )
    prefix = _text_prefix( checkpoint )

    if prefix != 'model.':
        print( f"  Note: multimodal packaging detected; text prefix is '{prefix}'" )

    # One head width: resolve_qwen_geometry refuses a checkpoint whose key and value
    # head dims differ, so linear_head_dim serves both.
    key_dim = key_heads * key_head_dim
    value_dim = value_heads * key_head_dim

    mappings = expand_qwen_tensor_map( num_layers, interval, key_dim, value_dim, prefix )

    raw_name = model_name.rsplit( '/', 1 )[ -1 ]
    model_id = raw_name.replace( '.', '_' ).replace( '-', '_' )

    writer = MilaStreamingWeightWriter( output_path )
    writer.set_metadata( qwen_mila_metadata( geometry, dtype, model_id ) )

    # ---- Declaration pass: shapes from the shard headers, no tensor data ----
    for mapping in mappings:
        writer.declare( mapping.mila, dtype,
            _output_shape( mapping, checkpoint, head_dim ) )

    _verify_geometry( writer.entries, num_layers, interval, hidden_size, vocab_size,
        intermediate, num_heads, num_kv_heads, head_dim, key_dim, value_dim, conv_kernel,
        key_head_dim, value_heads )

    print( f'\n  {len( writer.entries )} Mila tensors, '
           f'{writer.total_data_bytes() / 1024**3:.2f} GiB payload' )

    # ---- Data pass: one tensor at a time, source -> transform -> file ----
    consumed = set()
    reported_layer = -1

    with writer:
        for mapping in mappings:
            if mapping.mila.startswith( 'tf_layer_' ):
                layer = int( mapping.mila.split( '.', 1 )[ 0 ].removeprefix( 'tf_layer_' ) )

                if layer != reported_layer:
                    kind = 'full attention' if ((layer + 1) % interval) == 0 else 'DeltaNet'
                    print( f'  Converting layer {layer}/{num_layers - 1} ({kind})...' )
                    reported_layer = layer

            sources = [ checkpoint.tensor( source ) for source in mapping.sources ]
            consumed.update( mapping.sources )

            tensor = apply_transform( mapping, sources, head_dim )
            writer.write( mapping.mila, _tensor_to_numpy( tensor, dtype ) )

    _report_unconsumed( checkpoint, consumed, num_layers, max_layers )

    print( '\nConversion complete!' )
    print( f'  Output: {output_path}' )
    print( f'  Model:  {model_id}  dtype: {dtype}' )


def _output_shape( mapping, checkpoint: ShardedCheckpoint, head_dim: int ):
    """The shape a mapping's transform will produce, from the source shapes alone."""
    shapes = [ checkpoint.shape( source ) for source in mapping.sources ]

    if mapping.transform == 'concat':
        rows = sum( shape[ 0 ] for shape in shapes )

        return ( rows, ) + tuple( shapes[ 0 ][ 1: ] )

    if mapping.transform == 'rows':
        start, stop = mapping.rows

        return ( stop - start, ) + tuple( shapes[ 0 ][ 1: ] )

    if mapping.transform == 'conv_rows':
        start, stop = mapping.rows

        return ( stop - start, shapes[ 0 ][ -1 ] )

    if mapping.transform == 'gated_qkv':
        # The gate half stays: de-interleaving rearranges rows, it drops none.
        rows = sum( shape[ 0 ] for shape in shapes )

        return ( rows, shapes[ 0 ][ 1 ] )

    raise ValueError( f"unknown transform '{mapping.transform}' for {mapping.mila}" )


def _verify_geometry( entries, num_layers, interval, hidden_size, vocab_size,
                      intermediate, num_heads, num_kv_heads, head_dim,
                      key_dim, value_dim, conv_kernel, linear_head_dim, value_heads ):
    """Check every declared shape against what the Qwen components allocate.

    The declaration pass derives shapes from the checkpoint; this derives them from
    the config, independently. A disagreement means the checkpoint is not the geometry
    the config describes -- which is worth an hour of conversion to find out first.
    """
    q_width = num_heads * head_dim
    kv_width = num_kv_heads * head_dim

    expected = {
        'temb.wte':            ( vocab_size, hidden_size ),
        'rmsn_final.weight':   ( hidden_size, ),
        'lm_head.weight':      ( vocab_size, hidden_size ),
        'input_norm.weight':   ( hidden_size, ),
        'post_attn_norm.weight': ( hidden_size, ),
        'fc_gate_up.weight':   ( 2 * intermediate, hidden_size ),
        'fc_down.weight':      ( hidden_size, intermediate ),
        # Full attention: [query | gate | key | value], query and gate contiguous.
        'fc_qkv_proj.weight':  ( 2 * q_width + 2 * kv_width, hidden_size ),
        'q_norm.weight':       ( head_dim, ),
        'k_norm.weight':       ( head_dim, ),
        'fc_o_proj.weight':    ( hidden_size, q_width ),
        # Gated DeltaNet.
        'fc_in_proj_qk.weight': ( 2 * key_dim, hidden_size ),
        'fc_in_proj_v.weight':  ( value_dim, hidden_size ),
        'fc_in_proj_z.weight':  ( value_dim, hidden_size ),
        'fc_out_proj.weight':   ( hidden_size, value_dim ),
        'conv_qk.weight':       ( 2 * key_dim, conv_kernel ),
        'conv_v.weight':        ( value_dim, conv_kernel ),
        # The gated norm normalizes one value head, so its width is the head, not the
        # stream -- and it is the one norm in this family stored raw on both sides.
        'norm_gate.weight':     ( linear_head_dim, ),
        'delta_rule.A_log':     ( value_heads, ),
        'delta_rule.dt_bias':   ( value_heads, ),
    }

    for name, _, shape in entries:
        stem = name.split( '.', 1 )[ 1 ] if name.startswith( 'tf_layer_' ) else name
        want = expected.get( stem )

        if want is None:
            continue

        if shape != want:
            raise ValueError(
                f'{name}: checkpoint gives {shape}, the config implies {want}' )

    tensor_names = { name for name, _, _ in entries }
    full_attention_layers = sum( 1 for i in range( num_layers )
                                 if ((i + 1) % interval) == 0 )

    if len( tensor_names ) != len( entries ):
        raise ValueError( 'duplicate tensor names in the declared index' )

    declared_full = sum( 1 for name in tensor_names if name.endswith( 'fc_qkv_proj.weight' ) )

    if declared_full != full_attention_layers:
        raise ValueError(
            f'{declared_full} full-attention layers declared, config implies '
            f'{full_attention_layers}' )


def _report_unconsumed( checkpoint: ShardedCheckpoint, consumed, num_layers, max_layers ):
    """Account for every checkpoint tensor: consumed, deliberately skipped, or a gap.

    A tensor family this converter does not know about would otherwise be dropped in
    silence, which is the failure mode that produces a model that loads and is wrong.
    """
    skipped = { name for name in checkpoint.names() if name.startswith( SKIPPED_PREFIXES ) }
    unconsumed = checkpoint.names() - consumed - skipped

    if max_layers:
        # A truncated conversion leaves the layers past the cut unconsumed by design.
        unconsumed = { name for name in unconsumed
                       if not _past_layer_cut( name, num_layers ) }

    print( f'\n  Checkpoint tensors: {len( consumed )} consumed, '
           f'{len( skipped )} skipped (vision tower and MTP head)' )

    if unconsumed:
        sample = '\n    '.join( sorted( unconsumed )[ :10 ] )
        raise ValueError(
            f'{len( unconsumed )} checkpoint tensors were neither consumed nor '
            f'skipped:\n    {sample}' )


def _past_layer_cut( name: str, num_layers: int ) -> bool:
    """True for a text-model tensor belonging to a layer a truncated run stopped short of."""
    match = re.search( r'layers\.(\d+)\.', name )

    return match is not None and int( match.group( 1 ) ) >= num_layers


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description='Convert Qwen 3.8 weights to Mila format' )
    parser.add_argument( '--model', type=str, required=True,
        help='HuggingFace model name (or a local checkpoint directory)' )
    parser.add_argument( '--output', type=str, required=True,
        help='Output path for the Mila weight file' )
    parser.add_argument( '--dtype', type=str, default='bfloat16',
        choices=[ 'float32', 'bfloat16' ], help='Target dtype (default: bfloat16)' )
    parser.add_argument( '--max-layers', type=int, default=0,
        help='Convert only the first N layers -- a structural smoke test, not a model' )

    args = parser.parse_args()
    convert_qwen( args.model, args.output, args.dtype, args.max_layers )
