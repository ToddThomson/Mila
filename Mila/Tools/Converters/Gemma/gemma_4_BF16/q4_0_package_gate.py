#!/usr/bin/env python3
"""
Gate for a Mila Q4_0 Gemma 4 build: every Q4_0 tensor in Google's GGUF must equal the Mila weights in every code
and every FP16 scale bit.

Mila fuses q|k|v into qkv_proj and gate|up into fc_gate_up; the GGUF keeps them separate, so each GGUF tensor is
compared against its row range of the fused Mila tensor. The global layers carry no attn_v (K=V). The layouts
differ -- Mila stores codes [N, K/2] with the even column in the low nibble and scales in their own plane, the GGUF
interleaves 18-byte blocks -- so both are decoded to per-element codes and scale bits before comparing.

Usage:
    python q4_0_package_gate.py --gguf <gemma-4-12b-it-qat-q4_0.gguf> --weights <mila q4_0 .safetensors>
"""

import argparse
import json
import struct
import sys

import numpy as np

GGUF_TYPE_Q4_0 = 2

# GGUF scalar value types: struct format and size.
SCALARS = { 0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d' }
STRING, ARRAY = 8, 9

# GGUF projection -> ( Mila tensor, position in its fused row order ).
PROJECTIONS = {
    'attn_q': ( 'qkv_proj', 0 ), 'attn_k': ( 'qkv_proj', 1 ), 'attn_v': ( 'qkv_proj', 2 ),
    'attn_output': ( 'o_proj', 0 ),
    'ffn_gate': ( 'ffn.mlp.fc_gate_up', 0 ), 'ffn_up': ( 'ffn.mlp.fc_gate_up', 1 ),
    'ffn_down': ( 'ffn.mlp.fc_down', 0 ),
}


def read( f, fmt ):
    return struct.unpack( fmt, f.read( struct.calcsize( fmt ) ) )[ 0 ]


def read_string( f ):
    return f.read( read( f, '<Q' ) ).decode( 'utf-8' )


def read_value( f, value_type ):
    if value_type in SCALARS:
        return read( f, SCALARS[ value_type ] )

    if value_type == STRING:
        return read_string( f )

    if value_type == ARRAY:
        element_type = read( f, '<I' )
        return [ read_value( f, element_type ) for _ in range( read( f, '<Q' ) ) ]

    raise ValueError( f'unknown GGUF value type {value_type}' )


def gguf_index( path ):
    with open( path, 'rb' ) as f:
        if f.read( 4 ) != b'GGUF':
            raise ValueError( f'{path} is not a GGUF file' )

        read( f, '<I' )
        tensor_count = read( f, '<Q' )
        key_count = read( f, '<Q' )
        alignment = 32

        for _ in range( key_count ):
            key = read_string( f )
            value = read_value( f, read( f, '<I' ) )

            if key == 'general.alignment':
                alignment = value

        tensors = {}

        for _ in range( tensor_count ):
            name = read_string( f )
            dims = [ read( f, '<Q' ) for _ in range( read( f, '<I' ) ) ]
            tensors[ name ] = ( dims, read( f, '<I' ), read( f, '<Q' ) )

        data_start = ( f.tell() + alignment - 1 ) // alignment * alignment

    return tensors, data_start


def gguf_q4_0( f, data_start, dims, offset ):
    """Codes [N, K] (0..15) and scale bits [N, K/32] of a Q4_0 tensor."""
    columns, rows = dims[ 0 ], dims[ 1 ]
    f.seek( data_start + offset )
    blocks = np.frombuffer( f.read( rows * columns // 32 * 18 ), dtype=np.uint8 ).reshape( -1, 18 )
    scale_bits = blocks[ :, :2 ].copy().view( np.uint16 ).reshape( rows, columns // 32 )
    packed = blocks[ :, 2: ]
    codes = np.concatenate( [ packed & 0x0F, packed >> 4 ], axis=1 )  # element j low nibble, j + 16 high

    return codes.reshape( rows, columns ), scale_bits


class MilaWeights:
    """Reads single tensors out of a safetensors file without loading the rest."""

    def __init__( self, path ):
        self.path = path

        with open( path, 'rb' ) as f:
            header_bytes = read( f, '<Q' )
            self.header = json.loads( f.read( header_bytes ) )
            self.data_start = 8 + header_bytes

        self.metadata = self.header.pop( '__metadata__', {} )

    def raw( self, name ):
        entry = self.header[ name ]
        begin, end = entry[ 'data_offsets' ]

        with open( self.path, 'rb' ) as f:
            f.seek( self.data_start + begin )
            return entry, f.read( end - begin )

    def q4_0( self, name ):
        """Codes [N, K] and scale bits [N, K/32] of a Mila PerGroupInt4<32> tensor."""
        entry, data = self.raw( f'{name}.weight' )
        rows, half_columns = entry[ 'shape' ]

        if entry[ 'dtype' ] != 'U8':
            raise ValueError( f'{name}.weight is {entry[ "dtype" ]}, not packed U8' )

        packed = np.frombuffer( data, dtype=np.uint8 ).reshape( rows, half_columns )
        codes = np.empty( ( rows, half_columns * 2 ), dtype=np.uint8 )
        codes[ :, 0::2 ] = packed & 0x0F
        codes[ :, 1::2 ] = packed >> 4

        scale_entry, scale_data = self.raw( f'{name}.weight_scale' )

        if scale_entry[ 'dtype' ] != 'F16':
            raise ValueError( f'{name}.weight_scale is {scale_entry[ "dtype" ]}, not F16' )

        scale_bits = np.frombuffer( scale_data, dtype=np.uint16 ).reshape( scale_entry[ 'shape' ] )

        return codes, scale_bits


def main():
    parser = argparse.ArgumentParser( description=__doc__.strip().splitlines()[ 0 ] )
    parser.add_argument( '--gguf', required=True )
    parser.add_argument( '--weights', required=True )
    args = parser.parse_args()

    weights = MilaWeights( args.weights )
    scheme = weights.metadata.get( 'mila_quantization' )

    if scheme != 'q4_0':
        print( f'FAIL: the weights declare quantization {scheme!r}, not q4_0' )
        return 1

    tensors, data_start = gguf_index( args.gguf )
    q4_0 = sorted( ( name for name, ( _, kind, _ ) in tensors.items() if kind == GGUF_TYPE_Q4_0 ),
        key=lambda name: ( int( name.split( '.' )[ 1 ] ), name ) )

    # Group each layer's GGUF tensors by the fused Mila tensor they came from, in fused row order.
    fused = {}

    for name in q4_0:
        _, layer, projection, _ = name.split( '.' )
        mila, position = PROJECTIONS[ projection ]
        fused.setdefault( ( int( layer ), mila ), [] ).append( ( position, name ) )

    compared = 0
    differing_codes = 0
    differing_scales = 0
    failures = []

    with open( args.gguf, 'rb' ) as f:
        for ( layer, mila ), parts in sorted( fused.items() ):
            codes, scale_bits = weights.q4_0( f'tf_layer_{layer}.{mila}' )
            row = 0

            for _, name in sorted( parts ):
                dims, _, offset = tensors[ name ]
                expected_codes, expected_scales = gguf_q4_0( f, data_start, dims, offset )
                rows = expected_codes.shape[ 0 ]

                code_mismatch = int( np.count_nonzero( codes[ row:row + rows ] != expected_codes ) )
                scale_mismatch = int( np.count_nonzero( scale_bits[ row:row + rows ] != expected_scales ) )

                differing_codes += code_mismatch
                differing_scales += scale_mismatch
                compared += 1

                if code_mismatch or scale_mismatch:
                    failures.append( f'{name} -> tf_layer_{layer}.{mila} rows {row}..{row + rows}: '
                                     f'{code_mismatch} codes, {scale_mismatch} scales differ' )

                row += rows

            if row != codes.shape[ 0 ]:
                failures.append( f'tf_layer_{layer}.{mila}: {codes.shape[ 0 ]} rows, GGUF accounts for {row}' )

    print( f'{compared} of {len( q4_0 )} GGUF Q4_0 tensors compared' )
    print( f'codes differing: {differing_codes}   scale bits differing: {differing_scales}' )

    for failure in failures[ :20 ]:
        print( '  ' + failure )

    if failures or compared != len( q4_0 ):
        print( 'FAIL' )
        return 1

    print( 'PASS: every Q4_0 tensor equals the GGUF in every code and scale bit' )

    return 0


if __name__ == '__main__':
    sys.exit( main() )
