#!/usr/bin/env python3
"""
Gate for a Mila Q4_0 Llama build: every Q4_0 tensor of the GGUF must equal the Mila weights in every code and every
FP16 scale bit (ModelFamilyParity.md 8.4, L3, test 3).

Gemma's gate (q4_0_package_gate.py) reads both files; this adds what differs for Llama. Mila names its fused tensors
fc_qkv_proj, fc_out_proj, fc_gate_up and fc_down, and keeps HuggingFace's row order for query and key, where
llama.cpp permutes those rows into interleaved rotary pairs -- so the GGUF's attn_q and attn_k rows are put back in
HuggingFace's order before comparing. A row permutation moves codes and scales together.

Usage:
    python llama_q4_0_package_gate.py --gguf <llama31_8b_instruct_q4_0.gguf> --weights <mila q4_0 .safetensors>
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert( 0, str( Path( __file__ ).resolve().parents[ 1 ] / 'Gemma' / 'gemma_4_BF16' ) )
from q4_0_package_gate import GGUF_TYPE_Q4_0, MilaWeights, gguf_index, gguf_q4_0

HEADS = 32
KV_HEADS = 8

# GGUF projection -> ( Mila tensor, position in its fused row order, heads whose rows llama.cpp permuted ).
PROJECTIONS = {
    'attn_q': ( 'fc_qkv_proj', 0, HEADS ), 'attn_k': ( 'fc_qkv_proj', 1, KV_HEADS ), 'attn_v': ( 'fc_qkv_proj', 2, None ),
    'attn_output': ( 'fc_out_proj', 0, None ),
    'ffn_gate': ( 'fc_gate_up', 0, None ), 'ffn_up': ( 'fc_gate_up', 1, None ),
    'ffn_down': ( 'fc_down', 0, None ),
}


def unpermute( rows: np.ndarray, heads: int ) -> np.ndarray:
    """Inverse of convert_hf_to_gguf.py's permute: interleaved rotary pairs back to rotate-half order."""
    return rows.reshape( heads, rows.shape[ 0 ] // heads // 2, 2, *rows.shape[ 1: ] ).swapaxes( 1, 2 ).reshape( rows.shape )


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

    fused = {}

    for name in q4_0:
        _, layer, projection, _ = name.split( '.' )
        mila, position, heads = PROJECTIONS[ projection ]
        fused.setdefault( ( int( layer ), mila ), [] ).append( ( position, name, heads ) )

    compared = 0
    differing_codes = 0
    differing_scales = 0
    failures = []

    with open( args.gguf, 'rb' ) as f:
        for ( layer, mila ), parts in sorted( fused.items() ):
            codes, scale_bits = weights.q4_0( f'tf_layer_{layer}.{mila}' )
            row = 0

            for _, name, heads in sorted( parts ):
                dims, _, offset = tensors[ name ]
                expected_codes, expected_scales = gguf_q4_0( f, data_start, dims, offset )

                if heads is not None:
                    expected_codes = unpermute( expected_codes, heads )
                    expected_scales = unpermute( expected_scales, heads )

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
