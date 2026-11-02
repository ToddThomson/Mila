"""HuggingFace's Gemma 4 drafter on the inputs Mila's drafter used, step by step (Specifications/Gemma4Mtp.md 5.1).

`Drafting drafter-parity` runs the target over a prompt and the drafter K steps from its state, and dumps each
step's inputs -- the target's scaled embedding of the step's token and the hidden state it joins -- the target's
two caches in position order, and Mila's logits. This feeds the same inputs to `Gemma4AssistantForCausalLM.forward`
in FP32 on the CPU and compares the logits, so the comparison is the drafter's forward alone: not HuggingFace's
generation loop, and not the target, whose weights Mila holds in a different format.

Mila computes in BF16, so the yardstick is HuggingFace's own BF16 run: per step, Mila's mean logit difference from the
FP32 reference may not exceed twice the BF16 reference's, and the argmax must equal the FP32 reference's. A top-5
set that differs is reported with the gap at fifth place, since a near-tie there reorders under rounding.

Usage:
    python Gemma/hf_gemma_drafter_reference.py --dump <directory> \
        [--model google/gemma-4-12B-it-qat-q4_0-unquantized-assistant]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def bf16_file( path: Path, shape ):
    raw = np.fromfile( path, dtype=np.uint16 ).astype( np.uint32 ) << 16

    return torch.from_numpy( raw.view( np.float32 ).reshape( shape ).copy() )


def f32_file( path: Path ):
    return torch.from_numpy( np.fromfile( path, dtype=np.float32 ).copy() )


def main():
    parser = argparse.ArgumentParser( description=__doc__.splitlines()[ 0 ] )
    parser.add_argument( '--dump', type=Path, required=True )
    parser.add_argument( '--model', default='google/gemma-4-12B-it-qat-q4_0-unquantized-assistant' )
    args = parser.parse_args()

    from transformers.models.gemma4_unified_assistant.modeling_gemma4_unified_assistant import (
        Gemma4UnifiedAssistantForCausalLM )

    manifest = json.loads( ( args.dump / 'manifest.json' ).read_text() )
    models = { dtype: Gemma4UnifiedAssistantForCausalLM.from_pretrained( args.model, dtype=dtype ).eval()
               for dtype in ( torch.float32, torch.bfloat16 ) }

    def caches_as( dtype ):
        caches = {}

        for name, layer_type in ( ( 'sliding', 'sliding_attention' ), ( 'global', 'full_attention' ) ):
            geometry = manifest[ name ]
            shape = ( 1, geometry[ 'num_kv_heads' ], geometry[ 'count' ], geometry[ 'head_size' ] )
            caches[ layer_type ] = ( bf16_file( args.dump / f'{name}_k.bf16', shape ).to( dtype ),
                                     bf16_file( args.dump / f'{name}_v.bf16', shape ).to( dtype ) )

        return caches

    caches = { dtype: caches_as( dtype ) for dtype in models }
    width = manifest[ 'target_model_dim' ]
    failures = 0

    print( f"position {manifest[ 'position' ]}; sliding keys {manifest[ 'sliding' ][ 'count' ]}, "
           f"global keys {manifest[ 'global' ][ 'count' ]}" )

    for step in range( manifest[ 'steps' ] ):
        embedding = f32_file( args.dump / f'embedding_{step}.f32' ).reshape( 1, 1, width )
        hidden = f32_file( args.dump / f'hidden_{step}.f32' ).reshape( 1, 1, width )
        mila = f32_file( args.dump / f'logits_{step}.f32' )

        logits = {}

        for dtype, model in models.items():
            with torch.no_grad():
                output = model( inputs_embeds=torch.cat( [ embedding, hidden ], dim=-1 ).to( dtype ),
                                position_ids=torch.tensor( [ [ manifest[ 'position' ] ] ] ),
                                shared_kv_states=caches[ dtype ], attention_mask=None )

            logits[ dtype ] = output.logits[ 0, 0 ].float()

        reference = logits[ torch.float32 ]
        mila_error = ( mila - reference ).abs().mean()
        bf16_error = ( logits[ torch.bfloat16 ] - reference ).abs().mean()
        same_argmax = int( mila.argmax() ) == int( reference.argmax() )
        passed = same_argmax and mila_error <= 2.0 * bf16_error
        failures += 0 if passed else 1

        top_mila = set( torch.topk( mila, 5 ).indices.tolist() )
        ranked = torch.topk( reference, 6 ).values
        top_note = '' if top_mila == set( torch.topk( reference, 5 ).indices.tolist() ) \
            else f'; top-5 differs, reference gap at fifth place {ranked[ 4 ] - ranked[ 5 ]:.4f}'

        print( f'step {step}: argmax mila {int( mila.argmax() )} hf {int( reference.argmax() )}; mean |difference| '
               f'from FP32: mila {mila_error:.5f}, HF BF16 {bf16_error:.5f}{top_note}  {"PASS" if passed else "FAIL"}' )

    print( 'PASS' if failures == 0 else f'FAIL: {failures} of {manifest[ "steps" ]} steps' )
    sys.exit( 1 if failures else 0 )


if __name__ == '__main__':
    main()
