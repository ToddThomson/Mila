# Gemma 4 12B per-layer hidden states: Mila's FP8 prefill against HuggingFace on the same weights.
#
# GemmaLogLikelihoodCudaTests.DISABLED_LayerDump_Fp8 writes nine blocks' outputs, at every position, for a
# 128-token wikitext segment and its first 64 tokens. This runs HuggingFace on the same token ids, either with its
# own BF16 weights or with every text Linear and the tied table replaced by what Mila's FP8 prefill multiplies by --
# bf16( fp8( w / s ) * s ) with s = absmax / 448 per output row, Mila's quantizer -- and saves the same layers.
# Then it compares: Mila against HuggingFace on identical weights (parity), and each one's dependence on how many
# rows the prefill had (the 64-token prefix against the same positions inside the 128).
#
#   python hf_fp8_layer_comparison.py run --weights fp8
#   python hf_fp8_layer_comparison.py run --weights bf16
#   python hf_fp8_layer_comparison.py compare

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

LAYERS = [ 0, 5, 11, 17, 23, 29, 35, 41, 47 ]
MODEL_DIM = 3840
PREFIX = 64
DUMP = Path( tempfile.gettempdir() )


def mila_layers( name: str, positions: int, format: str = 'fp8' ) -> np.ndarray:
    return np.fromfile( DUMP / f'mila_{format}_layers_{name}.f32', dtype=np.float32 ).reshape( len( LAYERS ), positions, MODEL_DIM )


def quantize_like_mila( weight: torch.Tensor ) -> torch.Tensor:
    """Per output row: scale = absmax / 448, fp8 = e4m3( w / scale ), and the prefill's bf16( fp8 * scale )."""
    rows = weight.float()
    absmax = rows.abs().amax( dim=1, keepdim=True )
    scale = torch.where( absmax > 0, absmax / 448.0, torch.ones_like( absmax ) )
    codes = ( rows * ( 1.0 / scale ) ).to( torch.float8_e4m3fn )

    return ( codes.float() * scale ).to( torch.bfloat16 )


def quantize_fp4_like_mila( weight: torch.Tensor, group: int = 128 ) -> torch.Tensor:
    """Per 128-element group of a row: scale = absmax / 6, the nearest E2M1 level with Mila's breakpoints."""
    rows = weight.float().reshape( weight.shape[ 0 ], -1, group )
    absmax = rows.abs().amax( dim=2, keepdim=True )
    scale = torch.where( absmax > 0, absmax / 6.0, torch.ones_like( absmax ) )
    scaled = rows * ( 1.0 / scale )

    magnitude = scaled.abs()
    levels = torch.tensor( [ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 ] )
    breakpoints = torch.tensor( [ 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0 ] )
    index = torch.bucketize( magnitude, breakpoints, right=True )
    decoded = torch.sign( scaled ) * levels[ index ]

    return ( decoded * scale ).reshape( weight.shape ).to( torch.bfloat16 )


def fp8_round( values: torch.Tensor ) -> torch.Tensor:
    return values.to( torch.float8_e4m3fn ).float()


def per_token_fp8( module, inputs ):
    """Mila's W4A8 activation step: each token's row divided by absmax / 448 and rounded to E4M3."""
    x = inputs[ 0 ]
    rows = x.float()
    scale = rows.abs().amax( dim=-1, keepdim=True ).clamp_min( 1e-12 ) / 448.0

    return ( ( fp8_round( rows / scale ) * scale ).to( x.dtype ), ) + tuple( inputs[ 1: ] )


def apply_w4a8( text ):
    """
    Mila's FP4 prefill arithmetic on HuggingFace's layers: the FP4 weights re-expressed in E4M3 under one scale per
    Mila tensor -- q, k and v are one fused tensor there, as are gate and up -- and every quantized layer's input
    rounded to E4M3 per token. sB = 6 * max( group scale ) / 448, which is absmax( FP4 weights ) / 448.
    """
    with torch.no_grad():
        for layer in text.layers:
            attention = layer.self_attn
            fused = [ [ module for module in ( getattr( attention, name, None ) for name in ( 'q_proj', 'k_proj', 'v_proj' ) )
                        if isinstance( module, torch.nn.Linear ) ],
                      [ attention.o_proj ],
                      [ layer.mlp.gate_proj, layer.mlp.up_proj ],
                      [ layer.mlp.down_proj ] ]

            for group in fused:
                fp4 = [ quantize_fp4_like_mila( module.weight ).float() for module in group ]
                weight_scale = max( values.abs().max().item() for values in fp4 ) / 448.0

                for module, values in zip( group, fp4 ):
                    module.weight.copy_( ( fp8_round( values / weight_scale ) * weight_scale ).to( torch.bfloat16 ) )
                    module.register_forward_pre_hook( per_token_fp8 )

            # The tied table stays FP8 per row, as for every quantized body.
        text.embed_tokens.weight.copy_( quantize_like_mila( text.embed_tokens.weight ) )


def load( weights: str ):
    """HuggingFace's Gemma 4 12B it with the weights Mila's prefill multiplies by, dispatched over the card and CPU."""
    from accelerate import dispatch_model, infer_auto_device_map
    from transformers import AutoModelForCausalLM

    os.environ.setdefault( 'CUDA_VISIBLE_DEVICES', 'GPU-9a81c7d1-9db2-16b3-c256-2f991ec2a22c' )

    model = AutoModelForCausalLM.from_pretrained( 'google/gemma-4-12b-it', dtype=torch.bfloat16, device_map='cpu' ).eval()
    text = model.model.language_model

    if weights in ( 'fp8', 'fp4' ):
        body = quantize_like_mila if weights == 'fp8' else quantize_fp4_like_mila

        with torch.no_grad():
            for layer in text.layers:
                for module in list( layer.self_attn.children() ) + list( layer.mlp.children() ):
                    if isinstance( module, torch.nn.Linear ):
                        module.weight.copy_( body( module.weight ) )

            # A quantized body keeps its tied table at FP8 per row, whichever the body's format.
            text.embed_tokens.weight.copy_( quantize_like_mila( text.embed_tokens.weight ) )

    elif weights == 'w4a8':
        apply_w4a8( text )

    device_map = infer_auto_device_map( model, max_memory={ 0: '14GiB', 'cpu': '26GiB' },
        no_split_module_classes=model._no_split_modules )

    # A layer split across devices leaves the buffers it holds directly (layer_scalar) unplaced; they go with a sibling.
    names = [ name for name, _ in model.named_parameters() ] + [ name for name, _ in model.named_buffers() ]

    for name in names:
        if any( name == key or name.startswith( key + '.' ) for key in device_map ):
            continue

        parent = name.rsplit( '.', 1 )[ 0 ]
        device_map[ name ] = next( device for key, device in device_map.items() if key.startswith( parent + '.' ) )

    return dispatch_model( model, device_map=device_map ), text


def perplexity( weights: str ):
    """The segments DISABLED_SegmentPerplexity wrote, scored the way Mila scores them: every next token, after the softcap."""
    segments = np.fromfile( DUMP / 'gemma_perplexity_segments.f32', dtype=np.float32 ).astype( np.int64 ).reshape( -1, 1024 )

    model, _ = load( weights )

    total = 0.0
    positions = 0

    with torch.no_grad():
        for ids in segments:
            tokens = torch.tensor( [ ids.tolist() ] ).to( model.device )
            logits = model( input_ids=tokens, use_cache=False ).logits[ 0, :-1 ].double()
            log_probabilities = torch.log_softmax( logits, dim=-1 ).gather( 1, tokens[ 0, 1: ].unsqueeze( 1 ).to( logits.device ) )
            total += log_probabilities.sum().item()
            positions += len( ids ) - 1

    mean = -total / positions
    print( f'  HuggingFace {weights}: {positions} positions, mean negative log-likelihood {mean:.6f}, perplexity {np.exp( mean ):.3f}' )


def run( weights: str ):
    tokens = np.fromfile( DUMP / 'mila_fp8_layers_tokens.f32', dtype=np.float32 ).astype( np.int64 ).tolist()

    model, text = load( weights )

    captured = {}

    def hook( index ):
        def record( module, inputs, output ):
            hidden = output[ 0 ] if isinstance( output, tuple ) else output
            captured[ index ] = hidden[ 0 ].float().cpu().numpy()
        return record

    for index in LAYERS:
        text.layers[ index ].register_forward_hook( hook( index ) )

    saved = {}

    with torch.no_grad():
        for name, ids in [ ( 'longer', tokens ), ( 'prefix', tokens[ :PREFIX ] ) ]:
            captured.clear()
            model( input_ids=torch.tensor( [ ids ] ).to( model.device ), use_cache=False )
            saved[ name ] = np.stack( [ captured[ index ] for index in LAYERS ] )

    np.savez( DUMP / f'hf_{weights}_layers.npz', **saved )
    print( f'saved HuggingFace ({weights} weights) layers to {DUMP}' )


def projections():
    """
    Layer 0's four FP4 projections, recomputed from the package's own FP4 weights and Mila's captured inputs:
    as W4A16 (the FP4 weights times BF16 activations, what decode computes) and as W4A8 exactly as Mila's prefill
    specifies it, every dot product in FP64. Mila's own output is compared against both, so the error of its GEMM
    is measured apart from the rounding W4A8 is designed to do.
    """
    from safetensors import safe_open

    package = Path( r'D:\Repos\Mila\Data\models\gemma\gemma4_12b_it_fp4.safetensors' )
    pairs = [ ( 'qkv_proj', 'input_norm' ), ( 'o_proj', 'gqa' ), ( 'fc_gate_up', 'pre_ffn_norm' ), ( 'fc_down', 'geglu' ) ]
    levels = torch.tensor( [ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 ], dtype=torch.float32 )

    def decode( nibbles: torch.Tensor ) -> torch.Tensor:
        magnitude = levels[ ( nibbles & 7 ).long() ]
        return torch.where( nibbles >= 8, -magnitude, magnitude )

    def bf16( values: torch.Tensor ) -> torch.Tensor:
        return values.to( torch.bfloat16 ).to( torch.float64 )

    print( '\n  Layer 0 projections, 128 tokens: Mila\'s output against exact recomputations, relative L2 (all rows)\n' )
    print( '  projection  | Mila vs exact W4A8 | Mila vs exact W4A16 | exact W4A8 vs exact W4A16' )

    with safe_open( str( package ), framework='pt', device='cpu' ) as weights:
        for projection, source in pairs:
            packed = weights.get_tensor( f'tf_layer_0.{projection}.weight' )
            group_scales = weights.get_tensor( f'tf_layer_0.{projection}.weight_scale' )

            codes = torch.stack( [ decode( packed & 0xF ), decode( packed >> 4 ) ], dim=-1 ).reshape( packed.shape[ 0 ], -1 )
            out_features, in_features = codes.shape

            x = torch.from_numpy( np.fromfile( DUMP / f'gemma_projection_{source}.f32', dtype=np.float32 ) ).reshape( -1, in_features )
            mila = torch.from_numpy( np.fromfile( DUMP / f'gemma_projection_{projection}.f32', dtype=np.float32 ) ) \
                .reshape( -1, out_features ).double()

            # W4A16: the FP4 value times its group scale, in FP32 as the decode matvec forms it.
            expanded_scales = group_scales.repeat_interleave( 128, dim=1 )
            weight = codes * expanded_scales
            w4a16 = bf16( x.double() @ weight.double().T )

            # W4A8 as the prefill specifies it: sB = 6 * max( group scale ) / 448, weights fp4 * ( gs * (1/sB) ) rounded
            # to E4M3; activations x * (1/sa) rounded to E4M3 with sa = row absmax / 448; the product scaled by sB and
            # rounded to BF16, then by sa and rounded to BF16.
            weight_scale = torch.tensor( group_scales.max().item() * 6.0 / 448.0, dtype=torch.float32 )
            weight_fp8 = ( codes * ( expanded_scales * ( 1.0 / weight_scale ) ) ).to( torch.float8_e4m3fn ).double()

            token_scales = ( x.abs().amax( dim=1, keepdim=True ).clamp_min( 1e-12 ) / 448.0 ).float()
            x_fp8 = ( x * ( 1.0 / token_scales ) ).to( torch.float8_e4m3fn ).double()

            raw = x_fp8 @ weight_fp8.T
            w4a8 = bf16( bf16( raw * weight_scale.double() ).float() * token_scales ).double()

            def rel( left, right ):
                return ( ( left - right ).norm() / right.norm() ).item()

            print( f'  {projection:<11} | {rel( mila, w4a8 ):>18.3e} | {rel( mila, w4a16 ):>19.3e} | {rel( w4a8, w4a16 ):>24.3e}' )


def relative( left: np.ndarray, right: np.ndarray ) -> np.ndarray:
    """Relative L2 per position, left as the reference."""
    return np.linalg.norm( left - right, axis=-1 ) / np.linalg.norm( left, axis=-1 )


def compare():
    nan = float( 'nan' )
    hf = { name: np.load( DUMP / f'hf_{name}_layers.npz' ) for name in [ 'bf16', 'fp8', 'fp4', 'w4a8' ]
           if ( DUMP / f'hf_{name}_layers.npz' ).exists() }
    mila = { name: ( mila_layers( 'longer', 128, name ), mila_layers( 'prefix', PREFIX, name ) ) for name in [ 'fp8', 'fp4' ]
             if ( DUMP / f'mila_{name}_layers_longer.f32' ).exists() }

    def parity( name, row ):
        if name not in hf or name not in mila:
            return nan
        return relative( hf[ name ][ 'longer' ][ row ], mila[ name ][ 0 ][ row ] )[ 1: ].mean()

    def quantizing( name, row ):
        if name not in hf or 'bf16' not in hf:
            return nan
        return relative( hf[ 'bf16' ][ 'longer' ][ row ], hf[ name ][ 'longer' ][ row ] )[ 1: ].mean()

    def row_count( pair, row ):
        return relative( pair[ 0 ][ row, :PREFIX ], pair[ 1 ][ row ] )[ 1: ].mean()

    print( '\n  Parity on the 128-token prefill: Mila against HuggingFace with the same weights, relative L2' )
    print( '  (mean over positions 1-127; position 0 is <bos>), beside what quantizing does to HuggingFace itself\n' )
    print( '  layer | Mila FP8 vs HF-fp8 | HF-fp8 vs HF-bf16 | Mila FP4 vs HF-fp4 | HF-fp4 vs HF-bf16 | Mila FP4 vs HF-w4a8' )

    for row, layer in enumerate( LAYERS ):
        w4a8 = relative( hf[ 'w4a8' ][ 'longer' ][ row ], mila[ 'fp4' ][ 0 ][ row ] )[ 1: ].mean()             if 'w4a8' in hf and 'fp4' in mila else nan
        print( f'  {layer:>5} | {parity( "fp8", row ):>18.3e} | {quantizing( "fp8", row ):>17.3e} | '
               f'{parity( "fp4", row ):>18.3e} | {quantizing( "fp4", row ):>17.3e} | {w4a8:>18.3e}' )

    print( '\n  Row-count dependence: the 64-token prefix against the same positions inside the 128, relative L2' )
    print( '  (mean over positions 1-63)\n' )
    print( '  layer |  Mila FP8  |  Mila FP4  |   HF fp8   |   HF fp4   |   HF bf16' )

    for row, layer in enumerate( LAYERS ):
        values = [ row_count( mila[ name ], row ) if name in mila else nan for name in [ 'fp8', 'fp4' ] ]
        values += [ row_count( ( hf[ name ][ 'longer' ], hf[ name ][ 'prefix' ] ), row ) if name in hf else nan
                    for name in [ 'fp8', 'fp4', 'bf16' ] ]
        print( f'  {layer:>5} | ' + ' | '.join( f'{value:>10.3e}' for value in values ) )


def main():
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( 'mode', choices=[ 'run', 'compare', 'perplexity', 'projections' ] )
    parser.add_argument( '--weights', choices=[ 'bf16', 'fp8', 'fp4', 'w4a8' ], default='fp8' )
    arguments = parser.parse_args()

    if arguments.mode == 'run':
        run( arguments.weights )
    elif arguments.mode == 'perplexity':
        perplexity( arguments.weights )
    elif arguments.mode == 'projections':
        projections()
    else:
        compare()


if __name__ == '__main__':
    main()
