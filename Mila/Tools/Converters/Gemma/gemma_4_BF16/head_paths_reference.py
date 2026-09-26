# The tied FP8 head's two paths against the table itself (ModelFamilyParity.md 8.2, G1 result).
#
# Reads what GemmaLogLikelihoodCudaTests.DISABLED_HeadPathsDump_Fp4 wrote to the temp directory -- 64 normalized
# rows and the logits Mila's head produced from them at window 1 (the decode matvec) and window 64 (the staged
# GEMM) -- and computes exact logits from `temb.wte` and `temb.wte_scale` in FP64. Reports each path's error
# against the exact logits, beside the error of simply rounding the exact logits to BF16, which is the best a
# head that writes BF16 logits can do; then what each path does to the log-probability of the actual next token
# after the softcap, which is what a perplexity measures.
#
#   python head_paths_reference.py --weights <repo>/Data/models/gemma/gemma4_12b_it_fp4.safetensors

import argparse
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

VOCABULARY = 262144
MODEL_DIM = 3840
SOFTCAP = 30.0


def read( directory: Path, name: str, columns: int ) -> np.ndarray:
    return np.fromfile( directory / name, dtype=np.float32 ).reshape( -1, columns )


def to_bfloat16( values: torch.Tensor ) -> torch.Tensor:
    return values.to( torch.bfloat16 ).to( torch.float64 )


def target_log_probabilities( logits: torch.Tensor, targets: torch.Tensor ) -> torch.Tensor:
    capped = SOFTCAP * torch.tanh( logits / SOFTCAP )
    return torch.log_softmax( capped, dim=-1 ).gather( 1, targets.unsqueeze( 1 ) ).squeeze( 1 )


def main():
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( '--weights', type=Path, required=True )
    parser.add_argument( '--dump-dir', type=Path, default=Path( tempfile.gettempdir() ) )
    arguments = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    normalized = torch.from_numpy( read( arguments.dump_dir, 'gemma_head_normalized.f32', MODEL_DIM ) ).to( device, torch.float64 )
    window_1 = torch.from_numpy( read( arguments.dump_dir, 'gemma_head_logits_w1.f32', VOCABULARY ) ).to( device, torch.float64 )
    window_64 = torch.from_numpy( read( arguments.dump_dir, 'gemma_head_logits_w64.f32', VOCABULARY ) ).to( device, torch.float64 )
    targets = torch.from_numpy( np.fromfile( arguments.dump_dir / 'gemma_head_targets.f32', dtype=np.float32 ) ).to( device ).long()

    rows = normalized.shape[ 0 ]
    exact = torch.empty( rows, VOCABULARY, dtype=torch.float64, device=device )
    staged_like = torch.empty_like( exact )

    with safe_open( str( arguments.weights ), framework='pt', device='cpu' ) as weights:
        table = weights.get_tensor( 'temb.wte' )
        scales = weights.get_tensor( 'temb.wte_scale' ).to( device, torch.float64 )

    chunk = 16384

    for begin in range( 0, VOCABULARY, chunk ):
        codes = table[ begin : begin + chunk ].to( device ).to( torch.float64 )
        scale = scales[ begin : begin + chunk ]

        # Exact: the FP8 value times its row scale, in FP64, as the decode matvec does in FP32.
        exact[ :, begin : begin + chunk ] = ( normalized @ codes.T ) * scale

        # What the staged path's weights are: each FP8 value times its scale, rounded to BF16 before the GEMM.
        rounded_weights = to_bfloat16( codes * scale.unsqueeze( 1 ) )
        staged_like[ :, begin : begin + chunk ] = normalized @ rounded_weights.T

    floor = to_bfloat16( exact )

    def report( label: str, logits: torch.Tensor ):
        error = ( logits - exact ).abs()
        log_probability_error = target_log_probabilities( logits, targets ) - target_log_probabilities( exact, targets )

        print( f'  {label:<44} max |error| {error.max().item():.5f}   mean |error| {error.mean().item():.6f}   '
               f'mean log-prob error {log_probability_error.mean().item():+.3e} nats   '
               f'mean |log-prob error| {log_probability_error.abs().mean().item():.3e}' )

    print( f'\n  {rows} positions, exact logits in FP64 from temb.wte\n' )
    report( 'exact, rounded to BF16 (the floor)', floor )
    report( 'window 1: decode matvec (Mila)', window_1 )
    report( 'window 64: staged BF16 GEMM (Mila)', window_64 )
    report( 'staged weights, exact GEMM (model of window 64)', to_bfloat16( staged_like ) )

    exact_nll = -target_log_probabilities( exact, targets ).mean().item()
    window_1_nll = -target_log_probabilities( window_1, targets ).mean().item()
    window_64_nll = -target_log_probabilities( window_64, targets ).mean().item()

    print( f'\n  mean negative log-likelihood: exact {exact_nll:.6f}, window 1 {window_1_nll:.6f}, window 64 {window_64_nll:.6f}' )
    print( f'  relative perplexity against exact: window 1 {math.expm1( window_1_nll - exact_nll ):+.3e}, '
           f'window 64 {math.expm1( window_64_nll - exact_nll ):+.3e}' )


if __name__ == '__main__':
    main()
