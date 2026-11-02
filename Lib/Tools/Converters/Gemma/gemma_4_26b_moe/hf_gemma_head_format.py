#!/usr/bin/env python3
# Gemma 4's tied head in other weight formats, against BF16, on the same final hidden states.
#
# The decoder runs layer by layer from the checkpoint's shards (hf_gemma_layer_stream.py), so the 26B-A4B runs on one
# card; its final normalized hidden state at every position of a PG-19 book is kept, and every head format is applied
# to those same states. Only the head differs between arms. Scored as Mila scores: the true next token after the final
# softcap. The head's decode cost is its bytes, since its matvec already streams at the card's bandwidth.
#
#   python Gemma/gemma_4_26b_moe/hf_gemma_head_format.py --model <snapshot of google/gemma-4-26B-A4B-it-qat-q4_0-unquantized>
#   python Gemma/gemma_4_26b_moe/hf_gemma_head_format.py --model <snapshot of google/gemma-4-12B-it-qat-q4_0-unquantized>

import sys
from pathlib import Path
sys.path.insert( 0, str( Path( __file__ ).resolve().parent.parent.parent ) )
sys.path.insert( 0, str( Path( __file__ ).resolve().parent.parent ) )
sys.path.insert( 0, str( Path( __file__ ).resolve().parent.parent / 'gemma_4_BF16' ) )

import argparse
import os

import torch

from hf_fp8_layer_comparison import quantize_like_mila, quantize_q4_0
from hf_gemma_layer_stream import StreamedReference, _open_checkpoint, _release
from hf_long_context_loss import BOOK_TURN, join_wraps

BOOKS = Path( __file__ ).resolve().parents[ 5 ] / 'Data' / 'Datasets' / 'PG19' / 'raw' / 'test'
BEGIN_OF_TEXT = 2
POSITIONS_PER_PASS = 256


def quantize_q8_0( weight: torch.Tensor, group: int = 32 ) -> torch.Tensor:
    """Q8_0: per 32 elements of a row, d = absmax / 127 stored FP16, q = round( x / d ). 8.5 bits a weight."""
    rows = weight.float().reshape( weight.shape[ 0 ], -1, group )
    d = ( rows.abs().amax( dim=2, keepdim=True ) / 127.0 ).half().float()
    inverse = torch.where( d != 0, 1.0 / d, torch.zeros_like( d ) )

    return ( torch.round( rows * inverse ).clamp( -127, 127 ) * d ).reshape( weight.shape ).to( torch.bfloat16 )


def quantize_int6( weight: torch.Tensor, group: int = 32 ) -> torch.Tensor:
    """Six bits per 32 elements of a row, Q4_0's rule widened: d = the signed extreme / -32 stored FP16, q in [-32, 31]."""
    rows = weight.float().reshape( weight.shape[ 0 ], -1, group )
    extreme = torch.gather( rows, 2, rows.abs().argmax( dim=2, keepdim=True ) )
    d = ( extreme / -32.0 ).half().float()
    inverse = torch.where( d != 0, 1.0 / d, torch.zeros_like( d ) )

    return ( torch.round( rows * inverse ).clamp( -32, 31 ) * d ).reshape( weight.shape ).to( torch.bfloat16 )


def quantize_q6_k_like( weight: torch.Tensor, super_block: int = 256, block: int = 16 ) -> torch.Tensor:
    """
    Q6_K's layout: per 256 elements one FP16 scale, per 16 an INT8 sub-scale of it, codes in [-32, 31]; 6.5625 bits a
    weight. Each sub-block's scale is set by its signed extreme, without llama.cpp's search over candidate scales.
    """
    rows = weight.float().reshape( weight.shape[ 0 ], -1, super_block // block, block )
    extreme = torch.gather( rows, 3, rows.abs().argmax( dim=3, keepdim=True ) )
    wanted = extreme / -32.0
    d = ( wanted.abs().amax( dim=2, keepdim=True ) / 127.0 ).half().float()
    sub = torch.where( d != 0, torch.round( wanted / torch.where( d != 0, d, torch.ones_like( d ) ) ),
                       torch.zeros_like( wanted ) ).clamp( -128, 127 )
    scale = d * sub
    inverse = torch.where( scale != 0, 1.0 / scale, torch.zeros_like( scale ) )

    return ( torch.round( rows * inverse ).clamp( -32, 31 ) * scale ).reshape( weight.shape ).to( torch.bfloat16 )


# Name, bits a weight, quantizer. BF16 is the reference every other arm is scored against.
ARMS = [
    ( 'bf16', 16.0, None ),
    ( 'fp8 per row (Mila)', 8.0, quantize_like_mila ),
    ( 'q8_0', 8.5, quantize_q8_0 ),
    ( 'q6_k-like', 6.5625, quantize_q6_k_like ),
    ( 'int6 per 32', 6.5, quantize_int6 ),
    ( 'q4_0', 4.5, quantize_q4_0 ),
]


class AllPositions( StreamedReference ):
    """The streamed driver, keeping the final normalized hidden state at every position instead of the last."""

    @torch.no_grad()
    def _head( self, hidden ):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

        with torch.device( 'meta' ):
            norm = Gemma4RMSNorm( self.config.hidden_size, eps=self.config.rms_norm_eps )

        norm.load_state_dict(
            { 'weight': self.weights.tensor( f'{self.prefix}norm.weight' ).to( device=self.device, dtype=self.dtype ) },
            assign=True )

        return norm( hidden )[ 0 ].clone(), None


def book_ids( tokenizer, book: str, tokens: int ) -> list[ int ]:
    """The book in a model turn, as the G2 harnesses give it to an instruct model."""
    with open( BOOKS / f'{book}.txt', 'rb' ) as f:
        text = join_wraps( f.read( tokens * 8 ).decode( 'utf-8', errors='ignore' ) )

    return [ BEGIN_OF_TEXT ] + tokenizer.encode( BOOK_TURN + text, add_special_tokens=False ).ids[ :tokens - 1 ]


@torch.no_grad()
def score( head: torch.Tensor, reference: torch.Tensor, states, softcap: float ):
    """Mean NLL of the next token, mean KL from the BF16 head, and top-1 agreement with it, over every position."""
    nll = kl = agree = count = 0.0

    def capped( logits ):
        logits = logits.float()

        return torch.tanh( logits / softcap ) * softcap if softcap else logits

    for hidden, ids in states:
        for start in range( 0, hidden.shape[ 0 ] - 1, POSITIONS_PER_PASS ):
            rows = hidden[ start:min( start + POSITIONS_PER_PASS, hidden.shape[ 0 ] - 1 ) ]
            targets = ids[ start + 1:start + 1 + rows.shape[ 0 ] ]
            wanted = torch.log_softmax( capped( rows @ reference.T ), dim=-1 )
            got = torch.log_softmax( capped( rows @ head.T ), dim=-1 )

            nll -= got.gather( 1, targets.unsqueeze( 1 ) ).sum().item()
            kl += ( wanted.exp() * ( wanted - got ) ).sum().item()
            agree += ( wanted.argmax( dim=-1 ) == got.argmax( dim=-1 ) ).sum().item()
            count += rows.shape[ 0 ]

    return nll / count, kl / count, agree / count, int( count )


def main():
    parser = argparse.ArgumentParser( description='Gemma 4 tied head formats against BF16 on the same hidden states' )
    parser.add_argument( '--model', required=True, help='Local checkpoint directory (Google\'s QAT BF16 weights)' )
    parser.add_argument( '--books', nargs='+', default=[ '30312', '3608' ] )
    parser.add_argument( '--tokens', type=int, default=2048 )
    parser.add_argument( '--device', default='cuda' )
    args = parser.parse_args()

    os.environ.setdefault( 'CUDA_VISIBLE_DEVICES', 'GPU-9a81c7d1-9db2-16b3-c256-2f991ec2a22c' )

    from tokenizers import Tokenizer

    root, config, weights, prefix = _open_checkpoint( args.model )
    tokenizer = Tokenizer.from_file( str( root / 'tokenizer.json' ) )
    driver = AllPositions( config, weights, prefix, args.device, torch.bfloat16 )

    states = []

    for book in args.books:
        ids = book_ids( tokenizer, book, args.tokens )
        hidden = driver.run( ids, config.num_hidden_layers, progress=False )[ 'final_hidden' ]
        states.append( ( hidden, torch.tensor( ids, device=args.device ) ) )
        print( f'  book {book}: {len( ids )} tokens through {config.num_hidden_layers} layers', flush=True )

    table_name = f'{prefix}embed_tokens.weight' if config.tie_word_embeddings else 'lm_head.weight'
    reference = weights.tensor( table_name ).to( device=args.device, dtype=torch.bfloat16 )
    softcap = getattr( config, 'final_logit_softcapping', None ) or 0.0
    reference_bytes = reference.numel() * 2

    print( f'\n  {root.name}: head {tuple( reference.shape )}, softcap {softcap}\n' )
    print( f'  {"head":<20} {"bits":>7} {"MiB":>7}  {"NLL":>8} {"KL from BF16":>13} {"top-1 agree":>12}' )

    for name, bits, quantize in ARMS:
        head = reference if quantize is None else quantize( reference )
        nll, kl, agree, count = score( head, reference, states, softcap )
        mib = reference_bytes * bits / 16.0 / 2 ** 20

        print( f'  {name:<20} {bits:>7.4g} {mib:>7.0f}  {nll:>8.5f} {kl:>13.3e} {agree:>12.5f}', flush=True )

        if quantize is not None:
            del head
            _release( args.device )

    print( f'\n  {count} positions scored per arm' )


if __name__ == '__main__':
    main()
