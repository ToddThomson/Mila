# Llama 3.1 8B Instruct's loss along a PG-19 book in HuggingFace, by weight format (ModelFamilyParity.md 8.4, L3).
#
# The book goes into a user turn as LlamaLogLikelihoodCudaTests builds it: <|begin_of_text|>, a user turn "Continue
# this book.", the assistant header, then the book with Gutenberg's line wraps joined -- no system block, as Chat
# renders it. The body's projections are rounded as Mila rounds them for the chosen format; the embedding table and
# the head stay BF16, as in Mila's Llama packages. The sequence is fed in 1024-token chunks through the KV cache and
# the loss is reported per 8K band, as hf_long_context_loss.py does for Gemma.
#
#   python hf_llama_long_context_loss.py --weights fp4 --book Data/Datasets/PG19/raw/test/30312.txt --tokens 32768

import argparse
import os
import sys
from pathlib import Path

BOOK_TURN = ( '<|start_header_id|>user<|end_header_id|>\n\nContinue this book.<|eot_id|>'
              '<|start_header_id|>assistant<|end_header_id|>\n\n' )
BAND = 8192
CHUNK = 1024
LOG_SOFTMAX_ROWS = 256
CHECKPOINT = 'meta-llama/Llama-3.1-8B-Instruct'

# The 5060 Ti, then the 4070: BF16 8B does not fit one card with a 32K cache.
DEVICES = 'GPU-9a81c7d1-9db2-16b3-c256-2f991ec2a22c,GPU-11770557-a821-1b71-35db-34e342bae260'

GEMMA_SCRIPTS = Path( __file__ ).resolve().parents[ 1 ] / 'Gemma' / 'gemma_4_BF16'


def join_wraps( crlf: str ) -> str:
    """The book text as every G2 and L3 harness joins it (hf_long_context_loss.py)."""
    sys.path.insert( 0, str( GEMMA_SCRIPTS ) )
    from hf_long_context_loss import join_wraps as shared

    return shared( crlf )


def load( weights: str, memory: list[ str ] ):
    import torch
    from accelerate import dispatch_model, infer_auto_device_map
    from transformers import AutoModelForCausalLM

    # The quantizers Gemma's G2 arms used, so the two families' arms are rounded by the same code.
    sys.path.insert( 0, str( GEMMA_SCRIPTS ) )
    from hf_fp8_layer_comparison import quantize_fp4_like_mila, quantize_like_mila, quantize_q4_0

    model = AutoModelForCausalLM.from_pretrained( CHECKPOINT, dtype=torch.bfloat16, device_map='cpu' ).eval()

    body = { 'fp8': quantize_like_mila, 'fp4': quantize_fp4_like_mila, 'fp4-fp8-attention': quantize_fp4_like_mila,
             'q4_0': quantize_q4_0 }.get( weights )

    if body is not None:
        with torch.no_grad():
            for layer in model.model.layers:
                for sublayer in ( layer.self_attn, layer.mlp ):
                    for projection, module in sublayer.named_children():
                        if not isinstance( module, torch.nn.Linear ):
                            continue

                        attention = projection in ( 'q_proj', 'k_proj', 'v_proj', 'o_proj' )
                        quantize = quantize_like_mila if weights == 'fp4-fp8-attention' and attention else body
                        module.weight.copy_( quantize( module.weight ) )

    device_map = infer_auto_device_map( model, max_memory={ 0: memory[ 0 ], 1: memory[ 1 ], 'cpu': '24GiB' },
        no_split_module_classes=model._no_split_modules )

    return dispatch_model( model, device_map=device_map )


def main():
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( '--weights', choices=[ 'bf16', 'fp8', 'fp4', 'fp4-fp8-attention', 'q4_0' ], default='fp4' )
    parser.add_argument( '--book', required=True )
    parser.add_argument( '--tokens', type=int, default=32768 )
    # Weights per card. 12 GiB on the 5060 Ti left too little for attention at 24K.
    parser.add_argument( '--gpu-memory', nargs=2, default=[ '10GiB', '7GiB' ] )
    parser.add_argument( '--short-context', type=int, default=0,
        help="also score test 1's short arm: each block of this many targets after only this many book tokens" )
    arguments = parser.parse_args()

    os.environ.setdefault( 'CUDA_VISIBLE_DEVICES', DEVICES )
    os.environ.setdefault( 'PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True' )

    import torch
    from transformers import AutoTokenizer, DynamicCache

    tokenizer = AutoTokenizer.from_pretrained( CHECKPOINT )

    with open( arguments.book, 'rb' ) as book:
        text = join_wraps( book.read( arguments.tokens * 6 ).decode( 'utf-8', errors='replace' ) )

    prompt = [ tokenizer.bos_token_id ] + tokenizer.encode( BOOK_TURN, add_special_tokens=False )
    body = tokenizer.encode( text, add_special_tokens=False )
    ids = ( prompt + body )[ : arguments.tokens ]

    print( f'  {arguments.weights}: prompt ids {" ".join( str( id ) for id in prompt )}' )
    print( f'  {len( ids )} tokens, {len( prompt )} of prompt', flush=True )

    model = load( arguments.weights, arguments.gpu_memory )

    cache = DynamicCache( config=model.config )
    tokens = torch.tensor( [ ids ] )
    log_probabilities = torch.empty( len( ids ) - 1, dtype=torch.float64 )

    with torch.no_grad():
        for start in range( 0, len( ids ), CHUNK ):
            chunk = tokens[ :, start : start + CHUNK ].to( model.device )
            logits = model( input_ids=chunk, past_key_values=cache, use_cache=True ).logits[ 0 ]

            # Row r predicts the token at start + r + 1; the last row of the last chunk predicts nothing.
            targets = tokens[ 0, start + 1 : start + 1 + logits.shape[ 0 ] ]

            for row in range( 0, targets.shape[ 0 ], LOG_SOFTMAX_ROWS ):
                rows = logits[ row : row + LOG_SOFTMAX_ROWS ].double()
                wanted = targets[ row : row + rows.shape[ 0 ] ].unsqueeze( 1 ).to( rows.device )
                scored = torch.log_softmax( rows[ : wanted.shape[ 0 ] ], dim=-1 ).gather( 1, wanted )
                log_probabilities[ start + row : start + row + wanted.shape[ 0 ] ] = scored[ :, 0 ].cpu()

            if ( start + CHUNK ) % BAND == 0:
                print( f'  scored to {start + CHUNK}', flush=True )

        cache = None

        short = score_short_context( model, ids, len( prompt ), arguments.short_context ) if arguments.short_context else None

    # Entry p predicts token p + 1, so a band of target tokens [s, e) is entries [s - 1, e - 1), as Mila's prefix
    # differences count them; the book's first token is predicted at entry len( prompt ) - 1.
    first = len( prompt ) - 1

    for band_start in range( 0, len( ids ), BAND ):
        low = max( band_start - 1, first )
        high = min( band_start + BAND, len( ids ) ) - 1
        band = log_probabilities[ low : high ]
        line = f'  {band_start:>7} - {band_start + BAND:>7}: {band.numel():>5} positions, {-band.mean().item():.4f} nats/token'

        if short is not None:
            short_nats = -short[ low : high ].mean().item()
            line += f', short context {short_nats:.4f} ({"pass" if -band.mean().item() <= short_nats else "FAIL"})'

        print( line )


def score_short_context( model, ids: list[ int ], prompt_length: int, span: int ):
    """
    L3's test 1 short arm, as LlamaQualityCudaTests builds it: targets in blocks of `span`, each block scored after
    the prompt and only the `span` book tokens before it. Entry p holds the log-probability of token p + 1, as the
    whole-book array does, so the two compare position for position.
    """
    import torch

    short = torch.full( ( len( ids ) - 1, ), float( 'nan' ), dtype=torch.float64 )
    prompt = ids[ : prompt_length ]

    for block in range( 0, len( ids ), span ):
        first_target = max( block, prompt_length )
        end = min( block + span, len( ids ) )

        if first_target >= end:
            continue

        context_start = max( prompt_length, block - span )
        sequence = prompt + ids[ context_start : first_target ] + ids[ first_target : end ]
        first_row = prompt_length + ( first_target - context_start ) - 1

        logits = model( input_ids=torch.tensor( [ sequence ] ).to( model.device ) ).logits[ 0 ]
        targets = torch.tensor( sequence[ first_row + 1 : ] )

        for row in range( 0, targets.shape[ 0 ], LOG_SOFTMAX_ROWS ):
            rows = logits[ first_row + row : first_row + row + LOG_SOFTMAX_ROWS ].double()
            wanted = targets[ row : row + rows.shape[ 0 ] ].unsqueeze( 1 ).to( rows.device )
            scored = torch.log_softmax( rows[ : wanted.shape[ 0 ] ], dim=-1 ).gather( 1, wanted )
            short[ first_target - 1 + row : first_target - 1 + row + wanted.shape[ 0 ] ] = scored[ :, 0 ].cpu()

    return short


if __name__ == '__main__':
    main()
