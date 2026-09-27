# Gemma 4 12B's loss along a PG-19 book in HuggingFace, the reference for Mila's G2 curve (ModelFamilyParity.md 8.2).
#
# The book goes into a model turn exactly as GemmaLogLikelihoodCudaTests builds it (bookSegment): <bos>, a user turn
# "Continue this book.", the model primer with thinking off, then the book with Gutenberg's line wraps joined. The
# sequence is fed in 1024-token chunks through the model's KV cache, so no attention is wider than one chunk against
# the context, and every book token is scored after the final-logit softcap. Loss is reported per 8K band.
#
#   python hf_long_context_loss.py --weights fp4 --book Data/Datasets/PG19/raw/test/30312.txt --tokens 32768

import argparse
import os

BOOK_TURN = '<|turn>user\nContinue this book.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>'
BAND = 8192
CHUNK = 1024


def join_wraps( stored: str ) -> str:
    """A lone newline becomes a space; a blank line stays a paragraph. Mila's joinWraps."""
    joined = list( stored )

    for index, character in enumerate( stored ):
        before = index > 0 and stored[ index - 1 ] == '\n'
        after = index + 1 < len( stored ) and stored[ index + 1 ] == '\n'

        if character == '\n' and not before and not after:
            joined[ index ] = ' '

    return ''.join( joined )


def main():
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( '--weights', choices=[ 'bf16', 'fp8', 'fp4', 'fp4-attention', 'fp4-mlp', 'fp4-fp8-attention', 'q4_0', 'nvfp4', 'nvfp4-best-scale', 'nvfp4-scale-up', 'q4_0-fp8-attention', 'fp4-fp8-query-key', 'fp4-query-key', 'w4a8' ], default='fp4' )
    parser.add_argument( '--checkpoint', default='google/gemma-4-12b-it' )
    parser.add_argument( '--book', required=True )
    parser.add_argument( '--tokens', type=int, default=32768 )
    parser.add_argument( '--gpu-memory', default='6GiB' )
    arguments = parser.parse_args()

    os.environ.setdefault( 'CUDA_VISIBLE_DEVICES', 'GPU-11770557-a821-1b71-35db-34e342bae260' )

    # Imported here so that llama_cpp_long_context_loss.py can share the template without loading PyTorch, whose
    # OpenMP runtime conflicts with llama.cpp's in one process.
    import torch

    from transformers import AutoTokenizer
    from hf_fp8_layer_comparison import load

    tokenizer = AutoTokenizer.from_pretrained( 'google/gemma-4-12b-it' )

    with open( arguments.book, 'rb' ) as book:
        text = join_wraps( book.read( arguments.tokens * 6 ).decode( 'utf-8', errors='replace' ) )

    prompt = [ tokenizer.bos_token_id ] + tokenizer.encode( BOOK_TURN, add_special_tokens=False )
    body = tokenizer.encode( text, add_special_tokens=False )
    ids = ( prompt + body )[ : arguments.tokens ]

    print( f'  prompt ids: {" ".join( str( id ) for id in prompt )}' )
    print( f'  {len( ids )} tokens, {len( prompt )} of prompt', flush=True )

    model, _ = load( arguments.weights, arguments.gpu_memory, arguments.checkpoint )

    from transformers import DynamicCache

    cache = DynamicCache( config=model.config )
    tokens = torch.tensor( [ ids ] )
    log_probabilities = torch.empty( len( ids ) - 1, dtype=torch.float64 )

    with torch.no_grad():
        for start in range( 0, len( ids ), CHUNK ):
            chunk = tokens[ :, start : start + CHUNK ].to( model.device )
            logits = model( input_ids=chunk, past_key_values=cache, use_cache=True ).logits[ 0 ].double()

            # Row r predicts the token at start + r + 1; the last row of the last chunk predicts nothing.
            targets = tokens[ 0, start + 1 : start + 1 + logits.shape[ 0 ] ]
            rows = targets.shape[ 0 ]
            scored = torch.log_softmax( logits[ : rows ], dim=-1 ).gather( 1, targets.unsqueeze( 1 ).to( logits.device ) )
            log_probabilities[ start : start + rows ] = scored[ :, 0 ].cpu()

            if ( start + CHUNK ) % BAND == 0:
                print( f'  scored to {start + CHUNK}', flush=True )

    # Entry p predicts token p + 1, so a band of target tokens [s, e) is entries [s - 1, e - 1), as Mila's prefix
    # differences count them; the book's first token is predicted at entry len( prompt ) - 1.
    first = len( prompt ) - 1

    for band_start in range( 0, len( ids ), BAND ):
        low = max( band_start - 1, first )
        high = min( band_start + BAND, len( ids ) ) - 1
        band = log_probabilities[ low : high ]
        print( f'  {band_start:>7} - {band_start + BAND:>7}: {band.numel():>5} positions, {-band.mean().item():.4f} nats/token' )


if __name__ == '__main__':
    main()
